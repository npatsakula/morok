//! DiariZen segmentation model: WavLM backbone + weighted-layer-sum +
//! projection + LayerNorm + Conformer + powerset classifier.
//!
//! Direct port of `Model` from
//! `submodules/DiariZen/diarizen/models/eend/model_wavlm_conformer.py:25-264`.
//! Forward path on `(B, channels, samples)`:
//!
//! 1. Take `waveform[:, selected_channel, :]` → `(B, samples)`.
//! 2. WavLM `extract_features_stacked` → `(B, T, embed_dim, num_layers + 1)`.
//! 3. `weight_sum: Linear(num_layers + 1, 1, bias=False)` over the last axis
//!    → `(B, T, embed_dim, 1)` → squeeze → `(B, T, embed_dim)`.
//! 4. `proj: Linear(embed_dim, attention_in)` + `lnorm: LayerNorm(attention_in)`.
//! 5. ConformerEncoder → `(B, T, attention_in)`.
//! 6. `classifier: Linear(attention_in, K)` → `(B, T, K)`. `K` is the powerset
//!    class count.
//! 7. Final activation: `LogSoftmax(-1)` — pyannote's default for powerset
//!    segmentation. The parity reference captures the post-activation tensor
//!    so we match here.

use std::path::Path;

use snafu::ResultExt;
use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::nn::{Layer, LayerNorm, Linear, Module, StateDict};
use svod_tensor::{BoundVariable, Tensor, s};

use crate::init::{fan_in_uniform, ones, zeros};
use crate::wavlm::{WavLm, drop_inert_keys};

use super::config::DiariZenConfig;
use super::conformer::ConformerEncoder;
use super::error::{PickleSnafu, Result};

/// Every intermediate tensor captured during a [`DiariZenSegmentationModel`]
/// forward. Matches the keys saved by `scripts/diarizen_dump_reference.py`.
#[derive(Clone)]
pub struct ForwardIntermediates {
    /// `(B, T, D)` per layer; length = `wavlm_layer_num()` = 25 for s80-md-v2.
    /// Index 0 is the post-pos-conv input; the last entry is pre-final-LN.
    pub wavlm_intermediates: Vec<Tensor>,
    /// `(B, T, D)` after weight_sum + squeeze.
    pub weighted_sum: Tensor,
    /// `(B, T, attention_in)` after the projection Linear.
    pub proj_out: Tensor,
    /// `(B, T, attention_in)` after LayerNorm.
    pub lnorm_out: Tensor,
    /// One `(B, T, attention_in)` per Conformer block.
    pub conformer_blocks: Vec<Tensor>,
    /// `(B, T, K)` pre-activation classifier logits.
    pub classifier_logits: Tensor,
    /// `(B, T, K)` post log-softmax.
    pub final_out: Tensor,
}

/// The published checkpoint nests the backbone under `wavlm_model.` (Python
/// `Model.__init__` assigns `self.wavlm_model`), so that is the derived key.
#[derive(Clone, Module)]
pub struct DiariZenSegmentationModel {
    #[module(skip)]
    pub config: DiariZenConfig,
    #[module(key = "wavlm_model")]
    pub wavlm: WavLm,
    /// `(1, num_layers + 1)` — `Linear(layers + 1 → 1, bias=False)` applied
    /// along the last axis of the stacked WavLM intermediates.
    #[module(key = "weight_sum.weight")]
    pub weight_sum: Tensor,
    pub proj: Linear,
    pub lnorm: LayerNorm,
    pub conformer: ConformerEncoder,
    pub classifier: Linear,
}

impl DiariZenSegmentationModel {
    pub fn empty(config: DiariZenConfig) -> Self {
        let wavlm = WavLm::empty(config.wavlm.clone());
        let lnum = config.wavlm_layer_num();
        let feat_dim = config.wavlm_feat_dim();
        let attn_in = config.attention_in;
        let k = config.powerset_class_count();

        let linear = |out: usize, inp: usize| {
            Linear::new(fan_in_uniform(&[out, inp], inp, DType::Float32), Some(zeros(&[out], DType::Float32)))
        };
        Self {
            wavlm,
            weight_sum: fan_in_uniform(&[1, lnum], lnum, DType::Float32),
            proj: linear(attn_in, feat_dim),
            lnorm: LayerNorm::new(ones(&[attn_in], DType::Float32), Some(zeros(&[attn_in], DType::Float32)), 1e-5),
            conformer: ConformerEncoder::empty(
                attn_in,
                config.ffn_hidden,
                config.num_head,
                config.num_layer,
                config.kernel_size,
            ),
            classifier: linear(k, attn_in),
            config,
        }
    }

    /// Eager forward on `(B, channels, samples)`. Returns `(B, T, K)`
    /// log-probabilities over the powerset of speaker subsets.
    /// `selected_channel` is hardcoded to 0 (matches the published config).
    pub fn forward(&self, waveforms: &Tensor) -> Result<Tensor> {
        let waveforms = self.select_channel(waveforms)?;
        let stacked = self.wavlm.extract_features_stacked(&waveforms)?;
        self.head_forward(&stacked)
    }

    /// JIT-path variant of [`forward`]. `waveforms` is sized for the JIT
    /// plan's `max_batch`; `batch` shrinks the leading dim at execute time.
    pub fn forward_batch(&self, waveforms: &Tensor, batch: &BoundVariable) -> Result<Tensor> {
        let b = batch.as_sint();
        let waveforms = waveforms.try_shrink([Some((SInt::Const(0), b)), None, None])?;
        self.forward(&waveforms)
    }

    fn select_channel(&self, waveforms: &Tensor) -> Result<Tensor> {
        // Py: waveforms[:, 0, :]
        Ok(waveforms.getitem(s![.., 0, ..])?)
    }

    fn head_forward(&self, stacked: &Tensor) -> Result<Tensor> {
        Ok(self.head_logits(stacked)?.log_softmax(-1)?)
    }

    /// The head up to (but not including) the final log-softmax.
    fn head_logits(&self, stacked: &Tensor) -> Result<Tensor> {
        // weight_sum: Linear(L+1 → 1, bias=False) over the last axis.
        let summed = stacked.linear().weight(&self.weight_sum).call()?.try_squeeze(Some(-1))?;
        let h = self.lnorm.forward(&self.proj.forward(&summed)?)?;
        Ok(self.classifier.forward(&self.conformer.forward(&h)?)?)
    }

    /// Eager forward that returns every intermediate stage. Used by the
    /// `diarizen_parity` example to measure stage-by-stage MSE against the
    /// Python reference dump.
    pub fn forward_with_intermediates(&self, waveforms: &Tensor) -> Result<ForwardIntermediates> {
        let waveforms = self.select_channel(waveforms)?;

        let wavlm_intermediates = self.wavlm.extract_features(&waveforms)?;
        let unsq: Vec<Tensor> =
            wavlm_intermediates.iter().map(|t| t.try_unsqueeze(-1)).collect::<svod_tensor::error::Result<_>>()?;
        let stacked = Tensor::cat(&unsq.iter().collect::<Vec<_>>(), -1)?;

        let weighted_sum = stacked.linear().weight(&self.weight_sum).call()?.try_squeeze(Some(-1))?;
        let proj_out = self.proj.forward(&weighted_sum)?;
        let lnorm_out = self.lnorm.forward(&proj_out)?;

        let (conformer_out, conformer_blocks) = self.conformer.forward_with_block_outputs(&lnorm_out)?;

        let classifier_logits = self.classifier.forward(&conformer_out)?;
        let final_out = classifier_logits.log_softmax(-1)?;

        Ok(ForwardIntermediates {
            wavlm_intermediates,
            weighted_sum,
            proj_out,
            lnorm_out,
            conformer_blocks,
            classifier_logits,
            final_out,
        })
    }

    /// Forward returning the un-activated classifier logits (pre log-softmax).
    /// Useful for parity testing the classifier output directly.
    pub fn forward_logits(&self, waveforms: &Tensor) -> Result<Tensor> {
        let waveforms = self.select_channel(waveforms)?;
        let stacked = self.wavlm.extract_features_stacked(&waveforms)?;
        self.head_logits(&stacked)
    }

    /// Download the published DiariZen segmentation checkpoint and load it.
    pub fn from_hub(model_id: &str, config: DiariZenConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: DiariZenConfig) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;
        let weights_path = repo.get("pytorch_model.bin")?;
        Self::from_pytorch_bin(&weights_path, config)
    }

    /// Load from a local DiariZen `pytorch_model.bin`. The file is a torch
    /// pickle with a nested `{"state_dict": OrderedDict(...)}`; the pickle
    /// loader returns the inner dict, whose keys already carry the module
    /// layout this model derives.
    pub fn from_pytorch_bin(path: &Path, config: DiariZenConfig) -> Result<Self> {
        let raw_sd = crate::wespeaker::pickle::load_flat_pytorch_bin(path, "").context(PickleSnafu)?;
        Self::from_state_dict(&raw_sd, config)
    }

    pub fn from_state_dict(sd: &StateDict, config: DiariZenConfig) -> Result<Self> {
        let mut model = Self::empty(config);
        model.load_state_dict(&drop_inert_keys(sd), "")?;
        Ok(model)
    }
}
