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
use svod_tensor::{BoundVariable, Tensor, s};

use crate::init::{fan_in_uniform, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};
use crate::wavlm::{LayerNormWeights, WavLm};

use super::config::DiariZenConfig;
use super::conformer::ConformerEncoder;
use super::error::{PickleSnafu, Result, WavLmSnafu};

use super::remap::split_diarizen_state_dict;

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

#[derive(Clone)]
pub struct DiariZenSegmentationModel {
    pub config: DiariZenConfig,
    pub wavlm: WavLm,
    /// `(1, num_layers + 1)` — `Linear(layers + 1 → 1, bias=False)` applied
    /// along the last axis of the stacked WavLM intermediates.
    pub weight_sum_weight: Tensor,
    pub proj_weight: Tensor,
    pub proj_bias: Tensor,
    pub lnorm: LayerNormWeights,
    pub conformer: ConformerEncoder,
    pub classifier_weight: Tensor,
    pub classifier_bias: Tensor,
}

impl DiariZenSegmentationModel {
    pub fn empty(config: DiariZenConfig) -> Self {
        let wavlm = WavLm::empty(config.wavlm.clone());
        let lnum = config.wavlm_layer_num();
        let feat_dim = config.wavlm_feat_dim();
        let attn_in = config.attention_in;
        let k = config.powerset_class_count();

        let weight_sum_weight = fan_in_uniform(&[1, lnum], lnum, DType::Float32);
        let proj_weight = fan_in_uniform(&[attn_in, feat_dim], feat_dim, DType::Float32);
        let proj_bias = zeros(&[attn_in], DType::Float32);
        let lnorm = LayerNormWeights::empty(attn_in);
        let conformer =
            ConformerEncoder::empty(attn_in, config.ffn_hidden, config.num_head, config.num_layer, config.kernel_size);
        let classifier_weight = fan_in_uniform(&[k, attn_in], attn_in, DType::Float32);
        let classifier_bias = zeros(&[k], DType::Float32);

        Self {
            config,
            wavlm,
            weight_sum_weight,
            proj_weight,
            proj_bias,
            lnorm,
            conformer,
            classifier_weight,
            classifier_bias,
        }
    }

    /// Eager forward on `(B, channels, samples)`. Returns `(B, T, K)`
    /// log-probabilities over the powerset of speaker subsets.
    /// `selected_channel` is hardcoded to 0 (matches the published config).
    pub fn forward(&self, waveforms: &Tensor) -> Result<Tensor> {
        let waveforms = self.select_channel(waveforms)?;
        let stacked = self.wavlm.extract_features_stacked(&waveforms).context(WavLmSnafu)?;
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
        // weight_sum: Linear(L+1 → 1, bias=False) over last axis.
        let summed = stacked.linear().weight(&self.weight_sum_weight).call()?;
        let summed = summed.try_squeeze(Some(-1))?;

        let h = summed.linear().weight(&self.proj_weight).bias(&self.proj_bias).call()?;
        let h = self.lnorm.apply(&h)?;
        let h = self.conformer.forward(&h)?;
        let logits = h.linear().weight(&self.classifier_weight).bias(&self.classifier_bias).call()?;
        Ok(logits.log_softmax(-1)?)
    }

    /// Eager forward that returns every intermediate stage. Used by the
    /// `diarizen_parity` example to measure stage-by-stage MSE against the
    /// Python reference dump.
    pub fn forward_with_intermediates(&self, waveforms: &Tensor) -> Result<ForwardIntermediates> {
        let waveforms = self.select_channel(waveforms)?;

        let wavlm_intermediates = self.wavlm.extract_features(&waveforms).context(WavLmSnafu)?;
        let unsq: Vec<Tensor> =
            wavlm_intermediates.iter().map(|t| Ok(t.try_unsqueeze(-1)?)).collect::<Result<Vec<_>>>()?;
        let refs: Vec<&Tensor> = unsq.iter().collect();
        let stacked = Tensor::cat(&refs, -1)?;

        let weighted_sum = stacked.linear().weight(&self.weight_sum_weight).call()?;
        let weighted_sum = weighted_sum.try_squeeze(Some(-1))?;

        let proj_out = weighted_sum.linear().weight(&self.proj_weight).bias(&self.proj_bias).call()?;
        let lnorm_out = self.lnorm.apply(&proj_out)?;

        let (conformer_out, conformer_blocks) = self.conformer.forward_with_block_outputs(&lnorm_out)?;

        let classifier_logits =
            conformer_out.linear().weight(&self.classifier_weight).bias(&self.classifier_bias).call()?;
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
        let stacked = self.wavlm.extract_features_stacked(&waveforms).context(WavLmSnafu)?;
        let summed = stacked.linear().weight(&self.weight_sum_weight).call()?;
        let summed = summed.try_squeeze(Some(-1))?;
        let h = summed.linear().weight(&self.proj_weight).bias(&self.proj_bias).call()?;
        let h = self.lnorm.apply(&h)?;
        let h = self.conformer.forward(&h)?;
        Ok(h.linear().weight(&self.classifier_weight).bias(&self.classifier_bias).call()?)
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
    /// loader returns the inner dict, and `split_diarizen_state_dict` peels
    /// the `wavlm_model.` prefix for the backbone keys.
    pub fn from_pytorch_bin(path: &Path, config: DiariZenConfig) -> Result<Self> {
        let raw_sd = crate::wespeaker::pickle::load_flat_pytorch_bin(path, "").context(PickleSnafu)?;
        Self::from_state_dict(&raw_sd, config)
    }

    pub fn from_state_dict(sd: &StateDict, config: DiariZenConfig) -> Result<Self> {
        let (wavlm_sd, head_sd) = split_diarizen_state_dict(sd.clone())?;
        // PyTorch checkpoints carry raw `running_var`; fold to `invstd` (value
        // transform + key rename) once at load. Round-tripped state dicts
        // already use `invstd` keys and skip this call.
        let head_sd = crate::blocks::remap::fold_batchnorm(head_sd)?;
        let mut model = Self::empty(config);
        model.wavlm.load_state_dict(&wavlm_sd, "")?;
        model.load_head_state_dict(&head_sd)?;
        Ok(model)
    }

    fn load_head_state_dict(&mut self, sd: &StateDict) -> Result<()> {
        self.weight_sum_weight = get_tensor(sd, "weight_sum.weight")?;
        self.proj_weight = get_tensor(sd, "proj.weight")?;
        self.proj_bias = get_tensor(sd, "proj.bias")?;
        self.lnorm.load_state_dict(sd, "lnorm")?;
        self.conformer.load_state_dict(sd, "conformer")?;
        self.classifier_weight = get_tensor(sd, "classifier.weight")?;
        self.classifier_bias = get_tensor(sd, "classifier.bias")?;
        Ok(())
    }
}

impl HasStateDict for DiariZenSegmentationModel {
    fn state_dict(&self, prefix: &str) -> StateDict {
        // Emit keys in the upstream DiariZen layout — including `wavlm_model.`
        // prefix on the backbone — so round-trip mirrors what the published
        // checkpoint stores.
        let wavlm_prefix = if prefix.is_empty() { "wavlm_model".to_string() } else { format!("{prefix}.wavlm_model") };
        let mut sd = self.wavlm.state_dict(&wavlm_prefix);
        sd.insert(prefixed(prefix, "weight_sum.weight"), self.weight_sum_weight.clone());
        sd.insert(prefixed(prefix, "proj.weight"), self.proj_weight.clone());
        sd.insert(prefixed(prefix, "proj.bias"), self.proj_bias.clone());
        sd.extend(self.lnorm.state_dict(&prefixed(prefix, "lnorm")));
        sd.extend(self.conformer.state_dict(&prefixed(prefix, "conformer")));
        sd.insert(prefixed(prefix, "classifier.weight"), self.classifier_weight.clone());
        sd.insert(prefixed(prefix, "classifier.bias"), self.classifier_bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        // Split: wavlm-prefixed keys go to the backbone; the rest stay.
        let wavlm_prefix =
            if prefix.is_empty() { "wavlm_model.".to_string() } else { format!("{prefix}.wavlm_model.") };
        let mut wavlm_sd = StateDict::new();
        let mut head_sd = StateDict::new();
        for (k, v) in sd {
            if let Some(rest) = k.strip_prefix(&wavlm_prefix) {
                wavlm_sd.insert(rest.to_string(), v.clone());
            } else if prefix.is_empty() {
                head_sd.insert(k.clone(), v.clone());
            } else if let Some(rest) = k.strip_prefix(&format!("{prefix}.")) {
                head_sd.insert(rest.to_string(), v.clone());
            }
        }
        self.wavlm.load_state_dict(&wavlm_sd, "")?;
        self.weight_sum_weight = get_tensor(&head_sd, "weight_sum.weight")?;
        self.proj_weight = get_tensor(&head_sd, "proj.weight")?;
        self.proj_bias = get_tensor(&head_sd, "proj.bias")?;
        self.lnorm.load_state_dict(&head_sd, "lnorm")?;
        self.conformer.load_state_dict(&head_sd, "conformer")?;
        self.classifier_weight = get_tensor(&head_sd, "classifier.weight")?;
        self.classifier_bias = get_tensor(&head_sd, "classifier.bias")?;
        Ok(())
    }
}
