//! Root WavLM backbone: optional waveform LayerNorm → conv feature extractor
//! → transformer encoder. Exposes `extract_features` returning all
//! `num_layers + 1` intermediate tensors (the last one is **pre**-final-LN).

use std::path::Path;

use snafu::ResultExt;
use svod_ir::SInt;
use svod_tensor::nn::{Module, StateDict};
use svod_tensor::{BoundVariable, Tensor};

use super::config::WavLmConfig;
use super::encoder::Encoder;
use super::error::{PickleSnafu, Result};
use super::feature_extractor::FeatureExtractor;

#[derive(Clone, Module)]
pub struct WavLm {
    #[module(skip)]
    pub config: WavLmConfig,
    pub feature_extractor: FeatureExtractor,
    pub encoder: Encoder,
}

impl WavLm {
    pub fn empty(config: WavLmConfig) -> Self {
        let feature_extractor = FeatureExtractor::empty(&config);
        let encoder = Encoder::empty(&config);
        Self { config, feature_extractor, encoder }
    }

    /// Run the convolutional feature extractor + transformer encoder and
    /// return all `num_layers + 1` intermediate tensors (see
    /// [`Encoder::extract_features`] for layer-0 / final-LN semantics).
    ///
    /// Eager path: `waveform` is `(B, samples)` with whatever concrete batch
    /// dim the caller already shaped. Use [`extract_features_batch`] for the
    /// JIT path where the batch dim is a symbolic [`BoundVariable`].
    pub fn extract_features(&self, waveform: &Tensor) -> Result<Vec<Tensor>> {
        let normed = if self.config.normalize_waveform { waveform.layernorm(-1, 1e-5)? } else { waveform.clone() };
        let features = self.feature_extractor.forward(&normed)?;
        self.encoder.extract_features(&features)
    }

    /// `extract_features` then stack along a new last axis. Shape:
    /// `(B, T, embed_dim, num_layers + 1)`. Convenient for `weight_sum`-style
    /// downstream heads.
    pub fn extract_features_stacked(&self, waveform: &Tensor) -> Result<Tensor> {
        stack_last(self.extract_features(waveform)?)
    }

    /// JIT-path variant of [`extract_features`]. `waveform` is sized for the
    /// JIT plan's `max_batch`; `batch` shrinks the leading dim to the live
    /// value at execute time. Mirrors gigaam's `forward_batch` pattern.
    pub fn extract_features_batch(&self, waveform: &Tensor, batch: &BoundVariable) -> Result<Vec<Tensor>> {
        let waveform = waveform.try_shrink([Some((SInt::Const(0), batch.as_sint())), None])?;
        self.extract_features(&waveform)
    }

    /// JIT-path variant of [`extract_features_stacked`].
    pub fn extract_features_stacked_batch(&self, waveform: &Tensor, batch: &BoundVariable) -> Result<Tensor> {
        stack_last(self.extract_features_batch(waveform, batch)?)
    }

    /// Download `pytorch_model.bin` from a HuggingFace Hub repository and
    /// load it. Expects DiariZen's nested `{"state_dict": ..., ...}` format
    /// with `wavlm_model.`-prefixed keys.
    pub fn from_hub(model_id: &str, config: WavLmConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: WavLmConfig) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;
        let weights_path = repo.get("pytorch_model.bin")?;
        Self::from_pytorch_bin(&weights_path, config)
    }

    /// Load from a torch-pickled checkpoint with DiariZen's
    /// `{"state_dict": ..., "config": ..., ...}` wrapping. Strips the
    /// `wavlm_model.` prefix from each key.
    pub fn from_pytorch_bin(path: &Path, config: WavLmConfig) -> Result<Self> {
        let sd = crate::wespeaker::pickle::load_pyannote_pytorch_bin(path, "wavlm_model.").context(PickleSnafu)?;
        Self::from_state_dict(&sd, config)
    }

    /// Build from a preloaded state dict (keys without the `wavlm_model.`
    /// prefix).
    pub fn from_state_dict(sd: &StateDict, config: WavLmConfig) -> Result<Self> {
        let mut model = Self::empty(config);
        model.load_state_dict(&drop_inert_keys(sd), "")?;
        Ok(model)
    }
}

/// Drop the buffers torchaudio writes but our forward never consults:
/// `*.num_batches_tracked` (PyTorch BN metadata) and `*.hard_concrete_for_*`
/// (the head-pruning gates, already baked into the pruned linear shapes).
pub(crate) fn drop_inert_keys(sd: &StateDict) -> StateDict {
    let inert = |k: &str| k.ends_with("num_batches_tracked") || k.contains("hard_concrete_for_");
    sd.iter().filter(|(k, _)| !inert(k)).map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// Stack same-shaped tensors along a new trailing axis.
fn stack_last(layers: Vec<Tensor>) -> Result<Tensor> {
    let unsqueezed: Vec<Tensor> =
        layers.iter().map(|t| t.try_unsqueeze(-1)).collect::<svod_tensor::error::Result<_>>()?;
    Ok(Tensor::cat(&unsqueezed.iter().collect::<Vec<_>>(), -1)?)
}
