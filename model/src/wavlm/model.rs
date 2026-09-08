//! Root WavLM backbone: optional waveform LayerNorm → conv feature extractor
//! → transformer encoder. Exposes `extract_features` returning all
//! `num_layers + 1` intermediate tensors (the last one is **pre**-final-LN;

use std::path::Path;

use snafu::ResultExt;
use svod_ir::SInt;
use svod_tensor::{BoundVariable, Tensor};

use crate::state::{self, HasStateDict, StateDict};

use super::config::WavLmConfig;
use super::encoder::Encoder;
use super::error::{HubSnafu, PickleSnafu, Result, StateSnafu, TensorSnafu};
use super::feature_extractor::FeatureExtractor;

#[derive(Clone)]
pub struct WavLm {
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
        let normed = if self.config.normalize_waveform {
            waveform.layernorm(-1, 1e-5).context(TensorSnafu)?
        } else {
            waveform.clone()
        };
        let features = self.feature_extractor.forward(&normed)?;
        self.encoder.extract_features(&features)
    }

    /// `extract_features` then stack along a new last axis. Shape:
    /// `(B, T, embed_dim, num_layers + 1)`. Convenient for `weight_sum`-style
    /// downstream heads.
    pub fn extract_features_stacked(&self, waveform: &Tensor) -> Result<Tensor> {
        let layers = self.extract_features(waveform)?;
        let unsq: Result<Vec<Tensor>> = layers.iter().map(|t| t.try_unsqueeze(-1).context(TensorSnafu)).collect();
        let unsq = unsq?;
        let refs: Vec<&Tensor> = unsq.iter().collect();
        Tensor::cat(&refs, -1).context(TensorSnafu)
    }

    /// JIT-path variant of [`extract_features`]. `waveform` is sized for the
    /// JIT plan's `max_batch`; `batch` shrinks the leading dim to the live
    /// value at execute time. Mirrors gigaam's `forward_batch` pattern.
    pub fn extract_features_batch(&self, waveform: &Tensor, batch: &BoundVariable) -> Result<Vec<Tensor>> {
        let b = batch.as_sint();
        let waveform = waveform.try_shrink([Some((SInt::Const(0), b)), None]).context(TensorSnafu)?;
        self.extract_features(&waveform)
    }

    /// JIT-path variant of [`extract_features_stacked`].
    pub fn extract_features_stacked_batch(&self, waveform: &Tensor, batch: &BoundVariable) -> Result<Tensor> {
        let layers = self.extract_features_batch(waveform, batch)?;
        let unsq: Result<Vec<Tensor>> = layers.iter().map(|t| t.try_unsqueeze(-1).context(TensorSnafu)).collect();
        let unsq = unsq?;
        let refs: Vec<&Tensor> = unsq.iter().collect();
        Tensor::cat(&refs, -1).context(TensorSnafu)
    }

    /// Download `pytorch_model.bin` from a HuggingFace Hub repository and
    /// load it. Expects DiariZen's nested `{"state_dict": ..., ...}` format
    /// with `wavlm_model.`-prefixed keys.
    pub fn from_hub(model_id: &str, config: WavLmConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: WavLmConfig) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision).context(HubSnafu)?;
        let weights_path = repo.get("pytorch_model.bin").context(HubSnafu)?;
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
    /// prefix). Skips a small set of inert buffers
    /// (`*.num_batches_tracked`, `*.hard_concrete_for_*`) that torchaudio
    /// writes but our forward never consults.
    pub fn from_state_dict(sd: &StateDict, config: WavLmConfig) -> Result<Self> {
        let sd: StateDict = sd.iter().filter(|(k, _)| !is_inert_key(k)).map(|(k, v)| (k.clone(), v.clone())).collect();
        let mut model = Self::empty(config);
        model.load_state_dict(&sd, "").context(StateSnafu)?;
        Ok(model)
    }
}

fn is_inert_key(key: &str) -> bool {
    key.ends_with("num_batches_tracked") || key.contains("hard_concrete_for_")
}

impl HasStateDict for WavLm {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.feature_extractor.state_dict(&prefix_or(prefix, "feature_extractor"));
        sd.extend(self.encoder.state_dict(&prefix_or(prefix, "encoder")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.feature_extractor.load_state_dict(sd, &prefix_or(prefix, "feature_extractor"))?;
        self.encoder.load_state_dict(sd, &prefix_or(prefix, "encoder"))?;
        Ok(())
    }
}

fn prefix_or(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() { suffix.to_string() } else { format!("{prefix}.{suffix}") }
}
