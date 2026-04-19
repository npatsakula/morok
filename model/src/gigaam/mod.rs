mod encoder;
mod error;
mod head;
pub(crate) mod remap;
mod rope;

pub use encoder::*;
pub use error::{Error, Result};
pub use head::*;
pub use rope::*;

extern crate self as morok_model;

use std::path::Path;

use morok_dtype::DType;
use morok_ir::SInt;
use morok_macros::jit_wrapper;
use morok_tensor::{BoundVariable, Tensor};
use snafu::ResultExt;

use crate::audio::{MelConfig, MelSpectrogram};
use crate::state::{self, HasStateDict, StateDict};

use error::{ConfigIoSnafu, ConfigSnafu, StateSnafu, TensorSnafu};

pub enum SubsamplingMode {
    Conv1d,
    Conv2d,
}

pub enum ConvNormType {
    LayerNorm,
    BatchNorm,
}

pub struct GigaAmConfig {
    pub max_batch_size: usize,
    pub n_mels: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub conv_kernel: usize,
    pub subsampling_factor: usize,
    pub subsampling_mode: SubsamplingMode,
    pub subs_kernel_size: usize,
    pub conv_norm_type: ConvNormType,
    pub vocab_size: usize,
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub mel_center: bool,
    pub max_seq_len: usize,
}

impl GigaAmConfig {
    pub fn from_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).context(ConfigIoSnafu)?;
        let root: serde_json::Value = serde_json::from_str(&data).context(ConfigSnafu)?;
        let cfg = &root["cfg"]["model"]["cfg"];
        let pre = &cfg["preprocessor"];
        let enc = &cfg["encoder"];
        let head = &cfg["head"];

        let d_model = enc["d_model"].as_u64().expect("d_model") as usize;
        let ff_expansion_factor = enc["ff_expansion_factor"].as_u64().expect("ff_expansion_factor") as usize;

        let subsampling_str = enc["subsampling"].as_str().unwrap_or("conv2d");
        let subsampling_mode = match subsampling_str {
            "conv1d" => SubsamplingMode::Conv1d,
            _ => SubsamplingMode::Conv2d,
        };

        let conv_norm_str = enc["conv_norm_type"].as_str().unwrap_or("batch_norm");
        let conv_norm_type = match conv_norm_str {
            "layer_norm" => ConvNormType::LayerNorm,
            _ => ConvNormType::BatchNorm,
        };

        Ok(Self {
            max_batch_size: enc["max_batch_size"].as_u64().unwrap_or(32) as usize,
            n_mels: pre["features"].as_u64().expect("features") as usize,
            d_model,
            n_heads: enc["n_heads"].as_u64().expect("n_heads") as usize,
            n_layers: enc["n_layers"].as_u64().expect("n_layers") as usize,
            d_ff: d_model * ff_expansion_factor,
            conv_kernel: enc["conv_kernel_size"].as_u64().expect("conv_kernel_size") as usize,
            subsampling_factor: enc["subsampling_factor"].as_u64().expect("subsampling_factor") as usize,
            subsampling_mode,
            subs_kernel_size: enc["subs_kernel_size"].as_u64().unwrap_or(3) as usize,
            conv_norm_type,
            vocab_size: head["num_classes"].as_u64().expect("num_classes") as usize,
            sample_rate: pre["sample_rate"].as_u64().expect("sample_rate") as usize,
            n_fft: pre["n_fft"].as_u64().expect("n_fft") as usize,
            hop_length: pre["hop_length"].as_u64().expect("hop_length") as usize,
            win_length: pre["win_length"].as_u64().expect("win_length") as usize,
            mel_center: pre["center"].as_bool().unwrap_or(true),
            max_seq_len: enc["pos_emb_max_len"].as_u64().unwrap_or(5000) as usize,
        })
    }
}

/// GigaAM model: Conformer encoder + CTC head.
pub struct GigaAm {
    pub config: GigaAmConfig,
    pub mel: MelSpectrogram,
    pub subsampling: StridingSubsampling,
    pub layers: Vec<ConformerLayer>,
    pub cos_cache: Tensor,
    pub sin_cache: Tensor,
    pub head: CTCHead,
}

impl GigaAm {
    /// Load from a HuggingFace Hub repository.
    pub fn from_hub(model_id: &str) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main")
    }

    /// Load from a HuggingFace Hub repository at a specific branch/revision.
    pub fn from_hub_with_revision(model_id: &str, revision: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context(error::HubSnafu)?;
        let repo =
            api.repo(hf_hub::Repo::with_revision(model_id.to_string(), hf_hub::RepoType::Model, revision.to_string()));
        let config_path = repo.get("config.json").context(error::HubSnafu)?;
        let weights_path = repo.get("model.safetensors").context(error::HubSnafu)?;
        let config = GigaAmConfig::from_json(&config_path)?;
        Self::from_safetensors(&weights_path, config)
    }

    /// Load from a safetensors file with a config.json in the same directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.json");
        let weights_path = dir.join("model.safetensors");
        let config = GigaAmConfig::from_json(&config_path)?;
        Self::from_safetensors(&weights_path, config)
    }

    /// Load a GigaAM model from a safetensors file.
    pub fn from_safetensors(path: &Path, config: GigaAmConfig) -> Result<Self> {
        let sd = state::load_safetensors(path).context(StateSnafu)?;
        Self::from_state_dict(&sd, config)
    }

    /// Build from a pre-loaded state dict.
    ///
    /// Auto-detects PyTorch key format (keys starting with `encoder.` or `model.encoder.`) and remaps.
    pub fn from_state_dict(sd: &StateDict, config: GigaAmConfig) -> Result<Self> {
        let is_pytorch = sd.keys().any(|k| k.starts_with("encoder.") || k.starts_with("model.encoder."));
        let sd_owned = if is_pytorch { remap::remap_pytorch(sd.clone(), &config)? } else { sd.clone() };
        let sd = &sd_owned;
        let mel = MelSpectrogram::new(&MelConfig {
            sample_rate: config.sample_rate,
            n_fft: config.n_fft,
            hop_length: config.hop_length,
            win_length: config.win_length,
            n_mels: config.n_mels,
            center: config.mel_center,
        });
        let (cos_cache, sin_cache) = build_rope_cache(&config);

        let mut subsampling = StridingSubsampling::empty(&config);
        subsampling.load_state_dict(sd, "subsampling").context(StateSnafu)?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let mut layer = ConformerLayer::empty(&config);
            layer.load_state_dict(sd, &format!("layers.{i}")).context(StateSnafu)?;
            layers.push(layer);
        }

        let mut head = CTCHead::empty(&config);
        head.load_state_dict(sd, "head").context(StateSnafu)?;

        Ok(Self { config, mel, subsampling, layers, cos_cache, sin_cache, head })
    }

    pub fn with_random_weights(config: GigaAmConfig) -> Self {
        let mel = MelSpectrogram::new(&MelConfig {
            sample_rate: config.sample_rate,
            n_fft: config.n_fft,
            hop_length: config.hop_length,
            win_length: config.win_length,
            n_mels: config.n_mels,
            center: config.mel_center,
        });
        let (cos_cache, sin_cache) = build_rope_cache(&config);
        let subsampling = StridingSubsampling::empty(&config);
        let layers = (0..config.n_layers).map(|_| ConformerLayer::empty(&config)).collect();
        let head = CTCHead::empty(&config);
        Self { config, mel, subsampling, layers, cos_cache, sin_cache, head }
    }

    /// Run full inference: waveform -> CTC log-probabilities.
    ///
    /// Input: raw audio samples at 16kHz, mono, float32.
    /// Output: lazy tensor `[1, vocab_size, T/4]` of log-probabilities.
    pub fn forward(&self, waveform: &[f32], mel_tensor: &mut Tensor) -> Result<Tensor> {
        {
            let mut view = mel_tensor.array_view_mut::<f32>().context(TensorSnafu)?;
            self.mel.forward_into(waveform, &mut view);
        }
        let encoded = self.encode(mel_tensor)?;
        self.head.forward(&encoded)
    }

    /// Encoder-only: mel features -> encoded representation.
    ///
    /// Input: tensor `[B, n_mels, T]` from MelSpectrogram.
    /// Output: lazy tensor `[B, d_model, T/4]`.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let x = mel.try_transpose(-1, -2).context(TensorSnafu)?;
        let x = self.subsampling.forward(&x)?;

        let shape = x.shape().context(TensorSnafu)?;
        let batch = shape[0].clone();
        let seq_len = shape[1].clone();

        let d_half = self.config.d_model / self.config.n_heads / 2;

        let cos = self
            .cos_cache
            .try_shrink([
                (SInt::Const(0), seq_len.clone()),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(d_half)),
            ])
            .context(TensorSnafu)?;
        let sin = self
            .sin_cache
            .try_shrink([
                (SInt::Const(0), seq_len.clone()),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(d_half)),
            ])
            .context(TensorSnafu)?;

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin, None, None, batch.clone(), seq_len.clone())?;
        }

        x.try_transpose(-1, -2).context(TensorSnafu) // [B, d_model, T/4]
    }

    pub fn subsampling_output_length(&self, mel_frames: usize) -> usize {
        self.subsampling.output_length(mel_frames)
    }

    /// Batched encoder path with dynamic batch and mel-frame length.
    ///
    /// Input:
    /// - `mel`: `[B, n_mels, T_mel]`
    /// - `lengths`: `[B]` valid lengths in mel frames
    ///
    /// Output:
    /// - `[B, d_model, T_sub]` where `T_sub` is subsampled from `T_mel`.
    pub fn encode_batch(
        &self,
        mel: &Tensor,
        lengths: &Tensor,
        batch: &BoundVariable,
        seq_len: &BoundVariable,
    ) -> Result<Tensor> {
        let b = batch.as_sint();
        let t = seq_len.as_sint();

        let lengths = lengths.try_shrink([Some((SInt::Const(0), b.clone()))]).context(TensorSnafu)?;
        let lengths = lengths.cast(DType::Index).context(TensorSnafu)?;

        let two_t = Tensor::from_slice([2i64]).cast(DType::Index).context(TensorSnafu)?;
        let one_t = Tensor::from_slice([1i64]).cast(DType::Index).context(TensorSnafu)?;

        let mut lengths_sub = lengths;
        for _ in 0..2 {
            lengths_sub = lengths_sub.try_add(&one_t).context(TensorSnafu)?.try_div(&two_t).context(TensorSnafu)?;
        }

        let mel = mel
            .try_shrink([Some((SInt::Const(0), b.clone())), None, Some((SInt::Const(0), t.clone()))])
            .context(TensorSnafu)?;
        let x = mel.try_transpose(-1, -2).context(TensorSnafu)?;
        let x = self.subsampling.forward(&x)?;

        let shape = x.shape().context(TensorSnafu)?;
        let t_sub = shape[1].clone();

        let range = Tensor::arange(self.config.max_seq_len as i64, None, None).context(TensorSnafu)?;
        let range = range.cast(DType::Index).context(TensorSnafu)?;
        let range = range.try_shrink([(SInt::Const(0), t_sub.clone())]).context(TensorSnafu)?;
        let range = range.try_reshape([SInt::Const(1), t_sub.clone()]).context(TensorSnafu)?;
        let lens = lengths_sub;
        let lens = lens.try_reshape([b.clone(), SInt::Const(1)]).context(TensorSnafu)?;
        let pad_valid = range.try_lt(&lens).context(TensorSnafu)?;

        let pv1 = pad_valid.try_unsqueeze(1).context(TensorSnafu)?;
        let pv2 = pad_valid.try_unsqueeze(2).context(TensorSnafu)?;
        let att_mask = if batch.value() > 1 {
            let att_mask = pv1
                .bitwise_and(&pv2)
                .context(TensorSnafu)?
                .logical_not()
                .context(TensorSnafu)?
                .try_unsqueeze(1)
                .context(TensorSnafu)?;
            Some(
                att_mask
                    .try_expand([b.clone(), SInt::Const(self.config.n_heads), t_sub.clone(), t_sub.clone()])
                    .context(TensorSnafu)?,
            )
        } else {
            None
        };
        let pad_mask = pad_valid.logical_not().context(TensorSnafu)?;

        let d_half = self.config.d_model / self.config.n_heads / 2;
        let cos = self
            .cos_cache
            .try_shrink([
                (SInt::Const(0), t_sub.clone()),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(d_half)),
            ])
            .context(TensorSnafu)?;
        let sin = self
            .sin_cache
            .try_shrink([
                (SInt::Const(0), t_sub.clone()),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(d_half)),
            ])
            .context(TensorSnafu)?;

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin, att_mask.as_ref(), Some(&pad_mask), b.clone(), t_sub.clone())?;
        }

        x.try_transpose(-1, -2).context(TensorSnafu)
    }
}

jit_wrapper! {
    GigaAmJit(GigaAm) {
        mel: Tensor,

        build(mel) {
            let encoded = model.encode(mel)?;
            model.head.forward(&encoded)
        }
    }
}

jit_wrapper! {
    GigaAmBatchedJit(GigaAm) {
        mel: Tensor,
        lengths: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
            t: (1, model.config.max_seq_len),
        }

        build(mel, lengths, b, t) {
            let encoded = model.encode_batch(mel, lengths, &b, &t)?;
            model.head.forward(&encoded)
        }
    }
}
