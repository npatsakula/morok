//! Whisper model composite: encoder + decoder + dimensions.

use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::state::scoped;

use super::config::ModelDimensions;
use super::decoder::TextDecoder;
use super::encoder::AudioEncoder;
use super::error::Result;

/// The Whisper model: encoder + decoder + alignment heads.
#[derive(Clone, Module)]
pub struct Whisper {
    #[module(skip)]
    pub dims: ModelDimensions,
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
}

impl Whisper {
    pub fn empty(dims: ModelDimensions) -> Self {
        Self { encoder: AudioEncoder::empty(&dims), decoder: TextDecoder::empty(&dims), dims }
    }

    /// Encode mel spectrogram → audio features `[B, n_audio_ctx, D]`.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        scoped("encoder", || self.encoder.forward(mel))
    }

    /// Decode tokens given audio features → logits `[B, L, n_vocab]`.
    pub fn decode(&self, tokens: &Tensor, audio_features: &Tensor, offset: usize) -> Result<Tensor> {
        scoped("decoder", || self.decoder.forward(tokens, audio_features, offset))
    }

    /// Teacher-forced alignment using retained packed cross-attention K/V.
    pub fn align_with_cross_kv(
        &self,
        tokens: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        alignment_heads: &[(usize, usize)],
    ) -> Result<Tensor> {
        scoped("decoder", || self.decoder.forward_alignment(tokens, cross_k, cross_v, alignment_heads))
    }

    /// Project encoder features into packed cross-attention K/V once per window.
    pub fn project_cross_kv(&self, audio_features: &Tensor) -> Result<(Tensor, Tensor)> {
        scoped("decoder", || self.decoder.project_cross_kv(audio_features))
    }

    /// Decode logits using packed cross-attention K/V.
    pub fn decode_with_cross_kv(&self, tokens: &Tensor, cross_k: &Tensor, cross_v: &Tensor) -> Result<Tensor> {
        scoped("decoder", || self.decoder.forward_with_cross_kv(tokens, cross_k, cross_v, 0))
    }

    /// Prefill: initial tokens → logits + packed K/V caches.
    pub fn decode_prefill(
        &self,
        tokens: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        scoped("decoder", || self.decoder.forward_prefill(tokens, cross_k, cross_v, offset))
    }

    /// Single-token step with KV cache → (logits, new_self_k, new_self_v).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step(
        &self,
        token: &Tensor,
        pos_emb: &Tensor,
        self_k_cache: &Tensor,
        self_v_cache: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        self_key_lens: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        scoped("decoder", || {
            self.decoder.forward_step(token, pos_emb, self_k_cache, self_v_cache, cross_k, cross_v, self_key_lens)
        })
    }

    /// Full forward: encode + decode.
    pub fn forward(&self, mel: &Tensor, tokens: &Tensor) -> Result<Tensor> {
        let audio_features = self.encode(mel)?;
        self.decode(tokens, &audio_features, 0)
    }

    pub fn is_multilingual(&self) -> bool {
        self.dims.is_multilingual()
    }
}
