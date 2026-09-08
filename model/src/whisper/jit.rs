//! JIT wrappers for Whisper.
//!
//! Every wrapper is prepared at a concrete capacity. Recognition projects
//! encoder features into cross-attention caches once, then reuses them for
//! language detection, token prefill, fixed-slot decoder steps, and alignment.
#![allow(clippy::too_many_arguments)]

use svod_macros::jit_wrapper;

use super::model::Whisper;

#[derive(Clone)]
pub struct WhisperAlignmentModel {
    model: Whisper,
    alignment_heads: Vec<(usize, usize)>,
}

impl WhisperAlignmentModel {
    pub fn new(model: Whisper, alignment_heads: Vec<(usize, usize)>) -> Self {
        Self { model, alignment_heads }
    }

    fn forward(
        &self,
        cross_k: &svod_tensor::Tensor,
        cross_v: &svod_tensor::Tensor,
        tokens: &svod_tensor::Tensor,
    ) -> super::error::Result<svod_tensor::Tensor> {
        self.model.align_with_cross_kv(tokens, cross_k, cross_v, &self.alignment_heads)
    }
}

// Encoder-only JIT: mel `[B, n_mels, T]` → `[B, T/2, D]`.
jit_wrapper! {
    WhisperEncoderJit(Whisper) {
        mel: Tensor,

        build(mel) {
            model.encode(mel)
        }
    }
}

// Encoder-feature projection JIT. Its concrete packed outputs are reused by
// every prefill attempt and decoder step for one audio window.
jit_wrapper! {
    WhisperCrossKvJit(Whisper) {
        audio_features: Tensor,

        outputs { cross_k, cross_v }

        build(audio_features) {
            model.project_cross_kv(audio_features)
        }
    }
}

// Static teacher-forced alignment replay. The graph shape and selected heads
// are fixed at construction; valid token/audio lengths are host metadata.
jit_wrapper! {
    WhisperAlignmentJit(WhisperAlignmentModel) {
        cross_k: Tensor,
        cross_v: Tensor,
        tokens: Tensor,

        build(cross_k, cross_v, tokens) {
            model.forward(cross_k, cross_v, tokens)
        }
    }
}

// Decoder-only JIT for language detection: prepared cross K/V + tokens → logits.
// This remains separate from prefill and step because it is prepared for one SOT
// token and returns exactly one vocabulary row.
jit_wrapper! {
    WhisperDecoderJit(Whisper) {
        prepared_cross_k: Tensor,
        prepared_cross_v: Tensor,
        tokens: Tensor,

        build(prepared_cross_k, prepared_cross_v, tokens) {
            model.decode_with_cross_kv(tokens, prepared_cross_k, prepared_cross_v)
        }
    }
}

// Prefill JIT: initial tokens [1, init_len] + prepared cross K/V → logits + caches.
// Outputs: logits [1, init_len, n_vocab] and packed self K/V. Prepared cross
// K/V remain in the device-local inputs and are also bound into decoder steps.
// Compiled once at fixed init_len; the plan owns all buffers, reused per window.
// No realize() needed — logits read via copyout, K/V copied to step JIT caches.
jit_wrapper! {
    WhisperPrefillJit(Whisper) {
        tokens: Tensor,
        prepared_cross_k: Tensor,
        prepared_cross_v: Tensor,

        outputs { logits, self_k, self_v }

        build(tokens, prepared_cross_k, prepared_cross_v) {
            model.decode_prefill(tokens, prepared_cross_k, prepared_cross_v, 0)
        }
    }
}

// KV-cached decoder step JIT: single-token forward with K/V cache recycling.
// Inputs: token [B,1], pos_emb [B,1,D], self/cross K/V caches, self key lengths [B].
// Outputs: logits [B,n_vocab], new_self_k [B,1,n_layer*H,Dh], new_self_v [...].
// After execute: copy_output_to_self_k_cache/v_cache to append new K/V at pos.
jit_wrapper! {
    WhisperDecoderStepJit(Whisper) {
        token: Tensor,
        pos_emb: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        self_key_lens: Tensor,

        outputs { logits, new_self_k, new_self_v }

        build(token, pos_emb, self_k_cache, self_v_cache, cross_k, cross_v, self_key_lens) {
            model.decode_step(token, pos_emb, self_k_cache, self_v_cache, cross_k, cross_v, self_key_lens)
        }
    }
}
