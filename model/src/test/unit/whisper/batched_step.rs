//! Fixed-capacity decoder-step graph tests.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::whisper::decoder::{StepAttentionMode, cached_step_mask};
use crate::whisper::{ModelDimensions, Whisper, WhisperSize};

/// Tiny config so the CPU JIT graph compiles in seconds. `n_text_ctx` is kept
/// small (8) to shrink the self-attention buffers — the step JIT only needs
/// one position of cache populated for this test.
fn tiny_dims() -> ModelDimensions {
    // Start from WhisperSize::Tiny's structural dims, but shrink the text
    // context and vocab so the compile graph is minimal. The step JIT's cache
    // buffers scale with n_text_ctx.
    let mut dims = ModelDimensions::for_size(WhisperSize::Tiny);
    dims.n_text_ctx = 8;
    dims.n_vocab = 64;
    dims
}

#[test]
fn forward_step_fixed_batch_keeps_batch_concrete() {
    let dims = tiny_dims();
    let model = Whisper::empty(dims.clone());
    let (batch, n_audio_ctx) = (2usize, 8usize);
    let d_head = dims.n_text_state / dims.n_text_head;
    let layer_heads = dims.n_text_layer * dims.n_text_head;
    let token = Tensor::zeros(&[batch, 1], DType::Int32);
    let pos_emb = Tensor::zeros(&[batch, 1, dims.n_text_state], DType::Float32);
    let self_k = Tensor::zeros(&[batch, dims.n_text_ctx, layer_heads, d_head], DType::Float32);
    let self_v = Tensor::zeros(&[batch, dims.n_text_ctx, layer_heads, d_head], DType::Float32);
    let cross_k = Tensor::zeros(&[batch, n_audio_ctx, layer_heads, d_head], DType::Float32);
    let cross_v = Tensor::zeros(&[batch, n_audio_ctx, layer_heads, d_head], DType::Float32);
    let key_lens = Tensor::zeros(&[batch], DType::Int32);

    let (logits, new_k, new_v) =
        model.decode_step(&token, &pos_emb, &self_k, &self_v, &cross_k, &cross_v, &key_lens).unwrap();
    assert_eq!(logits.dim_const(0).unwrap(), batch);
    assert_eq!(new_k.dim_const(0).unwrap(), batch);
    assert_eq!(new_v.dim_const(0).unwrap(), batch);
    logits.realize().unwrap();
    assert!(logits.as_vec::<f32>().unwrap().into_iter().all(f32::is_finite));
}

#[test]
fn cached_step_key_lengths_mask_only_prefix_and_appended_key() {
    let key_lens = Tensor::from_slice([0i32, 3]);
    let mask = cached_step_mask(&key_lens, 2, 6).unwrap();
    assert_eq!(mask.dims().unwrap(), [2, 1, 1, 6]);
    mask.realize().unwrap();
    assert_eq!(
        mask.as_vec::<bool>().unwrap(),
        [true, true, true, true, true, false, false, false, false, true, true, false]
    );
}

#[test]
#[ignore = "GPU: custom single-query self/cross attention vs generic SDPA on a supported AMD device"]
fn decoder_step_attention_modes_match_generic_gpu_sdpa() {
    let Some(arch) = svod_tensor::config::amd_test_arch() else {
        eprintln!("skip: no AMD device");
        return;
    };
    if !matches!(arch, svod_dtype::AmdArch::Gfx942 | svod_dtype::AmdArch::Gfx1151) {
        eprintln!("skip: {arch:?} is not supported by single-query attention");
        return;
    }

    let mut dims = tiny_dims();
    // Production cross-attention length: split=4 creates 375-key chunks, which
    // also exercises the partial kernel's ragged subgroup tile.
    dims.n_audio_ctx = 1500;
    dims.n_text_ctx = 7;
    dims.dtype = DType::Float32;
    let model = Whisper::empty(dims.clone());
    let (batch, d_head) = (2, dims.n_text_state / dims.n_text_head);
    let layer_heads = dims.n_text_layer * dims.n_text_head;
    let token = Tensor::from_slice([1i32, 2]).try_reshape([batch, 1]).unwrap();
    let pos_emb = Tensor::randn(&[batch, 1, dims.n_text_state]).unwrap();
    let self_k = Tensor::randn(&[batch, dims.n_text_ctx, layer_heads, d_head]).unwrap();
    let self_v = Tensor::randn(&[batch, dims.n_text_ctx, layer_heads, d_head]).unwrap();
    let cross_k = Tensor::randn(&[batch, dims.n_audio_ctx, layer_heads, d_head]).unwrap();
    let cross_v = Tensor::randn(&[batch, dims.n_audio_ctx, layer_heads, d_head]).unwrap();
    let key_lens = Tensor::from_slice([2i32, 5]);

    let outputs = [
        StepAttentionMode::Generic,
        StepAttentionMode::CustomSelf,
        StepAttentionMode::CustomCross { split: 1 },
        StepAttentionMode::CustomCross { split: 4 },
        StepAttentionMode::CustomBoth { split: 1 },
        StepAttentionMode::CustomBoth { split: 4 },
    ]
    .map(|mode| {
        model
            .decoder
            .forward_step_with_attention_mode(&token, &pos_emb, &self_k, &self_v, &cross_k, &cross_v, &key_lens, mode)
            .unwrap()
            .0
    });
    Tensor::realize_batch(outputs.iter()).unwrap();
    let reference = outputs[0].as_vec::<f32>().unwrap();
    for (mode, output) in [
        StepAttentionMode::CustomSelf,
        StepAttentionMode::CustomCross { split: 1 },
        StepAttentionMode::CustomCross { split: 4 },
        StepAttentionMode::CustomBoth { split: 1 },
        StepAttentionMode::CustomBoth { split: 4 },
    ]
    .into_iter()
    .zip(&outputs[1..])
    {
        let got = output.as_vec::<f32>().unwrap();
        let max_abs = got.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_abs < 3e-4, "{mode:?} logits differ from generic SDPA by {max_abs:e}");
    }
}
