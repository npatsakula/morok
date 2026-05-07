use morok_arch::ctc::{CtcDecoder, GreedyDecoder};
use morok_dtype::DType;
use morok_tensor::{Tensor, Variable};

use crate::gigaam::{ConvNormType, GigaAm, GigaAmBatchedJit, GigaAmConfig, SubsamplingMode};

fn test_config() -> GigaAmConfig {
    GigaAmConfig {
        max_batch_size: 8,
        n_mels: 64,
        d_model: 32,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
        conv_kernel: 5,
        subsampling_factor: 4,
        subsampling_mode: SubsamplingMode::Conv1d,
        subs_kernel_size: 5,
        conv_norm_type: ConvNormType::LayerNorm,
        vocab_size: 34,
        sample_rate: 16000,
        n_fft: 320,
        hop_length: 160,
        win_length: 320,
        mel_center: false,
        max_mel_frames: 512,
        max_encoder_frames: 128,
        decoder: CtcDecoder::Greedy(GreedyDecoder::new(Vec::new())),
    }
}

fn model_with_random_weights() -> GigaAm {
    GigaAm::with_random_weights(test_config())
}

fn read_prefix_f32(t: &Tensor, len: usize) -> Vec<f32> {
    let buf = t.buffer().unwrap();
    buf.as_array::<f32>().unwrap().as_slice().unwrap()[..len].to_vec()
}

#[test]
fn test_output_length_matches_forward() {
    let model = GigaAm::with_random_weights(test_config());

    let x = Tensor::full(&[1, 100, 64], 0.0f32, DType::Float32).unwrap();
    let out = model.subsampling.forward(&x).unwrap();
    let actual_t = out.shape().unwrap()[1].as_const().unwrap();
    assert_eq!(model.subsampling_output_length(100), actual_t);

    let x2 = Tensor::full(&[1, 50, 64], 0.0f32, DType::Float32).unwrap();
    let out2 = model.subsampling.forward(&x2).unwrap();
    let actual_t2 = out2.shape().unwrap()[1].as_const().unwrap();
    assert_eq!(model.subsampling_output_length(50), actual_t2);
}

#[test]
fn test_batched_jit_prepare_and_execute() {
    let model = model_with_random_weights();
    let mut jit = GigaAmBatchedJit::new(model);

    let (b, t, n_mels) = (2, 10, test_config().n_mels);
    let mut mel = Tensor::full(&[b, n_mels, t], 0.5f32, DType::Float32).unwrap();
    mel.realize().unwrap();
    let lengths = Tensor::from_slice([10i32, 8]);

    jit.prepare(&mel, &lengths).unwrap();
    jit.execute_with_vars(&[("b", b as i64), ("t", t as i64)]).unwrap();

    let output = jit.output().unwrap();
    assert!(output.size() > 0);
}

#[test]
fn test_batched_jit_prepare_large_shape() {
    let model = model_with_random_weights();
    let mut jit = GigaAmBatchedJit::new(model);

    let cfg = test_config();
    let mut mel = Tensor::full(&[cfg.max_batch_size, cfg.n_mels, cfg.max_mel_frames], 0.0f32, DType::Float32).unwrap();
    mel.realize().unwrap();
    let lengths = Tensor::from_slice(vec![cfg.max_mel_frames as i32; cfg.max_batch_size]);

    jit.prepare(&mel, &lengths).unwrap();
    jit.execute_with_vars(&[("b", cfg.max_batch_size as i64), ("t", cfg.max_mel_frames as i64)]).unwrap();

    let output = jit.output().unwrap();
    assert!(output.size() > 0);
}

#[test]
fn test_batched_jit_t_bound_is_mel_frames() {
    let model = model_with_random_weights();
    let cfg = test_config();
    assert!(cfg.max_mel_frames > cfg.max_encoder_frames);

    let mut jit = GigaAmBatchedJit::new(model);
    let mut mel = Tensor::full(&[1, cfg.n_mels, cfg.max_mel_frames], 0.0f32, DType::Float32).unwrap();
    mel.realize().unwrap();
    let lengths = Tensor::from_slice([cfg.max_mel_frames as i32]);

    jit.prepare(&mel, &lengths).unwrap();
    jit.execute_with_vars(&[("b", 1), ("t", cfg.max_mel_frames as i64)]).unwrap();
}

#[test]
fn test_batched_jit_rejects_t_above_max_mel_frames() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let mut jit = GigaAmBatchedJit::new(model);
    let mut mel = Tensor::full(&[1, cfg.n_mels, cfg.max_mel_frames], 0.0f32, DType::Float32).unwrap();
    mel.realize().unwrap();
    let lengths = Tensor::from_slice([cfg.max_mel_frames as i32]);

    jit.prepare(&mel, &lengths).unwrap();
    let err = jit.execute_with_vars(&[("b", 1), ("t", cfg.max_mel_frames as i64 + 1)]).unwrap_err();
    match err {
        crate::jit::JitError::Runtime { source: morok_runtime::Error::Execution { reason } } => {
            assert!(reason.contains("outside bounds"), "unexpected runtime error: {reason}");
        }
        other => panic!("expected runtime bounds error, got {other:?}"),
    }
}

#[test]
fn test_rope_cache_uses_encoder_bound() {
    let model = model_with_random_weights();
    let cfg = test_config();

    assert_eq!(model.cos_cache.shape().unwrap()[0].as_const().unwrap(), cfg.max_encoder_frames);
    assert_eq!(model.sin_cache.shape().unwrap()[0].as_const().unwrap(), cfg.max_encoder_frames);
    assert_ne!(cfg.max_encoder_frames, cfg.max_mel_frames);
}

#[test]
fn test_encode_batch_near_max_mel_stays_within_encoder_bound() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let t = cfg.max_mel_frames;
    let t_sub = model.subsampling_output_length(t);
    assert!(t_sub <= cfg.max_encoder_frames);

    let x = Tensor::full(&[1, cfg.n_mels, t], 0.1f32, DType::Float32).unwrap();
    let lengths = Tensor::from_slice([t as i32]);
    let b_var = Variable::new("B", 1, cfg.max_batch_size as i64);
    let t_var = Variable::new("T", 1, cfg.max_mel_frames as i64);
    let b1 = b_var.bind(1).unwrap();
    let t_bound = t_var.bind(t as i64).unwrap();

    let mut out = model.encode_batch(&x, &lengths, &b1, &t_bound).unwrap();
    out.realize().unwrap();
    assert!(out.buffer().unwrap().size() > 0);
}

#[test]
fn test_single_vs_batch_consistency() {
    let model = model_with_random_weights();
    let d = test_config().d_model;
    let n_mels = test_config().n_mels;
    let t = 10;
    let t_sub = model.subsampling_output_length(t);

    let x1 = Tensor::full(&[1, n_mels, t], 0.5f32, DType::Float32).unwrap();
    let x2 = Tensor::full(&[1, n_mels, t], 0.3f32, DType::Float32).unwrap();
    let lengths_single = Tensor::from_slice([t as i32]);

    let b_var = Variable::new("B", 1, test_config().max_batch_size as i64);
    let t_var = Variable::new("T", 1, test_config().max_mel_frames as i64);
    let b1 = b_var.bind(1).unwrap();
    let t1 = t_var.bind(t as i64).unwrap();

    let mut out1 = model.encode_batch(&x1, &lengths_single, &b1, &t1).unwrap();
    out1.realize().unwrap();
    let data1 = read_prefix_f32(&out1, d * t_sub);

    let mut out2 = model.encode_batch(&x2, &lengths_single, &b1, &t1).unwrap();
    out2.realize().unwrap();
    let data2 = read_prefix_f32(&out2, d * t_sub);

    let batch = {
        let mut x1r = x1.clone();
        x1r.realize().unwrap();
        let d1 = x1r.as_vec::<f32>().unwrap();
        let mut x2r = x2.clone();
        x2r.realize().unwrap();
        let d2 = x2r.as_vec::<f32>().unwrap();
        let mut batch_data = vec![0.0f32; 2 * n_mels * t];
        batch_data[..n_mels * t].copy_from_slice(&d1);
        batch_data[n_mels * t..].copy_from_slice(&d2);
        ndarray::Array3::from_shape_vec((2, n_mels, t), batch_data).unwrap()
    };
    let batch_tensor = Tensor::from_ndarray(&batch);
    let batch_lengths = Tensor::from_slice([t as i32, t as i32]);

    let b2 = b_var.bind(2).unwrap();
    let mut batch_out = model.encode_batch(&batch_tensor, &batch_lengths, &b2, &t1).unwrap();
    batch_out.realize().unwrap();
    let batch_data = read_prefix_f32(&batch_out, 2 * d * t_sub);

    assert_eq!(data1.len() * 2, batch_data.len());

    for (i, (&b, &s)) in batch_data[..data1.len()].iter().zip(data1.iter()).enumerate() {
        assert!((b - s).abs() < 1e-4, "batch[0] mismatch at {}: batch={} single={}", i, b, s);
    }
    for (i, (&b, &s)) in batch_data[data1.len()..].iter().zip(data2.iter()).enumerate() {
        assert!((b - s).abs() < 1e-4, "batch[1] mismatch at {}: batch={} single={}", i, b, s);
    }
}

#[test]
fn test_encode_batch_full_lengths_finite() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let t = 256usize;

    let x = Tensor::full(&[2, cfg.n_mels, t], 0.1f32, DType::Float32).unwrap();
    let lengths = Tensor::from_slice([t as i32, t as i32]);

    let b_var = Variable::new("B", 1, cfg.max_batch_size as i64);
    let t_var = Variable::new("T", 1, cfg.max_mel_frames as i64);
    let b2 = b_var.bind(2).unwrap();
    let t_bound = t_var.bind(t as i64).unwrap();

    let mut out = model.encode_batch(&x, &lengths, &b2, &t_bound).unwrap();
    out.realize().unwrap();

    let buf = out.buffer().unwrap();
    let data = buf.as_array::<f32>().unwrap();
    for v in data.as_slice().unwrap() {
        assert!(v.is_finite(), "encode_batch produced non-finite value: {v}");
    }
}

#[test]
fn test_encode_batch_respects_dynamic_seq_len() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let t_dynamic = 64usize;

    let mut jit = GigaAmBatchedJit::new(model);
    let mut mel = Tensor::full(&[cfg.max_batch_size, cfg.n_mels, cfg.max_mel_frames], 0.0f32, DType::Float32).unwrap();
    mel.realize().unwrap();
    let lengths = Tensor::from_slice(vec![cfg.max_mel_frames as i32; cfg.max_batch_size]);

    jit.prepare(&mel, &lengths).unwrap();
    let profiles = jit.execute_with_vars_profiled(&[("b", 1), ("t", t_dynamic as i64)]).unwrap();

    assert!(!profiles.is_empty(), "expected kernels for profiled dynamic execute");
    assert!(
        profiles.iter().any(|p| p.kernel.var_names.iter().any(|name| name == "t")),
        "expected at least one kernel to keep dynamic seq var 't'"
    );
}
