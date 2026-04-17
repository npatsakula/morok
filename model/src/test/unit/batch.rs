use morok_dtype::DType;
use morok_tensor::Tensor;

use crate::gigaam::{ConvNormType, GigaAm, GigaAmBatchedJit, GigaAmConfig, SubsamplingMode};

fn test_config() -> GigaAmConfig {
    GigaAmConfig {
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
        max_seq_len: 5000,
    }
}

fn model_with_random_weights() -> GigaAm {
    GigaAm::with_random_weights(test_config())
}

#[test]
fn test_output_length_matches_forward() {
    let model = GigaAm::with_random_weights(test_config());
    let d = test_config().d_model;

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
#[ignore = "JIT wrapper buffer ID mismatch after prepare_batch normalization — needs macro fix"]
fn test_batched_jit_prepare_and_execute() {
    let model = model_with_random_weights();
    let mut jit = GigaAmBatchedJit::new(model);

    let (b, t, d) = (2, 10, 32);
    let mut features = Tensor::full(&[b, t, d], 0.5f32, DType::Float32).unwrap();
    features.realize().unwrap();
    let lengths = Tensor::from_slice(&[10i32, 8]);

    jit.prepare(&features, &lengths).unwrap();
    jit.execute().unwrap();

    let output = jit.output().unwrap();
    assert!(output.size() > 0);
}

#[test]
fn test_single_vs_batch_consistency() {
    let model = model_with_random_weights();
    let d = 32;
    let t = 10;

    let x1 = Tensor::full(&[1, t, d], 0.5f32, DType::Float32).unwrap();
    let x2 = Tensor::full(&[1, t, d], 0.3f32, DType::Float32).unwrap();
    let lengths_single = Tensor::from_slice(&[t as i32]);

    let mut out1 = model.encode_batch(&x1, &lengths_single).unwrap();
    out1.realize().unwrap();
    let data1 = out1.as_vec::<f32>().unwrap();

    let mut out2 = model.encode_batch(&x2, &lengths_single).unwrap();
    out2.realize().unwrap();
    let data2 = out2.as_vec::<f32>().unwrap();

    let batch = {
        let mut x1r = x1.clone();
        x1r.realize().unwrap();
        let d1 = x1r.as_vec::<f32>().unwrap();
        let mut x2r = x2.clone();
        x2r.realize().unwrap();
        let d2 = x2r.as_vec::<f32>().unwrap();
        let mut batch_data = vec![0.0f32; 2 * t * d];
        batch_data[..t * d].copy_from_slice(&d1);
        batch_data[t * d..].copy_from_slice(&d2);
        ndarray::Array3::from_shape_vec((2, t, d), batch_data).unwrap()
    };
    let batch_tensor = Tensor::from_ndarray(&batch);
    let batch_lengths = Tensor::from_slice(&[t as i32, t as i32]);

    let mut batch_out = model.encode_batch(&batch_tensor, &batch_lengths).unwrap();
    batch_out.realize().unwrap();
    let batch_data = batch_out.as_vec::<f32>().unwrap();

    assert_eq!(data1.len() * 2, batch_data.len());

    for (i, (&b, &s)) in batch_data[..data1.len()].iter().zip(data1.iter()).enumerate() {
        assert!((b - s).abs() < 1e-4, "batch[0] mismatch at {}: batch={} single={}", i, b, s);
    }
    for (i, (&b, &s)) in batch_data[data1.len()..].iter().zip(data2.iter()).enumerate() {
        assert!((b - s).abs() < 1e-4, "batch[1] mismatch at {}: batch={} single={}", i, b, s);
    }
}
