use std::path::Path;

use crate::gigaam::{ConvNormType, GigaAmConfig, SubsamplingMode};

#[test]
fn test_config_from_json() {
    let config = GigaAmConfig::from_json(Path::new("tests/gigaam/ctc_config.json")).unwrap();

    assert_eq!(config.n_mels, 64);
    assert_eq!(config.max_batch_size, 32);
    assert_eq!(config.d_model, 768);
    assert_eq!(config.n_heads, 16);
    assert_eq!(config.n_layers, 16);
    assert_eq!(config.d_ff, 3072);
    assert_eq!(config.conv_kernel, 5);
    assert_eq!(config.vocab_size, 34);
    assert_eq!(config.sample_rate, 16000);
    assert_eq!(config.n_fft, 320);
    assert_eq!(config.hop_length, 160);
    assert_eq!(config.win_length, 320);
    assert!(!config.mel_center);
    assert!(matches!(config.subsampling_mode, SubsamplingMode::Conv1d));
    assert!(matches!(config.conv_norm_type, ConvNormType::LayerNorm));
    assert_eq!(config.subs_kernel_size, 5);
    assert_eq!(config.subsampling_factor, 4);
    assert_eq!(config.max_encoder_frames, 5000);
    // No explicit `max_mel_frames` / `max_seq_len` in this config — the loader
    // derives the pre-subsampling bound from the post-subsampling encoder bound
    // by scaling up by `subsampling_factor`, so audio approaching the encoder
    // cap isn't rejected at the JIT input stage.
    assert_eq!(config.max_mel_frames, 5000 * config.subsampling_factor);
}
