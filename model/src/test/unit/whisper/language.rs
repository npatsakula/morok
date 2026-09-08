//! Language detection unit tests.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::whisper::{ModelDimensions, Whisper, WhisperSize};

#[test]
fn encoder_forward_for_detection() {
    // Verify a single-position decoder forward (as used by detect_language)
    // produces logits of the right shape.
    let dims = ModelDimensions::for_size(WhisperSize::Tiny);
    let model = Whisper::empty(dims.clone());

    let mel = Tensor::zeros(&[1, dims.n_mels, 3000], DType::Float32);
    let features = model.encode(&mel).unwrap();

    // Single SOT token, like detect_language does
    let sot = 50258i32; // <|startoftranscript|> for multilingual
    let tokens = Tensor::from_slice([sot]).try_reshape([1usize, 1]).unwrap().cast(DType::Int32);

    let logits = model.decode(&tokens, &features, 0).unwrap();
    let shape = logits.shape().unwrap();
    assert_eq!(shape[0].as_const(), Some(1));
    assert_eq!(shape[1].as_const(), Some(1));
    assert_eq!(shape[2].as_const(), Some(dims.n_vocab));
}

#[test]
fn dims_language_table() {
    // Multilingual models (non-.en) should have language tokens
    let tiny = ModelDimensions::for_size(WhisperSize::Tiny);
    assert!(tiny.is_multilingual());
    assert!(tiny.num_languages() > 0);

    let tiny_en = ModelDimensions::for_size(WhisperSize::TinyEn);
    assert!(!tiny_en.is_multilingual());
}
