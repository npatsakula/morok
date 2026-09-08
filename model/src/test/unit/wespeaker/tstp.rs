//! Hand-checked smoke tests for WeSpeaker's TSTP / StatsPool head.
//!
//! Reference: `pyannote.audio.models.blocks.pooling.StatsPool._pool` —
//! weighted mean and unbiased (Bessel-corrected) std with epsilon `1e-8`
//! applied to both the `weights.sum` denominator and the variance denominator
//! `v1 - v2 / v1 + eps`.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::wespeaker::WeSpeakerResNet34;

/// Build the forward graph (without realizing) and confirm the output shape
/// is `[B, 256]`. Cheap — only walks the symbolic shape pipeline through
/// permute → unsqueeze → stem → 4 stages → TSTP → linear; no kernel compile.
#[test]
fn forward_zero_weights_shape() {
    let model = WeSpeakerResNet34::with_zero_weights(crate::wespeaker::WeSpeakerConfig::new().with_max_batch_size(1));

    let feats = Tensor::zeros(&[1, 1598, 80], DType::Float32).unwrap();
    let weights = Tensor::ones(&[1, 799], DType::Float32).unwrap();

    let var = svod_tensor::Variable::new("b", 1, 1);
    let b = var.bind(1).unwrap();

    let out = model.forward(&feats, &weights, &b).unwrap();

    let shape = crate::test::max_dims(&out);
    assert_eq!(shape, vec![1, 256]);
}

/// Same as above but materialises the graph through the CPU JIT. Gated behind
/// `--ignored` because compiling the full 32-conv ResNet34 graph over the
/// `1×80×1598` spectrogram input is several minutes of kernel work.
#[test]
#[ignore = "heavy: full WeSpeaker ResNet34 graph compile through the CPU backend"]
fn forward_zero_weights_realize() {
    let model = WeSpeakerResNet34::with_zero_weights(crate::wespeaker::WeSpeakerConfig::new().with_max_batch_size(1));

    let feats = Tensor::zeros(&[1, 1598, 80], DType::Float32).unwrap();
    let weights = Tensor::ones(&[1, 799], DType::Float32).unwrap();

    let var = svod_tensor::Variable::new("b", 1, 1);
    let b = var.bind(1).unwrap();

    let out = model.forward(&feats, &weights, &b).unwrap();
    out.realize().unwrap();

    let shape = crate::test::max_dims(&out);
    assert_eq!(shape, vec![1, 256]);
}
