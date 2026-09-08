//! Hand-checked smoke tests for WeSpeaker's TSTP / StatsPool head.
//!
//! Reference: `pyannote.audio.models.blocks.pooling.StatsPool._pool` —
//! weighted mean and unbiased (Bessel-corrected) std with epsilon `1e-8`
//! applied to both the `weights.sum` denominator and the variance denominator
//! `v1 - v2 / v1 + eps`.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{CoordinateTransformMode, NearestMode, ResizeMode};

use crate::wespeaker::WeSpeakerResNet34;

/// Build the forward graph (without realizing) and confirm the output shape
/// is `[B, 256]`. Cheap — only walks the symbolic shape pipeline through
/// permute → unsqueeze → stem → 4 stages → TSTP → linear; no kernel compile.
#[test]
fn forward_zero_weights_shape() {
    let model = WeSpeakerResNet34::with_zero_weights(crate::wespeaker::WeSpeakerConfig::new().with_max_batch_size(1));

    let feats = Tensor::zeros(&[1, 1598, 80], DType::Float32);
    let weights = Tensor::ones(&[1, 799], DType::Float32);

    let out = model.forward(&feats, &weights).unwrap();

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

    let feats = Tensor::zeros(&[1, 1598, 80], DType::Float32);
    let weights = Tensor::ones(&[1, 799], DType::Float32);

    let out = model.forward(&feats, &weights).unwrap();
    out.realize().unwrap();

    let shape = crate::test::max_dims(&out);
    assert_eq!(shape, vec![1, 256]);
}

/// The nearest-mode resample the TSTP head runs on the attention weights must
/// reproduce `F.interpolate(..., mode="nearest")` index-for-index — an
/// asymmetric coordinate transform with floor rounding, i.e. the one-hot map
/// `src = floor(o * T_in / T_out)` this head used to build by hand.
#[test]
fn nearest_resample_matches_floor_index_map() {
    let (t_in, t_out) = (799usize, 200usize);
    let src: Vec<f32> = (0..t_in).map(|i| i as f32).collect();
    let weights = Tensor::from_slice(&src).try_reshape([1, t_in as isize]).unwrap();

    let resampled = weights
        .resize()
        .axes(&[1])
        .sizes(&[t_out])
        .mode(ResizeMode::Nearest)
        .nearest_mode(NearestMode::Floor)
        .coordinate_transformation_mode(CoordinateTransformMode::Asymmetric)
        .call()
        .unwrap();

    // The ramp carries its own source index, so the output *is* the index map.
    let expected: Vec<f32> = (0..t_out).map(|o| ((o * t_in) / t_out) as f32).collect();
    assert_eq!(resampled.to_vec::<f32>().unwrap(), expected);
}
