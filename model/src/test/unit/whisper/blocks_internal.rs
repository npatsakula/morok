//! Whisper block internals: FP16 LayerNorm and Linear must keep their affine
//! and bias epilogues in FP32 and round only once, at the final cast.

use crate::whisper::blocks::{LayerNormWeights, linear_with_bias};
use svod_dtype::DType;
use svod_tensor::Tensor;

fn realized_f32(tensor: Tensor) -> Vec<f32> {
    let tensor = tensor.cast(DType::Float32).unwrap();
    tensor.realize().unwrap();
    tensor.as_vec::<f32>().unwrap()
}

fn fp16(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(values).try_reshape(shape.to_vec()).unwrap().cast(DType::Float16).unwrap()
}

/// `legacy` is the all-FP16 formulation the epilogue replaced: it must differ,
/// or the fixture would pass without proving anything.
fn assert_rounds_once(actual: Tensor, reference: Tensor, legacy: Tensor, epilogue: &str) {
    let reference = realized_f32(reference);
    assert_eq!(realized_f32(actual), reference, "{epilogue} must run on the FP32 accumulator");
    assert_ne!(realized_f32(legacy), reference, "fixture must detect rounding before the {epilogue}");
}

#[test]
fn fp16_layernorm_keeps_affine_in_fp32_until_final_cast() {
    let x = fp16(&[0.1013, -0.2037, 0.3071, 1.913], &[1, 4]);
    let layer = LayerNormWeights {
        weight: fp16(&[17.25, -31.5, 47.75, -63.0], &[4]),
        bias: fp16(&[0.03125, -0.0625, 0.09375, -0.125], &[4]),
        eps: 1e-5,
    };
    let f32_of = |t: &Tensor| t.cast(DType::Float32).unwrap();

    let reference = f32_of(&x)
        .layernorm(-1, layer.eps)
        .unwrap()
        .try_mul(&f32_of(&layer.weight))
        .unwrap()
        .try_add(&f32_of(&layer.bias))
        .unwrap()
        .cast(DType::Float16)
        .unwrap();
    let legacy = x.layernorm(-1, layer.eps).unwrap().try_mul(&layer.weight).unwrap().try_add(&layer.bias).unwrap();

    assert_rounds_once(layer.apply(&x).unwrap(), reference, legacy, "affine epilogue");
}

#[test]
fn fp16_linear_keeps_bias_epilogue_in_fp32_until_final_cast() {
    let x = fp16(&[0.3333, -0.1428, 0.0909, 0.0769], &[1, 4]);
    let weight = fp16(&[3.0, -5.0, 7.0, -11.0, -2.0, 4.0, -6.0, 8.0], &[2, 4]);
    let bias = fp16(&[-2.5, 2.0], &[2]);

    let reference = x
        .linear()
        .weight(&weight)
        .dtype(DType::Float32)
        .call()
        .unwrap()
        .try_add(&bias.cast(DType::Float32).unwrap())
        .unwrap()
        .cast(DType::Float16)
        .unwrap();
    let legacy = x.linear().weight(&weight).bias(&bias).call().unwrap();

    assert_rounds_once(linear_with_bias(&x, &weight, &bias).unwrap(), reference, legacy, "bias addition");
}
