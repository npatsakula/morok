//! Kernel-count pins for rangeify's range assignment.
//!
//! Every expected count is tinygrad's, measured at `8c8b43de` with
//! `len(Tensor.schedule_linear().src)` on the same graph; morok already
//! matches each one. These rows exist so a future change to
//! `schedule/src/rangeify/indexing.rs` cannot silently move a fusion
//! boundary.

use crate::Tensor;
use crate::reduce::AxisSpec;
use svod_dtype::DType;
use svod_ir::{Op, UOp};
use test_case::test_case;

/// Kernels the rangeified graph would launch: one `Op::Call` per kernel,
/// the same collection `create_pre_schedule` walks.
fn count_kernels(t: &Tensor) -> usize {
    let sink = UOp::sink(vec![t.uop().contiguous()]);
    let rangeified = svod_schedule::rangeify_with_map(sink).expect("rangeify");
    let (kernels, _) = svod_schedule::try_get_kernel_graph(rangeified.sink).expect("kernel graph");
    kernels.toposort_call_aware(false).iter().filter(|n| matches!(n.op(), Op::Call(..))).count()
}

fn f32_in(shape: &[usize]) -> Tensor {
    Tensor::empty(shape, DType::Float32)
}

fn ln(x: &Tensor) -> Tensor {
    x.layernorm(-1, 1e-5).expect("layernorm")
}

/// Layernorm with the per-channel affine the conformer conv module applies.
fn ln_affine(x: &Tensor) -> Tensor {
    let c = x.shape().expect("shape").last().and_then(|d| d.as_const()).expect("static channels");
    let (w, b) = (f32_in(&[c]), f32_in(&[c]));
    ln(x).try_mul(&w).expect("scale").try_add(&b).expect("shift")
}

fn silu(x: &Tensor) -> Tensor {
    x.silu().expect("silu")
}

fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    a.try_mul(b).expect("mul")
}

fn transpose(x: &Tensor) -> Tensor {
    x.try_transpose(-1, -2).expect("transpose")
}

fn sub_mean(x: &Tensor) -> Tensor {
    let mean = x.mean_with().axes(AxisSpec::Single(-1)).keepdim(true).call().expect("mean");
    x.try_sub(&mean).expect("sub")
}

/// P1: layernorm -> silu.
fn ln_plain() -> Tensor {
    silu(&ln(&f32_in(&[1, 125, 768])))
}

/// P2: permute -> layernorm -> permute -> silu.
fn ln_transposed() -> Tensor {
    let x = f32_in(&[1, 768, 125]);
    silu(&transpose(&ln(&transpose(&x))))
}

/// P3: P2 read twice by one consumer.
fn ln_transposed_consumer() -> Tensor {
    let y = ln_transposed();
    mul(&y, &y)
}

/// P4: P1 read twice by one consumer.
fn ln_plain_consumer() -> Tensor {
    let y = ln_plain();
    mul(&y, &y)
}

/// P5: permute -> (x - mean) -> permute.
fn meanonly_transposed() -> Tensor {
    let x = f32_in(&[1, 768, 125]);
    transpose(&sub_mean(&transpose(&x)))
}

/// P6: x - mean.
fn meanonly_plain() -> Tensor {
    sub_mean(&f32_in(&[1, 125, 768]))
}

fn two() -> Tensor {
    Tensor::const_(2.0f32, DType::Float32)
}

/// B1: broadcast operand is itself elementwise-computed.
fn bcast_elemwise_operand() -> Tensor {
    let (x, y) = (f32_in(&[1, 125, 768]), f32_in(&[1, 125, 1]));
    mul(&x, &mul(&y, &two()))
}

/// B2: broadcast operand is a plain buffer.
fn bcast_buffer_operand() -> Tensor {
    mul(&f32_in(&[1, 125, 768]), &f32_in(&[1, 125, 1]))
}

/// B3: rank-lifted broadcast of an elementwise-computed vector.
fn bcast_vec_elemwise() -> Tensor {
    let (x, g) = (f32_in(&[1, 125, 768]), f32_in(&[768]));
    mul(&x, &mul(&g, &two()).try_reshape([1, 1, 768]).expect("reshape"))
}

/// B4: rank-lifted broadcast of a buffer vector.
fn bcast_vec_buffer() -> Tensor {
    let (x, g) = (f32_in(&[1, 125, 768]), f32_in(&[768]));
    mul(&x, &g.try_reshape([1, 1, 768]).expect("reshape"))
}

/// Conformer conv module: transposed layernorm feeding a pointwise conv1d.
fn ln_conv() -> Tensor {
    let x = f32_in(&[1, 768, 125]);
    let pw = f32_in(&[768, 768, 1]);
    let y = silu(&transpose(&ln_affine(&transpose(&x))));
    y.conv2d().weight(&pw).call().expect("pointwise conv")
}

/// Conformer conv module with the depthwise conv in front.
fn dw_ln_conv() -> Tensor {
    let x = f32_in(&[1, 768, 125]);
    let dw = f32_in(&[768, 1, 5]);
    let pw = f32_in(&[768, 768, 1]);
    let y = x.conv2d().weight(&dw).groups(768).padding(&[(2, 2)]).call().expect("depthwise conv");
    let y = silu(&transpose(&ln_affine(&transpose(&y))));
    y.conv2d().weight(&pw).call().expect("pointwise conv")
}

#[test_case(super::ln_plain, 3 ; "ln_plain")]
#[test_case(super::ln_transposed, 3 ; "ln_transposed")]
#[test_case(super::ln_transposed_consumer, 3 ; "ln_transposed_consumer")]
#[test_case(super::ln_plain_consumer, 3 ; "ln_plain_consumer")]
#[test_case(super::meanonly_transposed, 2 ; "meanonly_transposed")]
#[test_case(super::meanonly_plain, 2 ; "meanonly_plain")]
#[test_case(super::bcast_elemwise_operand, 1 ; "bcast_elemwise_operand")]
#[test_case(super::bcast_buffer_operand, 1 ; "bcast_buffer_operand")]
#[test_case(super::bcast_vec_elemwise, 1 ; "bcast_vec_elemwise")]
#[test_case(super::bcast_vec_buffer, 1 ; "bcast_vec_buffer")]
#[test_case(super::ln_conv, 4 ; "ln_conv")]
#[test_case(super::dw_ln_conv, 5 ; "dw_ln_conv")]
fn rangeify_kernel_count(build: fn() -> Tensor, tinygrad_kernels: usize) {
    assert_eq!(count_kernels(&build()), tinygrad_kernels);
}
