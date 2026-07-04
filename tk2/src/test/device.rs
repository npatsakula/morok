//! Hardware-gated end-to-end tests on gfx942 (`SVOD_DEVICE=AMD:0 ... --ignored`).
//!
//! `elementwise_add_runs_on_gfx942` is the "it works" proof: the tiled add lowers,
//! compiles, and runs on the real GPU, matching a CPU reference bit-for-bit (f32
//! add is exact). `sum_reduce_runs_on_gfx942` additionally exercises the
//! loop-carried accumulator on device.

use std::sync::Arc;

use svod_device::Buffer;
use svod_dtype::DType;
use svod_ir::Op;
use svod_tensor::Tensor;
use svod_tensor::testing::allclose_f32;

use crate::kernels::{elementwise_add, sum_reduce};
use crate::launch;

/// Realize an input tensor from host data and return its concrete buffer.
fn input(data: &[f32]) -> (Tensor, Buffer) {
    let mut t = Tensor::from_slice(data);
    t.realize().expect("realize input");
    let buf = t.buffer().expect("input buffer");
    buf.ensure_allocated().expect("input allocated");
    (t, buf)
}

/// Allocate + register a fresh output buffer for an empty tensor (the svod analog
/// of tinygrad's `b.allocate()` inside `run_linear`; mirrors tk's `realize_buffer`).
fn output(n: usize) -> (Tensor, Buffer) {
    let t = Tensor::empty(&[n], DType::Float32);
    if let Some(buf) = t.buffer() {
        buf.ensure_allocated().expect("output allocated");
        return (t, buf);
    }
    let base = t.uop().base();
    let Op::Buffer { device, size, .. } = base.op() else { panic!("output tensor has no BUFFER uop") };
    let Op::Device(spec) = device.op() else { panic!("output BUFFER has no DEVICE") };
    let dtype = base.dtype();
    let allocator = svod_device::registry::registry().get(spec).expect("allocator for output device");
    let buffer = Buffer::allocate(allocator, dtype, vec![*size], Default::default()).expect("allocate output");
    let buffer = Arc::new(buffer);
    svod_tensor::tensor_registry::register_buffer_by_uop_id(base.id, buffer.clone());
    (t, (*buffer).clone())
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::elementwise_add_runs_on_gfx942 --nocapture`
#[test]
#[ignore]
fn elementwise_add_runs_on_gfx942() {
    let (tile, n_tiles) = (64usize, 8usize);
    let n = tile * n_tiles;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();

    let program = elementwise_add(tile, n_tiles);
    let (_ta, a_buf) = input(&a);
    let (_tb, b_buf) = input(&b);
    let (out, c_buf) = output(n);

    // ABI order: output first, then inputs (the builder's `global()` order).
    launch::run(&program, &[c_buf, a_buf, b_buf]).expect("elementwise add dispatch");

    let got = out.as_vec::<f32>().expect("read output");
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let report = allclose_f32(&got, &expected, 0.0, 0.0);
    println!("elementwise add N={n}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
    assert!(report.ok, "{}", report.message);
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::sum_reduce_runs_on_gfx942 --nocapture`
#[test]
#[ignore]
fn sum_reduce_runs_on_gfx942() {
    let n = 1024usize;
    let a: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.25 - 1.0).collect();

    let program = sum_reduce(n);
    let (_ta, a_buf) = input(&a);
    let (out, o_buf) = output(1);

    launch::run(&program, &[o_buf, a_buf]).expect("sum reduce dispatch");

    let got = out.as_vec::<f32>().expect("read output");
    let expected = vec![a.iter().sum::<f32>()];
    // f32 accumulation order differs from the sequential reference → tolerance.
    let report = allclose_f32(&got, &expected, 1e-2, 1e-4);
    println!("sum reduce N={n}: got={} expected={} ok={}", got[0], expected[0], report.ok);
    assert!(report.ok, "{}", report.message);
}
