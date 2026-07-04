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

use crate::kernels::{Program, elementwise_add, matmul, sum_reduce};
use crate::launch;
use crate::passes::optimize_addressing;

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

/// Realize a random bf16 tensor and return it + its concrete (allocated) buffer.
fn bf16_input(shape: &[usize]) -> (Tensor, Buffer) {
    let mut t = Tensor::rand_with(shape, DType::BFloat16, svod_dtype::default_device::default_device())
        .expect("rand bf16 tensor");
    t.realize().expect("realize bf16 input");
    let buf = t.buffer().expect("bf16 input buffer");
    buf.ensure_allocated().expect("bf16 input allocated");
    (t, buf)
}

/// f32 ground-truth `A·B` over the SAME bf16-rounded operands (the kernel and the
/// reference both see the realized bf16 values, cast up to f32).
fn matmul_reference(a: &Tensor, b: &Tensor) -> Vec<f32> {
    let af = a.cast(DType::Float32).expect("a→f32");
    let bf = b.cast(DType::Float32).expect("b→f32");
    let mut reference = af.matmul(&bf).expect("reference matmul");
    reference.realize().expect("realize reference");
    reference.as_vec::<f32>().expect("read reference")
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::matmul_runs_on_gfx942 --nocapture`
///
/// The naive bf16→f32 tile matmul (fragment gather + 16×16×16 MFMA + loop-carried
/// accumulator) runs on the real GPU and matches an f32 reference over the same
/// bf16-rounded operands, at the tk matmul tolerance (`atol ≈ 0.02·√K`, `rtol = 2e-2`).
#[test]
#[ignore]
fn matmul_runs_on_gfx942() {
    for (m, n, k) in [(64usize, 64usize, 64usize), (128, 128, 128)] {
        let (a, a_buf) = bf16_input(&[m, k]);
        let (b, b_buf) = bf16_input(&[k, n]);
        let (out, c_buf) = output(m * n);

        // ABI order: output C first, then inputs A, B (the builder's `global()` order).
        let program = matmul(m, n, k);
        launch::run(&program, &[c_buf, a_buf, b_buf]).expect("matmul dispatch");

        let got = out.as_vec::<f32>().expect("read C");
        let expected = matmul_reference(&a, &b);
        let atol = 0.02 * (k as f32).sqrt();
        let report = allclose_f32(&got, &expected, atol, 2e-2);
        println!(
            "matmul M={m} N={n} K={k}: ok={} max_abs_err={:e} (atol={atol:e}, rtol=2e-2)",
            report.ok, report.max_abs_err
        );
        assert!(report.ok, "{}", report.message);
    }
}

/// The matmul with the two addressing passes (unroll → const-fold) applied.
fn optimized_matmul(m: usize, n: usize, k: usize) -> Program {
    let mut p = matmul(m, n, k);
    let root = optimize_addressing(&mut p.ir, p.sink).expect("addressing pipeline");
    Program { ir: p.ir, sink: root, name: p.name }
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::matmul_passes_bit_exact_on_gfx942 --nocapture`
///
/// Proof (b): the unroll + const-fold passes are semantics-preserving — the
/// optimized matmul still runs on gfx942 and matches the f32 reference EXACTLY
/// (`max_abs_err = 0`), same as the un-optimized kernel.
#[test]
#[ignore]
fn matmul_passes_bit_exact_on_gfx942() {
    for (m, n, k) in [(64usize, 64usize, 64usize), (128, 128, 128)] {
        let (a, a_buf) = bf16_input(&[m, k]);
        let (b, b_buf) = bf16_input(&[k, n]);
        let (out, c_buf) = output(m * n);

        let program = optimized_matmul(m, n, k);
        launch::run(&program, &[c_buf, a_buf, b_buf]).expect("optimized matmul dispatch");

        let got = out.as_vec::<f32>().expect("read C");
        let expected = matmul_reference(&a, &b);
        let report = allclose_f32(&got, &expected, 0.0, 0.0);
        println!("matmul[unroll+fold] M={m} N={n} K={k}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
        assert!(report.ok, "optimized matmul must be bit-exact: {}", report.message);
    }
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::matmul_passes_walltime_128 --nocapture`
///
/// Proof (d): a coarse best-of-N wall-time at 128³, before vs after the passes.
/// NOTE: `launch::run` recompiles every call, so this includes compilation — it is a
/// baseline datapoint, NOT a kernel-time comparison (unrolling grows the kernel, so
/// compile time may rise). Reported, not tuned against (the profiler-gated levers are
/// the coordinator's).
#[test]
#[ignore]
fn matmul_passes_walltime_128() {
    let (m, n, k) = (128usize, 128, 128);
    let iters = 20;
    let (_a, a_buf) = bf16_input(&[m, k]);
    let (_b, b_buf) = bf16_input(&[k, n]);
    let (_out, c_buf) = output(m * n);

    let best = |program: &Program| -> f64 {
        let bufs = [c_buf.clone(), a_buf.clone(), b_buf.clone()];
        launch::run(program, &bufs).expect("warm-up"); // populate the compile cache
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = std::time::Instant::now();
            launch::run(program, &bufs).expect("dispatch");
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };

    let base = matmul(m, n, k);
    let opt = optimized_matmul(m, n, k);
    let t_base = best(&base);
    let t_opt = best(&opt);
    println!(
        "matmul wall-time 128³ (best of {iters}, INCLUDES compile): rolled={t_base:.3} ms, unroll+fold={t_opt:.3} ms"
    );
}
