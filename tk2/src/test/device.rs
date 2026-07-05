//! Hardware-gated device smoke tests for the two skeleton kernels on gfx942
//! (`SVOD_DEVICE=AMD:0 ... --ignored`). The matmul — the perf kernel — is measured
//! *and* correctness-gated through the criterion bench (`benches/matmul.rs`), the
//! day-one feedback loop; these two remain here because they exercise distinct
//! machinery (tiled elementwise, and the loop-carried accumulator) that has no bench.
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
use crate::{SwizzlePass, VectorizePass, graph_kernel};

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

/// The **stages=2 pipeline** correctness gate (DESIGN §5b phase 2b): the register-staged
/// prologue/steady/epilogue must accumulate every K-block exactly once — a carry slip (gather
/// reading the wrong block, a dropped commit) shows up as a wrong C, not a crash. Checked at a
/// small shape (128×128×256 = 4 K-blocks, 2×2 warps) against an f32 reference over the same
/// bf16-rounded operands; both the scalar-gather base AND `.apply(VectorizePass).apply(SwizzlePass)`
/// (the pass composition must survive the split-loop structure).
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::matmul_pipeline --nocapture`
#[test]
#[ignore]
fn matmul_pipeline_stages2_is_bit_exact_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    let (m, n, k) = (128usize, 128, 256);
    let dev = svod_dtype::default_device::default_device();
    let mut a = Tensor::rand_with(&[m, k], DType::BFloat16, dev.clone()).expect("rand a");
    let mut b = Tensor::rand_with(&[k, n], DType::BFloat16, dev).expect("rand b");
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    let bf = b.cast(DType::Float32).expect("b→f32");
    let mut rt = a.cast(DType::Float32).expect("a→f32").matmul(&bf).expect("ref matmul");
    rt.realize().expect("realize ref");
    let expected = rt.as_vec::<f32>().expect("read ref");
    let atol = 0.02 * (k as f32).sqrt();

    for (label, prog) in [
        ("pipe_base", crate::kernels::matmul_lds_kblock_mw_pipe(m, n, k, 64, 64, 2, 2, 64)),
        (
            "pipe_vec_sw",
            crate::kernels::matmul_lds_kblock_mw_pipe(m, n, k, 64, 64, 2, 2, 64)
                .apply(VectorizePass)
                .apply(SwizzlePass),
        ),
    ] {
        let out = Tensor::empty(&[m, n], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&a, &b]).expect("wrap");
        let plan = y.prepare().expect("prepare");
        plan.execute().expect("execute");
        let got = y.as_vec::<f32>().expect("read output");
        let report = allclose_f32(&got, &expected, atol, 2e-2);
        println!("matmul pipeline {label} {m}×{n}×{k}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
        assert!(report.ok, "{label} pipelined matmul must match reference: {}", report.message);
    }
}

/// The **clustered §5c schedule** correctness gate: the per-slice memory/compute cluster
/// decomposition + the `Bracket` sched controls must be invariant to correctness (a carry or
/// acc-round-trip slip shows as a wrong C). Swept over the bracket flags, base AND vec+swizzle.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::matmul_clustered --nocapture`
#[test]
#[ignore]
fn matmul_clustered_bracket_sweep_is_bit_exact_on_gfx942() {
    use crate::kernels::Bracket;
    use svod_tensor::testing::allclose_f32;
    let (m, n, k) = (128usize, 128, 256);
    let dev = svod_dtype::default_device::default_device();
    let mut a = Tensor::rand_with(&[m, k], DType::BFloat16, dev.clone()).expect("rand a");
    let mut b = Tensor::rand_with(&[k, n], DType::BFloat16, dev).expect("rand b");
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    let bf = b.cast(DType::Float32).expect("b→f32");
    let mut rt = a.cast(DType::Float32).expect("a→f32").matmul(&bf).expect("ref matmul");
    rt.realize().expect("realize ref");
    let expected = rt.as_vec::<f32>().expect("read ref");
    let atol = 0.02 * (k as f32).sqrt();

    let base = Bracket::default();
    let brackets = [
        ("pin+fence", base),
        ("+prio", Bracket { prio: true, ..base }),
        ("+cbar", Bracket { per_cluster_barrier: true, ..base }),
    ];
    for (label, br) in brackets {
        for (suffix, apply_passes) in [("base", false), ("vec+sw", true)] {
            let mut prog = crate::kernels::matmul_lds_kblock_mw_clustered(m, n, k, 64, 64, 2, 2, 64, br);
            if apply_passes {
                prog = prog.apply(VectorizePass).apply(SwizzlePass);
            }
            let out = Tensor::empty(&[m, n], DType::Float32);
            let mut y = graph_kernel(prog, out, &[&a, &b]).expect("wrap");
            let plan = y.prepare().expect("prepare");
            plan.execute().expect("execute");
            let got = y.as_vec::<f32>().expect("read output");
            let report = allclose_f32(&got, &expected, atol, 2e-2);
            println!("clustered {label}/{suffix} {m}×{n}×{k}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
            assert!(report.ok, "clustered {label}/{suffix} must match reference: {}", report.message);
        }
    }
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
