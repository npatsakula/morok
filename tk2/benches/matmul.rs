//! Criterion GPU-device-time bench for the tk2 naive matmul and its addressing
//! passes (unroll → const-fold) — the perf-canary AND the day-one measurement
//! feedback loop (DESIGN.md §7). Every variant is **correctness-gated** (allclose vs
//! an f32 reference over the same bf16-rounded operands) before it is timed, so a
//! broken schedule fails the bench rather than reporting a fast-but-wrong number —
//! the "check tensor values" gate that lets a criterion bench double as the device
//! correctness test (no `#[ignore]` device tests, no custom timing harness). See
//! [`common`] for device-time stamping, self-skip, and the `--profile-time` PMC hook.
//!
//! Run:  `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk2 --bench matmul`
//! PMC:  `SVOD_DEVICE=AMD:0 SVOD_PMC=1 SVOD_PMC_FORCE=1 cargo bench -p svod-tk2 --bench matmul -- --profile-time 5`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_runtime::ExecutionPlan;
use svod_tensor::Tensor;
use svod_tensor::testing::allclose_f32;

mod common;
use common::{bench_plan, rand_bf16, requirements_met};

use svod_tk2::{Program, SwizzlePass, graph_kernel, matmul, matmul_lds_kblock_ks, optimize_addressing};

/// f32 ground truth `A·B` over the SAME bf16-rounded operands (both kernel and
/// reference see the realized bf16 values cast up to f32).
fn reference(a: &Tensor, b: &Tensor) -> Vec<f32> {
    let bf = b.cast(DType::Float32).expect("b→f32");
    let mut r = a.cast(DType::Float32).expect("a→f32").matmul(&bf).expect("reference matmul");
    r.realize().expect("realize reference");
    r.as_vec::<f32>().expect("read reference")
}

/// Wrap a tk2 matmul `Program` as a graph-node Tensor over `(a, b)` with a fresh f32
/// output template, and prepare its execution plan.
fn plan_of(program: Program, m: usize, n: usize, a: &Tensor, b: &Tensor) -> (Tensor, ExecutionPlan) {
    let out = Tensor::empty(&[m, n], DType::Float32);
    let mut y = graph_kernel(program, out, &[a, b]).expect("wrap matmul as graph node");
    let plan = y.prepare().expect("prepare execution plan");
    (y, plan)
}

/// Correctness gate: execute the plan once and allclose the wired output vs the f32
/// reference (tk matmul tolerance `atol ≈ 0.02·√K`, `rtol = 2e-2`). Panics (failing
/// the bench) on mismatch — a broken schedule cannot be silently timed.
fn assert_correct(y: &Tensor, plan: &ExecutionPlan, expected: &[f32], k: usize, label: &str) {
    plan.execute().expect("execute for correctness");
    let got = y.as_vec::<f32>().expect("read output");
    let atol = 0.02 * (k as f32).sqrt();
    let report = allclose_f32(&got, expected, atol, 2e-2);
    assert!(report.ok, "{label} matmul must match reference: {}", report.message);
}

/// tk2 matmul, rolled (naive per-K-step gather) vs unroll+const-fold addressing.
/// Square `M = N = K`, bf16 in, f32 accumulate; multiples of 16 (the MFMA edge).
fn bench_matmul(c: &mut Criterion) {
    if !requirements_met() {
        eprintln!("svod-tk2 matmul bench: skipped (device is not a supported gfx942 GPU)");
        return;
    }
    let mut group = c.benchmark_group("tk2_matmul");
    for &n in &[1024usize, 2048, 4096] {
        group.throughput(Throughput::Elements((2.0 * (n as f64).powi(3)) as u64)); // 2·M·N·K
        let a = rand_bf16(&[n, n]);
        let b = rand_bf16(&[n, n]);
        let expected = reference(&a, &b);

        // Rolled: the naive K-loop with per-step div/mod gather addressing.
        let (y0, p0) = plan_of(matmul(n, n, n), n, n, &a, &b);
        assert_correct(&y0, &p0, &expected, n, "rolled");
        group.bench_with_input(BenchmarkId::new("rolled", n), &n, |bch, _| bench_plan(bch, &p0));

        // Unroll + const-fold: the two addressing passes, applied to the tile-IR.
        let mut opt = matmul(n, n, n);
        let root = optimize_addressing(&mut opt.ir, opt.sink).expect("addressing pipeline");
        let opt = Program { ir: opt.ir, sink: root, name: opt.name };
        let (y1, p1) = plan_of(opt, n, n, &a, &b);
        assert_correct(&y1, &p1, &expected, n, "unroll+fold");
        group.bench_with_input(BenchmarkId::new("unroll+fold", n), &n, |bch, _| bench_plan(bch, &p1));

        // K-blocked LDS reuse (step 1b-ii): 64×64 tile, A/B strips re-staged per K-block,
        // reuse across the 4×4 accumulator grid, two barriers per block (RAW + WAR). At
        // K_STEP=64 (the 4×-fewer-barriers win), compare the flat-layout BASE vs the same
        // base `.apply(SwizzlePass)` — the swizzle is a composable layout pass now, so this
        // is the top-level `.apply` model measured end-to-end.
        let base = matmul_lds_kblock_ks(n, n, n, 64, 64, 64);
        let (yb, pb) = plan_of(base, n, n, &a, &b);
        assert_correct(&yb, &pb, &expected, n, "kblock_ks64");
        group.bench_with_input(BenchmarkId::new("kblock_ks64", n), &n, |bch, _| bench_plan(bch, &pb));

        let swizzled = matmul_lds_kblock_ks(n, n, n, 64, 64, 64).apply(SwizzlePass);
        let (ysw, psw) = plan_of(swizzled, n, n, &a, &b);
        assert_correct(&ysw, &psw, &expected, n, "kblock_sw64");
        group.bench_with_input(BenchmarkId::new("kblock_sw64", n), &n, |bch, _| bench_plan(bch, &psw));
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_matmul
}
criterion_main!(benches);
