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

use svod_tk2::{
    Program, SwizzlePass, VectorizePass, graph_kernel, matmul_lds_kblock_mw_clustered, matmul_lds_kblock_mw_pipe2,
};

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
    // 8192 added for the grid-FULL fair comparison vs HK (at 4096 the 256² tile makes only 256 WGs
    // for 304 CUs → ~24% idle → device-wide mfmautil deflated; at 8192 the grid fills). The slow
    // single-warp / naive variants are gated to n≤4096 (they'd take minutes at 8192).
    for &n in &[1024usize, 2048, 4096, 8192] {
        group.throughput(Throughput::Elements((2.0 * (n as f64).powi(3)) as u64)); // 2·M·N·K
        let a = rand_bf16(&[n, n]);
        let b = rand_bf16(&[n, n]);
        // The `matmul_lds_kblock*` family takes B as [N,K] and computes A·Bᵀ (HK's contract), so its
        // reference transposes B (fast GPU matmul, not a host loop — this runs at N=4096).
        let expected_abt = reference(&a, &b.try_transpose(0, 1).expect("Bᵀ for A·Bᵀ reference"));
        // Compiler-visible HK copy (DESIGN.md 2026-07-09): pipe2 now authors HK's full 8-cluster
        // dot-slice pipeline + 8-wave ping-pong via schedule::pipeline, with INTRINSIC MFMAs (the
        // asm-free bet). HK's own tiling (bm=128/bn=64/wm=2/wn=4 → 256² tile, 2 warp-rows) so the
        // warp-phase ping-pong is valid. Measured: does compiler-visible beat the asm clustered (500)?
        let mw256p2 = matmul_lds_kblock_mw_pipe2(n, n, n, 128, 64, 2, 4, 64).apply(VectorizePass).apply(SwizzlePass);
        let (ymw2p2, pmw2p2) = plan_of(mw256p2, n, n, &a, &b);
        assert_correct(&ymw2p2, &pmw2p2, &expected_abt, n, "kblock_mw256_pipe2");
        group.bench_with_input(BenchmarkId::new("kblock_mw256_pipe2", n), &n, |bch, _| bench_plan(bch, &pmw2p2));
        // §5c clustered HK replica (256² tile, HK tiling bm=128/bn=64/wm=2/wn=4): the COMPLETE
        // schedule — 8 clusters + per-cluster s_barrier + set_prio + sched_fence + the warp-phase
        // ping-pong. Now bit-exact, so measured: does the full co-designed schedule reach HK's 0.65?
        if std::env::var("SVOD_SKIP_CLUSTERED").is_err() {
            let hk = matmul_lds_kblock_mw_clustered(n, n, n, 128, 64, 2, 4, 64).apply(VectorizePass).apply(SwizzlePass);
            let (yhk, phk) = plan_of(hk, n, n, &a, &b);
            assert_correct(&yhk, &phk, &expected_abt, n, "kblock_hk");
            group.bench_with_input(BenchmarkId::new("kblock_hk", n), &n, |bch, _| bench_plan(bch, &phk));
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_matmul
}
criterion_main!(benches);
