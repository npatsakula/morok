//! Criterion GPU-device-time bench for the **clustered** DSL matmul (`matmul_lds_kblock_mw_clustered`,
//! the production 256² config bm=128/bn=64/wm=2/wn=4/k_step=64, 8 warps) on the production Llama-70B
//! GEMM shapes — rectangular `C = A·Bᵀ` at token counts T. The clustered kernel is rectangular-capable
//! (`kblock_impl` takes independent m,n,k, asserting only bm/bn/k ÷16 and m/n ÷ (bm·wm)/(bn·wn)=256).
//! Correctness-gated (allclose vs an f32 reference over the same bf16-rounded operands) before it is
//! timed, so a broken schedule fails the bench rather than reporting a fast-but-wrong number. See
//! [`common`] for device-time stamping, self-skip, and the `--profile-time` PMC hook. Vec+swizzle
//! refinements are applied (they are bit-exact bijections). F32 C output.
//!
//! Run:  `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk2 --bench matmul`
//! PMC:  `SVOD_DEVICE=AMD:0 SVOD_PMC=1 SVOD_PMC_FORCE=1 cargo bench -p svod-tk2 --bench matmul -- --profile-time 5`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_runtime::ExecutionPlan;
use svod_tensor::Tensor;
use svod_tensor::testing::allclose_f32;

mod common;
use common::{bench_plan, plan_gpu_ns, rand_bf16, requirements_met};

use svod_tk2::{
    Program, SwizzlePass, Tiling, VectorizePass, graph_kernel, matmul_lds_kblock_mw_clustered, tiling_for_mn,
};

/// f32 ground truth `A·B` over the SAME bf16-rounded operands (both kernel and
/// reference see the realized bf16 values cast up to f32).
fn reference(a: &Tensor, b: &Tensor) -> Vec<f32> {
    let bf = b.cast(DType::Float32).expect("b→f32");
    let mut r = a.cast(DType::Float32).expect("a→f32").matmul(&bf).expect("reference matmul");
    r.realize().expect("realize reference");
    r.as_vec::<f32>().expect("read reference")
}

/// Wrap the clustered matmul `Program` as a graph-node Tensor over `(a, b)` with a fresh f32 output
/// template (the clustered kernel stores F32 C), and prepare its execution plan.
fn plan_of(program: Program, m: usize, n: usize, a: &Tensor, b: &Tensor) -> (Tensor, ExecutionPlan) {
    let out = Tensor::empty(&[m, n], DType::Float32);
    let mut y = graph_kernel(program, out, &[a, b]).expect("wrap matmul as graph node");
    let plan = y.prepare().expect("prepare execution plan");
    (y, plan)
}

/// Correctness gate: execute, then read the f32 output and allclose vs the f32 reference
/// (`atol ≈ 0.02·√K`, `rtol = 2e-2`). Panics (failing the bench) on mismatch — a broken schedule
/// cannot be silently timed. The clustered kernel is bit-exact, so a failure is a real regression.
fn assert_correct(y: &Tensor, plan: &ExecutionPlan, expected: &[f32], k: usize, label: &str) {
    plan.execute().expect("execute for correctness");
    let got = y.as_vec::<f32>().expect("read output");
    let atol = 0.02 * (k as f32).sqrt();
    let report = allclose_f32(&got, expected, atol, 2e-2);
    assert!(report.ok, "{label} matmul must match reference: {}", report.message);
}

/// One benched GEMM problem: the `A[m,k]·Bᵀ` shape, its realized bf16 operands, and the f32 reference.
struct Case<'a> {
    m: usize,
    n: usize,
    k: usize,
    a: &'a Tensor,
    b: &'a Tensor,
    expected: &'a [f32],
}

/// Compiled TFLOP/s from `2·m·n·k` FLOPs over a device-time `ns`.
fn tflops(m: usize, n: usize, k: usize, ns: u64) -> f64 {
    (2.0 * m as f64 * n as f64 * k as f64) / (ns as f64 * 1e3)
}

/// Build the clustered matmul at `tiling` with the production vec+swizzle passes, allclose-gate it against
/// the case's reference, then measure its steady GPU device time (ns) over 40 `plan.profile` replays (one
/// warmup discarded). Returns the mean-per-replay ns and the prepared plan (so criterion can re-time the
/// winner). The returned `Tensor` MUST be kept alive while the plan is used (it owns the output buffer).
fn measure_tiling(tiling: Tiling, case: &Case, label: &str) -> (u64, Tensor, ExecutionPlan) {
    let Case { m, n, k, a, b, expected } = *case;
    let prog = matmul_lds_kblock_mw_clustered(m, n, k, tiling).apply(VectorizePass).apply(SwizzlePass);
    let (y, plan) = plan_of(prog, m, n, a, b);
    assert_correct(&y, &plan, expected, k, label);
    let _ = plan_gpu_ns(&plan, 2); // warmup (caches/clocks) — discarded
    const ITERS: u64 = 40;
    let ns = plan_gpu_ns(&plan, ITERS) / ITERS;
    (ns, y, plan)
}

/// RECTANGULAR, ML-realistic bench proving tk1's small-N 128²-tile port: the three Llama-70B GEMM types
/// `C[m,n] = A[m,k]·Bᵀ` (B stored `[n,k]`) at three token counts `T = m ∈ {256, 2048, 8192}` — 9 configs
/// spanning the grid-starved (m=256), mid (m=2048), and saturated (m=8192) regimes. For every config it
/// measures BOTH the 256² [`Tiling::default`] tile and the 128² [`svod_tk2::SMALL`] tile, prints a
/// `256²-vs-128²` TFLOP/s comparison + the crossover, and then criterion-benches the tile the
/// [`tiling_for_mn`] auto-selector picks (so the benched path is the shipped one). The clustered matmul
/// derives the grid from `m/n` and the K-loop from `k`; all shapes satisfy `m,n ÷256`, `k ÷64`.
fn bench_matmul_rect(c: &mut Criterion) {
    if !requirements_met() {
        eprintln!("svod-tk2 rectangular matmul bench: skipped (device is not a supported gfx942 GPU)");
        return;
    }
    // (name, n = output cols, k = contraction). All `n` ÷256, `k` ÷64 — the clustered grid/K-loop bounds.
    let shapes = [("attn_out", 8192usize, 8192usize), ("ffn_up", 28672, 8192), ("ffn_down", 8192, 28672)];
    let mut group = c.benchmark_group("tk2_matmul_rect");
    println!("\n=== 256² (default) vs 128² (SMALL) tile — device TFLOP/s (2·m·n·k / gpu_ns) ===");
    println!(
        "{:<10} {:>6} {:>7} {:>6} {:>10} {:>10} {:>9}  winner @ auto",
        "shape", "m", "n", "wg256", "256² TF/s", "128² TF/s", "128²/256²",
    );
    for &(name, n, k) in &shapes {
        for &t in &[256usize, 2048, 8192] {
            let m = t; // tokens = the M rows
            group.throughput(Throughput::Elements(2 * m as u64 * n as u64 * k as u64)); // 2·M·N·K FLOPs
            let a = rand_bf16(&[m, k]); // A is [M,K]
            let b = rand_bf16(&[n, k]); // B is [N,K] (the A·Bᵀ contract — K contiguous)
            // A·Bᵀ f32 reference: B is [N,K], so transpose to [K,N] for the generic matmul.
            let expected = reference(&a, &b.try_transpose(0, 1).expect("Bᵀ for A·Bᵀ reference"));
            let case = Case { m, n, k, a: &a, b: &b, expected: &expected };

            // Measure both tiles (each allclose-gated). wg256 = the 256²-tile workgroup count.
            let (ns256, _, _) = measure_tiling(Tiling::default(), &case, "256²");
            let (ns128, _, _) = measure_tiling(svod_tk2::SMALL, &case, "128²");
            let (tf256, tf128) = (tflops(m, n, k, ns256), tflops(m, n, k, ns128));
            let wg256 = (m / 256) * (n / 256);
            let auto = tiling_for_mn(m, n);
            let picked = if auto.bm == svod_tk2::SMALL.bm { "128²" } else { "256²" };
            println!(
                "{name:<10} {m:>6} {n:>7} {wg256:>6} {tf256:>10.1} {tf128:>10.1} {:>8.2}x  {picked}",
                tf128 / tf256,
            );

            // Criterion-bench the AUTO-selected tile — the shipped path via `tiling_for_mn`. `_y` is bound
            // (not `_`) so the output tensor outlives `plan` during the criterion timing loop.
            let (_ns, _y, plan) = measure_tiling(auto, &case, "auto");
            group.bench_with_input(BenchmarkId::new(format!("auto/{name}"), t), &t, |bch, _| bench_plan(bch, &plan));
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_matmul_rect
}
criterion_main!(benches);
