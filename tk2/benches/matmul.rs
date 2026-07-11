//! Criterion GPU-device-time bench for `hk::micro_tk` (the HipKittens BF16→FP32 GEMM port) on the
//! production Llama-70B GEMM shapes — rectangular `C = A·Bᵀ` at token counts T. Correctness-gated
//! (allclose vs an f32 reference over the same bf16-rounded operands) before it is timed, so a
//! broken schedule fails the bench rather than reporting a fast-but-wrong number. See [`common`]
//! for device-time stamping, self-skip, and the `--profile-time` PMC hook. (The square DSL dev
//! canary — kblock variants + hk_micro_tk square, DESIGN.md §7 — was removed: no transformer layer
//! is square M=N=K; the rectangular shapes are the production comparison.)
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

use svod_tk2::hk::micro_tk;
use svod_tk2::{Program, SwizzlePass, graph_kernel};

/// f32 ground truth `A·B` over the SAME bf16-rounded operands (both kernel and
/// reference see the realized bf16 values cast up to f32).
fn reference(a: &Tensor, b: &Tensor) -> Vec<f32> {
    let bf = b.cast(DType::Float32).expect("b→f32");
    let mut r = a.cast(DType::Float32).expect("a→f32").matmul(&bf).expect("reference matmul");
    r.realize().expect("realize reference");
    r.as_vec::<f32>().expect("read reference")
}

/// Wrap `hk::micro_tk`'s `Program` as a graph-node Tensor over `(a, b)` with a fresh bf16 output
/// template (HK's truncating fp32→bf16 C store), and prepare its execution plan.
fn plan_of_bf16(program: Program, m: usize, n: usize, a: &Tensor, b: &Tensor) -> (Tensor, ExecutionPlan) {
    let out = Tensor::empty(&[m, n], DType::BFloat16);
    let mut y = graph_kernel(program, out, &[a, b]).expect("wrap matmul as graph node");
    let plan = y.prepare().expect("prepare execution plan");
    (y, plan)
}

/// Correctness gate: execute, then read the bf16 output widened to f32 (bf16→f32 is exact) and
/// allclose vs the f32 reference (`atol ≈ 0.02·√K`, `rtol = 2e-2`). Panics (failing the bench) on
/// mismatch — a broken schedule cannot be silently timed.
fn assert_correct_bf16(y: &Tensor, plan: &ExecutionPlan, expected: &[f32], k: usize, label: &str) {
    plan.execute().expect("execute for correctness");
    // `execute()` submits async (wait=false — svod's sync-on-read model). The `y.cast(f32).realize()`
    // below is a device op that reads `y` WITHOUT waiting for the kernel, so it races it (flaky zero
    // tiles at the long 8192 kernel). Synchronize the output buffer before that read — what the f32
    // gate gets for free (its direct `as_vec` → `copyout` synchronizes). Kernel is correct; gate-only.
    plan.output_buffer().expect("output buffer").synchronize().expect("sync before read");
    let mut f = y.cast(DType::Float32).expect("bf16→f32 widen");
    f.realize().expect("realize widened output");
    let got = f.as_vec::<f32>().expect("read output");
    let atol = 0.02 * (k as f32).sqrt();
    let report = allclose_f32(&got, expected, atol, 2e-2);
    assert!(report.ok, "{label} matmul must match reference: {}", report.message);
}

/// RECTANGULAR, ML-realistic bench: the three Llama-70B GEMM types `C[m,n] = A[m,k]·Bᵀ`
/// (B stored `[n,k]`) at three token counts `T = m ∈ {256, 2048, 8192}` — 9 configs. Only
/// `hk::micro_tk` is rectangular-capable: it derives the grid from `m/n` and the K-loop from
/// `k` (asserting `m,n` ÷256 and `k` ÷64 with `k/64 ≥ 2`), so a distinct `m≠n≠k` is native.
/// `svod_tk::matmul` is square-only (`NotSquareSnafu` on `an≠am`), so it is not benched here.
/// bf16→f32 build + truncating fp32→bf16 C store + allclose correctness gate.
fn bench_matmul_rect(c: &mut Criterion) {
    if !requirements_met() {
        eprintln!("svod-tk2 rectangular matmul bench: skipped (device is not a supported gfx942 GPU)");
        return;
    }
    // (name, n = output cols, k = contraction). All `n` ÷256, `k` ÷64 — the micro_tk grid/K-loop bounds.
    let shapes = [("attn_out", 8192usize, 8192usize), ("ffn_up", 28672, 8192), ("ffn_down", 8192, 28672)];
    let mut group = c.benchmark_group("tk2_matmul_rect");
    for &(name, n, k) in &shapes {
        for &t in &[256usize, 2048, 8192] {
            let m = t; // tokens = the M rows
            group.throughput(Throughput::Elements(2 * m as u64 * n as u64 * k as u64)); // 2·M·N·K FLOPs
            let a = rand_bf16(&[m, k]); // A is [M,K]
            let b = rand_bf16(&[n, k]); // B is [N,K] (the A·Bᵀ contract — K contiguous)
            // A·Bᵀ f32 reference: B is [N,K], so transpose to [K,N] for the generic matmul.
            let expected = reference(&a, &b.try_transpose(0, 1).expect("Bᵀ for A·Bᵀ reference"));
            // hk::micro_tk: SwizzlePass-only (its asm gathers are already vectorized), bf16 C store.
            let hk_tk = micro_tk(m, n, k).apply(SwizzlePass);
            let (yhk_tk, phk_tk) = plan_of_bf16(hk_tk, m, n, &a, &b);
            assert_correct_bf16(&yhk_tk, &phk_tk, &expected, k, "hk_micro_tk_rect");
            group.bench_with_input(BenchmarkId::new(format!("hk_micro_tk/{name}"), t), &t, |bch, _| {
                bench_plan(bch, &phk_tk)
            });
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
