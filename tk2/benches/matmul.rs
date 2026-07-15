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
use common::{bench_plan, rand_bf16, requirements_met};

use svod_tk2::{Program, SwizzlePass, Tiling, VectorizePass, graph_kernel, matmul_lds_kblock_mw_clustered};

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

/// RECTANGULAR, ML-realistic bench: the three Llama-70B GEMM types `C[m,n] = A[m,k]·Bᵀ`
/// (B stored `[n,k]`) at three token counts `T = m ∈ {256, 2048, 8192}` — 9 configs. The clustered
/// matmul derives the grid from `m/n` and the K-loop from `k`, so a distinct `m≠n≠k` is native as
/// long as `m,n` ÷256 and `k` ÷64 (all shapes below satisfy this). bf16 operands → f32 C.
fn bench_matmul_rect(c: &mut Criterion) {
    if !requirements_met() {
        eprintln!("svod-tk2 rectangular matmul bench: skipped (device is not a supported gfx942 GPU)");
        return;
    }
    // (name, n = output cols, k = contraction). All `n` ÷256, `k` ÷64 — the clustered grid/K-loop bounds.
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
            // Clustered 256² config, vec+swizzle (bit-exact bijections), F32 C store.
            let prog =
                matmul_lds_kblock_mw_clustered(m, n, k, Tiling::default()).apply(VectorizePass).apply(SwizzlePass);
            let (y, plan) = plan_of(prog, m, n, &a, &b);
            assert_correct(&y, &plan, &expected, k, "clustered_rect");
            group.bench_with_input(BenchmarkId::new(format!("clustered/{name}"), t), &t, |bch, _| {
                bench_plan(bch, &plan)
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
