//! Criterion GPU-device-time bench for the **Flash-Attention forward schedule**
//! ([`svod_tk2::kernels_fa::flash_attention_fwd`]) on gfx942 — the experiment kernel authored on the
//! ClusterCx pipeline. Modeled on [`super::matmul`]'s harness (shared [`common`]: device-time
//! stamping, self-skip, `--profile-time` PMC hook).
//!
//! ## ⚠️ THIS MEASURES *SCHEDULE THROUGHPUT*, NOT VALIDATED ATTENTION ⚠️
//! The kernel is structurally complete (streams K/V → QKᵀ → online-softmax → P·V → normalize) and
//! runs without fault, but its P·V matmul uses tk2's only MMA (`A·Bᵀ`) where correct FA needs
//! `Vᵀ·att` (contraction over kv) — tk2 has no `mma_atb`/transpose (a catalogued vocabulary gap). So
//! the numbers below are a **ceiling proxy** for the schedule's cost, NOT attention correctness.
//! There is therefore **NO correctness gate** here (it would rightly fail); instead a finite-output
//! sanity check (no NaN/Inf, no GPU fault), exactly like the `fa_forward_launches_on_gfx942` test.
//!
//! Why the throughput is still a fair proxy: BOTH matmuls (QKᵀ and P·V) have the SAME FLOP count
//! regardless of P·V's contraction orientation (`2·B·H·S²·d` MACs each), and the softmax/movement
//! cost is orientation-independent — so fixing the transpose changes *correctness*, not the FLOP bill
//! this schedule pays. The TF figure is what a correct FA on this schedule would cost, modulo that fix.
//!
//! ## Shape-support LIMITS (an explicit kernel/DSL finding — see the experiment report)
//! The minimal kernel supports ONLY: head_dim `d = 16` (single d-fragment — multi-d needs the
//! inner-d-contraction loop the ClusterCx "slice" model doesn't host), `B = H = 1`, one 64-lane warp,
//! non-causal. So this benches the LARGEST it supports: `d = 16`, `S ∈ {512,1024,2048,4096}`. Realistic
//! attention (`d ∈ {64,128}`, multi-head batches, causal) is OUT of the current kernel's reach.
//!
//! Run:  `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk2 --bench fa`
//! PMC:  `SVOD_DEVICE=AMD:0 SVOD_PMC=1 SVOD_PMC_FORCE=1 cargo bench -p svod-tk2 --bench fa -- --profile-time 5`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_runtime::ExecutionPlan;
use svod_tensor::Tensor;

mod common;
use common::{bench_plan, plan_gpu_ns, rand_bf16, requirements_met};

use svod_tk2::graph_kernel;
use svod_tk2::kernels_fa::flash_attention_fwd;

/// FA-forward FLOPs, **non-causal**: `4·B·H·S²·d`. Two matmuls each cost `2·B·H·S²·d` FLOPs
/// (QKᵀ: `S×S` scores over `d` MACs; P·V: `S×d` outputs over `S` MACs) → `2 MACs/FLOP`. Causal
/// halves it (triangular sweep). This kernel is `B=H=1`, non-causal, so FLOPs `= 4·S²·d`.
fn fa_flops_noncausal(s: usize, d: usize) -> u64 {
    4 * (s as u64) * (s as u64) * (d as u64)
}

/// Wrap the FA `Program` as a graph-node Tensor over `(q, k, v)` with an f32 `[S,d]` output template,
/// and prepare its execution plan. (Q/K/V are bf16 `[S,d]`; the kernel is `B=H=1`.)
fn plan_of_fa(s: usize, d: usize, q: &Tensor, k: &Tensor, v: &Tensor) -> (Tensor, ExecutionPlan) {
    let prog = flash_attention_fwd(s, d);
    let out = Tensor::empty(&[s, d], DType::Float32);
    let mut y = graph_kernel(prog, out, &[q, k, v]).expect("wrap FA as graph node");
    let plan = y.prepare().expect("prepare FA execution plan");
    (y, plan)
}

/// SANITY gate (NOT correctness): execute, read the output, assert every element is finite (no
/// NaN/Inf, no GPU fault). A broken schedule (fault / all-garbage) fails the bench; a *numerically
/// wrong but finite* attention (the known P·V-orientation gap) is allowed through — see module docs.
fn assert_finite(y: &Tensor, plan: &ExecutionPlan, label: &str) {
    plan.execute().expect("execute for sanity");
    plan.output_buffer().expect("output buffer").synchronize().expect("sync before read");
    let got = y.as_vec::<f32>().expect("read FA output");
    let finite = got.iter().filter(|x| x.is_finite()).count();
    assert!(finite == got.len(), "{label}: FA output must be all-finite (got {finite}/{} finite)", got.len());
}

/// The FA schedule-throughput bench across `S ∈ {512,1024,2048,4096}` at `d = 16` (the kernel's max
/// supported shape). Prints device µs + a SCHEDULE-THROUGHPUT TF proxy per config (loudly caveated),
/// then hands the prepared plan to criterion for device-time-stamped sampling.
fn bench_fa_schedule(c: &mut Criterion) {
    if !requirements_met() {
        eprintln!("svod-tk2 FA-forward bench: skipped (device is not a supported gfx942 GPU)");
        return;
    }
    let d = 16; // kernel limit: single d-fragment
    let seqlens = [512usize, 1024, 2048, 4096];
    eprintln!(
        "\n=== tk2 FA-forward: SCHEDULE THROUGHPUT (a ceiling PROXY — NOT validated attention; \
         P·V orientation is a known gap) ===\n    shape limit: d=16, B=H=1, 1 warp, non-causal\n"
    );
    let mut group = c.benchmark_group("tk2_fa_schedule");
    for &s in &seqlens {
        let flops = fa_flops_noncausal(s, d);
        group.throughput(Throughput::Elements(flops));
        let q = rand_bf16(&[s, d]);
        let k = rand_bf16(&[s, d]);
        let v = rand_bf16(&[s, d]);
        let (y, plan) = plan_of_fa(s, d, &q, &k, &v);
        assert_finite(&y, &plan, &format!("fa_S{s}_d{d}"));

        // Explicit device-time + TF-proxy summary line (criterion also stamps device time below).
        let iters = 50u64;
        let total_ns = plan_gpu_ns(&plan, iters);
        let avg_ns = total_ns as f64 / iters as f64;
        let us = avg_ns / 1e3;
        let tf = flops as f64 / avg_ns / 1e3; // FLOPs / avg_ns / 1e3 = TFLOP/s
        eprintln!(
            "  FA schedule S={s:>4} d={d}: {us:>8.2} µs  |  {tf:>7.1} TF  \
             [SCHEDULE-ONLY proxy, NOT validated attention]"
        );

        group.bench_with_input(BenchmarkId::new(format!("fa_fwd/d{d}"), s), &s, |bch, _| bench_plan(bch, &plan));
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_fa_schedule
}
criterion_main!(benches);
