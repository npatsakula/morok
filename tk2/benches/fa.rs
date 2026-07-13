//! Criterion GPU-device-time bench for the **Flash-Attention forward** kernel
//! ([`svod_tk2::kernels_fa::flash_attention_fwd`]) on gfx942 — the correct, single-warp FA authored on
//! the ClusterCx pipeline. Modeled on [`super::matmul`]'s harness (shared [`common`]: device-time
//! stamping, self-skip, `--profile-time` PMC hook).
//!
//! ## Validated attention — with a REAL correctness gate
//! Unlike the earlier "schedule proxy" (whose P·V was the wrong contraction orientation), this kernel
//! is NUMERICALLY CORRECT: QKᵀ contracts over `d` (`d/16` `mma` K-steps), the online softmax runs the
//! two `ds_bpermute` column reductions + the running rescale, and P·V is `mma_atb` (the QKᵀ f32
//! accumulator `P` feeds the operand with only a bf16 cast; `V` is gathered transposed). Each config is
//! **gated against an f32 reference** (`allclose`) before timing, so a wrong result fails the bench.
//!
//! ## Shape support / honest perf caveat
//! Correct for head_dim `d` any multiple of 16 (benched at `d ∈ {64,128}`), `B = H = 1`, ONE 64-lane
//! warp, non-causal. This is the **correctness base** (Phase A): single-warp, no ping-pong / async-LDS
//! / swizzle / multi-warp occupancy. The TF below is therefore an HONEST single-warp figure — far below
//! a production multi-warp FA (Phase B), not a competitive number. It measures a *correct* kernel's cost.
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

/// Realize `t` as an f32 host vector (bf16 operands cast to f32 first) — the reference input.
fn as_f32(t: &Tensor) -> Vec<f32> {
    let mut f = t.cast(DType::Float32).expect("→f32");
    f.realize().expect("realize f32");
    f.as_vec::<f32>().expect("read f32")
}

/// Host f32 reference for non-causal FA-forward: `O[q,dd] = Σ_k softmax_k(Q[q]·K[k]/√d)·V[k,dd]` over
/// the SAME bf16-rounded operands (`q`/`k`/`v` `[s,d]` row-major).
fn fa_ref(qf: &[f32], kf: &[f32], vf: &[f32], s: usize, d: usize) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let mut o = vec![0f32; s * d];
    for qi in 0..s {
        let mut sc = vec![0f32; s];
        for ki in 0..s {
            let mut acc = 0f32;
            for di in 0..d {
                acc += qf[qi * d + di] * kf[ki * d + di];
            }
            sc[ki] = acc * scale;
        }
        let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for x in &mut sc {
            *x = (*x - m).exp();
            sum += *x;
        }
        for di in 0..d {
            let mut acc = 0f32;
            for ki in 0..s {
                acc += sc[ki] * vf[ki * d + di];
            }
            o[qi * d + di] = acc / sum;
        }
    }
    o
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

/// CORRECTNESS gate: execute once, `allclose` the output against the f32 reference (`atol = 0.02·√d`,
/// `rtol = 2e-2` — matmul style). A wrong result fails the bench BEFORE any timing is reported.
fn gate_correct(y: &Tensor, plan: &ExecutionPlan, q: &Tensor, k: &Tensor, v: &Tensor, s: usize, d: usize) {
    use svod_tensor::testing::allclose_f32;
    plan.execute().expect("execute for gate");
    plan.output_buffer().expect("output buffer").synchronize().expect("sync before read");
    let got = y.as_vec::<f32>().expect("read FA output");
    let expected = fa_ref(&as_f32(q), &as_f32(k), &as_f32(v), s, d);
    let report = allclose_f32(&got, &expected, 0.02 * (d as f32).sqrt(), 2e-2);
    assert!(report.ok, "FA S={s} d={d}: correctness gate FAILED before timing: {}", report.message);
}

/// The FA bench across `S ∈ {512,1024,2048,4096}` at `d ∈ {64,128}`. Correctness-gates each `d` at the
/// smallest `S` (the numerics are shape-invariant modulo tiling, which the device test covers at both
/// `d`), prints device µs + REAL attention TF per config, then hands the prepared plan to criterion.
fn bench_fa(c: &mut Criterion) {
    if !requirements_met() {
        eprintln!("svod-tk2 FA-forward bench: skipped (device is not a supported gfx942 GPU)");
        return;
    }
    let seqlens = [512usize, 1024, 2048, 4096];
    eprintln!(
        "\n=== tk2 FA-forward: VALIDATED attention (allclose-gated), HONEST single-warp TF ===\n    \
         B=H=1, 1 warp, non-causal — the Phase-A correctness base (no ping-pong / multi-warp / swizzle)\n"
    );
    let mut group = c.benchmark_group("tk2_fa");
    for &d in &[64usize, 128] {
        for (i, &s) in seqlens.iter().enumerate() {
            let flops = fa_flops_noncausal(s, d);
            group.throughput(Throughput::Elements(flops));
            let q = rand_bf16(&[s, d]);
            let k = rand_bf16(&[s, d]);
            let v = rand_bf16(&[s, d]);
            let (y, plan) = plan_of_fa(s, d, &q, &k, &v);
            if i == 0 {
                gate_correct(&y, &plan, &q, &k, &v, s, d); // gate this d at the smallest S
            } else {
                plan.execute().expect("execute");
            }

            let iters = 50u64;
            let avg_ns = plan_gpu_ns(&plan, iters) as f64 / iters as f64;
            let tf = flops as f64 / avg_ns / 1e3; // FLOPs / avg_ns / 1e3 = TFLOP/s
            let gated = if i == 0 { "  [correctness-gated ✓]" } else { "" };
            eprintln!("  FA S={s:>4} d={d:>3}: {:>8.2} µs  |  {tf:>7.1} TF{gated}", avg_ns / 1e3);

            group.bench_with_input(BenchmarkId::new(format!("fa_fwd/d{d}"), s), &s, |bch, _| bench_plan(bch, &plan));
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_fa
}
criterion_main!(benches);
