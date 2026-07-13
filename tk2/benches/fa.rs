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
/// halves it (triangular sweep). `bh = B·H` independent attentions, so FLOPs `= 4·bh·S²·d`.
fn fa_flops_noncausal(bh: usize, s: usize, d: usize) -> u64 {
    4 * (bh as u64) * (s as u64) * (s as u64) * (d as u64)
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

/// Wrap the FA `Program` for `bh` stacked `[s,d]` attentions as a graph-node Tensor over `(q,k,v)`
/// (each bf16 `[bh·s, d]`) with an f32 `[bh·s, d]` output, and prepare its execution plan.
fn plan_of_fa(bh: usize, s: usize, d: usize, q: &Tensor, k: &Tensor, v: &Tensor) -> (Tensor, ExecutionPlan) {
    let prog = flash_attention_fwd(bh, s, d);
    let out = Tensor::empty(&[bh * s, d], DType::Float32);
    let mut y = graph_kernel(prog, out, &[q, k, v]).expect("wrap FA as graph node");
    let plan = y.prepare().expect("prepare FA execution plan");
    (y, plan)
}

/// CORRECTNESS gate at a SMALL shape (the host reference is `O(bh·s²·d)`, infeasible at the timed
/// realistic shapes — the device test `flash_attention_matches_reference_on_gfx942` is the full gate):
/// run `bh` independent `[s,d]` attentions and `allclose` the whole `[bh,s,d]` output before any timing.
fn gate_small(bh: usize, s: usize, d: usize) {
    use svod_tensor::testing::allclose_f32;
    let (q, k, v) = (rand_bf16(&[bh * s, d]), rand_bf16(&[bh * s, d]), rand_bf16(&[bh * s, d]));
    let (y, plan) = plan_of_fa(bh, s, d, &q, &k, &v);
    plan.execute().expect("execute for gate");
    plan.output_buffer().expect("output buffer").synchronize().expect("sync before read");
    let got = y.as_vec::<f32>().expect("read FA output");
    let (qf, kf, vf) = (as_f32(&q), as_f32(&k), as_f32(&v));
    let mut expected = vec![0f32; bh * s * d];
    for z in 0..bh {
        let o = z * s * d;
        expected[o..o + s * d].copy_from_slice(&fa_ref(&qf[o..o + s * d], &kf[o..o + s * d], &vf[o..o + s * d], s, d));
    }
    let report = allclose_f32(&got, &expected, 0.02 * (d as f32).sqrt(), 2e-2);
    assert!(report.ok, "FA gate bh={bh} S={s} d={d} FAILED before timing: {}", report.message);
    eprintln!("  gate bh={bh} S={s} d={d}: allclose ✓ (max_abs_err {:e})", report.max_abs_err);
}

/// The FA bench over REALISTIC `(b, h, S, d)` attention shapes (grid = `b·h·(S/128)` workgroups). At
/// B=H=1 the grid under-fills the 304-CU MI300X below S≈39k (the #5 finding); the realistic shapes fill
/// it, so this measures whether the 8-warp per-CU win (the ~80 TF #5 ceiling) is reached. Prints device
/// µs + REAL attention TF (`4·b·h·S²·d`); the numerics are gated once at a small shape up front.
fn bench_fa(c: &mut Criterion) {
    if !requirements_met() {
        eprintln!("svod-tk2 FA-forward bench: skipped (device is not a supported gfx942 GPU)");
        return;
    }
    eprintln!(
        "\n=== tk2 FA-forward: 8-warp split-Q, REAL TF over (b, h, S, d) attention shapes ===\n    \
         non-causal, single-buffer. Grid = b·h·(S/128) workgroups; realistic b·h fills the 304-CU MI300X.\n"
    );
    gate_small(3, 512, 64); // correctness sanity (the device test is the full gate)
    gate_small(2, 512, 128);

    // (label, b, h, S, d). The first two are B=H=1 (continuity: the under-fill + the large-n #5 ceiling);
    // the rest are realistic MI300X attention shapes whose grid fills the machine.
    let configs = [
        ("B=H=1  ", 1usize, 1usize, 4096usize, 128usize),
        ("B=H=1  ", 1, 1, 32768, 64),
        ("b2·h16 ", 2, 16, 2048, 128),
        ("b1·h32 ", 1, 32, 4096, 128),
        ("b4·h32 ", 4, 32, 2048, 64),
    ];
    let mut group = c.benchmark_group("tk2_fa");
    for (label, bb, hh, s, d) in configs {
        let bh = bb * hh;
        let flops = fa_flops_noncausal(bh, s, d);
        group.throughput(Throughput::Elements(flops));
        let (q, k, v) = (rand_bf16(&[bh * s, d]), rand_bf16(&[bh * s, d]), rand_bf16(&[bh * s, d]));
        let (_y, plan) = plan_of_fa(bh, s, d, &q, &k, &v);
        plan.execute().expect("execute");

        let iters = 50u64;
        let avg_ns = plan_gpu_ns(&plan, iters) as f64 / iters as f64;
        let tf = flops as f64 / avg_ns / 1e3; // FLOPs / avg_ns / 1e3 = TFLOP/s
        let wgs = bh * (s / 128);
        eprintln!(
            "  {label} b={bb} h={hh:>2} S={s:>5} d={d:>3}  ({wgs:>4} wgs): {:>9.1} µs  |  {tf:>7.1} TF",
            avg_ns / 1e3
        );
        group.bench_with_input(BenchmarkId::new(format!("fa/{label}_S{s}_d{d}"), bh), &bh, |bch, _| {
            bench_plan(bch, &plan)
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_fa
}
criterion_main!(benches);
