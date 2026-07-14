//! Criterion GPU-device-time bench comparing the **32×32×8 FA** ([`svod_tk2::kernels_fa::flash_attention_fwd_32`],
//! §Step 6) against the frozen **16×16×16 FA** ([`svod_tk2::kernels_fa::flash_attention_fwd`]) at matched
//! attention shapes — the "does 32×32×8 close the gap" headline. Both are gated `allclose` at a small shape
//! before timing; device time via the shared [`common`] harness.
//!
//! HONEST caveat: `flash_attention_fwd_32` is the correctness-first ASSEMBLY (unrolled KV stream, scalar
//! global-load fills, no LDS swizzle, no ClusterCx pipeline / ping-pong). The 16×16 FA rides the tuned
//! pipeline (`VectorizePass.then(SwizzlePass)`, register-staged prefetch). This measures the un-tuned wide
//! core vs the tuned narrow core — a directional number, not the 32×32×8 ceiling.
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk2 --bench fa32`

use svod_dtype::DType;
use svod_runtime::ExecutionPlan;
use svod_tensor::Tensor;

mod common;
use common::{plan_gpu_ns, rand_bf16, requirements_met};

use svod_tk2::kernels_fa::{flash_attention_fwd, flash_attention_fwd_32};
use svod_tk2::{SwizzlePass, VectorizePass, graph_kernel};

fn fa_flops_noncausal(bh: usize, s: usize, d: usize) -> u64 {
    4 * (bh as u64) * (s as u64) * (s as u64) * (d as u64)
}

fn as_f32(t: &Tensor) -> Vec<f32> {
    let mut f = t.cast(DType::Float32).expect("→f32");
    f.realize().expect("realize f32");
    f.as_vec::<f32>().expect("read f32")
}

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

/// Prepare an execution plan for the given kernel variant (`wide` → FA-32, else the tuned 16×16 FA).
fn plan_of(wide: bool, bh: usize, s: usize, d: usize, q: &Tensor, k: &Tensor, v: &Tensor) -> (Tensor, ExecutionPlan) {
    let prog = if wide {
        flash_attention_fwd_32(bh, s, d)
    } else {
        flash_attention_fwd(bh, s, d).apply(VectorizePass).apply(SwizzlePass)
    };
    let out = Tensor::empty(&[bh * s, d], DType::Float32);
    let mut y = graph_kernel(prog, out, &[q, k, v]).expect("wrap FA as graph node");
    let plan = y.prepare().expect("prepare FA execution plan");
    (y, plan)
}

/// Gate a variant's numerics at a small shape before timing (the device test is the full gate).
fn gate(wide: bool, bh: usize, s: usize, d: usize) {
    use svod_tensor::testing::allclose_f32;
    let (q, k, v) = (rand_bf16(&[bh * s, d]), rand_bf16(&[bh * s, d]), rand_bf16(&[bh * s, d]));
    let (y, plan) = plan_of(wide, bh, s, d, &q, &k, &v);
    plan.execute().expect("execute for gate");
    plan.output_buffer().expect("output buffer").synchronize().expect("sync before read");
    let got = y.as_vec::<f32>().expect("read FA output");
    let (qf, kf, vf) = (as_f32(&q), as_f32(&k), as_f32(&v));
    let mut expected = vec![0f32; bh * s * d];
    for z in 0..bh {
        let o = z * s * d;
        expected[o..o + s * d].copy_from_slice(&fa_ref(&qf[o..o + s * d], &kf[o..o + s * d], &vf[o..o + s * d], s, d));
    }
    let report = allclose_f32(&got, &expected, 1e-2, 2e-2);
    let name = if wide { "FA-32" } else { "FA-16" };
    assert!(report.ok, "{name} gate bh={bh} S={s} d={d} FAILED before timing: {}", report.message);
    eprintln!("  gate {name} bh={bh} S={s} d={d}: allclose ✓ (max_abs_err {:e})", report.max_abs_err);
}

fn measure(wide: bool, bh: usize, s: usize, d: usize) -> f64 {
    let (q, k, v) = (rand_bf16(&[bh * s, d]), rand_bf16(&[bh * s, d]), rand_bf16(&[bh * s, d]));
    let (_y, plan) = plan_of(wide, bh, s, d, &q, &k, &v);
    plan.execute().expect("execute");
    let iters = 50u64;
    let avg_ns = plan_gpu_ns(&plan, iters) as f64 / iters as f64;
    fa_flops_noncausal(bh, s, d) as f64 / avg_ns / 1e3 // TF
}

fn main() {
    if !requirements_met() {
        eprintln!("svod-tk2 FA-32 bench: skipped (device is not a supported gfx942 GPU)");
        return;
    }
    eprintln!("\n=== tk2 FA: 32×32×8 (rolled ClusterCx pipeline) vs 16×16×16 (tuned pipeline) — REAL device TF ===\n");
    gate(true, 2, 128, 128);
    gate(false, 2, 128, 128);

    // (label, b, h, S, d) — the rolled FA-32 now scales past n=128 (large-S machine-filling shapes), plus
    // the old short-context shapes for the before→after comparison. `wgs = bh·S/128` fills the 304-CU MI300X.
    let configs = [
        ("b2·h16 ", 2usize, 16usize, 2048usize, 128usize),
        ("b2·h16 ", 2, 16, 2048, 64),
        ("b4·h16 ", 4, 16, 1024, 128),
        ("b8·h16 ", 8, 16, 512, 128),
        ("b16·h16", 16, 16, 128, 128),
        ("b8·h16 ", 8, 16, 256, 128),
        ("b16·h16", 16, 16, 128, 64),
        ("b8·h16 ", 8, 16, 256, 64),
    ];
    eprintln!(
        "  {:<9} {:>6} {:>4} {:>6}    {:>10}  {:>10}  {:>8}",
        "shape", "S", "d", "wgs", "FA-16 TF", "FA-32 TF", "ratio"
    );
    for (label, bb, hh, s, d) in configs {
        let bh = bb * hh;
        let wgs = bh * (s / 128);
        let tf16 = measure(false, bh, s, d);
        let tf32 = measure(true, bh, s, d);
        eprintln!("  {label} {s:>6} {d:>4} {wgs:>6}    {tf16:>10.1}  {tf32:>10.1}  {:>7.2}x", tf32 / tf16);
    }
}
