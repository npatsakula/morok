//! Criterion GPU-device-time bench comparing the **32×32×8 FA** ([`svod_tk2::kernels::fa::flash_attention_fwd_32`],
//! §Step 6) against the frozen **16×16×16 FA** ([`svod_tk2::kernels::fa::flash_attention_fwd`]) at matched
//! attention shapes — the "does 32×32×8 close the gap" headline. Both are gated `allclose` at a small shape
//! before timing; device time via the shared [`common`] harness.
//!
//! FA-32 rides the rolled ClusterCx KV pipeline with register-staged prefetch/commit, XOR-swizzled K,
//! padded-transposed V, d128 K double-buffering, a three-plane V rotation, and softmax EXP grouped under
//! the independent QK MFMA stream. A QK(0)-only warmup removes the empty seed softmax/PV for every
//! supported input; two-block domains use a direct warmup-to-epilogue transition, and ragged domains use
//! explicit lexical scopes around their loop/tail DAGs. The d64 path additionally remaps workgroups into
//! XCD-local Q-tile chunks for private-L2 K/V reuse. Phase staggering remains disabled because the
//! compiler-visible LDS path is not race-free under the asymmetric barrier schedule.
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk2 --bench fa32`

use svod_dtype::DType;
use svod_runtime::ExecutionPlan;
use svod_tensor::Tensor;

mod common;
use common::{plan_gpu_ns, rand_bf16, requirements_met};

use svod_tk2::kernels::fa::{flash_attention_fwd, flash_attention_fwd_32};
use svod_tk2::{SwizzlePass, VectorizePass, graph_kernel};

#[derive(Copy, Clone, Debug)]
enum InputDist {
    Sym,
    Normal,
    Normal10,
    U01,
    Zeros,
    Ones,
}

impl InputDist {
    fn from_env() -> Self {
        match std::env::var("SVOD_FA_INPUT").as_deref().unwrap_or("sym") {
            "sym" => Self::Sym,
            "normal" => Self::Normal,
            "normal10" => Self::Normal10,
            "u01" => Self::U01,
            "zeros" => Self::Zeros,
            "ones" => Self::Ones,
            value => panic!("SVOD_FA_INPUT must be one of sym, normal, normal10, u01, zeros, ones; got {value}"),
        }
    }

    fn make(self, shape: &[usize]) -> Tensor {
        let mut t = match self {
            Self::Sym => rand_bf16(shape),
            Self::Normal => Tensor::randn(shape).expect("normal input").cast(DType::BFloat16).expect("normal -> bf16"),
            Self::Normal10 => {
                let x = Tensor::randn(shape).expect("normal input");
                let scale = Tensor::full(shape, 10.0f32, DType::Float32).expect("normal scale");
                x.try_mul(&scale).expect("normal * 10").cast(DType::BFloat16).expect("normal10 -> bf16")
            }
            Self::U01 => Tensor::rand_with(shape, DType::BFloat16, svod_dtype::default_device::default_device())
                .expect("uniform [0,1) input"),
            Self::Zeros => Tensor::zeros(shape, DType::BFloat16).expect("zero input"),
            Self::Ones => Tensor::ones(shape, DType::BFloat16).expect("one input"),
        };
        t.realize().expect("realize FA input");
        t
    }

    fn inputs(self, shape: &[usize]) -> (Tensor, Tensor, Tensor) {
        (self.make(shape), self.make(shape), self.make(shape))
    }
}

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
        // FA-32 rides SwizzlePass ONLY: the K tile swizzles (cols = d, power of 2); the gathers are already
        // `ds_read_b64` (`load_lds_vec_after`) so VectorizePass has no fusible scalar run (it only touches the
        // loop-invariant Q prologue — measured negligible). V keeps its padded pitch (non-power-of-2, no XOR).
        flash_attention_fwd_32(bh, s, d).apply(SwizzlePass)
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

/// When `SVOD_PMC` requests Tier-4 counters, profile `plan` via [`ProfileOptions::from_env`] and echo
/// the rendered table — the plain-`main` bench's equivalent of the criterion `--profile-time`
/// `PlanProfiler` hook (`fa.rs`/`common.rs`). No-op on a bare `cargo bench` (empty/`0` selection).
fn profile_pmc(plan: &ExecutionPlan, label: &str) {
    let want = std::env::var("SVOD_PMC").map(|v| !matches!(v.trim(), "" | "0")).unwrap_or(false);
    if !want {
        return;
    }
    match plan.profile(&svod_runtime::ProfileOptions::from_env()) {
        Ok(report) => eprintln!("\nsvod PMC [{label}]:\n{}", report.render_table()),
        Err(e) => eprintln!("  PMC profile [{label}] failed: {e}"),
    }
}

#[derive(Copy, Clone)]
struct Measurement<'a> {
    label: &'a str,
    bh: usize,
    s: usize,
    d: usize,
}

fn measure(config: Measurement<'_>, wide: bool, q: &Tensor, k: &Tensor, v: &Tensor) -> f64 {
    let Measurement { label, bh, s, d } = config;
    let (_y, plan) = plan_of(wide, bh, s, d, q, k, v);
    plan.execute().expect("execute");
    let iters = 50u64;
    let avg_ns = plan_gpu_ns(&plan, iters) as f64 / iters as f64;
    // Counters for the FA-32 kernel under study (the 16×16 baseline is profiled by the `fa` bench).
    if wide {
        profile_pmc(&plan, &format!("FA-32 {} S{s} d{d}", label.trim()));
    }
    fa_flops_noncausal(bh, s, d) as f64 / avg_ns / 1e3 // TF
}

fn main() {
    if !requirements_met() {
        eprintln!("svod-tk2 FA-32 bench: skipped (device is not a supported gfx942 GPU)");
        return;
    }
    eprintln!("\n=== tk2 FA: 32×32×8 (rolled ClusterCx pipeline) vs 16×16×16 (tuned pipeline) — REAL device TF ===\n");
    let input_dist = InputDist::from_env();
    eprintln!("  input distribution: {input_dist:?} (select with SVOD_FA_INPUT)\n");
    gate(true, 2, 256, 128);
    gate(false, 2, 256, 128);

    // (label, b, h, S, d) — machine-filling large-S shapes plus short-context ones. All S are ≥256 and
    // 256-multiples (the 8-warp Q block), so the raw `bh·S` bench buffers cover every workgroup's fill.
    // `wgs = bh·S/256` fills the 304-CU MI300X.
    let configs = [
        ("b2·h16 ", 2usize, 16usize, 2048usize, 128usize),
        ("b2·h16 ", 2, 16, 2048, 64),
        ("b4·h16 ", 4, 16, 1024, 128),
        ("b8·h16 ", 8, 16, 512, 128),
        ("b16·h16", 16, 16, 256, 128),
        ("b8·h16 ", 8, 16, 256, 128),
        ("b16·h16", 16, 16, 256, 64),
        ("b8·h16 ", 8, 16, 256, 64),
    ];
    eprintln!(
        "  {:<9} {:>6} {:>4} {:>6}    {:>10}  {:>10}  {:>8}",
        "shape", "S", "d", "wgs", "FA-16 TF", "FA-32 TF", "ratio"
    );
    for (label, bb, hh, s, d) in configs {
        let bh = bb * hh;
        let wgs = bh * (s / 256);
        let (q, k, v) = input_dist.inputs(&[bh * s, d]);
        let measurement = Measurement { label, bh, s, d };
        let tf16 = measure(measurement, false, &q, &k, &v);
        let tf32 = measure(measurement, true, &q, &k, &v);
        eprintln!("  {label} {s:>6} {d:>4} {wgs:>6}    {tf16:>10.1}  {tf32:>10.1}  {:>7.2}x", tf32 / tf16);
    }
}
