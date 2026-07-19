//! GPU-device-time bench for the **32×32×8 FA** ([`svod_tk2::kernels::fa::flash_attention_fwd_32`]) at
//! realistic attention shapes, plus the aiter-matched **bf16 O** variant. Each config is gated `allclose`
//! at a small shape before timing; device time via the shared [`common`] harness.
//!
//! FA-32 rides the rolled ClusterCx KV pipeline with register-staged prefetch/commit, XOR-swizzled K,
//! padded-transposed V, d128 K double-buffering, a three-plane V rotation, and softmax EXP grouped under
//! the independent QK MFMA stream. A QK(0)-only warmup removes the empty seed softmax/PV for every
//! supported input; two-block domains use a direct warmup-to-epilogue transition, and ragged domains use
//! explicit lexical scopes around their loop/tail DAGs. The d64 path additionally remaps workgroups into
//! XCD-local Q-tile chunks for private-L2 K/V reuse. At d128 the production kernel ships the 8-wave two-crew
//! phase-stagger ping-pong over asm-opaque LDS movement (race-free where the compiler-visible path was not).
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk2 --bench fa32`

use svod_dtype::DType;
use svod_runtime::ExecutionPlan;
use svod_tensor::Tensor;

mod common;
use common::{plan_gpu_ns, rand_bf16, requirements_met};

use svod_tk2::kernels::fa::{flash_attention_fwd_32, flash_attention_fwd_32_bf16};
use svod_tk2::{SwizzlePass, graph_kernel};

/// The FA kernel under measurement. `N32` = the production 32×32×8 FA (d128 rides the 8-wave two-crew
/// phase-stagger ping-pong, d64 the single-crew path); `Bf16` = the same production config but with the
/// aiter-matched **bf16 O** store (half the O write bytes, d128).
#[derive(Copy, Clone, PartialEq, Debug)]
enum Fa {
    N32,
    Bf16,
}

impl Fa {
    fn name(self) -> &'static str {
        match self {
            Fa::N32 => "FA-32",
            Fa::Bf16 => "FA-32bf16",
        }
    }

    /// The output-O dtype: the bf16-O variant stores O as bf16, every other variant as f32.
    fn out_dtype(self) -> DType {
        match self {
            Fa::Bf16 => DType::BFloat16,
            _ => DType::Float32,
        }
    }
}

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

/// Prepare an execution plan for the given kernel variant.
fn plan_of(fa: Fa, bh: usize, s: usize, d: usize, q: &Tensor, k: &Tensor, v: &Tensor) -> (Tensor, ExecutionPlan) {
    let prog = match fa {
        // FA-32 rides SwizzlePass ONLY: the K tile swizzles (cols = d, power of 2); the gathers are already
        // `ds_read_b64` (`load_lds_vec_after`) so VectorizePass has no fusible scalar run (it only touches the
        // loop-invariant Q prologue — measured negligible). V keeps its padded pitch (non-power-of-2, no XOR).
        Fa::N32 => flash_attention_fwd_32(bh, s, d).apply(SwizzlePass),
        // bf16-O production FA-32: identical schedule, O truncated f32→bf16 at the store (half the O bytes).
        Fa::Bf16 => flash_attention_fwd_32_bf16(bh, s, d).apply(SwizzlePass),
    };
    let out = Tensor::empty(&[bh * s, d], fa.out_dtype());
    let mut y = graph_kernel(prog, out, &[q, k, v]).expect("wrap FA as graph node");
    let plan = y.prepare().expect("prepare FA execution plan");
    (y, plan)
}

/// Gate a variant's numerics at a small shape before timing (the device test is the full gate).
fn gate(fa: Fa, bh: usize, s: usize, d: usize) {
    use svod_tensor::testing::allclose_f32;
    let (q, k, v) = (rand_bf16(&[bh * s, d]), rand_bf16(&[bh * s, d]), rand_bf16(&[bh * s, d]));
    let (y, plan) = plan_of(fa, bh, s, d, &q, &k, &v);
    plan.execute().expect("execute for gate");
    plan.output_buffer().expect("output buffer").synchronize().expect("sync before read");
    // Cast the output to f32 for the comparison (identity for the f32-O variants; the bf16→f32 read for Bf16).
    let got = as_f32(&y);
    let (qf, kf, vf) = (as_f32(&q), as_f32(&k), as_f32(&v));
    let mut expected = vec![0f32; bh * s * d];
    for z in 0..bh {
        let o = z * s * d;
        expected[o..o + s * d].copy_from_slice(&fa_ref(&qf[o..o + s * d], &kf[o..o + s * d], &vf[o..o + s * d], s, d));
    }
    // bf16-O rounding (RTZ) adds ≤2^-7 relative error, so Bf16 gets the widened rtol; f32-O keeps the tight one.
    let rtol = if fa == Fa::Bf16 { 3e-2 } else { 2e-2 };
    let report = allclose_f32(&got, &expected, 1e-2, rtol);
    let name = fa.name();
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

fn measure(config: Measurement<'_>, fa: Fa, q: &Tensor, k: &Tensor, v: &Tensor) -> f64 {
    let Measurement { label, bh, s, d } = config;
    let (_y, plan) = plan_of(fa, bh, s, d, q, k, v);
    plan.execute().expect("execute");
    let iters = 50u64;
    let avg_ns = plan_gpu_ns(&plan, iters) as f64 / iters as f64;
    profile_pmc(&plan, &format!("{} {} S{s} d{d}", fa.name(), label.trim()));
    fa_flops_noncausal(bh, s, d) as f64 / avg_ns / 1e3 // TF
}

fn main() {
    if !requirements_met() {
        eprintln!("svod-tk2 FA-32 bench: skipped (device is not a supported gfx942 GPU)");
        return;
    }
    eprintln!("\n=== tk2 FA-32: 32×32×8 rolled ClusterCx pipeline — REAL device TF ===\n");
    let input_dist = InputDist::from_env();
    eprintln!("  input distribution: {input_dist:?} (select with SVOD_FA_INPUT)\n");
    gate(Fa::N32, 2, 256, 128);
    gate(Fa::N32, 2, 1024, 128); // the d128 production ping-pong at a longer stream
    gate(Fa::Bf16, 2, 256, 128); // bf16-O production FA-32 (aiter-matched O store)

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
    eprintln!("  {:<9} {:>6} {:>4} {:>6}    {:>10}", "shape", "S", "d", "wgs", "FA-32 TF");
    for (label, bb, hh, s, d) in configs {
        let bh = bb * hh;
        let wgs = bh * (s / 256);
        let (q, k, v) = input_dist.inputs(&[bh * s, d]);
        let measurement = Measurement { label, bh, s, d };
        let tf32 = measure(measurement, Fa::N32, &q, &k, &v);
        eprintln!("  {label} {s:>6} {d:>4} {wgs:>6}    {tf32:>10.1}");
    }

    // ── bf16-O headline: the aiter-API match. FA-32 is memory-bound and O is ~11% of its traffic as f32,
    //    so a bf16 O store (half those bytes) removes ~5% of total DRAM traffic. Compare the f32-O production
    //    FA-32 (`N32`) against the bf16-O production FA-32 (`Bf16`) at matched d128 shapes under the SAME
    //    input, machine-filling (wgs = 256). Run once per distribution via SVOD_FA_INPUT (sym default, u01 =
    //    aiter's). ──
    eprintln!("\n  === FA-32 O dtype: f32 vs bf16 (aiter-API match) — {input_dist:?} input ===");
    eprintln!(
        "  {:<9} {:>6} {:>4} {:>6}    {:>15}  {:>15}  {:>8}",
        "shape", "S", "d", "wgs", "FA-32(f32-O) TF", "FA-32bf16 TF", "gain"
    );
    for (label, bb, hh, s, d) in
        [("b8·h16 ", 8usize, 16usize, 512usize, 128usize), ("b4·h16 ", 4, 16, 1024, 128), ("b2·h16 ", 2, 16, 2048, 128)]
    {
        let bh = bb * hh;
        let wgs = bh * (s / 256);
        let (q, k, v) = input_dist.inputs(&[bh * s, d]);
        let measurement = Measurement { label, bh, s, d };
        let tf_f32 = measure(measurement, Fa::N32, &q, &k, &v);
        let tf_bf16 = measure(measurement, Fa::Bf16, &q, &k, &v);
        let gain = (tf_bf16 / tf_f32 - 1.0) * 100.0;
        eprintln!("  {label} {s:>6} {d:>4} {wgs:>6}    {tf_f32:>15.1}  {tf_bf16:>15.1}  {gain:>+7.1}%");
    }
}
