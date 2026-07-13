//! SCRATCH bench — DCC / delta-color-compression inflation of tk2's **DSL** matmul kernel
//! (`matmul_lds_kblock_mw_clustered`, the production 256² config bm=128/bn=64/wm=2/wn=4/k_step=64,
//! 8 warps) at square N ∈ {4096, 8192} under THREE input-data fills:
//!   * `u01`  — uniform [0,1) bf16 (`rand_with`); all-positive ⇒ low-variance C ⇒ inflated.
//!   * `sym`  — centered [-1,1) bf16 (`2*x - 1`); honest signed data (the number to trust).
//!   * `ones` — all 1.0 bf16 (constant C ⇒ maximally DCC-compressible output).
//!
//! The kernel outputs **F32** C (A·Bᵀ, B stored [N,K]). Every (N, fill) is correctness-gated
//! (allclose vs an f32 reference over the SAME bf16-rounded operands). The bit-exact `clustered`
//! kernel is HARD-gated (a failure aborts the bench). Device time via `common::plan_gpu_ns`
//! (PM4 HW stamps) — explicit MIN over replays printed per row, plus criterion's `bench_plan`.
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk2 --bench matmul_dsl_dcc`
//!
//! The rig that answered DESIGN.md's "re-measure the DSL @8192 on [-1,1]" question: honest
//! `sym` DSL is ~473 TF, FLAT with N, vs tk-asm ~639 @8192 — the ~1.4× gap HOLDS and widens
//! at scale. See `tk2/DESIGN.md` (2026-07-12).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_runtime::ExecutionPlan;
use svod_tensor::Tensor;
use svod_tensor::testing::allclose_f32;

mod common;
use common::{bench_plan, plan_gpu_ns, requirements_met};

use svod_tk2::{Program, SwizzlePass, VectorizePass, graph_kernel, matmul_lds_kblock_mw_clustered};

/// The env-selected device (`SVOD_DEVICE`, else default) — same source as `common::rand_bf16`.
fn dev() -> svod_dtype::DeviceSpec {
    svod_dtype::default_device::default_device()
}

/// The three input-data fills.
#[derive(Clone, Copy)]
enum Fill {
    U01,
    Sym,
    Ones,
}

impl Fill {
    fn name(self) -> &'static str {
        match self {
            Fill::U01 => "u01",
            Fill::Sym => "sym",
            Fill::Ones => "ones",
        }
    }

    /// Build ONE realized bf16 `[n, n]` operand of this fill on the device.
    fn make(self, n: usize) -> Tensor {
        let mut t = match self {
            // Uniform [0,1) bf16 via `rand_with` — the all-positive fill the harness used
            // BEFORE the DCC fix (`common::rand_bf16` is now centered [-1,1)); kept here to
            // measure how much that old fill inflated the number.
            Fill::U01 => Tensor::rand_with(&[n, n], DType::BFloat16, dev()).expect("rand u01"),
            // Centered [-1,1): 2*x - 1 over a device rand tensor (realizes a fresh [n,n] buffer).
            Fill::Sym => {
                let x = Tensor::rand_with(&[n, n], DType::BFloat16, dev()).expect("rand for sym");
                let two = Tensor::full(&[n, n], 2.0f32, DType::BFloat16).expect("const 2");
                let one = Tensor::full(&[n, n], 1.0f32, DType::BFloat16).expect("const 1");
                x.try_mul(&two).expect("2*x").try_sub(&one).expect("2*x-1")
            }
            // All 1.0: a pure const; realize() materializes a real [n,n] GPU buffer.
            Fill::Ones => Tensor::ones(&[n, n], DType::BFloat16).expect("ones"),
        };
        t.realize().expect("realize fill");
        t
    }
}

/// The DSL matmul kernel under test (outputs F32 C, the production 256² config).
#[derive(Clone, Copy)]
enum Kernel {
    Clustered,
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Kernel::Clustered => "clustered",
        }
    }

    /// bm=128, bn=64, wm=2, wn=4, k_step=64 ⇒ 256×256 tile, 8 warps — the production config.
    fn build(self, n: usize) -> Program {
        match self {
            Kernel::Clustered => {
                matmul_lds_kblock_mw_clustered(n, n, n, 128, 64, 2, 4, 64).apply(VectorizePass).apply(SwizzlePass)
            }
        }
    }

    /// The bit-exact `clustered` kernel HARD-gates (abort on correctness fail).
    fn hard_gate(self) -> bool {
        true
    }
}

/// f32 ground truth `A·B` over the SAME bf16-rounded operands (kernel + reference both see
/// the realized bf16 values cast up to f32). Caller passes B already transposed for A·Bᵀ.
fn reference(a: &Tensor, b: &Tensor) -> Vec<f32> {
    let bf = b.cast(DType::Float32).expect("b→f32");
    let mut r = a.cast(DType::Float32).expect("a→f32").matmul(&bf).expect("reference matmul");
    r.realize().expect("realize reference");
    r.as_vec::<f32>().expect("read reference")
}

/// Wrap a tk2 matmul `Program` as a graph-node Tensor over `(a, b)` with a fresh f32 output
/// template, and prepare its execution plan.
fn plan_of(program: Program, m: usize, n: usize, a: &Tensor, b: &Tensor) -> (Tensor, ExecutionPlan) {
    let out = Tensor::empty(&[m, n], DType::Float32);
    let mut y = graph_kernel(program, out, &[a, b]).expect("wrap matmul as graph node");
    let plan = y.prepare().expect("prepare execution plan");
    (y, plan)
}

/// Correctness gate: execute once and allclose the wired output vs the f32 reference
/// (atol ≈ 0.02·√K, rtol = 2e-2). Returns `(ok, max_abs_err)`. On a `hard`-gated failure,
/// panics (a bit-exact kernel producing wrong output is a real regression); on a soft-gated
/// failure it reports the delta and continues (the documented pipe2 ping-pong race).
fn gate(y: &Tensor, plan: &ExecutionPlan, expected: &[f32], k: usize, label: &str, hard: bool) -> (bool, f32) {
    plan.execute().expect("execute for correctness");
    let got = y.as_vec::<f32>().expect("read output");
    let atol = 0.02 * (k as f32).sqrt();
    let report = allclose_f32(&got, expected, atol, 2e-2);
    if report.ok {
        eprintln!("[DCC] gate PASS: {label:<22} max_abs_err={:.4} (atol={:.3})", report.max_abs_err, atol);
    } else if hard {
        panic!("{label}: BIT-EXACT kernel failed correctness gate: {}", report.message);
    } else {
        eprintln!(
            "[DCC] gate SOFT-FAIL: {label:<18} max_abs_err={:.4} (atol={:.3}) — {}",
            report.max_abs_err, atol, report.message
        );
    }
    (report.ok, report.max_abs_err)
}

/// Print min/max/first-few of a tiny 4×4 of each fill — CONFIRM ranges before big-N timings.
fn spot_check() {
    eprintln!("[DCC] --- 4x4 value spot-check (as_vec::<f32> after cast) ---");
    for fill in [Fill::U01, Fill::Sym, Fill::Ones] {
        let t = fill.make(4);
        let mut f = t.cast(DType::Float32).expect("cast f32");
        f.realize().expect("realize tiny f32");
        let v = f.as_vec::<f32>().expect("as_vec");
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in &v {
            lo = lo.min(x);
            hi = hi.max(x);
        }
        eprintln!(
            "[DCC] {:>4}: n={} min={:.4} max={:.4} sample={:?}",
            fill.name(),
            v.len(),
            lo,
            hi,
            &v[..v.len().min(6)]
        );
    }
}

/// One measured row.
struct Row {
    kernel: &'static str,
    n: usize,
    fill: &'static str,
    us: f64,
    tf: f64,
    ok: bool,
    max_err: f32,
}

/// The DSL DCC bench.
fn bench_matmul_dsl_dcc(c: &mut Criterion) {
    if !requirements_met() {
        eprintln!("svod-tk2 matmul_dsl_dcc bench: skipped (device is not a supported gfx942 GPU)");
        return;
    }

    spot_check();

    let mut rows: Vec<Row> = Vec::new();
    let mut group = c.benchmark_group("matmul_dsl_dcc");

    for &n in &[4096usize, 8192] {
        let flop = 2.0 * (n as f64).powi(3);
        group.throughput(Throughput::Elements(flop as u64));
        for fill in [Fill::U01, Fill::Sym, Fill::Ones] {
            // Fresh random operands per fill; same fill for both A and B.
            let a = fill.make(n);
            let b = fill.make(n);
            // Kernel computes C = A·Bᵀ (B stored [N,K]) ⇒ reference transposes B.
            let expected = reference(&a, &b.try_transpose(0, 1).expect("Bᵀ for A·Bᵀ reference"));

            for kernel in [Kernel::Clustered] {
                let label = format!("{}_{}_n{}", kernel.name(), fill.name(), n);
                let prog = kernel.build(n);
                let (y, plan) = plan_of(prog, n, n, &a, &b);
                let (ok, max_err) = gate(&y, &plan, &expected, n, &label, kernel.hard_gate());

                // Explicit MIN device time over replays (warm up, then min of HW stamps).
                for _ in 0..5 {
                    let _ = plan_gpu_ns(&plan, 1);
                }
                let mut min_ns = u64::MAX;
                for _ in 0..100 {
                    min_ns = min_ns.min(plan_gpu_ns(&plan, 1));
                }
                let us = min_ns as f64 / 1_000.0;
                let tf = flop / (min_ns as f64 * 1_000.0); // FLOP/ns → /1000 = TFLOP/s
                eprintln!(
                    "[DCC] N={n:>4} {:>9} {:>4} : min {us:8.1} us  {tf:6.1} TF  ({})",
                    kernel.name(),
                    fill.name(),
                    if ok { "correct" } else { "RACY" }
                );
                rows.push(Row { kernel: kernel.name(), n, fill: fill.name(), us, tf, ok, max_err });

                let id = format!("{}_{}", kernel.name(), fill.name());
                group.bench_with_input(BenchmarkId::new(id, n), &n, |bch, _| bench_plan(bch, &plan));
            }
        }
    }
    group.finish();

    // ---- Summary ----
    eprintln!("\n[DCC] ================ SUMMARY (min device time) ================");
    eprintln!("[DCC]   kernel      N    fill    min_us       TF   correct  max_err");
    for r in &rows {
        eprintln!(
            "[DCC] {:>9}  {:>5}   {:>4}   {:9.1}   {:6.1}   {:>7}  {:.3}",
            r.kernel,
            r.n,
            r.fill,
            r.us,
            r.tf,
            if r.ok { "yes" } else { "RACY" },
            r.max_err
        );
    }

    let get = |k: &str, n: usize, f: &str| rows.iter().find(|r| r.kernel == k && r.n == n && r.fill == f).map(|r| r.tf);

    eprintln!("\n[DCC] ---- HONEST (sym) TF + inflation factors ----");
    for &n in &[4096usize, 8192] {
        for k in ["clustered"] {
            if let (Some(u), Some(s), Some(o)) = (get(k, n, "u01"), get(k, n, "sym"), get(k, n, "ones")) {
                eprintln!("[DCC] {k:>9} N={n}: sym={s:6.1}TF  u01={u:6.1}({:.3}x)  ones={o:6.1}({:.3}x)", u / s, o / s);
            }
        }
    }

    eprintln!("\n[DCC] ---- honest-best DSL vs landmarks (tk-asm / HK) ----");
    let landmarks = [(4096usize, 631.0, 520.0), (8192usize, 639.0, 613.0)];
    for (n, tkasm, hk) in landmarks {
        let best = get("clustered", n, "sym").unwrap_or(0.0);
        if best > 0.0 {
            eprintln!(
                "[DCC] N={n}: best-DSL sym={best:.1}TF | tk-asm={tkasm:.0} (DSL/tk-asm={:.3}, tk-asm/DSL={:.3}x) | HK={hk:.0} (DSL/HK={:.3})",
                best / tkasm,
                tkasm / best,
                best / hk
            );
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_matmul_dsl_dcc
}
criterion_main!(benches);
