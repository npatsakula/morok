//! SCRATCH bench — DCC / delta-color-compression inflation of svod-tk's hand-written asm
//! GEMM (`gemm_core_asm`, the `BLOCK256_CFG` path at large square N). Times
//! `svod_tk::matmul` (square M=N=K) at N ∈ {4096, 8192} under THREE input-data fills:
//!   * `u01`  — uniform [0,1) bf16 (`rand_with`); all-positive, reproduces the harness.
//!   * `sym`  — centered [-1,1) bf16 (`2*x - 1`); honest signed data.
//!   * `ones` — all 1.0 bf16 (constant C ⇒ maximally DCC-compressible output).
//!
//! Device time via `common::bench_plan` (PM4 HW stamps) plus an explicit MIN over replays
//! (`common::plan_gpu_ns`) printed per (N, fill) as `[DCC]` lines / a summary table.
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk --bench matmul_dcc`
//!
//! This is the DCC-provenance rig: `sym` is the honest number; `u01`/`ones` quantify the
//! memory-compression inflation that made the vendor/HK/"805 TF" numbers untrustworthy. It
//! is WHY `common::rand_bf16` now fills centered `[-1,1)`. See `tk2/DESIGN.md` (2026-07-12).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_tensor::Tensor;

mod common;
use common::{bench_plan, plan_gpu_ns, requirements_met};

/// The env-selected device (`SVOD_DEVICE`, else default) — same source as `common::rand_bf16`.
fn dev() -> svod_dtype::DeviceSpec {
    svod_dtype::default_device::default_device()
}

/// The three fills, each a realized (materialized-on-device) `[n, n]` bf16 tensor.
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
            // Centered [-1,1): 2*x - 1 over a device rand tensor (depends on x's buffer, so
            // this realizes into a fresh [n,n] buffer — not a stride-0 const).
            Fill::Sym => {
                let x = Tensor::rand_with(&[n, n], DType::BFloat16, dev()).expect("rand for sym");
                let two = Tensor::full(&[n, n], 2.0f32, DType::BFloat16).expect("const 2");
                let one = Tensor::full(&[n, n], 1.0f32, DType::BFloat16).expect("const 1");
                x.try_mul(&two).expect("2*x").try_sub(&one).expect("2*x-1")
            }
            // All 1.0: a pure const; `realize()` wraps it in CONTIGUOUS and materializes a
            // real [n,n] GPU buffer (deviceless const → scheduler falls back to default_device).
            Fill::Ones => Tensor::ones(&[n, n], DType::BFloat16).expect("ones"),
        };
        t.realize().expect("realize fill");
        t
    }
}

/// Print min/max/first-few of a tiny 4×4 of each fill (cast to f32 to read) — CONFIRM the
/// values are actually [0,1) / [-1,1) / all-ones before trusting the big-N timings.
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

/// Confirm `svod_tk::matmul` dispatched the hand asm `gemm_core` kernel (not a fallback):
/// profile once, print the entry point + whether the generated code carries inline MFMA asm.
fn confirm_asm_dispatch(plan: &svod_runtime::ExecutionPlan) {
    use svod_runtime::{PmcSelection, ProfileOptions};
    let opts = ProfileOptions { iters: 1, static_analysis: false, counters: PmcSelection::None };
    let report = plan.profile(&opts).expect("profile for dispatch check");
    for stage in &report.stages {
        for k in &stage.kernels {
            let code = &k.kernel.code;
            let has_mfma = code.contains("mfma") || code.contains("v_mfma");
            let has_asm = code.contains("asm") || code.contains("inline");
            eprintln!(
                "[DCC] dispatch: entry_point={:?} device={:?} bufs={} code_len={} mfma_asm={} asm_marker={}",
                k.kernel.entry_point,
                k.device,
                k.num_buffers,
                code.len(),
                has_mfma,
                has_asm
            );
        }
    }
}

/// The DCC bench: for each N and fill, time `svod_tk::matmul(a, b)` (both operands the
/// same fill) by device time. Reports the MIN device time / TFLOPS explicitly and also
/// runs the criterion `bench_plan` harness for CI/outlier-rejected numbers.
fn bench_matmul_dcc(c: &mut Criterion) {
    if !requirements_met(svod_tk::kernels::matmul::MATMUL_SUPPORTED_ARCHS) {
        eprintln!("svod-tk matmul_dcc bench: skipped (no supported AMD GPU / toolchain)");
        return;
    }

    spot_check();

    let mut summary: Vec<(usize, &'static str, f64, f64)> = Vec::new();
    let mut checked_dispatch = false;
    let mut group = c.benchmark_group("matmul_dcc");
    for &n in &[4096usize, 8192] {
        let flop = 2.0 * (n as f64).powi(3);
        group.throughput(Throughput::Elements(flop as u64));
        for fill in [Fill::U01, Fill::Sym, Fill::Ones] {
            let a = fill.make(n);
            let b = fill.make(n);
            let mut cc = match svod_tk::matmul(&a, &b) {
                Ok(Some(t)) => t,
                Ok(None) => {
                    eprintln!("[DCC] N={n} fill={}: matmul returned Ok(None) — kernel not applicable!", fill.name());
                    continue;
                }
                Err(e) => {
                    eprintln!("[DCC] N={n} fill={}: matmul ERROR: {e}", fill.name());
                    continue;
                }
            };
            let plan = cc.prepare().expect("prepare matmul");

            if !checked_dispatch {
                confirm_asm_dispatch(&plan);
                checked_dispatch = true;
            }

            // Explicit MIN device time over replays (warm up first, then take the min).
            for _ in 0..5 {
                let _ = plan_gpu_ns(&plan, 1);
            }
            let mut min_ns = u64::MAX;
            for _ in 0..100 {
                min_ns = min_ns.min(plan_gpu_ns(&plan, 1));
            }
            let us = min_ns as f64 / 1_000.0;
            let tf = flop / (min_ns as f64); // 2n³ / ns = TFLOP/s
            eprintln!("[DCC] N={n} fill={:>4} : min {us:8.1} us  {tf:6.1} TF", fill.name());
            summary.push((n, fill.name(), us, tf));

            group.bench_with_input(BenchmarkId::new(fill.name(), n), &n, |bch, _| bench_plan(bch, &plan));
        }
    }
    group.finish();

    eprintln!("\n[DCC] ================ SUMMARY (min device time) ================");
    eprintln!("[DCC]   N     fill      min_us      TF");
    for (n, fill, us, tf) in &summary {
        eprintln!("[DCC] {n:>5}   {fill:>4}   {us:9.1}   {tf:6.1}");
    }
    // Inflation factors relative to the honest `sym` fill, per N.
    for &n in &[4096usize, 8192] {
        let get = |f: &str| summary.iter().find(|(nn, ff, _, _)| *nn == n && *ff == f).map(|(_, _, _, tf)| *tf);
        if let (Some(u), Some(s), Some(o)) = (get("u01"), get("sym"), get("ones")) {
            eprintln!("[DCC] N={n}: sym={s:.1}TF  u01/sym={:.3}  ones/sym={:.3}", u / s, o / s);
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_matmul_dcc
}
criterion_main!(benches);
