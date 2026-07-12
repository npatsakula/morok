//! Shared harness for svod-tk2's criterion benches (currently `matmul`). Each bench
//! drives a tk2 kernel through its graph-node [`Tensor`] (`graph_kernel` → `prepare`
//! → `plan.profile`): GPU device time comes from per-kernel HW stamps, so criterion's
//! outlier rejection / CIs operate on real on-device time, not host wall-clock. Under
//! `cargo bench --profile-time`, the full layered profiler (roofline / occupancy /
//! PMC, configured via [`ProfileOptions::from_env`]) is captured and rendered. A
//! tk-free port of `tk/benches/common.rs` — tk2 depends on runtime/tensor/device,
//! never tk.
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk2 --bench matmul`
//! Benches self-skip (record no samples) when the device is not a supported gfx942 GPU.

// Shared harness: each bench binary re-compiles this module and uses only a subset of it
// (e.g. the DCC benches don't call `rand_bf16`), so per-binary dead-code is expected.
#![allow(dead_code)]

use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use criterion::Bencher;
use criterion::profiler::Profiler;
use svod_dtype::{AmdArch, DType, DeviceSpec};
use svod_runtime::{ExecutionPlan, PmcSelection, ProfileOptions, RunProfile};
use svod_tensor::Tensor;

/// A realized random bf16 tensor on the env-selected device, centered in `[-1, 1)` —
/// `2·x − 1` over a native-bf16 [`Tensor::rand_with`] `[0,1)` draw. Centered (signed) data
/// is MANDATORY, not cosmetic: all-positive operands make the GEMM's C output concentrate
/// near `N/4`, which the GPU's delta-color compression (DCC) shrinks on write — inflating
/// throughput ~4–10% (worse at small N; ~+30% at all-ones). Measured effect + the fills:
/// the `matmul_dsl_dcc` bench.
pub fn rand_bf16(shape: &[usize]) -> Tensor {
    let dev = svod_dtype::default_device::default_device();
    let x = Tensor::rand_with(shape, DType::BFloat16, dev).expect("rand bf16");
    let two = Tensor::full(shape, 2.0f32, DType::BFloat16).expect("const 2");
    let one = Tensor::full(shape, 1.0f32, DType::BFloat16).expect("const 1");
    let mut t = x.try_mul(&two).expect("2*x").try_sub(&one).expect("2*x-1");
    t.realize().expect("realize");
    t
}

/// Whether the env-selected device is gfx942 with the AMD-LLVM toolchain. tk2's
/// matmul hardcodes the gfx942 16×16×16 MFMA (DESIGN.md §2.8), so the bench
/// self-skips elsewhere — `cargo bench` has no `#[ignore]`, so this replaces it.
pub fn requirements_met() -> bool {
    let DeviceSpec::Amd { device_id } = Tensor::empty(&[1], DType::Float32).device() else { return false };
    let Ok(arch) = svod_device::registry::resolve_amd_arch_from_topology(device_id) else { return false };
    arch == AmdArch::Gfx942 && svod_runtime::amd::has_amdgpu_target()
}

/// Sum GPU device time (ns) over `iters` replays of a prepared plan, via
/// `plan.profile`'s per-kernel HW stamps. Timing only — no static analysis or
/// counters (mirrors `tk/benches/common.rs::plan_gpu_ns`).
pub fn plan_gpu_ns(plan: &ExecutionPlan, iters: u64) -> u64 {
    let opts = ProfileOptions { iters: 1, static_analysis: false, counters: PmcSelection::None };
    let mut total = 0u64;
    for _ in 0..iters {
        let report = plan.profile(&opts).expect("plan.profile");
        // Pure on-device time: sum HW stamps only, skipping unstamped dispatches.
        for stage in &report.stages {
            for k in &stage.kernels {
                if let (Some(s), Some(e)) = (k.gpu_start_ns, k.gpu_end_ns) {
                    total += e - s;
                }
            }
        }
    }
    total
}

/// Bench `plan` by GPU device time. Under `cargo bench --profile-time`, also capture
/// the plan's full profile (roofline / occupancy / PMC via `ProfileOptions::from_env`)
/// into the shared [`bench_profiler`]. Plain runs are unaffected.
pub fn bench_plan(bencher: &mut Bencher<'_>, plan: &ExecutionPlan) {
    bench_profiler().maybe_capture(plan);
    bencher.iter_custom(|iters| Duration::from_nanos(black_box(plan_gpu_ns(plan, iters))));
}

/// Process-global profiler shared between criterion (via `Criterion::with_profiler`)
/// and the bench routines (which capture into it).
pub fn bench_profiler() -> PlanProfiler {
    static P: OnceLock<PlanProfiler> = OnceLock::new();
    P.get_or_init(PlanProfiler::default).clone()
}

/// Criterion `--profile-time` hook: while profiling one benchmark, profile its plan on
/// every invocation and accumulate by per-kernel min ([`RunProfile::merge_min`]). On
/// stop, render the merged table to `<dir>/svod-profile.txt` and echo it to stderr.
#[derive(Clone, Default)]
pub struct PlanProfiler {
    active: Arc<AtomicBool>,
    result: Arc<Mutex<Option<RunProfile>>>,
}

impl PlanProfiler {
    fn maybe_capture(&self, plan: &ExecutionPlan) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        let Ok(run) = plan.profile(&ProfileOptions::from_env()) else { return };
        let mut slot = self.result.lock().expect("profile slot");
        match slot.as_mut() {
            Some(acc) => acc.merge_min(run),
            None => *slot = Some(run),
        }
    }
}

impl Profiler for PlanProfiler {
    fn start_profiling(&mut self, _id: &str, _dir: &Path) {
        *self.result.lock().expect("profile slot") = None;
        self.active.store(true, Ordering::Relaxed);
    }

    fn stop_profiling(&mut self, id: &str, dir: &Path) {
        self.active.store(false, Ordering::Relaxed);
        if let Some(report) = self.result.lock().expect("profile slot").take() {
            let table = report.render_table();
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(dir.join("svod-profile.txt"), &table);
            eprintln!("svod profile [{id}]:\n{table}");
        }
    }
}
