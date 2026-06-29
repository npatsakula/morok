//! Standalone **vendor-floor** benches (hipBLASLt GEMM) to compare against the
//! svod `tk` rows. This is its own bench binary and uses **no svod AMD backend**:
//! svod is KFD-direct and HIP cannot enumerate the GPU alongside it in one
//! process (HIP reports 0 devices once svod is live), and svod's device VAs
//! aren't valid in HIP's context. So the shim allocates its own HIP buffers and
//! here we pass only shapes. Compare the device-ns rows below to the `tk` rows
//! from `cargo bench --bench {matmul,kmeans,knn}` — criterion merges them by
//! group name.
//!
//! Timing is HIP-event device time over the matmul only (allocation is one-time
//! setup, excluded) — the same class of on-device measurement as the tk rows'
//! PM4 stamps. Needs `libsvod_hipblaslt_shim.so` on `LD_LIBRARY_PATH` (the flake
//! `bench` dev shell); a missing shim or no GPU self-skips the bench. Run:
//!   nix develop .#bench --command cargo bench -p svod-tk --bench vendor

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use libloading::Library;

// FFI contract — matches the extern "C" shim in benches/shims/hipblaslt_shim.cpp.
type HblDeviceOk = unsafe extern "C" fn() -> c_int;
type HblCreate = unsafe extern "C" fn(i64, i64, i64, c_int, c_int, u64) -> *mut c_void;
type HblRunNs = unsafe extern "C" fn(*mut c_void, c_int, c_int) -> f64;
type HblDestroy = unsafe extern "C" fn(*mut c_void);
// Complete knn floor: hipBLASLt GEMM + rocPRIM segmented sort (N, M, D, max_ws).
type KnnCreate = unsafe extern "C" fn(i64, i64, i64, u64) -> *mut c_void;
type KnnRunNs = unsafe extern "C" fn(*mut c_void, c_int, c_int) -> f64;
type KnnDestroy = unsafe extern "C" fn(*mut c_void);
// Complete kmeans-assign floor: hipBLASLt GEMM + rocPRIM segmented arg-min (N, K, D, max_ws).
type KmeansCreate = unsafe extern "C" fn(i64, i64, i64, u64) -> *mut c_void;
type KmeansRunNs = unsafe extern "C" fn(*mut c_void, c_int, c_int) -> f64;
type KmeansDestroy = unsafe extern "C" fn(*mut c_void);

struct HblShim {
    _lib: Library,
    device_ok: HblDeviceOk,
    create: HblCreate,
    run_ns: HblRunNs,
    destroy: HblDestroy,
    knn_create: KnnCreate,
    knn_run_ns: KnnRunNs,
    knn_destroy: KnnDestroy,
    kmeans_create: KmeansCreate,
    kmeans_run_ns: KmeansRunNs,
    kmeans_destroy: KmeansDestroy,
}

fn load_hbl() -> Option<HblShim> {
    // SAFETY: a trusted, flake-built lib whose `extern "C"` symbols match the
    // signatures above. fn-pointers are Copy + 'static, so `*sym` copies them out
    // before `lib` is moved into the struct (which keeps it loaded).
    unsafe {
        let lib = Library::new("libsvod_hipblaslt_shim.so").ok()?;
        let device_ok = *lib.get::<HblDeviceOk>(b"hbl_device_ok\0").ok()?;
        let create = *lib.get::<HblCreate>(b"hbl_gemm_create\0").ok()?;
        let run_ns = *lib.get::<HblRunNs>(b"hbl_gemm_run_ns\0").ok()?;
        let destroy = *lib.get::<HblDestroy>(b"hbl_gemm_destroy\0").ok()?;
        let knn_create = *lib.get::<KnnCreate>(b"knn_create\0").ok()?;
        let knn_run_ns = *lib.get::<KnnRunNs>(b"knn_run_ns\0").ok()?;
        let knn_destroy = *lib.get::<KnnDestroy>(b"knn_destroy\0").ok()?;
        let kmeans_create = *lib.get::<KmeansCreate>(b"kmeans_create\0").ok()?;
        let kmeans_run_ns = *lib.get::<KmeansRunNs>(b"kmeans_run_ns\0").ok()?;
        let kmeans_destroy = *lib.get::<KmeansDestroy>(b"kmeans_destroy\0").ok()?;
        Some(HblShim {
            _lib: lib,
            device_ok,
            create,
            run_ns,
            destroy,
            knn_create,
            knn_run_ns,
            knn_destroy,
            kmeans_create,
            kmeans_run_ns,
            kmeans_destroy,
        })
    }
}

/// The GEMM shim is usable: loaded AND HIP enumerates a device in this process.
fn gemm_ready() -> Option<&'static HblShim> {
    static S: OnceLock<Option<HblShim>> = OnceLock::new();
    let shim = S.get_or_init(load_hbl).as_ref()?;
    // SAFETY: a no-arg C predicate.
    if unsafe { (shim.device_ok)() } != 0 { Some(shim) } else { None }
}

const MAX_WS: u64 = 256 * 1024 * 1024; // hipBLASLt heuristic workspace cap
const WARMUP: c_int = 3;
// Mirrors TILE in shims/hipblaslt_shim.cpp: the shim tiles output n into ≤2048
// chunks to dodge the gfx1151 large-N fault, so rows with n above this run tiled
// and are labelled "vendor_tiled" (not a silent cap).
const VENDOR_TILE: i64 = 2048;

fn iters_to_c(iters: u64) -> c_int {
    iters.min(c_int::MAX as u64) as c_int
}

/// GEMM problem `C[m,n] = op(A)·op(B)`; transposes are hipBLASLt op flags
/// (0 = N, 1 = T), matching svod's `x·cᵀ` (trans_b = 1) without a materialised
/// transpose.
struct GemmShape {
    m: i64,
    n: i64,
    k: i64,
    trans_a: c_int,
    trans_b: c_int,
}

/// Bench one GEMM (bf16 in, f32 out) via the shim, row id `<base>` (or
/// `<base>_tiled` when the output is tiled to dodge the gfx1151 fault). Median
/// device ns from HIP events.
fn bench_gemm(group: &mut BenchmarkGroup<'_, WallTime>, shim: &HblShim, size: usize, s: GemmShape, base: &str) {
    // SAFETY: the shim self-allocates; create returns null on an unsupported
    // config / OOM, handled below.
    let plan = unsafe { (shim.create)(s.m, s.n, s.k, s.trans_a, s.trans_b, MAX_WS) };
    if plan.is_null() {
        eprintln!("svod-tk vendor bench: hipBLASLt has no solution for {}x{}x{}; skipping size {size}", s.m, s.n, s.k);
        return;
    }
    let id = if s.n > VENDOR_TILE {
        eprintln!("svod-tk {base}: n={} tiled into ≤{} chunks (hipBLASLt gfx1151 large-N fault)", s.n, VENDOR_TILE);
        format!("{base}_tiled")
    } else {
        base.to_string()
    };
    group.bench_with_input(BenchmarkId::new(id, size), &size, |b, _| {
        b.iter_custom(|iters| {
            // SAFETY: `plan` is valid for the bench's duration; device VAs are the
            // shim's own.
            let ns = unsafe { (shim.run_ns)(plan, WARMUP, iters_to_c(iters)) };
            Duration::from_nanos((ns * iters as f64) as u64)
        });
    });
    // SAFETY: `plan` came from `create` and is destroyed exactly once.
    unsafe { (shim.destroy)(plan) };
}

/// Bench the COMPLETE vendor knn for `[N,D]` queries vs `[M,D]` corpus: hipBLASLt
/// GEMM (x·cᵀ -> [N,M]) + rocPRIM segmented sort (the top-K). Row id `vendor`.
/// The shim column-tiles the GEMM internally for large M.
fn bench_knn_full(group: &mut BenchmarkGroup<'_, WallTime>, shim: &HblShim, n: i64, m: i64, d: i64) {
    // SAFETY: the shim self-allocates; null on unsupported config / OOM.
    let plan = unsafe { (shim.knn_create)(n, m, d, MAX_WS) };
    if plan.is_null() {
        eprintln!("svod-tk vendor knn: hipBLASLt+rocPRIM knn unavailable for N={n} M={m}; skipping");
        return;
    }
    group.bench_with_input(BenchmarkId::new("vendor", m as usize), &m, |b, _| {
        b.iter_custom(|iters| {
            // SAFETY: `plan` valid for the bench's duration.
            let ns = unsafe { (shim.knn_run_ns)(plan, WARMUP, iters_to_c(iters)) };
            Duration::from_nanos((ns * iters as f64) as u64)
        });
    });
    // SAFETY: from knn_create, destroyed once.
    unsafe { (shim.knn_destroy)(plan) };
}

/// Bench the COMPLETE vendor kmeans-assign for `[N,D]` points vs `[K,D]`
/// centroids: hipBLASLt GEMM (x·cᵀ -> [N,K] cross) + rocPRIM segmented arg-min
/// over K (the nearest-centroid assignment). Row id `vendor`. The shim
/// column-tiles the GEMM internally for large K.
fn bench_kmeans_full(group: &mut BenchmarkGroup<'_, WallTime>, shim: &HblShim, n: i64, k: i64, d: i64) {
    // SAFETY: the shim self-allocates; null on unsupported config / OOM.
    let plan = unsafe { (shim.kmeans_create)(n, k, d, MAX_WS) };
    if plan.is_null() {
        eprintln!("svod-tk vendor kmeans: hipBLASLt+rocPRIM kmeans-assign unavailable for N={n} K={k}; skipping");
        return;
    }
    group.bench_with_input(BenchmarkId::new("vendor", k as usize), &k, |b, _| {
        b.iter_custom(|iters| {
            // SAFETY: `plan` valid for the bench's duration.
            let ns = unsafe { (shim.kmeans_run_ns)(plan, WARMUP, iters_to_c(iters)) };
            Duration::from_nanos((ns * iters as f64) as u64)
        });
    });
    // SAFETY: from kmeans_create, destroyed once.
    unsafe { (shim.kmeans_destroy)(plan) };
}

/// Square `C[n,n] = A·B` — the hipBLASLt floor for `svod_tk::matmul`.
fn bench_matmul(c: &mut Criterion) {
    let Some(shim) = gemm_ready() else {
        eprintln!("svod-tk vendor matmul: skipped (no hipBLASLt shim / GPU)");
        return;
    };
    let mut group = c.benchmark_group("matmul");
    for &n in &[1024usize, 2048] {
        group.throughput(Throughput::Elements((2.0 * (n as f64).powi(3)) as u64)); // 2·M·N·K
        let i = n as i64;
        bench_gemm(&mut group, shim, n, GemmShape { m: i, n: i, k: i, trans_a: 0, trans_b: 0 }, "vendor");
    }
    group.finish();
}

/// kmeans-assign floor: `vendor` = complete hipBLASLt GEMM (`x·cᵀ -> [N,K] cross`)
/// then rocPRIM segmented arg-min over K — apples-to-apples vs `svod_tk::kmeans_assign`,
/// which fuses GEMM with the nearest-centroid argmin. `vendor_gemm` = the cross GEMM
/// only, so the GEMM-vs-argmin split is visible on the vendor side too.
fn bench_kmeans(c: &mut Criterion) {
    let Some(shim) = gemm_ready() else {
        eprintln!("svod-tk vendor kmeans: skipped (no hipBLASLt shim / GPU)");
        return;
    };
    let (n, d) = (2048usize, 64usize);
    let mut group = c.benchmark_group("kmeans");
    for &k in &[64usize, 256, 1024, 4096] {
        group.throughput(Throughput::Elements((2.0 * (n * k * d) as f64) as u64));
        // Complete vendor kmeans-assign: hipBLASLt GEMM + rocPRIM segmented arg-min.
        bench_kmeans_full(&mut group, shim, n as i64, k as i64, d as i64);
        // Cross GEMM score floor only (no argmin), for reference / the split.
        bench_gemm(
            &mut group,
            shim,
            k,
            GemmShape { m: n as i64, n: k as i64, k: d as i64, trans_a: 0, trans_b: 1 },
            "vendor_gemm",
        );
    }
    group.finish();
}

/// knn floor: `vendor` = complete hipBLASLt GEMM + rocPRIM segmented-sort top-K
/// (apples-to-apples vs `svod_tk::knn`); `vendor_gemm` = the score GEMM only, so
/// the GEMM-vs-topK split is visible on the vendor side too.
fn bench_knn(c: &mut Criterion) {
    let Some(shim) = gemm_ready() else {
        eprintln!("svod-tk vendor knn: skipped (no hipBLASLt shim / GPU)");
        return;
    };
    let (n, d) = (2048usize, 64usize);
    let mut group = c.benchmark_group("knn");
    for &m in &[512usize, 1024, 2048, 16384] {
        group.throughput(Throughput::Elements((2.0 * (n * m * d) as f64) as u64));
        // Complete vendor knn: hipBLASLt GEMM + rocPRIM segmented sort (top-K).
        bench_knn_full(&mut group, shim, n as i64, m as i64, d as i64);
        // GEMM score floor only (no top-K), for reference / the split.
        bench_gemm(
            &mut group,
            shim,
            m,
            GemmShape { m: n as i64, n: m as i64, k: d as i64, trans_a: 0, trans_b: 1 },
            "vendor_gemm",
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_matmul, bench_kmeans, bench_knn
}
criterion_main!(benches);
