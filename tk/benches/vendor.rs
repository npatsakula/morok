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

// ── CK ck_tile FMHA-forward vendor floor (separate .so: libsvod_ck_fmha_shim.so) ──
// create(batch, nhead, nhead_kv, seqlen, hdim, causal) — self-allocates q/k/v/o; null when
// CK has no instance for the shape (or no HIP device). run_ns(plan, warmup, iters) → median ns.
type CkCreate = unsafe extern "C" fn(c_int, c_int, c_int, c_int, c_int, c_int) -> *mut c_void;
type CkRunNs = unsafe extern "C" fn(*mut c_void, c_int, c_int) -> f64;
type CkDestroy = unsafe extern "C" fn(*mut c_void);

struct CkShim {
    _lib: Library,
    create: CkCreate,
    run_ns: CkRunNs,
    destroy: CkDestroy,
}

fn load_ck() -> Option<CkShim> {
    // SAFETY: a trusted, flake-built lib whose `extern "C"` symbols match the signatures above.
    unsafe {
        let lib = Library::new("libsvod_ck_fmha_shim.so").ok()?;
        let create = *lib.get::<CkCreate>(b"ck_fmha_create\0").ok()?;
        let run_ns = *lib.get::<CkRunNs>(b"ck_fmha_run_ns\0").ok()?;
        let destroy = *lib.get::<CkDestroy>(b"ck_fmha_destroy\0").ok()?;
        Some(CkShim { _lib: lib, create, run_ns, destroy })
    }
}

fn ck_ready() -> Option<&'static CkShim> {
    static S: OnceLock<Option<CkShim>> = OnceLock::new();
    S.get_or_init(load_ck).as_ref()
}

const MAX_WS: u64 = 256 * 1024 * 1024; // hipBLASLt heuristic workspace cap
const WARMUP: c_int = 3;
// Mirrors TILE in shims/hipblaslt_shim.cpp: the shim tiles output n into ≤2048
// chunks to dodge the large-output GEMM fault (gfx1151 illegal-access AND the
// MI300X-VF Tensile host-init SIGSEGV), so rows with n above this run tiled and are
// labelled "vendor_tiled" (not a silent cap).
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

/// Square `C[n,n] = A·Bᵀ` (B in `[N,K]`, the B[N,K] contract) — the hipBLASLt floor for
/// `svod_tk::matmul`. `trans_b: 1` makes the vendor compute the identical `A·Bᵀ`.
fn bench_matmul(c: &mut Criterion) {
    let Some(shim) = gemm_ready() else {
        eprintln!("svod-tk vendor matmul: skipped (no hipBLASLt shim / GPU)");
        return;
    };
    let mut group = c.benchmark_group("matmul");
    // hipBLASLt/Tensile host-init SIGSEGVs past ~2048 on this VF. The shim tiles the
    // output-n to ≤2048 to dodge it (see `arch_tile_cap`), which rescues knn/kmeans
    // (their only large dim IS the output n). But a SQUARE matmul is m=n=k: n-tiling
    // shrinks only the output — m and k stay large and still fault — so the vendor
    // matmul floor stays capped at the runnable sizes here (full 3D m/n/k tiling would
    // be needed for 4096/8192). The tk `--bench matmul` covers 4096/8192 natively.
    for &n in &[1024usize, 2048] {
        group.throughput(Throughput::Elements((2.0 * (n as f64).powi(3)) as u64)); // 2·M·N·K
        let i = n as i64;
        bench_gemm(&mut group, shim, n, GemmShape { m: i, n: i, k: i, trans_a: 0, trans_b: 1 }, "vendor");
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

/// Flash-attention vendor floor: CK ck_tile `fmha_fwd` (fused, bf16) at HK's shape —
/// B=4, H=32, H_KV=8 (GQA 4:1), N=1024 — for the 4 configs (d ∈ {64,128} × causal). Row id
/// `vendor`, group `fa_hk`. FLOPs = 4·B·H·N²·d (H = query heads; GQA shares KV but each query
/// head does full N² attention), halved for causal. Compare to the svod `tk` fa rows measured
/// at the same shape. Self-skips if the CK shim / a matching instance is absent.
fn bench_fa(c: &mut Criterion) {
    let Some(shim) = ck_ready() else {
        eprintln!("svod-tk vendor fa: skipped (no libsvod_ck_fmha_shim.so — build the ckFmhaShim / bench-fa shell)");
        return;
    };
    let (b, h, h_kv, n) = (4i64, 32i64, 8i64, 1024i64);
    let mut group = c.benchmark_group("fa_hk");
    for &d in &[64i64, 128] {
        for &causal in &[0i32, 1] {
            let f = 4.0 * (b * h * d) as f64 * (n as f64).powi(2) * if causal == 1 { 0.5 } else { 1.0 };
            group.throughput(Throughput::Elements(f as u64));
            // SAFETY: the shim self-allocates; null == no CK instance for this shape / no device.
            let plan = unsafe { (shim.create)(b as c_int, h as c_int, h_kv as c_int, n as c_int, d as c_int, causal) };
            if plan.is_null() {
                eprintln!("svod-tk vendor fa (CK): no fmha instance for d={d} causal={causal}; skipping");
                continue;
            }
            let id = format!("d{d}/{}", if causal == 1 { "causal" } else { "noncausal" });
            group.bench_with_input(BenchmarkId::new("vendor", id), &d, |bch, _| {
                bch.iter_custom(|iters| {
                    // SAFETY: `plan` valid for the bench's duration; the shim's own device VAs.
                    let ns = unsafe { (shim.run_ns)(plan, WARMUP, iters_to_c(iters)) };
                    Duration::from_nanos((ns * iters as f64) as u64)
                });
            });
            // SAFETY: from create, destroyed once.
            unsafe { (shim.destroy)(plan) };
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_matmul, bench_kmeans, bench_knn, bench_fa
}
criterion_main!(benches);
