//! Criterion bench for `svod_tk::knn` — the fused brute-force top-K (running top-K from
//! an x²-free score, no `[N, M]` HBM matrix) — vs the generic GEMM-topk path it replaces,
//! plus a native ndarray CPU baseline. The `tk`/`generic` rows are GPU device time (see
//! [`common`]); the `cpu_ndarray` row is **host wall-clock** (`iter`, not `iter_custom`)
//! — a deliberately different metric, an absolute reference point that records even with
//! no GPU present.
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk --bench knn`

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::{Array2, Axis};
use svod_dtype::DType;
use svod_tensor::Tensor;

mod common;
use common::{bench_plan, rand_bf16, requirements_met};

/// The generic-graph **GEMM-topk** KNN baseline — what you'd write *without* the fused
/// kernel: materialise the full `[N, M]` squared-L2 distance matrix in HBM
/// (`‖x‖² + ‖c‖² − 2·x·cᵀ`; bf16 operands → f32 accumulate, like the matmul bench) then
/// `topk(largest = false)`. Returns the `[N, k]` nearest **squared distances** — the
/// same quantity `svod_tk::knn` returns — so the `tk` and `generic` rows are comparable.
fn knn_generic_ref(xb: &Tensor, cb: &Tensor, k: usize) -> Tensor {
    let f32 = DType::Float32;
    let xf = xb.cast(f32.clone()).expect("x→f32");
    let cf = cb.cast(f32.clone()).expect("c→f32");
    // Per-row squared norms (cheap [N,1]/[M,1] reductions; the cost is the GEMM + topk).
    let x_sq = xf.try_mul(&xf).expect("x²").sum_with().axes(1isize).keepdim(true).call().expect("Σx²"); // [N,1]
    let c_sq = cf.try_mul(&cf).expect("c²").sum_with().axes(1isize).keepdim(true).call().expect("Σc²"); // [M,1]
    let c_sq_row = c_sq.try_transpose(0, 1).expect("c_sq→[1,M]");
    // Cross term x·cᵀ — bf16 in, f32 accumulate, the same operand dtypes as `knn`/matmul.
    let ct = cb.try_transpose(0, 1).expect("cᵀ");
    let cross = xb.matmul_with().other(&ct).dtype(f32).call().expect("x·cᵀ"); // [N,M] f32
    let two_cross = cross.try_add(&cross).expect("2·cross");
    let dist = x_sq.try_add(&c_sq_row).expect("‖x‖²+‖c‖²").try_sub(&two_cross).expect("−2·cross"); // [N,M]
    let (vals, _idxs) = dist.topk(k, 1, false).expect("topk(largest=false)");
    vals
}

/// Deterministic pseudo-random `[rows, cols]` f32 matrix in ~`[-1, 1)` via a small LCG
/// (no `rand` dep). Timing for the dense CPU baseline is data-independent, so this only
/// needs to be cheap and reproducible.
fn rand_array2(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed | 1;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    })
}

/// Native-Rust brute-force squared-L2 top-K — the GEMM-topk baseline in pure ndarray:
/// materialise the `[N, M]` distance matrix, then per row partition out the `k` smallest.
/// Returns the running sum of the selected distances so the work isn't optimised away.
fn knn_cpu_ndarray(x: &Array2<f32>, c: &Array2<f32>, k: usize) -> f32 {
    let x_sq = (x * x).sum_axis(Axis(1)); // [N]
    let c_sq = (c * c).sum_axis(Axis(1)); // [M]
    let cross = x.dot(&c.t()); // [N, M]
    let (n, m) = (x.nrows(), c.nrows());
    let kk = k.min(m);
    let mut acc = 0.0f32;
    let mut row: Vec<f32> = Vec::with_capacity(m);
    for i in 0..n {
        row.clear();
        row.extend((0..m).map(|j| x_sq[i] + c_sq[j] - 2.0 * cross[[i, j]]));
        row.select_nth_unstable_by(kk - 1, |a, b| a.partial_cmp(b).expect("no NaN distances"));
        acc += row[..kk].iter().sum::<f32>();
    }
    acc
}

/// Fused brute-force KNN as the HDBSCAN pipeline will run it: `svod_tk::knn` (running
/// top-K from an x²-free score, no `[N, M]` HBM matrix) vs the generic GEMM-topk path it
/// replaces — both timed by GPU device time — plus a native ndarray CPU baseline (host
/// wall-clock; see the module docs on the metric difference). `D = 64`, `k = 16` (the
/// max), square-ish `N = 2048` queries, sweeping the streamed corpus `M`.
fn bench_knn(c: &mut Criterion) {
    let archs = svod_tk::kernels::knn::KNN_SUPPORTED_ARCHS;
    let gpu = requirements_met(archs);
    if !gpu {
        eprintln!("svod-tk knn bench: GPU rows skipped (no supported AMD GPU / toolchain); CPU baseline still runs");
    }
    let (n, d, k) = (2048usize, 64usize, 16usize);
    let mut group = c.benchmark_group("knn");
    for &m in &[512usize, 1024, 2048, 16384] {
        // GEMM-equivalent work: 2·N·M·D for the cross term (the dominant cost).
        group.throughput(Throughput::Elements((2.0 * (n * m * d) as f64) as u64));

        if gpu {
            let xb = rand_bf16(&[n, d]);
            let cb = rand_bf16(&[m, d]);

            // tk: fused running top-K. `prepare()`-ing `dists` realises the kernel + the
            // sort/gather tail (the indices are a shared intermediate of the same graph).
            let (mut dists, _idxs) = svod_tk::knn(&xb, &cb, k).expect("tk knn").expect("knn applies for bench shape");
            let plan = dists.prepare().expect("prepare tk knn");
            group.bench_with_input(BenchmarkId::new("tk", m), &m, |bencher, _| bench_plan(bencher, &plan));

            // Reference: the generic GEMM-topk path (materialises the [N,M] matrix).
            let mut reft = knn_generic_ref(&xb, &cb, k);
            let ref_plan = reft.prepare().expect("prepare generic knn");
            group.bench_with_input(BenchmarkId::new("generic", m), &m, |bencher, _| bench_plan(bencher, &ref_plan));
        }

        // Native CPU baseline (host wall-clock — a different metric than the GPU rows
        // above; an absolute reference point). Distinct seeds per `M`; timing is
        // data-independent so it need not match the GPU operands.
        let xh = rand_array2(n, d, 0x5eed_0001 ^ m as u64);
        let ch = rand_array2(m, d, 0x5eed_0002 ^ m as u64);
        group.bench_with_input(BenchmarkId::new("cpu_ndarray", m), &m, |bencher, _| {
            bencher.iter(|| black_box(knn_cpu_ndarray(&xh, &ch, k)));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_knn
}
criterion_main!(benches);
