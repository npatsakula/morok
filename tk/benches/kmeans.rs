//! Criterion bench for `svod_tk::kmeans_assign` — the fused brute-force argmin
//! (running top-1 from an x²-free score, no `[N, K]` HBM matrix) — vs the
//! generic GEMM-argmin path it replaces. The `tk`/`generic` rows are GPU device
//! time (see [`common`]).
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk --bench kmeans`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_tensor::Tensor;

mod common;
use common::{bench_plan, rand_bf16, requirements_met};

/// The generic-graph **GEMM-argmin** k-means assignment baseline — what you'd
/// write *without* the fused kernel: materialise the full `[N, K]` squared-L2
/// distance matrix in HBM (`‖x‖² + ‖c‖² − 2·x·cᵀ`) then `argmin` over K. Returns
/// the `[N]` nearest squared distances — the same quantity `svod_tk::kmeans_assign`
/// returns — so the `tk` and `generic` rows are comparable.
fn kmeans_generic_ref(xb: &Tensor, cb: &Tensor) -> Tensor {
    let f32 = DType::Float32;
    let xf = xb.cast(f32.clone()).expect("x→f32");
    let cf = cb.cast(f32.clone()).expect("c→f32");
    let x_sq = xf.try_mul(&xf).expect("x²").sum_with().axes(1isize).keepdim(true).call().expect("Σx²"); // [N,1]
    let c_sq = cf.try_mul(&cf).expect("c²").sum_with().axes(1isize).keepdim(true).call().expect("Σc²"); // [K,1]
    let c_sq_row = c_sq.try_transpose(0, 1).expect("c_sq→[1,K]");
    let ct = cb.try_transpose(0, 1).expect("cᵀ");
    let cross = xb.matmul_with().other(&ct).dtype(f32).call().expect("x·cᵀ"); // [N,K] f32
    let two_cross = cross.try_add(&cross).expect("2·cross");
    let dist = x_sq.try_add(&c_sq_row).expect("‖x‖²+‖c‖²").try_sub(&two_cross).expect("−2·cross"); // [N,K]

    dist.min(1).expect("min over K")
}

/// Fused brute-force k-means assignment: `svod_tk::kmeans_assign` (running
/// argmin from an x²-free score, no `[N, K]` HBM matrix) vs the generic
/// GEMM-argmin path it replaces — both timed by GPU device time. `D = 64`,
/// square-ish `N = 2048` points, sweeping the centroid count `K`.
fn bench_kmeans(c: &mut Criterion) {
    let archs = svod_tk::kernels::kmeans::KMEANS_SUPPORTED_ARCHS;
    let gpu = requirements_met(archs);
    if !gpu {
        eprintln!("svod-tk kmeans bench: GPU rows skipped (no supported AMD GPU / toolchain)");
    }
    let (n, d) = (2048usize, 64usize);
    let mut group = c.benchmark_group("kmeans");
    for &k in &[64usize, 256, 1024, 4096] {
        // GEMM-equivalent work: 2·N·K·D for the cross term (the dominant cost).
        group.throughput(Throughput::Elements((2.0 * (n * k * d) as f64) as u64));

        if gpu {
            let xb = rand_bf16(&[n, d]);
            let cb = rand_bf16(&[k, d]);

            // tk: fused running argmin.
            let (_ids, mut dists) =
                svod_tk::kmeans_assign(&xb, &cb).expect("tk kmeans").expect("kmeans applies for bench shape");
            let plan = dists.prepare().expect("prepare tk kmeans");
            group.bench_with_input(BenchmarkId::new("tk", k), &k, |bencher, _| bench_plan(bencher, &plan));

            // Reference: the generic GEMM-argmin path (materialises the [N,K] matrix).
            // Optional — the generic codegen hits a WMMA-intrinsic redeclaration on
            // some archs (gfx1151); skip the row instead of failing the whole bench.
            let mut reft = kmeans_generic_ref(&xb, &cb);
            match reft.prepare() {
                Ok(ref_plan) => {
                    group.bench_with_input(BenchmarkId::new("generic", k), &k, |bencher, _| {
                        bench_plan(bencher, &ref_plan)
                    });
                }
                Err(e) => eprintln!("svod-tk kmeans bench: skip generic row for K={k} ({e})"),
            }
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_kmeans
}
criterion_main!(benches);
