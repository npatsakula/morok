//! Criterion GPU-device-time bench for `svod_tk::matmul` — the hand kernel (a DSL
//! perf-canary; the model itself uses the generic optimizer via `Tensor::linear`) — vs
//! svod's generic bf16→f32 GEMM. See [`common`] for device-time stamping and self-skip.
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk --bench matmul`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;

mod common;
use common::{bench_plan, randn_bf16, requirements_met};

/// The svod-tk hand matmul (`svod_tk::matmul`, a graph-native `custom_kernel` node) vs
/// svod's generic `Tensor::matmul` reference — both timed through `prepare()` →
/// `execute_profiled`. The hand kernel carries no production load (the model uses the
/// generic optimizer); this is its DSL perf-canary. Square `M = N = K`, bf16 in,
/// f32 accumulate.
fn bench_matmul(c: &mut Criterion) {
    if !requirements_met(svod_tk::kernels::matmul::MATMUL_SUPPORTED_ARCHS) {
        eprintln!("svod-tk matmul bench: skipped (no supported AMD GPU / toolchain)");
        return;
    }
    let mut group = c.benchmark_group("matmul");
    for &n in &[1024usize, 2048, 4096, 8192] {
        group.throughput(Throughput::Elements((2.0 * (n as f64).powi(3)) as u64)); // 2·M·N·K
        let a = randn_bf16(&[n, n]);
        let b = randn_bf16(&[n, n]);

        let mut y = svod_tk::matmul(&a, &b).expect("tk matmul").expect("matmul kernel applies");
        let plan = y.prepare().expect("prepare matmul");
        group.bench_with_input(BenchmarkId::new("tk", n), &n, |bencher, _| bench_plan(bencher, &plan));

        // Reference: svod's generic bf16→f32 GEMM (the matmul a user would write).
        // `matmul` now follows the HK contract `C = A·Bᵀ` (B in [N,K]), so the generic
        // reference transposes B too — same op, apples-to-apples throughput.
        let bt = b.try_permute(&[1, 0]).expect("bᵀ");
        let mut reft = a.matmul_with().other(&bt).dtype(DType::Float32).call().expect("ref matmul");
        let ref_plan = reft.prepare().expect("prepare ref");
        group.bench_with_input(BenchmarkId::new("generic", n), &n, |bencher, _| bench_plan(bencher, &ref_plan));
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_matmul
}
criterion_main!(benches);
