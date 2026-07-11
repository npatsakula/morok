//! Criterion GPU-device-time bench for svod's generic `Tensor::matmul` (BEAM-optimized) on the
//! production Llama-70B GEMM shapes — rectangular `C = A·Bᵀ` at token counts T. The square
//! hand-kernel canary was removed (no transformer layer is square M=N=K; the rectangular shapes
//! are the production comparison). See [`common`] for device-time stamping and self-skip.
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk --bench matmul`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_schedule::{OptStrategy, OptimizerConfig};
use svod_tensor::PrepareConfig;

mod common;
use common::{bench_plan, rand_bf16, requirements_met};

/// BEAM optimizer config for the generic `Tensor::matmul` — searches kernel schedules
/// (expensive at `prepare_with`, free at run) to give the generic path its best number.
/// Beam only helps the generic op; the hand kernel/HK port are already hand-scheduled.
fn beam_config() -> PrepareConfig {
    OptimizerConfig::builder().strategy(OptStrategy::Beam { width: 2 }).build().into()
}

/// RECTANGULAR, ML-realistic bench: the generic `Tensor::matmul` (BEAM-optimized) on the three
/// Llama-70B GEMM types `C[m,n] = A[m,k]·Bᵀ` (B stored `[n,k]`) at `T = m ∈ {256, 2048, 8192}`.
/// The hand `svod_tk::matmul` is square-only, so only the generic path is benched here (vs the
/// separately-measured `tk2::hk::micro_tk` and hipBLASLt on the same shapes).
fn bench_matmul_rect(c: &mut Criterion) {
    if !requirements_met(svod_tk::kernels::matmul::MATMUL_SUPPORTED_ARCHS) {
        eprintln!("svod-tk rectangular matmul bench: skipped (no supported AMD GPU / toolchain)");
        return;
    }
    let shapes = [("attn_out", 8192usize, 8192usize), ("ffn_up", 28672, 8192), ("ffn_down", 8192, 28672)];
    let mut group = c.benchmark_group("matmul_rect");
    for &(name, n, k) in &shapes {
        for &t in &[256usize, 2048, 8192] {
            let m = t;
            group.throughput(Throughput::Elements(2 * m as u64 * n as u64 * k as u64));
            let a = rand_bf16(&[m, k]); // A is [M,K]
            let b = rand_bf16(&[n, k]); // B is [N,K]
            let bt = b.try_permute(&[1, 0]).expect("bᵀ"); // [K,N] so a·bᵀ = A·Bᵀ
            let mut beamt = a.matmul_with().other(&bt).dtype(DType::Float32).call().expect("beam matmul");
            let beam_plan = beamt.prepare_with(&beam_config()).expect("prepare beam");
            group.bench_with_input(BenchmarkId::new(format!("generic_beam/{name}"), t), &t, |bch, _| {
                bench_plan(bch, &beam_plan)
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_matmul_rect
}
criterion_main!(benches);
