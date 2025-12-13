//! Benchmark comparing beam search vs heuristic optimization.
//!
//! Measures EXECUTION time only (not compilation/optimization time).
//! Reports throughput in GFLOPS.
//!
//! Run with: `cargo bench -p morok-tensor`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use morok_tensor::Tensor;

/// Create a test matrix of given size with sequential values.
fn create_matrix(rows: usize, cols: usize) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.01).collect();
    Tensor::from_slice(&data).try_reshape(&[rows as isize, cols as isize]).expect("reshape should succeed")
}

/// Calculate FLOPs for matrix multiplication.
/// For [M, K] @ [K, N] -> [M, N]: 2 * M * N * K (one mul + one add per output element, K times)
fn matmul_flops(m: usize, k: usize, n: usize) -> u64 {
    2 * (m as u64) * (k as u64) * (n as u64)
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_optimization");
    let mut executor = morok_runtime::global_executor();

    for size in [32, 64, 128, 256, 512] {
        let flops = matmul_flops(size, size, size);
        group.throughput(Throughput::Elements(flops));

        let a = create_matrix(size, size);
        let b = create_matrix(size, size);

        // HEURISTIC: Prepare OUTSIDE timing (compilation happens here)
        unsafe { std::env::remove_var("MOROK_BEAM") };
        let result_h = a.matmul(&b).expect("matmul should succeed");
        let plan_h = result_h.prepare().expect("prepare should succeed");

        group.bench_with_input(BenchmarkId::new("heuristic", size), &size, |bencher, _| {
            bencher.iter(|| plan_h.execute(&mut executor).expect("execute should succeed"));
        });

        // BEAM: Prepare OUTSIDE timing (beam search + compilation happens here)
        unsafe { std::env::set_var("MOROK_BEAM", "2") };
        let result_b = a.matmul(&b).expect("matmul should succeed");
        let plan_b = result_b.prepare().expect("prepare should succeed");

        group.bench_with_input(BenchmarkId::new("beam_w2", size), &size, |bencher, _| {
            bencher.iter(|| plan_b.execute(&mut executor).expect("execute should succeed"));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
