//! Benchmark comparing beam search vs heuristic optimization.
//!
//! Measures EXECUTION time only (not compilation/optimization time).
//! Reports throughput in GFLOPS.
//!
//! Run with: `cargo bench -p morok-tensor`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use morok_schedule::{OptStrategy, OptimizerConfig};
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

    // Typed optimizer configurations (no environment variables needed)
    let heuristic_config = OptimizerConfig::default();
    let beam_config = OptimizerConfig::builder().strategy(OptStrategy::Beam { width: 2 }).build();

    for size in [64] {
        let flops = matmul_flops(size, size, size);
        group.throughput(Throughput::Elements(flops));

        // Scope tensors and plans so they're dropped before cleanup
        {
            let a = create_matrix(size, size);
            let b = create_matrix(size, size);

            // HEURISTIC: Prepare OUTSIDE timing (compilation happens here)
            let result_h = a.matmul(&b).expect("matmul should succeed");
            let plan_h = result_h.prepare_with(&heuristic_config).expect("prepare should succeed");

            // DEBUG: Print kernel info for heuristic
            eprintln!("\n=== HEURISTIC (size={}) ===", size);
            eprintln!("Kernel count: {}", plan_h.kernels().count());
            eprintln!("UOp tree:\n{}", result_h.uop().tree());
            for (i, kernel) in plan_h.kernels().enumerate() {
                eprintln!("  Kernel {}: {}", i, kernel.entry_point);
            }

            group.bench_with_input(BenchmarkId::new("heuristic", size), &size, |bencher, _| {
                bencher.iter(|| plan_h.execute(&mut executor).expect("execute should succeed"));
            });

            // BEAM: Prepare OUTSIDE timing (beam search + compilation happens here)
            let result_b = a.matmul(&b).expect("matmul should succeed");
            let plan_b = result_b.prepare_with(&beam_config).expect("prepare should succeed");

            // DEBUG: Print kernel info for beam
            eprintln!("\n=== BEAM (size={}) ===", size);
            eprintln!("Kernel count: {}", plan_b.kernels().count());
            for (i, kernel) in plan_b.kernels().enumerate() {
                eprintln!("  Kernel {}: {}", i, kernel.entry_point);
            }

            group.bench_with_input(BenchmarkId::new("beam_w2", size), &size, |bencher, _| {
                bencher.iter(|| plan_b.execute(&mut executor).expect("execute should succeed"));
            });
        }

        // Cleanup between sizes.
        // With weak references in both UOp cache and tensor registry (Tinygrad-aligned),
        // entries are auto-cleaned when dropped. We still clean up dead refs for hygiene.
        morok_ir::uop::gc_dead_refs();
        morok_tensor::tensor_registry::gc_dead_refs();
        let live_ids = morok_ir::uop::live_uop_ids();
        morok_runtime::kernel_cache::gc_unused_kernels(&live_ids);
    }

    group.finish();
}

#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
