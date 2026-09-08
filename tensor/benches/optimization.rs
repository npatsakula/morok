//! Benchmark comparing beam search vs heuristic optimization.
//!
//! Measures EXECUTION time only (not compilation/optimization time).
//! Reports throughput in GFLOPS.
//!
//! Run with: `cargo bench -p svod-tensor`
//!
//! If you want to see the UOp tree, set the TREE environment variable.

const KEY: &str = "TREE";

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::{Array, Dim};
use std::env;
use svod_schedule::{HeuristicsConfig, OptStrategy, OptimizerConfig, TcOptLevel};
use svod_tensor::{PrepareConfig, Tensor};

/// Create a test matrix of given size with sequential values.
fn create_matrix(rows: usize, cols: usize) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.01).collect();
    Tensor::from_slice(&data).try_reshape([rows as isize, cols as isize]).expect("reshape should succeed")
}

/// Create a ndarray matrix of given size with sequential values.
fn create_ndarray(rows: usize, cols: usize) -> ndarray::Array<f32, Dim<[usize; 2]>> {
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.01).collect();
    Array::from_shape_vec((rows, cols), data).expect("array from vec should succeed")
}

/// Calculate FLOPs for matrix multiplication.
/// For [M, K] @ [K, N] -> [M, N]: 2 * M * N * K (one mul + one add per output element, K times)
fn matmul_flops(m: usize, k: usize, n: usize) -> u64 {
    2 * (m as u64) * (k as u64) * (n as u64)
}

fn print_tree(config: &str, size: usize, plan: &svod_runtime::ExecutionPlan, result: &Tensor) {
    if env::var(KEY).is_ok() {
        // DEBUG: Print kernel info
        eprintln!("\n=== {config} (size={size}) ===");
        eprintln!("Kernel count: {}", plan.kernels().count());
        eprintln!("UOp tree:\n{}", result.uop().tree());

        for (i, kernel) in plan.prepared_kernels().iter().enumerate() {
            eprintln!("UOp tree:\n{}", kernel.ast.tree());
            eprintln!("  Kernel {}: {}", i, kernel.kernel.entry_point);
            eprintln!("{}", kernel.kernel.code);
        }
    }
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_optimization");

    // Typed optimizer configurations (no environment variables needed).
    // Don't override thread_count — the default is the `SVOD_THREADS` budget,
    // matching `renderer.global_max[0]` on CPU.
    let heuristic_config: PrepareConfig = OptimizerConfig::builder()
        .strategy(OptStrategy::Heuristic)
        // TC_OPT is pinned: the heuristic default (Strict, tinygrad `helpers.py:238`)
        // is what these numbers measure, not the BEAM action space's TC_OPT=2.
        .heuristics(HeuristicsConfig::builder().tc_opt(TcOptLevel::Strict).build())
        .build()
        .into();

    const BEAM_WIDTH: usize = 2;
    let beam_config: PrepareConfig =
        OptimizerConfig::builder().strategy(OptStrategy::Beam { width: BEAM_WIDTH }).build().into();

    for size in [256, 512, 1024] {
        let flops = matmul_flops(size, size, size);
        group.throughput(Throughput::Elements(flops));

        // Scope tensors and plans so they're dropped before cleanup
        {
            let a = create_matrix(size, size);
            let b = create_matrix(size, size);

            // HEURISTIC: Prepare OUTSIDE timing (compilation happens here)
            let result_h = a.matmul(&b).expect("matmul should succeed");
            let plan_h = result_h.prepare_with(&heuristic_config).expect("prepare should succeed");

            print_tree("HEURISTIC", size, &plan_h, &result_h);

            group.bench_with_input(BenchmarkId::new("heuristic", size), &plan_h, |bencher, plan_h| {
                bencher.iter(|| plan_h.execute().expect("execute should succeed"));
            });

            // BEAM: Prepare OUTSIDE timing (beam search + compilation happens here)
            let result_b = a.matmul(&b).expect("matmul should succeed");
            let plan_b = result_b.prepare_with(&beam_config).expect("prepare should succeed");

            print_tree("BEAM", size, &plan_b, &result_b);

            group.bench_with_input(
                BenchmarkId::new(format!("beam_w{BEAM_WIDTH}"), size),
                &plan_b,
                |bencher, plan_b| {
                    bencher.iter(|| plan_b.execute().expect("execute should succeed"));
                },
            );

            let a = create_ndarray(size, size);
            let b = create_ndarray(size, size);

            group.bench_with_input(
                BenchmarkId::new("ndarray multiplication".to_string(), size),
                &(a, b),
                |bencher, (a, b)| {
                    bencher.iter(|| a.dot(b));
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
