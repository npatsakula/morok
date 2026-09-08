//! Conformer FFN forward benchmark — measures execution time of the hot
//! matmul pair that dominates the GigaAM encoder.
//!
//! Mirrors the slow kernels we observed in `model/examples/gigaam_infer.rs`:
//!   - FFN expansion:   `[B*T, D] @ [D, FF]    → [B*T, FF]`     (768→3072)
//!   - silu activation
//!   - FFN contraction: `[B*T, FF] @ [FF, D]   → [B*T, D]`     (3072→768)
//!
//! Default shapes match GigaAM-V3 fp32 inference (B=5, T=550, D=768, FF=3072).
//! Override via env vars: `BT=2750 D=768 FF=3072`.
//!
//! Configurations exercised:
//!   - heuristic   (svod's hand-coded heuristic, default)
//!   - beam_w1    (beam width 1 — single-best candidate per step)
//!   - beam_w2    (beam width 2)
//!   - beam_w4    (beam width 4)
//!
//! Run with:
//!   `cargo bench -p svod-tensor --bench conformer_ffn`

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_schedule::{HeuristicsConfig, OptStrategy, OptimizerConfig, TcOptLevel};
use svod_tensor::{PrepareConfig, Tensor};

/// Shape envelope for the FFN forward pass.
#[derive(Clone, Copy, Debug)]
struct FfnShape {
    bt: usize,
    d: usize,
    ff: usize,
}

impl FfnShape {
    /// FLOPs for two matmuls (mul+add per element). The activation is
    /// negligible vs the matmuls.
    fn flops(&self) -> u64 {
        2 * (self.bt as u64) * (self.d as u64) * (self.ff as u64) * 2
    }
}

/// Build the FFN forward graph and pre-realize the weight tensors so timing
/// only covers the activation matmul + silu + projection matmul.
fn build_ffn(shape: FfnShape) -> Tensor {
    // Synthetic data — exact values don't matter for timing.
    let x_data: Vec<f32> = (0..shape.bt * shape.d).map(|i| (i as f32) * 1e-4).collect();
    let w1_data: Vec<f32> = (0..shape.d * shape.ff).map(|i| (i as f32) * 1e-4).collect();
    let w2_data: Vec<f32> = (0..shape.ff * shape.d).map(|i| (i as f32) * 1e-4).collect();

    let x = Tensor::from_slice(&x_data).try_reshape([shape.bt as isize, shape.d as isize]).expect("reshape x");
    let w1 = Tensor::from_slice(&w1_data).try_reshape([shape.d as isize, shape.ff as isize]).expect("reshape w1");
    let w2 = Tensor::from_slice(&w2_data).try_reshape([shape.ff as isize, shape.d as isize]).expect("reshape w2");

    // Realize inputs/weights so they aren't included in the FFN graph timing.
    x.realize().expect("realize x");
    w1.realize().expect("realize w1");
    w2.realize().expect("realize w2");

    // FFN macaron without LayerNorm / residual: matmul -> silu -> matmul.
    let h = x.matmul(&w1).expect("matmul w1");
    let h = h.silu().expect("silu");
    h.matmul(&w2).expect("matmul w2")
}

fn parse_env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn make_configs() -> Vec<(&'static str, PrepareConfig)> {
    let mut out: Vec<(&'static str, PrepareConfig)> = Vec::new();

    let heuristic: PrepareConfig = OptimizerConfig::builder()
        .strategy(OptStrategy::Heuristic)
        // TC_OPT is pinned: the heuristic default (Strict, tinygrad `helpers.py:238`)
        // is what these numbers measure, not the BEAM action space's TC_OPT=2.
        .heuristics(HeuristicsConfig::builder().tc_opt(TcOptLevel::Strict).build())
        .build()
        .into();
    out.push(("heuristic", heuristic));

    let beam: PrepareConfig = OptimizerConfig::builder().strategy(OptStrategy::Beam { width: 4 }).build().into();
    out.push(("beam_w4", beam));
    out
}

fn bench_ffn(c: &mut Criterion) {
    let shape = FfnShape { bt: parse_env("BT", 2750), d: parse_env("D", 768), ff: parse_env("FF", 3072) };
    eprintln!("\n=== Conformer FFN bench  shape: B*T={} D={} FF={} dtype=f32 ===", shape.bt, shape.d, shape.ff);
    eprintln!("Per-iter FLOPs: {:.2} G", shape.flops() as f64 / 1e9);

    let mut group = c.benchmark_group("conformer_ffn");
    // Reduce sample size for beam configs that have multi-second prepare time.
    group.sample_size(20).measurement_time(Duration::from_secs(8)).warm_up_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(shape.flops()));

    for (label, config) in make_configs() {
        // Build + prepare OUTSIDE timing. Beam search compilation cost is
        // not what we want to measure (and it's non-trivial: 200s+ per kernel
        // shape with width=4 in some cases).
        let result = build_ffn(shape);
        let prepare_start = std::time::Instant::now();
        let plan = result.prepare_with(&config).expect("prepare should succeed");
        let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;

        // Print the kernels svod generated for this config so we can see
        // what tile shapes / opts the optimizer landed on.
        eprintln!("\n--- {label} ---  prepare={:.1}ms  kernels={}", prepare_ms, plan.kernels().count());
        for (i, prepared) in plan.prepared_kernels().into_iter().enumerate() {
            // Just the entry-point name; full code is huge.
            eprintln!("  k{i}: {}", prepared.kernel.entry_point);
            println!("{}", prepared.ast.tree());
            println!("{}", prepared.kernel.code);
        }

        let label_owned = label.to_string();
        group.bench_with_input(BenchmarkId::new(label_owned, shape.bt), &shape.bt, |bencher, _| {
            bencher.iter(|| plan.execute().expect("execute should succeed"));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_ffn);
criterion_main!(benches);

// Silence unused-import warning when `DType` isn't referenced (kept for
// downstream variants that may want fp16).
#[allow(dead_code)]
fn _dtype_marker() -> DType {
    DType::Float32
}
