//! Minimum reproducible example: does the schedule fuse `Q@K^T → softmax → @V`
//! into a single tiled kernel, or does it materialize the
//! `[B, n_heads, T, T]` scores tensor to RAM?
//!
//! Compares two paths against a fixed shape:
//!   1. `Tensor::scaled_dot_product_attention` (the library SDPA — what GigaAM uses)
//!   2. Hand-rolled Q@K^T + softmax + @V (same math, expanded by hand)
//!
//! For each: prepare the execution plan and print kernel count + the
//! intermediate allocations sorted by size. If the scores buffer
//! `[B, H, T, T] × dtype_bytes` appears in the intermediates, fusion did
//! not happen.
//!
//! Run: `cargo run --release --example sdpa_fusion_mre -p svod-tensor`

use std::collections::HashMap;

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::{Tensor, Variable};

/// Build a Q/K/V-shaped tensor that the constant-folder *cannot* collapse.
/// `Tensor::full(0.5)` is recognized as a constant and the entire SDPA gets
/// folded to a single scalar — useless as an MRE. We bypass that by writing
/// non-uniform bytes through a realized buffer.
fn opaque_input(shape: &[usize], seed: u32) -> Result<Tensor, Box<dyn std::error::Error>> {
    let numel: usize = shape.iter().product();
    let mut data: Vec<f32> = Vec::with_capacity(numel);
    let mut state = seed.wrapping_mul(2654435761);
    for _ in 0..numel {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let bits = (state >> 1) | 0x3f800000; // [1.0, 2.0)
        data.push(f32::from_bits(bits) - 1.5); // ~[-0.5, 0.5]
    }
    let t = Tensor::from_slice(data);
    t.realize()?;
    Ok(t.try_reshape(shape.iter().map(|&d| SInt::Const(d)).collect::<Vec<_>>())?)
}

// Default = small "library smoke test" dims. Override with env vars to
// probe larger shapes:
//   SDPA_B=4 SDPA_H=16 SDPA_T=540 SDPA_D=48  cargo run ...
fn dims() -> (usize, usize, usize, usize) {
    fn env(name: &str, default: usize) -> usize {
        std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    (env("SDPA_B", 2), env("SDPA_H", 4), env("SDPA_T", 128), env("SDPA_D", 32))
}

fn fmt_bytes(n: usize) -> String {
    if n >= 1024 * 1024 {
        format!("{:.2} MiB", n as f64 / (1024.0 * 1024.0))
    } else if n >= 1024 {
        format!("{:.2} KiB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

fn dump_plan(label: &str, output: Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let plan = output.prepare()?;

    let kernels = plan.prepared_kernels();
    let buffers = plan.buffers();

    let (b, h, t, d) = dims();
    let scores_bytes = b * h * t * t * 4; // f32

    let mut by_storage: HashMap<u64, usize> = HashMap::new();
    for buf in buffers {
        by_storage.entry(buf.storage_id().0).or_insert_with(|| buf.total_size());
    }
    let mut allocs: Vec<(u64, usize)> = by_storage.into_iter().collect();
    allocs.sort_by_key(|(_, sz)| std::cmp::Reverse(*sz));

    println!("\n=== {label} ===");
    println!("shape:          [B={b}, H={h}, T={t}, head_dim={d}]");
    println!("expected scores: [{b}, {h}, {t}, {t}] f32 = {}", fmt_bytes(scores_bytes));
    println!("compiled kernels: {}", kernels.len());
    println!("distinct allocations: {} (top 10 by size):", allocs.len());
    for (i, (sid, sz)) in allocs.iter().take(10).enumerate() {
        let marker = if *sz == scores_bytes { "  <-- exact scores-tensor size" } else { "" };
        println!("  #{:<2} storage_id={:<5} {:>10}{marker}", i, sid, fmt_bytes(*sz));
    }

    // Tolerance: alignment can pad the storage above the exact element count.
    let scores_materialized = allocs.iter().any(|(_, sz)| {
        let ratio = *sz as f64 / scores_bytes as f64;
        (0.95..=1.10).contains(&ratio)
    });
    if scores_materialized {
        println!("VERDICT: scores tensor IS materialized (no SDPA fusion).");
    } else {
        println!("VERDICT: scores tensor NOT materialized (likely fused).");
    }

    // Per-kernel summary so the fused vs. split structure is readable.
    println!("kernels:");
    for (i, k) in kernels.iter().enumerate() {
        println!("  [{i}] {} ({}B IR; vars={:?})", k.kernel.entry_point, k.kernel.code.len(), k.kernel.var_names);
    }
    if std::env::var("SDPA_DUMP_KERNELS").is_ok() {
        for k in &kernels {
            println!("\n--- {} ---\n{}\n", k.kernel.entry_point, k.kernel.code);
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Library SDPA path.
    {
        let (b, h, t, d) = dims();
        let q = Tensor::full(&[b, h, t, d], 0.5f32, DType::Float32);
        let k = Tensor::full(&[b, h, t, d], 0.5f32, DType::Float32);
        let v = Tensor::full(&[b, h, t, d], 0.5f32, DType::Float32);
        let out = q.scaled_dot_product_attention().key(&k).value(&v).call()?;
        dump_plan("Tensor::scaled_dot_product_attention", out)?;
    }

    // 2. Hand-rolled Q@K^T → softmax → @V (identical math, no helper).
    {
        let (b, h, t, d) = dims();
        let q = Tensor::full(&[b, h, t, d], 0.5f32, DType::Float32);
        let k = Tensor::full(&[b, h, t, d], 0.5f32, DType::Float32);
        let v = Tensor::full(&[b, h, t, d], 0.5f32, DType::Float32);
        let kt = k.try_transpose(-1, -2)?;
        let scale = Tensor::const_(1.0 / (d as f64).sqrt(), DType::Float32);
        let scores = q.matmul(&kt)?.try_mul(&scale)?;
        let weights = scores.softmax(-1)?;
        let out = weights.matmul(&v)?;
        dump_plan("hand-rolled Q@K^T + softmax + @V", out)?;
    }

    // 3a. Q/K/V come from linear projections off a shared input — mirrors
    //     the lineage in GigaAM's encoder.rs (q = x @ q_proj, etc.) before
    //     reshape/transpose into [B, H, T, d_k]. If the scheduler decides
    //     to materialize Q/K outputs at the projection kernel boundary,
    //     SDPA loses its fused tile-local scores.
    {
        let (b, h, t, d) = dims();
        let d_model = h * d;
        let x = opaque_input(&[b, t, d_model], 10)?;
        let q_proj = opaque_input(&[d_model, d_model], 11)?;
        let k_proj = opaque_input(&[d_model, d_model], 12)?;
        let v_proj = opaque_input(&[d_model, d_model], 13)?;
        let q = x.linear().weight(&q_proj).call()?;
        let k = x.linear().weight(&k_proj).call()?;
        let v = x.linear().weight(&v_proj).call()?;
        let q = q.try_reshape([b, t, h, d])?.try_transpose(1, 2)?;
        let k = k.try_reshape([b, t, h, d])?.try_transpose(1, 2)?;
        let v = v.try_reshape([b, t, h, d])?.try_transpose(1, 2)?;
        let out = q.scaled_dot_product_attention().key(&k).value(&v).call()?;
        dump_plan("Q/K/V from linear projections + SDPA", out)?;
    }

    // 3. Same as (1), but Q/K/V have SYMBOLIC B and T axes — this mirrors how
    //    the GigaAM encoder shapes flow (Variable-bound batch and sequence
    //    length). If fusion is shape-symbol-sensitive, this is the case that
    //    would expose it.
    {
        let (b, h, t, d) = dims();
        let b_var = Variable::new("B", 1, b as i64);
        let t_var = Variable::new("T", 1, t as i64);
        let b_sym = b_var.bind(b as i64)?.as_sint();
        let t_sym = t_var.bind(t as i64)?.as_sint();

        let make = |seed: u32| -> Result<Tensor, Box<dyn std::error::Error>> {
            let tnsr = opaque_input(&[b, h, t, d], seed)?;
            Ok(tnsr.try_shrink([
                Some((SInt::Const(0), b_sym.clone())),
                None,
                Some((SInt::Const(0), t_sym.clone())),
                None,
            ])?)
        };
        let q = make(20)?;
        let k = make(21)?;
        let v = make(22)?;
        let out = q.scaled_dot_product_attention().key(&k).value(&v).call()?;
        dump_plan("symbolic-shape Tensor::scaled_dot_product_attention", out)?;
    }

    Ok(())
}
