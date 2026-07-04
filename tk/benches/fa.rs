//! Criterion GPU-device-time bench for `svod_tk::flash_attention_with` — the production
//! attention kernel — vs the SDPA fallback the model would otherwise take. See [`common`]
//! for how device time is stamped and when the bench self-skips.
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk --bench fa`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_tensor::Tensor;

mod common;
use common::{bench_plan, rand_bf16, requirements_met};

/// Attention as the model runs it: `svod_tk::flash_attention_with` — the exact GigaAM
/// encoder call (non-causal, `[B, T, H, d_k]` layout, optional key padding; here
/// unpadded) — vs the **non-causal** SDPA fallback the model would otherwise take. Both
/// timed through `prepare()` → `execute_profiled`.
fn bench_fa(c: &mut Criterion) {
    if !requirements_met(svod_tk::kernels::fa::FA_SUPPORTED_ARCHS) {
        eprintln!("svod-tk flash_attention bench: skipped (target does not meet the kernel's AMD-LLVM requirement)");
        return;
    }
    let (b, h, d) = (1usize, 16usize, 64usize);
    let mut group = c.benchmark_group("flash_attention");
    for &n in &[512usize, 1024, 2048] {
        // Non-causal attention FLOPs: QKᵀ + P·V, each 2·B·H·N²·d.
        group.throughput(Throughput::Elements((4.0 * (b * h * d) as f64 * (n as f64).powi(2)) as u64));
        let (q, k, v) = (rand_bf16(&[b, n, h, d]), rand_bf16(&[b, n, h, d]), rand_bf16(&[b, n, h, d]));

        // The model's exact call: non-causal, no key padding.
        let mut fa = svod_tk::flash_attention_with(&q, &k, &v, svod_tk::FaOpts { causal: false, key_lens: None })
            .expect("flash_attention_with")
            .expect("FA kernel applies for bench shape");
        let fa_plan = fa.prepare().expect("prepare fa");
        group.bench_with_input(BenchmarkId::new("tk", n), &n, |bencher, _| bench_plan(bencher, &fa_plan));

        // The model's fallback: non-causal SDPA. SDPA wants `[B, H, T, d]`.
        let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("perm");
        let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
        let refb = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(false).call().expect("sdpa");
        let mut ref_t = refb.try_permute(&[0, 2, 1, 3]).expect("perm back");
        let ref_plan = ref_t.prepare().expect("prepare ref");
        group.bench_with_input(BenchmarkId::new("sdpa", n), &n, |bencher, _| bench_plan(bencher, &ref_plan));
    }
    group.finish();
    let mut vg = c.benchmark_group("fa_hk_svod");
    for &d in &[64usize, 128] {
        let (b, h, hk, n) = (4usize, 32usize, 8usize, 1024usize);
        vg.throughput(Throughput::Elements((4.0 * (b * h * d) as f64 * (n as f64).powi(2)) as u64));
        let q = rand_bf16(&[b, n, h, d]);
        let k = rand_bf16(&[b, n, hk, d]);
        let v = rand_bf16(&[b, n, hk, d]);
        if let Ok(Some(mut fa)) =
            svod_tk::flash_attention_with(&q, &k, &v, svod_tk::FaOpts { causal: false, key_lens: None })
        {
            let pl = fa.prepare().unwrap();
            vg.bench_with_input(BenchmarkId::new("tk", d), &d, |bch, _| bench_plan(bch, &pl));
        }
    }
    vg.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_fa
}
criterion_main!(benches);
