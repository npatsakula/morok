//! Criterion GPU-device-time bench for AMD **aiter**'s hand-tuned bf16 GEMMs,
//! loaded straight into svod's KFD-direct launcher via [`AmdProgram::load_external`]
//! and the metadata-parsed kernarg layout — **no HIP runtime**. The `.co`'s
//! interleaved 16-byte-strided ABI (pointers `D,C,A,B`; f32 α/β; i32
//! byte-strides/dims; `Bias`; explicit padding between every field) is
//! packed at the exact offsets its `NT_AMDGPU_METADATA` note declares
//! (`parse_amdgpu_metadata` → `KernargLayout::Explicit`), which svod's
//! buffers-first convention could not express.
//!
//! Benches three aiter configs on the same `rand_bf16` operands / f32-reference
//! allclose gate / device-timestamp measurement as `tk2_matmul_rect`:
//!
//! - `pf3-64` — the plain-B single-pass small-tile kernel (aiter's low-M path).
//! - `bshuf-128`, `bshuf-160` — aiter's **peak** large-tile kernels, which
//!   require B pre-shuffled `shuffle_weight(B,(16,16))`. That relayout is a
//!   one-time weight prep in real inference, so it is done ONCE per shape
//!   (on-device, untimed) via `reshape→permute→contiguous→reshape`, and only
//!   the GEMM is timed. aiter's own heuristic picks 128×64 at M=256 and
//!   160×64 at M≥2048 (split-K is never selected here — N fills the 304 CUs).
//!
//! All compute `C[m,n] = A[m,k]·B[n,k]ᵀ` (α=1, β=0, F32 C), K contiguous in A/B.
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk2 --bench vendor_gemm`
//! Self-skips (records no samples) when the device is not gfx942 or the `.co`s
//! are absent.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_device::amd::{AmdAllocator, AmdProgram, parse_amdgpu_metadata};
use svod_dtype::{DType, DeviceSpec};
use svod_tensor::Tensor;
use svod_tensor::testing::allclose_f32;

mod common;
use common::{rand_bf16, requirements_met};

/// Directory holding aiter's prebuilt gfx942 bf16 GEMM code objects.
fn co_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../submodules/aiter/hsa/gfx942/bf16gemm")
}

/// One loaded aiter GEMM kernel + how to drive it.
struct AiterKernel {
    prog: AmdProgram,
    /// Output tile height (tileN is always 64); grid Y = ceil(M / tile_m).
    tile_m: usize,
    /// Whether this kernel consumes a pre-shuffled B (`bshuffle` kernels) vs
    /// plain row-major B (`pf3`).
    bshuffle: bool,
    label: &'static str,
}

impl AiterKernel {
    /// Load `<co_dir>/<file>` into svod's launcher with its parsed kernarg layout,
    /// or `None` if the file is missing (bench self-skips).
    fn load(file: &str, tile_m: usize, bshuffle: bool, label: &'static str) -> Option<Self> {
        let bytes = std::fs::read(co_dir().join(file)).ok()?;
        let meta = parse_amdgpu_metadata(&bytes).expect("parse aiter metadata").into_iter().next()?;
        let DeviceSpec::Amd { device_id } = Tensor::empty(&[1], DType::Float32).device() else {
            return None;
        };
        let alloc = AmdAllocator::new(device_id).expect("amd allocator");
        let device = Arc::clone(&alloc.dev);
        let prog = AmdProgram::load_external(device, &alloc, &bytes, &meta).expect("load_external aiter gemm");
        Some(Self { prog, tile_m, bshuffle, label })
    }

    /// Dispatch `C[m,n] = A[m,k]·B[n,k]ᵀ` (α=1, β=0, F32 out) once and return the
    /// kernel's on-device ns. `b` must already be in this kernel's expected layout
    /// (plain for pf3, shuffled for bshuffle). The tensors MUST outlive the call.
    fn dispatch_ns(&self, a: &Tensor, b: &Tensor, out: &Tensor, m: usize, n: usize, k: usize) -> u64 {
        let (ab, bb, ob) = (a.buffer().expect("a"), b.buffer().expect("b"), out.buffer().expect("out"));
        ab.ensure_allocated().expect("a alloc");
        bb.ensure_allocated().expect("b alloc");
        ob.ensure_allocated().expect("out alloc");
        // SAFETY: the three tensors outlive this dispatch (held by the caller).
        let (a_va, b_va, o_va) = unsafe { (ab.as_raw_ptr(), bb.as_raw_ptr(), ob.as_raw_ptr()) };

        // Pointer args, kernarg-declaration order: D(out), C(null,β=0), A, B, Bias(null).
        let buffers: [*mut u8; 5] = [o_va, std::ptr::null_mut(), a_va, b_va, std::ptr::null_mut()];
        // Scalar args, declaration order (aiter `KernelArgs`): α, β (f32 bit
        // patterns), row byte-strides D/C/A/B (inner dim contiguous ⇒ *1 = 0; B
        // stays K·2 even when shuffled), M, N, K, splitk=1 (single-pass),
        // is_out_b16=0 (f32 out), add_bias=0.
        let vals: [i64; 16] = [
            0x3f80_0000, // alpha = 1.0f32
            0,           // beta  = 0.0f32
            (n * 4) as i64,
            0, // strideD0 = N·4 (f32), strideD1
            (n * 4) as i64,
            0, // strideC0, strideC1
            (k * 2) as i64,
            0, // strideA0 = K·2 (bf16), strideA1
            (k * 2) as i64,
            0, // strideB0 = K·2 (bf16), strideB1
            m as i64,
            n as i64,
            k as i64,
            1, // splitk (single-pass)
            0, // is_out_b16 = 0 → f32 output
            0, // add_bias
        ];
        // Grid in workgroups: (ceil(N/64), ceil(M/tileM), 1); block = 256 threads.
        let global = [n.div_ceil(64), m.div_ceil(self.tile_m), 1];
        let local = [256, 1, 1];
        // SAFETY: buffers are live GPU VAs, vals match the parsed scalar arity,
        // dims are valid for the kernel's reqd 256-thread workgroup.
        unsafe { self.prog.execute_timed(&buffers, &vals, Some(global), Some(local)) }.expect("aiter dispatch")
    }
}

/// aiter's `shuffle_weight(B, layout=(16,16))` for bf16, applied on-device: view
/// `B[N,K]` as `[N/16, 16, K/32, 4, 8]`, `permute(0,2,3,1,4)`, materialize
/// row-major, reshape back to `[N,K]`. A pure element (u16) permutation — the
/// bshuffle kernels read B through this layout. Verified bijective vs aiter's
/// `aiter/ops/shuffle.py`.
fn shuffle_b(b: &Tensor, n: usize, k: usize) -> Tensor {
    let mut s = b
        .try_reshape([n / 16, 16, k / 32, 4, 8])
        .expect("reshape 5d")
        .try_permute(&[0, 2, 3, 1, 4])
        .expect("permute")
        .contiguous()
        .try_reshape([n, k])
        .expect("reshape 2d");
    s.realize().expect("realize shuffled B");
    s
}

/// A realized, zero-initialized F32 `[m,n]` output tensor (owns a device buffer
/// the kernel writes; `as_vec` copies it back for the correctness gate).
fn zeros_f32(m: usize, n: usize) -> Tensor {
    let mut t = Tensor::full(&[m, n], 0.0f32, DType::Float32).expect("out f32");
    t.realize().expect("realize out");
    t
}

/// f32 ground truth `A·Bᵀ` over the SAME bf16-rounded operands, using the PLAIN
/// (unshuffled) B (B stored `[n,k]`, transposed to `[k,n]` for the matmul).
fn reference(a: &Tensor, b_nk: &Tensor) -> Vec<f32> {
    let b_kn = b_nk.try_transpose(0, 1).expect("Bᵀ");
    let bf = b_kn.cast(DType::Float32).expect("b→f32");
    let mut r = a.cast(DType::Float32).expect("a→f32").matmul(&bf).expect("reference matmul");
    r.realize().expect("realize reference");
    r.as_vec::<f32>().expect("read reference")
}

/// Compiled TFLOP/s from `2·m·n·k` FLOPs over a device-time `ns`.
fn tflops(m: usize, n: usize, k: usize, ns: u64) -> f64 {
    (2.0 * m as f64 * n as f64 * k as f64) / (ns as f64 * 1e3)
}

/// Measure one kernel's steady device-time TFLOP/s over 40 replays, after an
/// allclose gate vs the f32 reference (a wrong kernarg layout / bad shuffle fails
/// the bench rather than reporting fast-but-wrong).
fn gate_and_measure(
    ker: &AiterKernel,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
    exp: &[f32],
    [m, n, k]: [usize; 3],
) -> f64 {
    let _ = ker.dispatch_ns(a, b, out, m, n, k);
    let got = out.as_vec::<f32>().expect("read aiter output");
    let atol = 0.02 * (k as f32).sqrt();
    let report = allclose_f32(&got, exp, atol, 2e-2);
    assert!(report.ok, "aiter {} m={m} n={n} k={k}: output must match reference: {}", ker.label, report.message);
    for _ in 0..2 {
        let _ = ker.dispatch_ns(a, b, out, m, n, k); // warmup
    }
    const ITERS: u64 = 40;
    let mut total = 0u64;
    for _ in 0..ITERS {
        total += ker.dispatch_ns(a, b, out, m, n, k);
    }
    tflops(m, n, k, total / ITERS)
}

/// RECTANGULAR Llama-70B GEMM shapes `C[m,n] = A[m,k]·Bᵀ` at token counts
/// `m ∈ {256, 2048, 8192}` — the SAME 9 configs as `tk2_matmul_rect`, dispatched
/// through three aiter asm kernels via the external-`.co` launch path. Prints a
/// per-config TFLOP/s table (pf3 / bshuf-128 / bshuf-160 / aiter-peak) and
/// criterion-benches the empirical peak.
fn bench_vendor_gemm(c: &mut Criterion) {
    if !requirements_met() {
        eprintln!("aiter vendor GEMM bench: skipped (device is not a supported gfx942 GPU)");
        return;
    }
    let kernels: Vec<AiterKernel> = [
        ("bf16gemm_fp32bf16_tn_64x64_pf3_splitk.co", 64, false, "pf3-64"),
        ("bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk.co", 128, true, "bshuf-128"),
        ("bf16gemm_fp32bf16_tn_160x64_bshuffle_splitk.co", 160, true, "bshuf-160"),
    ]
    .into_iter()
    .filter_map(|(f, tm, bs, l)| AiterKernel::load(f, tm, bs, l))
    .collect();
    if kernels.is_empty() {
        eprintln!("aiter vendor GEMM bench: skipped (no aiter .co under {})", co_dir().display());
        return;
    }

    let shapes = [("attn_out", 8192usize, 8192usize), ("ffn_up", 28672, 8192), ("ffn_down", 8192, 28672)];
    let mut group = c.benchmark_group("vendor_gemm");
    println!("\n=== aiter bf16 GEMM — device TFLOP/s (2·m·n·k / gpu_ns), B pre-shuffled once for bshuf ===");
    println!(
        "{:<10} {:>6} {:>7} {:>7}   {:>8} {:>10} {:>10}   {:>10}",
        "shape", "m", "n", "k", "pf3-64", "bshuf-128", "bshuf-160", "aiter-peak"
    );
    for &(name, n, k) in &shapes {
        assert!(n % 64 == 0 && k % 64 == 0, "{name}: aiter needs N÷64, K÷64");
        let b_plain = rand_bf16(&[n, k]);
        let b_shuf = shuffle_b(&b_plain, n, k); // one-time weight relayout (untimed)
        for &m in &[256usize, 2048, 8192] {
            group.throughput(Throughput::Elements(2 * m as u64 * n as u64 * k as u64));
            let a = rand_bf16(&[m, k]);
            let out = zeros_f32(m, n);
            let expected = reference(&a, &b_plain);

            let mut tfs = vec![f64::NAN; kernels.len()];
            let (mut peak_tf, mut peak_i) = (0.0f64, 0usize);
            for (i, ker) in kernels.iter().enumerate() {
                let b = if ker.bshuffle { &b_shuf } else { &b_plain };
                let tf = gate_and_measure(ker, &a, b, &out, &expected, [m, n, k]);
                tfs[i] = tf;
                if tf > peak_tf {
                    (peak_tf, peak_i) = (tf, i);
                }
            }
            let g = |label: &str| kernels.iter().position(|kk| kk.label == label).map_or(f64::NAN, |i| tfs[i]);
            println!(
                "{name:<10} {m:>6} {n:>7} {k:>7}   {:>8.1} {:>10.1} {:>10.1}   {:>7.1} ({})",
                g("pf3-64"),
                g("bshuf-128"),
                g("bshuf-160"),
                peak_tf,
                kernels[peak_i].label,
            );

            // Criterion-bench the empirical peak kernel for this (shape, m).
            let ker = &kernels[peak_i];
            let b = if ker.bshuffle { &b_shuf } else { &b_plain };
            group.bench_with_input(BenchmarkId::new(format!("aiter-peak/{name}"), m), &m, |bch, _| {
                bch.iter_custom(|iters| {
                    let mut total = 0u64;
                    for _ in 0..iters {
                        total += ker.dispatch_ns(&a, b, &out, m, n, k);
                    }
                    Duration::from_nanos(total)
                });
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_vendor_gemm
}
criterion_main!(benches);
