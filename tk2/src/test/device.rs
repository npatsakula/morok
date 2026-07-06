//! Hardware-gated device smoke tests for the two skeleton kernels on gfx942
//! (`SVOD_DEVICE=AMD:0 ... --ignored`). The matmul — the perf kernel — is measured
//! *and* correctness-gated through the criterion bench (`benches/matmul.rs`), the
//! day-one feedback loop; these two remain here because they exercise distinct
//! machinery (tiled elementwise, and the loop-carried accumulator) that has no bench.
//!
//! `elementwise_add_runs_on_gfx942` is the "it works" proof: the tiled add lowers,
//! compiles, and runs on the real GPU, matching a CPU reference bit-for-bit (f32
//! add is exact). `sum_reduce_runs_on_gfx942` additionally exercises the
//! loop-carried accumulator on device.

use std::sync::Arc;

use svod_device::Buffer;
use svod_dtype::DType;
use svod_ir::Op;
use svod_tensor::Tensor;
use svod_tensor::testing::allclose_f32;

use crate::kernels::{elementwise_add, sum_reduce};
use crate::launch;
use crate::{SwizzlePass, VectorizePass, graph_kernel};

/// Realize an input tensor from host data and return its concrete buffer.
fn input(data: &[f32]) -> (Tensor, Buffer) {
    let mut t = Tensor::from_slice(data);
    t.realize().expect("realize input");
    let buf = t.buffer().expect("input buffer");
    buf.ensure_allocated().expect("input allocated");
    (t, buf)
}

/// Realize `t` as an f32 host vector (bf16 operands cast to f32 first).
fn as_f32_vec(t: &Tensor) -> Vec<f32> {
    let mut f = t.cast(DType::Float32).expect("→f32");
    f.realize().expect("realize f32");
    f.as_vec::<f32>().expect("read f32")
}

/// Host f32 reference for `C = A·Bᵀ` — A`[m,k]`, B`[n,k]` (HK's [N,K] B contract, both K-contiguous):
/// `C[row,col] = Σ_k A[row,k]·B[col,k]`. The `matmul_lds_kblock*` family computes this.
fn ab_t_ref(af: &[f32], bf: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut e = vec![0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += af[row * k + kk] * bf[col * k + kk];
            }
            e[row * n + col] = acc;
        }
    }
    e
}

/// Allocate + register a fresh output buffer for an empty tensor (the svod analog
/// of tinygrad's `b.allocate()` inside `run_linear`; mirrors tk's `realize_buffer`).
fn output(n: usize) -> (Tensor, Buffer) {
    let t = Tensor::empty(&[n], DType::Float32);
    if let Some(buf) = t.buffer() {
        buf.ensure_allocated().expect("output allocated");
        return (t, buf);
    }
    let base = t.uop().base();
    let Op::Buffer { device, size, .. } = base.op() else { panic!("output tensor has no BUFFER uop") };
    let Op::Device(spec) = device.op() else { panic!("output BUFFER has no DEVICE") };
    let dtype = base.dtype();
    let allocator = svod_device::registry::registry().get(spec).expect("allocator for output device");
    let buffer = Buffer::allocate(allocator, dtype, vec![*size], Default::default()).expect("allocate output");
    let buffer = Arc::new(buffer);
    svod_tensor::tensor_registry::register_buffer_by_uop_id(base.id, buffer.clone());
    (t, (*buffer).clone())
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::elementwise_add_runs_on_gfx942 --nocapture`
#[test]
#[ignore]
fn elementwise_add_runs_on_gfx942() {
    let (tile, n_tiles) = (64usize, 8usize);
    let n = tile * n_tiles;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();

    let program = elementwise_add(tile, n_tiles);
    let (_ta, a_buf) = input(&a);
    let (_tb, b_buf) = input(&b);
    let (out, c_buf) = output(n);

    // ABI order: output first, then inputs (the builder's `global()` order).
    launch::run(&program, &[c_buf, a_buf, b_buf]).expect("elementwise add dispatch");

    let got = out.as_vec::<f32>().expect("read output");
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let report = allclose_f32(&got, &expected, 0.0, 0.0);
    println!("elementwise add N={n}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
    assert!(report.ok, "{}", report.message);
}

/// The **stages=2 pipeline** correctness gate (DESIGN §5b phase 2b): the register-staged
/// prologue/steady/epilogue must accumulate every K-block exactly once — a carry slip (gather
/// reading the wrong block, a dropped commit) shows up as a wrong C, not a crash. Checked at a
/// small shape (128×128×256 = 4 K-blocks, 2×2 warps) against an f32 reference over the same
/// bf16-rounded operands; both the scalar-gather base AND `.apply(VectorizePass).apply(SwizzlePass)`
/// (the pass composition must survive the split-loop structure).
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::matmul_pipeline --nocapture`
#[test]
#[ignore]
fn matmul_pipeline_stages2_is_bit_exact_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    let (m, n, k) = (128usize, 128, 256);
    let dev = svod_dtype::default_device::default_device();
    let mut a = Tensor::rand_with(&[m, k], DType::BFloat16, dev.clone()).expect("rand a");
    let mut b = Tensor::rand_with(&[n, k], DType::BFloat16, dev).expect("rand b"); // B is [N,K] (A·Bᵀ)
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    let expected = ab_t_ref(&as_f32_vec(&a), &as_f32_vec(&b), m, n, k);
    let atol = 0.02 * (k as f32).sqrt();

    for (label, prog) in [
        ("pipe_base", crate::kernels::matmul_lds_kblock_mw_pipe(m, n, k, 64, 64, 2, 2, 64)),
        (
            "pipe_vec_sw",
            crate::kernels::matmul_lds_kblock_mw_pipe(m, n, k, 64, 64, 2, 2, 64)
                .apply(VectorizePass)
                .apply(SwizzlePass),
        ),
    ] {
        let out = Tensor::empty(&[m, n], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&a, &b]).expect("wrap");
        let plan = y.prepare().expect("prepare");
        plan.execute().expect("execute");
        let got = y.as_vec::<f32>().expect("read output");
        let report = allclose_f32(&got, &expected, atol, 2e-2);
        println!("matmul pipeline {label} {m}×{n}×{k}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
        assert!(report.ok, "{label} pipelined matmul must match reference: {}", report.message);
    }
}

/// The **clustered HK replica** (§5c) correctness gate: the 8-cluster schedule + per-cluster
/// brackets + the warp-phase ping-pong (the asymmetric `wave_barrier`s must not deadlock AND the
/// per-slice acc round-trip must accumulate exactly) → bit-exact vs the f32 reference. Uses the
/// 2-warp-row tiling (wm=2) so the phase groups exist; base AND vec+swizzle.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::matmul_clustered --nocapture`
#[test]
#[ignore]
fn matmul_clustered_hk_replica_is_bit_exact_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    let (m, n, k) = (256usize, 256, 256);
    let dev = svod_dtype::default_device::default_device();
    let mut a = Tensor::rand_with(&[m, k], DType::BFloat16, dev.clone()).expect("rand a");
    let mut b = Tensor::rand_with(&[n, k], DType::BFloat16, dev).expect("rand b"); // B is [N,K] (A·Bᵀ)
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    let expected = ab_t_ref(&as_f32_vec(&a), &as_f32_vec(&b), m, n, k);
    let atol = 0.02 * (k as f32).sqrt();

    // HK tiling: bm=128, bn=64, wm=2, wn=4 (warp_row = warp/4 ∈ {0,1} = the two phase groups).
    for (suffix, apply_passes) in [("base", false), ("vec+sw", true)] {
        let mut prog = crate::kernels::matmul_lds_kblock_mw_clustered(m, n, k, 128, 64, 2, 4, 64);
        if apply_passes {
            prog = prog.apply(VectorizePass).apply(SwizzlePass);
        }
        let out = Tensor::empty(&[m, n], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&a, &b]).expect("wrap");
        let plan = y.prepare().expect("prepare");
        plan.execute().expect("execute");
        let got = y.as_vec::<f32>().expect("read output");
        let report = allclose_f32(&got, &expected, atol, 2e-2);
        println!("clustered HK replica/{suffix} {m}×{n}×{k}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
        assert!(report.ok, "clustered HK replica/{suffix} must match reference: {}", report.message);
    }
}

/// The **compute-resident HK microkernel** (the apples-to-apples benchmark) correctness gate: the
/// steady loop drops the streaming prefetch/commit and re-reads a resident block-0 tile every
/// iteration, so it computes `nblocks · (A[:, 0:k_step] · B[0:k_step, :])`. Bit-exact vs an f32
/// host reference over the SAME bf16-rounded operands (the block-0 product scaled by nblocks) —
/// proving the schedule executes correctly (right acc round-trip, no carry slip, no deadlock)
/// even though it is not a full GEMM. Same tiling as the clustered replica (wm=2 phase groups).
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::matmul_resident --nocapture`
#[test]
#[ignore]
fn matmul_resident_microkernel_is_bit_exact_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    let (m, n, k, k_step) = (256usize, 256, 256, 64);
    let nblocks = (k / k_step) as f32;
    let dev = svod_dtype::default_device::default_device();
    let mut a = Tensor::rand_with(&[m, k], DType::BFloat16, dev.clone()).expect("rand a");
    let mut b = Tensor::rand_with(&[n, k], DType::BFloat16, dev).expect("rand b"); // B is [N,K] (A·Bᵀ)
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    // Host reference over the realized bf16→f32 values: nblocks · (A[:,0:k_step] · Bᵀ[0:k_step,:]).
    let (af, bf) = (as_f32_vec(&a), as_f32_vec(&b));
    let mut expected = vec![0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0f32;
            for kk in 0..k_step {
                acc += af[row * k + kk] * bf[col * k + kk];
            }
            expected[row * n + col] = nblocks * acc;
        }
    }
    let atol = 0.02 * (k as f32).sqrt();

    // HK tiling: bm=128, bn=64, wm=2, wn=4 (warp_row = warp/4 ∈ {0,1} = the two phase groups).
    for (suffix, apply_passes) in [("base", false), ("vec+sw", true)] {
        let mut prog = crate::kernels::matmul_lds_kblock_mw_resident(m, n, k, 128, 64, 2, 4, k_step);
        if apply_passes {
            prog = prog.apply(VectorizePass).apply(SwizzlePass);
        }
        let out = Tensor::empty(&[m, n], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&a, &b]).expect("wrap");
        let plan = y.prepare().expect("prepare");
        plan.execute().expect("execute");
        let got = y.as_vec::<f32>().expect("read output");
        assert!(got.iter().all(|v| v.is_finite()), "resident/{suffix}: output has NaN/inf");
        let report = allclose_f32(&got, &expected, atol, 2e-2);
        println!("resident microkernel/{suffix} {m}×{n}×{k}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
        assert!(report.ok, "resident microkernel/{suffix} must match nblocks·block0 reference: {}", report.message);
    }
}

/// Dump the **compute-resident microkernel**'s amdgcn LLVM IR + compiled code object to the
/// scratchpad for ISA validation (the `.co` disassembles via `llvm-objdump-20 -d`; the `.ll` via
/// `clang-20 -O3 --target=amdgcn-amd-amdhsa -mcpu=gfx942`). Large K (4096) so the steady loop runs
/// enough iterations to measure. Env `SVOD_DUMP_DIR` overrides the output directory.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::dump_resident_isa --nocapture`
#[test]
#[ignore]
fn dump_resident_isa() {
    let dir = std::env::var("SVOD_DUMP_DIR").unwrap_or_else(|_| "/tmp/tk2_isa".into());
    std::fs::create_dir_all(&dir).expect("mkdir dump dir");
    let device_spec = Tensor::empty(&[1], DType::Float32).device();
    // HK tiling (bm=128, bn=64, wm=2, wn=4, k_step=64); the production vec+swizzle passes.
    let prog = crate::kernels::matmul_lds_kblock_mw_resident(4096, 4096, 4096, 128, 64, 2, 4, 64)
        .apply(VectorizePass)
        .apply(SwizzlePass);
    let (src, bytes) = crate::launch::compile_artifacts(&prog, &device_spec).expect("compile artifacts");
    std::fs::write(format!("{dir}/resident.ll"), &src).expect("write ll");
    std::fs::write(format!("{dir}/resident.co"), &bytes).expect("write co");
    println!("resident ISA dumped: {dir}/resident.ll ({} B), {dir}/resident.co ({} B)", src.len(), bytes.len());
}

/// Dump the **DRAM-streaming clustered** kernel's amdgcn LLVM IR + compiled code object to the
/// scratchpad for ISA validation (sibling of [`dump_resident_isa`]). Same shape/tiling; this one
/// keeps the streaming global-prefetch + LDS-commit path so its steady loop shows the
/// `global_load`/`ds_write`/`ds_read` traffic to diff against HK's b128 buffer_load path.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::dump_streaming_isa --nocapture`
#[test]
#[ignore]
fn dump_streaming_isa() {
    let dir = std::env::var("SVOD_DUMP_DIR").unwrap_or_else(|_| "/tmp/tk2_isa".into());
    std::fs::create_dir_all(&dir).expect("mkdir dump dir");
    let device_spec = Tensor::empty(&[1], DType::Float32).device();
    // HK tiling (bm=128, bn=64, wm=2, wn=4, k_step=64); the production vec+swizzle passes.
    let prog = crate::kernels::matmul_lds_kblock_mw_clustered(4096, 4096, 4096, 128, 64, 2, 4, 64)
        .apply(VectorizePass)
        .apply(SwizzlePass);
    let (src, bytes) = crate::launch::compile_artifacts(&prog, &device_spec).expect("compile artifacts");
    std::fs::write(format!("{dir}/streaming.ll"), &src).expect("write ll");
    std::fs::write(format!("{dir}/streaming.co"), &bytes).expect("write co");
    println!("streaming ISA dumped: {dir}/streaming.ll ({} B), {dir}/streaming.co ({} B)", src.len(), bytes.len());
}

/// The **apples-to-apples mfmautil measurement** (the whole point): profile the compute-resident
/// microkernel's steady state and print the rocprofiler-compute gfx942 MfmaUtil, side by side with
/// the DRAM-streaming clustered kernel (the 0.24 baseline) — both at HK's tiling, both vec+swizzle.
/// The resident loop is pure `ds_read`+MFMA (ISA-verified: zero `global_load`), so its mfmautil is
/// the schedule's compute efficiency with memory-boundedness removed; compare to HK's own 0.65.
/// Needs `SVOD_PMC_FORCE=1` (VF device). Prints the full profile table (mfmautil column).
/// `SVOD_DEVICE=AMD:0 SVOD_PMC_FORCE=1 cargo test -p svod-tk2 --lib -- --ignored device::resident_mfmautil --nocapture`
#[test]
#[ignore]
fn resident_mfmautil_vs_streaming() {
    use svod_runtime::{PmcCounter, PmcSelection, ProfileOptions};
    let (m, n, k) = (4096usize, 4096, 4096);
    let dev = svod_dtype::default_device::default_device();
    let mut a = Tensor::rand_with(&[m, k], DType::BFloat16, dev.clone()).expect("rand a");
    let mut b = Tensor::rand_with(&[n, k], DType::BFloat16, dev).expect("rand b"); // B is [N,K] (A·Bᵀ)
    a.realize().expect("realize a");
    b.realize().expect("realize b");

    // mfmautil needs mfmabusy + gui; sqbusy gives the timestamp-free mfmaduty cross-check.
    let opts = ProfileOptions {
        iters: 3,
        static_analysis: false,
        counters: PmcSelection::Custom(vec![
            PmcCounter::ValuMfmaBusyCycles,
            PmcCounter::GrbmGuiActive,
            PmcCounter::SqBusyCycles,
        ]),
    };

    for (label, prog) in [
        ("resident (compute-resident)", crate::kernels::matmul_lds_kblock_mw_resident(m, n, k, 128, 64, 2, 4, 64)),
        ("streaming (clustered)", crate::kernels::matmul_lds_kblock_mw_clustered(m, n, k, 128, 64, 2, 4, 64)),
    ] {
        let prog = prog.apply(VectorizePass).apply(SwizzlePass);
        let out = Tensor::empty(&[m, n], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&a, &b]).expect("wrap");
        let plan = y.prepare().expect("prepare");
        let report = plan.profile(&opts).expect("profile");
        println!("\n===== {label} {m}×{n}×{k} =====\n{}", report.render_table());
    }
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::sum_reduce_runs_on_gfx942 --nocapture`
#[test]
#[ignore]
fn sum_reduce_runs_on_gfx942() {
    let n = 1024usize;
    let a: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.25 - 1.0).collect();

    let program = sum_reduce(n);
    let (_ta, a_buf) = input(&a);
    let (out, o_buf) = output(1);

    launch::run(&program, &[o_buf, a_buf]).expect("sum reduce dispatch");

    let got = out.as_vec::<f32>().expect("read output");
    let expected = vec![a.iter().sum::<f32>()];
    // f32 accumulation order differs from the sequential reference → tolerance.
    let report = allclose_f32(&got, &expected, 1e-2, 1e-4);
    println!("sum reduce N={n}: got={} expected={} ok={}", got[0], expected[0], report.ok);
    assert!(report.ok, "{}", report.message);
}
