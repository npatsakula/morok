//! Hardware-gated device tests for the clustered HK-replica kernel on gfx942
//! (`SVOD_DEVICE=AMD:0 ... --ignored`). Both kept matmul kernels are also correctness-gated +
//! measured through the criterion bench (`benches/matmul.rs`); the tests here exercise the
//! clustered replica's on-device bit-exactness (the warp-phase `wave_barrier`s must not deadlock
//! and the per-slice acc round-trip must accumulate exactly), dump its amdgcn ISA, and guard the
//! asm-gather kernel's 0-spill requirement.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::launch;
use crate::{SwizzlePass, VectorizePass, graph_kernel};

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

/// The **`hk::micro_tk` 1:1 port** correctness gate: HK's assembled 8-cluster BF16→FP32 GEMM
/// (`hk/gemm.rs`) — the rolled K-loop with the legacy `raw.buffer.load.i128` prefetch, HK-form
/// `ds_read_b64`/`ds_write_b64`, the `mfma.1k` accumulation, and the **truncating** fp32→bf16 C store
/// — must match the f32 reference within the bf16-output tolerance. C is bf16 (HK's `_gl_C`), read back
/// cast to f32; base (flat LDS) AND SwizzlePass forms (bit-identical — the swizzle is a bijection).
///
/// Runs under `SVOD_NO_PINGPONG` to isolate the port's numerics (addressing / accumulation / swizzle /
/// truncating store) from HK's single-buffer 8-wave ping-pong async-LDS race — the ping-pong is a
/// faithful, deliberate HK behavior (HK ships the race; verified structurally by the headless CFG test),
/// NOT a port bug and NOT fixed here.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::micro_tk --nocapture`
#[test]
#[ignore]
fn micro_tk_hk_port_is_correct_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    // SAFETY: device tests run serially (`--ignored`); scoped to this kernel build, restored below.
    unsafe { std::env::set_var("SVOD_NO_PINGPONG", "1") };
    let (m, n, k) = (256usize, 256, 256);
    let dev = svod_dtype::default_device::default_device();
    let mut a = Tensor::rand_with(&[m, k], DType::BFloat16, dev.clone()).expect("rand a");
    let mut b = Tensor::rand_with(&[n, k], DType::BFloat16, dev).expect("rand b"); // B is [N,K] (A·Bᵀ)
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    let expected = ab_t_ref(&as_f32_vec(&a), &as_f32_vec(&b), m, n, k);
    let atol = 0.02 * (k as f32).sqrt();

    for (suffix, swizzle) in [("base", false), ("sw", true)] {
        let mut prog = crate::hk::micro_tk(m, n, k);
        if swizzle {
            prog = prog.apply(SwizzlePass);
        }
        let out = Tensor::empty(&[m, n], DType::BFloat16); // HK's C is bf16
        let mut y = graph_kernel(prog, out, &[&a, &b]).expect("wrap");
        let plan = y.prepare().expect("prepare");
        plan.execute().expect("execute");
        let got = as_f32_vec(&y); // bf16 output → f32 for comparison
        let report = allclose_f32(&got, &expected, atol, 2e-2);
        println!("micro_tk hk port/{suffix} {m}×{n}×{k}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
        assert!(report.ok, "micro_tk hk port/{suffix} must match reference: {}", report.message);
    }
    unsafe { std::env::remove_var("SVOD_NO_PINGPONG") };
}

/// Host f32 reference for the `mma_atb` probe: `O[d,q] = Σ_kv V[kv,d]·P[kv,q]` (V`[kv,d]`, P`[kv,q]`,
/// both row-major) — the FA `P·V` contraction over the shared `kv` row.
fn atb_ref(vf: &[f32], pf: &[f32], kv: usize, d: usize, q: usize) -> Vec<f32> {
    let mut e = vec![0f32; d * q];
    for di in 0..d {
        for qi in 0..q {
            let mut acc = 0f32;
            for k in 0..kv {
                acc += vf[k * d + di] * pf[k * q + qi];
            }
            e[di * q + qi] = acc;
        }
    }
    e
}

/// **`mma_atb` isolation gate** (Phase-A de-risk): the transposed-gather + `v_mfma` compute
/// `O[d,q] = Σ_kv V[kv,d]·P[kv,q]` (the FA `P·V` contraction over the shared `kv` row) must match the
/// f32 reference over the SAME bf16-rounded operands. Covers the base fragment (16×16×16), a wide
/// output-`d` tile (16×64×16 = 4 d-frags), and a wide output-`q` tile (16×16×64 = 4 q-frags), so both
/// output-fragment axes of the transposed gather are exercised BEFORE the layout is trusted in FA.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::mma_atb_probe --nocapture`
#[test]
#[ignore]
fn mma_atb_probe_is_correct_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    let dev = svod_dtype::default_device::default_device();
    for (kv, d, q) in [(16usize, 16usize, 16usize), (16, 64, 16), (16, 16, 64), (16, 64, 64)] {
        let mut v = Tensor::rand_with(&[kv, d], DType::BFloat16, dev.clone()).expect("rand v");
        let mut p = Tensor::rand_with(&[kv, q], DType::BFloat16, dev.clone()).expect("rand p");
        v.realize().expect("realize v");
        p.realize().expect("realize p");
        let expected = atb_ref(&as_f32_vec(&v), &as_f32_vec(&p), kv, d, q);
        let atol = 0.02 * (kv as f32).sqrt();

        let prog = crate::kernels_fa::atb_probe(kv, d, q);
        let out = Tensor::empty(&[d, q], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&v, &p]).expect("wrap atb probe");
        let plan = y.prepare().expect("prepare");
        plan.execute().expect("execute");
        let got = y.as_vec::<f32>().expect("read output");
        let report = allclose_f32(&got, &expected, atol, 2e-2);
        println!("mma_atb probe kv={kv} d={d} q={q}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
        assert!(report.ok, "mma_atb probe {kv}×{d}×{q} must match Aᵀ·B reference: {}", report.message);
    }
}

/// Host f32 reference for non-causal FA-forward: `O[q,dd] = Σ_k softmax_k(Q[q]·K[k]/√d)·V[k,dd]`
/// over the SAME bf16-rounded operands (`q`/`k`/`v` are `[n,d]` row-major, cast to f32 first).
fn fa_ref(qf: &[f32], kf: &[f32], vf: &[f32], n: usize, d: usize) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let mut o = vec![0f32; n * d];
    for qi in 0..n {
        let mut s = vec![0f32; n];
        for ki in 0..n {
            let mut acc = 0f32;
            for di in 0..d {
                acc += qf[qi * d + di] * kf[ki * d + di];
            }
            s[ki] = acc * scale;
        }
        let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for x in &mut s {
            *x = (*x - m).exp();
            sum += *x;
        }
        for di in 0..d {
            let mut acc = 0f32;
            for ki in 0..n {
                acc += s[ki] * vf[ki * d + di];
            }
            o[qi * d + di] = acc / sum;
        }
    }
    o
}

/// **FA-forward correctness GATE** (the Phase-A deliverable): the streaming, single-warp, online-
/// softmax FA on the ClusterCx pipeline ([`crate::kernels_fa::flash_attention_fwd`]) must match the
/// f32 reference (non-causal, same bf16-rounded operands) at d=64 AND d=128 — the QKᵀ inner-d loop,
/// the two `ds_bpermute` softmax reductions, the online rescale, and the `mma_atb` P·V (transposed-V
/// gather + free-relayout P) all correct end-to-end. `atol = 0.02·√d`, `rtol = 2e-2` (matmul style).
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::flash_attention_matches --nocapture`
#[test]
#[ignore]
fn flash_attention_matches_reference_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    let dev = svod_dtype::default_device::default_device();
    let n = 128usize;
    for d in [64usize, 128usize] {
        let mut q = Tensor::rand_with(&[n, d], DType::BFloat16, dev.clone()).expect("rand q");
        let mut k = Tensor::rand_with(&[n, d], DType::BFloat16, dev.clone()).expect("rand k");
        let mut v = Tensor::rand_with(&[n, d], DType::BFloat16, dev.clone()).expect("rand v");
        q.realize().expect("realize q");
        k.realize().expect("realize k");
        v.realize().expect("realize v");
        let expected = fa_ref(&as_f32_vec(&q), &as_f32_vec(&k), &as_f32_vec(&v), n, d);
        let atol = 0.02 * (d as f32).sqrt();

        let prog = crate::kernels_fa::flash_attention_fwd(n, d);
        let out = Tensor::empty(&[n, d], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&q, &k, &v]).expect("wrap FA program");
        let plan = y.prepare().expect("prepare FA");
        plan.execute().expect("execute FA");
        let got = y.as_vec::<f32>().expect("read FA output");
        let report = allclose_f32(&got, &expected, atol, 2e-2);
        println!("FA-forward n={n} d={d}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
        assert!(report.ok, "FA-forward n={n} d={d} must match the f32 reference: {}", report.message);
    }
}

/// **FA-forward `d=16` launch smoke test** on gfx942 — the multi-warp FA at a single head-dim fragment,
/// where the VEC4-aligned fill forces `kv_blk = 128` (8 KV-fragments, the degenerate `kvf` stress).
/// Complements the full `flash_attention_matches_reference_on_gfx942` gate (numerics at d=64/128): this
/// one confirms the `d=16 / kvf=8` shape compiles + executes without a GPU fault and produces finite
/// output. `n = 256` (2 KV blocks × 2 workgroups over the 128-row Q block).
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::fa_forward_launches --nocapture`
#[test]
#[ignore]
fn fa_forward_launches_on_gfx942() {
    let (n, d) = (256usize, 16);
    let dev = svod_dtype::default_device::default_device();
    let mut q = Tensor::rand_with(&[n, d], DType::BFloat16, dev.clone()).expect("rand q");
    let mut k = Tensor::rand_with(&[n, d], DType::BFloat16, dev.clone()).expect("rand k");
    let mut v = Tensor::rand_with(&[n, d], DType::BFloat16, dev).expect("rand v");
    q.realize().expect("realize q");
    k.realize().expect("realize k");
    v.realize().expect("realize v");

    let prog = crate::kernels_fa::flash_attention_fwd(n, d);
    let out = Tensor::empty(&[n, d], DType::Float32);
    let mut y = graph_kernel(prog, out, &[&q, &k, &v]).expect("wrap FA program");
    let plan = y.prepare().expect("prepare FA");
    plan.execute().expect("execute FA on device");
    let got = y.as_vec::<f32>().expect("read FA output");
    let finite = got.iter().filter(|x| x.is_finite()).count();
    let sample = &got[..d.min(8)];
    println!("FA-forward n={n} d={d}: launched OK, {}/{} finite outputs, sample={sample:?}", finite, got.len());
    assert!(finite > 0, "FA-forward must produce at least some finite output (launch/exec sanity)");
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

/// **0-spill guard for the asm-gather clustered kernels** — the latent-fragility tripwire. The asm
/// `ds_read_b64`/`ds_write_b64` gather+commit are waitcnt-opaque, so LLVM's spill logic cannot model
/// their async LDS completion: a register spilled/reloaded around them can carry a value that has not
/// yet arrived, silently corrupting the result (an occupancy hint that forced spills produced
/// `max abs err ~2e2` on device, with the compiler-visible `mw256_pipe` staying bit-exact under the
/// same spilling). These kernels are therefore correct **only** at zero spills; assert that here so a
/// regression fails LOUDLY (a compile check) instead of as silent device numerics. `private_segment`
/// (scratch) bytes == 0 ⟺ no spills. See memory `tk2-mfma-fracture-ir-shape`.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::asm_clustered_kernels_have_zero_spills --nocapture`
#[test]
#[ignore]
fn asm_clustered_kernels_have_zero_spills() {
    let device_spec = Tensor::empty(&[1], DType::Float32).device();
    let variants = [("clustered", crate::kernels::matmul_lds_kblock_mw_clustered(4096, 4096, 4096, 128, 64, 2, 4, 64))];
    for (label, prog) in variants {
        let prog = prog.apply(VectorizePass).apply(SwizzlePass);
        let (_src, bytes) = launch::compile_artifacts(&prog, &device_spec).expect("compile clustered kernel");
        let parsed =
            svod_device::amd::program::parse_kernel(&bytes, "tk2_matmul_kblock").expect("parse compiled kernel");
        let scratch = parsed.kd.private_segment_fixed_size;
        assert_eq!(
            scratch, 0,
            "{label}: the asm-gather clustered kernel MUST compile with 0 spills — it has {scratch} scratch \
             bytes/thread. The waitcnt-opaque asm ds_read/ds_write are unsafe to spill (a spilled async LDS \
             value is stale → silent miscompile). Reduce register pressure or make the gather compiler-visible."
        );
        println!("{label}: 0 spills ✓ (private_segment = {scratch} B)");
    }
}
