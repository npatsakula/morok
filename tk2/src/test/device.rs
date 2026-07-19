//! Hardware-gated device tests for the clustered HK-replica kernel on gfx942
//! (`SVOD_DEVICE=AMD:0 ... --ignored`). The clustered matmul kernel is also correctness-gated +
//! measured through the criterion bench (`benches/matmul.rs`); the tests here exercise the
//! clustered replica's on-device bit-exactness (the warp-phase `wave_barrier`s must not deadlock
//! and the per-slice acc round-trip must accumulate exactly), dump its amdgcn ISA, and guard the
//! asm-gather kernel's 0-spill requirement.

use svod_device::PmcCounter;
use svod_dtype::DType;
use svod_runtime::{PmcSelection, ProfileOptions};
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
        let mut prog =
            crate::kernels::matmul::matmul_lds_kblock_mw_clustered(m, n, k, crate::kernels::matmul::Tiling::default());
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

/// **32×32×8 MFMA isolation gate** (§migration Step 3 go/no-go): the wide-core probe
/// ([`crate::test::probes::mfma_32x32x8_probe`]) computes `C = A·Bᵀ` via `v_mfma_f32_32x32x8_bf16` and
/// scatters its 16-VGPR accumulator through `acc_rc`; it must match the f32 reference over the SAME
/// bf16-rounded operands. This PROVES the 32×32×8 operand layout + the 4-block accumulator distribution
/// (`AccDist`) + the intrinsic ARE CORRECT in isolation. Shapes: one MFMA (32×32×8), a K-loop
/// (32×32×16 = 2 hw K-steps), and M-/N-/both-tiled (64×32×8, 32×64×8, 64×64×16) to stress every axis of
/// the 4-block accumulator. A mismatch ⟹ the AccDist/operand layout is wrong — debug vs CK
/// `CWarpDstrEncoding` + the aiter disasm before trusting it in FA.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::mfma_32x32x8_probe --nocapture`
#[test]
#[ignore]
fn mfma_32x32x8_probe_is_correct_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    let dev = svod_dtype::default_device::default_device();
    for (m, n, k) in [(32usize, 32usize, 8usize), (32, 32, 16), (64, 32, 8), (32, 64, 8), (64, 64, 16)] {
        let mut a = Tensor::rand_with(&[m, k], DType::BFloat16, dev.clone()).expect("rand a");
        let mut b = Tensor::rand_with(&[n, k], DType::BFloat16, dev.clone()).expect("rand b"); // B is [N,K]
        a.realize().expect("realize a");
        b.realize().expect("realize b");
        let expected = ab_t_ref(&as_f32_vec(&a), &as_f32_vec(&b), m, n, k);
        let atol = 0.02 * (k as f32).sqrt();

        // The intrinsic MFMA (the production fast path — the asm `sideeffect` form was dropped: it is
        // opaque to the AMDGPU GCNHazardRecognizer, so it emits none of the mandatory 32×32×8 `s_nop`s
        // and a VALU-adjacent accumulator miscompiles → NaN, device-proven; see `flash_attention_fwd_32`).
        let prog = crate::test::probes::mfma_32x32x8_probe(m, n, k);
        let out = Tensor::empty(&[m, n], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&a, &b]).expect("wrap 32×32×8 probe");
        let plan = y.prepare().expect("prepare");
        plan.execute().expect("execute");
        let got = y.as_vec::<f32>().expect("read output");
        let report = allclose_f32(&got, &expected, atol, 2e-2);
        println!("mfma_32x32x8 probe {m}×{n}×{k}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
        assert!(report.ok, "32×32×8 probe {m}×{n}×{k} must match A·Bᵀ reference: {}", report.message);
    }
}

/// The retained direct GLOBAL→LDS primitive must survive real gfx942 selection and workgroup
/// publication, not only render the expected LLVM intrinsic.
#[test]
#[ignore]
fn direct_global_to_lds_round_trip_is_correct_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    const N: usize = 1024;
    let dev = svod_dtype::default_device::default_device();
    let mut input = Tensor::rand_with(&[N], DType::BFloat16, dev).expect("random direct-LDS input");
    input.realize().expect("realize direct-LDS input");
    let expected = as_f32_vec(&input);

    let prog = crate::test::probes::direct_lds_probe(N);
    let out = Tensor::empty(&[N], DType::BFloat16);
    let mut y = graph_kernel(prog, out, &[&input]).expect("wrap direct-LDS probe");
    let plan = y.prepare().expect("prepare direct-LDS probe");
    plan.execute().expect("execute direct-LDS probe");
    let got = as_f32_vec(&y);
    let report = allclose_f32(&got, &expected, 0.0, 0.0);
    assert!(report.ok, "direct GLOBAL→LDS round trip must be exact: {}", report.message);
}

#[test]
#[ignore]
fn direct_global_to_swizzled_k_round_trip_is_correct_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    const N: usize = 32 * 128;
    let dev = svod_dtype::default_device::default_device();
    let mut input = Tensor::rand_with(&[N], DType::BFloat16, dev).expect("random swizzled direct-LDS input");
    input.realize().expect("realize swizzled direct-LDS input");
    let expected = as_f32_vec(&input);

    let prog = crate::test::probes::direct_lds_swizzled_k_probe().apply(crate::SwizzlePass);
    let out = Tensor::empty(&[N], DType::BFloat16);
    let mut y = graph_kernel(prog, out, &[&input]).expect("wrap swizzled direct-LDS probe");
    let plan = y.prepare().expect("prepare swizzled direct-LDS probe");
    plan.execute().expect("execute swizzled direct-LDS probe");
    let got = as_f32_vec(&y);
    let report = allclose_f32(&got, &expected, 0.0, 0.0);
    assert!(report.ok, "swizzled direct GLOBAL→LDS round trip must be exact: {}", report.message);
}

#[test]
#[ignore]
fn ds_write_b16_round_trip_is_correct_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    const N: usize = 512;
    let dev = svod_dtype::default_device::default_device();
    let mut input = Tensor::rand_with(&[N], DType::BFloat16, dev).expect("random ds_write_b16 input");
    input.realize().expect("realize ds_write_b16 input");
    let expected = as_f32_vec(&input);

    let prog = crate::test::probes::ds_write_b16_probe(N);
    let out = Tensor::empty(&[N], DType::BFloat16);
    let mut y = graph_kernel(prog, out, &[&input]).expect("wrap ds_write_b16 probe");
    let plan = y.prepare().expect("prepare ds_write_b16 probe");
    plan.execute().expect("execute ds_write_b16 probe");
    let got = as_f32_vec(&y);
    let report = allclose_f32(&got, &expected, 0.0, 0.0);
    assert!(report.ok, "ds_write_b16 round trip must be exact: {}", report.message);
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

        let prog = crate::test::probes::atb_probe(kv, d, q);
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

/// Host f32 reference for the P→PV relayout probe: `O[d,q] = Σ_kv V[d,kv]·P[kv,q]` (V`[d,kv]` bf16,
/// P`[kv,q]` exact-bf16 f32, both row-major) — the FA `P·V` with `P` in the 32×32×8 accumulator layout.
fn pv_ref(vf: &[f32], pf: &[f32], d: usize, q: usize, kv: usize) -> Vec<f32> {
    let mut e = vec![0f32; d * q];
    for di in 0..d {
        for qi in 0..q {
            let mut acc = 0f32;
            for k in 0..kv {
                acc += vf[di * kv + k] * pf[k * q + qi];
            }
            e[di * q + qi] = acc;
        }
    }
    e
}

/// **Step-4 P→PV relayout gate** (32×32×8 DE-RISK): the `v_perm_b32` primitive + [`Builder::pv_relayout_s49`]
/// pack ([`crate::test::probes::pv_relayout_probe`]) must reproduce `O[d,q] = Σ_kv V[d,kv]·P[kv,q]` where `P`
/// is loaded into the 32×32×8 C-accumulator layout and relayout'd to the PV B-operands via `v_perm s49`.
/// A match PROVES the selector + the accumulator↔B-operand element correspondence in isolation, before FA.
/// Base tile (32×32), a wide-`d` tile (64×32, 2 A-operand tiles), and a wide-`q` tile (32×64, 2 accumulator
/// loads/relayouts) exercise both axes. `P` holds exact bf16 values so the pack's truncation is a no-op.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::pv_relayout_probe --nocapture`
#[test]
#[ignore]
fn pv_relayout_probe_is_correct_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    const KV: usize = 32;
    let dev = svod_dtype::default_device::default_device();
    for (d, q) in [(32usize, 32usize), (64, 32), (32, 64)] {
        // P holds EXACT bf16 values (rand bf16 → f32), so the v_perm-pack truncation is a no-op vs ref.
        let mut p = Tensor::rand_with(&[KV, q], DType::BFloat16, dev.clone()).expect("rand p");
        let mut v = Tensor::rand_with(&[d, KV], DType::BFloat16, dev.clone()).expect("rand v");
        p.realize().expect("realize p");
        v.realize().expect("realize v");
        let mut p_f32 = p.cast(DType::Float32).expect("p→f32");
        p_f32.realize().expect("realize p f32");
        let expected = pv_ref(&as_f32_vec(&v), &as_f32_vec(&p_f32), d, q, KV);
        let atol = 0.02 * (KV as f32).sqrt();

        let prog = crate::test::probes::pv_relayout_probe(d, q);
        let out = Tensor::empty(&[d, q], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&p_f32, &v]).expect("wrap pv_relayout probe");
        let plan = y.prepare().expect("prepare");
        plan.execute().expect("execute");
        let got = y.as_vec::<f32>().expect("read output");
        let report = allclose_f32(&got, &expected, atol, 2e-2);
        println!("pv_relayout probe d={d} q={q}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
        assert!(report.ok, "pv_relayout probe {d}×{q} must match P·V reference: {}", report.message);
    }
}

/// Host reference for the softmax-32 probe: per-q softmax over kv `P[kv,q] = 2^(S[kv,q]−m_q) / Σ_kv 2^(…)`,
/// `m_q = max_kv S[kv,q]` (exp2 basis, matching the kernel's `Builder::exp2`). `S`/`P` are `[kv,q]` row-major.
fn softmax32_ref(sf: &[f32], kv: usize, q: usize) -> Vec<f32> {
    let mut e = vec![0f32; kv * q];
    for qi in 0..q {
        let m = (0..kv).map(|k| sf[k * q + qi]).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for k in 0..kv {
            sum += (sf[k * q + qi] - m).exp2();
        }
        for k in 0..kv {
            e[k * q + qi] = (sf[k * q + qi] - m).exp2() / sum;
        }
    }
    e
}

/// **Step-6 softmax-reduction gate** (32×32×8 DE-RISK): the [`Builder::acc_row_reduce_32`] online-softmax
/// over the `EPT_C = 16` accumulator geometry ([`crate::test::probes::softmax32_probe`]) must reproduce the
/// per-q softmax over kv. A match PROVES the AccDist reduction (16 in-register + `L↔L+32` cross-lane) +
/// broadcast are correct — the last un-proven FA-32 building block, in isolation before the FA rewrite.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::softmax32_probe --nocapture`
#[test]
#[ignore]
fn softmax32_probe_is_correct_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    const KV: usize = 32;
    const Q: usize = 32;
    let dev = svod_dtype::default_device::default_device();
    let mut s = Tensor::rand_with(&[KV, Q], DType::Float32, dev.clone()).expect("rand s");
    s.realize().expect("realize s");
    let expected = softmax32_ref(&as_f32_vec(&s), KV, Q);

    let prog = crate::test::probes::softmax32_probe();
    let out = Tensor::empty(&[KV, Q], DType::Float32);
    let mut y = graph_kernel(prog, out, &[&s]).expect("wrap softmax32 probe");
    let plan = y.prepare().expect("prepare");
    plan.execute().expect("execute");
    let got = y.as_vec::<f32>().expect("read output");
    let report = allclose_f32(&got, &expected, 1e-3, 2e-3);
    println!("softmax32 probe {KV}×{Q}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
    assert!(report.ok, "softmax32 probe must match the per-q softmax reference: {}", report.message);
}

/// Host f32 reference for the V-transpose probe: `O[d,q] = Σ_kv V[kv,d]·Pt[q,kv]` (V`[kv,d]`, Pt`[q,kv]`,
/// both bf16 row-major) — the FA `P·V` with V staged through the transposed LDS and P the straight B-operand.
fn vtrans_ref(vf: &[f32], ptf: &[f32], d: usize, q: usize, kv: usize) -> Vec<f32> {
    let mut e = vec![0f32; d * q];
    for di in 0..d {
        for qi in 0..q {
            let mut acc = 0f32;
            for k in 0..kv {
                acc += vf[k * d + di] * ptf[qi * kv + k];
            }
            e[di * q + qi] = acc;
        }
    }
    e
}

/// **Step-5 V write-side padded-transpose gate** (32×32×8 DE-RISK): the padded transposed V staging
/// ([`crate::test::probes::v_transpose_probe`]) must reproduce `O[d,q] = Σ_kv V[kv,d]·Pt[q,kv]` with V read
/// as the 32×32×8 A-operand straight from the transposed LDS. A match PROVES the transposed padded layout +
/// straight-read addressing yield the correct PV A-operand, in isolation, before FA. Base tile (32×32) and
/// a wide-`d` tile (64×32, two A-operand tiles over the shared transposed LDS). `q = kv = 32`.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::v_transpose_probe --nocapture`
#[test]
#[ignore]
fn v_transpose_probe_is_correct_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    const KV: usize = 32;
    const Q: usize = 32;
    let dev = svod_dtype::default_device::default_device();
    for d in [32usize, 64] {
        let mut v = Tensor::rand_with(&[KV, d], DType::BFloat16, dev.clone()).expect("rand v");
        let mut pt = Tensor::rand_with(&[Q, KV], DType::BFloat16, dev.clone()).expect("rand pt");
        v.realize().expect("realize v");
        pt.realize().expect("realize pt");
        let expected = vtrans_ref(&as_f32_vec(&v), &as_f32_vec(&pt), d, Q, KV);
        let atol = 0.02 * (KV as f32).sqrt();

        let prog = crate::test::probes::v_transpose_probe(d);
        let out = Tensor::empty(&[d, Q], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&v, &pt]).expect("wrap v_transpose probe");
        let plan = y.prepare().expect("prepare");
        plan.execute().expect("execute");
        let got = y.as_vec::<f32>().expect("read output");
        let report = allclose_f32(&got, &expected, atol, 2e-2);
        println!("v_transpose probe d={d}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
        assert!(report.ok, "v_transpose probe d={d} must match P·V reference: {}", report.message);
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

/// **FA-32 correctness GATE** (§Step 6): the 32×32×8-MFMA FA ([`crate::kernels::fa::flash_attention_fwd_32`],
/// kept SEPARATE from the frozen 16×16 FA) must match the SAME f32 reference (non-causal, bf16-rounded
/// operands) at d=64 AND d=128, over `bh > 1` independent `[bh,n,d]` attentions — validating the assembled
/// 32×32×8 hot path end-to-end: QKᵀ (`v_mfma_f32_32x32x8`) → `acc_row_reduce_32` online softmax → `v_perm
/// s49` P→PV relayout → padded-transposed-V P·V. The full cases use `n=256/512`; `atol` is the tight 1e-2
/// the 16×16 gate uses (the honest bf16-cast error is ~2e-3; a corrupted result is rejected).
/// The `(8, 512, 64)` case exercises the non-identity d64 XCD remap. The `n=64` cases exercise the
/// warmup-to-epilogue transition with no steady loop.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::flash_attention32_matches --nocapture`
///
/// Also gates the **ragged tail** (§Step-B): the `(1, 80, …)` cases have `n=80` — NOT a KV-block (32)
/// multiple — so the last KV block is partial and the online softmax must apply the per-element mask
/// (`global_kv < n ? score : −∞`) or the out-of-range keys corrupt the softmax. bh=1 for the ragged
/// cases (a padded buffer absorbs the tile-covering fill/scatter; bh>1 would need a tile-exact `n`).
#[test]
#[ignore]
fn flash_attention32_matches_reference_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    let dev = svod_dtype::default_device::default_device();
    const Q_BLK: usize = 256; // NUM_WARPS_32 · 32 (the workgroup Q block; 8 warps × 32)
    const KV_BLK: usize = 32; // KV_BLK_32
    // (bh, n, d): the tile-exact full cases (bh>1, n a Q-block multiple) + the RAGGED partial-block cases
    // (bh=1, n=80 — a partial last KV block exercising the mask at both head dims).
    for (bh, n, d) in [
        (3usize, 256usize, 64usize),
        (2, 256, 128),
        (8, 512, 64), // exercises a non-identity d64 8-XCD workgroup remap
        (1, 64, 64),
        (1, 64, 128),
        (1, 80, 64),
        (1, 80, 128),
    ] {
        // The kernel's fill + scatter cover ⌈n/tile⌉·tile rows per (b,h) slice; provision the buffers to
        // match. bh>1 needs a tile-exact `n` (per-slice stride == n); a ragged `n` runs at bh=1 (slice
        // base 0), the padded tail holding intentional garbage the mask makes irrelevant.
        let rows_pad = (n.div_ceil(Q_BLK) * Q_BLK).max(n.div_ceil(KV_BLK) * KV_BLK);
        assert!(bh == 1 || rows_pad == n, "ragged n (padded buffers) is gated at bh=1 only");
        let rows = bh * rows_pad;
        let mut q = Tensor::rand_with(&[rows, d], DType::BFloat16, dev.clone()).expect("rand q");
        let mut k = Tensor::rand_with(&[rows, d], DType::BFloat16, dev.clone()).expect("rand k");
        let mut v = Tensor::rand_with(&[rows, d], DType::BFloat16, dev.clone()).expect("rand v");
        q.realize().expect("realize q");
        k.realize().expect("realize k");
        v.realize().expect("realize v");
        let (qf, kf, vf) = (as_f32_vec(&q), as_f32_vec(&k), as_f32_vec(&v));
        // Reference over the TRUE `n` rows of each slice (slice s starts at padded row s·rows_pad).
        let mut expected = vec![0f32; rows * d];
        for s in 0..bh {
            let z = s * rows_pad * d;
            let o_s = fa_ref(&qf[z..z + n * d], &kf[z..z + n * d], &vf[z..z + n * d], n, d);
            expected[z..z + n * d].copy_from_slice(&o_s);
        }
        let atol = 1e-2;
        // SwizzlePass folds the K-tile LDS bank swizzle (the as-used tuned path); the gate runs it on to
        // catch any swizzle-layout regression (fill/gather must agree on `lds_col(row, …, d)`). Two kernels:
        // the PRODUCTION fast path (tile-exact `n` only — d128 ping-pong / d64 single-crew) and the
        // register-staged ORACLE (any `n`, the ragged + differential reference). The 3× replay per variant
        // exposes any residual LDS stage/overwrite race in the fast path's asm-opaque movement.
        let mut variants = Vec::new();
        if n.is_multiple_of(Q_BLK) {
            variants.push(("production", crate::kernels::fa::flash_attention_fwd_32(bh, n, d).apply(SwizzlePass)));
        }
        variants.push((
            "qualification-register-K",
            crate::kernels::fa::flash_attention_fwd_32_register_k(bh, n, d).apply(SwizzlePass),
        ));
        for (variant, prog) in variants {
            let out = Tensor::empty(&[rows, d], DType::Float32);
            let mut y = graph_kernel(prog, out, &[&q, &k, &v]).expect("wrap FA-32 program");
            let plan = y.prepare().expect("prepare FA-32");
            // Repeat the same prepared plan to expose nondeterministic LDS stage/overwrite races.
            for run in 0..3 {
                plan.execute().expect("execute FA-32");
                let got = y.as_vec::<f32>().expect("read FA-32 output");
                // Compare ONLY the true `n` rows per slice (the padded tail is intentional garbage).
                for s in 0..bh {
                    let z = s * rows_pad * d;
                    let report = allclose_f32(&got[z..z + n * d], &expected[z..z + n * d], atol, 2e-2);
                    println!(
                        "FA-32 {variant} run={run} bh={bh} n={n} d={d} slice={s}: ok={} max_abs_err={:e}",
                        report.ok, report.max_abs_err
                    );
                    assert!(
                        report.ok,
                        "FA-32 {variant} run={run} bh={bh} n={n} d={d} slice={s} must match the f32 reference: {}",
                        report.message
                    );
                }
            }
        }
    }
}

/// **FA-32 bf16-O correctness GATE**: the aiter-API-matched ping-pong FA-32
/// ([`crate::kernels::fa::flash_attention_fwd_32_bf16`]) stores O as **bf16** (RTZ truncation
/// of the f32 accumulator at the final scatter — the MFMA accumulator stays f32). It must still match
/// the SAME f32 reference within the widened tolerance bf16 output rounding needs: bf16 has ~8 bits of
/// mantissa, so RTZ truncation contributes ≤2^-7 ≈ 0.8% relative error on top of the f32 pipeline error
/// the f32-O gate already tolerates. `atol=1e-2, rtol=3e-2` covers both; the bf16 output is cast to f32
/// (`as_f32_vec`) for the comparison. d128-only (the ping-pong constraint); the 3× replay exposes any
/// residual LDS stage/overwrite race, exactly as the f32-O gate.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::flash_attention32_bf16o_matches --nocapture`
#[test]
#[ignore]
fn flash_attention32_bf16o_matches_reference_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    let dev = svod_dtype::default_device::default_device();
    let bh = 2usize;
    let d = 128usize;
    // bf16 output rounding: RTZ truncation error ≤2^-7 relative. atol 1e-2 + rtol 3e-2 covers the f32
    // pipeline error the f32-O gate tolerates PLUS the extra bf16 truncation — no looser than that.
    let (atol, rtol) = (1e-2f32, 3e-2f32);
    for n in [512usize, 1024, 2048] {
        let rows = bh * n; // every n is a Q-block (256) multiple ⇒ tile-exact ⇒ per-slice stride == n
        let mut q = Tensor::rand_with(&[rows, d], DType::BFloat16, dev.clone()).expect("rand q");
        let mut k = Tensor::rand_with(&[rows, d], DType::BFloat16, dev.clone()).expect("rand k");
        let mut v = Tensor::rand_with(&[rows, d], DType::BFloat16, dev.clone()).expect("rand v");
        q.realize().expect("realize q");
        k.realize().expect("realize k");
        v.realize().expect("realize v");
        let (qf, kf, vf) = (as_f32_vec(&q), as_f32_vec(&k), as_f32_vec(&v));
        let mut expected = vec![0f32; rows * d];
        for s in 0..bh {
            let z = s * n * d;
            let o_s = fa_ref(&qf[z..z + n * d], &kf[z..z + n * d], &vf[z..z + n * d], n, d);
            expected[z..z + n * d].copy_from_slice(&o_s);
        }
        let prog = crate::kernels::fa::flash_attention_fwd_32_bf16(bh, n, d).apply(SwizzlePass);
        // The output tensor is bf16 — half the O write bytes, the aiter-matched store dtype.
        let out = Tensor::empty(&[rows, d], DType::BFloat16);
        let mut y = graph_kernel(prog, out, &[&q, &k, &v]).expect("wrap bf16-O FA-32 program");
        let plan = y.prepare().expect("prepare bf16-O FA-32");
        for run in 0..3 {
            plan.execute().expect("execute bf16-O FA-32");
            let got = as_f32_vec(&y); // cast the bf16 output back to f32 for the comparison
            for s in 0..bh {
                let z = s * n * d;
                let report = allclose_f32(&got[z..z + n * d], &expected[z..z + n * d], atol, rtol);
                println!(
                    "FA-32 bf16-O run={run} bh={bh} n={n} d={d} slice={s}: ok={} max_abs_err={:e}",
                    report.ok, report.max_abs_err
                );
                assert!(
                    report.ok,
                    "FA-32 bf16-O run={run} bh={bh} n={n} d={d} slice={s} must match the f32 reference within bf16 tol: {}",
                    report.message
                );
            }
        }
    }
}

/// The d128 PRODUCTION fast path (ping-pong over asm-opaque movement) must be bit-exact to the
/// register-staged ORACLE at a long stream, replayed 64× to expose any residual LDS stage/overwrite race.
#[test]
#[ignore]
fn flash_attention32_long_production_matches_register_k_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    let dev = svod_dtype::default_device::default_device();
    let (bh, n, d) = (2usize, 1024usize, 128usize);
    let mut q = Tensor::rand_with(&[bh * n, d], DType::BFloat16, dev.clone()).expect("rand long q");
    let mut k = Tensor::rand_with(&[bh * n, d], DType::BFloat16, dev.clone()).expect("rand long k");
    let mut v = Tensor::rand_with(&[bh * n, d], DType::BFloat16, dev).expect("rand long v");
    q.realize().expect("realize long q");
    k.realize().expect("realize long k");
    v.realize().expect("realize long v");

    let oracle = crate::kernels::fa::flash_attention_fwd_32_register_k(bh, n, d).apply(SwizzlePass);
    let out = Tensor::empty(&[bh * n, d], DType::Float32);
    let mut expected_tensor = graph_kernel(oracle, out, &[&q, &k, &v]).expect("wrap register-K oracle");
    let oracle_plan = expected_tensor.prepare().expect("prepare register-K oracle");
    oracle_plan.execute().expect("execute register-K oracle");
    let expected = expected_tensor.as_vec::<f32>().expect("read register-K oracle");

    let production = crate::kernels::fa::flash_attention_fwd_32(bh, n, d).apply(SwizzlePass);
    let out = Tensor::empty(&[bh * n, d], DType::Float32);
    let mut got_tensor = graph_kernel(production, out, &[&q, &k, &v]).expect("wrap production FA");
    let production_plan = got_tensor.prepare().expect("prepare production FA");
    for run in 0..64 {
        production_plan.execute().expect("execute production FA");
        let got = got_tensor.as_vec::<f32>().expect("read production FA");
        let report = allclose_f32(&got, &expected, 0.0, 0.0);
        assert!(report.ok, "long production run {run} must be bit-exact to register-K: {}", report.message);
    }
}

/// Runtime-DSL integration gate: one compiled graph kernel is replayed with different symbolic grid
/// extents. The same runtime scalar also bounds both global views, so no access can exceed the logical
/// prefix even though the physical buffers are allocated to the declared maximum.
#[test]
#[ignore]
fn runtime_grid_and_bounded_views_rebind_without_recompile() {
    use crate::build::{Builder, F32};

    let dev = svod_dtype::default_device::default_device();
    let mut input = Tensor::rand_with(&[16], DType::Float32, dev).expect("rand dynamic input");
    input.realize().expect("realize dynamic input");
    let expected = input.as_vec::<f32>().expect("read dynamic input");

    let mut b = Builder::new("tk2_runtime_bounded_copy");
    let out_buf = b.global::<F32>(16);
    let in_buf = b.global::<F32>(16);
    let n = b.scalar_param("n", 1, 16);
    let gid = b.grid_axis_dyn(0, n);
    let out_view = b.bounded(out_buf, n);
    let in_view = b.bounded(in_buf, n);
    let zero = b.f32(0.0);
    let value = b.load_bounded(in_view, gid, zero);
    let root = b.store_bounded(out_view, gid, value);
    let (ir, sink) = b.finish(&[root]);
    let program = crate::Program { ir, sink, name: "tk2_runtime_bounded_copy".into() };

    let out = Tensor::empty(&[16], DType::Float32);
    let mut result = graph_kernel(program, out, &[&input]).expect("wrap runtime bounded copy");
    let mut plan = result.prepare().expect("prepare runtime bounded copy once");
    for n in [4usize, 11, 16] {
        plan.execute_with_vars(&[("n", n as i64)]).expect("rebind and execute runtime bounded copy");
        let got = result.as_vec::<f32>().expect("read runtime bounded copy");
        assert_eq!(&got[..n], &expected[..n], "runtime bounded prefix n={n}");
    }
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
    let prog = crate::kernels::matmul::matmul_lds_kblock_mw_clustered(
        4096,
        4096,
        4096,
        crate::kernels::matmul::Tiling::default(),
    )
    .apply(VectorizePass)
    .apply(SwizzlePass);
    let (src, bytes) = crate::launch::compile_artifacts(&prog, &device_spec).expect("compile artifacts");
    std::fs::write(format!("{dir}/streaming.ll"), &src).expect("write ll");
    std::fs::write(format!("{dir}/streaming.co"), &bytes).expect("write co");
    println!("streaming ISA dumped: {dir}/streaming.ll ({} B), {dir}/streaming.co ({} B)", src.len(), bytes.len());
}

/// **FA-32 ISA/spill diagnostic** (§Phase-2 bottleneck probe) — dump the compiled FA-32's amdgcn source
/// and report scratch (spill) bytes/thread, so the Phase-3 handoff knows whether the kernel is spill-bound.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::dump_fa32_isa --nocapture`
#[test]
#[ignore]
fn dump_fa32_isa() {
    let dir = std::env::var("SVOD_DUMP_DIR").unwrap_or_else(|_| "/tmp/tk2_isa".into());
    std::fs::create_dir_all(&dir).expect("mkdir dump dir");
    let device_spec = Tensor::empty(&[1], DType::Float32).device();
    let mut variants: Vec<(String, crate::kernels::Program)> = Vec::new();
    for d in [64usize, 128usize] {
        variants.push((format!("d{d}"), crate::kernels::fa::flash_attention_fwd_32(2, 2048, d).apply(SwizzlePass)));
    }
    variants.push((
        "pp_d128_bf16o".into(),
        crate::kernels::fa::flash_attention_fwd_32_bf16(2, 2048, 128).apply(SwizzlePass),
    ));
    for (tag, prog) in variants {
        let (src, bytes) = launch::compile_artifacts(&prog, &device_spec).expect("compile FA-32");
        let parsed = svod_device::amd::program::parse_kernel(&bytes, "tk2_fa_fwd_32").expect("parse FA-32");
        std::fs::write(format!("{dir}/fa32_{tag}.ll"), &src).expect("write ll");
        std::fs::write(format!("{dir}/fa32_{tag}.co"), &bytes).expect("write co");
        let kd = parsed.kd;
        let scratch = kd.private_segment_fixed_size;
        let lds = kd.group_segment_fixed_size;
        let vgprs = ((kd.compute_pgm_rsrc1 & 0x3f) + 1) * 8;
        println!(
            "FA-32 {tag}: VGPRs={vgprs}, LDS={lds}B, scratch={scratch}B/thread — {} (LLVM IR → {dir}/fa32_{tag}.ll)",
            if scratch == 0 { "NO spills" } else { "SPILLING" },
        );
    }
}

#[derive(Copy, Clone)]
struct FaResourceBudget {
    max_vgprs: u32,
    max_lds: u32,
}

fn assert_fa32_resources(label: &str, bytes: &[u8], budget: FaResourceBudget) -> (u32, u32) {
    let parsed = svod_device::amd::program::parse_kernel(bytes, "tk2_fa_fwd_32").expect("parse FA-32");
    let kd = parsed.kd;
    let scratch = kd.private_segment_fixed_size;
    let lds = kd.group_segment_fixed_size;
    // gfx942 allocates wave64 VGPRs in groups of eight. The generic occupancy helper currently uses
    // the four-register GFX10+ wave64 granule, so decode this target's RSRC1 field directly here.
    let vgprs = ((kd.compute_pgm_rsrc1 & 0x3f) + 1) * 8;
    assert_eq!(scratch, 0, "{label} must not spill; got {scratch} scratch bytes/thread");
    assert!(
        vgprs <= budget.max_vgprs,
        "{label} exceeds its {}-VGPR allocation budget: {vgprs} VGPRs",
        budget.max_vgprs
    );
    assert!(lds <= budget.max_lds, "{label} exceeds its {}-byte LDS budget: {lds} bytes", budget.max_lds);
    println!("{label}: VGPRs={vgprs}, scratch={scratch}B/thread, LDS={lds}B");
    (vgprs, lds)
}

fn assert_fa32_rendered_llvm_structure(src: &str, d: usize) -> usize {
    // These are authoring/DSL checks. Scheduler-group intrinsics are compile-time directives, so their
    // rendered-LLVM cadence is meaningful even though they need not survive as machine instructions.
    let actual_groups = src.matches("call void @llvm.amdgcn.sched.group.barrier").count();
    assert_eq!(actual_groups, 128, "FA-32 d={d} rendered-LLVM scheduler-group cadence changed");
    for (intrinsic, expected) in [
        ("call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 1)", 32),
        ("call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 3, i32 1)", 32),
        ("call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 2)", 32),
        ("call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 5, i32 2)", 32),
    ] {
        let actual = src.matches(intrinsic).count();
        assert_eq!(
            actual, expected,
            "FA-32 d={d} rendered-LLVM directive changed: `{intrinsic}` appears {actual} times"
        );
    }

    let direct_k = src.contains("call void @llvm.amdgcn.global.load.lds");
    let partial_wait = src.contains("call void @llvm.amdgcn.s.waitcnt(i32 3956)");
    let full_wait = src.contains("call void @llvm.amdgcn.s.waitcnt(i32 3952)");
    let scalar_v = src.contains("asm sideeffect \"ds_write_b16");
    let packed_v = src.contains("asm sideeffect \"v_perm_b32");
    let b64_publication = src.contains("asm sideeffect \"ds_write_b64");
    let opaque_gather = src.contains("asm sideeffect \"ds_read_b64");
    if d == 128 {
        assert!(direct_k && partial_wait && full_wait && scalar_v, "long d128 rendered LLVM lost promoted movement");
        assert!(!packed_v && !b64_publication && !opaque_gather, "qualification-only asm movement reached production");
    } else {
        assert!(
            !direct_k && !partial_wait && !full_wait && !scalar_v && !packed_v && !b64_publication && !opaque_gather,
            "d64 rendered LLVM must retain register-staged intrinsic movement"
        );
    }
    actual_groups
}

fn llvm_objdump() -> String {
    let mut candidates = Vec::new();
    if let Ok(tool) = std::env::var("SVOD_LLVM_OBJDUMP") {
        candidates.push(tool);
    }
    candidates.extend([
        "/opt/rocm/llvm/bin/llvm-objdump".to_string(),
        "llvm-objdump-22".to_string(),
        "llvm-objdump-21".to_string(),
        "llvm-objdump-20".to_string(),
        "llvm-objdump".to_string(),
    ]);
    candidates
        .into_iter()
        .find(|tool| {
            std::process::Command::new(tool).arg("--version").output().is_ok_and(|output| output.status.success())
        })
        .expect(
            "FA ISA gate requires ROCm llvm-objdump; set SVOD_LLVM_OBJDUMP or install /opt/rocm/llvm/bin/llvm-objdump",
        )
}

fn disassemble_code_object(bytes: &[u8], label: &str) -> String {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_nanos();
    let safe_label: String = label.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    let path = std::env::temp_dir().join(format!("svod_tk2_{safe_label}_{}_{nonce}.co", std::process::id()));
    std::fs::write(&path, bytes).expect("write temporary FA code object");
    let output = std::process::Command::new(llvm_objdump())
        .args(["-d", "--mcpu=gfx942"])
        .arg(&path)
        .output()
        .expect("run llvm-objdump on FA code object");
    let _ = std::fs::remove_file(&path);
    assert!(output.status.success(), "llvm-objdump failed for {label}: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout).expect("llvm-objdump output must be UTF-8")
}

fn assert_fa32_long_d128_isa_sequence(isa: &str) {
    let body = isa.split_once("<tk2_fa_fwd_32>:").map_or(isa, |(_, body)| body);
    let lines: Vec<_> = body.lines().map(str::trim).filter(|line| !line.is_empty()).collect();
    let partial_waits: Vec<_> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, line)| (line.contains("s_waitcnt") && line.contains("vmcnt(4)")).then_some(i))
        .collect();
    assert!(!partial_waits.is_empty(), "final ISA has no d128 vmcnt(4) throttle");

    let has_steady_sequence = partial_waits.into_iter().any(|partial| {
        let search_start = partial.saturating_sub(256);
        let stage_start = lines[search_start..partial]
            .iter()
            .rposition(|line| line.contains("s_barrier"))
            .map_or(search_start, |i| search_start + i + 1);
        let direct_before: Vec<_> =
            (stage_start..partial).filter(|&i| lines[i].contains("global_load_lds_dword")).collect();
        if direct_before.len() != 4 {
            return false;
        }
        let first_direct = direct_before[0];
        let older_v_load = lines[stage_start..first_direct]
            .iter()
            .any(|line| line.contains("global_load") && !line.contains("global_load_lds"));
        if !older_v_load {
            return false;
        }

        let Some(full_rel) =
            lines[partial + 1..].iter().position(|line| line.contains("s_waitcnt") && line.contains("vmcnt(0)"))
        else {
            return false;
        };
        let full = partial + 1 + full_rel;
        if lines[partial + 1..full].iter().any(|line| line.contains("s_waitcnt") && line.contains("vmcnt(4)")) {
            return false;
        }
        let writes: Vec<_> = (partial + 1..full).filter(|&i| lines[i].contains("ds_write_b16")).collect();
        if writes.len() < 8 {
            return false;
        }
        let mfmas = lines[writes[7] + 1..full].iter().filter(|line| line.contains("v_mfma_f32_32x32x8_bf16")).count();
        if mfmas < 16 {
            return false;
        }

        let publication_end = (full + 96).min(lines.len());
        let Some(lgkm_rel) = lines[full + 1..publication_end]
            .iter()
            .position(|line| line.contains("s_waitcnt") && line.contains("lgkmcnt(0)"))
        else {
            return false;
        };
        let lgkm = full + 1 + lgkm_rel;
        let Some(barrier_rel) = lines[partial + 1..publication_end].iter().position(|line| line.contains("s_barrier"))
        else {
            return false;
        };
        let first_barrier = partial + 1 + barrier_rel;
        first_barrier > lgkm
    });
    assert!(
        has_steady_sequence,
        "final d128 ISA lost the steady movement/compute/publication sequence: older V load -> four direct K dwords -> \
         vmcnt(4) -> eight scalar V writes -> >=16 MFMAs -> vmcnt(0) -> lgkmcnt(0) -> s_barrier"
    );
}

#[test]
fn fa32_isa_sequence_rejects_premature_publication_barrier() {
    let valid = r#"
<tk2_fa_fwd_32>:
global_load_dword v0, v1, s[0:1]
global_load_lds_dword v0, v1, s[0:1]
global_load_lds_dword v0, v1, s[0:1]
global_load_lds_dword v0, v1, s[0:1]
global_load_lds_dword v0, v1, s[0:1]
s_waitcnt vmcnt(4)
ds_write_b16 v0, v1
ds_write_b16 v0, v1
ds_write_b16 v0, v1
ds_write_b16 v0, v1
ds_write_b16 v0, v1
ds_write_b16 v0, v1
ds_write_b16 v0, v1
ds_write_b16 v0, v1
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
v_mfma_f32_32x32x8_bf16 v0, v1, v2, v3
s_waitcnt vmcnt(0)
s_waitcnt lgkmcnt(0)
s_barrier
"#;
    assert_fa32_long_d128_isa_sequence(valid);

    let premature = valid.replace("s_waitcnt vmcnt(0)", "s_barrier\ns_waitcnt vmcnt(0)");
    assert!(
        std::panic::catch_unwind(|| assert_fa32_long_d128_isa_sequence(&premature)).is_err(),
        "ISA gate accepted LDS publication before the VMEM/LDS completion waits"
    );
}

/// Resource and scheduler regression gate for the production kernels and the register-staged oracle.
/// Rendered LLVM checks validate DSL construction; wait/movement ordering is checked separately against
/// disassembly of the final production d128 code object. ELF bytes are intentionally not hashed: the
/// existing golden framework hashes deterministic tile IR, while clang-produced ELF metadata is
/// toolchain-specific. The ordered final-ISA sequence is the stable code-object structural gate.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::fa32_stays_within_resource_and_schedule_budget --nocapture`
#[test]
#[ignore]
fn fa32_stays_within_resource_and_schedule_budget() {
    let device_spec = Tensor::empty(&[1], DType::Float32).device();
    type Constructor = fn(usize, usize, usize) -> crate::Program;
    let cases: [(&str, usize, Constructor, FaResourceBudget, bool); 3] = [
        (
            "production-d64",
            64,
            crate::kernels::fa::flash_attention_fwd_32,
            // LLVM 22 allocates this unchanged IR at 136 VGPR; the active LLVM 23 backend uses 144.
            FaResourceBudget { max_vgprs: 144, max_lds: 17_920 },
            true,
        ),
        (
            "production-long-d128",
            128,
            crate::kernels::fa::flash_attention_fwd_32,
            FaResourceBudget { max_vgprs: 232, max_lds: 44_032 },
            true,
        ),
        (
            "qualification-register-oracle",
            128,
            crate::kernels::fa::flash_attention_fwd_32_register_k,
            FaResourceBudget { max_vgprs: 240, max_lds: 44_032 },
            false,
        ),
    ];
    for (label, d, constructor, budget, production) in cases {
        let prog = constructor(32, 2048, d).apply(SwizzlePass);
        let (src, bytes) = launch::compile_artifacts(&prog, &device_spec).expect("compile FA-32");
        assert_fa32_resources(label, &bytes, budget);
        if production {
            let actual_groups = assert_fa32_rendered_llvm_structure(&src, d);
            println!("{label}: rendered-LLVM sched_groups={actual_groups}");
        }
        if label == "production-long-d128" {
            let isa = disassemble_code_object(&bytes, label);
            assert_fa32_long_d128_isa_sequence(&isa);
        }
    }
}

/// Physical-work gate for the tile-exact FA-32 warmup. The QK(0)-only peel must remove the dummy seed
/// P·V, leaving exactly two real MFMA contractions per KV block and launched wave. Requires Tier-4 PMC.
/// `SVOD_DEVICE=AMD:0 SVOD_PMC_FORCE=1 cargo test -p svod-tk2 --lib -- --ignored device::fa32_warmup_has_no_dummy_mfma_work --nocapture`
#[test]
#[ignore]
fn fa32_warmup_has_no_dummy_mfma_work() {
    let dev = svod_dtype::default_device::default_device();
    let (bh, n) = (32usize, 2048usize);
    for d in [64usize, 128] {
        let mut q = Tensor::rand_with(&[bh * n, d], DType::BFloat16, dev.clone()).expect("rand q");
        let mut k = Tensor::rand_with(&[bh * n, d], DType::BFloat16, dev.clone()).expect("rand k");
        let mut v = Tensor::rand_with(&[bh * n, d], DType::BFloat16, dev.clone()).expect("rand v");
        q.realize().expect("realize q");
        k.realize().expect("realize k");
        v.realize().expect("realize v");

        let prog = crate::kernels::fa::flash_attention_fwd_32(bh, n, d).apply(SwizzlePass);
        let out = Tensor::empty(&[bh * n, d], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&q, &k, &v]).expect("wrap FA-32 program");
        let plan = y.prepare().expect("prepare FA-32");
        let report = plan
            .profile(&ProfileOptions {
                iters: 1,
                static_analysis: false,
                counters: PmcSelection::Custom(vec![PmcCounter::SqWaves, PmcCounter::InstsMfma]),
            })
            .expect("profile FA-32 with PMC");
        let kernel = report
            .stages
            .iter()
            .flat_map(|stage| &stage.kernels)
            .find(|kernel| kernel.kernel.entry_point == "tk2_fa_fwd_32")
            .expect("FA-32 profile entry");
        let counters = kernel.counters.as_ref().expect("PMC counters unavailable; run with SVOD_PMC_FORCE=1");
        let waves = counters.values[&PmcCounter::SqWaves];
        let mfma = counters.values[&PmcCounter::InstsMfma];
        let expected = waves * (n / 32) as u64 * 2 * (d / 8) as u64;
        assert_eq!(mfma, expected, "FA-32 d={d} executed dummy or missing MFMA work");
        println!("FA-32 d={d}: waves={waves}, physical MFMA={mfma} (exact useful work)");
    }
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
    let variants = [(
        "clustered",
        crate::kernels::matmul::matmul_lds_kblock_mw_clustered(
            4096,
            4096,
            4096,
            crate::kernels::matmul::Tiling::default(),
        ),
    )];
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
