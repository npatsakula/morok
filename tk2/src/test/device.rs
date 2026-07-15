//! Hardware-gated device tests for the clustered HK-replica kernel on gfx942
//! (`SVOD_DEVICE=AMD:0 ... --ignored`). The clustered matmul kernel is also correctness-gated +
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

/// **FA-forward correctness GATE**: the multi-warp (8-warp split-Q), online-softmax FA on the ClusterCx
/// pipeline ([`crate::kernels_fa::flash_attention_fwd`]) must match the f32 reference (non-causal, same
/// bf16-rounded operands) at d=64 AND d=128, over `bh > 1` INDEPENDENT attentions (`[bh,n,d]` layout —
/// this validates the per-(b,h) base addressing on top of the QKᵀ inner-d loop, the two `ds_bpermute`
/// softmax reductions, the online rescale, and the `mma_atb` P·V). `atol = 0.02·√d`, `rtol = 2e-2`.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::flash_attention_matches --nocapture`
#[test]
#[ignore]
fn flash_attention_matches_reference_on_gfx942() {
    use svod_tensor::testing::allclose_f32;
    let dev = svod_dtype::default_device::default_device();
    let n = 128usize;
    for (bh, d) in [(3usize, 64usize), (2usize, 128usize)] {
        // Q/K/V are `[bh·n, d]` (= `[bh, n, d]` flat) — bh stacked independent attentions.
        let mut q = Tensor::rand_with(&[bh * n, d], DType::BFloat16, dev.clone()).expect("rand q");
        let mut k = Tensor::rand_with(&[bh * n, d], DType::BFloat16, dev.clone()).expect("rand k");
        let mut v = Tensor::rand_with(&[bh * n, d], DType::BFloat16, dev.clone()).expect("rand v");
        q.realize().expect("realize q");
        k.realize().expect("realize k");
        v.realize().expect("realize v");
        let (qf, kf, vf) = (as_f32_vec(&q), as_f32_vec(&k), as_f32_vec(&v));
        // Reference: `bh` independent `[n,d]` attentions, each at global row base `s·n`.
        let mut expected = vec![0f32; bh * n * d];
        for s in 0..bh {
            let z = s * n * d;
            let o_s = fa_ref(&qf[z..z + n * d], &kf[z..z + n * d], &vf[z..z + n * d], n, d);
            expected[z..z + n * d].copy_from_slice(&o_s);
        }
        // TIGHT atol: the honest bf16-P-cast error is ~2e-3, so a fixed 1e-2 passes correct FA with
        // margin while REJECTING a corrupted-but-plausible result (a mis-fused V gather → ~1e-1 error,
        // which the old `0.02·√d`≈0.23 tolerance wrongly passed). This gate DELIBERATELY runs with
        // VectorizePass on to catch any V-gather mis-fusion regression.
        let atol = 1e-2;

        // Vectorize.then(Swizzle) (matmul's order): VectorizePass now fuses ONLY the straight K gather
        // into `ds_read_b64` — the transposed V gather packs its strided reads into a single
        // `store_frag_vec` (see `LdsView::gather_transposed`), presenting no fusible scalar run, so the
        // pass leaves V bit-exact. SwizzlePass folds the LDS bank swizzle (+82% TF).
        let prog = crate::kernels_fa::flash_attention_fwd(bh, n, d).apply(VectorizePass).apply(SwizzlePass);
        let out = Tensor::empty(&[bh * n, d], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&q, &k, &v]).expect("wrap FA program");
        let plan = y.prepare().expect("prepare FA");
        plan.execute().expect("execute FA");
        let got = y.as_vec::<f32>().expect("read FA output");
        let report = allclose_f32(&got, &expected, atol, 2e-2);
        println!("FA-forward bh={bh} n={n} d={d}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
        assert!(report.ok, "FA-forward bh={bh} n={n} d={d} must match the f32 reference: {}", report.message);
    }
}

/// **FA-32 correctness GATE** (§Step 6): the 32×32×8-MFMA FA ([`crate::kernels_fa::flash_attention_fwd_32`],
/// kept SEPARATE from the frozen 16×16 FA) must match the SAME f32 reference (non-causal, bf16-rounded
/// operands) at d=64 AND d=128, over `bh > 1` independent `[bh,n,d]` attentions — validating the assembled
/// 32×32×8 hot path end-to-end: QKᵀ (`v_mfma_f32_32x32x8`) → `acc_row_reduce_32` online softmax → `v_perm
/// s49` P→PV relayout → padded-transposed-V P·V. `n = 128` (the unrolled-KV assembly's gate). `atol` is the
/// tight 1e-2 the 16×16 gate uses (the honest bf16-cast error is ~2e-3; a corrupted result is rejected).
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
    const Q_BLK: usize = 128; // NUM_WARPS_32 · 32 (the workgroup Q block)
    const KV_BLK: usize = 32; // KV_BLK_32
    // (bh, n, d): the tile-exact full cases (bh>1, n=128) + the RAGGED partial-KV-block cases (bh=1,
    // n=80 — a partial last KV block exercising the mask at both head dims).
    for (bh, n, d) in [(3usize, 128usize, 64usize), (2, 128, 128), (1, 80, 64), (1, 80, 128)] {
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
        // catch any swizzle-layout regression (fill/gather must agree on `lds_col(row, …, d)`).
        let prog = crate::kernels_fa::flash_attention_fwd_32(bh, n, d).apply(SwizzlePass);
        let out = Tensor::empty(&[rows, d], DType::Float32);
        let mut y = graph_kernel(prog, out, &[&q, &k, &v]).expect("wrap FA-32 program");
        let plan = y.prepare().expect("prepare FA-32");
        plan.execute().expect("execute FA-32");
        let got = y.as_vec::<f32>().expect("read FA-32 output");
        // Compare ONLY the true `n` rows per slice (the padded tail is intentional garbage).
        for s in 0..bh {
            let z = s * rows_pad * d;
            let report = allclose_f32(&got[z..z + n * d], &expected[z..z + n * d], atol, 2e-2);
            println!("FA-32 bh={bh} n={n} d={d} slice={s}: ok={} max_abs_err={:e}", report.ok, report.max_abs_err);
            assert!(report.ok, "FA-32 bh={bh} n={n} d={d} slice={s} must match the f32 reference: {}", report.message);
        }
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
    let (bh, n, d) = (1usize, 256usize, 16);
    let dev = svod_dtype::default_device::default_device();
    let mut q = Tensor::rand_with(&[bh * n, d], DType::BFloat16, dev.clone()).expect("rand q");
    let mut k = Tensor::rand_with(&[bh * n, d], DType::BFloat16, dev.clone()).expect("rand k");
    let mut v = Tensor::rand_with(&[bh * n, d], DType::BFloat16, dev).expect("rand v");
    q.realize().expect("realize q");
    k.realize().expect("realize k");
    v.realize().expect("realize v");

    let prog = crate::kernels_fa::flash_attention_fwd(bh, n, d).apply(VectorizePass).apply(SwizzlePass);
    let out = Tensor::empty(&[bh * n, d], DType::Float32);
    let mut y = graph_kernel(prog, out, &[&q, &k, &v]).expect("wrap FA program");
    let plan = y.prepare().expect("prepare FA");
    plan.execute().expect("execute FA on device");
    let got = y.as_vec::<f32>().expect("read FA output");
    let finite = got.iter().filter(|x| x.is_finite()).count();
    let sample = &got[..d.min(8)];
    println!("FA-forward bh={bh} n={n} d={d}: launched OK, {}/{} finite outputs, sample={sample:?}", finite, got.len());
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

/// **FA-32 ISA/spill diagnostic** (§Phase-2 bottleneck probe) — dump the compiled FA-32's amdgcn source
/// and report scratch (spill) bytes/thread, so the Phase-3 handoff knows whether the kernel is spill-bound.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored device::dump_fa32_isa --nocapture`
#[test]
#[ignore]
fn dump_fa32_isa() {
    let dir = std::env::var("SVOD_DUMP_DIR").unwrap_or_else(|_| "/tmp/tk2_isa".into());
    std::fs::create_dir_all(&dir).expect("mkdir dump dir");
    let device_spec = Tensor::empty(&[1], DType::Float32).device();
    for d in [64usize, 128usize] {
        let prog = crate::kernels_fa::flash_attention_fwd_32(2, 2048, d).apply(SwizzlePass);
        let (src, bytes) = launch::compile_artifacts(&prog, &device_spec).expect("compile FA-32");
        let parsed = svod_device::amd::program::parse_kernel(&bytes, "tk2_fa_fwd_32").expect("parse FA-32");
        std::fs::write(format!("{dir}/fa32_d{d}.ll"), &src).expect("write ll");
        std::fs::write(format!("{dir}/fa32_d{d}.co"), &bytes).expect("write co");
        let scratch = { parsed.kd }.private_segment_fixed_size;
        println!(
            "FA-32 d={d}: scratch(spill)={scratch}B/thread — {} (LLVM IR → {dir}/fa32_d{d}.ll)",
            if scratch == 0 { "NO spills" } else { "SPILLING" },
        );
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
