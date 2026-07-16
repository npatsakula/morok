//! **Isolation de-risk probes** — standalone single-warp kernels that each prove ONE building block of
//! the 32×32×8 / FA pipeline against a device allclose reference. Relocated out of the production
//! `kernels::matmul`/`kernels::fa` modules (they are built ONLY for the device / lower / byte-identity test
//! gates, never by a kernel or bench). Relocating a probe does NOT change the [`Program`] it builds —
//! every emitted-IR signature (the `atb_probe` byte-identity gate, the device oracles) is unchanged.

use crate::build::{BF16, Buf, Builder, Effect, F32, Idx, Val};
use crate::ir::{FragMap, TileId};
use crate::kernels::{EDGE, Program, offset_by};
use crate::shape::{Mfma16x16x16Bf16, MfmaShape};
use crate::tile::BCol;
use crate::tile_move::{commit, gather, prefetch};

/// One 64-lane warp — the probes are single-warp isolation kernels.
const WARP: usize = 64;
/// gfx942 elements-per-thread for the 16×16 fragment (`= 4`) — the `atb_probe` accumulator width.
const EPT: usize = Mfma16x16x16Bf16::EPT_C;

/// Load ONE MFMA operand fragment (`map.ept` wide) straight from a global `[outer, K]` (row-major, row
/// stride `k_dim`) into a register fragment, addressed by `map`'s `lane_rc`. For a Row (A) map the lane
/// pair is `(M-row, K)`; for a Col (B) map it is `(K-spread, N-flat)` — so `outer` is the M (A) / N (B)
/// index and `kk` the K contribution, and the global offset is `(outer_base + outer)·k_dim + (k_base +
/// kk)`. Used only by the 32×32×8 isolation probe (a self-contained gather, free of the EDGE-coupled
/// movement layer, so it exercises the marker's operand maps directly).
pub(crate) fn load_op_frag(
    b: &mut Builder,
    src: Buf<BF16>,
    map: FragMap,
    outer_base: usize,
    k_base: usize,
    k_dim: usize,
    lane: Idx,
) -> Val<BF16> {
    let frag = b.define_frag::<BF16>(map);
    let k_c = b.idx_const(k_dim as i64);
    let stores: Vec<TileId> = (0..map.ept)
        .map(|e| {
            let e_idx = b.idx_const(e as i64);
            let (r, cc) = b.lane_rc(map, lane, e_idx);
            let (outer, kk) = if map.transpose { (cc, r) } else { (r, cc) };
            let row = offset_by(b, outer, outer_base);
            let col = offset_by(b, kk, k_base);
            let off = b.idx_mul(row, k_c);
            let off = b.idx_add(off, col);
            let v = b.load(src, off);
            b.store_frag_elem(frag, e_idx, v).dep()
        })
        .collect();
    b.load_frag_vec_after(frag, &stores)
}

/// **32×32×8 MFMA isolation probe** (§migration Step 3 DE-RISK): a standalone single-warp kernel that
/// computes `C = A·Bᵀ` (`A[m,k]`, `B[n,k]`, both K-contiguous) tiled as `(m/32)×(n/32)` output blocks,
/// each ONE `v_mfma_f32_32x32x8_bf16` per K-slice accumulated over the `k/8` slices, then scatters the
/// 16-VGPR accumulator via [`Builder::acc_rc`]. It PROVES the 32×32×8 operand layout + accumulator
/// distribution IN ISOLATION (device-gated allclose vs an f32 reference) before any FA work — the
/// wide-core analog of `atb_probe`. `m`/`n` multiples of 32, `k` a multiple of 8. Emitted with the
/// intrinsic MFMA ([`Builder::mma_of`], the production fast path).
#[allow(clippy::needless_range_loop)]
pub(crate) fn mfma_32x32x8_probe(m: usize, n: usize, k: usize) -> Program {
    use crate::shape::Mfma32x32x8Bf16 as S;
    assert!(
        m.is_multiple_of(S::M) && n.is_multiple_of(S::N) && k.is_multiple_of(S::K),
        "probe dims must tile by 32×32×8"
    );
    let mut b = Builder::new("tk2_mfma_32x32x8_probe");
    // ABI: output C[m,n] first, then inputs A[m,k], B[n,k] (B is [N,K] — A·Bᵀ, both K-contiguous).
    let c = b.global::<F32>(m * n);
    let a = b.global::<BF16>(m * k);
    let bmat = b.global::<BF16>(n * k);
    let _wg = b.grid_axis(0, 1);
    let lane = b.block_axis(WARP as i64);

    let a_map = S::a_map();
    let b_map = S::b_map();
    let dist = S::acc_dist();
    let n_c = b.idx_const(n as i64);
    let mut roots = Vec::new();
    for mi in 0..m / S::M {
        for ni in 0..n / S::N {
            // Accumulate `k/8` MFMAs into a 16-wide f32 value (the C accumulator, register-carried).
            let mut acc = {
                let zs: Vec<Val<F32>> = (0..S::EPT_C).map(|_| b.f32(0.0)).collect();
                b.vec_build(&zs)
            };
            for ki in 0..k / S::K {
                let af = load_op_frag(&mut b, a, a_map, mi * S::M, ki * S::K, k, lane);
                let bf = load_op_frag(&mut b, bmat, b_map, ni * S::N, ki * S::K, k, lane);
                acc = b.mma_of::<S>(af, bf, acc);
            }
            // Scatter the 16-VGPR accumulator: element `i` → C[mi·32 + row, ni·32 + col] via acc_rc.
            for i in 0..S::EPT_C {
                let (row, col) = b.acc_rc(dist, lane, i);
                let m_idx = offset_by(&mut b, row, mi * S::M);
                let n_idx = offset_by(&mut b, col, ni * S::N);
                let off = b.idx_mul(m_idx, n_c);
                let off = b.idx_add(off, n_idx);
                let v = b.vec_extract(acc, i);
                roots.push(b.store(c, off, v));
            }
        }
    }
    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_mfma_32x32x8_probe".into() }
}

/// **Step-4 P→PV relayout isolation probe** (32×32×8 DE-RISK): proves the `v_perm_b32` primitive AND the
/// [`Builder::pv_relayout_s49`] pack are correct, IN ISOLATION, before FA trusts them. It computes
/// `O[d,q] = Σ_kv V[d,kv]·P[kv,q]` where `P` is loaded straight into the 32×32×8 **C-accumulator** layout
/// (`acc_rc`, the exact distribution a QKᵀ MFMA would leave it in), then relayout'd via `v_perm s49` into
/// the PV **B-operands**, while `V` is the A-operand — one 32×32×8 MFMA per `K=8` slice accumulated over
/// the four slices of the 32-row KV block. A device allclose vs the f32 reference proves the selector +
/// the accumulator↔B-operand element correspondence are right (a wrong selector → high error on a tiny
/// probe, not silent FA corruption). `P` holds exact bf16 values (f32 buffer) so the pack's truncation is
/// a no-op vs the reference. `d`/`q` multiples of 32; `kv = 32` (one KV block, four K-slices).
#[allow(clippy::needless_range_loop)]
pub(crate) fn pv_relayout_probe(d: usize, q: usize) -> Program {
    use crate::shape::Mfma32x32x8Bf16 as S;
    const KV: usize = 32; // one 32-row KV block = four hardware K=8 slices
    assert!(d.is_multiple_of(S::M) && q.is_multiple_of(S::N), "pv_relayout_probe: d, q multiples of 32");
    let mut b = Builder::new("tk2_pv_relayout_probe");
    // ABI: output O[d,q], then P[kv,q] (f32, exact-bf16 values) and V[d,kv] (bf16, the A operand).
    let o = b.global::<F32>(d * q);
    let p = b.global::<F32>(KV * q);
    let v = b.global::<BF16>(d * KV);
    let _wg = b.grid_axis(0, 1);
    let lane = b.block_axis(WARP as i64);

    let a_map = S::a_map();
    let dist = S::acc_dist();
    let q_c = b.idx_const(q as i64);
    let mut roots = Vec::new();
    for qt in 0..q / S::N {
        // ── load P's [kv=32, q-tile] into the 32×32×8 C-accumulator layout (f32), then relayout via
        //    v_perm s49 into the four PV B-operands (one per K=8 slice). ──
        let p_acc = {
            let els: Vec<Val<F32>> = (0..S::EPT_C)
                .map(|i| {
                    let (row, col) = b.acc_rc(dist, lane, i); // row = kv, col = q-in-tile
                    let q_idx = offset_by(&mut b, col, qt * S::N);
                    let off = b.idx_mul(row, q_c);
                    let off = b.idx_add(off, q_idx);
                    b.load(p, off)
                })
                .collect();
            b.vec_build(&els)
        };
        let b_ops = b.pv_relayout_s49(p_acc); // 4 × <4×bf16>, one per K-slice

        for dt in 0..d / S::M {
            let mut acc = {
                let zs: Vec<Val<F32>> = (0..S::EPT_C).map(|_| b.f32(0.0)).collect();
                b.vec_build(&zs)
            };
            for s in 0..KV / S::K {
                let a_s = load_op_frag(&mut b, v, a_map, dt * S::M, s * S::K, KV, lane);
                acc = b.mma_of::<S>(a_s, b_ops[s], acc);
            }
            // scatter O[d,q]: acc element i → O[dt·32 + row(=d), qt·32 + col(=q)] via acc_rc.
            for i in 0..S::EPT_C {
                let (row, col) = b.acc_rc(dist, lane, i);
                let d_idx = offset_by(&mut b, row, dt * S::M);
                let q_idx = offset_by(&mut b, col, qt * S::N);
                let off = b.idx_mul(d_idx, q_c);
                let off = b.idx_add(off, q_idx);
                let val = b.vec_extract(acc, i);
                roots.push(b.store(o, off, val));
            }
        }
    }
    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_pv_relayout_probe".into() }
}

/// **`mma_atb` isolation probe** (DE-RISK, per the Phase-A brief): a standalone single-warp kernel that
/// computes `O[d,q] = Σ_kv V[kv,d]·P[kv,q]` — EXACTLY the FA `P·V` contraction (over the shared `kv`
/// row) — via the transposed operand gather ([`LdsView::gather_transposed`]) + the SAME
/// `v_mfma_f32_16x16x16` [`Builder::mma`]. It proves, against an f32 host reference, that landing the
/// contraction axis on the MFMA spread lane-axis (the "register transpose" done as a transposed LDS
/// read) computes `Aᵀ·B` correctly, BEFORE the layout is trusted inside FA. `kv = 16` (one contraction
/// fragment); `d`/`q` are the output extents (multiples of 16), tiled as `d/16 × q/16` output fragments.
#[allow(clippy::needless_range_loop)] // the fragment index also drives the d·16 / q·16 tile base
pub(crate) fn atb_probe(kv: usize, d: usize, q: usize) -> Program {
    assert!(kv == EDGE, "atb_probe: single kv contraction fragment (kv = 16)");
    assert!(d.is_multiple_of(EDGE) && q.is_multiple_of(EDGE), "d, q must be multiples of 16");
    let mut b = Builder::new("tk2_atb_probe");
    // ABI: output O[d,q] first, then inputs V[kv,d], P[kv,q].
    let o = b.global::<F32>(d * q);
    let v = b.global::<BF16>(kv * d);
    let p = b.global::<BF16>(kv * q);
    let _wg = b.grid_axis(0, 1);
    let lane = b.block_axis(WARP as i64);
    let zero = b.idx_const(0);

    let col = Mfma16x16x16Bf16::c_map(); // the Col map, still driving the epilogue scatter below
    let v_smem = b.define_local::<BF16>(kv * d);
    let p_smem = b.define_local::<BF16>(kv * q);

    // ── fill V[kv,d] and P[kv,q] into LDS via the register-staged movement ops (prefetch → commit).
    //    (Reg←Global) prefetch stages the whole tile in VGPRs; (Lds←Reg) commit writes it to LDS. ──
    let vl = prefetch(&mut b, v_smem, d, v, d as i64, kv * d / WARP, lane, zero, zero, &[]);
    let pl = prefetch(&mut b, p_smem, q, p, q as i64, kv * q / WARP, lane, zero, zero, &[]);
    let vf = commit(&mut b, v_smem, d, v, d as i64, kv * d / WARP, lane, zero, &vl, &[]);
    let pf = commit(&mut b, p_smem, q, p, q as i64, kv * q / WARP, lane, zero, &pl, &[]);
    let fill: Vec<TileId> = vf.iter().chain(pf.iter()).map(|e| e.dep()).collect();
    let bar = b.barrier(Effect(fill[0]), &fill[1..]);

    // ── transposed gather via the (Reg←Lds) op: the BCol operand role derives the transposed read,
    //    landing kv (contraction) on the spread lane-axis and d/q (output) on flat, stacked as frags. ──
    let (v_frags, _) =
        gather::<BF16, BCol, Mfma16x16x16Bf16>(&mut b, v_smem, d, EDGE, d, None, lane, &[bar.dep()], 0, false);
    let (p_frags, _) =
        gather::<BF16, BCol, Mfma16x16x16Bf16>(&mut b, p_smem, q, EDGE, q, None, lane, &[bar.dep()], 0, false);

    // ── O[d,q] = Σ_kv V·P: one MFMA per (d-frag, q-frag) output tile (single kv contraction). ──
    let zero_c = {
        let zs: Vec<Val<F32>> = (0..EPT).map(|_| b.f32(0.0)).collect();
        b.vec_build(&zs)
    };
    let q_c = b.idx_const(q as i64);
    let mut roots = Vec::new();
    for df in 0..d / EDGE {
        for qf in 0..q / EDGE {
            let res = b.mma(v_frags[df], p_frags[qf], zero_c, EPT); // Col-out [d=spread, q=flat]
            let d_base = b.idx_const((df * EDGE) as i64);
            let q_base = b.idx_const((qf * EDGE) as i64);
            for inner in 0..EPT {
                let inner_c = b.idx_const(inner as i64);
                let (row, ccol) = b.lane_rc(col, lane, inner_c); // (d in-frag = spread, q in-frag = flat)
                let d_idx = b.idx_add(d_base, row);
                let q_idx = b.idx_add(q_base, ccol);
                let off = b.idx_mul(d_idx, q_c);
                let off = b.idx_add(off, q_idx);
                let val = b.vec_extract(res, inner);
                roots.push(b.store(o, off, val));
            }
        }
    }
    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_atb_probe".into() }
}

/// **Step-6 softmax-reduction isolation probe** (32×32×8 DE-RISK): proves the [`Builder::acc_row_reduce_32`]
/// online-softmax reduce over the `EPT_C = 16` accumulator geometry (the FA-32 row-reduce over kv), IN
/// ISOLATION, before FA. It loads `S[kv, q]` into the 32×32×8 C-accumulator layout (kv on M), computes the
/// full softmax over kv per q — `P[kv,q] = exp2(S[kv,q] − max_kv) / Σ_kv exp2(…)` — via the 16-in-register +
/// `L↔L+32` cross-lane reduce, and scatters `P`. A device allclose vs the host softmax proves the AccDist
/// reduction geometry + broadcast are correct (a missing cross-lane term → the norm sums 16 of 32 kv →
/// ~2× error). `kv = q = 32` (one 32×32 tile). This is the last un-proven FA-32 building block.
#[allow(clippy::needless_range_loop)]
pub(crate) fn softmax32_probe() -> Program {
    use crate::shape::Mfma32x32x8Bf16 as S;
    const KV: usize = 32;
    const Q: usize = 32;
    let mut b = Builder::new("tk2_softmax32_probe");
    // ABI: output P[kv,q], then input S[kv,q] (f32 scores in the accumulator layout).
    let out = b.global::<F32>(KV * Q);
    let s = b.global::<F32>(KV * Q);
    let _wg = b.grid_axis(0, 1);
    let lane = b.block_axis(WARP as i64);

    let dist = S::acc_dist();
    let q_c = b.idx_const(Q as i64);
    // Load S into the 16-wide accumulator (row = kv, col = q).
    let s_acc = {
        let els: Vec<Val<F32>> = (0..S::EPT_C)
            .map(|i| {
                let (row, col) = b.acc_rc(dist, lane, i);
                let off = b.idx_mul(row, q_c);
                let off = b.idx_add(off, col);
                b.load(s, off)
            })
            .collect();
        b.vec_build(&els)
    };
    // Online softmax over kv (per q): max, exp2(S−max), sum, normalize — all over the AccDist geometry.
    let neg_inf = {
        let ni = b.f32(f32::NEG_INFINITY);
        let cs: Vec<Val<F32>> = (0..S::EPT_C).map(|_| ni).collect();
        b.vec_build(&cs)
    };
    let zero = {
        let z = b.f32(0.0);
        let cs: Vec<Val<F32>> = (0..S::EPT_C).map(|_| z).collect();
        b.vec_build(&cs)
    };
    let m = b.acc_row_reduce_32(s_acc, lane, neg_inf, false); // per-q max, broadcast to 16 slots
    let shifted = b.sub(s_acc, m);
    let p = b.exp2(shifted); // exp2(S − max), 16-wide
    let sum = b.acc_row_reduce_32(p, lane, zero, true); // per-q Σ, broadcast
    let recip = b.recip(sum);
    let p_norm = b.mul(p, recip); // P / Σ
    // Scatter P[kv,q].
    let mut roots = Vec::new();
    for i in 0..S::EPT_C {
        let (row, col) = b.acc_rc(dist, lane, i);
        let off = b.idx_mul(row, q_c);
        let off = b.idx_add(off, col);
        let val = b.vec_extract(p_norm, i);
        roots.push(b.store(out, off, val));
    }
    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_softmax32_probe".into() }
}

/// **Step-5 V write-side padded-transpose isolation probe** (32×32×8 DE-RISK): proves that staging V
/// through a **padded transposed LDS** layout produces the correct 32×32×8 PV **A-operand**, IN ISOLATION,
/// before FA trusts it. It computes `O[d,q] = Σ_kv V[kv,d]·P[kv,q]` where `V[kv,d]` (natural layout) is
/// staged into LDS TRANSPOSED to `[d, kv]` with a **padded row pitch** (`kv + PAD`, the conflict-free
/// bank layout — an un-padded `[d,kv]` transpose bank-conflicts and regresses), then read back **straight**
/// as the A-operand via a contiguous `ds_read_b64` (4 kv per lane); `P` is the straight B-operand. A device
/// allclose vs the f32 reference proves the transposed layout + the straight-read addressing yield the
/// right operand (the read-side of aiter's V relayout; the v_perm-deinterleaved *b64* WRITE is a
/// write-bandwidth refinement layered on this correct layout, not a correctness requirement of the probe).
/// `d` a multiple of 32; `q = 32`; `kv = 32` (one KV block, four hardware K=8 slices).
#[allow(clippy::needless_range_loop)]
pub(crate) fn v_transpose_probe(d: usize) -> Program {
    use crate::shape::Mfma32x32x8Bf16 as S;
    const KV: usize = 32;
    const Q: usize = 32;
    const PAD: usize = 4; // pitch padding: kv(32)+4 = 36 → b64 dword-stride 18, gcd(18,32)=2 → 16 distinct banks (conflict-free; matches the kernel's VT_PAD)
    let pitch = KV + PAD; // transposed LDS row pitch (d rows, kv+pad cols)
    assert!(d.is_multiple_of(S::M), "v_transpose_probe: d a multiple of 32");
    let mut b = Builder::new("tk2_v_transpose_probe");
    // ABI: O[d,q], then V[kv,d] (natural, transposed here) and Pt[q,kv] (the straight B-operand [N=q,K=kv]).
    let o = b.global::<F32>(d * Q);
    let v = b.global::<BF16>(KV * d);
    let pt = b.global::<BF16>(Q * KV);
    let _wg = b.grid_axis(0, 1);
    let lane = b.block_axis(WARP as i64);
    let zero = b.idx_const(0);

    let a_map = S::a_map();
    let b_map = S::b_map();
    let dist = S::acc_dist();

    // ── stage V[kv,d] → LDS transposed to [d, kv] with padded pitch. Each lane writes its coalesced share
    //    of V (source index `flat = kv·d + d_idx`) to the transposed slot `LDS_T[d_idx·pitch + kv]`. ──
    let vt = b.define_local::<BF16>(d * pitch);
    let epl = KV * d / WARP; // elements per lane
    let epl_c = b.idx_const(epl as i64);
    let d_c = b.idx_const(d as i64);
    let pitch_c = b.idx_const(pitch as i64);
    let lane_epl = b.idx_mul(lane, epl_c);
    let fills: Vec<TileId> = (0..epl)
        .map(|i| {
            let i_c = b.idx_const(i as i64);
            let flat = b.idx_add(lane_epl, i_c); // source V flat index (kv·d + d_idx)
            let kv = b.idx_div(flat, d_c);
            let d_idx = b.idx_mod(flat, d_c);
            let val = b.load(v, flat);
            let dst = b.idx_mul(d_idx, pitch_c);
            let dst = b.idx_add(dst, kv); // transposed padded slot
            b.store_lds(vt, dst, val).dep()
        })
        .collect();
    let bar = b.barrier(Effect(fills[0]), &fills[1..]);

    // ── P·V: A-operand from the straight (contiguous) transposed read, B-operand P from global. ──
    let q_c = b.idx_const(Q as i64);
    let mut roots = Vec::new();
    for dt in 0..d / S::M {
        let mut acc = {
            let zs: Vec<Val<F32>> = (0..S::EPT_C).map(|_| b.f32(0.0)).collect();
            b.vec_build(&zs)
        };
        for s in 0..KV / S::K {
            // A-operand for K-slice s: LDS_T[d = dt·32 + lane%32][kv = 8s + (lane/32)·4 .. +4] (contiguous).
            let (row, kloc) = b.lane_rc(a_map, lane, zero); // row = lane%32 = d-in-tile, kloc = (lane/32)·4
            let d_row = offset_by(&mut b, row, dt * S::M);
            let base = b.idx_mul(d_row, pitch_c);
            let kv_base = offset_by(&mut b, kloc, s * S::K); // (lane/32)·4 + 8s
            let base = b.idx_add(base, kv_base);
            let a_s = b.load_lds_vec_after(vt, base, S::EPT_A, &[bar.dep()]);
            let b_s = load_op_frag(&mut b, pt, b_map, 0, s * S::K, KV, lane);
            acc = b.mma_of::<S>(a_s, b_s, acc);
        }
        for i in 0..S::EPT_C {
            let (row, col) = b.acc_rc(dist, lane, i);
            let d_idx = offset_by(&mut b, row, dt * S::M);
            let off = b.idx_mul(d_idx, q_c);
            let off = b.idx_add(off, col);
            let val = b.vec_extract(acc, i);
            roots.push(b.store(o, off, val));
        }
    }
    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_v_transpose_probe".into() }
}
