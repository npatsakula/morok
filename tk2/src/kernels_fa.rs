//! **Flash-Attention forward on the ClusterCx declarative pipeline** — a NUMERICALLY CORRECT,
//! MULTI-WARP (8-warp split-Q) FA-forward proving [`crate::pipeline`] (built for the HK GEMM)
//! generalises to a second, differently-shaped kernel. Streams K/V blocks: QKᵀ → online-softmax → P·V
//! accumulate → normalize → write O. Non-causal, `bh = batch·heads` independent attentions (`[bh,n,d]`
//! layout), head dim `d` any multiple of 16.
//!
//! ## The two matmuls (both `mma_atb` — contraction over the shared row)
//! - **QKᵀ** contracts over `d`: `K` (row map) and `Q` (col map) are gathered with `d` on the MFMA
//!   spread (contraction) lane-axis, summed over `d/16` `mma` K-steps → `att[kv, q]` (`q` on flat).
//! - **P·V** contracts over `kv`: the QKᵀ f32 accumulator `P` feeds PV's operand with ONLY a bf16
//!   cast — its lane distribution already equals the required operand layout (the CK "free relayout",
//!   DESIGN §5) — while `V` is gathered TRANSPOSED ([`LdsView::gather_transposed`]) so `kv` lands on
//!   the spread axis. Both use the SAME `v_mfma_f32_16x16x16`; the orientation is purely the operand
//!   lane layout, NOT a new hardware op (proven in isolation by `atb_probe` + its device gate).
//!
//! ## Multi-warp split-Q (§1.2)
//! A workgroup owns a `q_blk = 128`-row Q block over 8 warps (512 threads); warp `w` owns Q rows
//! `[w·16, w·16+16)`. K/V are shared in LDS — loaded ONCE per KV block by a collaborative 512-thread
//! fill (the 8× occupancy + K/V-bandwidth lever). Each warp keeps its OWN o/m/l registers (the warp
//! offset rides the Q-gather + O-scatter addressing, not the slot count) and reduces WITHIN its 64
//! lanes (`frag_col_reduce`/`ds_bpermute` is single-warp — warps run disjoint Q rows, so no cross-warp
//! reduction). `kv_blk` = 16 (`d=128`) or 32 (`d=64`, 2 KV-fragments) so the 512-thread fill is
//! VEC4-aligned. No ping-pong (`warp_row=None`): there is no co-resident load/compute pair to steer.
//!
//! Device-gated by `flash_attention_matches_reference_on_gfx942` (allclose vs an f32 reference at
//! d=64 and d=128). Single-buffer; softmax-under-MFMA interleave + swizzle are later Phase-B items.

use crate::build::{BF16, Buf, Builder, Effect, F32, Frag, Idx, Lds, Val};
use crate::ir::{FragMap, TileId};
use crate::kernels::{EDGE, Program, offset_by};
use crate::movement::{Drain, LdsStage, LdsView, SharedTile};
use crate::pipeline::{AccSlot, CommitDrain, Compute, Hooks, Mem, SlotVal, pipeline};
use crate::shape::{Mfma16x16x16Bf16, MfmaShape};

const WARP: usize = 64;
/// Warps per workgroup for the multi-warp split-Q FA (§1.2): 8 warps × 64 lanes = 512 threads. Each
/// warp owns 16 Q rows; all 8 share ONE K/V LDS tile (loaded once per KV block by the 512-thread
/// collaborative fill — the 8× occupancy + K/V-bandwidth lever). No cross-warp reduction: the softmax
/// `frag_col_reduce` (`ds_bpermute` over 64 lanes) stays within each warp (disjoint Q rows).
const NUM_WARPS: usize = 8;

/// One K-slice's gathered fragments — the [`Hooks::Op`]. Slice 0 = K (QKᵀ operand),
/// slice 1 = V (PV operand). Both are `Vec<Val<BF16>>` (one per outer fragment).
type FaOp = Vec<Val<BF16>>;

/// The register-staged fill carried prefetch→commit: block k+1's K and V chunks in VGPRs.
struct FaFill {
    k: Vec<Val<BF16>>,
    v: Vec<Val<BF16>>,
}

/// FA's [`Hooks`] — the ONLY kernel-specific movement. It rides the SAME [`crate::movement`]
/// handles matmul does (proving the movement layer is not GEMM-bound); the compute math rides the
/// [`Compute`] bodies below, so `Hooks` grows no per-cluster compute method. `PREFETCH_TILES = 2`
/// (K, V) as in matmul (A, B), but the two tiles are the two *operands of two different matmuls*
/// (K feeds QKᵀ, V feeds PV) rather than the A/B of one — the first shape strain (see report).
struct FaHooks {
    k_view: LdsView<BF16>,
    v_view: LdsView<BF16>,
    k_stage: LdsStage<BF16>,
    v_stage: LdsStage<BF16>,
    /// `d / 16` — the head-dim fragment count. QKᵀ contracts over `d` as this many `mma` K-steps
    /// (K row-map slices); PV's `V` is gathered as this many transposed output-`d` fragments.
    dfrags: usize,
    /// `kv_blk / 16` — the KV-block fragment count. K's row-map view stacks `kvf` kv-fragments (its
    /// `n_frags`); V is gathered `kvf × dfrags` (a transposed gather per kv-slice). `kvf = 1` for the
    /// single-fragment KV block (`d = 128`); `2` when `kv_blk = 32` (`d = 64`, so the 512-thread fill
    /// is VEC4-aligned). K is returned indexed `[s·kvf + kf]`, V indexed `[kf·dfrags + df]`.
    kvf: usize,
}

impl Hooks for FaHooks {
    type Op = FaOp;
    type Reg = FaFill;
    const PREFETCH_TILES: usize = 2; // 0 = K, 1 = V

    fn prefetch(
        &mut self,
        b: &mut Builder,
        k_base: Idx,
        tile: usize,
        prev: Option<FaFill>,
        order: &[TileId],
    ) -> (FaFill, Vec<TileId>) {
        let mut reg = prev.unwrap_or(FaFill { k: Vec::new(), v: Vec::new() });
        let loaded = match tile {
            0 => {
                reg.k = self.k_stage.prefetch(b, k_base, order);
                &reg.k
            }
            1 => {
                reg.v = self.v_stage.prefetch(b, k_base, order);
                &reg.v
            }
            _ => panic!("FA prefetch: tile ∈ {{0=K, 1=V}}, got {tile}"),
        };
        let anchors = loaded.iter().map(|v| v.id).collect();
        (reg, anchors)
    }

    fn commit(&mut self, b: &mut Builder, _k_base: Idx, reg: &FaFill, war: &[TileId]) -> Vec<Effect> {
        // Intrinsic commit (an `s_barrier` auto-drains the `ds_write` `lgkmcnt(0)`) — the simplest
        // policy; the asm/deferred-drain machinery is a matmul-perf concern, orthogonal to FA's shape.
        let fk = self.k_stage.commit(b, &reg.k, war);
        let fv = self.v_stage.commit(b, &reg.v, war);
        fk.into_iter().chain(fv).collect()
    }

    fn gather(&mut self, b: &mut Builder, slice: usize, raw: &[TileId]) -> (FaOp, Vec<TileId>, TileId) {
        let mut vecs = Vec::new();
        let mut g = Vec::new();
        match slice {
            // K: QKᵀ's A operand as `dfrags · kvf` fragments (row map, contraction `d` on the spread
            // lane-axis; `slice(s)` selects d-columns `[s·16, s·16+16)`, the view's `n_frags = kvf`
            // stacking the KV-block rows). Indexed `[s·kvf + kf]`; QKᵀ = Σ_s mma(K[s,kf], Q_s) per `kf`.
            0 => {
                for s in 0..self.dfrags {
                    let (v, gg) = self.k_view.slice(s).gather(b, raw); // kvf kv-fragments for d-slice s
                    vecs.extend(v);
                    g.extend(gg);
                }
            }
            // V: PV's A operand as `kvf · dfrags` TRANSPOSED fragments (contraction `kv` on the spread
            // lane-axis — the `mma_atb` orientation). `slice(kf)` sets the kv-row base `kf·16`, then a
            // transposed gather yields the `dfrags` output-d fragments. Indexed `[kf·dfrags + df]`.
            1 => {
                for kf in 0..self.kvf {
                    let (v, gg) = self.v_view.slice(kf).gather_transposed(b, raw);
                    vecs.extend(v);
                    g.extend(gg);
                }
            }
            _ => panic!("FA gather: slice ∈ {{0=K, 1=V}}, got {slice}"),
        }
        let anchor = vecs[0].id;
        (vecs, g, anchor)
    }
}

/// gfx942 elements-per-thread for the 16×16 fragment — DERIVED from the shape marker (§Step 1; `= 4`).
const EPT: usize = Mfma16x16x16Bf16::EPT_C;

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
pub fn pv_relayout_probe(d: usize, q: usize) -> Program {
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
                let a_s = crate::kernels::load_op_frag(&mut b, v, a_map, dt * S::M, s * S::K, KV, lane);
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
pub fn atb_probe(kv: usize, d: usize, q: usize) -> Program {
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

    let col = Mfma16x16x16Bf16::c_map(); // the Col (B/accumulator) map, derived from the marker
    let v_smem = b.define_local::<BF16>(kv * d);
    let p_smem = b.define_local::<BF16>(kv * q);
    let v_tile = SharedTile::new(v_smem, d);
    let p_tile = SharedTile::new(p_smem, q);

    // ── fill V[kv,d] and P[kv,q] into LDS (single warp, whole tile) via the collaborative stage. ──
    let v_stage = v_tile.stage_view(v, kv * d / WARP, lane, zero, d as i64, Drain::Intrinsic);
    let p_stage = p_tile.stage_view(p, kv * q / WARP, lane, zero, q as i64, Drain::Intrinsic);
    let vl = v_stage.prefetch(&mut b, zero, &[]);
    let pl = p_stage.prefetch(&mut b, zero, &[]);
    let vf = v_stage.commit(&mut b, &vl, &[]);
    let pf = p_stage.commit(&mut b, &pl, &[]);
    let fill: Vec<TileId> = vf.iter().chain(pf.iter()).map(|e| e.dep()).collect();
    let bar = b.barrier(Effect(fill[0]), &fill[1..]);

    // ── transposed gather: kv (contraction) → spread; d/q (output) → flat, stacked as fragments. ──
    let v_view = v_tile.gather_view(col, d / EDGE, None, lane, false);
    let p_view = p_tile.gather_view(col, q / EDGE, None, lane, false);
    let (v_frags, _gv) = v_view.gather_transposed(&mut b, &[bar.dep()]);
    let (p_frags, _gp) = p_view.gather_transposed(&mut b, &[bar.dep()]);

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

/// Load this warp's QKᵀ B-operand Q fragment for d-slice `s` straight from GLOBAL into registers
/// (Probe B — no LDS staging). Reproduces exactly the col-map fragment the LDS gather produced, but
/// sourced from global Q at `[q_row_base + flat, s·16 + spread]` (row-major, stride `d`): each warp's
/// 16 Q rows are warp-PRIVATE (disjoint across the 8 warps), so LDS-staging Q bought NO cross-warp
/// sharing — only a 32KB (d=128) LDS tile that capped occupancy at 1 wg/CU. The fragment is
/// loop-invariant (the QKᵀ B operand), so it is hoisted once and lives in VGPRs for the whole KV stream.
fn load_q_frag_global(
    b: &mut Builder,
    q: Buf<BF16>,
    map: FragMap,
    q_row_base: Idx,
    d: usize,
    s: usize,
    lane: Idx,
) -> Val<BF16> {
    let frag = b.define_frag::<BF16>(map);
    let d_c = b.idx_const(d as i64);
    // Col map ⇒ `lane_rc` yields (spread = contraction d-row, flat = Q-row within the 16-row sub-block).
    let stores: Vec<TileId> = (0..map.ept)
        .map(|e| {
            let e_idx = b.idx_const(e as i64);
            let (spread, flat) = b.lane_rc(map, lane, e_idx);
            let row = b.idx_add(q_row_base, flat); // global Q row = this warp's origin + flat
            let col = offset_by(b, spread, s * EDGE); // global d column = spread + s·16
            let off = b.idx_mul(row, d_c);
            let off = b.idx_add(off, col);
            let v = b.load(q, off);
            b.store_frag_elem(frag, e_idx, v).dep()
        })
        .collect();
    b.load_frag_vec_after(frag, &stores)
}

/// **Step-6 softmax-reduction isolation probe** (32×32×8 DE-RISK): proves the [`Builder::acc_row_reduce_32`]
/// online-softmax reduce over the `EPT_C = 16` accumulator geometry (the FA-32 row-reduce over kv), IN
/// ISOLATION, before FA. It loads `S[kv, q]` into the 32×32×8 C-accumulator layout (kv on M), computes the
/// full softmax over kv per q — `P[kv,q] = exp2(S[kv,q] − max_kv) / Σ_kv exp2(…)` — via the 16-in-register +
/// `L↔L+32` cross-lane reduce, and scatters `P`. A device allclose vs the host softmax proves the AccDist
/// reduction geometry + broadcast are correct (a missing cross-lane term → the norm sums 16 of 32 kv →
/// ~2× error). `kv = q = 32` (one 32×32 tile). This is the last un-proven FA-32 building block.
#[allow(clippy::needless_range_loop)]
pub fn softmax32_probe() -> Program {
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
pub fn v_transpose_probe(d: usize) -> Program {
    use crate::shape::Mfma32x32x8Bf16 as S;
    const KV: usize = 32;
    const Q: usize = 32;
    const PAD: usize = 8; // pitch padding (mult-of-4 kept for the b64 straight read; breaks bank conflicts)
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
            let b_s = crate::kernels::load_op_frag(&mut b, pt, b_map, 0, s * S::K, KV, lane);
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

/// **Streaming FA-forward** (non-causal, 8-warp split-Q, head-dim `d` a multiple of 16) authored on the
/// [`crate::pipeline`] ClusterCx combinator. `bh = batch·heads` independent attentions over Q/K/V/O
/// laid out `[bh, n, d]` row-major; `n` = sequence length. Grid = `bh × (n/q_blk)` workgroups, each
/// owning one `q_blk=128`-row Q block within its (b,h) slice. NUMERICALLY CORRECT: streams K/V blocks,
/// computing QKᵀ (Σ over `d/16` `mma` K-steps), the online-softmax running max/rescale/norm, and the PV
/// accumulate as `mma_atb` (the QKᵀ f32 accumulator `P` feeds PV's operand with only a bf16 cast — no
/// data transpose — while `V` is gathered transposed so `kv` lands on the MFMA contraction axis).
/// Device-gated by `flash_attention_matches_reference_on_gfx942`. Returns a lowerable [`Program`].
///
/// **PASS REQUIREMENT**: apply `.apply(VectorizePass).apply(SwizzlePass)` (matmul's order). SwizzlePass
/// folds the LDS bank-conflict swizzle (~81% → ~29% conflicts, +82% TF). VectorizePass fuses the
/// **straight K gather** into `ds_read_b64` (halving the scalar `ds_read_u16` LDS-wait traffic —
/// `ds_read_u16` 128→64, `ds_read_b64` 0→16 at d=128); it leaves the **transposed V gather** untouched
/// because [`LdsView::gather_transposed`] packs its `ept` column-strided reads into a single
/// `store_frag_vec` (no fusible `ept`-scalar-store run), so V stays bit-exact (the naive per-element V
/// gather is what made VectorizePass corrupt numerics 3.9e-4 → ~1e-1 before). V's strided read is the
/// residual scalar gather (its register transpose is barrier-bound — see report).
///
/// Accumulator carry (all `Frag<F32>`, Col map): `[o_0 .. o_{d/16−1}, att, max, norm]`. `o_df` is the
/// `[d, q]`-layout PV accumulator for output d-fragment `df` (`q` on the flat lane-axis, matching `att`
/// / `max` / `norm`); `att` is the per-KV-block QKᵀ temporary (re-zeroed each block, then overwritten
/// with the softmax weights `P` for PV to consume); `max`/`norm` are the per-Q online-softmax stats
/// (broadcast across the fragment). O is normalised (`o/norm`) and transpose-scattered in the epilogue.
#[allow(clippy::needless_range_loop)] // the d-fragment index also drives the output-d tile base
pub fn flash_attention_fwd(bh: usize, n: usize, d: usize) -> Program {
    assert!(d.is_multiple_of(EDGE), "head dim d must be a multiple of 16");
    assert!(bh >= 1, "bh (batch·heads) must be ≥ 1");
    let nthreads = NUM_WARPS * WARP; // 512
    let q_blk = NUM_WARPS * EDGE; // 128 — the workgroup Q block (8 warps × 16 rows)
    // The collaborative 512-thread K/V fill must be VEC4-aligned: `kv_blk·d % (nthreads·4) == 0`. Pick
    // the smallest multiple of 16 that satisfies it (d=128 → 16 = 1 fragment; d=64 → 32 = 2 fragments —
    // exactly tk1's KV_BLK=32). Larger `kv_blk` is processed as `kvf = kv_blk/16` KV-fragments.
    let kv_blk = {
        let mut kb = EDGE;
        while !(kb * d).is_multiple_of(nthreads * 4) {
            kb += EDGE;
        }
        kb
    };
    let (dfrags, kvf) = (d / EDGE, kv_blk / EDGE);
    assert!(n.is_multiple_of(q_blk), "N must be a multiple of the workgroup Q block ({q_blk})");
    assert!(n / kv_blk >= 2, "N must give ≥2 KV blocks (n / kv_blk ≥ 2)");

    let mut b = Builder::new("tk2_fa_fwd");
    // ABI: output O first, then inputs Q, K, V — each `[bh, n, d]` row-major (bh stacked `[n,d]` slices).
    let o = b.global::<F32>(bh * n * d);
    let q = b.global::<BF16>(bh * n * d);
    let k = b.global::<BF16>(bh * n * d);
    let v = b.global::<BF16>(bh * n * d);

    // Grid = `bh × (n/q_blk)` workgroups over ONE flat axis, decoded into `(bh_idx, qwg)` — each (b,h)
    // owns `n/q_blk` consecutive workgroups. `bh_idx·n` is this slice's global ROW base (folded into the
    // Q origin + K/V stage origin, so the O scatter — expressed off `q_origin` — inherits it for free).
    let nqb = n / q_blk;
    let wgid = b.grid_axis(0, (bh * nqb) as i64);
    let nqb_c = b.idx_const(nqb as i64);
    let bh_idx = b.idx_div(wgid, nqb_c);
    let qwg = b.idx_mod(wgid, nqb_c);
    let n_c = b.idx_const(n as i64);
    let bh_row = b.idx_mul(bh_idx, n_c); // this (b,h) slice's row base = bh_idx·n

    // 8 warps (512 threads). `tid → (warp, wlane)`: warp `w` owns Q rows `[w·16, w·16+16)` (offset
    // `warp_qoff`) within its Q block, and reduces/scatters over its own 64 lanes.
    let tid = b.block_axis(nthreads as i64);
    let warp_c = b.idx_const(WARP as i64);
    let edge_c = b.idx_const(EDGE as i64);
    let warp = b.idx_div(tid, warp_c);
    let wlane = b.idx_mod(tid, warp_c); // wave-local lane (the ds_bpermute reduce stays per-warp)
    let warp_qoff = b.idx_mul(warp, edge_c); // this warp's Q-row offset into the [q_blk, d] tile

    // ── LDS tiles: K/V shared [kv_blk, d]. Q is NOT staged through LDS (Probe B): each warp's 16 Q
    //    rows are warp-PRIVATE, so LDS-staging Q amortised nothing — it only wasted a 32KB (d=128) tile
    //    that capped occupancy at 1 wg/CU. Q is loaded global→VGPR directly in the prologue below. ──
    let k_smem = b.define_local::<BF16>(kv_blk * d);
    let v_smem = b.define_local::<BF16>(kv_blk * d);

    let row_map = Mfma16x16x16Bf16::a_map(); // A operand (contraction on the spread lane-axis)
    let col_map = Mfma16x16x16Bf16::c_map(); // B / C / accumulator operands (the Col map)

    // Movement handles. K (QKᵀ's A, row map): `n_frags = kvf` (the KV-block rows) × d-slices. V (PV's A,
    // col map): transposed gather (`kv` on spread), `dfrags` output frags per kv-slice. K/V shared (no
    // warp off). Q has no LDS view — it is register-resident (loaded straight from global below).
    let k_tile = SharedTile::new(k_smem, d);
    let v_tile = SharedTile::new(v_smem, d);
    let k_view = k_tile.gather_view(row_map, kvf, None, wlane, false);
    let v_view = v_tile.gather_view(col_map, dfrags, None, wlane, false);

    let epl_kv = kv_blk * d / nthreads; // 512-thread collaborative fill
    // K/V stream: staged by ALL 512 threads (`tid`); origin = this (b,h)'s row base `bh_row`, the
    // per-block advance rides `k_base` (so block k reads rows `[bh_idx·n + k·kv_blk, ...]`).
    let k_stage = k_tile.stage_view(k, epl_kv, tid, bh_row, d as i64, Drain::Intrinsic);
    let v_stage = v_tile.stage_view(v, epl_kv, tid, bh_row, d as i64, Drain::Intrinsic);
    // Q-row origin: the whole [q_blk, d] block base for this workgroup; each warp owns rows
    // [warp_qoff, warp_qoff+16) within it.
    let qblk_c = b.idx_const(q_blk as i64);
    let q_off = b.idx_mul(qwg, qblk_c);
    let q_origin = b.idx_add(bh_row, q_off); // global Q-row origin = bh_idx·n + qwg·q_blk (in rows)

    // ── prologue: EACH warp loads its own 16 Q rows (its `dfrags` d-slices) global→VGPR directly
    //    (Probe B — NO LDS staging, NO barrier). `q_row_base` = this warp's global Q-row origin; the
    //    QKᵀ B-operand fragment (col map) is loop-invariant, so it is hoisted once into VGPRs. ──
    let q_row_base = b.idx_add(q_origin, warp_qoff);
    let q_frags: Vec<Val<BF16>> =
        (0..dfrags).map(|s| load_q_frag_global(&mut b, q, col_map, q_row_base, d, s, wlane)).collect();

    // ── the heterogeneous slot set (DESIGN §3.2). CARRIED (seeded, loop-carried): `o_0..o_{dfrags-1}`
    //    (f32), `m` (running max), `l` (running norm). TEMPORARIES (no seed, produced+consumed within
    //    one KV block): `s_0..s_{kvf-1}` (f32 QKᵀ scores, one per KV-fragment), `p_0..p_{kvf-1}` (BF16
    //    softmax weights — native in the channel, no cast-in-PV). Per-warp registers give each warp its
    //    own o/m/l over its 16 Q rows (the warp offset rides the addressing, NOT the slot count). ──
    let o_frags: Vec<Frag<F32>> = (0..dfrags).map(|_| b.define_frag::<F32>(col_map)).collect();
    let (m_frag, l_frag) = (b.define_frag::<F32>(col_map), b.define_frag::<F32>(col_map));
    let s_frags: Vec<Frag<F32>> = (0..kvf).map(|_| b.define_frag::<F32>(col_map)).collect();
    let p_frags: Vec<Frag<BF16>> = (0..kvf).map(|_| b.define_frag::<BF16>(col_map)).collect();
    let slot_m = dfrags;
    let slot_l = dfrags + 1;
    let slot_s = |kf: usize| dfrags + 2 + kf; // s_0..s_{kvf-1}
    let slot_p = |kf: usize| dfrags + 2 + kvf + kf; // p_0..p_{kvf-1}
    let mut accs: Vec<AccSlot> = o_frags.iter().map(|&f| AccSlot::F32(f)).collect();
    accs.extend([AccSlot::F32(m_frag), AccSlot::F32(l_frag)]);
    accs.extend(s_frags.iter().map(|&f| AccSlot::F32(f)));
    accs.extend(p_frags.iter().map(|&f| AccSlot::BF16(f)));
    let mut inited: Vec<Option<Effect>> = o_frags.iter().map(|&f| Some(b.zero_init_frag(f))).collect();
    inited.push(Some(b.const_init_frag(m_frag, f32::NEG_INFINITY))); // running max seed = −∞
    inited.push(Some(b.zero_init_frag(l_frag))); // running norm seed = 0
    inited.extend((0..2 * kvf).map(|_| None)); // s_*, p_*: temporaries (produced fresh each block)

    // Per-cluster read/write slot sets (asymmetric — the point of §3.2). QKᵀ writes only `s_*`; softmax
    // reads {s_*,m,l,o_*} writes {m,l,p_*,o_*}; PV reads {p_*,o_*} writes {o_*}.
    let o_idx: Vec<usize> = (0..dfrags).collect();
    let sm_reads: Vec<usize> = (0..kvf).map(slot_s).chain([slot_m, slot_l]).chain(0..dfrags).collect();
    let sm_writes: Vec<usize> = [slot_m, slot_l].into_iter().chain((0..kvf).map(slot_p)).chain(0..dfrags).collect();
    let pv_reads: Vec<usize> = (0..kvf).map(slot_p).chain(0..dfrags).collect();

    // Softmax scale folded into the QKᵀ scores: exp2(score·log2(e)/√d) == exp(score/√d).
    let scale = std::f32::consts::LOG2_E / (d as f32).sqrt();
    let scale_bcast = {
        let s = b.f32(scale);
        let cs: Vec<Val<F32>> = (0..EPT).map(|_| s).collect();
        b.vec_build(&cs)
    };
    let zero_c = {
        // QKᵀ's C = 0 (att re-zeroed every KV block — `s_*` are temporaries, not read from the carry).
        let zs: Vec<Val<F32>> = (0..EPT).map(|_| b.f32(0.0)).collect();
        b.vec_build(&zs)
    };

    // ── the three compute bodies, each declaring ONLY the slots it touches ──
    // QKᵀ: att[kf] = K[kf]·Qᵀ over the `kvf` KV-fragments (each contracting `d` = Σ over `dfrags` `mma`
    // K-steps). reads nothing (re-zeros); writes the `kvf` `s_*`. K indexed `[s·kvf + kf]`.
    let qk = Compute::<FaHooks>::new(
        0,
        vec![],
        (0..kvf).map(slot_s).collect::<Vec<_>>(),
        move |b: &mut Builder, op: Option<&FaOp>, _reads: &[SlotVal]| {
            let k_frags = op.expect("QKᵀ consumes gathered K");
            (0..kvf)
                .map(|kf| {
                    let mut att = zero_c;
                    for s in 0..dfrags {
                        att = b.mma(k_frags[s * kvf + kf], q_frags[s], att, EPT); // d-slice K·Qᵀ
                    }
                    SlotVal::F32(att)
                })
                .collect()
        },
    );

    // Online softmax (operand = None): reads {s_*, m, l, o_*}, writes {m, l, p_*, o_*}. Reduce over ALL
    // `kvf·16` KV rows by CHAINING `frag_col_reduce` across the KV-fragments (each fold takes the prior
    // as its running init) — barrier-free `ds_bpermute`, per warp. Produces `p_*` as BF16.
    let softmax = Compute::<FaHooks>::new(
        None,
        sm_reads,
        sm_writes,
        move |b: &mut Builder, _op: Option<&FaOp>, reads: &[SlotVal]| {
            let (max_old, norm_old) = (reads[kvf].f32(), reads[kvf + 1].f32());
            let s: Vec<Val<F32>> = (0..kvf).map(|kf| b.mul(reads[kf].f32(), scale_bcast)).collect();
            let mut m = max_old; // running max over the kvf fragments (chained reductions)
            for &sk in &s {
                m = b.frag_col_reduce(sk, wlane, m, false);
            }
            let corr = b.sub(max_old, m);
            let scale_f = b.exp2(corr); // exp2(max_old − max_new)
            let mut norm = b.mul(norm_old, scale_f); // rescale, then fold Σ_kv P over the fragments
            let mut p_bf: Vec<Val<BF16>> = Vec::with_capacity(kvf);
            for &sk in &s {
                let sm = b.sub(sk, m);
                let p = b.exp2(sm); // softmax weights P (f32)
                norm = b.frag_col_reduce(p, wlane, norm, true);
                p_bf.push(b.cast_vec_bf16(p)); // cast into the channel (PV reads bf16)
            }
            let mut out = vec![SlotVal::F32(m), SlotVal::F32(norm)];
            out.extend(p_bf.into_iter().map(SlotVal::BF16));
            for i in 0..dfrags {
                out.push(SlotVal::F32(b.mul(reads[kvf + 2 + i].f32(), scale_f))); // O *= corr
            }
            out
        },
    );

    // P·V accumulate (`mma_atb` over `kv`): reads {p_*, o_*}, writes {o_*}. Contract over the `kvf`
    // KV-fragments: `o_df += Σ_kf mma(V[kf,df], p_kf)`. `P` is native bf16; V is the transposed gather
    // indexed `[kf·dfrags + df]`.
    let pv = Compute::<FaHooks>::new(
        1,
        pv_reads,
        o_idx.clone(),
        move |b: &mut Builder, op: Option<&FaOp>, reads: &[SlotVal]| {
            let v_frags = op.expect("PV consumes gathered V");
            (0..dfrags)
                .map(|df| {
                    let mut o = reads[kvf + df].f32();
                    for kf in 0..kvf {
                        o = b.mma(v_frags[kf * dfrags + df], reads[kf].bf16(), o, EPT);
                    }
                    SlotVal::F32(o)
                })
                .collect()
        },
    );

    let hooks = FaHooks { k_view, v_view, k_stage, v_stage, dfrags, kvf };
    let acc_final = pipeline(
        &mut b,
        n / kv_blk, // nblocks (streaming over KV)
        kv_blk * d, // k_step: the FLAT per-block advance (kv_blk rows · d)
        2,          // ksteps: gather slices (K, V)
        &accs,
        &inited,
        None,  // warp_row: FA wants NO wave-phase ping-pong — warps run disjoint Q rows, so there is
        false, // asm_gather   no co-resident load/compute pair to steer (the brief: keep it None)
        false, // resident
        CommitDrain::IntrinsicAuto,
        false, // bare_seals
        false, // pin_mfma
        hooks,
    )
    .cluster(Mem::builder().prefetch([0, 1]).gathers([0, 1]).commit(true).build())
    .cluster(qk)
    .cluster(softmax)
    .cluster(pv)
    .build();

    // ── post-loop: normalize O = o / norm (per q, broadcast across d) and transpose-scatter to O. Each
    //    warp writes its own 16 Q rows: `q_global = q_origin + warp_qoff + q_in_frag`. ──
    let norm_vec = b.load_frag_vec(acc_final[slot_l].f32());
    let recip_norm = b.recip(norm_vec);
    let d_c = b.idx_const(d as i64);
    // `q_row_base` (this warp's global Q-row origin) was computed in the prologue (Probe B).
    let mut roots = Vec::new();
    for df in 0..dfrags {
        let o_vec = b.load_frag_vec(acc_final[df].f32());
        let o_norm = b.mul(o_vec, recip_norm);
        for inner in 0..EPT {
            let inner_c = b.idx_const(inner as i64);
            // o_df is Col-map [d = spread = row, q = flat = col]; scatter to O[q_global, d_global].
            let (row, col) = b.lane_rc(col_map, wlane, inner_c);
            let q_global = b.idx_add(q_row_base, col);
            let qg_d = b.idx_mul(q_global, d_c);
            let d_global = crate::kernels::offset_by(&mut b, row, df * EDGE);
            let off = b.idx_add(qg_d, d_global);
            let val = b.vec_extract(o_norm, inner);
            roots.push(b.store(o, off, val));
        }
    }

    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_fa_fwd".into() }
}

/// Warps per workgroup for the **32×32×8 FA** (§Step 6): 4 warps × 32 Q rows = a `q_blk = 128` block
/// (matching the 16×16 FA's Q-block for a fair comparison). Each warp owns 32 Q rows and computes its
/// own 32(kv)×32(q) MFMA tile; all 4 share ONE K/V LDS tile per KV block. No cross-warp reduction (the
/// softmax `acc_row_reduce_32` stays within each 64-lane warp — disjoint Q rows).
const NUM_WARPS_32: usize = 4;
/// The 32×32×8 KV-block size (one MFMA N/M tile = aiter's `ts_kv`). Four hardware `K = 8` slices.
const KV_BLK_32: usize = 32;
/// The transposed-V LDS row padding (mult-of-4 kept for the b64 straight read; breaks bank conflicts —
/// an un-padded `[d, kv]` transpose regresses, proven by `v_transpose_probe`).
const VT_PAD: usize = 8;

/// One K/V-slice's gathered A-operands — the FA-32 [`Hooks::Op`]. Slice 0 = K (`dslices` QKᵀ A-operands,
/// contraction `d` over `K = 8`); slice 1 = V (the `dtiles·ksl` transposed-V A-operands, indexed
/// `[dt·ksl + s]`, contraction `kv` over `K = 8`). Both `Vec<Val<BF16>>` (`EPT_A = 4` each).
type Fa32Op = Vec<Val<BF16>>;

/// The register-staged fill carried prefetch→commit: block k+1's K and V chunks in VGPRs (the
/// collaborative 256-thread fill's per-thread `epl` scalar elements).
struct Fa32Fill {
    k: Vec<Val<BF16>>,
    v: Vec<Val<BF16>>,
}

/// FA-32's [`Hooks`] — the 32×32×8 movement (natural-K LDS + padded-transposed-V LDS). Rides the SAME
/// ClusterCx pipeline the 16×16 [`flash_attention_fwd`] drives, but with the 32×32×8 fragment addressing:
/// each A-operand is a contiguous `EPT_A = 4` LDS run (`ds_read` straight, no register transpose — V's
/// transpose is done write-side into LDS) rather than the movement layer's 16×16 [`LdsView`] gather. The
/// fills/gathers are SCALAR in this pass (the vectorized b128 fill + b64 gather are Phase-2 refinements);
/// gathered operands round-trip through a fragment so the WAR barrier consumes a proper store token
/// (the [`Hooks::gather`] contract), exactly as the 16×16 FA — SROA elides the store/load.
struct Fa32Hooks {
    k: Buf<BF16>,
    v: Buf<BF16>,
    k_lds: Lds<BF16>,
    vt_lds: Lds<BF16>,
    /// `bh_row · d` — this workgroup's (b,h) slice flat base (folded into every fill global offset so the
    /// pipeline's FLAT per-block `k_base` lands at the right `[bh, n, d]` row).
    bh_row_d: Idx,
    /// The collaborative fill thread id (all `nthreads`) and its per-thread element run `epl`.
    tid: Idx,
    epl: usize,
    /// The per-warp gather axis parts: `q_in = wlane % 32` (kv/d-in-tile), `half_off = (wlane/32)·4`.
    q_in: Idx,
    half_off: Idx,
    d: usize,
    pitch: usize,
    dslices: usize,
    dtiles: usize,
    ksl: usize,
}

impl Hooks for Fa32Hooks {
    type Op = Fa32Op;
    type Reg = Fa32Fill;
    const PREFETCH_TILES: usize = 2; // 0 = K, 1 = V

    fn prefetch(
        &mut self,
        b: &mut Builder,
        k_base: Idx,
        tile: usize,
        prev: Option<Fa32Fill>,
        _order: &[TileId],
    ) -> (Fa32Fill, Vec<TileId>) {
        // global→VGPR: this thread's coalesced `epl` scalar loads at flat `bh_row·d + k_base + tid·epl`
        // (the FLAT global index into `[bh, n, d]` — block `k`'s first element is `bh_row·d + k_base`).
        let mut reg = prev.unwrap_or(Fa32Fill { k: Vec::new(), v: Vec::new() });
        let epl_c = b.idx_const(self.epl as i64);
        let lane_epl = b.idx_mul(self.tid, epl_c);
        let flat_base = b.idx_add(self.bh_row_d, k_base);
        let flat_base = b.idx_add(flat_base, lane_epl);
        let buf = match tile {
            0 => self.k,
            1 => self.v,
            _ => panic!("FA-32 prefetch: tile ∈ {{0=K, 1=V}}, got {tile}"),
        };
        let loaded: Vec<Val<BF16>> = (0..self.epl)
            .map(|i| {
                let off = offset_by(b, flat_base, i);
                b.load(buf, off)
            })
            .collect();
        let anchors = loaded.iter().map(|v| v.id).collect();
        match tile {
            0 => reg.k = loaded,
            1 => reg.v = loaded,
            _ => unreachable!(),
        }
        (reg, anchors)
    }

    fn commit(&mut self, b: &mut Builder, _k_base: Idx, reg: &Fa32Fill, war: &[TileId]) -> Vec<Effect> {
        // VGPR→LDS behind the WAR barrier: K natural `[kv, d]`, V transposed `[d, kv]` with padded pitch.
        // `flat = tid·epl + i → (kv, d_idx)` is block-relative (k_base rides only the global load).
        let k_lds = if war.is_empty() { self.k_lds } else { b.lds_after(self.k_lds, war) };
        let vt_lds = if war.is_empty() { self.vt_lds } else { b.lds_after(self.vt_lds, war) };
        let epl_c = b.idx_const(self.epl as i64);
        let lane_epl = b.idx_mul(self.tid, epl_c);
        let d_c = b.idx_const(self.d as i64);
        let pitch_c = b.idx_const(self.pitch as i64);
        let mut effs = Vec::with_capacity(2 * self.epl);
        for i in 0..self.epl {
            let flat = offset_by(b, lane_epl, i);
            let kv = b.idx_div(flat, d_c);
            let d_idx = b.idx_mod(flat, d_c);
            // K natural [kv, d] through the `LdsCol` swizzle hole (cols = d, a power of 2 for d ∈ {64,128}):
            // `SwizzlePass` folds it to the XOR bank-spread, the gather reads the SAME `lds_col(kv, …, d)`.
            // (V's transpose keeps its padded pitch — non-power-of-2, so the XOR swizzle is not applied.)
            let k_col = b.lds_col(kv, d_idx, self.d);
            let k_dst = b.idx_mul(kv, d_c);
            let k_dst = b.idx_add(k_dst, k_col);
            effs.push(b.store_lds(k_lds, k_dst, reg.k[i]));
            let v_dst = b.idx_mul(d_idx, pitch_c);
            let v_dst = b.idx_add(v_dst, kv); // V transposed [d, kv]
            effs.push(b.store_lds(vt_lds, v_dst, reg.v[i]));
        }
        effs
    }

    fn gather(&mut self, b: &mut Builder, slice: usize, raw: &[TileId]) -> (Fa32Op, Vec<TileId>, TileId) {
        use crate::shape::Mfma32x32x8Bf16 as S;
        let d_c = b.idx_const(self.d as i64);
        let pitch_c = b.idx_const(self.pitch as i64);
        // Read each contiguous `EPT_A` LDS run, then round-trip through a fragment so the WAR barrier gets
        // a proper store token (the gather contract; SROA elides the store/load into a straight operand).
        let read = |b: &mut Builder, lds: Lds<BF16>, base: Idx, gathers: &mut Vec<TileId>| -> Val<BF16> {
            let v = b.load_lds_vec_after(lds, base, S::EPT_A, raw);
            let frag = b.define_frag::<BF16>(S::a_map());
            let st = b.store_frag_vec(frag, v).dep();
            gathers.push(st);
            b.load_frag_vec_after(frag, &[st])
        };
        let mut vecs = Vec::new();
        let mut gathers = Vec::new();
        match slice {
            // K A-operand: k_lds[kv = q_in, d = ki·8 + half_off .. +4] (contiguous), `dslices` of them.
            // The 4-run start `col_base` is 4-aligned and the swizzle `delta(row)` is 4-aligned (d ∈
            // {64,128}), so the swizzled base + [0..4) stays the fill's contiguous 4-run — `ds_read_b64` safe.
            0 => {
                for ki in 0..self.dslices {
                    let col_base = offset_by(b, self.half_off, ki * S::K);
                    let kcol = b.lds_col(self.q_in, col_base, self.d);
                    let base = b.idx_mul(self.q_in, d_c);
                    let base = b.idx_add(base, kcol);
                    vecs.push(read(b, self.k_lds, base, &mut gathers));
                }
            }
            // V A-operand: vt_lds[d = dt·32 + q_in, kv = s·8 + half_off .. +4] (contiguous), `dtiles·ksl`.
            1 => {
                for dt in 0..self.dtiles {
                    for s in 0..self.ksl {
                        let d_row = offset_by(b, self.q_in, dt * S::M);
                        let base = b.idx_mul(d_row, pitch_c);
                        let kvcol = offset_by(b, self.half_off, s * S::K);
                        let base = b.idx_add(base, kvcol);
                        vecs.push(read(b, self.vt_lds, base, &mut gathers));
                    }
                }
            }
            _ => panic!("FA-32 gather: slice ∈ {{0=K, 1=V}}, got {slice}"),
        }
        let anchor = vecs[0].id;
        (vecs, gathers, anchor)
    }
}

/// **Streaming FA-forward on the 32×32×8 MFMA** (§Step 6 — the wide-core variant, KEPT SEPARATE from the
/// frozen 16×16 [`flash_attention_fwd`]). Non-causal, `bh = batch·heads` independent `[bh,n,d]` attentions;
/// 4 warps × 32 Q rows = a `q_blk = 128` block. Assembles the four device-proven 32×32×8 primitives:
/// QKᵀ (`v_mfma_f32_32x32x8`, `S[kv,q]`, kv on M) → online softmax over kv via [`Builder::acc_row_reduce_32`]
/// (the `EPT_C = 16` AccDist reduce) → P→PV relayout via [`Builder::pv_relayout_s49`] (`v_perm s49`) → P·V
/// with V staged through the padded transposed LDS (read straight).
///
/// **Phase 1 (the ClusterCx port):** the KV stream is a **rolled [`crate::pipeline`] loop** (like the 16×16
/// FA) rather than the correctness-first unrolled assembly — O/m/l are the loop-carried accumulators, the
/// per-KV `s`/`p` are pipeline TEMPORARIES, and [`Fa32Hooks`] supplies the register-staged prefetch, commit,
/// and gather over the SAME Mem/QKᵀ/softmax/PV cluster structure. This scales past `n = 128` and inherits the
/// prefetch/commit/WAR/RAW machinery. Device-gated by `flash_attention32_matches_reference_on_gfx942`
/// (tight `atol` at d=64 AND d=128); no ping-pong (`warp_row = None`: disjoint-Q warps).
///
/// **PASS REQUIREMENT**: apply `.apply(SwizzlePass)`. It folds the **K-tile** LDS bank swizzle (`cols = d`,
/// a power of 2) into the XOR bank-spread — the fill and gather both route K's column through
/// `lds_col(row, …, d)`, so they stay in agreement. The **V tile** keeps its padded pitch (non-power-of-2,
/// so the XOR is not applied — the pad is what breaks its conflicts). VectorizePass is unnecessary: the
/// K/V gathers are already `ds_read_b64` ([`Builder::load_lds_vec_after`]) and the fills are `store_lds`
/// (no fusible scalar frag-store run), so the pass touches only the loop-invariant Q prologue (negligible).
#[allow(clippy::needless_range_loop)]
pub fn flash_attention_fwd_32(bh: usize, n: usize, d: usize) -> Program {
    use crate::shape::Mfma32x32x8Bf16 as S;
    assert!(d.is_multiple_of(S::M), "FA-32 head dim d must be a multiple of 32");
    assert!(bh >= 1, "bh must be ≥ 1");
    let (m32, k8) = (S::M, S::K); // 32, 8
    let nthreads = NUM_WARPS_32 * WARP; // 256
    let q_blk = NUM_WARPS_32 * m32; // 128
    let pitch = KV_BLK_32 + VT_PAD; // transposed-V LDS row pitch
    assert!(n.is_multiple_of(q_blk), "n must be a multiple of the Q block ({q_blk})");
    assert!(n.is_multiple_of(KV_BLK_32) && n / KV_BLK_32 >= 2, "n must give ≥2 KV blocks (the rolled pipeline)");
    let dslices = d / k8; // QKᵀ K-steps (contract d over K=8)
    let dtiles = d / m32; // PV output d-tiles (32 each)
    let ksl = KV_BLK_32 / k8; // 4 PV K-slices per KV block

    let mut b = Builder::new("tk2_fa_fwd_32");
    // ABI: O[bh·n, d] then Q, K, V — each `[bh, n, d]` row-major.
    let o = b.global::<F32>(bh * n * d);
    let q = b.global::<BF16>(bh * n * d);
    let k = b.global::<BF16>(bh * n * d);
    let v = b.global::<BF16>(bh * n * d);

    // Grid = bh × (n/q_blk); decode (bh_idx, qwg). bh_idx·n = this (b,h) slice's global row base.
    let nqb = n / q_blk;
    let wgid = b.grid_axis(0, (bh * nqb) as i64);
    let nqb_c = b.idx_const(nqb as i64);
    let bh_idx = b.idx_div(wgid, nqb_c);
    let qwg = b.idx_mod(wgid, nqb_c);
    let n_c = b.idx_const(n as i64);
    let bh_row = b.idx_mul(bh_idx, n_c);

    let tid = b.block_axis(nthreads as i64);
    let warp_c = b.idx_const(WARP as i64);
    let m32_c = b.idx_const(m32 as i64);
    let warp = b.idx_div(tid, warp_c);
    let wlane = b.idx_mod(tid, warp_c);
    let warp_qoff = b.idx_mul(warp, m32_c); // this warp's 32-row Q offset in the [q_blk, d] block

    // This warp's global Q-row origin.
    let qblk_c = b.idx_const(q_blk as i64);
    let q_off = b.idx_mul(qwg, qblk_c);
    let q_origin = b.idx_add(bh_row, q_off);
    let q_row_base = b.idx_add(q_origin, warp_qoff);

    let d_c = b.idx_const(d as i64);
    // Per-lane axis parts (shared by every fragment gather): q = wlane%32, dloc/kvloc = (wlane/32)·4.
    let (q_in, half_off) = {
        let n_lanes = b.idx_const(m32 as i64); // 32
        let q_in = b.idx_mod(wlane, n_lanes);
        let half = b.idx_div(wlane, n_lanes);
        let four = b.idx_const(4);
        (q_in, b.idx_mul(half, four))
    };

    // ── prologue: Q B-operands (loop-invariant, hoisted). q_frags[ki] = Q[q_in, 8ki + half_off .. +4]. ──
    let q_frags: Vec<Val<BF16>> = (0..dslices)
        .map(|ki| {
            let q_row = b.idx_add(q_row_base, q_in);
            let base = b.idx_mul(q_row, d_c);
            let dcol = offset_by(&mut b, half_off, ki * k8);
            let base = b.idx_add(base, dcol);
            let frag = b.define_frag::<BF16>(S::b_map());
            let stores: Vec<TileId> = (0..S::EPT_B)
                .map(|e| {
                    let e_c = b.idx_const(e as i64);
                    let off = b.idx_add(base, e_c);
                    let val = b.load(q, off);
                    b.store_frag_elem(frag, e_c, val).dep()
                })
                .collect();
            b.load_frag_vec_after(frag, &stores)
        })
        .collect();

    // Shared LDS: K natural [kv,d], V transposed padded [d, pitch].
    let k_lds = b.define_local::<BF16>(KV_BLK_32 * d);
    let vt_lds = b.define_local::<BF16>(d * pitch);
    let epl = KV_BLK_32 * d / nthreads; // collaborative fill, per thread

    // Softmax scale folded into exp2: exp2(score·log2(e)/√d) == exp(score/√d). 16-wide broadcast.
    let scale = std::f32::consts::LOG2_E / (d as f32).sqrt();
    let scale_bcast = {
        let c = b.f32(scale);
        let cs: Vec<Val<F32>> = (0..S::EPT_C).map(|_| c).collect();
        b.vec_build(&cs)
    };

    // ── the heterogeneous slot set (DESIGN §3.2). CARRIED: `o_0..o_{dtiles-1}` (16-wide f32 PV acc),
    //    `m` (running max), `l` (running norm). TEMPORARIES (produced+consumed within one KV block):
    //    `s` (QKᵀ scores, 16-wide f32), `p` (softmax weights, 16-wide f32 → v_perm-packed in PV). All f32
    //    on the `EPT_C = 16` accumulator (`c_map`); each warp keeps its OWN o/m/l over its 32 Q rows. ──
    let o_frags: Vec<Frag<F32>> = (0..dtiles).map(|_| b.define_frag::<F32>(S::c_map())).collect();
    let (m_frag, l_frag) = (b.define_frag::<F32>(S::c_map()), b.define_frag::<F32>(S::c_map()));
    let (s_frag, p_frag) = (b.define_frag::<F32>(S::c_map()), b.define_frag::<F32>(S::c_map()));
    let slot_m = dtiles;
    let slot_l = dtiles + 1;
    let slot_s = dtiles + 2;
    let slot_p = dtiles + 3;
    let mut accs: Vec<AccSlot> = o_frags.iter().map(|&f| AccSlot::F32(f)).collect();
    accs.extend([AccSlot::F32(m_frag), AccSlot::F32(l_frag), AccSlot::F32(s_frag), AccSlot::F32(p_frag)]);
    let mut inited: Vec<Option<Effect>> = o_frags.iter().map(|&f| Some(b.zero_init_frag(f))).collect();
    inited.push(Some(b.const_init_frag(m_frag, f32::NEG_INFINITY))); // running max seed = −∞
    inited.push(Some(b.zero_init_frag(l_frag))); // running norm seed = 0
    inited.push(None); // s: temporary
    inited.push(None); // p: temporary

    // Per-cluster read/write slot sets (asymmetric — the §3.2 point). QKᵀ writes only `s`; softmax reads
    // {s,m,l,o_*} writes {m,l,p,o_*}; PV reads {p,o_*} writes {o_*}.
    let o_idx: Vec<usize> = (0..dtiles).collect();
    let sm_reads: Vec<usize> = [slot_s, slot_m, slot_l].into_iter().chain(0..dtiles).collect();
    let sm_writes: Vec<usize> = [slot_m, slot_l, slot_p].into_iter().chain(0..dtiles).collect();
    let pv_reads: Vec<usize> = [slot_p].into_iter().chain(0..dtiles).collect();

    // ── the three compute bodies, each declaring ONLY the slots it touches ──
    // QKᵀ: S[kv,q] = Σ_ki mma(K_ki, Q_ki) (K A-operand, Q B-operand). reads nothing (re-zeros); writes `s`.
    let qk = Compute::<Fa32Hooks>::new(
        0,
        vec![],
        vec![slot_s],
        move |b: &mut Builder, op: Option<&Fa32Op>, _reads: &[SlotVal]| {
            let k_frags = op.expect("QKᵀ consumes gathered K");
            let zeros: Vec<Val<F32>> = (0..S::EPT_C).map(|_| b.f32(0.0)).collect();
            let mut s_acc = b.vec_build(&zeros);
            for ki in 0..dslices {
                s_acc = b.mma_of::<S>(k_frags[ki], q_frags[ki], s_acc);
            }
            vec![SlotVal::F32(s_acc)]
        },
    );

    // Online softmax (operand None): reads {s,m,l,o_*}, writes {m,l,p,o_*}. scale, running max via
    // `acc_row_reduce_32` (the C=16 AccDist reduce over kv), O/l rescale, P = exp2(S−max), l += Σ_kv P.
    let softmax = Compute::<Fa32Hooks>::new(
        None,
        sm_reads,
        sm_writes,
        move |b: &mut Builder, _op: Option<&Fa32Op>, reads: &[SlotVal]| {
            let (s_acc, m_run, l_run) = (reads[0].f32(), reads[1].f32(), reads[2].f32());
            let s_scaled = b.mul(s_acc, scale_bcast);
            let m_new = b.acc_row_reduce_32(s_scaled, wlane, m_run, false);
            let corr = {
                let diff = b.sub(m_run, m_new);
                b.exp2(diff) // exp2(max_old − max_new)
            };
            let l_resc = b.mul(l_run, corr);
            let p = {
                let sh = b.sub(s_scaled, m_new);
                b.exp2(sh) // softmax weights P (f32, 16-wide)
            };
            let l_new = b.acc_row_reduce_32(p, wlane, l_resc, true);
            let mut out = vec![SlotVal::F32(m_new), SlotVal::F32(l_new), SlotVal::F32(p)];
            for i in 0..dtiles {
                out.push(SlotVal::F32(b.mul(reads[3 + i].f32(), corr))); // O *= corr
            }
            out
        },
    );

    // P·V (`mma_atb` over kv): reads {p,o_*}, writes {o_*}. `pv_relayout_s49(p)` → 4 B-operands; contract
    // the `ksl` K-slices: `o_dt += Σ_s mma(V[dt·ksl+s], b_ops[s])`.
    let pv = Compute::<Fa32Hooks>::new(
        1,
        pv_reads,
        o_idx.clone(),
        move |b: &mut Builder, op: Option<&Fa32Op>, reads: &[SlotVal]| {
            let v_frags = op.expect("PV consumes gathered V");
            let b_ops = b.pv_relayout_s49(reads[0].f32());
            (0..dtiles)
                .map(|dt| {
                    let mut o = reads[1 + dt].f32();
                    for s in 0..ksl {
                        o = b.mma_of::<S>(v_frags[dt * ksl + s], b_ops[s], o);
                    }
                    SlotVal::F32(o)
                })
                .collect()
        },
    );

    let bh_row_d = b.idx_mul(bh_row, d_c);
    let hooks = Fa32Hooks { k, v, k_lds, vt_lds, bh_row_d, tid, epl, q_in, half_off, d, pitch, dslices, dtiles, ksl };
    let acc_final = pipeline(
        &mut b,
        n / KV_BLK_32, // nblocks (streaming over KV)
        KV_BLK_32 * d, // k_step: the FLAT per-block advance (kv_blk rows · d)
        2,             // ksteps: gather slices (K, V)
        &accs,
        &inited,
        None,  // warp_row: no ping-pong (disjoint-Q warps, no co-resident load/compute pair)
        false, // asm_gather
        false, // resident
        CommitDrain::IntrinsicAuto,
        false, // bare_seals
        false, // pin_mfma
        hooks,
    )
    .cluster(Mem::builder().prefetch([0, 1]).gathers([0, 1]).commit(true).build())
    .cluster(qk)
    .cluster(softmax)
    .cluster(pv)
    .build();

    // ── epilogue: O /= l (per q, broadcast across d), transpose-scatter to O[q_global, d_global]. ──
    let recip_l = {
        let l_vec = b.load_frag_vec(acc_final[slot_l].f32());
        b.recip(l_vec)
    };
    let dist = S::acc_dist();
    let mut roots = Vec::new();
    for dt in 0..dtiles {
        let o_vec = b.load_frag_vec(acc_final[dt].f32());
        let o_norm = b.mul(o_vec, recip_l);
        for i in 0..S::EPT_C {
            let (row, col) = b.acc_rc(dist, wlane, i); // row = d-in-tile, col = q-in-tile
            let d_global = offset_by(&mut b, row, dt * m32);
            let q_global = b.idx_add(q_row_base, col);
            let off = b.idx_mul(q_global, d_c);
            let off = b.idx_add(off, d_global);
            let val = b.vec_extract(o_norm, i);
            roots.push(b.store(o, off, val));
        }
    }
    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_fa_fwd_32".into() }
}
