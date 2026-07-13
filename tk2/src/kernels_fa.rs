//! **Flash-Attention forward on the ClusterCx declarative pipeline** — a NUMERICALLY CORRECT,
//! MULTI-WARP (8-warp split-Q) FA-forward proving [`crate::pipeline`] (built for the HK GEMM)
//! generalises to a second, differently-shaped kernel. Streams K/V blocks: QKᵀ → online-softmax → P·V
//! accumulate → normalize → write O. Non-causal, `b = h = 1`, head dim `d` any multiple of 16.
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

use crate::build::{BF16, Builder, Effect, F32, Frag, Idx, Val};
use crate::ir::{FragMap, TileId};
use crate::kernels::{EDGE, Program};
use crate::movement::{Drain, LdsStage, LdsView, SharedTile};
use crate::pipeline::{AccSlot, CommitDrain, Compute, Hooks, Mem, SlotVal, pipeline};

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

/// gfx942 elements-per-thread for the 16×16 fragment.
const EPT: usize = 4;

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

    let col = FragMap::gfx942_16x16(true);
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

/// **Streaming FA-forward** (non-causal, `b = h = 1`, one warp, `q_blk = kv_blk = 16`, head-dim `d` a
/// multiple of 16) authored on the [`crate::pipeline`] ClusterCx combinator. `n` = sequence length
/// (KV blocks = `n/16 ≥ 2`). NUMERICALLY CORRECT: streams K/V blocks, computing QKᵀ (Σ over `d/16`
/// `mma` K-steps), the online-softmax running max/rescale/norm, and the PV accumulate as `mma_atb`
/// (the QKᵀ f32 accumulator `P` feeds PV's operand with only a bf16 cast — no data transpose — while
/// `V` is gathered transposed so `kv` lands on the MFMA contraction axis). Device-gated by
/// `flash_attention_matches_reference_on_gfx942`. Returns a lowerable [`Program`].
///
/// Accumulator carry (all `Frag<F32>`, Col map): `[o_0 .. o_{d/16−1}, att, max, norm]`. `o_df` is the
/// `[d, q]`-layout PV accumulator for output d-fragment `df` (`q` on the flat lane-axis, matching `att`
/// / `max` / `norm`); `att` is the per-KV-block QKᵀ temporary (re-zeroed each block, then overwritten
/// with the softmax weights `P` for PV to consume); `max`/`norm` are the per-Q online-softmax stats
/// (broadcast across the fragment). O is normalised (`o/norm`) and transpose-scattered in the epilogue.
#[allow(clippy::needless_range_loop)] // the d-fragment index also drives the output-d tile base
pub fn flash_attention_fwd(n: usize, d: usize) -> Program {
    assert!(d.is_multiple_of(EDGE), "head dim d must be a multiple of 16");
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
    // ABI: output O first, then inputs Q, K, V (all flat [n, d]).
    let o = b.global::<F32>(n * d);
    let q = b.global::<BF16>(n * d);
    let k = b.global::<BF16>(n * d);
    let v = b.global::<BF16>(n * d);

    // One workgroup per `q_blk`-row Q block; 8 warps (512 threads). `tid → (warp, wlane)`: warp `w`
    // owns Q rows `[w·16, w·16+16)` (offset `warp_qoff`), and reduces/scatters over its own 64 lanes.
    let qwg = b.grid_axis(0, (n / q_blk) as i64);
    let tid = b.block_axis(nthreads as i64);
    let warp_c = b.idx_const(WARP as i64);
    let edge_c = b.idx_const(EDGE as i64);
    let warp = b.idx_div(tid, warp_c);
    let wlane = b.idx_mod(tid, warp_c); // wave-local lane (the ds_bpermute reduce stays per-warp)
    let warp_qoff = b.idx_mul(warp, edge_c); // this warp's Q-row offset into the [q_blk, d] tile

    // ── LDS tiles: K/V shared [kv_blk, d]; Q the whole [q_blk, d] workgroup block. ──
    let k_smem = b.define_local::<BF16>(kv_blk * d);
    let v_smem = b.define_local::<BF16>(kv_blk * d);
    let q_smem = b.define_local::<BF16>(q_blk * d);

    let row_map = FragMap::gfx942_16x16(false); // A operand (contraction on the spread lane-axis)
    let col_map = FragMap::gfx942_16x16(true); // B / C / accumulator operands

    // Movement handles. K (QKᵀ's A, row map): `n_frags = kvf` (the KV-block rows) × d-slices. Q (QKᵀ's
    // B, col map): this warp's single 16-row sub-block via `warp_off = warp_qoff`. V (PV's A, col map):
    // transposed gather (`kv` on spread), `dfrags` output frags per kv-slice. K/V shared (no warp off).
    let k_tile = SharedTile::new(k_smem, d);
    let v_tile = SharedTile::new(v_smem, d);
    let q_tile = SharedTile::new(q_smem, d);
    let k_view = k_tile.gather_view(row_map, kvf, None, wlane, false);
    let v_view = v_tile.gather_view(col_map, dfrags, None, wlane, false);
    let q_view = q_tile.gather_view(col_map, 1, Some(warp_qoff), wlane, false);

    let (epl_kv, epl_q) = (kv_blk * d / nthreads, q_blk * d / nthreads); // 512-thread collaborative fill
    let zero = b.idx_const(0);
    // K/V stream: staged by ALL 512 threads (`tid`); origin row 0, the per-block advance rides `k_base`.
    let k_stage = k_tile.stage_view(k, epl_kv, tid, zero, d as i64, Drain::Intrinsic);
    let v_stage = v_tile.stage_view(v, epl_kv, tid, zero, d as i64, Drain::Intrinsic);
    // Q: the whole [q_blk, d] block staged ONCE by all 512 threads from Q[qwg·q_blk .., :].
    let qblk_c = b.idx_const(q_blk as i64);
    let q_origin = b.idx_mul(qwg, qblk_c); // workgroup Q-row origin (in rows)
    let q_stage = q_tile.stage_view(q, epl_q, tid, q_origin, d as i64, Drain::Intrinsic);

    // ── prologue: stage + commit Q once, then EACH warp gathers its own 16-row sub-block's `dfrags`
    //    d-slices (loop-invariant QKᵀ B; `warp_qoff` in the view selects this warp's Q rows). ──
    let q_loaded = q_stage.prefetch(&mut b, zero, &[]);
    let q_fill = q_stage.commit(&mut b, &q_loaded, &[]);
    let q_fill_deps: Vec<TileId> = q_fill[1..].iter().map(|e| e.dep()).collect();
    let q_bar = b.barrier(q_fill[0], &q_fill_deps);
    let q_frags: Vec<Val<BF16>> = (0..dfrags)
        .map(|s| {
            let (qv, _g) = q_view.slice(s).gather(&mut b, &[q_bar.dep()]);
            qv[0]
        })
        .collect();

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
    let q_row_base = b.idx_add(q_origin, warp_qoff); // this warp's global Q-row origin
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
