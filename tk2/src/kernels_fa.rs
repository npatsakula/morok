//! **Flash-Attention forward on the ClusterCx declarative pipeline** — a NUMERICALLY CORRECT,
//! single-warp FA-forward proving [`crate::pipeline`] (built for the HK GEMM) generalises to a second,
//! differently-shaped kernel. Streams K/V blocks: QKᵀ → online-softmax → P·V accumulate → normalize →
//! write O. Non-causal, `b = h = 1`, `q_blk = kv_blk = 16`, head dim `d` any multiple of 16.
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
//! Device-gated by `flash_attention_matches_reference_on_gfx942` (allclose vs an f32 reference at
//! d=64 and d=128). Perf / multi-warp / ping-pong / swizzle are Phase B — this is the correctness base.

use crate::build::{BF16, Builder, Effect, F32, Frag, Idx, Val};
use crate::ir::{FragMap, TileId};
use crate::kernels::{EDGE, Program};
use crate::movement::{Drain, LdsStage, LdsView, SharedTile};
use crate::pipeline::{AccSlot, CommitDrain, Compute, Hooks, Mem, SlotVal, pipeline};

const WARP: usize = 64;

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
        let (vecs, g) = match slice {
            // K: the `dfrags` d-fragments of QKᵀ's A operand (row map, contraction `d` on the spread
            // lane-axis; each `slice(s)` selects d-columns `[s·16, s·16+16)`). QKᵀ = Σ_s mma(K_s, Q_s).
            0 => {
                let mut vecs = Vec::new();
                let mut g = Vec::new();
                for s in 0..self.dfrags {
                    let (v, gg) = self.k_view.slice(s).gather(b, raw);
                    vecs.extend(v);
                    g.extend(gg);
                }
                (vecs, g)
            }
            // V: the `dfrags` output-d fragments of PV's A operand, gathered TRANSPOSED (contraction
            // `kv` on the spread lane-axis — the `mma_atb` orientation the isolation gate proved).
            1 => self.v_view.gather_transposed(b, raw),
            _ => panic!("FA gather: slice ∈ {{0=K, 1=V}}, got {slice}"),
        };
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
    assert!(n.is_multiple_of(EDGE) && n / EDGE >= 2, "N must be a multiple of 16 with ≥2 KV blocks");
    let (q_blk, kv_blk) = (EDGE, EDGE);
    let dfrags = d / EDGE; // head-dim fragment count (QKᵀ K-steps / PV output-d fragments)

    let mut b = Builder::new("tk2_fa_fwd");
    // ABI: output O first, then inputs Q, K, V (all flat [n, d]).
    let o = b.global::<F32>(n * d);
    let q = b.global::<BF16>(n * d);
    let k = b.global::<BF16>(n * d);
    let v = b.global::<BF16>(n * d);

    // One workgroup per Q-block; single 64-lane warp.
    let qwg = b.grid_axis(0, (n / q_blk) as i64);
    let lane = b.block_axis(WARP as i64);

    // ── LDS tiles ([16, d] each) ──
    let k_smem = b.define_local::<BF16>(kv_blk * d);
    let v_smem = b.define_local::<BF16>(kv_blk * d);
    let q_smem = b.define_local::<BF16>(q_blk * d);

    let row_map = FragMap::gfx942_16x16(false); // A operand (contraction on the spread lane-axis)
    let col_map = FragMap::gfx942_16x16(true); // B / C / accumulator operands

    // Movement handles — the SAME `SharedTile`→view/stage machinery matmul uses. K is QKᵀ's A (row
    // map, `d` on spread; gathered per d-slice); Q is QKᵀ's B (col map, `d` on spread; loop-invariant);
    // V is PV's A (col map, gathered TRANSPOSED so `kv` lands on spread — the `mma_atb` relayout).
    let k_tile = SharedTile::new(k_smem, d);
    let v_tile = SharedTile::new(v_smem, d);
    let q_tile = SharedTile::new(q_smem, d);
    let k_view = k_tile.gather_view(row_map, kv_blk / EDGE, None, lane, false); // 1 kv-frag / d-slice
    let v_view = v_tile.gather_view(col_map, dfrags, None, lane, false); // d/16 transposed output frags
    let q_view = q_tile.gather_view(col_map, q_blk / EDGE, None, lane, false); // 1 q-frag / d-slice

    let epl = kv_blk * d / WARP; // vectorised fill elements per lane
    let zero = b.idx_const(0);
    // K/V stream: origin row 0 (single head/batch); the per-block row advance rides `k_base` (below).
    let k_stage = k_tile.stage_view(k, epl, lane, zero, d as i64, Drain::Intrinsic);
    let v_stage = v_tile.stage_view(v, epl, lane, zero, d as i64, Drain::Intrinsic);
    // Q origin row = qwg·q_blk (this workgroup's query rows); staged ONCE (loop-invariant).
    let qblk_c = b.idx_const(q_blk as i64);
    let q_origin = b.idx_mul(qwg, qblk_c);
    let q_stage = q_tile.stage_view(q, epl, lane, q_origin, d as i64, Drain::Intrinsic);

    // ── prologue: stage + commit Q once, gather its `dfrags` d-slices (the loop-invariant QKᵀ B). ──
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
    //    (f32), `m` (f32 running max), `l` (f32 running norm). TEMPORARIES (no seed, produced+consumed
    //    within one KV block): `s` (f32 QKᵀ scores), `p` (BF16 softmax weights — flows through the
    //    channel natively, no cast-in-PV). Only the clusters that touch a slot round-trip it. ──
    let o_frags: Vec<Frag<F32>> = (0..dfrags).map(|_| b.define_frag::<F32>(col_map)).collect();
    let (m_frag, l_frag, s_frag) =
        (b.define_frag::<F32>(col_map), b.define_frag::<F32>(col_map), b.define_frag::<F32>(col_map));
    let p_frag = b.define_frag::<BF16>(col_map);
    let (slot_m, slot_l, slot_s, slot_p) = (dfrags, dfrags + 1, dfrags + 2, dfrags + 3);
    let mut accs: Vec<AccSlot> = o_frags.iter().map(|&f| AccSlot::F32(f)).collect();
    accs.extend([AccSlot::F32(m_frag), AccSlot::F32(l_frag), AccSlot::F32(s_frag), AccSlot::BF16(p_frag)]);
    let mut inited: Vec<Option<Effect>> = o_frags.iter().map(|&f| Some(b.zero_init_frag(f))).collect();
    inited.push(Some(b.const_init_frag(m_frag, f32::NEG_INFINITY))); // running max seed = −∞
    inited.push(Some(b.zero_init_frag(l_frag))); // running norm seed = 0
    inited.push(None); // s: temporary (QKᵀ produces it fresh each block)
    inited.push(None); // p: temporary (softmax produces it fresh each block)

    // Per-cluster read/write slot sets (asymmetric — the point of §3.2): QKᵀ writes only `s`; softmax
    // reads {s,m,l,o} writes {m,l,p,o}; PV reads {p,o} writes {o}. `m`/`l` are untouched by QKᵀ+PV and
    // `s`/`p` never round-trip through a cluster that doesn't use them.
    let o_idx: Vec<usize> = (0..dfrags).collect();
    let sm_reads: Vec<usize> = [slot_s, slot_m, slot_l].into_iter().chain(0..dfrags).collect();
    let sm_writes: Vec<usize> = [slot_m, slot_l, slot_p].into_iter().chain(0..dfrags).collect();
    let pv_reads: Vec<usize> = std::iter::once(slot_p).chain(0..dfrags).collect();

    // Softmax scale folded into the QKᵀ scores: exp2(score·log2(e)/√d) == exp(score/√d).
    let scale = std::f32::consts::LOG2_E / (d as f32).sqrt();
    let scale_bcast = {
        let s = b.f32(scale);
        let cs: Vec<Val<F32>> = (0..EPT).map(|_| s).collect();
        b.vec_build(&cs)
    };
    let zero_c = {
        // QKᵀ's C = 0 (att re-zeroed every KV block — `s` is a temporary, not read from the carry).
        let zs: Vec<Val<F32>> = (0..EPT).map(|_| b.f32(0.0)).collect();
        b.vec_build(&zs)
    };

    // ── the three compute bodies, each declaring ONLY the slots it touches ──
    // QKᵀ: att = K·Qᵀ (contraction over `d` = Σ over `dfrags` `mma` K-steps). reads nothing (re-zeros
    // its output); writes only `s`. Operand = gathered K.
    let qk = Compute::<FaHooks>::new(
        0,
        vec![],
        vec![slot_s],
        move |b: &mut Builder, op: Option<&FaOp>, _reads: &[SlotVal]| {
            let k_frags = op.expect("QKᵀ consumes gathered K");
            let mut att = zero_c;
            for s in 0..dfrags {
                att = b.mma(k_frags[s], q_frags[s], att, EPT); // accumulate the d-slice K·Qᵀ
            }
            vec![SlotVal::F32(att)]
        },
    );

    // Online softmax (operand = None): reads {s, m, l, o_*}, writes {m, l, p, o_*}. The running-max
    // rescale + exp2 + the two `ds_bpermute` column reductions — barrier-free. Produces `p` as BF16
    // (the cast lives HERE now, so PV reads a native bf16 operand) and rescales every `o_df` by corr.
    let softmax = Compute::<FaHooks>::new(
        None,
        sm_reads,
        sm_writes,
        move |b: &mut Builder, _op: Option<&FaOp>, reads: &[SlotVal]| {
            let (s_raw, max_old, norm_old) = (reads[0].f32(), reads[1].f32(), reads[2].f32());
            let s = b.mul(s_raw, scale_bcast); // scaled scores
            let m = b.frag_col_reduce(s, lane, max_old, false); // running max (per q, broadcast)
            let corr = b.sub(max_old, m);
            let scale_f = b.exp2(corr); // exp2(max_old − max_new)
            let norm2 = b.mul(norm_old, scale_f); // rescale running norm
            let sm = b.sub(s, m);
            let p = b.exp2(sm); // softmax weights P (f32)
            let norm3 = b.frag_col_reduce(p, lane, norm2, true); // norm += Σ_kv P
            let p_bf16 = b.cast_vec_bf16(p); // cast into the channel (PV reads bf16)
            let mut out = vec![SlotVal::F32(m), SlotVal::F32(norm3), SlotVal::BF16(p_bf16)];
            for i in 0..dfrags {
                out.push(SlotVal::F32(b.mul(reads[3 + i].f32(), scale_f))); // O *= corr
            }
            out
        },
    );

    // P·V accumulate (`mma_atb` over `kv`): reads {p, o_*}, writes {o_*}. Operand = the `dfrags`
    // transposed V fragments (`kv` on spread). `P` arrives as a native bf16 channel value (no local
    // cast); V feeds PV's A operand transposed. Each `o_df` gets O[d in df, q] += Σ_kv V[kv,d]·P[kv,q].
    let pv = Compute::<FaHooks>::new(
        1,
        pv_reads,
        o_idx.clone(),
        move |b: &mut Builder, op: Option<&FaOp>, reads: &[SlotVal]| {
            let v_frags = op.expect("PV consumes gathered V");
            let p = reads[0].bf16(); // native bf16 P from the channel
            (0..dfrags).map(|df| SlotVal::F32(b.mma(v_frags[df], p, reads[1 + df].f32(), EPT))).collect()
        },
    );

    let hooks = FaHooks { k_view, v_view, k_stage, v_stage, dfrags };
    let acc_final = pipeline(
        &mut b,
        n / kv_blk, // nblocks (streaming over KV)
        kv_blk * d, // k_step: the FLAT per-block advance (kv_blk rows · d)
        2,          // ksteps: gather slices (K, V)
        &accs,
        &inited,
        None,  // warp_row: no wave-phase ping-pong (single warp)
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

    // ── post-loop: normalize O = o / norm (per q, broadcast across d) and transpose-scatter to O. ──
    let norm_vec = b.load_frag_vec(acc_final[slot_l].f32());
    let recip_norm = b.recip(norm_vec);
    let d_c = b.idx_const(d as i64);
    let mut roots = Vec::new();
    for df in 0..dfrags {
        let o_vec = b.load_frag_vec(acc_final[df].f32());
        let o_norm = b.mul(o_vec, recip_norm);
        for inner in 0..EPT {
            let inner_c = b.idx_const(inner as i64);
            // o_df is Col-map [d = spread = row, q = flat = col]; scatter to O[q_global, d_global].
            let (row, col) = b.lane_rc(col_map, lane, inner_c);
            let q_global = b.idx_add(q_origin, col);
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
