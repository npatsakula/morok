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
use crate::partition::RowPartition;
use crate::pipeline::{
    AccSlot, BlockCounter, CommitDrain, Compute, Hooks, Init, Mem, Sched, SlotSet, SlotVal, pipeline,
};
use crate::shape::{Mfma16x16x16Bf16, MfmaShape};
use crate::tile::{ARow, BCol, Plain, Xor};
use crate::tile_move::{commit, commit_run, gather, gather_run, prefetch};

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

/// FA's [`Hooks`] — the ONLY kernel-specific movement. It rides the SAME [`crate::tile_move`]
/// handles matmul does (proving the movement layer is not GEMM-bound); the compute math rides the
/// [`Compute`] bodies below, so `Hooks` grows no per-cluster compute method. `PREFETCH_TILES = 2`
/// (K, V) as in matmul (A, B), but the two tiles are the two *operands of two different matmuls*
/// (K feeds QKᵀ, V feeds PV) rather than the A/B of one — the first shape strain (see report).
struct FaHooks {
    /// The shared K/V LDS tiles `[kv_blk, d]` and their global sources — the raw handles the
    /// `tile_move` prefetch/commit/gather forwards address. They REPLACE the pre-built `LdsView`/
    /// `LdsStage` (which the forwards now rebuild internally from these + the params below — the
    /// `SharedTile`/`gather_view`/`slice` builders emit no IR, so the emission is byte-identical).
    k_smem: Lds<BF16>,
    v_smem: Lds<BF16>,
    k: Buf<BF16>,
    v: Buf<BF16>,
    /// The collaborative 512-thread fill addressing (prefetch/commit): `tid` = fill thread id,
    /// `bh_row` = this (b,h)'s row origin, `epl_kv` = elements per lane. `d` = the LDS tile inner width
    /// / global row stride (and V's transposed-gather `tile_cols`); K's gather `tile_rows` is `kvf·16`.
    tid: Idx,
    bh_row: Idx,
    epl_kv: usize,
    /// The per-warp gather lane (the gather/`ds_bpermute` stays within each 64-lane warp).
    wlane: Idx,
    d: usize,
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
        let (d, s) = (self.d, self.d as i64);
        let loaded = match tile {
            0 => {
                reg.k = prefetch(b, self.k_smem, d, self.k, s, self.epl_kv, self.tid, self.bh_row, k_base, order);
                &reg.k
            }
            1 => {
                reg.v = prefetch(b, self.v_smem, d, self.v, s, self.epl_kv, self.tid, self.bh_row, k_base, order);
                &reg.v
            }
            _ => panic!("FA prefetch: tile ∈ {{0=K, 1=V}}, got {tile}"),
        };
        let anchors = loaded.iter().map(|v| v.id).collect();
        (reg, anchors)
    }

    fn commit(&mut self, b: &mut Builder, _k_base: Idx, reg: &FaFill, war: &[TileId]) -> Vec<Effect> {
        // Intrinsic commit (an `s_barrier` auto-drains the `ds_write` `lgkmcnt(0)`) — `tile_move::commit`
        // pins `Drain::Intrinsic`; the asm/deferred-drain machinery is a matmul-perf concern, orthogonal.
        let (d, s) = (self.d, self.d as i64);
        let fk = commit(b, self.k_smem, d, self.k, s, self.epl_kv, self.tid, self.bh_row, &reg.k, war);
        let fv = commit(b, self.v_smem, d, self.v, s, self.epl_kv, self.tid, self.bh_row, &reg.v, war);
        fk.into_iter().chain(fv).collect()
    }

    fn gather(
        &mut self,
        b: &mut Builder,
        slice: usize,
        _block: BlockCounter,
        raw: &[TileId],
    ) -> (FaOp, Vec<TileId>, TileId) {
        let mut vecs = Vec::new();
        let mut g = Vec::new();
        match slice {
            // K: QKᵀ's A operand as `dfrags · kvf` fragments (row map, contraction `d` on the spread
            // lane-axis; `slice(s)` selects d-columns `[s·16, s·16+16)`, the view's `n_frags = kvf`
            // stacking the KV-block rows). Indexed `[s·kvf + kf]`; QKᵀ = Σ_s mma(K[s,kf], Q_s) per `kf`.
            0 => {
                // `ARow` derives the row map + straight gather; `tile_rows = kvf·16` gives `n_frags = kvf`.
                for s in 0..self.dfrags {
                    let (v, gg) = gather::<BF16, ARow, Mfma16x16x16Bf16>(
                        b,
                        self.k_smem,
                        self.d,
                        self.kvf * EDGE,
                        EDGE,
                        None,
                        self.wlane,
                        raw,
                        s,
                        false,
                    ); // kvf kv-fragments for d-slice s
                    vecs.extend(v);
                    g.extend(gg);
                }
            }
            // V: PV's A operand as `kvf · dfrags` TRANSPOSED fragments (contraction `kv` on the spread
            // lane-axis — the `mma_atb` orientation). `slice(kf)` sets the kv-row base `kf·16`, then a
            // transposed gather yields the `dfrags` output-d fragments. Indexed `[kf·dfrags + df]`.
            1 => {
                // `BCol` derives the col map + transposed gather; `tile_cols = d` gives `n_frags = dfrags`.
                for kf in 0..self.kvf {
                    let (v, gg) = gather::<BF16, BCol, Mfma16x16x16Bf16>(
                        b,
                        self.v_smem,
                        self.d,
                        EDGE,
                        self.d,
                        None,
                        self.wlane,
                        raw,
                        kf,
                        false,
                    );
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

    let col_map = Mfma16x16x16Bf16::c_map(); // B / C / accumulator operands (the Col map)

    let epl_kv = kv_blk * d / nthreads; // 512-thread collaborative fill
    // Movement rides the `tile_move` vocabulary: `FaHooks` carries the raw K/V LDS tiles, their global
    // sources, and the fill/gather addressing (`tid`/`bh_row`/`epl_kv`/`wlane`/`d`), and
    // `tile_move::{prefetch, commit, gather}` rebuild the `LdsStage`/`LdsView` internally — byte-identical
    // to a pre-built handle (the builders emit no IR). K = QKᵀ's A (row map, straight gather, `kvf`
    // KV-frags per d-slice); V = PV's A (col map, transposed gather, `dfrags` output frags per kv-slice);
    // K/V shared (no warp off); staged by ALL 512 threads (`tid`), origin = this (b,h)'s row base
    // `bh_row`, the per-block advance riding `k_base`. Q has no LDS view — it is register-resident.
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
        move |b: &mut Builder, op: Option<&FaOp>, _reads: &[SlotVal], _blk: BlockCounter| {
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
        move |b: &mut Builder, _op: Option<&FaOp>, reads: &[SlotVal], _blk: BlockCounter| {
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
        move |b: &mut Builder, op: Option<&FaOp>, reads: &[SlotVal], _blk: BlockCounter| {
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

    let hooks = FaHooks { k_smem, v_smem, k, v, tid, bh_row, epl_kv, wlane, d, dfrags, kvf };
    let acc_final = pipeline(
        &mut b,
        n / kv_blk, // nblocks (streaming over KV)
        kv_blk * d, // k_step: the FLAT per-block advance (kv_blk rows · d)
        2,          // ksteps: gather slices (K, V)
        &accs,
        &inited,
        None, // warp_row: FA wants NO wave-phase ping-pong — warps run disjoint Q rows, so there is
        // no co-resident load/compute pair to steer (the brief: keep it None)
        Sched {
            asm_gather: false,
            resident: false,
            commit_drain: CommitDrain::IntrinsicAuto,
            bare_seals: false,
            pin_mfma: false,
        },
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
/// The transposed-V LDS row padding. Pitch = `KV_BLK_32 + VT_PAD` must keep the per-lane b64 V read
/// conflict-free: consecutive lanes stride `pitch/2` dwords, so only a pitch whose dword-stride has
/// `gcd(·, 32) = 2` (not 4) spreads 16 lanes across 16 distinct LDS banks — the same spread the XOR
/// swizzle gives K. `VT_PAD = 8` → pitch 40 → stride 20 → gcd 4 → 8 banks → ~2-way conflict (measured
/// PMC bankconf ≈ 1.6); `VT_PAD = 4` → pitch 36 → stride 18 → gcd 2 → conflict-free. Still mult-of-4 for
/// b64 alignment. (An un-padded `[d, kv]` transpose regresses — proven by `v_transpose_probe`.)
const VT_PAD: usize = 4;

/// The **softmax-under-MFMA interleave ratios** (plan §2.5/§5-lever-1, HipKittens' `sched_barrier_pairs`
/// counts). `interleave_exp<PAIRS,CNT>` folds the online `exp2` under the block's MFMAs; `interleave_valu`
/// folds the reduction VALU under the P·V MFMAs. These are the ONLY perf-tuning knobs (step-6 sweep); the
/// values here are HK's starting cadence, and the hints are numerics-INERT (verified: the gate is unchanged).
/// `PAIRS = 16` covers the full 16-MFMA count of each matmul at d=128 (QKᵀ = d/8, P·V = dtiles·ksl =
/// d/8), the cadence device-swept as the robust optimum (Lever 1): +2.5% d128 / +4.2% d64 over a
/// no-hint build. A fixed 16 also beats the "exact" d64 count (8), which hits a scheduler cliff (~-8%
/// on d64) — measured, not assumed. `CNT` (exp=3, valu=5) is HK's canonical ratio; the sweep confirmed
/// it is flat across 2–6, so the HK values stand.
const FA_EXP_PAIRS: u32 = 16;
const FA_EXP_CNT: u32 = 3;
const FA_VALU_PAIRS: u32 = 16;
const FA_VALU_CNT: u32 = 5;

/// One K/V-slice's gathered A-operands — the FA-32 [`Hooks::Op`]. Slice 0 = K (`dslices` QKᵀ A-operands,
/// contraction `d` over `K = 8`); slice 1 = V (the `dtiles·ksl` transposed-V A-operands, indexed
/// `[dt·ksl + s]`, contraction `kv` over `K = 8`). Both `Vec<Val<BF16>>` (`EPT_A = 4` each).
type Fa32Op = Vec<Val<BF16>>;

/// The register-staged fill carried prefetch→commit: block k+1's K and V chunks in VGPRs (the
/// collaborative 256-thread fill's per-thread `epl` elements as `gvec`-wide (b128/b64) load chunks).
struct Fa32Fill {
    k: Vec<Val<BF16>>,
    v: Vec<Val<BF16>>,
}

/// FA-32's [`Hooks`] — the 32×32×8 movement (natural-K LDS + padded-transposed-V LDS). Rides the SAME
/// ClusterCx pipeline the 16×16 [`flash_attention_fwd`] drives, but with the 32×32×8 fragment addressing:
/// each A-operand is a contiguous `EPT_A = 4` `ds_read_b64` run (no register transpose — V's transpose is
/// done write-side into LDS) rather than the movement layer's 16×16 [`LdsView`] gather. The global fill
/// reads are **b128** coalesced (`load_vec_after`, Phase-2c); the LDS stores stay scalar (the coalesced
/// write is a fill refinement the barrier-bound kernel doesn't need). Gathered operands round-trip through
/// a fragment so the WAR barrier consumes a proper store token (the [`Hooks::gather`] contract) — SROA
/// elides the store/load, exactly as the 16×16 FA.
struct Fa32Hooks {
    k: Buf<BF16>,
    v: Buf<BF16>,
    k_lds: Lds<BF16>,
    vt_lds: Lds<BF16>,
    /// Lever-3 LDS double-buffer: when set, `k_lds`/`vt_lds` are allocated 2× and a runtime parity
    /// offset (`(block±1)%2 · tile`) selects the read/write half so commit(k+1) writes the OTHER buffer
    /// than gather(k) reads — removing the WAR hazard and letting the two staggered warp-groups read
    /// non-overwritten buffers. Off ⇒ single tile, parity offset always 0 (bit-identical to pre-lever).
    double_buf: bool,
    /// `bh_row · d` — this workgroup's (b,h) slice flat base (folded into every fill global offset so the
    /// pipeline's FLAT per-block `k_base` lands at the right `[bh, n, d]` row).
    bh_row_d: Idx,
    /// The collaborative fill thread id (all `nthreads`) and its per-thread element run `epl`.
    tid: Idx,
    epl: usize,
    /// The per-warp gather axis parts: `q_in = wlane % 32` (kv/d-in-tile), `half_off = (wlane/32)·4` —
    /// i.e. `lane_rc(S::a_map(), wlane, 0)`, the `(row, col)` the derived [`gather_run`] addresses with.
    /// The fragment count / per-fragment tile-offsets are DERIVED from the tile dims (`d`/`pitch` +
    /// `KV_BLK_32`/`S::M`), so `dslices`/`dtiles`/`ksl` no longer ride the struct.
    q_in: Idx,
    half_off: Idx,
    d: usize,
    pitch: usize,
}

impl Fa32Hooks {
    /// The runtime parity offset (in elements) into the double-sized K / Vt LDS tiles for `block`:
    /// `(block % 2) · tile_size`. Both zero when double-buffering is off (so the single-tile addressing
    /// is bit-identical to the pre-lever kernel).
    fn parity_off(&self, b: &mut Builder, block: Idx) -> (Idx, Idx) {
        if !self.double_buf {
            let z = b.idx_const(0);
            return (z, z);
        }
        let two = b.idx_const(2);
        let par = b.idx_mod(block, two);
        let k_tile = b.idx_const((KV_BLK_32 * self.d) as i64);
        let vt_tile = b.idx_const((self.d * self.pitch) as i64);
        (b.idx_mul(par, k_tile), b.idx_mul(par, vt_tile))
    }
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
        order: &[TileId],
    ) -> (Fa32Fill, Vec<TileId>) {
        // global→VGPR: this thread's COALESCED b128 (or b64) vector loads at flat `bh_row·d + k_base +
        // tid·epl` (block `k`'s first element is `bh_row·d + k_base`). `gvec = 8` (b128) when `epl` tiles it,
        // else 4 (b64) — the `order` edge pins the loads into this cluster (the ClusterCx load-pin anchors
        // on the returned values). Each chunk stays within one `kv` row (`gvec ≤ d`), consumed by `commit`.
        let mut reg = prev.unwrap_or(Fa32Fill { k: Vec::new(), v: Vec::new() });
        let gvec = if self.epl.is_multiple_of(8) { 8 } else { 4 };
        let epl_c = b.idx_const(self.epl as i64);
        let lane_epl = b.idx_mul(self.tid, epl_c);
        let flat_base = b.idx_add(self.bh_row_d, k_base);
        let flat_base = b.idx_add(flat_base, lane_epl);
        let buf = match tile {
            0 => self.k,
            1 => self.v,
            _ => panic!("FA-32 prefetch: tile ∈ {{0=K, 1=V}}, got {tile}"),
        };
        let loaded: Vec<Val<BF16>> = (0..self.epl / gvec)
            .map(|cg| {
                let off = offset_by(b, flat_base, cg * gvec);
                b.load_vec_after(buf, off, gvec, order)
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

    fn commit(&mut self, b: &mut Builder, k_base: Idx, reg: &Fa32Fill, war: &[TileId]) -> Vec<Effect> {
        // VGPR→LDS behind the WAR barrier via the derived fused K/V commit. The block-dependent runtime
        // params stay in the hook: the WAR `lds_after` re-bind and the Lever-3 double-buffer parity —
        // commit writes block k+1 → parity `(k+1)%2` (`k_base = (k+1)·k_step`, so the committed block index
        // is `k_base / k_step`; its parity offset selects the write half, 0 when single-buffered). The
        // K-natural-swizzled + V-transposed-padded addressing (and the SCALAR `store_lds` — a coalesced
        // write regressed the barrier-bound kernel) is derived by `commit_run` from the K (`Xor`) / V
        // (`Plain`, padded `pitch`) tile types.
        let k_lds = if war.is_empty() { self.k_lds } else { b.lds_after(self.k_lds, war) };
        let vt_lds = if war.is_empty() { self.vt_lds } else { b.lds_after(self.vt_lds, war) };
        let (k_off, vt_off) = {
            let kstep = b.idx_const((KV_BLK_32 * self.d) as i64);
            let blk1 = b.idx_div(k_base, kstep);
            self.parity_off(b, blk1)
        };
        commit_run::<BF16, Xor, Plain>(
            b, k_lds, self.d, vt_lds, self.pitch, &reg.k, &reg.v, self.epl, self.tid, k_off, vt_off,
        )
    }

    fn gather(
        &mut self,
        b: &mut Builder,
        slice: usize,
        block: BlockCounter,
        raw: &[TileId],
    ) -> (Fa32Op, Vec<TileId>, TileId) {
        use crate::shape::Mfma32x32x8Bf16 as S;
        // Lever-3: gather reads block k → parity `k%2` selects the read half (0 when single-buffered) —
        // the one genuinely block-dependent runtime param, so it stays in the hook.
        let (k_off, vt_off) = {
            let blk = block.idx(b);
            self.parity_off(b, blk)
        };
        // The contiguous-run addressing is DERIVED by `gather_run` from the ARow role + the tile types: K is
        // the natural `[kv=32, d]` `Xor`-swizzled tile (`dslices = d/8` col-fragments), V the transposed
        // `[d, kv=32]` `Plain` padded-pitch tile (`dtiles·ksl` fragments). `q_in`/`half_off` = the lane
        // partition `lane_rc(S::a_map(), wlane, 0)`; the 4-run start stays 4-aligned so the swizzled K read
        // remains a contiguous `ds_read_b64`.
        let (vecs, gathers) = match slice {
            0 => gather_run::<BF16, Xor, S>(b, self.k_lds, self.d, S::M, self.d, self.q_in, self.half_off, k_off, raw),
            1 => gather_run::<BF16, Plain, S>(
                b,
                self.vt_lds,
                self.pitch,
                self.d,
                KV_BLK_32,
                self.q_in,
                self.half_off,
                vt_off,
                raw,
            ),
            _ => panic!("FA-32 gather: slice ∈ {{0=K, 1=V}}, got {slice}"),
        };
        let anchor = vecs[0].id;
        (vecs, gathers, anchor)
    }
}

/// **Ragged-tail KV mask** (§Step-B): add `−∞` to every score whose global KV index
/// `block·KV_BLK_32 + kv_in_tile` is ≥ `n`, so a partial last KV block's out-of-range keys `exp→0`
/// and contribute to NEITHER the running max NOR the softmax sum (masking `P` alone would let an
/// out-of-range score pollute the online max and wipe the carried accumulator — the mask MUST precede
/// the max). `s` is the scaled `EPT_C`-wide score accumulator (`c_map`, so `acc_rc` row = kv-in-tile);
/// the additive mask is a per-element [`Builder::select_lt`] built from the routed [`BlockCounter`]. A
/// no-op for a tile-exact `n`, so [`flash_attention_fwd_32`] emits it only when the sequence is ragged.
fn mask_ragged_kv(b: &mut Builder, s: Val<F32>, blk: BlockCounter, wlane: Idx, n: usize) -> Val<F32> {
    use crate::shape::Mfma32x32x8Bf16 as S;
    let dist = S::acc_dist();
    let base_kv = {
        let block = blk.idx(b);
        let kvb = b.idx_const(KV_BLK_32 as i64);
        b.idx_mul(block, kvb) // block · KV_BLK_32
    };
    let n_c = b.idx_const(n as i64);
    let zero = b.f32(0.0);
    let ninf = b.f32(f32::NEG_INFINITY);
    let els: Vec<Val<F32>> = (0..S::EPT_C)
        .map(|i| {
            let (row, _col) = b.acc_rc(dist, wlane, i); // row = kv-in-tile
            let gkv = b.idx_add(base_kv, row);
            b.select_lt(gkv, n_c, zero, ninf) // 0 if global_kv < n, else −∞
        })
        .collect();
    let mask = b.vec_build(&els);
    b.add(s, mask)
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
    // Ragged tail supported: `n` need NOT be a multiple of the Q block or the KV block. The Q grid and
    // the KV stream both round UP (`div_ceil`); the last KV block is masked past the true `n` (the score
    // mask below), and partial Q rows are computed-but-not-scattered-to-valid-output. PRECONDITION (as
    // everywhere in tk2 — the fill's raw global loads are unbounded): the caller provisions Q/K/V/O
    // buffers covering `⌈n/tile⌉·tile` rows per (b,h) slice so the tile-covering fill + scatter stay
    // in-buffer (the device gate pads the tensors; a tile-exact `n` needs no padding).
    let nblocks = n.div_ceil(KV_BLK_32);
    assert!(nblocks >= 2, "n must give ≥2 KV blocks (the rolled pipeline needs nblocks ≥ 2)");
    let ragged = !n.is_multiple_of(KV_BLK_32); // a partial last KV block ⇒ emit the ragged-tail score mask
    let dslices = d / k8; // QKᵀ K-steps (contract d over K=8)
    let dtiles = d / m32; // PV output d-tiles (32 each)
    let ksl = KV_BLK_32 / k8; // 4 PV K-slices per KV block

    let mut b = Builder::new("tk2_fa_fwd_32");
    // ABI: O[bh·n, d] then Q, K, V — each `[bh, n, d]` row-major.
    let o = b.global::<F32>(bh * n * d);
    let q = b.global::<BF16>(bh * n * d);
    let k = b.global::<BF16>(bh * n * d);
    let v = b.global::<BF16>(bh * n * d);

    // Grid = bh × ⌈n/q_blk⌉ workgroups over ONE flat axis; a `RowPartition` (Phase 2) derives the grid and
    // decodes `wgid → pid = (slice = bh_idx, tile = qwg, row_origin = q_origin)`. `row_origin =
    // bh_idx·n + qwg·q_blk` is the tile's global Q-row origin; `pid.slice·n` is this (b,h) slice's row base.
    // The Q grid rounds UP (div_ceil) so a ragged `n` is fully covered — the partial last workgroup's
    // excess Q rows compute into the (caller-provisioned) buffer tail, not the compared output.
    let part = RowPartition { slices: bh, rows_per_slice: n, tile_rows: q_blk };
    let wgid = b.grid_axis(0, part.grid_size() as i64);
    let pid = part.decode(&mut b, wgid);

    let tid = b.block_axis(nthreads as i64);
    let warp_c = b.idx_const(WARP as i64);
    let m32_c = b.idx_const(m32 as i64);
    let warp = b.idx_div(tid, warp_c);
    let wlane = b.idx_mod(tid, warp_c);
    let warp_qoff = b.idx_mul(warp, m32_c); // this warp's 32-row Q offset in the [q_blk, d] block
    // Lever-3 LDS double-buffer: commit(k+1) writes the parity half gather(k) is NOT reading, so the
    // per-iteration WAR `s_barrier` is dropped and the LDS write overlaps compute. ENABLED for the
    // occupancy-1 regime (`d ≥ 128`): there the workgroup is register-pressure-bound to 1 wave/SIMD, so
    // hiding the LDS-write latency instruction-locally is a net win (device-measured +5% at S2048 d128).
    // DISABLED for the occupancy-2 small-`d` shapes (`d = 64`): there a 2nd resident wave already hides
    // that latency, so the double-buffer's extra VGPRs + parity math only regress (~-9%, measured). No
    // ping-pong stagger: device-measured a NET LOSS here — at occupancy 1 the within-workgroup two-group
    // offset gives no intra-SIMD MFMA overlap (1 wave/SIMD), so its added compute-cluster barriers don't
    // pay for themselves. (Reaching aiter's stagger win needs occupancy ≥2 first — a VGPR-O redesign.)
    let double_buf = d >= 128;
    let warp_row: Option<Idx> = None;

    // This warp's global Q-row origin = the tile's row origin + this warp's 32-row Q offset.
    let q_row_base = b.idx_add(pid.row_origin, warp_qoff);

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

    // Shared LDS: K natural [kv,d], V transposed padded [d, pitch]. Doubled (parity-swapped) under the
    // Lever-3 double-buffer so commit(k+1) and gather(k) touch disjoint halves.
    let nbuf = if double_buf { 2 } else { 1 };
    let k_lds = b.define_local::<BF16>(nbuf * KV_BLK_32 * d);
    let vt_lds = b.define_local::<BF16>(nbuf * d * pitch);
    let epl = KV_BLK_32 * d / nthreads; // collaborative fill, per thread

    // Softmax scale folded into exp2: exp2(score·log2(e)/√d) == exp(score/√d). 16-wide broadcast.
    let scale = std::f32::consts::LOG2_E / (d as f32).sqrt();
    let scale_bcast = {
        let c = b.f32(scale);
        crate::tile_ops::splat::<S>(&mut b, c)
    };

    // ── the heterogeneous slot set (DESIGN §3.2). CARRIED: `o_0..o_{dtiles-1}` (16-wide f32 PV acc),
    //    `m` (running max), `l` (running norm). TEMPORARIES (produced+consumed within one KV block):
    //    `s` (QKᵀ scores, 16-wide f32), `p` (softmax weights, 16-wide f32 → v_perm-packed in PV). All f32
    //    on the `EPT_C = 16` accumulator (`c_map`); each warp keeps its OWN o/m/l over its 32 Q rows. ──
    // Declared, not hand-numbered (Phase 3): `SlotSet` allocates each fragment as it is declared but
    // DEFERS every init to `finish`, so the emission stays "all define_frags, then all inits" (o×dtiles,
    // m, l, s, p → zero o×dtiles, const m, zero l) — byte-identical to the old hand-written slot table.
    let mut slots = SlotSet::new();
    let o_idx = slots.carried_group(&mut b, dtiles, S::c_map(), Init::Zero);
    let slot_m = slots.carried(&mut b, S::c_map(), Init::Const(f32::NEG_INFINITY)); // running max seed = −∞
    let slot_l = slots.carried(&mut b, S::c_map(), Init::Zero); // running norm seed = 0
    let slot_s = slots.temp(&mut b, S::c_map()); // QKᵀ scores (temporary)
    let slot_p = slots.temp(&mut b, S::c_map()); // softmax weights (temporary)
    let (accs, inited) = slots.finish(&mut b);

    // Per-cluster read/write slot sets (asymmetric — the §3.2 point). QKᵀ writes only `s`; softmax reads
    // {s,m,l,o_*} writes {m,l,p,o_*}; PV reads {p,o_*} writes {o_*}. `o_idx` is the carried_group's
    // returned slot indices (0..dtiles).
    let sm_reads: Vec<usize> = [slot_s, slot_m, slot_l].into_iter().chain(0..dtiles).collect();
    let sm_writes: Vec<usize> = [slot_m, slot_l, slot_p].into_iter().chain(0..dtiles).collect();
    let pv_reads: Vec<usize> = [slot_p].into_iter().chain(0..dtiles).collect();

    // ── the three compute bodies, each declaring ONLY the slots it touches ──
    // QKᵀ: S[kv,q] = Σ_ki mma(K_ki, Q_ki) (K A-operand, Q B-operand). reads nothing (re-zeros); writes `s`.
    // INTRINSIC MFMA (the production fast path — the asm `sideeffect` form is a dead end here: it is
    // OPAQUE to the AMDGPU GCNHazardRecognizer, so it emits NONE of the mandatory 32×32×8 `s_nop`s and
    // a VALU-adjacent accumulator is read before the MFMA result lands → silent miscompile, device-proven
    // — the PV NaN was exactly this, and QKᵀ-asm only survived by luck of instruction spacing). The
    // softmax-under-MFMA interleave (`sched_group_barrier`) interleaves fine with the intrinsic.
    let qk = Compute::<Fa32Hooks>::new(
        0,
        vec![],
        vec![slot_s],
        move |b: &mut Builder, op: Option<&Fa32Op>, _reads: &[SlotVal], _blk: BlockCounter| {
            let k_frags = op.expect("QKᵀ consumes gathered K");
            let z = b.f32(0.0);
            let mut s_acc = crate::tile_ops::splat::<S>(b, z);
            for ki in 0..dslices {
                s_acc = crate::tile_ops::mma::<S>(b, k_frags[ki], q_frags[ki], s_acc);
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
        move |b: &mut Builder, _op: Option<&Fa32Op>, reads: &[SlotVal], blk: BlockCounter| {
            let (s_acc, m_run, l_run) = (reads[0].f32(), reads[1].f32(), reads[2].f32());
            let s_scaled = b.mul(s_acc, scale_bcast);
            // Ragged-tail mask: force scores of keys past the true `n` to −∞ BEFORE the max/exp, so they
            // drop out of both the online max and the sum. Emitted only when `n` is not a KV-block
            // multiple; the routed `blk` counter makes it a runtime no-op on every full block.
            let s_scaled = if ragged { mask_ragged_kv(b, s_scaled, blk, wlane, n) } else { s_scaled };
            let m_new = crate::tile_ops::row_reduce::<S>(b, s_scaled, wlane, m_run, false);
            let corr = {
                let diff = b.sub(m_run, m_new);
                b.exp2(diff) // exp2(max_old − max_new)
            };
            let l_resc = b.mul(l_run, corr);
            let p = {
                let sh = b.sub(s_scaled, m_new);
                b.exp2(sh) // softmax weights P (f32, 16-wide)
            };
            let l_new = crate::tile_ops::row_reduce::<S>(b, p, wlane, l_resc, true);
            // Declarative softmax-under-MFMA (plan §2.5): fold the online-exp2 under the block's MFMAs —
            // `interleave_exp<pairs, exp>` in SyncID group 1. The hint emits NO instruction; it is kept
            // live by routing it into the carried `p` value (`val_after`), so it survives DCE and sits in
            // the wall-free compute region. Ratio is a perf knob (step-6 tuning); the mechanism is here.
            let p = match b.interleave_exp(FA_EXP_PAIRS, FA_EXP_CNT, 1, &[s_scaled.id]) {
                Some(h) => b.val_after(p, &[h.dep()]),
                None => p,
            };
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
        move |b: &mut Builder, op: Option<&Fa32Op>, reads: &[SlotVal], _blk: BlockCounter| {
            let v_frags = op.expect("PV consumes gathered V");
            let b_ops = crate::tile_ops::relayout::<S>(b, reads[0].f32());
            let mut anchor = None;
            let mut out: Vec<SlotVal> = (0..dtiles)
                .map(|dt| {
                    // O accumulator uses the INTRINSIC MFMA. The asm `sideeffect` form is opaque to the
                    // AMDGPU GCNHazardRecognizer, so it emits none of the mandatory 32×32×8 hazard `s_nop`s;
                    // the loop-carried O is VALU-adjacent (softmax `O*=corr` writes it, the epilogue `O/=l`
                    // reads it), so both hazards fire → device-proven NaN at d64 (0 spill, NOT SROA/pressure;
                    // `opt -O3` gives byte-identical <16×f32> phis either way). VGPR-resident O (aiter's
                    // 0-AGPR) additionally overflows the budget at d128 (260 VGPR). O-on-intrinsic is thus a
                    // correctness + budget requirement, not avoidance; the interleave hints below interleave
                    // fine with the intrinsic MFMA (it is schedulable, unlike the pinned asm form).
                    let mut o = reads[1 + dt].f32();
                    for s in 0..ksl {
                        o = crate::tile_ops::mma::<S>(b, v_frags[dt * ksl + s], b_ops[s], o);
                    }
                    anchor.get_or_insert(o.id);
                    SlotVal::F32(o)
                })
                .collect();
            // Declarative reduction-under-MFMA: fold the softmax-max/sum VALU under the P·V MFMAs —
            // `interleave_valu<pairs, valu>` in SyncID group 2, kept live by routing into O[0].
            if let (Some(a), Some(SlotVal::F32(o0))) = (anchor, out.first().copied())
                && let Some(h) = b.interleave_valu(FA_VALU_PAIRS, FA_VALU_CNT, 2, &[a])
            {
                out[0] = SlotVal::F32(b.val_after(o0, &[h.dep()]));
            }
            out
        },
    );

    // This (b,h) slice's flat base (rows·d). `bh_row = pid.slice·n` reuses the decode's interned nodes.
    let n_c = b.idx_const(n as i64);
    let bh_row = b.idx_mul(pid.slice, n_c);
    let bh_row_d = b.idx_mul(bh_row, d_c);
    let hooks = Fa32Hooks { k, v, k_lds, vt_lds, double_buf, bh_row_d, tid, epl, q_in, half_off, d, pitch };
    let acc_final = pipeline(
        &mut b,
        nblocks,       // nblocks (streaming over KV; ⌈n/kv_blk⌉ — the last block may be ragged)
        KV_BLK_32 * d, // k_step: the FLAT per-block advance (kv_blk rows · d)
        2,             // ksteps: gather slices (K, V)
        &accs,
        &inited,
        warp_row, // Lever-2: Some(warp/2) enables the two-group phase stagger (env FA_STAGGER); None = off
        Sched {
            asm_gather: false,
            resident: false,
            // Lever-3: double-buffered ⇒ drop the WAR seal (commit/gather touch disjoint parity halves).
            commit_drain: if double_buf { CommitDrain::IntrinsicNoWar } else { CommitDrain::IntrinsicAuto },
            bare_seals: false,
            pin_mfma: false,
        },
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
    // Scheduling-coherence gate (plan §2.6): setprio balance + interleave sanity + hint purity ⇒
    // deleting every interleave/prio hint leaves a still-correct kernel. A build-time panic on a bug.
    let root_ids: Vec<TileId> = roots.iter().map(|e| e.dep()).collect();
    crate::schedule::verify_v2(&b.ir, &root_ids);
    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_fa_fwd_32".into() }
}
