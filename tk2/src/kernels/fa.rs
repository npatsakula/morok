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

use crate::build::{BF16, Buf, Builder, Effect, F32, Frag, Idx, Lds, Scope, Val};
use crate::ir::{FragMap, TileId};
use crate::kernels::{EDGE, Program, offset_by};
use crate::partition::RowPartition;
use crate::pipeline::{
    AccSlot, BlockCounter, CommitBatch, CommitCompletion, CommitDrain, Compute, Hooks, Init, Mem, Sched, SlotSet,
    SlotVal, pipeline,
};
use crate::shape::{Mfma16x16x16Bf16, MfmaShape};
use crate::tile::{ARow, BCol, Plain, Xor};
use crate::tile_move::{
    commit, commit_run, commit_run_asm, commit_transposed_run_asm, commit_transposed_v4_asm, gather, gather_run,
    prefetch,
};

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

    fn commit(&mut self, b: &mut Builder, _k_base: Idx, reg: &FaFill, war: &[TileId]) -> CommitBatch {
        // Intrinsic commit (an `s_barrier` auto-drains the `ds_write` `lgkmcnt(0)`) — `tile_move::commit`
        // pins `Drain::Intrinsic`; the asm/deferred-drain machinery is a matmul-perf concern, orthogonal.
        let (d, s) = (self.d, self.d as i64);
        let fk = commit(b, self.k_smem, d, self.k, s, self.epl_kv, self.tid, self.bh_row, &reg.k, war);
        let fv = commit(b, self.v_smem, d, self.v, s, self.epl_kv, self.tid, self.bh_row, &reg.v, war);
        CommitBatch::new(fk.into_iter().chain(fv).collect(), CommitCompletion::Intrinsic)
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

    fn ready_after_lgkm(&mut self, _b: &mut Builder, _op: Self::Op, _wait: TileId) -> Self::Op {
        panic!("FA-16 uses compiler-visible gathers and has no opaque readiness path")
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

/// Warps per workgroup for the **32×32×8 FA**: 8 warps × 32 Q rows = a `q_blk = 256` block (aiter/HK's
/// split-Q tiling — Route B). One 512-thread block yields 8 waves = 2 waves/SIMD (**occupancy-2**), the
/// substrate for the two-group phase stagger (group A's softmax VALU hidden under group B's MFMAs). Each
/// warp owns 32 Q rows and computes its own 32(kv)×32(q) MFMA tile; all 8 share ONE K/V LDS tile per KV
/// block. No cross-warp reduction (the softmax `acc_row_reduce_32` stays within each 64-lane warp —
/// disjoint Q rows), so 4→8 warps leaves every per-wave live value unchanged (≈160 VGPR/wave); only the
/// collaborative fill's per-lane run shrinks (`epl = KV·d/nthreads`, 512 threads).
const NUM_WARPS_32: usize = 8;
/// The 32×32×8 KV-block size (one MFMA N/M tile = aiter's `ts_kv`). Four hardware `K = 8` slices.
const KV_BLK_32: usize = 32;
/// The transposed-V LDS row padding. Pitch = `KV_BLK_32 + VT_PAD` must keep the per-lane b64 V read
/// conflict-free: consecutive lanes stride `pitch/2` dwords, so only a pitch whose dword-stride has
/// `gcd(·, 32) = 2` (not 4) spreads 16 lanes across 16 distinct LDS banks — the same spread the XOR
/// swizzle gives K. `VT_PAD = 8` → pitch 40 → stride 20 → gcd 4 → 8 banks → ~2-way conflict (measured
/// PMC bankconf ≈ 1.6); `VT_PAD = 4` → pitch 36 → stride 18 → gcd 2 → conflict-free. Still mult-of-4 for
/// b64 alignment. (An un-padded `[d, kv]` transpose regresses — proven by `v_transpose_probe`.)
const VT_PAD: usize = 4;

/// **V's LDS buffer count for the compute rotation** — TRIPLE-buffered (vs. K's double). The rotation
/// gathers `V(i−1)` (for P·V(i−1)) while committing `V(i+1)`, so THREE V blocks are live at once
/// (`i−1`, `i`, `i+1`). V loads/commits block `i+1` exactly like K (no prefetch skew); only the GATHER
/// reads one block behind, at parity `(counter+2) mod 3 == (counter−1) mod 3`. A bonus: the drain's
/// `V(nblocks−1)` (committed in the last steady iteration) is still resident, so no post-loop V re-fetch.
const V_NBUF: usize = 3;

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
/// collaborative 512-thread fill's per-thread `epl` elements as `gvec`-wide (b128/b64) load chunks).
struct Fa32Fill {
    k: Vec<Val<BF16>>,
    k_dma: Vec<Effect>,
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
    /// K LDS double-buffer: when set, a runtime block parity selects the read/write half so commit(k+1)
    /// writes the other buffer from gather(k). V has its independent, always-three-plane rotation.
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
    /// d128 experimental path: K writes directly to its alternate LDS plane while V remains
    /// register-staged for the required write-side transpose.
    direct_k: bool,
    /// Use waitcnt-opaque LDS operand reads with explicit pipeline readiness waits.
    asm_gather: bool,
    /// Register-staged K/V publish through waitcnt-opaque asm writes.
    asm_commit: bool,
    /// Four-row dword V prefetch plus packed b64 transposed commit.
    packed_v: bool,
    /// Two-crew phase stagger with the 2-cluster MERGED compute (softmax+QKᵀ+P·V in one 32-MFMA cluster so
    /// the `s_barrier` handoff amortizes like the matmul clone — small clusters lose to the barrier tax).
    /// When set, `gather(0)` returns K++V concatenated (the merged cluster consumes both).
    ping_pong: bool,
}

impl Fa32Hooks {
    /// K's LDS parity offset (elements) for `block`: `(block % 2) · k_tile` when double-buffered, else `0`
    /// (single tile — bit-identical to the pre-lever kernel).
    fn k_parity(&self, b: &mut Builder, block: Idx) -> Idx {
        if !self.double_buf {
            return b.idx_const(0);
        }
        let two = b.idx_const(2);
        let par = b.idx_mod(block, two);
        let k_tile = b.idx_const((KV_BLK_32 * self.d) as i64);
        b.idx_mul(par, k_tile)
    }

    /// V's LDS parity offset (elements) into the TRIPLE-buffered `vt_lds`: `(block % 3) · vt_tile`. The
    /// commit passes `block = i+1` (writes `V(i+1)`); the gather passes `block = counter+2` so it reads
    /// `(counter−1) mod 3` (the rotation's `V(i−1)`). Always triple-buffered — the three live V blocks
    /// are always disjoint parities, so no WAR seal is needed for V regardless of K's buffering.
    fn v_parity(&self, b: &mut Builder, block: Idx) -> Idx {
        let three = b.idx_const(V_NBUF as i64);
        let par = b.idx_mod(block, three);
        let vt_tile = b.idx_const((self.d * self.pitch) as i64);
        b.idx_mul(par, vt_tile)
    }
}

impl Hooks for Fa32Hooks {
    type Op = Fa32Op;
    type Reg = Fa32Fill;
    const PREFETCH_TILES: usize = 2; // 0 = K, 1 = V

    fn prologue_prefetch_tiles(&self) -> Vec<usize> {
        if self.direct_k { vec![1, 0] } else { vec![0, 1] }
    }

    fn prefetch(
        &mut self,
        b: &mut Builder,
        k_base: Idx,
        tile: usize,
        prev: Option<Fa32Fill>,
        order: &[TileId],
    ) -> (Fa32Fill, Vec<TileId>) {
        let mut reg = prev.unwrap_or(Fa32Fill { k: Vec::new(), k_dma: Vec::new(), v: Vec::new() });
        let gvec = if self.epl.is_multiple_of(8) { 8 } else { 4 };
        let epl_c = b.idx_const(self.epl as i64);
        let lane_epl = b.idx_mul(self.tid, epl_c);
        let flat_base = b.idx_add(self.bh_row_d, k_base);
        let flat_base = b.idx_add(flat_base, lane_epl);

        if tile == 0 && self.direct_k {
            let d_c = b.idx_const(self.d as i64);
            let two = b.idx_const(2);
            let kstep = b.idx_const((KV_BLK_32 * self.d) as i64);
            let blk = b.idx_div(k_base, kstep);
            let parity = self.k_parity(b, blk);
            let mut deps = order.to_vec();
            // V is issued first so vmcnt(4) can complete those older register loads while the four
            // younger direct-K DMAs remain in flight. The prologue has no cluster fence between its
            // tile prefetches, so carry the V values explicitly into the first DMA's issue order.
            deps.extend(reg.v.iter().map(|value| value.id));
            let mut dma = Vec::with_capacity(self.epl / 2);
            for call in 0..self.epl / 2 {
                // Each instruction must target one wave-coalesced run of physical LDS dwords. Invert
                // the XOR layout on the global source because XOR is self-inverse: physical LDS column
                // `p` receives logical source column `lds_col(row, p)`.
                let dword = if call == 0 { self.tid } else { offset_by(b, self.tid, call * NUM_WARPS_32 * WARP) };
                let elem = b.idx_mul(dword, two);
                let row = b.idx_div(elem, d_c);
                let physical_col = b.idx_mod(elem, d_c);
                let logical_col = b.lds_col(row, physical_col, self.d);
                let row_off = b.idx_mul(row, d_c);
                let src = b.idx_add(row_off, logical_col);
                let src = b.idx_add(k_base, src);
                let src = b.idx_add(self.bh_row_d, src);
                let dst = b.idx_add(parity, elem);
                let copy = b.global_load_lds_dword(self.k, src, self.k_lds, dst, &deps);
                deps = vec![copy.dep()];
                dma.push(copy);
            }
            let anchors = dma.iter().map(|e| e.dep()).collect();
            reg.k_dma = dma;
            return (reg, anchors);
        }

        if tile == 1 && self.packed_v {
            let warp_c = b.idx_const(WARP as i64);
            let four = b.idx_const(4);
            let two = b.idx_const(2);
            let d_c = b.idx_const(self.d as i64);
            let warp = b.idx_div(self.tid, warp_c);
            let lane = b.idx_mod(self.tid, warp_c);
            let first_row = b.idx_mul(warp, four);
            let d_pair = b.idx_mul(lane, two);
            let block_base = b.idx_add(self.bh_row_d, k_base);
            reg.v = (0..4)
                .map(|row| {
                    let kv = offset_by(b, first_row, row);
                    let row_base = b.idx_mul(kv, d_c);
                    let off = b.idx_add(block_base, row_base);
                    let off = b.idx_add(off, d_pair);
                    b.load_vec_after(self.v, off, 2, order)
                })
                .collect();
            let anchors = reg.v.iter().map(|v| v.id).collect();
            return (reg, anchors);
        }

        // Register-staged global load. K uses this path for d64; V always uses it because gfx942 needs
        // a write-side transpose before LDS publication.
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

    fn commit(&mut self, b: &mut Builder, k_base: Idx, reg: &Fa32Fill, war: &[TileId]) -> CommitBatch {
        let k_lds = if war.is_empty() { self.k_lds } else { b.lds_after(self.k_lds, war) };
        // Keep baseline interning order byte-identical: the intrinsic path historically rebound both
        // LDS handles before constructing parity expressions.
        let vt_lds = (!self.direct_k).then(|| if war.is_empty() { self.vt_lds } else { b.lds_after(self.vt_lds, war) });
        let (k_off, vt_off) = {
            let kstep = b.idx_const((KV_BLK_32 * self.d) as i64);
            let blk1 = b.idx_div(k_base, kstep); // committed block index (K & V both write block i+1)
            (self.k_parity(b, blk1), self.v_parity(b, blk1))
        };
        if self.direct_k {
            assert!(war.is_empty(), "direct K uses disjoint K2/V3 stages and must not emit a WAR seal");
            let last_dma = reg.k_dma.last().expect("direct K commit requires issued DMA").dep();
            let allowed = u8::try_from(reg.k_dma.len()).expect("direct K DMA batch must fit vmcnt");
            let v_ready = b.swait_vmcnt_allowed(Effect(last_dma), allowed);
            let mut effects = reg.k_dma.clone();
            if self.packed_v {
                effects.extend(commit_transposed_v4_asm(
                    b,
                    self.vt_lds,
                    self.pitch,
                    &reg.v,
                    self.tid,
                    vt_off,
                    Some(v_ready.dep()),
                ));
            } else {
                effects.extend(commit_transposed_run_asm(
                    b,
                    self.vt_lds,
                    self.pitch,
                    self.d,
                    &reg.v,
                    self.epl,
                    self.tid,
                    vt_off,
                    Some(v_ready.dep()),
                ));
            }
            return CommitBatch::new(effects, CommitCompletion::DirectAndOpaque);
        }
        if self.asm_commit {
            assert!(war.is_empty(), "asm K2/V3 commit must target disjoint stages without a WAR seal");
            return CommitBatch::new(
                commit_run_asm::<Xor>(
                    b,
                    k_lds,
                    self.d,
                    self.vt_lds,
                    self.pitch,
                    &reg.k,
                    &reg.v,
                    self.epl,
                    self.tid,
                    k_off,
                    vt_off,
                    self.packed_v,
                ),
                CommitCompletion::Opaque,
            );
        }
        CommitBatch::new(
            commit_run::<BF16, Xor, Plain>(
                b,
                k_lds,
                self.d,
                vt_lds.expect("intrinsic FA commit has a V LDS handle"),
                self.pitch,
                &reg.k,
                &reg.v,
                self.epl,
                self.tid,
                k_off,
                vt_off,
            ),
            CommitCompletion::Intrinsic,
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
        let scope = block.scope();
        let q_in = b.scope_idx(self.q_in, scope);
        let half_off = b.scope_idx(self.half_off, scope);
        let k_lds = b.scope_lds(self.k_lds, scope);
        let vt_lds = b.scope_lds(self.vt_lds, scope);
        // K gathers the CURRENT block (parity `counter % 2`, for QKᵀ(i)); V gathers ONE BLOCK BEHIND (for
        // P·V(i−1)) — parity `(counter−1) % 3`, computed as `(counter+2) % 3` to stay non-negative. This is
        // the compute-rotation's V skew: it rides the triple-buffer, no prefetch/commit skew, no clamp.
        let (k_off, vt_off) = {
            let blk = block.idx(b);
            let two = b.idx_const(2);
            let blk_v = b.idx_add(blk, two); // (counter+2) ≡ (counter−1) (mod 3)
            (self.k_parity(b, blk), self.v_parity(b, blk_v))
        };
        // The contiguous-run addressing is DERIVED by `gather_run` from the ARow role + the tile types: K is
        // the natural `[kv=32, d]` `Xor`-swizzled tile (`dslices = d/8` col-fragments), V the transposed
        // `[d, kv=32]` `Plain` padded-pitch tile (`dtiles·ksl` fragments). `q_in`/`half_off` = the lane
        // partition `lane_rc(S::a_map(), wlane, 0)`; the 4-run start stays 4-aligned so the swizzled K read
        // remains a contiguous `ds_read_b64`.
        let gather_k = |b: &mut Builder, q_in: Idx, half_off: Idx| {
            gather_run::<BF16, Xor, S>(b, k_lds, self.d, S::M, self.d, q_in, half_off, k_off, self.asm_gather, raw)
        };
        let gather_v = |b: &mut Builder, q_in: Idx, half_off: Idx| {
            gather_run::<BF16, Plain, S>(
                b,
                vt_lds,
                self.pitch,
                self.d,
                KV_BLK_32,
                q_in,
                half_off,
                vt_off,
                self.asm_gather,
                raw,
            )
        };
        let (vecs, gathers) = match slice {
            // 2-cluster ping-pong: the single merged compute cluster consumes BOTH K (QKᵀ) and V (P·V), so
            // gather them together — K fragments first (`dslices`), then V (`dtiles·ksl`).
            0 if self.ping_pong => {
                let (mut kv, mut kg) = gather_k(b, q_in, half_off);
                let (vv, vg) = gather_v(b, q_in, half_off);
                kv.extend(vv);
                kg.extend(vg);
                (kv, kg)
            }
            0 => gather_k(b, q_in, half_off),
            1 => gather_v(b, q_in, half_off),
            _ => panic!("FA-32 gather: slice ∈ {{0=K, 1=V}}, got {slice}"),
        };
        let anchor = vecs[0].id;
        (vecs, gathers, anchor)
    }

    fn ready_after_lgkm(&mut self, b: &mut Builder, op: Self::Op, wait: TileId) -> Self::Op {
        op.into_iter().map(|v| b.opaque_ready_b64(v, wait)).collect()
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
/// 8 warps × 32 Q rows = a `q_blk = 256` block. Assembles the four device-proven 32×32×8 primitives:
/// QKᵀ (`v_mfma_f32_32x32x8`, `S[kv,q]`, kv on M) → online softmax over kv via [`Builder::acc_row_reduce_32`]
/// (the `EPT_C = 16` AccDist reduce) → P→PV relayout via [`Builder::pv_relayout_s49`] (`v_perm s49`) → P·V
/// with V staged through the padded transposed LDS (read straight).
///
/// **Phase 1 (the ClusterCx port):** the KV stream is a **rolled [`crate::pipeline`] loop** (like the 16×16
/// FA) rather than the correctness-first unrolled assembly — O/m/l are the loop-carried accumulators, the
/// per-KV `s`/`p` ride the rotated QK/softmax/PV dataflow, and [`Fa32Hooks`] supplies the register-staged
/// prefetch, commit, and gather. This scales past `n = 128` and inherits the
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
    assert!(matches!(d, 64 | 128), "FA-32 supports only head dimensions d=64 or d=128");
    let q_blk = NUM_WARPS_32 * crate::shape::Mfma32x32x8Bf16::M;
    assert!(
        n > 0 && n.is_multiple_of(q_blk),
        "FA-32 sequence length n must be a positive multiple of the Q tile ({q_blk})"
    );
    // d128: the MERGED two-crew ping-pong is the fastest at EVERY d128 size (device-measured 1.02–1.09× over
    // the single-crew packed-V path). Public `n` is a Q-block (256) multiple ⇒ never ragged ⇒ always
    // ping-pong-eligible (nblocks ≥ 8 ≥ 3). d64 stays single-crew (the ping-pong is d128-only).
    if d == 128 {
        return flash_attention_fwd_32_pingpong(bh, n, d);
    }
    // Direct K amortizes its address/wait machinery only on long d128 streams. Centered-data A/B
    // measurements cross the 3% gate at S>=1024; shorter shapes retain the faster register path.
    // d128 long path: direct-to-LDS K + PACKED transposed V (v_perm register-transpose + wide `ds_write_b64`).
    // The packed V write replaces the scalar `ds_write_b16` transpose whose consecutive fill lanes address
    // d-rows 8 apart (dword stride 144 ≡ 16 mod 32 → ~16-way LDS bank conflict); the b64 packed write drops
    // that to 2-way and measures +8% at S2048 d128 (301→325). See the two-plane skew (VT_SKEW) for the
    // conflict-free finish. Below the S≥1024 crossover the register-staged path stays faster.
    let long_d128 = d == 128 && n >= 1024;
    flash_attention_fwd_32_impl(bh, n, d, long_d128, false, false, long_d128, false, false)
}

/// Ping-pong FA-32: the 8-wave two-crew phase stagger (`warp_row = warp/4`) layered on the asm-opaque
/// register-staged movement — HK's schedule adapted to gfx942. Reuses the warmup + drain; the engine runs
/// the warmup in lockstep and offsets the crews after it, and every steady cluster seal becomes the
/// workgroup `s_barrier` that carries the phase. The opaque (`asm_gather`/`asm_commit`) LDS movement is
/// what makes the single-parity-half commit race-free under a lagged crew (compiler-visible LDS reschedules
/// past any barrier — see the matmul clone). d128 only for now (K2/V3 asm staging).
pub fn flash_attention_fwd_32_pingpong(bh: usize, n: usize, d: usize) -> Program {
    assert_eq!(d, 128, "FA-32 ping-pong currently requires d128 (register asm K2/V3 staging)");
    let q_blk = NUM_WARPS_32 * crate::shape::Mfma32x32x8Bf16::M;
    assert!(
        n > 0 && n.is_multiple_of(q_blk),
        "FA-32 sequence length n must be a positive multiple of the Q tile ({q_blk})"
    );
    // Asm-opaque register-staged K + PACKED transposed V (v_perm + wide `ds_write_b64`, the bank-conflict
    // fix) under the phase stagger. Packed V rides the asm_commit branch (stagger-safe, opaque), so the
    // two-crew overlap no longer pays the narrow-`ds_write_b16` V-write tax. (direct_k stays incompatible
    // with ping-pong: its separate publish cluster violates the phased-tail-depends-on-publication invariant.)
    flash_attention_fwd_32_impl(bh, n, d, false, true, true, true, true, false)
}

/// aiter-API-matched **bf16 O** ping-pong FA-32: identical to [`flash_attention_fwd_32_pingpong`] except
/// the final O scatter truncates the f32 accumulator to bf16 (RTZ, `bits>>16`), halving the O write
/// bytes. FA-32 is memory-bound and O is ~11% of its traffic as f32, so this removes ~5% of total
/// bytes — the fair-API match to aiter (which stores bf16-RTZ O). The MFMA accumulator stays f32; only
/// the store casts. Allocate the output tensor as `DType::BFloat16`. Same `SwizzlePass` requirement,
/// same d128-only constraint as the f32 ping-pong.
pub fn flash_attention_fwd_32_pingpong_bf16o(bh: usize, n: usize, d: usize) -> Program {
    assert_eq!(d, 128, "FA-32 ping-pong currently requires d128 (register asm K2/V3 staging)");
    let q_blk = NUM_WARPS_32 * crate::shape::Mfma32x32x8Bf16::M;
    assert!(
        n > 0 && n.is_multiple_of(q_blk),
        "FA-32 sequence length n must be a positive multiple of the Q tile ({q_blk})"
    );
    flash_attention_fwd_32_impl(bh, n, d, false, true, true, true, true, true)
}

/// Test-only constructor forcing direct K below the production crossover for warmup/ragged coverage.
#[cfg(test)]
pub(crate) fn flash_attention_fwd_32_direct_k(bh: usize, n: usize, d: usize) -> Program {
    assert_eq!(d, 128, "ordered direct K is currently defined only for d128");
    flash_attention_fwd_32_impl(bh, n, d, true, false, false, false, false, false)
}

/// Test-only direct-K constructor qualifying the packed transposed-V movement experiment.
#[cfg(test)]
pub(crate) fn flash_attention_fwd_32_direct_k_packed_v(bh: usize, n: usize, d: usize) -> Program {
    assert_eq!(d, 128, "packed V is currently defined only for d128");
    flash_attention_fwd_32_impl(bh, n, d, true, false, false, true, false, false)
}

/// Test-only direct-K constructor qualifying waitcnt-opaque K/V gathers before phase staggering.
#[cfg(test)]
pub(crate) fn flash_attention_fwd_32_direct_k_asm_gather(bh: usize, n: usize, d: usize) -> Program {
    assert_eq!(d, 128, "ordered direct K is currently defined only for d128");
    flash_attention_fwd_32_impl(bh, n, d, true, true, false, true, false, false)
}

/// Test-only long-shape oracle retaining the pre-optimization register-staged K path.
#[cfg(test)]
pub(crate) fn flash_attention_fwd_32_register_k(bh: usize, n: usize, d: usize) -> Program {
    flash_attention_fwd_32_impl(bh, n, d, false, false, false, false, false, false)
}

/// Test-only register-staged asm movement without asymmetric wave phasing.
#[cfg(test)]
pub(crate) fn flash_attention_fwd_32_register_asm(bh: usize, n: usize, d: usize) -> Program {
    assert_eq!(d, 128, "register asm movement currently requires d128 K2/V3 staging");
    flash_attention_fwd_32_impl(bh, n, d, false, true, true, false, false, false)
}

/// The output-O buffer handle, monomorphized over its store dtype so the ping-pong FA can ship an
/// aiter-matched **bf16 O** (half the O-write bytes — the one unblocked memory-side lever for this
/// memory-bound kernel) alongside the f32 default without a second impl. The MFMA accumulator stays
/// f32; only the FINAL scatter re-types — the bf16 path truncates (RTZ, [`Builder::bf16_trunc`], the
/// `bits>>16` cast) at the store, matching aiter's bf16-RTZ O store.
#[derive(Copy, Clone)]
enum OutO {
    F32(Buf<F32>),
    Bf16(Buf<BF16>),
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn flash_attention_fwd_32_impl(
    bh: usize,
    n: usize,
    d: usize,
    direct_k: bool,
    asm_gather: bool,
    asm_commit: bool,
    packed_v: bool,
    ping_pong: bool,
    bf16_o: bool,
) -> Program {
    use crate::shape::Mfma32x32x8Bf16 as S;
    assert!(d.is_multiple_of(S::M), "FA-32 head dim d must be a multiple of 32");
    assert!(bh >= 1, "bh must be ≥ 1");
    let (m32, k8) = (S::M, S::K); // 32, 8
    let nthreads = NUM_WARPS_32 * WARP; // 512
    let q_blk = NUM_WARPS_32 * m32; // 256
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
    // ABI: O[bh·n, d] then Q, K, V — each `[bh, n, d]` row-major. O binds the first ABI slot regardless
    // of its store dtype (f32 default, or aiter-matched bf16 under `bf16_o`); Q/K/V follow unchanged.
    let o = if bf16_o { OutO::Bf16(b.global::<BF16>(bh * n * d)) } else { OutO::F32(b.global::<F32>(bh * n * d)) };
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
    // d64 is memory-heavier and benefits when one XCD reuses consecutive Q tiles' common K/V stream.
    // d128 is compute-heavy and measured slower under this remap, so retain the native order there.
    let logical_wgid = if d == 64 { part.xcd_swizzle(&mut b, wgid, 8, 16) } else { wgid };
    let pid = part.decode(&mut b, logical_wgid);

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
    // that latency, so the double-buffer's extra VGPRs + parity math only regress (~-9%, measured).
    let double_buf = d >= 128 || direct_k;
    // Lever-2: the 8-wave two-crew phase stagger. `warp_row = warp/4 ∈ {0,1}` splits the 8 waves into a
    // leader crew (warps 0-3) and follower crew (warps 4-7), one wave per SIMD each — HK's `warpid()/4`.
    // The engine runs the warmup in lockstep, then offsets the crews (eq=1 after warmup, eq=0 rebalance in
    // the epilogue), and every steady cluster seal becomes the workgroup `s_barrier` that carries the phase.
    let warp_row: Option<Idx> = if ping_pong {
        let four = b.idx_const(4);
        Some(b.idx_div(warp, four))
    } else {
        None
    };

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

    // Shared LDS: K natural [kv,d] (double-buffered under Lever-3 so commit(i+1)/gather(i) touch disjoint
    // halves), V transposed padded [d, pitch] TRIPLE-buffered (the rotation keeps 3 V blocks live: i−1
    // consumed by P·V, i, i+1 committed — see `V_NBUF`).
    let k_nbuf = if double_buf { 2 } else { 1 };
    let k_lds = b.define_local::<BF16>(k_nbuf * KV_BLK_32 * d);
    let vt_lds = b.define_local::<BF16>(V_NBUF * d * pitch);
    let epl = KV_BLK_32 * d / nthreads; // collaborative fill, per thread

    // Softmax scale folded into exp2: exp2(score·log2(e)/√d) == exp(score/√d). 16-wide broadcast.
    let scale = std::f32::consts::LOG2_E / (d as f32).sqrt();
    let scale_bcast = {
        let c = b.f32(scale);
        crate::tile_ops::splat::<S>(&mut b, c)
    };

    // ── the heterogeneous slot set (DESIGN §3.2). CARRIED: `o_0..o_{dtiles-1}` (16-wide f32 PV acc),
    //    `m` (running max), `l` (running norm), and rotated `s` (QKᵀ scores, 16-wide f32). TEMPORARY:
    //    `p` (softmax weights, 16-wide f32 → v_perm-packed in PV). All f32
    //    on the `EPT_C = 16` accumulator (`c_map`); each warp keeps its OWN o/m/l over its 32 Q rows. ──
    // Declared, not hand-numbered (Phase 3): `SlotSet` allocates each fragment as it is declared but
    // DEFERS every init to `finish`, so the emission stays "all define_frags, then all inits" (o×dtiles,
    // m, l, s, p → zero o×dtiles, const m, zero l) — byte-identical to the old hand-written slot table.
    let mut slots = SlotSet::new();
    let o_idx = slots.carried_group(&mut b, dtiles, S::c_map(), Init::Zero);
    // Running max seed = −FLT_MAX (FINITE, not −∞): the rotation's seed block (softmax of the not-yet-real
    // block 0 at iteration 0) must be inert. With `s` seeded −∞ and `m` finite: `m_new = max(−FLT_MAX,
    // reduce_max(−∞)) = −FLT_MAX`, so `corr = exp2(−FLT_MAX − (−FLT_MAX)) = exp2(0) = 1` (no `∞−∞` NaN) and
    // `p = exp2(−∞ − (−FLT_MAX)) = 0` (no phantom mass). The first REAL block's max always exceeds −FLT_MAX,
    // so the running max is numerically identical to a −∞ seed.
    let slot_m = slots.carried(&mut b, S::c_map(), Init::Const(-f32::MAX));
    let slot_l = slots.carried(&mut b, S::c_map(), Init::Zero); // running norm seed = 0
    // `s` is CARRIED (double-buffer): the fused QKᵀ∥softmax cluster reads block i−1's scores (carry-in)
    // and writes block i's fresh QKᵀ, so softmax(i−1)'s VALU is independent of QKᵀ(i)'s MFMAs — the
    // interleave's independent-MFMA stream. Seed −∞ (an empty block: exp2(−∞−m)=0, no phantom mass).
    let slot_s = slots.carried(&mut b, S::c_map(), Init::Const(f32::NEG_INFINITY));
    let slot_p = slots.temp(&mut b, S::c_map()); // softmax weights (temporary)
    let (accs, inited) = slots.finish(&mut b);

    // Per-cluster read/write slot sets (asymmetric — the §3.2 point). The FUSED QKᵀ∥softmax cluster reads
    // {s(i−1),m,l,o_*} and writes {s(i),m,l,p,o_*}; PV reads {p,o_*} writes {o_*}. `o_idx` is the
    // carried_group's returned slot indices (0..dtiles).
    let qk_reads: Vec<usize> = [slot_s, slot_m, slot_l].into_iter().chain(0..dtiles).collect();
    let qk_writes: Vec<usize> = [slot_s, slot_m, slot_l, slot_p].into_iter().chain(0..dtiles).collect();
    let pv_reads: Vec<usize> = [slot_p].into_iter().chain(0..dtiles).collect();

    // Warmup QK(0), with no seed softmax and no zero-weight P·V. The warmup memory cluster gathers K0
    // before committing K1/V1, so d64's single K buffer cannot be overwritten before its operands are
    // resident in VGPRs. The regular fused rotation then begins at block 1 with S(0) carried.
    let q_warmup = q_frags.clone();
    let qk_warmup = Compute::<Fa32Hooks>::new(
        0,
        vec![],
        vec![slot_s],
        move |b: &mut Builder, op: Option<&Fa32Op>, _reads: &[SlotVal], _blk: BlockCounter| {
            let k_frags = op.expect("FA warmup QK consumes gathered K0");
            let z = b.f32(0.0);
            let mut s_acc = crate::tile_ops::splat::<S>(b, z);
            for ki in 0..dslices {
                s_acc = crate::tile_ops::mma::<S>(b, k_frags[ki], q_warmup[ki], s_acc);
            }
            vec![SlotVal::F32(s_acc)]
        },
    )
    .no_prio();

    // ── the FUSED QKᵀ∥softmax cluster + the PV cluster (the self-contained-cluster restructure) ──
    // ONE self-contained compute region: (a) online softmax on the CARRIED s (block i−1's raw scores);
    // (b) QKᵀ(i) → the new s. The 16 QKᵀ MFMAs are the INDEPENDENT stream (they read K(i)/Q, not s(i−1)),
    // so softmax(i−1)'s exp2/reduce VALU folds UNDER them — the `interleave_exp` hint (group 1) is now
    // anchored on the QKᵀ MFMA output (`s_acc.id`), a real matrix op in the SAME region, so LLVM honors it
    // (vs. the old anchor on the non-MFMA `s_scaled.id`, which had no MFMA to group against → dropped).
    // INTRINSIC MFMA (the asm `sideeffect` form is opaque to the GCNHazardRecognizer → no 32×32×8 hazard
    // `s_nop`s → device-proven NaN; the intrinsic is schedulable, so the interleave binds fine with it).
    let q_frags_pp = q_frags.clone(); // the merged (2-cluster ping-pong) compute also needs Q; both closures own it
    let qk_softmax = Compute::<Fa32Hooks>::new(
        0,
        qk_reads,
        qk_writes,
        move |b: &mut Builder, op: Option<&Fa32Op>, reads: &[SlotVal], blk: BlockCounter| {
            let wlane = b.scope_idx(wlane, blk.scope());
            let k_frags = op.expect("QKᵀ consumes gathered K");
            let (s_prev, m_run, l_run) = (reads[0].f32(), reads[1].f32(), reads[2].f32());
            // (a) online softmax of block i−1 (the carried scores). scale, running max via
            // `acc_row_reduce_32` (the C=16 AccDist reduce over kv), O/l rescale, P = exp2(S−max), l += ΣP.
            // NO ragged mask here: the rotation processes softmax for blocks 0..nblocks−2 only (the loop's
            // fused cluster never touches the last block); the ONLY possibly-ragged block, nblocks−1, is
            // softmaxed in the post-loop DRAIN, which applies the mask there.
            let s_scaled = b.mul(s_prev, scale_bcast);
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
            // (b) QKᵀ(i): fresh scores into the CARRIED s buffer — the independent MFMA stream.
            let z = b.f32(0.0);
            let mut s_acc = crate::tile_ops::splat::<S>(b, z);
            for ki in 0..dslices {
                s_acc = crate::tile_ops::mma::<S>(b, k_frags[ki], q_frags[ki], s_acc);
            }
            // Softmax-under-MFMA: fold softmax(i−1)'s exp2 under QKᵀ(i)'s MFMAs — `interleave_exp` (group 1)
            // anchored on the QKᵀ MFMA output. The hint emits NO instruction; kept live by routing into `p`.
            let p = match b.interleave_exp(FA_EXP_PAIRS, FA_EXP_CNT, 1, &[s_acc.id]) {
                Some(h) => b.val_after(p, &[h.dep()]),
                None => p,
            };
            // writes: [s(new), m, l, p, o_*].
            let mut out = vec![SlotVal::F32(s_acc), SlotVal::F32(m_new), SlotVal::F32(l_new), SlotVal::F32(p)];
            for i in 0..dtiles {
                out.push(SlotVal::F32(b.mul(reads[3 + i].f32(), corr))); // O *= corr
            }
            out
        },
    )
    // Only the P·V cluster raises `s_setprio` — HK's convention, and it holds under ping-pong too: raising
    // priority during QKᵀ as well starves the offset crew's memory phase of SIMD issue slots (it can't
    // prefetch far enough ahead), measured a net loss. QKᵀ stays prio-neutral (also keeps the `verify_v2`
    // setprio balance).
    .no_prio();

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

    // 2-cluster ping-pong MERGED compute: softmax(i−1) + QKᵀ(i) + P·V(i−1) in ONE 32-MFMA compute cluster,
    // so the phase-stagger `s_barrier` amortizes over 32 MFMAs (matching the matmul clone) instead of the
    // 4-cluster's 16 — the small-cluster barrier tax is why the split ping-pong lost to single-crew
    // production. Operand 0 is the merged K++V gather (`ksteps=1`); `p` is a local, not a written slot.
    let merged_reads: Vec<usize> = [slot_s, slot_m, slot_l].into_iter().chain(o_idx.iter().copied()).collect();
    let merged = Compute::<Fa32Hooks>::new(
        0,
        merged_reads.clone(),
        merged_reads,
        move |b: &mut Builder, op: Option<&Fa32Op>, reads: &[SlotVal], blk: BlockCounter| {
            let wlane = b.scope_idx(wlane, blk.scope());
            let all = op.expect("merged ping-pong consumes K++V");
            let k_frags = &all[0..dslices];
            let v_frags = &all[dslices..];
            let (s_prev, m_run, l_run) = (reads[0].f32(), reads[1].f32(), reads[2].f32());
            // softmax(i−1) on the carried scores.
            let s_scaled = b.mul(s_prev, scale_bcast);
            let m_new = crate::tile_ops::row_reduce::<S>(b, s_scaled, wlane, m_run, false);
            let corr = {
                let diff = b.sub(m_run, m_new);
                b.exp2(diff)
            };
            let l_resc = b.mul(l_run, corr);
            let p = {
                let sh = b.sub(s_scaled, m_new);
                b.exp2(sh)
            };
            let l_new = crate::tile_ops::row_reduce::<S>(b, p, wlane, l_resc, true);
            // QKᵀ(i) → the fresh carried scores (independent MFMA stream; softmax exp folds under it, group 1).
            let z = b.f32(0.0);
            let mut s_acc = crate::tile_ops::splat::<S>(b, z);
            for ki in 0..dslices {
                s_acc = crate::tile_ops::mma::<S>(b, k_frags[ki], q_frags_pp[ki], s_acc);
            }
            let s_acc = match b.interleave_exp(FA_EXP_PAIRS, FA_EXP_CNT, 1, &[s_acc.id]) {
                Some(h) => b.val_after(s_acc, &[h.dep()]),
                None => s_acc,
            };
            // P·V(i−1): O = O·corr + P·V (relayout/reduce VALU folds under the PV MFMAs, group 2).
            let b_ops = crate::tile_ops::relayout::<S>(b, p);
            let mut anchor = None;
            let mut o_out: Vec<SlotVal> = (0..dtiles)
                .map(|dt| {
                    let mut o = b.mul(reads[3 + dt].f32(), corr);
                    for s in 0..ksl {
                        o = crate::tile_ops::mma::<S>(b, v_frags[dt * ksl + s], b_ops[s], o);
                    }
                    anchor.get_or_insert(o.id);
                    SlotVal::F32(o)
                })
                .collect();
            if let (Some(a), Some(SlotVal::F32(o0))) = (anchor, o_out.first().copied())
                && let Some(h) = b.interleave_valu(FA_VALU_PAIRS, FA_VALU_CNT, 2, &[a])
            {
                o_out[0] = SlotVal::F32(b.val_after(o0, &[h.dep()]));
            }
            let mut out = vec![SlotVal::F32(s_acc), SlotVal::F32(m_new), SlotVal::F32(l_new)];
            out.extend(o_out);
            out
        },
    );

    // This (b,h) slice's flat base (rows·d). `bh_row = pid.slice·n` reuses the decode's interned nodes.
    let n_c = b.idx_const(n as i64);
    let bh_row = b.idx_mul(pid.slice, n_c);
    let bh_row_d = b.idx_mul(bh_row, d_c);
    let hooks = Fa32Hooks {
        k,
        v,
        k_lds,
        vt_lds,
        double_buf,
        bh_row_d,
        tid,
        epl,
        q_in,
        half_off,
        d,
        pitch,
        direct_k,
        asm_gather,
        asm_commit,
        packed_v,
        ping_pong,
    };
    let pipe = pipeline(
        &mut b,
        nblocks,       // nblocks (streaming over KV; ⌈n/kv_blk⌉ — the last block may be ragged)
        KV_BLK_32 * d, // k_step: the FLAT per-block advance (kv_blk rows · d)
        // gather slices: 2 (K, V) normally; 1 MERGED (K++V) for the 2-cluster ping-pong.
        if ping_pong { 1 } else { 2 },
        &accs,
        &inited,
        warp_row, // Some(warp/4) for the ping-pong path enables the two-crew phase stagger; None = single-crew
        Sched {
            asm_gather,
            resident: false,
            // Lever-3: double-buffered ⇒ drop the WAR seal (commit/gather touch disjoint parity halves).
            commit_drain: if asm_commit {
                CommitDrain::AsmPublishedNoWar
            } else if direct_k {
                CommitDrain::DirectDeferred
            } else if double_buf {
                CommitDrain::IntrinsicNoWar
            } else {
                CommitDrain::IntrinsicAuto
            },
            bare_seals: asm_commit,
            pin_mfma: false,
        },
        hooks,
    );
    let pipe = if direct_k {
        pipe.warmup_cluster(Mem::builder().prefetch([1]).gathers([0]).prefetch_after_gathers([0]).commit(true).build())
            .warmup_cluster(qk_warmup)
            .warmup_cluster(Mem::builder().publish(true).build())
            .warmup_seed(slot_s)
    } else {
        pipe.warmup_cluster(Mem::builder().prefetch([0, 1]).gathers([0]).commit(true).build())
            .warmup_cluster(qk_warmup)
            .warmup_seed(slot_s)
    };
    let pipe = if ragged { pipe.scoped_regions() } else { pipe };
    let acc_final = if direct_k {
        pipe.cluster(Mem::builder().prefetch([1]).gathers([0, 1]).prefetch_after_gathers([0]).commit(true).build())
            .cluster(qk_softmax)
            .cluster(pv)
            .cluster(Mem::builder().publish(true).build())
            .build()
    } else if ping_pong {
        // 2-cluster ping-pong: Mem (prefetch next K+V, gather the MERGED K++V, commit) + the merged 32-MFMA
        // compute. 2 `s_barrier`s/block, amortizing the phase handoff over 32 MFMAs (the matmul-clone ratio,
        // which beats single-crew). The 3-cluster variant (separate softmax/PV clusters for aiter's cross-crew
        // softmax-under-MFMA hiding) was device-measured a NET LOSS — 335→285 — its extra barrier dominates.
        pipe.cluster(Mem::builder().prefetch([0, 1]).gathers([0]).commit(true).build()).cluster(merged).build()
    } else {
        pipe.cluster(Mem::builder().prefetch([0, 1]).gathers([0, 1]).commit(true).build())
            .cluster(qk_softmax)
            .cluster(pv)
            .build()
    };

    // ── DRAIN: block nblocks−1's softmax + P·V (the rotation leaves it undone — its QKᵀ ran in the
    //    epilogue → carried `s`; softmax/P·V of the LAST block have no following QKᵀ to fuse with). The
    //    last block's V(nblocks−1) is still resident in the triple-buffered vt_lds (committed in the last
    //    steady iteration, at parity (nblocks−1)%3), so no re-fetch — just gather it and finish. This is
    //    also the ONLY block that can be ragged, so the ragged mask lives here. ──
    let vt_off_drain = b.idx_const(((nblocks - 1) % V_NBUF * (d * pitch)) as i64);
    let s_last = b.load_frag_vec(acc_final[slot_s].f32()); // block nblocks−1's QKᵀ scores (carried, observes End)
    let m_run = b.load_frag_vec(acc_final[slot_m].f32());
    let l_run = b.load_frag_vec(acc_final[slot_l].f32());
    // The drain's V gather must be emitted in POST-LOOP scope with FRESH nodes — `gather_run` hash-conses
    // its LDS-address nodes on `(q_in, half_off, vt_lds)`, which the loop's gather already interned INSIDE
    // the loop body; reusing them post-loop breaks SSA dominance (hard fail at some `n`) and mis-addresses
    // the read (a ~5%-of-last-block soft error at others). Re-bind the shared leaves to one anchored
    // lexical scope so all drain address/reduction DAGs are distinct and dominated post-pipeline.
    let end_dep = [s_last.id];
    let (drain_scope, q_in_d, half_off_d, vt_lds_d, wlane_d) = if ragged {
        let scope = b.scope(&end_dep);
        (
            scope,
            b.scope_idx(q_in, scope),
            b.scope_idx(half_off, scope),
            b.scope_lds(vt_lds, scope),
            b.scope_idx(wlane, scope),
        )
    } else {
        // Preserve the tuned tile-exact DAG: these ordering rebinds predate lexical scopes and keep the
        // drain distinct without perturbing the steady-state scheduler topology.
        (
            Scope::ROOT,
            b.idx_after(q_in, &end_dep),
            b.idx_after(half_off, &end_dep),
            b.lds_after(vt_lds, &end_dep),
            wlane,
        )
    };
    let (v_frags, gathers) = gather_run::<BF16, Plain, S>(
        &mut b,
        vt_lds_d,
        pitch,
        d,
        KV_BLK_32,
        q_in_d,
        half_off_d,
        vt_off_drain,
        asm_gather,
        &end_dep,
    );
    // The V `ds_read`s are async — WAIT for them (`lgkmcnt(0)`) before the P·V MMA consumes `v_frags`.
    let drain_wait = b.swait_lgkmcnt(*gathers.last().expect("V gather emits ≥1 read")).dep();
    let v_frags: Vec<Val<BF16>> =
        if asm_gather { v_frags.into_iter().map(|v| b.opaque_ready_b64(v, drain_wait)).collect() } else { v_frags };
    let s_scaled = b.mul(s_last, scale_bcast);
    let s_scaled = if ragged {
        mask_ragged_kv(&mut b, s_scaled, BlockCounter::Epilogue((nblocks - 1) as i64, drain_scope), wlane_d, n)
    } else {
        s_scaled
    };
    let m_new = crate::tile_ops::row_reduce::<S>(&mut b, s_scaled, wlane_d, m_run, false);
    let corr = {
        let diff = b.sub(m_run, m_new);
        b.exp2(diff)
    };
    let l_resc = b.mul(l_run, corr);
    let p = {
        let sh = b.sub(s_scaled, m_new);
        b.exp2(sh)
    };
    let l_drain = crate::tile_ops::row_reduce::<S>(&mut b, p, wlane_d, l_resc, true);
    let b_ops = crate::tile_ops::relayout::<S>(&mut b, p);
    // Route each P·V result THROUGH its accumulator fragment (store then load) — the loop's `compute`
    // wrapper does exactly this (mma → `store_frag_vec` → later `load_frag_vec` → VALU), which gives the
    // AMDGPU GCNHazardRecognizer a clean MFMA→store boundary; the drain's direct mma→VALU (scatter `mul`)
    // otherwise trips the 32×32×8 "VALU reads accumulator before the MFMA result lands" hazard.
    let o_drain: Vec<Frag<F32>> = (0..dtiles)
        .map(|dt| {
            let o_run = b.load_frag_vec(acc_final[dt].f32());
            let o = b.mul(o_run, corr); // O *= corr for the new running max
            let mut o = b.val_after(o, &[drain_wait]); // order the MMA after the V-read lgkmcnt drain
            for s in 0..ksl {
                o = crate::tile_ops::mma::<S>(&mut b, v_frags[dt * ksl + s], b_ops[s], o);
            }
            let st = b.store_frag_vec(acc_final[dt].f32(), o).dep();
            b.frag_after(acc_final[dt].f32(), &[st])
        })
        .collect();

    // ── normalize O /= l (per q, broadcast across d) and transpose-scatter to O[q_global, d_global],
    //    reading the DRAINED final l/O. ──
    let recip_l = b.recip(l_drain);
    let dist = S::acc_dist();
    let mut roots = Vec::new();
    for dt in 0..dtiles {
        let o_vec = b.load_frag_vec(o_drain[dt]);
        let o_norm = b.mul(o_vec, recip_l);
        for i in 0..S::EPT_C {
            let (row, col) = b.acc_rc(dist, wlane_d, i); // row = d-in-tile, col = q-in-tile
            let d_global = offset_by(&mut b, row, dt * m32);
            let q_global = b.idx_add(q_row_base, col);
            let off = b.idx_mul(q_global, d_c);
            let off = b.idx_add(off, d_global);
            let val = b.vec_extract(o_norm, i);
            // The MFMA accumulator is f32; only the store re-types. bf16 O truncates (RTZ) here.
            let eff = match o {
                OutO::F32(buf) => b.store(buf, off, val),
                OutO::Bf16(buf) => {
                    let val = b.bf16_trunc(val);
                    b.store(buf, off, val)
                }
            };
            roots.push(eff);
        }
    }
    // Scheduling-coherence gate (plan §2.6): setprio balance + interleave sanity + hint purity ⇒
    // deleting every interleave/prio hint leaves a still-correct kernel. A build-time panic on a bug.
    let root_ids: Vec<TileId> = roots.iter().map(|e| e.dep()).collect();
    crate::schedule::verify_v2(&b.ir, &root_ids);
    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_fa_fwd_32".into() }
}
