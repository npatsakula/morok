//! **The movement half of the tile-op vocabulary** (`scratchpad/tile_layer_design.md` §3) — the
//! residency-dispatched `load`/`store` ops. Each op is keyed on a `(Dst, Src)` **residency pair** and
//! is a faithful, behaviour-preserving forward to the proven [`crate::tile_move`] primitive that already
//! performs that transfer; the STATIC choices (operand map, `n_frags` geometry, `ept`, swizzle, the
//! straight-vs-transposed gather) are DERIVED from the tile TYPES ([`RegLayout`]/[`Swizzle`]/
//! [`MfmaShape`], exactly as [`crate::tile_ops`] derives its compute widths from `S`), while the
//! genuinely-runtime addressing (lane, origin, base, `epl`, ordering deps) stays as explicit params.
//!
//! It mirrors [`crate::tile_ops`]'s style: thin, marker-generic free functions over the raw handles.
//! The probes/kernels this surface serves are *runtime*-shaped (`atb_probe(kv, d, q)`), so the dims
//! ride as values and the [`crate::tile`] marker only carries the operand role / swizzle policy. No
//! addressing/swizzle/asm logic is re-implemented here — that all lives in
//! [`crate::tile_move`] and [`crate::build`], which these ops call unchanged.
//!
//! ## The §3.2 residency table (design), and what each op forwards to
//!
//! | `(Dst::RES, Src::RES)` | op | forwards to |
//! |---|---|---|
//! | `(Reg, Lds)` gather | [`gather`] | [`SharedTile::gather_view`] → `.gather()` (ARow) / `.gather_transposed()` (BCol) |
//! | `(Reg, Global)` staged prefetch | [`prefetch`] | [`SharedTile::stage_view`] → [`LdsStage::prefetch`] |
//! | `(Lds, Reg)` staged commit | [`commit`] | [`SharedTile::stage_view`] → [`LdsStage::commit`] |
//! | `(Global, Reg)` epilogue scatter | [`scatter`] | [`crate::kernels::scatter_frag`] |
//!
//! [`LdsStage::prefetch`]: crate::tile_move::LdsStage::prefetch
//! [`LdsStage::commit`]: crate::tile_move::LdsStage::commit
//! [`SharedTile::gather_view`]: crate::tile_move::SharedTile::gather_view
//! [`SharedTile::stage_view`]: crate::tile_move::SharedTile::stage_view

use crate::build::{Buf, Builder, Effect, Elem, F32, Frag, Idx, Lds, Val};
use crate::ir::{FragMap, TileId};
use crate::kernels::{EDGE, add_opt, offset_by};
use crate::shape::MfmaShape;
use crate::tile::{RegLayout, Swizzle};

/// **`(Reg ← Lds)` — the operand gather.** Reads `Src`'s LDS run into the register operand fragments,
/// choosing `.gather()` (straight, ARow) or `.gather_transposed()` (BCol) **by the operand role `L`**:
/// the [`FragMap`](crate::ir::FragMap) `L` derives carries `transpose`, so the direction is a property
/// of the tile type, not a threaded flag. `n_frags` is `L::n_frags(tile_rows, tile_cols)`; the map and
/// `ept` come from `L`+`S`; `lds_cols` is the LDS tile's inner width. `slice` selects the K-run (inner
/// base `slice·EDGE`, via [`LdsView::slice`](crate::tile_move)) — `0` reads the tile's leading run (a
/// whole-tile probe), `s`/`kf` streams a multi-slice operand (FA's K d-slice / V kv-slice). `asm` is the
/// arch dispatch threaded onto the view: gfx942's waitcnt-opaque `ds_read_b64` gather vs the
/// compiler-visible intrinsic (a barrier's `lgkmcnt` auto-drains) — and it also gates the straight-vs-
/// transposed choice below (asm implies the straight contiguous gather, per the note there).
/// Returns one operand `Val` per fragment AND the store-fence tokens (the WAR barrier consumes them at
/// the pipeline commit; a whole-tile probe drops them).
#[allow(clippy::too_many_arguments)]
pub fn gather<E: Elem, L: RegLayout, S: MfmaShape>(
    b: &mut Builder,
    src: Lds<E>,
    lds_cols: usize,
    tile_rows: usize,
    tile_cols: usize,
    warp_off: Option<Idx>,
    lane: Idx,
    deps: &[TileId],
    slice: usize,
    asm: bool,
) -> (Vec<Val<E>>, Vec<TileId>) {
    let map = L::frag::<S>().expect("gather: Src must fill an operand tile (ARow/BCol), not an accumulator");
    let n_frags = L::n_frags::<S>(tile_rows, tile_cols);
    let view = SharedTile::new(src, lds_cols).gather_view(map, n_frags, warp_off, lane, asm).slice(slice);
    // The strided register-transpose gather (FA's V) is taken ONLY for a transposed map on the
    // compiler-visible path; the asm `ds_read_b64` reads a CONTIGUOUS run, so it serves the STRAIGHT
    // gather even for a transposed map (matmul's `Bᵀ`: a Col map that is K-contiguous in LDS —
    // `gather_asm` swaps the lane axes and reads it straight).
    if map.transpose && !asm { view.gather_transposed(b, deps) } else { view.gather(b, deps) }
}

/// **`(Reg ← Lds)` — the 32×32×8 contiguous-run operand gather** (the wide-MFMA sibling of [`gather`]).
/// Each `S::EPT_A`-wide fragment is ONE `ds_read_b64` of a CONTIGUOUS LDS run — the register transpose
/// the PV `V` needs is done write-side into LDS (see [`commit_run`]), so BOTH K and V read STRAIGHT here
/// (unlike [`gather`]'s per-element scalar/asm read + the transposed-`V` register transpose). The whole
/// address is derived from the tile TYPES: the operand [`FragMap`] + `EPT_A` from `S`, the
/// `(tile_rows/M)×(tile_cols/K)` fragment tiling with per-fragment tile-offset `(rt·M, ct·K)` from the
/// [`ARow`](crate::tile::ARow) role, and the swizzle (`lds_col` vs plain) from `Sw` — so K (`Xor`) reads
/// through the bank-swizzle hole and V (`Plain`, padded `inner` pitch) reads flat. `row`/`col` are the
/// lane partition `lane_rc(S::a_map(), lane, 0)` (supplied once by the kernel as `q_in`/`half_off`);
/// `parity` is the block-dependent double-buffer element offset (`0` when single-buffered). Returns one
/// operand `Val` and one store-fence token per fragment (the WAR-barrier contract, as [`gather`]).
#[allow(clippy::too_many_arguments)]
pub fn gather_run<E: Elem, Sw: Swizzle, S: MfmaShape>(
    b: &mut Builder,
    src: Lds<E>,
    inner: usize,
    tile_rows: usize,
    tile_cols: usize,
    row: Idx,
    col: Idx,
    parity: Idx,
    raw: &[TileId],
) -> (Vec<Val<E>>, Vec<TileId>) {
    let map = S::a_map();
    let swizzled = !Sw::layout(inner).transforms.is_empty();
    let inner_c = b.idx_const(inner as i64);
    let (rt_n, ct_n) = (tile_rows / S::M, tile_cols / S::K);
    let mut vecs = Vec::with_capacity(rt_n * ct_n);
    let mut gathers = Vec::with_capacity(rt_n * ct_n);
    for rt in 0..rt_n {
        for ct in 0..ct_n {
            let row_full = offset_by(b, row, rt * S::M); // rt·M (0 for a single-M-tile operand, e.g. K)
            let base = if swizzled {
                // Natural tile (K): col-offset → swizzle hole → row·inner (matches the hand-rolled order).
                let col_full = offset_by(b, col, ct * S::K);
                let col_part = b.lds_col(row_full, col_full, inner);
                let ri = b.idx_mul(row_full, inner_c);
                let base = b.idx_add(ri, col_part);
                b.idx_add(base, parity)
            } else {
                // Transposed padded tile (V): row·inner → col-offset (no swizzle; the pad breaks conflicts).
                let ri = b.idx_mul(row_full, inner_c);
                let col_full = offset_by(b, col, ct * S::K);
                let base = b.idx_add(ri, col_full);
                b.idx_add(base, parity)
            };
            // One `ds_read_b64` of the contiguous `EPT_A` run, round-tripped through a fragment so the WAR
            // barrier gets a proper store token (SROA elides the store/load into a straight operand).
            let v = b.load_lds_vec_after(src, base, S::EPT_A, raw);
            let frag = b.define_frag::<E>(map);
            let st = b.store_frag_vec(frag, v).dep();
            gathers.push(st);
            vecs.push(b.load_frag_vec_after(frag, &[st]));
        }
    }
    (vecs, gathers)
}

/// **`(Lds ← Reg)` — FA-32's fused K/V staged commit** (the wide-MFMA sibling of [`commit`]). Writes the
/// prefetched `gvec`-chunks of the next block into BOTH shared tiles in ONE interleaved pass: K NATURAL
/// `[kv, d]` through its swizzle hole (`SwK`), V TRANSPOSED to `[d, kv]` at the padded `vt_pitch`
/// (`SwV`). Both destinations decode the SAME source `[kv, d]` chunk (`flat → (kv, d_base)`), so the
/// index math is shared and the two stores stay interleaved (the emission the pipeline's RAW barrier
/// fences); the per-tile layout (swizzle / transpose / pitch) is DERIVED from the tile types. `k_lds`/
/// `vt_lds` are the caller's (already WAR-ordered) LDS handles; `k_parity`/`vt_parity` the block-dependent
/// double-buffer element offsets (`0` when single-buffered). Scalar `store_lds` (the coalesced write is a
/// fill refinement the barrier-bound kernel doesn't need). Returns the store effects the RAW barrier fences.
#[allow(clippy::too_many_arguments)]
pub fn commit_run<E: Elem, SwK: Swizzle, SwV: Swizzle>(
    b: &mut Builder,
    k_lds: Lds<E>,
    k_cols: usize,
    vt_lds: Lds<E>,
    vt_pitch: usize,
    k_chunks: &[Val<E>],
    v_chunks: &[Val<E>],
    epl: usize,
    tid: Idx,
    k_parity: Idx,
    vt_parity: Idx,
) -> Vec<Effect> {
    let swiz_k = !SwK::layout(k_cols).transforms.is_empty();
    let swiz_v = !SwV::layout(vt_pitch).transforms.is_empty();
    let gvec = if epl.is_multiple_of(8) { 8 } else { 4 };
    let epl_c = b.idx_const(epl as i64);
    let lane_epl = b.idx_mul(tid, epl_c);
    let d_c = b.idx_const(k_cols as i64);
    let pitch_c = b.idx_const(vt_pitch as i64);
    let mut effs = Vec::with_capacity(2 * epl);
    for cg in 0..epl / gvec {
        let flat = offset_by(b, lane_epl, cg * gvec);
        let kv = b.idx_div(flat, d_c);
        let d_base = b.idx_mod(flat, d_c);
        let kv_row = b.idx_mul(kv, d_c);
        let (kchunk, vchunk) = (k_chunks[cg], v_chunks[cg]);
        for j in 0..gvec {
            let d_idx = offset_by(b, d_base, j);
            // K natural [kv, d] through the swizzle hole.
            let k_col = if swiz_k { b.lds_col(kv, d_idx, k_cols) } else { d_idx };
            let k_dst = b.idx_add(kv_row, k_col);
            let k_dst = b.idx_add(k_dst, k_parity);
            let kval = b.vec_extract(kchunk, j);
            effs.push(b.store_lds(k_lds, k_dst, kval));
            // V transposed [d, kv] at the padded pitch.
            let v_row = b.idx_mul(d_idx, pitch_c);
            let v_col = if swiz_v { b.lds_col(d_idx, kv, vt_pitch) } else { kv };
            let v_dst = b.idx_add(v_row, v_col);
            let v_dst = b.idx_add(v_dst, vt_parity);
            let vval = b.vec_extract(vchunk, j);
            effs.push(b.store_lds(vt_lds, v_dst, vval));
        }
    }
    effs
}

/// **`(Reg ← Global)` — the register-staged prefetch.** Issues the coalesced global loads for the
/// `dst` LDS tile's share, staging them in VGPRs (no LDS write yet); pair with [`commit`]. `grow_stride`
/// is the global row stride, `epl` the per-lane element run, `origin` the tile's row base, `k_base` the
/// per-iteration K-column base. The staged chunks' layout mirrors `dst` (hence `lds_cols`), so both the
/// LDS destination and the global source are named. Forwards to [`LdsStage::prefetch`](crate::tile_move).
#[allow(clippy::too_many_arguments)]
pub fn prefetch<E: Elem>(
    b: &mut Builder,
    dst: Lds<E>,
    lds_cols: usize,
    src: Buf<E>,
    grow_stride: i64,
    epl: usize,
    lane: Idx,
    origin: Idx,
    k_base: Idx,
    order: &[TileId],
) -> Vec<Val<E>> {
    SharedTile::new(dst, lds_cols)
        .stage_view(src, epl, lane, origin, grow_stride, Drain::Intrinsic)
        .prefetch(b, k_base, order)
}

/// **`(Lds ← Reg)` — the staged commit.** Writes the [`prefetch`]ed VGPR `chunks` into `dst` LDS at the
/// swizzle-safe `ds_write` granularity, ordered after the WAR barrier `war`. The `src`/`grow_stride`/
/// `origin` args define the same stage [`prefetch`] built (they select no bytes on the write side); the
/// commit reads only `dst`/`lds_cols`/`epl`/`lane`. Compiler-visible ([`Drain::Intrinsic`] — an
/// `s_barrier` auto-drains its `lgkmcnt`). Forwards to [`LdsStage::commit`](crate::tile_move).
#[allow(clippy::too_many_arguments)]
pub fn commit<E: Elem>(
    b: &mut Builder,
    dst: Lds<E>,
    lds_cols: usize,
    src: Buf<E>,
    grow_stride: i64,
    epl: usize,
    lane: Idx,
    origin: Idx,
    chunks: &[Val<E>],
    war: &[TileId],
) -> Vec<Effect> {
    SharedTile::new(dst, lds_cols)
        .stage_view(src, epl, lane, origin, grow_stride, Drain::Intrinsic)
        .commit(b, chunks, war)
}

/// **`(Lds ← Reg)` — the waitcnt-opaque asm commit** (§5c). The [`Drain::Asm`] twin of [`commit`]: each
/// prefetched chunk is written with an `asm ds_write_b64` the RAW barrier can NOT auto-drain (HK's
/// waitcnt-opaque write — the drain PLACEMENT is the schedule, owned by the pipeline's `CommitDrain`, so
/// this op emits none). The `sideeffect` writes chain in program order via `prev0`: the clustered commit
/// threads A's last write into B's `prev0` so ONE later drain reaches BOTH (the A→B prev-chain). Same
/// `src`/`origin`/`grow_stride`/`epl`/`lane` stage as [`prefetch`]/[`commit`] (they select no bytes on
/// the write side — the commit reads only `dst`/`lds_cols`/`epl`/`lane`). Forwards to
/// [`LdsStage::commit_asm`](crate::tile_move).
#[allow(clippy::too_many_arguments)]
pub fn commit_asm<E: Elem>(
    b: &mut Builder,
    dst: Lds<E>,
    lds_cols: usize,
    src: Buf<E>,
    grow_stride: i64,
    epl: usize,
    lane: Idx,
    origin: Idx,
    chunks: &[Val<E>],
    war: &[TileId],
    prev0: Option<TileId>,
) -> Vec<Effect> {
    SharedTile::new(dst, lds_cols)
        .stage_view(src, epl, lane, origin, grow_stride, Drain::Asm)
        .commit_asm(b, chunks, war, prev0)
}

/// **`(Global ← Reg)` — the accumulator scatter epilogue.** Writes a register accumulator fragment back
/// to global via its `acc_dist` map. Forwards to the proven [`scatter_frag`](crate::kernels::scatter_frag)
/// (16×16×16 accumulator distribution). Not exercised by `atb_probe` (whose probe epilogue scatters raw
/// values element-wise); it completes the §3.2 table for the GEMM/FA epilogue migration (steps 4–5).
pub fn scatter(b: &mut Builder, acc: Frag<F32>, dst: Buf<F32>, base: Idx, row_stride: i64, lane: Idx) -> Vec<Effect> {
    crate::kernels::scatter_frag(b, acc, dst, base, row_stride, lane)
}

// ===========================================================================================
// The movement primitives (relocated verbatim from the former `movement.rs`, DESIGN.md §5c):
// the LDS↔register/global handles the vocabulary ops above forward to. Only `tile_move` uses
// them, so they stay crate-private here.
// ===========================================================================================

/// A **rich view over an LDS operand tile** — the movement handle. It carries, as DATA, everything
/// [`Self::gather`] needs to derive each fragment's LDS address, so the call site names none of it:
///
/// - `lds` — the shared-memory buffer (residency `Lds`); many views share one `Lds` (A vs B, and a
///   `.slice` per K-run).
/// - `map` — the operand's MFMA lane→(row,col) [`FragMap`] (A = Row, B/C = Col — drives `lane_rc`
///   and the operand `ept`).
/// - `n_frags` — the number of 16×16 fragments stacked along the tile's **outer** axis (matmul's
///   `ri`/`cj`; each fragment `i` sits at outer offset `i·EDGE`). `gather` returns one operand
///   `Val` per fragment.
/// - `inner` — the LDS tile's inner (column) width (`k_step`): the row stride of the flat layout.
/// - `warp_off` — the multi-warp wave's runtime outer (row for A / col for B) offset into the
///   shared tile; `None` on the single-warp path (kept byte-identical, no `+0`).
/// - `run` — the current K-slice's inner base (`s·EDGE`); selected by [`Self::slice`].
/// - `lane` — the intra-warp lane (`wlane`).
/// - `asm` — the arch/residency dispatch: gfx942 emits the `ds_read_b64` asm gather; `false` is the
///   compiler-visible scalar path (`VectorizePass` fuses it to `ds_read_b64` for RDNA/fallback).
///
/// It is a `Copy` VIEW: `.slice(s)` returns a fresh view over the same `Lds` at a new K-run — no
/// mutation, no allocation.
#[derive(Copy, Clone, Debug)]
pub(crate) struct LdsView<E: Elem> {
    lds: Lds<E>,
    map: FragMap,
    n_frags: usize,
    inner: usize,
    warp_off: Option<Idx>,
    run: usize,
    lane: Idx,
    asm: bool,
}

impl<E: Elem> LdsView<E> {
    /// Re-view the same LDS tile at K-slice `s` (inner base `s·EDGE`) — the `.slice(i, s)` selector.
    pub(crate) fn slice(self, s: usize) -> Self {
        LdsView { run: s * EDGE, ..self }
    }

    /// **Gather** this slice's `n_frags` operand fragments LDS→register, ordered after `raw` (the
    /// RAW barrier / the carried `[raw_seed, range]`). Returns the operand `Val`s (one per fragment,
    /// the WMMA operands) and the store-fence tokens the WAR barrier consumes. Dispatches on the
    /// view's `asm`: the `ds_read_b64 offset:N` asm gather (gfx942) or the scalar intrinsic path.
    pub(crate) fn gather(self, b: &mut Builder, raw: &[TileId]) -> (Vec<Val<E>>, Vec<TileId>) {
        if self.asm { self.gather_asm(b, raw) } else { self.gather_scalar(b, raw) }
    }

    /// The **scalar** (intrinsic/fallback) gather: per fragment, `ept` per-element
    /// `load_lds_after` then `store_frag_elem` at `outer·inner + LdsCol(outer, run+e)`, then one
    /// `load_frag_vec_after`. The per-element `LdsCol` is the composable hole `SwizzlePass` and
    /// `VectorizePass` refine (§5b).
    /// Subsumes `gather_frag_lds_run` (bit-for-bit — same nodes, same edges).
    fn gather_scalar(self, b: &mut Builder, raw: &[TileId]) -> (Vec<Val<E>>, Vec<TileId>) {
        let frags: Vec<Frag<E>> = (0..self.n_frags).map(|_| b.define_frag::<E>(self.map)).collect();
        self.gather_scalar_into(b, raw, &frags)
    }

    /// The **pooled** scalar gather: identical to [`Self::gather_scalar`] but stores into caller-owned,
    /// REUSED `slots` (a [`crate::schedule::TilePool`] phase) instead of minting a fresh fragment per
    /// slice — the compiler-visible counterpart to [`Self::gather_asm_into`]. This is the path a
    /// compiler-visible kernel (asm=false) MUST use for pooling: its `ds_read` is an intrinsic whose
    /// `lgkmcnt` the `s_barrier` auto-drains, unlike the waitcnt-opaque asm gather. `slots.len()` must
    /// equal the view's `n_frags`.
    pub(crate) fn gather_scalar_into(
        self,
        b: &mut Builder,
        raw: &[TileId],
        slots: &[Frag<E>],
    ) -> (Vec<Val<E>>, Vec<TileId>) {
        assert_eq!(slots.len(), self.n_frags, "TilePool slots must match the view's fragment count");
        let inner_c = b.idx_const(self.inner as i64);
        let mut gathers = Vec::new();
        let vecs = (0..self.n_frags)
            .map(|f| {
                let frag = slots[f];
                let stores: Vec<TileId> = (0..self.map.ept)
                    .map(|e| {
                        let e_idx = b.idx_const(e as i64);
                        let (frag_row, frag_col) = b.lane_rc(self.map, self.lane, e_idx);
                        let (outer_frag, run_frag) =
                            if self.map.transpose { (frag_col, frag_row) } else { (frag_row, frag_col) };
                        // fixed axis: intra-wave lane_rc + compile-time sub-tile base (fragment f)
                        // + (multi-warp) the wave's runtime offset into the shared LDS tile.
                        let outer = offset_by(b, outer_frag, f * EDGE);
                        let outer = add_opt(b, outer, self.warp_off);
                        let run = offset_by(b, run_frag, self.run);
                        let col_part = b.lds_col(outer, run, self.inner); // the swizzle/vectorise hole
                        let row_off = b.idx_mul(outer, inner_c);
                        let off = b.idx_add(row_off, col_part);
                        // `raw` = the fill RAW barrier (stages=1) or the carried `[raw_seed, range]`
                        // (stages=2) — the read observes the previous iteration's commit either way.
                        let v = b.load_lds_after(self.lds, off, raw);
                        b.store_frag_elem(frag, e_idx, v).dep()
                    })
                    .collect();
                let v = b.load_frag_vec_after(frag, &stores);
                gathers.extend(stores);
                v
            })
            .collect();
        (vecs, gathers)
    }

    /// The **transposed scalar gather** (FA's PV `V` operand — the "register transpose" the naive
    /// FA lacked): gather `n_frags` fragments whose CONTRACTION axis is the tile's LEADING (row) axis,
    /// stacking the output fragments along the tile's INNER (column) axis. For a `[kv, d]` V tile the
    /// gather reads `reg(lane, e) = tile[row = spread = (lane/16)·stride + e, col = f·EDGE + flat = lane%16]`,
    /// i.e. contraction `kv` lands on the MFMA spread (contraction) lane-axis and the output `d` on the
    /// flat lane-axis — exactly the layout an mma A/B operand needs to contract over the tile's row.
    ///
    /// This is the mirror of [`Self::gather_scalar`] with the tile `(row, col)` roles SWAPPED: there the
    /// contraction is the tile's trailing axis (`QKᵀ` over `d`); here it is the leading axis (`PV` over
    /// `kv`). The `ept` run is column-STRIDED (4 consecutive rows, `inner` elements apart), so it is a
    /// scalar 4-load gather — a `ds_read_b64` reads a contiguous run and cannot serve it (a perf, not a
    /// correctness, concern; Phase A is single-warp correctness). `self.map` must be the Col map so
    /// `lane_rc` yields `(spread, flat)`; `run` offsets the contraction (a kv-slice, 0 for a 16-row block).
    pub(crate) fn gather_transposed(self, b: &mut Builder, raw: &[TileId]) -> (Vec<Val<E>>, Vec<TileId>) {
        let inner_c = b.idx_const(self.inner as i64);
        let mut gathers = Vec::new();
        let vecs = (0..self.n_frags)
            .map(|f| {
                let frag = b.define_frag::<E>(self.map);
                // The `ept` column-strided scalar reads, packed into ONE `vec_build`→`store_frag_vec`
                // (a `StoreRegVec`, NOT `ept` scalar `store_frag_elem`s). Numerically bit-identical to
                // the per-element store — the same 4 loads in the same `e=0..3` order — but it presents
                // NO fusible `ept`-scalar-store run to [`crate::VectorizePass`], so the pass leaves this
                // strided gather alone (fusing it would mis-read `ept` contiguous LDS, corrupting V —
                // the reason FA formerly had to omit VectorizePass entirely) while STILL fusing the
                // straight K gather. The strided V read stays scalar `ds_read_u16`; K vectorises.
                let loaded: Vec<Val<E>> = (0..self.map.ept)
                    .map(|e| {
                        let e_idx = b.idx_const(e as i64);
                        // Col map ⇒ (frag_row = spread = contraction, frag_col = flat = output).
                        let (row, flat) = b.lane_rc(self.map, self.lane, e_idx);
                        let row = offset_by(b, row, self.run); // contraction-slice base (kv-slice; 0 here)
                        let col = offset_by(b, flat, f * EDGE); // output-fragment `f` on the column axis
                        let col = add_opt(b, col, self.warp_off);
                        let col_part = b.lds_col(row, col, self.inner); // swizzle hole (flat = col at base)
                        let row_off = b.idx_mul(row, inner_c);
                        let off = b.idx_add(row_off, col_part);
                        b.load_lds_after(self.lds, off, raw)
                    })
                    .collect();
                let packed = b.vec_build(&loaded);
                let st = b.store_frag_vec(frag, packed).dep();
                gathers.push(st);
                b.load_frag_vec_after(frag, &[st])
            })
            .collect();
        (vecs, gathers)
    }

    /// The **asm `ds_read_b64` gather** (gfx942 §5c — HK's only asm): all `n_frags` fragments differ
    /// by a COMPILE-TIME outer offset (fragment `i` at LDS row `i·EDGE`), so the lane's base LDS
    /// address is materialised **once** (`lane_rc(elem 0) + warp/run`, through [`Builder::lds_col`]
    /// so `SwizzlePass` folds the fragment-invariant XOR delta into that single base), then fragment
    /// `i` reads `ds_read_b64 $d, $base offset:(i·EDGE·inner·itemsize)`. ONE base VGPR + immediates
    /// replaces the per-fragment div/mod address the scalar path spills under the barrier walls.
    /// Subsumes `gather_frags_asm`.
    fn gather_asm(self, b: &mut Builder, raw: &[TileId]) -> (Vec<Val<E>>, Vec<TileId>) {
        let frags: Vec<Frag<E>> = (0..self.n_frags).map(|_| b.define_frag::<E>(self.map)).collect();
        self.gather_asm_into(b, raw, &frags)
    }

    /// The **pooled** asm gather: identical addressing/reads to [`Self::gather_asm`] but stores into
    /// caller-owned, REUSED `slots` (a [`crate::schedule::TilePool`] phase) instead of minting a fresh
    /// fragment per slice. Reusing fixed slots is what makes the read-ahead register working-set
    /// explicit and bounded: `raw` carries both the LDS-RAW carry AND the slot's recycle edge (the
    /// previous occupant's consuming MMA), so the base pointer — hence every `ds_read` — lands after
    /// the live operand it overwrites is drained. `slots.len()` must equal the view's `n_frags`.
    pub(crate) fn gather_asm_into(
        self,
        b: &mut Builder,
        raw: &[TileId],
        slots: &[Frag<E>],
    ) -> (Vec<Val<E>>, Vec<TileId>) {
        assert_eq!(slots.len(), self.n_frags, "TilePool slots must match the view's fragment count");
        // base LDS element offset at fragment 0, element 0: the lane's slot + the wave/run offset,
        // with the swizzle hole at `lds_col` (flat = `run`; SwizzlePass = `run ^ delta`).
        let zero = b.idx_const(0);
        let (frag_row, frag_col) = b.lane_rc(self.map, self.lane, zero);
        let (outer_frag, run_frag) = if self.map.transpose { (frag_col, frag_row) } else { (frag_row, frag_col) };
        let outer0 = add_opt(b, outer_frag, self.warp_off);
        let run0 = offset_by(b, run_frag, self.run);
        let inner_c = b.idx_const(self.inner as i64);
        let col_part = b.lds_col(outer0, run0, self.inner);
        let row_off = b.idx_mul(outer0, inner_c);
        let base_off = b.idx_add(row_off, col_part);
        // ONE base VGPR (addr(3) cast), After-wrapped by `raw` so the reads land past the RAW barrier.
        let base_ptr = b.lds_ptr_as3(self.lds, base_off, raw);

        let itemsize = E::dtype().bytes() as i64;
        let step_bytes = EDGE as i64 * self.inner as i64 * itemsize; // fragment-row `offset:` step
        let mut vecs = Vec::with_capacity(slots.len());
        let mut stores = Vec::with_capacity(slots.len());
        let mut prev: Option<TileId> = None;
        for (i, &f) in slots.iter().enumerate() {
            let off_bytes = i as i64 * step_bytes;
            let v: Val<E> = b.ds_read_b64(base_ptr, off_bytes, self.map.ept, prev);
            let st = b.store_frag_vec(f, v);
            prev = Some(st.dep());
            stores.push(st.dep());
            vecs.push(b.load_frag_vec_after(f, &[st.dep()]));
        }
        (vecs, stores)
    }
}

/// The **shared LDS operand tile** — the layout owner. It mints BOTH movement handles ([`LdsView`]
/// for the read/gather side, [`LdsStage`] for the write/commit side) from one place, so the fill and
/// the gather **cannot disagree** on the tile's `cols`/swizzle (a desync = silent bank corruption).
/// A `Copy` descriptor: many views share one tile (A vs B).
#[derive(Copy, Clone, Debug)]
pub(crate) struct SharedTile<E: Elem> {
    lds: Lds<E>,
    cols: usize, // the inner (column) width — `k_step`; the flat-layout row stride AND the swizzle period
}

impl<E: Elem> SharedTile<E> {
    pub(crate) fn new(lds: Lds<E>, cols: usize) -> Self {
        SharedTile { lds, cols }
    }

    /// Mint the **read/gather** view (the operand-fragment handle). `inner`/swizzle come from the tile.
    pub(crate) fn gather_view(
        self,
        map: FragMap,
        n_frags: usize,
        warp_off: Option<Idx>,
        lane: Idx,
        asm: bool,
    ) -> LdsView<E> {
        LdsView { lds: self.lds, map, n_frags, inner: self.cols, warp_off, run: 0, lane, asm }
    }

    /// Mint the **write/commit** stage (the collaborative global→LDS fill handle). `src` is the global
    /// source (K-contiguous `[M,K]`/`[N,K]`), `origin` the tile's M/N row base, `grow_stride` the
    /// global row stride (`K`); `epl` elements per lane. The K-column base is passed per iteration to
    /// [`LdsStage::prefetch`].
    pub(crate) fn stage_view(
        self,
        src: Buf<E>,
        epl: usize,
        lane: Idx,
        origin: Idx,
        grow_stride: i64,
        drain: Drain,
    ) -> LdsStage<E> {
        LdsStage { src, lds: self.lds, epl, lane, cols: self.cols as i64, origin, grow_stride, drain }
    }
}

/// The commit's **drain policy** (DESIGN §5c) — how the collaborative fill's LDS writes are made
/// visible before the RAW barrier. [`Drain::Intrinsic`] is the compiler-visible `ds_write` (an
/// `s_barrier` auto-drains its `lgkmcnt(0)`); [`Drain::Asm`] is HK's waitcnt-opaque `asm ds_write_b64`
/// (the barrier can NOT auto-drain it, so the caller drains manually via `swait_lgkmcnt` — the safe
/// foundation before a later step defers that drain to hide it).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Drain {
    Intrinsic,
    Asm,
}

/// A **rich view over the write/commit side** of a [`SharedTile`] — the movement handle symmetric to
/// [`LdsView`]. It carries, as DATA, the collaborative fill's addressing (`src`/`origin`/`grow_stride`
/// /`epl`/`lane`/`cols`), so the call site names none of it — the write-side answer to HK's
/// `load(st, gl)`. The fill is the trivial b128 `load_vec`→`ds_write_b64` (post-`[N,K]`-B: K-contiguous,
/// no transpose, no `v_perm`) shared by A and B.
///
/// The two halves mirror [`LdsView::gather`]'s split: [`Self::prefetch`] (global→VGPR, no ordering)
/// and [`Self::commit`] (VGPR→LDS, the WAR ordering ridden onto the LDS handle via `lds_after` — the
/// write-side analog of `gather`'s `raw` param). Returns the store-fence tokens the RAW barrier
/// consumes (the inverse of `gather`, which consumes the RAW and returns fence tokens).
#[derive(Copy, Clone, Debug)]
pub(crate) struct LdsStage<E: Elem> {
    src: Buf<E>,
    lds: Lds<E>,
    epl: usize,
    lane: Idx,
    cols: i64,
    origin: Idx,
    grow_stride: i64,
    drain: Drain,
}

impl<E: Elem> LdsStage<E> {
    /// The **prefetch** half (global→VGPR): issue the `epl/gvec` coalesced global `load_vec`s at the
    /// K-column base `k_base`, no LDS write. Global-load width is **b128** (8 elems, `buffer_load_dwordx4`,
    /// matching HK's `raw.buffer.load.i128`) when the lane run tiles it, else b64 — the wide load is
    /// split back into VEC-wide b64 chunks (register-only; the b128 lands in adjacent VGPRs) so the
    /// swizzle-safe [`Self::commit`] stores them independently. Returns the chunks in commit order (the
    /// commit's `r`/`c` addressing hash-cons-shares this half's nodes).
    pub(crate) fn prefetch(self, b: &mut Builder, k_base: Idx, order: &[TileId]) -> Vec<Val<E>> {
        const VEC: usize = 4; // b64 = 4 elems: the LDS-store / swizzle granularity
        let gvec = if self.epl.is_multiple_of(8) { 8 } else { VEC };
        assert!(
            self.epl.is_multiple_of(VEC) && (self.cols as usize).is_multiple_of(gvec),
            "vectorised fill needs VEC-aligned epl and gvec-aligned cols"
        );
        let epl_c = b.idx_const(self.epl as i64);
        let lane_epl = b.idx_mul(self.lane, epl_c);
        let cols_c = b.idx_const(self.cols);
        let gstride = b.idx_const(self.grow_stride);
        // The MUBUF split (§DRAM-prefetch — HK's `buffer_load` over FLAT `global_load`): the buffer
        // descriptor + the workgroup-uniform byte soffset are hoisted/uniform (SGPR base, advanced by a
        // scalar `s_add` = the K-tile stride); each chunk's per-lane byte voffset is K-invariant (hoisted
        // VGPR). soffset + voffset reproduce the FLAT `goff = (origin+r)·K + k_base + c` (in bytes), but
        // with NO per-iteration 64-bit VGPR address — the load no longer parks on a `v_add` dependency.
        let item_c = b.idx_const(E::dtype().bytes() as i64);
        // MUBUF prefetch (HK's `buffer_load` over FLAT `global_load`) with the **advancing base** (HK's
        // exact scheme): fold the workgroup-uniform `(origin·K + k_base)` into the descriptor base —
        // uniform + loop-variant ⇒ an SGPR advanced per K-tile by a scalar `s_add`, leaving
        // `voffset = (r·K + c)` per-lane and LOOP-INVARIANT (no per-iteration 64-bit VGPR address, so the
        // load never parks on a `v_add` dependency — the +20% over a fixed base with the whole offset in
        // `voffset`). Safe under the ping-pong overlap because the asm-opaque gather/commit pins the LDS
        // reads and the commit's last-gather `lgkmcnt(0)` read-drain completes them before the single-buffer
        // overwrite (the compiler-visible variant that could not honour that ordering was retired).
        let orig_k = b.idx_mul(self.origin, gstride); // origin·K
        let base_off = b.idx_add(orig_k, k_base); // origin·K + k_base — uniform ⇒ SGPR s_add
        let rsrc = b.make_buffer_rsrc(self.src, base_off);
        let mut out = Vec::with_capacity(self.epl / VEC);
        for cg in 0..self.epl / gvec {
            let ec = b.idx_const((cg * gvec) as i64);
            let flat = b.idx_add(lane_epl, ec); // gvec-aligned chunk start (stays in one row)
            let r = b.idx_div(flat, cols_c);
            let c = b.idx_mod(flat, cols_c);
            // Within-tile per-lane element offset r·K + c (loop-invariant; origin+k_base ride the base).
            let rk = b.idx_mul(r, gstride);
            let goff = b.idx_add(rk, c);
            let voff = b.idx_mul(goff, item_c);
            let wide = b.buffer_load_raw(rsrc, voff, gvec, order); // ONE b128 (or b64) MUBUF load
            for h in 0..gvec / VEC {
                if gvec == VEC {
                    out.push(wide);
                } else {
                    let half: Vec<Val<E>> = (0..VEC).map(|e| b.vec_extract(wide, h * VEC + e)).collect();
                    out.push(b.vec_build(&half));
                }
            }
        }
        out
    }

    /// The **commit** half (VGPR→LDS): `ds_write_b64` each prefetched chunk into LDS at
    /// `r·cols + LdsCol(r, c)` (swizzle-safe b64 granularity). `war` is the WAR barrier the writes must
    /// observe — ridden onto the LDS handle ONCE via `lds_after` (empty on the prologue block-0 commit,
    /// kept byte-identical). Returns the store effects the RAW barrier fences.
    pub(crate) fn commit(self, b: &mut Builder, loaded: &[Val<E>], war: &[TileId]) -> Vec<Effect> {
        match self.drain {
            Drain::Intrinsic => self.commit_intrinsic(b, loaded, war),
            Drain::Asm => self.commit_asm(b, loaded, war, None),
        }
    }

    /// The compiler-visible commit (`store_lds_vec` → `ds_write`): an `s_barrier` auto-drains its
    /// `lgkmcnt(0)`. Returns the store effects the RAW barrier fences. Byte-identical to the original.
    fn commit_intrinsic(self, b: &mut Builder, loaded: &[Val<E>], war: &[TileId]) -> Vec<Effect> {
        const VEC: usize = 4;
        let lds = if war.is_empty() { self.lds } else { b.lds_after(self.lds, war) };
        let epl_c = b.idx_const(self.epl as i64);
        let lane_epl = b.idx_mul(self.lane, epl_c);
        let cols_c = b.idx_const(self.cols);
        (0..self.epl / VEC)
            .map(|cc| {
                let ec = b.idx_const((cc * VEC) as i64);
                let flat = b.idx_add(lane_epl, ec);
                let r = b.idx_div(flat, cols_c);
                let c = b.idx_mod(flat, cols_c);
                let col = b.lds_col(r, c, self.cols as usize);
                let rc = b.idx_mul(r, cols_c);
                let dst_off = b.idx_add(rc, col);
                b.store_lds_vec(lds, dst_off, loaded[cc])
            })
            .collect()
    }

    /// The **asm commit** (§5c — HK's waitcnt-opaque `asm ds_write_b64`): SAME `epl/VEC` chunk loop and
    /// SAME `flat`/`r`/`c`/`lds_col`/`dst_off` addressing as [`Self::commit_intrinsic`] (hash-cons-shared
    /// with the intrinsic path's index nodes), but each store is an `asm ds_write_b64` from ONE base
    /// pointer (`lds_ptr_as3`, `off_bytes=0`). The `sideeffect` writes chain in program order via `prev`
    /// (seeded by `prev0` — the caller threads A's tail into B so BOTH survive one drain). The RAW
    /// barrier can NOT auto-drain these — the caller emits ONE `swait_lgkmcnt` on the last write instead.
    pub(crate) fn commit_asm(
        self,
        b: &mut Builder,
        loaded: &[Val<E>],
        war: &[TileId],
        prev0: Option<TileId>,
    ) -> Vec<Effect> {
        const VEC: usize = 4;
        let lds = if war.is_empty() { self.lds } else { b.lds_after(self.lds, war) };
        let epl_c = b.idx_const(self.epl as i64);
        let lane_epl = b.idx_mul(self.lane, epl_c);
        let cols_c = b.idx_const(self.cols);
        let mut prev = prev0;
        (0..self.epl / VEC)
            .map(|cc| {
                let ec = b.idx_const((cc * VEC) as i64);
                let flat = b.idx_add(lane_epl, ec);
                let r = b.idx_div(flat, cols_c);
                let c = b.idx_mod(flat, cols_c);
                let col = b.lds_col(r, c, self.cols as usize);
                let rc = b.idx_mul(r, cols_c);
                let dst_off = b.idx_add(rc, col);
                let base_ptr = b.lds_ptr_as3(lds, dst_off, &[]);
                let w = b.ds_write_b64(base_ptr, 0, loaded[cc], prev);
                prev = Some(w.dep());
                w
            })
            .collect()
    }
}
