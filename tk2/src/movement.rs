//! The **tile-view movement layer** (DESIGN.md §5c): the LDS→register fragment gather as
//! a method on a rich handle that carries its addressing CONTEXT as data, replacing the
//! free-function param-scatter (`gather_frag_lds_run` / `gather_frags_asm` and their
//! `outer_base`/`warp_off`/`run_base`/`inner`/`lane` parameters threaded at every call site).
//!
//! This is svod's answer to HipKittens' `load(rt, st)`: two typed handles, ZERO addressing
//! params, because the tile carries its layout+position and the op derives the address. Per
//! DESIGN §OPEN-2 ("gentle typing") the shape/layout/residency ride as **data on the handle**,
//! never as compile-time type parameters — so an [`LdsView`] is a plain `Copy` struct, and a
//! K-slice or an arch choice is a field, not a monomorphisation.
//!
//! [`LdsView::gather`] DISPATCHES on the view's residency+arch (the "arch is a trait over the
//! ~18% that varies" rule, §2.8): on gfx942 it emits the `ds_read_b64 offset:N` asm gather (ONE
//! base VGPR + a per-fragment immediate — the addressing-VGPR collapse that stops the spill), and
//! the scalar-then-`VectorizePass` intrinsic path is the RDNA/fallback. The **same** method serves
//! matmul's A/B operands and (by construction — a different tile shape/origin) FA's K/V gather.

use crate::build::{Buf, Builder, Effect, Elem, Frag, Idx, Lds, Val};
use crate::ir::{FragMap, TileId};
use crate::kernels::{EDGE, add_opt, offset_by};

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

    /// Gather into caller-owned `slots` (a [`crate::schedule::TilePool`] phase), dispatching on the
    /// view's `asm` exactly as [`Self::gather`] does — so a pooled gather takes the SAME path (asm vs
    /// compiler-visible) the kernel is built for. Returns operands + store-fence tokens.
    pub(crate) fn gather_into(self, b: &mut Builder, raw: &[TileId], slots: &[Frag<E>]) -> (Vec<Val<E>>, Vec<TileId>) {
        if self.asm { self.gather_asm_into(b, raw, slots) } else { self.gather_vec_into(b, raw, slots) }
    }

    /// The **pooled vector gather** (compiler-visible): emits directly what `gather_scalar` +
    /// [`crate::VectorizePass`] would fuse to — ONE `LoadVecAt` (intrinsic `ds_read_b64`) per fragment
    /// from the e=0 element's swizzled base, into a caller-owned REUSED slot. The pool MUST emit the
    /// vector form up front because `VectorizePass`'s fusion needs exactly `ept` stores per fragment
    /// buffer, which a reused slot (written by several dot-slices) violates — so the fuse-later path
    /// silently declines and falls back to scalar. `slots.len()` must equal the view's `n_frags`.
    pub(crate) fn gather_vec_into(
        self,
        b: &mut Builder,
        raw: &[TileId],
        slots: &[Frag<E>],
    ) -> (Vec<Val<E>>, Vec<TileId>) {
        assert_eq!(slots.len(), self.n_frags, "TilePool slots must match the view's fragment count");
        // Fragment 0's swizzled LDS base. Every fragment sits `f·EDGE` rows further down the SAME
        // column, and the swizzle delta (a function of `row % 16`) is identical for all of them (they
        // are EDGE=16 rows apart), so fragment f's base is `base0 + f·EDGE·inner` — a COMPILE-TIME
        // offset LLVM folds into the `ds_read offset:` immediate. That collapses the per-fragment
        // address to ONE base VGPR (HK's asm-gather addressing, but compiler-visible), the spill cure
        // the over-read needs, instead of a live base register per fragment.
        let zero = b.idx_const(0);
        let (frag_row, frag_col) = b.lane_rc(self.map, self.lane, zero);
        let (outer_frag, run_frag) = if self.map.transpose { (frag_col, frag_row) } else { (frag_row, frag_col) };
        let outer0 = add_opt(b, outer_frag, self.warp_off);
        let run0 = offset_by(b, run_frag, self.run);
        let inner_c = b.idx_const(self.inner as i64);
        let col_part = b.lds_col(outer0, run0, self.inner);
        let row_off = b.idx_mul(outer0, inner_c);
        let base0 = b.idx_add(row_off, col_part);
        let step = EDGE as i64 * self.inner as i64; // fragment row stride (elements)
        let mut stores = Vec::with_capacity(slots.len());
        let vecs = (0..self.n_frags)
            .map(|f| {
                let frag = slots[f];
                let base = if f == 0 {
                    base0
                } else {
                    let off = b.idx_const(f as i64 * step);
                    b.idx_add(base0, off)
                };
                let vec = b.load_lds_vec_after(self.lds, base, self.map.ept, raw);
                let st = b.store_frag_vec(frag, vec);
                stores.push(st.dep());
                b.load_frag_vec_after(frag, &[st.dep()])
            })
            .collect();
        (vecs, stores)
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
