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

use crate::build::{Builder, Elem, Frag, Idx, Lds, Val};
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
    /// Construct a view over the `lds` operand tile: `n_frags` fragments along the outer axis, the
    /// LDS tile `inner`-wide, addressed by `lane` (+ the wave's `warp_off`), with the arch dispatch
    /// `asm`. The K-run starts at slice 0; select another with [`Self::slice`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        lds: Lds<E>,
        map: FragMap,
        n_frags: usize,
        inner: usize,
        warp_off: Option<Idx>,
        lane: Idx,
        asm: bool,
    ) -> Self {
        LdsView { lds, map, n_frags, inner, warp_off, run: 0, lane, asm }
    }

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
        let inner_c = b.idx_const(self.inner as i64);
        let mut gathers = Vec::new();
        let vecs = (0..self.n_frags)
            .map(|f| {
                let frag = b.define_frag::<E>(self.map);
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
        let mut vecs = Vec::with_capacity(frags.len());
        let mut stores = Vec::with_capacity(frags.len());
        let mut prev: Option<TileId> = None;
        for (i, &f) in frags.iter().enumerate() {
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
