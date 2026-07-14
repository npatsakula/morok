//! **The movement half of the tile-op vocabulary** (`scratchpad/tile_layer_design.md` §3) — the
//! residency-dispatched `load`/`store` ops. Each op is keyed on a `(Dst, Src)` **residency pair** and
//! is a faithful, behaviour-preserving forward to the proven [`crate::movement`] primitive that already
//! performs that transfer; the STATIC choices (operand map, `n_frags` geometry, `ept`, swizzle, the
//! straight-vs-transposed gather) are DERIVED from the tile TYPES ([`RegLayout`]/[`Swizzle`]/
//! [`MfmaShape`], exactly as [`crate::tile_ops`] derives its compute widths from `S`), while the
//! genuinely-runtime addressing (lane, origin, base, `epl`, ordering deps) stays as explicit params.
//!
//! It mirrors [`crate::tile_ops`]'s style (thin, marker-generic free functions over the raw handles),
//! NOT the const-generic [`crate::tile::Tile`] handle: the probes/kernels this surface serves are
//! *runtime*-shaped (`atb_probe(kv, d, q)`), so the dims ride as values and the type only carries the
//! shape/role/swizzle. No addressing/swizzle/asm logic is re-implemented here — that all lives in
//! [`crate::movement`] and [`crate::build`], which these ops call unchanged.
//!
//! ## The §3.2 residency table (design), and what each op forwards to
//!
//! | `(Dst::RES, Src::RES)` | op | forwards to |
//! |---|---|---|
//! | `(Reg, Lds)` gather | [`gather`] | [`SharedTile::gather_view`] → `.gather()` (ARow) / `.gather_transposed()` (BCol) |
//! | `(Reg, Global)` staged prefetch | [`prefetch`] | [`SharedTile::stage_view`] → [`LdsStage::prefetch`] |
//! | `(Lds, Reg)` staged commit | [`commit`] | [`SharedTile::stage_view`] → [`LdsStage::commit`] |
//! | `(Lds, Global)` direct-to-LDS | [`fill_direct`] | [`Builder::buffer_load_lds`] (Plain-only, [`DirectFill`]) |
//! | `(Global, Reg)` epilogue scatter | [`scatter`] | [`crate::kernels::scatter_frag`] |
//!
//! [`LdsStage::prefetch`]: crate::movement::LdsStage::prefetch
//! [`LdsStage::commit`]: crate::movement::LdsStage::commit
//! [`SharedTile::gather_view`]: crate::movement::SharedTile::gather_view
//! [`SharedTile::stage_view`]: crate::movement::SharedTile::stage_view

use crate::build::{Buf, Builder, Effect, Elem, F32, Frag, Idx, Lds, Val};
use crate::ir::TileId;
use crate::movement::{Drain, SharedTile};
use crate::shape::MfmaShape;
use crate::tile::{Plain, RegLayout, Swizzle};

/// The **residency-move legality gate** for the one pair the handle types don't already constrain
/// (design §3.2's `MovePath` bound). Every other pair is forbidden structurally — an op that expects
/// an [`Lds`] cannot be handed a [`Buf`]. The exception is the direct-to-LDS fill: `buffer_load…lds`
/// pins lane `L → m0 + L·4`, so it CANNOT apply a per-lane bank XOR on the fill side (this session's
/// hardware finding). Only [`Plain`] implements this trait, so a [`fill_direct`] over a swizzled
/// ([`crate::tile::Xor`]) tile does not compile.
pub trait DirectFill: Swizzle {}
impl DirectFill for Plain {}

/// **`(Reg ← Lds)` — the operand gather.** Reads `Src`'s LDS run into the register operand fragments,
/// choosing `.gather()` (straight, ARow) or `.gather_transposed()` (BCol) **by the operand role `L`**:
/// the [`FragMap`](crate::ir::FragMap) `L` derives carries `transpose`, so the direction is a property
/// of the tile type, not a threaded flag. `n_frags` is `L::n_frags(tile_rows, tile_cols)`; the map and
/// `ept` come from `L`+`S`; `lds_cols` is the LDS tile's inner width. `slice` selects the K-run (inner
/// base `slice·EDGE`, via [`LdsView::slice`](crate::movement)) — `0` reads the tile's leading run (a
/// whole-tile probe), `s`/`kf` streams a multi-slice operand (FA's K d-slice / V kv-slice). The
/// compiler-visible (`asm = false`) path — the intrinsic gather a barrier's `lgkmcnt` auto-drains.
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
) -> (Vec<Val<E>>, Vec<TileId>) {
    let map = L::frag::<S>().expect("gather: Src must fill an operand tile (ARow/BCol), not an accumulator");
    let n_frags = L::n_frags::<S>(tile_rows, tile_cols);
    let view = SharedTile::new(src, lds_cols).gather_view(map, n_frags, warp_off, lane, false).slice(slice);
    if map.transpose { view.gather_transposed(b, deps) } else { view.gather(b, deps) }
}

/// **`(Reg ← Global)` — the register-staged prefetch.** Issues the coalesced global loads for the
/// `dst` LDS tile's share, staging them in VGPRs (no LDS write yet); pair with [`commit`]. `grow_stride`
/// is the global row stride, `epl` the per-lane element run, `origin` the tile's row base, `k_base` the
/// per-iteration K-column base. The staged chunks' layout mirrors `dst` (hence `lds_cols`), so both the
/// LDS destination and the global source are named. Forwards to [`LdsStage::prefetch`](crate::movement).
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
/// `s_barrier` auto-drains its `lgkmcnt`). Forwards to [`LdsStage::commit`](crate::movement).
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

/// **`(Lds ← Global)` — the direct-to-LDS fill (register diet).** One `raw.ptr.buffer.load.lds` DMA
/// straight into LDS, NO intermediate VGPR — the register-bypass twin of [`prefetch`]+[`commit`]. Only
/// legal for a [`Plain`] tile (the [`DirectFill`] bound): the hardware pins lane `L → m0 + L·4`, so a
/// swizzled direct fill can't be authored. `rsrc`/`voffset`/`lds_dst` are the caller's runtime
/// addressing (buffer descriptor, per-lane byte offset, per-wave LDS base); the dword-granular width is
/// derived from `E`. Forwards to [`Builder::buffer_load_lds`].
pub fn fill_direct<E: Elem, Sw: DirectFill>(
    b: &mut Builder,
    rsrc: Idx,
    voffset: Idx,
    lds_dst: Idx,
    order: &[TileId],
) -> Effect {
    let ept = 4 / E::dtype().bytes(); // dword-granular: ept·sizeof(E) == 4
    b.buffer_load_lds::<E>(rsrc, voffset, lds_dst, ept, order)
}

/// **`(Global ← Reg)` — the accumulator scatter epilogue.** Writes a register accumulator fragment back
/// to global via its `acc_dist` map. Forwards to the proven [`scatter_frag`](crate::kernels::scatter_frag)
/// (16×16×16 accumulator distribution). Not exercised by `atb_probe` (whose probe epilogue scatters raw
/// values element-wise); it completes the §3.2 table for the GEMM/FA epilogue migration (steps 4–5).
pub fn scatter(b: &mut Builder, acc: Frag<F32>, dst: Buf<F32>, base: Idx, row_stride: i64, lane: Idx) -> Vec<Effect> {
    crate::kernels::scatter_frag(b, acc, dst, base, row_stride, lane)
}
