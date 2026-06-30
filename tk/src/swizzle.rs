//! Shared-tile (ST) swizzles.
//!
//! A swizzle remaps the `(row, col)` of an element within a base tile to avoid
//! LDS bank conflicts. The MVP path uses [`Swizzle::Identity`] (`ST_16X16`); the
//! XOR variants port the HipKittens scheme (`st.cuh:88-97`): the element's byte
//! offset within the tile is XORed with `((addr % repeat) >> 7) << 3`, a
//! bijection applied identically on every LDS store and load, so it never
//! changes the numeric result — it only re-lays-out the banks.

use std::sync::Arc;

use svod_dtype::ScalarDType;
use svod_ir::UOp;

use crate::index::cidx;

/// The five predefined ST base-tile swizzles (see tinygrad `tiles.py`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Swizzle {
    /// No remapping (`ST_16X16`).
    Identity,
    /// `ST_16X16_SWIZZLED` (bf16 XOR).
    Sw16x16,
    /// `ST_32X32` (bf16 double XOR).
    Sw32x32,
    /// `ST_16X32` (bf16 XOR).
    Sw16x32,
    /// `ST_32X16` (bf16 XOR).
    Sw32x16,
}

/// HipKittens `swizzle_bytes` (`st.cuh:74-86`): the bank-conflict-avoiding XOR
/// period in bytes, selected from the tile's underlying width (in 16-col
/// fragments) and element size. `cols` is the (per-fragment) column count.
///
/// # Panics
/// `itemsize` must be 1/2/4 bytes — i.e. a swizzled shared (`ST`) tile's dtype
/// must be a 1/2/4-byte type (bf16/f16/f32 in practice). An 8-byte element
/// (f64/i64) panics; this is a kernel-authoring precondition, the USE-face
/// kernels only allocate bf16/f32 LDS tiles.
fn swizzle_bytes(cols: usize, itemsize: i64) -> i64 {
    let uw = cols / 16; // underlying width in 16-col tiles
    match itemsize {
        1 | 2 => {
            if uw.is_multiple_of(4) {
                128
            } else if uw.is_multiple_of(2) {
                64
            } else {
                32
            }
        }
        4 => {
            if uw.is_multiple_of(2) {
                128
            } else {
                64
            }
        }
        other => panic!("swizzle: unsupported itemsize {other} (bf16/f32 only)"),
    }
}

impl Swizzle {
    /// The XOR swizzle period in bytes for a `cols`-wide / `itemsize`-byte base
    /// tile (`None` for [`Swizzle::Identity`], which has no period). Used by the
    /// vectorized fill; asserts 16-byte group alignment.
    pub(crate) fn period_bytes(&self, cols: usize, itemsize: i64) -> Option<i64> {
        match self {
            Swizzle::Identity => None,
            _ => Some(swizzle_bytes(cols, itemsize)),
        }
    }

    /// The subtile column width (`swizzle_bytes / itemsize`) of the whole-tile
    /// layout — the contiguous column block, and (×`base_rows`) the element stride
    /// between adjacent fragment-rows. Because the XOR delta depends only on
    /// `row % 16` (invariant under a +16 row step) and the swizzled column part is
    /// always `< subtile_cols`, `tile_offset(row+16, col) == tile_offset(row, col)
    /// + 16*subtile_cols` exactly — a lane-uniform constant the gather lifts into
    /// the `ds_read offset:` immediate. `None` for [`Swizzle::Identity`].
    pub(crate) fn subtile_cols(&self, cols: usize, scalar: ScalarDType) -> Option<i64> {
        let is = scalar.bytes() as i64;
        match self {
            Swizzle::Identity => None,
            _ => Some(swizzle_bytes(cols, is) / is),
        }
    }

    /// Map a WHOLE-TILE in-tile `(row, col)` (`0 ≤ row < rows`, `0 ≤ col < cols`,
    /// the full ST tile dims) to the flat element offset within the LDS tile,
    /// applying HipKittens' subtile-structured XOR bank swizzle (`st.cuh:88-104`).
    ///
    /// The tile is laid out as `subtile_cols`-wide column subtiles, each stored
    /// `rows × subtile_cols` contiguous; the element's byte address inside that
    /// layout is XORed with `((addr % repeat) >> 7) << 3`. Computing the swizzle
    /// over the *whole-tile* `subtile_cols`-wide address (not a 16-col base
    /// fragment) is what spreads the gfx942 LDS banks: every row gets a distinct
    /// XOR delta, so the per-warp MFMA gather hits 32 distinct banks instead of
    /// collapsing rows `r, r+4, …` onto the same bank.
    ///
    /// A bijection on `[0,rows)×[0,cols)`, applied identically on every LDS store
    /// and load, so it never changes the numeric result — it only re-lays-out the
    /// banks. [`Swizzle::Identity`] is handled by the caller (plain fragment-major
    /// layout) and is not a valid receiver here.
    ///
    /// # Panics
    /// Panics on [`Swizzle::Identity`], or if the scalar itemsize is not 1/2/4
    /// bytes (only bf16/f16/f32 LDS tiles are swizzled).
    pub fn tile_offset(&self, row: Arc<UOp>, col: Arc<UOp>, rows: usize, cols: usize, scalar: ScalarDType) -> Arc<UOp> {
        assert!(!matches!(self, Swizzle::Identity), "tile_offset: Identity uses the caller's plain layout");
        let itemsize = scalar.bytes() as i64;
        let sb = swizzle_bytes(cols, itemsize);
        let subtile = sb / itemsize; // subtile_cols (st.cuh:104)
        let repeat = sb << 4; // swizzle_repeat (st.cuh:87)
        debug_assert_eq!(cols as i64 % subtile, 0, "tile_offset: cols {cols} not a multiple of subtile {subtile}");
        // Subtile-major element address: `addr = (col/subtile)*rows*subtile +
        // row*subtile + col%subtile` (st.cuh:101). When the tile is exactly one
        // subtile wide (`cols == subtile`, the common K_STEP=64 gemm operand) the
        // outer index is statically 0 and `col%subtile == col`, so we drop that
        // whole sub-expression — keeping the per-`ds_read` address arithmetic in
        // the unrolled MFMA cluster cheap (the compiler can't prove `col < subtile`
        // at runtime, so it would otherwise emit the divide/mul/mask).
        let subtile_u = cidx(subtile);
        if cols as i64 == subtile {
            // Single-subtile-wide tile (the K_STEP=64 gemm operand): `outer == 0`,
            // `col < subtile`, and `row*subtile` is a `subtile` multiple, so the
            // byte address `row*sb + col*itemsize` has `col*itemsize < sb`. The
            // swizzle's `>>7` therefore sees ONLY the `row` term — the XOR delta
            // `(((row%16)*sb >> 7) << 3)/itemsize` depends on `row` alone and is
            // `< subtile`, so `addr = row*subtile + (col ^ delta)` (no carry into
            // the row bits). Folding the XOR into `col` keeps `delta` loop-invariant
            // (it hoists out of the unrolled MFMA cluster) and collapses the live
            // range, avoiding the per-`ds_read` address recompute + register spills.
            let r16 = row.mod_(&cidx(16));
            let dbytes = r16.mul(&cidx(subtile * itemsize)).mod_(&cidx(repeat)).shr(&cidx(7)).shl(&cidx(3));
            let delta = dbytes.idiv(&cidx(itemsize));
            return row.mul(&subtile_u).add(&col.xor(&delta));
        }
        // General (multi-subtile) layout: `addr = (col/subtile)*rows*subtile +
        // row*subtile + col%subtile` (st.cuh:101). XOR its byte address per
        // st.cuh:103 and divide back to elements (the delta is a multiple of
        // `itemsize`, so the element-space XOR equals the byte-space one).
        let outer = col.idiv(&subtile_u);
        let addr = outer.mul(&cidx(rows as i64 * subtile)).add(&row.mul(&subtile_u)).add(&col.mod_(&subtile_u));
        let sw_bytes = addr.mul(&cidx(itemsize)).mod_(&cidx(repeat)).shr(&cidx(7)).shl(&cidx(3));
        addr.xor(&sw_bytes.idiv(&cidx(itemsize)))
    }
}
