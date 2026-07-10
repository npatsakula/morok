//! HipKittens tile types (`types/shared/st.cuh`, `types/register/rt.cuh`) as thin tk2 handles.
//! Shapes/layouts ride as DATA (tk2's "gentle typing"), so `st_bf`/`rt_bf`/`rt_fl` are plain structs,
//! not const-generic monomorphisations — but they keep HK's names + the same `height`/`width` geometry.

#![allow(non_camel_case_types)]

use crate::build::{BF16, Builder, F32, Frag, Idx, Lds};
use crate::ir::FragMap;

/// A **shared (LDS) tile** — HK's `st_bf<H, W>` (`st.cuh:261`), backed by a tk2 `DefineLocal`.
/// `rows`/`cols` are the tile geometry; `SharedTile`-style swizzle addressing rides on `cols`.
#[derive(Copy, Clone, Debug)]
pub struct st_bf {
    pub lds: Lds<BF16>,
    pub rows: usize,
    pub cols: usize,
}

impl st_bf {
    /// `al.allocate<st_bf<rows, cols>>()` — one LDS allocation (the bump-allocated `As`/`Bs`).
    pub fn new(b: &mut Builder, rows: usize, cols: usize) -> Self {
        let lds = b.define_local::<BF16>(rows * cols);
        st_bf { lds, rows, cols }
    }

    /// `subtile_inplace<sub_rows, sub_cols>(self, {blk_row, blk_col})` (`conversions.cuh:51`) — a
    /// pure address view selecting the `sub_rows × sub_cols` sub-tile at block offset
    /// `{blk_row·sub_rows, blk_col·sub_cols}`. No hardware op (index arithmetic only).
    pub fn subtile_inplace(self, sub_rows: usize, sub_cols: usize, blk_row: usize, blk_col: usize) -> st_subtile {
        st_subtile { parent: self, sub_rows, sub_cols, blk_row, blk_col, blk_row_dyn: None }
    }

    /// `subtile_inplace<sub_rows, sub_cols>(self, {warp_row, blk_col})` with a **runtime** row-block —
    /// HK's `micro_tk` selects each warp's subtile at `{warp_row, slice}` / `{warp_col, slice}`, and
    /// `warp_row`/`warp_col` are per-warp runtime values (`warpid()`-derived), not compile-time. The
    /// row offset becomes `blk_row · sub_rows` (uniform per warp); `blk_col` (the K-slice) stays
    /// compile-time. Additive over [`Self::subtile_inplace`] (whose `blk_row_dyn` is `None`).
    pub fn subtile_inplace_dyn(self, sub_rows: usize, sub_cols: usize, blk_row: Idx, blk_col: usize) -> st_subtile {
        st_subtile { parent: self, sub_rows, sub_cols, blk_row: 0, blk_col, blk_row_dyn: Some(blk_row) }
    }
}

/// A **shared-tile sub-view** — HK's `st_subtile<ST, sub_rows, sub_cols>` (`st.cuh:158`). Carries the
/// parent tile + the block offsets; addressing uses the parent's `underlying_rows`/`underlying_cols`
/// stride (`st.cuh:210`). `row_offset = blk_row·sub_rows`, `col_offset = blk_col·sub_cols`.
#[derive(Copy, Clone, Debug)]
pub struct st_subtile {
    pub parent: st_bf,
    pub sub_rows: usize,
    pub sub_cols: usize,
    pub blk_row: usize,
    pub blk_col: usize,
    /// A per-warp **runtime** row-block index (`warp_row`/`warp_col`), added as `blk_row_dyn·sub_rows`
    /// to the compile-time `row_offset` by [`crate::hk::memory::load`]. `None` for the compile-time-
    /// only [`st_bf::subtile_inplace`] (byte-identical to before this field existed).
    pub blk_row_dyn: Option<Idx>,
}

impl st_subtile {
    /// The parent tile's inner (column) width — the flat-layout row stride + swizzle period.
    pub fn underlying_cols(self) -> usize {
        self.parent.cols
    }
    /// Fragment count along the sub-tile's outer (row) axis — `sub_rows / 16`.
    pub fn n_frags(self) -> usize {
        self.sub_rows / 16
    }
    /// Row offset in elements (`blk_row · sub_rows`).
    pub fn row_offset(self) -> usize {
        self.blk_row * self.sub_rows
    }
    /// Column offset in elements (`blk_col · sub_cols`).
    pub fn col_offset(self) -> usize {
        self.blk_col * self.sub_cols
    }
}

/// A **register operand tile** — HK's `rt_bf<rows, cols>` (Row layout), an array of `rows/16 × cols/16`
/// 16×16 bf16 fragments (each 4 bf16/lane in 2 VGPRs). In `micro_tk`, `tiles[8] : rt_bf<64, 16>` = 4×1.
#[derive(Clone, Debug)]
pub struct rt_bf {
    pub frags: Vec<Frag<BF16>>,
    pub rows: usize,
    pub cols: usize,
}

impl rt_bf {
    /// `rt_bf<rows, cols>` — mint `(rows/16)·(cols/16)` Row (`transpose = false`) fragments.
    pub fn new(b: &mut Builder, rows: usize, cols: usize) -> Self {
        let n = (rows / 16) * (cols / 16);
        let frags = (0..n).map(|_| b.define_frag::<BF16>(FragMap::gfx942_16x16(false))).collect();
        rt_bf { frags, rows, cols }
    }
}

/// A **register accumulator tile** — HK's `rt_fl<rows, cols, col>` (Col layout), an array of
/// `rows/16 × cols/16` 16×16 f32 fragments (each 4 f32/lane in 4 VGPRs). In `micro_tk`,
/// `C_accum[2] : rt_fl<64, 64, col>` = 4×4.
#[derive(Clone, Debug)]
pub struct rt_fl {
    pub frags: Vec<Frag<F32>>,
    pub rows: usize,
    pub cols: usize,
}

impl rt_fl {
    /// `rt_fl<rows, cols, col>` — mint `(rows/16)·(cols/16)` Col (`transpose = true`) fragments.
    pub fn new(b: &mut Builder, rows: usize, cols: usize) -> Self {
        let n = (rows / 16) * (cols / 16);
        let frags = (0..n).map(|_| b.define_frag::<F32>(FragMap::gfx942_16x16(true))).collect();
        rt_fl { frags, rows, cols }
    }
    /// The number of 16×16 fragments (`(rows/16)·(cols/16)`).
    pub fn n_frags(&self) -> usize {
        self.frags.len()
    }
}

/// HK's `coord` (`util.cuh:38`) — a named tensor index `{b, d, r, c}` in tile units. A plain data
/// carrier; the flat element offset is computed by the caller (tk2 addresses buffers flat).
#[derive(Copy, Clone, Debug, Default)]
pub struct coord {
    pub b: i64,
    pub d: i64,
    pub r: i64,
    pub c: i64,
}
