//! `RowPartition` — a cuTile-style workgroup partition descriptor (DSL-simplification Phase 2).
//!
//! A row-major tensor whose rows are `slices` independent blocks of `rows_per_slice` rows each,
//! tiled into `tile_rows`-row Q/output tiles: FA's `(batch·heads) × (seq/tile)` launch. The descriptor
//! DERIVES the 1-D workgroup grid (ceil-div) and, from a workgroup id, the `(slice, tile-in-slice,
//! global row-origin)` — replacing the hand-written `wgid → (bh_idx, qwg)` grid decode and the
//! `q_origin = bh_idx·n + qwg·q_blk` row-origin arithmetic with one reusable descriptor.

use crate::build::{Builder, Idx};

/// A workgroup partition of a row-major tensor: `slices` independent row-blocks of `rows_per_slice`
/// rows, each tiled into `tile_rows`-row tiles (one tile per workgroup). For [`flash_attention_fwd_32`]:
/// `slices = bh` (batch·heads), `rows_per_slice = n` (sequence length), `tile_rows = q_blk`.
///
/// [`flash_attention_fwd_32`]: crate::kernels::fa::flash_attention_fwd_32
#[derive(Copy, Clone, Debug)]
pub struct RowPartition {
    /// Independent row-blocks (FA: `bh`).
    pub slices: usize,
    /// Rows in each slice (FA: `n`).
    pub rows_per_slice: usize,
    /// Rows per Q/output tile, i.e. per workgroup (FA: `q_blk`).
    pub tile_rows: usize,
}

/// A decoded workgroup position: its `slice`, its `tile` within that slice, and the global
/// `row_origin` of the tile (`slice·rows_per_slice + tile·tile_rows`). The small three-`Idx` value
/// [`RowPartition::decode`] returns — FA's former `(bh_idx, qwg, q_origin)` triple.
#[derive(Copy, Clone, Debug)]
pub struct TilePos {
    /// The independent slice this workgroup owns (FA: `bh_idx`).
    pub slice: Idx,
    /// The tile index within that slice (FA: `qwg`).
    pub tile: Idx,
    /// The tile's global row origin `slice·rows_per_slice + tile·tile_rows` (FA: `q_origin`).
    pub row_origin: Idx,
}

impl RowPartition {
    /// Tiles covering one slice — rounds UP, so a ragged `rows_per_slice` is fully covered (the partial
    /// last tile's excess rows are computed-but-not-scattered by the caller).
    pub fn tiles_per_slice(&self) -> usize {
        self.rows_per_slice.div_ceil(self.tile_rows)
    }

    /// The 1-D workgroup grid size `slices · tiles_per_slice` — the launch bound for the flat grid axis.
    pub fn grid_size(&self) -> usize {
        self.slices * self.tiles_per_slice()
    }

    /// Remap the physical round-robin workgroup id into XCD-local chunks of logical row tiles. AMD
    /// assigns consecutive workgroups across XCDs; de-interleaving by `num_xcds` makes each XCD consume
    /// `chunk` consecutive logical tiles, preserving K/V cache locality for FA. The transform is used
    /// only when it is a bijection over the full grid and chunks do not cross slice boundaries.
    pub fn xcd_swizzle(&self, b: &mut Builder, wgid: Idx, num_xcds: usize, max_chunk: usize) -> Idx {
        if num_xcds == 0 || max_chunk == 0 {
            return wgid;
        }
        let tiles = self.tiles_per_slice();
        let chunk = tiles.min(max_chunk);
        let block = num_xcds * chunk;
        if chunk == 0 || !tiles.is_multiple_of(chunk) || !self.grid_size().is_multiple_of(block) {
            return wgid;
        }

        let nx = b.idx_const(num_xcds as i64);
        let ch = b.idx_const(chunk as i64);
        let bl = b.idx_const(block as i64);
        let xcd = b.idx_mod(wgid, nx);
        let local = b.idx_div(wgid, nx);
        let chunk_idx = b.idx_div(local, ch);
        let pos = b.idx_mod(local, ch);
        let hi = b.idx_mul(chunk_idx, bl);
        let mid = b.idx_mul(xcd, ch);
        let grouped = b.idx_add(hi, mid);
        b.idx_add(grouped, pos)
    }

    /// Decode a flat workgroup id into `(slice, tile, row_origin)`:
    /// `slice = wgid / tiles_per_slice`, `tile = wgid % tiles_per_slice`, and
    /// `row_origin = slice·rows_per_slice + tile·tile_rows`.
    pub fn decode(&self, b: &mut Builder, wgid: Idx) -> TilePos {
        let tps = b.idx_const(self.tiles_per_slice() as i64);
        let slice = b.idx_div(wgid, tps);
        let tile = b.idx_mod(wgid, tps);
        let rows = b.idx_const(self.rows_per_slice as i64);
        let slice_row = b.idx_mul(slice, rows); // slice·rows_per_slice
        let trows = b.idx_const(self.tile_rows as i64);
        let tile_row = b.idx_mul(tile, trows); // tile·tile_rows
        let row_origin = b.idx_add(slice_row, tile_row);
        TilePos { slice, tile, row_origin }
    }
}
