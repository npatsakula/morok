//! HipKittens' shared-tile XOR bank swizzle — `st::idx()` (`types/shared/st.cuh:106`) — as a pure
//! function, cross-checked against tk2's [`SwizzlePass`](crate::passes::SwizzlePass) formula.
//!
//! For `st_bf<256, 64>` (bf16): `swizzle_bytes = 128`, `swizzle_repeat = 2048`, `subtile_cols = 64`.
//! HK computes the swizzle on the **byte** address; tk2's `SwizzlePass` computes it on the **element**
//! column (`col ^ delta`, `delta = (((row%16)·cols·2 >> 7 << 3) >> 1)`). They are the same map in
//! element space (`hk_idx / sizeof == tk2_offset`) — the leaf verified by test T2.

/// HK's `st::idx(ptr, {r, c})` (`st.cuh:106`) — the swizzled **byte** offset of element `(r, c)` in a
/// `rows × ·` tile with `subtile_cols`-wide swizzle sub-tiles (bf16, `sizeof = 2`,
/// `swizzle_repeat = 2048`). `ptr` is the tile base byte offset (0 for a fresh tile).
pub fn idx(ptr: u32, r: u32, c: u32, rows: u32, subtile_cols: u32) -> u32 {
    const SIZEOF_T: u32 = 2; // bf16
    const SWIZZLE_REPEAT: u32 = 2048;
    let outer_idx = c / subtile_cols;
    let addr = ptr + SIZEOF_T * (outer_idx * rows * subtile_cols + r * subtile_cols + c % subtile_cols);
    let swizzle = ((addr % SWIZZLE_REPEAT) >> 7) << 3;
    addr ^ swizzle
}

/// tk2's [`SwizzlePass`](crate::passes::SwizzlePass) element-space delta:
/// `delta = (((row % 16) · cols · 2) >> 7 << 3) >> 1`. XORed with the element column.
pub fn tk2_delta(row: u32, cols: u32) -> u32 {
    (((row % 16) * cols * 2) >> 7 << 3) >> 1
}

/// tk2's swizzled **element** offset of `(r, c)` in a `cols`-wide tile: `r·cols + (c ^ delta(r, cols))`
/// — the flat offset [`Builder::lds_col`](crate::build::Builder::lds_col) yields after `SwizzlePass`.
pub fn tk2_offset(r: u32, c: u32, cols: u32) -> u32 {
    r * cols + (c ^ tk2_delta(r, cols))
}
