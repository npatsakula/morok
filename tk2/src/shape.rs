//! **`MfmaShape` — the matrix-core shape as a compile-time deriving device** (migration Step 1).
//!
//! tk2 is presently hardwired to the gfx942 **16×16×16 bf16** MFMA: `EDGE = 16`, `ept = 4`, one
//! [`FragMap::gfx942_16x16`], one intrinsic. Supporting a second shape (32×32×8) cleanly means the
//! shape-dependent constants — the operand/accumulator lane maps, the elements-per-thread triple, the
//! intrinsic dims — must be *derived* from a shape marker, not hand-copied per kernel.
//!
//! Per DESIGN §OPEN-2 (`tk2/DESIGN.md`), which splits "tile dims → in types" from "hardware-shape
//! instruction selection → in data + verifier", a marker is an **authoring-only deriving device**: it
//! *computes* the constants ([`FragMap`], [`AccDist`], ept, dims), which then ride the IR as ordinary
//! DATA. The marker never parameterises a handle ([`crate::build::Frag`]) or a [`crate::ir::Node`], so
//! the interned IR, the lowering, the verifier, and the passes stay data-driven exactly as today — no
//! monomorphisation of the movement layer, no generic explosion. Types compute the constants; the IR
//! stays data.
//!
//! Step 1 lands ONLY [`Mfma16x16x16Bf16`], and every 16×16×16 site is re-derived from it *byte-
//! identically* (the `test::byte_identity` gate). `Mfma32x32x8Bf16` is a later step.

use crate::ir::{AccDist, FragMap};

/// A gfx942 MFMA shape, resolved at authoring time to the fragment maps / ept / intrinsic dims the IR
/// needs as data. The A/B operand maps and the C-accumulator ([`Self::c_map`]) FragMap are *derived*
/// from the associated consts by default (one `impl` per shape sets only the consts + [`Self::acc_dist`]).
pub trait MfmaShape: Copy + 'static {
    /// Output-tile rows (MFMA `M`).
    const M: usize;
    /// Output-tile cols (MFMA `N`).
    const N: usize;
    /// Contraction depth (MFMA `K`).
    const K: usize;
    /// A-operand (Row) elements per lane.
    const EPT_A: usize;
    /// B-operand (Col) elements per lane.
    const EPT_B: usize;
    /// C-accumulator elements per lane (`= M·N/64`). **16×16×16: 4; 32×32×8: 16** — the field that
    /// breaks the `ept_A == ept_B == ept_C` invariant tk2 was built on.
    const EPT_C: usize;

    /// The A-operand (Row) fragment map: `row = lane % M`, `col = (lane / M)·ept + inner`.
    fn a_map() -> FragMap {
        FragMap { rows: Self::M, cols: Self::K, ept: Self::EPT_A, stride: Self::EPT_A, transpose: false }
    }
    /// The B-operand (Col) fragment map: `row = (lane / N)·ept + inner`, `col = lane % N`.
    fn b_map() -> FragMap {
        FragMap { rows: Self::K, cols: Self::N, ept: Self::EPT_B, stride: Self::EPT_B, transpose: true }
    }
    /// The C-accumulator fragment map carried by its `DefineFrag` — an `EPT_C`-wide `Col` carrier. Its
    /// lane→(row,col) *addressing* comes from [`Self::acc_dist`], NOT this map (which the FragMap cannot
    /// express beyond the 16×16 degenerate case); the map exists so the accumulator reg carries the
    /// correct width. For 16×16×16 this equals `FragMap::gfx942_16x16(true)` exactly.
    fn c_map() -> FragMap {
        FragMap { rows: Self::M, cols: Self::N, ept: Self::EPT_C, stride: Self::EPT_C, transpose: true }
    }
    /// The accumulator lane→(row,col) distribution (the two-level M-block split the FragMap can't hold).
    fn acc_dist() -> AccDist;

    /// The `(M, N, K)` intrinsic dims — the `wmma_desc` / `resolve_intrinsic` selector for `Node::Mma`.
    fn dims() -> (usize, usize, usize) {
        (Self::M, Self::N, Self::K)
    }
}

/// The gfx942 **16×16×16 bf16→f32** MFMA (`v_mfma_f32_16x16x16_bf16`) — tk2's default and, in Step 1,
/// its ONLY shape. Every current hardcoded 16×16×16 constant is re-derived from this marker.
#[derive(Copy, Clone, Debug)]
pub struct Mfma16x16x16Bf16;

impl MfmaShape for Mfma16x16x16Bf16 {
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 16;
    const EPT_A: usize = 4;
    const EPT_B: usize = 4;
    const EPT_C: usize = 4;

    // `kCM0PerLane=1, kCMLane=4, kCM1PerLane=4, kCNLane=16` ⇒ `row = (lane/16)·4 + i`, `col = lane%16`
    // (`m_blocks=1` ⇒ the block term is 0, so this is identical to the `transpose` FragMap `lane_rc`).
    fn acc_dist() -> AccDist {
        AccDist { m_blocks: 1, m_block_stride: 16, m_inner: 4, lane_m_stride: 4, n_lanes: 16 }
    }
}
