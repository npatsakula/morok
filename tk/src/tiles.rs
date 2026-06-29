//! Tile shape descriptors and layouts.
//!
//! These are the pure, data-only building blocks shared by every tile kind. A
//! [`BaseShape`] is one WMMA-sized fragment (e.g. 16×16); a full tile is a grid
//! of base shapes. The concrete tile wrappers (GL/ST/RT/RV) that bind a buffer
//! and a [`crate::Kernel`] live alongside the builder.
//!
//! `elements_per_thread` is carried **explicitly** per shape rather than derived
//! `num_elements / WARP_THREADS`, because it is a function of the matrix-core
//! fragment layout, which differs by arch: CDNA wave64 16×16 = 4/lane; RDNA
//! wave32 = 8/lane for the accumulator and **16/lane for the (replicated) WMMA
//! inputs** (256/32 × the 0-15≡16-31 wave-half replication). The `_W32_*`
//! constants below are the RDNA (gfx11) shapes; the unsuffixed ones are gfx942.

use crate::swizzle::Swizzle;

/// Register-tile element layout within a warp.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileLayout {
    Row,
    Col,
}

/// Register-vector layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VecLayout {
    Ortho,
}

/// A WMMA-sized base fragment, carrying its per-lane element count (`ept`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BaseShape {
    pub rows: usize,
    pub cols: usize,
    /// Elements each lane holds for one base fragment — arch/layout-specific (see
    /// the module docs), NOT always `num_elements / wave_size` (RDNA inputs are
    /// replicated, so `ept > num_elements / wave_size`).
    pub ept: usize,
}

impl BaseShape {
    pub const fn num_elements(&self) -> usize {
        self.rows * self.cols
    }
    /// Elements each thread (lane) holds for one base fragment.
    pub const fn elements_per_thread(&self) -> usize {
        self.ept
    }
}

/// Shared-tile base fragment: a [`BaseShape`] plus its LDS [`Swizzle`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct STBaseShape {
    pub base: BaseShape,
    pub swizzle: Swizzle,
}

/// Register-tile base fragment: a [`BaseShape`] plus the per-lane fragment
/// `stride` (the lane-group step in `lane_rc`) and the `interleave`/`interleave_t`
/// flags. gfx942 spreads K across lane-groups (stride = ept); RDNA holds all K in
/// one lane (`stride = 0` for the replicated inputs). `interleave` selects the
/// RDNA WMMA f32 accumulator's even/odd row map (`m = 2·j + lane/16, n = lane%16`),
/// which no `stride` can express; `interleave_t` is its **transpose**
/// (`row = lane%16, col = 2·j + lane/16`) — the layout for storing an RDNA
/// accumulator to memory along the transposed (N-major) axis, e.g. the FA output
/// tile `O[q,d]` from the `[d,q]` PV accumulator (see `lane_rc`). At most one of
/// the two is set.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RTBaseShape {
    pub base: BaseShape,
    pub stride: usize,
    pub interleave: bool,
    pub interleave_t: bool,
}

impl RTBaseShape {
    pub const fn elements_per_thread(&self) -> usize {
        self.base.elements_per_thread()
    }
    pub const fn num_strides(&self) -> usize {
        // `stride == 0` (RDNA replicated inputs: all K in one lane) ⇒ a single run.
        match self.elements_per_thread().checked_div(self.stride) {
            Some(n) => n,
            None => 1,
        }
    }
}

// ── gfx942 (CDNA3, wave64) base shapes — ept = num_elements / 64 ──────────────

// Predefined shared-tile base shapes.
pub const ST_16X16: STBaseShape =
    STBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 4 }, swizzle: Swizzle::Identity };
pub const ST_16X16_SWIZZLED: STBaseShape =
    STBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 4 }, swizzle: Swizzle::Sw16x16 };
pub const ST_32X32: STBaseShape =
    STBaseShape { base: BaseShape { rows: 32, cols: 32, ept: 16 }, swizzle: Swizzle::Sw32x32 };
pub const ST_16X32: STBaseShape =
    STBaseShape { base: BaseShape { rows: 16, cols: 32, ept: 8 }, swizzle: Swizzle::Sw16x32 };
pub const ST_32X16: STBaseShape =
    STBaseShape { base: BaseShape { rows: 32, cols: 16, ept: 8 }, swizzle: Swizzle::Sw32x16 };

/// The WMMA matrix-core tile edge (16): the fragment dimension every supported arch
/// shares (gfx942 MFMA and gfx1151 WMMA are both 16×16). The single source the
/// kernels' structural `BLK` tiles derive from, rather than re-declaring `16`.
pub const WMMA_EDGE: usize = RT_16X16.base.rows;

// Predefined register-tile base shapes.
pub const RT_16X16: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 4 }, stride: 4, interleave: false, interleave_t: false };
pub const RT_32X32: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 32, cols: 32, ept: 16 }, stride: 4, interleave: false, interleave_t: false };
pub const RT_16X32: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 32, ept: 8 }, stride: 8, interleave: false, interleave_t: false };
pub const RT_32X16: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 32, cols: 16, ept: 8 }, stride: 8, interleave: false, interleave_t: false };

// ── RDNA (gfx11, wave32) base shapes — for the gfx1151 WMMA matmul ────────────
//
// Accumulator: ept = 256/32 = 8, `interleave` ⇒ `lane_rc` gives the RDNA3 WMMA
// f32 even/odd row map `m = 2·j + lane/16, n = lane%16` (tinygrad `ops_python`
// `c_map = (lane%16, lane//16 + 2·elem)`; NOT the gfx12/CK contiguous layout).
// Inputs: ept = 16 (replicated across wave-halves), stride = 0 ⇒ lane = M/N, the
// 16 elements = the K run, identical for lanes L and L+16.

/// LDS strip fragment for the wave32 matmul (`ept = 256/32 = 8`).
pub const ST_16X16_SWIZZLED_W32: STBaseShape =
    STBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 8 }, swizzle: Swizzle::Sw16x16 };
/// wave32 WMMA f32 accumulator fragment: even/odd row interleave (`interleave`).
/// `stride` is unused (the interleave map ignores it).
pub const RT_16X16_W32_ACC: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 8 }, stride: 1, interleave: true, interleave_t: false };
/// wave32 WMMA input fragment: 16 K/lane, replicated across the two wave-halves.
pub const RT_16X16_W32_IN: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 16 }, stride: 0, interleave: false, interleave_t: false };
/// wave32 WMMA f32 accumulator, **transposed** for an N-major memory store
/// (`interleave_t`): `row = lane%16, col = 2·j + lane/16`. Used for the FA output
/// tile (`o_reg_t`, `O[q,d]`) — the transpose of the `[d,q]` PV accumulator
/// ([`RT_16X16_W32_ACC`]). gfx942 reaches the same transposed store through the
/// plain stride map, so this is RDNA-only.
pub const RT_16X16_W32_ACC_T: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 8 }, stride: 1, interleave: false, interleave_t: true };
