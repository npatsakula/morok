//! **Operand-role + swizzle markers** — the zero-sized types the movement/compute ops use to *derive*
//! fragment addressing from the [`crate::shape::MfmaShape`]. A [`RegLayout`] marker (`ARow`/`BCol`/
//! `Acc`) computes the operand `FragMap` / accumulator `AccDist` / `ept` / `n_frags` for a register
//! tile in its role; a [`Swizzle`] marker (`Plain`/`Xor`) computes the LDS bank layout. No
//! [`crate::ir::Node`] gains a type parameter (Recommendation C, `shape.rs:8-14`): the marker computes
//! the constants, the IR stays data-driven, so the arena/lowering/verifier are untouched.
//!
//! The kernels are *runtime*-shaped (`atb_probe(kv, d, q)`), so dims ride as `usize` values and the
//! marker only carries the operand role / swizzle policy — [`crate::tile_move`]'s `gather`/`gather_run`
//! call `L::frag::<S>()` / `L::n_frags::<S>()` and `Sw::layout(cols)` directly off these markers.

use crate::ir::{AccDist, FragMap, Layout, Transform};
use crate::shape::MfmaShape;

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Register-tile operand roles (HK's `rt_layout` row/col/accumulator) — each derives its FragMap /
// AccDist / ept / n_frags from the MFMA shape, so the role marker is all the kernel names.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// A register tile's operand role, deriving the shape-dependent fragment data.
pub trait RegLayout: Copy + 'static {
    /// The operand `FragMap` — `Some` for Row/Col operands, `None` for the accumulator.
    fn frag<S: MfmaShape>() -> Option<FragMap>;
    /// The accumulator distribution — `Some` only for [`Acc`].
    fn acc<S: MfmaShape>() -> Option<AccDist>;
    /// Per-lane element run in this role (`EPT_A`/`EPT_B`/`EPT_C`).
    fn ept<S: MfmaShape>() -> usize;
    /// MFMA base fragments an `R×C` tile spans in this role.
    fn n_frags<S: MfmaShape>(rows: usize, cols: usize) -> usize;
}

/// The A-operand (Row) role — a straight `M×K` fragment (`ds_read_b64`, no register transpose).
#[derive(Copy, Clone, Debug)]
pub struct ARow;
/// The B-operand (Col) role — a transposed `K×N` fragment.
#[derive(Copy, Clone, Debug)]
pub struct BCol;
/// The C-accumulator role — an `M×N` fragment addressed by the [`AccDist`] distribution.
#[derive(Copy, Clone, Debug)]
pub struct Acc;

impl RegLayout for ARow {
    fn frag<S: MfmaShape>() -> Option<FragMap> {
        Some(S::a_map())
    }
    fn acc<S: MfmaShape>() -> Option<AccDist> {
        None
    }
    fn ept<S: MfmaShape>() -> usize {
        S::EPT_A
    }
    fn n_frags<S: MfmaShape>(rows: usize, cols: usize) -> usize {
        (rows / S::M) * (cols / S::K)
    }
}

impl RegLayout for BCol {
    fn frag<S: MfmaShape>() -> Option<FragMap> {
        Some(S::b_map())
    }
    fn acc<S: MfmaShape>() -> Option<AccDist> {
        None
    }
    fn ept<S: MfmaShape>() -> usize {
        S::EPT_B
    }
    fn n_frags<S: MfmaShape>(rows: usize, cols: usize) -> usize {
        (rows / S::K) * (cols / S::N)
    }
}

impl RegLayout for Acc {
    fn frag<S: MfmaShape>() -> Option<FragMap> {
        None
    }
    fn acc<S: MfmaShape>() -> Option<AccDist> {
        Some(S::acc_dist())
    }
    fn ept<S: MfmaShape>() -> usize {
        S::EPT_C
    }
    fn n_frags<S: MfmaShape>(rows: usize, cols: usize) -> usize {
        (rows / S::M) * (cols / S::N)
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// LDS-tile swizzle (HK's `st_shape::swizzle`) — a property of the *type*, so one tile owns the
// bank XOR and fill/gather cannot disagree.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// An LDS tile's bank-swizzle policy.
pub trait Swizzle: Copy + 'static {
    /// The layout for a `cols`-wide tile.
    fn layout(cols: usize) -> Layout;
}

/// Contiguous (no XOR) — the padded layout, used by V (flat `inner` pitch). The register-staged fill
/// can use either swizzle.
#[derive(Copy, Clone, Debug)]
pub struct Plain;
/// The HK/CK bank-conflict XOR swizzle (`col ^ delta(row)`).
#[derive(Copy, Clone, Debug)]
pub struct Xor;

impl Swizzle for Plain {
    fn layout(_cols: usize) -> Layout {
        Layout::contiguous()
    }
}

impl Swizzle for Xor {
    fn layout(cols: usize) -> Layout {
        let mut transforms = smallvec::SmallVec::new();
        transforms.push(Transform::Xor { cols });
        Layout { transforms }
    }
}
