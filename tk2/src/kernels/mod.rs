//! The tk2 kernel core shared by the device kernels: the finished-[`Program`] handle + its `.apply`
//! pass composition, the gfx942 MFMA `EDGE`/`WARP` constants, and the accumulator-scatter / index
//! helpers ([`scatter_frag`], [`offset_by`], [`add_opt`]) both submodules ride. The kernels themselves
//! live in [`matmul`] (the asm `clustered` HK matmul) and [`fa`] (the Flash-Attention forwards).

use crate::build::{Buf, Builder, F32, Frag, Idx};
use crate::ir::{TileId, TileIr};
use crate::pass::Pass;
use crate::shape::{Mfma16x16x16Bf16, MfmaShape};

pub mod fa;
pub mod matmul;

/// A finished tile-IR program: the arena, its sink root, and the kernel name.
pub struct Program {
    pub ir: TileIr,
    pub sink: TileId,
    pub name: String,
}

impl Program {
    /// Apply a refinement [`Pass`] to this program, returning the transformed program —
    /// the top-level `.apply` composition (DESIGN §1): `matmul_staged(cfg).apply(SwizzlePass)
    /// .apply(VectorizePass)`. The pass's `requires`/`ensures` contracts are checked around
    /// it (a failed contract or pass is a kernel-authoring bug, so it panics rather than
    /// threading a Result through the fluent chain — mirrors `run_kernel`'s expect idiom).
    pub fn apply(mut self, pass: impl Pass) -> Self {
        assert!(pass.requires(&self.ir, self.sink), "pass {}: precondition failed", pass.name());
        let root = pass.apply(&mut self.ir, self.sink).unwrap_or_else(|e| panic!("pass {}: {e:?}", pass.name()));
        assert!(pass.ensures(&self.ir, root), "pass {}: postcondition failed", pass.name());
        self.sink = root;
        self
    }
}

/// The gfx942 MFMA edge — one 16×16×16 fragment per workgroup, one 64-lane warp. Now DERIVED from the
/// [`Mfma16x16x16Bf16`] marker (`M == N == K == 16`) instead of a bare literal (§migration Step 1): the
/// type computes what was hardcoded, and every `EDGE` call site is byte-identical.
pub(crate) const EDGE: usize = Mfma16x16x16Bf16::M;
const WARP: usize = 64;

/// Scatter the accumulated f32 fragment `acc` (already ordered after the K-loop `End`)
/// back to GLOBAL `dst` via the same `lane_rc` map (the C tile is Col-layout, so this
/// is the transposed store) — the mirror of [`gather_frag`]. Returns the terminal
/// store effects (the kernel's sink roots).
pub(crate) fn scatter_frag(
    b: &mut Builder,
    acc: Frag<F32>,
    dst: Buf<F32>,
    base: Idx,
    row_stride: i64,
    lane: Idx,
) -> Vec<crate::build::Effect> {
    let rs = b.idx_const(row_stride);
    // The accumulator C-site re-derived from the marker: `acc_rc(acc_dist())` replaces the FragMap
    // `lane_rc`. For 16×16×16 (`m_blocks == 1`) it interns to the identical index nodes (byte-identical);
    // for 32×32×8 the AccDist carries the 4×4 block layout a single FragMap run cannot.
    let dist = Mfma16x16x16Bf16::acc_dist();
    (0..Mfma16x16x16Bf16::EPT_C)
        .map(|inner| {
            let inner_idx = b.idx_const(inner as i64);
            let (row, col) = b.acc_rc(dist, lane, inner);
            let row_off = b.idx_mul(row, rs);
            let off = b.idx_add(base, row_off);
            let off = b.idx_add(off, col);
            let v = b.load_frag_elem(acc, inner_idx);
            b.store(dst, off, v)
        })
        .collect()
}

/// `idx + base` (folding the `base == 0` identity so the flat path stays clean).
pub(crate) fn offset_by(b: &mut Builder, idx: Idx, base: usize) -> Idx {
    if base == 0 {
        idx
    } else {
        let c = b.idx_const(base as i64);
        b.idx_add(idx, c)
    }
}

/// `idx + off` when a runtime offset is present (the multi-warp wave offset); the identity
/// when `None` (the single-warp path, kept byte-identical — no spurious `+0` node).
pub(crate) fn add_opt(b: &mut Builder, idx: Idx, off: Option<Idx>) -> Idx {
    match off {
        Some(o) => b.idx_add(idx, o),
        None => idx,
    }
}
