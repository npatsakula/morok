//! **The tile-op vocabulary** (`scratchpad/tile_layer_design.md` §2) — shape-generic compute ops whose
//! width/opcode is derived from the MFMA shape TYPE, not a threaded `ept`/`EDGE` parameter. Each op is a
//! faithful forward to the proven `build.rs` primitive selected by the shape, so migrating a kernel onto
//! this surface is behaviour-preserving (device-gated). The value is twofold: (1) a kernel written
//! `<S: MfmaShape>` picks its reduce/relayout by *shape*, not a hand-copied `acc_row_reduce_32` vs
//! `frag_col_reduce` call — the seam a shape swap needs; (2) the `const EPT` footgun (a blindly-applied
//! width) becomes a shape-SELECTED width, so applying the 16-wide reduce to a 4-wide fragment (or vice
//! versa) can't be authored.
//!
//! Movement ops (`load`/`store` residency dispatch) are a sibling concern landed in a later step; this
//! module is the compute half.

use crate::build::{BF16, Builder, F32, Idx, Val};
use crate::shape::MfmaShape;

/// MMA `D = A·B + C` at shape `S` — width from `S::EPT_C`. The uniform typed surface for both QKᵀ and
/// P·V; the intrinsic dims come from `S`, no `ept` argument.
pub fn mma<S: MfmaShape>(b: &mut Builder, a: Val<BF16>, bb: Val<BF16>, c: Val<F32>) -> Val<F32> {
    b.mma(a, bb, c, S::EPT_C)
}

/// Row-reduce the accumulator along its implicit contraction axis, WIDTH + geometry SELECTED BY `S`:
/// `EPT_C = 4` (16×16×16) → the [`Builder::frag_col_reduce`] lane-tree over the Col map; `EPT_C = 16`
/// (32×32×8) → the [`Builder::acc_row_reduce_32`] two-level `AccDist` reduce (16 in-register + the one
/// `L↔L+32` partner). `add = false` ⇒ running max; `add = true` ⇒ running sum. This is the seam that
/// replaces a hand-picked reduce call: the shape chooses the correct geometry, so the "apply the 16-wide
/// reduce to a 4-wide fragment" silent bug cannot be authored.
pub fn row_reduce<S: MfmaShape>(b: &mut Builder, val: Val<F32>, lane: Idx, init: Val<F32>, add: bool) -> Val<F32> {
    match S::EPT_C {
        4 => b.frag_col_reduce(val, lane, init, add),
        16 => b.acc_row_reduce_32(val, lane, init, add),
        n => panic!("row_reduce: no reduce geometry for EPT_C={n} (shapes are 16×16×16 → 4, 32×32×8 → 16)"),
    }
}

/// Relayout the softmax weights `P` (an f32 accumulator) into the P·V B-operand(s), MECHANISM SELECTED
/// BY `S`: 16×16×16 → the free f32→bf16 cast ([`Builder::cast_vec_bf16`], ONE operand, the accumulator
/// feeds B directly); 32×32×8 → the `v_perm s49` pack ([`Builder::pv_relayout_s49`], FOUR bf16 operands,
/// one per hardware K=8 slice, fusing the f32→bf16 cast with the 16→8 repack). The caller consumes the
/// returned run (1 or `KV_BLK/K` operands) per its K-loop — the count is a property of the shape.
pub fn relayout<S: MfmaShape>(b: &mut Builder, p: Val<F32>) -> Vec<Val<BF16>> {
    match (S::M, S::N, S::K) {
        (16, 16, 16) => vec![b.cast_vec_bf16(p)],
        (32, 32, 8) => b.pv_relayout_s49(p),
        (m, n, k) => panic!("relayout: no P→PV mechanism for shape {m}×{n}×{k}"),
    }
}

/// Broadcast scalar `v` across the `S::EPT_C`-wide accumulator vector — the softmax scale/zero splat
/// (`(0..EPT_C).map(|_| v)` → `vec_build`). The width is the shape's accumulator EPT, not a threaded
/// const, so it can't be applied at the wrong width.
pub fn splat<S: MfmaShape>(b: &mut Builder, v: Val<F32>) -> Val<F32> {
    let cs: Vec<Val<F32>> = (0..S::EPT_C).map(|_| v).collect();
    b.vec_build(&cs)
}
