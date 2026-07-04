//! The two skeleton kernels authored via the [`Builder`], each a proof of one
//! half of the architecture spine (DESIGN.md §7 step 1).
//!
//! - [`elementwise_add`] — a *tiled* `C = A + B`: grid over tiles, an inner loop
//!   over each tile's elements. Proves the end-to-end "it works" path (naive
//!   tile-IR → verified lowering → device-UOp → runs on gfx942, matches CPU).
//! - [`sum_reduce`] — `out = Σ A[i]` with a **loop-carried register accumulator**.
//!   Proves the Range/End + ordering-edge machinery is structurally present — the
//!   loop-carry read routes through `reg.after([prev_store, range])` and the
//!   post-loop read through `reg.after([end])`, the exact edges whose omission is a
//!   silent miscompile (mirrors `tk/src/group/reduce.rs`).

use crate::build::{Builder, F32};
use crate::ir::{TileId, TileIr};

/// A finished tile-IR program: the arena, its sink root, and the kernel name.
pub struct Program {
    pub ir: TileIr,
    pub sink: TileId,
    pub name: String,
}

/// Tiled elementwise add: `n_tiles` workgroups, each looping over `tile` elements
/// computing `C[g*tile + i] = A[..] + B[..]`. Total length `n_tiles * tile`.
pub fn elementwise_add(tile: usize, n_tiles: usize) -> Program {
    let n = tile * n_tiles;
    let mut b = Builder::new("tk2_add");

    // ABI slots: output C (0) is bound before inputs A (1), B (2).
    let c = b.global::<F32>(n);
    let a = b.global::<F32>(n);
    let bb = b.global::<F32>(n);

    // One workgroup per tile (grid geometry rides on this Axis bound); an inner
    // loop over the tile's elements. `off = g*tile + i`.
    let g = b.grid_axis(0, n_tiles as i64);
    let r = b.range(tile as i64);
    let tile_c = b.idx_const(tile as i64);
    let base = b.idx_mul(g, tile_c);
    let off = b.idx_add(base, b.counter(r));

    let av = b.load(a, off);
    let bv = b.load(bb, off);
    let cv = b.add(av, bv);
    let st = b.store(c, off, cv);
    let ended = b.end(st, &[r]); // one END closes the inner loop.

    let (ir, sink) = b.finish(&[ended]);
    Program { ir, sink, name: "tk2_add".into() }
}

/// 1-D sum reduction with a loop-carried f32 register accumulator. A single
/// workgroup (grid [1,1,1]) folds `A[0..n]` into `reg`, then stores it to `out[0]`.
pub fn sum_reduce(n: usize) -> Program {
    let mut b = Builder::new("tk2_sum");

    let out = b.global::<F32>(1); // slot 0
    let a = b.global::<F32>(n); // slot 1
    let reg = b.define_reg::<F32>(1);

    let zero_idx = b.idx_const(0);

    // (init) `reg[0] = 0`, scoped by a 1-trip range (the `g.zero`/reduce.rs init
    // shape) so it is a self-contained pre-loop store, not a loose entry write.
    let init_r = b.range(1);
    let init_c = b.counter(init_r);
    let zero = b.f32(0.0);
    let s_init = b.store_reg(reg, init_c, zero);
    let inited = b.end(s_init, &[init_r]);

    // (loop) fold `reg += A[i]` over `i in 0..n`.
    let r = b.range(n as i64);
    let rc = b.counter(r);
    // (b) the accumulate READ routes through [prev_store, range]: the prior-store
    // edge is the loop-carry chain; the range edge keeps the read in the loop body.
    let acc = b.load_reg_after(reg, zero_idx, &[inited.dep(), r.dep()]);
    let x = b.load(a, rc);
    let new = b.add(acc, x);
    // (c) exactly one END closes the reduction range around the accumulate store.
    let s_acc = b.store_reg(reg, zero_idx, new);
    let ended = b.end(s_acc, &[r]);

    // (d) the post-loop read routes through the loop-closing END.
    let reg_final = b.reg_after(reg, &[ended.dep()]);
    let result = b.load_reg(reg_final, zero_idx);
    let out_st = b.store(out, zero_idx, result);

    let (ir, sink) = b.finish(&[out_st]);
    Program { ir, sink, name: "tk2_sum".into() }
}
