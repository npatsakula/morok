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

use crate::build::{BF16, Buf, Builder, F32, Frag, Idx};
use crate::ir::{FragMap, TileId, TileIr};

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

/// A cross-lane LDS round-trip: one workgroup of `n` lanes stages `in[lane] → lds[lane]`,
/// a workgroup **barrier** fences every lane's write, then each lane reads its
/// neighbour `lds[(lane+1) % n]` back — a read only correct *past* the barrier — and
/// stores it to `out[lane]`. Result: `out[i] = in[(i+1) % n]` (a rotation). This is the
/// proof that the `DefineLocal` + `Barrier` + LDS load/store machinery and the
/// `store → barrier → load` ordering edge work (the reuse-lever foundation).
pub fn lds_roundtrip(n: usize) -> Program {
    let mut b = Builder::new("tk2_lds_roundtrip");

    // ABI slots: output (0) before input (1).
    let out = b.global::<F32>(n);
    let inp = b.global::<F32>(n);
    let lds = b.define_local::<F32>(n);

    // One workgroup, `n` lanes. Each lane fills its own LDS slot from global.
    let lane = b.block_axis(n as i64);
    let v = b.load(inp, lane);
    let staged = b.store_lds(lds, lane, v);

    // Fence: every lane's LDS write completes before any cross-lane read.
    let bar = b.barrier(staged, &[]);

    // Neighbour read `lds[(lane+1) % n]`, ordered AFTER the barrier (the cross-lane edge).
    let one = b.idx_const(1);
    let np = b.idx_const(n as i64);
    let lp1 = b.idx_add(lane, one);
    let nbr = b.idx_mod(lp1, np);
    let rot = b.load_lds_after(lds, nbr, &[bar.dep()]);
    let st = b.store(out, lane, rot);

    let (ir, sink) = b.finish(&[st]);
    Program { ir, sink, name: "tk2_lds_roundtrip".into() }
}

/// The gfx942 MFMA edge — one 16×16×16 fragment per workgroup, one 64-lane warp.
const EDGE: usize = 16;
const WARP: usize = 64;

/// Gather one 16×16 bf16 fragment straight from GLOBAL into the register fragment
/// `dst` (NO LDS, NO swizzle, NO vectorization — the deliberately-naive direct load;
/// re-run every K-step). For each per-lane element `inner`, `lane_rc` gives the
/// in-tile `(row, col)`, and the flat global offset is `base + row·row_stride + col`
/// (`base` = the tile origin `tile_row·EDGE·row_stride + tile_col·EDGE`). Mirrors
/// `tk/src/group/movement.rs::load_global_to_reg`. Returns the `ept` store edges the
/// operand read routes through.
fn gather_frag(b: &mut Builder, dst: Frag<BF16>, src: Buf<BF16>, base: Idx, row_stride: i64, lane: Idx) -> Vec<TileId> {
    let rs = b.idx_const(row_stride);
    (0..dst.map.ept)
        .map(|inner| {
            let inner_idx = b.idx_const(inner as i64);
            let (row, col) = b.lane_rc(dst.map, lane, inner_idx);
            let row_off = b.idx_mul(row, rs);
            let off = b.idx_add(base, row_off);
            let off = b.idx_add(off, col);
            let v = b.load(src, off);
            b.store_frag_elem(dst, inner_idx, v).dep()
        })
        .collect()
}

/// Scatter the accumulated f32 fragment `acc` (already ordered after the K-loop `End`)
/// back to GLOBAL `dst` via the same `lane_rc` map (the C tile is Col-layout, so this
/// is the transposed store) — the mirror of [`gather_frag`]. Returns the terminal
/// store effects (the kernel's sink roots).
fn scatter_frag(
    b: &mut Builder,
    acc: Frag<F32>,
    dst: Buf<F32>,
    base: Idx,
    row_stride: i64,
    lane: Idx,
) -> Vec<crate::build::Effect> {
    let rs = b.idx_const(row_stride);
    (0..acc.map.ept)
        .map(|inner| {
            let inner_idx = b.idx_const(inner as i64);
            let (row, col) = b.lane_rc(acc.map, lane, inner_idx);
            let row_off = b.idx_mul(row, rs);
            let off = b.idx_add(base, row_off);
            let off = b.idx_add(off, col);
            let v = b.load_frag_elem(acc, inner_idx);
            b.store(dst, off, v)
        })
        .collect()
}

/// A **naive, correct, slow** tiled matmul `C[M,N] = A[M,K] · B[K,N]` (bf16 inputs,
/// f32 accumulate) — the correctness scaffold a later perf phase optimizes via passes.
///
/// Structure (one 16×16 output tile per workgroup, one 64-lane warp):
/// - grid `(M/16) × (N/16)` output tiles (`gidx0`, `gidx1`); block `64` lanes (`lidx0`);
/// - a K-loop over `K/16` fragments with a **loop-carried f32 register accumulator**
///   (the exact `sum_reduce` edge pattern: init `End`, in-loop `acc.after([init, range])`
///   read, one `End`, post-loop `acc.after([end])` read);
/// - each K-step gathers an A-fragment (Row) and a B-fragment (Col) **directly from
///   global** and issues ONE 16×16×16 MFMA into the accumulator;
/// - after the loop, the accumulator fragment scatters to C.
///
/// Deliberately slow — NO LDS, swizzle, double-buffering, async, asm-pin, or AGPR (all
/// deferred to the perf phase). `m`/`n`/`k` must be multiples of 16.
pub fn matmul(m: usize, n: usize, k: usize) -> Program {
    assert!(
        m.is_multiple_of(EDGE) && n.is_multiple_of(EDGE) && k.is_multiple_of(EDGE),
        "matmul dims must be multiples of {EDGE}"
    );
    let mut b = Builder::new("tk2_matmul");

    // ABI slots: output C (0, f32) before inputs A (1, bf16), B (2, bf16).
    let c = b.global::<F32>(m * n);
    let a = b.global::<BF16>(m * k);
    let bmat = b.global::<BF16>(k * n);

    // One 16×16 output tile per workgroup; a 64-lane warp per tile.
    let tile_m = b.grid_axis(0, (m / EDGE) as i64);
    let tile_n = b.grid_axis(1, (n / EDGE) as i64);
    let lane = b.block_axis(WARP as i64);

    // Fragment tiles carrying their MFMA lane-maps (A = Row, B/C = Col — the
    // `mma_ab` operand layouts, mirrored from tk's `test_mma_ab_wmma_graph_shape`).
    let a_map = FragMap::gfx942_16x16(false);
    let bc_map = FragMap::gfx942_16x16(true);
    let a_frag = b.define_frag::<BF16>(a_map);
    let b_frag = b.define_frag::<BF16>(bc_map);
    let acc = b.define_frag::<F32>(bc_map);
    let ept = a_map.ept;

    // ── init: acc[0..ept] = 0, scoped by an `ept`-trip range (the sum_reduce init
    //    shape — a self-contained pre-loop store, not a loose entry write). ──
    let init_r = b.range(ept as i64);
    let init_c = b.counter(init_r);
    let zero = b.f32(0.0);
    let s_init = b.store_frag_elem(acc, init_c, zero);
    let inited = b.end(s_init, &[init_r]);

    // ── K-loop over `K/16` fragments (the loop-carried accumulator). ──
    let kr = b.range((k / EDGE) as i64);
    let tk = b.counter(kr);

    // Tile-origin base offsets: A at (tile_m, tk), B at (tk, tile_n).
    let e = b.idx_const(EDGE as i64);
    let ek = b.idx_const((EDGE * k) as i64);
    let en = b.idx_const((EDGE * n) as i64);
    let tk_e = b.idx_mul(tk, e);
    let tm_ek = b.idx_mul(tile_m, ek);
    let base_a = b.idx_add(tm_ek, tk_e); // tile_m·16·K + tk·16
    let tk_en = b.idx_mul(tk, en);
    let tn_e = b.idx_mul(tile_n, e);
    let base_b = b.idx_add(tk_en, tn_e); // tk·16·N + tile_n·16

    let a_stores = gather_frag(&mut b, a_frag, a, base_a, k as i64, lane);
    let b_stores = gather_frag(&mut b, b_frag, bmat, base_b, n as i64, lane);

    let a_vec = b.load_frag_vec_after(a_frag, &a_stores);
    let b_vec = b.load_frag_vec_after(b_frag, &b_stores);
    // The accumulator read routes through [init, K-range] — the loop-carry edge:
    // the init store on the first trip, the range keeping it inside the K loop.
    let acc_vec = b.load_frag_vec_after(acc, &[inited.dep(), kr.dep()]);
    let out = b.mma(a_vec, b_vec, acc_vec, ept);
    let s_acc = b.store_frag_vec(acc, out);
    let ended = b.end(s_acc, &[kr]); // exactly one END closes the K loop.

    // ── post-loop: read the accumulator after the loop END, scatter to C. ──
    let acc_final = b.frag_after(acc, &[ended.dep()]);
    let tm_en = b.idx_mul(tile_m, en);
    let base_c = b.idx_add(tm_en, tn_e); // tile_m·16·N + tile_n·16
    let roots = scatter_frag(&mut b, acc_final, c, base_c, n as i64, lane);

    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_matmul".into() }
}
