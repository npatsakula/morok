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

use crate::build::{BF16, Buf, Builder, Effect, F32, Frag, Idx, Lds, Val};
use crate::ir::{FragMap, TileId, TileIr};
use crate::pass::Pass;

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

/// Collaboratively fill an LDS tile `lds` (logical `[rows, cols_lds]`, flat row-major)
/// from `src`: the 64 lanes partition the `rows·cols_lds` elements into `epl` each. The
/// flat index `flat = lane·epl + j` maps to LDS position `flat` and global element
/// `(tile_row_base + flat/cols_lds)·grow_stride + tile_col_base + flat%cols_lds`. One
/// store per iteration → the single-END fill loop; returns its closing effect.
#[allow(clippy::too_many_arguments)]
fn fill_lds(
    b: &mut Builder,
    lds: Lds<BF16>,
    src: Buf<BF16>,
    epl: i64,
    lane: Idx,
    cols_lds: i64,
    tile_row_base: Idx,
    tile_col_base: Idx,
    grow_stride: i64,
) -> Effect {
    let fr = b.range(epl);
    let j = b.counter(fr);
    let epl_c = b.idx_const(epl);
    let lane_epl = b.idx_mul(lane, epl_c);
    let flat = b.idx_add(lane_epl, j);
    let cols_c = b.idx_const(cols_lds);
    let r = b.idx_div(flat, cols_c);
    let c = b.idx_mod(flat, cols_c);
    let gstride = b.idx_const(grow_stride);
    let grow = b.idx_add(tile_row_base, r);
    let goff = b.idx_mul(grow, gstride);
    let goff = b.idx_add(goff, tile_col_base);
    let goff = b.idx_add(goff, c);
    let v = b.load(src, goff);
    let st = b.store_lds(lds, flat, v);
    b.end(st, &[fr])
}

/// Unrolled collaborative global→LDS fill (NO inner range — usable inside the K-loop
/// without a loop nest): each lane writes `epl` elements `flat = lane·epl + e`, mapping
/// LDS position `flat` to global `(tile_row_base + flat/cols_lds)·grow_stride +
/// tile_col_base + flat%cols_lds`. `tile_col_base`/`tile_row_base` may be per-K-block
/// (carry the loop counter) so the strip is re-staged each iteration. Returns the `epl`
/// store effects to fence with a barrier.
#[allow(clippy::too_many_arguments)]
fn fill_lds_unrolled(
    b: &mut Builder,
    lds: Lds<BF16>,
    src: Buf<BF16>,
    epl: usize,
    lane: Idx,
    cols_lds: i64,
    tile_row_base: Idx,
    tile_col_base: Idx,
    grow_stride: i64,
) -> Vec<Effect> {
    let epl_c = b.idx_const(epl as i64);
    let lane_epl = b.idx_mul(lane, epl_c);
    let cols_c = b.idx_const(cols_lds);
    let gstride = b.idx_const(grow_stride);
    (0..epl)
        .map(|e| {
            let ec = b.idx_const(e as i64);
            let flat = b.idx_add(lane_epl, ec);
            let r = b.idx_div(flat, cols_c);
            let c = b.idx_mod(flat, cols_c);
            let grow = b.idx_add(tile_row_base, r);
            let goff = b.idx_mul(grow, gstride);
            let goff = b.idx_add(goff, tile_col_base);
            let goff = b.idx_add(goff, c);
            let v = b.load(src, goff);
            // LDS store position `r·cols + LdsCol(r, c)` — flat by default, or the bank
            // swizzle once `SwizzlePass` materialises the LdsCol (a composable refinement).
            let col = b.lds_col(r, c, cols_lds as usize);
            let rc = b.idx_mul(r, cols_c);
            let dst_off = b.idx_add(rc, col);
            b.store_lds(lds, dst_off, v)
        })
        .collect()
}

/// Gather one 16×16 bf16 fragment from an already-staged LDS tile `lds` into `dst`,
/// ordered after the fill `bar`. The tile origin is compile-time `(row_base, col_base)`
/// in a `[.., cols]` LDS tile (the K-blocked strip's per-fragment offset), so the full
/// in-tile coord is `(row_base + frag_row, col_base + frag_col)` and the LDS offset is
/// `wr·cols + (wc | swizzle_col(wr,wc,cols))`. `swizzle` must match the fill.
#[allow(clippy::too_many_arguments)]
fn gather_frag_lds_sw(
    b: &mut Builder,
    dst: Frag<BF16>,
    lds: Lds<BF16>,
    row_base: usize,
    col_base: usize,
    cols: usize,
    lane: Idx,
    bar: Effect,
) -> Vec<TileId> {
    let cols_c = b.idx_const(cols as i64);
    (0..dst.map.ept)
        .map(|inner| {
            let inner_idx = b.idx_const(inner as i64);
            let (frag_row, frag_col) = b.lane_rc(dst.map, lane, inner_idx);
            // Whole-tile (row, col) — the swizzle XORs the *whole* column as a unit
            // (col_base spans multiple 16-fragments in the B strip), so fold the const base in.
            let wr = if row_base == 0 {
                frag_row
            } else {
                let rb = b.idx_const(row_base as i64);
                b.idx_add(frag_row, rb)
            };
            let wc = if col_base == 0 {
                frag_col
            } else {
                let cb = b.idx_const(col_base as i64);
                b.idx_add(frag_col, cb)
            };
            // `wr·cols + LdsCol(wr, wc)` — flat until `SwizzlePass` swizzles the LdsCol.
            let col_part = b.lds_col(wr, wc, cols);
            let row_off = b.idx_mul(wr, cols_c);
            let off = b.idx_add(row_off, col_part);
            let v = b.load_lds_after(lds, off, &[bar.dep()]);
            b.store_frag_elem(dst, inner_idx, v).dep()
        })
        .collect()
}

/// Gather one 16×16 bf16 fragment from an already-staged LDS tile `lds` into `dst`,
/// ordered after the fill `bar` (the cross-lane LDS read edge). Same
/// `base + row·stride + col` addressing as [`gather_frag`], but reading LDS not global.
/// Returns the `ept` store edges the operand read routes through.
fn gather_frag_lds(
    b: &mut Builder,
    dst: Frag<BF16>,
    lds: Lds<BF16>,
    base: Idx,
    row_stride: i64,
    lane: Idx,
    bar: Effect,
) -> Vec<TileId> {
    let rs = b.idx_const(row_stride);
    (0..dst.map.ept)
        .map(|inner| {
            let inner_idx = b.idx_const(inner as i64);
            let (row, col) = b.lane_rc(dst.map, lane, inner_idx);
            let row_off = b.idx_mul(row, rs);
            let off = b.idx_add(base, row_off);
            let off = b.idx_add(off, col);
            let v = b.load_lds_after(lds, off, &[bar.dep()]);
            b.store_frag_elem(dst, inner_idx, v).dep()
        })
        .collect()
}

/// **LDS-staged** naive matmul (correctness stepping stone, DESIGN.md §5b step 1a): one
/// 16×16 output tile per workgroup, but the A row-strip `[16, K]` and B col-strip
/// `[K, 16]` are staged into LDS **once** (before the K-loop, so a single fill barrier —
/// NO per-K-step refill, hence NO single-buffer WAR), then the K-loop reads fragments
/// from LDS instead of re-gathering from global. Single f32 accumulator (the naive
/// carry). This isolates the LDS-in-matmul + fill + barrier machinery from the
/// multi-accumulator + K-blocking-WAR complexity (step 1b). NOT a perf win at one output
/// tile (zero reuse — that arrives with the bigger tile); a device-correctness stone.
/// `K` must fit LDS (`64·K` bytes ≤ 64 KB ⇒ K ≤ 1024); `m`/`n`/`k` multiples of 16.
pub fn matmul_lds(m: usize, n: usize, k: usize) -> Program {
    assert!(
        m.is_multiple_of(EDGE) && n.is_multiple_of(EDGE) && k.is_multiple_of(EDGE),
        "matmul dims must be multiples of {EDGE}"
    );
    assert!(64 * k <= 64 * 1024, "matmul_lds stages the full K-strip; K={k} exceeds the LDS budget");
    let mut b = Builder::new("tk2_matmul_lds");

    let c = b.global::<F32>(m * n);
    let a = b.global::<BF16>(m * k);
    let bmat = b.global::<BF16>(k * n);

    let tile_m = b.grid_axis(0, (m / EDGE) as i64);
    let tile_n = b.grid_axis(1, (n / EDGE) as i64);
    let lane = b.block_axis(WARP as i64);

    let a_map = FragMap::gfx942_16x16(false);
    let bc_map = FragMap::gfx942_16x16(true);
    let a_frag = b.define_frag::<BF16>(a_map);
    let b_frag = b.define_frag::<BF16>(bc_map);
    let acc = b.define_frag::<F32>(bc_map);
    let ept = a_map.ept;

    // ── stage the full A[16,K] and B[K,16] tile-strips into LDS (once). ──
    let lds_a = b.define_local::<BF16>(EDGE * k);
    let lds_b = b.define_local::<BF16>(k * EDGE);
    let epl = (EDGE * k / WARP) as i64; // elements per lane (= K/4)
    let e16 = b.idx_const(EDGE as i64);
    let zero = b.idx_const(0);
    let tm16 = b.idx_mul(tile_m, e16);
    let tn16 = b.idx_mul(tile_n, e16);
    // A row-strip: LDS [16, K], global rows (tile_m·16 + r), stride K, col base 0.
    let fa = fill_lds(&mut b, lds_a, a, epl, lane, k as i64, tm16, zero, k as i64);
    // B col-strip: LDS [K, 16], global rows (kk), stride N, col base tile_n·16.
    let fb = fill_lds(&mut b, lds_b, bmat, epl, lane, EDGE as i64, zero, tn16, n as i64);
    // Fence: both strips fully staged before any lane reads a fragment.
    let bar = b.barrier(fa, &[fb.dep()]);

    // ── init: acc = 0 (the sum_reduce init shape). ──
    let init_r = b.range(ept as i64);
    let init_c = b.counter(init_r);
    let zero_f = b.f32(0.0);
    let s_init = b.store_frag_elem(acc, init_c, zero_f);
    let inited = b.end(s_init, &[init_r]);

    // ── K-loop: gather A/B fragments from LDS, one MFMA per K-fragment. ──
    let kr = b.range((k / EDGE) as i64);
    let tk = b.counter(kr);
    let e256 = b.idx_const((EDGE * EDGE) as i64);
    let a_base = b.idx_mul(tk, e16); // A-frag K-column base: tk·16 (row stride K)
    let b_base = b.idx_mul(tk, e256); // B-frag base: tk·256 (row stride 16)
    let a_stores = gather_frag_lds(&mut b, a_frag, lds_a, a_base, k as i64, lane, bar);
    let b_stores = gather_frag_lds(&mut b, b_frag, lds_b, b_base, EDGE as i64, lane, bar);

    let a_vec = b.load_frag_vec_after(a_frag, &a_stores);
    let b_vec = b.load_frag_vec_after(b_frag, &b_stores);
    let acc_vec = b.load_frag_vec_after(acc, &[inited.dep(), kr.dep()]);
    let out = b.mma(a_vec, b_vec, acc_vec, ept);
    let s_acc = b.store_frag_vec(acc, out);
    let ended = b.end(s_acc, &[kr]);

    // ── post-loop: scatter the accumulator to C. ──
    let acc_final = b.frag_after(acc, &[ended.dep()]);
    let en = b.idx_const((EDGE * n) as i64);
    let tm_en = b.idx_mul(tile_m, en);
    let tn_e = b.idx_mul(tile_n, e16);
    let base_c = b.idx_add(tm_en, tn_e);
    let roots = scatter_frag(&mut b, acc_final, c, base_c, n as i64, lane);

    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_matmul_lds".into() }
}

/// **LDS-staged, block-tiled** matmul (DESIGN.md §5b step 1b, the reuse lever): one
/// `bm × bn` output tile per workgroup (one 64-lane warp), computing an `(bm/16) ×
/// (bn/16)` grid of 16×16 accumulator fragments. A[bm,K] and B[K,bn] are staged into
/// LDS **once** (single fill barrier, no per-K-step refill ⇒ no WAR — K bounded by LDS),
/// then per K-fragment the warp gathers `bm/16` A-fragments and `bn/16` B-fragments from
/// LDS and issues `(bm/16)·(bn/16)` MFMAs — **each staged A-fragment is reused across all
/// `bn/16` columns, each B-fragment across all `bm/16` rows** (the reuse the naive/1a
/// kernels lack). The K-loop carries all `(bm/16)·(bn/16)` accumulators; its single `End`
/// closes around them via [`Builder::combine`] (one END per RANGE). Multi-accumulator +
/// reuse in isolation from the K-blocking WAR (that is step 1b-ii). `m/n` multiples of
/// `bm/bn`; `bm/bn/k` multiples of 16; `(bm·K + K·bn)·2 ≤ 64 KB`.
// The fragment-grid loops index `a_vecs[i]`/`b_vecs[j]` while also needing `i`/`j` for the
// accumulator index and the C tile origin, so a range loop is clearer than a zip.
#[allow(clippy::needless_range_loop)]
pub fn matmul_lds_tiled(m: usize, n: usize, k: usize, bm: usize, bn: usize) -> Program {
    assert!(bm.is_multiple_of(EDGE) && bn.is_multiple_of(EDGE) && k.is_multiple_of(EDGE), "tile dims multiples of 16");
    assert!(m.is_multiple_of(bm) && n.is_multiple_of(bn), "m/n must tile by bm/bn");
    assert!(2 * (bm * k + k * bn) <= 64 * 1024, "staged A+B strips ({bm}×{k} + {k}×{bn} bf16) exceed 64 KB LDS");
    let mut b = Builder::new("tk2_matmul_tiled");

    let c = b.global::<F32>(m * n);
    let a = b.global::<BF16>(m * k);
    let bmat = b.global::<BF16>(k * n);

    let tile_m = b.grid_axis(0, (m / bm) as i64);
    let tile_n = b.grid_axis(1, (n / bn) as i64);
    let lane = b.block_axis(WARP as i64);

    let a_map = FragMap::gfx942_16x16(false);
    let bc_map = FragMap::gfx942_16x16(true);
    let ept = a_map.ept;
    let (ri, cj) = (bm / EDGE, bn / EDGE); // fragment-grid rows / cols

    // ── stage the full A[bm,K] and B[K,bn] tile-strips into LDS (once). ──
    let lds_a = b.define_local::<BF16>(bm * k);
    let lds_b = b.define_local::<BF16>(k * bn);
    let zero = b.idx_const(0);
    let bm_c = b.idx_const(bm as i64);
    let bn_c = b.idx_const(bn as i64);
    let tm_bm = b.idx_mul(tile_m, bm_c); // A row base (rows): tile_m·bm
    let tn_bn = b.idx_mul(tile_n, bn_c); // B col base (cols): tile_n·bn
    let fa = fill_lds(&mut b, lds_a, a, (bm * k / WARP) as i64, lane, k as i64, tm_bm, zero, k as i64);
    let fb = fill_lds(&mut b, lds_b, bmat, (k * bn / WARP) as i64, lane, bn as i64, zero, tn_bn, n as i64);
    let bar = b.barrier(fa, &[fb.dep()]);

    // ── accumulators: one 16×16 f32 fragment per (i,j), each zero-initialised. ──
    let acc: Vec<Frag<F32>> = (0..ri * cj).map(|_| b.define_frag::<F32>(bc_map)).collect();
    let inited: Vec<Effect> = acc
        .iter()
        .map(|&ac| {
            let init_r = b.range(ept as i64);
            let init_c = b.counter(init_r);
            let zf = b.f32(0.0);
            let s = b.store_frag_elem(ac, init_c, zf);
            b.end(s, &[init_r])
        })
        .collect();

    // ── K-loop: gather A/B fragments from LDS (reused), one MFMA per (i,j,k-frag). ──
    let kr = b.range((k / EDGE) as i64);
    let tk = b.counter(kr);
    let e16 = b.idx_const(EDGE as i64);
    let e16bn = b.idx_const((EDGE * bn) as i64);
    let tk16 = b.idx_mul(tk, e16); // A-frag K-column base: tk·16 (stride K)
    let tk16bn = b.idx_mul(tk, e16bn); // B-frag K-row base: tk·16·bn (stride bn)

    // A-fragment i: LDS base i·16·K + tk·16, row stride K — reused across all columns j.
    let a_frags: Vec<Frag<BF16>> = (0..ri).map(|_| b.define_frag::<BF16>(a_map)).collect();
    let a_vecs: Vec<Val<BF16>> = (0..ri)
        .map(|i| {
            let row_off = b.idx_const((i * EDGE * k) as i64);
            let base = b.idx_add(row_off, tk16);
            let st = gather_frag_lds(&mut b, a_frags[i], lds_a, base, k as i64, lane, bar);
            b.load_frag_vec_after(a_frags[i], &st)
        })
        .collect();
    // B-fragment j: LDS base tk·16·bn + j·16, row stride bn — reused across all rows i.
    let b_frags: Vec<Frag<BF16>> = (0..cj).map(|_| b.define_frag::<BF16>(bc_map)).collect();
    let b_vecs: Vec<Val<BF16>> = (0..cj)
        .map(|j| {
            let col_off = b.idx_const((j * EDGE) as i64);
            let base = b.idx_add(tk16bn, col_off);
            let st = gather_frag_lds(&mut b, b_frags[j], lds_b, base, bn as i64, lane, bar);
            b.load_frag_vec_after(b_frags[j], &st)
        })
        .collect();

    // One MFMA per accumulator; collect the stores to close the loop around all of them.
    let mut stores: Vec<Effect> = Vec::with_capacity(ri * cj);
    for i in 0..ri {
        for j in 0..cj {
            let idx = i * cj + j;
            let acc_read = b.load_frag_vec_after(acc[idx], &[inited[idx].dep(), kr.dep()]);
            let out = b.mma(a_vecs[i], b_vecs[j], acc_read, ept);
            stores.push(b.store_frag_vec(acc[idx], out));
        }
    }
    // One END per RANGE: combine the accumulators' stores into the last, close around it.
    let last = *stores.last().expect("at least one accumulator");
    let others: Vec<TileId> = stores[..stores.len() - 1].iter().map(|e| e.dep()).collect();
    let combined = b.combine(last, &others);
    let ended = b.end(combined, &[kr]);

    // ── post-loop: each accumulator reads its final value after the loop, scatters to C. ──
    let n_c = b.idx_const(n as i64);
    let mut roots = Vec::new();
    for i in 0..ri {
        for j in 0..cj {
            let idx = i * cj + j;
            let acc_final = b.frag_after(acc[idx], &[ended.dep()]);
            // base_c = (tile_m·bm + i·16)·N + tile_n·bn + j·16
            let i16 = b.idx_const((i * EDGE) as i64);
            let row = b.idx_add(tm_bm, i16);
            let row_n = b.idx_mul(row, n_c);
            let j16 = b.idx_const((j * EDGE) as i64);
            let col = b.idx_add(tn_bn, j16);
            let base_c = b.idx_add(row_n, col);
            roots.extend(scatter_frag(&mut b, acc_final, c, base_c, n as i64, lane));
        }
    }

    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_matmul_tiled".into() }
}

/// **K-blocked, LDS-staged, block-tiled** matmul (DESIGN.md §5b step 1b-ii — the
/// occupancy win). Like [`matmul_lds_tiled`] (bm×bn tile, `(bm/16)×(bn/16)` reused
/// accumulators) but the A/B strips are re-staged **per K-fragment inside the K-loop**
/// (K_STEP = 16) instead of the whole K at once — so the LDS footprint is a tiny
/// `(bm·16 + 16·bn)·2` bytes **independent of K** (bm=bn=64 ⇒ 4 KB ⇒ ~16 resident
/// workgroups), keeping occupancy high at any K. This is the fix for 1b-i's occupancy
/// collapse. The single LDS buffer is reused every iteration, so each K-block needs two
/// workgroup barriers (mirroring tk's `gemm_core`): a **RAW** fence after the fill (reads
/// see the staged data) and a **WAR** fence after the LDS reads (the next fill must not
/// overwrite until every lane finished reading). The WAR fence is routed into the
/// accumulator reads so it is scoped inside the K-loop. `m/n` multiples of `bm/bn`;
/// `bm/bn/k` multiples of 16. Emits the LDS addressing through [`Builder::lds_col`], so
/// the flat layout is the base; `.apply(`[`SwizzlePass`](crate::passes::SwizzlePass)`)`
/// turns it into the bank-swizzled one — the swizzle is a **composable refinement**, not
/// hand-woven here (bm/bn/k_step ∈ {16,32,64} for the single-subtile swizzle).
#[allow(clippy::needless_range_loop)]
fn kblock_impl(m: usize, n: usize, k: usize, bm: usize, bn: usize, k_step: usize) -> Program {
    assert!(bm.is_multiple_of(EDGE) && bn.is_multiple_of(EDGE) && k.is_multiple_of(EDGE), "tile dims multiples of 16");
    assert!(k_step.is_multiple_of(EDGE) && k.is_multiple_of(k_step), "k_step multiple of 16, K multiple of k_step");
    assert!(m.is_multiple_of(bm) && n.is_multiple_of(bn), "m/n must tile by bm/bn");
    let mut b = Builder::new("tk2_matmul_kblock");

    let c = b.global::<F32>(m * n);
    let a = b.global::<BF16>(m * k);
    let bmat = b.global::<BF16>(k * n);

    let tile_m = b.grid_axis(0, (m / bm) as i64);
    let tile_n = b.grid_axis(1, (n / bn) as i64);
    let lane = b.block_axis(WARP as i64);

    let a_map = FragMap::gfx942_16x16(false);
    let bc_map = FragMap::gfx942_16x16(true);
    let ept = a_map.ept;
    let (ri, cj) = (bm / EDGE, bn / EDGE);
    let ksteps = k_step / EDGE; // K-fragments per staged block (amortises the 2 barriers)

    // Single-buffered K_STEP strips: A[bm,k_step], B[k_step,bn]. Footprint ∝ k_step, ⊥ K.
    let a_smem = b.define_local::<BF16>(bm * k_step);
    let b_smem = b.define_local::<BF16>(k_step * bn);
    let epl_a = bm * k_step / WARP;
    let epl_b = k_step * bn / WARP;

    let bm_c = b.idx_const(bm as i64);
    let bn_c = b.idx_const(bn as i64);
    let tm_bm = b.idx_mul(tile_m, bm_c); // A row base (rows): tile_m·bm
    let tn_bn = b.idx_mul(tile_n, bn_c); // B col base (cols): tile_n·bn

    // ── accumulators: one 16×16 f32 fragment per (i,j), zero-initialised. ──
    let acc: Vec<Frag<F32>> = (0..ri * cj).map(|_| b.define_frag::<F32>(bc_map)).collect();
    let inited: Vec<Effect> = acc
        .iter()
        .map(|&ac| {
            let init_r = b.range(ept as i64);
            let init_c = b.counter(init_r);
            let zf = b.f32(0.0);
            let s = b.store_frag_elem(ac, init_c, zf);
            b.end(s, &[init_r])
        })
        .collect();

    // ── K-loop over K/k_step blocks; re-stage the k_step-wide strips each iteration. ──
    let kr = b.range((k / k_step) as i64);
    let tk = b.counter(kr);
    let ks_c = b.idx_const(k_step as i64);
    let tk_ks = b.idx_mul(tk, ks_c); // K-block column/row base: tk·k_step

    // Fill A[bm,k_step] (K-col base tk·k_step, stride K) and B[k_step,bn] (K-row base, stride N).
    let mut fill_stores: Vec<TileId> = Vec::with_capacity(epl_a + epl_b);
    let fa = fill_lds_unrolled(&mut b, a_smem, a, epl_a, lane, k_step as i64, tm_bm, tk_ks, k as i64);
    let fb = fill_lds_unrolled(&mut b, b_smem, bmat, epl_b, lane, bn as i64, tk_ks, tn_bn, n as i64);
    let fill_head = fa[0];
    fill_stores.extend(fa.iter().skip(1).chain(fb.iter()).map(|e| e.dep()));
    // RAW fence: the whole strip is staged before any lane gathers a fragment.
    let fill_bar = b.barrier(fill_head, &fill_stores);

    // ── read the reused fragments from LDS (after the RAW fence): A[i][kf], B[kf][j].
    //    Each A-fragment is reused across all cols j, each B-fragment across all rows i;
    //    pre-gathering the whole block lets the WAR fence route into the accumulator reads. ──
    let a_frags: Vec<Vec<Frag<BF16>>> =
        (0..ri).map(|_| (0..ksteps).map(|_| b.define_frag::<BF16>(a_map)).collect()).collect();
    let b_frags: Vec<Vec<Frag<BF16>>> =
        (0..ksteps).map(|_| (0..cj).map(|_| b.define_frag::<BF16>(bc_map)).collect()).collect();
    let mut gathers: Vec<TileId> = Vec::new();
    let a_vecs: Vec<Vec<Val<BF16>>> = (0..ri)
        .map(|i| {
            (0..ksteps)
                .map(|kf| {
                    // A-frag (i,kf): a_smem[bm,k_step] row-block i, K-block kf (cols k_step).
                    let st =
                        gather_frag_lds_sw(&mut b, a_frags[i][kf], a_smem, i * EDGE, kf * EDGE, k_step, lane, fill_bar);
                    gathers.extend(st.iter().copied());
                    b.load_frag_vec_after(a_frags[i][kf], &st)
                })
                .collect()
        })
        .collect();
    let b_vecs: Vec<Vec<Val<BF16>>> = (0..ksteps)
        .map(|kf| {
            (0..cj)
                .map(|j| {
                    // B-frag (kf,j): b_smem[k_step,bn] K-block kf, col-block j (cols bn).
                    let st =
                        gather_frag_lds_sw(&mut b, b_frags[kf][j], b_smem, kf * EDGE, j * EDGE, bn, lane, fill_bar);
                    gathers.extend(st.iter().copied());
                    b.load_frag_vec_after(b_frags[kf][j], &st)
                })
                .collect()
        })
        .collect();

    // WAR fence: every lane finished reading LDS before the next K-block's fill.
    let war_head = Effect(gathers[0]);
    let war_bar = b.barrier(war_head, &gathers[1..]);

    // ── per accumulator, chain `ksteps` MFMAs (D = A·B + C over the block's K-fragments);
    //    the acc read routes through the WAR fence + its init/range loop-carry edges. ──
    let mut stores: Vec<Effect> = Vec::with_capacity(ri * cj);
    for i in 0..ri {
        for j in 0..cj {
            let idx = i * cj + j;
            let mut c_acc = b.load_frag_vec_after(acc[idx], &[inited[idx].dep(), kr.dep(), war_bar.dep()]);
            for kf in 0..ksteps {
                c_acc = b.mma(a_vecs[i][kf], b_vecs[kf][j], c_acc, ept);
            }
            stores.push(b.store_frag_vec(acc[idx], c_acc));
        }
    }
    let last = *stores.last().expect("at least one accumulator");
    let others: Vec<TileId> = stores[..stores.len() - 1].iter().map(|e| e.dep()).collect();
    let combined = b.combine(last, &others);
    let ended = b.end(combined, &[kr]);

    // ── post-loop: each accumulator reads its final value, scatters to C. ──
    let n_c = b.idx_const(n as i64);
    let mut roots = Vec::new();
    for i in 0..ri {
        for j in 0..cj {
            let idx = i * cj + j;
            let acc_final = b.frag_after(acc[idx], &[ended.dep()]);
            let i16 = b.idx_const((i * EDGE) as i64);
            let row = b.idx_add(tm_bm, i16);
            let row_n = b.idx_mul(row, n_c);
            let j16 = b.idx_const((j * EDGE) as i64);
            let col = b.idx_add(tn_bn, j16);
            let base_c = b.idx_add(row_n, col);
            roots.extend(scatter_frag(&mut b, acc_final, c, base_c, n as i64, lane));
        }
    }

    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_matmul_kblock".into() }
}

/// K-blocked LDS-reuse matmul (DESIGN.md §5b step 1b-ii), **flat LDS layout** (the base),
/// K_STEP=16 — the first tk2 matmul to beat naive at scale. See [`kblock_impl`]. Compose
/// `.apply(`[`SwizzlePass`](crate::passes::SwizzlePass)`)` for the bank-swizzled variant.
pub fn matmul_lds_kblock(m: usize, n: usize, k: usize, bm: usize, bn: usize) -> Program {
    kblock_impl(m, n, k, bm, bn, EDGE)
}

/// [`matmul_lds_kblock_ks`] with the [`SwizzlePass`](crate::passes::SwizzlePass) layout
/// refinement composed via [`Program::apply`] — the swizzle is now a top-level `.apply`
/// pass, not hand-woven. `bm/bn/k_step ∈ {16,32,64}` (single-subtile).
pub fn matmul_lds_kblock_sw(m: usize, n: usize, k: usize, bm: usize, bn: usize, k_step: usize) -> Program {
    matmul_lds_kblock_ks(m, n, k, bm, bn, k_step).apply(crate::passes::SwizzlePass)
}

/// The tunable K-blocked kernel (the **base**, flat layout): `k_step`-wide strips staged
/// per outer K-block, `k_step/16` chained MFMAs per accumulator — **amortising the two
/// per-block barriers over `k_step/16` K-fragments** (the K_STEP=16 barrier flood was the
/// measured matrix starve). Apply [`SwizzlePass`](crate::passes::SwizzlePass) for the
/// swizzle. `k_step ∈ {16,32,64}` if swizzling; bigger `k_step` = fewer barriers, more LDS
/// + operand VGPR (the occupancy trade the harness resolves).
pub fn matmul_lds_kblock_ks(m: usize, n: usize, k: usize, bm: usize, bn: usize, k_step: usize) -> Program {
    kblock_impl(m, n, k, bm, bn, k_step)
}
