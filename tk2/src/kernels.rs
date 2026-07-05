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
/// without a loop nest): each lane writes `epl` elements `flat = lane·epl + e` into the
/// LDS tile position `flat` (logical `[.., cols_lds]`, `r = flat/cols_lds`, `c =
/// flat%cols_lds`). The **global** element is `(tile_row_base + gr)·grow_stride +
/// tile_col_base + gc`, where `(gr, gc) = (r, c)` for a straight fill or `(c, r)` when
/// `transpose` (so a `[bn, k_step]` LDS tile is filled from the `[K, N]` B strip: the LDS
/// row `r` is the N coordinate, the LDS col `c` is the K coordinate). `tile_*_base` may
/// carry the K-loop counter so the strip is re-staged each iteration. Returns the `epl`
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
    transpose: bool,
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
            // Global coord: (r, c) straight, or (c, r) transposed (B's [bn,k_step] tile).
            let (gr, gc) = if transpose { (c, r) } else { (r, c) };
            let grow = b.idx_add(tile_row_base, gr);
            let goff = b.idx_mul(grow, gstride);
            let goff = b.idx_add(goff, tile_col_base);
            let goff = b.idx_add(goff, gc);
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

/// **Vectorised** collaborative global→LDS fill for the **non-transposed** (A / Row) tile,
/// whose lane run is contiguous on *both* sides: each lane's `epl` elements form contiguous
/// columns within one LDS row, so the run splits into `epl/VEC` b64 chunks — ONE
/// `<VEC×bf16>` global `load_vec` (coalesced) + ONE `<VEC×bf16>` `store_lds_vec`
/// (`ds_write_b64`) per chunk, replacing `VEC` scalar load/store pairs. The store routes
/// its chunk base through `lds_col` at b64 granularity so it stays swizzle-safe (the delta
/// is `VEC`-aligned; §5b). Requires `epl % VEC == 0` and `cols_lds % VEC == 0` (so a chunk
/// never straddles a row). B stays on the scalar [`fill_lds_unrolled`] — its transpose makes
/// the global read strided, so it can't be a contiguous vector load. Returns the chunk store
/// effects to fence.
/// The **prefetch** half (global→VGPR): issue the `epl/VEC` coalesced b64 global `load_vec`s
/// into registers, no LDS write. Returns the loaded chunks in `cc` order for [`fill_lds_vec_commit`]
/// (which shares the addressing nodes via hash-consing, so prefetch-then-commit == the old
/// fused fill bit-for-bit). Splitting them lets a `stages=2` pipeline hoist the load ahead of
/// the MFMAs (register-staged prefetch; §5b).
#[allow(clippy::too_many_arguments)]
fn fill_lds_vec_prefetch(
    b: &mut Builder,
    src: Buf<BF16>,
    epl: usize,
    lane: Idx,
    cols_lds: i64,
    tile_row_base: Idx,
    tile_col_base: Idx,
    grow_stride: i64,
) -> Vec<Val<BF16>> {
    const VEC: usize = 4; // b64 = 4 bf16 (KPack / swizzle granularity)
    assert!(
        epl.is_multiple_of(VEC) && (cols_lds as usize).is_multiple_of(VEC),
        "vectorised fill needs VEC-aligned epl/cols"
    );
    let epl_c = b.idx_const(epl as i64);
    let lane_epl = b.idx_mul(lane, epl_c);
    let cols_c = b.idx_const(cols_lds);
    let gstride = b.idx_const(grow_stride);
    (0..epl / VEC)
        .map(|cc| {
            let ec = b.idx_const((cc * VEC) as i64);
            let flat = b.idx_add(lane_epl, ec); // chunk start (VEC-aligned ⇒ stays in one row)
            let r = b.idx_div(flat, cols_c);
            let c = b.idx_mod(flat, cols_c);
            // Global: contiguous `VEC`-run at (tile_row_base + r)·stride + tile_col_base + c.
            let grow = b.idx_add(tile_row_base, r);
            let goff = b.idx_mul(grow, gstride);
            let goff = b.idx_add(goff, tile_col_base);
            let goff = b.idx_add(goff, c);
            b.load_vec(src, goff, VEC)
        })
        .collect()
}

/// The **commit** half (VGPR→LDS): `ds_write_b64` each prefetched chunk into LDS at
/// `r·cols + LdsCol(r, c)` (swizzle-safe b64 granularity). `loaded` is [`fill_lds_vec_prefetch`]'s
/// output; the `r`/`c` addressing hash-cons-shares that half's nodes. Returns the store effects.
fn fill_lds_vec_commit(b: &mut Builder, lds: Lds<BF16>, loaded: &[Val<BF16>], epl: usize, lane: Idx, cols_lds: i64) -> Vec<Effect> {
    const VEC: usize = 4;
    let epl_c = b.idx_const(epl as i64);
    let lane_epl = b.idx_mul(lane, epl_c);
    let cols_c = b.idx_const(cols_lds);
    (0..epl / VEC)
        .map(|cc| {
            let ec = b.idx_const((cc * VEC) as i64);
            let flat = b.idx_add(lane_epl, ec);
            let r = b.idx_div(flat, cols_c);
            let c = b.idx_mod(flat, cols_c);
            let col = b.lds_col(r, c, cols_lds as usize);
            let rc = b.idx_mul(r, cols_c);
            let dst_off = b.idx_add(rc, col);
            b.store_lds_vec(lds, dst_off, loaded[cc])
        })
        .collect()
}

/// True when the register-transpose B fill tiles the strip exactly (every one of the
/// `nthreads` threads owns the same whole number of `VEC×VEC` micro-tiles).
/// `(k_step·bn)/VEC² % nthreads == 0`. `bn` here is the whole-workgroup B tile (bn·wn).
fn b_transpose_vec_ok(k_step: usize, bn: usize, nthreads: usize) -> bool {
    const VEC: usize = 4;
    k_step.is_multiple_of(VEC) && bn.is_multiple_of(VEC) && (k_step * bn).is_multiple_of(nthreads * VEC * VEC)
}

/// **Register-transpose** vectorised fill for the transposed B tile `b_smem[bn, k_step]`.
/// B's global read is N-contiguous but its transposed LDS write is K-contiguous — different
/// axes — so a plain vector copy can't serve both. Each lane instead cooperatively loads a
/// `VEC×VEC` micro-tile (VEC coalesced b64 rows, one per K), **transposes it in registers**
/// (`vec_extract` a column out of each row-vector, `vec_build` it back), and stores VEC b64
/// to the transposed LDS (one contiguous k_step run per N-row) — `2·VEC` vector mem ops + a
/// register shuffle replacing `2·VEC²` scalar load/stores. The LDS row (N) still exchanges
/// across lanes in shared memory; the register transpose only reconciles the fill's own
/// load/store axes. Gated by [`b_transpose_vec_ok`]. Returns the store effects to fence.
/// The **prefetch** half of the register-transpose B fill (global→VGPR): per `VEC×VEC`
/// micro-tile, load the `VEC` coalesced b64 rows into registers. Returns them per micro-tile
/// for [`fill_lds_transpose_vec_commit`] (the register transpose + `ds_write` half).
#[allow(clippy::too_many_arguments)]
fn fill_lds_transpose_vec_prefetch(
    b: &mut Builder,
    src: Buf<BF16>,
    tid: Idx,
    nthreads: usize,
    k_step: usize,
    bn: usize,
    k_base: Idx,
    n_base: Idx,
    grow_stride: i64,
) -> Vec<Vec<Val<BF16>>> {
    const VEC: usize = 4;
    let nb_count = bn / VEC;
    let per_lane = k_step * bn / (VEC * VEC) / nthreads;
    let (nb_c, vec_c, gstride) = (b.idx_const(nb_count as i64), b.idx_const(VEC as i64), b.idx_const(grow_stride));
    (0..per_lane)
        .map(|t| {
            let bi = offset_by(b, tid, t * nthreads);
            let kb = b.idx_div(bi, nb_c);
            let nb = b.idx_mod(bi, nb_c);
            let k0 = b.idx_mul(kb, vec_c);
            let n0 = b.idx_mul(nb, vec_c);
            (0..VEC)
                .map(|i| {
                    let ic = b.idx_const(i as i64);
                    let ki = b.idx_add(k0, ic);
                    let ki = b.idx_add(k_base, ki);
                    let goff = b.idx_mul(ki, gstride);
                    let goff = b.idx_add(goff, n_base);
                    let goff = b.idx_add(goff, n0);
                    b.load_vec(src, goff, VEC)
                })
                .collect()
        })
        .collect()
}

/// The **commit** half: register-transpose each micro-tile's rows (`vec_extract` a column,
/// `vec_build` it) and `ds_write_b64` to the transposed LDS. `loaded` is the prefetch output;
/// the `kb`/`nb`/`k0`/`n0` addressing hash-cons-shares that half's nodes.
fn fill_lds_transpose_vec_commit(
    b: &mut Builder,
    lds: Lds<BF16>,
    loaded: &[Vec<Val<BF16>>],
    tid: Idx,
    nthreads: usize,
    k_step: usize,
    bn: usize,
) -> Vec<Effect> {
    const VEC: usize = 4;
    let nb_count = bn / VEC;
    let (nb_c, vec_c, kstep_c) = (b.idx_const(nb_count as i64), b.idx_const(VEC as i64), b.idx_const(k_step as i64));
    let mut stores = Vec::with_capacity(loaded.len() * VEC);
    for (t, rows) in loaded.iter().enumerate() {
        let bi = offset_by(b, tid, t * nthreads);
        let kb = b.idx_div(bi, nb_c);
        let nb = b.idx_mod(bi, nb_c);
        let k0 = b.idx_mul(kb, vec_c);
        let n0 = b.idx_mul(nb, vec_c);
        // Transpose in registers + store VEC b64 to LDS row (n0+j), k_step run k0..k0+VEC.
        for j in 0..VEC {
            let col: Vec<Val<BF16>> = rows.iter().map(|&r| b.vec_extract(r, j)).collect();
            let bt = b.vec_build(&col);
            let jc = b.idx_const(j as i64);
            let nrow = b.idx_add(n0, jc);
            let col_part = b.lds_col(nrow, k0, k_step);
            let nrow_off = b.idx_mul(nrow, kstep_c);
            let dst = b.idx_add(nrow_off, col_part);
            stores.push(b.store_lds_vec(lds, dst, bt));
        }
    }
    stores
}

/// `idx + base` (folding the `base == 0` identity so the flat path stays clean).
fn offset_by(b: &mut Builder, idx: Idx, base: usize) -> Idx {
    if base == 0 {
        idx
    } else {
        let c = b.idx_const(base as i64);
        b.idx_add(idx, c)
    }
}

/// `idx + off` when a runtime offset is present (the multi-warp wave offset); the identity
/// when `None` (the single-warp path, kept byte-identical — no spurious `+0` node).
fn add_opt(b: &mut Builder, idx: Idx, off: Option<Idx>) -> Idx {
    match off {
        Some(o) => b.idx_add(idx, o),
        None => idx,
    }
}

/// **Scalar** gather of one 16×16 fragment from an `[outer, inner]` row-major LDS tile — the
/// fusible base form: `ept` per-element `load_lds_after` + `store_frag_elem`, each at
/// `outer·inner + LdsCol(outer, run+e, inner)`. It serves **both** operands via `map.transpose`
/// (A/Row runs its ept along `k_step` columns; B/Col, staged transposed as `b_smem[bn,k_step]`,
/// runs along its `k_step` rows — `transpose` picks the fixed `outer` vs the run).
///
/// The per-element `LdsCol` is the composable hole: `.apply(SwizzlePass)` XORs each element's
/// column, and `.apply(`[`VectorizePass`](crate::passes::VectorizePass)`)` fuses the `ept`
/// contiguous loads into ONE `ds_read_b64`. Both compose because the swizzle `delta` is
/// `ept`-aligned (`>>7<<3>>1` ⇒ ×4 = ept) and the run start is `ept`-aligned, so the b64 chunk
/// relocates as a unit and the run stays contiguous (§5b). Returns the `ept` store edges.
#[allow(clippy::too_many_arguments)]
fn gather_frag_lds_run(
    b: &mut Builder,
    dst: Frag<BF16>,
    lds: Lds<BF16>,
    outer_base: usize,
    outer_warp: Option<Idx>,
    run_base: usize,
    inner: usize,
    lane: Idx,
    bar: Effect,
) -> Vec<TileId> {
    let inner_c = b.idx_const(inner as i64);
    (0..dst.map.ept)
        .map(|e| {
            let e_idx = b.idx_const(e as i64);
            let (frag_row, frag_col) = b.lane_rc(dst.map, lane, e_idx);
            let (outer_frag, run_frag) = if dst.map.transpose { (frag_col, frag_row) } else { (frag_row, frag_col) };
            // The fixed-axis coordinate: intra-wave lane_rc + compile-time sub-tile base +
            // (multi-warp) the wave's runtime row/col offset into the shared LDS tile.
            let outer = offset_by(b, outer_frag, outer_base);
            let outer = add_opt(b, outer, outer_warp);
            let run = offset_by(b, run_frag, run_base);
            let col_part = b.lds_col(outer, run, inner); // the swizzle/vectorise hole
            let row_off = b.idx_mul(outer, inner_c);
            let off = b.idx_add(row_off, col_part);
            let v = b.load_lds_after(lds, off, &[bar.dep()]);
            b.store_frag_elem(dst, e_idx, v).dep()
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

/// The register-staged fill bundle carried between [`kloop`]'s prefetch and commit hooks:
/// A's b64 chunks and (vectorised path) B's per-micro-tile rows, held in VGPRs. `b = None`
/// selects the scalar B fallback (committed straight from global). At `stages=2` the prefetch
/// runs a K-block ahead of the commit so the global-load latency overlaps the MFMAs.
struct FillRegs {
    a: Vec<Val<BF16>>,
    b: Option<Vec<Vec<Val<BF16>>>>,
}

/// The **K-reduction map-reduce combinator** (DESIGN §5b — the first-class loop, so
/// ping-pong is a `stages` field, not a fragile whole-loop `.apply` pass). It owns the
/// structural skeleton — the `Range`/`End`, the per-block K base `tk·k_step`, the RAW
/// (post-fill) + WAR (post-gather) barriers, the loop-carried accumulator reads
/// (`[init, range, war]`) and their `combine`d `End`, and the post-loop `frag_after` — while
/// the caller supplies the three kernel-specific hooks: `stage` (global→LDS fill, returns its
/// fence edges; the first is the barrier head), `gather` (LDS→fragment reads, returns the
/// operand bundle `Op` + its fence edges), and `mma` (the MFMA chains over the carried
/// accumulators). This is `stages=1` — the single-buffer kernel; `stages=2` (ping-pong) will
/// reuse the SAME three hooks under a prologue/steady/epilogue expansion with a `buf%2`
/// rotation (the `slot` argument, ignored here). Returns the accumulators re-bound after the
/// loop `End` (for the caller's scatter).
#[allow(clippy::too_many_arguments)]
fn kloop<Op, Reg>(
    b: &mut Builder,
    nblocks: usize,
    k_step: usize,
    stages: usize,
    accs: &[Frag<F32>],
    inited: &[Effect],
    mut prefetch: impl FnMut(&mut Builder, Idx) -> Reg,
    mut commit: impl FnMut(&mut Builder, Idx, &Reg) -> Vec<Effect>,
    mut gather: impl FnMut(&mut Builder, usize, Effect) -> (Op, Vec<TileId>),
    mut mma: impl FnMut(&mut Builder, &Op, &[Val<F32>]) -> Vec<Val<F32>>,
) -> Vec<Frag<F32>> {
    assert_eq!(stages, 1, "kloop: only stages=1 is implemented (the stages=2 pipeline is phase 2b)");
    let kr = b.range(nblocks as i64);
    let tk = b.counter(kr);
    let ks_c = b.idx_const(k_step as i64);
    let k_base = b.idx_mul(tk, ks_c); // K-block base: tk·k_step

    // fill the K-block strip into LDS: prefetch (global→VGPR) then commit (VGPR→ds_write),
    // fused at stages=1 → RAW fence (the whole strip staged before any gather).
    let reg = prefetch(b, k_base);
    let fill = commit(b, k_base, &reg);
    let fill_deps: Vec<TileId> = fill[1..].iter().map(|e| e.dep()).collect();
    let raw = b.barrier(fill[0], &fill_deps);

    // gather the reused fragments → WAR fence (every lane read before the next block's fill).
    let (op, gathers) = gather(b, 0, raw);
    let war = b.barrier(Effect(gathers[0]), &gathers[1..]);

    // carried accumulator reads route [init, range, WAR]; chain the MFMAs; store back.
    let acc_reads: Vec<Val<F32>> = accs
        .iter()
        .enumerate()
        .map(|(i, a)| b.load_frag_vec_after(*a, &[inited[i].dep(), kr.dep(), war.dep()]))
        .collect();
    let new = mma(b, &op, &acc_reads);
    let stores: Vec<Effect> = accs.iter().zip(new).map(|(a, v)| b.store_frag_vec(*a, v)).collect();

    // one `End` per RANGE: combine the accumulators' stores into the last, close around it.
    let last = *stores.last().expect("at least one accumulator");
    let others: Vec<TileId> = stores[..stores.len() - 1].iter().map(|e| e.dep()).collect();
    let combined = b.combine(last, &others);
    let ended = b.end(combined, &[kr]);
    accs.iter().map(|a| b.frag_after(*a, &[ended.dep()])).collect()
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
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn kblock_impl(m: usize, n: usize, k: usize, bm: usize, bn: usize, wm: usize, wn: usize, k_step: usize) -> Program {
    assert!(bm.is_multiple_of(EDGE) && bn.is_multiple_of(EDGE) && k.is_multiple_of(EDGE), "tile dims multiples of 16");
    assert!(k_step.is_multiple_of(EDGE) && k.is_multiple_of(k_step), "k_step multiple of 16, K multiple of k_step");
    assert!(wm >= 1 && wn >= 1, "at least one warp per axis");
    // Workgroup output tile = (bm·wm) × (bn·wn), computed by a wm×wn grid of 64-lane warps.
    let (big_m, big_n, nthreads) = (bm * wm, bn * wn, wm * wn * WARP);
    assert!(m.is_multiple_of(big_m) && n.is_multiple_of(big_n), "m/n must tile by (bm·wm)/(bn·wn)");
    let mut b = Builder::new("tk2_matmul_kblock");

    let c = b.global::<F32>(m * n);
    let a = b.global::<BF16>(m * k);
    let bmat = b.global::<BF16>(k * n);

    let tile_m = b.grid_axis(0, (m / big_m) as i64);
    let tile_n = b.grid_axis(1, (n / big_n) as i64);
    let tid = b.block_axis(nthreads as i64);

    // Warp split: the fill spans all `nthreads`; each warp computes one bm×bn sub-tile at
    // (warp_row·bm, warp_col·bn). Single-warp keeps `wlane = tid` and no runtime offset
    // (byte-identical to the pre-multi-warp kernel).
    let (wlane, warp_row_off, warp_col_off) = if wm * wn == 1 {
        (tid, None, None)
    } else {
        let warp_c = b.idx_const(WARP as i64);
        let wn_c = b.idx_const(wn as i64);
        let bm_c = b.idx_const(bm as i64);
        let bn_c = b.idx_const(bn as i64);
        let warp = b.idx_div(tid, warp_c);
        let wlane = b.idx_mod(tid, warp_c);
        let warp_row = b.idx_div(warp, wn_c);
        let warp_col = b.idx_mod(warp, wn_c);
        let row_off = b.idx_mul(warp_row, bm_c);
        let col_off = b.idx_mul(warp_col, bn_c);
        (wlane, Some(row_off), Some(col_off))
    };

    let a_map = FragMap::gfx942_16x16(false);
    let bc_map = FragMap::gfx942_16x16(true);
    let ept = a_map.ept;
    let (ri, cj) = (bm / EDGE, bn / EDGE); // per-warp accumulator grid
    let ksteps = k_step / EDGE; // K-fragments per staged block (amortises the 2 barriers)

    // Single-buffered K_STEP strips over the WHOLE workgroup tile: A[big_m,k_step], B[big_n,k_step].
    let a_smem = b.define_local::<BF16>(big_m * k_step);
    let b_smem = b.define_local::<BF16>(k_step * big_n);
    let epl_a = big_m * k_step / nthreads;
    let epl_b = k_step * big_n / nthreads;

    let big_m_c = b.idx_const(big_m as i64);
    let big_n_c = b.idx_const(big_n as i64);
    let tm_bm = b.idx_mul(tile_m, big_m_c); // workgroup A row origin: tile_m·big_m
    let tn_bn = b.idx_mul(tile_n, big_n_c); // workgroup B col origin: tile_n·big_n

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

    // ── the K-reduction, via the map-reduce combinator (DESIGN §5b). The three hooks are
    //    this kernel's fill / gather / MFMA; the combinator owns the loop + barriers + carry. ──
    let acc_final = kloop(
        &mut b,
        k / k_step,
        k_step,
        1, // stages=1 (single-buffer); stages=2 register-staged prefetch is phase 2b
        &acc,
        &inited,
        // prefetch (global→VGPR): A b64 chunks + B micro-tile rows (or None → scalar B in commit).
        |b, k_base| {
            let a = fill_lds_vec_prefetch(b, a, epl_a, tid, k_step as i64, tm_bm, k_base, k as i64);
            let bt = b_transpose_vec_ok(k_step, big_n, nthreads)
                .then(|| fill_lds_transpose_vec_prefetch(b, bmat, tid, nthreads, k_step, big_n, k_base, tn_bn, n as i64));
            FillRegs { a, b: bt }
        },
        // commit (VGPR→ds_write LDS): A + transposed B; the scalar B fallback fills from global here.
        |b, k_base, reg| {
            let fa = fill_lds_vec_commit(b, a_smem, &reg.a, epl_a, tid, k_step as i64);
            let fb = match &reg.b {
                Some(bt) => fill_lds_transpose_vec_commit(b, b_smem, bt, tid, nthreads, k_step, big_n),
                None => fill_lds_unrolled(b, b_smem, bmat, epl_b, tid, k_step as i64, k_base, tn_bn, n as i64, true),
            };
            fa.into_iter().chain(fb).collect()
        },
        // gather: the reused A[i][kf] (warp_row rows) + B[kf][j] (warp_col N-rows) fragments —
        // scalar runs VectorizePass fuses. Returns the operand bundle + the gather (WAR) edges.
        |b, _slot, raw| {
            let a_frags: Vec<Vec<Frag<BF16>>> =
                (0..ri).map(|_| (0..ksteps).map(|_| b.define_frag::<BF16>(a_map)).collect()).collect();
            let b_frags: Vec<Vec<Frag<BF16>>> =
                (0..ksteps).map(|_| (0..cj).map(|_| b.define_frag::<BF16>(bc_map)).collect()).collect();
            let mut gathers: Vec<TileId> = Vec::new();
            let a_vecs: Vec<Vec<Val<BF16>>> = (0..ri)
                .map(|i| {
                    (0..ksteps)
                        .map(|kf| {
                            let s = gather_frag_lds_run(
                                b,
                                a_frags[i][kf],
                                a_smem,
                                i * EDGE,
                                warp_row_off,
                                kf * EDGE,
                                k_step,
                                wlane,
                                raw,
                            );
                            gathers.extend(s.iter().copied());
                            b.load_frag_vec_after(a_frags[i][kf], &s)
                        })
                        .collect()
                })
                .collect();
            let b_vecs: Vec<Vec<Val<BF16>>> = (0..ksteps)
                .map(|kf| {
                    (0..cj)
                        .map(|j| {
                            let s = gather_frag_lds_run(
                                b,
                                b_frags[kf][j],
                                b_smem,
                                j * EDGE,
                                warp_col_off,
                                kf * EDGE,
                                k_step,
                                wlane,
                                raw,
                            );
                            gathers.extend(s.iter().copied());
                            b.load_frag_vec_after(b_frags[kf][j], &s)
                        })
                        .collect()
                })
                .collect();
            ((a_vecs, b_vecs), gathers)
        },
        // mma: chain `ksteps` MFMAs per accumulator over the block's K-fragments.
        |b, (a_vecs, b_vecs), acc_reads| {
            let mut out = Vec::with_capacity(ri * cj);
            for i in 0..ri {
                for j in 0..cj {
                    let mut c_acc = acc_reads[i * cj + j];
                    for kf in 0..ksteps {
                        c_acc = b.mma(a_vecs[i][kf], b_vecs[kf][j], c_acc, ept);
                    }
                    out.push(c_acc);
                }
            }
            out
        },
    );

    // ── post-loop: each accumulator scatters its final value to C. ──
    let n_c = b.idx_const(n as i64);
    let mut roots = Vec::new();
    for i in 0..ri {
        for j in 0..cj {
            let idx = i * cj + j;
            // C row/col = workgroup origin + this warp's sub-tile offset + fragment block i/j.
            let i16 = b.idx_const((i * EDGE) as i64);
            let row = b.idx_add(tm_bm, i16);
            let row = add_opt(&mut b, row, warp_row_off);
            let row_n = b.idx_mul(row, n_c);
            let j16 = b.idx_const((j * EDGE) as i64);
            let col = b.idx_add(tn_bn, j16);
            let col = add_opt(&mut b, col, warp_col_off);
            let base_c = b.idx_add(row_n, col);
            roots.extend(scatter_frag(&mut b, acc_final[idx], c, base_c, n as i64, wlane));
        }
    }

    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_matmul_kblock".into() }
}

/// K-blocked LDS-reuse matmul (DESIGN.md §5b step 1b-ii), **flat LDS layout** (the base),
/// K_STEP=16 — the first tk2 matmul to beat naive at scale. See [`kblock_impl`]. Compose
/// `.apply(`[`SwizzlePass`](crate::passes::SwizzlePass)`)` for the bank-swizzled variant.
pub fn matmul_lds_kblock(m: usize, n: usize, k: usize, bm: usize, bn: usize) -> Program {
    kblock_impl(m, n, k, bm, bn, 1, 1, EDGE)
}

/// **Multi-warp** K-blocked matmul (DESIGN.md §5b, the bigger-tile lever): a `wm×wn` grid
/// of 64-lane warps (block size `wm·wn·64`) collaboratively fills one shared
/// `(bm·wm)×(bn·wn)` LDS tile, then each warp computes its own `bm×bn` sub-tile — 4× the
/// output tile per workgroup at the same per-warp VGPR, amortising the two barriers over
/// `wm·wn×` more MFMAs and lifting low-N occupancy. The scalar-gather base; compose
/// `.apply(VectorizePass).apply(SwizzlePass)` for the production variant.
#[allow(clippy::too_many_arguments)]
pub fn matmul_lds_kblock_mw(
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    wm: usize,
    wn: usize,
    k_step: usize,
) -> Program {
    kblock_impl(m, n, k, bm, bn, wm, wn, k_step)
}

/// The **vectorised** K-blocked kernel: the scalar-gather base with
/// [`VectorizePass`](crate::passes::VectorizePass) fusing each `ept` gather run into one
/// `ds_read_b64`. The `.apply` composition of the gather refinement (fills are already
/// vectorised in the builder — structural, not a pass; DESIGN §5b).
pub fn matmul_lds_kblock_vec(m: usize, n: usize, k: usize, bm: usize, bn: usize, k_step: usize) -> Program {
    matmul_lds_kblock_ks(m, n, k, bm, bn, k_step).apply(crate::passes::VectorizePass)
}

/// The production K-blocked kernel: scalar-gather base `.apply(VectorizePass).apply(SwizzlePass)`
/// — vectorise the gathers to `ds_read_b64` **then** bank-swizzle the b64 chunks (order
/// matters: VectorizePass fuses the clean `LdsCol(outer, run+e)` runs; after swizzle the
/// `^delta` form isn't cleanly fusible). `bm/bn/k_step ∈ {16,32,64}` (single-subtile).
pub fn matmul_lds_kblock_sw(m: usize, n: usize, k: usize, bm: usize, bn: usize, k_step: usize) -> Program {
    matmul_lds_kblock_ks(m, n, k, bm, bn, k_step).apply(crate::passes::VectorizePass).apply(crate::passes::SwizzlePass)
}

/// The tunable K-blocked kernel (the **base**, flat layout): `k_step`-wide strips staged
/// per outer K-block, `k_step/16` chained MFMAs per accumulator — **amortising the two
/// per-block barriers over `k_step/16` K-fragments** (the K_STEP=16 barrier flood was the
/// measured matrix starve). Apply [`SwizzlePass`](crate::passes::SwizzlePass) for the
/// swizzle. `k_step ∈ {16,32,64}` if swizzling; bigger `k_step` = fewer barriers, more LDS
/// + operand VGPR (the occupancy trade the harness resolves).
pub fn matmul_lds_kblock_ks(m: usize, n: usize, k: usize, bm: usize, bn: usize, k_step: usize) -> Program {
    kblock_impl(m, n, k, bm, bn, 1, 1, k_step)
}
