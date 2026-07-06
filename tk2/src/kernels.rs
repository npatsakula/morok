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
use crate::movement::{Drain, LdsStage, LdsView, SharedTile};
use crate::pass::Pass;
use crate::pipeline::{CommitDrain, Compute, Hooks, Mem, pipeline};

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

/// **LDS-carry-through-barrier proof** (DESIGN §5b — the `kloop stages=2` prerequisite): a
/// `T`-iteration loop that reads LDS written by the *previous* iteration through a **loop-
/// carried RAW barrier**, plus a same-iteration WAR ([`Builder::store_lds_after`]) — the exact
/// edge pattern the register-staged-prefetch pipeline needs, isolated on a toy so a carry bug
/// surfaces here, not as silent garbage in the 385-TF kernel.
///
/// Per iteration `t`: gather the neighbour `lds[(lane+1)%n]` (ordered after the carried RAW
/// barrier), accumulate it, then **overwrite** `lds[lane]` with it (a rotation) *after* the WAR
/// fence, and RAW-fence the write for the next iteration. With `lds[lane] = in[lane]` seeded,
/// `S_t[lane] = in[(lane+t)%n]`, so `out[lane] = Σ_{t=0}^{T-1} in[(lane+1+t)%n]`. If `broken`,
/// the gather's loop-carry `range` edge is omitted so it always re-reads the *seed* state —
/// `out[lane] = T·in[(lane+1)%n]` — proving the test distinguishes a correct carry from a stale one.
pub fn lds_carry_loop(n: usize, t: usize, broken: bool) -> Program {
    let mut b = Builder::new("tk2_lds_carry");

    let out = b.global::<F32>(n); // slot 0
    let inp = b.global::<F32>(n); // slot 1
    let lds = b.define_local::<F32>(n);
    let acc = b.define_reg::<F32>(1);

    let lane = b.block_axis(n as i64);
    let zero_idx = b.idx_const(0);
    let one = b.idx_const(1);
    let np = b.idx_const(n as i64);

    // ── init acc = 0 (the sum_reduce init shape). ──
    let ir = b.range(1);
    let ic = b.counter(ir);
    let zf = b.f32(0.0);
    let s0 = b.store_reg(acc, ic, zf);
    let acc_init = b.end(s0, &[ir]);

    // ── prologue: lds[lane] = in[lane]; barrier (the RAW-carry seed = state S_0). ──
    let v0 = b.load(inp, lane);
    let staged = b.store_lds(lds, lane, v0);
    let raw_seed = b.barrier(staged, &[]);

    // ── loop t in 0..T. ──
    let tr = b.range(t as i64);
    let _tk = b.counter(tr);
    let lp1 = b.idx_add(lane, one);
    let nbr = b.idx_mod(lp1, np);

    // gather the neighbour, ordered after the *loop-carried* RAW barrier ([seed, range] → the
    // previous iteration's commit). `broken` drops the range edge → always re-reads S_0.
    let gather_deps: Vec<TileId> = if broken { vec![raw_seed.dep()] } else { vec![raw_seed.dep(), tr.dep()] };
    let v = b.load_lds_after(lds, nbr, &gather_deps);

    // acc += v (register carry, proven).
    let acc_c = b.load_reg_after(acc, zero_idx, &[acc_init.dep(), tr.dep()]);
    let na = b.add(acc_c, v);
    let s_acc = b.store_reg(acc, zero_idx, na);

    // WAR fence (s_acc depends on v ⇒ the gather is done), then overwrite lds[lane]=v after it,
    // and RAW-fence the write as the carry-out for the next iteration's gather.
    let war = b.barrier(s_acc, &[]);
    let commit = b.store_lds_after(lds, lane, v, &[war.dep()]);
    let raw_next = b.barrier(commit, &[]);

    // one End per RANGE: combine the acc store (register carry) + raw_next (LDS-RAW carry).
    let combined = b.combine(s_acc, &[raw_next.dep()]);
    let ended = b.end(combined, &[tr]);

    // ── post-loop: out[lane] = acc. ──
    let acc_f = b.reg_after(acc, &[ended.dep()]);
    let res = b.load_reg(acc_f, zero_idx);
    let ost = b.store(out, lane, res);

    let (ir2, sink) = b.finish(&[ost]);
    Program { ir: ir2, sink, name: "tk2_lds_carry".into() }
}

/// The gfx942 MFMA edge — one 16×16×16 fragment per workgroup, one 64-lane warp.
pub(crate) const EDGE: usize = 16;
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

    // ── init: acc = 0 via ONE constant-index `<ept×f32>` vector store — SROA-promotable, so the
    //    accumulator's load→mfma→store round-trip becomes the loop-carried phi (a variable-index
    //    scalar init loop blocks promotion, fracturing the MFMA clusters). See `zero_init_frag`. ──
    let inited = b.zero_init_frag(acc);

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

    // ── init: acc = 0 via ONE constant-index vector store (SROA-promotable; see `zero_init_frag`). ──
    let inited = b.zero_init_frag(acc);

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
    let inited: Vec<Effect> = acc.iter().map(|&ac| b.zero_init_frag(ac)).collect();

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

/// The register-staged fill bundle carried between [`kloop`]'s prefetch and commit hooks: A's and
/// B's b64/b128 chunks held in VGPRs. Since B is taken **`[N,K]`** (HK's pre-transposed layout), its
/// fill is the SAME trivial coalesced `load→ds_write` as A — no register transpose, no `v_perm`. At
/// `stages=2` the prefetch runs a K-block ahead of the commit so the global-load latency overlaps.
struct FillRegs {
    a: Vec<Val<BF16>>,
    b: Vec<Val<BF16>>,
}

/// matmul's [`Hooks`] impl for the §5c clustered [`pipeline`] — the ONLY kernel-specific part of the
/// clustered schedule. It rides the [`crate::movement`] handles (so the prefetch/commit/gather bodies
/// name no addressing) and carries `ri`/`cj`/`ept` for the per-accumulator MFMA grid. `Op` is the
/// `(A-vecs, B-vecs)` operand bundle of one K-slice; `Reg` is [`FillRegs`]. The bodies are the SAME
/// emission the whole-block path uses (`prefetch_fn`/`commit_fn` + the per-slice gather/mma) — the
/// clustered kernel is byte-identical to the pre-refactor interpreter under hash-consing.
/// One K-slice's matmul operand bundle: the `ri` A-fragments + `cj` B-fragments the compute MFMAs.
type MatmulOp = (Vec<Val<BF16>>, Vec<Val<BF16>>);

struct MatmulHooks {
    a_view: LdsView<BF16>,
    b_view: LdsView<BF16>,
    a_stage: LdsStage<BF16>,
    b_stage: LdsStage<BF16>,
    /// Phase C: HK's waitcnt-opaque `asm ds_write_b64` commit + an EXPOSED manual drain. When set, the
    /// stages (built with `Drain::Asm`) emit asm writes chained A→B into ONE prev chain, and `commit`
    /// appends ONE `s_waitcnt lgkmcnt(0)` on the last write (the RAW barrier can't auto-drain the asm).
    asm_commit: bool,
}

impl Hooks for MatmulHooks {
    type Op = MatmulOp;
    type Reg = FillRegs;

    const PREFETCH_TILES: usize = 2;

    fn prefetch(
        &mut self,
        b: &mut Builder,
        k_base: Idx,
        tile: usize,
        prev: Option<FillRegs>,
        order: &[TileId],
    ) -> (FillRegs, Vec<TileId>) {
        // Two operand tiles: 0 = A, 1 = B. HK loads A@C0 and B@C4 so each global load hides under a
        // different compute cluster; the schedule names which cluster stages which tile, and the fill
        // accumulates across them (`prev`) for the single C6 commit that writes BOTH to LDS. `order`
        // (the cluster entry) pins each tile's load into its cluster so the split survives lowering.
        let mut reg = prev.unwrap_or(FillRegs { a: Vec::new(), b: Vec::new() });
        let loaded = match tile {
            0 => {
                reg.a = self.a_stage.prefetch(b, k_base, order);
                &reg.a
            }
            1 => {
                reg.b = self.b_stage.prefetch(b, k_base, order);
                &reg.b
            }
            _ => panic!("matmul prefetch: tile ∈ {{0=A, 1=B}}, got {tile}"),
        };
        // The load result values — the `sched_fence(0)` load-pin anchors on these so LLVM cannot sink
        // the global load down to its consumer (the commit), exposing the DRAM latency.
        let anchors: Vec<TileId> = loaded.iter().map(|v| v.id).collect();
        (reg, anchors)
    }

    fn commit(&mut self, b: &mut Builder, _k_base: Idx, reg: &FillRegs, war: &[TileId]) -> Vec<Effect> {
        if self.asm_commit {
            // Asm commit (§5c): chain A then B writes into ONE `prev` chain (thread A's tail into B) so a
            // single drain reaches BOTH. Return the WRITE effects (last = `fill.last()`) — the pipeline
            // combinator owns the drain now (`CommitDrain`: exposed at C6, or deferred to C7's tail), since
            // the RAW barrier can't auto-drain the waitcnt-opaque asm and WHERE it drains is the schedule.
            let fa = self.a_stage.commit_asm(b, &reg.a, war, None);
            let a_last = fa.last().map(|e| e.dep());
            let fb = self.b_stage.commit_asm(b, &reg.b, war, a_last);
            fa.into_iter().chain(fb).collect()
        } else {
            let fa = self.a_stage.commit(b, &reg.a, war);
            let fb = self.b_stage.commit(b, &reg.b, war);
            fa.into_iter().chain(fb).collect()
        }
    }

    fn gather(&mut self, b: &mut Builder, slice: usize, raw: &[TileId]) -> (Self::Op, Vec<TileId>, TileId) {
        // One gather per operand VIEW at K-slice `slice` — the view's `asm` field dispatches the
        // `ds_read_b64` asm gather vs the scalar fallback. No addressing params: they ride the view.
        let mut gathers: Vec<TileId> = Vec::new();
        let (a_vecs, ga) = self.a_view.slice(slice).gather(b, raw);
        let (b_vecs, gb) = self.b_view.slice(slice).gather(b, raw);
        gathers.extend(ga);
        gathers.extend(gb);
        // op_anchor = an operand VALUE (the first A fragment) for `set_prio` to anchor on.
        let op_anchor = a_vecs[0].id;
        ((a_vecs, b_vecs), gathers, op_anchor)
    }
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
    prefetch: impl FnMut(&mut Builder, Idx) -> (Reg, Vec<TileId>),
    commit: impl FnMut(&mut Builder, Idx, &Reg, &[TileId]) -> Vec<Effect>,
    gather: impl FnMut(&mut Builder, usize, &[TileId]) -> (Op, Vec<TileId>),
    mma: impl FnMut(&mut Builder, &Op, &[Val<F32>]) -> Vec<Val<F32>>,
) -> Vec<Frag<F32>> {
    match stages {
        1 => kloop_1stage(b, nblocks, k_step, accs, inited, prefetch, commit, gather, mma),
        2 => kloop_2stage(b, nblocks, k_step, accs, inited, prefetch, commit, gather, mma),
        s => panic!("kloop: unsupported stages={s} (1 = single-buffer, 2 = register-staged pipeline)"),
    }
}

/// The **single-buffer** K-reduction (`stages=1`): each iteration fills its own K-block into
/// LDS (RAW fence), gathers, WAR fence, MFMAs. Two barriers per block; no cross-block overlap.
#[allow(clippy::too_many_arguments)]
fn kloop_1stage<Op, Reg>(
    b: &mut Builder,
    nblocks: usize,
    k_step: usize,
    accs: &[Frag<F32>],
    inited: &[Effect],
    mut prefetch: impl FnMut(&mut Builder, Idx) -> (Reg, Vec<TileId>),
    mut commit: impl FnMut(&mut Builder, Idx, &Reg, &[TileId]) -> Vec<Effect>,
    mut gather: impl FnMut(&mut Builder, usize, &[TileId]) -> (Op, Vec<TileId>),
    mut mma: impl FnMut(&mut Builder, &Op, &[Val<F32>]) -> Vec<Val<F32>>,
) -> Vec<Frag<F32>> {
    let kr = b.range(nblocks as i64);
    let tk = b.counter(kr);
    let ks_c = b.idx_const(k_step as i64);
    let k_base = b.idx_mul(tk, ks_c); // K-block base: tk·k_step

    // fill the K-block strip into LDS: prefetch (global→VGPR) then commit (VGPR→ds_write),
    // fused at stages=1 → RAW fence (the whole strip staged before any gather). No load-pin at
    // stages=1 (the fill is consumed this same iteration by design — no cross-block overlap).
    let (reg, _anchors) = prefetch(b, k_base);
    let fill = commit(b, k_base, &reg, &[]);
    let fill_deps: Vec<TileId> = fill[1..].iter().map(|e| e.dep()).collect();
    let raw = b.barrier(fill[0], &fill_deps);

    // gather the reused fragments → WAR fence (every lane read before the next block's fill).
    let (op, gathers) = gather(b, 0, &[raw.dep()]);
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

/// The **register-staged software pipeline** (`stages=2`, DESIGN §5b — the HK ping-pong shape,
/// proven expressible by the `lds_carry_loop` microkernel). ONE LDS buffer, but block `k+1`'s
/// global load is hoisted into VGPRs (`prefetch`) so it flies in-flight across block `k`'s MFMAs,
/// and its `ds_write` (`commit`) is deferred behind the WAR barrier — hiding the DRAM latency the
/// single-buffer kernel exposes. The loop carries THREE things across the back-edge: the register
/// accumulators (register carry, `combine`/`End`), and the RAW barrier that lets iteration `t`'s
/// gather read iteration `t-1`'s commit (LDS carry, `[raw_seed, range]`). Structure:
/// **prologue** commits block 0; **steady** `range(nblocks-1)` prefetches `k+1`, gathers `k` via the
/// carried RAW, MFMAs, WARs, commits `k+1` after the WAR; **epilogue** gathers + MFMAs the last
/// block (whose commit was the loop's final iteration). Requires `nblocks ≥ 2`.
#[allow(clippy::too_many_arguments)]
fn kloop_2stage<Op, Reg>(
    b: &mut Builder,
    nblocks: usize,
    k_step: usize,
    accs: &[Frag<F32>],
    inited: &[Effect],
    mut prefetch: impl FnMut(&mut Builder, Idx) -> (Reg, Vec<TileId>),
    mut commit: impl FnMut(&mut Builder, Idx, &Reg, &[TileId]) -> Vec<Effect>,
    mut gather: impl FnMut(&mut Builder, usize, &[TileId]) -> (Op, Vec<TileId>),
    mut mma: impl FnMut(&mut Builder, &Op, &[Val<F32>]) -> Vec<Val<F32>>,
) -> Vec<Frag<F32>> {
    assert!(nblocks >= 2, "kloop_2stage needs nblocks ≥ 2 (single-block K falls back to stages=1)");
    let ks_c = b.idx_const(k_step as i64);
    let one = b.idx_const(1);

    // ── prologue: commit block 0 into LDS (plain, no WAR — nothing to overwrite yet). No pin:
    //    the prologue loads are committed immediately, before the loop, so they need not stay
    //    in flight. ──
    let zero = b.idx_const(0);
    let (reg0, _) = prefetch(b, zero);
    let fill0 = commit(b, zero, &reg0, &[]);
    let fill0_deps: Vec<TileId> = fill0[1..].iter().map(|e| e.dep()).collect();
    let raw_seed = b.barrier(fill0[0], &fill0_deps);

    // ── steady loop over blocks 0..nblocks-1: gather k (carried RAW), MFMA, commit k+1. ──
    let kr = b.range((nblocks - 1) as i64);
    let tk = b.counter(kr);
    let k_next_idx = b.idx_add(tk, one);
    let k_next = b.idx_mul(k_next_idx, ks_c); // next block base: (tk+1)·k_step

    // prefetch block k+1 EARLY (global→VGPR, in-flight across the MFMAs) — the latency hide.
    // (The `anchors` — prefetch load tiles — feed the cluster combinator's load-pin; a lone
    // `sched_fence` here regresses, so single-buffer stages=2 stays fence-free. See §5c/3b.)
    let (reg_next, _anchors) = prefetch(b, k_next);

    // gather block k via the loop-carried RAW ([raw_seed, range] → the previous commit).
    let (op, gathers) = gather(b, 0, &[raw_seed.dep(), kr.dep()]);
    let war = b.barrier(Effect(gathers[0]), &gathers[1..]);

    // carried accumulator reads [init, range, WAR]; chain the MFMAs; store back (register carry).
    let acc_reads: Vec<Val<F32>> = accs
        .iter()
        .enumerate()
        .map(|(i, a)| b.load_frag_vec_after(*a, &[inited[i].dep(), kr.dep(), war.dep()]))
        .collect();
    let new = mma(b, &op, &acc_reads);
    let stores: Vec<Effect> = accs.iter().zip(new).map(|(a, v)| b.store_frag_vec(*a, v)).collect();

    // commit block k+1 AFTER the WAR (single buffer: the overwrite must follow this block's reads),
    // then RAW-fence it as the carry-out for the next iteration's gather.
    let fill_next = commit(b, k_next, &reg_next, &[war.dep()]);
    let fill_next_deps: Vec<TileId> = fill_next[1..].iter().map(|e| e.dep()).collect();
    let raw_next = b.barrier(fill_next[0], &fill_next_deps);

    // one `End` per RANGE: fold the other accumulators' stores AND raw_next (the LDS carry) into
    // the last store's combine, so the single End carries both the register and LDS state.
    let last = *stores.last().expect("at least one accumulator");
    let mut carried: Vec<TileId> = stores[..stores.len() - 1].iter().map(|e| e.dep()).collect();
    carried.push(raw_next.dep());
    let combined = b.combine(last, &carried);
    let ended = b.end(combined, &[kr]);
    let acc_loop: Vec<Frag<F32>> = accs.iter().map(|a| b.frag_after(*a, &[ended.dep()])).collect();

    // ── epilogue: gather + MFMA the last block (committed by the loop's final iteration; read via
    //    the End's carried RAW). No commit — nothing overwrites the strip after this. ──
    let (op_e, gathers_e) = gather(b, 0, &[ended.dep()]);
    let war_e = b.barrier(Effect(gathers_e[0]), &gathers_e[1..]);
    let acc_reads_e: Vec<Val<F32>> = acc_loop.iter().map(|a| b.load_frag_vec_after(*a, &[war_e.dep()])).collect();
    let new_e = mma(b, &op_e, &acc_reads_e);
    let stores_e: Vec<Effect> = acc_loop.iter().zip(new_e).map(|(a, v)| b.store_frag_vec(*a, v)).collect();
    acc_loop.iter().zip(stores_e).map(|(a, s)| b.frag_after(*a, &[s.dep()])).collect()
}

/// **XCD / L2 grid swizzle** (HK `GEMM:50-65` / `util.cuh:90`, ported from `tk/src/grid.rs`): remap a
/// flattened 1-D workgroup id to a `(tile_m, tile_n)` block coordinate so co-scheduled workgroups share
/// an XCD/L2 slice — gfx942 has 8 XCDs with private L2, and naive row-major block ordering gets only
/// ~36% L2 hit rate (the HK paper's ~19% chiplet win). Pure index arithmetic + a bijection over the
/// grid, so the computed C is unchanged (bit-exact — just *which* workgroup computes *which* tile).
///
/// Caller gates on `grid_m % WGM == 0`, so `group_size_m == WGM` (drops tk's `imin`); the chiplet
/// transform is applied only when `num_wgs` is a whole multiple of `NUM_XCDS·chunk` (drops tk's `where`
/// guard — a sub-`block` grid already fits one XCD sweep, so identity there is fine).
fn l2_swizzle(b: &mut Builder, wgid: Idx, grid_m: i64, grid_n: i64) -> (Idx, Idx) {
    const NUM_XCDS: i64 = 8; // gfx942 chiplet count
    const WGM: i64 = 4; // grouped-M L2 swizzle group width (HK `GEMM:48`)
    let chunk = WGM * WGM; // 16
    let block = NUM_XCDS * chunk; // 128
    // ── chiplet transform: reorder so each run of `chunk` ids lands on one XCD (exact when the grid is
    //    a whole multiple of `block`; else identity — the grid fits inside one XCD sweep). ──
    let wgid = if (grid_m * grid_n) % block == 0 {
        let (nx, ch, bl) = (b.idx_const(NUM_XCDS), b.idx_const(chunk), b.idx_const(block));
        let xcd = b.idx_mod(wgid, nx);
        let local = b.idx_div(wgid, nx);
        let chunk_idx = b.idx_div(local, ch);
        let pos = b.idx_mod(local, ch);
        let hi = b.idx_mul(chunk_idx, bl);
        let mid = b.idx_mul(xcd, ch);
        let himid = b.idx_add(hi, mid);
        b.idx_add(himid, pos)
    } else {
        wgid
    };
    // ── L2 super-group (Triton grouped-M); `group_size_m == WGM` by the caller's `grid_m % WGM == 0`. ──
    let in_group = b.idx_const(WGM * grid_n);
    let wgm_c = b.idx_const(WGM);
    let group_id = b.idx_div(wgid, in_group);
    let first_pid_m = b.idx_mul(group_id, wgm_c);
    let local = b.idx_mod(wgid, in_group);
    let local_m = b.idx_mod(local, wgm_c);
    let tile_m = b.idx_add(first_pid_m, local_m);
    let tile_n = b.idx_div(local, wgm_c);
    (tile_m, tile_n)
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
fn kblock_impl(
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    wm: usize,
    wn: usize,
    k_step: usize,
    stages: usize,
    clustered: bool,
    asm_gather: bool,
    resident: bool,
    commit_drain: CommitDrain,
    bare_seals: bool,
    pin_mfma: bool,
) -> Program {
    assert!(bm.is_multiple_of(EDGE) && bn.is_multiple_of(EDGE) && k.is_multiple_of(EDGE), "tile dims multiples of 16");
    assert!(k_step.is_multiple_of(EDGE) && k.is_multiple_of(k_step), "k_step multiple of 16, K multiple of k_step");
    assert!(wm >= 1 && wn >= 1, "at least one warp per axis");
    assert!(stages == 1 || stages == 2, "kblock: stages ∈ {{1, 2}}");
    // stages=2 (register-staged pipeline) needs ≥2 K-blocks to overlap; single-block K = stages=1.
    let stages = if k / k_step >= 2 { stages } else { 1 };
    // The clustered §5c schedule only applies at stages=2 (it decomposes the register-staged body);
    // it falls back to the whole-block hooks when the pipeline collapses to stages=1.
    let clustered = clustered && stages == 2;
    // The asm `ds_read_b64 offset:N` gather is the clustered path's spill cure (§5c) — it only
    // steers the per-slice `gather_slice` (the whole-block kloop keeps the compiler-visible gather).
    // gfx942-only; tk2 hardcodes gfx942, so the flag alone gates it.
    let asm_gather = asm_gather && clustered;
    // Compute-residency (stage the tile once, no steady-loop global load) only makes sense for the
    // clustered pipeline — it decomposes that body into the HK schedule with prefetch/commit dropped.
    let resident = resident && clustered;
    // The asm `ds_write_b64` commit (§5c Phase C) is the clustered path's waitcnt-opaque write. Gated to
    // clustered so the whole-block `commit_fn` (which does not drain) only ever sees `Drain::Intrinsic`.
    // The pipeline's `CommitDrain` (drain PLACEMENT) and the stage's `Drain` (asm-vs-intrinsic write) are
    // derived from ONE source so they cannot disagree: any asm policy ⟹ `Drain::Asm` + `asm_commit`.
    let commit_drain = if clustered { commit_drain } else { CommitDrain::IntrinsicAuto };
    // HK bare cluster seals (§5c): a bare `s_barrier` + explicit `lgkmcnt(0)` drains vs the fenced
    // barrier's implicit 9×/K-block drain. Only meaningful for the clustered per-cluster schedule.
    let bare_seals = bare_seals && clustered;
    // MFMA-cluster pin (§5c ISA fix): bracket each 32-MFMA run with a leading + trailing `sched.barrier(0)`
    // so LLVM can't fracture it. Only meaningful for the clustered per-cluster schedule.
    let pin_mfma = pin_mfma && clustered;
    let asm_commit = commit_drain != CommitDrain::IntrinsicAuto;
    let stage_drain = if asm_commit { Drain::Asm } else { Drain::Intrinsic };
    // Workgroup output tile = (bm·wm) × (bn·wn), computed by a wm×wn grid of 64-lane warps.
    let (big_m, big_n, nthreads) = (bm * wm, bn * wn, wm * wn * WARP);
    assert!(m.is_multiple_of(big_m) && n.is_multiple_of(big_n), "m/n must tile by (bm·wm)/(bn·wn)");
    let mut b = Builder::new("tk2_matmul_kblock");

    let c = b.global::<F32>(m * n);
    let a = b.global::<BF16>(m * k);
    // B is taken **`[N,K]`** (HK's pre-transposed contract): K contiguous, so the fill is the trivial
    // coalesced copy A uses — no in-kernel transpose, no `v_perm`. The whole `matmul_lds_kblock*`
    // family therefore computes `A·Bᵀ` (distinct from the pedagogical `matmul`/`matmul_lds*`, A·B).
    let bmat = b.global::<BF16>(n * k);

    // XCD/L2 grid swizzle when the M-grid is WGM(4)-aligned (all square power-of-2 shapes): flatten to
    // a 1-D grid and remap wgid→(tile_m,tile_n) for L2/chiplet locality (bit-exact). Else naive 2-D.
    let (grid_m, grid_n) = ((m / big_m) as i64, (n / big_n) as i64);
    let (tile_m, tile_n) = if grid_m % 4 == 0 {
        let wgid = b.grid_axis(0, grid_m * grid_n);
        l2_swizzle(&mut b, wgid, grid_m, grid_n)
    } else {
        (b.grid_axis(0, grid_m), b.grid_axis(1, grid_n))
    };
    let tid = b.block_axis(nthreads as i64);

    // Warp split: the fill spans all `nthreads`; each warp computes one bm×bn sub-tile at
    // (warp_row·bm, warp_col·bn). Single-warp keeps `wlane = tid` and no runtime offset
    // (byte-identical to the pre-multi-warp kernel).
    let (wlane, warp_row_off, warp_col_off, warp_row) = if wm * wn == 1 {
        (tid, None, None, None)
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
        // warp_row (the phase group, 0/1 for wm=2) — surfaced for the §5c wave-phase ping-pong.
        (wlane, Some(row_off), Some(col_off), Some(warp_row))
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

    // The read + write movement handles, minted from ONE `SharedTile` per operand so the fill and the
    // gather share the tile's `cols`/swizzle and cannot desync. `gather_view` → `LdsView` (HK's
    // `load(rt,st)`); `stage_view` → `LdsStage` (the collaborative global→LDS fill, `load(st,gl)`) —
    // the tile origin/warp-offset/K-run/map/residency ride as DATA on the handles, so the gather AND
    // commit call sites name NO addressing params. `asm_gather` is the gather's arch dispatch (gfx942
    // `ds_read_b64` vs the scalar intrinsic). The SAME handles serve the whole-block and clustered
    // paths (and FA's K/V gather + stage, by how they are constructed).
    let a_tile = SharedTile::new(a_smem, k_step);
    let b_tile = SharedTile::new(b_smem, k_step);
    let a_view = a_tile.gather_view(a_map, ri, warp_row_off, wlane, asm_gather);
    let b_view = b_tile.gather_view(bc_map, cj, warp_col_off, wlane, asm_gather);
    // A[M,K] and B[N,K] are BOTH K-contiguous → the identical trivial coalesced fill: `origin` = the
    // M/N row base, `grow_stride` = K. No transpose, no `v_perm` — B's fill IS A's fill.
    let a_stage = a_tile.stage_view(a, epl_a, tid, tm_bm, k as i64, stage_drain);
    let b_stage = b_tile.stage_view(bmat, epl_b, tid, tn_bn, k as i64, stage_drain);

    // ── accumulators: one 16×16 f32 fragment per (i,j), zero-initialised. ──
    let acc: Vec<Frag<F32>> = (0..ri * cj).map(|_| b.define_frag::<F32>(bc_map)).collect();
    let inited: Vec<Effect> = acc.iter().map(|&ac| b.zero_init_frag(ac)).collect();

    // ── the K-reduction, via the map-reduce combinator (DESIGN §5b/§5c). The prefetch/commit fills
    //    ride the `LdsStage` handle; `bracket=None` runs the whole-block gather/mma (byte-identical
    //    under hash-consing), the clustered schedule (§5c) drives the per-slice path. ──
    // prefetch (global→VGPR): the load-pin anchors returned for the sched fence.
    let mut prefetch_fn = |b: &mut Builder, k_base: Idx| {
        let a = a_stage.prefetch(b, k_base, &[]);
        let bt = b_stage.prefetch(b, k_base, &[]);
        let mut anchors: Vec<TileId> = a.iter().map(|v| v.id).collect();
        anchors.extend(bt.iter().map(|v| v.id));
        (FillRegs { a, b: bt }, anchors)
    };
    // commit (VGPR→ds_write LDS): the WAR `deps` ride each stage's LDS handle via `lds_after`.
    let mut commit_fn = |b: &mut Builder, _k_base: Idx, reg: &FillRegs, deps: &[TileId]| {
        let fa = a_stage.commit(b, &reg.a, deps);
        let fb = b_stage.commit(b, &reg.b, deps);
        fa.into_iter().chain(fb).collect()
    };

    let acc_final = if !clustered {
        // Whole-block hooks: gather all `ksteps` slices → chain all MFMAs (the stages≤2 base).
        kloop(
            &mut b,
            k / k_step,
            k_step,
            stages,
            &acc,
            &inited,
            &mut prefetch_fn,
            &mut commit_fn,
            |b, _slot, raw| {
                // Gather each K-slice via the operand views (all `ri`/`cj` fragments per slice).
                // `a_slices[kf][i]` / `b_slices[kf][j]` — the addressing rides the view.
                let mut gathers: Vec<TileId> = Vec::new();
                let a_slices: Vec<Vec<Val<BF16>>> = (0..ksteps)
                    .map(|kf| {
                        let (v, g) = a_view.slice(kf).gather(b, raw);
                        gathers.extend(g);
                        v
                    })
                    .collect();
                let b_slices: Vec<Vec<Val<BF16>>> = (0..ksteps)
                    .map(|kf| {
                        let (v, g) = b_view.slice(kf).gather(b, raw);
                        gathers.extend(g);
                        v
                    })
                    .collect();
                ((a_slices, b_slices), gathers)
            },
            |b, (a_slices, b_slices), acc_reads| {
                let mut out = Vec::with_capacity(ri * cj);
                for i in 0..ri {
                    for j in 0..cj {
                        let mut c_acc = acc_reads[i * cj + j];
                        for kf in 0..ksteps {
                            // Intrinsic MFMA — the pipe path's chained accumulators + interleaved gather
                            // let LLVM scatter the MFMAs among the memory ops, which HIDES latency (the
                            // scatter is load-bearing here: asm-pinning it measured 4.8× SLOWER, mfmautil
                            // 0.51→0.08). Asm MFMA helps only the rigid clustered schedule, not this one.
                            c_acc = b.mma(a_slices[kf][i], b_slices[kf][j], c_acc, ept);
                        }
                        out.push(c_acc);
                    }
                }
                out
            },
        )
    } else {
        // The §5c clustered HK replica: the SAME movement handles + fills as the whole-block path,
        // decomposed into the 8-cluster HK schedule (prefetch k+1 + gather slice 0 at C0, gathers
        // spread C0/C2/C4 with slice 3 read early for C7, deferred commit at C6, the four MFMA slices
        // at C1/C3/C5/C7). The pipeline combinator owns ALL placement (per-cluster barrier + set_prio
        // brackets, warp-phase ping-pong, End-fold, resident fork) and runs the completeness verifier
        // at `.build()`; the author declares only the schedule + the `MatmulHooks` (§5c cluster model).
        let hooks = MatmulHooks { a_view, b_view, a_stage, b_stage, asm_commit };
        // The compute clusters carry the kernel math (the `ri×cj` MFMA loop) as an edge-free `body` —
        // the combinator brackets it with `set_prio` + the acc round-trip. This is what makes the
        // compute side pluggable: FA's softmax/PV clusters carry their own body, `Hooks` never grows a
        // compute method. `mma(s)` mints a compute cluster over gathered slice `s` (always `Some`).
        let mma = |s: usize| -> Compute<MatmulHooks> {
            Compute::new(s, move |b: &mut Builder, op: Option<&MatmulOp>, reads: &[Val<F32>]| {
                let (a_vecs, b_vecs) = op.expect("matmul compute consumes a gathered operand");
                let mut out = Vec::with_capacity(ri * cj);
                for i in 0..ri {
                    for j in 0..cj {
                        // Asm-sideeffect MFMA (opaque to LLVM's scheduler → the 32-run cannot be
                        // fractured; tk's `mma_abt_asm` pin, verified: the intrinsic path is unpinnable).
                        out.push(b.mma_asm(a_vecs[i], b_vecs[j], reads[i * cj + j], ept));
                    }
                }
                out
            })
        };
        pipeline(
            &mut b,
            k / k_step,
            k_step,
            ksteps,
            &acc,
            &inited,
            warp_row,
            asm_gather,
            resident,
            commit_drain,
            bare_seals,
            pin_mfma,
            hooks,
        )
        .cluster(Mem::builder().prefetch([0]).gathers([0]).build()) // C0: load A, gather slice 0
        .cluster(mma(0)) // C1
        .cluster(Mem::builder().gathers([1]).build()) // C2
        .cluster(mma(1)) // C3
        .cluster(Mem::builder().prefetch([1]).gathers([2, 3]).build()) // C4: load B (HK split), gather 2,3
        .cluster(mma(2)) // C5
        .cluster(Mem::builder().commit(true).build()) // C6
        .cluster(mma(3)) // C7
        .build()
    };

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
    // The wave-phase balance + carry-completeness are verified inside `Pipeline::build` (§5c/3c).
    Program { ir, sink, name: "tk2_matmul_kblock".into() }
}

/// K-blocked LDS-reuse matmul (DESIGN.md §5b step 1b-ii), **flat LDS layout** (the base),
/// K_STEP=16 — the first tk2 matmul to beat naive at scale. See [`kblock_impl`]. Compose
/// `.apply(`[`SwizzlePass`](crate::passes::SwizzlePass)`)` for the bank-swizzled variant.
pub fn matmul_lds_kblock(m: usize, n: usize, k: usize, bm: usize, bn: usize) -> Program {
    kblock_impl(m, n, k, bm, bn, 1, 1, EDGE, 1, false, false, false, CommitDrain::IntrinsicAuto, false, false)
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
    kblock_impl(m, n, k, bm, bn, wm, wn, k_step, 1, false, false, false, CommitDrain::IntrinsicAuto, false, false)
}

/// **Register-staged pipelined** multi-warp K-blocked matmul (`stages=2`, DESIGN §5b): the
/// [`matmul_lds_kblock_mw`] tile shape, but each K-block's global load is prefetched into VGPRs
/// ahead of the MFMAs and its `ds_write` deferred behind the WAR barrier — the HK ping-pong
/// latency hide, proven expressible by `lds_carry_loop`. The scalar-gather base; compose
/// `.apply(VectorizePass).apply(SwizzlePass)` for the production variant.
#[allow(clippy::too_many_arguments)]
pub fn matmul_lds_kblock_mw_pipe(
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    wm: usize,
    wn: usize,
    k_step: usize,
) -> Program {
    kblock_impl(m, n, k, bm, bn, wm, wn, k_step, 2, false, false, false, CommitDrain::IntrinsicAuto, false, false)
}

/// The **clustered HK replica** (DESIGN §5c): [`matmul_lds_kblock_mw_pipe`]'s tile + stages=2
/// overlap, but the steady body is decomposed into the [`hk_schedule`] 8-cluster memory/compute
/// sequence with ALL scheduling placed by one interpreter — the per-cluster `sched_fence(0)` then
/// `s_barrier` boundary, the `set_prio` compute brackets, and the warp-phase ping-pong (one
/// asymmetric `wave_barrier` per warp-row). Use HK's tiling `(bm=128, bn=64, wm=2, wn=4, k_step=64)`
/// so `warp_row = warp/4` in `{0,1}` gives the two phase groups. Balance is verified at build.
#[allow(clippy::too_many_arguments)]
pub fn matmul_lds_kblock_mw_clustered(
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    wm: usize,
    wn: usize,
    k_step: usize,
) -> Program {
    kblock_impl(m, n, k, bm, bn, wm, wn, k_step, 2, true, true, false, CommitDrain::AsmDeferred, false, false)
}

/// The **bare-seal clustered HK replica** (DESIGN §5c) — identical to [`matmul_lds_kblock_mw_clustered`]
/// EXCEPT the per-cluster acq-rel-fenced `s_barrier`s are swapped for HK's bare `s_barrier` +
/// baked `sched.barrier(0)` wall, and the LDS ordering the fences dropped is re-supplied by an explicit
/// `s_waitcnt lgkmcnt(0)` before each compute cluster with undrained gathers. The fenced replica forces
/// `lgkmcnt(0)` at all 9 seals/K-block (the acquire fence); HK drains only 3× and uses a bare seal for
/// the rest — the fence is a machine-scheduler barrier that throttles MFMA overlap (`tk/arch/gfx9.rs:55`).
/// The A/B twin isolating that one variable. Use HK's tiling `(bm=128, bn=64, wm=2, wn=4, k_step=64)`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_lds_kblock_mw_clustered_bare(
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    wm: usize,
    wn: usize,
    k_step: usize,
) -> Program {
    kblock_impl(m, n, k, bm, bn, wm, wn, k_step, 2, true, true, false, CommitDrain::AsmDeferred, true, false)
}

/// The **MFMA-pinned clustered HK replica** (DESIGN §5c, the ISA-diff fix) — identical to
/// [`matmul_lds_kblock_mw_clustered`] EXCEPT each 32-MFMA compute cluster is bracketed by a LEADING +
/// trailing `sched.barrier(0)`. The ISA diff (2026-07-06) showed LLVM re-batches the fenced replica's
/// independent MFMAs into `1+31 / 1+31 / 1 / 63` runs and sinks an `s_barrier` between mfma#1 and #2 —
/// serializing LDS-read against compute (mfmautil 0.355 vs HK's 0.65). The double-sided pin makes each
/// run indivisible so neither a barrier nor an `lgkmcnt(0)` can sink in (`kernel_instr.md §2`: a single
/// positional fence did NOT hold intrinsic MFMAs). Use HK's tiling `(bm=128, bn=64, wm=2, wn=4, k_step=64)`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_lds_kblock_mw_clustered_pin(
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    wm: usize,
    wn: usize,
    k_step: usize,
) -> Program {
    kblock_impl(m, n, k, bm, bn, wm, wn, k_step, 2, true, true, false, CommitDrain::AsmDeferred, false, true)
}

/// The **compute-resident HK microkernel** (the apples-to-apples benchmark): identical to
/// [`matmul_lds_kblock_mw_clustered`] — same 256² tile, HK tiling, 8-cluster [`hk_schedule`],
/// per-cluster `s_barrier`/`set_prio`, warp-phase ping-pong, asm `ds_read_b64` gather — EXCEPT the
/// whole tile (block 0) is staged into LDS ONCE in the prologue and the steady loop **drops the
/// prefetch/commit**: it re-reads that resident block every iteration, so the loop is pure
/// `ds_read` + MFMA with **ZERO `global_load` and ZERO `ds_write`**. This isolates the clustered
/// SCHEDULE quality from memory-boundedness (the streaming kernel measures mfmautil 0.24 at 4096
/// with 32 global-loads/iter; HK's own compute-resident micro measures 0.65). NOT a full GEMM: it
/// computes `nblocks · (A[:, 0:k_step] · B[0:k_step, :])` (the resident block-0 product accumulated
/// `nblocks` times) — a well-defined, bit-exact-checkable reduction. Use HK's tiling
/// `(bm=128, bn=64, wm=2, wn=4, k_step=64)`; `k/k_step ≥ 2` (else no steady loop to measure).
#[allow(clippy::too_many_arguments)]
pub fn matmul_lds_kblock_mw_resident(
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    wm: usize,
    wn: usize,
    k_step: usize,
) -> Program {
    kblock_impl(m, n, k, bm, bn, wm, wn, k_step, 2, true, true, true, CommitDrain::AsmDeferred, false, false)
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
    kblock_impl(m, n, k, bm, bn, 1, 1, k_step, 1, false, false, false, CommitDrain::IntrinsicAuto, false, false)
}
