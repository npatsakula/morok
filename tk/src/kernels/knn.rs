//! A fused brute-force KNN: the two tile-kernel stages plus the public
//! lazy-[`Tensor`] entry point [`knn`] (its host-side prep + generic-graph tail).
//!
//! **Stage 1** ([`build_knn_score`]) is the **x²-free score tile**. For query
//! rows `x[query, d]` and corpus rows `c[corpus, d]` the score is
//! `score[m, n] = ‖c[m]‖² − 2·⟨x[n], c[m]⟩`. The query self-term `‖x[n]‖²` is
//! dropped (it is constant per query row `n`, so it never changes the argmin over
//! the corpus `m` that the running top-K in Stage 2 takes). The dominant distance
//! term `‖c[m]‖²` (`c_sq`) is precomputed in **f32** outside the kernel and passed
//! in (an augmentation that smuggled it through a bf16 WMMA operand would lose its
//! precision), replicated along the query axis so every `(m, n)` reads `c_sq[m]`.
//!
//! **Stage 2** ([`build_knn_topk`]) streams the corpus in [`TM`]`= 16`-wide tiles
//! and keeps, per query, a running top-K **sorted ascending** by a counted
//! **bitonic sorting network** (no `loop_dynamic`, no GPU-hang risk): each tile's
//! candidate scores are bitonic-argsorted once and merged into the running top-K
//! ([`crate::group`]'s `bitonic_argsort` / `bitonic_merge_topk`), replacing the old
//! serialized k-step argmin-insert. The final exact re-sort/distances still run in
//! the generic graph (Stage 3).
//!
//! Stage 2 computes the score **query-major** (`score[n, m]`, query the row, corpus
//! the **column**): the corpus axis then lands on the 16-wide lane axis
//! (`laneid % 16`) of ONE WMMA fragment, exactly the axis a cross-lane butterfly
//! sorts, so the bitonic network folds it with no LDS relayout. The cross MMA is
//! `mma_atb(score, xᵀ, cᵀ)` (the swapped-operand transpose of Stage 1), and the
//! running top-K's `[query, K-slot]` orientation makes the `[query, k]` output a
//! direct (masked) store — no output transpose.
//!
//! Stage 1 ([`build_knn_score`]) keeps the original corpus-major `score[m, n]`
//! orientation; [`score_tile`]'s `query_major` flag selects between the two. The
//! `c_sq` global is loaded into the SAME accumulator fragment as the cross MMA
//! output, so it aligns lane-for-lane; the combine `score = c_sq − 2·cross` is a
//! pair of per-lane f32 elementwise ops. Arch-portable (gfx942 wave64 / gfx1151
//! wave32) via the role-based fragment shortcuts — no hardcoded fragment.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{ConstValue, UOp};
use svod_tensor::Tensor;

use crate::Group;
use crate::arch::FragRole;
use crate::group::MoveIdx;
use crate::index::{Idx, cidx};
use crate::kernel::Kernel;
use crate::scaffold::GlSpec;
use crate::tile::{GL, RT, RegTile};
use crate::tiles::TileLayout;

/// The WMMA tile edge (K=16); the cross MMA operates on 16×16 fragments, so the
/// corpus / query / D dims must each be a multiple of it. Also the query width and
/// the top-K slot padding (`K_pad`). The corpus stream tile height is [`TM`].
const BLK: usize = 16;

/// The corpus stream-tile width for Stage 2 ([`build_knn_topk`]). The corpus is
/// streamed in `M/TM` tiles, and per tile the [`TM`]-wide candidate scores are sorted
/// by a cross-lane bitonic network and merged into the sorted running top-K. The
/// bitonic sort folds the corpus along the 16-wide lane axis (`laneid % 16`) of ONE
/// WMMA fragment, so `TM` is pinned to [`BLK`] (16): a wider tile would span multiple
/// corpus fragments, which the single-fragment butterfly cannot reach across without
/// an extra inter-fragment merge. (Tunable only by extending the network — see the
/// merge helpers in [`crate::group`].)
const TM: usize = BLK;

/// The GPU arch(es) this kernel is built for: gfx942 (CDNA3 MFMA, wave64) and
/// gfx1151 (RDNA3.5 WMMA, wave32). Both resolve the accumulator/operand fragments
/// by role through [`crate::ArchCaps`]; the launcher gates against this list.
/// Validated on gfx942 (CDNA3) and gfx1151 (RDNA3.5).
pub const KNN_SUPPORTED_ARCHS: &[svod_dtype::AmdArch] = &[svod_dtype::AmdArch::Gfx942, svod_dtype::AmdArch::Gfx1151];

const POS_INF: f64 = f64::INFINITY;

fn fconst(dt: &DType, v: f64) -> Arc<UOp> {
    UOp::const_(dt.clone(), ConstValue::Float(v))
}
fn iconst32(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(v))
}

/// Anchor a register tile's next op on the corpus-loop range carried by `m_blk`:
/// `t.after([m_blk])` when `m_blk` is the rolled loop index (`Idx::Uop`), a no-op for
/// a `Const` tile index (single-tile Stage 1). The rolled-loop re-init footgun
/// ([`crate::loop_scope`]): a constant fill with no loop dependency hoists to
/// `run_count = 1`.
fn reinit_on<'k>(t: RT<'k>, m_blk: &Idx) -> RT<'k> {
    match m_blk {
        Idx::Uop(u) => t.after(u),
        Idx::Const(_) => t,
    }
}

/// The cross-term + `c_sq` combine yielding one `[m_rows, query]` Col f32 score tile
/// `score[m, n] = ‖c[m]‖² − 2·⟨x[n], c[m]⟩` for corpus rows `[m_blk·m_rows, +m_rows)`
/// — the Stage-1 machinery, factored so Stage 2 calls it per [`TM`]-tall corpus tile
/// (and [`build_knn_score`] still uses it for the whole corpus in one tile).
/// `m_rows` is the corpus-tile height (the full `corpus` for Stage 1, [`BLK`] per
/// stream tile for Stage 2). `x_reg_t` is the loop-invariant query operand `[d,
/// query]` (the caller loads it once); `m_blk` is the corpus tile index (its
/// row-block offset, in units of `m_rows`). `masked` gates the GLOBAL→LDS/REG hops
/// against the true corpus extent so a ragged final tile reads `0.0` instead of
/// touching out-of-bounds memory (the caller then masks those rows to `+∞` for the
/// argmin).
#[allow(clippy::too_many_arguments)]
fn score_tile<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    m_rows: usize,
    query: usize,
    d: usize,
    x_reg_t: &RT<'k>,
    c_gl: &GL,
    c_sq_gl: &GL,
    m_blk: &Idx,
    masked: bool,
    query_major: bool,
) -> RT<'k> {
    let bf16 = DType::BFloat16;
    let (row, col) = (TileLayout::Row, TileLayout::Col);

    // GLOBAL(corpus tile m_blk) → LDS (swizzled) → REG, transposed to the `[d, m_rows]`
    // Col operand for the contraction over D (mirrors `fa_qk`'s Kᵀ).
    let c_smem = ker.shared_sw((m_rows, d), bf16.clone(), row);
    let c_reg = ker.operand((m_rows, d), bf16.clone(), row);
    let c_reg_t = ker.operand((d, m_rows), bf16.clone(), col);

    let c_smem = warp.load(c_smem, c_gl.clone(), MoveIdx::block((0, 0, m_blk.clone(), 0), 2));
    let c_reg = warp.load(c_reg, c_smem, MoveIdx::default());
    let c_reg_t = warp.transpose(c_reg_t, &c_reg);

    // The score `‖c[m]‖² − 2·⟨x[n],c[m]⟩` in one of two orientations:
    // - `query_major = false` (Stage 1 [`build_knn_score`]): `score[m, n]` — corpus
    //   the row, query the col (`mma_atb(cross, cᵀ, xᵀ)`); `c_sq` rides axis 2.
    // - `query_major = true` (Stage 2 [`build_knn_topk`]): `score[n, m]` — query the
    //   row, **corpus the col** (`mma_atb(cross, xᵀ, cᵀ)`), so the corpus axis lands
    //   on `laneid % 16` and the running-top-K's cross-lane bitonic sort folds it;
    //   `c_sq[m]` rides axis 3 (replicated along the query rows).
    // The MMA accumulator must be RE-ZEROED each corpus iteration: when `m_blk` is the
    // rolled corpus-loop index, anchor the zero-fill on it (`cross.after([loop_range])`,
    // exactly `fa_qk`'s `warp.zero(lp.reinit(att))`) or the constant fill hoists out of
    // the loop (`run_count = 1`) and the MMA accumulates the cross term across ALL tiles.
    // A `Const` `m_blk` (the single-tile Stage-1 path) adds no dependency.
    let acc_dims = if query_major { (query, m_rows) } else { (m_rows, query) };
    let cross = warp.zero(reinit_on(ker.acc(acc_dims, col), m_blk));
    let cross =
        if query_major { warp.mma_atb(cross, x_reg_t, &c_reg_t) } else { warp.mma_atb(cross, &c_reg_t, x_reg_t) };

    // Load c_sq_rep[m_blk] into the SAME accumulator fragment + layout as the cross
    // MMA output, so the two align lane-for-lane (both index the accumulator frag's
    // `lane_rc`; orientation-robust per the reductions/masked tests). The corpus-tile
    // offset rides the corpus axis: axis 2 when corpus is the row, axis 3 when col.
    let cs_mi = if query_major {
        MoveIdx::block((0, 0, 0, m_blk.clone()), 2)
    } else {
        MoveIdx::block((0, 0, m_blk.clone(), 0), 2)
    };
    let cs_mi = if masked { cs_mi.masked() } else { cs_mi };
    let c_sq = warp.load(ker.acc(acc_dims, col), c_sq_gl.clone(), cs_mi);

    // score = c_sq − 2·cross, all f32.
    let cross = warp.mul_scalar(cross, -2.0);
    warp.add(cross, &c_sq)
}

/// Build the x²-free KNN score-tile kernel into the bound ABI.
///
/// ABI (outputs then inputs, fixed by [`Kernel::bind_abi`]):
/// - `score` (`[1, 1, corpus, query]`, f32) — the output `‖c[m]‖² − 2·⟨x[n],c[m]⟩`.
/// - `x` (`[1, 1, query, d]`, bf16) — the query rows.
/// - `c` (`[1, 1, corpus, d]`, bf16) — the corpus rows.
/// - `c_sq_rep` (`[1, 1, corpus, query]`, f32) — `‖c[m]‖²` precomputed outside the
///   kernel and replicated along the query axis (each `(m, n)` holds `c_sq[m]`).
///
/// Single-warp; `corpus`, `query`, `d` must each be a multiple of [`BLK`] (16).
///
/// # Panics
/// Panics unless `corpus`, `query`, and `d` are each a multiple of 16.
pub fn build_knn_score(ker: &Kernel, corpus: usize, query: usize, d: usize) {
    Kernel::assert_divisible(corpus, BLK, "KNN corpus");
    Kernel::assert_divisible(query, BLK, "KNN query");
    Kernel::assert_divisible(d, BLK, "KNN D");

    let bf16 = DType::BFloat16;
    let f32 = DType::Float32;
    let warp = ker.warp();

    // ABI: output (score, f32) then inputs (x, c — bf16; c_sq_rep — f32).
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[1, 1, corpus, query], f32.clone())],
        &[
            GlSpec::new(&[1, 1, query, d], bf16.clone()),
            GlSpec::new(&[1, 1, corpus, d], bf16.clone()),
            GlSpec::new(&[1, 1, corpus, query], f32.clone()),
        ],
    );
    let (score_gl, x_gl, c_gl, c_sq_gl): (GL, GL, GL, GL) =
        (outs[0].clone(), ins[0].clone(), ins[1].clone(), ins[2].clone());

    // Query tile loaded once and transposed to its `[d, query]` Col fragment.
    let x_reg_t = load_query_t(ker, &warp, query, d, &x_gl, &Idx::Const(0));

    // The whole corpus in one `(corpus, query)` tile (the Stage-1 single-store shape).
    let score = score_tile(ker, &warp, corpus, query, d, &x_reg_t, &c_gl, &c_sq_gl, &Idx::Const(0), false, false);
    let _ = warp.store(score_gl, score, MoveIdx::block((0, 0, 0, 0), 2));
}

/// Load the query tile `[query, d]` and transpose it to its `[d, query]` Col
/// operand fragment for the cross contraction over D — loop-invariant, so both
/// builders load it once. `q_blk` is the query-block index this workgroup owns (in
/// query-tile-height units): the GLOBAL→LDS load offsets axis 2 by it, so the grid's
/// `block_idx[0]` selects each 16-query block of a wide `[Npad, d]` query input
/// (block-unit offset ⇒ the element offset is `q_blk·query`, exactly `q_blk·16`).
/// `Idx::Const(0)` is the single-block (Stage-1 / `N ≤ 16`) path.
fn load_query_t<'k>(ker: &'k Kernel, warp: &Group<'k>, query: usize, d: usize, x_gl: &GL, q_blk: &Idx) -> RT<'k> {
    let bf16 = DType::BFloat16;
    let (row, col) = (TileLayout::Row, TileLayout::Col);
    let x_smem = ker.shared_sw((query, d), bf16.clone(), row);
    let x_reg = ker.operand((query, d), bf16.clone(), row);
    let x_reg_t = ker.operand((d, query), bf16.clone(), col);
    let x_smem = warp.load(x_smem, x_gl.clone(), MoveIdx::block((0, 0, q_blk.clone(), 0), 2));
    let x_reg = warp.load(x_reg, x_smem, MoveIdx::default());
    warp.transpose(x_reg_t, &x_reg)
}

/// Per-query running top-K state: two `Col`-layout `[query, K_pad=BLK]` register
/// tiles — `val` (f32) and `idx` (Int32) — kept **sorted ascending** along the
/// K-slot (column = `laneid % 16`) axis. Slots `[0, k)` hold the running `k` smallest
/// (ascending); slots `[k, 16)` are `+∞` padding (idx `−1`). Each corpus tile's
/// candidates are bitonic-argsorted and merged in via [`Group::bitonic_merge_topk`],
/// which keeps the 16 smallest of the two ascending runs.
struct TopK<'k> {
    val: RT<'k>,
    idx: RT<'k>,
}

/// Build the x²-free KNN running-top-K kernel into the bound ABI.
///
/// ABI (outputs then inputs):
/// - `idx` (`[1, 1, query, k]`, Int32) — the K nearest corpus indices per query,
///   sorted ascending by score (the final exact re-sort still runs in Stage 3).
/// - `val` (`[1, 1, query, k]`, f32) — their x²-free scores (ascending).
/// - `x` (`[1, 1, query, d]`, bf16) — the query rows.
/// - `c` (`[1, 1, corpus, d]`, bf16) — the corpus rows.
/// - `c_sq_rep` (`[1, 1, query, corpus]`, f32) — `‖c[m]‖²` replicated along query
///   (the query-major orientation: corpus is the last axis).
///
/// Single-warp, correctness-first, arch-portable via role fragments. The corpus is
/// streamed in [`TM`]`= 16`-wide tiles through a [`crate::loop_scope::Loop`]; per
/// tile the candidate scores are **bitonic-argsorted** once and merged into the
/// sorted running top-K ([`TopK`]) — replacing the serialized k-step argmin-insert
/// with a counted sorting network. Built **rolled**.
///
/// The score is computed **query-major** (`score[n, m]`, corpus the column), so the
/// corpus axis lands on the 16-wide lane axis (`laneid % 16`) of one WMMA fragment
/// and the cross-lane bitonic network folds it directly. The running top-K then
/// already has query rows / K-slot columns, so the `[query, k]` output is a direct
/// (masked) store with no transpose.
///
/// **Query-block grid tiling:** each workgroup processes ONE `query`(= [`BLK`]) block,
/// selected by `block_idx[0]` — the grid is `[ceil(Npad/16), 1, 1]`, so it covers a
/// wide `[Npad, *]` query input. The block index offsets ONLY the query (x) load and
/// the output store (query-independent steps — the corpus stream, score, `c_sq` load,
/// sort/merge — are block-relative); a `[1,1,1]` grid ⇒ block 0 ⇒ the single-block
/// path. The `[1,1,query,*]` x/output globals address the wider real buffers because
/// the offset rides the (identical) row stride, not the declared extent.
///
/// # Panics
/// Panics unless `query`/`d` are multiples of [`BLK`], `corpus > 0`, `1 ≤ k ≤ BLK`,
/// and `query ≤ BLK` (the single-query-fragment constraint).
pub fn build_knn_topk(ker: &Kernel, corpus: usize, query: usize, d: usize, k: usize) {
    Kernel::assert_divisible(query, BLK, "KNN topk query");
    Kernel::assert_divisible(d, BLK, "KNN topk D");
    Kernel::assert_divisible(TM, BLK, "KNN topk TM");
    assert!(corpus > 0, "KNN topk corpus must be > 0");
    assert!((1..=BLK).contains(&k), "KNN topk k must be in 1..=16");
    assert!(query <= BLK, "KNN topk query must be <= 16 (single query fragment)");

    let bf16 = DType::BFloat16;
    let f32 = DType::Float32;
    let i32 = DType::Int32;
    let col = TileLayout::Col;
    let warp = ker.warp();
    let acc_frag = ker.caps.frag(FragRole::Accumulator);

    // ABI: outputs (idx i32, val f32) then inputs (x, c — bf16; c_sq_rep — f32). The
    // c_sq is query-major (`[1,1,query,corpus]`): `‖c[m]‖²` replicated along query.
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[1, 1, query, k], i32.clone()), GlSpec::new(&[1, 1, query, k], f32.clone())],
        &[
            GlSpec::new(&[1, 1, query, d], bf16.clone()),
            GlSpec::new(&[1, 1, corpus, d], bf16.clone()),
            GlSpec::new(&[1, 1, query, corpus], f32.clone()),
        ],
    );
    let (idx_gl, val_gl, x_gl, c_gl, c_sq_gl): (GL, GL, GL, GL, GL) =
        (outs[0].clone(), outs[1].clone(), ins[0].clone(), ins[1].clone(), ins[2].clone());

    let q_blk = Idx::Uop(ker.block_idx[0].clone());
    let x_reg_t = load_query_t(ker, &warp, query, d, &x_gl, &q_blk);

    // Running top-K state (Col `[query, K_pad=BLK]`, K-slot = column = `laneid % 16`),
    // sorted ascending. All 16 slots seed to `+∞` (idx `−1`): the bitonic merge keeps
    // the 16 smallest, so real candidates always beat the `+∞` padding and settle into
    // the leading `k` slots. No `−∞` padding trick is needed (that was the Max-evict's).
    let val0 = warp.map(ker.acc((query, BLK), col), |x, _| fconst(&x.dtype(), POS_INF));
    let idx0 = warp.map(ker.rt((query, BLK), i32.clone(), col, acc_frag), |_, _| iconst32(-1));
    let topk = TopK { val: val0, idx: idx0 };

    // Stream the corpus in TM-wide tiles via the FA running-state Loop carry.
    let tiles = corpus.div_ceil(TM);
    let masked = !corpus.is_multiple_of(TM);
    let lp = ker.loop_static(tiles as i64);
    let m_tile = lp.index().clone();
    let topk = TopK { val: lp.reinit(topk.val), idx: lp.reinit(topk.idx) };

    let topk = topk_merge(ker, &warp, corpus, query, d, &x_reg_t, &c_gl, &c_sq_gl, &m_tile, masked, topk);

    // Close the loop once: `bitonic_merge_topk`'s final `arg_compare_exchange` is the
    // body's terminal grouped (val, idx) store; the single loop-closing END scopes the
    // whole sort/merge body. Both carried tiles read their post-loop value via `.after`.
    let ended = lp.close();
    let idx_after = topk.idx.after(&ended);
    let val_after = topk.val.after(&ended);

    store_topk(&warp, k, &idx_gl, &val_gl, &idx_after, &val_after, &q_blk);
}

/// One corpus tile's sort-and-merge: compute the query-major score sub-tile
/// (`[query, TM]`, corpus = column = `laneid % 16`), mask ragged corpus columns to
/// `+∞`, tag each candidate with its global corpus index, bitonic-argsort the
/// candidates ascending, then merge them into the sorted running top-K (keep the 16
/// smallest). Returns the updated running top-K.
#[allow(clippy::too_many_arguments)]
fn topk_merge<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    corpus: usize,
    query: usize,
    d: usize,
    x_reg_t: &RT<'k>,
    c_gl: &GL,
    c_sq_gl: &GL,
    m_tile: &Arc<UOp>,
    masked: bool,
    topk: TopK<'k>,
) -> TopK<'k> {
    let i32 = DType::Int32;
    let col = TileLayout::Col;
    let acc_frag = ker.caps.frag(FragRole::Accumulator);

    // score[query, corpus] for this corpus tile (query-major, corpus = column).
    let mut score =
        score_tile(ker, warp, TM, query, d, x_reg_t, c_gl, c_sq_gl, &Idx::Uop(m_tile.clone()), masked, true);
    if masked {
        score = mask_ragged_cols(warp, score, m_tile, corpus);
    }

    // Each candidate's global corpus index `m_tile·TM + col` (the in-tile column is the
    // arch-correct `lane_col`, lifted by the corpus-tile block via `map_position`).
    let cand_idx = warp.map_position(
        ker.rt((query, TM), i32, col, acc_frag),
        Idx::Const(0),
        Idx::Uop(m_tile.clone()),
        |_, _, _, corpus_col| corpus_col.cast(DType::Int32),
    );

    // Sort the candidates ascending, then merge into the running top-K (smallest 16).
    let (sorted_v, sorted_i) = warp.bitonic_argsort(score, cand_idx);
    // The merge yields FRESH buffers; copy them back into the loop-carried `topk`
    // tiles (the rolled-loop carry needs the SAME buffer read-then-written each trip,
    // so the next iteration sees this tile's result rather than the `+∞` seed). The
    // idx copy chains after the val copy so one loop-closing END scopes both.
    let (merged_v, merged_i) = warp.bitonic_merge_topk(&topk.val, &topk.idx, &sorted_v, &sorted_i);
    let val = warp.copy(topk.val, &merged_v);
    let idx = warp.copy(topk.idx.after(&val), &merged_i);
    TopK { val, idx }
}

/// Mask ragged corpus columns (`global_m ≥ corpus`) of a query-major Col
/// `[query, TM]` score tile to `+∞` via [`Group::mask_where`], so the bitonic sort
/// never ranks the padding the masked score load zeroed. The per-element corpus
/// column (`global_m = m_tile·TM + lane_col`) is computed arch-correctly inside
/// `mask_where`; the corpus-tile block offset rides the column (`col_blk`).
fn mask_ragged_cols<'k>(warp: &Group<'k>, score: RT<'k>, m_tile: &Arc<UOp>, corpus: usize) -> RT<'k> {
    let bound = cidx(corpus as i64);
    warp.mask_where(score, Idx::Const(0), Idx::Uop(m_tile.clone()), POS_INF, move |_, global_m| global_m.ge(&bound))
}

/// Store the sorted running top-K to the `[1, 1, query, k]` outputs. The running
/// tiles are already Col `[query, K_slot=BLK]` (query the row, K-slot the column), so
/// the `[query, k]` output is a DIRECT store of the leading `k` columns — no
/// transpose (the query-major orientation's payoff). `k == BLK` needs no mask; a
/// partial `k` gates the trailing columns via the boundary mask. The query-row block
/// is offset by `q_blk` (the store-side mirror of `load_query_t`).
fn store_topk<'k>(
    warp: &Group<'k>,
    k: usize,
    idx_gl: &GL,
    val_gl: &GL,
    idx_after: &RT<'k>,
    val_after: &RT<'k>,
    q_blk: &Idx,
) {
    let mi = if k.is_multiple_of(BLK) {
        MoveIdx::block((0, 0, q_blk.clone(), 0), 2)
    } else {
        MoveIdx::block((0, 0, q_blk.clone(), 0), 2).masked()
    };
    // Both global stores come last on the store stack, so `finish(2)` pops exactly
    // these two output writes as the SINK sources.
    let _ = warp.store(val_gl.clone(), val_after.clone(), mi.clone());
    let _ = warp.store(idx_gl.clone(), idx_after.clone(), mi);
}

/// Round `x` up to the next multiple of [`BLK`] (16) — the WMMA tile edge the
/// kernel's `D`/query block geometry requires.
fn pad16(x: usize) -> usize {
    x.div_ceil(BLK) * BLK
}

// =============================================================================
// Stage 3 — the public lazy-Tensor KNN entry point + the generic-graph tail.
// =============================================================================

/// **Graph-native** fused brute-force K-nearest-neighbors — the matmul/FA peer for
/// KNN, returning lazy output [`Tensor`]s (the tile kernel is a `custom_kernel` /
/// `Op::Call` node, the K-ordering + exact distances are ordinary generic-graph ops).
///
/// For `N` query rows `x` (`[N, D]`) and `M` corpus rows `c` (`[M, D]`, **any float
/// dtype**) it returns `Some((dists, idxs))`:
/// - `idxs` (`[N, k]`, i32) — the `k` nearest corpus rows per query, **sorted
///   ascending by distance** (ties → smaller corpus index, matching a brute-force
///   reference / [`Tensor::topk`]).
/// - `dists` (`[N, k]`, f32) — their **true** squared-L2 distances
///   `‖x[n] − c[idxs[n,j]]‖²`, recomputed exactly in f32 (the kernel's x²-free score
///   only orders the corpus; the self-term `‖x‖²` is re-added here).
///
/// The kernel streams the corpus and keeps the running top-K from the x²-free score
/// `‖c[m]‖² − 2·⟨x[n],c[m]⟩` ([`build_knn_topk`]); this entry owns the host-side prep
/// (cast → bf16, zero-pad `D`/`N` to the WMMA edge, the f32 `‖c‖²`) and the
/// generic-graph tail (sort the K, gather the sorted corpus rows, exact f32
/// distances). The corpus `M` is NOT padded — the kernel ragged-masks its final tile.
///
/// Like [`crate::matmul`] / [`crate::flash_attention_with`], the outcome is three-way
/// (via [`crate::launch_custom`]):
/// - `Ok(Some((dists, idxs)))` — ran (lazy nodes; `prepare()` to realize).
/// - `Ok(None)` — the device isn't a supported arch ([`KNN_SUPPORTED_ARCHS`] —
///   gfx942 / gfx1151 with the AMD toolchain). The caller substitutes its own KNN.
/// - `Err` — a malformed request on a supported device: `x`/`c` not statically-shaped
///   rank-2 tensors, mismatched `D`, `k > M`, or `k` outside the kernel's `1..=16`.
///   These are caller bugs (a genuine kernel build/dispatch failure also returns `Err`).
///
/// ```no_run
/// use svod_tensor::Tensor;
/// let x = Tensor::randn(&[40, 20]).unwrap(); // 40 queries, dim 20
/// let c = Tensor::randn(&[100, 20]).unwrap(); // 100 corpus rows
/// if let Some((mut dists, mut idxs)) = svod_tk::knn(&x, &c, 5).unwrap() {
///     dists.prepare().unwrap(); // [40, 5] f32 squared-L2 to the 5 nearest
///     idxs.prepare().unwrap();  // [40, 5] i32 corpus indices (ascending by distance)
/// }
/// ```
pub fn knn(x: &Tensor, c: &Tensor, k: usize) -> crate::LaunchResult<Option<(Tensor, Tensor)>> {
    use snafu::{ResultExt, ensure};

    let xd = crate::launch::concrete_dims(x, "knn", "x", 2)?;
    let cd = crate::launch::concrete_dims(c, "knn", "c", 2)?;
    let (n, dx) = (xd[0], xd[1]);
    let (m, dc) = (cd[0], cd[1]);

    // Structural validity (`Err`) — checked BEFORE arch resolution, like `concrete_dims`:
    // D mismatch and the k bounds are FIXED request properties, so a violation is a
    // caller bug regardless of the device (never silently `None`).
    ensure!(dx == dc, crate::launch::OperandDimMismatchSnafu { kernel: "knn", dim: "D", a: dx, b: dc });
    ensure!(
        (1..=BLK).contains(&k),
        crate::launch::DimMultipleSnafu { kernel: "knn", dim: "k (must be 1..=16)", value: k, multiple: 1usize }
    );
    ensure!(
        k <= m,
        crate::launch::DimMultipleSnafu { kernel: "knn", dim: "k (must be <= corpus M)", value: k, multiple: m }
    );

    // The three-way policy (cf. `launch_custom`), inlined because the build yields a
    // tuple (`launch_custom` is single-Tensor): `None` for the wrong arch/toolchain
    // (caller's fallback), `Err` for a malformed request (handled above), `Some` when run.
    let Some(arch) = crate::target::resolve_supported_arch(&x.device(), KNN_SUPPORTED_ARCHS).ok() else {
        return Ok(None);
    };

    let caps = crate::ArchCaps::for_arch(arch);
    let (f32, bf16) = (DType::Float32, DType::BFloat16);
    let d_pad = pad16(dx);
    let n_pad = pad16(n);

    // f32 copies for the exact-distance tail (corpus stays unpadded — the tail gathers
    // TRUE D-rows; the query keeps its N rows).
    let x_f32 = x.cast(f32.clone()).context(crate::launch::OperandSnafu)?;
    let c_f32 = c.cast(f32.clone()).context(crate::launch::OperandSnafu)?;

    // Kernel bf16 operands, zero-padded to the WMMA edge. Zeros contribute 0 to ⟨x,c⟩
    // and to ‖c‖², so the score is unchanged; padded query rows produce junk top-Ks the
    // tail slices off. `try_pad` pads with zeros.
    let x_bf = pad_operand(&x.cast(bf16.clone()).context(crate::launch::OperandSnafu)?, n, dx, n_pad, d_pad)?;
    let c_bf = pad_operand(&c.cast(bf16.clone()).context(crate::launch::OperandSnafu)?, m, dc, m, d_pad)?;

    // c_sq[m] = Σ_d c[m,d]² in f32 (query-independent), replicated to the kernel's
    // [1,1,M,BLK] (one query-block width — every query block reads the same slice).
    let c_sq_rep = c_sq_replicated(&c_f32, m)?;

    let idx_t = Tensor::empty(&[1, 1, n_pad, k], DType::Int32);
    let val_t = Tensor::empty(&[1, 1, n_pad, k], f32.clone());
    let grid = [(n_pad / BLK) as i64, 1, 1];
    let block = caps.wave_size as i64;

    // The kernel processes ONE 16-query block per workgroup (`query = BLK`);
    // `block_idx[0]` selects the block, so its declared `[1,1,BLK,*]` x/output globals
    // address the wider real `[1,1,Npad,*]` buffers (identical row stride).
    let outs = crate::graph_launch_multi(
        "knn_topk",
        grid,
        block,
        vec![idx_t, val_t],
        &[&x_bf, &c_bf, &c_sq_rep],
        caps,
        move |ker| {
            build_knn_topk(ker, m, BLK, d_pad, k);
            ker.finish(2)
        },
    )?;
    let (idx_raw, val_raw) = (outs[0].clone(), outs[1].clone());

    knn_tail(&idx_raw, &val_raw, &x_f32, &c_f32, n, dx, k).map(Some)
}

/// Zero-pad a `[rows, d]` tensor's last (`D`) axis to `d_pad` and its leading (row)
/// axis to `rows_pad`, then add the kernel's `[1, 1, …]` leading singleton axes —
/// the bf16 kernel operand layout. Zeros are the additive identity in ⟨x,c⟩/‖c‖², so
/// the padding leaves the score unchanged; padded query rows are sliced off in the tail.
fn pad_operand(t: &Tensor, rows: usize, d: usize, rows_pad: usize, d_pad: usize) -> crate::LaunchResult<Tensor> {
    use snafu::ResultExt;
    let padded = t
        .try_pad(&[(0, (rows_pad - rows) as isize), (0, (d_pad - d) as isize)])
        .context(crate::launch::OperandSnafu)?;
    padded.try_reshape([1isize, 1, rows_pad as isize, d_pad as isize]).context(crate::launch::OperandSnafu)
}

/// `c_sq[m] = Σ_d c_f32[m,d]²` in f32, replicated to the kernel's **query-major**
/// `[1, 1, BLK, M]` `c_sq_rep` operand (each `(n, m)` reads `c_sq[m]` — corpus the
/// last axis, broadcast along the `BLK = 16` query rows). `c_sq` is query-independent,
/// so one query-block height suffices regardless of `N` — every query block reads the
/// same slice. The query-major layout matches [`build_knn_topk`]'s `score[n, m]` tile.
fn c_sq_replicated(c_f32: &Tensor, m: usize) -> crate::LaunchResult<Tensor> {
    use snafu::ResultExt;
    let c_sq = c_f32
        .try_mul(c_f32)
        .context(crate::launch::OperandSnafu)?
        .sum_with()
        .axes(1isize)
        .keepdim(true)
        .call()
        .context(crate::launch::OperandSnafu)?; // [M, 1]
    c_sq.try_reshape([1isize, 1, 1, m as isize])
        .context(crate::launch::OperandSnafu)?
        .try_expand([1isize, 1, BLK as isize, m as isize])
        .context(crate::launch::OperandSnafu)
}

/// The generic-graph tail over the kernel's UNSORTED top-K (`idx_raw`/`val_raw`,
/// `[1,1,Npad,k]`): slice off the padded query rows, sort the `k` per query ascending
/// by the x²-free score (its order equals the true-distance order — `‖x‖²` is constant
/// per query), gather the sorted corpus rows, and recompute the EXACT f32 squared-L2.
/// Returns `(dists [N,k] f32, idx_sorted [N,k] i32)`.
fn knn_tail(
    idx_raw: &Tensor,
    val_raw: &Tensor,
    x_f32: &Tensor,
    c_f32: &Tensor,
    n: usize,
    d: usize,
    k: usize,
) -> crate::LaunchResult<(Tensor, Tensor)> {
    use snafu::ResultExt;
    // Each tensor-op `?` boxes into the launch `Error` (`OperandSnafu`) inline — the
    // launch enum keeps its sources boxed (`clippy::result_large_err`), so the tail
    // never surfaces the large `svod_tensor` Result.
    let op = crate::launch::OperandSnafu;

    // 1. [1,1,Npad,k] → [Npad,k] → [N,k] (drop the padded-query rows).
    let idx = idx_raw
        .try_reshape([-1, k as isize])
        .context(op)?
        .try_shrink([(0, n as isize), (0, k as isize)])
        .context(op)?;
    let val = val_raw
        .try_reshape([-1, k as isize])
        .context(op)?
        .try_shrink([(0, n as isize), (0, k as isize)])
        .context(op)?;

    // 2. Sort the k per query ascending by the x²-free score; reorder the indices to
    //    match. The score order == the true-distance order (‖x‖² is a per-query const).
    let (_val_sorted, perm) = val.sort(1, false).context(op)?;
    let idx_sorted = idx.gather(1, &perm).context(op)?;

    // 3. Exact f32 distances for the sorted indices: gather the TRUE (unpadded) corpus
    //    rows c_f32[idx_sorted] → [N, k, D], then ‖x[n] − c_gathered‖² over D.
    let idx_flat = idx_sorted.try_reshape([(n * k) as isize]).context(op)?;
    let c_gathered =
        c_f32.index_select(0, &idx_flat).context(op)?.try_reshape([n as isize, k as isize, d as isize]).context(op)?;
    let diff = x_f32.try_reshape([n as isize, 1, d as isize]).context(op)?.try_sub(&c_gathered).context(op)?;
    let dists = diff.try_mul(&diff).context(op)?.sum_with().axes(2isize).dtype(DType::Float32).call().context(op)?;

    Ok((dists, idx_sorted))
}
