//! The arch-generic kernel machinery for the tile matmul: the single-buffered
//! GEMM core ([`gemm_core`]) and its split-K sibling ([`gemm_core_splitk`]), the
//! thin ABI-binding builders ([`build_matmul_cfg`] / [`build_matmul_splitk`]),
//! the split-K size heuristic ([`split_k_for`]), and the
//! shared block/accumulator coordinate helpers ([`acc_row`] / [`block_coords`]).
//! One builder serves every arch — the tile shortcuts ([`Kernel::acc`] /
//! [`Kernel::operand`] / [`Kernel::shared_sw`]) resolve the right WMMA fragment per
//! arch through `caps.frag`.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::UOp;

use super::MatmulCfg;
use crate::index::{Idx, cidx};
use crate::tiles::TileLayout;
use crate::{GL, GlSpec, Group, Kernel, MoveIdx, RT, RegTile};

/// The M-row C-block coordinate of accumulator `a` (`warp_row + a*wave_rows`,
/// in `reg`-block units) — the wave sub-tile row selection.
pub(crate) fn acc_row(warp_row: &Arc<UOp>, a: usize, cfg: &MatmulCfg) -> Arc<UOp> {
    if a == 0 { warp_row.clone() } else { warp_row.add(&cidx((a * cfg.wave_rows) as i64)) }
}

/// Store each col-major accumulator to global C at its `reg`-block coordinate
/// `{row*bps + warp_row + a*wave_rows, col*bps + warp_col}`. `lead` is the optional
/// leading store index — `None` for the whole-C tile store, `Some(k_slice)` for the
/// split-K partial-tile store into `scratch[k_slice]`. Shared by the generic core,
/// the split-K core, and the gfx942 asm microkernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn store_accs<'k>(
    g: &Group<'k>,
    c_gl: GL,
    final_accs: Vec<RT<'k>>,
    row: &Arc<UOp>,
    col: &Arc<UOp>,
    warp_row: &Arc<UOp>,
    warp_col: &Arc<UOp>,
    cfg: &MatmulCfg,
    lead: Option<Arc<UOp>>,
) {
    let bps = cfg.blocks_per_side() as i64;
    let nidx = col.mul(&cidx(bps)).add(warp_col);
    let lead = lead.unwrap_or_else(|| cidx(0));
    let mut c_t = c_gl;
    for (a, c) in final_accs.into_iter().enumerate() {
        let mrow = row.mul(&cidx(bps)).add(&acc_row(warp_row, a, cfg));
        c_t = g.store(c_t, c, MoveIdx::block((lead.clone(), 0, mrow.clone(), nidx.clone()), 2));
    }
}

/// The `(pid_m, pid_n)` C-block coordinate (in `block` units) for this workgroup
/// — the chiplet/L2 [`l2_swizzle`](crate::grid::l2_swizzle) off a flattened 1-D
/// grid (`block_idx[0]`) when enabled, else the plain 2-D `block_idx`. Generalized
/// to a non-square `m × n` C (the swizzle takes the `gm × gn` block grid; the plain
/// path reads `block_idx[1]` = pid_m, `block_idx[0]` = pid_n per [`grid_dims_mn`](MatmulCfg::grid_dims_mn)).
pub(crate) fn block_coords(ker: &Kernel, m: usize, n: usize, cfg: &MatmulCfg) -> (Arc<UOp>, Arc<UOp>) {
    if cfg.l2_swizzle {
        let (gm, gn) = ((m / cfg.block) as i64, (n / cfg.block) as i64);
        crate::grid::l2_swizzle(ker.block_idx[0].clone(), gm * gn, gm, gn)
    } else {
        (ker.block_idx[1].clone(), ker.block_idx[0].clone())
    }
}

/// The parametrized multi-wave matmul. One `cfg.block × cfg.block` C
/// tile per workgroup, `cfg.n_accum` col-major `reg × reg` accumulators/wave
/// reduced over a tracked K-loop; each wave streams its A-strip rows and shared
/// B-strip cols out of XOR-swizzled LDS. A single `END` closes the K-loop around
/// the last accumulator's store; the rest stay scoped inside it by chaining
/// their A-inputs through the prior accumulator's MFMA (a `RANGE` admits one
/// `END`). The epilogue stores each accumulator to global C at its `reg`-block.
///
/// # Panics
/// Panics on the same preconditions as [`gemm_core`].
pub fn build_matmul_cfg(ker: &Kernel, n: usize, cfg: MatmulCfg) {
    // ABI: output (c, f32) then inputs (a, b — bf16), fixed by construction. Tiles in
    // `gemm_core` are declared by ROLE via the scaffold shortcuts (`ker.acc`/`operand`/
    // `shared_sw`), which resolve the arch fragment through `caps.frag` (gfx942 CDNA
    // MFMA vs gfx11 RDNA WMMA) — so the kernel names no physical fragment constant.
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[1, 1, n, n], DType::Float32)],
        &[GlSpec::new(&[1, 1, n, n], DType::BFloat16), GlSpec::new(&[1, 1, n, n], DType::BFloat16)],
    );
    gemm_core(ker, n, n, n, cfg, cfg.k_step(), outs[0].clone(), ins[0].clone(), ins[1].clone());
}

/// The parametrized `C[m,n] = A[m,k] · B[n,k]` (`mma_abt`, B in `[N,K]` layout) GEMM
/// core for the square matmul, into the already-bound `c_gl`. One `cfg.block × cfg.block` C tile per
/// workgroup, `cfg.n_accum` col-major `reg × reg` accumulators/wave reduced over a
/// tracked `k_step`-strip K-loop out of XOR-swizzled LDS; a single `END` closes the
/// loop around the last accumulator's store, the rest scoped inside by chaining their
/// A-inputs through the prior accumulator's MMA.
///
/// # Panics
/// Panics unless: `m` and `n` are each a multiple of `cfg.block`; `k_step` is a
/// multiple of 16 (the WMMA K-edge); `k` is a multiple of `k_step`; and
/// `cfg.wave_cols == cfg.wave_rows * cfg.n_accum`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_core(
    ker: &Kernel,
    m: usize,
    k: usize,
    n: usize,
    cfg: MatmulCfg,
    k_step: usize,
    c_gl: GL,
    a_gl: GL,
    b_gl: GL,
) {
    assert_eq!(m % cfg.block, 0, "gemm M={m} must be a multiple of the {} block", cfg.block);
    assert_eq!(n % cfg.block, 0, "gemm N={n} must be a multiple of the {} block", cfg.block);
    assert_eq!(k_step % 16, 0, "k_step={k_step} must be a multiple of 16 (the WMMA K-edge)");
    assert_eq!(k % k_step, 0, "gemm K={k} must be a multiple of k_step={k_step}");
    assert_eq!(cfg.wave_cols, cfg.wave_rows * cfg.n_accum, "config invariant wave_cols == wave_rows*n_accum");
    let reg = cfg.reg();
    let g = ker.group_2d(cfg.wave_rows, cfg.wave_cols);
    let bf16 = DType::BFloat16;

    // A strip [block×k_step] = [M-block, K-strip]; B strip [block×k_step] = [N-block,
    // K-strip] (B is [N,K]); both XOR-swizzled, K contiguous, single-buffered.
    let a_smem = ker.shared_sw((cfg.block, k_step), bf16.clone(), TileLayout::Row);
    let b_smem = ker.shared_sw((cfg.block, k_step), bf16.clone(), TileLayout::Row);

    let (row, col) = block_coords(ker, m, n, &cfg); // (pid_m, pid_n) in block units
    let warp_row = g.warp_row();
    let warp_col = g.warp_col();

    // `n_accum` col-major reg×reg f32 accumulators per wave.
    let accs: Vec<RT> = (0..cfg.n_accum).map(|_| g.zero(ker.acc((reg, reg), TileLayout::Col))).collect();

    let lp = ker.loop_static((k / k_step) as i64);
    let tile = lp.index().clone();

    // Collaborative GLOBAL→LDS fill over all threads (each ends in a barrier);
    // Uses 128-bit vectorized loads for the large-N strips. B is in [N,K] layout
    // (the B[N,K] contract), indexed as [N-block, K-strip] at (col, tile).
    let (a_smem, b_smem) = if cfg.vec_load {
        (
            g.fill_local_vec(a_smem, a_gl, &[Idx::Const(0), Idx::Const(0), Idx::from(&row), Idx::from(&tile)], 2),
            g.fill_local_vec(b_smem, b_gl, &[Idx::Const(0), Idx::Const(0), Idx::from(&col), Idx::from(&tile)], 2),
        )
    } else {
        (
            g.load(a_smem, a_gl, MoveIdx::block((0, 0, row.clone(), tile.clone()), 2)),
            g.load(b_smem, b_gl, MoveIdx::block((0, 0, col.clone(), tile.clone()), 2)),
        )
    };

    // Shared B sub-tile (N row-block {warp_col}, same for every accumulator) read as a
    // [reg, k_step] Row fragment (K contiguous → `ds_read_b64`), and per-accumulator A
    // sub-tiles (M row-block {warp_row + a*wave_rows}). `mma_abt` consumes B as B[N,k].
    let bb = g.load(
        ker.operand((reg, k_step), bf16.clone(), TileLayout::Row),
        b_smem.subtile((reg, k_step), (warp_col.clone(), 0)),
        MoveIdx::default(),
    );
    let a_subs: Vec<RT> = (0..cfg.n_accum)
        .map(|a| {
            g.load(
                ker.operand((reg, k_step), bf16.clone(), TileLayout::Row),
                a_smem.subtile((reg, k_step), (acc_row(&warp_row, a, &cfg), 0)),
                MoveIdx::default(),
            )
        })
        .collect();

    // Cross-wave WAR barrier: every wave must finish reading LDS before the next
    // K iteration's collaborative fill overwrites it.
    let mut bar_deps: smallvec::SmallVec<[Arc<UOp>; 4]> = smallvec![bb.uop().clone()];
    bar_deps.extend(a_subs.iter().skip(1).map(|t| t.uop().clone()));
    let sync = a_subs[0].uop().barrier(bar_deps);
    let bb = bb.after(smallvec![sync.clone()]);
    let a_subs: Vec<RT> = a_subs.into_iter().map(|t| t.after(smallvec![sync.clone()])).collect();

    // MMA-accumulate each accumulator over the K sub-steps; chain accumulator `a`'s
    // A-input through accumulator `a-1`'s MMA so a single `END` scopes them all inside
    // the K-loop.
    let mut prev_out: Option<Arc<UOp>> = None;
    for (a, a_sub) in a_subs.iter().enumerate() {
        let a_sub = match &prev_out {
            Some(p) => a_sub.after(smallvec![p.clone()]),
            None => a_sub.clone(),
        };
        prev_out = Some(g.mma_abt(accs[a].clone(), &a_sub, &bb).uop().clone());
    }
    let ended = lp.close();
    // Each accumulator reads its fully-reduced register value *outside* the loop.
    let final_accs: Vec<RT> = accs.iter().map(|c| c.after(smallvec![ended.clone()])).collect();

    // Epilogue: store each col-major accumulator to global C at its reg-block coords.
    store_accs(&g, c_gl, final_accs, &row, &col, &warp_row, &warp_col, &cfg, None);
}

/// Split-K [`gemm_core`]: partitions the K reduction across `k_splits` workgroups
/// per C tile so a small `(n/block)²` grid still fills the machine. Workgroup
/// `(tile, k_slice)` (the latter on `block_idx[1]`, so this requires `l2_swizzle`
/// — which frees `block_idx[1]`) computes the `block×block` tile's *partial* sum
/// over K-strips `[k_slice·spk, (k_slice+1)·spk)` and writes it to
/// `scratch[k_slice]`. The caller sums `scratch` over `k_slice` (a graph reduce).
/// Single-buffered, same per-tile inner loop as [`gemm_core`].
#[allow(clippy::too_many_arguments)]
fn gemm_core_splitk(
    ker: &Kernel,
    n: usize,
    cfg: MatmulCfg,
    k_step: usize,
    k_splits: usize,
    c_gl: GL,
    a_gl: GL,
    b_gl: GL,
) {
    assert_eq!(n % cfg.block, 0, "split-K N={n} must be a multiple of block {}", cfg.block);
    assert_eq!(k_step % 16, 0, "k_step={k_step} must be a multiple of 16");
    assert!(cfg.l2_swizzle, "split-K needs l2_swizzle (block_idx[1] carries k_slice)");
    let strips = n / k_step;
    assert_eq!(strips % k_splits, 0, "split-K: strips {strips} must divide k_splits {k_splits}");
    let spk = (strips / k_splits) as i64; // K-strips per slice
    let reg = cfg.reg();
    let g = ker.group_2d(cfg.wave_rows, cfg.wave_cols);
    let bf16 = DType::BFloat16;

    let a_smem = ker.shared_sw((cfg.block, k_step), bf16.clone(), TileLayout::Row);
    let b_smem = ker.shared_sw((cfg.block, k_step), bf16.clone(), TileLayout::Row);

    let (row, col) = block_coords(ker, n, n, &cfg);
    let warp_row = g.warp_row();
    let warp_col = g.warp_col();
    let k_slice = ker.block_idx[1].clone(); // 0..k_splits
    let k_origin = k_slice.mul(&cidx(spk)); // first global K-strip of this slice

    let accs: Vec<RT> = (0..cfg.n_accum).map(|_| g.zero(ker.acc((reg, reg), TileLayout::Col))).collect();

    let lp = ker.loop_static(spk);
    let strip = k_origin.add(lp.index()); // global K-strip = k_slice*spk + t

    let (a_smem, b_smem) = if cfg.vec_load {
        (
            g.fill_local_vec(a_smem, a_gl, &[Idx::Const(0), Idx::Const(0), Idx::from(&row), Idx::from(&strip)], 2),
            g.fill_local_vec(b_smem, b_gl, &[Idx::Const(0), Idx::Const(0), Idx::from(&col), Idx::from(&strip)], 2),
        )
    } else {
        (
            g.load(a_smem, a_gl, MoveIdx::block((0, 0, row.clone(), strip.clone()), 2)),
            g.load(b_smem, b_gl, MoveIdx::block((0, 0, col.clone(), strip.clone()), 2)),
        )
    };

    let bb = g.load(
        ker.operand((reg, k_step), bf16.clone(), TileLayout::Row),
        b_smem.subtile((reg, k_step), (warp_col.clone(), 0)),
        MoveIdx::default(),
    );
    let a_subs: Vec<RT> = (0..cfg.n_accum)
        .map(|a| {
            g.load(
                ker.operand((reg, k_step), bf16.clone(), TileLayout::Row),
                a_smem.subtile((reg, k_step), (acc_row(&warp_row, a, &cfg), 0)),
                MoveIdx::default(),
            )
        })
        .collect();

    let mut bar_deps: smallvec::SmallVec<[Arc<UOp>; 4]> = smallvec![bb.uop().clone()];
    bar_deps.extend(a_subs.iter().skip(1).map(|t| t.uop().clone()));
    let sync = a_subs[0].uop().barrier(bar_deps);
    let bb = bb.after(smallvec![sync.clone()]);
    let a_subs: Vec<RT> = a_subs.into_iter().map(|t| t.after(smallvec![sync.clone()])).collect();

    let mut prev_out: Option<Arc<UOp>> = None;
    for (a, a_sub) in a_subs.iter().enumerate() {
        let a_sub = match &prev_out {
            Some(p) => a_sub.after(smallvec![p.clone()]),
            None => a_sub.clone(),
        };
        prev_out = Some(g.mma_abt(accs[a].clone(), &a_sub, &bb).uop().clone());
    }
    let ended = lp.close();
    let final_accs: Vec<RT> = accs.iter().map(|c| c.after(smallvec![ended.clone()])).collect();

    // Epilogue: store the partial tile into `scratch[k_slice]` (leading dim = k_slice).
    store_accs(&g, c_gl, final_accs, &row, &col, &warp_row, &warp_col, &cfg, Some(k_slice));
}

/// Bind the split-K ABI — output `scratch[k_splits, 1, n, n]` (f32 partials), inputs
/// `a`/`b` `[1,1,n,n]` bf16 — and run [`gemm_core_splitk`]. The caller launches with
/// grid `[(n/block)², k_splits, 1]` and reduces `scratch` over axis 0.
pub fn build_matmul_splitk(ker: &Kernel, n: usize, cfg: MatmulCfg, k_splits: usize) {
    let bf16 = DType::BFloat16;
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[k_splits, 1, n, n], DType::Float32)],
        &[GlSpec::new(&[1, 1, n, n], bf16.clone()), GlSpec::new(&[1, 1, n, n], bf16)],
    );
    gemm_core_splitk(ker, n, cfg, cfg.k_step(), k_splits, outs[0].clone(), ins[0].clone(), ins[1].clone());
}

/// Choose `k_splits` so the `n_tiles · k_splits` grid covers ~2× the CUs when the
/// plain `(n/block)²` grid is too small to fill the machine. `cu_target` is the
/// device CU count (topology-derived via [`crate::DeviceProfile::cu_count`]); a smaller
/// SKU just over-splits slightly (more reduce traffic), so an exact count isn't
/// needed. Returns 1 (no split) when the grid is already large enough or K can't be
/// divided usefully. Picks the largest divisor of `strips` not exceeding the
/// saturation target, bounding the reduce overhead.
pub(super) fn split_k_for(n: usize, cfg: MatmulCfg, cu_target: usize) -> usize {
    if !cfg.l2_swizzle {
        return 1; // split-K needs block_idx[1] free
    }
    let strips = n / cfg.k_step();
    let n_tiles = (n / cfg.block).pow(2);
    // Only split when the plain grid is *well* under the machine — otherwise the
    // scratch round-trip of the reduce outweighs the added parallelism (at n_tiles ≈
    // CU the grid already fills the GPU).
    if n_tiles >= cu_target / 2 || strips <= 1 {
        return 1;
    }
    // Target ~1·CU total workgroups: past that the scratch round-trip of the reduce
    // costs more than the extra parallelism buys (measured: k=4 beats k=8 at N=1024).
    let target = cu_target.div_ceil(n_tiles).max(1);
    // Largest divisor of `strips` that is ≤ target (and ≥ 1).
    (1..=strips).rev().find(|d| strips.is_multiple_of(*d) && *d <= target).unwrap_or(1)
}
