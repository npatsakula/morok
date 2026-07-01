//! The gfx942 (CDNA3, wave64) specialization: the size-adaptive configs
//! ([`M1_CFG`] / [`SMALL_CFG`] / [`MID_CFG`]) and their selector ([`cfg_for_n`]),
//! plus the inline-`asm` HipKittens-pipeline **reference** microkernel
//! ([`gemm_core_hk`] / [`build_matmul_hk`] / [`build_matmul_hk_phase`]) — the
//! readable demonstration of tk's asm MFMA / `ds_read_b64` gather / register-staged
//! prefetch infrastructure.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::UOp;

use super::super::common::{acc_row, block_coords};
use super::super::{K_STEP, MatmulCfg};
use crate::index::{Idx, cidx};
use crate::tiles::TileLayout;
use crate::{GL, GlSpec, Kernel, MoveIdx, RT, RegTile};

/// 8-wave (2×4) 256×256 block, two 64×64 accumulators/wave, 512
/// threads, the chiplet/L2 grid swizzle, and 128-bit vectorized LDS fills.
pub const M1_CFG: MatmulCfg =
    MatmulCfg { block: 256, wave_rows: 2, wave_cols: 4, n_accum: 2, l2_swizzle: true, vec_load: true, k_step: K_STEP };
/// The HK pipeline at a **128×128** block (HK's small-N choice — `kernel_1024/2048.cpp`
/// set `BLOCK_SIZE=128`, byte-identical body). Same 8-wave / 2-accumulator / `K_STEP=64`
/// choreography as [`M1_CFG`], only `reg = 32` and 32 KB LDS (→ 2 WG/CU). Its `(n/128)²`
/// grid is 4× M1's at a given N, so it keeps the machine fed where the 256² block starves
/// (`gemm_core_hk` at 256² is grid-starved below ~4096; this is the small-N variant).
pub const HK128_CFG: MatmulCfg =
    MatmulCfg { block: 128, wave_rows: 2, wave_cols: 4, n_accum: 2, l2_swizzle: true, vec_load: true, k_step: K_STEP };
/// Small-N: single-warp 64×64 block, one 64×64 accumulator, 64 threads — the
/// grid is `(n/64)²` workgroups, ~16× the large-N config's at a given N, so a small N keeps the
/// 304-CU machine fed instead of collapsing to a handful of 256×256 blocks.
/// Keeps the plain 2-D grid + scalar fill (the swizzle/vec wins are large-N).
pub const SMALL_CFG: MatmulCfg =
    MatmulCfg { block: 64, wave_rows: 1, wave_cols: 1, n_accum: 1, l2_swizzle: false, vec_load: false, k_step: K_STEP };

/// Mid-N (gfx942): 128×128 block, 2×2 waves (4 waves / 256 threads), one 64×64
/// accumulator/wave, L2 swizzle + 128-bit vec fills. 32 KB LDS (vs M1's 64 KB) so
/// two workgroups fit per CU, and the grid is `(n/128)²` — 4× M1's tile count at a
/// given N. Fills the saturation gap between SMALL (64×64) and M1 (256×256).
pub const MID_CFG: MatmulCfg =
    MatmulCfg { block: 128, wave_rows: 2, wave_cols: 2, n_accum: 1, l2_swizzle: true, vec_load: true, k_step: K_STEP };

/// Size-adaptive config for gfx942 (CDNA wave64). [`MID_CFG`] (128×128) is the
/// tuned block for all but small N: its 32 KB LDS admits two workgroups per CU and
/// its `(n/128)²` grid keeps the machine fed, beating the 256×256 [`M1_CFG`] by
/// 6–8× across 1024–8192 (M1's 64 KB LDS pins one single-buffered workgroup per CU,
/// so it starves the GPU — it is no longer auto-selected). [`SMALL_CFG`] (64×64)
/// keeps the grid fed for small N, where 128-blocks leave too few tiles. The block
/// must divide N, so the chain also degrades gracefully (128 → 64) for sizes that
/// aren't a multiple of 128.
pub fn cfg_for_n(n: usize) -> MatmulCfg {
    if n <= 768 && n.is_multiple_of(SMALL_CFG.block) {
        SMALL_CFG
    } else if n.is_multiple_of(MID_CFG.block) {
        MID_CFG
    } else if n.is_multiple_of(SMALL_CFG.block) {
        SMALL_CFG
    } else {
        M1_CFG
    }
}

/// Bind the square `n×n` bf16→f32 ABI and run the inline-`asm` HK-pipeline **reference**
/// kernel ([`gemm_core_hk`]) — M1_CFG geometry (256²/8-warp/K_STEP=64/2-accumulator). This
/// is the readable HK-mirror that exercises tk's asm GEMM infrastructure; it is **not** a
/// production path (see [`gemm_core_hk`] for why it is not wired into
/// [`cfg_for_arch`](super::super::cfg_for_arch)).
///
/// # Panics
/// Panics unless `n` is a multiple of both 256 and [`K_STEP`] (64).
pub fn build_matmul_hk(ker: &Kernel, n: usize) {
    build_matmul_hk_phase(ker, n, true);
}

/// [`build_matmul_hk`] with HK's wave-phase ping-pong (`offset`) togglable — the
/// **bisect hook** for validating the offset independently of the pipelined skeleton.
/// `offset == false` sets the conditional `wave_phase_barrier`s never-matching (no phase
/// shift, counts still balanced), so any numeric difference vs `offset == true` isolates
/// to the offset, not the restructure. [`build_matmul_hk`] wires `offset == true` (HK's
/// real choreography).
pub fn build_matmul_hk_phase(ker: &Kernel, n: usize, offset: bool) {
    build_matmul_hk_cfg(ker, n, M1_CFG, offset);
}

/// The HK pipeline at the **128²** block ([`HK128_CFG`]) — the small-N variant (`kernel_*.cpp`
/// switch to `BLOCK_SIZE=128` for N ≤ 2048). Same asm choreography as [`build_matmul_hk`],
/// only the tile is 128² so the `(n/128)²` grid keeps small N fed.
pub fn build_matmul_hk128(ker: &Kernel, n: usize) {
    build_matmul_hk_cfg(ker, n, HK128_CFG, true);
}

/// Bind the square `n×n` bf16→f32 ABI and run [`gemm_core_hk`] at the given `cfg` (tile
/// geometry) and ping-pong `offset`. The tile-parametrized entry both [`build_matmul_hk_phase`]
/// (256²) and [`build_matmul_hk128`] (128²) go through.
fn build_matmul_hk_cfg(ker: &Kernel, n: usize, cfg: MatmulCfg, offset: bool) {
    let bf16 = DType::BFloat16;
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[1, 1, n, n], DType::Float32)],
        &[GlSpec::new(&[1, 1, n, n], bf16.clone()), GlSpec::new(&[1, 1, n, n], bf16)],
    );
    gemm_core_hk(ker, n, outs[0].clone(), ins[0].clone(), ins[1].clone(), cfg, offset);
}

/// **The HipKittens-pipeline reference kernel** (`C = A·Bᵀ`, B in `[N,K]`) — a flat,
/// hand-unrolled, cluster-for-cluster port of HK's `cdna3/.../256_256_64_16.cpp` inner
/// loop. It is the readable demonstration of — and the real user/test for — tk's
/// inline-`asm` GEMM primitives: the asm MFMA ([`crate::group::Group::mma_abt_asm`]), the
/// asm `ds_read_b64` gather ([`crate::group::Group::gather_local_asm`]), and the asm
/// register-staged global prefetch ([`crate::Group::stage_global_to_reg_vec_asm`] /
/// [`crate::Group::commit_reg_to_local_vec_asm`]). Each is an explicit per-call gfx942
/// primitive — no kernel-global asm mode.
///
/// Pipeline (HK's exact three-phase skeleton): a SINGLE swizzled LDS tile per operand at
/// [`K_STEP`]=64 (no 2-deep ring), with the next strip's GLOBAL load **register-staged**
/// (asm `global_load_dwordx4`) and committed (asm `ds_write_b64` carrying a deferred
/// `s_waitcnt vmcnt(0) lgkmcnt(0)`) back into that SAME tile. The key move: all four 16-K
/// substeps' fragments are **pre-gathered** into registers across clusters 0/2/4 (asm
/// `ds_read_b64`) *before* the cluster-6 commit overwrites the tile — so the commit has no
/// WAR stall even with one buffer. **Prologue** (1 fill barrier) → **steady
/// `loop_static(num_tiles-1)`** (8 clusters/strip, each closing with one workgroup
/// `s_barrier`; M1_CFG → 2 accumulators × 16 MFMA = 32-MFMA clusters at 1/3/5/7, 128
/// MFMAs/strip; `s_setprio(1/0)` brackets the MFMA clusters) → **drained epilogue** (the
/// final strip, gathers + MFMAs, no prefetch/commit, 7 barriers). The two warp-rows are
/// run one cluster-barrier out of phase by HK's **wave-phase ping-pong**
/// ([`crate::arch::gfx9::wave_phase_barrier`], `offset`): an extra `if(warp_row==1) s_barrier` in
/// the prologue and a matching `if(warp_row==0) s_barrier` in the epilogue, so per-warp-row
/// the total `s_barrier` count is identical (16 unconditional sites + 1 conditional each =
/// 8·num_tiles+1) — balanced, no deadlock — while one row's MFMAs overlap the other row's
/// memory/commit. Hardcoded M1_CFG geometry — flat, not parametrized.
///
/// # Why it is not wired into [`cfg_for_n`] (a reference, not a production path)
/// It is the readable existence-test for the asm GEMM infrastructure + the wave-phase
/// ping-pong; `cfg_for_n` stays on the size-adaptive [`gemm_core`](super::super::common::gemm_core) /
/// [`cfg_for_arch`](super::super::cfg_for_arch)
/// (the 256² block under-occupies small N). The earlier "collaborative-LDS-vs-ping-pong
/// incompatibility" wall was a misdiagnosis: AMD `s_barrier` is a *counting* barrier, so
/// the collaborative all-512-thread commit is split across two adjacent barrier windows and
/// both halves land before any read — given HK's exact balanced topology (which the rolled
/// core, with TWO prologue fill barriers and a wrap-around loop, did not have).
///
/// `offset` gates the ping-pong for the bisect ([`build_matmul_hk_phase`]); the reference
/// [`build_matmul_hk`] wires it on.
///
/// # Panics
/// Panics unless `n` is a multiple of both 256 and [`K_STEP`] (64).
fn gemm_core_hk(ker: &Kernel, n: usize, c_gl: GL, a_gl: GL, b_gl: GL, cfg: MatmulCfg, offset: bool) {
    assert_eq!(n % cfg.block, 0, "HK matmul N={n} must be a multiple of the block {}", cfg.block);
    assert_eq!(n % K_STEP, 0, "HK matmul N={n} must be a multiple of K_STEP={K_STEP}");
    let reg = cfg.reg(); // 64 at 256², 32 at 128²
    let g = ker.group_2d(cfg.wave_rows, cfg.wave_cols); // 2×4 = 8 warps
    let bf16 = DType::BFloat16;

    let (row, col) = block_coords(ker, n, n, &cfg);
    let warp_row = g.warp_row();
    let warp_col = g.warp_col();
    let mut accs: Vec<RT> = (0..cfg.n_accum).map(|_| g.zero(ker.acc((reg, reg), TileLayout::Col))).collect();

    let a_smem = ker.shared_sw((cfg.block, K_STEP), bf16.clone(), TileLayout::Row);
    let b_smem = ker.shared_sw((cfg.block, K_STEP), bf16.clone(), TileLayout::Row);
    let num_tiles = n / K_STEP;

    // ── Prologue (HK :73-79): collaboratively fill strip 0 into the single LDS tile,
    //    closed by ONE workgroup barrier. Collapsing the previous TWO per-`fill_local_vec`
    //    barriers into a single fence is the off-by-one that broke M7 — HK fences the two
    //    fills exactly once (:75), so every later commit→read pairing lands one barrier
    //    event later than it did under the rolled core.
    let a_smem = g.fill_local_vec_nobar(
        a_smem,
        a_gl.clone(),
        &[Idx::Const(0), Idx::Const(0), Idx::from(&row), Idx::Const(0)],
        2,
    );
    let b_smem = g.fill_local_vec_nobar(
        b_smem,
        b_gl.clone(),
        &[Idx::Const(0), Idx::Const(0), Idx::from(&col), Idx::Const(0)],
        2,
    );
    // One workgroup fence over both fills, consumed in the smem value chain (so the
    // `Barrier` is scheduled — an ordering-only dep on a Custom would orphan it).
    let prol_bar = a_smem.uop().barrier(smallvec![b_smem.uop().clone()]);
    let a_smem = a_smem.rewrap(a_smem.uop().after(smallvec![prol_bar.clone()]));
    let b_smem = b_smem.rewrap(b_smem.uop().after(smallvec![prol_bar]));
    // HK's wave-phase ping-pong (:77-79): warp-row 1 takes an extra prologue `s_barrier`
    // (eq=1) so it runs one cluster-barrier behind warp-row 0 for the whole steady loop —
    // one row's MFMA clusters then overlap the other row's memory/commit clusters (the
    // overlap a single LDS buffer otherwise can't get). AMD `s_barrier` is a COUNTING
    // barrier (a wave's k-th s_barrier pairs with every wave's k-th by execution count, not
    // program location), so the collaborative all-512-thread fill/commit is split across
    // two adjacent barrier windows — both halves land before any read, given balanced
    // counts + HK's exact topology. `offset == false` sets eq never-matching (warp_row ∈
    // {0,1}, eq=2) → no conditional fires → no phase shift (the bisect control: balanced
    // either way, so it isolates the skeleton restructure from the offset).
    let wp_pro =
        crate::arch::gfx9::wave_phase_barrier(warp_row.clone(), if offset { 1 } else { 2 }, a_smem.uop().clone());
    let a_smem = a_smem.rewrap(a_smem.uop().after(smallvec![wp_pro.clone()]));
    let b_smem = b_smem.rewrap(b_smem.uop().after(smallvec![wp_pro]));

    // ── Steady loop (HK :82 `tile < num_tiles - 1`): tiles 0..num_tiles-1 — the final
    //    strip is drained flat in the epilogue. Each strip's MFMAs overlap the NEXT strip's
    //    register-staged prefetch + late single-buffer commit. `pf = tile + 1` is always in
    //    range (tile ∈ [0, num_tiles-2]), so the commit only writes a strip the next
    //    iteration / epilogue consumes — no wrap-around.
    let lp = ker.loop_static((num_tiles - 1) as i64);
    let tile = lp.index().clone();
    let pf = tile.add(&cidx(1));

    // Helpers: asm `ds_read_b64` gather of one 16-K subtile from the single LDS tile,
    // threaded after `dep` (cluster ordering + loop-scoping so the fixed-address read is
    // not hoisted). A = As[warp_row + acc*2, k]; B = Bs[warp_col, k].
    let ga = |dep: &Arc<UOp>, acc: usize, k: i64| -> RT {
        g.gather_local_asm(
            ker.operand((reg, 16), bf16.clone(), TileLayout::Row),
            a_smem
                .rewrap(a_smem.uop().after(smallvec![dep.clone()]))
                .subtile((reg, 16), (acc_row(&warp_row, acc, &cfg), k)),
            MoveIdx::default(),
        )
    };
    let gb = |dep: &Arc<UOp>, k: i64| -> RT {
        g.gather_local_asm(
            ker.operand((reg, 16), bf16.clone(), TileLayout::Row),
            b_smem.rewrap(b_smem.uop().after(smallvec![dep.clone()])).subtile((reg, 16), (warp_col.clone(), k)),
            MoveIdx::default(),
        )
    };
    // Workgroup barrier closing a cluster: drain inline-asm LDS traffic (`lgkmcnt(0)`)
    // FIRST, then the workgroup fence. The gathers (`ds_read`) and the commit (`ds_write`)
    // are `asm sideeffect` — OPAQUE to LLVM — so the `Barrier`'s own `fence release` never
    // lowers to a wait for them (`SIInsertWaitcnts` can't see asm LDS ops; only a *consuming*
    // register use is auto-waited, a cluster too late). Under the wave-phase offset the two
    // warp-rows are one cluster apart, so a read/commit one phase later would race the
    // still-in-flight asm reads/writes (the RAW + WAR edges) without this drain — the
    // misdiagnosed "collaboration incompatibility". HK gets the identical pre-barrier drain
    // implicitly (its `ds` ops are compiler-tracked); the offset hides the stall under the
    // OTHER warp-row's MFMAs. The asm side-effects still hold program order (M2–M5), so no
    // `sched_barrier` is needed. `pass` carries the cluster's last LDS op (its chain).
    let cbar = |pass: &Arc<UOp>, deps: smallvec::SmallVec<[Arc<UOp>; 4]>| -> Arc<UOp> {
        crate::arch::gfx9::s_waitcnt_lgkmcnt(0, pass.clone()).barrier(deps)
    };
    // One 32-MFMA cluster (2 accumulators) on `(a0,a1)·b`, opened after the workgroup
    // barrier `$bar` (the prior cluster's fence — threaded onto the B operand via `.after`,
    // NOT passed to a Custom, since a `Barrier` registers no value), bracketed by
    // `s_setprio(1/0)`; binds the post-`s_setprio(0)` tail to `$tail`. A macro (not a
    // closure) so the `RT<'k>` borrows unify. `mma_abt` chains accumulator 1's A-input
    // through accumulator 0's MFMA so one loop `END` scopes both.
    macro_rules! mma_cluster {
        ($tail:ident, $a0:expr, $a1:expr, $b:expr, $bar:expr) => {
            let bb = $b.after(&$bar); // order the cluster after the prior barrier (value dep)
            let p1 = crate::arch::gfx9::s_setprio(1, bb.uop().clone());
            accs[0] = g.mma_abt_asm(accs[0].clone(), &$a0.after(&p1), &bb);
            accs[1] = g.mma_abt_asm(accs[1].clone(), &$a1.after(&accs[0].uop().clone()), &bb);
            let $tail = crate::arch::gfx9::s_setprio(0, accs[1].uop().clone());
        };
    }

    // ── Cluster 0: stage A(k+1)→VGPR; pre-gather substep 0 (B0, A0a, A0b). ──
    let s_a = g.stage_global_to_reg_vec_asm(
        &a_smem,
        &a_gl,
        &[Idx::Const(0), Idx::Const(0), Idx::from(&row), Idx::from(&pf)],
        2,
        None,
    );
    let b0 = gb(&tile, 0);
    let a0a = ga(&b0.uop().clone(), 0, 0);
    let a0b = ga(&a0a.uop().clone(), 1, 0);
    let bar0 = cbar(&a0b.uop().clone(), smallvec![b0.uop().clone(), a0a.uop().clone(), s_a.clone()]);

    // ── Cluster 1: MMA substep 0. ──
    mma_cluster!(m1, a0a, a0b, &b0, bar0);
    // HK's C2 boundary carries NO drain (MMA read no LDS → counter already 0); bare barrier.
    let bar1 = crate::arch::gfx9::s_barrier_bare(smallvec![m1.clone()]);

    // ── Cluster 2: pre-gather substep 1 (B1,A1a,A1b) + part of substep 2 (B2,A2a). ──
    let b1 = gb(&bar1, 1);
    let a1a = ga(&b1.uop().clone(), 0, 1);
    let a1b = ga(&a1a.uop().clone(), 1, 1);
    let b2 = gb(&a1b.uop().clone(), 2);
    let a2a = ga(&b2.uop().clone(), 0, 2);
    // A2b gathered HERE (was C4) so all of C5's operands are resident 2 clusters ahead (HK depth).
    let a2b = ga(&a2a.uop().clone(), 1, 2);
    let bar2 = cbar(
        &a2b.uop().clone(),
        smallvec![b1.uop().clone(), a1a.uop().clone(), a1b.uop().clone(), b2.uop().clone(), a2a.uop().clone()],
    );

    // ── Cluster 3: MMA substep 1. ──
    mma_cluster!(m3, a1a, a1b, &b1, bar2);
    // Bare barrier (HK's C4 boundary): counter already 0.
    let bar3 = crate::arch::gfx9::s_barrier_bare(smallvec![m3.clone()]);

    // ── Cluster 4: stage B(k+1)→VGPR; pre-gather rest (A2b, B3, A3a, A3b). ──
    // Anchor B's prefetch load to `bar3` (cluster-3's barrier) so the toposort
    // emits it HERE at cluster 4 — interleaved between the MFMA clusters — instead
    // of floating it to the loop top bunched with A (latency then hides behind the
    // C5/C7 MFMAs, HK's mid-loop `BLOAD` placement).
    let s_b = g.stage_global_to_reg_vec_asm(
        &b_smem,
        &b_gl,
        &[Idx::Const(0), Idx::Const(0), Idx::from(&col), Idx::from(&pf)],
        2,
        Some(&bar3),
    );
    // C4 gathers only substep 3 (A2b moved to C2); these are consumed at C7, drained at bar6.
    let b3 = gb(&bar3, 3);
    let a3a = ga(&b3.uop().clone(), 0, 3);
    let a3b = ga(&a3a.uop().clone(), 1, 3);
    // Bare barrier (HK's C5 boundary): C5's operands all resident from C2.
    let bar4 = crate::arch::gfx9::s_barrier_bare(smallvec![
        a3b.uop().clone(),
        b3.uop().clone(),
        a3a.uop().clone(),
        s_b.clone()
    ]);

    // ── Cluster 5: MMA substep 2 (all current-strip reads now in registers). ──
    mma_cluster!(m5, a2a, a2b, &b2, bar4);
    let bar5 = cbar(&m5, smallvec![]);

    // ── Cluster 6: commit strip k+1 into the SAME tile, ordered after the cluster-5
    //    barrier (all waves past their reads). No WAR stall — every current-strip read is
    //    already in registers; the commit's first `ds_write` bakes `s_waitcnt vmcnt(0)
    //    lgkmcnt(0)` (global load arrived + gathers drained) so it can overwrite safely. ──
    let a_after = a_smem.rewrap(a_smem.uop().after(smallvec![bar5.clone()]));
    let b_after = b_smem.rewrap(b_smem.uop().after(smallvec![bar5]));
    let commit_a = g.commit_reg_to_local_vec_asm(a_after, &s_a);
    // Chain B's commit after A's so `cbar`'s single `lgkmcnt(0)` drain (on `commit_b`, the
    // later one) covers BOTH tiles' asm `ds_write`s before `bar6` — the RAW edge the next
    // strip's cross-warp-row gather depends on.
    let b_after = b_after.rewrap(b_after.uop().after(smallvec![commit_a.uop().clone()]));
    let commit_b = g.commit_reg_to_local_vec_asm(b_after, &s_b);
    let bar6 = cbar(&commit_b.uop().clone(), smallvec![commit_a.uop().clone()]);

    // ── Cluster 7: MMA substep 3. ──
    mma_cluster!(m7, a3a, a3b, &b3, bar6);

    // Loop tail = cluster 7's workgroup barrier (the 8th per-iteration fence; RAW — the
    // strip-(tile+1) commit is visible to the next trip's C0 gather / the epilogue's gather).
    let ended = lp.close_barrier(smallvec![m7, commit_a.uop().clone(), commit_b.uop().clone()]);
    // Read the loop-carried accumulators' post-loop value, threaded IN PLACE on the same
    // `accs` binding — the `mma_cluster` macro's `accs[i] = …` writes resolve (by macro
    // hygiene) to this outer binding, so the epilogue MFMAs must continue ON it, not a shadow.
    accs = accs.iter().map(|c| c.after(smallvec![ended.clone()])).collect();

    // ── Drained epilogue (HK :159-219): the final strip (num_tiles-1) is already in LDS —
    //    gather every substep and run the 4 MFMA clusters, with NO prefetch and NO commit.
    //    Seven cluster barriers (C0,C1,C2,C3,C4,C5,C7 — there is no C6 commit), matching
    //    HK's epilogue count so the per-warp-row barrier total stays balanced.
    let b0 = gb(&ended, 0);
    let a0a = ga(&b0.uop().clone(), 0, 0);
    let a0b = ga(&a0a.uop().clone(), 1, 0);
    let eb0 = cbar(&a0b.uop().clone(), smallvec![b0.uop().clone(), a0a.uop().clone()]);

    mma_cluster!(em1, a0a, a0b, &b0, eb0);
    let eb1 = cbar(&em1, smallvec![]);

    let b1 = gb(&eb1, 1);
    let a1a = ga(&b1.uop().clone(), 0, 1);
    let a1b = ga(&a1a.uop().clone(), 1, 1);
    let eb2 = cbar(&a1b.uop().clone(), smallvec![b1.uop().clone(), a1a.uop().clone()]);

    mma_cluster!(em3, a1a, a1b, &b1, eb2);
    let eb3 = cbar(&em3, smallvec![]);

    let b2 = gb(&eb3, 2);
    let a2a = ga(&b2.uop().clone(), 0, 2);
    let a2b = ga(&a2a.uop().clone(), 1, 2);
    let b3 = gb(&a2b.uop().clone(), 3);
    let a3a = ga(&b3.uop().clone(), 0, 3);
    let a3b = ga(&a3a.uop().clone(), 1, 3);
    let eb4 = cbar(
        &a3b.uop().clone(),
        smallvec![b2.uop().clone(), a2a.uop().clone(), a2b.uop().clone(), b3.uop().clone(), a3a.uop().clone()],
    );

    mma_cluster!(em5, a2a, a2b, &b2, eb4);
    let eb5 = cbar(&em5, smallvec![]);

    mma_cluster!(em7, a3a, a3b, &b3, eb5);
    let eb7 = cbar(&em7, smallvec![]);
    // Consume the 7th epilogue barrier in the accumulator value chain (so it is scheduled).
    accs = accs.iter().map(|c| c.after(smallvec![eb7.clone()])).collect();

    // HK's wave-phase rebalance (:221-223): warp-row 0 takes the matching extra barrier
    // (eq=0) so both warp-rows execute the IDENTICAL total count (no deadlock) and re-sync
    // before the store. `offset == false` → eq never-matching (2), as in the prologue.
    let wp_epi =
        crate::arch::gfx9::wave_phase_barrier(warp_row.clone(), if offset { 0 } else { 2 }, accs[0].uop().clone());
    let final_accs: Vec<RT> = accs.into_iter().map(|c| c.after(smallvec![wp_epi.clone()])).collect();

    // Store each col-major accumulator to global C.
    let bps = cfg.blocks_per_side() as i64;
    let nidx = col.mul(&cidx(bps)).add(&warp_col);
    let mut c_t = c_gl;
    for (a, c) in final_accs.into_iter().enumerate() {
        let mrow = row.mul(&cidx(bps)).add(&acc_row(&warp_row, a, &cfg));
        c_t = g.store(c_t, c, MoveIdx::block((0, 0, mrow.clone(), nidx.clone()), 2));
    }
}
