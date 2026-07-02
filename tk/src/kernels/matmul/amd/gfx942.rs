//! The gfx942 (CDNA3, wave64) specialization: the **production** software-pipelined
//! inline-`asm` MFMA microkernel ([`gemm_core_asm`], reached via [`build_matmul_asm_cfg`])
//! — tk's asm MFMA / `ds_read_b64` gather / register-staged prefetch infrastructure —
//! with the 256²/128² configs ([`BLOCK256_CFG`] / [`BLOCK128_CFG`]) and their by-N selector
//! ([`cfg_for_gfx942`]). [`SMALL_CFG`] is a small generic-tile config used by the
//! generic-`gemm_core` correctness tests.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::UOp;

use super::super::common::{acc_row, block_coords, store_accs};
use super::super::{K_STEP, MatmulCfg};
use crate::index::{Idx, cidx};
use crate::tiles::TileLayout;
use crate::{GL, GlSpec, Kernel, MoveIdx, RT, RegTile};

/// 8-wave (2×4) 256×256 block, two 64×64 accumulators/wave, 512
/// threads, the chiplet/L2 grid swizzle, and 128-bit vectorized LDS fills.
pub const BLOCK256_CFG: MatmulCfg =
    MatmulCfg { block: 256, wave_rows: 2, wave_cols: 4, n_accum: 2, l2_swizzle: true, vec_load: true, k_step: K_STEP };
/// The software-pipeline at a **128×128** block — the small-N choice. Same 8-wave /
/// 2-accumulator / `K_STEP=64` choreography as [`BLOCK256_CFG`], only `reg = 32` and 32 KB
/// LDS (→ 2 WG/CU). Its `(n/128)²` grid is 4× the 256² block's at a given N, so it keeps
/// the machine fed where the 256² block starves (`gemm_core_asm` at 256² is grid-starved
/// below ~4096; this is the small-N variant).
pub const BLOCK128_CFG: MatmulCfg =
    MatmulCfg { block: 128, wave_rows: 2, wave_cols: 4, n_accum: 2, l2_swizzle: true, vec_load: true, k_step: K_STEP };
/// Small-N: single-warp 64×64 block, one 64×64 accumulator, 64 threads — the
/// grid is `(n/64)²` workgroups, ~16× the large-N config's at a given N, so a small N keeps the
/// 304-CU machine fed instead of collapsing to a handful of 256×256 blocks.
/// Keeps the plain 2-D grid + scalar fill (the swizzle/vec wins are large-N).
pub const SMALL_CFG: MatmulCfg =
    MatmulCfg { block: 64, wave_rows: 1, wave_cols: 1, n_accum: 1, l2_swizzle: false, vec_load: false, k_step: K_STEP };

/// gfx942 (CDNA3) production selector for the inline-`asm` MFMA microkernel
/// ([`gemm_core_asm`]). The 256² [`BLOCK256_CFG`] for large N (≥4096, where its grid fills
/// the 304-CU machine and it reaches ~805 TF @8192); the 128² [`BLOCK128_CFG`] below
/// that (its `(n/128)²` grid keeps small/mid N fed while the per-workgroup MFMA duty
/// still beats the legacy generic path — measured a strict win at every size). `n`
/// must be a multiple of the returned block (256 or 128); the dispatch enforces it.
pub(crate) fn cfg_for_gfx942(n: usize) -> MatmulCfg {
    if n >= 4096 && n.is_multiple_of(BLOCK256_CFG.block) { BLOCK256_CFG } else { BLOCK128_CFG }
}

/// Bind the square `n×n` bf16→f32 ABI and run [`gemm_core_asm`] at the given `cfg` (tile
/// geometry) and wave-phase ping-pong `offset`. The production dispatch reaches
/// [`gemm_core_asm`] through here (256²/128² by N via [`cfg_for_gfx942`]); the tests call
/// it directly with a fixed `cfg` (256²/128²) and `offset` — the latter being the
/// ping-pong bisect toggle (`false` runs the pipeline with no phase shift).
///
/// # Panics
/// Panics unless `n` is a multiple of both `cfg.block` and [`K_STEP`] (64).
pub(crate) fn build_matmul_asm_cfg(ker: &Kernel, n: usize, cfg: MatmulCfg, offset: bool) {
    let bf16 = DType::BFloat16;
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[1, 1, n, n], DType::Float32)],
        &[GlSpec::new(&[1, 1, n, n], bf16.clone()), GlSpec::new(&[1, 1, n, n], bf16)],
    );
    gemm_core_asm(ker, n, outs[0].clone(), ins[0].clone(), ins[1].clone(), cfg, offset);
}

/// **The software-pipelined inline-`asm` MFMA microkernel** (`C = A·Bᵀ`, B in `[N,K]` —
/// the B[N,K] contract, both operands K-contiguous) — a flat, hand-unrolled,
/// cluster-for-cluster GEMM inner loop. Adapted from the HipKittens gfx942 software
/// pipeline. It is the readable demonstration of — and the real user/test for — tk's
/// inline-`asm` GEMM primitives: the asm MFMA ([`crate::group::Group::mma_abt_asm`]), the
/// asm `ds_read_b64` gather ([`crate::group::Group::gather_local_asm`]), and the asm
/// register-staged global prefetch ([`crate::Group::stage_global_to_reg_vec_asm`] /
/// [`crate::Group::commit_reg_to_local_vec_asm`]). Each is an explicit per-call gfx942
/// primitive — no kernel-global asm mode.
///
/// Pipeline (a three-phase skeleton): a SINGLE swizzled LDS tile per operand at
/// [`K_STEP`]=64 (no 2-deep ring), with the next strip's GLOBAL load **register-staged**
/// (asm `global_load_dwordx4`) and committed (asm `ds_write_b64` carrying a deferred
/// `s_waitcnt vmcnt(0) lgkmcnt(0)`) back into that SAME tile. The key move — a **single-LDS
/// pre-gather**: all four 16-K substeps' fragments are gathered into registers across
/// clusters 0/2/4 (asm `ds_read_b64`) *before* the cluster-6 commit overwrites the tile —
/// so the commit has no WAR stall even with one buffer. **Prologue** (1 fill barrier) →
/// **steady `loop_static(num_tiles-1)`** (8 clusters/strip, each closing with one workgroup
/// `s_barrier`; BLOCK256_CFG → 2 accumulators × 16 MFMA = 32-MFMA clusters at 1/3/5/7, 128
/// MFMAs/strip; `s_setprio(1/0)` brackets the MFMA clusters) → **drained epilogue** (the
/// final strip, gathers + MFMAs, no prefetch/commit, 7 barriers). The two warp-rows are
/// run one cluster-barrier out of phase by the **wave-phase ping-pong**
/// ([`crate::arch::gfx9::wave_phase_prologue`] / [`crate::arch::gfx9::wave_phase_epilogue`],
/// gated by `offset`): a predicated `s_barrier` on warp-row 1 in the prologue and a matching
/// one on warp-row 0 in the epilogue, so per-warp-row the total `s_barrier` count is
/// identical (16 unconditional sites + 1 conditional each = 8·num_tiles+1) — balanced, no
/// deadlock — while one row's MFMAs overlap the other row's memory/commit. Parametrized by
/// `cfg` (256²/128²) via [`cfg_for_gfx942`].
///
/// # The gfx942 production path
/// This IS the gfx942 matmul: [`crate::matmul`] routes CDNA3 through here (256² for large
/// N ≈805 TF @8192, 128² below via [`cfg_for_gfx942`]); the generic
/// [`gemm_core`](super::super::common::gemm_core) is now only the gfx1151 / non-CDNA path.
/// The collaborative all-512-thread commit is compatible with the ping-pong because AMD
/// `s_barrier` is a *counting* barrier: the commit is split across two adjacent barrier
/// windows and both halves land before any read, given the exact balanced barrier topology
/// this skeleton maintains (a rolled core with two prologue fill barriers and a wrap-around
/// loop would not).
///
/// `offset` gates the ping-pong for the bisect (see [`build_matmul_asm_cfg`]); production
/// wires it on.
///
/// # Panics
/// Panics unless `n` is a multiple of both `cfg.block` and [`K_STEP`] (64).
fn gemm_core_asm(ker: &Kernel, n: usize, c_gl: GL, a_gl: GL, b_gl: GL, cfg: MatmulCfg, offset: bool) {
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

    // ── Prologue: collaboratively fill strip 0 into the single LDS tile, closed by ONE
    //    workgroup barrier. Fencing the two fills exactly once (rather than once per fill)
    //    lands every later commit→read pairing one barrier event later, which is what keeps
    //    the single-buffer pipeline's phasing correct.
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
    // Wave-phase ping-pong (prologue half): warp-row 1 takes an extra prologue `s_barrier`
    // so it runs one cluster-barrier behind warp-row 0 for the whole steady loop — one row's
    // MFMA clusters then overlap the other row's memory/commit clusters (the overlap a single
    // LDS buffer otherwise can't get). AMD `s_barrier` is a COUNTING barrier (a wave's k-th
    // s_barrier pairs with every wave's k-th by execution count, not program location), so the
    // collaborative all-512-thread fill/commit is split across two adjacent barrier windows —
    // both halves land before any read, given balanced counts + this exact topology.
    // `offset == false` runs the pipeline with no phase shift (the bisect control: balanced
    // either way, so it isolates the skeleton restructure from the offset).
    let wp_pro = crate::arch::gfx9::wave_phase_prologue(warp_row.clone(), offset, a_smem.uop().clone());
    let a_smem = a_smem.rewrap(a_smem.uop().after(smallvec![wp_pro.clone()]));
    let b_smem = b_smem.rewrap(b_smem.uop().after(smallvec![wp_pro]));

    // ── Steady loop (`tile < num_tiles - 1`): tiles 0..num_tiles-1 — the final
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
    // FIRST, then the workgroup fence (see [`drained_barrier`]). The gathers (`ds_read`) and
    // the commit (`ds_write`) are `asm sideeffect` — OPAQUE to LLVM — so the `Barrier`'s own
    // `fence release` never lowers to a wait for them (`SIInsertWaitcnts` can't see asm LDS
    // ops; only a *consuming* register use is auto-waited, a cluster too late). Under the
    // wave-phase offset the two warp-rows are one cluster apart, so a read/commit one phase
    // later would race the still-in-flight asm reads/writes (the RAW + WAR edges) without this
    // drain; the offset then hides the drain latency under the OTHER warp-row's MFMAs. The asm
    // side-effects still hold program order, so no `sched_barrier` is needed. `pass` carries
    // the cluster's last LDS op (its chain).
    let cbar = |pass: &Arc<UOp>, deps: smallvec::SmallVec<[Arc<UOp>; 4]>| -> Arc<UOp> {
        crate::arch::gfx9::drained_barrier(pass.clone(), deps)
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
    // This cluster boundary carries NO drain (the MMA read no LDS → counter already 0); bare barrier.
    let bar1 = crate::arch::gfx9::s_barrier_bare(smallvec![m1.clone()]);

    // ── Cluster 2: pre-gather substep 1 (B1,A1a,A1b) + part of substep 2 (B2,A2a). ──
    let b1 = gb(&bar1, 1);
    let a1a = ga(&b1.uop().clone(), 0, 1);
    let a1b = ga(&a1a.uop().clone(), 1, 1);
    let b2 = gb(&a1b.uop().clone(), 2);
    let a2a = ga(&b2.uop().clone(), 0, 2);
    // A2b gathered HERE (not at cluster 4) so all of cluster 5's operands are resident 2 clusters ahead.
    let a2b = ga(&a2a.uop().clone(), 1, 2);
    let bar2 = cbar(
        &a2b.uop().clone(),
        smallvec![b1.uop().clone(), a1a.uop().clone(), a1b.uop().clone(), b2.uop().clone(), a2a.uop().clone()],
    );

    // ── Cluster 3: MMA substep 1. ──
    mma_cluster!(m3, a1a, a1b, &b1, bar2);
    // Bare barrier (cluster-3 boundary): counter already 0.
    let bar3 = crate::arch::gfx9::s_barrier_bare(smallvec![m3.clone()]);

    // ── Cluster 4: stage B(k+1)→VGPR; pre-gather rest (A2b, B3, A3a, A3b). ──
    // Anchor B's prefetch load to `bar3` (cluster-3's barrier) so the toposort
    // emits it HERE at cluster 4 — interleaved between the MFMA clusters — instead
    // of floating it to the loop top bunched with A (latency then hides behind the
    // cluster-5/7 MFMAs — the interleaved mid-loop global-load placement).
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
    // Bare barrier (cluster-4 boundary): cluster-5's operands all resident from cluster 2.
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

    // ── Drained epilogue: the final strip (num_tiles-1) is already in LDS — gather every
    //    substep and run the 4 MFMA clusters, with NO prefetch and NO commit. Seven cluster
    //    barriers (clusters 0,1,2,3,4,5,7 — there is no cluster-6 commit), the count that
    //    keeps the per-warp-row barrier total balanced.
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

    // Wave-phase rebalance (epilogue half): warp-row 0 takes the matching extra barrier so
    // both warp-rows execute the IDENTICAL total count (no deadlock) and re-sync before the
    // store. `offset == false` → both halves never-match, as in the prologue.
    let wp_epi = crate::arch::gfx9::wave_phase_epilogue(warp_row.clone(), offset, accs[0].uop().clone());
    let final_accs: Vec<RT> = accs.into_iter().map(|c| c.after(smallvec![wp_epi.clone()])).collect();

    // Store each col-major accumulator to global C.
    store_accs(&g, c_gl, final_accs, &row, &col, &warp_row, &warp_col, &cfg, None);
}
