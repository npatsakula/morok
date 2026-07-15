//! tk2's production matmul, computing `C = A·Bᵀ` on gfx942 (DESIGN.md §5c): the HipKittens 8-cluster
//! dot-slice pipeline + 8-wave ping-pong, authored declaratively over the [`crate::pipeline`] driver.
//!
//! - [`matmul_lds_kblock_mw_clustered`] — the **asm clustered HK replica**: the 8-cluster schedule via
//!   the [`crate::pipeline`] combinator with asm-opaque `ds_read_b64`/`ds_write_b64` gather+commit and
//!   `mma_asm` (backed by [`MatmulHooks`]). Asm-opacity is load-bearing for BOTH
//!   correctness and speed: it pins the LDS reads/writes so the single-buffer commit can't race a
//!   ping-pong-lagged read (the intrinsic "asm-free" variant could not be made race-safe — LLVM
//!   reschedules compiler-visible LDS ops past any authored barrier/drain), and it keeps the 32-MFMA
//!   run unbroken.

use super::{EDGE, Program, WARP, add_opt, scatter_frag};
use crate::build::{BF16, Buf, Builder, Effect, F32, Idx, Lds, Val};
use crate::ir::TileId;
use crate::pipeline::{BlockCounter, CommitDrain, Compute, Hooks, Mem, Sched, SlotVal, pipeline};
use crate::shape::{Mfma16x16x16Bf16, MfmaShape};
use crate::tile::{ARow, BCol};
use crate::tile_move::{commit_asm, gather, prefetch};

/// The register-staged fill bundle carried between the pipeline's prefetch and commit: the two operands'
/// b64/b128 chunks held in VGPRs, indexed by tile (`[A, B]`). Since B is taken **`[N,K]`** (HK's
/// pre-transposed layout), its fill is the SAME trivial coalesced `load→ds_write` as A — no register
/// transpose, no `v_perm`. At `stages=2` the prefetch runs a K-block ahead of the commit.
struct FillRegs {
    chunks: [Vec<Val<BF16>>; 2],
}

/// One matmul operand's movement context — its shared LDS tile, global source, workgroup row/col origin,
/// and per-lane element run. A[M,K] and B[N,K] share the SAME fill (both K-contiguous), so prefetch and
/// commit index `operands[tile]` rather than duplicating an A-arm and a B-arm.
#[derive(Copy, Clone)]
struct Operand {
    smem: Lds<BF16>,
    src: Buf<BF16>,
    origin: Idx,
    epl: usize,
}

/// matmul's [`Hooks`] impl for the §5c clustered [`pipeline`] — the ONLY kernel-specific part of the
/// clustered schedule. It rides the [`crate::tile_move`] handles (so the prefetch/commit/gather bodies
/// name no addressing) and carries `ri`/`cj`/`ept` for the per-accumulator MFMA grid. `Op` is the
/// `(A-vecs, B-vecs)` operand bundle of one K-slice; `Reg` is [`FillRegs`]. The prefetch/commit
/// bodies stage the register-buffered fill; the gather emits one K-slice's `ds_read` operands.
/// One K-slice's matmul operand bundle: the `ri` A-fragments + `cj` B-fragments the compute MFMAs.
type MatmulOp = (Vec<Val<BF16>>, Vec<Val<BF16>>);

struct MatmulHooks {
    /// The two operands' movement context (`[A, B]`) — LDS tile, global source, origin, per-lane run —
    /// the raw handles the `tile_move` prefetch/commit forwards address (they rebuild the `SharedTile`/
    /// `stage_view` internally; those emit no IR, so the emission is byte-identical). A[M,K] / B[N,K] are
    /// both K-contiguous, so the fill is symmetric and prefetch/commit index by tile.
    operands: [Operand; 2],
    /// `k_step` = the LDS tile inner width (`lds_cols`, the flat-layout row stride); `grow` = the global
    /// row stride `K` (the fill's `grow_stride`, shared by both operands). `tid` = the fill thread id.
    k_step: usize,
    grow: i64,
    tid: Idx,
    /// The gather addressing: `wlane` = the intra-warp lane, `bm`/`bn` = the per-warp sub-tile extents
    /// (A is `(bm, EDGE)` → `ri` fragments, B is `(EDGE, bn)` → `cj` — role-asymmetric, so NOT grouped),
    /// `warp_row_off`/`warp_col_off` = the multi-warp wave's runtime offset into the shared tile.
    wlane: Idx,
    bm: usize,
    bn: usize,
    warp_row_off: Option<Idx>,
    warp_col_off: Option<Idx>,
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
        // Tile `0 = A`, `1 = B` (the pipeline's `PREFETCH_TILES = 2` numeric ids). HK loads A@C0 and B@C4
        // so each global load hides under a different compute cluster; the fill accumulates across them
        // (`prev`) for the single C6 commit that writes BOTH. `order` (the cluster entry) pins each tile's
        // load into its cluster. The operands' fill is symmetric, so index `operands[tile]` — no per-tile
        // arm, no `_ => panic` (the array bound IS the operand count).
        let mut reg = prev.unwrap_or_else(|| FillRegs { chunks: [Vec::new(), Vec::new()] });
        let op = self.operands[tile];
        reg.chunks[tile] =
            prefetch(b, op.smem, self.k_step, op.src, self.grow, op.epl, self.tid, op.origin, k_base, order);
        // The load result values — the `sched_fence(0)` load-pin anchors on these so LLVM cannot sink
        // the global load down to its consumer (the commit), exposing the DRAM latency.
        let anchors: Vec<TileId> = reg.chunks[tile].iter().map(|v| v.id).collect();
        (reg, anchors)
    }

    fn commit(&mut self, b: &mut Builder, _k_base: Idx, reg: &FillRegs, war: &[TileId]) -> Vec<Effect> {
        // HK's waitcnt-opaque asm `ds_write_b64` commit (§5c — the only commit path; the production config
        // bake fixed the clustered kernel to asm, so there is no intrinsic fallback). Chain A then B writes
        // into ONE `prev` chain (thread A's tail into B) so a single drain reaches BOTH; return the WRITE
        // effects — the pipeline owns the drain (`CommitDrain::AsmDeferred`, since the RAW barrier can't
        // auto-drain the waitcnt-opaque asm and WHERE it drains is the schedule).
        let [op_a, op_b] = self.operands;
        let fa = commit_asm(
            b,
            op_a.smem,
            self.k_step,
            op_a.src,
            self.grow,
            op_a.epl,
            self.tid,
            op_a.origin,
            &reg.chunks[0],
            war,
            None,
        );
        let a_last = fa.last().map(|e| e.dep());
        let fb = commit_asm(
            b,
            op_b.smem,
            self.k_step,
            op_b.src,
            self.grow,
            op_b.epl,
            self.tid,
            op_b.origin,
            &reg.chunks[1],
            war,
            a_last,
        );
        fa.into_iter().chain(fb).collect()
    }

    fn gather(
        &mut self,
        b: &mut Builder,
        slice: usize,
        _block: BlockCounter,
        raw: &[TileId],
    ) -> (Self::Op, Vec<TileId>, TileId) {
        // One gather per operand at K-slice `slice`, via `tile_move::gather` in its asm `ds_read_b64` form
        // (the production path — the `true` below; asm ⇒ straight, so B's Col map routes through the
        // STRAIGHT contiguous gather, not the FA register-transpose). A = ARow over
        // `(bm, EDGE)` → `ri` frags; B = BCol over `(EDGE, bn)` → `cj` frags — `n_frags` derived by role.
        let mut gathers: Vec<TileId> = Vec::new();
        let (a_vecs, ga) = gather::<BF16, ARow, Mfma16x16x16Bf16>(
            b,
            self.operands[0].smem,
            self.k_step,
            self.bm,
            EDGE,
            self.warp_row_off,
            self.wlane,
            raw,
            slice,
            true,
        );
        let (b_vecs, gb) = gather::<BF16, BCol, Mfma16x16x16Bf16>(
            b,
            self.operands[1].smem,
            self.k_step,
            EDGE,
            self.bn,
            self.warp_col_off,
            self.wlane,
            raw,
            slice,
            true,
        );
        gathers.extend(ga);
        gathers.extend(gb);
        // op_anchor = an operand VALUE (the first A fragment) for `set_prio` to anchor on.
        let op_anchor = a_vecs[0].id;
        ((a_vecs, b_vecs), gathers, op_anchor)
    }
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
pub(crate) fn l2_swizzle(b: &mut Builder, wgid: Idx, grid_m: i64, grid_n: i64) -> (Idx, Idx) {
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

/// The clustered matmul's tiling config — the per-warp sub-tile dims + warp grid + K-step. Grouped so
/// the easily-swapped bm/bn and wm/wn are named at the call site; `Default` is the production config.
#[derive(Copy, Clone, Debug)]
pub struct Tiling {
    pub bm: usize,
    pub bn: usize,
    pub wm: usize,
    pub wn: usize,
    pub k_step: usize,
}

impl Default for Tiling {
    fn default() -> Self {
        Tiling { bm: 128, bn: 64, wm: 2, wn: 4, k_step: 64 } // the §5c HK config
    }
}

/// The **clustered HK replica** (DESIGN §5c): a 256²-tile stages=2 overlap whose steady body is
/// decomposed into the 8-cluster memory/compute
/// sequence with ALL scheduling placed by the [`crate::pipeline`] driver — the per-cluster `sched_fence(0)` then
/// `s_barrier` boundary, the `set_prio` compute brackets, and the warp-phase ping-pong (one
/// asymmetric `wave_barrier` per warp-row). Use HK's tiling `(bm=128, bn=64, wm=2, wn=4, k_step=64)`
/// so `warp_row = warp/4` in `{0,1}` gives the two phase groups. Balance is verified at build.
///
/// **K-blocked, LDS-staged, block-tiled** (DESIGN.md §5b step 1b-ii — the occupancy win): the A/B
/// strips are re-staged **per K-fragment inside the K-loop** (`k_step`) instead of the whole K at once
/// — so the LDS footprint is a tiny `(bm·k_step + k_step·bn)·2` bytes **independent of K**, keeping
/// occupancy high at any K. The single LDS buffer is reused every iteration, so each K-block needs two
/// workgroup barriers (mirroring tk's `gemm_core`): a **RAW** fence after the fill (reads see the staged
/// data) and a **WAR** fence after the LDS reads (the next fill must not overwrite until every lane
/// finished reading). `m/n` multiples of `bm·wm`/`bn·wn`; `bm/bn/k` multiples of 16. Emits the LDS
/// addressing through [`Builder::lds_col`], so the flat layout is the base; `.apply(`[`SwizzlePass`](crate::passes::SwizzlePass)`)`
/// turns it into the bank-swizzled one — the swizzle is a **composable refinement**, not hand-woven
/// here (bm/bn/k_step ∈ {16,32,64} for the single-subtile swizzle).
#[allow(clippy::needless_range_loop)]
pub fn matmul_lds_kblock_mw_clustered(m: usize, n: usize, k: usize, t: Tiling) -> Program {
    let Tiling { bm, bn, wm, wn, k_step } = t;
    assert!(bm.is_multiple_of(EDGE) && bn.is_multiple_of(EDGE) && k.is_multiple_of(EDGE), "tile dims multiples of 16");
    assert!(k_step.is_multiple_of(EDGE) && k.is_multiple_of(k_step), "k_step multiple of 16, K multiple of k_step");
    assert!(wm >= 1 && wn >= 1, "at least one warp per axis");
    // Production bake (§5c clustered HK replica — "one config, as aiter/hk do"): the scheduling knobs are
    // fixed here — stages=2, clustered, `asm_gather`, `bare_seals`, `AsmDeferred` commit (⇒ `asm_commit`);
    // `resident`/`pin_mfma` off. The register-staged pipeline needs ≥2 K-blocks to overlap; a single-block
    // K would trip the pipeline's `nblocks ≥ 2` assert at construction (production always satisfies this).
    assert!(k / k_step >= 2, "kblock: clustered pipeline needs ≥2 K-blocks (k/k_step ≥ 2)");
    // Workgroup output tile = (bm·wm) × (bn·wn), computed by a wm×wn grid of 64-lane warps.
    let (big_m, big_n, nthreads) = (bm * wm, bn * wn, wm * wn * WARP);
    assert!(m.is_multiple_of(big_m) && n.is_multiple_of(big_n), "m/n must tile by (bm·wm)/(bn·wn)");
    // Distinct module name from pipe2 (which also uses "tk2_matmul_kblock") so IR dumps don't collide.
    let mut b = Builder::new("tk2_matmul_clustered");

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

    // The accumulator lane map + width, DERIVED from the shape marker (§Step 1). The A/B operand maps
    // (A = Row, B = Col) are now derived inside `tile_move::gather` from the `ARow`/`BCol` roles — the
    // hooks name the role, not the `FragMap`. For 16×16×16 `c_map` equals `FragMap::gfx942_16x16(true)`.
    let c_map = Mfma16x16x16Bf16::c_map();
    let ept = Mfma16x16x16Bf16::EPT_C;
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

    // The movement RAW ingredients ride as DATA on `MatmulHooks` (below); the `tile_move` prefetch/
    // commit/gather forwards rebuild the per-op `SharedTile`/`gather_view`/`stage_view`/`slice` handles
    // internally (those emit NO IR, so the emission is byte-identical) and name NO addressing at the call
    // site. `asm_gather` is the gather's arch dispatch (gfx942 `ds_read_b64` vs scalar); A[M,K] and B[N,K]
    // are BOTH K-contiguous, so the fill is the identical coalesced copy (`origin` = M/N base, `grow` = K,
    // no transpose, no `v_perm`).

    // ── accumulators: one 16×16 f32 fragment per (i,j), zero-initialised. ALL carried, and every
    //    compute cluster reads+writes the full set — GEMM is the UNIFORM special case of the §3.2
    //    per-cluster read/write contract (`reads = writes = 0..ri·cj`), so it emits byte-identically. ──
    let mut slots = crate::pipeline::SlotSet::new();
    let all_slots = slots.carried_group(&mut b, ri * cj, c_map, crate::pipeline::Init::Zero);
    let (accs, inited) = slots.finish(&mut b);

    let acc_final = {
        // The §5c clustered HK replica: the movement handles + fills feed the 8-cluster HK schedule
        // (prefetch k+1 + gather slice 0 at C0, gathers
        // spread C0/C2/C4 with slice 3 read early for C7, deferred commit at C6, the four MFMA slices
        // at C1/C3/C5/C7). The pipeline combinator owns ALL placement (per-cluster barrier + set_prio
        // brackets, warp-phase ping-pong, End-fold, resident fork) and runs the completeness verifier
        // at `.build()`; the author declares only the schedule + the `MatmulHooks` (§5c cluster model).
        let hooks = MatmulHooks {
            operands: [
                Operand { smem: a_smem, src: a, origin: tm_bm, epl: epl_a },
                Operand { smem: b_smem, src: bmat, origin: tn_bn, epl: epl_b },
            ],
            k_step,
            grow: k as i64,
            tid,
            wlane,
            bm,
            bn,
            warp_row_off,
            warp_col_off,
        };
        // The compute clusters carry the kernel math (the `ri×cj` MFMA loop) as an edge-free `body` —
        // the combinator brackets it with `set_prio` + the acc round-trip. This is what makes the
        // compute side pluggable: FA's softmax/PV clusters carry their own body, `Hooks` never grows a
        // compute method. `mma(s)` mints a compute cluster over gathered slice `s` (always `Some`).
        let mma = |s: usize| -> Compute<MatmulHooks> {
            Compute::new(
                s,
                all_slots.clone(),
                all_slots.clone(),
                move |b: &mut Builder, op: Option<&MatmulOp>, reads: &[SlotVal], _blk: BlockCounter| {
                    let (a_vecs, b_vecs) = op.expect("matmul compute consumes a gathered operand");
                    let mut out = Vec::with_capacity(ri * cj);
                    for i in 0..ri {
                        for j in 0..cj {
                            // Asm-sideeffect MFMA (opaque to LLVM's scheduler → the 32-run cannot be
                            // fractured; tk's `mma_abt_asm` pin, verified: the intrinsic path is unpinnable).
                            out.push(SlotVal::F32(b.mma_asm(a_vecs[i], b_vecs[j], reads[i * cj + j].f32(), ept)));
                        }
                    }
                    out
                },
            )
        };
        pipeline(
            &mut b,
            k / k_step,
            k_step,
            ksteps,
            &accs,
            &inited,
            warp_row,
            Sched {
                asm_gather: true,
                resident: false,
                commit_drain: CommitDrain::AsmDeferred,
                bare_seals: true,
                pin_mfma: false,
            },
            hooks,
        )
        .cluster(Mem::builder().prefetch([0]).gathers([0]).build()) // C0: load A, gather slice 0
        .cluster(mma(0)) // C1
        .cluster(Mem::builder().gathers([1, 2]).build()) // C2: lead slice 2 (3-cluster read latency to hide, per HK)
        .cluster(mma(1)) // C3
        .cluster(Mem::builder().prefetch([1]).gathers([3]).build()) // C4: load B (HK split), gather slice 3 only
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
            roots.extend(scatter_frag(&mut b, acc_final[idx].f32(), c, base_c, n as i64, wlane));
        }
    }

    let (ir, sink) = b.finish(&roots);
    // The wave-phase balance + carry-completeness are verified inside `Pipeline::build` (§5c/3c).
    Program { ir, sink, name: "tk2_matmul_kblock".into() }
}
