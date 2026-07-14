//! tk2's production matmul, computing `C = A·Bᵀ` on gfx942 (DESIGN.md §5c): the HipKittens 8-cluster
//! dot-slice pipeline + 8-wave ping-pong, authored declaratively over the [`crate::pipeline`] driver.
//!
//! - [`matmul_lds_kblock_mw_clustered`] — the **asm clustered HK replica**: the 8-cluster schedule via
//!   the [`crate::pipeline`] combinator with asm-opaque `ds_read_b64`/`ds_write_b64` gather+commit and
//!   `mma_asm` (backed by [`kblock_impl`] + [`MatmulHooks`]). Asm-opacity is load-bearing for BOTH
//!   correctness and speed: it pins the LDS reads/writes so the single-buffer commit can't race a
//!   ping-pong-lagged read (the intrinsic "asm-free" variant could not be made race-safe — LLVM
//!   reschedules compiler-visible LDS ops past any authored barrier/drain), and it keeps the 32-MFMA
//!   run unbroken. The hand-inline reference at the same ceiling is [`crate::hk`] (`hk/gemm.rs`, 638 TF),
//!   which rides the typed [`crate::schedule::pipeline`] driver with the same asm emission.

use crate::build::{BF16, Buf, Builder, Effect, F32, Frag, Idx, Val};
use crate::ir::{FragMap, TileId, TileIr};
use crate::movement::{Drain, LdsStage, LdsView, SharedTile};
use crate::pass::Pass;
use crate::pipeline::{AccSlot, CommitDrain, Compute, Hooks, Mem, SlotVal, pipeline};
use crate::shape::{Mfma16x16x16Bf16, MfmaShape};

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

/// The gfx942 MFMA edge — one 16×16×16 fragment per workgroup, one 64-lane warp. Now DERIVED from the
/// [`Mfma16x16x16Bf16`] marker (`M == N == K == 16`) instead of a bare literal (§migration Step 1): the
/// type computes what was hardcoded, and every `EDGE` call site is byte-identical.
pub(crate) const EDGE: usize = Mfma16x16x16Bf16::M;
const WARP: usize = 64;

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
    // The accumulator C-site re-derived from the marker: `acc_rc(acc_dist())` replaces the FragMap
    // `lane_rc`. For 16×16×16 (`m_blocks == 1`) it interns to the identical index nodes (byte-identical);
    // for 32×32×8 the AccDist carries the 4×4 block layout a single FragMap run cannot.
    let dist = Mfma16x16x16Bf16::acc_dist();
    (0..Mfma16x16x16Bf16::EPT_C)
        .map(|inner| {
            let inner_idx = b.idx_const(inner as i64);
            let (row, col) = b.acc_rc(dist, lane, inner);
            let row_off = b.idx_mul(row, rs);
            let off = b.idx_add(base, row_off);
            let off = b.idx_add(off, col);
            let v = b.load_frag_elem(acc, inner_idx);
            b.store(dst, off, v)
        })
        .collect()
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

/// Load ONE MFMA operand fragment (`map.ept` wide) straight from a global `[outer, K]` (row-major, row
/// stride `k_dim`) into a register fragment, addressed by `map`'s `lane_rc`. For a Row (A) map the lane
/// pair is `(M-row, K)`; for a Col (B) map it is `(K-spread, N-flat)` — so `outer` is the M (A) / N (B)
/// index and `kk` the K contribution, and the global offset is `(outer_base + outer)·k_dim + (k_base +
/// kk)`. Used only by the 32×32×8 isolation probe (a self-contained gather, free of the EDGE-coupled
/// movement layer, so it exercises the marker's operand maps directly).
pub(crate) fn load_op_frag(
    b: &mut Builder,
    src: Buf<BF16>,
    map: FragMap,
    outer_base: usize,
    k_base: usize,
    k_dim: usize,
    lane: Idx,
) -> Val<BF16> {
    let frag = b.define_frag::<BF16>(map);
    let k_c = b.idx_const(k_dim as i64);
    let stores: Vec<TileId> = (0..map.ept)
        .map(|e| {
            let e_idx = b.idx_const(e as i64);
            let (r, cc) = b.lane_rc(map, lane, e_idx);
            let (outer, kk) = if map.transpose { (cc, r) } else { (r, cc) };
            let row = offset_by(b, outer, outer_base);
            let col = offset_by(b, kk, k_base);
            let off = b.idx_mul(row, k_c);
            let off = b.idx_add(off, col);
            let v = b.load(src, off);
            b.store_frag_elem(frag, e_idx, v).dep()
        })
        .collect();
    b.load_frag_vec_after(frag, &stores)
}

/// **32×32×8 MFMA isolation probe** (§migration Step 3 DE-RISK): a standalone single-warp kernel that
/// computes `C = A·Bᵀ` (`A[m,k]`, `B[n,k]`, both K-contiguous) tiled as `(m/32)×(n/32)` output blocks,
/// each ONE `v_mfma_f32_32x32x8_bf16` per K-slice accumulated over the `k/8` slices, then scatters the
/// 16-VGPR accumulator via [`Builder::acc_rc`]. It PROVES the 32×32×8 operand layout + accumulator
/// distribution IN ISOLATION (device-gated allclose vs an f32 reference) before any FA work — the
/// wide-core analog of `atb_probe`. `m`/`n` multiples of 32, `k` a multiple of 8. Emitted with the
/// intrinsic MFMA ([`Builder::mma_of`], the production fast path).
#[allow(clippy::needless_range_loop)]
pub fn mfma_32x32x8_probe(m: usize, n: usize, k: usize) -> Program {
    use crate::shape::Mfma32x32x8Bf16 as S;
    assert!(
        m.is_multiple_of(S::M) && n.is_multiple_of(S::N) && k.is_multiple_of(S::K),
        "probe dims must tile by 32×32×8"
    );
    let mut b = Builder::new("tk2_mfma_32x32x8_probe");
    // ABI: output C[m,n] first, then inputs A[m,k], B[n,k] (B is [N,K] — A·Bᵀ, both K-contiguous).
    let c = b.global::<F32>(m * n);
    let a = b.global::<BF16>(m * k);
    let bmat = b.global::<BF16>(n * k);
    let _wg = b.grid_axis(0, 1);
    let lane = b.block_axis(WARP as i64);

    let a_map = S::a_map();
    let b_map = S::b_map();
    let dist = S::acc_dist();
    let n_c = b.idx_const(n as i64);
    let mut roots = Vec::new();
    for mi in 0..m / S::M {
        for ni in 0..n / S::N {
            // Accumulate `k/8` MFMAs into a 16-wide f32 value (the C accumulator, register-carried).
            let mut acc = {
                let zs: Vec<Val<F32>> = (0..S::EPT_C).map(|_| b.f32(0.0)).collect();
                b.vec_build(&zs)
            };
            for ki in 0..k / S::K {
                let af = load_op_frag(&mut b, a, a_map, mi * S::M, ki * S::K, k, lane);
                let bf = load_op_frag(&mut b, bmat, b_map, ni * S::N, ki * S::K, k, lane);
                acc = b.mma_of::<S>(af, bf, acc);
            }
            // Scatter the 16-VGPR accumulator: element `i` → C[mi·32 + row, ni·32 + col] via acc_rc.
            for i in 0..S::EPT_C {
                let (row, col) = b.acc_rc(dist, lane, i);
                let m_idx = offset_by(&mut b, row, mi * S::M);
                let n_idx = offset_by(&mut b, col, ni * S::N);
                let off = b.idx_mul(m_idx, n_c);
                let off = b.idx_add(off, n_idx);
                let v = b.vec_extract(acc, i);
                roots.push(b.store(c, off, v));
            }
        }
    }
    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_mfma_32x32x8_probe".into() }
}

/// The register-staged fill bundle carried between the pipeline's prefetch and commit: A's and
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
/// `(A-vecs, B-vecs)` operand bundle of one K-slice; `Reg` is [`FillRegs`]. The prefetch/commit
/// bodies stage the register-buffered fill; the gather emits one K-slice's `ds_read` operands.
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
    // `kblock_impl` is now the clustered §5c path only. `clustered` stays true for any valid call
    // (k/k_step ≥ 2 ⇒ stages=2); the gate is defensive — a stages=1 collapse would trip the
    // pipeline's `nblocks ≥ 2` assert at construction, never silently miscompile.
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

    // Operand + accumulator lane maps and the accumulator width, DERIVED from the shape marker (§Step 1):
    // A = Row map, B = Col map, C = the accumulator carrier. For 16×16×16 these equal the former
    // `FragMap::gfx942_16x16(false/true)` / `ept = 4` byte-for-byte.
    let a_map = Mfma16x16x16Bf16::a_map();
    let b_map = Mfma16x16x16Bf16::b_map();
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
    let b_view = b_tile.gather_view(b_map, cj, warp_col_off, wlane, asm_gather);
    // A[M,K] and B[N,K] are BOTH K-contiguous → the identical trivial coalesced fill: `origin` = the
    // M/N row base, `grow_stride` = K. No transpose, no `v_perm` — B's fill IS A's fill.
    let a_stage = a_tile.stage_view(a, epl_a, tid, tm_bm, k as i64, stage_drain);
    let b_stage = b_tile.stage_view(bmat, epl_b, tid, tn_bn, k as i64, stage_drain);

    // ── accumulators: one 16×16 f32 fragment per (i,j), zero-initialised. ALL carried, and every
    //    compute cluster reads+writes the full set — GEMM is the UNIFORM special case of the §3.2
    //    per-cluster read/write contract (`reads = writes = 0..ri·cj`), so it emits byte-identically. ──
    let acc: Vec<Frag<F32>> = (0..ri * cj).map(|_| b.define_frag::<F32>(c_map)).collect();
    let accs: Vec<AccSlot> = acc.iter().map(|&f| AccSlot::F32(f)).collect();
    let inited: Vec<Option<Effect>> = acc.iter().map(|&ac| Some(b.zero_init_frag(ac))).collect();
    let all_slots: Vec<usize> = (0..ri * cj).collect();

    let acc_final = {
        // The §5c clustered HK replica: the movement handles + fills feed the 8-cluster HK schedule
        // (prefetch k+1 + gather slice 0 at C0, gathers
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
            Compute::new(
                s,
                all_slots.clone(),
                all_slots.clone(),
                move |b: &mut Builder, op: Option<&MatmulOp>, reads: &[SlotVal]| {
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
            asm_gather,
            resident,
            commit_drain,
            bare_seals,
            pin_mfma,
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

/// The **clustered HK replica** (DESIGN §5c): a 256²-tile stages=2 overlap whose steady body is
/// decomposed into the 8-cluster memory/compute
/// sequence with ALL scheduling placed by the [`crate::pipeline`] driver — the per-cluster `sched_fence(0)` then
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
    kblock_impl(m, n, k, bm, bn, wm, wn, k_step, 2, true, true, false, CommitDrain::AsmDeferred, true, false)
}
