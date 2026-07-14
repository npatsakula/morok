//! The **explicit-schedule combinator** (DESIGN.md conversation 2026-07-08): clusters as
//! closure scopes, dependencies as value-flow tokens, cross-iteration edges as named carries.
//!
//! The three pieces:
//! - **cluster scopes** ([`mem_cluster`]/[`compute_cluster`] → [`MemScope`]/[`ComputeScope`]) — a
//!   lexical scope a cluster body authors inside `|m|`/`|c| { ... }`. The scheduling surface is gated
//!   by KIND (mem: `fence`/`drain_lgkm`; compute: `set_prio`), and both deref to [`Builder`] so the
//!   movement handles (`LdsView`/`LdsStage`) work unchanged. A cluster ends with a `seal` (a barrier).
//! - **Tokens** ([`Gathered`]/[`InFlight`]/[`Committed`]) — move-only wrappers carrying both
//!   the values and the dependency edges. A `Gathered` produced in one cluster is consumed by
//!   exactly one later cluster; Rust's move semantics make double-use a compile error.
//! - [`pipeline`] — owns the loop skeleton (Range/End + prologue/epilogue + carry-fold + the 8-wave
//!   ping-pong wave barriers); the body authors the steady iteration as a sequence of clusters.
//!
//! `matmul_lds_kblock_mw_pipe2` authors HipKittens' 8-cluster dot-slice pipeline over this DSL.

use crate::build::{Builder, Effect, Elem, F32, Frag, Idx, Val};
use crate::ir::{FragMap, Node, TileId, TileIr};
use crate::movement::LdsView;

// ══════════════════════════ tokens (move-only value-flow) ══════════════════════════

/// A **gather's result** — the operand values for the MMA + the store-edge tokens a WAR/commit
/// consumes. Produced by a gather inside a cluster; consumed by exactly one compute cluster
/// (`.operands()`) or fed into a commit's WAR. Move-only (not `Clone`) — the operands can't be
/// MMA'd twice, and the edges can't be double-fenced. (`edges` is a `Vec` because one `gather`
/// emits several `ds_read` completion tokens; a single WAR seal covers them all.)
#[derive(Debug)]
pub struct Gathered<E: crate::build::Elem> {
    operands: Vec<Val<E>>,
    edges: Vec<TileId>,
}

impl<E: crate::build::Elem> Gathered<E> {
    /// Construct from a gather's `(operands, store-edges)`.
    pub fn new(operands: Vec<Val<E>>, edges: Vec<TileId>) -> Self {
        Gathered { operands, edges }
    }
    /// Take the operand values for the compute MMA. Consumes `self` (one-shot).
    pub fn operands(self) -> Vec<Val<E>> {
        self.operands
    }
    /// The gather's store-edge tokens (for a WAR/commit's dependency list). Borrowed — does not
    /// consume the operands, so a compute + a WAR can both reference them.
    pub fn edges(&self) -> &[TileId] {
        &self.edges
    }
}

/// A prefetch load's result — data in-flight in VGPRs (the global load issued, the `ds_write`
/// commit not yet). Consumed by exactly one `.commit()`. Move-only (not `Clone`) — the same
/// in-flight load must not be committed to LDS twice.
#[derive(Debug)]
pub struct InFlight<E: crate::build::Elem>(pub(crate) Vec<Val<E>>);

impl<E: crate::build::Elem> InFlight<E> {
    /// Construct from the prefetch's loaded VGPR chunks.
    pub fn new(chunks: Vec<Val<E>>) -> Self {
        InFlight(chunks)
    }
    /// The loaded VGPR chunks (for `LdsStage::commit`).
    pub fn chunks(&self) -> &[Val<E>] {
        &self.0
    }
}

/// A committed fill — data written to LDS. Its store edges feed the RAW barrier the pipeline
/// folds into the loop carry.
#[derive(Debug)]
pub struct Committed {
    stores: Vec<Effect>,
}

impl Committed {
    /// Construct from the commit's store effects.
    pub fn new(stores: Vec<Effect>) -> Self {
        Committed { stores }
    }
    /// The first store (barrier body) + the rest (barrier deps) — for the RAW fence.
    pub fn barrier_parts(&self) -> (Effect, Vec<TileId>) {
        (self.stores[0], self.stores[1..].iter().map(|e| e.dep()).collect())
    }
}

// ══════════════════════════ register-tile pool (residency) ══════════════════════════
//
// The THIRD schedule dimension, alongside ordering (barriers/fences) and value-flow (the tokens
// above): register RESIDENCY. `TilePool` makes the operand working set an explicit object — a fixed
// set of register-tile slots minted ONCE and reused across dot-slices — so the read-ahead DEPTH is a
// single number whose live VGPR cost (`vgprs()`) is computable and build-checkable, instead of an
// emergent property of the register allocator. svod's analog of HipKittens' `rt tiles[8]` pool.

/// A fixed-capacity pool of operand register tiles — the explicit read-ahead working set. `depth`
/// dot-slices are resident at once (`depth · n_frags` slots), so its live VGPR cost is bounded and
/// asserted against the hardware ceiling at build. Slots are reused across slices (phase `s % depth`);
/// the caller threads each stage's ordering-in edges (the LDS-RAW carry + the slot's recycle edge).
pub(crate) struct TilePool<E: Elem> {
    slots: Vec<Vec<Frag<E>>>,     // [phase][fragment] — `depth` phases × `n_frags` reused slots
    recycle: Vec<Vec<TileId>>,    // [phase] — the resident slice's consuming-MMA edges (the WAR source)
    occupant: Vec<Option<usize>>, // [phase] — the slice holding the slots, until `consumed` frees it
    depth: usize,
    frag_vgprs: usize,
}

impl<E: Elem> TilePool<E> {
    /// Mint a `depth`-deep pool of `n_frags` slots per phase over MFMA lane-map `map`. `frag_vgprs`
    /// is one fragment's hardware VGPR width (bf16 `<4×bf16>` = 2), for the budget check.
    pub(crate) fn new(b: &mut Builder, map: FragMap, depth: usize, n_frags: usize, frag_vgprs: usize) -> Self {
        assert!(depth >= 1, "pool depth ≥ 1");
        let slots = (0..depth).map(|_| (0..n_frags).map(|_| b.define_frag::<E>(map)).collect()).collect();
        TilePool { slots, recycle: vec![Vec::new(); depth], occupant: vec![None; depth], depth, frag_vgprs }
    }

    /// Gather dot-slice `s` into its phase (`s % depth`) slots, ordering the `ds_read`s after `deps`
    /// (the LDS-RAW carry) AND — when the phase is being REUSED — the previous occupant's recycle edge.
    /// That recycle edge is the register WAR: sequencing this gather's READS after the prior slice's
    /// consuming MMA means its stores (value-dependent on those reads) cannot clobber an operand the
    /// MMA has not yet read, and the prior operands are dead so their VGPRs are free to reuse. The
    /// ordering rides the READ path — the same one the LDS-RAW carry uses — not a store re-bind: an
    /// `After` on a store buffer is not an honored scheduling edge, whereas read-after is. **Panics**
    /// if the phase is still OCCUPIED — reading `depth` slices ahead without consuming exhausts the
    /// pool, a schedule bug we surface LOUDLY at build rather than let a stale slot corrupt numerics.
    /// Returns the operand `Val`s + the store-fence tokens the seal consumes.
    pub(crate) fn stage(
        &mut self,
        b: &mut Builder,
        view: LdsView<E>,
        s: usize,
        deps: &[TileId],
    ) -> (Vec<Val<E>>, Vec<TileId>) {
        let phase = s % self.depth;
        assert!(
            self.occupant[phase].is_none(),
            "TilePool depth {} exhausted: slice {s} needs phase {phase}, still held by slice {} \
             (read-ahead exceeds pool depth — raise depth or consume() the resident slice first)",
            self.depth,
            self.occupant[phase].expect("occupied"),
        );
        let mut raw = deps.to_vec();
        raw.extend_from_slice(&self.recycle[phase]);
        self.occupant[phase] = Some(s);
        view.slice(s).gather_into(b, &raw, &self.slots[phase])
    }

    /// Mark slice `s`'s operands drained by its MMA: free the phase and arm its recycle edge (`edges`,
    /// the consuming accumulator stores) for the next slice that reuses the slots. **Panics** if the
    /// phase does not currently hold slice `s` — `consumed` must pair with the matching `stage`.
    pub(crate) fn consumed(&mut self, s: usize, edges: &[TileId]) {
        let phase = s % self.depth;
        assert_eq!(
            self.occupant[phase],
            Some(s),
            "TilePool.consumed({s}) but phase {phase} holds {:?} — stage/consume order mismatch",
            self.occupant[phase],
        );
        self.occupant[phase] = None;
        self.recycle[phase] = edges.to_vec();
    }

    /// The pool's live VGPR cost — `depth · n_frags · frag_vgprs`. For the register-budget assert.
    pub(crate) fn vgprs(&self) -> usize {
        self.depth * self.slots.first().map_or(0, Vec::len) * self.frag_vgprs
    }
}

// ══════════════════════════════ cluster scopes (typestate) ══════════════════════════════
//
// A cluster is one of two KINDS, and the kind is a TYPE, not a runtime flag. The kind gates the
// **seal vocabulary** — the FA-redesign §2.2 correctness/scheduling SPLIT:
//
//   CORRECTNESS (hard — a dropped one is a silent wrong answer)
//     · `MemScope::seal_fence`   — the acq-rel `s_barrier` (`Node::Barrier`, implicit `lgkmcnt(0)`):
//        the K/V LDS RAW/WAR fence. The ONE seal that guards real cross-wave shared-LDS traffic.
//     · `MemScope::rendezvous`   — the bare `s_barrier` metronome (`Node::BareBarrier`): a workgroup
//        rendezvous with NO acq-rel fence — the two-warp-group phase-lock carrier.
//   SCHEDULING (soft — a dropped one costs perf, never correctness; emits NO instruction)
//     · `ComputeScope::seal_ordering` — a pure `Node::After` combine (ZERO instructions): the
//        MFMA-grid seal is scheduling-TRANSPARENT, so the softmax VALU/exp interleave under the
//        MFMAs is not walled. The accumulator RAW is carried per-slot (the store edge), NOT here.
//     · `ComputeScope::interleave_valu` / `interleave_exp` — the `sched_group_barrier` ratio hint.
//     · `ComputeScope::set_prio` — the `s_setprio` MFMA-burst priority.
//
// So a `set_prio`/interleave on a memory cluster or a `seal_fence` on a compute cluster is a
// *compile* error, not a convention — the exact "scheduling can attach anywhere" hole in the
// untyped `crate::pipeline`. Both deref to `&mut Builder` so the movement handles work unchanged.

/// A **memory cluster scope** — loads/stores/gathers. Exposes the memory-side correctness seals
/// (`seal_fence` LDS-RAW/WAR, `rendezvous` metronome) + the load-pin `fence`/`drain_lgkm`; NOT the
/// compute scheduling surface. Derefs to `&mut Builder` for the movement handles.
pub struct MemScope<'b> {
    b: &'b mut Builder,
}

/// A **compute cluster scope** — the MFMA grid. Exposes ONLY the scheduling surface (`seal_ordering`,
/// `interleave_valu`/`interleave_exp`, `set_prio`); no LDS fence (that is memory-side). Derefs to
/// `&mut Builder` for `mma`/frag ops.
pub struct ComputeScope<'b> {
    b: &'b mut Builder,
}

macro_rules! deref_to_builder {
    ($scope:ident) => {
        impl<'b> std::ops::Deref for $scope<'b> {
            type Target = Builder;
            fn deref(&self) -> &Builder {
                self.b
            }
        }
        impl<'b> std::ops::DerefMut for $scope<'b> {
            fn deref_mut(&mut self) -> &mut Builder {
                self.b
            }
        }
    };
}
deref_to_builder!(MemScope);
deref_to_builder!(ComputeScope);

impl<'b> MemScope<'b> {
    /// The **LDS-RAW/WAR correctness seal** — the acq-rel `s_barrier` (`Node::Barrier`): its implicit
    /// `lgkmcnt(0)` fences the K/V fill before the next gather reads it. The one seal that guards real
    /// cross-wave shared-LDS traffic. `body` is the last store; `deps` the additional edges.
    pub fn seal_fence(&mut self, body: Effect, deps: &[TileId]) -> Effect {
        self.b.barrier(body, deps)
    }

    /// The **two-warp-group metronome** — a bare `s_barrier` (`Node::BareBarrier`, no acq-rel fence):
    /// a pure workgroup rendezvous that phase-locks the two stagger groups. Correctness (LDS ordering)
    /// is supplied separately by `drain_lgkm` at the RAW/WAR points — this only rendezvouses.
    pub fn rendezvous(&mut self, body: Effect, deps: &[TileId]) -> Effect {
        self.b.bare_barrier(body, deps)
    }

    /// A **scheduler fence** (`sched_barrier(mask)`) — the load-pin: forbids the machine scheduler
    /// sinking a prefetch load down to the commit's `vmcnt(0)` wait. Memory-cluster-only.
    pub fn fence(&mut self, mask: i64, anchors: &[TileId]) -> Effect {
        self.b.sched_fence(mask, anchors)
    }

    /// The LDS drain (`s_waitcnt lgkmcnt(0)`) — wait for outstanding gather reads before a commit
    /// overwrites LDS (the WAR guard). Memory-cluster-only.
    pub fn drain_lgkm(&mut self, prev: TileId) -> Effect {
        self.b.swait_lgkmcnt(prev)
    }
}

impl<'b> ComputeScope<'b> {
    /// The **scheduling-transparent seal** — a pure `Node::After` ordering combine (ZERO instructions):
    /// the MFMA grid's cluster boundary, folding `body` (the last acc store) + `deps` into one token so
    /// the carry threads on, but emitting NO `s_barrier`/wall — so LLVM is free to interleave the softmax
    /// VALU/exp under the MFMAs. The accumulator RAW is carried per-slot (the store edge), not by this.
    pub fn seal_ordering(&mut self, body: Effect, deps: &[TileId]) -> Effect {
        self.b.combine(body, deps)
    }

    /// The **interleave ratio** (`sched_group_barrier`): `pairs`×{ 1 MFMA, then `valu` VALU } in
    /// scheduling group `group` — the softmax reduction VALU folded under the MFMA. Returns the final
    /// hint effect to thread onward (keeps it live). Compute-cluster-only.
    pub fn interleave_valu(&mut self, pairs: u32, valu: u32, group: i64, anchors: &[TileId]) -> Option<Effect> {
        self.b.interleave_valu(pairs, valu, group, anchors)
    }

    /// The **exp interleave** (`sched_group_barrier`): `pairs`×{ 1 MFMA, then `exp` transcendental } —
    /// the softmax `exp2` folded under the P·V MFMA. Compute-cluster-only.
    pub fn interleave_exp(&mut self, pairs: u32, exp: u32, group: i64, anchors: &[TileId]) -> Option<Effect> {
        self.b.interleave_exp(pairs, exp, group, anchors)
    }

    /// **Wave priority** (`s_setprio`) — bracket the MFMA run so the compute wave wins SIMD issue
    /// slots over the co-resident loading wave. `level=1` before the run, `level=0` after.
    /// Compute-cluster-only: a memory cluster has no `set_prio`.
    pub fn set_prio(&mut self, level: i64, after: &[TileId]) -> Effect {
        self.b.set_prio(level, after)
    }
}

/// **Open a memory cluster.** `body` authors loads/gathers/commits inside `|m| { … }`
/// (`m: &mut MemScope` derefs to `&mut Builder`) and returns whatever tokens/ports the next
/// cluster consumes (e.g. `(Gathered…, war_edge)` or `(Committed, raw_next)`).
pub fn mem_cluster<R>(b: &mut Builder, body: impl FnOnce(&mut MemScope<'_>) -> R) -> R {
    body(&mut MemScope { b })
}

/// **Open a compute cluster.** `body` authors the MFMA grid inside `|c| { … }`
/// (`c: &mut ComputeScope`), bracketed by `set_prio`; returns the accumulator stores (the
/// register carry-out).
pub fn compute_cluster<R>(b: &mut Builder, body: impl FnOnce(&mut ComputeScope<'_>) -> R) -> R {
    body(&mut ComputeScope { b })
}

// ══════════════════════════════ carry channels ══════════════════════════════

/// A **named loop-back carry** — the cross-iteration edge the pipeline owns. Read by name from
/// the steady context; the pipeline folds the carry-out into the loop `End`. Zero-cost: the
/// `TileId`s of the seed + range edges, threaded by the pipeline so the body reads them by name.
#[derive(Debug, Clone)]
pub struct Carry<M> {
    deps: Vec<TileId>,
    _m: std::marker::PhantomData<M>,
}

impl<M> Carry<M> {
    /// The carry edges — pass to the consuming op (gather, acc-read).
    pub fn deps(&self) -> &[TileId] {
        &self.deps
    }

    pub(crate) fn wrap(deps: Vec<TileId>) -> Self {
        Carry { deps, _m: std::marker::PhantomData }
    }
}

/// Marker: the LDS-RAW carry (commit → next-iteration gather).
pub struct Raw;
/// Marker: the register-accumulator carry (the MFMA fold).
pub struct Acc;

// ══════════════════════════════ pipeline ══════════════════════════════

/// The **steady-body context** — what the steady body receives. Carries the loop counter, the
/// next-block K-base, and the named carry channels.
pub struct PipelineCx {
    /// The K-loop counter (block k within the steady range `0..nblocks-1`).
    pub counter: Idx,
    /// Block k+1's K-base `(counter+1)·k_step` — pass to `prefetch`.
    pub next_base: Idx,
    /// The LDS-RAW carry (the previous iteration's commit barrier + range). Pass to `gather`.
    pub raw: Carry<Raw>,
    /// The register-accumulator carries (init-edge + range, one per accumulator). Append the
    /// intra-iteration WAR edge, then pass to `load_frag_vec_after`.
    pub accs: Vec<Carry<Acc>>,
}

impl PipelineCx {
    /// The number of accumulators.
    pub fn n_acc(&self) -> usize {
        self.accs.len()
    }
}

/// What the steady body returns: the carry-OUTs the pipeline folds into the `End`.
#[derive(Debug)]
pub struct SteadyOut {
    /// The accumulator stores (register carry-out), one per accumulator.
    pub acc_stores: Vec<Effect>,
    /// The RAW barrier after this iteration's commit (LDS carry-out).
    pub raw_next: TileId,
}

/// The **register-staged pipeline** (`stages=2`). Owns the loop skeleton; the body authors the
/// steady iteration as clusters. The prologue and epilogue are kernel-specific, so they're
/// closures the caller supplies — `prologue` returns the `raw_seed` (the block-0 commit barrier);
/// `epilogue` gathers+MFMAs the last block via the loop `End`'s carried RAW. Requires `nblocks ≥ 2`.
/// `warp_row` (`Some` only for HK's 2-warp-row config) enables the **8-wave ping-pong**: an `eq=1`
/// wave-phase barrier after the prologue offsets warp-row 1 by one cluster (only it executes the
/// barrier — the counter-based `s_barrier` then pairs the two groups offset-by-one, so one warp-row
/// runs a compute cluster while its SIMD-partner runs the paired memory cluster), rebalanced by an
/// `eq=0` barrier the epilogue emits. The pair is balanced (verified) so the workgroup can't deadlock.
#[allow(clippy::too_many_arguments)]
pub fn pipeline(
    b: &mut Builder,
    nblocks: usize,
    k_step: usize,
    accs: &[Frag<F32>],
    inited: &[Effect],
    warp_row: Option<Idx>,
    prologue: impl FnOnce(&mut Builder) -> TileId,
    steady: impl FnOnce(&mut Builder, &PipelineCx) -> SteadyOut,
    epilogue: impl FnOnce(&mut Builder, TileId, &[Frag<F32>], Option<Idx>) -> Vec<Frag<F32>>,
) -> Vec<Frag<F32>> {
    assert!(nblocks >= 2, "pipeline needs nblocks ≥ 2");

    let ks_c = b.idx_const(k_step as i64);
    let one = b.idx_const(1);

    // ── prologue (kernel-specific): returns the block-0 raw_seed. ──
    let raw_seed = prologue(b);
    // ── ping-pong seed: eq=1 wave-phase barrier offsets warp-row 1 one cluster (balanced by the
    //    epilogue's eq=0). Ordered after the prologue commit; rides as the steady RAW carry seed. ──
    let loop_seed = match warp_row {
        Some(wr) => b.wave_barrier(wr, 1, &[raw_seed]).dep(),
        None => raw_seed,
    };

    // ── steady loop ──
    let kr = b.range((nblocks - 1) as i64);
    let tk = b.counter(kr);
    let k_next_idx = b.idx_add(tk, one);
    let k_next = b.idx_mul(k_next_idx, ks_c);

    let cx = PipelineCx {
        counter: tk,
        next_base: k_next,
        raw: Carry::<Raw>::wrap(vec![loop_seed, kr.dep()]),
        accs: (0..accs.len()).map(|i| Carry::<Acc>::wrap(vec![inited[i].dep(), kr.dep()])).collect(),
    };

    let steady_out = steady(b, &cx);

    // ── End-fold: fold acc stores + raw_next into one End. ──
    let last = *steady_out.acc_stores.last().expect("steady body must produce ≥1 acc store");
    let mut carried: Vec<TileId> =
        steady_out.acc_stores[..steady_out.acc_stores.len() - 1].iter().map(|e| e.dep()).collect();
    carried.push(steady_out.raw_next);
    let combined = b.combine(last, &carried);
    let ended = b.end(combined, &[kr]);
    let acc_loop: Vec<Frag<F32>> = accs.iter().map(|a| b.frag_after(*a, &[ended.dep()])).collect();

    // ── epilogue (kernel-specific): gather+MFMA the last block via the End-carried RAW + emit the
    //    eq=0 rebalance barrier (if ping-pong), returning the accumulators rebound past its store. ──
    let out = epilogue(b, ended.dep(), &acc_loop, warp_row);

    // ── wave-phase balance (deadlock guard): equal eq=0/eq=1 wave barriers reachable from the output. ──
    if warp_row.is_some() {
        let mut reach: std::collections::HashSet<TileId> = std::collections::HashSet::new();
        for f in &out {
            reach.extend(crate::passes::reachable(&b.ir, f.id));
        }
        let count = |want: i64| {
            reach
                .iter()
                .filter(|&&id| matches!(b.ir.node(id), crate::ir::Node::WaveBarrier { eq, .. } if *eq == want))
                .count()
        };
        let (n0, n1) = (count(0), count(1));
        assert_eq!(n0, n1, "wave-phase barriers unbalanced (eq=0: {n0}, eq=1: {n1}) — would deadlock the workgroup");
    }
    out
}

// ══════════════════════════════ verify_v2 (scheduling coherence) ══════════════════════════════

/// **Scheduling-coherence verifier** (FA-redesign §2.6) — the teeth of the correctness/scheduling
/// split, run at kernel build alongside `pipeline`'s carry-completeness + wave-phase-balance checks.
/// Three structural checks over the reachable DAG guarantee that **deleting every scheduling hint
/// leaves a still-correct kernel** (so §5 perf work can never change numerics):
/// - **setprio balance**: every `s_setprio(1)` RAISE has a matching reachable `s_setprio(0)` DROP
///   (`prio1 ≤ prio0`) — a wave stuck at raised priority starves its ping-pong partner. A lone DROP is
///   benign (a softmax cluster with no MFMA burst emits only the closing prio0).
/// - **interleave sanity**: per scheduling group, Σ `sched_group_barrier(MFMA).size` ≤ the MFMA count
///   emitted — an over-promise means LLVM silently drops the tail hints (a wrong interleave, not a
///   crash). A necessary whole-kernel bound (cluster tags are not in the IR).
/// - **hint purity**: no value-bearing node (`meta.dtype.is_some()`) consumes a scheduling hint
///   (`SchedGroupBarrier`/`SetPrio`/`SchedFence`) as a child — so removing every hint changes no
///   computed value. (The typed `Builder` already enforces this — `Effect` ≠ `Val`; this is the
///   defense-in-depth structural proof.)
///
/// A build-time panic (a kernel-authoring bug, not recoverable), matching the carry verifiers.
pub(crate) fn verify_v2(ir: &TileIr, roots: &[TileId]) {
    let mut reach: std::collections::HashSet<TileId> = std::collections::HashSet::new();
    for &r in roots {
        reach.extend(crate::passes::reachable(ir, r));
    }
    let is_hint =
        |n: &Node| matches!(n, Node::SchedGroupBarrier { .. } | Node::SetPrio { .. } | Node::SchedFence { .. });

    // (1) setprio balance — every RAISE (prio1) must have a matching DROP (prio0); a lone prio0 is
    //     benign (already at base), a lone prio1 leaves the wave stuck at raised priority.
    let prio = |lvl: i64| {
        reach.iter().filter(|&&id| matches!(ir.node(id), Node::SetPrio { level, .. } if *level == lvl)).count()
    };
    let (p1, p0) = (prio(1), prio(0));
    assert!(
        p1 <= p0,
        "s_setprio unbalanced (prio1: {p1} > prio0: {p0}) — a wave stuck at raised priority starves its partner"
    );

    // (2) interleave sanity: per group, Σ MFMA-mask hint size ≤ MFMA count.
    let n_mma = reach.iter().filter(|&&id| matches!(ir.node(id), Node::Mma { .. })).count();
    let mut promised: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
    for &id in &reach {
        if let Node::SchedGroupBarrier { mask, size, group, .. } = ir.node(id)
            && *mask == Builder::SG_MFMA
        {
            *promised.entry(*group).or_default() += *size;
        }
    }
    for (g, sum) in promised {
        assert!(
            sum as usize <= n_mma,
            "interleave group {g} promises {sum} MFMAs but only {n_mma} emitted — LLVM will drop the tail hints"
        );
    }

    // (3) hint purity: no COMPUTATION consumes a hint as a value operand. `Node::After` is EXEMPT — it
    //     is a pure ordering passthrough (its value == its `val` child regardless of the hints in its
    //     ordering deps), so a hint reachable only through an `After`'s deps changes no computed value.
    //     Routing a hint into a carried value via `After` (the `val_after` liveness idiom) is thus pure.
    for &id in &reach {
        if ir.meta(id).dtype.is_some() && !matches!(ir.node(id), Node::After { .. }) {
            for c in TileIr::children(ir.node(id)) {
                assert!(
                    !is_hint(ir.node(c)),
                    "value node {} consumes scheduling hint {} — violates the correctness/scheduling split",
                    id.0,
                    c.0
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{BF16, Builder, F32};
    use crate::shape::{Mfma32x32x8Bf16 as S, MfmaShape};

    #[test]
    fn gathered_is_move_only() {
        let g = Gathered::<crate::build::BF16> { operands: Vec::new(), edges: vec![TileId(0)] };
        let _ops = g.operands(); // consumes g
        // let _dup = g.operands(); // would not compile: borrow of moved value
    }

    /// A tiny compute cluster authored through the typed scopes — the seal-split + interleave + prio
    /// vocabulary — must pass `verify_v2`. Proves the API composes + the coherence checks accept a
    /// well-formed schedule BEFORE FA rides it (step 4). Two MFMAs (intrinsic) bracketed by set_prio,
    /// an `interleave_valu<2,5>` hint, sealed scheduling-transparently (`seal_ordering`).
    #[test]
    fn typed_scopes_compose_and_verify_v2() {
        let mut b = Builder::new("tk2_scope_probe");
        let c = b.global::<F32>(S::M * S::N);
        let a = b.global::<BF16>(S::M * 2 * S::K);
        let bmat = b.global::<BF16>(S::N * 2 * S::K);
        let _wg = b.grid_axis(0, 1);
        let lane = b.block_axis(64);
        let (a_map, b_map, dist) = (S::a_map(), S::b_map(), S::acc_dist());
        let mut roots = Vec::new();
        compute_cluster(&mut b, |c_scope| {
            let entry = c_scope.f32(0.0).id;
            let prio1 = c_scope.set_prio(1, &[entry]).dep();
            let mut acc = {
                let zs: Vec<_> = (0..S::EPT_C).map(|_| c_scope.f32(0.0)).collect();
                c_scope.vec_build(&zs)
            };
            for ki in 0..2 {
                let af = crate::kernels::load_op_frag(c_scope, a, a_map, 0, ki * S::K, 2 * S::K, lane);
                let bf = crate::kernels::load_op_frag(c_scope, bmat, b_map, 0, ki * S::K, 2 * S::K, lane);
                acc = c_scope.mma_of::<S>(af, bf, acc);
            }
            let hint = c_scope.interleave_valu(2, 5, 1, &[prio1]).expect("pairs>0");
            let n_c = c_scope.idx_const(S::N as i64);
            for i in 0..S::EPT_C {
                let (row, col) = c_scope.acc_rc(dist, lane, i);
                let rn = c_scope.idx_mul(row, n_c);
                let off = c_scope.idx_add(rn, col);
                let v = c_scope.vec_extract(acc, i);
                roots.push(c_scope.store(c, off, v));
            }
            let last = *roots.last().expect("stores");
            let prio0 = c_scope.set_prio(0, &[last.dep()]).dep();
            let seal = c_scope.seal_ordering(last, &[prio0, hint.dep()]);
            roots.push(seal);
        });
        let ids: Vec<TileId> = roots.iter().map(|e| e.dep()).collect();
        verify_v2(&b.ir, &ids); // must accept a balanced, pure schedule
    }

    /// `verify_v2` REJECTS an unbalanced `s_setprio` (a prio-1 with no reachable prio-0) — the wave
    /// would stay at raised priority into the loop-back memory phase, starving its partner.
    #[test]
    #[should_panic(expected = "s_setprio unbalanced")]
    fn verify_v2_rejects_unbalanced_setprio() {
        let mut b = Builder::new("t");
        let s = b.f32(0.0);
        let frag = b.define_frag::<F32>(crate::ir::FragMap::gfx942_16x16(true));
        let z: Vec<_> = (0..4).map(|_| b.f32(0.0)).collect();
        let zvec = b.vec_build(&z);
        let store = b.store_frag_vec(frag, zvec);
        let p1 = b.set_prio(1, &[s.id]); // prio-1, no matching prio-0
        let root = b.combine(store, &[p1.dep()]);
        verify_v2(&b.ir, &[root.dep()]);
    }
}
