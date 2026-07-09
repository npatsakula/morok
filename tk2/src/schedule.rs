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

use crate::build::{Builder, Effect, F32, Frag, Idx, Val};
use crate::ir::TileId;

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

// ══════════════════════════════ cluster scopes (typestate) ══════════════════════════════
//
// A cluster is one of two KINDS, and the kind is a TYPE, not a runtime flag. `MemScope` (loads /
// stores / gathers) exposes the memory-side scheduling primitives — the load-pin `fence` and the
// `drain_lgkm`; `ComputeScope` (the MFMA grid) exposes ONLY the `set_prio` bracket. So `set_prio`
// on a memory cluster or a load-pin on a compute cluster is a *compile* error, not a convention —
// the exact "scheduling can attach anywhere" hole in the untyped `crate::pipeline`. Both deref to
// `&mut Builder` so the kernel-specific movement handles (`LdsView`/`LdsStage`) work unchanged;
// only the SCHEDULING surface is gated by kind.

/// A **memory cluster scope** — loads/stores/gathers. Exposes the memory-side scheduling
/// primitives (`fence` load-pin, `drain_lgkm`) and the cluster `seal`; NOT `set_prio` (that is
/// compute-only). Derefs to `&mut Builder` for the movement handles.
pub struct MemScope<'b> {
    b: &'b mut Builder,
}

/// A **compute cluster scope** — the MFMA grid. Exposes ONLY the `set_prio` bracket + the `seal`;
/// no load-pin or drain (those are memory-side). Derefs to `&mut Builder` for `mma`/frag ops.
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
        impl<'b> $scope<'b> {
            /// **Seal the cluster** with the workgroup `s_barrier` boundary. `body` is the last op
            /// before the barrier; `deps` are additional edges. Returns the barrier effect.
            pub fn seal(&mut self, body: Effect, deps: &[TileId]) -> Effect {
                self.b.barrier(body, deps)
            }
        }
    };
}
deref_to_builder!(MemScope);
deref_to_builder!(ComputeScope);

impl<'b> MemScope<'b> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gathered_is_move_only() {
        let g = Gathered::<crate::build::BF16> { operands: Vec::new(), edges: vec![TileId(0)] };
        let _ops = g.operands(); // consumes g
        // let _dup = g.operands(); // would not compile: borrow of moved value
    }
}
