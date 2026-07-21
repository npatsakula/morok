//! **Scheduling-coherence verification** ([`verify_v2`]) — the teeth of the FA-redesign §2.2
//! correctness/scheduling split. Three structural checks over the reachable tile-IR DAG guarantee
//! that deleting every scheduling hint (`s_setprio` / `sched_group_barrier` / `sched_barrier`)
//! leaves a still-correct kernel, so perf tuning can never change numerics. Run at kernel build —
//! the FA-32 kernel (`kernels/fa.rs`) rides it.

use crate::build::Builder;
use crate::ir::{Node, TileId, TileIr};

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
    use proptest::prelude::*;

    use super::*;
    use crate::build::{Builder, Edge, Effect, F32};
    use crate::ir::FragMap;

    /// `verify_v2` REJECTS an unbalanced `s_setprio` (a prio-1 with no reachable prio-0) — the wave
    /// would stay at raised priority into the loop-back memory phase, starving its partner.
    #[test]
    #[should_panic(expected = "s_setprio unbalanced")]
    fn verify_v2_rejects_unbalanced_setprio() {
        let mut b = Builder::new("t");
        let s = b.f32(0.0);
        let frag = b.define_frag::<F32>(FragMap::gfx942_16x16(true));
        let z: Vec<_> = (0..4).map(|_| b.f32(0.0)).collect();
        let zvec = b.vec_build(&z);
        let store = b.store_frag_vec(frag, zvec);
        let p1 = b.set_prio(1, &[Edge::anchor(s.id)]); // prio-1, no matching prio-0
        let root = b.combine(store, &[p1.dep()]);
        verify_v2(&b.ir, &[root.dep().raw()]);
    }

    proptest! {
        /// **Edge-threading invariant of the typed `Edge` conversion:** every ordering op must intern ALL
        /// the edges it is handed as node children — none silently dropped. Each op is fed the SAME set of
        /// `n_src` DISTINCT source effects as its ordering input; the constructed node's children must then
        /// contain every source id. This directly exercises the drop-a-token failure mode the `&[Edge]`
        /// conversion could regress (a truncated slice, a missed `.raw()`) — which plain reachability
        /// cannot see, since a hint folded into the sink stays reachable even if it dropped its own inputs.
        #[test]
        fn ordering_ops_thread_every_input_edge(n_src in 2usize..6) {
            let mut b = Builder::new("edge_thread_probe");
            let frag = b.define_frag::<F32>(FragMap::gfx942_16x16(true));
            // `n_src` DISTINCT stores (distinct payload values ⇒ distinct interned effects); their dep-edges
            // are the ordering input every op below must thread into its node.
            let sources: Vec<Effect> = (0..n_src)
                .map(|i| {
                    let vs: Vec<_> = (0..4).map(|j| b.f32((i * 4 + j) as f32)).collect();
                    let v = b.vec_build(&vs);
                    b.store_frag_vec(frag, v)
                })
                .collect();
            let edges: Vec<Edge> = sources.iter().map(|e| e.dep()).collect();
            let src_ids: Vec<TileId> = edges.iter().map(|e| e.raw()).collect();
            // A body/val distinct from every source (for the ops whose first arg is separate from the deps).
            let bv: Vec<_> = (0..4).map(|j| b.f32((1000 + j) as f32)).collect();
            let bvec = b.vec_build(&bv);
            let body = b.store_frag_vec(frag, bvec);
            let wr = b.block_axis(64);

            // One node per ordering-op kind, each fed `edges` as its ordering input.
            let nodes: Vec<(&str, Edge)> = vec![
                ("barrier", b.barrier(body, &edges).dep()),
                ("bare_barrier", b.bare_barrier(body, &edges).dep()),
                ("sched_fence", b.sched_fence(0, &edges).dep()),
                ("sched_group", b.sched_group(Builder::SG_VALU, 1, 0, &edges).dep()),
                ("set_prio", b.set_prio(1, &edges).dep()),
                ("wave_barrier", b.wave_barrier(wr, 1, &edges).dep()),
                ("combine", b.combine(body, &edges).dep()),
            ];
            for (name, node) in &nodes {
                let children: std::collections::HashSet<TileId> =
                    TileIr::children(b.ir.node(node.raw())).into_iter().collect();
                for id in &src_ids {
                    prop_assert!(children.contains(id), "{name} dropped input edge {} from its node children", id.0);
                }
            }
        }
    }
}
