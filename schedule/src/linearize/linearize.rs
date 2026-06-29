//! Direct port of tinygrad's linearizer (codegen/late/linearizer.py).
//!
//! Converts a UOp DAG into a linear instruction sequence using:
//! 1. Priority + tuplize-based "ideal order" sort
//! 2. Heap toposort respecting data dependencies

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

use svod_ir::UOp;
use svod_ir::op::Op;
use svod_ir::types::ConstValue;

/// Op discriminant matching tinygrad's `Ops` enum values.
/// Used as first element of the recursive `tuplize` key.
fn op_value(op: &Op) -> u32 {
    match op {
        Op::Special { .. } => 2,
        Op::DefineLocal(_) | Op::Buffer { .. } | Op::BufferView { .. } => 3,
        Op::DefineReg { .. } => 4,
        Op::Param { .. } => 6,
        Op::DefineVar { .. } => 7,
        Op::Function { .. } => 7,
        Op::Call { .. } => 8,
        Op::Program { .. } => 9,
        Op::Source { .. } => 11,
        Op::Sink { .. } => 13,
        Op::After { .. } => 14,
        Op::Group { .. } => 15,
        Op::Gep { .. } => 16,
        Op::Vectorize { .. } => 17,
        Op::Tuple { .. } => 18,
        Op::GetTuple { .. } => 19,
        Op::PointerIndex { .. } => 20,
        Op::Index { .. } => 21,
        Op::Shrink { .. } => 22,
        Op::Load { .. } => 23,
        Op::Store { .. } => 24,
        Op::Wmma { .. } => 25,
        Op::Cast { .. } => 27,
        Op::BitCast { .. } => 28,
        Op::Unary(_, _) => 34,
        Op::Binary(_, _, _) => 36,
        Op::Ternary(_, _, _, _) => 39,
        Op::Barrier { .. } => 57,
        Op::Range { .. } => 58,
        Op::If { .. } => 59,
        Op::End { .. } => 60,
        Op::EndIf { .. } => 61,
        Op::Const(_) | Op::VConst { .. } => 63,
        Op::Custom { .. } => 64,
        Op::CustomI { .. } => 65,
        Op::Unique(_) => 67,
        Op::Device(_) => 68,
        Op::LUnique(_) => 69,
        Op::Contiguous { .. } => 70,
        Op::ContiguousBackward { .. } => 71,
        Op::Detach { .. } => 72,
        Op::Bufferize { .. } => 73,
        Op::Copy { .. } => 74,
        Op::MSelect { .. } => 76,
        Op::MStack { .. } => 77,
        Op::CustomFunction { .. } => 78,
        Op::Reshape { .. } => 79,
        Op::Permute { .. } => 80,
        Op::Expand { .. } => 81,
        Op::Pad { .. } => 82,
        Op::Flip { .. } => 83,
        Op::Multi { .. } => 84,
        Op::ReduceAxis { .. } | Op::Reduce { .. } => 85,
        Op::AllReduce { .. } => 86,
        Op::Unroll { .. } => 87,
        Op::Contract { .. } => 88,
        Op::Cat { .. } => 89,
        Op::PtrCat { .. } => 90,
        _ => 50,
    }
}

/// Compute recursive structural sort keys (tinygrad's `tuplize`).
///
/// Key = `[op_value, src_count, src0_key..., src1_key..., ...]`
///
/// Computed bottom-up from toposort (children before parents).
/// Lexicographic `Vec<u32>` comparison matches Python's tuple comparison.
///
/// Keys are bounded to `MAX_KEY_LEN` elements to prevent exponential blowup
/// on large DAGs (shared children expand recursively). The bound is generous
/// enough that semantic ordering is preserved for all practical graph depths.
fn compute_tuplize(nodes: &[Arc<UOp>]) -> HashMap<u64, Vec<u32>> {
    const MAX_KEY_LEN: usize = 128;
    let mut keys: HashMap<u64, Vec<u32>> = HashMap::with_capacity(nodes.len());
    for node in nodes {
        let srcs = node.op().sources();
        let mut key = Vec::with_capacity(16);
        key.push(op_value(node.op()));
        key.push(srcs.len() as u32);
        for src in &srcs {
            if key.len() >= MAX_KEY_LEN {
                break;
            }
            if let Some(ck) = keys.get(&src.id) {
                let remaining = MAX_KEY_LEN - key.len();
                key.extend_from_slice(&ck[..ck.len().min(remaining)]);
            }
        }
        keys.insert(node.id, key);
    }
    keys
}

/// Compute run_count: `prod(int(r.vmax)+1 for r in u.ranges)`.
///
/// Mirrors tinygrad's `run_count = prod([int(r.vmax)+1 for r in u.ranges])`,
/// applied uniformly to every op. [`InScopeRangesProperty`] is the faithful
/// port of tinygrad's `u.ranges` (`_ranges`): it merges the sources' in-scope
/// ranges and pops the op's `ended_ranges()`. For an AFTER that pop already
/// drops the ranges its deps close (e.g. a post-loop `acc.after(end_R)` yields
/// the empty set → run_count 1), so AFTER needs no special handling — the
/// generic path places it outside the loop, not nested inside it.
///
/// [`InScopeRangesProperty`]: svod_ir::uop::properties::InScopeRangesProperty
fn run_count(uop: &Arc<UOp>) -> u64 {
    use svod_ir::uop::cached_property::CachedProperty;
    use svod_ir::uop::properties::InScopeRangesProperty;

    #[allow(clippy::mutable_key_type)]
    let in_scope = InScopeRangesProperty::get(uop);

    // Saturating: a range with an unbounded/symbolic `vmax` (e.g. a data-dependent
    // dynamic-loop bound with no sound range) reports `i64::MAX`, whose trip product
    // overflows `u64` when nested. `run_count` is only a "place deepest in loops" sort
    // key, so saturating to `u64::MAX` is correct; for every kernel whose trips fit in
    // `u64` it is identical to the old `.product()`, and it fixes the prior wrap (debug
    // panicked; release silently mis-ordered).
    in_scope
        .iter()
        .map(|key| match key.0.vmax() {
            ConstValue::Int(v) if *v >= 0 => (*v as u64).saturating_add(1),
            ConstValue::UInt(v) => (*v).saturating_add(1),
            _ => 1,
        })
        .fold(1u64, |acc, n| acc.saturating_mul(n))
}

/// Priority assignment matching tinygrad's linearizer.py:24-32.
/// Returns `(priority, extra)` where extra is `Some(slot)` for PARAM.
fn priority(uop: &Arc<UOp>) -> (i32, Option<i64>) {
    match uop.op() {
        Op::Param { slot, device: None, .. } => (-20, Some(*slot as i64)),
        Op::DefineLocal(_) => (-18, None),
        Op::DefineReg { .. } => (-17, None),
        Op::Load { .. } => (-1, None),
        Op::Store { .. } => (1, None),
        Op::Range { .. } => (5, None),
        Op::End { .. } => (-5, None),
        _ => (0, None),
    }
}

/// Direct port of tinygrad's `linearize()` (linearizer.py:8-51).
pub fn linearize(sink: Arc<UOp>) -> Vec<Arc<UOp>> {
    let lst = sink.toposort();
    if lst.is_empty() {
        return vec![sink];
    }

    // Compute out_degree and priorities.
    let mut out_degree: HashMap<u64, usize> = HashMap::new();
    let mut priorities: HashMap<u64, (u64, i32, Option<i64>)> = HashMap::new();

    for u in &lst {
        for s in u.op().sources() {
            *out_degree.entry(s.id).or_default() += 1;
        }
    }
    for u in &lst {
        let rc = run_count(u);
        let (p, extra) = priority(u);
        priorities.insert(u.id, (rc, p, extra));
    }

    // Compute tuplize keys (bottom-up).
    let tuplize = compute_tuplize(&lst);

    // Sort all nodes by (run_count, priority, extra, tuplize) — the "ideal order".
    // Assign sequential nkey based on sorted position.
    let mut sorted: Vec<u64> = lst.iter().map(|u| u.id).collect();
    sorted.sort_by(|&a, &b| {
        let pa = &priorities[&a];
        let pb = &priorities[&b];
        pa.cmp(pb).then_with(|| tuplize[&a].cmp(&tuplize[&b]))
    });

    let nkey: HashMap<u64, usize> = sorted.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Heap toposort: pop highest nkey first (max-heap), reverse at end.
    let id_map: HashMap<u64, Arc<UOp>> = lst.iter().map(|u| (u.id, u.clone())).collect();

    let mut heap: BinaryHeap<(usize, u64)> = BinaryHeap::new();
    heap.push((nkey[&sink.id], sink.id));

    let mut newlst: Vec<Arc<UOp>> = Vec::with_capacity(lst.len());
    let mut visited: HashSet<u64> = HashSet::new();

    while let Some((_, uid)) = heap.pop() {
        if !visited.insert(uid) {
            continue;
        }
        let u = &id_map[&uid];
        newlst.push(u.clone());

        for v in u.op().sources() {
            let deg = out_degree.entry(v.id).or_default();
            *deg = deg.saturating_sub(1);
            if *deg == 0 && !visited.contains(&v.id) {
                heap.push((nkey[&v.id], v.id));
            }
        }
    }

    newlst.reverse();

    newlst
}

#[cfg(test)]
#[path = "../test/unit/linearize/linearize_internal.rs"]
mod tests;
