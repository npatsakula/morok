//! `SliceMemo` must agree with a gated toposort on every node of a shared DAG.

use std::sync::Arc;

use svod_ir::{BinaryOp, Op, UOp};
use test_case::test_case;

use crate::passes::slice_memo::SliceMemo;

fn is_add(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Binary(BinaryOp::Add, ..))
}

fn not_mul(uop: &Arc<UOp>) -> bool {
    !matches!(uop.op(), Op::Binary(BinaryOp::Mul, ..))
}

fn ids(nodes: &[Arc<UOp>]) -> Vec<u64> {
    let mut ids: Vec<u64> = nodes.iter().map(|uop| uop.id).collect();
    ids.sort_unstable();
    ids
}

/// `((x + y) * (x + y) + (x + y)) + (y + 1)`: the inner ADD is shared three
/// times and once hidden behind a MUL, the outer-right ADD is reachable only
/// through gate-passing nodes.
fn shared_dag() -> Arc<UOp> {
    let (x, y) = (UOp::index_const(2), UOp::index_const(3));
    let inner = x.add(&y);
    inner.mul(&inner).add(&inner).add(&y.add(&UOp::index_const(1)))
}

#[test_case(true ; "gated on MUL")]
#[test_case(false ; "ungated")]
fn matches_gated_toposort_on_every_node(gated: bool) {
    let root = shared_dag();
    let mut memo = if gated { SliceMemo::new(is_add, not_mul) } else { SliceMemo::ungated(is_add) };
    for node in root.toposort() {
        let expected: Vec<Arc<UOp>> = if gated {
            node.toposort_filtered(not_mul).into_iter().filter(is_add).collect()
        } else {
            node.toposort().into_iter().filter(is_add).collect()
        };
        assert_eq!(ids(&memo.get(&node)), ids(&expected), "slice of {}", node.tree());
    }
}

#[test]
fn memoized_children_feed_the_parent_query() {
    let root = shared_dag();
    let mut fresh = SliceMemo::ungated(is_add);
    let mut warmed = SliceMemo::ungated(is_add);
    for child in root.op().children() {
        warmed.get(child);
    }
    assert_eq!(ids(&warmed.get(&root)), ids(&fresh.get(&root)));
}

#[test]
fn gated_root_is_empty() {
    let product = UOp::index_const(2).add(&UOp::index_const(3)).mul(&UOp::index_const(4));
    assert!(SliceMemo::new(is_add, not_mul).get(&product).is_empty());
}
