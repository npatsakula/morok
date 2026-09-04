use super::*;
use crate::{BinaryOp, Op};
use test_case::test_case;

/// `len` distinct variables summed left-to-right: a graph deep enough that a
/// recursive walk would overflow the real stack.
fn deep_chain(len: usize) -> (Vec<Arc<UOp>>, Arc<UOp>) {
    let vars: Vec<Arc<UOp>> = (0..len).map(|i| UOp::define_var(format!("v{i}"), 1, 1024)).collect();
    let root = vars.iter().skip(1).fold(vars[0].clone(), |acc, v| acc.try_add(v).expect("index add"));
    (vars, root)
}

/// `((x + y) * (x + y) + (x + y)) + (y + 1)`: the inner ADD is shared three
/// times and once hidden behind a MUL, the outer-right ADD is reachable only
/// through gate-passing nodes.
fn shared_dag() -> Arc<UOp> {
    let (x, y) = (UOp::index_const(2), UOp::index_const(3));
    let inner = x.try_add(&y).unwrap();
    let product = inner.try_mul(&inner).unwrap();
    product.try_add(&inner).unwrap().try_add(&y.try_add(&UOp::index_const(1)).unwrap()).unwrap()
}

fn is_add(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Binary(BinaryOp::Add, ..))
}

fn not_mul(uop: &Arc<UOp>) -> bool {
    !matches!(uop.op(), Op::Binary(BinaryOp::Mul, ..))
}

fn not_add(uop: &Arc<UOp>) -> bool {
    !is_add(uop)
}

fn ids(nodes: &[Arc<UOp>]) -> Vec<u64> {
    let mut ids: Vec<u64> = nodes.iter().map(|uop| uop.id).collect();
    ids.sort_unstable();
    ids
}

#[test]
fn reaching_covers_self_and_ancestors() {
    let x = UOp::define_var("x".to_string(), 1, 16);
    let y = UOp::define_var("y".to_string(), 1, 16);
    let sum = x.try_add(&y).unwrap();

    assert!(reaching(&x).contains(&x));
    assert!(reaching(&x).contains(&sum));
    assert!(reaching(&y).contains(&sum));
    assert!(!reaching(&sum).contains(&x));
}

#[test]
fn memo_is_reusable_across_roots() {
    // Diamond: both sides share `shared`, so the memo answers the second root
    // from entries filled by the first.
    let shared = UOp::define_var("shared".to_string(), 1, 16);
    let other = UOp::define_var("other".to_string(), 1, 16);
    let left = shared.try_add(&UOp::index_const(1)).unwrap();
    let right = shared.try_mul(&UOp::index_const(3)).unwrap();
    let diamond = left.try_add(&right).unwrap();

    let mut memo = reaching(&shared);
    for root in [&left, &right, &diamond, &other] {
        assert_eq!(memo.contains(root), root.any_in_subtree(|n| n.id == shared.id), "root {}", root.id);
    }
}

#[test_case(0; "first")]
#[test_case(500; "middle")]
#[test_case(999; "last")]
fn deep_chain_finds_every_leaf(idx: usize) {
    let (vars, root) = deep_chain(1000);
    assert!(root.node_count() > 1000, "chain collapsed: {}", root.node_count());
    assert!(reaching(&vars[idx]).contains(&root));
    assert_eq!(ids(&reaching_each(&[vars[idx].clone()]).get(&root)), vec![vars[idx].id]);
}

#[test]
fn deep_chain_rejects_foreign_node() {
    let (_, root) = deep_chain(1000);
    let outsider = UOp::define_var("outsider".to_string(), 1, 16);
    assert!(!reaching(&outsider).contains(&root));
    assert!(reaching_each(&[outsider]).get(&root).is_empty());
}

#[test]
fn multi_target_predicate_matches_any_in_subtree() {
    let x = UOp::define_var("x".to_string(), 1, 16);
    let y = UOp::define_var("y".to_string(), 1, 16);
    let z = UOp::define_var("z".to_string(), 1, 16);
    let xy = x.try_add(&y).unwrap();
    let targets: std::collections::HashSet<u64> = [x.id, z.id].into_iter().collect();

    let mut memo = SubtreeMemo::new(|n: &Arc<UOp>| targets.contains(&n.id));
    for root in [&x, &y, &z, &xy] {
        assert_eq!(memo.contains(root), root.any_in_subtree(|n| targets.contains(&n.id)), "root {}", root.id);
    }
}

/// Both entry types must agree with a gated toposort on every node of a shared
/// DAG; a gated node is excluded even when it is selected itself (`not_add`).
#[test_case(None; "ungated")]
#[test_case(Some(not_mul as Pred); "gated on MUL")]
#[test_case(Some(not_add as Pred); "gated on ADD")]
fn matches_gated_toposort_on_every_node(gate: Option<Pred>) {
    let root = shared_dag();
    let gate = gate.unwrap_or(|_| true);
    let mut nodes: SliceMemo<Nodes> = SliceMemo::gated(is_add, gate);
    let mut any: SubtreeMemo = SliceMemo::gated(is_add, gate);
    for node in root.toposort() {
        let expected: Vec<Arc<UOp>> = node.toposort_filtered(gate).into_iter().filter(is_add).collect();
        assert_eq!(ids(&nodes.get(&node)), ids(&expected), "slice of {}", node.tree());
        assert_eq!(any.contains(&node), !expected.is_empty(), "membership of {}", node.tree());
    }
}

#[test]
fn memoized_children_feed_the_parent_query() {
    let root = shared_dag();
    let mut fresh: SliceMemo<Nodes> = SliceMemo::new(is_add);
    let mut warmed: SliceMemo<Nodes> = SliceMemo::new(is_add);
    for child in root.op().children() {
        warmed.get(child);
    }
    assert_eq!(ids(&warmed.get(&root)), ids(&fresh.get(&root)));
}

#[test]
fn reaching_each_lists_exactly_the_reached_targets() {
    let root = shared_dag();
    let targets: Vec<Arc<UOp>> = root.toposort().into_iter().filter(|n| n.op().children().is_empty()).collect();
    let mut memo = reaching_each(&targets);
    for node in root.toposort() {
        let expected: Vec<Arc<UOp>> = targets.iter().filter(|t| reaching(t).contains(&node)).cloned().collect();
        assert_eq!(ids(&memo.get(&node)), ids(&expected), "targets below {}", node.tree());
    }
}
