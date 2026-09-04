use super::*;
use test_case::test_case;

/// `len` distinct variables summed left-to-right: a graph deep enough that a
/// recursive walk would overflow the real stack.
fn deep_chain(len: usize) -> (Vec<Arc<UOp>>, Arc<UOp>) {
    let vars: Vec<Arc<UOp>> = (0..len).map(|i| UOp::define_var(format!("v{i}"), 1, 1024)).collect();
    let root = vars.iter().skip(1).fold(vars[0].clone(), |acc, v| acc.try_add(v).expect("index add"));
    (vars, root)
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
}

#[test]
fn deep_chain_rejects_foreign_node() {
    let (_, root) = deep_chain(1000);
    let outsider = UOp::define_var("outsider".to_string(), 1, 16);
    assert!(!reaching(&outsider).contains(&root));
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
