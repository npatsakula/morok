use morok_ir::{ConstValue, DType, UOp};

#[test]
fn test_tree_simple() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    let sum = a.try_add(&b).unwrap();

    let tree = sum.tree();
    println!("Tree output:\n{}", tree);
    assert!(tree.contains("Add"));
    assert!(tree.contains("CONST"));
}

#[test]
fn test_tree_shared_nodes() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let shared = a.try_add(&a).unwrap();

    // Compact tree should show back-reference
    let compact = shared.tree();
    println!("Compact tree:\n{}", compact);
    assert!(compact.contains("see above"));

    // Full tree should NOT show back-reference
    let full = shared.tree_full();
    println!("Full tree:\n{}", full);
    assert!(!full.contains("see above"));
}
