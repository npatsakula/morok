use super::*;
use morok_dtype::DType;
use morok_ir::ConstValue;

#[test]
fn test_register_and_get() {
    crate::test::helpers::test_setup();

    let uop = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let entry = register_tensor(uop.clone());

    let retrieved = get_tensor(entry.id).expect("Should find tensor");
    assert_eq!(retrieved.id, entry.id);
    assert!(Arc::ptr_eq(&*retrieved.uop.read(), &uop));
}

#[test]
fn test_apply_map_updates_tensors() {
    crate::test::helpers::test_setup();

    // Create two tensors sharing a common UOp
    let shared = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let t1_uop = shared.neg();
    let t2_uop = shared.neg(); // Same as t1_uop due to hash consing

    let t1 = register_tensor(t1_uop.clone());
    let t2 = register_tensor(t2_uop.clone());

    // Create a replacement for the shared const
    let replacement = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    #[allow(clippy::mutable_key_type)]
    let mut becomes_map = HashMap::new();
    becomes_map.insert(UOpKey(shared.clone()), replacement.clone());

    // Apply the map
    apply_map_to_tensors(&becomes_map);

    // Both tensors should now reference the replacement
    let t1_new = t1.uop.read();
    let t2_new = t2.uop.read();

    // The root NEG should now have the replacement as its source
    assert!(!Arc::ptr_eq(&*t1_new, &t1_uop), "t1 should be updated");
    assert!(!Arc::ptr_eq(&*t2_new, &t2_uop), "t2 should be updated");
}
