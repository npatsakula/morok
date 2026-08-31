use super::*;
use svod_dtype::DType;
use svod_ir::ConstValue;

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

/// Regression: two threads realizing hash-cons-identical `zeros` graphs must
/// keep distinct buffer AND graph identities. Before the Phase-4 CAS in
/// `apply_map_to_tensors_inner`, one realize's broadcast could clobber the
/// other tensor's just-finalized entry, converging both onto one buffer UOp
/// (the model-JIT parallel-test flake).
#[test]
fn concurrent_zeros_realizes_keep_distinct_identities() {
    crate::test::helpers::test_setup();
    let cfg = crate::PrepareConfig::default();
    // Warm the schedule cache so both threads race through apply_map together.
    {
        let mut warm = crate::Tensor::zeros(&[3], DType::Float32).unwrap();
        warm.realize_with(&cfg).unwrap();
    }
    for _ in 0..100 {
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let results = [(); 2]
            .map(|()| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let cfg = crate::PrepareConfig::default();
                    let mut t = crate::Tensor::zeros(&[3], DType::Float32).unwrap();
                    barrier.wait();
                    t.realize_with(&cfg).unwrap();
                    (t.buffer().unwrap().id().0, t.uop().base().id)
                })
            })
            .map(|handle| handle.join().unwrap());
        assert_ne!(results[0].0, results[1].0, "device buffers must differ");
        assert_ne!(results[0].1, results[1].1, "graph identities must not converge (apply_map clobber)");
    }
}

/// Buffer registry entries must expire automatically when their BUFFER UOp is
/// dropped (the `set_uop_drop_hook` path) — no manual release required.
#[test]
fn buffer_entry_expires_when_uop_drops() {
    crate::test::helpers::test_setup();
    let uop = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32);
    let id = uop.id;
    let buffer = svod_device::Buffer::new(
        svod_device::registry::cpu().expect("cpu allocator"),
        DType::Float32,
        vec![4],
        Default::default(),
    );
    register_buffer_by_uop_id(id, Arc::new(buffer));
    assert!(get_buffer(id).is_some());

    drop(uop);
    assert!(get_buffer(id).is_none(), "entry must expire with its UOp");
}
