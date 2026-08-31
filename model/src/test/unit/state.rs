use svod_tensor::Tensor;

#[test]
fn test_safetensors_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.safetensors");

    // Create and realize tensors
    let mut w = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape([2, 2]).unwrap();
    w.realize().unwrap();
    let mut b = Tensor::from_slice([0.5f32, -0.5]);
    b.realize().unwrap();

    let w_data = w.as_vec::<f32>().unwrap();
    let b_data = b.as_vec::<f32>().unwrap();

    // Build TensorView map — data must outlive the views
    let w_bytes: &[u8] = bytemuck::cast_slice(&w_data);
    let b_bytes: &[u8] = bytemuck::cast_slice(&b_data);

    let tensors = std::collections::HashMap::from([
        (
            "weight".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], w_bytes).unwrap(),
        ),
        ("bias".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], b_bytes).unwrap()),
    ]);

    safetensors::serialize_to_file(&tensors, None::<std::collections::HashMap<String, String>>, &path).unwrap();

    // Load back
    let loaded = crate::state::load_safetensors(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    assert!(loaded.contains_key("weight"));
    assert!(loaded.contains_key("bias"));

    let mut loaded_w = loaded["weight"].clone();
    loaded_w.realize().unwrap();
    let loaded_vals = loaded_w.as_vec::<f32>().unwrap();
    assert_eq!(loaded_vals, vec![1.0, 2.0, 3.0, 4.0]);

    let mut loaded_b = loaded["bias"].clone();
    loaded_b.realize().unwrap();
    let loaded_bvals = loaded_b.as_vec::<f32>().unwrap();
    assert_eq!(loaded_bvals, vec![0.5, -0.5]);
}

#[test]
fn test_load_safetensors_fp8() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fp8.safetensors");
    let data = [0x00u8, 0x38, 0x40, 0xb8];
    let tensors = std::collections::HashMap::from([(
        "weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F8_E4M3, vec![2, 2], &data).unwrap(),
    )]);
    safetensors::serialize_to_file(&tensors, None::<std::collections::HashMap<String, String>>, &path).unwrap();

    let loaded = crate::state::load_safetensors(&path).unwrap();
    assert_eq!(loaded["weight"].uop().dtype(), svod_dtype::DType::FP8E4M3);
}

/// Phase-3 acceptance: loading one checkpoint into two model instances
/// uploads each weight once — the tensors share ONE sealed device storage,
/// writes into it fail loudly, and computation through it stays correct.
#[test]
fn loading_same_checkpoint_twice_shares_immutable_storage() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared.safetensors");
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tensors = std::collections::HashMap::from([(
        "w".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4], bytes).unwrap(),
    )]);
    safetensors::serialize_to_file(&tensors, None::<std::collections::HashMap<String, String>>, &path).unwrap();

    let sd1 = crate::state::load_safetensors(&path).unwrap();
    let sd2 = crate::state::load_safetensors(&path).unwrap();
    let b1 = sd1["w"].buffer().unwrap();
    let b2 = sd2["w"].buffer().unwrap();
    assert_eq!(b1.storage_id(), b2.storage_id(), "one checkpoint must upload each weight once");
    assert!(b1.is_immutable());
    let mut writable = b1.clone();
    assert!(writable.copyin(&[0u8; 16]).is_err(), "shared weights must refuse host writes");

    // Both instances still evaluate correctly through the shared storage.
    let mut sum = &sd1["w"] + &sd2["w"];
    sum.realize().unwrap();
    assert_eq!(sum.as_vec::<f32>().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
}
