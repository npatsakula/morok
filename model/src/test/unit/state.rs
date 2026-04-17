use morok_tensor::Tensor;

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
