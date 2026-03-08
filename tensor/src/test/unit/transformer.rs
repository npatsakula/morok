//! Tests for transformer building blocks: embedding, attention, rotary embeddings, rms_norm.

use crate::Tensor;

// =========================================================================
// RMS Norm tests
// =========================================================================

#[test]
fn test_rms_norm_basic() {
    // rms_norm(x) = x * rsqrt(mean(x^2) + eps)
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[1, 4]).unwrap();
    let result = x.rms_norm(-1, 1e-5).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 4]);

    // Manual: mean([1,4,9,16]) = 7.5, rsqrt(7.5 + 1e-5) ≈ 0.36514837
    let rms_inv = 1.0 / (7.5f32 + 1e-5).sqrt();
    for i in 0..4 {
        let expected = (i + 1) as f32 * rms_inv;
        assert!((arr[[0, i]] - expected).abs() < 1e-4, "rms_norm[{i}]: got {}, expected {}", arr[[0, i]], expected);
    }
}

#[test]
fn test_rms_norm_axis() {
    // (2, 3), normalize over last axis
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
    let result = x.rms_norm(-1, 1e-5).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[2, 3]);

    // Row 0: mean([1,4,9]) = 14/3, rsqrt(14/3 + 1e-5) ≈ 0.4629
    let rms0 = 1.0 / (14.0f32 / 3.0 + 1e-5).sqrt();
    assert!((arr[[0, 0]] - 1.0 * rms0).abs() < 1e-4);
    assert!((arr[[0, 1]] - 2.0 * rms0).abs() < 1e-4);

    // Row 1: mean([16,25,36]) = 77/3, rsqrt(77/3 + 1e-5)
    let rms1 = 1.0 / (77.0f32 / 3.0 + 1e-5).sqrt();
    assert!((arr[[1, 0]] - 4.0 * rms1).abs() < 1e-4);
}

// =========================================================================
// Embedding tests
// =========================================================================

#[test]
fn test_embedding_basic() {
    // Weight: [3, 4] (3 vocab, 4 embed_dim)
    let weight_data: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let weight = Tensor::from_slice(&weight_data).try_reshape(&[3, 4]).unwrap();
    // Indices: [2, 0] -> should return rows 2 and 0
    let indices = Tensor::from_slice([2i32, 0]);
    let result = weight.embedding(&indices).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[2, 4]);
    // Row 0 = weight[2] = [8, 9, 10, 11]
    assert_eq!(arr[[0, 0]], 8.0);
    assert_eq!(arr[[0, 3]], 11.0);
    // Row 1 = weight[0] = [0, 1, 2, 3]
    assert_eq!(arr[[1, 0]], 0.0);
    assert_eq!(arr[[1, 3]], 3.0);
}

#[test]
fn test_embedding_2d_indices() {
    // Weight: [4, 2] (4 vocab, 2 embed_dim)
    let weight = Tensor::from_slice([0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).try_reshape(&[4, 2]).unwrap();
    // Indices: [2, 3] (batch=2, seq=3)
    let indices = Tensor::from_slice([0i32, 1, 2, 3, 2, 1]).try_reshape(&[2, 3]).unwrap();
    let result = weight.embedding(&indices).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[2, 3, 2]);
    // [0,0] = weight[0] = [0, 1]
    assert_eq!(arr[[0, 0, 0]], 0.0);
    assert_eq!(arr[[0, 0, 1]], 1.0);
    // [0,2] = weight[2] = [4, 5]
    assert_eq!(arr[[0, 2, 0]], 4.0);
    // [1,0] = weight[3] = [6, 7]
    assert_eq!(arr[[1, 0, 0]], 6.0);
    assert_eq!(arr[[1, 0, 1]], 7.0);
}

// =========================================================================
// Scaled Dot-Product Attention tests
// =========================================================================

#[test]
fn test_sdpa_basic() {
    // Q, K, V: [1, 1, 2, 2] (batch=1, head=1, seq=2, dim=2)
    let q = Tensor::from_slice([1.0f32, 0.0, 0.0, 1.0]).try_reshape(&[1, 1, 2, 2]).unwrap();
    let k = q.clone();
    let v = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[1, 1, 2, 2]).unwrap();

    let result = q.scaled_dot_product_attention().key(&k).value(&v).call().unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 1, 2, 2]);
    // With identity-like Q=K, attention should weight both rows
}

#[test]
fn test_sdpa_causal() {
    // Q, K, V: [1, 1, 3, 2] — verify causal masking zeros upper triangle
    let q = Tensor::from_slice([1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]).try_reshape(&[1, 1, 3, 2]).unwrap();
    let k = q.clone();
    let v = Tensor::from_slice([1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0]).try_reshape(&[1, 1, 3, 2]).unwrap();

    let result = q.scaled_dot_product_attention().key(&k).value(&v).is_causal(true).call().unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 1, 3, 2]);
    // Position 0 can only attend to position 0 -> output[0] = V[0] = [1, 0]
    assert!((arr[[0, 0, 0, 0]] - 1.0).abs() < 1e-4);
    assert!((arr[[0, 0, 0, 1]] - 0.0).abs() < 1e-4);
}

#[test]
fn test_sdpa_softcap() {
    // Verify softcap bounds the attention scores
    let q = Tensor::from_slice([10.0f32, 0.0, 0.0, 10.0]).try_reshape(&[1, 1, 2, 2]).unwrap();
    let k = q.clone();
    let v = Tensor::from_slice([1.0f32, 0.0, 0.0, 1.0]).try_reshape(&[1, 1, 2, 2]).unwrap();

    // With softcap, large scores get capped via tanh
    let result = q.scaled_dot_product_attention().key(&k).value(&v).softcap(1.0).call().unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 1, 2, 2]);
    // Should still produce valid output (no NaN/Inf)
    for val in arr.iter() {
        assert!(val.is_finite(), "softcap produced non-finite value: {val}");
    }
}

// =========================================================================
// Rotary Embedding tests
// =========================================================================

#[test]
fn test_rotary_emb_split() {
    // Non-interleaved: [1, 1, 4] -> split into [1, 1, 2] halves
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[1, 1, 4]).unwrap();
    // cos = [1, 0], sin = [0, 1] (identity-like rotation)
    let cos = Tensor::from_slice([1.0f32, 0.0]).try_reshape(&[1, 1, 2]).unwrap();
    let sin = Tensor::from_slice([0.0f32, 0.0]).try_reshape(&[1, 1, 2]).unwrap();

    let result = x.apply_rotary_emb(&cos, &sin, false).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 1, 4]);
    // With cos=[1,0], sin=[0,0]:
    // real = x1*cos - x2*sin = [1*1 - 3*0, 2*0 - 4*0] = [1, 0]
    // imag = x1*sin + x2*cos = [1*0 + 3*1, 2*0 + 4*0] = [3, 0]
    // Hmm, actually cos/sin broadcast element-wise to x1 and x2
    // x1 = [1, 2], x2 = [3, 4], cos = [1, 0], sin = [0, 0]
    // real = [1*1 - 3*0, 2*0 - 4*0] = [1, 0]
    // imag = [1*0 + 3*1, 2*0 + 4*0] = [3, 0]
    // cat = [1, 0, 3, 0]
    assert!((arr[[0, 0, 0]] - 1.0).abs() < 1e-5);
    assert!((arr[[0, 0, 1]] - 0.0).abs() < 1e-5);
    assert!((arr[[0, 0, 2]] - 3.0).abs() < 1e-5);
    assert!((arr[[0, 0, 3]] - 0.0).abs() < 1e-5);
}

#[test]
fn test_rotary_emb_interleaved() {
    // Interleaved: [1, 1, 4] -> reshape [1,1,2,2] -> split -> squeeze
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[1, 1, 4]).unwrap();
    // cos = [1, 1], sin = [0, 0] (identity rotation)
    let cos = Tensor::from_slice([1.0f32, 1.0]).try_reshape(&[1, 1, 2]).unwrap();
    let sin = Tensor::from_slice([0.0f32, 0.0]).try_reshape(&[1, 1, 2]).unwrap();

    let result = x.apply_rotary_emb(&cos, &sin, true).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 1, 4]);
    // Interleaved: x1 = [1, 3] (even), x2 = [2, 4] (odd)
    // real = x1*cos - x2*sin = [1, 3]
    // imag = x1*sin + x2*cos = [2, 4]
    // stack on last dim -> [[1,2], [3,4]] -> flatten -> [1, 2, 3, 4]
    assert!((arr[[0, 0, 0]] - 1.0).abs() < 1e-5);
    assert!((arr[[0, 0, 1]] - 2.0).abs() < 1e-5);
    assert!((arr[[0, 0, 2]] - 3.0).abs() < 1e-5);
    assert!((arr[[0, 0, 3]] - 4.0).abs() < 1e-5);
}

#[test]
fn test_rotary_emb_rotation() {
    // 90-degree rotation: cos=0, sin=1
    let x = Tensor::from_slice([1.0f32, 0.0, 0.0, 1.0]).try_reshape(&[1, 1, 4]).unwrap();
    let cos = Tensor::from_slice([0.0f32, 0.0]).try_reshape(&[1, 1, 2]).unwrap();
    let sin = Tensor::from_slice([1.0f32, 1.0]).try_reshape(&[1, 1, 2]).unwrap();

    let result = x.apply_rotary_emb(&cos, &sin, false).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    // x1 = [1, 0], x2 = [0, 1]
    // real = x1*cos - x2*sin = [0-0, 0-1] = [0, -1]
    // imag = x1*sin + x2*cos = [1+0, 0+0] = [1, 0]
    // cat = [0, -1, 1, 0]
    assert!((arr[[0, 0, 0]] - 0.0).abs() < 1e-5);
    assert!((arr[[0, 0, 1]] - (-1.0)).abs() < 1e-5);
    assert!((arr[[0, 0, 2]] - 1.0).abs() < 1e-5);
    assert!((arr[[0, 0, 3]] - 0.0).abs() < 1e-5);
}
