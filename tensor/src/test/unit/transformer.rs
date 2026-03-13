//! Tests for transformer building blocks: embedding, attention, rotary embeddings, rms_norm.

use crate::Tensor;
use ndarray::{Array2, array};

crate::codegen_tests! {
    // =========================================================================
    // RMS Norm tests
    // =========================================================================

    fn test_rms_norm_basic(config) {
        // rms_norm(x) = x * rsqrt(mean(x^2) + eps)
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0, 4.0]]);
        let result = x.rms_norm(-1, 1e-5).unwrap();
        let result = result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 4]);

        // Manual: mean([1,4,9,16]) = 7.5, rsqrt(7.5 + 1e-5) ≈ 0.36514837
        let rms_inv = 1.0 / (7.5f32 + 1e-5).sqrt();
        for i in 0..4 {
            let expected = (i + 1) as f32 * rms_inv;
            assert!((view[[0, i]] - expected).abs() < 1e-4, "rms_norm[{i}]: got {}, expected {}", view[[0, i]], expected);
        }
    }

    fn test_rms_norm_axis(config) {
        // (2, 3), normalize over last axis
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = x.rms_norm(-1, 1e-5).unwrap();
        let result = result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[2, 3]);

        // Row 0: mean([1,4,9]) = 14/3, rsqrt(14/3 + 1e-5) ≈ 0.4629
        let rms0 = 1.0 / (14.0f32 / 3.0 + 1e-5).sqrt();
        assert!((view[[0, 0]] - 1.0 * rms0).abs() < 1e-4);
        assert!((view[[0, 1]] - 2.0 * rms0).abs() < 1e-4);

        // Row 1: mean([16,25,36]) = 77/3, rsqrt(77/3 + 1e-5)
        let rms1 = 1.0 / (77.0f32 / 3.0 + 1e-5).sqrt();
        assert!((view[[1, 0]] - 4.0 * rms1).abs() < 1e-4);
    }

    // =========================================================================
    // Embedding tests
    // =========================================================================

    fn test_embedding_basic(config) {
        // Weight: [3, 4] (3 vocab, 4 embed_dim)
        let weight_data: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let weight = Tensor::from_ndarray(&Array2::from_shape_vec((3, 4), weight_data).unwrap());
        // Indices: [2, 0] -> should return rows 2 and 0
        let indices = Tensor::from_slice([2i32, 0]);
        let result = weight.embedding(&indices).unwrap();
        let result = result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[2, 4]);
        // Row 0 = weight[2] = [8, 9, 10, 11]
        assert_eq!(view[[0, 0]], 8.0);
        assert_eq!(view[[0, 3]], 11.0);
        // Row 1 = weight[0] = [0, 1, 2, 3]
        assert_eq!(view[[1, 0]], 0.0);
        assert_eq!(view[[1, 3]], 3.0);
    }

    fn test_embedding_2d_indices(config) {
        // Weight: [4, 2] (4 vocab, 2 embed_dim)
        let weight = Tensor::from_ndarray(&array![[0.0f32, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]);
        // Indices: [2, 3] (batch=2, seq=3)
        let indices = Tensor::from_ndarray(&array![[0i32, 1, 2], [3, 2, 1]]);
        let result = weight.embedding(&indices).unwrap();
        let result = result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[2, 3, 2]);
        // [0,0] = weight[0] = [0, 1]
        assert_eq!(view[[0, 0, 0]], 0.0);
        assert_eq!(view[[0, 0, 1]], 1.0);
        // [0,2] = weight[2] = [4, 5]
        assert_eq!(view[[0, 2, 0]], 4.0);
        // [1,0] = weight[3] = [6, 7]
        assert_eq!(view[[1, 0, 0]], 6.0);
        assert_eq!(view[[1, 0, 1]], 7.0);
    }

    // =========================================================================
    // Scaled Dot-Product Attention tests
    // =========================================================================

    fn test_sdpa_basic(config) {
        // Q, K, V: [1, 1, 2, 2] (batch=1, head=1, seq=2, dim=2)
        let q = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[1.0f32, 2.0], [3.0, 4.0]]]]);

        let result = q.scaled_dot_product_attention().key(&k).value(&v).call().unwrap();
        let result = result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 2, 2]);
        // With identity-like Q=K, attention should weight both rows
    }

    fn test_sdpa_causal(config) {
        // Q, K, V: [1, 1, 3, 2] — verify causal masking zeros upper triangle
        let q = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0], [0.0, 0.0]]]]);

        let result = q.scaled_dot_product_attention().key(&k).value(&v).is_causal(true).call().unwrap();
        let result = result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 3, 2]);
        // Position 0 can only attend to position 0 -> output[0] = V[0] = [1, 0]
        assert!((view[[0, 0, 0, 0]] - 1.0).abs() < 1e-4);
        assert!((view[[0, 0, 0, 1]] - 0.0).abs() < 1e-4);
    }

    fn test_sdpa_softcap(config) {
        // Verify softcap bounds the attention scores
        let q = Tensor::from_ndarray(&array![[[[10.0f32, 0.0], [0.0, 10.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]]]]);

        // With softcap, large scores get capped via tanh
        let result = q.scaled_dot_product_attention().key(&k).value(&v).softcap(1.0).call().unwrap();
        let result = result.realize_with(&config).unwrap();
        // Should still produce valid output (no NaN/Inf)
        for val in result.to_vec::<f32>().unwrap() {
            assert!(val.is_finite(), "softcap produced non-finite value: {val}");
        }
    }

    // =========================================================================
    // Rotary Embedding tests
    // =========================================================================

    fn test_rotary_emb_split(config) {
        // Non-interleaved: [1, 1, 4] -> split into [1, 1, 2] halves
        let x = Tensor::from_ndarray(&array![[[1.0f32, 2.0, 3.0, 4.0]]]);
        // cos = [1, 0], sin = [0, 1] (identity-like rotation)
        let cos = Tensor::from_ndarray(&array![[[1.0f32, 0.0]]]);
        let sin = Tensor::from_ndarray(&array![[[0.0f32, 0.0]]]);

        let result = x.apply_rotary_emb(&cos, &sin, false).unwrap();
        let result = result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 4]);
        // With cos=[1,0], sin=[0,0]:
        // real = x1*cos - x2*sin = [1*1 - 3*0, 2*0 - 4*0] = [1, 0]
        // imag = x1*sin + x2*cos = [1*0 + 3*1, 2*0 + 4*0] = [3, 0]
        // Hmm, actually cos/sin broadcast element-wise to x1 and x2
        // x1 = [1, 2], x2 = [3, 4], cos = [1, 0], sin = [0, 0]
        // real = [1*1 - 3*0, 2*0 - 4*0] = [1, 0]
        // imag = [1*0 + 3*1, 2*0 + 4*0] = [3, 0]
        // cat = [1, 0, 3, 0]
        assert!((view[[0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((view[[0, 0, 1]] - 0.0).abs() < 1e-5);
        assert!((view[[0, 0, 2]] - 3.0).abs() < 1e-5);
        assert!((view[[0, 0, 3]] - 0.0).abs() < 1e-5);
    }

    fn test_rotary_emb_interleaved(config) {
        // Interleaved: [1, 1, 4] -> reshape [1,1,2,2] -> split -> squeeze
        let x = Tensor::from_ndarray(&array![[[1.0f32, 2.0, 3.0, 4.0]]]);
        // cos = [1, 1], sin = [0, 0] (identity rotation)
        let cos = Tensor::from_ndarray(&array![[[1.0f32, 1.0]]]);
        let sin = Tensor::from_ndarray(&array![[[0.0f32, 0.0]]]);

        let result = x.apply_rotary_emb(&cos, &sin, true).unwrap();
        let result = result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 4]);
        // Interleaved: x1 = [1, 3] (even), x2 = [2, 4] (odd)
        // real = x1*cos - x2*sin = [1, 3]
        // imag = x1*sin + x2*cos = [2, 4]
        // stack on last dim -> [[1,2], [3,4]] -> flatten -> [1, 2, 3, 4]
        assert!((view[[0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((view[[0, 0, 1]] - 2.0).abs() < 1e-5);
        assert!((view[[0, 0, 2]] - 3.0).abs() < 1e-5);
        assert!((view[[0, 0, 3]] - 4.0).abs() < 1e-5);
    }

    fn test_rotary_emb_rotation(config) {
        // 90-degree rotation: cos=0, sin=1
        let x = Tensor::from_ndarray(&array![[[1.0f32, 0.0, 0.0, 1.0]]]);
        let cos = Tensor::from_ndarray(&array![[[0.0f32, 0.0]]]);
        let sin = Tensor::from_ndarray(&array![[[1.0f32, 1.0]]]);

        let result = x.apply_rotary_emb(&cos, &sin, false).unwrap();
        let result = result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        // x1 = [1, 0], x2 = [0, 1]
        // real = x1*cos - x2*sin = [0-0, 0-1] = [0, -1]
        // imag = x1*sin + x2*cos = [1+0, 0+0] = [1, 0]
        // cat = [0, -1, 1, 0]
        assert!((view[[0, 0, 0]] - 0.0).abs() < 1e-5);
        assert!((view[[0, 0, 1]] - (-1.0)).abs() < 1e-5);
        assert!((view[[0, 0, 2]] - 1.0).abs() < 1e-5);
        assert!((view[[0, 0, 3]] - 0.0).abs() < 1e-5);
    }
}
