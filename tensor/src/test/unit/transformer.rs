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
        let mut result = x.rms_norm(-1, 1e-5).unwrap();
        result.realize_with(&config).unwrap();
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
        let mut result = x.rms_norm(-1, 1e-5).unwrap();
        result.realize_with(&config).unwrap();
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
        let mut result = weight.embedding(&indices).unwrap();
        result.realize_with(&config).unwrap();
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
        let mut result = weight.embedding(&indices).unwrap();
        result.realize_with(&config).unwrap();
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

        let mut result = q.scaled_dot_product_attention().key(&k).value(&v).call().unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 2, 2]);
        // With identity-like Q=K, attention should weight both rows
    }

    fn test_sdpa_causal(config) {
        // Q, K, V: [1, 1, 3, 2] — verify causal masking zeros upper triangle
        let q = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0], [0.0, 0.0]]]]);

        let mut result = q.scaled_dot_product_attention().key(&k).value(&v).is_causal(true).call().unwrap();
        result.realize_with(&config).unwrap();
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
        let mut result = q.scaled_dot_product_attention().key(&k).value(&v).softcap(1.0).call().unwrap();
        result.realize_with(&config).unwrap();
        // Should still produce valid output (no NaN/Inf)
        for val in result.as_vec::<f32>().unwrap() {
            assert!(val.is_finite(), "softcap produced non-finite value: {val}");
        }
    }

    fn test_sdpa_bool_mask_true_masks_out(config) {
        let q = Tensor::from_ndarray(&array![[[[1.0f32, 0.0]]]]);
        let k = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let v = Tensor::from_ndarray(&array![[[[10.0f32, 1.0], [1.0, 10.0]]]]);
        // True means masked, False means visible.
        let mask = Tensor::from_ndarray(&array![[[[true, false]]]]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .maybe_attn_mask(Some(&mask))
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 1, 2]);
        assert!((view[[0, 0, 0, 0]] - 1.0).abs() < 1e-4);
        assert!((view[[0, 0, 0, 1]] - 10.0).abs() < 1e-4);
    }

    fn test_sdpa_bool_mask_all_masked_row_finite(config) {
        let q = Tensor::from_ndarray(&array![[[[1.0f32, 0.0]]]]);
        let k = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let v = Tensor::from_ndarray(&array![[[[10.0f32, 1.0], [1.0, 10.0]]]]);
        let mask = Tensor::from_ndarray(&array![[[[true, true]]]]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .maybe_attn_mask(Some(&mask))
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        for v in result.as_vec::<f32>().unwrap() {
            assert!(v.is_finite(), "expected finite attention output, got {v}");
        }
    }

    fn test_sdpa_bool_mask_all_masked_with_causal_finite(config) {
        let q = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[10.0f32, 1.0], [1.0, 10.0]]]]);
        let mask = Tensor::from_ndarray(&array![[[[true, true], [true, true]]]]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .is_causal(true)
            .maybe_attn_mask(Some(&mask))
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        for v in result.as_vec::<f32>().unwrap() {
            assert!(v.is_finite(), "expected finite attention output with causal+mask, got {v}");
        }
    }

    fn test_sdpa_rejects_non_float_qkv(_config) {
        let qf = Tensor::from_ndarray(&array![[[[1.0f32, 0.0]]]]);
        let kf = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let vf = Tensor::from_ndarray(&array![[[[10.0f32, 1.0], [1.0, 10.0]]]]);

        let qi = Tensor::from_ndarray(&array![[[[1i32, 0]]]]);
        let ki = Tensor::from_ndarray(&array![[[[1i32, 0], [0, 1]]]]);
        let vi = Tensor::from_ndarray(&array![[[[10i32, 1], [1, 10]]]]);

        let err_q = match qi.scaled_dot_product_attention().key(&kf).value(&vf).call() {
            Ok(_) => panic!("expected query dtype error"),
            Err(err) => err,
        };
        assert!(matches!(err_q, crate::Error::FloatDTypeRequired { arg: "query", .. }));

        let err_k = match qf.scaled_dot_product_attention().key(&ki).value(&vf).call() {
            Ok(_) => panic!("expected key dtype error"),
            Err(err) => err,
        };
        assert!(matches!(err_k, crate::Error::FloatDTypeRequired { arg: "key", .. }));

        let err_v = match qf.scaled_dot_product_attention().key(&kf).value(&vi).call() {
            Ok(_) => panic!("expected value dtype error"),
            Err(err) => err,
        };
        assert!(matches!(err_v, crate::Error::FloatDTypeRequired { arg: "value", .. }));
    }

    fn test_sdpa_window_masks_far_keys(config) {
        // Seq len 4, head dim 1. Q=K=ones so raw scores are uniform; the only
        // thing distinguishing which keys are attended is the window band.
        // window=(0,0) → each query attends ONLY to itself. With V = [0,10,20,30]
        // the output equals the value at the query's own position.
        let q = Tensor::from_ndarray(&array![[[[1.0f32], [1.0], [1.0], [1.0]]]]); // [1,1,4,1]
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[0.0f32], [10.0], [20.0], [30.0]]]]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .window((0usize, 0usize))
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 4, 1]);
        // Self-only attention: output[q] = v[q].
        assert!((view[[0, 0, 0, 0]] - 0.0).abs() < 1e-4, "q0 leaked far key: {}", view[[0, 0, 0, 0]]);
        assert!((view[[0, 0, 1, 0]] - 10.0).abs() < 1e-4);
        assert!((view[[0, 0, 2, 0]] - 20.0).abs() < 1e-4);
        assert!((view[[0, 0, 3, 0]] - 30.0).abs() < 1e-4);
    }

    fn test_sdpa_window_band_attends_neighbors(config) {
        // window=(1,1): each query attends to itself and its immediate
        // neighbours. v = [0,10,20,30]; q=1 only sees keys 0,1,2 → mean of
        // (0,10,20)/3 = 10.0 (scores uniform). q=0 sees only keys 0,1 →
        // mean(0,10)/2 = 5.0.
        let q = Tensor::from_ndarray(&array![[[[1.0f32], [1.0], [1.0], [1.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[0.0f32], [10.0], [20.0], [30.0]]]]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .window((1usize, 1usize))
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        // q0: keys {0,1} → (0+10)/2 = 5
        assert!((view[[0, 0, 0, 0]] - 5.0).abs() < 1e-4, "q0: {}", view[[0, 0, 0, 0]]);
        // q1: keys {0,1,2} → (0+10+20)/3 = 10
        assert!((view[[0, 0, 1, 0]] - 10.0).abs() < 1e-4, "q1: {}", view[[0, 0, 1, 0]]);
        // q3: keys {2,3} → (20+30)/2 = 25
        assert!((view[[0, 0, 3, 0]] - 25.0).abs() < 1e-4, "q3: {}", view[[0, 0, 3, 0]]);
    }

    fn test_sdpa_window_intersects_bool_mask(config) {
        // window=(0,1) keeps keys {q, q+1}. A bool mask removes keys ≥2
        // everywhere. So q=0 keeps {0,1}∩{0,1}={0,1} → mean(0,10)=5; q=1 keeps
        // {1,2}∩{0,1}={1} → v[1]=10 (the window allowed key 2 but the mask
        // stripped it — this is the intersection under test).
        let q = Tensor::from_ndarray(&array![[[[1.0f32], [1.0], [1.0], [1.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[0.0f32], [10.0], [20.0], [30.0]]]]);
        // True = masked out. Keys ≥2 masked everywhere; key 1 also masked for q0.
        let mask = Tensor::from_ndarray(&array![
            [[[false, true, true, true], [false, false, true, true], [false, false, true, true], [false, false, true, true]]]
        ]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .window((0usize, 1usize))
            .maybe_attn_mask(Some(&mask))
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        // q0: window {0,1} ∩ mask-keep {0} = {0} → v[0] = 0.
        assert!((view[[0, 0, 0, 0]] - 0.0).abs() < 1e-4, "q0 intersect: {}", view[[0, 0, 0, 0]]);
        // q1: window {1,2} ∩ mask-keep {0,1} = {1} → v[1] = 10.
        assert!((view[[0, 0, 1, 0]] - 10.0).abs() < 1e-4, "q1 intersect: {}", view[[0, 0, 1, 0]]);
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

        let mut result = x.apply_rotary_emb(&cos, &sin, false).unwrap();
        result.realize_with(&config).unwrap();
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

        let mut result = x.apply_rotary_emb(&cos, &sin, true).unwrap();
        result.realize_with(&config).unwrap();
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

        let mut result = x.apply_rotary_emb(&cos, &sin, false).unwrap();
        result.realize_with(&config).unwrap();
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

// ─── symbolic-batch embedding (regression for the dropped-JIT-batching bug) ────
//
// `Tensor::embedding` (src/transformer.rs) used to demand a *concrete* leading
// dim on `indices`: its flatten (`try_reshape([-1])`) and output-shape rebuild
// (`idx_shape[i].as_const()`) needed a concrete total element count. A symbolic
// (rebindable) batch dim — what a JIT plan with `vars { b: (1, max_batch) }`
// threads through `forward_batch`'s `try_shrink` — used to bail with
// `SymbolicShapeUnsupported { operation: "reshape with -1 inference" }`, which
// blocked the entire ModernBERT backbone JIT path. `embedding` now carries the
// symbolic leading dim through to the gather; this guards that it stays fixed.
use crate::{SInt, Variable};

#[test]
fn embedding_supports_symbolic_batch_dim() {
    // Weight table [vocab=4, embed_dim=2] (concrete — only the *index* side matters).
    let weight = Tensor::from_ndarray(&array![[0.0f32, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]);

    // Concrete index tensor [max_b=2, seq=3], then shrink the leading dim to a
    // bound (symbolic) variable — the exact shape `forward_batch` produces for a
    // JIT plan with `vars { b: (1, 2) }`.
    let indices = Tensor::from_ndarray(&array![[0i32, 1, 2], [3, 2, 1]]);
    let bound = Variable::new("b", 1, 2).bind(2).unwrap();
    let indices = indices.try_shrink([Some((SInt::Const(0), bound.as_sint())), None]).unwrap();
    // indices is now (Symbolic(b), 3) — the batch dim is rebindable.

    let mut t = weight.embedding(&indices).expect("embedding must carry a symbolic batch dim through");
    t.realize_with(&crate::PrepareConfig::from_env()).expect("realize symbolic-batch embedding");
    // The realized shape is still symbolic (BIND on dim 0), so array_view
    // returns the flat buffer — read it flat and index manually.
    let view = t.array_view::<f32>().expect("readout");
    assert_eq!(view.shape(), &[12]);
    // A [2,3] index into a [4,2] weight yields [2,3,2]; row-major flat = b*6+s*2+e.
    // Row 0 picks weight rows [0,1,2] = [0,1],[2,3],[4,5].
    assert_eq!(view[0], 0.0); // [0,0,0]
    assert_eq!(view[3], 3.0); // [0,1,1]
    assert_eq!(view[4], 4.0); // [0,2,0]
    // Row 1 picks weight rows [3,2,1] = [6,7],[4,5],[2,3].
    assert_eq!(view[7], 7.0); // [1,0,1]
    assert_eq!(view[10], 2.0); // [1,2,0]
}
