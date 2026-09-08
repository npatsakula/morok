//! Phase 3 tests: attention_helpers.
//!
//! Every helper is checked against a host reference computed the way the model
//! code it replaces computes it, so the model can swap to the tensor helper
//! without a numeric change.

use crate::test::helpers::*;
use crate::{Tensor, Variable};
use svod_dtype::DType;
use svod_ir::SInt;
use test_case::test_case;

// =========================================================================
// Host references
// =========================================================================

/// `modernbert::rotary::RotaryTable::new` verbatim: `inv_freq` in f64, angles
/// accumulated in f32, row-major `[seq_len, head_dim / 2]`.
fn rope_reference(theta: f64, seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half).map(|i| theta.powf(-2.0 * i as f64 / head_dim as f64) as f32).collect();
    let mut cos = Vec::with_capacity(seq_len * half);
    let mut sin = Vec::with_capacity(seq_len * half);
    for s in 0..seq_len {
        for freq in &inv_freq {
            let angle = s as f32 * freq;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

/// Row-major strides for `dims`.
fn strides(dims: &[usize]) -> Vec<usize> {
    let mut out = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        out[i] = out[i + 1] * dims[i + 1];
    }
    out
}

/// `torch.repeat_interleave` on a row-major buffer.
fn repeat_interleave_reference(data: &[f32], dims: &[usize], repeats: usize, axis: usize) -> Vec<f32> {
    let mut out_dims = dims.to_vec();
    out_dims[axis] *= repeats;
    let (in_strides, out_strides) = (strides(dims), strides(&out_dims));
    let total: usize = out_dims.iter().product();
    (0..total)
        .map(|flat| {
            let source: usize = (0..out_dims.len())
                .map(|d| {
                    let mut index = flat / out_strides[d] % out_dims[d];
                    if d == axis {
                        index /= repeats;
                    }
                    index * in_strides[d]
                })
                .sum();
            data[source]
        })
        .collect()
}

/// `modernbert::pooling::masked_mean` on `[B, L, D]` with a `[B, L]` mask.
fn masked_mean_reference(data: &[f32], mask: &[f32], batch: usize, len: usize, depth: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * depth];
    for b in 0..batch {
        let count: f32 = (0..len).map(|l| mask[b * len + l]).sum::<f32>().max(1e-9);
        for d in 0..depth {
            let sum: f32 = (0..len).map(|l| data[(b * len + l) * depth + d] * mask[b * len + l]).sum();
            out[b * depth + d] = sum / count;
        }
    }
    out
}

/// A deterministic, well-spread filler — cheaper than wiring up an RNG and
/// stable across the two sides of every equality test.
fn ramp(count: usize) -> Vec<f32> {
    (0..count).map(|i| ((i as f32) * 0.37).sin() + (i as f32) * 0.01).collect()
}

// =========================================================================
// rope_table
// =========================================================================

crate::codegen_tests! {
    #[test_case(10_000.0, 8, 16; "modernbert_global")]
    #[test_case(100_000.0, 6, 8; "modernbert_local")]
    #[test_case(1_000_000.0, 5, 8; "qwen3")]
    #[test_case(64.0, 4, 4; "small_base")]
    fn test_rope_table_matches_host(config, theta: f64, seq_len: usize, head_dim: usize) {
        test_setup();
        let (cos, sin) = Tensor::rope_table(theta, seq_len, head_dim, DType::Float32).unwrap();
        assert_eq!(cos.dims().unwrap(), vec![1, 1, seq_len, head_dim / 2]);
        assert_eq!(sin.dims().unwrap(), vec![1, 1, seq_len, head_dim / 2]);

        cos.realize_with(&config).unwrap();
        sin.realize_with(&config).unwrap();
        let (expected_cos, expected_sin) = rope_reference(theta, seq_len, head_dim);
        assert_close_f32(&cos.as_vec::<f32>().unwrap(), &expected_cos, 1e-5);
        assert_close_f32(&sin.as_vec::<f32>().unwrap(), &expected_sin, 1e-5);
    }

    /// The table must land in `apply_rotary_emb`'s broadcast slot: a
    /// `[B, H, L, head_dim]` q/k rotates without an explicit expand.
    fn test_rope_table_applies_to_qk(config) {
        test_setup();
        let (cos, sin) = Tensor::rope_table(10_000.0, 4, 8, DType::Float32).unwrap();
        let q = Tensor::from_slice(ramp(2 * 3 * 4 * 8)).try_reshape([2, 3, 4, 8]).unwrap();
        let rotated = q.apply_rotary_emb(&cos, &sin, false).unwrap();
        assert_eq!(rotated.dims().unwrap(), vec![2, 3, 4, 8]);
        rotated.realize_with(&config).unwrap();

        // Position 0 has angle 0 → cos = 1, sin = 0 → the rotation is identity.
        let got = rotated.as_vec::<f32>().unwrap();
        let src = q.as_vec::<f32>().unwrap();
        for h in 0..3 {
            let base = h * 4 * 8;
            assert_close_f32(&got[base..base + 8], &src[base..base + 8], 1e-5);
        }
    }
}

#[test_case(0; "zero")]
#[test_case(7; "odd")]
fn test_rope_table_rejects_bad_head_dim(head_dim: usize) {
    let err = Tensor::rope_table(10_000.0, 4, head_dim, DType::Float32).unwrap_err();
    assert!(matches!(err.kind(), crate::ErrorKind::ParamRange { param: "head_dim", .. }), "{err}");
}

#[test]
fn test_rope_table_rejects_empty_sequence() {
    let err = Tensor::rope_table(10_000.0, 0, 8, DType::Float32).unwrap_err();
    assert!(matches!(err.kind(), crate::ErrorKind::ParamRange { param: "seq_len", .. }), "{err}");
}

// =========================================================================
// split_heads / merge_heads
// =========================================================================

crate::codegen_tests! {
    #[test_case(2, 3, 8, 4; "four_heads")]
    #[test_case(1, 5, 6, 2; "two_heads")]
    #[test_case(3, 2, 12, 1; "single_head")]
    fn test_split_heads_layout(config, batch: usize, len: usize, features: usize, heads: usize) {
        test_setup();
        let head_dim = features / heads;
        let data = ramp(batch * len * features);
        let x = Tensor::from_slice(data.clone()).try_reshape([batch, len, features]).unwrap();

        let split = x.split_heads(heads).unwrap();
        assert_eq!(split.dims().unwrap(), vec![batch, heads, len, head_dim]);
        split.realize_with(&config).unwrap();

        // split[b, h, l, d] == x[b, l, h * head_dim + d]
        let got = split.as_vec::<f32>().unwrap();
        let mut expected = Vec::with_capacity(got.len());
        for b in 0..batch {
            for h in 0..heads {
                for l in 0..len {
                    for d in 0..head_dim {
                        expected.push(data[(b * len + l) * features + h * head_dim + d]);
                    }
                }
            }
        }
        assert_close_f32(&got, &expected, 1e-6);
    }

    #[test_case(2, 3, 8, 4; "four_heads")]
    #[test_case(1, 5, 6, 3; "three_heads")]
    fn test_split_merge_round_trip(config, batch: usize, len: usize, features: usize, heads: usize) {
        test_setup();
        let data = ramp(batch * len * features);
        let x = Tensor::from_slice(data.clone()).try_reshape([batch, len, features]).unwrap();

        let round_trip = x.split_heads(heads).unwrap().merge_heads().unwrap();
        assert_eq!(round_trip.dims().unwrap(), vec![batch, len, features]);
        round_trip.realize_with(&config).unwrap();
        assert_close_f32(&round_trip.as_vec::<f32>().unwrap(), &data, 1e-6);
    }

    /// A JIT-style symbolic batch must survive the split/merge pair: only the
    /// feature axis is required to be concrete.
    fn test_split_merge_symbolic_batch(config) {
        test_setup();
        let batch = Variable::new("B", 1, 4);
        let shape = [batch.bind(2).unwrap().as_sint(), SInt::from(3), SInt::from(8)];
        let x = Tensor::empty_dynamic(&shape, DType::Float32);
        let data = ramp(2 * 3 * 8);
        x.assign(&Tensor::from_slice(data.clone()).try_reshape([2, 3, 8]).unwrap());

        let split = x.split_heads(4).unwrap();
        let split_shape = split.shape().unwrap();
        assert!(split_shape[0].is_symbolic(), "batch must stay symbolic: {split_shape:?}");
        assert_eq!(split_shape[1].as_const(), Some(4));
        assert_eq!(split_shape[3].as_const(), Some(2));

        let round_trip = split.merge_heads().unwrap();
        assert!(round_trip.shape().unwrap()[0].is_symbolic());

        // A symbolic-shaped buffer can't be read back elementwise; fold the
        // batch axis away and compare the per-position sums.
        let folded = round_trip.sum(0).unwrap();
        folded.realize_with(&config).unwrap();
        let row = 3 * 8;
        let expected: Vec<f32> = (0..row).map(|i| data[i] + data[row + i]).collect();
        assert_close_f32(&folded.as_vec::<f32>().unwrap(), &expected, 1e-5);
    }
}

#[test]
fn test_split_heads_rejects_indivisible_features() {
    let x = Tensor::zeros(&[1, 2, 6], DType::Float32);
    let err = x.split_heads(4).unwrap_err();
    assert!(matches!(err.kind(), crate::ErrorKind::Divisibility { op: "split_heads", .. }), "{err}");
}

#[test_case(2; "rank_two")]
#[test_case(4; "rank_four")]
fn test_split_heads_rejects_wrong_rank(rank: usize) {
    let x = Tensor::zeros(&vec![2usize; rank], DType::Float32);
    let err = x.split_heads(2).unwrap_err();
    assert!(matches!(err.kind(), crate::ErrorKind::NdimExact { op: "split_heads", expected: 3, .. }), "{err}");
}

// =========================================================================
// causal_mask / sequence_mask
// =========================================================================

crate::codegen_tests! {
    #[test_case(1; "single")]
    #[test_case(4; "four")]
    fn test_causal_mask_values(config, len: usize) {
        test_setup();
        let mask = Tensor::causal_mask(len, DType::Float32).unwrap();
        assert_eq!(mask.dims().unwrap(), vec![1, 1, len, len]);
        mask.realize_with(&config).unwrap();

        let values = mask.as_vec::<f32>().unwrap();
        for query in 0..len {
            for key in 0..len {
                let value = values[query * len + key];
                if key <= query {
                    assert_eq!(value, 0.0, "visible ({query}, {key}) must be 0");
                } else {
                    assert!(value.is_infinite() && value.is_sign_negative(), "masked ({query}, {key}) = {value}");
                }
            }
        }
    }

    /// The additive mask must forbid exactly what `is_causal` forbids.
    fn test_causal_mask_matches_is_causal(config) {
        test_setup();
        let q = Tensor::from_slice(ramp(8)).try_reshape([1, 1, 4, 2]).unwrap();
        let v = Tensor::from_slice(ramp(8)).try_reshape([1, 1, 4, 2]).unwrap();
        let additive = Tensor::causal_mask(4, DType::Float32).unwrap();

        let flagged = q.scaled_dot_product_attention().key(&q).value(&v).is_causal(true).call().unwrap();
        let masked = q.scaled_dot_product_attention().key(&q).value(&v).attn_mask(&additive).call().unwrap();
        flagged.realize_with(&config).unwrap();
        masked.realize_with(&config).unwrap();
        assert_close_f32(&masked.as_vec::<f32>().unwrap(), &flagged.as_vec::<f32>().unwrap(), 1e-5);
    }

    #[test_case(&[3i32, 1, 0], 3; "mixed")]
    #[test_case(&[4i32, 4], 4; "all_valid")]
    #[test_case(&[0i32], 2; "empty_row")]
    #[test_case(&[5i32, 2], 3; "length_over_max")]
    fn test_sequence_mask(config, lengths: &[i32], max_len: usize) {
        test_setup();
        let lens = Tensor::from_slice(lengths);
        let mask = Tensor::sequence_mask(&lens, max_len).unwrap();
        assert_eq!(mask.dtype(), DType::Bool);
        assert_eq!(mask.dims().unwrap(), vec![lengths.len(), max_len]);

        let as_float = mask.cast(DType::Float32);
        as_float.realize_with(&config).unwrap();
        let expected: Vec<f32> = lengths
            .iter()
            .flat_map(|&len| (0..max_len).map(move |p| f32::from(p < len.max(0) as usize)))
            .collect();
        assert_close_f32(&as_float.as_vec::<f32>().unwrap(), &expected, 1e-6);
    }
}

#[test]
fn test_sequence_mask_rejects_non_vector_lengths() {
    let lengths = Tensor::zeros(&[2, 2], DType::Int32);
    let err = Tensor::sequence_mask(&lengths, 4).unwrap_err();
    assert!(matches!(err.kind(), crate::ErrorKind::NdimExact { op: "sequence_mask", expected: 1, .. }), "{err}");
}

// =========================================================================
// repeat_interleave
// =========================================================================

crate::codegen_tests! {
    #[test_case(2, 0; "dim_0")]
    #[test_case(3, 1; "dim_1")]
    #[test_case(2, 2; "dim_2")]
    #[test_case(4, -1; "negative_dim")]
    #[test_case(1, 1; "identity")]
    fn test_repeat_interleave_vs_host(config, repeats: usize, dim: isize) {
        test_setup();
        let dims = [2usize, 3, 4];
        let data = ramp(dims.iter().product());
        let x = Tensor::from_slice(data.clone()).try_reshape(dims).unwrap();

        let result = x.repeat_interleave(repeats, dim).unwrap();
        let axis = if dim < 0 { (dims.len() as isize + dim) as usize } else { dim as usize };
        let mut expected_dims = dims.to_vec();
        expected_dims[axis] *= repeats;
        assert_eq!(result.dims().unwrap(), expected_dims);

        result.realize_with(&config).unwrap();
        let expected = repeat_interleave_reference(&data, &dims, repeats, axis);
        assert_close_f32(&result.as_vec::<f32>().unwrap(), &expected, 1e-6);
    }

    /// `qwen3::attention::repeat_kv` shape: a symbolic batch beside the
    /// concrete head axis being repeated.
    fn test_repeat_interleave_symbolic_batch(config) {
        test_setup();
        let batch = Variable::new("B", 1, 4);
        let shape = [batch.bind(2).unwrap().as_sint(), SInt::from(2), SInt::from(3), SInt::from(2)];
        let x = Tensor::empty_dynamic(&shape, DType::Float32);
        let data = ramp(2 * 2 * 3 * 2);
        x.assign(&Tensor::from_slice(data.clone()).try_reshape([2, 2, 3, 2]).unwrap());

        let result = x.repeat_interleave(3, 1).unwrap();
        let result_shape = result.shape().unwrap();
        assert!(result_shape[0].is_symbolic(), "batch must stay symbolic: {result_shape:?}");
        assert_eq!(result_shape[1].as_const(), Some(6));

        // Fold the symbolic batch away so the result can be read back.
        let folded = result.sum(0).unwrap();
        folded.realize_with(&config).unwrap();
        let reference = repeat_interleave_reference(&data, &[2, 2, 3, 2], 3, 1);
        let row = 6 * 3 * 2;
        let expected: Vec<f32> = (0..row).map(|i| reference[i] + reference[row + i]).collect();
        assert_close_f32(&folded.as_vec::<f32>().unwrap(), &expected, 1e-5);
    }
}

#[test]
fn test_repeat_interleave_rejects_zero_repeats() {
    let x = Tensor::zeros(&[2, 2], DType::Float32);
    let err = x.repeat_interleave(0, 0).unwrap_err();
    assert!(matches!(err.kind(), crate::ErrorKind::ParamRange { param: "repeats", .. }), "{err}");
}

// =========================================================================
// SDPA: key_padding_mask and GQA
// =========================================================================

crate::codegen_tests! {
    /// `key_padding_mask` (True = valid) must equal the hand-built 4-D
    /// `attn_mask` (True = masked out) the encoders assemble today.
    #[test_case(&[4i32, 2]; "one_padded_row")]
    #[test_case(&[1i32, 3]; "both_padded")]
    #[test_case(&[4i32, 4]; "nothing_padded")]
    fn test_sdpa_key_padding_mask_equals_attn_mask(config, lengths: &[i32]) {
        test_setup();
        let (batch, heads, q_len, k_len, head_dim) = (lengths.len(), 2usize, 3usize, 4usize, 2usize);
        let q = Tensor::from_slice(ramp(batch * heads * q_len * head_dim))
            .try_reshape([batch, heads, q_len, head_dim])
            .unwrap();
        let k = Tensor::from_slice(ramp(batch * heads * k_len * head_dim))
            .try_reshape([batch, heads, k_len, head_dim])
            .unwrap();
        let v = Tensor::from_slice(ramp(batch * heads * k_len * head_dim).iter().rev().copied().collect::<Vec<_>>())
            .try_reshape([batch, heads, k_len, head_dim])
            .unwrap();

        let lens = Tensor::from_slice(lengths);
        let padding = Tensor::sequence_mask(&lens, k_len).unwrap();
        // The encoders' form: invert to "True = masked out", then [B, 1, 1, Lk].
        let attn = padding.logical_not().unwrap().try_unsqueeze(1).unwrap().try_unsqueeze(1).unwrap();

        let via_padding = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .key_padding_mask(&padding)
            .call()
            .unwrap();
        let via_attn = q.scaled_dot_product_attention().key(&k).value(&v).attn_mask(&attn).call().unwrap();

        via_padding.realize_with(&config).unwrap();
        via_attn.realize_with(&config).unwrap();
        assert_close_f32(&via_padding.as_vec::<f32>().unwrap(), &via_attn.as_vec::<f32>().unwrap(), 1e-5);
    }

    /// The padding mask must intersect with, not replace, the causal mask.
    fn test_sdpa_key_padding_mask_intersects_causal(config) {
        test_setup();
        let q = Tensor::from_slice(ramp(6)).try_reshape([1, 1, 3, 2]).unwrap();
        let v = Tensor::from_slice(vec![1.0f32, 1.0, 10.0, 10.0, 100.0, 100.0])
            .try_reshape([1, 1, 3, 2])
            .unwrap();
        let padding = Tensor::sequence_mask(&Tensor::from_slice([2i32]), 3).unwrap();

        let out = q
            .scaled_dot_product_attention()
            .key(&q)
            .value(&v)
            .key_padding_mask(&padding)
            .is_causal(true)
            .call()
            .unwrap();
        out.realize_with(&config).unwrap();
        let got = out.as_vec::<f32>().unwrap();
        // Query 0 sees key 0 only (causal); key 2 is padded away for every query,
        // so no output may carry V[2] = 100.
        assert_close_f32(&got[0..2], &[1.0, 1.0], 1e-5);
        for value in &got {
            assert!(*value < 11.0, "padded key 2 leaked into the output: {got:?}");
        }
    }

    /// `enable_gqa` must equal repeating K/V by hand, the way
    /// `qwen3::attention::repeat_kv` does today.
    #[test_case(4, 2; "two_groups")]
    #[test_case(4, 1; "multi_query")]
    #[test_case(2, 2; "no_grouping")]
    fn test_sdpa_gqa_equals_manual_repeat(config, q_heads: usize, kv_heads: usize) {
        test_setup();
        let (batch, q_len, k_len, head_dim) = (2usize, 3usize, 4usize, 2usize);
        let q = Tensor::from_slice(ramp(batch * q_heads * q_len * head_dim))
            .try_reshape([batch, q_heads, q_len, head_dim])
            .unwrap();
        let k = Tensor::from_slice(ramp(batch * kv_heads * k_len * head_dim))
            .try_reshape([batch, kv_heads, k_len, head_dim])
            .unwrap();
        let v = Tensor::from_slice(&ramp(batch * kv_heads * k_len * head_dim + 1)[1..])
            .try_reshape([batch, kv_heads, k_len, head_dim])
            .unwrap();

        let grouped = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .enable_gqa(true)
            .call()
            .unwrap();

        let repeats = q_heads / kv_heads;
        let manual = q
            .scaled_dot_product_attention()
            .key(&k.repeat_interleave(repeats, 1).unwrap())
            .value(&v.repeat_interleave(repeats, 1).unwrap())
            .call()
            .unwrap();

        assert_eq!(grouped.dims().unwrap(), vec![batch, q_heads, q_len, head_dim]);
        grouped.realize_with(&config).unwrap();
        manual.realize_with(&config).unwrap();
        assert_close_f32(&grouped.as_vec::<f32>().unwrap(), &manual.as_vec::<f32>().unwrap(), 1e-5);
    }
}

#[test]
fn test_sdpa_gqa_rejects_indivisible_heads() {
    let q = Tensor::zeros(&[1, 3, 2, 2], DType::Float32);
    let kv = Tensor::zeros(&[1, 2, 2, 2], DType::Float32);
    let err = q.scaled_dot_product_attention().key(&kv).value(&kv).enable_gqa(true).call().unwrap_err();
    assert!(matches!(err.kind(), crate::ErrorKind::Divisibility { op: "scaled_dot_product_attention", .. }), "{err}");
}

#[test]
fn test_sdpa_rejects_non_2d_key_padding_mask() {
    let q = Tensor::zeros(&[1, 1, 2, 2], DType::Float32);
    let mask = Tensor::zeros(&[1, 1, 2], DType::Bool);
    let err = q.scaled_dot_product_attention().key(&q).value(&q).key_padding_mask(&mask).call().unwrap_err();
    assert!(matches!(err.kind(), crate::ErrorKind::NdimExact { expected: 2, .. }), "{err}");
}

// =========================================================================
// masked_mean / take_index
// =========================================================================

crate::codegen_tests! {
    #[test_case(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 0.0]; "one_padded")]
    #[test_case(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0]; "single_token_rows")]
    #[test_case(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0]; "nothing_padded")]
    #[test_case(&[0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; "empty_first_row")]
    fn test_masked_mean_vs_host(config, mask_values: &[f32]) {
        test_setup();
        let (batch, len, depth) = (2usize, 3usize, 4usize);
        let data = ramp(batch * len * depth);
        let x = Tensor::from_slice(data.clone()).try_reshape([batch, len, depth]).unwrap();
        let mask = Tensor::from_slice(mask_values)
            .try_reshape([batch, len])
            .unwrap()
            .try_gt(0.5f32)
            .unwrap();

        let pooled = x.masked_mean(&mask, 1).unwrap();
        assert_eq!(pooled.dims().unwrap(), vec![batch, depth]);
        pooled.realize_with(&config).unwrap();
        let expected = masked_mean_reference(&data, mask_values, batch, len, depth);
        assert_close_f32(&pooled.as_vec::<f32>().unwrap(), &expected, 1e-5);
    }

    /// A float 0/1 mask must pool exactly like the boolean one.
    fn test_masked_mean_accepts_float_mask(config) {
        test_setup();
        let x = Tensor::from_slice(ramp(2 * 3 * 4)).try_reshape([2, 3, 4]).unwrap();
        let values = vec![1.0f32, 1.0, 0.0, 1.0, 0.0, 0.0];
        let float_mask = Tensor::from_slice(values.clone()).try_reshape([2, 3]).unwrap();
        let bool_mask = float_mask.try_gt(0.5f32).unwrap();

        let from_float = x.masked_mean(&float_mask, 1).unwrap();
        let from_bool = x.masked_mean(&bool_mask, 1).unwrap();
        from_float.realize_with(&config).unwrap();
        from_bool.realize_with(&config).unwrap();
        assert_close_f32(&from_float.as_vec::<f32>().unwrap(), &from_bool.as_vec::<f32>().unwrap(), 1e-6);
    }

    #[test_case(1, 0; "cls")]
    #[test_case(1, -1; "last_token")]
    #[test_case(1, 2; "middle")]
    #[test_case(0, -1; "last_batch_row")]
    #[test_case(-1, 1; "feature_axis")]
    fn test_take_index_vs_host(config, axis: isize, index: isize) {
        test_setup();
        let dims = [2usize, 4, 3];
        let data = ramp(dims.iter().product());
        let x = Tensor::from_slice(data.clone()).try_reshape(dims).unwrap();

        let taken = x.take_index(axis, index).unwrap();
        let ax = if axis < 0 { (dims.len() as isize + axis) as usize } else { axis as usize };
        let position = if index < 0 { (dims[ax] as isize + index) as usize } else { index as usize };
        let expected_dims: Vec<usize> =
            dims.iter().enumerate().filter(|(d, _)| *d != ax).map(|(_, size)| *size).collect();
        assert_eq!(taken.dims().unwrap(), expected_dims);

        taken.realize_with(&config).unwrap();
        let in_strides = strides(&dims);
        let out_strides = strides(&expected_dims);
        let total: usize = expected_dims.iter().product();
        let expected: Vec<f32> = (0..total)
            .map(|flat| {
                let mut source = position * in_strides[ax];
                let mut out_axis = 0;
                for (d, size) in dims.iter().enumerate() {
                    if d == ax {
                        continue;
                    }
                    let _ = size;
                    source += (flat / out_strides[out_axis] % expected_dims[out_axis]) * in_strides[d];
                    out_axis += 1;
                }
                data[source]
            })
            .collect();
        assert_close_f32(&taken.as_vec::<f32>().unwrap(), &expected, 1e-6);
    }

    /// CLS and last-token pooling, the two shapes the embedders use.
    fn test_take_index_pooling_positions(config) {
        test_setup();
        let data = ramp(2 * 3 * 2);
        let x = Tensor::from_slice(data.clone()).try_reshape([2, 3, 2]).unwrap();

        let cls = x.take_index(1, 0).unwrap();
        let last = x.take_index(1, -1).unwrap();
        cls.realize_with(&config).unwrap();
        last.realize_with(&config).unwrap();
        assert_close_f32(&cls.as_vec::<f32>().unwrap(), &[data[0], data[1], data[6], data[7]], 1e-6);
        assert_close_f32(&last.as_vec::<f32>().unwrap(), &[data[4], data[5], data[10], data[11]], 1e-6);
    }
}

#[test]
fn test_take_index_negative_needs_concrete_axis() {
    let batch = Variable::new("B", 1, 4);
    let shape = [batch.bind(2).unwrap().as_sint(), SInt::from(3)];
    let x = Tensor::empty_dynamic(&shape, DType::Float32);
    let err = x.take_index(0, -1).unwrap_err();
    assert!(matches!(err.kind(), crate::ErrorKind::SymbolicShapeUnsupported { .. }), "{err}");
}

#[test]
fn test_take_index_rejects_out_of_range_negative() {
    let x = Tensor::zeros(&[2, 3], DType::Float32);
    let err = x.take_index(1, -4).unwrap_err();
    assert!(matches!(err.kind(), crate::ErrorKind::ParamRange { param: "index", .. }), "{err}");
}
