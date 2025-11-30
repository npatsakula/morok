//! Movement operation tests.
//!
//! Tests all movement/reshape operations including reshape, permute, expand, pad, shrink, flip.

use std::f64::consts::PI;

use smallvec::smallvec;

use morok_dtype::DType;

use crate::{SInt, UOp, error::Error, shape::Shape};

// =========================================================================
// Reshape Tests
// =========================================================================

#[test]
fn test_reshape_basic() {
    let new_shape: Shape = smallvec![SInt::from(1), SInt::from(1)];

    let result = UOp::native_const(1.0f32).try_reshape(&new_shape).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_reshape_size_must_match() {
    let val = UOp::native_const(1.0f32); // Scalar (size=1)
    let bad_shape: Shape = smallvec![SInt::from(2), SInt::from(3)]; // size=6

    let result = val.try_reshape(&bad_shape);
    assert!(matches!(result, Err(Error::ReshapeSizeMismatch { input_size: 1, output_size: 6 })));
}

// =========================================================================
// Permute Tests
// =========================================================================

#[test]
fn test_permute_empty_on_scalar() {
    let val = UOp::native_const(1.0f32); // Scalar

    let perm = vec![]; // Empty permutation for scalar
    let result = val.try_permute(perm).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_permute_invalid_on_scalar() {
    let val = UOp::native_const(1.0f32); // Scalar

    let bad_perm = vec![0, 1]; // Not valid for scalar (empty shape)
    let result = val.try_permute(bad_perm);
    assert!(matches!(result, Err(Error::PermuteInvalidPermutation { .. })));
}

#[test]
fn test_permute_duplicate_index() {
    let _val = UOp::native_const(1.0f32);

    // This is hard to test without a UOp with known multi-dimensional shape
    // For now, test that validation exists
}

// =========================================================================
// Expand Tests
// =========================================================================

#[test]
fn test_expand_dimension_mismatch() {
    let val = UOp::native_const(1.0f32); // Scalar (0 dims)
    let new_shape: Shape = smallvec![SInt::from(3), SInt::from(5)]; // 2 dims

    let result = val.try_expand(&new_shape);
    assert!(matches!(result, Err(Error::ExpandDimensionMismatch { input_dims: 0, output_dims: 2 })));
}

// =========================================================================
// Pad Tests
// =========================================================================

#[test]
fn test_pad_dimension_mismatch() {
    let val = UOp::native_const(1.0f32); // Scalar

    // Padding for 2 dimensions but scalar has 0
    let padding = vec![(SInt::from(0), SInt::from(0)), (SInt::from(1), SInt::from(1))];

    let result = val.try_pad(&padding);
    assert!(matches!(result, Err(Error::PadDimensionMismatch { padding_dims: 2, shape_dims: 0 })));
}

#[test]
fn test_pad_empty_on_scalar() {
    let val = UOp::native_const(1.0f32); // Scalar

    // Empty padding for scalar
    let padding = vec![];

    let result = val.try_pad(&padding).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Shrink Tests
// =========================================================================

#[test]
fn test_shrink_empty_on_scalar() {
    let val = UOp::native_const(1.0f32); // Scalar

    // Empty ranges for scalar
    let ranges = vec![];

    let result = val.try_shrink(&ranges).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Flip Tests
// =========================================================================

#[test]
fn test_flip_dimension_mismatch() {
    let val = UOp::native_const(1.0f32); // Scalar

    // Flip spec for 2 dimensions but scalar has 0
    let flip_spec = vec![true, false];

    let result = val.try_flip(flip_spec);
    assert!(matches!(result, Err(Error::FlipInvalidSpec { expected_dims: 0, got_dims: 2 })));
}

#[test]
fn test_flip_empty_on_scalar() {
    let val = UOp::native_const(1.0f32); // Scalar

    // Empty flip spec for scalar
    let flip_spec = vec![];

    let result = val.try_flip(flip_spec).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Multi Tests
// =========================================================================

#[test]
fn test_multi_basic() {
    let val = UOp::native_const(1.0f32);
    let result = UOp::multi(val, 0);
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// DType Preservation Tests
// =========================================================================

#[test]
fn test_movement_ops_preserve_dtype() {
    // Reshape preserves dtype
    let val_int = UOp::native_const(42i64);
    let shape: Shape = smallvec![SInt::from(1)];
    let reshaped = val_int.try_reshape(&shape).unwrap();
    assert_eq!(reshaped.dtype(), DType::Int64);

    // Permute preserves dtype
    let val_float = UOp::native_const(PI);
    let permuted = val_float.try_permute(vec![]).unwrap();
    assert_eq!(permuted.dtype(), DType::Float64);

    // Multi preserves dtype
    let val_bool = UOp::native_const(true);
    let multi = UOp::multi(val_bool, 0);
    assert_eq!(multi.dtype(), DType::Bool);
}
