//! Reduction operation tests.
//!
//! Tests tensor-form reduction operations.

use std::f64::consts::PI;

use svod_dtype::DType;

use crate::ops;
use crate::{ReduceOp, UOp};

// =========================================================================
// Tensor REDUCE tests
// =========================================================================

#[test]
fn test_reduce_axis_basic() {
    // Reduce on empty axes for scalar (no-op)
    let result = UOp::native_const(1.0f32).try_reduce_axis(ReduceOp::Add, vec![]).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_reduce_axis_preserves_dtype() {
    let val_int = UOp::native_const(42i32);
    let result = val_int.try_reduce_axis(ReduceOp::Add, vec![]).unwrap();
    assert_eq!(result.dtype(), DType::Int32);

    let val_float = UOp::native_const(PI);
    let result = val_float.try_reduce_axis(ReduceOp::Max, vec![]).unwrap();
    assert_eq!(result.dtype(), DType::Float64);
}

#[test]
fn test_reduce_ops() {
    let val = UOp::native_const(1.0f32);

    // Test different reduce operations
    let add = val.try_reduce_axis(ReduceOp::Add, vec![]).unwrap();
    assert_eq!(add.dtype(), DType::Float32);

    let max = val.try_reduce_axis(ReduceOp::Max, vec![]).unwrap();
    assert_eq!(max.dtype(), DType::Float32);

    let mul = val.try_reduce_axis(ReduceOp::Mul, vec![]).unwrap();
    assert_eq!(mul.dtype(), DType::Float32);

    let min = val.try_reduce_axis(ReduceOp::Min, vec![]).unwrap();
    assert_eq!(min.dtype(), DType::Float32);
}

// =========================================================================
// Early-Return Pattern Tests (Tinygrad alignment)
// =========================================================================

#[test]
fn test_reduce_axis_full_reduction_is_leading_tensor_reduce() {
    use crate::SInt;
    use crate::shape::shape_to_uop;
    use smallvec::smallvec;

    // Create a UOp with a known shape [2, 3]
    let shape = smallvec![SInt::Const(2), SInt::Const(3)];
    let src = UOp::native_const(1.0f32);
    let shaped = UOp::reshape(src, shape_to_uop(&shape));

    let result = shaped.try_reduce_axis(ReduceOp::Add, vec![0, 1]).unwrap();

    assert_eq!(result.dtype(), shaped.dtype());
    assert!(matches!(result.op(), crate::Op::Reduce(ops::Reduce { src, ranges, num_axes: 2, .. })
        if std::sync::Arc::ptr_eq(src, &shaped) && ranges.is_empty()));
    assert!(result.shape().unwrap().unwrap().is_empty());
}

#[test]
fn test_reduce_axis_size_one_dims_filtered() {
    use crate::SInt;
    use crate::shape::shape_to_uop;
    use smallvec::smallvec;

    // Create a UOp with shape [1, 3, 1, 4]
    let shape = smallvec![SInt::Const(1), SInt::Const(3), SInt::Const(1), SInt::Const(4)];
    let src = UOp::native_const(1.0f32);
    let shaped = UOp::reshape(src, shape_to_uop(&shape));

    let result = shaped.try_reduce_axis(ReduceOp::Add, vec![0, 2]).unwrap();

    assert_eq!(result.dtype(), shaped.dtype());
    assert!(
        matches!(result.op(), crate::Op::Reshape(ops::Reshape { src, .. }) if std::sync::Arc::ptr_eq(src, &shaped))
    );
    assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[SInt::Const(3), SInt::Const(4)]);
}

#[test]
fn test_reduce_axis_mixed_size_dims() {
    use crate::SInt;
    use crate::op::Op;
    use crate::shape::shape_to_uop;
    use smallvec::smallvec;

    // Create a UOp with shape [1, 3, 1, 4]
    let shape = smallvec![SInt::Const(1), SInt::Const(3), SInt::Const(1), SInt::Const(4)];
    let src = UOp::native_const(1.0f32);
    let shaped = UOp::reshape(src, shape_to_uop(&shape));

    let result = shaped.try_reduce_axis(ReduceOp::Add, vec![0, 1, 2]).unwrap();

    let Op::Reshape(ops::Reshape { src: reduced, .. }) = result.op() else {
        panic!("expected singleton-removing RESHAPE")
    };
    let Op::Reduce(ops::Reduce { src: permuted, ranges, num_axes: 1, .. }) = reduced.op() else {
        panic!("expected tensor REDUCE")
    };
    assert!(ranges.is_empty());
    assert!(matches!(permuted.op(), Op::Permute(ops::Permute { src, axes })
        if std::sync::Arc::ptr_eq(src, &shaped) && axes == &[1, 0, 2, 3]));
    assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[SInt::Const(4)]);
}

#[test]
fn test_reduce_axis_rejects_duplicate_axes_like_tinygrad_permute() {
    use crate::SInt;
    use crate::shape::shape_to_uop;
    use smallvec::smallvec;

    let shaped = UOp::reshape(UOp::native_const(1.0f32), shape_to_uop(&smallvec![SInt::Const(2), SInt::Const(3)]));
    assert!(shaped.try_reduce_axis(ReduceOp::Add, vec![0, 0]).is_err());
}
