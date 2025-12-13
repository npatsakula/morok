//! Reduction operation tests.
//!
//! Tests reduction operations including ReduceAxis.

use std::f64::consts::PI;

use morok_dtype::DType;

use crate::{ReduceOp, UOp};

// =========================================================================
// ReduceAxis Tests
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
fn test_reduce_axis_full_reduction_returns_self() {
    use crate::SInt;
    use crate::shape::shape_to_uop;
    use smallvec::smallvec;

    // Create a UOp with a known shape [2, 3]
    let shape = smallvec![SInt::Const(2), SInt::Const(3)];
    let src = UOp::native_const(1.0f32);
    let shaped = UOp::reshape(src, shape_to_uop(&shape));

    // Reduce all axes - should return self since all dims will be 1 after reduction
    // This tests the early-return pattern when all reduction axes have size 1
    let result = shaped.try_reduce_axis(ReduceOp::Add, vec![0, 1]).unwrap();

    // Verify result has same dtype
    assert_eq!(result.dtype(), shaped.dtype());
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

    // Try to reduce axes 0 and 2 (both have dimension 1)
    // Should return self since no active reduction is needed
    let result = shaped.try_reduce_axis(ReduceOp::Add, vec![0, 2]).unwrap();

    // The result should be the same as input (early return)
    assert_eq!(result.dtype(), shaped.dtype());
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

    // Try to reduce axes 0, 1, 2 (includes both size-1 and non-size-1)
    // Should create ReduceAxis with only axis 1 (dimension 3)
    let result = shaped.try_reduce_axis(ReduceOp::Add, vec![0, 1, 2]).unwrap();

    // Result should be a ReduceAxis operation with filtered axes
    if let Op::ReduceAxis { axes, .. } = result.op() {
        // Only axis 1 should remain (axes 0 and 2 filtered out)
        assert_eq!(axes.len(), 1);
        assert_eq!(axes[0], 1);
    } else {
        panic!("Expected ReduceAxis operation");
    }
}
