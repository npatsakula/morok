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
