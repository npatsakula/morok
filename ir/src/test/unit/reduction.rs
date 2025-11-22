//! Reduction operation tests.
//!
//! Tests reduction operations including ReduceAxis.

use morok_dtype::DType;

use crate::{ConstValue, ReduceOp, UOp};

// =========================================================================
// ReduceAxis Tests
// =========================================================================

#[test]
fn test_reduce_axis_basic() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Reduce on empty axes for scalar (no-op)
    let result = val.try_reduce_axis(ReduceOp::Add, vec![]).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_reduce_axis_preserves_dtype() {
    let val_int = UOp::const_(DType::Int32, ConstValue::Int(42));
    let result = val_int.try_reduce_axis(ReduceOp::Add, vec![]).unwrap();
    assert_eq!(result.dtype(), DType::Int32);

    let val_float = UOp::const_(DType::Float64, ConstValue::Float(std::f64::consts::PI));
    let result = val_float.try_reduce_axis(ReduceOp::Max, vec![]).unwrap();
    assert_eq!(result.dtype(), DType::Float64);
}

#[test]
fn test_reduce_ops() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(1.0));

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
