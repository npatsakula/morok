//! Control flow operation tests.
//!
//! Tests control flow operations: If, EndIf, Range, End, Barrier.

use smallvec::smallvec;

use morok_dtype::DType;

use crate::{AxisType, ConstValue, Op, UOp};

// =========================================================================
// Basic If/EndIf Tests
// =========================================================================

#[test]
fn test_if_basic_construction() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let body_stmt = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    let if_op = UOp::new(Op::If { condition: condition.clone(), body: smallvec![body_stmt] }, DType::Void);

    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_if_with_comparison_condition() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(5));
    let b = UOp::const_(DType::Int32, ConstValue::Int(10));

    // Create comparison condition: a < b
    let condition = a.try_cmplt(&b).unwrap();
    assert_eq!(condition.dtype(), DType::Bool);

    let body_stmt = UOp::const_(DType::Float32, ConstValue::Float(42.0));

    let if_op = UOp::new(Op::If { condition, body: smallvec![body_stmt] }, DType::Void);

    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_if_empty_body() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(false));

    let if_op = UOp::new(Op::If { condition, body: smallvec![] }, DType::Void);

    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_if_single_statement_body() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let stmt = UOp::const_(DType::Int32, ConstValue::Int(100));

    let if_op = UOp::new(Op::If { condition, body: smallvec![stmt] }, DType::Void);

    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_if_multiple_statements() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let stmt1 = UOp::const_(DType::Int32, ConstValue::Int(1));
    let stmt2 = UOp::const_(DType::Int32, ConstValue::Int(2));
    let stmt3 = UOp::const_(DType::Int32, ConstValue::Int(3));

    let if_op = UOp::new(Op::If { condition, body: smallvec![stmt1, stmt2, stmt3] }, DType::Void);

    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_endif_basic() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let body_stmt = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    let if_op = UOp::new(Op::If { condition, body: smallvec![body_stmt] }, DType::Void);

    let endif = UOp::new(Op::EndIf { if_op: if_op.clone() }, DType::Void);

    assert_eq!(endif.dtype(), DType::Void);
}

#[test]
fn test_if_returns_void() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let body = smallvec![UOp::const_(DType::Int32, ConstValue::Int(42))];

    let if_op = UOp::new(Op::If { condition, body }, DType::Void);

    // Verify If has DType::Void
    assert_eq!(if_op.dtype(), DType::Void);
}

// =========================================================================
// Range Operation Tests
// =========================================================================

#[test]
fn test_range_global_axis() {
    let end = UOp::const_(DType::Int32, ConstValue::Int(10));

    let range_op = UOp::new(Op::Range { end, axis_id: 0, axis_type: AxisType::Global }, DType::Index);

    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_warp_axis() {
    let end = UOp::const_(DType::Int32, ConstValue::Int(32));

    let range_op = UOp::new(Op::Range { end, axis_id: 1, axis_type: AxisType::Warp }, DType::Index);

    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_local_axis() {
    let end = UOp::const_(DType::Int32, ConstValue::Int(256));

    let range_op = UOp::new(Op::Range { end, axis_id: 0, axis_type: AxisType::Local }, DType::Index);

    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_loop_axis() {
    let end = UOp::const_(DType::Int32, ConstValue::Int(100));

    let range_op = UOp::new(Op::Range { end, axis_id: 2, axis_type: AxisType::Loop }, DType::Index);

    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_reduce_axis() {
    let end = UOp::const_(DType::Int32, ConstValue::Int(1024));

    let range_op = UOp::new(Op::Range { end, axis_id: 0, axis_type: AxisType::Reduce }, DType::Index);

    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_unroll_axis() {
    let end = UOp::const_(DType::Int32, ConstValue::Int(4));

    let range_op = UOp::new(Op::Range { end, axis_id: 3, axis_type: AxisType::Unroll }, DType::Index);

    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_thread_axis() {
    let end = UOp::const_(DType::Int32, ConstValue::Int(8));

    let range_op = UOp::new(Op::Range { end, axis_id: 1, axis_type: AxisType::Thread }, DType::Index);

    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_dtype_is_index() {
    // Verify all Range operations return DType::Index
    let end = UOp::const_(DType::Int32, ConstValue::Int(10));

    let axis_types = vec![
        AxisType::Global,
        AxisType::Warp,
        AxisType::Local,
        AxisType::Loop,
        AxisType::GroupReduce,
        AxisType::Reduce,
        AxisType::Upcast,
        AxisType::Unroll,
        AxisType::Thread,
    ];

    for (idx, axis_type) in axis_types.into_iter().enumerate() {
        let range_op = UOp::new(Op::Range { end: end.clone(), axis_id: idx, axis_type }, DType::Index);

        assert_eq!(range_op.dtype(), DType::Index);
    }
}

// =========================================================================
// End Operation Tests
// =========================================================================

#[test]
fn test_end_of_range() {
    let end_val = UOp::const_(DType::Int32, ConstValue::Int(10));
    let range_op = UOp::new(Op::Range { end: end_val, axis_id: 0, axis_type: AxisType::Loop }, DType::Index);

    // Create a simple computation (NOOP)
    let computation = UOp::noop();

    // Create END with computation and ranges
    let end_op = UOp::new(Op::End { computation, ranges: smallvec![range_op] }, DType::Void);

    assert_eq!(end_op.dtype(), DType::Void);
}

#[test]
fn test_end_preserves_dtype() {
    // End operation should have DType::Void
    let end_val = UOp::const_(DType::Int32, ConstValue::Int(5));
    let range_op = UOp::new(Op::Range { end: end_val, axis_id: 0, axis_type: AxisType::Global }, DType::Index);

    // Create a simple computation (NOOP)
    let computation = UOp::noop();

    // Create END with computation and ranges
    let end_op = UOp::new(Op::End { computation, ranges: smallvec![range_op] }, DType::Void);

    assert_eq!(end_op.dtype(), DType::Void);
}

#[test]
fn test_end_returns_void() {
    let end_val = UOp::const_(DType::Int32, ConstValue::Int(100));
    let range_op = UOp::new(Op::Range { end: end_val, axis_id: 1, axis_type: AxisType::Reduce }, DType::Index);

    // Create a simple computation (NOOP)
    let computation = UOp::noop();

    // Create END with computation and ranges
    let end_op = UOp::new(Op::End { computation, ranges: smallvec![range_op] }, DType::Void);

    // Verify End has DType::Void
    assert_eq!(end_op.dtype(), DType::Void);
}

// =========================================================================
// Barrier Tests
// =========================================================================

#[test]
fn test_barrier_basic() {
    let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    let barrier = UOp::new(Op::Barrier { src: src.clone(), deps: smallvec![] }, DType::Float32);

    // Barrier preserves src dtype
    assert_eq!(barrier.dtype(), DType::Float32);
}

#[test]
fn test_barrier_with_single_dep() {
    let src = UOp::const_(DType::Int32, ConstValue::Int(42));
    let dep = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as f64));

    let barrier = UOp::new(Op::Barrier { src: src.clone(), deps: smallvec![dep] }, DType::Int32);

    assert_eq!(barrier.dtype(), DType::Int32);
}

#[test]
fn test_barrier_with_multiple_deps() {
    let src = UOp::const_(DType::Float64, ConstValue::Float(std::f64::consts::E));
    let dep1 = UOp::const_(DType::Int32, ConstValue::Int(1));
    let dep2 = UOp::const_(DType::Int32, ConstValue::Int(2));
    let dep3 = UOp::const_(DType::Int32, ConstValue::Int(3));

    let barrier = UOp::new(Op::Barrier { src: src.clone(), deps: smallvec![dep1, dep2, dep3] }, DType::Float64);

    assert_eq!(barrier.dtype(), DType::Float64);
}

#[test]
fn test_barrier_preserves_dtype() {
    // Test that Barrier preserves various dtypes
    let dtypes = vec![
        (DType::Int8, ConstValue::Int(1)),
        (DType::Int32, ConstValue::Int(100)),
        (DType::Float32, ConstValue::Float(std::f32::consts::PI as f64)),
        (DType::Float64, ConstValue::Float(std::f64::consts::E)),
        (DType::UInt32, ConstValue::UInt(42)),
    ];

    for (dtype, value) in dtypes {
        let src = UOp::const_(dtype.clone(), value);
        let barrier = UOp::new(Op::Barrier { src, deps: smallvec![] }, dtype.clone());

        assert_eq!(barrier.dtype(), dtype);
    }
}

// =========================================================================
// Nested Control Flow
// =========================================================================

#[test]
fn test_nested_if() {
    let outer_cond = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let inner_cond = UOp::const_(DType::Bool, ConstValue::Bool(false));

    let inner_body = UOp::const_(DType::Int32, ConstValue::Int(42));

    let inner_if = UOp::new(Op::If { condition: inner_cond, body: smallvec![inner_body] }, DType::Void);

    // Outer if contains inner if in its body
    let outer_if = UOp::new(Op::If { condition: outer_cond, body: smallvec![inner_if] }, DType::Void);

    assert_eq!(outer_if.dtype(), DType::Void);
}

#[test]
fn test_range_inside_if() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let range_end = UOp::const_(DType::Int32, ConstValue::Int(10));

    let range_op = UOp::new(Op::Range { end: range_end, axis_id: 0, axis_type: AxisType::Loop }, DType::Index);

    let if_op = UOp::new(Op::If { condition, body: smallvec![range_op] }, DType::Void);

    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_multiple_sequential_ranges() {
    let end1 = UOp::const_(DType::Int32, ConstValue::Int(10));
    let end2 = UOp::const_(DType::Int32, ConstValue::Int(20));
    let end3 = UOp::const_(DType::Int32, ConstValue::Int(30));

    let range1 = UOp::new(Op::Range { end: end1, axis_id: 0, axis_type: AxisType::Global }, DType::Index);

    let range2 = UOp::new(Op::Range { end: end2, axis_id: 1, axis_type: AxisType::Local }, DType::Index);

    let range3 = UOp::new(Op::Range { end: end3, axis_id: 2, axis_type: AxisType::Loop }, DType::Index);

    // All ranges should be valid
    assert_eq!(range1.dtype(), DType::Index);
    assert_eq!(range2.dtype(), DType::Index);
    assert_eq!(range3.dtype(), DType::Index);
}

// =========================================================================
// Dtype Validation
// =========================================================================

#[test]
fn test_if_dtype_is_void() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let body = smallvec![UOp::const_(DType::Float32, ConstValue::Float(1.0))];

    let if_op = UOp::new(Op::If { condition, body }, DType::Void);

    // Confirm If dtype
    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_endif_dtype_is_void() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let if_op = UOp::new(Op::If { condition, body: smallvec![] }, DType::Void);

    let endif = UOp::new(Op::EndIf { if_op }, DType::Void);

    // Confirm EndIf dtype
    assert_eq!(endif.dtype(), DType::Void);
}

#[test]
fn test_range_confirms_index_dtype() {
    let end = UOp::const_(DType::Int32, ConstValue::Int(100));

    let range_op = UOp::new(Op::Range { end, axis_id: 0, axis_type: AxisType::Loop }, DType::Index);

    // Confirm Range dtype
    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_end_dtype_is_void() {
    let end_val = UOp::const_(DType::Int32, ConstValue::Int(10));
    let range_op = UOp::new(Op::Range { end: end_val, axis_id: 0, axis_type: AxisType::Global }, DType::Index);

    // Create a simple computation (NOOP)
    let computation = UOp::noop();

    // Create END with computation and ranges
    let end_op = UOp::new(Op::End { computation, ranges: smallvec![range_op] }, DType::Void);

    // Confirm End dtype
    assert_eq!(end_op.dtype(), DType::Void);
}

#[test]
fn test_barrier_dtype_preservation() {
    // Test that Barrier preserves src dtype across different types
    let int_src = UOp::const_(DType::Int32, ConstValue::Int(42));
    let int_barrier = UOp::new(Op::Barrier { src: int_src, deps: smallvec![] }, DType::Int32);
    assert_eq!(int_barrier.dtype(), DType::Int32);

    let float_src = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as f64));
    let float_barrier = UOp::new(Op::Barrier { src: float_src, deps: smallvec![] }, DType::Float32);
    assert_eq!(float_barrier.dtype(), DType::Float32);
}
