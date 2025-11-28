use std::rc::Rc;

use morok_ir::{BinaryOp, BufferizeOpts, ConstValue, DType, Op, UOp};

use crate::rangeify::helpers::{get_const_value, is_const, is_identity_value, is_zero_value};

/// Count occurrences of ops matching a predicate in a UOp graph.
///
/// Recursively traverses the graph and counts all UOps where `predicate` returns true.
pub fn count_ops<F>(uop: &Rc<UOp>, predicate: F) -> usize
where
    F: Fn(&Op) -> bool + Copy,
{
    let mut count = if predicate(uop.op()) { 1 } else { 0 };

    // Count in all source UOps
    for src in uop.op().sources() {
        count += count_ops(&src, predicate);
    }

    count
}

/// Count KERNEL operations in a UOp graph.
pub fn count_kernels(uop: &Rc<UOp>) -> usize {
    count_ops(uop, |op| matches!(op, Op::Kernel { .. }))
}

/// Count DEFINE_GLOBAL operations in a UOp graph.
pub fn count_define_globals(uop: &Rc<UOp>) -> usize {
    count_ops(uop, |op| matches!(op, Op::DefineGlobal(_)))
}

/// Count DEFINE_LOCAL operations in a UOp graph.
pub fn count_define_locals(uop: &Rc<UOp>) -> usize {
    count_ops(uop, |op| matches!(op, Op::DefineLocal(_)))
}

/// Count STORE operations in a UOp graph.
pub fn count_stores(uop: &Rc<UOp>) -> usize {
    count_ops(uop, |op| matches!(op, Op::Store { .. }))
}

/// Count END operations in a UOp graph.
pub fn count_ends(uop: &Rc<UOp>) -> usize {
    count_ops(uop, |op| matches!(op, Op::End { .. }))
}

/// Count BUFFERIZE operations in a UOp graph.
pub fn count_bufferizes(uop: &Rc<UOp>) -> usize {
    count_ops(uop, |op| matches!(op, Op::Bufferize { .. }))
}

// ============================================================================
// Test UOp Construction Helpers
// ============================================================================

/// Create a constant UOp with the given value.
pub fn create_const(val: i64) -> Rc<UOp> {
    UOp::const_(DType::Index, ConstValue::Int(val))
}

/// Create a RANGE operation with constant end value.
pub fn create_range(end: i64, axis_id: usize) -> Rc<UOp> {
    UOp::range_const(end, axis_id)
}

/// Create a RANGE operation with symbolic end value.
pub fn create_range_symbolic(end: Rc<UOp>, axis_id: usize) -> Rc<UOp> {
    UOp::range(end, axis_id)
}

/// Create a BUFFERIZE operation with global address space.
pub fn create_bufferize(compute: Rc<UOp>, ranges: Vec<Rc<UOp>>) -> Rc<UOp> {
    UOp::bufferize_global(compute, ranges)
}

/// Create a BUFFERIZE operation with custom options.
pub fn create_bufferize_opts(compute: Rc<UOp>, ranges: Vec<Rc<UOp>>, opts: BufferizeOpts) -> Rc<UOp> {
    UOp::bufferize(compute, ranges, opts)
}

// ============================================================================
// Helper Function Tests
// ============================================================================

#[test]
fn test_is_identity_value() {
    // Add identity
    assert!(is_identity_value(&ConstValue::Int(0), &BinaryOp::Add, false));
    assert!(is_identity_value(&ConstValue::Int(0), &BinaryOp::Add, true));
    assert!(is_identity_value(&ConstValue::Float(0.0), &BinaryOp::Add, false));

    // Mul identity
    assert!(is_identity_value(&ConstValue::Int(1), &BinaryOp::Mul, false));
    assert!(is_identity_value(&ConstValue::Int(1), &BinaryOp::Mul, true));
    assert!(is_identity_value(&ConstValue::Float(1.0), &BinaryOp::Mul, false));

    // Sub only has right identity
    assert!(!is_identity_value(&ConstValue::Int(0), &BinaryOp::Sub, false));
    assert!(is_identity_value(&ConstValue::Int(0), &BinaryOp::Sub, true));

    // Idiv only has right identity
    assert!(!is_identity_value(&ConstValue::Int(1), &BinaryOp::Idiv, false));
    assert!(is_identity_value(&ConstValue::Int(1), &BinaryOp::Idiv, true));

    // Non-identity values
    assert!(!is_identity_value(&ConstValue::Int(2), &BinaryOp::Add, false));
    assert!(!is_identity_value(&ConstValue::Int(0), &BinaryOp::Mul, false));
}

#[test]
fn test_is_zero_value() {
    // Mul zero
    assert!(is_zero_value(&ConstValue::Int(0), &BinaryOp::Mul));
    assert!(is_zero_value(&ConstValue::Float(0.0), &BinaryOp::Mul));

    // And zero
    assert!(is_zero_value(&ConstValue::Int(0), &BinaryOp::And));

    // Non-zero values
    assert!(!is_zero_value(&ConstValue::Int(1), &BinaryOp::Mul));
    assert!(!is_zero_value(&ConstValue::Int(0), &BinaryOp::Add));
}

#[test]
fn test_get_const_value() {
    let c = UOp::const_(DType::Int32, ConstValue::Int(42));
    assert_eq!(get_const_value(&c), Some(ConstValue::Int(42)));

    let x = UOp::define_global(0, DType::Float32);
    assert_eq!(get_const_value(&x), None);
}

#[test]
fn test_is_const() {
    let c = UOp::const_(DType::Int32, ConstValue::Int(42));
    assert!(is_const(&c, &ConstValue::Int(42)));
    assert!(!is_const(&c, &ConstValue::Int(0)));
}
