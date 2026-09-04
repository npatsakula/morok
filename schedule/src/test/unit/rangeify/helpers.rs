use std::sync::Arc;

use svod_ir::{BinaryOp, BufferizeOpts, ConstValue, DType, Op, UOp};
use test_case::test_case;

use crate::rangeify::indexing::{get_const_value, is_const, is_identity_value, is_zero_value};
use svod_ir::ops;

/// Count occurrences of ops matching a predicate in a UOp graph.
///
/// Recursively traverses the graph and counts all UOps where `predicate` returns true.
pub fn count_ops<F>(uop: &Arc<UOp>, predicate: F) -> usize
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

/// Count CALL operations in a UOp graph.
pub fn count_kernels(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |op| matches!(op, Op::Call(..)))
}

/// Extract the first CALL from a pipeline result.
///
/// The kernel split pipeline may return:
/// - CALL directly
/// - AFTER(DEFINE_GLOBAL, [END(CALL)])
/// - SINK([AFTER(...)])
///
/// This helper extracts the first CALL found in any of these structures.
pub fn extract_kernel(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    match uop.op() {
        // Direct callable wrapper
        Op::Call(..) => Some(uop.clone()),
        // AFTER(passthrough, deps) - check deps for END(CALL)
        Op::After(ops::After { deps, .. }) => {
            for dep in deps.iter() {
                if let Op::End(ops::End { computation, .. }) = dep.op()
                    && matches!(computation.op(), Op::Call(..))
                {
                    return Some(computation.clone());
                }
                // Also check if dep is directly a callable wrapper
                if matches!(dep.op(), Op::Call(..)) {
                    return Some(dep.clone());
                }
            }
            None
        }
        // SINK - check sources
        Op::Sink(ops::Sink { sources, .. }) => {
            for src in sources.iter() {
                if let Some(kernel) = extract_kernel(src) {
                    return Some(kernel);
                }
            }
            None
        }
        // END(CALL)
        Op::End(ops::End { computation, .. }) if matches!(computation.op(), Op::Call(..)) => Some(computation.clone()),
        _ => None,
    }
}

/// Count codegen PARAM operations (device: None) in a UOp graph.
pub fn count_codegen_params(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |op| matches!(op, Op::Param(ops::Param { arg, .. }) if arg.device.is_none()))
}

/// Count DEFINE_LOCAL operations in a UOp graph.
pub fn count_define_locals(uop: &Arc<UOp>) -> usize {
    count_ops(
        uop,
        |op| matches!(op, Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_dtype::AddrSpace::Local)),
    )
}

/// Count STORE operations in a UOp graph.
pub fn count_stores(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |op| matches!(op, Op::Store(..)))
}

/// Count END operations in a UOp graph.
pub fn count_ends(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |op| matches!(op, Op::End(..)))
}

/// Count STAGE operations in a UOp graph.
pub fn count_bufferizes(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |op| matches!(op, Op::Stage(..)))
}

// ============================================================================
// Test UOp Construction Helpers
// ============================================================================

/// Create a constant UOp with the given value.
pub fn create_const(val: i64) -> Arc<UOp> {
    UOp::index_const(val)
}

/// Create a RANGE operation with constant end value.
pub fn create_range(end: i64, axis_id: usize) -> Arc<UOp> {
    UOp::range_const(end, axis_id)
}

/// Create a RANGE operation with symbolic end value.
pub fn create_range_symbolic(end: Arc<UOp>, axis_id: usize) -> Arc<UOp> {
    UOp::range(end, axis_id)
}

/// Create a STAGE operation with global address space.
pub fn create_bufferize(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::stage_global(compute, ranges)
}

/// Create a STAGE operation with custom options.
pub fn create_bufferize_opts(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>, opts: BufferizeOpts) -> Arc<UOp> {
    UOp::stage(compute, ranges, opts)
}

// ============================================================================
// Tests for the indexing helpers these builders feed
// ============================================================================

/// `is_identity_value` is per-operator and side-aware: `-` and `//` have a right
/// identity only.
#[test_case(ConstValue::Int(0), BinaryOp::Add, false, true ; "zero is a left add identity")]
#[test_case(ConstValue::Int(0), BinaryOp::Add, true, true ; "zero is a right add identity")]
#[test_case(ConstValue::Float(0.0), BinaryOp::Add, false, true ; "float zero is an add identity")]
#[test_case(ConstValue::Int(1), BinaryOp::Mul, false, true ; "one is a left mul identity")]
#[test_case(ConstValue::Int(1), BinaryOp::Mul, true, true ; "one is a right mul identity")]
#[test_case(ConstValue::Float(1.0), BinaryOp::Mul, false, true ; "float one is a mul identity")]
#[test_case(ConstValue::Int(0), BinaryOp::Sub, false, false ; "sub has no left identity")]
#[test_case(ConstValue::Int(0), BinaryOp::Sub, true, true ; "sub has a right identity")]
#[test_case(ConstValue::Int(1), BinaryOp::FloorDiv, false, false ; "div has no left identity")]
#[test_case(ConstValue::Int(1), BinaryOp::FloorDiv, true, true ; "div has a right identity")]
#[test_case(ConstValue::Int(2), BinaryOp::Add, false, false ; "two is not an add identity")]
#[test_case(ConstValue::Int(0), BinaryOp::Mul, false, false ; "zero is not a mul identity")]
fn identity_values(value: ConstValue, op: BinaryOp, right: bool, expected: bool) {
    assert_eq!(is_identity_value(&value, &op, right), expected);
}

/// `is_zero_value` is the absorbing element, not the literal zero.
#[test_case(ConstValue::Int(0), BinaryOp::Mul, true ; "zero absorbs mul")]
#[test_case(ConstValue::Float(0.0), BinaryOp::Mul, true ; "float zero absorbs mul")]
#[test_case(ConstValue::Int(0), BinaryOp::And, true ; "zero absorbs and")]
#[test_case(ConstValue::Int(1), BinaryOp::Mul, false ; "one does not absorb mul")]
#[test_case(ConstValue::Int(0), BinaryOp::Add, false ; "zero does not absorb add")]
fn zero_values(value: ConstValue, op: BinaryOp, expected: bool) {
    assert_eq!(is_zero_value(&value, &op), expected);
}

#[test]
fn only_constants_have_a_const_value() {
    let c = UOp::native_const(42i32);
    assert_eq!(get_const_value(&c), Some(ConstValue::Int(42)));
    assert!(is_const(&c, &ConstValue::Int(42)));
    assert!(!is_const(&c, &ConstValue::Int(0)));

    assert_eq!(get_const_value(&UOp::param(0, 1, DType::Float32, None)), None);
}
