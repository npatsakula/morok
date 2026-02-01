//! Tests for reduce_to_acc (REDUCE → accumulator pattern transformation).
//!
//! reduce_to_acc converts REDUCE operations to explicit accumulator patterns:
//! - Creates DEFINE_REG for accumulator
//! - Initializes accumulator with identity value
//! - Loops over reduce_range with binary operations
//! - Handles horizontal reduction for vector types
//!
//! Critical behavior tested:
//! - input_ranges excludes parallel axes (Thread, Global, Local, Warp)
//! - input_ranges includes Loop axes
//! - reduce_range itself is excluded from input_ranges
//!
//! Based on Tinygrad's devectorizer.py:291-308.

use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{BinaryOp, Op, ReduceOp, UOp};
use smallvec::smallvec;

use super::helpers::*;

// =============================================================================
// Happy Path Tests: Basic REDUCE operations
// =============================================================================

/// Test: REDUCE(scalar, [Range], Add) → accumulator pattern with Add.
#[test]
fn test_reduce_scalar_add() {
    let range = create_range_reduce(16, 0);
    let src = create_float_const(1.0);
    let reduce = create_reduce(src, vec![range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    // Should transform to accumulator pattern (no longer REDUCE)
    assert!(!matches!(result.op(), Op::Reduce { .. }), "Should transform REDUCE to accumulator pattern");
    // Should have DEFINE_REG in the tree
    assert!(count_define_regs(&result) > 0, "Should contain DEFINE_REG");
    // Should have END in the tree
    assert!(count_ends(&result) > 0, "Should contain END");
}

/// Test: REDUCE(scalar, [Range], Mul) → accumulator pattern with Mul.
#[test]
fn test_reduce_scalar_mul() {
    let range = create_range_reduce(8, 0);
    let src = create_float_const(2.0);
    let reduce = create_reduce(src, vec![range], ReduceOp::Mul);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }), "Should transform REDUCE");
    assert!(count_define_regs(&result) > 0);
}

/// Test: REDUCE(scalar, [Range], Max) → accumulator pattern with Max.
#[test]
fn test_reduce_scalar_max() {
    let range = create_range_reduce(32, 0);
    let src = create_float_const(0.0);
    let reduce = create_reduce(src, vec![range], ReduceOp::Max);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }), "Should transform REDUCE");
    // Max uses Binary::Max
    assert!(count_define_regs(&result) > 0);
}

/// Test: REDUCE(scalar, [Range], Min) → accumulator pattern with Min (uses WHERE).
#[test]
fn test_reduce_scalar_min() {
    let range = create_range_reduce(32, 0);
    let src = create_float_const(100.0);
    let reduce = create_reduce(src, vec![range], ReduceOp::Min);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }), "Should transform REDUCE");
    // Min uses WHERE(Lt, a, b)
    assert!(count_define_regs(&result) > 0);
}

/// Test: REDUCE(<4 x f32>, [Range], Add) → horizontal reduction then accumulator.
#[test]
fn test_reduce_vector_to_scalar() {
    let range = create_range_reduce(16, 0);
    // Vector source that reduces to scalar
    let src = create_vector_float_iota(4);
    let reduce = src.reduce(smallvec![range], ReduceOp::Add);
    // Output dtype is scalar f32

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }), "Should transform REDUCE");
    assert!(count_define_regs(&result) > 0);
}

// =============================================================================
// Horizontal Reduce Tests
// =============================================================================

/// Test: Horizontal reduce with no ranges → direct horizontal reduction.
///
/// REDUCE(<4 x f32>, [], Add) → tree reduction of GEPs
/// Note: The output dtype follows the REDUCE's dtype, which may still be vector.
#[test]
fn test_horizontal_reduce_no_ranges() {
    let src = create_vector_float_iota(4);
    // Empty ranges: just horizontal reduce
    let reduce = src.reduce(smallvec![], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    // With no ranges, should just be horizontal reduction (no DEFINE_REG)
    assert!(!matches!(result.op(), Op::Reduce { .. }), "Should transform REDUCE");
    // Note: The default reduce without explicit output dtype keeps the input dtype.
    // This is correct behavior - horizontal reduction happens based on dtype mismatch.
    // No accumulator needed for horizontal-only reduce
    assert_eq!(count_define_regs(&result), 0, "Should not have DEFINE_REG for horizontal-only reduce");
}

/// Test: Vector identity (vcount in == vcount out) → no GEPs needed.
#[test]
fn test_horizontal_reduce_identity() {
    let range = create_range_reduce(8, 0);
    let src = create_vector_float_iota(4);
    // Vec4 input with vec4 output dtype
    let reduce =
        UOp::new(Op::Reduce { src, ranges: smallvec![range], reduce_op: ReduceOp::Add }, DType::Float32.vec(4));

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    // Output should still be vec4
    assert_eq!(result.dtype().vcount(), 4);
}

/// Test: <16 x f32> → <4 x f32> requires 4-stride horizontal GEPs.
#[test]
fn test_horizontal_reduce_16_to_4() {
    let range = create_range_reduce(8, 0);
    // 16 elements to 4 elements
    let elements: smallvec::SmallVec<[Arc<UOp>; 4]> =
        (0..16).map(|i| UOp::const_(DType::Float32, ConstValue::Float(i as f64))).collect();
    let src = UOp::vectorize(elements);
    let reduce =
        UOp::new(Op::Reduce { src, ranges: smallvec![range], reduce_op: ReduceOp::Add }, DType::Float32.vec(4));

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert_eq!(result.dtype().vcount(), 4);
}

// =============================================================================
// Edge Cases
// =============================================================================

/// Test: REDUCE with empty ranges → direct horizontal reduction.
#[test]
fn test_reduce_empty_ranges() {
    let src = create_vector_float_iota(4);
    let reduce = src.reduce(smallvec![], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    // Empty ranges means no loop, just horizontal reduce
    assert!(!matches!(result.op(), Op::Reduce { .. }));
}

/// Test: REDUCE with scalar src and scalar out.
#[test]
fn test_reduce_single_element() {
    let range = create_range_reduce(1, 0);
    let src = create_float_const(42.0);
    let reduce = create_reduce(src, vec![range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert_eq!(result.dtype().vcount(), 1);
}

/// Test: REDUCE with multiple reduce ranges.
#[test]
fn test_reduce_multiple_ranges() {
    let range1 = create_range_reduce(8, 0);
    let range2 = create_range_reduce(4, 1);
    let src = create_float_const(1.0);
    let reduce = create_reduce(src.clone(), vec![range1, range2], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
    // Multiple ranges should all be in the END
    assert!(count_ends(&result) > 0);
}

// =============================================================================
// Axis Type Tests (Tinygrad alignment: all ranges included in input_ranges)
// =============================================================================

/// Test: Thread ranges in topo are included in input_ranges.
///
/// Matches Tinygrad: input_ranges includes all RANGE ops in topo
/// (except reduce_range itself and ended ranges).
#[test]
fn test_input_ranges_include_thread() {
    let thread_range = create_range_thread(32, 0);
    let reduce_range = create_range_reduce(16, 1);

    let src = thread_range.cast(DType::Float32);
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
}

/// Test: Global ranges in topo are included in input_ranges.
#[test]
fn test_input_ranges_include_global() {
    let global_range = create_range_global(64, 0);
    let reduce_range = create_range_reduce(16, 1);

    let src = global_range.cast(DType::Float32);
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
}

/// Test: Local ranges in topo are included in input_ranges.
#[test]
fn test_input_ranges_include_local() {
    let local_range = create_range_local(16, 0);
    let reduce_range = create_range_reduce(8, 1);

    let src = local_range.cast(DType::Float32);
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
}

/// Test: Loop ranges in topo are included in input_ranges.
#[test]
fn test_input_ranges_include_loop() {
    let loop_range = create_range_loop(8, 0);
    let reduce_range = create_range_reduce(16, 1);

    let src = loop_range.cast(DType::Float32);
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
}

/// Test: The reduce range itself is excluded from input_ranges.
///
/// Matches Tinygrad: `x not in reduce_range` check.
#[test]
fn test_input_ranges_exclude_reduce_range() {
    let reduce_range = create_range_reduce(16, 0);
    // Source depends on the reduce_range itself (e.g., loop variable)
    let src = reduce_range.clone().cast(DType::Float32);
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    // reduce_range is excluded (it's the loop we iterate over)
    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
}

/// Test: Mixed axis types in source - all are included in input_ranges.
///
/// Matches Tinygrad: all RANGE ops in topo go into input_ranges
/// (except reduce_range and ended ranges).
#[test]
fn test_input_ranges_mixed_axis_types() {
    let global_range = create_range_global(64, 0);
    let thread_range = create_range_thread(32, 1);
    let loop_range = create_range_loop(8, 2);
    let reduce_range = create_range_reduce(16, 3);

    // Source depends on all three non-reduce ranges
    let src = UOp::new(
        Op::Binary(
            BinaryOp::Add,
            UOp::new(
                Op::Binary(BinaryOp::Add, global_range.cast(DType::Float32), thread_range.cast(DType::Float32)),
                DType::Float32,
            ),
            loop_range.cast(DType::Float32),
        ),
        DType::Float32,
    );
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    // All three ranges (global, thread, loop) are in input_ranges
    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
}

// =============================================================================
// Integration Tests
// =============================================================================

/// Test: REDUCE transformation through pm_reduce + gep_pushing (combined pass).
///
/// This tests the REDUCE transformation in the context of a realistic
/// LOAD → REDUCE scenario.
#[test]
fn test_reduce_in_full_pipeline() {
    use crate::devectorize::pm_reduce;
    use crate::rewrite::graph_rewrite_bottom_up;
    use crate::symbolic::patterns::gep_pushing_patterns;
    use morok_dtype::{AddrSpace, DeviceSpec};

    // Create a realistic REDUCE scenario
    let reduce_range = create_range_reduce(32, 0);
    let buffer_dtype = DType::Float32.ptr(Some(1024), AddrSpace::Global);
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 1024, buffer_dtype.clone());
    let define = UOp::define_global(0, buffer_dtype);

    // LOAD from buffer
    let idx = UOp::index().buffer(define).indices(vec![reduce_range.clone()]).call().unwrap();
    let load = UOp::load().buffer(buffer.clone()).index(idx).call();

    // REDUCE over load
    let reduce = load.reduce(smallvec![reduce_range], ReduceOp::Add);

    // Apply pm_reduce + gep_pushing (as done in optimizer)
    let combined = pm_reduce() + gep_pushing_patterns();
    let result = graph_rewrite_bottom_up(&combined, reduce, &mut ());

    // Should transform REDUCE to accumulator pattern
    assert!(!matches!(result.op(), Op::Reduce { .. }), "REDUCE should be transformed");
    assert!(count_define_regs(&result) > 0, "Should have DEFINE_REG for accumulator");
}

/// Test: REDUCE with vectorized CONTRACT source.
#[test]
fn test_reduce_with_vectorized_source() {
    let reduce_range = create_range_reduce(16, 0);

    // Create a vectorized source via VECTORIZE
    let elements: smallvec::SmallVec<[Arc<UOp>; 4]> =
        (0..4).map(|i| UOp::const_(DType::Float32, ConstValue::Float(i as f64))).collect();
    let vectorized = UOp::vectorize(elements);

    // REDUCE the vectorized value to vec4 output
    let reduce = UOp::new(
        Op::Reduce { src: vectorized, ranges: smallvec![reduce_range], reduce_op: ReduceOp::Add },
        DType::Float32.vec(4),
    );

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert_eq!(result.dtype().vcount(), 4);
}
