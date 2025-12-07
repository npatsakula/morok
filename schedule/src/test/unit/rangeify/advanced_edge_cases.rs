//! Advanced edge case tests for rangeify.
//!
//! Tests for IR-level edge cases that aren't covered by basic tests:
//! - Symbolic (variable-sized) ranges
//! - Nested BUFFERIZE operations
//! - Multi-consumer patterns
//! - Complex indexing scenarios

use morok_ir::{DType, Op, UOp};

use crate::rangeify::transform::rangeify;

use super::helpers::{create_bufferize, create_const, create_range, create_range_symbolic};

// ============================================================================
// Symbolic Range Size Tests
// ============================================================================

#[test]
fn test_symbolic_range_size() {
    // Test BUFFERIZE with symbolic (variable) range size
    // This tests that rangeify doesn't crash on non-constant range sizes

    let size_var = UOp::var("size", DType::Index, 1024);
    let compute = UOp::native_const(1.0f32);

    // Create range with symbolic size
    let range = create_range_symbolic(size_var, 0);
    let bufferized = create_bufferize(compute, vec![range]);

    // Symbolic ranges work correctly and create kernels
    let (_result, _ctx) = rangeify(bufferized, None).unwrap();

    // Note: Dead-axis optimization now works for provably-dead symbolic ranges
    // (uses vmax analysis - see test_is_dead_axis_symbolic_bounded test)
}

#[test]
fn test_symbolic_range_multiple() {
    // Test multiple symbolic ranges
    let size1 = UOp::var("size1", DType::Index, 1024);
    let size2 = UOp::var("size2", DType::Index, 1024);

    let compute = UOp::native_const(2.0f32);

    let range1 = create_range_symbolic(size1, 0);
    let range2 = create_range_symbolic(size2, 1);

    let bufferized = create_bufferize(compute.clone(), vec![range1, range2]);

    // Symbolic ranges work correctly with multiple dimensions
    let (_result, _ctx) = rangeify(bufferized, None).unwrap();

    // Note: Dead-axis optimization is skipped for symbolic ranges
    // TODO: Enhance dead-axis detection to handle provably-dead symbolic ranges
}

#[test]
fn test_symbolic_range_with_arithmetic() {
    // Test symbolic range size with arithmetic expression
    let n = UOp::var("n", DType::Index, 512);
    let size = n.try_mul(&create_const(2)).unwrap();

    let compute = UOp::native_const(3.0f32);
    let range = create_range_symbolic(size, 0);
    let bufferized = create_bufferize(compute, vec![range]);

    // Symbolic arithmetic expressions work correctly as range sizes
    let (_result, _ctx) = rangeify(bufferized, None).unwrap();

    // Note: Dead-axis optimization is skipped for symbolic ranges
    // TODO: Enhance dead-axis detection to handle provably-dead symbolic ranges
}

// ============================================================================
// Nested BUFFERIZE Tests
// ============================================================================

#[test]
fn test_nested_bufferize_different_ranges() {
    // Test BUFFERIZE(BUFFERIZE(x, R1), R2) where R1 != R2
    // This tests multi-level buffering with different iteration spaces

    let inner_compute = UOp::native_const(1.0f32);

    // Inner bufferize with range [0, 10)
    let inner_range = create_range(10, 0);
    let inner_buf = create_bufferize(inner_compute, vec![inner_range]);

    // Outer bufferize with different range [0, 20)
    let outer_range = create_range(20, 1);
    let outer_buf = create_bufferize(inner_buf, vec![outer_range]);

    // Should handle nested bufferization without crashing
    let (_result, _ctx) = rangeify(outer_buf, None).unwrap();

    // Note: Tests robustness - nested BUFFERIZE operations should be handled gracefully
}

#[test]
fn test_deeply_nested_bufferize() {
    // Test 3-level nesting: BUFFERIZE(BUFFERIZE(BUFFERIZE(x)))
    let compute = UOp::native_const(1.0f32);

    let r1 = create_range(5, 0);
    let buf1 = create_bufferize(compute, vec![r1]);

    let r2 = create_range(10, 1);
    let buf2 = create_bufferize(buf1, vec![r2]);

    let r3 = create_range(15, 2);
    let buf3 = create_bufferize(buf2, vec![r3]);

    // Should handle deep nesting without crashing
    let (_result, _ctx) = rangeify(buf3, None).unwrap();

    // Note: Tests that deeply nested BUFFERIZE operations don't cause stack overflow or panics
}

// ============================================================================
// Multi-Consumer Pattern Tests
// ============================================================================

#[test]
fn test_bufferize_multiple_consumers() {
    // Test single BUFFERIZE with multiple consumers
    // Pattern: buf = BUFFERIZE(x); y = f(buf); z = g(buf)

    let compute = UOp::native_const(1.0f32);
    let range = create_range(10, 0);
    let buf = create_bufferize(compute, vec![range]);

    // Two independent consumers of the same buffer
    let consumer1 = buf.try_add(&UOp::native_const(2.0f32)).unwrap();

    let consumer2 = buf.try_mul(&UOp::native_const(3.0f32)).unwrap();

    // Combine consumers with SINK
    let sink = UOp::sink(vec![consumer1, consumer2]);

    // Should handle multi-consumer pattern without crashing
    let (_result, _ctx) = rangeify(sink, None).unwrap();

    // Note: Tests that multiple consumers of the same BUFFERIZE don't cause issues
}

#[test]
fn test_operation_with_multiple_uses() {
    // Test intermediate operation used multiple times
    // Pattern: x = CONST; buf1 = BUFFERIZE(x); buf2 = BUFFERIZE(x)

    let compute = UOp::native_const(1.0f32);

    let r1 = create_range(10, 0);
    let buf1 = create_bufferize(compute.clone(), vec![r1]);

    let r2 = create_range(20, 1);
    let buf2 = create_bufferize(compute.clone(), vec![r2]);

    // Both bufferize the same compute
    let sink = UOp::sink(vec![buf1, buf2]);

    // Should handle same operation bufferized with different ranges
    let (_result, _ctx) = rangeify(sink, None).unwrap();

    // Note: Tests that same compute can be buffered with different iteration spaces
}

// ============================================================================
// Complex Indexing Tests
// ============================================================================

#[test]
fn test_index_with_multiple_ranges() {
    // Test INDEX operation with multiple range dimensions
    let compute = UOp::native_const(1.0f32);
    let r1 = create_range(10, 0);
    let r2 = create_range(20, 1);
    let r3 = create_range(5, 2);

    let bufferized = create_bufferize(compute, vec![r1.clone(), r2.clone(), r3.clone()]);

    // Create INDEX with all three ranges
    let index_op = UOp::new(
        Op::Index { buffer: bufferized.clone(), indices: vec![r1, r2, r3].into(), gate: None },
        DType::Float32,
    );

    let (_result, _ctx) = rangeify(index_op, None).unwrap();
}

#[test]
fn test_range_size_mismatch() {
    // Test BUFFERIZE with mixed constant and symbolic range sizes
    let const_range = create_range(10, 0);
    let sym_size = UOp::define_global(0, DType::Index);
    let sym_range = create_range_symbolic(sym_size, 1);

    let compute = UOp::native_const(1.0f32);
    let bufferized = create_bufferize(compute, vec![const_range, sym_range]);

    // Mixed constant and symbolic ranges work correctly
    let (_result, _ctx) = rangeify(bufferized, None).unwrap();
}

// ============================================================================
// Dead Axis Detection Tests (is_dead_axis with vmax analysis)
// ============================================================================

#[test]
fn test_is_dead_axis_constant_ranges() {
    use crate::rangeify::helpers::is_dead_axis;

    // Dead: RANGE(0) - vmax = -1
    let range_0 = create_range(0, 0);
    assert!(is_dead_axis(&range_0));

    // Dead: RANGE(1) - vmax = 0
    let range_1 = create_range(1, 0);
    assert!(is_dead_axis(&range_1));

    // Live: RANGE(2) - vmax = 1
    let range_2 = create_range(2, 0);
    assert!(!is_dead_axis(&range_2));

    // Live: RANGE(10) - vmax = 9
    let range_10 = create_range(10, 0);
    assert!(!is_dead_axis(&range_10));
}

#[test]
fn test_is_dead_axis_symbolic_bounded() {
    use crate::rangeify::helpers::is_dead_axis;

    // Dead: variable bounded to [1, 1]
    let size = UOp::var("size", DType::Index, 1);
    let range = create_range_symbolic(size, 0);
    assert!(is_dead_axis(&range));

    // Live: variable with max > 1
    let size = UOp::var("size", DType::Index, 1024);
    let range = create_range_symbolic(size, 0);
    assert!(!is_dead_axis(&range));

    // Live: variable with min > 1 (still live range)
    let size = UOp::var("size", DType::Index, 100);
    let range = create_range_symbolic(size, 0);
    assert!(!is_dead_axis(&range));
}

#[test]
fn test_is_dead_axis_non_range() {
    use crate::rangeify::helpers::is_dead_axis;

    // Non-RANGE operations should return false
    let const_op = UOp::index_const(0);
    assert!(!is_dead_axis(&const_op));

    let add_op = const_op.try_add(&const_op).unwrap();
    assert!(!is_dead_axis(&add_op));
}

#[test]
fn test_symbolic_dead_range_smoke_test() {
    // Smoke test: verify that symbolic dead ranges don't cause crashes
    // This tests that the is_dead_axis() vmax analysis works end-to-end,
    // but doesn't validate that the optimization actually happens.
    //
    // NOTE: Full validation would require checking that the dead axis
    // is actually removed from the result (e.g., verify kernel has 1D ranges
    // instead of 2D). This would depend on dead axis elimination passes that
    // may run in later optimization stages.

    let size = UOp::var("size", DType::Index, 1); // Bounded to [1, 1] - provably dead
    let compute = UOp::native_const(1.0f32);

    // Create BUFFERIZE with dead symbolic range and live range
    let dead_range = create_range_symbolic(size, 0);
    let live_range = create_range(10, 1);

    // Clone for later assertions (create_bufferize moves the ranges)
    let dead_range_clone = dead_range.clone();
    let live_range_clone = live_range.clone();

    let bufferized = create_bufferize(compute, vec![dead_range, live_range]);

    // Rangeify should process this without errors
    let bufferized_clone = bufferized.clone();
    let (result, _ctx) = rangeify(bufferized, None).unwrap();

    // Basic smoke test: verify transformation occurred
    assert!(!std::sync::Arc::ptr_eq(&result, &bufferized_clone), "Result should be transformed");

    // Verify is_dead_axis() correctly identifies the dead range
    use crate::rangeify::helpers::is_dead_axis;
    assert!(is_dead_axis(&dead_range_clone), "var[1,1] range should be detected as dead");
    assert!(!is_dead_axis(&live_range_clone), "Range(10) should be detected as live");
}
