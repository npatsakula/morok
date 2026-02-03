//! Tests for do_expand (UNROLL propagation through operations).
//!
//! Ported from Tinygrad's TestExpander class (test_uop_graph.py:663-811).
//!
//! do_expand replicates operations that have UNROLL inputs:
//! - Broadcasts scalar operands
//! - Swizzles UNROLL operands with different axes
//! - Wraps results in UNROLL
//!
//! Value assertions match Tinygrad's exact test expectations.

use super::helpers::*;
use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::UOp;

// =============================================================================
// Broadcast Expansion Tests
// =============================================================================

/// Test: UNROLL + scalar broadcast
///
/// Tinygrad: test_expand_add_broadcast
/// ```python
/// e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((1,4),))
/// sink = expander_rewrite(e1+3)
/// self.assertTupleEqual(sink.src[0].arg, (3,4,5,6))
/// ```
#[test]
fn test_expand_add_broadcast() {
    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let unroll = create_unroll_iota(1, 4);

    // Add scalar constant 3
    let scalar = UOp::const_(DType::Int64, ConstValue::Int(3));
    let add = unroll.try_add(&scalar).unwrap();

    // Apply expander
    let result = phase2_only(&add);

    // Assert exact values like Tinygrad: (3, 4, 5, 6)
    assert_result_values(&result, &[3, 4, 5, 6]);

    // Also verify UNROLL structure
    let (_, axes) = unwrap_unroll(&result);
    assert_eq!(axes, vec![(1, 4)], "Should preserve axis");
}

// =============================================================================
// Same-Axis Expansion Tests
// =============================================================================

/// Test: Two UNROLLs with same axis
///
/// Tinygrad: test_expand_same_axis
/// ```python
/// e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((1,4),))
/// e2 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x*4 for x in range(4))),), ((1,4),))
/// sink = expander_rewrite(e1+e2)
/// self.assertTupleEqual(sink.src[0].arg, (0, 5, 10, 15))
/// ```
#[test]
fn test_expand_same_axis() {
    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let e1 = create_unroll_iota(1, 4);

    // Create UNROLL(VCONST([0,4,8,12]), [(1,4)])
    let e2 = create_unroll_scaled(1, 4, 4);

    // Add them
    let add = e1.try_add(&e2).unwrap();

    // Apply expander
    let result = phase2_only(&add);

    // Assert exact values: 0+0=0, 1+4=5, 2+8=10, 3+12=15
    assert_result_values(&result, &[0, 5, 10, 15]);

    // Verify UNROLL structure
    let (_, axes) = unwrap_unroll(&result);
    assert_eq!(axes, vec![(1, 4)], "Should preserve axis");
}

// =============================================================================
// Different-Axis Expansion Tests
// =============================================================================

/// Test: Two UNROLLs with different axes
///
/// Tinygrad: test_expand_different_axis
/// ```python
/// e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x*4 for x in range(4))),), ((1,4),))
/// e2 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((2,4),))
/// sink = expander_rewrite(e1+e2)
/// self.assertTupleEqual(sink.arg, ((1, 4), (2, 4)))
/// self.assertTupleEqual(sink.src[0].arg, tuple(range(16)))
/// ```
#[test]
fn test_expand_different_axis() {
    // Create UNROLL(VCONST([0,4,8,12]), [(1,4)])
    let e1 = create_unroll_scaled(1, 4, 4);

    // Create UNROLL(VCONST([0,1,2,3]), [(2,4)])
    let e2 = create_unroll_iota(2, 4);

    // Add them
    let add = e1.try_add(&e2).unwrap();

    // Apply expander
    let result = phase2_only(&add);

    // When combining different axes, the result expands to 4*4=16 values.
    // Values follow the pattern: axis1_val + axis2_val
    // Row-major iteration: axis 1 is outer (slower), axis 2 is inner (faster)
    // (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (1,0)=4, (1,1)=5, ...
    // = 0+0, 0+1, 0+2, 0+3, 4+0, 4+1, 4+2, 4+3, 8+0, 8+1, 8+2, 8+3, 12+0, 12+1, 12+2, 12+3
    let expected: Vec<i64> = (0..16).collect();
    assert_result_values(&result, &expected);

    // Verify axes
    let (_, axes) = unwrap_unroll(&result);
    assert_eq!(axes, vec![(1, 4), (2, 4)], "Should have both axes");
}

/// Test: Two UNROLLs with different axes (operands flipped)
///
/// Tinygrad: test_expand_different_axis_flip
/// Same values but operands reversed.
#[test]
fn test_expand_different_axis_flip() {
    // Create UNROLL(VCONST([0,1,2,3]), [(2,4)])
    let e2 = create_unroll_iota(2, 4);

    // Create UNROLL(VCONST([0,4,8,12]), [(1,4)])
    let e1 = create_unroll_scaled(1, 4, 4);

    // Add them (flipped order)
    let add = e2.try_add(&e1).unwrap();

    // Apply expander
    let result = phase2_only(&add);

    // Same result as test_expand_different_axis (addition is commutative)
    let expected: Vec<i64> = (0..16).collect();
    assert_result_values(&result, &expected);

    // Verify axes
    let (_, axes) = unwrap_unroll(&result);
    assert_eq!(axes, vec![(1, 4), (2, 4)], "Should have both axes");
}

// =============================================================================
// Three-Axis Expansion Tests
// =============================================================================

/// Test: Three UNROLLs with different axes
///
/// This extends Tinygrad's pattern to verify 3D expansion.
#[test]
fn test_expand_three_axes() {
    // Create UNROLL with axis 1 (stride 4): [0, 4, 8, 12]
    let e1 = create_unroll_scaled(1, 4, 4);

    // Create UNROLL with axis 2 (stride 1): [0, 1, 2, 3]
    let e2 = create_unroll_iota(2, 4);

    // Create UNROLL with axis 3 (stride 16): [0, 16, 32, 48]
    let e3 = create_unroll_scaled(3, 4, 16);

    // Build: e1 + e2 + e3
    let sum = e1.try_add(&e2).unwrap().try_add(&e3).unwrap();

    // Apply expander
    let result = phase2_only(&sum);

    // Result should have 4*4*4=64 elements
    // Verify axes
    let (src, axes) = unwrap_unroll(&result);
    assert_eq!(axes, vec![(1, 4), (2, 4), (3, 4)], "Should have three axes");
    assert_eq!(src.dtype().vcount(), 64, "Inner should be vec64");
}

// =============================================================================
// Multiplication Expansion Tests
// =============================================================================

/// Test: UNROLL * scalar
#[test]
fn test_expand_mul_broadcast() {
    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let unroll = create_unroll_iota(1, 4);

    // Multiply by scalar 2
    let scalar = UOp::const_(DType::Int64, ConstValue::Int(2));
    let mul = unroll.try_mul(&scalar).unwrap();

    // Apply expander
    let result = phase2_only(&mul);

    // Expected: [0*2, 1*2, 2*2, 3*2] = [0, 2, 4, 6]
    assert_result_values(&result, &[0, 2, 4, 6]);
}

/// Test: Two UNROLLs multiplied (same axis)
#[test]
fn test_expand_mul_same_axis() {
    // Create UNROLL(VCONST([1,2,3,4]), [(1,4)])
    let e1 = create_unroll_values(1, vec![1, 2, 3, 4]);

    // Create UNROLL(VCONST([1,2,3,4]), [(1,4)])
    let e2 = create_unroll_values(1, vec![1, 2, 3, 4]);

    // Multiply them
    let mul = e1.try_mul(&e2).unwrap();

    // Apply expander
    let result = phase2_only(&mul);

    // Expected: [1*1, 2*2, 3*3, 4*4] = [1, 4, 9, 16]
    assert_result_values(&result, &[1, 4, 9, 16]);
}

// =============================================================================
// Subtraction Expansion Tests
// =============================================================================

/// Test: UNROLL - scalar
#[test]
fn test_expand_sub_broadcast() {
    // Create UNROLL(VCONST([10,20,30,40]), [(1,4)])
    let unroll = create_unroll_values(1, vec![10, 20, 30, 40]);

    // Subtract scalar 5
    let scalar = UOp::const_(DType::Int64, ConstValue::Int(5));
    let sub = unroll.try_sub(&scalar).unwrap();

    // Apply expander
    let result = phase2_only(&sub);

    // Expected: [10-5, 20-5, 30-5, 40-5] = [5, 15, 25, 35]
    assert_result_values(&result, &[5, 15, 25, 35]);
}

// =============================================================================
// Mixed Operations Expansion Tests
// =============================================================================

/// Test: (UNROLL + scalar) * UNROLL
///
/// Verifies that compound expressions expand correctly.
#[test]
fn test_expand_compound_expression() {
    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let e1 = create_unroll_iota(1, 4);

    // Create UNROLL(VCONST([2,2,2,2]), [(1,4)])
    let e2 = create_unroll_values(1, vec![2, 2, 2, 2]);

    // Build: (e1 + 1) * e2 = ([0,1,2,3] + 1) * [2,2,2,2]
    let scalar = UOp::const_(DType::Int64, ConstValue::Int(1));
    let sum = e1.try_add(&scalar).unwrap();
    let result = phase2_only(&sum.try_mul(&e2).unwrap());

    // Expected: [1*2, 2*2, 3*2, 4*2] = [2, 4, 6, 8]
    assert_result_values(&result, &[2, 4, 6, 8]);
}
