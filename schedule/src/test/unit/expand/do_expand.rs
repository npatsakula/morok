//! Tests for do_expand (UNROLL propagation through operations).
//!
//! Ported from Tinygrad's TestExpander class (test_uop_graph.py:663-811).
//!
//! do_expand replicates operations that have UNROLL inputs:
//! - Broadcasts scalar operands
//! - Swizzles UNROLL operands with different axes
//! - Wraps results in UNROLL

use super::helpers::*;
use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{BinaryOp, Op, UOp};

// =============================================================================
// Broadcast Expansion Tests
// =============================================================================

/// Test: UNROLL + scalar broadcast
///
/// Tinygrad: test_expand_add_broadcast
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((0,4),))
/// b = a + 3
/// # Result VCONST: (3, 4, 5, 6)
/// ```
#[test]
#[tracing_test::traced_test]
fn test_expand_add_broadcast() {
    // Create UNROLL(VCONST([0,1,2,3]), [(0,4)])
    let unroll = create_unroll_iota(0, 4);

    // Add scalar constant 3
    let scalar = UOp::const_(DType::Int64, ConstValue::Int(3));
    let add = UOp::new(Op::Binary(BinaryOp::Add, unroll, scalar), DType::Int64);

    // Apply expander
    let result = phase2_only(&add);

    // Result should be UNROLL(VCONST([3,4,5,6]) or Binary with vectorized sources)
    // After expansion: the scalar is broadcast, the Binary operates on vectors
    match result.op() {
        Op::Unroll { src, unroll_axes } => {
            assert_eq!(unroll_axes, &[(0, 4)], "Should preserve axis");
            // Inner should be Binary with vectorized operands
            match src.op() {
                Op::Binary(BinaryOp::Add, left, right) => {
                    // Left should be VCONST([0,1,2,3]) or similar
                    // Right should be broadcast VECTORIZE([3,3,3,3])
                    assert_eq!(left.dtype().vcount(), 4, "Left operand should be vec4");
                    assert_eq!(right.dtype().vcount(), 4, "Right operand should be vec4");
                }
                // Could also be optimized to VCONST directly
                Op::VConst { values } => {
                    let ints: Vec<i64> = values
                        .iter()
                        .map(|v| match v {
                            ConstValue::Int(i) => *i,
                            _ => panic!("Expected Int"),
                        })
                        .collect();
                    assert_eq!(ints, vec![3, 4, 5, 6]);
                }
                other => panic!("Expected Binary or VConst, got {:?}", other),
            }
        }
        // If no UNROLL wrapping, check for direct result
        Op::Binary(BinaryOp::Add, _, _) => {
            // Expansion happened but lifted UNROLL
            assert_eq!(result.dtype().vcount(), 4, "Result should be vec4");
        }
        other => panic!("Expected UNROLL or Binary, got {:?}", other),
    }
}

// =============================================================================
// Same-Axis Expansion Tests
// =============================================================================

/// Test: Two UNROLLs with same axis
///
/// Tinygrad: test_expand_same_axis
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((0,4),))
/// b = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x*5 for x in range(4))),), ((0,4),))
/// c = a + b
/// # Result: (0, 6, 12, 18) = (0+0, 1+5, 2+10, 3+15)
/// ```
#[test]
fn test_expand_same_axis() {
    // Create UNROLL(VCONST([0,1,2,3]), [(0,4)])
    let unroll_a = create_unroll_iota(0, 4);

    // Create UNROLL(VCONST([0,5,10,15]), [(0,4)])
    let unroll_b = create_unroll_scaled(0, 4, 5);

    // Add them
    let add = UOp::new(Op::Binary(BinaryOp::Add, unroll_a, unroll_b), DType::Int64);

    // Apply expander
    let result = phase2_only(&add);

    // Result should be UNROLL with sum [0, 6, 12, 18]
    // (0+0=0, 1+5=6, 2+10=12, 3+15=18)
    match result.op() {
        Op::Unroll { src, unroll_axes } => {
            assert_eq!(unroll_axes, &[(0, 4)], "Should preserve axis");
            match src.op() {
                Op::Binary(BinaryOp::Add, _, _) => {
                    assert_eq!(src.dtype().vcount(), 4, "Inner binary should be vec4");
                }
                Op::VConst { values } => {
                    let ints: Vec<i64> = values
                        .iter()
                        .map(|v| match v {
                            ConstValue::Int(i) => *i,
                            _ => panic!("Expected Int"),
                        })
                        .collect();
                    assert_eq!(ints, vec![0, 6, 12, 18]);
                }
                other => panic!("Expected Binary or VConst, got {:?}", other),
            }
        }
        other => panic!("Expected UNROLL, got {:?}", other),
    }
}

// =============================================================================
// Different-Axis Expansion Tests
// =============================================================================

/// Test: Two UNROLLs with different axes
///
/// Tinygrad: test_expand_different_axis
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((1,4),))
/// b = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((2,4),))
/// c = a + b
/// # Result: axes=((1,4),(2,4)), values=(0..16)
/// ```
#[test]
fn test_expand_different_axis() {
    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let unroll_a = create_unroll_iota(1, 4);

    // Create UNROLL(VCONST([0,1,2,3]), [(2,4)])
    let unroll_b = create_unroll_iota(2, 4);

    // Add them
    let add = UOp::new(Op::Binary(BinaryOp::Add, unroll_a, unroll_b), DType::Int64);

    // Apply expander
    let result = phase2_only(&add);

    // When combining different axes, the result has both axes
    // The expansion size is 4*4=16
    match result.op() {
        Op::Unroll { src, unroll_axes } => {
            // Combined axes should be [(1,4), (2,4)] (sorted by axis id)
            assert_eq!(unroll_axes.len(), 2, "Should have two axes");
            assert!(unroll_axes.contains(&(1, 4)), "Should contain axis 1");
            assert!(unroll_axes.contains(&(2, 4)), "Should contain axis 2");
            // Inner should be vec16
            assert_eq!(src.dtype().vcount(), 16, "Inner should be vec16");
        }
        other => panic!("Expected UNROLL, got {:?}", other),
    }
}

/// Test: Two UNROLLs with different axes (operands flipped)
///
/// Tinygrad: test_expand_different_axis_flip
/// Same as above but with operand order reversed - result should be identical.
#[test]
fn test_expand_different_axis_flip() {
    // Create UNROLL(VCONST([0,1,2,3]), [(2,4)])
    let unroll_b = create_unroll_iota(2, 4);

    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let unroll_a = create_unroll_iota(1, 4);

    // Add them (flipped order vs test_expand_different_axis)
    let add = UOp::new(Op::Binary(BinaryOp::Add, unroll_b, unroll_a), DType::Int64);

    // Apply expander
    let result = phase2_only(&add);

    // Same result as test_expand_different_axis
    match result.op() {
        Op::Unroll { src, unroll_axes } => {
            assert_eq!(unroll_axes.len(), 2, "Should have two axes");
            assert!(unroll_axes.contains(&(1, 4)), "Should contain axis 1");
            assert!(unroll_axes.contains(&(2, 4)), "Should contain axis 2");
            assert_eq!(src.dtype().vcount(), 16, "Inner should be vec16");
        }
        other => panic!("Expected UNROLL, got {:?}", other),
    }
}
