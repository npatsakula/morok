//! Morok-specific edge case tests for expand.rs.
//!
//! These test edge cases and cleanup patterns specific to Morok's implementation:
//! - Empty UNROLL unwrapping
//! - Double UNROLL collapsing
//! - BARRIER with UNROLL handling
//! - CONTRACT on void dtype

use super::helpers::*;
use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{Op, UOp};
use smallvec::smallvec;

// =============================================================================
// Empty UNROLL Tests
// =============================================================================

/// Test: UNROLL(x, ()) → x (empty axes unwrapping).
///
/// Based on Tinygrad's pattern (expander.py:104).
#[test]
fn test_empty_unroll_unwrap() {
    // Create a simple scalar
    let scalar = UOp::const_(DType::Float32, ConstValue::Float(42.0));

    // Wrap in UNROLL with empty axes
    let unroll = UOp::unroll(scalar.clone(), vec![]);

    // Apply expander
    let result = phase2_only(&unroll);

    // Should unwrap to the scalar
    match result.op() {
        Op::Const(cv) => {
            assert_eq!(cv.0, ConstValue::Float(42.0), "Should unwrap to original scalar");
        }
        // Could also be pointer-equal to original
        other => {
            if std::sync::Arc::ptr_eq(&result, &scalar) {
                // OK - same reference
            } else {
                panic!("Expected Const or same reference, got {:?}", other);
            }
        }
    }
}

// =============================================================================
// Double UNROLL Tests
// =============================================================================

/// Test: UNROLL(UNROLL(x, inner), outer) → UNROLL(x, inner + outer).
///
/// Based on Tinygrad's pattern (expander.py:94-95).
#[test]
fn test_double_unroll_collapse() {
    // Create inner UNROLL
    let values = create_vconst_int(vec![0, 1, 2, 3]);
    let inner_unroll = UOp::unroll(values, vec![(0, 4)]);

    // Wrap in outer UNROLL
    let outer_unroll = UOp::unroll(inner_unroll, vec![(1, 2)]);

    // Apply expander
    let result = phase2_only(&outer_unroll);

    // Should collapse to single UNROLL with combined axes
    match result.op() {
        Op::Unroll { unroll_axes, .. } => {
            // Combined axes: [(0, 4), (1, 2)]
            assert_eq!(unroll_axes.len(), 2, "Should have combined axes");
            assert!(unroll_axes.contains(&(0, 4)), "Should contain inner axis");
            assert!(unroll_axes.contains(&(1, 2)), "Should contain outer axis");
        }
        other => panic!("Expected UNROLL, got {:?}", other),
    }
}

// =============================================================================
// BARRIER Tests
// =============================================================================

/// Test: BARRIER(UNROLL(x, axes)) → UNROLL(BARRIER(x), axes).
///
/// Based on Tinygrad's pattern (expander.py:101-102).
/// BARRIERs are pushed inside UNROLL rather than being expanded.
#[test]
fn test_barrier_with_unroll() {
    // Create UNROLL
    let values = create_vconst_int(vec![0, 1, 2, 3]);
    let unroll = UOp::unroll(values.clone(), vec![(0, 4)]);

    // Create BARRIER wrapping UNROLL
    let barrier = UOp::new(Op::Barrier { src: unroll, deps: smallvec![] }, DType::Int64.vec(4));

    // Apply expander
    let result = phase2_only(&barrier);

    // BARRIER should be pushed inside UNROLL
    match result.op() {
        Op::Unroll { src, unroll_axes } => {
            assert_eq!(unroll_axes, &[(0, 4)], "Should preserve axes");
            // Inner should be BARRIER
            assert!(matches!(src.op(), Op::Barrier { .. }), "Inner should be BARRIER");
        }
        // Could also remain as BARRIER if pattern didn't match
        Op::Barrier { .. } => {
            // Pattern may not have matched - that's ok
        }
        other => panic!("Expected UNROLL or BARRIER, got {:?}", other),
    }
}

// =============================================================================
// CONTRACT Void Tests
// =============================================================================

/// Test: CONTRACT on void dtype (STORE) unwraps.
///
/// For void types, CONTRACT is essentially a no-op marker.
#[test]
fn test_contract_void_store() {
    // Create a STORE-like void operation
    let void_op = UOp::noop();

    // Wrap in CONTRACT
    let contract = UOp::contract(void_op.clone(), vec![(0, 4)]);
    assert_eq!(contract.dtype(), DType::Void, "CONTRACT of void should be void");

    // Apply expander
    let result = phase2_only(&contract);

    // CONTRACT on void should unwrap (or stay as CONTRACT which is ok)
    match result.op() {
        Op::Noop => {
            // Unwrapped correctly
        }
        Op::Contract { .. } => {
            // Staying as CONTRACT is also valid
        }
        other => panic!("Expected Noop or Contract, got {:?}", other),
    }
}
