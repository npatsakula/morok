//! Tests for do_contract (CONTRACT → GEP transformation).
//!
//! Ported from Tinygrad's TestExpander class (test_uop_graph.py:663-811).
//!
//! CONTRACT collapses UNROLL operations back to scalar/vector form:
//! - Full contraction: CONTRACT(UNROLL, same_axes) → GEP
//! - Partial contraction: CONTRACT(UNROLL, subset_axes) → UNROLL(GEP, remaining_axes)
//! - Non-UNROLL source: CONTRACT(src, axes) → VECTORIZE

use super::helpers::*;
use morok_ir::{Op, UOp};

// =============================================================================
// Full Contraction Tests
// =============================================================================

/// Test: CONTRACT(UNROLL, same axes) → VCONST
///
/// Tinygrad: test_contract_simple
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((0,4),))
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(4), (a,), ((0,4),))
/// # Result: VCONST([0,1,2,3])
/// ```
#[test]
fn test_contract_simple() {
    // Create UNROLL(VCONST([0,1,2,3]), [(0,4)])
    let unroll = create_unroll_iota(0, 4);

    // Create CONTRACT with same axes
    let contract = create_contract(unroll, vec![(0, 4)]);

    // Apply expander
    let result = phase2_only(&contract);

    // Should produce GEP that extracts to VCONST
    // After GEP optimization, result should be the original VCONST [0,1,2,3]
    // The GEP extracts indices [0,1,2,3] from VCONST[0,1,2,3]
    match result.op() {
        Op::Gep { indices, .. } => {
            assert_eq!(indices, &[0, 1, 2, 3], "GEP should extract sequential indices");
        }
        Op::VConst { values } => {
            let ints: Vec<i64> = values
                .iter()
                .map(|v| match v {
                    morok_ir::types::ConstValue::Int(i) => *i,
                    _ => panic!("Expected Int"),
                })
                .collect();
            assert_eq!(ints, vec![0, 1, 2, 3]);
        }
        other => panic!("Expected GEP or VCONST, got {:?}", other),
    }
}

// =============================================================================
// Partial Contraction Tests
// =============================================================================

/// Test: CONTRACT axis 1 of 2-axis UNROLL
///
/// Tinygrad: test_contract_axis_1
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(range(16))),), ((1,4),(2,4)))
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(4), (a,), ((1,4),))
/// # Result swizzle: [0:4] = (0, 4, 8, 12)
/// ```
#[test]
fn test_contract_partial_axis_1() {
    // Create UNROLL(VCONST([0..15]), [(1,4), (2,4)])
    let unroll = create_unroll_multi_axis(vec![(1, 4), (2, 4)]);

    // CONTRACT axis 1 only
    let contract = create_contract(unroll, vec![(1, 4)]);

    let result = phase2_only(&contract);

    // Should be UNROLL(GEP(...), [(2,4)])
    let (gep, remaining_axes) = unwrap_unroll(&result);
    assert_eq!(remaining_axes, vec![(2, 4)], "Should have axis 2 remaining");

    // GEP indices for axis 1 contraction:
    // Contracting axis 1 while keeping axis 2 variable means:
    // - Axis 2 is excluded (zeroed), axis 1 iterates 0..4
    // Expected: swizzle_args([(1,4)], [(1,4),(2,4)], [2])
    // For cargs [(1,4)]: iterates {1:0}, {1:1}, {1:2}, {1:3}
    // With axis 2 zeroed → eargs indices: 0*4+0=0, 1*4+0=4, 2*4+0=8, 3*4+0=12
    let (_, indices) = unwrap_gep(&gep);
    assert_eq!(indices, vec![0, 4, 8, 12], "GEP indices for axis 1 contraction");
}

/// Test: CONTRACT axis 2 of 2-axis UNROLL
///
/// Tinygrad: test_contract_axis_2
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(range(16))),), ((1,4),(2,4)))
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(4), (a,), ((2,4),))
/// # Result swizzle: [0:4] = (0, 1, 2, 3)
/// ```
#[test]
fn test_contract_partial_axis_2() {
    // Create UNROLL(VCONST([0..15]), [(1,4), (2,4)])
    let unroll = create_unroll_multi_axis(vec![(1, 4), (2, 4)]);

    // CONTRACT axis 2 only
    let contract = create_contract(unroll, vec![(2, 4)]);

    let result = phase2_only(&contract);

    // Should be UNROLL(GEP(...), [(1,4)])
    let (gep, remaining_axes) = unwrap_unroll(&result);
    assert_eq!(remaining_axes, vec![(1, 4)], "Should have axis 1 remaining");

    // GEP indices for axis 2 contraction (axis 1 zeroed):
    // For cargs [(2,4)]: iterates {2:0}, {2:1}, {2:2}, {2:3}
    // With axis 1 zeroed → eargs indices: 0*4+0=0, 0*4+1=1, 0*4+2=2, 0*4+3=3
    let (_, indices) = unwrap_gep(&gep);
    assert_eq!(indices, vec![0, 1, 2, 3], "GEP indices for axis 2 contraction");
}

/// Test: CONTRACT axis 2 of 4-axis UNROLL
///
/// Tinygrad: test_contract_axis_2_big
/// ```python
/// axes = ((1,2),(2,2),(3,2),(4,2))
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(range(16))),), axes)
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(2), (a,), ((2,2),))
/// # Remaining axes: ((1,2),(3,2),(4,2))
/// ```
#[test]
fn test_contract_four_axes() {
    // Create UNROLL(VCONST([0..15]), [(1,2), (2,2), (3,2), (4,2)])
    let unroll = create_unroll_multi_axis(vec![(1, 2), (2, 2), (3, 2), (4, 2)]);

    // CONTRACT axis 2 only
    let contract = create_contract(unroll, vec![(2, 2)]);

    let result = phase2_only(&contract);

    // Should have remaining axes [(1,2), (3,2), (4,2)]
    let (_, remaining_axes) = unwrap_unroll(&result);
    assert_eq!(remaining_axes, vec![(1, 2), (3, 2), (4, 2)], "Should have axes 1, 3, 4 remaining");
}

// =============================================================================
// Multi-Axis Contraction Tests
// =============================================================================

/// Test: CONTRACT 2 axes at once (order 1)
///
/// Tinygrad: test_contract_multi_axis (first sub-test)
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(8), tuple(range(8))),), ((1,2),(2,2),(3,2)))
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(4), (a,), ((1,2),(2,2)))
/// ```
#[test]
fn test_contract_multi_axis_order_1() {
    // Create UNROLL(VCONST([0..7]), [(1,2), (2,2), (3,2)])
    let unroll = create_unroll_multi_axis(vec![(1, 2), (2, 2), (3, 2)]);

    // CONTRACT axes 1 and 2
    let contract = create_contract(unroll, vec![(1, 2), (2, 2)]);

    let result = phase2_only(&contract);

    // Should have remaining axis [(3,2)]
    let (_, remaining_axes) = unwrap_unroll(&result);
    assert_eq!(remaining_axes, vec![(3, 2)], "Should have axis 3 remaining");
}

/// Test: CONTRACT 2 axes at once (order 2)
///
/// Tinygrad: test_contract_multi_axis (second sub-test)
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(8), tuple(range(8))),), ((1,2),(2,2),(3,2)))
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(4), (a,), ((2,2),(3,2)))
/// ```
#[test]
fn test_contract_multi_axis_order_2() {
    // Create UNROLL(VCONST([0..7]), [(1,2), (2,2), (3,2)])
    let unroll = create_unroll_multi_axis(vec![(1, 2), (2, 2), (3, 2)]);

    // CONTRACT axes 2 and 3
    let contract = create_contract(unroll, vec![(2, 2), (3, 2)]);

    let result = phase2_only(&contract);

    // Should have remaining axis [(1,2)]
    let (_, remaining_axes) = unwrap_unroll(&result);
    assert_eq!(remaining_axes, vec![(1, 2)], "Should have axis 1 remaining");
}

/// Test: CONTRACT middle axis
///
/// Tinygrad: test_contract_mid
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(8), tuple(range(8))),), ((1,2),(2,2),(3,2)))
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(2), (a,), ((2,2),))
/// # Swizzle: (0, 2, 1, 3, 4, 6, 5, 7)
/// ```
#[test]
fn test_contract_middle_axis() {
    // Create UNROLL(VCONST([0..7]), [(1,2), (2,2), (3,2)])
    let unroll = create_unroll_multi_axis(vec![(1, 2), (2, 2), (3, 2)]);

    // CONTRACT middle axis 2
    let contract = create_contract(unroll, vec![(2, 2)]);

    let result = phase2_only(&contract);

    // Should have remaining axes [(1,2), (3,2)]
    let (gep, remaining_axes) = unwrap_unroll(&result);
    assert_eq!(remaining_axes, vec![(1, 2), (3, 2)], "Should have axes 1, 3 remaining");

    // GEP indices: swizzle_args([(2,2)], [(1,2),(2,2),(3,2)], [1,3])
    // Contract axis 2, exclude axes 1 and 3 (zeroed)
    let (_, indices) = unwrap_gep(&gep);
    assert_eq!(indices, vec![0, 2], "GEP indices for middle axis contraction");
}

// =============================================================================
// Non-UNROLL Source Tests
// =============================================================================

/// Test: CONTRACT without UNROLL source
///
/// Tinygrad: test_contract_no_expand
/// ```python
/// a = UOp.const(dtypes.int, 4)
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(4), (a,), ((0,4),))
/// # Result: VECTORIZE([4, 4, 4, 4])
/// ```
#[test]
fn test_contract_non_unroll_source() {
    // Create a simple scalar constant
    let scalar = UOp::const_(morok_dtype::DType::Int64, morok_ir::types::ConstValue::Int(4));

    // CONTRACT without UNROLL source
    let contract = create_contract(scalar, vec![(0, 4)]);

    let result = phase2_only(&contract);

    // Should produce VECTORIZE with 4 copies
    assert_is_vectorize(&result, 4);
}

/// Test: CONTRACT with partial expansion (extra axis)
///
/// Tinygrad: test_contract_half_expand
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((1,4),))
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(8), (a,), ((0,2),(1,4)))
/// # Expected: duplication pattern because axis 0 not in UNROLL
/// ```
#[test]
fn test_contract_partial_expansion() {
    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let unroll = create_unroll_iota(1, 4);

    // CONTRACT with extra axis 0 that's not in UNROLL
    let contract = create_contract(unroll, vec![(0, 2), (1, 4)]);

    let result = phase2_only(&contract);

    // Axis 0 is not in UNROLL's axes, so contraction should still work
    // but axis 0 gets zeroed (it defaults to 0 in swizzle_args)
    // swizzle_args([(0,2),(1,4)], [(1,4)], [])
    // cargs iterate: (0,0)..(0,3), (1,0)..(1,3)
    // eargs only have axis 1, so axis 0 defaults to 0
    // Result indices: 0,1,2,3,0,1,2,3 (duplicated because axis 0 not in source)
    match result.op() {
        Op::Gep { indices, .. } => {
            assert_eq!(indices, &[0, 1, 2, 3, 0, 1, 2, 3], "Should duplicate for missing axis");
        }
        other => panic!("Expected GEP, got {:?}", other),
    }
}

// =============================================================================
// Dtype Validation Tests
// =============================================================================

/// Test: Partial contraction dtype matches remaining axes product.
///
/// Bug fix: UNROLL wrapper dtype should match remaining_axes product,
/// not the CONTRACT axes product.
#[test]
fn test_contract_partial_dtype_validation() {
    // UNROLL with axes [(1,4), (2,2)] → 8 elements
    // CONTRACT with axes [(1,4)] → contract dtype vec4
    // remaining_axes = [(2,2)] → remaining_product = 2
    let unroll = create_unroll_multi_axis(vec![(1, 4), (2, 2)]);
    let contract = create_contract(unroll, vec![(1, 4)]);

    let result = phase2_only(&contract);

    // Validate dtype matches remaining axes product (2), not contract axes product (4)
    assert_vcount(&result, 2);

    let (_, remaining_axes) = unwrap_unroll(&result);
    assert_eq!(remaining_axes, vec![(2, 2)]);
}

/// Test: Partial contraction with equal axis sizes validates dtype correctly.
///
/// This test would pass even with the bug since contract_product == remaining_product.
/// Included for completeness.
#[test]
fn test_contract_partial_dtype_same_sizes() {
    // UNROLL with axes [(1,4), (2,4)] → 16 elements
    // CONTRACT with axes [(1,4)] → contract dtype vec4
    // remaining_axes = [(2,4)] → remaining_product = 4
    let unroll = create_unroll_multi_axis(vec![(1, 4), (2, 4)]);
    let contract = create_contract(unroll, vec![(1, 4)]);

    let result = phase2_only(&contract);

    // Both products are 4, so this validates correctly with or without the fix
    assert_vcount(&result, 4);

    let (_, remaining_axes) = unwrap_unroll(&result);
    assert_eq!(remaining_axes, vec![(2, 4)]);
}

/// Test: Void dtype is preserved for STORE-like operations.
#[test]
fn test_contract_void_dtype_preserved() {
    use super::helpers::create_contract_void;
    use super::helpers::create_unroll_multi_axis_with_dtype;
    use morok_dtype::DType;

    // UNROLL with Void dtype (like STORE)
    let unroll = create_unroll_multi_axis_with_dtype(vec![(1, 4), (2, 4)], DType::Void);
    let contract = create_contract_void(unroll, vec![(1, 4)]);

    let result = phase2_only(&contract);

    // Void dtype should be preserved
    assert_eq!(result.dtype(), DType::Void);

    let (_, remaining_axes) = unwrap_unroll(&result);
    assert_eq!(remaining_axes, vec![(2, 4)]);
}

/// Test: Full contraction uses output dtype (scalar).
#[test]
fn test_contract_full_uses_output_dtype() {
    // UNROLL with axes [(1,4)]
    // CONTRACT with same axes [(1,4)] → full contraction
    let unroll = create_unroll_iota(1, 4);
    let contract = create_contract(unroll, vec![(1, 4)]);

    let result = phase2_only(&contract);

    // Full contraction produces GEP (no UNROLL wrapper)
    // The result should be the vectorized dtype from CONTRACT
    assert_vcount(&result, 4);
}
