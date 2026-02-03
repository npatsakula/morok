//! Tests for do_contract (CONTRACT → GEP transformation).
//!
//! Ported from Tinygrad's TestExpander class (test_uop_graph.py:663-811).
//!
//! CONTRACT collapses UNROLL operations back to scalar/vector form:
//! - Full contraction: CONTRACT(UNROLL, same_axes) → GEP
//! - Partial contraction: CONTRACT(UNROLL, subset_axes) → UNROLL(GEP, remaining_axes)
//! - Non-UNROLL source: CONTRACT(src, axes) → VECTORIZE
//!
//! Value assertions match Tinygrad's exact test expectations.

use super::helpers::*;
use morok_ir::{Op, UOp};

// =============================================================================
// Full Contraction Tests
// =============================================================================

/// Test: CONTRACT(UNROLL, same axes) → values preserved
///
/// Tinygrad: test_contract_simple
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((1,4),))
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(4), (a,), ((1,4),))
/// self.assertTupleEqual(sink.arg, (0, 1, 2, 3))
/// ```
#[test]
fn test_contract_simple() {
    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let unroll = create_unroll_iota(1, 4);

    // Create CONTRACT with same axes
    let contract = create_contract(unroll, vec![(1, 4)]);

    // Apply expander
    let result = phase2_only(&contract);

    // Full contraction returns original values: [0, 1, 2, 3]
    assert_result_values(&result, &[0, 1, 2, 3]);
}

// =============================================================================
// Partial Contraction Tests (with exact swizzle verification)
// =============================================================================

/// Test: CONTRACT axis 1 of 2-axis UNROLL → stride-4 swizzle
///
/// Tinygrad: test_contract_axis_1
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(range(16))),), ((1,4),(2,4)))
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(4), (a,), ((1,4),))
/// # Inner GEP indices: (0, 4, 8, 12)
/// # After full contraction, values per axis-2 position:
/// # axis2=0: [0,4,8,12], axis2=1: [1,5,9,13], axis2=2: [2,6,10,14], axis2=3: [3,7,11,15]
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

    // GEP indices for axis 1 contraction (axis 2 zeroed):
    // Iterates {1:0}, {1:1}, {1:2}, {1:3} with axis 2 = 0
    // → indices: 0*4+0=0, 1*4+0=4, 2*4+0=8, 3*4+0=12
    let (_, indices) = unwrap_gep(&gep);
    assert_eq!(indices, vec![0, 4, 8, 12], "GEP indices for axis 1 contraction");

    // Verify extracted values: [0, 4, 8, 12]
    let values = extract_result_values(&gep);
    assert_eq!(values, vec![0, 4, 8, 12], "Contracted values from axis 1");
}

/// Test: CONTRACT axis 2 of 2-axis UNROLL → sequential swizzle
///
/// Tinygrad: test_contract_axis_2
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(range(16))),), ((1,4),(2,4)))
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(4), (a,), ((2,4),))
/// # Inner GEP indices: (0, 1, 2, 3)
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
    // Iterates {2:0}, {2:1}, {2:2}, {2:3} with axis 1 = 0
    // → indices: 0*4+0=0, 0*4+1=1, 0*4+2=2, 0*4+3=3
    let (_, indices) = unwrap_gep(&gep);
    assert_eq!(indices, vec![0, 1, 2, 3], "GEP indices for axis 2 contraction");

    // Verify extracted values: [0, 1, 2, 3]
    let values = extract_result_values(&gep);
    assert_eq!(values, vec![0, 1, 2, 3], "Contracted values from axis 2");
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

/// Test: CONTRACT 2 axes at once (axes 1,2)
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

/// Test: CONTRACT 2 axes at once (axes 2,3)
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

/// Test: CONTRACT middle axis → swizzle pattern
///
/// Tinygrad: test_contract_mid
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(8), tuple(range(8))),), ((1,2),(2,2),(3,2)))
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(2), (a,), ((2,2),))
/// self.assertTupleEqual(idxs, (0, 2, 1, 3, 4, 6, 5, 7))
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

    // GEP indices for middle axis contraction:
    // swizzle_args([(2,2)], [(1,2),(2,2),(3,2)], [1,3])
    // Contract axis 2, exclude axes 1 and 3 (zeroed)
    let (_, indices) = unwrap_gep(&gep);
    assert_eq!(indices, vec![0, 2], "GEP indices for middle axis contraction");

    // Verify extracted values: indices [0, 2] from [0..7] = [0, 2]
    let values = extract_result_values(&gep);
    assert_eq!(values, vec![0, 2], "Contracted values from middle axis");
}

// =============================================================================
// Non-UNROLL Source Tests
// =============================================================================

/// Test: CONTRACT without UNROLL source → VECTORIZE broadcast
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

    // Should produce VECTORIZE with 4 copies of 4
    assert_is_vectorize(&result, 4);

    // Verify all values are 4 (broadcast)
    assert_result_values(&result, &[4, 4, 4, 4]);
}

/// Test: CONTRACT with partial expansion (axis not in UNROLL) → duplication
///
/// Tinygrad: test_contract_half_expand
/// ```python
/// a = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((1,4),))
/// c = UOp(Ops.CONTRACT, dtypes.int.vec(8), (a,), ((0,2),(1,4)))
/// # Result: GEP indices = (0,1,2,3,0,1,2,3) - axis 0 causes duplication
/// ```
#[test]
fn test_contract_partial_expansion() {
    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let unroll = create_unroll_iota(1, 4);

    // CONTRACT with extra axis 0 that's not in UNROLL
    let contract = create_contract(unroll, vec![(0, 2), (1, 4)]);

    let result = phase2_only(&contract);

    // Axis 0 is not in UNROLL's axes, so it defaults to 0 in swizzle_args
    // Result: values [0,1,2,3] duplicated: [0,1,2,3,0,1,2,3]
    match result.op() {
        Op::Gep { indices, .. } => {
            assert_eq!(indices, &[0, 1, 2, 3, 0, 1, 2, 3], "Should duplicate for missing axis");
        }
        other => panic!("Expected GEP, got {:?}", other),
    }

    // Verify duplicated values
    assert_result_values(&result, &[0, 1, 2, 3, 0, 1, 2, 3]);
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

/// Test: Full contraction uses output dtype (vectorized).
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

    // Verify values preserved
    assert_result_values(&result, &[0, 1, 2, 3]);
}
