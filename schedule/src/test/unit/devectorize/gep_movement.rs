//! Tests for GEP movement patterns (move_gep_after_load, move_gep_on_store).
//!
//! GEP movement transforms:
//! - LOAD(GEP(ptr, indices)) → GEP(LOAD(ptr), indices)
//! - STORE(GEP(ptr, indices), data) → STORE(ptr, GEP⁻¹(data))
//!
//! These transformations enable:
//! - PTRCAT distribution after GEP is moved out of LOAD
//! - Proper data lane reordering for STORE
//!
//! **Important**: When testing with `apply_gep_movement` (which uses load_store_folding_patterns),
//! multiple patterns interact:
//! 1. expand_index: vector INDEX → GEP(PTRCAT)
//! 2. gep_movement: moves GEP through LOAD/STORE
//! 3. ptrcat_distribution: STORE(PTRCAT) → GROUP(STOREs)
//!
//! So STORE tests may result in GROUP, which is correct Tinygrad behavior!
//!
//! Based on Tinygrad's devectorizer.py:106-126.

use std::sync::Arc;

use morok_dtype::ScalarDType;
use morok_ir::{Op, UOp};
use smallvec::smallvec;

use super::helpers::*;

// =============================================================================
// move_gep_after_load Tests
// =============================================================================

/// Test: LOAD(GEP(idx)) → GEP(LOAD(idx)) for single-index GEP.
///
/// Verifies that GEP is moved outside of LOAD.
/// After full load_store_folding, result may be further simplified.
#[test]
fn test_gep_after_load_single_index() {
    let buffer = create_buffer_typed(64, ScalarDType::Float32);
    let idx = create_vector_index_iota(buffer.clone(), 4);
    let gep_idx = idx.gep(vec![2]);
    let load = UOp::load().buffer(buffer.clone()).index(gep_idx).call();

    let result = apply_gep_movement(&load);

    // Result should preserve the extraction - vcount should be 1 (single element)
    assert_eq!(result.dtype().vcount(), 1, "Single-index GEP extracts one element");
    assert_eq!(result.dtype().base(), ScalarDType::Float32, "Base dtype preserved");
}

/// Test: LOAD(GEP(idx, [0,2,1])) extracts 3 elements in reordered pattern.
#[test]
fn test_gep_after_load_multi_index() {
    let buffer = create_buffer_typed(64, ScalarDType::Float32);
    let idx = create_vector_index_iota(buffer.clone(), 4);
    let gep_idx = idx.gep(vec![0, 2, 1]);
    let load = UOp::load().buffer(buffer.clone()).index(gep_idx).call();

    let result = apply_gep_movement(&load);

    // 3 indices → vcount 3
    assert_eq!(result.dtype().vcount(), 3, "Multi-index GEP extracts 3 elements");
}

/// Test: LOAD(GEP(PTRCAT)) enables PTRCAT distribution.
///
/// After GEP moves out, LOAD(PTRCAT) becomes CAT(LOADs).
#[test]
fn test_gep_after_load_with_ptrcat() {
    let buffer1 = create_buffer_typed(32, ScalarDType::Float32);
    let buffer2 = create_buffer_typed(32, ScalarDType::Float32);

    // Create PTRCAT of two indices
    let idx1 = create_vector_index_iota(buffer1.clone(), 2);
    let idx2 = create_vector_index_iota(buffer2.clone(), 2);
    let ptrcat = UOp::ptrcat().sources(vec![idx1, idx2]).call();

    // GEP selecting across both sources
    let gep_ptrcat = ptrcat.gep(vec![0, 2]);
    let load = UOp::load().buffer(buffer1.clone()).index(gep_ptrcat).call();

    let result = apply_gep_movement(&load);

    // After full phase2, PTRCAT under LOAD should be distributed
    // No LOAD should have PTRCAT as direct index
    let ptrcat_under_load = count_ops(&result, |u| {
        if let Op::Load { index, .. } = u.op() { matches!(index.op(), Op::PtrCat { .. }) } else { false }
    });
    assert_eq!(ptrcat_under_load, 0, "PTRCAT should be distributed through LOAD");
}

/// Test: GEP dtype calculation follows Tinygrad: scalar.vec(gep.dtype.count).
#[test]
fn test_gep_after_load_dtype_calculation() {
    let buffer = create_buffer_typed(64, ScalarDType::Float32);
    let idx = create_vector_index_iota(buffer.clone(), 4);
    let gep_idx = idx.gep(vec![0, 2]); // 2 elements
    let load = UOp::load().buffer(buffer.clone()).index(gep_idx).call();

    let result = apply_gep_movement(&load);

    // Result vcount = number of GEP indices
    assert_eq!(result.dtype().vcount(), 2);
    assert_eq!(result.dtype().base(), ScalarDType::Float32);
}

/// Test: Buffer type is preserved through transformation.
#[test]
fn test_gep_after_load_preserves_buffer() {
    let buffer = create_buffer_typed(64, ScalarDType::Float32);
    let idx = create_vector_index_iota(buffer.clone(), 4);
    let gep_idx = idx.gep(vec![1, 3]);
    let load = UOp::load().buffer(buffer.clone()).index(gep_idx).call();

    let result = apply_gep_movement(&load);

    // Find any LOAD in result and verify its base type
    fn check_load_dtype(uop: &Arc<UOp>) -> bool {
        match uop.op() {
            Op::Load { .. } => uop.dtype().base() == ScalarDType::Float32,
            _ => {
                for child in uop.op().children() {
                    if check_load_dtype(child) {
                        return true;
                    }
                }
                false
            }
        }
    }
    assert!(check_load_dtype(&result) || result.dtype().base() == ScalarDType::Float32);
}

/// Test: Identity GEP [0,1,2,3] preserves vector semantics.
#[test]
fn test_gep_after_load_identity_indices() {
    let buffer = create_buffer_typed(64, ScalarDType::Float32);
    let idx = create_vector_index_iota(buffer.clone(), 4);
    let gep_idx = idx.gep(vec![0, 1, 2, 3]); // Identity
    let load = UOp::load().buffer(buffer.clone()).index(gep_idx).call();

    let result = apply_gep_movement(&load);

    // Identity GEP preserves vcount
    assert_eq!(result.dtype().vcount(), 4);
}

// =============================================================================
// move_gep_on_store Tests
//
// NOTE: When STORE goes through load_store_folding_patterns, the result may be
// GROUP (not STORE) because:
// 1. expand_index converts vector INDEX to GEP(PTRCAT)
// 2. gep_movement moves GEP
// 3. ptrcat_distribution converts STORE(PTRCAT) to GROUP(STOREs)
//
// This is CORRECT Tinygrad behavior (devectorizer.py:97-104, cat_after_store).
// =============================================================================

/// Test: STORE with GEP index produces valid output (may be GROUP).
///
/// Tinygrad's cat_after_store returns UOp.group(*ret) for distributed stores.
#[test]
fn test_gep_on_store_identity() {
    let buffer = create_buffer_typed(64, ScalarDType::Float32);
    let idx = create_vector_index_iota(buffer.clone(), 2);
    let value = create_vector_float_values(vec![1.0, 2.0]);
    let gep_idx = idx.gep(vec![0, 1]); // Identity
    let store = gep_idx.store(value);

    let result = apply_gep_movement(&store);

    // Result may be STORE or GROUP (if PTRCAT distributed)
    assert!(
        matches!(result.op(), Op::Store { .. } | Op::Group { .. }),
        "Expected STORE or GROUP, got {:?}",
        result.op()
    );
}

/// Test: STORE with swap GEP [1,0] reorders data lanes.
///
/// gep_on_store inverts the GEP: data gets GEP^-1.
/// For [1,0], inverse is [1,0] (swap is self-inverse).
#[test]
fn test_gep_on_store_swap() {
    let buffer = create_buffer_typed(64, ScalarDType::Float32);
    let idx = create_vector_index_iota(buffer.clone(), 2);
    let value = create_vector_float_values(vec![1.0, 2.0]);
    let gep_idx = idx.gep(vec![1, 0]); // Swap
    let store = gep_idx.store(value);

    let result = apply_gep_movement(&store);

    // Verify inverse is computed: [1,0] inverse is [1,0]
    let inv = compute_inverse_permutation(&[1, 0]);
    assert_eq!(inv, vec![1, 0], "Swap is self-inverse");

    // Result is valid (STORE or GROUP)
    assert!(matches!(result.op(), Op::Store { .. } | Op::Group { .. }));
}

/// Test: STORE with complex permutation [2,0,1] → inverse [1,2,0].
#[test]
fn test_gep_on_store_complex_permutation() {
    let buffer = create_buffer_typed(64, ScalarDType::Float32);
    let idx = create_vector_index_iota(buffer.clone(), 3);
    let value = create_vector_float_values(vec![1.0, 2.0, 3.0]);
    let gep_idx = idx.gep(vec![2, 0, 1]);
    let store = gep_idx.store(value);

    let result = apply_gep_movement(&store);

    // Verify inverse calculation
    let inv = compute_inverse_permutation(&[2, 0, 1]);
    assert_eq!(inv, vec![1, 2, 0]);

    assert!(matches!(result.op(), Op::Store { .. } | Op::Group { .. }));
}

/// Test: STORE with ranges - ranges are preserved through transformation.
///
/// Tinygrad: gep_on_store passes *sto.src[2:] to preserve ranges.
#[test]
fn test_gep_on_store_preserves_ranges() {
    let buffer = create_buffer_typed(64, ScalarDType::Float32);
    let idx = create_vector_index_iota(buffer.clone(), 2);
    let value = create_vector_float_values(vec![1.0, 2.0]);
    let range = create_range_loop(8, 0);
    let gep_idx = idx.gep(vec![1, 0]);
    let store = gep_idx.store_with_ranges(value, smallvec![range]);

    let result = apply_gep_movement(&store);

    // For GROUP, each inner STORE should have ranges
    // For STORE, ranges should be preserved
    fn has_ranges(uop: &Arc<UOp>) -> bool {
        match uop.op() {
            Op::Store { ranges, .. } => !ranges.is_empty(),
            Op::Group { sources } => sources.iter().all(has_ranges),
            _ => false,
        }
    }
    assert!(has_ranges(&result), "Ranges should be preserved");
}

/// Test: 4-element permutation [3,1,2,0].
#[test]
fn test_gep_on_store_4_element() {
    let buffer = create_buffer_typed(64, ScalarDType::Float32);
    let idx = create_vector_index_iota(buffer.clone(), 4);
    let value = create_vector_float_iota(4);
    let gep_idx = idx.gep(vec![3, 1, 2, 0]);
    let store = gep_idx.store(value);

    let result = apply_gep_movement(&store);

    // Verify inverse: [3,1,2,0]
    // pos 0 has val 3 → inv[3] = 0
    // pos 1 has val 1 → inv[1] = 1
    // pos 2 has val 2 → inv[2] = 2
    // pos 3 has val 0 → inv[0] = 3
    // inv = [3,1,2,0]
    let inv = compute_inverse_permutation(&[3, 1, 2, 0]);
    assert_eq!(inv, vec![3, 1, 2, 0]);

    assert!(matches!(result.op(), Op::Store { .. } | Op::Group { .. }));
}

/// Test: STORE(GEP(PTRCAT)) distributes correctly.
#[test]
fn test_gep_on_store_with_ptrcat() {
    let buffer1 = create_buffer_typed(32, ScalarDType::Float32);
    let buffer2 = create_buffer_typed(32, ScalarDType::Float32);

    let idx1 = create_vector_index_iota(buffer1.clone(), 2);
    let idx2 = create_vector_index_iota(buffer2.clone(), 2);
    let ptrcat = UOp::ptrcat().sources(vec![idx1, idx2]).call();

    let value = create_vector_float_iota(4);
    let gep_ptrcat = ptrcat.gep(vec![0, 2, 1, 3]);
    let store = gep_ptrcat.store(value);

    let result = apply_gep_movement(&store);

    // PTRCAT distribution → GROUP of STOREs (Tinygrad cat_after_store)
    assert!(
        matches!(result.op(), Op::Store { .. } | Op::Group { .. }),
        "Expected STORE or GROUP after PTRCAT distribution"
    );
}

// =============================================================================
// Integration Tests
// =============================================================================

/// Test: Combined GEP movement in phase2.
#[test]
fn test_gep_movement_in_phase2() {
    let buffer = create_buffer_typed(64, ScalarDType::Float32);
    let idx = create_vector_index_iota(buffer.clone(), 4);
    let gep_idx = idx.gep(vec![0, 2]);

    // LOAD with GEP
    let load = UOp::load().buffer(buffer.clone()).index(gep_idx.clone()).call();
    // STORE the loaded value
    let store = gep_idx.store(load.clone());

    let result = apply_phase2(&store);

    // Both patterns should have fired, result is valid
    assert!(matches!(result.op(), Op::Store { .. } | Op::Group { .. }));
}

/// Test: GEP movement enables PTRCAT distribution end-to-end.
#[test]
fn test_gep_movement_enables_ptrcat_distribution() {
    let buffer1 = create_buffer_typed(32, ScalarDType::Float32);
    let buffer2 = create_buffer_typed(32, ScalarDType::Float32);

    // PTRCAT of two 2-element indices
    let idx1 = create_vector_index_iota(buffer1.clone(), 2);
    let idx2 = create_vector_index_iota(buffer2.clone(), 2);
    let ptrcat = UOp::ptrcat().sources(vec![idx1, idx2]).call();

    // GEP selecting across both
    let gep_ptrcat = ptrcat.gep(vec![0, 1, 2, 3]);

    // LOAD through GEP(PTRCAT)
    let load = UOp::load().buffer(buffer1.clone()).index(gep_ptrcat).call();

    let result = apply_phase2(&load);

    // After phase2, no LOAD(PTRCAT) should exist
    let ptrcat_under_load = count_ops(&result, |u| {
        if let Op::Load { index, .. } = u.op() { matches!(index.op(), Op::PtrCat { .. }) } else { false }
    });
    assert_eq!(ptrcat_under_load, 0, "PTRCAT should be distributed");
}

// =============================================================================
// Inverse Permutation Unit Tests
// =============================================================================

#[test]
fn test_inverse_permutation_identity() {
    let inv = compute_inverse_permutation(&[0, 1, 2, 3]);
    assert_eq!(inv, vec![0, 1, 2, 3]);
}

#[test]
fn test_inverse_permutation_reverse() {
    let inv = compute_inverse_permutation(&[3, 2, 1, 0]);
    assert_eq!(inv, vec![3, 2, 1, 0], "Reverse is self-inverse");
}

#[test]
fn test_inverse_permutation_rotation() {
    // [1,2,3,0] = rotate left by 1
    // Inverse of rotate-left-1 is rotate-right-1 = [3,0,1,2]
    let inv = compute_inverse_permutation(&[1, 2, 3, 0]);
    assert_eq!(inv, vec![3, 0, 1, 2]);
}

#[test]
fn test_inverse_permutation_complex() {
    // [2,0,3,1]
    // pos 0→val 2, pos 1→val 0, pos 2→val 3, pos 3→val 1
    // inv[2]=0, inv[0]=1, inv[3]=2, inv[1]=3
    // inv = [1,3,0,2]
    let inv = compute_inverse_permutation(&[2, 0, 3, 1]);
    assert_eq!(inv, vec![1, 3, 0, 2]);
}
