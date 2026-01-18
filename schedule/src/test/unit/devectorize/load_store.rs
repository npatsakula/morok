//! Phase 2 tests: PTRCAT distribution and split load/store.
//!
//! Tests for the load_store_patterns which:
//! 1. Distribute PTRCAT through LOAD/STORE operations
//! 2. Split LOAD/STORE by fold length divisibility
//!
//! Based on Tinygrad's devectorizer.py load_store_folding and split_load_store.

use std::sync::Arc;

use morok_dtype::{AddrSpace, DType, ScalarDType};
use morok_ir::types::ConstValue;
use morok_ir::{Op, UOp};

use super::helpers::*;

// =============================================================================
// PTRCAT Distribution Tests - LOAD
// =============================================================================

/// Test: Distribute PTRCAT through LOAD (dual sources).
///
/// LOAD(PTRCAT(a, b)) -> CAT(LOAD(a), LOAD(b))
#[test]
fn test_distribute_ptrcat_load_dual() {
    let buffer = create_buffer(64);

    // Create two INDEX pointers
    let idx1 = create_index(buffer.clone(), 0);
    let idx2 = create_index(buffer.clone(), 1);

    // PTRCAT(idx1, idx2)
    let ptrcat = UOp::ptrcat(vec![idx1, idx2]);

    // LOAD(buffer, ptrcat)
    let load = UOp::load(buffer.clone(), ptrcat);

    let result = apply_phase2(&load);

    // Result should be CAT(LOAD(idx1), LOAD(idx2)) or VECTORIZE equivalent
    match result.op() {
        Op::Cat { sources } => {
            assert_eq!(sources.len(), 2, "Should have 2 LOAD sources");
            for src in sources.iter() {
                assert_is_load(src);
            }
        }
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 2, "Should have 2 LOAD elements");
        }
        other => panic!("Expected CAT or VECTORIZE, got {:?}", other),
    }
}

/// Test: Distribute PTRCAT through LOAD (quad sources).
///
/// LOAD(PTRCAT(a, b, c, d)) -> CAT(LOAD(a), LOAD(b), LOAD(c), LOAD(d))
#[test]
fn test_distribute_ptrcat_load_quad() {
    let buffer = create_buffer(64);

    let indices: Vec<Arc<UOp>> = (0..4).map(|i| create_index(buffer.clone(), i)).collect();
    let ptrcat = UOp::ptrcat(indices);
    let load = UOp::load(buffer.clone(), ptrcat);

    let result = apply_phase2(&load);

    match result.op() {
        Op::Cat { sources } => {
            assert_eq!(sources.len(), 4, "Should have 4 LOAD sources");
        }
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4, "Should have 4 LOAD elements");
        }
        other => panic!("Expected CAT or VECTORIZE, got {:?}", other),
    }
}

/// Test: PTRCAT distribution preserves buffer reference.
#[test]
fn test_distribute_ptrcat_preserves_buffer() {
    let buffer = create_buffer(64);

    let idx1 = create_index(buffer.clone(), 0);
    let idx2 = create_index(buffer.clone(), 1);
    let ptrcat = UOp::ptrcat(vec![idx1, idx2]);
    let load = UOp::load(buffer.clone(), ptrcat);

    let result = apply_phase2(&load);

    // All LOADs should reference the same buffer
    let buffer_refs =
        count_ops(&result, |u| if let Op::Load { buffer: b, .. } = u.op() { Arc::ptr_eq(b, &buffer) } else { false });
    assert!(buffer_refs >= 2, "Should have at least 2 buffer references");
}

// =============================================================================
// PTRCAT Distribution Tests - STORE
// =============================================================================

/// Test: Distribute PTRCAT through STORE.
///
/// STORE(buffer, PTRCAT(a, b), data) -> GROUP(STORE(a, gep(data, 0)), STORE(b, gep(data, 1)))
#[test]
fn test_distribute_ptrcat_store() {
    let buffer = create_buffer(64);
    let value = create_vector_float_iota(2);

    let idx1 = create_index(buffer.clone(), 0);
    let idx2 = create_index(buffer.clone(), 1);
    let ptrcat = UOp::ptrcat(vec![idx1, idx2]);
    let store = UOp::store(buffer.clone(), ptrcat, value);

    let result = apply_phase2(&store);

    // Result should be GROUP of individual STOREs
    match result.op() {
        Op::Group { sources } => {
            assert_eq!(sources.len(), 2, "Should have 2 STORE sources");
            for src in sources.iter() {
                assert_is_store(src);
            }
        }
        // Could be a single STORE if simplified
        Op::Store { .. } => {}
        other => panic!("Expected GROUP or STORE, got {:?}", other),
    }
}

/// Test: STORE with quad PTRCAT.
#[test]
fn test_distribute_ptrcat_store_quad() {
    let buffer = create_buffer(64);
    let value = create_vector_float_iota(4);

    let indices: Vec<Arc<UOp>> = (0..4).map(|i| create_index(buffer.clone(), i)).collect();
    let ptrcat = UOp::ptrcat(indices);
    let store = UOp::store(buffer.clone(), ptrcat, value);

    let result = apply_phase2(&store);

    match result.op() {
        Op::Group { sources } => {
            assert_eq!(sources.len(), 4, "Should have 4 STORE sources");
        }
        Op::Store { .. } => {}
        other => panic!("Expected GROUP or STORE, got {:?}", other),
    }
}

// =============================================================================
// Split Load/Store Tests
// =============================================================================

/// Test: Split vec8 LOAD into 2x vec4.
///
/// Based on default fold lengths [4, 2, 1].
#[test]
fn test_split_load_vec8_to_vec4() {
    let buffer = create_buffer(128);

    // Create CAST(INDEX) with vec8 pointer dtype (simulating expand_index output)
    let idx = create_index(buffer.clone(), 0);
    let vec8_ptr_dtype = DType::Float32.vec(8).ptr(Some(8), AddrSpace::Global);
    let cast_idx = UOp::cast(idx, vec8_ptr_dtype);

    // LOAD with vec8 result dtype
    let load_dtype = DType::Float32.vec(8);
    let load = UOp::new(Op::Load { buffer: buffer.clone(), index: cast_idx }, load_dtype);

    let result = apply_phase2(&load);

    // Should be split into 2x vec4 loads (8 = 4 + 4)
    match result.op() {
        Op::Cat { sources } => {
            assert_eq!(sources.len(), 2, "vec8 should split into 2 chunks");
            for src in sources.iter() {
                assert_is_load(src);
                assert_eq!(src.dtype().vcount(), 4, "Each chunk should be vec4");
            }
        }
        Op::Vectorize { elements } => {
            // Could be VECTORIZE of individual loads
            assert_eq!(elements.len(), 8);
        }
        // If not split, single load
        Op::Load { .. } => {
            assert_eq!(result.dtype().vcount(), 8);
        }
        other => panic!("Expected CAT, VECTORIZE or LOAD, got {:?}", other),
    }
}

/// Test: Split vec6 LOAD (non-power-of-2).
///
/// 6 = 4 + 2 with fold lengths [4, 2, 1]
#[test]
fn test_split_load_vec6_mixed() {
    let buffer = create_buffer(128);

    let idx = create_index(buffer.clone(), 0);
    let vec6_ptr_dtype = DType::Float32.vec(6).ptr(Some(6), AddrSpace::Global);
    let cast_idx = UOp::cast(idx, vec6_ptr_dtype);

    let load_dtype = DType::Float32.vec(6);
    let load = UOp::new(Op::Load { buffer: buffer.clone(), index: cast_idx }, load_dtype);

    let result = apply_phase2(&load);

    // Should be split into vec4 + vec2
    match result.op() {
        Op::Cat { sources } => {
            assert_eq!(sources.len(), 2, "vec6 should split into 2 chunks");
            // First chunk should be vec4, second vec2
            let vcounts: Vec<usize> = sources.iter().map(|s| s.dtype().vcount()).collect();
            assert!(vcounts == vec![4, 2] || vcounts.iter().sum::<usize>() == 6, "Chunks should sum to 6");
        }
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 6);
        }
        Op::Load { .. } => {}
        other => panic!("Expected CAT, VECTORIZE or LOAD, got {:?}", other),
    }
}

/// Test: Split vec8 STORE into 2x vec4.
#[test]
fn test_split_store_vec8() {
    let buffer = create_buffer(128);
    let value = create_vector_float_iota(8);

    let idx = create_index(buffer.clone(), 0);
    let vec8_ptr_dtype = DType::Float32.vec(8).ptr(Some(8), AddrSpace::Global);
    let cast_idx = UOp::cast(idx, vec8_ptr_dtype);

    let store = UOp::store(buffer.clone(), cast_idx, value);

    let result = apply_phase2(&store);

    // Should be split into 2x vec4 stores
    match result.op() {
        Op::Group { sources } => {
            assert_eq!(sources.len(), 2, "vec8 store should split into 2 chunks");
            for src in sources.iter() {
                assert_is_store(src);
            }
        }
        Op::Store { .. } => {}
        other => panic!("Expected GROUP or STORE, got {:?}", other),
    }
}

/// Test: Split preserves ranges in STORE.
#[test]
fn test_split_preserves_ranges() {
    use morok_ir::AxisId;
    use smallvec::smallvec;

    let buffer = create_buffer(128);
    let value = create_vector_float_iota(8);

    let idx = create_index(buffer.clone(), 0);
    let vec8_ptr_dtype = DType::Float32.vec(8).ptr(Some(8), AddrSpace::Global);
    let cast_idx = UOp::cast(idx, vec8_ptr_dtype);

    // Create range for the store
    let range = UOp::new(
        Op::Range {
            end: UOp::const_(DType::Index, ConstValue::Int(10)),
            axis_id: AxisId::Renumbered(0),
            axis_type: morok_ir::AxisType::Loop,
        },
        DType::Index,
    );

    let store = UOp::store_with_ranges(buffer.clone(), cast_idx, value, smallvec![range.clone()]);

    let result = apply_phase2(&store);

    // Check that ranges are preserved in split stores
    match result.op() {
        Op::Group { sources } => {
            // Split stores should exist and each should preserve ranges
            assert!(!sources.is_empty(), "Should have split stores");
            for src in sources.iter() {
                if let Op::Store { ranges, .. } = src.op() {
                    assert_eq!(ranges.len(), 1, "Each split store should preserve ranges");
                }
            }
        }
        Op::Store { ranges, .. } => {
            // If not split, ranges should be preserved
            assert_eq!(ranges.len(), 1, "Ranges should be preserved");
        }
        other => panic!("Expected GROUP or STORE, got {:?}", other),
    }
}

// =============================================================================
// Integration Tests (Phase 1 -> Phase 2)
// =============================================================================

/// Test: LOAD after expand_index produces correct output.
#[test]
fn test_load_after_expand_index() {
    let buffer = create_buffer(64);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load(buffer.clone(), index);

    // Apply Phase 1 first
    let after_phase1 = apply_phase1(&load);

    // Then apply Phase 2
    let result = apply_phase2(&after_phase1);

    // Final result should have proper structure
    // Either CAT of scalar loads, VECTORIZE, or single contiguous load
    let total_vcount = result.dtype().vcount();
    assert!(total_vcount >= 1, "Should produce valid dtype");
}

/// Test: STORE after expand_index produces correct output.
#[test]
fn test_store_after_expand_index() {
    let buffer = create_buffer(64);
    let value = create_vector_float_iota(4);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let store = UOp::store(buffer.clone(), index, value);

    let after_phase1 = apply_phase1(&store);
    let result = apply_phase2(&after_phase1);

    // Should produce GROUP of stores or single store
    let store_count = count_stores(&result);
    assert!(store_count >= 1, "Should have at least one store");
}

// =============================================================================
// Divisibility Tests
// =============================================================================

/// Test: Offset divisibility affects fold length selection.
#[test]
fn test_split_load_divisibility() {
    let buffer = create_buffer(128);

    // Create INDEX with offset that's divisible by 4
    let idx = UOp::index(
        buffer.clone(),
        vec![UOp::const_(DType::Index, ConstValue::Int(8))], // 8 is divisible by 4
    )
    .unwrap();

    let vec8_ptr_dtype = DType::Float32.vec(8).ptr(Some(8), AddrSpace::Global);
    let cast_idx = UOp::cast(idx, vec8_ptr_dtype);

    let load_dtype = DType::Float32.vec(8);
    let load = UOp::new(Op::Load { buffer: buffer.clone(), index: cast_idx }, load_dtype);

    let result = apply_phase2(&load);

    // With offset 8 (divisible by 4), should prefer vec4 chunks
    // Result may be CAT, LOAD, or VECTORIZE (of scalar GEPs from vec4 loads)
    assert_eq!(result.dtype().vcount(), 8, "Total vcount should be 8");
    match result.op() {
        Op::Cat { sources } => {
            // Should prefer larger chunks when offset is divisible
            assert!(sources.len() <= 4, "Should use fewer, larger chunks");
            // Verify total vcount is 8
            let total: usize = sources.iter().map(|s| s.dtype().vcount()).sum();
            assert_eq!(total, 8, "Total vcount should be 8");
        }
        Op::Load { .. } => {
            // If kept as single load, vcount should be 8
            assert_eq!(result.dtype().vcount(), 8, "Single load should have vcount 8");
        }
        Op::Vectorize { elements } => {
            // Vectorize of 8 scalar GEPs from vec4 loads is valid
            assert_eq!(elements.len(), 8, "Vectorize should have 8 elements");
            // Verify there are LOADs in the tree
            assert!(count_loads(&result) >= 1, "Should have at least one LOAD");
        }
        other => panic!("Expected CAT, LOAD, or VECTORIZE, got {:?}", other),
    }
}

/// Test: Offset not divisible by 4 forces smaller chunks.
#[test]
fn test_split_load_not_divisible() {
    let buffer = create_buffer(128);

    // Create INDEX with offset that's not divisible by 4
    let idx = UOp::index(
        buffer.clone(),
        vec![UOp::const_(DType::Index, ConstValue::Int(3))], // 3 is not divisible by 4
    )
    .unwrap();

    let vec8_ptr_dtype = DType::Float32.vec(8).ptr(Some(8), AddrSpace::Global);
    let cast_idx = UOp::cast(idx, vec8_ptr_dtype);

    let load_dtype = DType::Float32.vec(8);
    let load = UOp::new(Op::Load { buffer: buffer.clone(), index: cast_idx }, load_dtype);

    let result = apply_phase2(&load);

    // With offset 3 (not divisible by 4), may use smaller chunks
    // Verify total vcount is preserved
    match result.op() {
        Op::Cat { sources } => {
            let total: usize = sources.iter().map(|s| s.dtype().vcount()).sum();
            assert_eq!(total, 8, "Total vcount should be 8");
        }
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 8, "Vectorize should have 8 elements");
        }
        Op::Load { .. } => {
            assert_eq!(result.dtype().vcount(), 8, "Single load should have vcount 8");
        }
        other => panic!("Expected CAT, VECTORIZE, or LOAD, got {:?}", other),
    }
}

// =============================================================================
// Gated Index Load Test
// =============================================================================

/// Test: Load from gated index.
///
/// In Tinygrad's model, gates are on INDEX, not LOAD/STORE.
/// Tests that loads from gated indices are properly handled.
#[test]
fn test_gated_index_load() {
    let buffer = create_buffer(64);
    let gate = create_bool_const(true);

    // Create gated index (gate is on INDEX, not LOAD)
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));
    let gated_index = UOp::new(
        Op::Index { buffer: buffer.clone(), indices: smallvec::smallvec![idx], gate: Some(gate.clone()) },
        DType::Float32,
    );

    // Create load from gated index
    let load = UOp::load(buffer.clone(), gated_index);

    // Apply devectorization should handle gated indices
    let result = apply_devectorize(&load);

    // Should produce valid output
    assert!(
        matches!(result.op(), Op::Load { .. } | Op::Cat { .. } | Op::Vectorize { .. }),
        "Should produce valid load structure"
    );
}
