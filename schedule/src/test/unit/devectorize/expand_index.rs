//! Phase 1 tests: expand_vector_index.
//!
//! Tests for the expand_index patterns which transform vector INDEX
//! operations into grouped PTRCAT structures for contiguous memory access.
//!
//! Based on Tinygrad's devectorizer.py expand_index (lines 59-95).

use std::sync::Arc;

use morok_dtype::{DType, ScalarDType};
use morok_ir::types::ConstValue;
use morok_ir::{AxisId, AxisType, BinaryOp, Op, UOp};
use smallvec::smallvec;

use super::helpers::*;

/// Unwrap GEP if present to get the inner PTRCAT or other node.
/// expand_vector_index returns GEP(PTRCAT(...)) to handle lane reordering;
/// the identity GEP is simplified later by gep_pushing_patterns.
fn unwrap_gep(uop: &Arc<UOp>) -> Arc<UOp> {
    match uop.op() {
        Op::Gep { vector, .. } => vector.clone(),
        _ => uop.clone(),
    }
}

// =============================================================================
// Contiguous Access Tests
// =============================================================================

/// Test: Contiguous vec4 index [0,1,2,3] -> single PTRCAT group.
///
/// When indices are consecutive, they should be grouped into a single
/// contiguous access with a CAST to vector pointer type.
#[test]
fn test_expand_contiguous_vec4() {
    let buffer = create_buffer(64);
    let index = create_vector_index_iota(buffer.clone(), 4);

    let result = apply_phase1(&index);

    // Result should be GEP(PTRCAT(...)) or PTRCAT with a single CAST(INDEX) pointer
    // representing a contiguous 4-element access.
    // The GEP reorders lanes; identity GEP is simplified by gep_pushing_patterns later.
    match result.op() {
        Op::Gep { vector, indices } => {
            // GEP wrapping PTRCAT - check inner PTRCAT has single CAST(INDEX) source
            assert_eq!(indices.len(), 4, "GEP should have 4 indices for vec4");
            match vector.op() {
                Op::PtrCat { sources } => {
                    assert_eq!(sources.len(), 1, "Contiguous indices should form single group");
                    assert_is_cast(&sources[0]);
                }
                other => panic!("Expected PTRCAT inside GEP, got {:?}", other),
            }
        }
        Op::PtrCat { sources } => {
            assert_eq!(sources.len(), 1, "Contiguous indices should form single group");
            assert_is_cast(&sources[0]);
            let (src, _dtype) = unwrap_cast(&sources[0]);
            assert_is_index(&src);
        }
        Op::Cast { src, .. } => {
            assert_is_index(src);
        }
        other => panic!("Expected GEP, PTRCAT or CAST, got {:?}", other),
    }
}

/// Test: Contiguous vec8 index [0..8] -> single group.
#[test]
fn test_expand_contiguous_vec8() {
    let buffer = create_buffer(128);
    let index = create_vector_index_iota(buffer.clone(), 8);

    let result = unwrap_gep(&apply_phase1(&index));

    match result.op() {
        Op::PtrCat { sources } => {
            assert_eq!(sources.len(), 1, "Contiguous vec8 should form single group");
        }
        Op::Cast { src, .. } => {
            assert_is_index(src);
        }
        other => panic!("Expected PTRCAT or CAST, got {:?}", other),
    }
}

/// Test: Contiguous with offset [10,11,12,13] -> offset group.
#[test]
fn test_expand_contiguous_with_offset() {
    let buffer = create_buffer(64);
    let index = create_vector_index_offset(buffer.clone(), 4, 10);

    let result = unwrap_gep(&apply_phase1(&index));

    // Should still form a single contiguous group
    match result.op() {
        Op::PtrCat { sources } => {
            assert_eq!(sources.len(), 1, "Offset contiguous indices should form single group");
        }
        Op::Cast { src, .. } => {
            assert_is_index(src);
        }
        other => panic!("Expected PTRCAT or CAST, got {:?}", other),
    }
}

/// Test: Contiguous expansion contains DEFINE_GLOBAL references.
///
/// After conversion to Tinygrad's structure (VECTORIZE(DEFINE_GLOBAL)),
/// the result should contain DEFINE_GLOBAL references.
#[test]
fn test_expand_contiguous_preserves_buffer() {
    let buffer = create_buffer(64);
    let index = create_vector_index_iota(buffer.clone(), 4);

    let result = apply_phase1(&index);

    // Verify DEFINE_GLOBAL is present in the result tree
    let define_count = count_ops(&result, |u| matches!(u.op(), Op::DefineGlobal(_)));
    assert!(define_count > 0, "DEFINE_GLOBAL reference should be present");
}

// =============================================================================
// Strided Access Tests
// =============================================================================

/// Test: Strided access [0,2,4,6] -> 4 separate groups.
///
/// Non-consecutive indices cannot be grouped together.
#[test]
fn test_expand_strided_access() {
    let buffer = create_buffer(64);
    let index = create_vector_index_scaled(buffer.clone(), 4, 2);

    let result = apply_phase1(&index);

    // With stride 2, each index is separate -> 4 groups
    match result.op() {
        Op::PtrCat { sources } => {
            assert_eq!(sources.len(), 4, "Strided indices should form 4 separate groups");
            // Each source should be a scalar INDEX (no CAST needed for scalar)
            for src in sources.iter() {
                assert_is_index(src);
            }
        }
        // Could be wrapped in GEP for reordering
        Op::Gep { vector, .. } => {
            if let Op::PtrCat { sources } = vector.op() {
                assert_eq!(sources.len(), 4);
            }
        }
        other => panic!("Expected PTRCAT or GEP(PTRCAT), got {:?}", other),
    }
}

/// Test: Mixed groups [0,1,5,6] -> 2 groups of 2.
#[test]
fn test_expand_mixed_groups() {
    let buffer = create_buffer(64);
    let index = create_vector_index_values(buffer.clone(), vec![0, 1, 5, 6]);

    let result = apply_phase1(&index);

    // [0,1] consecutive -> group 1 (size 2)
    // [5,6] consecutive -> group 2 (size 2)
    match result.op() {
        Op::PtrCat { sources } => {
            assert_eq!(sources.len(), 2, "Should form 2 groups");
        }
        Op::Gep { vector, .. } => {
            if let Op::PtrCat { sources } = vector.op() {
                assert_eq!(sources.len(), 2);
            }
        }
        other => panic!("Expected PTRCAT, got {:?}", other),
    }
}

/// Test: Reversed indices [3,2,1,0] -> 4 groups + GEP reorder.
#[test]
fn test_expand_reversed_indices() {
    let buffer = create_buffer(64);
    let index = create_vector_index_values(buffer.clone(), vec![3, 2, 1, 0]);

    let result = apply_phase1(&index);

    // Reversed indices are non-consecutive when traversed in order
    // They should be grouped as [0,1,2,3] with a GEP reorder [3,2,1,0]
    match result.op() {
        Op::Gep { vector, indices } => {
            assert_eq!(indices, &[3, 2, 1, 0], "Should have reorder GEP");
            if let Op::PtrCat { sources } = vector.op() {
                // Single contiguous group
                assert_eq!(sources.len(), 1);
            }
        }
        Op::PtrCat { sources } => {
            // Could also be 4 separate groups if not recognized as reversed contiguous
            assert!(sources.len() == 1 || sources.len() == 4);
        }
        other => panic!("Expected GEP or PTRCAT, got {:?}", other),
    }
}

/// Test: Scattered indices [0,5,3,7] -> 4 groups + reorder.
#[test]
fn test_expand_scattered_indices() {
    let buffer = create_buffer(64);
    let index = create_vector_index_values(buffer.clone(), vec![0, 5, 3, 7]);

    let result = apply_phase1(&index);

    // All scattered -> no consecutive groups
    match result.op() {
        Op::Gep { vector, .. } => {
            if let Op::PtrCat { sources } = vector.op() {
                assert_eq!(sources.len(), 4, "Scattered indices should form 4 groups");
            }
        }
        Op::PtrCat { sources } => {
            assert_eq!(sources.len(), 4);
        }
        other => panic!("Expected GEP(PTRCAT) or PTRCAT, got {:?}", other),
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/// Test: Scalar index (vcount=1) should not be transformed.
#[test]
fn test_expand_scalar_index_no_change() {
    let buffer = create_buffer(64);
    let index = create_index(buffer.clone(), 5);

    let result = apply_phase1(&index);

    // Scalar index should remain unchanged
    assert_is_index(&result);
    assert_eq!(result.dtype().vcount(), 1, "Should remain scalar");
}

/// Test: Gated index preserves gate.
#[test]
fn test_expand_gated_index() {
    let buffer = create_buffer(64);
    let gate = create_bool_const(true);
    let index = create_vector_index_gated(buffer.clone(), 4, gate);

    let result = apply_phase1(&index);

    // Gated indices should still be expanded
    match result.op() {
        Op::PtrCat { .. } | Op::Cast { .. } | Op::Gep { .. } => {
            // Expansion happened
        }
        Op::Index { gate, .. } => {
            // If not expanded, gate should be preserved
            assert!(gate.is_some(), "Gate should be preserved");
        }
        other => panic!("Unexpected result: {:?}", other),
    }
}

/// Test: Multi-index (>1 indices) is unsupported.
#[test]
fn test_expand_multi_index_unsupported() {
    let buffer = create_buffer(64);
    let idx1 = UOp::const_(DType::Index, ConstValue::Int(0));
    let idx2 = UOp::const_(DType::Index, ConstValue::Int(1));
    let index = UOp::new(Op::Index { buffer, indices: smallvec![idx1, idx2], gate: None }, DType::Float32);

    let result = apply_phase1(&index);

    // Multi-index should be unchanged
    let (_, indices, _) = unwrap_index(&result);
    assert_eq!(indices.len(), 2, "Multi-index should be preserved");
}

// =============================================================================
// Symbolic Root Tests
// =============================================================================

/// Test: Range-based index with symbolic root.
///
/// INDEX(VECTORIZE([def]*4), [range*4 + 0, range*4 + 1, range*4 + 2, range*4 + 3])
/// Should detect common root and group offsets.
#[test]
fn test_expand_range_based_index() {
    let buffer = create_buffer(256);

    // Create DEFINE_GLOBAL and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::define_global(def_id, buffer.dtype());
    let buf_vec = define.broadcast(4);

    // Create vector index: VECTORIZE([range*4+0, range*4+1, range*4+2, range*4+3])
    let range = UOp::new(
        Op::Range {
            end: UOp::const_(DType::Index, ConstValue::Int(64)),
            axis_id: AxisId::Renumbered(0),
            axis_type: AxisType::Loop,
            deps: smallvec::SmallVec::new(),
        },
        DType::Index,
    );

    let four = UOp::const_(DType::Index, ConstValue::Int(4));
    let base = UOp::new(Op::Binary(BinaryOp::Mul, range, four), DType::Index);

    let indices: smallvec::SmallVec<[Arc<UOp>; 4]> = (0..4)
        .map(|i| {
            if i == 0 {
                base.clone()
            } else {
                UOp::new(
                    Op::Binary(BinaryOp::Add, base.clone(), UOp::const_(DType::Index, ConstValue::Int(i))),
                    DType::Index,
                )
            }
        })
        .collect();

    let vec_idx = UOp::vectorize(indices);
    let index = UOp::new(Op::Index { buffer: buf_vec, indices: smallvec![vec_idx], gate: None }, DType::Float32);

    let result = unwrap_gep(&apply_phase1(&index));

    // Should form a single contiguous group
    match result.op() {
        Op::PtrCat { sources } => {
            assert_eq!(sources.len(), 1, "Symbolic contiguous should form single group");
        }
        Op::Cast { .. } => {
            // Single CAST(INDEX) without PTRCAT wrapper
        }
        other => panic!("Expected PTRCAT or CAST, got {:?}", other),
    }
}

/// Test: Same symbolic root groups together.
#[test]
fn test_expand_symbolic_root_grouping() {
    let buffer = create_buffer(256);

    // Create DEFINE_GLOBAL and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::define_global(def_id, buffer.dtype());
    let buf_vec = define.broadcast(4);

    // Create two groups with same root but different base offsets
    let range = UOp::new(
        Op::Range {
            end: UOp::const_(DType::Index, ConstValue::Int(64)),
            axis_id: AxisId::Renumbered(0),
            axis_type: AxisType::Loop,
            deps: smallvec::SmallVec::new(),
        },
        DType::Index,
    );

    // Group 1: [range+0, range+1]
    // Group 2: [range+10, range+11]
    let indices: smallvec::SmallVec<[Arc<UOp>; 4]> = [0i64, 1, 10, 11]
        .iter()
        .map(|&offset| {
            if offset == 0 {
                range.clone()
            } else {
                UOp::new(
                    Op::Binary(BinaryOp::Add, range.clone(), UOp::const_(DType::Index, ConstValue::Int(offset))),
                    DType::Index,
                )
            }
        })
        .collect();

    let vec_idx = UOp::vectorize(indices);
    let index = UOp::new(Op::Index { buffer: buf_vec, indices: smallvec![vec_idx], gate: None }, DType::Float32);

    let result = apply_phase1(&index);

    // Should form 2 groups: [0,1] and [10,11]
    match result.op() {
        Op::PtrCat { sources } => {
            assert_eq!(sources.len(), 2, "Should form 2 groups");
        }
        Op::Gep { vector, .. } => {
            if let Op::PtrCat { sources } = vector.op() {
                assert_eq!(sources.len(), 2);
            }
        }
        other => panic!("Expected PTRCAT, got {:?}", other),
    }
}

/// Test: Different roots produce separate groups.
#[test]
fn test_expand_different_roots_separate() {
    let buffer = create_buffer(256);

    // Create DEFINE_GLOBAL and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(3000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::define_global(def_id, buffer.dtype());
    let buf_vec = define.broadcast(4);

    // Two different range variables
    let range1 = UOp::new(
        Op::Range {
            end: UOp::const_(DType::Index, ConstValue::Int(64)),
            axis_id: AxisId::Renumbered(0),
            axis_type: AxisType::Loop,
            deps: smallvec::SmallVec::new(),
        },
        DType::Index,
    );

    let range2 = UOp::new(
        Op::Range {
            end: UOp::const_(DType::Index, ConstValue::Int(64)),
            axis_id: AxisId::Renumbered(1),
            axis_type: AxisType::Loop,
            deps: smallvec::SmallVec::new(),
        },
        DType::Index,
    );

    // [range1, range1+1, range2, range2+1]
    let indices: smallvec::SmallVec<[Arc<UOp>; 4]> = [
        range1.clone(),
        UOp::new(
            Op::Binary(BinaryOp::Add, range1.clone(), UOp::const_(DType::Index, ConstValue::Int(1))),
            DType::Index,
        ),
        range2.clone(),
        UOp::new(
            Op::Binary(BinaryOp::Add, range2.clone(), UOp::const_(DType::Index, ConstValue::Int(1))),
            DType::Index,
        ),
    ]
    .into_iter()
    .collect();

    let vec_idx = UOp::vectorize(indices);
    let index = UOp::new(Op::Index { buffer: buf_vec, indices: smallvec![vec_idx], gate: None }, DType::Float32);

    let result = apply_phase1(&index);

    // Different roots cannot be grouped together -> 2 groups
    match result.op() {
        Op::PtrCat { sources } => {
            assert_eq!(sources.len(), 2, "Different roots should form separate groups");
        }
        Op::Gep { vector, .. } => {
            if let Op::PtrCat { sources } = vector.op() {
                assert_eq!(sources.len(), 2);
            }
        }
        other => panic!("Expected PTRCAT, got {:?}", other),
    }
}

// =============================================================================
// Dtype Tests
// =============================================================================

/// Test: Different dtypes are handled correctly.
#[test]
fn test_expand_int32_buffer() {
    let buffer = create_buffer_typed(64, ScalarDType::Int32);
    let index = create_vector_index_iota(buffer.clone(), 4);

    let result = unwrap_gep(&apply_phase1(&index));

    // Should work with int32 dtype
    match result.op() {
        Op::PtrCat { sources } => {
            assert_eq!(sources.len(), 1);
        }
        Op::Cast { .. } => {}
        other => panic!("Expected PTRCAT or CAST, got {:?}", other),
    }
}

/// Test: Half precision (f16) buffer.
#[test]
fn test_expand_half_buffer() {
    let buffer = create_buffer_typed(64, ScalarDType::Float16);
    let index = create_vector_index_iota(buffer.clone(), 4);

    let result = unwrap_gep(&apply_phase1(&index));

    match result.op() {
        Op::PtrCat { sources } => {
            assert_eq!(sources.len(), 1);
        }
        Op::Cast { .. } => {}
        other => panic!("Expected PTRCAT or CAST, got {:?}", other),
    }
}
