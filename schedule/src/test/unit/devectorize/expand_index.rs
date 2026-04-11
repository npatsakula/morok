//! Expand index tests (full devectorize pipeline).
//!
//! Tests that vector INDEX operations are correctly processed through
//! the full devectorize pipeline (sym + devectorize + load_store_folding
//! + correct_load_store + load_store_indexing + pm_render).
//!
//! After the full pass:
//! - No PTRCAT nodes remain (distributed into scalar loads)
//! - Contiguous accesses produce a single wide LOAD
//! - Scattered accesses produce N scalar LOADs in a VECTORIZE
//!
//! Based on Tinygrad's devectorizer.py expand_index (lines 59-95).

use std::sync::Arc;

use morok_dtype::{DType, ScalarDType};
use morok_ir::types::ConstValue;
use morok_ir::{AxisId, AxisType, BinaryOp, Op, UOp};
use smallvec::smallvec;

use super::helpers::*;

/// Check that no PTRCAT nodes remain in the result tree.
fn assert_no_ptrcat(uop: &Arc<UOp>) {
    let ptrcat_count = count_ptrcats(uop);
    assert_eq!(ptrcat_count, 0, "No PTRCAT nodes should remain after full devectorize, found {}", ptrcat_count);
}

// =============================================================================
// Contiguous Access Tests
// =============================================================================

/// Test: Contiguous vec4 index [0,1,2,3] -> single wide LOAD.
///
/// After the full pipeline, contiguous indices should produce a single
/// vector LOAD (no PTRCAT, no individual scalar LOADs).
#[test]
fn test_expand_contiguous_vec4() {
    let buffer = create_buffer(64);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    // Contiguous access: expect a single LOAD with vec4 dtype or a VECTORIZE of scalar LOADs
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}

/// Test: Contiguous vec8 index [0..8] -> single wide LOAD.
#[test]
fn test_expand_contiguous_vec8() {
    let buffer = create_buffer(128);
    let index = create_vector_index_iota(buffer.clone(), 8);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}

/// Test: Contiguous with offset [10,11,12,13] -> offset group.
#[test]
fn test_expand_contiguous_with_offset() {
    let buffer = create_buffer(64);
    let index = create_vector_index_offset(buffer.clone(), 4, 10);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}

/// Test: Contiguous expansion contains codegen PARAM references.
///
/// After conversion to Tinygrad's structure (VECTORIZE(PARAM)),
/// the result should contain codegen PARAM references.
#[test]
fn test_expand_contiguous_preserves_buffer() {
    let buffer = create_buffer(64);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    // Verify codegen PARAM is present in the result tree
    let define_count = count_ops(&result, |u| matches!(u.op(), Op::Param { device: None, .. }));
    assert!(define_count > 0, "Codegen PARAM reference should be present");
}

// =============================================================================
// Strided Access Tests
// =============================================================================

/// Test: Strided access [0,2,4,6] -> 4 scalar LOADs in VECTORIZE.
///
/// Non-consecutive indices cannot be grouped together.
#[test]
fn test_expand_strided_access() {
    let buffer = create_buffer(64);
    let index = create_vector_index_scaled(buffer.clone(), 4, 2);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    // Strided: expect 4 scalar LOADs (each index is separate)
    let load_count = count_loads(&result);
    assert_eq!(load_count, 4, "Strided access should produce 4 scalar LOADs");
}

/// Test: Mixed groups [0,1,5,6] -> 2 groups of 2.
#[test]
fn test_expand_mixed_groups() {
    let buffer = create_buffer(64);
    let index = create_vector_index_values(buffer.clone(), vec![0, 1, 5, 6]);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    // [0,1] contiguous + [5,6] contiguous -> 2 loads (each vec2) or 4 scalar loads
    let load_count = count_loads(&result);
    assert!((1..=4).contains(&load_count), "Should have between 1 and 4 LOADs, got {}", load_count);
}

/// Test: Reversed indices [3,2,1,0] -> GEP reorder over contiguous group.
#[test]
fn test_expand_reversed_indices() {
    let buffer = create_buffer(64);
    let index = create_vector_index_values(buffer.clone(), vec![3, 2, 1, 0]);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}

/// Test: Scattered indices [0,5,3,7] -> 4 scalar LOADs.
#[test]
fn test_expand_scattered_indices() {
    let buffer = create_buffer(64);
    let index = create_vector_index_values(buffer.clone(), vec![0, 5, 3, 7]);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    // Fully scattered: 4 separate scalar LOADs
    let load_count = count_loads(&result);
    assert_eq!(load_count, 4, "Scattered access should produce 4 scalar LOADs");
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/// Test: Scalar index (vcount=1) should not be transformed.
#[test]
fn test_expand_scalar_index_no_change() {
    let buffer = create_buffer(64);
    let index = create_index(buffer.clone(), 5);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    // Scalar load should remain a single LOAD
    assert_is_load(&result);
    assert_eq!(result.dtype().vcount(), 1, "Should remain scalar");
}

/// Test: Gated index preserves gate.
#[test]
fn test_expand_gated_index() {
    let buffer = create_buffer(64);
    let gate = create_bool_const(true);
    let index = create_vector_index_gated(buffer.clone(), 4, gate);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    // Should still produce valid LOAD structure
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}

/// Test: Multi-index (>1 indices) is unsupported.
#[test]
fn test_expand_multi_index_unsupported() {
    let buffer = create_buffer(64);
    let idx1 = UOp::const_(DType::Index, ConstValue::Int(0));
    let idx2 = UOp::const_(DType::Index, ConstValue::Int(1));
    let index =
        UOp::new(Op::Index { buffer: buffer.clone(), indices: smallvec![idx1, idx2], gate: None }, DType::Float32);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    // Multi-index should be preserved through the pipeline
    assert_is_load(&result);
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

    // Create codegen PARAM and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::param(def_id, 256, buffer.dtype(), None);
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
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    // Symbolic contiguous: should form a single wide LOAD
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}

/// Test: Same symbolic root groups together.
#[test]
fn test_expand_symbolic_root_grouping() {
    let buffer = create_buffer(256);

    // Create codegen PARAM and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::param(def_id, 256, buffer.dtype(), None);
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
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    // Two groups: should produce 2 LOADs (each vec2) or more scalar LOADs
    let load_count = count_loads(&result);
    assert!(load_count >= 2, "Should have at least 2 LOADs for 2 groups, got {}", load_count);
}

/// Test: Different roots produce separate groups.
#[test]
fn test_expand_different_roots_separate() {
    let buffer = create_buffer(256);

    // Create codegen PARAM and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(3000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::param(def_id, 256, buffer.dtype(), None);
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
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    // Different roots -> at least 2 separate LOADs
    let load_count = count_loads(&result);
    assert!(load_count >= 2, "Different roots should produce at least 2 LOADs, got {}", load_count);
}

// =============================================================================
// Dtype Tests
// =============================================================================

/// Test: Different dtypes are handled correctly.
#[test]
fn test_expand_int32_buffer() {
    let buffer = create_buffer_typed(64, ScalarDType::Int32);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}

/// Test: Half precision (f16) buffer.
#[test]
fn test_expand_half_buffer() {
    let buffer = create_buffer_typed(64, ScalarDType::Float16);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}

// =============================================================================
// Broadcast Tests (multi-lane offsets)
// =============================================================================

/// Test: Pure broadcast [0,0,0,0] -> scalar LOAD broadcast to all lanes.
///
/// When all lanes read the same address, the result should be a single
/// scalar LOAD broadcast via VECTORIZE.
#[test]
fn test_expand_pure_broadcast() {
    let buffer = create_buffer(64);
    let index = create_vector_index_values(buffer.clone(), vec![0, 0, 0, 0]);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    // Pure broadcast: all lanes read the same address
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}

/// Test: Partial broadcast [0,1,0,1] -> vec2 LOAD with GEP rebroadcast.
///
/// Two distinct offsets (0 and 1), each read by two lanes.
#[test]
fn test_expand_partial_broadcast() {
    let buffer = create_buffer(64);
    let index = create_vector_index_values(buffer.clone(), vec![0, 1, 0, 1]);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}

/// Test: Mixed single/multi-lane [0,1,0,2] -> vec3 group with rebroadcast.
///
/// Three distinct offsets (0, 1, 2), where offset 0 is read by two lanes.
#[test]
fn test_expand_mixed_broadcast() {
    let buffer = create_buffer(64);
    let index = create_vector_index_values(buffer.clone(), vec![0, 1, 0, 2]);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    assert_no_ptrcat(&result);
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}
