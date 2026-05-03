//! Edge cases and regression tests.
//!
//! Tests for corner cases, boundary conditions, and regressions
//! in the devectorize pass.

use std::sync::Arc;

use morok_dtype::{DType, ScalarDType};
use morok_ir::types::ConstValue;
use morok_ir::{Op, UOp};
use smallvec::SmallVec;

use super::helpers::*;

// =============================================================================
// Scalar Passthrough Tests
// =============================================================================

/// Test: Scalar operations pass through unchanged.
#[test]
fn test_devectorize_scalar_passthrough() {
    let buffer = create_buffer(64);
    let index = create_index(buffer.clone(), 0);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    // Scalar LOAD should pass through
    assert_is_load(&result);
    assert_eq!(result.dtype().vcount(), 1);
}

/// Test: Scalar INDEX passes through.
#[test]
fn test_devectorize_scalar_index_passthrough() {
    let buffer = create_buffer(64);
    let index = create_index(buffer.clone(), 5);

    let result = apply_devectorize(&index);

    // Scalar INDEX should remain unchanged
    assert_is_index(&result);
}

// =============================================================================
// Empty/Trivial Tests
// =============================================================================

/// Test: Empty SINK passes through.
#[test]
fn test_devectorize_empty_sink() {
    let sink = UOp::sink(vec![]);

    let result = apply_devectorize(&sink);

    // Empty SINK should remain as SINK
    match result.op() {
        Op::Sink { sources, .. } => {
            assert!(sources.is_empty());
        }
        other => panic!("Expected SINK, got {:?}", other),
    }
}

/// Test: SINK with single NOOP.
#[test]
fn test_devectorize_sink_noop() {
    let noop = UOp::noop();
    let sink = UOp::sink(vec![noop]);

    let result = apply_devectorize(&sink);

    // NOOP is dropped from SINK by sym_phase3_patterns (Tinygrad sym lines 422-424)
    match result.op() {
        Op::Sink { sources, .. } => {
            assert_eq!(sources.len(), 0, "NOOP should be dropped from SINK");
        }
        Op::Noop => {}
        other => panic!("Expected empty SINK or Noop, got {:?}", other),
    }
}

// =============================================================================
// Precision Tests
// =============================================================================

/// Test: Half precision (f16) buffer handling.
#[test]
fn test_devectorize_half_precision() {
    let buffer = create_buffer_typed(64, ScalarDType::Float16);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    // Should preserve f16 dtype and produce 4 elements total
    assert_eq!(result.dtype().base(), ScalarDType::Float16, "Base dtype should be f16");
    assert_eq!(result.dtype().vcount(), 4, "Total vcount should be 4");
    assert!(count_loads(&result) >= 1, "Should have at least one LOAD");
}

/// Test: Int8 buffer handling.
#[test]
fn test_devectorize_int8() {
    let buffer = create_buffer_typed(64, ScalarDType::Int8);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    assert_eq!(result.dtype().base(), ScalarDType::Int8, "Base dtype should be i8");
    assert_eq!(result.dtype().vcount(), 4, "Total vcount should be 4");
}

/// Test: UInt8 buffer handling.
#[test]
fn test_devectorize_uint8() {
    let buffer = create_buffer_typed(64, ScalarDType::UInt8);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    assert_eq!(result.dtype().base(), ScalarDType::UInt8, "Base dtype should be u8");
    assert_eq!(result.dtype().vcount(), 4, "Total vcount should be 4");
}

// =============================================================================
// Mixed Dtype Tests
// =============================================================================

/// Test: Multiple dtypes in same kernel.
#[test]
fn test_devectorize_mixed_dtypes() {
    let buffer_f32 = create_buffer_typed(64, ScalarDType::Float32);
    let buffer_i32 = create_buffer_typed(64, ScalarDType::Int32);

    // Load f32
    let index_f32 = create_vector_index_iota(buffer_f32.clone(), 4);
    let load_f32 = UOp::load().buffer(buffer_f32.clone()).index(index_f32).call();

    // Load i32
    let index_i32 = create_vector_index_iota(buffer_i32.clone(), 4);
    let load_i32 = UOp::load().buffer(buffer_i32.clone()).index(index_i32).call();

    // Process both
    let result_f32 = apply_devectorize(&load_f32);
    let result_i32 = apply_devectorize(&load_i32);

    assert_eq!(result_f32.dtype().base(), ScalarDType::Float32);
    assert_eq!(result_i32.dtype().base(), ScalarDType::Int32);
}

// =============================================================================
// Address Space Tests
// =============================================================================

/// Test: Local (shared) memory handling.
#[test]
fn test_devectorize_local_memory() {
    let buffer = create_buffer_local(64, ScalarDType::Float32);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    // Should work with local memory, preserving vcount and dtype
    assert_eq!(result.dtype().vcount(), 4, "Total vcount should be 4");
    assert_eq!(result.dtype().base(), ScalarDType::Float32, "Base dtype should be f32");
    assert!(count_loads(&result) >= 1, "Should have at least one LOAD");
}

/// Test: Different address spaces in same kernel.
#[test]
fn test_devectorize_mixed_addrspaces() {
    let buffer_global = create_buffer(64);
    let buffer_local = create_buffer_local(64, ScalarDType::Float32);

    let index_global = create_vector_index_iota(buffer_global.clone(), 4);
    let load_global = UOp::load().buffer(buffer_global.clone()).index(index_global).call();

    let index_local = create_vector_index_iota(buffer_local.clone(), 4);
    let load_local = UOp::load().buffer(buffer_local.clone()).index(index_local).call();

    let result_global = apply_devectorize(&load_global);
    let result_local = apply_devectorize(&load_local);

    // Both should produce valid results with preserved vcount
    assert_eq!(result_global.dtype().vcount(), 4, "Global vcount should be 4");
    assert_eq!(result_local.dtype().vcount(), 4, "Local vcount should be 4");
    assert!(count_loads(&result_global) >= 1, "Global should have LOADs");
    assert!(count_loads(&result_local) >= 1, "Local should have LOADs");
}

// =============================================================================
// Large Vector Tests
// =============================================================================

/// Test: Very large vector (vec64).
#[test]
fn test_devectorize_very_large_vector() {
    let buffer = create_buffer(512);

    // Create codegen PARAM and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(10000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::param(def_id, 512, buffer.dtype(), None);
    let buf_vec = define.broadcast(64);

    // Create vec64 index
    let indices: SmallVec<[Arc<UOp>; 4]> = (0..64).map(|i| UOp::const_(DType::Index, ConstValue::Int(i))).collect();
    let vec_idx = UOp::vectorize(indices);

    let index =
        UOp::new(Op::Index { buffer: buf_vec, indices: smallvec::smallvec![vec_idx], gate: None }, DType::Float32);

    let load = UOp::load().buffer(define).index(index).call();

    let result = apply_devectorize(&load);

    // Should handle vec64 (will be split into smaller chunks)
    assert_eq!(result.dtype().vcount(), 64, "Total vcount should be 64");
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have LOADs");
}

/// Test: Vec32 access.
#[test]
fn test_devectorize_vec32() {
    let buffer = create_buffer(256);

    // Create codegen PARAM and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(11000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::param(def_id, 256, buffer.dtype(), None);
    let buf_vec = define.broadcast(32);

    let indices: SmallVec<[Arc<UOp>; 4]> = (0..32).map(|i| UOp::const_(DType::Index, ConstValue::Int(i))).collect();
    let vec_idx = UOp::vectorize(indices);

    let index =
        UOp::new(Op::Index { buffer: buf_vec, indices: smallvec::smallvec![vec_idx], gate: None }, DType::Float32);

    let load = UOp::load().buffer(define).index(index).call();

    let result = apply_devectorize(&load);

    assert_eq!(result.dtype().vcount(), 32, "Total vcount should be 32");
    assert!(count_loads(&result) >= 1, "Should have LOADs");
}

// =============================================================================
// Unaligned Access Tests
// =============================================================================

/// Test: Non-power-of-2 offset.
#[test]
fn test_devectorize_unaligned_access() {
    let buffer = create_buffer(64);

    // Index starting at 3 (not aligned to 4)
    let index = create_vector_index_offset(buffer.clone(), 4, 3);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    // Should still produce valid result with preserved vcount
    assert_eq!(result.dtype().vcount(), 4, "Total vcount should be 4");
    assert!(count_loads(&result) >= 1, "Should have LOADs");
}

/// Test: Odd vector size (vec3).
#[test]
fn test_devectorize_vec3() {
    let buffer = create_buffer(64);

    // Create codegen PARAM and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(12000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::param(def_id, 64, buffer.dtype(), None);
    let buf_vec = define.broadcast(3);

    let indices: SmallVec<[Arc<UOp>; 4]> = (0..3).map(|i| UOp::const_(DType::Index, ConstValue::Int(i))).collect();
    let vec_idx = UOp::vectorize(indices);

    let index =
        UOp::new(Op::Index { buffer: buf_vec, indices: smallvec::smallvec![vec_idx], gate: None }, DType::Float32);

    let load = UOp::load().buffer(define).index(index).call();

    let result = apply_devectorize(&load);

    // vec3 should be handled (split into smaller pieces)
    assert_eq!(result.dtype().vcount(), 3, "Total vcount should be 3");
    assert!(count_loads(&result) >= 1, "Should have LOADs");
}

/// Test: Vec5 (non-power-of-2, larger than 4).
#[test]
fn test_devectorize_vec5() {
    let buffer = create_buffer(64);

    // Create codegen PARAM and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(13000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::param(def_id, 64, buffer.dtype(), None);
    let buf_vec = define.broadcast(5);

    let indices: SmallVec<[Arc<UOp>; 4]> = (0..5).map(|i| UOp::const_(DType::Index, ConstValue::Int(i))).collect();
    let vec_idx = UOp::vectorize(indices);

    let index =
        UOp::new(Op::Index { buffer: buf_vec, indices: smallvec::smallvec![vec_idx], gate: None }, DType::Float32);

    let load = UOp::load().buffer(define).index(index).call();

    let result = apply_devectorize(&load);

    // vec5 = 4 + 1, should be split but total vcount preserved
    assert_eq!(result.dtype().vcount(), 5, "Total vcount should be 5");
    assert!(count_loads(&result) >= 1, "Should have LOADs");
}

// =============================================================================
// Constants and Special Values
// =============================================================================

/// Test: Zero index.
#[test]
fn test_devectorize_zero_index() {
    let buffer = create_buffer(64);
    let index = create_vector_index_offset(buffer.clone(), 4, 0);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    assert_eq!(result.dtype().vcount(), 4, "Total vcount should be 4");
    assert!(count_loads(&result) >= 1, "Should have LOADs");
}

/// Test: Large constant offset.
#[test]
fn test_devectorize_large_offset() {
    let buffer = create_buffer(10000);
    let index = create_vector_index_offset(buffer.clone(), 4, 9000);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    assert_eq!(result.dtype().vcount(), 4, "Total vcount should be 4");
    assert!(count_loads(&result) >= 1, "Should have LOADs");
}

// =============================================================================
// Idempotency Tests
// =============================================================================

/// Test: Applying devectorize twice produces same result.
#[test]
fn test_devectorize_idempotent() {
    let buffer = create_buffer(64);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result1 = apply_devectorize(&load);
    let result2 = apply_devectorize(&result1);

    // Second application should not change the result
    // (structure should be equivalent even if IDs differ)
    assert_eq!(result1.dtype(), result2.dtype());
    assert_eq!(count_loads(&result1), count_loads(&result2));
}

// =============================================================================
// Regression Tests
// =============================================================================

/// Regression: Ensure PTRCAT sources are preserved correctly.
#[test]
fn test_regression_ptrcat_sources() {
    let buffer = create_buffer(64);

    let idx1 = create_index(buffer.clone(), 0);
    let idx2 = create_index(buffer.clone(), 1);
    let idx3 = create_index(buffer.clone(), 2);
    let idx4 = create_index(buffer.clone(), 3);

    let ptrcat = UOp::ptrcat().sources(vec![idx1, idx2, idx3, idx4]).call();

    // PTRCAT should have 4 sources
    let sources = unwrap_ptrcat(&ptrcat);
    assert_eq!(sources.len(), 4);
}

/// Regression: GEP indices should be preserved during transforms.
#[test]
fn test_regression_gep_indices_preserved() {
    let vec = create_vector_float_iota(8);
    let gep = vec.gep(vec![1, 3, 5, 7]);

    let (_, indices) = unwrap_gep(&gep);
    assert_eq!(indices, vec![1, 3, 5, 7]);
}

// =============================================================================
// fold_expanded_index: contiguous grouping
// =============================================================================

/// Contiguous offsets [R+0, R+1, R+2, R+3] should be grouped into a single
/// wide PTRCAT entry (cast to vec4 pointer), NOT 4 separate scalar entries.
/// This is the key optimization that makes `contiguous_gep_load_patterns` redundant.
#[test]
fn test_fold_expanded_index_groups_contiguous() {
    use crate::devectorize::load_store_folding_patterns;
    use crate::rewrite::graph_rewrite;
    use morok_dtype::AddrSpace;

    let buf = UOp::param(0, 64, DType::Scalar(ScalarDType::Float32).ptr(Some(64), AddrSpace::Global), None);
    let r1 = UOp::range_axis(
        UOp::const_(DType::Index, ConstValue::Int(16)),
        morok_ir::AxisId::Renumbered(0),
        morok_ir::AxisType::Loop,
    );

    // VECTORIZE(INDEX(buf, R+0), INDEX(buf, R+1), INDEX(buf, R+2), INDEX(buf, R+3))
    // — 4 contiguous offsets, should become a single vec4 PTRCAT entry
    let indices: SmallVec<[Arc<UOp>; 4]> = (0..4)
        .map(|i| {
            let offset = if i == 0 { r1.clone() } else { r1.add(&UOp::const_(DType::Index, ConstValue::Int(i))) };
            UOp::index().buffer(buf.clone()).indices(vec![offset]).ptr(true).call().unwrap()
        })
        .collect();
    let vectorize = UOp::vectorize(indices);

    let result = graph_rewrite(load_store_folding_patterns(), vectorize, &mut ());

    // Should produce identity GEP(PTRCAT(single_wide_index)) or just the wide index.
    // The PTRCAT should have exactly 1 source (the grouped vec4 pointer).
    let ptrcat_nodes: Vec<_> = result.toposort().into_iter().filter(|n| matches!(n.op(), Op::PtrCat { .. })).collect();
    assert_eq!(ptrcat_nodes.len(), 1, "Expected 1 PTRCAT, got {}", ptrcat_nodes.len());
    let ptrcat_sources = match ptrcat_nodes[0].op() {
        Op::PtrCat { sources } => sources,
        _ => unreachable!(),
    };
    // Single source = all 4 offsets grouped into one wide pointer
    assert_eq!(
        ptrcat_sources.len(),
        1,
        "Contiguous offsets should be grouped into 1 PTRCAT entry, got {}",
        ptrcat_sources.len()
    );
}

/// Non-contiguous offsets [R+0, R+16, R+32, R+48] must NOT be grouped —
/// each becomes a separate scalar PTRCAT entry.
#[test]
fn test_fold_expanded_index_no_group_scattered() {
    use crate::devectorize::load_store_folding_patterns;
    use crate::rewrite::graph_rewrite;
    use morok_dtype::AddrSpace;

    let buf = UOp::param(0, 64, DType::Scalar(ScalarDType::Float32).ptr(Some(64), AddrSpace::Global), None);
    let r1 = UOp::range_axis(
        UOp::const_(DType::Index, ConstValue::Int(16)),
        morok_ir::AxisId::Renumbered(0),
        morok_ir::AxisType::Loop,
    );

    // VECTORIZE(INDEX(buf, R+0), INDEX(buf, R+16), INDEX(buf, R+32), INDEX(buf, R+48))
    // — 4 scattered offsets, should become 4 separate PTRCAT entries
    let offsets = [0i64, 16, 32, 48];
    let indices: SmallVec<[Arc<UOp>; 4]> = offsets
        .iter()
        .map(|&off| {
            let offset = if off == 0 { r1.clone() } else { r1.add(&UOp::const_(DType::Index, ConstValue::Int(off))) };
            UOp::index().buffer(buf.clone()).indices(vec![offset]).ptr(true).call().unwrap()
        })
        .collect();
    let vectorize = UOp::vectorize(indices);

    let result = graph_rewrite(load_store_folding_patterns(), vectorize, &mut ());

    let ptrcat_nodes: Vec<_> = result.toposort().into_iter().filter(|n| matches!(n.op(), Op::PtrCat { .. })).collect();
    assert_eq!(ptrcat_nodes.len(), 1, "Expected 1 PTRCAT node, got {}", ptrcat_nodes.len());
    let ptrcat_sources = match ptrcat_nodes[0].op() {
        Op::PtrCat { sources } => sources,
        _ => unreachable!(),
    };
    // 4 separate entries — no grouping
    assert_eq!(
        ptrcat_sources.len(),
        4,
        "Scattered offsets should produce 4 PTRCAT entries, got {}",
        ptrcat_sources.len()
    );
}

// =============================================================================
// PtrCat Distribution (ScatterND regression)
// =============================================================================

/// ScatterND regression: STORE(vector_INDEX, WHERE(cond, updates, LOAD(vector_INDEX)))
/// where both STORE and LOAD INDEXes share the same offset vector.
/// Reproduces the exact pm_add_loads output structure from the ScatterND IR dump.
///
/// Root cause was a rewrite engine bug: `handle_link` resolved stale results
/// before pattern chains completed, leaving PTR_CAT in the output.
#[test]
fn test_scatternd_ptrcat_elimination() {
    use crate::devectorize::devectorize;
    use morok_dtype::AddrSpace;

    let dg0 = UOp::param(0, 64, DType::Scalar(ScalarDType::Float32).ptr(Some(64), AddrSpace::Global), None);
    let dg1 = UOp::param(1, 2, DType::Scalar(ScalarDType::Int64).ptr(Some(2), AddrSpace::Global), None);
    let dg2 = UOp::param(2, 32, DType::Scalar(ScalarDType::Float32).ptr(Some(32), AddrSpace::Global), None);
    let dg3 = UOp::param(3, 64, DType::Scalar(ScalarDType::Float32).ptr(Some(64), AddrSpace::Global), None);

    let dg0_vec = UOp::vectorize(smallvec::smallvec![dg0.clone(); 4]);
    let dg3_vec = UOp::vectorize(smallvec::smallvec![dg3.clone(); 4]);

    let r1 = UOp::range_axis(
        UOp::const_(DType::Index, ConstValue::Int(16)),
        morok_ir::AxisId::Renumbered(1),
        morok_ir::AxisType::Loop,
    );

    // Shared offset vector (used by both STORE and LOAD INDEXes)
    let r1_vec = UOp::vectorize(smallvec::smallvec![r1.clone(); 4]);
    let offsets = UOp::vconst(
        vec![ConstValue::Int(0), ConstValue::Int(16), ConstValue::Int(32), ConstValue::Int(48)],
        DType::Index,
    );
    let shared_add = r1_vec.add(&offsets);

    let store_idx = UOp::new(
        Op::Index { buffer: dg0_vec.clone(), indices: smallvec::smallvec![shared_add.clone()], gate: None },
        DType::Scalar(ScalarDType::Float32).ptr(Some(64), AddrSpace::Global),
    );
    let load_idx = UOp::new(
        Op::Index { buffer: dg3_vec.clone(), indices: smallvec::smallvec![shared_add], gate: None },
        DType::Scalar(ScalarDType::Float32).ptr(Some(64), AddrSpace::Global).vec(4),
    );
    let load = UOp::load().buffer(dg3_vec).index(load_idx).dtype(DType::Scalar(ScalarDType::Float32).vec(4)).call();

    // Condition: Or(Eq(cast(indices[0]), row_vals), Eq(cast(indices[1]), row_vals))
    let idx1_0 =
        UOp::index().buffer(dg1.clone()).indices(vec![UOp::const_(DType::Index, ConstValue::Int(0))]).call().unwrap();
    let idx1_1 =
        UOp::index().buffer(dg1.clone()).indices(vec![UOp::const_(DType::Index, ConstValue::Int(1))]).call().unwrap();
    let cast0 = UOp::load()
        .buffer(dg1.clone())
        .index(idx1_0)
        .dtype(DType::Scalar(ScalarDType::Int64))
        .call()
        .cast(DType::Scalar(ScalarDType::Int32));
    let cast1 = UOp::load()
        .buffer(dg1.clone())
        .index(idx1_1)
        .dtype(DType::Scalar(ScalarDType::Int64))
        .call()
        .cast(DType::Scalar(ScalarDType::Int32));
    let row_vconst =
        UOp::vconst(vec![ConstValue::Int(0), ConstValue::Int(1), ConstValue::Int(2), ConstValue::Int(3)], DType::Index);
    let minus_ones = UOp::vconst(vec![ConstValue::Int(-1); 4], DType::Scalar(ScalarDType::Int32));
    let row_vals = row_vconst.cast(DType::Scalar(ScalarDType::Int32).vec(4)).add(&minus_ones);

    let eq0 = UOp::vectorize(smallvec::smallvec![cast0; 4]).eq(&row_vals);
    let eq1 = UOp::vectorize(smallvec::smallvec![cast1; 4]).eq(&row_vals);
    let cond = eq0.or_(&eq1);

    // Updates
    let idx2_a = UOp::index()
        .buffer(dg2.clone())
        .indices(vec![UOp::const_(DType::Index, ConstValue::Int(16)).add(&r1)])
        .call()
        .unwrap();
    let idx2_b = UOp::index().buffer(dg2.clone()).indices(vec![r1.clone()]).call().unwrap();
    let load2_a = UOp::load().buffer(dg2.clone()).index(idx2_a).dtype(DType::Scalar(ScalarDType::Float32)).call();
    let load2_b = UOp::load().buffer(dg2.clone()).index(idx2_b).dtype(DType::Scalar(ScalarDType::Float32)).call();
    let inner_where = UOp::try_where(
        eq1,
        UOp::vectorize(smallvec::smallvec![load2_a; 4]),
        UOp::vectorize(smallvec::smallvec![load2_b; 4]),
    )
    .unwrap();
    let outer_where = UOp::try_where(cond, inner_where, load).unwrap();

    let store = store_idx.store_with_ranges(outer_where, smallvec::smallvec![r1]);
    let result = devectorize(&UOp::sink(vec![store]));

    let has_ptrcat = result.toposort().iter().any(|n| matches!(n.op(), Op::PtrCat { .. }));
    assert!(!has_ptrcat, "PTR_CAT survived devectorize! Result:\n{}", result.tree());
}
