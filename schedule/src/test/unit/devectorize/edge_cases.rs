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
        Op::Sink { sources } => {
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

    // Should handle trivial SINK
    match result.op() {
        Op::Sink { sources } => {
            assert_eq!(sources.len(), 1);
        }
        Op::Noop => {}
        other => panic!("Expected SINK or Noop, got {:?}", other),
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

    // Create DEFINE_GLOBAL and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(10000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::define_global(def_id, buffer.dtype());
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

    // Create DEFINE_GLOBAL and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(11000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::define_global(def_id, buffer.dtype());
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

    // Create DEFINE_GLOBAL and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(12000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::define_global(def_id, buffer.dtype());
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

    // Create DEFINE_GLOBAL and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(13000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::define_global(def_id, buffer.dtype());
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
