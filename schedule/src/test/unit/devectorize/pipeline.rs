//! End-to-end devectorize() pipeline tests.
//!
//! Tests for the complete devectorize pass running all phases.
//! Verifies that the combined pipeline produces correct results.

use std::sync::Arc;

use morok_dtype::{DType, ScalarDType};
use morok_ir::types::ConstValue;
use morok_ir::{AxisId, AxisType, BinaryOp, Op, UOp};

use super::helpers::*;

// =============================================================================
// Contiguous Access Pipeline Tests
// =============================================================================

/// Test: Full pipeline for contiguous vec4 load.
///
/// INDEX(buffer, [0,1,2,3]) -> LOAD -> single contiguous vector load
#[test]
fn test_devectorize_contiguous_load() {
    let buffer = create_buffer(64);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    // Final result should have vcount 4 and be a valid load structure
    assert_vcount(&result, 4);

    // Should have LOADs somewhere in the tree
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}

/// Test: Full pipeline for contiguous vec4 store.
#[test]
fn test_devectorize_contiguous_store() {
    let buffer = create_buffer(64);
    let value = create_vector_float_iota(4);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let store = index.store(value);

    let result = apply_devectorize(&store);

    // Should have STOREs somewhere in the tree
    let store_count = count_stores(&result);
    assert!(store_count >= 1, "Should have at least one STORE");
}

/// Test: Full pipeline for strided vec4 access.
#[test]
fn test_devectorize_strided_load() {
    let buffer = create_buffer(128);
    // Strided access: [0, 2, 4, 6]
    let index = create_vector_index_scaled(buffer.clone(), 4, 2);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    // Strided access results in multiple scalar loads
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have LOADs for strided access");
}

// =============================================================================
// Complex Pattern Tests
// =============================================================================

/// Test: Matmul-like memory access pattern (8x8 tile).
///
/// Simulates typical tiled matmul memory access with output upcast.
#[test]
fn test_devectorize_matmul_pattern() {
    use crate::devectorize::{devectorize, pm_render};
    use crate::rewrite::graph_rewrite_bottom_up;

    let buffer = create_buffer(256);

    // Create 8 contiguous accesses (simulating 8-wide output upcast)
    let index = create_vector_index_iota(buffer.clone(), 8);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    // Step 1: Run devectorize (without pm_render)
    let after_devectorize = devectorize(&load);

    // Step 2: Run pm_render
    let result = graph_rewrite_bottom_up(&pm_render(), after_devectorize, &mut ());

    // Should produce vec8 result through devectorization
    assert_eq!(result.dtype().vcount(), 8, "Total vcount should be 8");
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD");
}

/// Test: Reduction with vector accumulator.
#[test]
fn test_devectorize_reduction_accumulator() {
    let buffer = create_buffer(64);

    // Load vec4 accumulator
    let acc_index = create_vector_index_iota(buffer.clone(), 4);
    let acc_load = UOp::load().buffer(buffer.clone()).index(acc_index).call();

    // Add to accumulator
    let values = create_vector_float_iota(4);
    let add = UOp::new(Op::Binary(BinaryOp::Add, acc_load, values), DType::Float32.vec(4));

    // Store back
    let store_index = create_vector_index_iota(buffer.clone(), 4);
    let store = store_index.store(add);

    let result = apply_devectorize(&store);

    // Should have both LOADs and STOREs
    let load_count = count_loads(&result);
    let store_count = count_stores(&result);
    assert!(load_count >= 1 && store_count >= 1);
}

/// Test: Multiple buffers in same kernel.
#[test]
fn test_devectorize_multiple_buffers() {
    let buffer_a = create_buffer(64);
    let buffer_b = create_buffer(64);
    let buffer_c = create_buffer(64);

    // Load from A
    let index_a = create_vector_index_iota(buffer_a.clone(), 4);
    let load_a = UOp::load().buffer(buffer_a.clone()).index(index_a).call();

    // Load from B
    let index_b = create_vector_index_iota(buffer_b.clone(), 4);
    let load_b = UOp::load().buffer(buffer_b.clone()).index(index_b).call();

    // Compute A + B
    let add = UOp::new(Op::Binary(BinaryOp::Add, load_a, load_b), DType::Float32.vec(4));

    // Store to C
    let index_c = create_vector_index_iota(buffer_c.clone(), 4);
    let store = index_c.store(add);

    let result = apply_devectorize(&store);

    // Should have multiple LOADs and STOREs
    let load_count = count_loads(&result);
    let store_count = count_stores(&result);
    assert!(load_count >= 2, "Should have LOADs from both A and B");
    assert!(store_count >= 1, "Should have STORE to C");
}

// =============================================================================
// Integration with pre_expand Tests
// =============================================================================

/// Test: Devectorize after pre_expand (simulated).
///
/// Tests the typical pipeline: pre_expand -> devectorize
#[test]
fn test_devectorize_after_pre_expand() {
    let buffer = create_buffer(64);

    // Create a simple kernel pattern that would come from pre_expand
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();
    let value = create_vector_float_iota(4);
    let add = UOp::new(Op::Binary(BinaryOp::Add, load, value), DType::Float32.vec(4));

    let store_index = create_vector_index_iota(buffer.clone(), 4);
    let store = store_index.store(add);

    // Apply devectorize
    let result = apply_devectorize(&store);

    // Should produce valid structure
    assert!(count_stores(&result) >= 1);
}

/// Test: Output upcast pattern.
///
/// Simulates output upcast where STORE has vectorized index.
#[test]
fn test_devectorize_with_output_upcast() {
    let buffer = create_buffer(256);

    // Create output upcast pattern: store vec8 to consecutive locations
    let index = create_vector_index_iota(buffer.clone(), 8);
    let value = create_vector_float_iota(8);
    let store = index.store(value);

    let result = apply_devectorize(&store);

    // Should handle vec8 output upcast
    let store_count = count_stores(&result);
    assert!(store_count >= 1);
}

// =============================================================================
// Symbolic Index Tests
// =============================================================================

/// Test: Loop-dependent index through full pipeline.
#[test]
fn test_devectorize_loop_index() {
    let buffer = create_buffer(256);

    // Create DEFINE_GLOBAL and broadcast to match Tinygrad's expand_index pattern
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(20000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::define_global(def_id, buffer.dtype());
    let buf_vec = define.broadcast(4);

    // Create index: range * 4 + [0,1,2,3]
    let range = UOp::new(
        Op::Range {
            end: UOp::const_(DType::Index, ConstValue::Int(64)),
            axis_id: AxisId::Renumbered(0),
            axis_type: AxisType::Loop,
        },
        DType::Index,
    );

    let base = UOp::new(Op::Binary(BinaryOp::Mul, range, UOp::const_(DType::Index, ConstValue::Int(4))), DType::Index);

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
    let index =
        UOp::new(Op::Index { buffer: buf_vec, indices: smallvec::smallvec![vec_idx], gate: None }, DType::Float32);

    let load = UOp::load().buffer(define).index(index).call();
    let result = apply_devectorize(&load);

    // Should produce valid vectorized load with vec4
    assert_eq!(result.dtype().vcount(), 4, "Total vcount should be 4");
    assert!(count_loads(&result) >= 1, "Should have at least one LOAD");
}

// =============================================================================
// Sink Tests
// =============================================================================

/// Test: Devectorize with SINK containing multiple stores.
#[test]
fn test_devectorize_sink_multiple_stores() {
    let buffer_a = create_buffer(64);
    let buffer_b = create_buffer(64);

    // Store to A
    let index_a = create_vector_index_iota(buffer_a.clone(), 4);
    let value_a = create_vector_float_iota(4);
    let store_a = index_a.store(value_a);

    // Store to B
    let index_b = create_vector_index_iota(buffer_b.clone(), 4);
    let value_b = create_vector_float_values(vec![10.0, 11.0, 12.0, 13.0]);
    let store_b = index_b.store(value_b);

    // SINK both stores
    let sink = UOp::sink(vec![store_a, store_b]);

    let result = apply_devectorize(&sink);

    // Should process both stores in SINK
    let store_count = count_stores(&result);
    assert!(store_count >= 2, "Should have stores from both operations");
}

// =============================================================================
// Dtype Preservation Tests
// =============================================================================

/// Test: Float16 through pipeline.
#[test]
fn test_devectorize_float16() {
    let buffer = create_buffer_typed(64, ScalarDType::Float16);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    // Base dtype should be preserved
    assert_eq!(result.dtype().base(), ScalarDType::Float16);
}

/// Test: Int32 through pipeline.
#[test]
fn test_devectorize_int32() {
    let buffer = create_buffer_typed(64, ScalarDType::Int32);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    assert_eq!(result.dtype().base(), ScalarDType::Int32);
}

/// Test: Bool through pipeline (special handling).
#[test]
fn test_devectorize_bool_pipeline() {
    let buffer = create_bool_buffer(64);
    let index = create_index(buffer.clone(), 0); // Scalar index for bool
    let load = UOp::load().buffer(buffer.clone()).index(index).call();

    let result = apply_devectorize(&load);

    // Bool loads go through special uint8 conversion
    assert!(
        result.dtype().base() == ScalarDType::Bool || result.dtype().base() == ScalarDType::UInt8,
        "Bool should be handled correctly"
    );
}
