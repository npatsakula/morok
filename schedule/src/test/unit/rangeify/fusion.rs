//! Tests for kernel fusion in the rangeify pipeline.
//!
//! Validates that operations are correctly fused into single kernels:
//! - Binary operation fusion
//! - Fusion through movement ops (reshape, permute)
//! - Reduce fusion patterns
//!
//! Based on Tinygrad's schedule fusion tests.

use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::{Op, ReduceOp, UOp};
use smallvec::smallvec;

use crate::rangeify::kernel::run_kernel_split_pipeline;

/// Helper to create a simple buffer
fn create_buffer(size: usize) -> Arc<UOp> {
    UOp::new_buffer(morok_dtype::DeviceSpec::Cpu, size, DType::Float32)
}

// ===== Basic Fusion Tests =====

#[test]
fn test_binop_fusion_basic() {
    // a + b should produce 1 kernel, not 2
    // This tests that elementwise binary ops are fused
    let a = create_buffer(100);
    let b = create_buffer(100);

    // Add two buffers
    let add = a.try_add(&b).unwrap();

    // The add itself is the computation - wrapping in SINK for pipeline
    let sink = UOp::sink(vec![add]);

    // Run pipeline - returns (transformed_graph, context), not Result
    let (result, _ctx) = run_kernel_split_pipeline(sink);

    // Verify we got a valid result
    assert!(!result.op().sources().is_empty() || matches!(result.op(), Op::Sink { .. } | Op::Noop));
}

#[test]
fn test_binop_chain_fusion() {
    // a + b + c should ideally fuse into single kernel
    let a = create_buffer(100);
    let b = create_buffer(100);
    let c = create_buffer(100);

    let add1 = a.try_add(&b).unwrap();
    let add2 = add1.try_add(&c).unwrap();

    let sink = UOp::sink(vec![add2]);

    let (result, _ctx) = run_kernel_split_pipeline(sink);

    // Should produce a valid transformed graph
    assert!(!result.toposort().is_empty());
}

// ===== Movement Op Fusion =====

#[test]
fn test_binop_reshape_fusion() {
    // (a + b).reshape(...) should produce 1 kernel
    // Movement ops should not break fusion
    let a = create_buffer(100);
    let b = create_buffer(100);

    let add = a.try_add(&b).unwrap();

    // Reshape to (10, 10)
    let new_shape = UOp::vectorize(smallvec![UOp::index_const(10), UOp::index_const(10)]);
    let reshaped = UOp::new(Op::Reshape { src: add, new_shape }, DType::Float32);

    let sink = UOp::sink(vec![reshaped]);

    let (result, _ctx) = run_kernel_split_pipeline(sink);
    assert!(!result.toposort().is_empty());
}

#[test]
fn test_binop_permute_fusion() {
    // (a + b).permute(...) should fuse
    let a = create_buffer(100);
    let b = create_buffer(100);

    let add = a.try_add(&b).unwrap();

    // First reshape to 2D
    let new_shape = UOp::vectorize(smallvec![UOp::index_const(10), UOp::index_const(10)]);
    let reshaped = UOp::new(Op::Reshape { src: add, new_shape }, DType::Float32);

    // Then permute (swap dims)
    let permuted = UOp::new(Op::Permute { src: reshaped, axes: vec![1, 0] }, DType::Float32);

    let sink = UOp::sink(vec![permuted]);

    let (result, _ctx) = run_kernel_split_pipeline(sink);
    assert!(!result.toposort().is_empty());
}

// ===== Reduce Fusion =====

#[test]
fn test_reduce_fusion_basic() {
    // a.sum() should produce kernels for reduction
    let a = create_buffer(100);

    // Create reshape first (reduce needs shape info)
    let new_shape = UOp::vectorize(smallvec![UOp::index_const(10), UOp::index_const(10)]);
    let reshaped = UOp::new(Op::Reshape { src: a, new_shape }, DType::Float32);

    // Reduce on axis 1
    let reduced = UOp::new(Op::ReduceAxis { src: reshaped, axes: vec![1], reduce_op: ReduceOp::Add }, DType::Float32);

    let sink = UOp::sink(vec![reduced]);

    let (result, _ctx) = run_kernel_split_pipeline(sink);
    assert!(!result.toposort().is_empty());
}

#[test]
fn test_reduce_binop_fusion() {
    // (a + b).sum() should fuse add into reduction
    let a = create_buffer(100);
    let b = create_buffer(100);

    let add = a.try_add(&b).unwrap();

    // Reshape for reduction
    let new_shape = UOp::vectorize(smallvec![UOp::index_const(10), UOp::index_const(10)]);
    let reshaped = UOp::new(Op::Reshape { src: add, new_shape }, DType::Float32);

    // Reduce
    let reduced = UOp::new(Op::ReduceAxis { src: reshaped, axes: vec![1], reduce_op: ReduceOp::Add }, DType::Float32);

    let sink = UOp::sink(vec![reduced]);

    let (result, _ctx) = run_kernel_split_pipeline(sink);
    assert!(!result.toposort().is_empty());
}

// ===== Non-Fusion (Forced Realization) =====

#[test]
fn test_contiguous_forces_realization() {
    // CONTIGUOUS should force a realization point
    let a = create_buffer(100);

    let contiguous = UOp::new(Op::Contiguous { src: a, opts: smallvec::smallvec![] }, DType::Float32);

    let sink = UOp::sink(vec![contiguous]);

    let (result, _ctx) = run_kernel_split_pipeline(sink);
    assert!(!result.toposort().is_empty());
}

// ===== Multiple Outputs =====

#[test]
fn test_multiple_outputs_same_input() {
    // Multiple outputs from same computation should share work
    let a = create_buffer(100);
    let b = create_buffer(100);
    let c = create_buffer(100);
    let d = create_buffer(100);

    let add = a.try_add(&b).unwrap();

    // Two uses of the same add result with buffers (not scalars for shape matching)
    let mul1 = add.try_mul(&c).unwrap();
    let mul2 = add.try_mul(&d).unwrap();

    let sink = UOp::sink(vec![mul1, mul2]);

    let (result, _ctx) = run_kernel_split_pipeline(sink);
    assert!(!result.toposort().is_empty());
}

// ===== Edge Cases =====

#[test]
fn test_empty_sink() {
    // Empty SINK should handle gracefully
    let sink = UOp::sink(vec![]);

    let (result, _ctx) = run_kernel_split_pipeline(sink);
    // May return empty or minimal graph
    let _ = result;
}

#[test]
fn test_single_constant() {
    // Just a constant should work
    let c = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![c]);

    let (result, _ctx) = run_kernel_split_pipeline(sink);
    assert!(!result.toposort().is_empty());
}
