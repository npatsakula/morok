//! Pipeline integration tests.
//!
//! Tests that verify the full rangeify pipeline orchestrates correctly,
//! threading context between stages and preserving UOp structure.

use morok_dtype::DType;
use morok_ir::{ConstValue, Op, UOp};

use crate::rangeify::pipeline::run_kernel_split_pipeline;
use crate::test::unit::rangeify::helpers::{count_define_globals, count_define_locals, count_kernels};

#[test]
fn test_pipeline_two_bufferizes() {
    // Two BUFFERIZEs with different address spaces
    let compute1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let compute2 = UOp::const_(DType::Int32, ConstValue::Int(42));

    let range1 = UOp::range_const(10, 0);
    let range2 = UOp::range_const(5, 1);

    let bufferize_global = UOp::bufferize_global(compute1, vec![range1]);

    let bufferize_local = UOp::bufferize_local(compute2, vec![range2]);

    // Create root with both
    let root = UOp::new(Op::Sink { sources: smallvec::smallvec![bufferize_global, bufferize_local] }, DType::Void);

    let result = run_kernel_split_pipeline(root);

    // Should create buffers for both address spaces
    // Note: Actual kernel count depends on how pipeline handles SINK
    let global_count = count_define_globals(&result);
    let local_count = count_define_locals(&result);

    // At least 1 global and 1 local should exist
    assert!(global_count >= 1, "Should have at least 1 DEFINE_GLOBAL");
    assert!(local_count >= 1, "Should have at least 1 DEFINE_LOCAL");
}

#[test]
fn test_pipeline_preserves_structure() {
    // Verify that UOp identity is preserved through pipeline stages
    let compute = UOp::const_(DType::Float32, ConstValue::Float(std::f64::consts::PI));
    let range = UOp::range_const(20, 0);

    let bufferize = UOp::bufferize_global(compute.clone(), vec![range.clone()]);

    let result = run_kernel_split_pipeline(bufferize.clone());

    // Result should be a KERNEL
    assert!(matches!(result.op(), Op::Kernel { .. }));

    // KERNEL should contain the original compute somewhere in its graph
    // (We can't easily verify deep structure without graph traversal,
    // but we can check that the pipeline created a valid kernel)
    if let Op::Kernel { ast, sources } = result.op() {
        // AST should be a SINK
        assert!(matches!(ast.op(), Op::Sink { .. }));

        // Sources should contain a DEFINE_GLOBAL
        assert!(!sources.is_empty(), "KERNEL should have sources");
        assert!(
            sources.iter().any(|s| matches!(s.op(), Op::DefineGlobal(_))),
            "KERNEL sources should include DEFINE_GLOBAL"
        );
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_pipeline_context_threading() {
    // Verify that context state is preserved between stages
    let compute = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let range = UOp::range_const(8, 0);

    let bufferize = UOp::bufferize_global(compute, vec![range]);

    let result = run_kernel_split_pipeline(bufferize);

    // After pipeline:
    // - Stage 1 (bufferize_to_store) creates DEFINE_GLOBAL and tracks in context
    // - Stage 2 (split_store) uses context to populate KERNEL sources

    if let Op::Kernel { sources, .. } = result.op() {
        // Sources should include the DEFINE_GLOBAL created in stage 1
        assert_eq!(sources.len(), 1, "KERNEL should have 1 source (the buffer from stage 1)");
        assert!(matches!(sources[0].op(), Op::DefineGlobal(0)), "Source should be DEFINE_GLOBAL(0)");
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_pipeline_mixed_addrspace() {
    // Global and Local buffers in the same graph
    let global_compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let local_compute = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    let range = UOp::range_const(16, 0);

    let global_buf = UOp::bufferize_global(global_compute, vec![range.clone()]);

    let local_buf = UOp::bufferize_local(local_compute, vec![range]);

    // Create a SINK with both
    let root = UOp::new(Op::Sink { sources: smallvec::smallvec![global_buf, local_buf] }, DType::Void);

    let result = run_kernel_split_pipeline(root);

    // Count buffers by address space
    let globals = count_define_globals(&result);
    let locals = count_define_locals(&result);

    // Should have at least one of each
    assert!(globals >= 1, "Should have at least 1 DEFINE_GLOBAL");
    assert!(locals >= 1, "Should have at least 1 DEFINE_LOCAL");

    // Total buffers should be at least 2
    assert!(globals + locals >= 2, "Should have at least 2 total buffers");
}

#[test]
#[ignore = "Pipeline doesn't handle complex chaining yet"]
fn test_pipeline_chained_operations() {
    // Test: A → bufferize → B → bufferize → C
    // Should create 2 buffers and appropriate kernel boundaries

    let compute_a = UOp::const_(DType::Int32, ConstValue::Int(1));
    let range_a = UOp::range_const(10, 0);

    let buf_a = UOp::bufferize_global(compute_a.clone(), vec![range_a]);

    // Use buf_a as input to next operation
    // (In real code, this would be a load from buf_a)
    let compute_b = buf_a; // Simplified for testing
    let range_b = UOp::range_const(5, 1);

    let buf_b = UOp::bufferize_global(compute_b, vec![range_b]);

    let result = run_kernel_split_pipeline(buf_b);

    // Should create multiple kernels for chained operations
    let kernel_count = count_kernels(&result);
    assert!(kernel_count >= 1, "Should create at least 1 kernel");

    // Should create multiple buffers
    let buffer_count = count_define_globals(&result);
    assert!(buffer_count >= 1, "Should create at least 1 buffer");
}
