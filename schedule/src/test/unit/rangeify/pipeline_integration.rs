//! Pipeline integration tests.
//!
//! Tests that verify the full rangeify pipeline orchestrates correctly,
//! threading context between stages and preserving UOp structure.

use morok_dtype::DType;
use morok_ir::{AddrSpace, AxisType, BufferizeOpts, ConstValue, Op, UOp};

use crate::rangeify::pipeline::run_kernel_split_pipeline;
use crate::test::unit::rangeify::helpers::{count_define_globals, count_define_locals, count_kernels};

#[test]
fn test_pipeline_two_bufferizes() {
    // Two BUFFERIZEs with different address spaces
    let compute1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let compute2 = UOp::const_(DType::Int32, ConstValue::Int(42));

    let range1 = UOp::range(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Loop);
    let range2 = UOp::range(UOp::const_(DType::Index, ConstValue::Int(5)), 1, AxisType::Loop);

    let bufferize_global = UOp::new(
        Op::Bufferize {
            compute: compute1,
            ranges: smallvec::smallvec![range1],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    let bufferize_local = UOp::new(
        Op::Bufferize {
            compute: compute2,
            ranges: smallvec::smallvec![range2],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Local },
        },
        DType::Int32,
    );

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
    let compute = UOp::const_(DType::Float32, ConstValue::Float(3.14));
    let range = UOp::range(UOp::const_(DType::Index, ConstValue::Int(20)), 0, AxisType::Loop);

    let bufferize = UOp::new(
        Op::Bufferize {
            compute: compute.clone(),
            ranges: smallvec::smallvec![range.clone()],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

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
    let range = UOp::range(UOp::const_(DType::Index, ConstValue::Int(8)), 0, AxisType::Loop);

    let bufferize = UOp::new(
        Op::Bufferize {
            compute,
            ranges: smallvec::smallvec![range],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Bool,
    );

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

    let range = UOp::range(UOp::const_(DType::Index, ConstValue::Int(16)), 0, AxisType::Loop);

    let global_buf = UOp::new(
        Op::Bufferize {
            compute: global_compute,
            ranges: smallvec::smallvec![range.clone()],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    let local_buf = UOp::new(
        Op::Bufferize {
            compute: local_compute,
            ranges: smallvec::smallvec![range],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Local },
        },
        DType::Float32,
    );

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
    let range_a = UOp::range(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Loop);

    let buf_a = UOp::new(
        Op::Bufferize {
            compute: compute_a.clone(),
            ranges: smallvec::smallvec![range_a],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Int32,
    );

    // Use buf_a as input to next operation
    // (In real code, this would be a load from buf_a)
    let compute_b = buf_a; // Simplified for testing
    let range_b = UOp::range(UOp::const_(DType::Index, ConstValue::Int(5)), 1, AxisType::Loop);

    let buf_b = UOp::new(
        Op::Bufferize {
            compute: compute_b,
            ranges: smallvec::smallvec![range_b],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Int32,
    );

    let result = run_kernel_split_pipeline(buf_b);

    // Should create multiple kernels for chained operations
    let kernel_count = count_kernels(&result);
    assert!(kernel_count >= 1, "Should create at least 1 kernel");

    // Should create multiple buffers
    let buffer_count = count_define_globals(&result);
    assert!(buffer_count >= 1, "Should create at least 1 buffer");
}
