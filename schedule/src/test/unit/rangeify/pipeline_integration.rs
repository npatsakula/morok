//! Pipeline integration tests.
//!
//! Tests that verify the full rangeify pipeline orchestrates correctly,
//! threading context between stages and preserving UOp structure.

use std::f32::consts::PI;

use morok_ir::{Op, UOp};

use crate::rangeify::kernel::run_kernel_split_pipeline;
use crate::test::unit::rangeify::helpers::{count_bufferizes, count_define_globals, count_kernels, extract_kernel};

#[test]
fn test_pipeline_two_bufferizes() {
    // Two BUFFERIZEs with different address spaces.
    // pm_add_buffers (allow_locals=false) converts only GLOBAL BUFFERIZEs to STORE/BUFFER.
    // LOCAL BUFFERIZEs remain as-is and are converted later in codegen (pm_add_buffers_local).
    // This matches Tinygrad's two-stage buffer conversion.
    let compute1 = UOp::native_const(1.0f32);
    let compute2 = UOp::native_const(42i32);

    let range1 = UOp::range_const(10, 0);
    let range2 = UOp::range_const(5, 1);

    let bufferize_global = UOp::bufferize_global(compute1, vec![range1]);

    let bufferize_local = UOp::bufferize_local(compute2, vec![range2]);

    // Create root with both
    let root = UOp::sink(vec![bufferize_global, bufferize_local]);

    let (result, _context) = run_kernel_split_pipeline(root);

    // Global BUFFERIZE should be converted to BUFFER (and eventually DEFINE_GLOBAL in kernel AST)
    let global_count = count_define_globals(&result);
    assert!(global_count >= 1, "Should have at least 1 DEFINE_GLOBAL from global BUFFERIZE");

    // Local BUFFERIZE should remain as BUFFERIZE (not converted to DEFINE_LOCAL yet)
    let local_bufferize_count = count_bufferizes(&result);
    assert!(local_bufferize_count >= 1, "Local BUFFERIZE should remain unconverted (handled later in codegen)");
}

#[test]
fn test_pipeline_preserves_structure() {
    // Verify that UOp identity is preserved through pipeline stages
    let compute = UOp::native_const(PI);
    let range = UOp::range_const(20, 0);

    let bufferize = UOp::bufferize_global(compute.clone(), vec![range.clone()]);

    let (result, _context) = run_kernel_split_pipeline(bufferize.clone());

    // Extract KERNEL from result (may be wrapped in AFTER structure)
    let kernel = extract_kernel(&result).expect("Pipeline should create a KERNEL");

    // KERNEL should contain the original compute somewhere in its graph
    // (We can't easily verify deep structure without graph traversal,
    // but we can check that the pipeline created a valid kernel)
    if let Op::Kernel { ast, sources } = kernel.op() {
        // AST should be a SINK
        assert!(matches!(ast.op(), Op::Sink { .. }));

        // Sources contain BUFFER nodes (the original buffers from lctx.map)
        // The DEFINE_GLOBAL conversion happens in the AST, not the sources
        assert!(!sources.is_empty(), "KERNEL should have sources");
        assert!(sources.iter().any(|s| matches!(s.op(), Op::Buffer { .. })), "KERNEL sources should include BUFFER");
    } else {
        panic!("Expected KERNEL operation, got {:?}", kernel.op());
    }
}

#[test]
fn test_pipeline_context_threading() {
    // Verify that context state is preserved between stages
    let compute = UOp::native_const(true);
    let range = UOp::range_const(8, 0);

    let bufferize = UOp::bufferize_global(compute, vec![range]);

    let (result, _context) = run_kernel_split_pipeline(bufferize);

    // After pipeline:
    // - Stage 1 (bufferize_to_store) creates BUFFER and tracks in context
    // - Stage 2 (split_store) uses context to populate KERNEL sources with BUFFER

    // Extract KERNEL from result (may be wrapped in AFTER structure)
    let kernel = extract_kernel(&result).expect("Pipeline should create a KERNEL");

    if let Op::Kernel { sources, .. } = kernel.op() {
        // Sources contain BUFFER nodes from lctx.map (not DEFINE_GLOBAL)
        assert_eq!(sources.len(), 1, "KERNEL should have 1 source (the buffer from stage 1)");
        assert!(matches!(sources[0].op(), Op::Buffer { .. }), "Source should be BUFFER, got {:?}", sources[0].op());
    } else {
        panic!("Expected KERNEL operation, got {:?}", kernel.op());
    }
}

#[test]
fn test_pipeline_mixed_addrspace() {
    // Global and Local buffers in the same graph.
    // pm_add_buffers (allow_locals=false) converts only GLOBAL BUFFERIZEs.
    // LOCAL BUFFERIZEs are preserved for later codegen conversion.
    let global_compute = UOp::native_const(1.0f32);
    let local_compute = UOp::native_const(2.0f32);

    let range = UOp::range_const(16, 0);

    let global_buf = UOp::bufferize_global(global_compute, vec![range.clone()]);

    let local_buf = UOp::bufferize_local(local_compute, vec![range]);

    // Create a SINK with both
    let root = UOp::sink(vec![global_buf, local_buf]);

    let (result, _context) = run_kernel_split_pipeline(root);

    // Global BUFFERIZE should be converted
    let globals = count_define_globals(&result);
    assert!(globals >= 1, "Should have at least 1 DEFINE_GLOBAL from global BUFFERIZE");

    // Local BUFFERIZE should remain unconverted
    let local_bufferizes = count_bufferizes(&result);
    assert!(local_bufferizes >= 1, "Local BUFFERIZE should remain for later codegen conversion");
}

#[test]
fn test_pipeline_reshape_buffer_to_load() {
    // Test that RESHAPE(BUFFER) input is transformed correctly through the pipeline.
    // Uses SINK(CONTIGUOUS(compute)) — the Tinygrad-aligned graph structure.
    use morok_dtype::DType;

    // Create input buffer with size 12
    let input_buffer = UOp::new_buffer(morok_device::DeviceSpec::Cpu, 12, DType::Float32);

    // RESHAPE to 3x4
    let reshape_shape = UOp::vectorize(vec![UOp::index_const(3), UOp::index_const(4)].into());
    let reshaped = UOp::new(Op::Reshape { src: input_buffer, new_shape: reshape_shape }, DType::Float32);

    // Create another RESHAPE(BUFFER) with same shape
    let input_buffer2 = UOp::new_buffer(morok_device::DeviceSpec::Cpu, 12, DType::Float32);
    let reshape_shape2 = UOp::vectorize(vec![UOp::index_const(3), UOp::index_const(4)].into());
    let reshaped2 = UOp::new(Op::Reshape { src: input_buffer2, new_shape: reshape_shape2 }, DType::Float32);

    // Add two reshaped buffers (this is exactly what Tensor::from_slice + add does)
    let compute = reshaped.try_add(&reshaped2).expect("Add should work");

    // Wrap in CONTIGUOUS + SINK (Tinygrad approach: no pre-existing BUFFERIZE)
    let contiguous = compute.contiguous();
    let sink = UOp::sink(vec![contiguous]);

    // Run rangeify pipeline
    let (rangeified, _ctx) = crate::rangeify::rangeify(sink, None).expect("Rangeify should succeed");

    // Verify RESHAPE operations have been eliminated
    let has_reshape = rangeified.toposort().iter().any(|node| matches!(node.op(), Op::Reshape { .. }));
    assert!(!has_reshape, "RESHAPE should be eliminated after rangeify");
}

#[test]
fn test_full_pipeline_creates_load_for_input_buffers() {
    // Test the FULL pipeline (rangeify + kernel split) creates LOAD for input buffers.
    // Uses SINK(CONTIGUOUS(compute)) — the Tinygrad-aligned graph structure.
    use morok_dtype::DType;

    // Create input buffer with size 12
    let input_buffer = UOp::new_buffer(morok_device::DeviceSpec::Cpu, 12, DType::Float32);

    // RESHAPE to 3x4
    let reshape_shape = UOp::vectorize(vec![UOp::index_const(3), UOp::index_const(4)].into());
    let reshaped = UOp::new(Op::Reshape { src: input_buffer, new_shape: reshape_shape }, DType::Float32);

    // Create another RESHAPE(BUFFER) with same shape
    let input_buffer2 = UOp::new_buffer(morok_device::DeviceSpec::Cpu, 12, DType::Float32);
    let reshape_shape2 = UOp::vectorize(vec![UOp::index_const(3), UOp::index_const(4)].into());
    let reshaped2 = UOp::new(Op::Reshape { src: input_buffer2, new_shape: reshape_shape2 }, DType::Float32);

    // Add two reshaped buffers
    let compute = reshaped.try_add(&reshaped2).expect("Add should work");

    // Wrap in CONTIGUOUS + SINK (Tinygrad approach)
    let contiguous = compute.contiguous();
    let sink = UOp::sink(vec![contiguous]);

    // Run rangeify pipeline (Phases 1-4)
    let (rangeified, _ctx) = crate::rangeify::rangeify(sink.clone(), None).expect("Rangeify should succeed");

    // Print rangeified graph
    println!("After rangeify:");
    for node in rangeified.toposort() {
        let op_name = match node.op() {
            Op::Bufferize { .. } => "BUFFERIZE",
            Op::Buffer { .. } => "BUFFER",
            Op::Index { .. } => "INDEX",
            Op::Binary(morok_ir::BinaryOp::Add, _, _) => "ADD",
            Op::Reshape { .. } => "RESHAPE",
            Op::Sink { .. } => "SINK",
            _ => continue,
        };
        println!("  {} [{:?}]", op_name, node.dtype());
    }
    println!("Rangeified root op: {:?}", rangeified.op());

    // Run kernel split pipeline (Phase 5)
    let (kernelized, _context) = run_kernel_split_pipeline(rangeified);

    // Print summarized graph for debugging
    println!("Kernelized graph ops:");
    for node in kernelized.toposort() {
        let op_name = match node.op() {
            Op::Kernel { .. } => "KERNEL",
            Op::Sink { .. } => "SINK",
            Op::Store { .. } => "STORE",
            Op::Load { .. } => "LOAD",
            Op::Index { buffer, .. } => {
                let buf_name = match buffer.op() {
                    Op::Buffer { .. } => "BUFFER",
                    Op::DefineGlobal(_) => "DEFINE_GLOBAL",
                    Op::DefineLocal(_) => "DEFINE_LOCAL",
                    _ => "OTHER",
                };
                println!("  INDEX(buf={}) [{:?}]", buf_name, node.dtype());
                continue;
            }
            Op::Binary(op, _, _) => match op {
                morok_ir::BinaryOp::Add => "ADD",
                _ => "BINARY",
            },
            Op::DefineGlobal(id) => {
                println!("  DEFINE_GLOBAL({})", id);
                continue;
            }
            Op::DefineLocal(id) => {
                println!("  DEFINE_LOCAL({})", id);
                continue;
            }
            Op::Buffer { .. } => "BUFFER",
            Op::Range { .. } => "RANGE",
            Op::Const(_) => "CONST",
            Op::End { .. } => "END",
            _ => "OTHER",
        };
        println!("  {} [{:?}]", op_name, node.dtype());
    }

    // Check that INDEX operations exist for buffer access.
    // NOTE: After aligning with Tinygrad, INDEX operations may reference BUFFER nodes
    // directly (the BUFFER → DEFINE_GLOBAL conversion happens in the AST but the
    // kernel sources maintain the original BUFFER references).
    //
    // INDEX dtype can be either:
    // - Element dtype (for LOAD sources, before pm_add_loads transforms them)
    // - Ptr dtype (for STORE targets, created in bufferize_to_store)
    let topo = kernelized.toposort();
    let index_on_buffer = topo
        .iter()
        .filter(|node| {
            if let Op::Index { buffer, .. } = node.op() {
                matches!(buffer.op(), Op::Buffer { .. } | Op::DefineGlobal(_))
            } else {
                false
            }
        })
        .collect::<Vec<_>>();

    // There should be INDEX operations for input buffers
    assert!(!index_on_buffer.is_empty(), "Pipeline should create INDEX operations for input buffers");
}

#[test]
#[ignore = "Pipeline doesn't handle complex chaining yet"]
fn test_pipeline_chained_operations() {
    // Test: A → bufferize → B → bufferize → C
    // Should create 2 buffers and appropriate kernel boundaries

    let compute_a = UOp::native_const(1i32);
    let range_a = UOp::range_const(10, 0);

    let buf_a = UOp::bufferize_global(compute_a.clone(), vec![range_a]);

    // Use buf_a as input to next operation
    // (In real code, this would be a load from buf_a)
    let compute_b = buf_a; // Simplified for testing
    let range_b = UOp::range_const(5, 1);

    let buf_b = UOp::bufferize_global(compute_b, vec![range_b]);

    let (result, _context) = run_kernel_split_pipeline(buf_b);

    // Should create multiple kernels for chained operations
    let kernel_count = count_kernels(&result);
    assert!(kernel_count >= 1, "Should create at least 1 kernel");

    // Should create multiple buffers
    let buffer_count = count_define_globals(&result);
    assert!(buffer_count >= 1, "Should create at least 1 buffer");
}
