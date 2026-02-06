//! Kernel count validation tests.
//!
//! Tests that verify the number of kernels created by the pipeline,
//! ensuring fusion decisions are correct without needing actual tensor data.

use morok_ir::{Op, UOp};

use crate::rangeify::{KernelContext, run_kernel_split_pipeline};
use crate::test::unit::rangeify::helpers::{count_kernels, count_stores};

#[test]
fn test_single_store_one_kernel() {
    // Single BUFFERIZE → Should create 1 KERNEL
    let compute = UOp::native_const(1.0f32);
    let range = UOp::range_const(10, 0);

    let bufferize = UOp::bufferize_global(compute, vec![range]);

    let (result, _context) = run_kernel_split_pipeline(bufferize);

    // Should create exactly 1 KERNEL
    assert_eq!(count_kernels(&result), 1);
}

#[test]
fn test_double_store_two_kernels() {
    // Two independent BUFFERIZEs → Should create 2 KERNELs
    let compute1 = UOp::native_const(1.0f32);
    let compute2 = UOp::native_const(2.0f32);

    let range1 = UOp::range_const(10, 0);
    let range2 = UOp::range_const(20, 1);

    let bufferize1 = UOp::bufferize_global(compute1, vec![range1]);

    let bufferize2 = UOp::bufferize_global(compute2, vec![range2]);

    // Create a root that references both (e.g., SINK)
    let root = UOp::sink(vec![bufferize1, bufferize2]);

    let (result, _context) = run_kernel_split_pipeline(root);

    // Should create 2 KERNELs (one per BUFFERIZE)
    assert_eq!(count_kernels(&result), 2);
}

#[test]
fn test_shared_buffer_one_kernel() {
    let mut ctx = KernelContext::new();

    // Same BUFFERIZE used twice → should reuse buffer
    let compute = UOp::native_const(42i32);
    let range = UOp::range_const(5, 0);

    let bufferize = UOp::bufferize_global(compute, vec![range]);

    // Convert to STORE twice (simulating reuse)
    use crate::rangeify::transforms::bufferize_to_store;

    let _result1 = bufferize_to_store(&bufferize, &mut ctx, true);
    let _result2 = bufferize_to_store(&bufferize, &mut ctx, true);

    // For BUFFER ops (global address space), global_counter is NOT incremented
    // But the buffer should still be tracked and reused
    assert!(ctx.has_buffer(&bufferize));

    // Getting the buffer twice should return the same one
    let buf1 = ctx.get_buffer(&bufferize).unwrap();
    let buf2 = ctx.get_buffer(&bufferize).unwrap();
    assert!(std::sync::Arc::ptr_eq(buf1, buf2));
}

#[test]
fn test_independent_buffers_separate() {
    let mut ctx = KernelContext::new();

    // Different BUFFERIZEs → separate buffers (BUFFER nodes, not DEFINE_GLOBAL)
    let compute1 = UOp::native_const(1.0f32);
    let compute2 = UOp::native_const(2.0f32);

    let range = UOp::range_const(10, 0);

    let bufferize1 = UOp::bufferize_global(compute1, vec![range.clone()]);

    let bufferize2 = UOp::bufferize_global(compute2, vec![range]);

    use crate::rangeify::transforms::bufferize_to_store;

    bufferize_to_store(&bufferize1, &mut ctx, true);
    bufferize_to_store(&bufferize2, &mut ctx, true);

    // For BUFFER ops, global_counter is NOT incremented
    // But both should be tracked separately
    assert!(ctx.has_buffer(&bufferize1));
    assert!(ctx.has_buffer(&bufferize2));

    // Buffers should be different BUFFER nodes
    let buf1 = ctx.get_buffer(&bufferize1).unwrap();
    let buf2 = ctx.get_buffer(&bufferize2).unwrap();
    assert!(!std::sync::Arc::ptr_eq(buf1, buf2));
}

#[test]
fn test_nested_end_operations() {
    // Nested END operations should each contribute to structure
    let store = UOp::noop();
    let range1 = UOp::range_const(4, 0);
    let range2 = UOp::range_const(8, 1);

    // Create nested ENDs (unusual but should handle)
    let end1 = store.end(smallvec::smallvec![range1.clone()]);
    let end2 = end1.clone().end(smallvec::smallvec![range2.clone()]);

    // Verify structure
    if let Op::End { computation, ranges } = end2.op() {
        // Outer END should have 1 range
        assert_eq!(ranges.len(), 1);
        assert!(std::sync::Arc::ptr_eq(&ranges[0], &range2));

        // Inner computation should be another END
        assert!(std::sync::Arc::ptr_eq(computation, &end1));

        if let Op::End { ranges: inner_ranges, .. } = computation.op() {
            assert_eq!(inner_ranges.len(), 1);
            assert!(std::sync::Arc::ptr_eq(&inner_ranges[0], &range1));
        }
    } else {
        panic!("Expected END operation");
    }
}

#[test]
fn test_pipeline_kernel_count() {
    // After full pipeline, count kernels
    // Use non-OUTER (Loop) range since OUTER ranges are skipped by split_store
    // (OUTER ranges are handled at a higher level in the scheduler)
    let compute = UOp::native_const(false);
    let range = UOp::range_const(100, 0); // Loop range, not OUTER

    let bufferize = UOp::bufferize_global(compute, vec![range]);

    let (result, _context) = run_kernel_split_pipeline(bufferize);

    // Verify exactly 1 KERNEL was created
    assert_eq!(count_kernels(&result), 1);

    // Verify STORE is inside the KERNEL (wrapped, not bare)
    // We expect 1 STORE inside the KERNEL body - this is correct!
    // The STORE represents the actual memory write operation.
    assert_eq!(count_stores(&result), 1, "STORE should be inside KERNEL");

    // Note: After aligning with Tinygrad, buffers may be BUFFER nodes in the graph
    // rather than DEFINE_GLOBAL. The count depends on the pattern rewriting behavior.
    // The important thing is that we have a valid kernel with sources.
}
