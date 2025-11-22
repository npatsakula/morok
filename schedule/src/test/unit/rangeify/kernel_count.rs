//! Kernel count validation tests.
//!
//! Tests that verify the number of kernels created by the pipeline,
//! ensuring fusion decisions are correct without needing actual tensor data.

use morok_dtype::DType;
use morok_ir::{AxisType, ConstValue, Op, UOp};

use crate::rangeify::{KernelContext, pipeline::run_kernel_split_pipeline};
use crate::test::unit::rangeify::helpers::{count_define_globals, count_ends, count_kernels, count_stores};

#[test]
fn test_single_store_one_kernel() {
    // Single BUFFERIZE → Should create 1 KERNEL
    let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let range = UOp::range_const(10, 0);

    let bufferize = UOp::bufferize_global(compute, vec![range]);

    let result = run_kernel_split_pipeline(bufferize);

    // Should create exactly 1 KERNEL
    assert_eq!(count_kernels(&result), 1);
}

#[test]
fn test_double_store_two_kernels() {
    // Two independent BUFFERIZEs → Should create 2 KERNELs
    let compute1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let compute2 = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    let range1 = UOp::range_const(10, 0);
    let range2 = UOp::range_const(20, 1);

    let bufferize1 = UOp::bufferize_global(compute1, vec![range1]);

    let bufferize2 = UOp::bufferize_global(compute2, vec![range2]);

    // Create a root that references both (e.g., SINK)
    let root = UOp::new(Op::Sink { sources: smallvec::smallvec![bufferize1, bufferize2] }, DType::Void);

    let result = run_kernel_split_pipeline(root);

    // Should create 2 KERNELs (one per BUFFERIZE)
    assert_eq!(count_kernels(&result), 2);
}

#[test]
fn test_shared_buffer_one_kernel() {
    let mut ctx = KernelContext::new();

    // Same BUFFERIZE used twice → should reuse buffer
    let compute = UOp::const_(DType::Int32, ConstValue::Int(42));
    let range = UOp::range_const(5, 0);

    let bufferize = UOp::bufferize_global(compute, vec![range]);

    // Convert to STORE twice (simulating reuse)
    use crate::rangeify::bufferize_to_store::bufferize_to_store;

    let _result1 = bufferize_to_store(&bufferize, &mut ctx);
    let _result2 = bufferize_to_store(&bufferize, &mut ctx);

    // Should only create 1 DEFINE_GLOBAL (buffer is tracked and reused)
    assert_eq!(ctx.global_counter, 1);
    assert!(ctx.has_buffer(&bufferize));

    // Getting the buffer twice should return the same one
    let buf1 = ctx.get_buffer(&bufferize).unwrap();
    let buf2 = ctx.get_buffer(&bufferize).unwrap();
    assert!(std::rc::Rc::ptr_eq(buf1, buf2));
}

#[test]
fn test_independent_buffers_separate() {
    let mut ctx = KernelContext::new();

    // Different BUFFERIZEs → separate buffers
    let compute1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let compute2 = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    let range = UOp::range_const(10, 0);

    let bufferize1 = UOp::bufferize_global(compute1, vec![range.clone()]);

    let bufferize2 = UOp::bufferize_global(compute2, vec![range]);

    use crate::rangeify::bufferize_to_store::bufferize_to_store;

    bufferize_to_store(&bufferize1, &mut ctx);
    bufferize_to_store(&bufferize2, &mut ctx);

    // Should create 2 separate DEFINE_GLOBALs
    assert_eq!(ctx.global_counter, 2);

    // Both should be tracked separately
    assert!(ctx.has_buffer(&bufferize1));
    assert!(ctx.has_buffer(&bufferize2));

    // Buffers should be different
    let buf1 = ctx.get_buffer(&bufferize1).unwrap();
    let buf2 = ctx.get_buffer(&bufferize2).unwrap();
    assert!(!std::rc::Rc::ptr_eq(buf1, buf2));
}

#[test]
fn test_nested_end_operations() {
    // Nested END operations should each contribute to structure
    let store = UOp::noop();
    let range1 = UOp::range_const(4, 0);
    let range2 = UOp::range_const(8, 1);

    // Create nested ENDs (unusual but should handle)
    let end1 = UOp::end(store, smallvec::smallvec![range1.clone()]);
    let end2 = UOp::end(end1.clone(), smallvec::smallvec![range2.clone()]);

    // Verify structure
    if let Op::End { computation, ranges } = end2.op() {
        // Outer END should have 1 range
        assert_eq!(ranges.len(), 1);
        assert!(std::rc::Rc::ptr_eq(&ranges[0], &range2));

        // Inner computation should be another END
        assert!(std::rc::Rc::ptr_eq(computation, &end1));

        if let Op::End { ranges: inner_ranges, .. } = computation.op() {
            assert_eq!(inner_ranges.len(), 1);
            assert!(std::rc::Rc::ptr_eq(&inner_ranges[0], &range1));
        }
    } else {
        panic!("Expected END operation");
    }
}

#[test]
fn test_pipeline_kernel_count() {
    // After full pipeline, count kernels
    // Use OUTER range so split_store will split at kernel boundary
    let compute = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(100)), 0, AxisType::Outer);

    let bufferize = UOp::bufferize_global(compute, vec![range]);

    let result = run_kernel_split_pipeline(bufferize);

    // Verify exactly 1 KERNEL was created
    assert_eq!(count_kernels(&result), 1);

    // Verify STORE is inside the KERNEL (wrapped, not bare)
    // We expect 1 STORE inside the KERNEL body - this is correct!
    // The STORE represents the actual memory write operation.
    assert_eq!(count_stores(&result), 1, "STORE should be inside KERNEL");

    // Verify END remains inside KERNEL (END wraps STORE in kernel body)
    // The END marks the range closure for the STORE operation
    assert_eq!(count_ends(&result), 1, "END should be inside KERNEL body");

    // Verify DEFINE_GLOBAL count (counts references, not unique nodes)
    // The same DEFINE_GLOBAL(0) appears 3 times due to hash-consing:
    // 1. In STORE buffer parameter
    // 2. In INDEX operation (indexing into the buffer)
    // 3. In KERNEL sources (as an argument)
    assert_eq!(count_define_globals(&result), 3, "DEFINE_GLOBAL referenced 3 times in hash-consed graph");
}
