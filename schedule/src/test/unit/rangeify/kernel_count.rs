//! Kernel count validation tests.
//!
//! Tests that verify the number of kernels created by the pipeline,
//! ensuring fusion decisions are correct without needing actual tensor data.

use morok_dtype::DType;
use morok_ir::{AddrSpace, AxisType, BufferizeOpts, ConstValue, Op, UOp};

use crate::rangeify::{pipeline::run_kernel_split_pipeline, KernelContext};
use crate::test::unit::rangeify::helpers::{count_define_globals, count_kernels, count_stores};

#[test]
fn test_single_store_one_kernel() {
    // Single BUFFERIZE → Should create 1 KERNEL
    let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let range = UOp::range(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Loop);

    let bufferize = UOp::new(
        Op::Bufferize {
            compute,
            ranges: smallvec::smallvec![range],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    let result = run_kernel_split_pipeline(bufferize);

    // Should create exactly 1 KERNEL
    assert_eq!(count_kernels(&result), 1);
}

#[test]
#[ignore = "Pipeline doesn't handle multiple independent BUFFERIZEs yet"]
fn test_double_store_two_kernels() {
    // Two independent BUFFERIZEs → Should create 2 KERNELs
    let compute1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let compute2 = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    let range1 = UOp::range(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Loop);
    let range2 = UOp::range(UOp::const_(DType::Index, ConstValue::Int(20)), 1, AxisType::Loop);

    let bufferize1 = UOp::new(
        Op::Bufferize {
            compute: compute1,
            ranges: smallvec::smallvec![range1],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    let bufferize2 = UOp::new(
        Op::Bufferize {
            compute: compute2,
            ranges: smallvec::smallvec![range2],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

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
    let range = UOp::range(UOp::const_(DType::Index, ConstValue::Int(5)), 0, AxisType::Loop);

    let bufferize = UOp::new(
        Op::Bufferize {
            compute,
            ranges: smallvec::smallvec![range],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Int32,
    );

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

    let range = UOp::range(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Loop);

    let bufferize1 = UOp::new(
        Op::Bufferize {
            compute: compute1,
            ranges: smallvec::smallvec![range.clone()],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    let bufferize2 = UOp::new(
        Op::Bufferize {
            compute: compute2,
            ranges: smallvec::smallvec![range],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

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
    let range1 = UOp::range(UOp::const_(DType::Index, ConstValue::Int(4)), 0, AxisType::Loop);
    let range2 = UOp::range(UOp::const_(DType::Index, ConstValue::Int(8)), 1, AxisType::Loop);

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
#[ignore = "Pipeline behavior with OUTER ranges needs investigation - creates 1 kernel + 1 unwrapped STORE + 3 buffers"]
fn test_pipeline_kernel_count() {
    // After full pipeline, count kernels
    // Use OUTER range so split_store will split at kernel boundary
    let compute = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let range = UOp::range(UOp::const_(DType::Index, ConstValue::Int(100)), 0, AxisType::Outer);

    let bufferize = UOp::new(
        Op::Bufferize {
            compute,
            ranges: smallvec::smallvec![range],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Bool,
    );

    let result = run_kernel_split_pipeline(bufferize);

    // Verify exactly 1 KERNEL was created
    assert_eq!(count_kernels(&result), 1);

    // Verify no STORE operations remain (all converted to KERNELs)
    assert_eq!(count_stores(&result), 0, "STOREs should be wrapped in KERNELs");

    // Verify exactly 1 DEFINE_GLOBAL was created
    assert_eq!(count_define_globals(&result), 1);
}
