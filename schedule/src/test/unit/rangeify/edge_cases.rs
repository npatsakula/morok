//! Edge case tests for rangeify pipeline.
//!
//! Tests zero-size operations, empty structures, and other corner cases
//! that could cause crashes or incorrect behavior.

use morok_dtype::DType;
use morok_ir::{AddrSpace, AxisType, BufferizeOpts, ConstValue, Op, UOp};
use smallvec::SmallVec;

use crate::rangeify::{KernelContext, bufferize_to_store::bufferize_to_store, pipeline::run_kernel_split_pipeline};

#[test]
fn test_zero_size_range() {
    // RANGE with end=0 should be handled gracefully
    let range_zero = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(0)), 0, AxisType::Loop);

    // Should create a valid RANGE operation
    assert!(matches!(range_zero.op(), Op::Range { .. }));

    if let Op::Range { end, .. } = range_zero.op() {
        // End should be 0
        if let Op::Const(cv) = end.op() {
            assert_eq!(cv.0, ConstValue::Int(0));
        } else {
            panic!("Expected const end");
        }
    }
}

#[test]
fn test_empty_bufferize() {
    let mut ctx = KernelContext::new();

    // BUFFERIZE with no ranges (scalar store)
    let compute = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let bufferize = UOp::new(
        Op::Bufferize {
            compute: compute.clone(),
            ranges: SmallVec::new(),
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    // Should convert successfully
    let result = bufferize_to_store(&bufferize, &mut ctx);
    assert!(result.is_some());

    // Result should be a STORE (no END wrapper since no ranges)
    if let Op::Store { value, .. } = result.unwrap().op() {
        assert!(std::rc::Rc::ptr_eq(value, &compute));
    } else {
        panic!("Expected STORE operation for empty ranges");
    }
}

#[test]
fn test_zero_size_index() {
    // INDEX with zero indices (direct buffer access)
    let buffer = UOp::define_global(0, DType::Float32);

    // Create INDEX with empty indices
    let index = UOp::index(buffer.clone(), vec![]).expect("INDEX with no indices should work");

    // Should be a valid INDEX
    if let Op::Index { buffer: idx_buf, indices, .. } = index.op() {
        assert!(std::rc::Rc::ptr_eq(idx_buf, &buffer));
        assert_eq!(indices.len(), 0);
    } else {
        panic!("Expected INDEX operation");
    }
}

#[test]
fn test_zero_size_end() {
    // END with zero ranges
    let store = UOp::noop();
    let end = UOp::end(store.clone(), SmallVec::new());

    // Should create valid END
    if let Op::End { computation, ranges } = end.op() {
        assert!(std::rc::Rc::ptr_eq(computation, &store));
        assert_eq!(ranges.len(), 0);
    } else {
        panic!("Expected END operation");
    }
}

#[test]
fn test_zero_size_pipeline() {
    // Full pipeline with zero-size BUFFERIZE
    let compute = UOp::const_(DType::Int32, ConstValue::Int(0));
    let bufferize = UOp::new(
        Op::Bufferize {
            compute,
            ranges: SmallVec::new(),
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Int32,
    );

    // Run through pipeline
    let result = run_kernel_split_pipeline(bufferize);

    // Should create a KERNEL even with zero ranges
    assert!(matches!(result.op(), Op::Kernel { .. }));
}

#[test]
fn test_bufferize_with_zero_range_inside() {
    let mut ctx = KernelContext::new();

    // Create BUFFERIZE with a zero-sized range
    let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let range_zero = UOp::range_const(0, 0);

    let bufferize = UOp::new(
        Op::Bufferize {
            compute: compute.clone(),
            ranges: smallvec::smallvec![range_zero.clone()],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    // Should convert (even though range is zero)
    let result = bufferize_to_store(&bufferize, &mut ctx);
    assert!(result.is_some());

    // Should create END with the zero range
    if let Op::End { ranges, .. } = result.unwrap().op() {
        assert_eq!(ranges.len(), 1);
        assert!(std::rc::Rc::ptr_eq(&ranges[0], &range_zero));
    } else {
        panic!("Expected END operation");
    }
}

#[test]
fn test_multiple_zero_ranges() {
    let mut ctx = KernelContext::new();

    // Create BUFFERIZE with multiple zero-sized ranges
    let compute = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let range1 = UOp::range_const(0, 0);
    let range2 = UOp::range_const(0, 1);

    let bufferize = UOp::new(
        Op::Bufferize {
            compute,
            ranges: smallvec::smallvec![range1.clone(), range2.clone()],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Local },
        },
        DType::Bool,
    );

    // Should convert and preserve both ranges
    let result = bufferize_to_store(&bufferize, &mut ctx);
    assert!(result.is_some());

    // Should be BARRIER wrapping END with 2 ranges
    if let Op::Barrier { src, .. } = result.unwrap().op() {
        if let Op::End { ranges, .. } = src.op() {
            assert_eq!(ranges.len(), 2);
        } else {
            panic!("Expected END inside BARRIER");
        }
    } else {
        panic!("Expected BARRIER for local buffer");
    }
}
