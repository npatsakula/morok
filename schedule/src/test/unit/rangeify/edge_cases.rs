//! Edge case tests for rangeify pipeline.
//!
//! Tests zero-size operations, empty structures, and other corner cases
//! that could cause crashes or incorrect behavior.

use morok_dtype::DType;
use morok_ir::{AddrSpace, AxisId, AxisType, BufferizeOpts, ConstValue, Op, UOp};
use smallvec::SmallVec;

use crate::rangeify::{KernelContext, bufferize_to_store, run_kernel_split_pipeline};

#[test]
fn test_zero_size_range() {
    // RANGE with end=0 should be handled gracefully
    let range_zero = UOp::range_axis(UOp::index_const(0), AxisId::Renumbered(0), AxisType::Loop);

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
    let compute = UOp::native_const(42.0f32);
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

    // Result should be AFTER(passthrough=DEFINE_GLOBAL, deps=[STORE])
    // No ranges means no END wrapper, but still AFTER structure
    let result = result.unwrap();
    let Op::After { passthrough, deps } = result.op() else {
        panic!("Expected AFTER operation, got {:?}", result.op());
    };
    assert!(matches!(passthrough.op(), Op::DefineGlobal(_)));
    assert_eq!(deps.len(), 1);

    // deps[0] should be STORE (no ranges = no END)
    let Op::Store { value, .. } = deps[0].op() else {
        panic!("Expected STORE in AFTER deps, got {:?}", deps[0].op());
    };
    assert!(std::sync::Arc::ptr_eq(value, &compute));
}

#[test]
fn test_zero_size_index() {
    // INDEX with zero indices (direct buffer access)
    let buffer = UOp::define_global(0, DType::Float32);

    // Create INDEX with empty indices
    let index = UOp::index(buffer.clone(), vec![]).expect("INDEX with no indices should work");

    // Should be a valid INDEX
    if let Op::Index { buffer: idx_buf, indices, .. } = index.op() {
        assert!(std::sync::Arc::ptr_eq(idx_buf, &buffer));
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
        assert!(std::sync::Arc::ptr_eq(computation, &store));
        assert_eq!(ranges.len(), 0);
    } else {
        panic!("Expected END operation");
    }
}

#[test]
fn test_zero_size_pipeline() {
    // Full pipeline with zero-size BUFFERIZE
    let compute = UOp::native_const(0i32);
    let bufferize = UOp::new(
        Op::Bufferize {
            compute,
            ranges: SmallVec::new(),
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Int32,
    );

    // Run through pipeline
    let (result, _context) = run_kernel_split_pipeline(bufferize);

    // Should create a KERNEL even with zero ranges
    assert!(matches!(result.op(), Op::Kernel { .. }));
}

#[test]
#[should_panic(expected = "Cannot allocate buffer with symbolic size")]
fn test_bufferize_with_zero_range_inside() {
    let mut ctx = KernelContext::new();

    // Create BUFFERIZE with a zero-sized range
    // Zero-sized buffers are invalid (Tinygrad: "assert size > 0")
    let compute = UOp::native_const(1.0f32);
    let range_zero = UOp::range_const(0, 0);

    let bufferize = UOp::new(
        Op::Bufferize {
            compute: compute.clone(),
            ranges: smallvec::smallvec![range_zero.clone()],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    // Should panic because zero-sized buffers are not allowed
    let _result = bufferize_to_store(&bufferize, &mut ctx);
}

#[test]
#[should_panic(expected = "Cannot allocate buffer with symbolic size")]
fn test_multiple_zero_ranges() {
    let mut ctx = KernelContext::new();

    // Create BUFFERIZE with multiple zero-sized ranges
    // Zero-sized buffers are invalid (Tinygrad: "assert size > 0")
    let compute = UOp::native_const(true);
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

    // Should panic because zero-sized buffers are not allowed
    let _result = bufferize_to_store(&bufferize, &mut ctx);
}
