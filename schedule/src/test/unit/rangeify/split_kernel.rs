use morok_dtype::DType;
use morok_ir::{AxisType, ConstValue, Op, UOp};
use smallvec::smallvec;

use crate::rangeify::{split_kernel::{split_store}, KernelContext};

#[test]
fn test_split_store_basic() {
    let mut ctx = KernelContext::new();

    // Create a simple STORE operation
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = UOp::new(Op::Store { buffer: buffer.clone(), index, value }, DType::Void);

    // Try to split
    let result = split_store(&store, &mut ctx);

    // Should return a KERNEL
    assert!(result.is_some());
    let kernel = result.unwrap();
    assert!(matches!(kernel.op(), Op::Kernel { .. }));
}

#[test]
fn test_split_store_non_store_returns_none() {
    let mut ctx = KernelContext::new();

    // Create a non-STORE operation
    let const_op = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Try to split
    let result = split_store(&const_op, &mut ctx);

    // Should return None
    assert!(result.is_none());
}

#[test]
fn test_split_store_end_operation() {
    let mut ctx = KernelContext::new();

    // Create an END operation
    let store = UOp::noop();
    let range = UOp::range(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Loop);
    let end = UOp::end(store, smallvec![range]);

    // Try to split
    let result = split_store(&end, &mut ctx);

    // Should process END operations
    assert!(result.is_some());
    let kernel = result.unwrap();
    assert!(matches!(kernel.op(), Op::Kernel { .. }));
}

#[test]
fn test_split_store_gated() {
    let mut ctx = KernelContext::new();

    // Create a STORE_GATED operation
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let store_gated = UOp::new(Op::StoreGated { buffer, index, value, gate }, DType::Void);

    // Try to split
    let result = split_store(&store_gated, &mut ctx);

    // Should return a KERNEL
    assert!(result.is_some());
    let kernel = result.unwrap();
    assert!(matches!(kernel.op(), Op::Kernel { .. }));
}

#[test]
fn test_split_store_creates_sink() {
    let mut ctx = KernelContext::new();

    // Create a STORE operation
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = UOp::new(Op::Store { buffer, index, value }, DType::Void);

    let result = split_store(&store, &mut ctx).unwrap();

    // Extract the AST from the KERNEL
    if let Op::Kernel { ast, .. } = result.op() {
        // The AST should be a SINK operation
        assert!(matches!(ast.op(), Op::Sink { .. }));

        // The SINK should wrap the original STORE
        if let Op::Sink { sources } = ast.op() {
            assert_eq!(sources.len(), 1);
            // The source should be the original STORE
            assert!(matches!(sources[0].op(), Op::Store { .. }));
        }
    } else {
        panic!("Expected KERNEL operation");
    }
}


#[test]
fn test_split_store_preserves_dtype() {
    let mut ctx = KernelContext::new();

    // Create STOREs with different value dtypes
    let test_cases = [
        (DType::Float32, ConstValue::Float(1.0)),
        (DType::Int32, ConstValue::Int(1)),
        (DType::Bool, ConstValue::Bool(true)),
    ];

    for (dtype, const_val) in test_cases {
        let buffer = UOp::unique(Some(0));
        let index = UOp::const_(DType::Index, ConstValue::Int(0));
        let value = UOp::const_(dtype, const_val);
        let store = UOp::new(Op::Store { buffer, index, value }, DType::Void);

        let result = split_store(&store, &mut ctx);

        // Should successfully create kernel regardless of value dtype
        assert!(result.is_some());
    }
}

#[test]
fn test_split_store_multiple_calls_independent() {
    let mut ctx = KernelContext::new();

    // Create two different STORE operations
    let buffer1 = UOp::unique(Some(1));
    let index1 = UOp::const_(DType::Index, ConstValue::Int(0));
    let value1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store1 = UOp::new(Op::Store { buffer: buffer1, index: index1, value: value1 }, DType::Void);

    let buffer2 = UOp::unique(Some(2));
    let index2 = UOp::const_(DType::Index, ConstValue::Int(0));
    let value2 = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let store2 = UOp::new(Op::Store { buffer: buffer2, index: index2, value: value2 }, DType::Void);

    // Split both
    let kernel1 = split_store(&store1, &mut ctx).unwrap();
    let kernel2 = split_store(&store2, &mut ctx).unwrap();

    // Both should be valid kernels
    assert!(matches!(kernel1.op(), Op::Kernel { .. }));
    assert!(matches!(kernel2.op(), Op::Kernel { .. }));

    // They should be different kernels (different UOps)
    assert!(!std::rc::Rc::ptr_eq(&kernel1, &kernel2));
}

#[test]
fn test_split_store_end_with_multiple_ranges() {
    let mut ctx = KernelContext::new();

    // Create END with multiple ranges
    let store = UOp::noop();
    let range1 = UOp::range(UOp::const_(DType::Index, ConstValue::Int(4)), 0, AxisType::Loop);
    let range2 = UOp::range(UOp::const_(DType::Index, ConstValue::Int(8)), 1, AxisType::Loop);
    let end = UOp::end(store, smallvec![range1, range2]);

    let result = split_store(&end, &mut ctx);

    // Should create a kernel
    assert!(result.is_some());
    let kernel = result.unwrap();
    assert!(matches!(kernel.op(), Op::Kernel { .. }));
}

#[test]
#[ignore = "Incomplete: proper AxisType::Outer filtering not implemented in split_store"]
fn test_split_store_end_with_outer_range() {
    let mut ctx = KernelContext::new();

    // Create END with OUTER range
    let store = UOp::noop();
    let range_outer = UOp::range(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Outer);
    let end = UOp::end(store, smallvec![range_outer]);

    let result = split_store(&end, &mut ctx);

    // Should still create a kernel (the simplified implementation doesn't skip OUTER ENDs yet)
    // TODO: Update this test when proper filtering is implemented
    assert!(result.is_some());
}

#[test]
#[ignore = "Incomplete: proper AxisType::Outer filtering not implemented in split_store"]
fn test_split_store_end_with_mixed_ranges() {
    let mut ctx = KernelContext::new();

    // Create END with mix of LOOP and OUTER ranges
    let store = UOp::noop();
    let range_loop = UOp::range(UOp::const_(DType::Index, ConstValue::Int(4)), 0, AxisType::Loop);
    let range_outer = UOp::range(UOp::const_(DType::Index, ConstValue::Int(8)), 1, AxisType::Outer);
    let end = UOp::end(store, smallvec![range_loop, range_outer]);

    let result = split_store(&end, &mut ctx);

    // Should create a kernel
    assert!(result.is_some());
}
