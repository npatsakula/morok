use std::{collections::HashMap, rc::Rc};

use morok_dtype::DType;
use morok_ir::{AxisType, ConstValue, Op, UOp};
use smallvec::smallvec;

use crate::{
    pattern::RewriteResult,
    rangeify::{
        KernelContext,
        split_patterns::{cleanup_const, debuf, handle_after, remove_zero_range, renumber_range, unbind_kernel},
    },
};

#[test]
fn test_debuf_global() {
    let mut ctx = KernelContext::new();

    // Create a BUFFER operation directly
    let unique = UOp::unique(Some(0));
    let device = UOp::device(morok_device::DeviceSpec::Cpu);
    let buffer = UOp::new(Op::Buffer { unique, device, size: 100 }, DType::Float32);

    // Create bindings
    let mut bindings = HashMap::new();
    bindings.insert("buf".to_string(), buffer.clone());

    // Apply debuf pattern
    let result = debuf(&bindings, &mut ctx);

    // Should return a DEFINE_GLOBAL
    match result {
        RewriteResult::Rewritten(op) => {
            assert!(matches!(op.op(), Op::DefineGlobal(_)));
            assert_eq!(ctx.global_counter, 1);
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_unbind_kernel() {
    let mut ctx = KernelContext::new();

    // Create a BIND operation
    let var = UOp::new(Op::DefineVar { name: "x".to_string(), min_val: 0, max_val: 10 }, DType::Index);
    let value = UOp::const_(DType::Index, ConstValue::Int(5));
    let bind = UOp::bind(var.clone(), value);

    // Create bindings
    let mut bindings = HashMap::new();
    bindings.insert("b".to_string(), bind);

    // Apply unbind_kernel pattern
    let result = unbind_kernel(&bindings, &mut ctx);

    // Should return just the variable
    match result {
        RewriteResult::Rewritten(op) => {
            assert!(matches!(op.op(), Op::DefineVar { .. }));
            assert!(ctx.has_var(&var));
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_renumber_range() {
    let mut ctx = KernelContext::new();

    // Create a RANGE with axis_id=5
    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range(end, 5, AxisType::Loop);

    // Create bindings
    let mut bindings = HashMap::new();
    bindings.insert("r".to_string(), range);

    // Apply renumber_range pattern
    let result = renumber_range(&bindings, &mut ctx);

    // Should return a RANGE with axis_id=0
    match result {
        RewriteResult::Rewritten(op) => {
            if let Op::Range { axis_id, .. } = op.op() {
                assert_eq!(*axis_id, 0);
            } else {
                panic!("Expected RANGE operation");
            }
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_remove_zero_range() {
    let mut ctx = KernelContext::new();

    // Create a RANGE with end=0
    let end = UOp::const_(DType::Index, ConstValue::Int(0));
    let range = UOp::range(end, 0, AxisType::Loop);

    // Create bindings
    let mut bindings = HashMap::new();
    bindings.insert("r".to_string(), range);

    // Apply remove_zero_range pattern
    let result = remove_zero_range(&bindings, &mut ctx);

    // Should return CONST(0)
    match result {
        RewriteResult::Rewritten(op) => {
            assert!(matches!(op.op(), Op::Const(_)));
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_cleanup_const_with_sources() {
    let mut ctx = KernelContext::new();

    // Create a CONST operation (normally has no sources)
    let const_op = UOp::const_(DType::Int32, ConstValue::Int(42));

    // Create bindings
    let mut bindings = HashMap::new();
    bindings.insert("c".to_string(), const_op.clone());

    // Apply cleanup_const pattern (should not match since const has no sources normally)
    let result = cleanup_const(&bindings, &mut ctx);

    // Should return NoMatch since the const has no sources
    assert!(matches!(result, RewriteResult::NoMatch));
}

#[test]
fn test_handle_after() {
    let mut ctx = KernelContext::new();

    // Create an AFTER operation
    let buffer = UOp::unique(Some(0));
    let store = UOp::noop();
    let after = UOp::new(Op::After { passthrough: buffer.clone(), deps: smallvec![store] }, buffer.dtype());

    // Create bindings
    let mut bindings = HashMap::new();
    bindings.insert("after".to_string(), after.clone());

    // Apply handle_after pattern
    let result = handle_after(&bindings, &mut ctx);

    // Should return the buffer
    match result {
        RewriteResult::Rewritten(op) => {
            assert!(matches!(op.op(), Op::Unique(_)));
            // Check that the buffer was mapped to the after operation
            assert!(ctx.has_buffer(&buffer));
            // Use Rc::ptr_eq for comparison
            assert!(Rc::ptr_eq(&ctx.get_buffer(&buffer).unwrap(), &after));
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_debuf_counter_increment() {
    let mut ctx = KernelContext::new();

    // Create first buffer
    let unique1 = UOp::unique(Some(1));
    let device1 = UOp::device(morok_device::DeviceSpec::Cpu);
    let buffer1 = UOp::new(Op::Buffer { unique: unique1, device: device1, size: 100 }, DType::Float32);

    // Create second buffer
    let unique2 = UOp::unique(Some(2));
    let device2 = UOp::device(morok_device::DeviceSpec::Cpu);
    let buffer2 = UOp::new(Op::Buffer { unique: unique2, device: device2, size: 200 }, DType::Float32);

    // Apply debuf to first buffer
    let mut bindings1 = HashMap::new();
    bindings1.insert("buf".to_string(), buffer1.clone());
    let result1 = debuf(&bindings1, &mut ctx);

    assert!(matches!(result1, RewriteResult::Rewritten(_)));
    assert_eq!(ctx.global_counter, 1);

    // Apply debuf to second buffer
    let mut bindings2 = HashMap::new();
    bindings2.insert("buf".to_string(), buffer2.clone());
    let result2 = debuf(&bindings2, &mut ctx);

    assert!(matches!(result2, RewriteResult::Rewritten(_)));
    assert_eq!(ctx.global_counter, 2);

    // Verify both buffers are mapped
    assert!(ctx.has_buffer(&buffer1));
    assert!(ctx.has_buffer(&buffer2));
}

#[test]
fn test_debuf_buffer_mapping() {
    let mut ctx = KernelContext::new();

    let unique = UOp::unique(Some(0));
    let device = UOp::device(morok_device::DeviceSpec::Cpu);
    let buffer = UOp::new(Op::Buffer { unique, device, size: 100 }, DType::Float32);

    let mut bindings = HashMap::new();
    bindings.insert("buf".to_string(), buffer.clone());

    debuf(&bindings, &mut ctx);

    // Buffer should be mapped to itself
    assert!(ctx.has_buffer(&buffer));
    let mapped = ctx.get_buffer(&buffer).unwrap();
    assert!(Rc::ptr_eq(&mapped, &buffer));
}

#[test]
fn test_handle_after_mstack_unwrap() {
    let mut ctx = KernelContext::new();

    // Create MSTACK with buffers
    let buf1 = UOp::unique(Some(1));
    let buf2 = UOp::unique(Some(2));
    let mstack = UOp::new(Op::MStack { buffers: smallvec![buf1.clone(), buf2] }, buf1.dtype());

    // Create AFTER wrapping MSTACK
    let store = UOp::noop();
    let after = UOp::new(Op::After { passthrough: mstack, deps: smallvec![store] }, buf1.dtype());

    let mut bindings = HashMap::new();
    bindings.insert("after".to_string(), after.clone());

    let result = handle_after(&bindings, &mut ctx);

    // Should unwrap to first buffer of MSTACK
    match result {
        RewriteResult::Rewritten(op) => {
            assert!(matches!(op.op(), Op::Unique(_)));
            assert!(Rc::ptr_eq(&op, &buf1));
            // buf1 should be mapped to after
            assert!(Rc::ptr_eq(&ctx.get_buffer(&buf1).unwrap(), &after));
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_handle_after_mselect_unwrap() {
    let mut ctx = KernelContext::new();

    // Create MSELECT
    let buffer = UOp::unique(Some(1));
    let mselect = UOp::new(Op::MSelect { buffer: buffer.clone(), device_index: 0 }, buffer.dtype());

    // Create AFTER wrapping MSELECT
    let store = UOp::noop();
    let after = UOp::new(Op::After { passthrough: mselect, deps: smallvec![store] }, buffer.dtype());

    let mut bindings = HashMap::new();
    bindings.insert("after".to_string(), after.clone());

    let result = handle_after(&bindings, &mut ctx);

    // Should unwrap to buffer
    match result {
        RewriteResult::Rewritten(op) => {
            assert!(Rc::ptr_eq(&op, &buffer));
            // buffer should be mapped to after
            assert!(Rc::ptr_eq(&ctx.get_buffer(&buffer).unwrap(), &after));
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_renumber_range_sequential() {
    let mut ctx = KernelContext::new();

    // Create three ranges with different axis IDs
    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range1 = UOp::range(end.clone(), 5, AxisType::Loop);
    let range2 = UOp::range(end.clone(), 10, AxisType::Reduce);
    let range3 = UOp::range(end.clone(), 3, AxisType::Outer);

    // Renumber all three
    let mut bindings1 = HashMap::new();
    bindings1.insert("r".to_string(), range1);
    let result1 = renumber_range(&bindings1, &mut ctx);

    let mut bindings2 = HashMap::new();
    bindings2.insert("r".to_string(), range2);
    let result2 = renumber_range(&bindings2, &mut ctx);

    let mut bindings3 = HashMap::new();
    bindings3.insert("r".to_string(), range3);
    let result3 = renumber_range(&bindings3, &mut ctx);

    // Should get sequential IDs 0, 1, 2
    if let RewriteResult::Rewritten(r) = result1 {
        if let Op::Range { axis_id, .. } = r.op() {
            assert_eq!(*axis_id, 0);
        }
    }

    if let RewriteResult::Rewritten(r) = result2 {
        if let Op::Range { axis_id, .. } = r.op() {
            assert_eq!(*axis_id, 1);
        }
    }

    if let RewriteResult::Rewritten(r) = result3 {
        if let Op::Range { axis_id, .. } = r.op() {
            assert_eq!(*axis_id, 2);
        }
    }
}

#[test]
fn test_renumber_range_different_axis_types() {
    let mut ctx = KernelContext::new();
    let end = UOp::const_(DType::Index, ConstValue::Int(10));

    // Test all three axis types
    for axis_type in [AxisType::Loop, AxisType::Reduce, AxisType::Outer] {
        let range = UOp::range(end.clone(), 99, axis_type);
        let mut bindings = HashMap::new();
        bindings.insert("r".to_string(), range);

        let result = renumber_range(&bindings, &mut ctx);

        if let RewriteResult::Rewritten(r) = result {
            if let Op::Range { axis_type: new_type, .. } = r.op() {
                // Axis type should be preserved
                assert_eq!(*new_type, axis_type);
            }
        } else {
            panic!("Expected Rewritten result for {:?}", axis_type);
        }
    }
}

#[test]
fn test_renumber_range_no_change_if_same() {
    let mut ctx = KernelContext::new();

    // First range will get ID 0
    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range1 = UOp::range(end.clone(), 5, AxisType::Loop);

    let mut bindings1 = HashMap::new();
    bindings1.insert("r".to_string(), range1);
    renumber_range(&bindings1, &mut ctx);

    // Now create a range that already has axis_id=1 (which would be the next ID)
    let range2 = UOp::range(end.clone(), 1, AxisType::Loop);
    let mut bindings2 = HashMap::new();
    bindings2.insert("r".to_string(), range2);

    let result = renumber_range(&bindings2, &mut ctx);

    // Should return NoMatch since the ID matches what we would assign
    assert!(matches!(result, RewriteResult::NoMatch));
}

#[test]
#[ignore = "Incomplete: only tests negative case, missing spurious sources test case"]
fn test_cleanup_const_define_var() {
    let mut ctx = KernelContext::new();

    // Create a DEFINE_VAR
    let define_var = UOp::new(Op::DefineVar { name: "x".to_string(), min_val: 0, max_val: 10 }, DType::Index);

    let mut bindings = HashMap::new();
    bindings.insert("c".to_string(), define_var.clone());

    // Without sources, should not match
    let result = cleanup_const(&bindings, &mut ctx);
    assert!(matches!(result, RewriteResult::NoMatch));

    // TODO: Test with spurious sources once we have a way to create them
}

#[test]
fn test_remove_zero_range_uint() {
    let mut ctx = KernelContext::new();

    // Create a RANGE with end=0 (UInt)
    let end = UOp::const_(DType::Index, ConstValue::UInt(0));
    let range = UOp::range(end, 0, AxisType::Loop);

    let mut bindings = HashMap::new();
    bindings.insert("r".to_string(), range);

    let result = remove_zero_range(&bindings, &mut ctx);

    // Should return CONST(0)
    match result {
        RewriteResult::Rewritten(op) => {
            assert!(matches!(op.op(), Op::Const(_)));
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_remove_zero_range_non_zero() {
    let mut ctx = KernelContext::new();

    // Create a RANGE with non-zero end
    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range(end, 0, AxisType::Loop);

    let mut bindings = HashMap::new();
    bindings.insert("r".to_string(), range);

    let result = remove_zero_range(&bindings, &mut ctx);

    // Should return NoMatch
    assert!(matches!(result, RewriteResult::NoMatch));
}
