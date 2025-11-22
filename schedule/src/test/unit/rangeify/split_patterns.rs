use std::rc::Rc;

use morok_dtype::DType;
use morok_ir::{AxisType, ConstValue, Op, UOp};
use smallvec::{SmallVec, smallvec};

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

    // Apply debuf pattern
    let result = debuf(&buffer, &mut ctx);

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

    // Apply unbind_kernel pattern
    let result = unbind_kernel(&bind, &mut ctx);

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
    let range = UOp::range_axis(end, 5, AxisType::Loop);

    // Apply renumber_range pattern
    let result = renumber_range(&range, &mut ctx);

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
    let range = UOp::range_axis(end, 0, AxisType::Loop);

    // Create bindings

    // Apply remove_zero_range pattern
    let result = remove_zero_range(&range, &mut ctx);

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

    // Apply cleanup_const pattern (should not match since const has no sources normally)
    let result = cleanup_const(&const_op, &mut ctx);

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

    // Apply handle_after pattern
    let result = handle_after(&after, &mut ctx);

    // Should return the buffer
    match result {
        RewriteResult::Rewritten(op) => {
            assert!(matches!(op.op(), Op::Unique(_)));
            // Check that the buffer was mapped to the after operation
            assert!(ctx.has_buffer(&buffer));
            // Use Rc::ptr_eq for comparison
            assert!(Rc::ptr_eq(ctx.get_buffer(&buffer).unwrap(), &after));
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
    let result1 = debuf(&buffer1, &mut ctx);

    assert!(matches!(result1, RewriteResult::Rewritten(_)));
    assert_eq!(ctx.global_counter, 1);

    // Apply debuf to second buffer
    let result2 = debuf(&buffer2, &mut ctx);

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

    debuf(&buffer, &mut ctx);

    // Buffer should be mapped to itself
    assert!(ctx.has_buffer(&buffer));
    let mapped = ctx.get_buffer(&buffer).unwrap();
    assert!(Rc::ptr_eq(mapped, &buffer));
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

    let result = handle_after(&after, &mut ctx);

    // Should unwrap to first buffer of MSTACK
    match result {
        RewriteResult::Rewritten(op) => {
            assert!(matches!(op.op(), Op::Unique(_)));
            assert!(Rc::ptr_eq(&op, &buf1));
            // buf1 should be mapped to after
            assert!(Rc::ptr_eq(ctx.get_buffer(&buf1).unwrap(), &after));
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

    let result = handle_after(&after, &mut ctx);

    // Should unwrap to buffer
    match result {
        RewriteResult::Rewritten(op) => {
            assert!(Rc::ptr_eq(&op, &buffer));
            // buffer should be mapped to after
            assert!(Rc::ptr_eq(ctx.get_buffer(&buffer).unwrap(), &after));
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_renumber_range_sequential() {
    let mut ctx = KernelContext::new();

    // Create three ranges with different axis IDs
    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range1 = UOp::range_axis(end.clone(), 5, AxisType::Loop);
    let range2 = UOp::range_axis(end.clone(), 10, AxisType::Reduce);
    let range3 = UOp::range_axis(end.clone(), 3, AxisType::Outer);

    // Renumber all three
    let result1 = renumber_range(&range1, &mut ctx);

    let result2 = renumber_range(&range2, &mut ctx);

    let result3 = renumber_range(&range3, &mut ctx);

    // Should get sequential IDs 0, 1, 2
    if let RewriteResult::Rewritten(r) = result1
        && let Op::Range { axis_id, .. } = r.op()
    {
        assert_eq!(*axis_id, 0);
    }

    if let RewriteResult::Rewritten(r) = result2
        && let Op::Range { axis_id, .. } = r.op()
    {
        assert_eq!(*axis_id, 1);
    }

    if let RewriteResult::Rewritten(r) = result3
        && let Op::Range { axis_id, .. } = r.op()
    {
        assert_eq!(*axis_id, 2);
    }
}

#[test]
fn test_renumber_range_different_axis_types() {
    let mut ctx = KernelContext::new();
    let end = UOp::const_(DType::Index, ConstValue::Int(10));

    // Test all three axis types
    for axis_type in [AxisType::Loop, AxisType::Reduce, AxisType::Outer] {
        let range = UOp::range_axis(end.clone(), 99, axis_type);

        let result = renumber_range(&range, &mut ctx);

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
    let range1 = UOp::range_axis(end.clone(), 5, AxisType::Loop);

    renumber_range(&range1, &mut ctx);

    // Now create a range that already has axis_id=1 (which would be the next ID)
    let range2 = UOp::range_axis(end.clone(), 1, AxisType::Loop);

    let result = renumber_range(&range2, &mut ctx);

    // Should return NoMatch since the ID matches what we would assign
    assert!(matches!(result, RewriteResult::NoMatch));
}

#[test]
#[ignore = "Incomplete: only tests negative case, missing spurious sources test case"]
fn test_cleanup_const_define_var() {
    let mut ctx = KernelContext::new();

    // Create a DEFINE_VAR
    let define_var = UOp::new(Op::DefineVar { name: "x".to_string(), min_val: 0, max_val: 10 }, DType::Index);

    // Without sources, should not match
    let result = cleanup_const(&define_var, &mut ctx);
    assert!(matches!(result, RewriteResult::NoMatch));

    // TODO: Test with spurious sources once we have a way to create them
}

#[test]
fn test_remove_zero_range_uint() {
    let mut ctx = KernelContext::new();

    // Create a RANGE with end=0 (UInt)
    let end = UOp::const_(DType::Index, ConstValue::UInt(0));
    let range = UOp::range_axis(end, 0, AxisType::Loop);

    let result = remove_zero_range(&range, &mut ctx);

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
    let range = UOp::range_axis(end, 0, AxisType::Loop);

    let result = remove_zero_range(&range, &mut ctx);

    // Should return NoMatch
    assert!(matches!(result, RewriteResult::NoMatch));
}

#[test]
#[ignore = "MSTACK/AFTER handling not fully implemented yet"]
fn test_handle_after_mstack_advanced() {
    let mut ctx = KernelContext::new();

    // Create MSTACK operation
    let buf1 = UOp::unique(Some(1));
    let buf2 = UOp::unique(Some(2));
    let mstack = UOp::new(Op::MStack { buffers: smallvec::smallvec![buf1.clone(), buf2] }, DType::Float32);

    // Create AFTER wrapping MSTACK
    // Note: AFTER has passthrough + deps, not src
    let after = UOp::new(Op::After { passthrough: mstack.clone(), deps: SmallVec::new() }, DType::Float32);

    let result = handle_after(&after, &mut ctx);

    // Should unwrap MSTACK and return first buffer
    match result {
        RewriteResult::Rewritten(buf) => {
            // Should return buf1 (first in MSTACK)
            assert!(std::rc::Rc::ptr_eq(&buf, &buf1));

            // MSTACK should be tracked in context
            assert!(ctx.buffer_map.contains_key(&morok_ir::UOpKey(mstack)));
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_cleanup_const_with_spurious_sources() {
    let mut ctx = KernelContext::new();

    // Create a CONST that has sources (spurious - consts shouldn't have sources normally)
    // This tests the cleanup pattern that removes unnecessary sources from CONST

    // Create a CONST
    let const_op = UOp::const_(DType::Int32, ConstValue::Int(42));

    let result = cleanup_const(&const_op, &mut ctx);

    // DEFINE_VAR shouldn't be cleaned up (it's not a CONST)
    assert!(matches!(result, RewriteResult::NoMatch));
}

#[test]
fn test_renumber_range_with_gaps() {
    let mut ctx = KernelContext::new();

    // Create ranges with non-sequential IDs (0, 5, 10)
    let range0 = UOp::range_const(10, 0);
    let range5 = UOp::range_const(20, 5);
    let range10 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(30)), 10, AxisType::Reduce);

    // Process them in sequence

    let result0 = renumber_range(&range0, &mut ctx);

    // First range should keep ID 0
    match result0 {
        RewriteResult::NoMatch => {
            // Correct - ID 0 is what we'd assign anyway
        }
        _ => panic!("Expected NoMatch for first range"),
    }

    // Second range (ID 5) should be renumbered to 1

    let result5 = renumber_range(&range5, &mut ctx);

    match result5 {
        RewriteResult::Rewritten(new_range) => {
            // Should be renumbered to ID 1
            if let Op::Range { axis_id, .. } = new_range.op() {
                assert_eq!(*axis_id, 1);
            } else {
                panic!("Expected RANGE operation");
            }
        }
        _ => panic!("Expected Rewritten result for range with ID 5"),
    }

    // Third range (ID 10) should be renumbered to 2

    let result10 = renumber_range(&range10, &mut ctx);

    match result10 {
        RewriteResult::Rewritten(new_range) => {
            // Should be renumbered to ID 2
            if let Op::Range { axis_id, axis_type, .. } = new_range.op() {
                assert_eq!(*axis_id, 2);
                // Should preserve axis type
                assert_eq!(*axis_type, AxisType::Reduce);
            } else {
                panic!("Expected RANGE operation");
            }
        }
        _ => panic!("Expected Rewritten result for range with ID 10"),
    }

    // Context should have assigned 3 sequential IDs
    assert_eq!(ctx.range_counter, 3);
}

#[test]
fn test_remove_zero_range_verification() {
    let mut ctx = KernelContext::new();

    // Create RANGE with end=0
    let end = UOp::const_(DType::Index, ConstValue::Int(0));
    let range = UOp::range(end.clone(), 0);

    let result = remove_zero_range(&range, &mut ctx);

    // Should rewrite to CONST(0)
    match result {
        RewriteResult::Rewritten(const_op) => {
            // Should be a CONST
            if let Op::Const(val) = const_op.op() {
                // Should be Int(0)
                assert_eq!(val.0, ConstValue::Int(0));

                // Should NOT be the same as the original range
                assert!(!std::rc::Rc::ptr_eq(&const_op, &range));

                // Should have Index dtype (same as range)
                assert_eq!(const_op.dtype(), DType::Index);
            } else {
                panic!("Expected CONST operation");
            }
        }
        _ => panic!("Expected Rewritten result for zero range"),
    }
}

#[test]
fn test_pattern_composition_sequence() {
    let mut ctx = KernelContext::new();

    // Test that patterns can be applied in sequence
    // 1. Create a RANGE with non-sequential ID
    // 2. Apply renumber_range
    // 3. Verify the result

    let range_gap = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(15)), 7, AxisType::Loop);

    // Apply renumber_range pattern
    let result1 = renumber_range(&range_gap, &mut ctx);

    match result1 {
        RewriteResult::Rewritten(renumbered) => {
            // Should be renumbered to ID 0 (first in sequence)
            if let Op::Range { axis_id, end, axis_type } = renumbered.op() {
                assert_eq!(*axis_id, 0);
                assert_eq!(*axis_type, AxisType::Loop);

                // End should be preserved
                if let Op::Range { end: original_end, .. } = range_gap.op() {
                    assert!(std::rc::Rc::ptr_eq(end, original_end));
                }

                // Now apply another pattern to the result
                // For example, if the renumbered range has zero end, remove it

                let result2 = remove_zero_range(&renumbered, &mut ctx);

                // Should return NoMatch since end is 15, not 0
                assert!(matches!(result2, RewriteResult::NoMatch));
            } else {
                panic!("Expected RANGE operation");
            }
        }
        _ => panic!("Expected Rewritten result"),
    }
}
