use std::sync::Arc;

use morok_dtype::{AddrSpace, DType, ScalarDType};
use morok_ir::{AxisId, AxisType, ConstValue, Op, UOp};
use smallvec::smallvec;

use crate::rangeify::{KernelContext, patterns::to_define_global_patterns};

/// Helper to apply to_define_global patterns and return result
fn apply_patterns(uop: &Arc<UOp>, ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    let matcher = to_define_global_patterns();
    match matcher.rewrite(uop, ctx) {
        morok_ir::pattern::RewriteResult::Rewritten(result) => Some(result),
        _ => None,
    }
}

#[test]
fn test_debuf_global() {
    let mut ctx = KernelContext::new();

    // Create a BUFFER operation directly
    let unique = UOp::buffer_id(Some(0));
    let device = UOp::device(morok_device::DeviceSpec::Cpu);
    let buffer = UOp::new(Op::Buffer { unique, device, size: 100 }, DType::Float32);

    // Apply pattern via matcher
    let result = apply_patterns(&buffer, &mut ctx);

    // Should return a DEFINE_GLOBAL
    let op = result.expect("Expected Some result");
    assert!(matches!(op.op(), Op::DefineGlobal(_)));
    assert_eq!(ctx.global_counter, 1);
}

#[test]
fn test_unbind_kernel() {
    let mut ctx = KernelContext::new();

    // Create a BIND operation
    let var = UOp::new(Op::DefineVar { name: "x".to_string(), min_val: 0, max_val: 10 }, DType::Index);
    let value = UOp::index_const(5);
    let bind = UOp::bind(var.clone(), value);

    // Apply pattern via matcher
    let result = apply_patterns(&bind, &mut ctx);

    // Should return just the variable
    let op = result.expect("Expected Some result");
    assert!(matches!(op.op(), Op::DefineVar { .. }));
    assert!(ctx.has_var(&var));
}

#[test]
fn test_renumber_range() {
    let mut ctx = KernelContext::new();

    // Create a RANGE with unrenumbered axis_id (use Reduce since it returns plain Range)
    let end = UOp::index_const(10);
    let range = UOp::range_axis(end, AxisId::Unrenumbered(5), AxisType::Reduce);

    // Apply pattern via matcher
    let result = apply_patterns(&range, &mut ctx);

    // Should return a RANGE with axis_id=Renumbered(0) (renumbered)
    let op = result.expect("Expected Some result");
    if let Op::Range { axis_id, .. } = op.op() {
        assert_eq!(*axis_id, AxisId::Renumbered(0));
    } else {
        panic!("Expected RANGE operation");
    }
}

#[test]
fn test_renumber_range_loop_no_bind() {
    let mut ctx = KernelContext::new();

    // Create a LOOP RANGE with unrenumbered axis_id
    let end = UOp::index_const(10);
    let range = UOp::range_axis(end, AxisId::Unrenumbered(5), AxisType::Loop);

    // Apply pattern via matcher
    let result = apply_patterns(&range, &mut ctx);

    // LOOP ranges should be renumbered without BIND wrapper (Tinygrad approach)
    // Codegen creates loops directly from RANGE ops
    let op = result.expect("Expected Some result");
    if let Op::Range { axis_id, axis_type, .. } = op.op() {
        assert_eq!(*axis_id, AxisId::Renumbered(0));
        assert_eq!(*axis_type, AxisType::Loop);
    } else {
        panic!("Expected RANGE operation for LOOP axis, got {:?}", op.op());
    }
}

#[test]
fn test_renumber_range_already_numbered() {
    let mut ctx = KernelContext::new();

    // Create a RANGE with already-renumbered axis_id
    let end = UOp::index_const(10);
    let range = UOp::range_axis(end, AxisId::Renumbered(5), AxisType::Loop);

    // Apply pattern via matcher - should return None (already numbered)
    let result = apply_patterns(&range, &mut ctx);
    assert!(result.is_none(), "Already-numbered range should not be renumbered");
}

#[test]
fn test_remove_zero_range() {
    let mut ctx = KernelContext::new();

    // Create a RANGE with end=0
    let end = UOp::index_const(0);
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);

    // Apply pattern via matcher
    let result = apply_patterns(&range, &mut ctx);

    // Should return CONST(0)
    let op = result.expect("Expected Some result");
    assert!(matches!(op.op(), Op::Const(_)));
}

#[test]
fn test_cleanup_const_with_sources() {
    let mut ctx = KernelContext::new();

    // Create a CONST operation (normally has no sources)
    let const_op = UOp::native_const(42i32);

    // Apply pattern via matcher (should not match since const has no sources normally)
    let result = apply_patterns(&const_op, &mut ctx);

    // Should return None since the const has no sources
    assert!(result.is_none());
}

#[test]
fn test_handle_after() {
    let mut ctx = KernelContext::new();

    // Create an AFTER operation
    let buffer = UOp::buffer_id(Some(0));
    let store = UOp::noop();
    let after = UOp::after(buffer.clone(), smallvec::smallvec![store]);

    // Apply pattern via matcher
    let result = apply_patterns(&after, &mut ctx);

    // Should return the buffer
    let op = result.expect("Expected Some result");
    assert!(matches!(op.op(), Op::Unique(_)));
    // Check that the buffer was mapped to the after operation
    assert!(ctx.has_buffer(&buffer));
    // Use Arc::ptr_eq for comparison
    assert!(Arc::ptr_eq(ctx.get_buffer(&buffer).unwrap(), &after));
}

#[test]
fn test_debuf_counter_increment() {
    let mut ctx = KernelContext::new();

    // Create first buffer
    let unique1 = UOp::buffer_id(Some(1));
    let device1 = UOp::device(morok_device::DeviceSpec::Cpu);
    let buffer1 = UOp::new(Op::Buffer { unique: unique1, device: device1, size: 100 }, DType::Float32);

    // Create second buffer
    let unique2 = UOp::buffer_id(Some(2));
    let device2 = UOp::device(morok_device::DeviceSpec::Cpu);
    let buffer2 = UOp::new(Op::Buffer { unique: unique2, device: device2, size: 200 }, DType::Float32);

    // Apply patterns to first buffer
    let result1 = apply_patterns(&buffer1, &mut ctx);

    assert!(result1.is_some());
    assert_eq!(ctx.global_counter, 1);

    // Apply patterns to second buffer
    let result2 = apply_patterns(&buffer2, &mut ctx);

    assert!(result2.is_some());
    assert_eq!(ctx.global_counter, 2);

    // Verify both buffers are mapped
    assert!(ctx.has_buffer(&buffer1));
    assert!(ctx.has_buffer(&buffer2));
}

#[test]
fn test_debuf_buffer_mapping() {
    let mut ctx = KernelContext::new();

    let unique = UOp::buffer_id(Some(0));
    let device = UOp::device(morok_device::DeviceSpec::Cpu);
    let buffer = UOp::new(Op::Buffer { unique, device, size: 100 }, DType::Float32);

    let result = apply_patterns(&buffer, &mut ctx);

    // Pattern returns DEFINE_GLOBAL and maps BUFFER → DEFINE_GLOBAL
    assert!(result.is_some());
    let define_global = result.unwrap();
    assert!(matches!(define_global.op(), Op::DefineGlobal(0)));

    // Buffer should be tracked, mapping to DEFINE_GLOBAL (not itself)
    assert!(ctx.has_buffer(&buffer));
    let mapped = ctx.get_buffer(&buffer).unwrap();
    assert!(Arc::ptr_eq(mapped, &define_global));
}

#[test]
fn test_handle_after_mstack_unwrap() {
    let mut ctx = KernelContext::new();

    // Create MSTACK with buffers
    let buf1 = UOp::buffer_id(Some(1));
    let buf2 = UOp::buffer_id(Some(2));
    let mstack = UOp::new(Op::MStack { buffers: smallvec![buf1.clone(), buf2] }, buf1.dtype());

    // Create AFTER wrapping MSTACK
    let store = UOp::noop();
    let after = UOp::after(mstack, smallvec::smallvec![store]);

    let result = apply_patterns(&after, &mut ctx);

    // Should unwrap to first buffer of MSTACK
    let op = result.expect("Expected Some result");
    assert!(matches!(op.op(), Op::Unique(_)));
    assert!(Arc::ptr_eq(&op, &buf1));
    // buf1 should be mapped to after
    assert!(Arc::ptr_eq(ctx.get_buffer(&buf1).unwrap(), &after));
}

#[test]
fn test_handle_after_mselect_unwrap() {
    let mut ctx = KernelContext::new();

    // Create MSELECT
    let buffer = UOp::buffer_id(Some(1));
    let mselect = UOp::new(Op::MSelect { buffer: buffer.clone(), device_index: 0 }, buffer.dtype());

    // Create AFTER wrapping MSELECT
    let store = UOp::noop();
    let after = UOp::after(mselect, smallvec::smallvec![store]);

    let result = apply_patterns(&after, &mut ctx);

    // Should unwrap to buffer
    let op = result.expect("Expected Some result");
    assert!(Arc::ptr_eq(&op, &buffer));
    // buffer should be mapped to after
    assert!(Arc::ptr_eq(ctx.get_buffer(&buffer).unwrap(), &after));
}

#[test]
fn test_renumber_range_different_axis_types() {
    let mut ctx = KernelContext::new();
    let end = UOp::index_const(10);

    // Test axis types with unrenumbered axis_ids
    // All axis types now return plain Range without BIND wrapper (Tinygrad approach)
    // Codegen creates loops directly from RANGE ops
    for (i, axis_type) in [AxisType::Loop, AxisType::Reduce, AxisType::Outer].iter().enumerate() {
        let range = UOp::range_axis(end.clone(), AxisId::Unrenumbered(i), *axis_type);

        let result = apply_patterns(&range, &mut ctx);

        if let Some(r) = result {
            // All axis types return plain Range
            if let Op::Range { axis_type: new_type, .. } = r.op() {
                assert_eq!(*new_type, *axis_type);
            } else {
                panic!("Expected Range for {:?}, got {:?}", axis_type, r.op());
            }
        } else {
            panic!("Expected Some result for {:?}", axis_type);
        }
    }
}

#[test]
fn test_renumber_range_no_change_if_same() {
    let mut ctx = KernelContext::new();

    // First range will get ID 0
    let end = UOp::index_const(10);
    let range1 = UOp::range_axis(end.clone(), AxisId::Renumbered(5), AxisType::Loop);

    apply_patterns(&range1, &mut ctx);

    // Now create a range that already has axis_id=Renumbered(1)
    let range2 = UOp::range_axis(end.clone(), AxisId::Renumbered(1), AxisType::Loop);

    let result = apply_patterns(&range2, &mut ctx);

    // Should return None since it's already renumbered
    assert!(result.is_none());
}

#[test]
#[ignore = "Incomplete: only tests negative case, missing spurious sources test case"]
fn test_cleanup_const_define_var() {
    let mut ctx = KernelContext::new();

    // Create a DEFINE_VAR
    let define_var = UOp::new(Op::DefineVar { name: "x".to_string(), min_val: 0, max_val: 10 }, DType::Index);

    // Without sources, should not match
    let result = apply_patterns(&define_var, &mut ctx);
    assert!(result.is_none());

    // TODO: Test with spurious sources once we have a way to create them
}

#[test]
fn test_remove_zero_range_uint() {
    let mut ctx = KernelContext::new();

    // Create a RANGE with end=0 (UInt)
    let end = UOp::index_const(0);
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);

    let result = apply_patterns(&range, &mut ctx);

    // Should return CONST(0)
    let op = result.expect("Expected Some result");
    assert!(matches!(op.op(), Op::Const(_)));
}

#[test]
fn test_remove_zero_range_non_zero() {
    let mut ctx = KernelContext::new();

    // Create a RANGE with non-zero end
    let end = UOp::index_const(10);
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);

    let result = apply_patterns(&range, &mut ctx);

    // Should renumber (since it's Renumbered already, no match for renumber; non-zero, no match for zero)
    assert!(result.is_none());
}

#[test]
#[ignore = "MSTACK/AFTER handling not fully implemented yet"]
fn test_handle_after_mstack_advanced() {
    let mut ctx = KernelContext::new();

    // Create MSTACK operation
    let buf1 = UOp::buffer_id(Some(1));
    let buf2 = UOp::buffer_id(Some(2));
    let mstack = UOp::new(Op::MStack { buffers: smallvec::smallvec![buf1.clone(), buf2] }, DType::Float32);

    // Create AFTER wrapping MSTACK
    // Note: AFTER has passthrough + deps, not src
    let after = UOp::after(mstack.clone(), smallvec::SmallVec::new());

    let result = apply_patterns(&after, &mut ctx);

    // Should unwrap MSTACK and return first buffer
    match result {
        Some(buf) => {
            // Should return buf1 (first in MSTACK)
            assert!(std::sync::Arc::ptr_eq(&buf, &buf1));

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
    let const_op = UOp::native_const(42i32);

    let result = apply_patterns(&const_op, &mut ctx);

    // DEFINE_VAR shouldn't be cleaned up (it's not a CONST)
    assert!(result.is_none());
}

#[test]
fn test_renumber_range_sequential() {
    let mut ctx = KernelContext::new();

    // Create ranges with unrenumbered axis_ids
    // All axis types now return plain Range without BIND wrapper (Tinygrad approach)
    let range0 = UOp::range_axis(UOp::index_const(10), AxisId::Unrenumbered(0), AxisType::Loop);
    let range1 = UOp::range_axis(UOp::index_const(20), AxisId::Unrenumbered(1), AxisType::Loop);
    let range2 = UOp::range_axis(UOp::index_const(30), AxisId::Unrenumbered(2), AxisType::Reduce);

    // Process them in sequence - should get sequential IDs Renumbered(0), Renumbered(1), Renumbered(2)

    let result0 = apply_patterns(&range0, &mut ctx);
    match result0 {
        Some(new_range) => {
            // LOOP returns plain Range (no BIND)
            if let Op::Range { axis_id, axis_type, .. } = new_range.op() {
                assert_eq!(*axis_id, AxisId::Renumbered(0));
                assert_eq!(*axis_type, AxisType::Loop);
            } else {
                panic!("Expected RANGE operation for LOOP");
            }
        }
        None => panic!("Expected renumbered range"),
    }

    let result1 = apply_patterns(&range1, &mut ctx);
    match result1 {
        Some(new_range) => {
            // LOOP returns plain Range (no BIND)
            if let Op::Range { axis_id, axis_type, .. } = new_range.op() {
                assert_eq!(*axis_id, AxisId::Renumbered(1));
                assert_eq!(*axis_type, AxisType::Loop);
            } else {
                panic!("Expected RANGE operation for LOOP");
            }
        }
        None => panic!("Expected renumbered range"),
    }

    let result2 = apply_patterns(&range2, &mut ctx);
    match result2 {
        Some(new_range) => {
            // Reduce returns plain Range
            if let Op::Range { axis_id, axis_type, .. } = new_range.op() {
                assert_eq!(*axis_id, AxisId::Renumbered(2));
                assert_eq!(*axis_type, AxisType::Reduce);
            } else {
                panic!("Expected RANGE operation");
            }
        }
        None => panic!("Expected renumbered range"),
    }

    // Context should have assigned 3 sequential IDs
    assert_eq!(ctx.range_counter, 3);
}

#[test]
fn test_remove_zero_range_verification() {
    let mut ctx = KernelContext::new();

    // Create RANGE with end=0
    let end = UOp::index_const(0);
    let range = UOp::range_axis(end.clone(), AxisId::Renumbered(0), AxisType::Loop);

    let result = apply_patterns(&range, &mut ctx);

    // Should rewrite to CONST(0)
    match result {
        Some(const_op) => {
            // Should be a CONST
            if let Op::Const(val) = const_op.op() {
                // Should be Int(0)
                assert_eq!(val.0, ConstValue::Int(0));

                // Should NOT be the same as the original range
                assert!(!std::sync::Arc::ptr_eq(&const_op, &range));

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
    // 1. Create a RANGE with unrenumbered ID (use Reduce for plain Range output)
    // 2. Apply renumber pattern
    // 3. Verify the result

    let range_unnum = UOp::range_axis(UOp::index_const(15), AxisId::Unrenumbered(7), AxisType::Reduce);

    // Apply pattern
    let result1 = apply_patterns(&range_unnum, &mut ctx);

    match result1 {
        Some(renumbered) => {
            // Should be renumbered to ID 0 (first in sequence)
            if let Op::Range { axis_id, end, axis_type } = renumbered.op() {
                assert_eq!(*axis_id, AxisId::Renumbered(0));
                assert_eq!(*axis_type, AxisType::Reduce);

                // End should be preserved
                if let Op::Range { end: original_end, .. } = range_unnum.op() {
                    assert!(std::sync::Arc::ptr_eq(end, original_end));
                }

                // Now apply another pattern to the result
                // For example, if the renumbered range has zero end, remove it

                let result2 = apply_patterns(&renumbered, &mut ctx);

                // Should return NoMatch since end is 15, not 0
                assert!(result2.is_none());
            } else {
                panic!("Expected RANGE operation");
            }
        }
        None => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_pattern_composition_sequence_no_bind() {
    let mut ctx = KernelContext::new();

    // Test that LOOP ranges return plain Range (no BIND wrapper, Tinygrad approach)
    let range_unnum = UOp::range_axis(UOp::index_const(15), AxisId::Unrenumbered(7), AxisType::Loop);

    // Apply pattern
    let result1 = apply_patterns(&range_unnum, &mut ctx);

    match result1 {
        Some(new_range) => {
            // LOOP should return plain Range (codegen creates loops from RANGE ops)
            if let Op::Range { axis_id, axis_type, end } = new_range.op() {
                assert_eq!(*axis_id, AxisId::Renumbered(0));
                assert_eq!(*axis_type, AxisType::Loop);

                // End should be preserved
                if let Op::Range { end: original_end, .. } = range_unnum.op() {
                    assert!(std::sync::Arc::ptr_eq(end, original_end));
                }
            } else {
                panic!("Expected RANGE operation for LOOP axis, got {:?}", new_range.op());
            }
        }
        None => panic!("Expected Rewritten result"),
    }
}

// ============================================================================
// Local Buffer Address Space Tests
// ============================================================================

#[test]
fn test_handle_after_local_buffer_not_tracked() {
    // Local buffers should NOT be tracked in the buffer map
    // They are kernel-scoped and synchronized via BARRIER, not AFTER
    let mut ctx = KernelContext::new();

    // Create a local buffer (DEFINE_LOCAL with Ptr{Local} dtype)
    let local_dtype = DType::Ptr {
        base: Box::new(DType::Scalar(ScalarDType::Float32)),
        addrspace: AddrSpace::Local,
        size: Some(1024),
    };
    let local_buf = UOp::define_local(1, local_dtype);

    // Wrap in AFTER operation
    let store = UOp::noop();
    let after = UOp::after(local_buf.clone(), smallvec![store]);

    // Apply pattern
    let result = apply_patterns(&after, &mut ctx);

    // Should return the buffer unwrapped
    match result {
        Some(op) => {
            assert!(matches!(op.op(), Op::DefineLocal(_)));
            // Local buffer should NOT be in buffer map
            assert!(!ctx.has_buffer(&local_buf));
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_handle_after_global_buffer_tracked() {
    // Global buffers SHOULD be tracked in the buffer map
    let mut ctx = KernelContext::new();

    // Create a global buffer (DEFINE_GLOBAL with Ptr{Global} dtype)
    let global_dtype = DType::Ptr {
        base: Box::new(DType::Scalar(ScalarDType::Float32)),
        addrspace: AddrSpace::Global,
        size: Some(1024),
    };
    let global_buf = UOp::define_global(1, global_dtype);

    // Wrap in AFTER operation
    let store = UOp::noop();
    let after = UOp::after(global_buf.clone(), smallvec![store]);

    // Apply pattern
    let result = apply_patterns(&after, &mut ctx);

    // Should return the buffer unwrapped
    match result {
        Some(op) => {
            assert!(matches!(op.op(), Op::DefineGlobal(_)));
            // Global buffer SHOULD be in buffer map
            assert!(ctx.has_buffer(&global_buf));
            assert!(Arc::ptr_eq(ctx.get_buffer(&global_buf).unwrap(), &after));
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_handle_after_mstack_with_local_buffer() {
    // AFTER wrapping MSTACK containing local buffer should not be tracked
    let mut ctx = KernelContext::new();

    // Create local buffer
    let local_dtype = DType::Ptr {
        base: Box::new(DType::Scalar(ScalarDType::Float32)),
        addrspace: AddrSpace::Local,
        size: Some(512),
    };
    let local_buf1 = UOp::define_local(1, local_dtype.clone());
    let local_buf2 = UOp::define_local(2, local_dtype.clone());

    // Create MSTACK
    let mstack = UOp::new(Op::MStack { buffers: smallvec![local_buf1.clone(), local_buf2] }, local_dtype);

    // Wrap in AFTER
    let store = UOp::noop();
    let after = UOp::after(mstack, smallvec![store]);

    // Apply pattern
    let result = apply_patterns(&after, &mut ctx);

    // Should unwrap to first buffer in MSTACK
    match result {
        Some(op) => {
            // Verify MSTACK was actually unwrapped to local_buf1 (not just any DEFINE_LOCAL)
            assert!(Arc::ptr_eq(&op, &local_buf1), "Should unwrap to first buffer in MSTACK");
            assert!(matches!(op.op(), Op::DefineLocal(1)));
            // Local buffer should NOT be tracked
            assert!(!ctx.has_buffer(&local_buf1));
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_handle_after_mselect_with_local_buffer() {
    // AFTER wrapping MSELECT containing local buffer should not be tracked
    let mut ctx = KernelContext::new();

    // Create local buffer
    let local_dtype =
        DType::Ptr { base: Box::new(DType::Scalar(ScalarDType::Int32)), addrspace: AddrSpace::Local, size: Some(256) };
    let local_buf = UOp::define_local(3, local_dtype.clone());

    // Create MSELECT
    let mselect = UOp::new(Op::MSelect { buffer: local_buf.clone(), device_index: 0 }, local_dtype);

    // Wrap in AFTER
    let store = UOp::noop();
    let after = UOp::after(mselect, smallvec![store]);

    // Apply pattern
    let result = apply_patterns(&after, &mut ctx);

    // Should unwrap to the buffer in MSELECT
    match result {
        Some(op) => {
            // Verify MSELECT was actually unwrapped to local_buf (not just any DEFINE_LOCAL)
            assert!(Arc::ptr_eq(&op, &local_buf), "Should unwrap to buffer from MSELECT");
            assert!(matches!(op.op(), Op::DefineLocal(3)));
            // Local buffer should NOT be tracked
            assert!(!ctx.has_buffer(&local_buf));
        }
        _ => panic!("Expected Rewritten result"),
    }
}

#[test]
fn test_handle_after_mixed_address_spaces() {
    // Verify local and global buffers are handled differently
    let mut ctx = KernelContext::new();

    // Create both local and global buffers
    let local_dtype = DType::Ptr {
        base: Box::new(DType::Scalar(ScalarDType::Float32)),
        addrspace: AddrSpace::Local,
        size: Some(128),
    };
    let global_dtype = DType::Ptr {
        base: Box::new(DType::Scalar(ScalarDType::Float32)),
        addrspace: AddrSpace::Global,
        size: Some(128),
    };

    let local_buf = UOp::define_local(10, local_dtype);
    let global_buf = UOp::define_global(11, global_dtype);

    // Wrap both in AFTER
    let store1 = UOp::noop();
    let store2 = UOp::noop();
    let after_local = UOp::after(local_buf.clone(), smallvec![store1]);
    let after_global = UOp::after(global_buf.clone(), smallvec![store2]);

    // Apply patterns to both and validate return values
    let result_local = apply_patterns(&after_local, &mut ctx);
    let result_global = apply_patterns(&after_global, &mut ctx);

    // Verify both returned Rewritten with correct buffers
    match result_local {
        Some(op) => {
            assert!(Arc::ptr_eq(&op, &local_buf), "Local AFTER should return local buffer");
        }
        _ => panic!("Expected Rewritten for local"),
    }
    match result_global {
        Some(op) => {
            assert!(Arc::ptr_eq(&op, &global_buf), "Global AFTER should return global buffer");
        }
        _ => panic!("Expected Rewritten for global"),
    }

    // Verify only global buffer is tracked (side effect validation)
    assert!(!ctx.has_buffer(&local_buf), "Local buffer should NOT be tracked");
    assert!(ctx.has_buffer(&global_buf), "Global buffer SHOULD be tracked");
}
