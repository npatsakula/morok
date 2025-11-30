//! Unit tests for the Scheduler (kernel optimization state manager).

use std::rc::Rc;

use morok_ir::{AxisId, AxisType, ConstValue, Op, ReduceOp, UOp};

use crate::optimizer::error::OptError;
use crate::optimizer::{OptOps, Renderer, Scheduler};

#[test]
fn test_scheduler_new() {
    let ast = UOp::native_const(1.0f32);
    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(ast, ren);

    assert_eq!(scheduler.applied_opts.len(), 0);
    assert!(!scheduler.dont_use_locals);
    assert_eq!(scheduler.shape_len(), 0); // No ranges
}

#[test]
fn test_scheduler_rngs_sorting() {
    // Create ranges with different types and IDs using range_axis
    let end_16 = UOp::index_const(16);
    let end_8 = UOp::index_const(8);
    let end_32 = UOp::index_const(32);
    let end_4 = UOp::index_const(4);

    let r_global = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::Global);
    let r_local = UOp::range_axis(end_8, AxisId::Renumbered(1), AxisType::Local);
    let r_reduce = UOp::range_axis(end_32, AxisId::Renumbered(2), AxisType::Reduce);
    let r_loop = UOp::range_axis(end_4, AxisId::Renumbered(3), AxisType::Loop);

    // Build a simple computation using all ranges
    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global, r_local, r_reduce, r_loop]);

    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(sink, ren);

    let rngs = scheduler.rngs();
    assert_eq!(rngs.len(), 4);

    // Verify sort order: Loop(-1) < Global(0) < Local(2) < Reduce(4)
    if let Op::Range { axis_type, .. } = rngs[0].op() {
        assert_eq!(*axis_type, AxisType::Loop);
    }
    if let Op::Range { axis_type, .. } = rngs[1].op() {
        assert_eq!(*axis_type, AxisType::Global);
    }
    if let Op::Range { axis_type, .. } = rngs[2].op() {
        assert_eq!(*axis_type, AxisType::Local);
    }
    if let Op::Range { axis_type, .. } = rngs[3].op() {
        assert_eq!(*axis_type, AxisType::Reduce);
    }
}

#[test]
fn test_scheduler_maxarg() {
    // Using range_const for convenience
    let r1 = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(5), AxisType::Loop);
    let r2 = UOp::range_axis(UOp::index_const(20), AxisId::Renumbered(2), AxisType::Global);
    let r3 = UOp::range_axis(UOp::index_const(30), AxisId::Renumbered(10), AxisType::Reduce);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r1, r2, r3]);

    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(sink, ren);

    assert_eq!(scheduler.maxarg(), 10); // Highest axis_id is 10
}

#[test]
fn test_scheduler_helper_properties() {
    // Create a reduction kernel: sum over axis
    let end_8 = UOp::index_const(8);
    let end_16 = UOp::index_const(16);
    let end_32 = UOp::index_const(32);

    let r_global = UOp::range_axis(end_16.clone(), AxisId::Renumbered(0), AxisType::Global);
    let r_local = UOp::range_axis(end_8.clone(), AxisId::Renumbered(1), AxisType::Local);
    let r_reduce = UOp::range_axis(end_32.clone(), AxisId::Renumbered(2), AxisType::Reduce);

    // Create a simple reduction: value to reduce
    let value = UOp::native_const(1.0f32);
    let reduce_op = UOp::reduce(value.clone(), vec![r_reduce.clone()].into(), ReduceOp::Add);

    // Wrap in sink with all ranges
    let sink = UOp::sink(vec![reduce_op, r_global, r_local, r_reduce]);

    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(sink, ren);

    // Test reduceop/reduceops
    assert!(scheduler.reduceop().is_some());
    assert_eq!(scheduler.reduceops().len(), 1);

    // Test output_shape (should exclude REDUCE axes)
    let output = scheduler.output_shape();
    assert_eq!(output.len(), 2); // Global(16) + Local(8), no Reduce
    assert_eq!(output[0], 16); // Global comes first (priority 0 < Local priority 2)
    assert_eq!(output[1], 8);

    // Test upcast_size (no UPCAST axes, should be 1)
    assert_eq!(scheduler.upcast_size(), 1);

    // Test group_for_reduces (no GROUP_REDUCE axes)
    assert_eq!(scheduler.group_for_reduces(), 0);

    // Test bufs (no INDEX ops in this simple kernel)
    assert_eq!(scheduler.bufs().len(), 0);
}

#[test]
fn test_scheduler_upcast_size() {
    // Create kernel with UPCAST axes
    let end_4 = UOp::index_const(4);
    let end_8 = UOp::index_const(8);

    let r_upcast1 = UOp::range_axis(end_4, AxisId::Renumbered(0), AxisType::Upcast);
    let r_upcast2 = UOp::range_axis(end_8, AxisId::Renumbered(1), AxisType::Upcast);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_upcast1, r_upcast2]);

    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(sink, ren);

    // upcast_size should be 4 * 8 = 32
    assert_eq!(scheduler.upcast_size(), 32);
}

#[test]
fn test_scheduler_group_for_reduces() {
    // Create kernel with GROUP_REDUCE axes
    let end_16 = UOp::index_const(16);
    let end_32 = UOp::index_const(32);

    let r_group = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::GroupReduce);
    let r_reduce = UOp::range_axis(end_32, AxisId::Renumbered(1), AxisType::Reduce);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_group, r_reduce]);

    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(sink, ren);

    // Should have 1 GROUP_REDUCE axis
    assert_eq!(scheduler.group_for_reduces(), 1);
}

#[test]
fn test_scheduler_axes_of() {
    // Create kernel with mixed axis types
    let end_16 = UOp::index_const(16);
    let end_8 = UOp::index_const(8);
    let end_32 = UOp::index_const(32);

    let r_global = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::Global);
    let r_local = UOp::range_axis(end_8, AxisId::Renumbered(1), AxisType::Local);
    let r_reduce = UOp::range_axis(end_32, AxisId::Renumbered(2), AxisType::Reduce);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global, r_local, r_reduce]);

    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(sink, ren);

    // Test axes_of for GLOBAL
    let global_axes = scheduler.axes_of(&[AxisType::Global]);
    assert_eq!(global_axes, vec![0]); // Global is at position 0

    // Test axes_of for reduction types
    let reduce_axes = scheduler.axes_of(&[AxisType::Reduce]);
    assert_eq!(reduce_axes, vec![2]); // Reduce is at position 2

    // Test axes_of for multiple types
    let parallel_axes = scheduler.axes_of(&[AxisType::Global, AxisType::Local]);
    assert_eq!(parallel_axes, vec![0, 1]); // Global + Local

    // Test ranges_of
    let reduce_rngs = scheduler.ranges_of(&[AxisType::Reduce]);
    assert_eq!(reduce_rngs.len(), 1);
    if let Op::Range { axis_type, .. } = reduce_rngs[0].op() {
        assert_eq!(*axis_type, AxisType::Reduce);
    }
}

#[test]
fn test_scheduler_upcastable_dims() {
    // Create kernel with various axis types
    let end_1 = UOp::index_const(1);
    let end_16 = UOp::index_const(16);
    let end_32 = UOp::index_const(32);

    let r_global = UOp::range_axis(end_16.clone(), AxisId::Renumbered(0), AxisType::Global);
    let r_loop = UOp::range_axis(end_32, AxisId::Renumbered(1), AxisType::Loop);
    let r_reduce = UOp::range_axis(end_16.clone(), AxisId::Renumbered(2), AxisType::Reduce);
    let r_size1 = UOp::range_axis(end_1, AxisId::Renumbered(3), AxisType::Global); // Size 1, not upcastable

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global, r_loop, r_reduce, r_size1]);

    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(sink, ren);

    // upcastable_dims: GLOBAL(16), LOOP(32) - excludes REDUCE and size-1 GLOBAL
    let upcastable = scheduler.upcastable_dims();
    assert_eq!(upcastable.len(), 2);
    assert!(upcastable.contains(&0)); // Global(16)
    assert!(upcastable.contains(&1)); // Loop(32)
}

#[test]
fn test_scheduler_unrollable_dims() {
    // Create kernel with reduction axes
    let end_16 = UOp::index_const(16);
    let end_32 = UOp::index_const(32);
    let end_1 = UOp::index_const(1);

    let r_global = UOp::range_axis(end_16.clone(), AxisId::Renumbered(0), AxisType::Global);
    let r_reduce1 = UOp::range_axis(end_32, AxisId::Renumbered(1), AxisType::Reduce);
    let r_reduce2 = UOp::range_axis(end_16, AxisId::Renumbered(2), AxisType::Reduce);
    let r_reduce_size1 = UOp::range_axis(end_1, AxisId::Renumbered(3), AxisType::Reduce); // Size 1, not unrollable

    // Create reduction
    let value = UOp::native_const(1.0f32);
    let reduce_op = UOp::reduce(value, vec![r_reduce1.clone(), r_reduce2.clone()].into(), ReduceOp::Add);

    let sink = UOp::sink(vec![reduce_op, r_global, r_reduce1, r_reduce2, r_reduce_size1]);

    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(sink, ren);

    // unrollable_dims: REDUCE(32), REDUCE(16) - excludes size-1 REDUCE
    let unrollable = scheduler.unrollable_dims();
    assert_eq!(unrollable.len(), 2);
    assert!(unrollable.contains(&1)); // Reduce(32)
    assert!(unrollable.contains(&2)); // Reduce(16)
}

#[test]
fn test_scheduler_real_axis() {
    // Create complex kernel
    let end_16 = UOp::index_const(16);
    let end_32 = UOp::index_const(32);

    let r_global = UOp::range_axis(end_16.clone(), AxisId::Renumbered(0), AxisType::Global);
    let r_loop = UOp::range_axis(end_16.clone(), AxisId::Renumbered(1), AxisType::Loop);
    let r_reduce1 = UOp::range_axis(end_32.clone(), AxisId::Renumbered(2), AxisType::Reduce);
    let r_reduce2 = UOp::range_axis(end_16, AxisId::Renumbered(3), AxisType::Reduce);

    let value = UOp::native_const(1.0f32);
    let reduce_op = UOp::reduce(value, vec![r_reduce1.clone(), r_reduce2.clone()].into(), ReduceOp::Add);

    let sink = UOp::sink(vec![reduce_op, r_global, r_loop, r_reduce1, r_reduce2]);

    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(sink, ren);

    // Test direct axis mapping (UPCAST, LOCAL, etc.)
    assert_eq!(scheduler.real_axis(OptOps::UPCAST, Some(1)).unwrap(), 1);
    assert_eq!(scheduler.real_axis(OptOps::LOCAL, Some(0)).unwrap(), 0);

    // Test UNROLL (logical index into unrollable_dims)
    // unrollable_dims = [2, 3] (both REDUCE axes)
    assert_eq!(scheduler.real_axis(OptOps::UNROLL, Some(0)).unwrap(), 2); // First unrollable
    assert_eq!(scheduler.real_axis(OptOps::UNROLL, Some(1)).unwrap(), 3); // Second unrollable

    // Test GROUP (logical index into REDUCE axes)
    assert_eq!(scheduler.real_axis(OptOps::GROUP, Some(0)).unwrap(), 2);
    assert_eq!(scheduler.real_axis(OptOps::GROUP, Some(1)).unwrap(), 3);

    // Test TC (no axis)
    assert_eq!(scheduler.real_axis(OptOps::TC, None).unwrap(), -1);

    // Test NOLOCALS (no axis)
    assert_eq!(scheduler.real_axis(OptOps::NOLOCALS, None).unwrap(), -1);

    // Test out of bounds
    assert!(scheduler.real_axis(OptOps::UPCAST, Some(10)).is_err());
    assert!(scheduler.real_axis(OptOps::UNROLL, Some(5)).is_err());
}

#[test]
fn test_scheduler_colored_shape() {
    // Create a typical reduction kernel: g16l8R32u4
    let end_16 = UOp::index_const(16);
    let end_8 = UOp::index_const(8);
    let end_32 = UOp::index_const(32);
    let end_4 = UOp::index_const(4);

    let r_global = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::Global);
    let r_local = UOp::range_axis(end_8, AxisId::Renumbered(1), AxisType::Local);
    let r_reduce = UOp::range_axis(end_32, AxisId::Renumbered(2), AxisType::Reduce);
    let r_upcast = UOp::range_axis(end_4, AxisId::Renumbered(3), AxisType::Upcast);

    let value = UOp::native_const(1.0f32);
    let reduce_op = UOp::reduce(value, vec![r_reduce.clone()].into(), ReduceOp::Add);

    let sink = UOp::sink(vec![reduce_op, r_global, r_local, r_reduce, r_upcast]);

    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(sink, ren);

    // Test colored_shape
    // Sort order: Global(0) < Local(2) < Upcast(3) < Reduce(4)
    let shape = scheduler.colored_shape();
    assert_eq!(shape, "g16l8u4R32");

    // Test shape_str
    let shape_vec = scheduler.shape_str();
    assert_eq!(shape_vec, vec!["g16", "l8", "u4", "R32"]);

    // Test kernel_type
    assert_eq!(scheduler.kernel_type(), "r"); // Has reduction

    // Test Display
    let display_str = format!("{}", scheduler);
    assert_eq!(display_str, "r_g16l8u4R32");
}

#[test]
fn test_scheduler_display_elementwise() {
    // Create an elementwise kernel: E_g256g256
    let end_256 = UOp::index_const(256);

    let r_global1 = UOp::range_axis(end_256.clone(), AxisId::Renumbered(0), AxisType::Global);
    let r_global2 = UOp::range_axis(end_256, AxisId::Renumbered(1), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global1, r_global2]);

    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(sink, ren);

    // Test kernel_type
    assert_eq!(scheduler.kernel_type(), "E"); // No reduction

    // Test Display
    let display_str = format!("{}", scheduler);
    assert_eq!(display_str, "E_g256g256");
}

#[test]
fn test_scheduler_display_complex() {
    // Create complex kernel with multiple axis types
    let end_2 = UOp::index_const(2);
    let end_4 = UOp::index_const(4);
    let end_8 = UOp::index_const(8);
    let end_16 = UOp::index_const(16);
    let end_32 = UOp::index_const(32);

    let r_loop = UOp::range_axis(end_2, AxisId::Renumbered(0), AxisType::Loop);
    let r_global = UOp::range_axis(end_32.clone(), AxisId::Renumbered(1), AxisType::Global);
    let r_local = UOp::range_axis(end_16, AxisId::Renumbered(2), AxisType::Local);
    let r_reduce = UOp::range_axis(end_32, AxisId::Renumbered(3), AxisType::Reduce);
    let r_upcast = UOp::range_axis(end_4, AxisId::Renumbered(4), AxisType::Upcast);
    let r_unroll = UOp::range_axis(end_8, AxisId::Renumbered(5), AxisType::Unroll);

    let value = UOp::native_const(1.0f32);
    let reduce_op = UOp::reduce(value, vec![r_reduce.clone(), r_unroll.clone()].into(), ReduceOp::Add);

    let sink = UOp::sink(vec![reduce_op, r_loop, r_global, r_local, r_reduce, r_upcast, r_unroll]);

    let ren = Renderer::cpu();
    let scheduler = Scheduler::new(sink, ren);

    // Verify sort order: Loop < Global < Local < Upcast < Reduce < Unroll
    let shape = scheduler.colored_shape();
    assert_eq!(shape, "L2g32l16u4R32r8");

    let display_str = format!("{}", scheduler);
    assert_eq!(display_str, "r_L2g32l16u4R32r8");
}

// =========================================================================
// shift_to() - Core Transformation Tests
// =========================================================================

#[test]
fn test_shift_to_basic_split() {
    // Create a simple kernel with a single Global(16) range
    let end_16 = UOp::index_const(16);
    let r_global = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::Global);

    // Use the range in a simple computation
    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global.clone()]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Verify initial state
    assert_eq!(scheduler.shape_len(), 1);
    assert_eq!(scheduler.maxarg(), 0);

    // Split Global(16) into Global(4) and Upcast(4)
    let result = scheduler.shift_to(r_global.clone(), 4, AxisType::Upcast, false, None);
    assert!(result.is_ok());

    let (replaced_rng, new_rng) = result.unwrap();

    // Verify the reduced range has size 4 (16 / 4)
    if let Op::Range { end, axis_id, axis_type } = replaced_rng.op() {
        assert_eq!(axis_id, &AxisId::Renumbered(0)); // Same axis_id
        assert_eq!(axis_type, &AxisType::Global); // Same type
        if let Op::Const(cv) = end.op()
            && let ConstValue::Int(sz) = cv.0
        {
            assert_eq!(sz, 4);
        } else {
            panic!("Expected constant size");
        }
    } else {
        panic!("Expected Range operation");
    }

    // Verify the new range has size 4 and type Upcast
    if let Op::Range { end, axis_id, axis_type } = new_rng.op() {
        assert_eq!(axis_id, &AxisId::Renumbered(1)); // New axis_id = maxarg + 1
        assert_eq!(axis_type, &AxisType::Upcast);
        if let Op::Const(cv) = end.op()
            && let ConstValue::Int(sz) = cv.0
        {
            assert_eq!(sz, 4);
        } else {
            panic!("Expected constant size");
        }
    } else {
        panic!("Expected Range operation");
    }

    // Verify we now have 2 ranges in the AST
    assert_eq!(scheduler.shape_len(), 2);

    // Verify maxarg was incremented
    assert_eq!(scheduler.maxarg(), 1);
}

#[test]
fn test_shift_to_top_order() {
    // Create a kernel with Global(16)
    let end_16 = UOp::index_const(16);
    let r_global = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global.clone()]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Split with top=true (new range is outer loop)
    let result = scheduler.shift_to(r_global.clone(), 4, AxisType::Local, true, None);
    assert!(result.is_ok());

    let (replaced_rng, new_rng) = result.unwrap();

    // Verify both ranges exist
    if let Op::Range { end, .. } = replaced_rng.op()
        && let Op::Const(cv) = end.op()
        && let ConstValue::Int(sz) = cv.0
    {
        assert_eq!(sz, 4); // 16 / 4
    } else {
        panic!("Expected constant size");
    }

    if let Op::Range { end, axis_type, .. } = new_rng.op() {
        assert_eq!(axis_type, &AxisType::Local);
        if let Op::Const(cv) = end.op()
            && let ConstValue::Int(sz) = cv.0
        {
            assert_eq!(sz, 4);
        } else {
            panic!("Expected constant size");
        }
    } else {
        panic!("Expected Range operation");
    }

    // Verify shape updated
    assert_eq!(scheduler.shape_len(), 2);
}

#[test]
fn test_shift_to_division_error() {
    // Create a kernel with Global(15)
    let end_15 = UOp::index_const(15);
    let r_global = UOp::range_axis(end_15, AxisId::Renumbered(0), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global.clone()]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Try to split by 4 (15 is not divisible by 4)
    let result = scheduler.shift_to(r_global.clone(), 4, AxisType::Upcast, false, None);

    // Should return DivisionError
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::DivisionError { .. }));
    }
}

#[test]
fn test_shift_to_substitution_in_ast() {
    // Create a kernel that actually uses the range value in a computation
    let end_16 = UOp::index_const(16);
    let r_global = UOp::range_axis(end_16.clone(), AxisId::Renumbered(0), AxisType::Global);

    // Create a computation that uses the range: r_global * 2
    let two = UOp::index_const(2);
    let compute = r_global.try_mul(&two).unwrap();

    let sink = UOp::sink(vec![compute.clone(), r_global.clone()]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Split the range
    let result = scheduler.shift_to(r_global.clone(), 4, AxisType::Upcast, false, None);
    assert!(result.is_ok());

    // Verify the AST was updated (old range should be replaced)
    // The computation should now use (replaced_rng * 4 + new_rng) instead of r_global
    let new_rngs = scheduler.rngs();
    assert_eq!(new_rngs.len(), 2);

    // The substitution should have replaced the old range with the reduced range
    // Verify that the range with axis_id=0 now has size 4 (not the original size 16)
    let all_nodes = scheduler.ast().toposort();
    let ranges_with_axis0: Vec<_> = all_nodes
        .iter()
        .filter_map(|node| {
            if let Op::Range { end, axis_id, .. } = node.op()
                && *axis_id == AxisId::Renumbered(0)
                && let Op::Const(cv) = end.op()
                && let ConstValue::Int(sz) = cv.0
            {
                return Some(sz);
            }
            None
        })
        .collect();

    // Should have exactly one range with axis_id=0, and its size should be 4 (the reduced size)
    assert_eq!(ranges_with_axis0.len(), 1);
    assert_eq!(ranges_with_axis0[0], 4);
}

#[test]
fn test_shift_to_cache_invalidation() {
    // Create a kernel with a single range
    let end_16 = UOp::index_const(16);
    let r_global = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global.clone()]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Access cached properties to populate caches
    let _rngs_before = scheduler.rngs();
    let _maxarg_before = scheduler.maxarg();
    let _shape_len_before = scheduler.shape_len();

    // Perform shift_to
    let result = scheduler.shift_to(r_global.clone(), 4, AxisType::Upcast, false, None);
    assert!(result.is_ok());

    // Verify caches were invalidated and recomputed correctly
    let rngs_after = scheduler.rngs();
    assert_eq!(rngs_after.len(), 2); // Should now have 2 ranges

    let maxarg_after = scheduler.maxarg();
    assert_eq!(maxarg_after, 1); // Should be incremented

    let shape_len_after = scheduler.shape_len();
    assert_eq!(shape_len_after, 2); // Should match new range count
}

#[test]
fn test_shift_to_with_custom_range() {
    // Test providing a custom new_rng (input_new_rng parameter)
    let end_16 = UOp::index_const(16);
    let r_global = UOp::range_axis(end_16.clone(), AxisId::Renumbered(0), AxisType::Global);

    // Create a custom range with specific axis_id
    let end_4 = UOp::index_const(4);
    let custom_rng = UOp::range_axis(end_4, AxisId::Renumbered(99), AxisType::Upcast);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global.clone()]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Use the custom range
    let result = scheduler.shift_to(r_global.clone(), 4, AxisType::Upcast, false, Some(custom_rng.clone()));
    assert!(result.is_ok());

    let (_replaced_rng, new_rng) = result.unwrap();

    // Verify the returned range is our custom one
    if let Op::Range { axis_id, .. } = new_rng.op() {
        assert_eq!(axis_id, &AxisId::Renumbered(99)); // Should use our custom axis_id
    } else {
        panic!("Expected Range operation");
    }
}

#[test]
fn test_shift_to_multiple_splits() {
    // Test multiple consecutive splits
    let end_64 = UOp::index_const(64);
    let r_global = UOp::range_axis(end_64, AxisId::Renumbered(0), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global.clone()]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // First split: Global(64) -> Global(16) + Upcast(4)
    let result1 = scheduler.shift_to(r_global.clone(), 4, AxisType::Upcast, false, None);
    assert!(result1.is_ok());
    let (global_16, _upcast_4) = result1.unwrap();

    assert_eq!(scheduler.shape_len(), 2);
    assert_eq!(scheduler.maxarg(), 1);

    // Second split: Global(16) -> Global(8) + Local(2)
    let result2 = scheduler.shift_to(global_16, 2, AxisType::Local, false, None);
    assert!(result2.is_ok());

    assert_eq!(scheduler.shape_len(), 3);
    assert_eq!(scheduler.maxarg(), 2);

    // Verify the final shape has 3 ranges
    let final_rngs = scheduler.rngs();
    assert_eq!(final_rngs.len(), 3);
}

// ============================================================================
// OptOps Tests (Phase 4)
// ============================================================================

use crate::optimizer::{Opt, OptArg, apply_opt};

/// Helper to extract axis_type from a Range UOp
fn get_axis_type(uop: &UOp) -> AxisType {
    if let Op::Range { axis_type, .. } = uop.op() {
        *axis_type
    } else {
        panic!("Expected Range operation");
    }
}

#[test]
fn test_upcast_basic() {
    // Create a kernel with Global(16)
    let end_16 = UOp::index_const(16);
    let r_global = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Apply UPCAST optimization
    let opt = Opt::upcast(0, 4);
    let result = apply_opt(&mut scheduler, &opt, true);
    assert!(result.is_ok());

    // Verify the optimization was recorded
    assert_eq!(scheduler.applied_opts.len(), 1);
    assert_eq!(scheduler.applied_opts[0], opt);

    // Verify the shape changed: Global(16) -> Global(4) + Upcast(4)
    assert_eq!(scheduler.shape_len(), 2);

    let rngs = scheduler.rngs();
    assert_eq!(get_axis_type(&rngs[0]), AxisType::Global);
    assert_eq!(get_axis_type(&rngs[1]), AxisType::Upcast);
}

#[test]
fn test_upcast_invalid_axis_type() {
    // Create a kernel with Reduce axis (cannot upcast reduce axes)
    let end_32 = UOp::index_const(32);
    let r_reduce = UOp::range_axis(end_32, AxisId::Renumbered(0), AxisType::Reduce);

    let compute = UOp::native_const(1.0f32);
    let reduce = UOp::reduce(compute, vec![r_reduce].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Try to upcast a Reduce axis (should fail)
    let opt = Opt::upcast(0, 4);
    let result = apply_opt(&mut scheduler, &opt, false);

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::ValidationFailed { op: "UPCAST", .. }));
    }
}

#[test]
fn test_upcast_device_limit() {
    // Create a kernel with Global(256)
    let end_256 = UOp::index_const(256);
    let r_global = UOp::range_axis(end_256, AxisId::Renumbered(0), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Try to upcast by more than device limit (cpu upcast_max is 16)
    let opt = Opt::upcast(0, 32);
    let result = apply_opt(&mut scheduler, &opt, false);

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::DeviceLimitExceeded { limit_type: "upcast", .. }));
    }
}

#[test]
fn test_local_basic() {
    // Create a kernel with Global(64) and a backend that supports local memory
    let end_64 = UOp::index_const(64);
    let r_global = UOp::range_axis(end_64, AxisId::Renumbered(0), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global]);

    let ren = Renderer::cuda(); // CUDA has local memory support
    let mut scheduler = Scheduler::new(sink, ren);

    // Apply LOCAL optimization
    let opt = Opt::local(0, 8);
    let result = apply_opt(&mut scheduler, &opt, true);
    assert!(result.is_ok());

    // Verify the optimization was recorded
    assert_eq!(scheduler.applied_opts.len(), 1);

    // Verify the shape changed: Global(64) -> Global(8) + Local(8)
    assert_eq!(scheduler.shape_len(), 2);

    let rngs = scheduler.rngs();
    assert_eq!(get_axis_type(&rngs[0]), AxisType::Global);
    assert_eq!(get_axis_type(&rngs[1]), AxisType::Local);
}

#[test]
fn test_local_no_backend_support() {
    // Create a kernel with CPU backend (no local memory)
    let end_64 = UOp::index_const(64);
    let r_global = UOp::range_axis(end_64, AxisId::Renumbered(0), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Try to apply LOCAL (should fail - CPU doesn't have local memory)
    let opt = Opt::local(0, 8);
    let result = apply_opt(&mut scheduler, &opt, false);

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::UnsupportedFeature { feature: "local memory" }));
    }
}

#[test]
fn test_local_invalid_axis_type() {
    // Create a kernel with Reduce axis
    let end_32 = UOp::index_const(32);
    let r_reduce = UOp::range_axis(end_32, AxisId::Renumbered(0), AxisType::Reduce);

    let compute = UOp::native_const(1.0f32);
    let reduce = UOp::reduce(compute, vec![r_reduce].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Try to localize a Reduce axis (should fail)
    let opt = Opt::local(0, 4);
    let result = apply_opt(&mut scheduler, &opt, false);

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::ValidationFailed { op: "LOCAL", .. }));
    }
}

#[test]
fn test_unroll_basic() {
    // Create a kernel with a reduction
    let end_32 = UOp::index_const(32);
    let r_reduce = UOp::range_axis(end_32, AxisId::Renumbered(0), AxisType::Reduce);

    let compute = UOp::native_const(1.0f32);
    let reduce = UOp::reduce(compute, vec![r_reduce].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Verify we have unrollable dimensions
    let unrollable = scheduler.unrollable_dims();
    assert_eq!(unrollable.len(), 1);

    // Apply UNROLL optimization (logical axis 0 = first unrollable dimension)
    let opt = Opt::unroll(0, 4);
    let result = apply_opt(&mut scheduler, &opt, true);
    assert!(result.is_ok());

    // Verify the optimization was recorded
    assert_eq!(scheduler.applied_opts.len(), 1);

    // Verify the shape changed: Reduce(32) -> Reduce(8) + Unroll(4)
    assert_eq!(scheduler.shape_len(), 2);

    let rngs = scheduler.rngs();
    assert_eq!(get_axis_type(&rngs[0]), AxisType::Reduce);
    assert_eq!(get_axis_type(&rngs[1]), AxisType::Unroll);
}

#[test]
fn test_unroll_axis_out_of_bounds() {
    // Create a kernel with a single reduction
    let end_32 = UOp::index_const(32);
    let r_reduce = UOp::range_axis(end_32, AxisId::Renumbered(0), AxisType::Reduce);

    let compute = UOp::native_const(1.0f32);
    let reduce = UOp::reduce(compute, vec![r_reduce].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Try to unroll axis 1 (only axis 0 exists)
    let opt = Opt::unroll(1, 4);
    let result = apply_opt(&mut scheduler, &opt, false);

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::AxisOutOfBounds { .. }));
    }
}

#[test]
fn test_unroll_excessive_amount() {
    // Create a kernel with a large reduction
    let end_128 = UOp::index_const(128);
    let r_reduce = UOp::range_axis(end_128, AxisId::Renumbered(0), AxisType::Reduce);

    let compute = UOp::native_const(1.0f32);
    let reduce = UOp::reduce(compute, vec![r_reduce].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Try to unroll by 64 (exceeds reasonable limit of 32)
    let opt = Opt::unroll(0, 64);
    let result = apply_opt(&mut scheduler, &opt, false);

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::DeviceLimitExceeded { limit_type: "unroll", .. }));
    }
}

#[test]
fn test_apply_opt_multiple_operations() {
    // Create a complex kernel with Global and Reduce axes
    let end_64 = UOp::index_const(64);
    let r_global = UOp::range_axis(end_64.clone(), AxisId::Renumbered(0), AxisType::Global);

    let end_32 = UOp::index_const(32);
    let r_reduce = UOp::range_axis(end_32, AxisId::Renumbered(1), AxisType::Reduce);

    let compute = UOp::native_const(1.0f32);
    let reduce = UOp::reduce(compute, vec![r_global, r_reduce].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Apply multiple optimizations
    let opt1 = Opt::upcast(0, 4); // Upcast Global axis
    assert!(apply_opt(&mut scheduler, &opt1, true).is_ok());

    let opt2 = Opt::unroll(0, 8); // Unroll Reduce axis
    assert!(apply_opt(&mut scheduler, &opt2, true).is_ok());

    // Verify both optimizations were recorded
    assert_eq!(scheduler.applied_opts.len(), 2);
    assert_eq!(scheduler.applied_opts[0], opt1);
    assert_eq!(scheduler.applied_opts[1], opt2);

    // Verify the shape: Global + Upcast + Reduce + Unroll = 4 ranges
    assert_eq!(scheduler.shape_len(), 4);
}

#[test]
fn test_apply_opt_invalid_arg_type() {
    // Test that OptArg type validation works
    let end_16 = UOp::index_const(16);
    let r_global = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Create an Opt with wrong arg type (TensorCore for UPCAST)
    let opt = Opt::new(OptOps::UPCAST, Some(0), OptArg::TensorCore { tc_select: 0, opt_level: 0, use_tc: 0 });

    let result = apply_opt(&mut scheduler, &opt, false);

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::InvalidArgType { expected: "Int", .. }));
    }
}

// ============================================================================
// Phase 5: Advanced OptOps Tests
// ============================================================================

#[test]
fn test_nolocals_basic() {
    // Create a simple kernel without any local axes
    let end_16 = UOp::index_const(16);
    let r_global = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global]);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Apply NOLOCALS
    let opt = Opt::nolocals();
    let result = apply_opt(&mut scheduler, &opt, true);
    assert!(result.is_ok());

    // Verify the flag is set
    assert!(scheduler.dont_use_locals);

    // Verify LOCAL optimization now fails
    let opt_local = Opt::local(0, 4);
    let result = apply_opt(&mut scheduler, &opt_local, false);
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::ValidationFailed { op: "LOCAL", .. }));
    }
}

#[test]
fn test_nolocals_with_existing_local() {
    // Create a kernel with a Local axis
    let end_64 = UOp::index_const(64);
    let r_global = UOp::range_axis(end_64.clone(), AxisId::Renumbered(0), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global]);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // First apply LOCAL optimization
    let opt_local = Opt::local(0, 8);
    assert!(apply_opt(&mut scheduler, &opt_local, true).is_ok());

    // Now try to apply NOLOCALS (should fail)
    let opt = Opt::nolocals();
    let result = apply_opt(&mut scheduler, &opt, false);
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::ValidationFailed { op: "NOLOCALS", .. }));
    }
}

#[test]
fn test_swap_basic() {
    // Create a kernel with two Global axes
    let end_16 = UOp::index_const(16);
    let end_32 = UOp::index_const(32);
    let r_global1 = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::Global);
    let r_global2 = UOp::range_axis(end_32, AxisId::Renumbered(1), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global1, r_global2]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Get sizes of ranges before swap
    let rngs_before = scheduler.rngs();
    let get_size = |rng: &Rc<UOp>| -> i64 {
        if let Op::Range { end, .. } = rng.op()
            && let Op::Const(cv) = end.op()
            && let morok_ir::ConstValue::Int(sz) = cv.0
        {
            return sz;
        }
        panic!("Expected Range with constant size");
    };

    let size0_before = get_size(&rngs_before[0]);
    let size1_before = get_size(&rngs_before[1]);

    // Apply SWAP optimization
    let opt = Opt::swap(0, 1); // Swap axes 0 and 1
    let result = apply_opt(&mut scheduler, &opt, true);
    assert!(result.is_ok());

    // Verify sizes are swapped (axis_id 0 now has size of what was axis_id 1, and vice versa)
    let rngs_after = scheduler.rngs();
    let size0_after = get_size(&rngs_after[0]);
    let size1_after = get_size(&rngs_after[1]);

    assert_eq!(size0_after, size1_before, "axis_id 0 should now have the size that axis_id 1 had");
    assert_eq!(size1_after, size0_before, "axis_id 1 should now have the size that axis_id 0 had");
}

#[test]
fn test_swap_invalid_axis() {
    // Create a kernel with two Global axes
    let end_16 = UOp::index_const(16);
    let end_32 = UOp::index_const(32);
    let r_global1 = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::Global);
    let r_global2 = UOp::range_axis(end_32, AxisId::Renumbered(1), AxisType::Global);

    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_global1, r_global2]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Try to swap with out-of-bounds axis
    let opt = Opt::swap(0, 5);
    let result = apply_opt(&mut scheduler, &opt, false);
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::AxisOutOfBounds { .. }));
    }
}

#[test]
fn test_swap_non_global_axis() {
    // Create a kernel with a Global and a Reduce axis
    let end_16 = UOp::index_const(16);
    let end_32 = UOp::index_const(32);
    let r_global = UOp::range_axis(end_16, AxisId::Renumbered(0), AxisType::Global);
    let r_reduce = UOp::range_axis(end_32, AxisId::Renumbered(1), AxisType::Reduce);

    let compute = UOp::native_const(1.0f32);
    let reduce = UOp::reduce(compute, vec![r_global.clone(), r_reduce].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Try to swap Global with Reduce (should fail)
    let opt = Opt::swap(0, 1);
    let result = apply_opt(&mut scheduler, &opt, false);
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::ValidationFailed { op: "SWAP", .. }));
    }
}

#[test]
fn test_group_basic() {
    // Create a reduction kernel for GPU
    let end_64 = UOp::index_const(64);
    let r_reduce = UOp::range_axis(end_64, AxisId::Renumbered(0), AxisType::Reduce);

    let compute = UOp::native_const(1.0f32);
    let reduce = UOp::reduce(compute, vec![r_reduce].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Apply GROUP optimization
    let opt = Opt::group(0, 8); // Group first reduce axis by 8
    let result = apply_opt(&mut scheduler, &opt, true);
    assert!(result.is_ok());

    // Verify the shape changed: Reduce(64) -> Reduce(8) + GroupReduce(8)
    assert_eq!(scheduler.shape_len(), 2);

    let rngs = scheduler.rngs();
    // GroupReduce has priority 2, Reduce has priority 4, so GroupReduce comes first
    assert_eq!(get_axis_type(&rngs[0]), AxisType::GroupReduce);
    assert_eq!(get_axis_type(&rngs[1]), AxisType::Reduce);
}

#[test]
fn test_group_no_shared_memory() {
    // Create a reduction kernel for CPU (no shared memory)
    let end_64 = UOp::index_const(64);
    let r_reduce = UOp::range_axis(end_64, AxisId::Renumbered(0), AxisType::Reduce);

    let compute = UOp::native_const(1.0f32);
    let reduce = UOp::reduce(compute, vec![r_reduce].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Try to apply GROUP optimization (should fail on CPU)
    let opt = Opt::group(0, 8);
    let result = apply_opt(&mut scheduler, &opt, false);
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, OptError::UnsupportedFeature { .. }));
    }
}

// ===== Phase 7: Initialization & Finalization Tests =====

#[test]
fn test_convert_loop_to_global_gpu() {
    // Create a simple kernel with LOOP axes
    let loop1 = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Loop);
    let loop2 = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(1), AxisType::Loop);

    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val, loop1.clone(), loop2.clone()]);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Convert LOOP to GLOBAL
    scheduler.convert_loop_to_global().unwrap();

    // Verify that LOOP axes were converted to GLOBAL
    let ranges = scheduler.rngs();
    assert_eq!(ranges.len(), 2);

    for rng in ranges {
        if let Op::Range { axis_type, .. } = rng.op() {
            assert_eq!(*axis_type, AxisType::Global);
        } else {
            panic!("Expected RANGE operation");
        }
    }
}

#[test]
fn test_convert_loop_to_global_cpu() {
    // Create a simple kernel with LOOP axes
    let loop1 = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Loop);
    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val, loop1.clone()]);

    let ren = Renderer::cpu();
    let mut scheduler = Scheduler::new(sink, ren);

    // Convert should be a no-op for CPU
    scheduler.convert_loop_to_global().unwrap();

    // Verify that LOOP axes were NOT converted (CPU doesn't have local memory)
    let ranges = scheduler.rngs();
    assert_eq!(ranges.len(), 1);

    if let Op::Range { axis_type, .. } = ranges[0].op() {
        assert_eq!(*axis_type, AxisType::Loop);
    }
}

#[test]
fn test_get_optimized_ast_reduce_kernel() {
    // Create a reduction kernel
    let r_global = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Global);
    let r_local = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(1), AxisType::Local);
    let r_reduce = UOp::range_axis(UOp::index_const(32), AxisId::Renumbered(2), AxisType::Reduce);
    let r_upcast = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(3), AxisType::Upcast);

    let val = UOp::native_const(1.0f32);
    let reduce = UOp::reduce(val, vec![r_reduce.clone()].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce, r_global, r_local, r_upcast]);

    let ren = Renderer::cuda();
    let scheduler = Scheduler::new(sink, ren);

    // Get optimized AST
    let optimized = scheduler.get_optimized_ast(None);

    // Verify metadata is attached
    use crate::optimizer::KernelInfo;
    let info = optimized.metadata::<KernelInfo>();
    assert!(info.is_some());

    let info = info.unwrap();
    // Kernel name should be: r_g16l8R32u4
    assert!(info.name.starts_with("r_"));
    assert!(info.name.contains("g16"));
    assert!(info.name.contains("l8"));
    assert!(info.name.contains("R32"));
    assert!(info.name.contains("u4"));
}

#[test]
fn test_get_optimized_ast_elementwise_kernel() {
    // Create an elementwise kernel (no reduction)
    let r_global = UOp::range_axis(UOp::index_const(256), AxisId::Renumbered(0), AxisType::Global);

    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val, r_global]);

    let ren = Renderer::cuda();
    let scheduler = Scheduler::new(sink, ren);

    // Get optimized AST
    let optimized = scheduler.get_optimized_ast(None);

    // Verify metadata is attached
    use crate::optimizer::KernelInfo;
    let info = optimized.metadata::<KernelInfo>();
    assert!(info.is_some());

    let info = info.unwrap();
    // Kernel name should be: E_g256
    assert!(info.name.starts_with("E_"));
    assert!(info.name.contains("g256"));
}

#[test]
fn test_get_optimized_ast_custom_name() {
    let r_global = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Global);
    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val, r_global]);

    let ren = Renderer::cuda();
    let scheduler = Scheduler::new(sink, ren);

    // Get optimized AST with custom name
    let optimized = scheduler.get_optimized_ast(Some("custom_kernel".to_string()));

    // Verify custom name is used
    use crate::optimizer::KernelInfo;
    let info = optimized.metadata::<KernelInfo>();
    assert!(info.is_some());

    let info = info.unwrap();
    assert_eq!(info.name, "custom_kernel");
}

#[test]
fn test_phase7_integration() {
    // Full Phase 7 integration test: LOOP → GLOBAL → optimize → finalize
    let loop1 = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Loop);
    let loop2 = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(1), AxisType::Loop);

    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val, loop1, loop2]);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // 1. Convert LOOP to GLOBAL
    scheduler.convert_loop_to_global().unwrap();

    // 2. Apply some optimizations
    let opt = Opt::upcast(0, 4);
    apply_opt(&mut scheduler, &opt, true).unwrap();

    // 3. Get optimized AST
    let optimized = scheduler.get_optimized_ast(None);

    // 4. Verify metadata contains applied opts
    use crate::optimizer::KernelInfo;
    let info = optimized.metadata::<KernelInfo>();
    assert!(info.is_some());

    let info = info.unwrap();
    assert_eq!(info.applied_opts.len(), 1);
    assert_eq!(info.applied_opts[0].op, OptOps::UPCAST);
}

#[test]
fn test_kernel_name_deduplication() {
    use crate::optimizer::{KernelInfo, clear_kernel_name_counts};

    // Clear the counter to ensure clean state for this test
    clear_kernel_name_counts();

    // Create two identical kernel shapes
    let r_global = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Global);
    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val, r_global]);

    let ren = Renderer::cuda();
    let scheduler1 = Scheduler::new(sink.clone(), ren.clone());
    let scheduler2 = Scheduler::new(sink.clone(), ren.clone());
    let scheduler3 = Scheduler::new(sink.clone(), ren);

    // Get optimized ASTs
    let opt1 = scheduler1.get_optimized_ast(None);
    let opt2 = scheduler2.get_optimized_ast(None);
    let opt3 = scheduler3.get_optimized_ast(None);

    // Extract names
    let info1 = opt1.metadata::<KernelInfo>().unwrap();
    let info2 = opt2.metadata::<KernelInfo>().unwrap();
    let info3 = opt3.metadata::<KernelInfo>().unwrap();

    // All three names should be different (deduplication working)
    assert_ne!(info1.name, info2.name, "Second kernel should have different name than first");
    assert_ne!(info2.name, info3.name, "Third kernel should have different name than second");
    assert_ne!(info1.name, info3.name, "Third kernel should have different name than first");

    // They should all start with the same base name
    assert!(info1.name.starts_with("E_g16"), "First kernel name should start with E_g16");
    assert!(info2.name.starts_with("E_g16"), "Second kernel name should start with E_g16");
    assert!(info3.name.starts_with("E_g16"), "Third kernel name should start with E_g16");

    // The second and third should have the deduplication suffix 'n'
    assert!(info2.name.contains('n'), "Second kernel should have deduplication suffix");
    assert!(info3.name.contains('n'), "Third kernel should have deduplication suffix");

    // Clean up for other tests
    clear_kernel_name_counts();
}

#[test]
fn test_globalizable_rngs_with_sink() {
    // Test that SINK operations are properly handled in globalizable_rngs
    let loop1 = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Loop);
    let loop2 = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(1), AxisType::Loop);

    let val = UOp::native_const(1.0f32);
    // Use SINK operation
    let sink = UOp::sink(vec![val, loop1.clone(), loop2.clone()]);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Convert LOOP to GLOBAL
    scheduler.convert_loop_to_global().unwrap();

    // Verify that LOOP axes were converted to GLOBAL
    let ranges = scheduler.rngs();
    assert_eq!(ranges.len(), 2);

    for rng in ranges {
        if let Op::Range { axis_type, .. } = rng.op() {
            assert_eq!(*axis_type, AxisType::Global, "LOOP axes in SINK should be converted to GLOBAL");
        }
    }
}

#[test]
fn test_flatten_ranges_store() {
    // Test that STORE operations with nested REDUCE are properly flattened
    let r_reduce = UOp::range_axis(UOp::index_const(32), AxisId::Renumbered(0), AxisType::Reduce);

    let val = UOp::native_const(1.0f32);
    let reduce = UOp::reduce(val.clone(), vec![r_reduce].into(), ReduceOp::Add);

    // Create a STORE operation with the reduce as its value
    let buffer = UOp::index_const(0); // Dummy buffer
    let index = UOp::index_const(0); // Dummy index
    let store = UOp::store(buffer, index, reduce);

    let ren = Renderer::cuda();
    let scheduler = Scheduler::new(store, ren);

    // Get optimized AST (which calls flatten_ranges)
    let optimized = scheduler.get_optimized_ast(None);

    // Verify the AST was processed without errors
    // The flattening should have recursively processed the STORE and its nested REDUCE
    use crate::optimizer::KernelInfo;
    let info = optimized.metadata::<KernelInfo>();
    assert!(info.is_some(), "STORE with nested REDUCE should be flattened successfully");
}
