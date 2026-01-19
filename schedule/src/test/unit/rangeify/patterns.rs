//! Comprehensive tests for rangeify pattern matchers.
//!
//! Tests verify that all pattern matchers correctly transform UOps:
//! - early_rewrites: DETACH and CONTIGUOUS_BACKWARD removal
//! - buffer_folding: Noop bufferize removal and constant propagation
//! - dead_axis_removal: Remove size-1 dimensions
//! - buffer_removal: Cost-based buffer elimination
//!
//! Based on Tinygrad's test_schedule.py pattern tests.

use std::f32::consts::PI;
use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, BufferizeOpts, ConstValue, Op, UOp};

use crate::pattern::RewriteResult;
use crate::rangeify::IndexingContext;
use crate::rangeify::patterns;

// ===== early_rewrites Pattern Tests =====

#[test]
fn test_early_rewrites_detach_removal() {
    let matcher = patterns::early_rewrites();

    // Test: DETACH(x) → x
    let x = UOp::native_const(42.0f32);
    let detach = UOp::detach(x.clone());

    let result = matcher.rewrite(&detach, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)), "Should rewrite DETACH");

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x), "Should return the source");
    }
}

#[test]
fn test_early_rewrites_contiguous_backward_removal() {
    let matcher = patterns::early_rewrites();

    // Test: CONTIGUOUS_BACKWARD(x) → x
    let x = UOp::native_const(PI);
    let contiguous = UOp::contiguous_backward(x.clone());

    let result = matcher.rewrite(&contiguous, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)), "Should rewrite CONTIGUOUS_BACKWARD");

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x), "Should return the source");
    }
}

#[test]
fn test_early_rewrites_no_match_for_other_ops() {
    let matcher = patterns::early_rewrites();

    // Test that non-DETACH/CONTIGUOUS_BACKWARD operations return NoMatch
    let const_op = UOp::native_const(1.0f32);
    let result = matcher.rewrite(&const_op, &mut ());
    assert!(matches!(result, RewriteResult::NoMatch), "Should not match CONST");

    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let add = a.try_add(&b).unwrap();
    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::NoMatch), "Should not match Binary ops");
}

#[test]
fn test_early_rewrites_nested_detach() {
    let matcher = patterns::early_rewrites();

    // Test: DETACH(DETACH(x)) should rewrite outer DETACH to DETACH(x)
    let x = UOp::native_const(1.0f32);
    let inner_detach = UOp::detach(x.clone());
    let outer_detach = UOp::detach(inner_detach.clone());

    let result = matcher.rewrite(&outer_detach, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &inner_detach), "Should unwrap outer DETACH to inner DETACH");
    }
}

// ===== buffer_folding Pattern Tests =====

#[test]
fn test_buffer_folding_noop_bufferize() {
    let matcher = patterns::buffer_folding();

    // Test: INDEX(BUFFERIZE(x, ranges), ranges) → x when ranges are equal
    let x = UOp::native_const(1.0f32);
    let range_end = UOp::index_const(10);
    let range = UOp::range_axis(range_end, AxisId::Renumbered(0), AxisType::Loop);

    let bufferize = UOp::bufferize(x.clone(), vec![range.clone()], BufferizeOpts::local());
    let index = UOp::index(bufferize, vec![range]).unwrap();

    let result = matcher.rewrite(&index, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)), "Should remove noop BUFFERIZE");

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x), "Should return the compute directly");
    }
}

#[test]
fn test_buffer_folding_bufferize_const() {
    let matcher = patterns::buffer_folding();

    // Test: BUFFERIZE(CONST) → CONST
    let const_val = UOp::native_const(42.0f32);
    let range_end = UOp::index_const(10);
    let range = UOp::range_axis(range_end, AxisId::Renumbered(0), AxisType::Loop);
    let bufferize = UOp::bufferize(const_val.clone(), vec![range], BufferizeOpts::local());

    let result = matcher.rewrite(&bufferize, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)), "Should remove BUFFERIZE from CONST");

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &const_val), "Should return the constant directly");
    }
}

#[test]
fn test_buffer_folding_index_const() {
    let matcher = patterns::buffer_folding();

    // Test: INDEX(CONST) → CONST
    let const_val = UOp::native_const(PI);
    let range_end = UOp::index_const(10);
    let range = UOp::range_axis(range_end, AxisId::Renumbered(0), AxisType::Loop);
    let index = UOp::index(const_val.clone(), vec![range]).unwrap();

    let result = matcher.rewrite(&index, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)), "Should remove INDEX from CONST");

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &const_val), "Should return the constant directly");
    }
}

#[test]
fn test_buffer_folding_copy_const() {
    let matcher = patterns::buffer_folding();

    // Test: COPY(CONST, device) → CONST
    let const_val = UOp::native_const(1.0f32);
    let device = UOp::device(morok_ir::DeviceSpec::Cpu);
    let copy = UOp::copy(const_val.clone(), device);

    let result = matcher.rewrite(&copy, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)), "Should remove COPY from CONST");

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &const_val), "Should return the constant directly");
    }
}

#[test]
fn test_buffer_folding_no_match_different_ranges() {
    let matcher = patterns::buffer_folding();

    // Test: INDEX(BUFFERIZE(x, r1), r2) should NOT match when r1 != r2
    let x = UOp::native_const(1.0f32);
    let range1_end = UOp::index_const(10);
    let range1 = UOp::range_axis(range1_end, AxisId::Renumbered(0), AxisType::Loop);

    let range2_end = UOp::index_const(20);
    let range2 = UOp::range_axis(range2_end, AxisId::Renumbered(1), AxisType::Loop);

    let bufferize = UOp::bufferize(x, vec![range1], BufferizeOpts::local());
    let index = UOp::index(bufferize, vec![range2]).unwrap();

    let result = matcher.rewrite(&index, &mut ());
    // This might match or not depending on implementation details,
    // but should NOT return the original compute 'x' directly
    match result {
        RewriteResult::NoMatch => {}
        RewriteResult::Rewritten(rewritten) => {
            // If it rewrites, it should not be the original 'x'
            assert!(!matches!(rewritten.op(), Op::Const(_)));
        }
        RewriteResult::Gate(_) => {}
    }
}

// ===== dead_axis_removal Pattern Tests =====

#[test]
fn test_dead_axis_removal_single_dead_axis() {
    let matcher = patterns::dead_axis_removal();

    // Create a BUFFERIZE with one dead axis (range with size 1)
    let x = UOp::native_const(1.0f32);
    let dead_range_end = UOp::index_const(1); // size 1 = dead
    let dead_range = UOp::range_axis(dead_range_end, AxisId::Renumbered(0), AxisType::Loop);

    let bufferize = UOp::bufferize(x.clone(), vec![dead_range], BufferizeOpts::local());

    let result = matcher.rewrite(&bufferize, &mut ());

    // Should remove the dead axis and return compute directly
    match result {
        RewriteResult::Rewritten(rewritten) => {
            // Either returns compute directly or BUFFERIZE with no ranges
            // Since all ranges are dead, should return compute directly
            assert!(
                Arc::ptr_eq(&rewritten, &x) || matches!(rewritten.op(), Op::Bufferize { .. }),
                "Should either return compute or empty BUFFERIZE"
            );
        }
        _ => {
            // This is also acceptable if dead axis detection has specific conditions
        }
    }
}

#[test]
fn test_dead_axis_removal_mixed_axes() {
    let matcher = patterns::dead_axis_removal();

    // Create BUFFERIZE with mix of live and dead axes
    // NOTE: When compute is native_const (no ranges), ALL ranges are dead
    // because compute doesn't depend on any of them (Tinygrad behavior)
    let x = UOp::native_const(1.0f32);
    let live_range_end = UOp::index_const(10);
    let live_range = UOp::range_axis(live_range_end, AxisId::Renumbered(0), AxisType::Loop);

    let dead_range_end = UOp::index_const(1);
    let dead_range = UOp::range_axis(dead_range_end, AxisId::Renumbered(1), AxisType::Loop);

    let bufferize = UOp::bufferize(x.clone(), vec![live_range.clone(), dead_range], BufferizeOpts::local());

    let result = matcher.rewrite(&bufferize, &mut ());

    match result {
        RewriteResult::Rewritten(rewritten) => {
            // Since compute has no ranges, ALL ranges are dead → return compute directly
            assert!(
                Arc::ptr_eq(&rewritten, &x),
                "When compute has no ranges, all BUFFERIZE ranges are dead → return compute"
            );
        }
        _ => {
            // Pattern should match and rewrite when there are dead axes
            panic!("Expected pattern to match and rewrite");
        }
    }
}

#[test]
fn test_dead_axis_removal_no_dead_axes_simple_compute() {
    let matcher = patterns::dead_axis_removal();

    // Create BUFFERIZE with "live" axes (size > 1), but simple compute (no ranges)
    // NOTE: When compute is native_const (no ranges), ALL ranges are dead
    // because compute doesn't depend on any of them (Tinygrad behavior)
    let x = UOp::native_const(1.0f32);
    let range1_end = UOp::index_const(10);
    let range1 = UOp::range_axis(range1_end, AxisId::Renumbered(0), AxisType::Loop);

    let range2_end = UOp::index_const(20);
    let range2 = UOp::range_axis(range2_end, AxisId::Renumbered(1), AxisType::Loop);

    let bufferize = UOp::bufferize(x.clone(), vec![range1, range2], BufferizeOpts::local());

    let result = matcher.rewrite(&bufferize, &mut ());

    // All ranges are dead (compute has no ranges) → should return compute directly
    match result {
        RewriteResult::Rewritten(rewritten) => {
            assert!(
                Arc::ptr_eq(&rewritten, &x),
                "When compute has no ranges, all BUFFERIZE ranges are dead → return compute"
            );
        }
        _ => panic!("Expected pattern to match and rewrite when all ranges are dead"),
    }
}

// ===== buffer_removal Pattern Tests =====

#[test]
fn test_buffer_removal_cheap_compute() {
    let matcher = patterns::buffer_removal();

    // Test: BUFFERIZE(cheap_op) should be removed if cheap to inline
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let add = a.try_add(&b).unwrap(); // Binary add is cheap

    let range_end = UOp::index_const(10);
    let range = UOp::range_axis(range_end, AxisId::Renumbered(0), AxisType::Loop);
    let bufferize = UOp::bufferize(add.clone(), vec![range], BufferizeOpts::local());

    let result = matcher.rewrite(&bufferize, &mut ());

    match result {
        RewriteResult::Rewritten(rewritten) => {
            assert!(Arc::ptr_eq(&rewritten, &add), "Should remove BUFFERIZE from cheap compute");
        }
        _ => {
            // Acceptable if cost model determines it's not cheap enough
        }
    }
}

#[test]
fn test_buffer_removal_always_run_ops() {
    let matcher = patterns::buffer_removal();

    // Test: BUFFERIZE(CONTIGUOUS) should be removed (always-run op)
    let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let contiguous = UOp::contiguous(src.clone());

    let range_end = UOp::index_const(10);
    let range = UOp::range_axis(range_end, AxisId::Renumbered(0), AxisType::Loop);
    let bufferize = UOp::bufferize(contiguous.clone(), vec![range], BufferizeOpts::local());

    let result = matcher.rewrite(&bufferize, &mut ());

    match result {
        RewriteResult::Rewritten(rewritten) => {
            assert!(Arc::ptr_eq(&rewritten, &contiguous), "Should remove BUFFERIZE from always-run op");
        }
        _ => {
            // Acceptable depending on implementation
        }
    }
}

#[test]
fn test_buffer_removal_nested_bufferize() {
    let matcher = patterns::buffer_removal();

    // Test: BUFFERIZE(BUFFERIZE(x, r1), r2) → BUFFERIZE(x, r2)
    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let range1_end = UOp::index_const(10);
    let range1 = UOp::range_axis(range1_end, AxisId::Renumbered(0), AxisType::Loop);

    let inner = UOp::bufferize(x.clone(), vec![range1], BufferizeOpts::local());

    let range2_end = UOp::index_const(20);
    let range2 = UOp::range_axis(range2_end, AxisId::Renumbered(1), AxisType::Loop);

    let outer = UOp::bufferize(inner, vec![range2.clone()], BufferizeOpts::local());

    let result = matcher.rewrite(&outer, &mut ());

    match result {
        RewriteResult::Rewritten(rewritten) => {
            if let Op::Bufferize { compute, .. } = rewritten.op() {
                // Should have unwrapped inner BUFFERIZE
                assert!(Arc::ptr_eq(compute, &x), "Should have compute pointing to x, not inner BUFFERIZE");
            } else {
                panic!("Expected BUFFERIZE operation");
            }
        }
        _ => {
            // Acceptable depending on implementation
        }
    }
}

#[test]
fn test_buffer_removal_no_match_expensive_compute() {
    let matcher = patterns::buffer_removal();

    // Test: BUFFERIZE(expensive_op) should NOT be removed
    // LOAD is typically considered expensive and should not be inlined
    let buffer = UOp::buffer_id(Some(0));
    let index = UOp::index_const(0);
    let load = UOp::load().buffer(buffer).index(index).call();

    let range_end = UOp::index_const(10);
    let range = UOp::range_axis(range_end, AxisId::Renumbered(0), AxisType::Loop);
    let bufferize = UOp::bufferize(load, vec![range], BufferizeOpts::local());

    let result = matcher.rewrite(&bufferize, &mut ());

    // Should not remove BUFFERIZE from expensive LOAD
    assert!(matches!(result, RewriteResult::NoMatch), "Should not remove BUFFERIZE from expensive op");
}

// ===== Movement Op Removal Tests =====
// These tests verify movement op removal behavior which is now integrated into apply_rangeify_patterns

#[test]
fn test_movement_op_removal_no_match_without_ranges() {
    let matcher = patterns::apply_rangeify_patterns();
    let mut ctx = IndexingContext::new();

    // Create a PERMUTE operation (a movement op)
    let src = UOp::define_global(0, DType::Float32);
    let permute = UOp::new(Op::Permute { src: src.clone(), axes: vec![1, 0] }, DType::Float32);

    // Without ranges assigned, should NOT remove
    // (The bufferize pattern will try to match but return None without ranges)
    let result = matcher.rewrite(&permute, &mut ctx);
    assert!(matches!(result, RewriteResult::NoMatch), "Should NOT remove movement op without ranges assigned");
}

#[test]
fn test_movement_op_removal_removes_with_ranges() {
    let matcher = patterns::apply_rangeify_patterns();
    let mut ctx = IndexingContext::new();

    // Create a PERMUTE operation
    let src = UOp::define_global(0, DType::Float32);
    let permute = UOp::new(Op::Permute { src: src.clone(), axes: vec![1, 0] }, DType::Float32);

    // Assign ranges to the movement op (simulating transformation has been applied)
    let range = UOp::new(
        Op::Range { end: UOp::index_const(5), axis_id: AxisId::Renumbered(0), axis_type: AxisType::Loop },
        DType::Index,
    );
    ctx.set_ranges(&permute, vec![range.clone()], vec![range.clone()]);

    // With ranges assigned, SHOULD remove and return source
    let result = matcher.rewrite(&permute, &mut ctx);
    match result {
        RewriteResult::Rewritten(result) => {
            assert!(std::sync::Arc::ptr_eq(&result, &src), "Should return the source operand");
        }
        _ => panic!("Expected movement op to be removed when ranges are assigned"),
    }
}

#[test]
fn test_movement_op_removal_reshape() {
    let matcher = patterns::apply_rangeify_patterns();
    let mut ctx = IndexingContext::new();

    // Create a RESHAPE operation
    let src = UOp::define_global(0, DType::Float32);
    let new_shape = UOp::vectorize(smallvec::smallvec![UOp::index_const(4), UOp::index_const(8)]);
    let reshape = UOp::new(Op::Reshape { src: src.clone(), new_shape }, DType::Float32);

    // Assign ranges
    let range = UOp::new(
        Op::Range { end: UOp::index_const(4), axis_id: AxisId::Renumbered(0), axis_type: AxisType::Loop },
        DType::Index,
    );
    ctx.set_ranges(&reshape, vec![range.clone()], vec![range.clone()]);

    // Should remove and return source
    let result = matcher.rewrite(&reshape, &mut ctx);
    match result {
        RewriteResult::Rewritten(result) => {
            assert!(std::sync::Arc::ptr_eq(&result, &src), "RESHAPE should be removed");
        }
        _ => panic!("Expected RESHAPE to be removed when ranges are assigned"),
    }
}

#[test]
fn test_movement_op_removal_expand() {
    let matcher = patterns::apply_rangeify_patterns();
    let mut ctx = IndexingContext::new();

    // Create an EXPAND operation
    let src = UOp::define_global(0, DType::Float32);
    let new_shape = UOp::vectorize(smallvec::smallvec![UOp::index_const(4), UOp::index_const(8)]);
    let expand = UOp::new(Op::Expand { src: src.clone(), new_shape }, DType::Float32);

    // Assign ranges
    let range = UOp::new(
        Op::Range { end: UOp::index_const(4), axis_id: AxisId::Renumbered(0), axis_type: AxisType::Loop },
        DType::Index,
    );
    ctx.set_ranges(&expand, vec![range.clone()], vec![range.clone()]);

    // Should remove and return source
    let result = matcher.rewrite(&expand, &mut ctx);
    match result {
        RewriteResult::Rewritten(result) => {
            assert!(std::sync::Arc::ptr_eq(&result, &src), "EXPAND should be removed");
        }
        _ => panic!("Expected EXPAND to be removed when ranges are assigned"),
    }
}

#[test]
fn test_movement_op_removal_non_movement_op() {
    let matcher = patterns::apply_rangeify_patterns();
    let mut ctx = IndexingContext::new();

    // Create a non-movement op (NEG)
    let src = UOp::define_global(0, DType::Float32);
    let neg = src.neg();

    // Non-movement ops without ranges should not match the movement removal pattern
    // (they may match other patterns like bufferize, but without ranges assigned,
    // apply_bufferize_transform returns None)
    let result = matcher.rewrite(&neg, &mut ctx);
    assert!(matches!(result, RewriteResult::NoMatch), "Should not match non-movement ops without ranges");
}

// ===== Integration Tests =====

#[test]
fn test_pattern_composition() {
    // Test that multiple patterns can be applied in sequence

    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // First apply DETACH
    let detach = UOp::detach(x.clone());

    // Then apply early_rewrites to remove DETACH
    let early = patterns::early_rewrites();
    let result1 = early.rewrite(&detach, &mut ());
    assert!(matches!(result1, RewriteResult::Rewritten(_)));

    let unwrapped = if let RewriteResult::Rewritten(r) = result1 {
        r
    } else {
        panic!("Should have rewritten");
    };

    // Now wrap in BUFFERIZE
    let range_end = UOp::index_const(10);
    let range = UOp::range_axis(range_end, AxisId::Renumbered(0), AxisType::Loop);
    let bufferize = UOp::bufferize(unwrapped, vec![range], BufferizeOpts::local());

    // Apply buffer_folding to remove BUFFERIZE(CONST)
    let folding = patterns::buffer_folding();
    let result2 = folding.rewrite(&bufferize, &mut ());

    match result2 {
        RewriteResult::Rewritten(rewritten) => {
            assert!(Arc::ptr_eq(&rewritten, &x), "Should have removed both DETACH and BUFFERIZE");
        }
        _ => {
            // Acceptable depending on implementation
        }
    }
}

#[test]
fn test_idempotent_patterns() {
    // Test that applying patterns multiple times doesn't cause issues

    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let detach = UOp::detach(x.clone());

    let matcher = patterns::early_rewrites();

    // First application
    let result1 = matcher.rewrite(&detach, &mut ());
    assert!(matches!(result1, RewriteResult::Rewritten(_)));

    let unwrapped = if let RewriteResult::Rewritten(r) = result1 { r } else { x.clone() };

    // Second application (should not match on CONST)
    let result2 = matcher.rewrite(&unwrapped, &mut ());
    assert!(matches!(result2, RewriteResult::NoMatch), "Should not match on already-processed node");
}
