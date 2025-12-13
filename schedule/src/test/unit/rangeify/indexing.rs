//! Tests for IndexingContext and range assignment.
//!
//! Validates:
//! - Range creation and ID assignment
//! - Realize map tracking
//! - Range map operations
//! - Symbolic size handling
//! - Axis types (Loop vs Reduce)

use std::sync::Arc;

use morok_ir::{AxisId, AxisType, DType, Op, SInt, UOp};

use crate::rangeify::IndexingContext;

// ===== Basic Range Creation =====

#[test]
fn test_indexing_context_new_range() {
    let mut ctx = IndexingContext::new();

    // Test constant size - ranges are created with AxisId::Unrenumbered
    let r1 = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    assert!(matches!(r1.op(), Op::Range { axis_id, .. } if *axis_id == AxisId::Unrenumbered(0)));

    let r2 = ctx.new_range(&SInt::Const(20), AxisType::Loop);
    assert!(matches!(r2.op(), Op::Range { axis_id, .. } if *axis_id == AxisId::Unrenumbered(1)));

    // Test size 1 optimization (returns const 0)
    let r3 = ctx.new_range(&SInt::Const(1), AxisType::Loop);
    assert!(matches!(r3.op(), Op::Const(_)));
}

#[test]
fn test_indexing_context_realize_map() {
    let mut ctx = IndexingContext::new();
    let x = UOp::var("x", DType::Float32, i64::MAX);

    assert!(!ctx.should_realize(&x));

    ctx.mark_realize_all(&x).unwrap();
    assert!(ctx.should_realize(&x));
}

// ===== Range Counter =====

#[test]
fn test_range_counter_increments() {
    let mut ctx = IndexingContext::new();

    assert_eq!(ctx.range_counter(), 0);

    ctx.new_range(&SInt::Const(10), AxisType::Loop);
    assert_eq!(ctx.range_counter(), 1);

    ctx.new_range(&SInt::Const(20), AxisType::Loop);
    assert_eq!(ctx.range_counter(), 2);

    // Size 1 should NOT increment counter (returns const 0)
    ctx.new_range(&SInt::Const(1), AxisType::Loop);
    assert_eq!(ctx.range_counter(), 2);
}

// ===== Axis Types =====

#[test]
fn test_range_axis_types() {
    let mut ctx = IndexingContext::new();

    // Loop axis
    let loop_range = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    if let Op::Range { axis_type, .. } = loop_range.op() {
        assert_eq!(*axis_type, AxisType::Loop);
    } else {
        panic!("Expected Range op");
    }

    // Reduce axis
    let reduce_range = ctx.new_range(&SInt::Const(10), AxisType::Reduce);
    if let Op::Range { axis_type, .. } = reduce_range.op() {
        assert_eq!(*axis_type, AxisType::Reduce);
    } else {
        panic!("Expected Range op");
    }
}

// ===== Symbolic Sizes =====

#[test]
fn test_symbolic_size_range() {
    let mut ctx = IndexingContext::new();

    // Create symbolic size
    let n = UOp::var("n", DType::Index, i64::MAX);
    let symbolic_size = SInt::Symbolic(n.clone());

    let range = ctx.new_range(&symbolic_size, AxisType::Loop);

    // Should create range with symbolic end
    if let Op::Range { end, .. } = range.op() {
        assert!(Arc::ptr_eq(end, &n));
    } else {
        panic!("Expected Range op");
    }
}

// ===== Range Map Operations =====

#[test]
fn test_set_get_ranges() {
    let mut ctx = IndexingContext::new();
    let x = UOp::var("x", DType::Float32, i64::MAX);

    let r0 = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let r1 = ctx.new_range(&SInt::Const(20), AxisType::Loop);

    // Initially no ranges
    assert!(ctx.get_ranges(&x).is_none());

    // Set ranges
    let input_ranges = vec![r0.clone(), r1.clone()];
    let output_ranges = vec![r0.clone()];
    ctx.set_ranges(&x, input_ranges.clone(), output_ranges.clone());

    // Get ranges
    let ranges = ctx.get_ranges(&x);
    assert!(ranges.is_some());

    let (inp, out) = ranges.unwrap();
    assert_eq!(inp.len(), 2);
    assert_eq!(out.len(), 1);
    assert!(Arc::ptr_eq(&inp[0], &r0));
    assert!(Arc::ptr_eq(&inp[1], &r1));
    assert!(Arc::ptr_eq(&out[0], &r0));
}

// ===== Realize Axes =====

#[test]
fn test_mark_realize_specific_axes() {
    let mut ctx = IndexingContext::new();
    let x = UOp::var("x", DType::Float32, i64::MAX);

    // Mark specific axes
    ctx.mark_realize(&x, vec![0, 2]);

    assert!(ctx.should_realize(&x));

    let axes = ctx.get_realize_axes(&x);
    assert!(axes.is_some());
    assert_eq!(axes.unwrap(), &[0, 2]);
}

#[test]
fn test_get_realize_axes_none() {
    let ctx = IndexingContext::new();
    let x = UOp::var("x", DType::Float32, i64::MAX);

    // Not in realize map
    assert!(ctx.get_realize_axes(&x).is_none());
}

// ===== Multi-Dimensional =====

#[test]
fn test_multi_dimensional_ranges() {
    let mut ctx = IndexingContext::new();

    // Create 3D ranges
    let r0 = ctx.new_range(&SInt::Const(32), AxisType::Loop);
    let r1 = ctx.new_range(&SInt::Const(64), AxisType::Loop);
    let r2 = ctx.new_range(&SInt::Const(128), AxisType::Loop);

    // Verify sequential IDs
    assert!(matches!(r0.op(), Op::Range { axis_id: AxisId::Unrenumbered(0), .. }));
    assert!(matches!(r1.op(), Op::Range { axis_id: AxisId::Unrenumbered(1), .. }));
    assert!(matches!(r2.op(), Op::Range { axis_id: AxisId::Unrenumbered(2), .. }));

    // Verify sizes (ConstValueHash is a tuple struct wrapping ConstValue)
    use morok_ir::ConstValue;
    if let Op::Range { end, .. } = r0.op() {
        assert!(matches!(end.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(32))));
    }
    if let Op::Range { end, .. } = r1.op() {
        assert!(matches!(end.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(64))));
    }
    if let Op::Range { end, .. } = r2.op() {
        assert!(matches!(end.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(128))));
    }
}

// ===== Edge Cases =====

#[test]
fn test_zero_size_range() {
    let mut ctx = IndexingContext::new();

    // Size 0 should still create a range (not optimized like size 1)
    let range = ctx.new_range(&SInt::Const(0), AxisType::Loop);
    assert!(matches!(range.op(), Op::Range { .. }));
}

#[test]
fn test_large_size_range() {
    let mut ctx = IndexingContext::new();

    // Very large size
    let range = ctx.new_range(&SInt::Const(1 << 30), AxisType::Loop);

    use morok_ir::ConstValue;
    if let Op::Range { end, .. } = range.op() {
        assert!(matches!(end.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(v) if v == 1 << 30)));
    }
}

#[test]
fn test_multiple_contexts_independent() {
    // Two separate contexts should be independent
    let mut ctx1 = IndexingContext::new();
    let mut ctx2 = IndexingContext::new();

    ctx1.new_range(&SInt::Const(10), AxisType::Loop);
    ctx1.new_range(&SInt::Const(20), AxisType::Loop);

    // ctx2 starts fresh
    assert_eq!(ctx2.range_counter(), 0);

    let r = ctx2.new_range(&SInt::Const(30), AxisType::Loop);
    assert!(matches!(r.op(), Op::Range { axis_id: AxisId::Unrenumbered(0), .. }));
}
