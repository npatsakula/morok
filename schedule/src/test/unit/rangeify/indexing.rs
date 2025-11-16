use morok_ir::{AxisType, DType, Op, SInt, UOp};

use crate::rangeify::IndexingContext;

#[test]
fn test_indexing_context_new_range() {
    let mut ctx = IndexingContext::new();

    // Test constant size
    let r1 = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    assert!(matches!(r1.op(), Op::Range { axis_id: 0, .. }));

    let r2 = ctx.new_range(&SInt::Const(20), AxisType::Loop);
    assert!(matches!(r2.op(), Op::Range { axis_id: 1, .. }));

    // Test size 1 optimization (returns const 0)
    let r3 = ctx.new_range(&SInt::Const(1), AxisType::Loop);
    assert!(matches!(r3.op(), Op::Const(_)));
}

#[test]
fn test_indexing_context_realize_map() {
    let mut ctx = IndexingContext::new();
    let x = UOp::define_global(0, DType::Float32);

    assert!(!ctx.should_realize(&x));

    ctx.mark_realize_all(&x);
    assert!(ctx.should_realize(&x));
}
