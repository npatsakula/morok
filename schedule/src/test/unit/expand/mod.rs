//! Pre-expander test suite (expand.rs).
//!
//! Tests for the UNROLL/CONTRACT expansion system, ported from Tinygrad's
//! TestExpander class (test_uop_graph.py:663-811).

pub mod do_contract;
pub mod do_expand;
pub mod edge_cases;
pub mod fix_reduce;
pub mod fix_store;
pub mod helpers;
pub mod shift_to_integration;
pub mod swizzle;

use crate::expand::*;
use morok_ir::{AxisType, prelude::*};

#[test]
fn test_pre_expand_passthrough() {
    // A simple REDUCE with proper Range ops should pass through unchanged
    let end = UOp::const_(DType::Index, ConstValue::Int(32));
    let range = UOp::range_axis(end, morok_ir::AxisId::Renumbered(0), AxisType::Reduce);
    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec::smallvec![range.clone()], ReduceOp::Add);

    let result = pre_expand(&reduce);

    // Should be unchanged (though may be a new node due to graph_rewrite)
    if let Op::Reduce { ranges, .. } = result.op() {
        assert_eq!(ranges.len(), 1);
        assert!(matches!(ranges[0].op(), Op::Range { axis_type: AxisType::Reduce, .. }));
    } else {
        panic!("Expected REDUCE op");
    }
}

#[test]
fn test_vectorize_expansion_with_mixed_sources() {
    // Test that VECTORIZE with mixed scalar/vector sources after expansion
    // produces CAT instead of invalid VECTORIZE.
    //
    // This tests the fix for: "Invalid VECTORIZE operand count: 2, expected 4"
    // which occurred when beam search used width >= 3.

    // Create an UNROLL operation (simulates expanded loop)
    let values = UOp::vconst(vec![ConstValue::Int(0), ConstValue::Int(1), ConstValue::Int(2)]);
    let unroll = values.unroll(vec![(0, 3)]);

    // Create a scalar constant
    let scalar = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Create a Binary op with UNROLL - this will trigger expansion
    // The scalar source will be broadcast, creating a vector
    let binary = UOp::new(Op::Binary(morok_ir::BinaryOp::Add, scalar.clone(), unroll.clone()), DType::Float32);

    // Run pre_expand - should not panic
    let result = pre_expand(&binary);

    // Result should be wrapped in UNROLL with expanded inner op
    assert!(
        matches!(result.op(), Op::Unroll { .. } | Op::Binary(..)),
        "Expected UNROLL or Binary, got {:?}",
        result.op()
    );
}

#[test]
fn test_vectorize_all_scalar_sources() {
    // When all sources are scalar after expansion, VECTORIZE should be used
    let scalar_a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let scalar_b = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    // Create VECTORIZE with scalars only (no UNROLL)
    let vectorize = UOp::vectorize(smallvec::smallvec![scalar_a, scalar_b]);

    // No expansion needed - should pass through unchanged
    let result = pre_expand(&vectorize);

    // Should still be VECTORIZE (or equivalent)
    assert_eq!(result.dtype().vcount(), 2);
}

#[test]
fn test_fix_reduce_unroll_returns_none_on_unextractable_binary() {
    // Test that fix_reduce_unroll returns None when ranges contain
    // unextractable Binary expressions. This prevents infinite rewrite loops.
    //
    // This tests the fix for: "Rewrite iteration limit (1000) exceeded"
    // which occurred when beam search used width >= 3.

    // Create a Binary expression that doesn't match any extraction pattern:
    // ADD(MUL(Const, Const), Const) - no Range ops at all!
    // This simulates a malformed expression that might result from some edge case.
    let unextractable = UOp::new(
        Op::Binary(
            morok_ir::BinaryOp::Add,
            UOp::new(
                Op::Binary(
                    morok_ir::BinaryOp::Mul,
                    UOp::const_(DType::Index, ConstValue::Int(2)),
                    UOp::const_(DType::Index, ConstValue::Int(3)),
                ),
                DType::Index,
            ),
            UOp::const_(DType::Index, ConstValue::Int(1)),
        ),
        DType::Index,
    );

    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec::smallvec![unextractable], ReduceOp::Add);

    // fix_reduce_unroll should return None because:
    // 1. The Binary expression can't be extracted (no Range ops)
    // 2. The range is kept as-is (no change)
    // 3. No CONTRACT is added
    // -> ranges_unchanged && !has_contract -> return None
    let result = fix_reduce_unroll(&reduce);
    assert!(result.is_none(), "Expected None when extraction fails and nothing changes");
}

#[test]
fn test_fix_reduce_unroll_handles_nested_binary() {
    // Test that nested Binary expressions (from multiple shift_to) are handled.
    // Pattern: ADD(ADD(MUL(reduce_range, 4), range_upcast1), range_upcast2)

    let end = UOp::const_(DType::Index, ConstValue::Int(64));
    let reduce_range = UOp::range_axis(end.clone(), morok_ir::AxisId::Renumbered(0), AxisType::Reduce);

    // Inner shift_to result: MUL(reduce_range, 4) + Range(Upcast)
    let upcast_end = UOp::const_(DType::Index, ConstValue::Int(4));
    let upcast_range = UOp::range_axis(upcast_end.clone(), morok_ir::AxisId::Renumbered(1), AxisType::Upcast);

    let mul = UOp::new(
        Op::Binary(morok_ir::BinaryOp::Mul, reduce_range.clone(), UOp::const_(DType::Index, ConstValue::Int(4))),
        DType::Index,
    );
    let inner_add = UOp::new(Op::Binary(morok_ir::BinaryOp::Add, mul, upcast_range.clone()), DType::Index);

    // Outer shift_to: inner_add + Range(Upcast)
    let upcast_range2 = UOp::range_axis(upcast_end, morok_ir::AxisId::Renumbered(2), AxisType::Upcast);
    let nested_binary = UOp::new(Op::Binary(morok_ir::BinaryOp::Add, inner_add, upcast_range2), DType::Index);

    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec::smallvec![nested_binary], ReduceOp::Add);

    // fix_reduce_unroll should extract the reduce_range and handle the upcast axes
    let result = fix_reduce_unroll(&reduce);
    assert!(result.is_some(), "Expected Some when nested Binary can be extracted");

    // Check the result has CONTRACT wrapper (for upcast axes)
    if let Some(fixed) = result
        && let Op::Reduce { src: fixed_src, .. } = fixed.op()
    {
        assert!(matches!(fixed_src.op(), Op::Contract { .. }), "Expected CONTRACT wrapper for upcast axes");
    }
}
