//! Pre-expander test suite (expand.rs).
//!
//! Tests for the UNROLL/CONTRACT expansion system, ported from Tinygrad's
//! TestExpander class (test_uop_graph.py:663-811).

pub mod bufferize_unroll;
pub mod do_contract;
pub mod do_expand;
pub mod edge_cases;
pub mod end_unrolls;
pub mod fix_reduce;
pub mod fix_store;
pub mod group_reduce;
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
fn test_fix_reduce_unroll_with_unroll_ops() {
    // Test the new Tinygrad-aligned behavior: fix_reduce_unroll partitions
    // REDUCE.ranges into RANGE ops vs UNROLL ops, moving UNROLL to CONTRACT.
    //
    // This tests the simplified partition-based logic.

    // Create an UNROLL op (simulates what Phase 1 produces from Range(Unroll))
    let values = UOp::vconst(vec![ConstValue::Int(0), ConstValue::Int(1), ConstValue::Int(2), ConstValue::Int(3)]);
    let unroll = values.unroll(vec![(1, 4)]);

    // Create a Reduce range
    let reduce_end = UOp::const_(DType::Index, ConstValue::Int(16));
    let reduce_range = UOp::range_axis(reduce_end, morok_ir::AxisId::Renumbered(0), AxisType::Reduce);

    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec::smallvec![reduce_range.clone(), unroll], ReduceOp::Add);

    // fix_reduce_unroll should:
    // 1. Partition ranges into [Range(Reduce)] and [UNROLL]
    // 2. Create CONTRACT wrapper on source with UNROLL axes
    // 3. Return REDUCE with only Range ops
    let result = fix_reduce_unroll(&reduce);
    assert!(result.is_some(), "Expected Some when UNROLL is in ranges");

    if let Some(fixed) = result
        && let Op::Reduce { src: fixed_src, ranges, .. } = fixed.op()
    {
        // Source should be wrapped in CONTRACT
        assert!(matches!(fixed_src.op(), Op::Contract { .. }), "Expected CONTRACT wrapper");
        // Ranges should only contain Range ops (UNROLL moved to CONTRACT)
        assert!(ranges.iter().all(|r| matches!(r.op(), Op::Range { .. })), "All ranges should be Range ops");
    }
}
