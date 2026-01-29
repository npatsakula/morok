//! Integration tests for shift_to → expand.rs pipeline.
//!
//! These tests verify the complete optimization path through the Scheduler.
//! The actual shift_to and expand behavior is tested via the real Scheduler infrastructure.

use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{AxisId, AxisType, Op, ReduceOp, UOp};
use smallvec::smallvec;

use crate::expand::pre_expand;
use crate::optimizer::Renderer;
use crate::optimizer::Scheduler;

/// Helper: Create a simple reduction sum([0..size]).
fn create_simple_reduce(size: usize, axis_id: usize) -> Arc<UOp> {
    let end = UOp::const_(DType::Index, ConstValue::Int(size as i64));
    let range = UOp::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Reduce);
    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    src.reduce(smallvec![range], ReduceOp::Add)
}

// =============================================================================
// Full Scheduler integration (real shift_to call)
// =============================================================================

/// Test the complete path through Scheduler.shift_to → pre_expand.
///
/// This uses the actual Scheduler infrastructure instead of manual expression building.
#[test]
fn test_scheduler_shift_to_integration() {
    // Create a simple kernel AST with STORE(REDUCE)
    let reduce = create_simple_reduce(16, 0);

    // Create simple index for STORE
    let index = UOp::index_const(0);

    // Create STORE with the REDUCE as value
    let loop_end = UOp::const_(DType::Index, ConstValue::Int(1));
    let loop_range = UOp::range_axis(loop_end, AxisId::Renumbered(1), AxisType::Loop);
    let store = index.store_with_ranges(reduce.clone(), smallvec![loop_range.clone()]);

    // Wrap in SINK (required for Scheduler)
    let ast = UOp::sink(vec![store]);

    // Create Scheduler with CPU renderer (no device limits)
    let renderer = Renderer::cpu();
    let mut scheduler = Scheduler::new(ast.clone(), renderer);

    // Get the reduce range
    let reduce_range = scheduler.rngs().iter().find(|r| {
        matches!(r.op(), Op::Range { axis_type: AxisType::Reduce, .. })
    }).cloned();

    if let Some(rng) = reduce_range {
        // Apply shift_to with UNROLL
        let result = scheduler.shift_to(rng, 4, AxisType::Unroll, false, None);
        assert!(result.is_ok(), "shift_to should succeed");

        // Get the optimized AST and run pre_expand
        let optimized = scheduler.get_optimized_ast(None);
        let expanded = pre_expand(&optimized);

        // Verify the expanded AST has proper structure
        // The REDUCE inside should have CONTRACT-wrapped source
        let mut found_reduce = false;
        for node in expanded.toposort() {
            if let Op::Reduce { src, ranges, .. } = node.op() {
                found_reduce = true;
                // Source should be expanded (CONTRACT or VECTORIZE)
                assert!(
                    matches!(src.op(), Op::Contract { .. } | Op::Vectorize { .. } | Op::Const(_)),
                    "REDUCE.src should be CONTRACT/VECTORIZE/Const after expansion, got {:?}",
                    src.op()
                );
                // Ranges should not contain arithmetic expressions
                for range in ranges.iter() {
                    assert!(
                        !matches!(range.op(), Op::Binary(..)),
                        "REDUCE.ranges should not contain Binary after expansion"
                    );
                }
            }
        }
        assert!(found_reduce, "Should find REDUCE in expanded AST");
    }
}
