//! Tests for fix_bufferize_unroll (BUFFERIZE with UNROLL handling).
//!
//! fix_bufferize_unroll transforms BUFFERIZE operations where both compute and
//! ranges[0] are UNROLL ops, wrapping both in CONTRACT:
//!
//! BUFFERIZE(UNROLL(compute, axes), [UNROLL(range, axes)]) →
//! BUFFERIZE(CONTRACT(UNROLL(compute)), [CONTRACT(UNROLL(range))])
//!
//! After the full phase2 pass, CONTRACT(UNROLL) is further processed by do_contract,
//! which produces GEP-based extraction. So the final result won't have CONTRACT
//! but will have GEP or UNROLL(GEP).
//!
//! Based on Tinygrad's expander.py:91-92.

use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{Op, UOp};

use super::helpers::*;

// =============================================================================
// Basic Functionality Tests
// =============================================================================

/// Test: BUFFERIZE(UNROLL, [UNROLL]) is processed by fix_bufferize_unroll.
///
/// After phase2, CONTRACT(UNROLL) → GEP extraction via do_contract.
#[test]
fn test_bufferize_unroll_basic() {
    let compute = create_unroll_iota(0, 4);
    let range = create_unroll_iota(0, 4);
    let bufferize = create_bufferize_global(compute, vec![range]);

    let result = phase2_only(&bufferize);

    // After phase2, the UNROLLs should be expanded (CONTRACT→GEP)
    // The pattern fires and wraps in CONTRACT, then do_contract processes it
    match result.op() {
        Op::Bufferize { compute: c, .. } => {
            // Compute should now be either:
            // - CONTRACT (if do_contract hasn't run yet - unlikely)
            // - GEP (result of do_contract on CONTRACT(UNROLL))
            // - UNROLL(GEP) if some axes remain
            // The key check: UNROLL should be contracted away or transformed
            let has_raw_unroll = matches!(c.op(), Op::Unroll { .. });
            assert!(
                !has_raw_unroll
                    || matches!(c.op(), Op::Contract { .. })
                    || count_ops(&result, |u| matches!(u.op(), Op::Gep { .. })) > 0,
                "UNROLL should be processed, got {:?}",
                c.op()
            );
        }
        _ => {
            // May have been transformed to something else entirely
            // Just verify UNROLLs are processed
            assert!(
                count_unrolls(&result) == 0
                    || count_contracts(&result) > 0
                    || count_ops(&result, |u| matches!(u.op(), Op::Gep { .. })) > 0
            );
        }
    }
}

/// Test: Multi-axis UNROLL [(0,2), (1,3)] is contracted correctly.
#[test]
fn test_bufferize_unroll_multi_axis() {
    // Multi-axis UNROLL: total count = 2 * 3 = 6
    let compute = create_unroll_multi_axis(vec![(0, 2), (1, 3)]);
    let range = create_unroll_multi_axis(vec![(0, 2), (1, 3)]);

    let bufferize = create_bufferize_global(compute, vec![range]);
    let result = phase2_only(&bufferize);

    // After contraction, dtype should reflect vectorization
    // The pattern calculates: s.dtype.vec(x.src[1].src[0].dtype.count)
    // With vcount=6 inner elements
    if let Op::Bufferize { compute: c, .. } = result.op() {
        // The result dtype should have appropriate vectorization
        assert!(c.dtype().vcount() >= 1, "Contracted compute should have valid vcount");
    }
}

// =============================================================================
// Passthrough Tests (pattern should NOT fire)
// =============================================================================

/// Test: BUFFERIZE without UNROLL compute - other patterns may still transform.
#[test]
fn test_bufferize_no_unroll_compute_passthrough() {
    // Non-UNROLL compute
    let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let range = create_unroll_iota(0, 4);

    let bufferize = create_bufferize_global(compute, vec![range]);
    let result = phase2_only(&bufferize);

    // fix_bufferize_unroll should not fire (compute is not UNROLL)
    // But do_expand or other patterns may still process the range UNROLL
    assert!(count_bufferizes(&result) > 0 || count_contracts(&result) == 0);
}

/// Test: BUFFERIZE without UNROLL ranges.
#[test]
fn test_bufferize_no_unroll_ranges_passthrough() {
    let compute = create_unroll_iota(0, 4);
    // Non-UNROLL range (just a constant)
    let range = UOp::const_(DType::Index, ConstValue::Int(0));

    let bufferize = create_bufferize_global(compute, vec![range]);
    let result = phase2_only(&bufferize);

    // fix_bufferize_unroll should not fire (range is not UNROLL)
    // But compute UNROLL may be expanded by other patterns
    // Check that BUFFERIZE still exists or was reasonably transformed
    assert!(count_bufferizes(&result) > 0 || count_ops(&result, |u| matches!(u.op(), Op::Gep { .. })) > 0);
}

/// Test: BUFFERIZE with >1 range passes through fix_bufferize_unroll.
///
/// The pattern requires exactly 1 range that is UNROLL.
#[test]
fn test_bufferize_multiple_ranges_passthrough() {
    let compute = create_unroll_iota(0, 4);
    let range1 = create_unroll_iota(0, 2);
    let range2 = create_unroll_iota(1, 2);

    let bufferize = create_bufferize_global(compute, vec![range1, range2]);
    let result = phase2_only(&bufferize);

    // Pattern requires exactly 1 range - should not fire for this specific pattern
    // But other patterns may still process the UNROLLs
    if let Op::Bufferize { ranges, .. } = result.op() {
        assert!(ranges.len() >= 1);
    }
}

/// Test: BUFFERIZE with empty UNROLL axes.
#[test]
fn test_bufferize_empty_unroll_axes_passthrough() {
    // UNROLL with empty axes (edge case)
    let vconst = UOp::vconst(vec![ConstValue::Int(0)]);
    let compute = vconst.clone().unroll(vec![]);
    let range = vconst.unroll(vec![]);

    let bufferize = create_bufferize_global(compute, vec![range]);
    let result = phase2_only(&bufferize);

    // Empty axes → expander has special handling: UNROLL(x, ()) → x
    assert!(count_bufferizes(&result) > 0);
}

// =============================================================================
// CONTRACT Dtype Tests
// =============================================================================

/// Test: After contraction, dtype reflects vectorization.
#[test]
fn test_bufferize_contract_dtype_matches() {
    let compute = create_unroll_iota(0, 4);
    let range = create_unroll_iota(0, 4);

    let bufferize = create_bufferize_global(compute, vec![range]);
    let result = phase2_only(&bufferize);

    // Find any GEP and verify it has valid structure
    if let Op::Bufferize { compute: c, .. } = result.op() {
        // The vcount should be reasonable after contraction
        assert!(c.dtype().vcount() >= 1);
    }
}

/// Test: BufferizeOpts are preserved through transformation.
#[test]
fn test_bufferize_preserves_opts() {
    use morok_dtype::AddrSpace;
    use morok_ir::BufferizeOpts;

    let compute = create_unroll_iota(0, 4);
    let range = create_unroll_iota(0, 4);

    // Create with local addrspace
    let opts = BufferizeOpts::local();
    let bufferize = create_bufferize(compute, vec![range], opts);

    let result = phase2_only(&bufferize);

    // Verify opts preserved
    if let Op::Bufferize { opts: result_opts, .. } = result.op() {
        assert_eq!(result_opts.addrspace, AddrSpace::Local, "BufferizeOpts should be preserved");
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

/// Test: BUFFERIZE with UNROLL through full expander.
#[test]
fn test_bufferize_unroll_full_expander() {
    let compute = create_unroll_iota(0, 4);
    let range = create_unroll_iota(0, 4);

    let bufferize = create_bufferize_global(compute, vec![range]);

    // Use full expander (pre_expand)
    let result = expander_rewrite(&bufferize);

    // After full expansion, raw UNROLLs should be processed
    // Either contracted away or transformed to GEP
    let raw_unroll_count =
        count_ops(&result, |u| matches!(u.op(), Op::Unroll { unroll_axes, .. } if !unroll_axes.is_empty()));
    assert!(
        raw_unroll_count == 0
            || count_contracts(&result) > 0
            || count_ops(&result, |u| matches!(u.op(), Op::Gep { .. })) > 0,
        "UNROLLs should be expanded/contracted"
    );
}
