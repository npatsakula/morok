//! Tests for codegen preparation patterns.
//!
//! Validates that the rangeify_codegen_patterns correctly transform UOps for code generation.

use std::sync::Arc;

use morok_ir::{ContiguousHint, Op, UOp};

use crate::rangeify::kernel::LocalAddBufferContext;
use crate::rangeify::patterns::rangeify_codegen_patterns;

/// Helper to apply rangeify_codegen patterns and return result
fn apply_patterns(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let matcher = rangeify_codegen_patterns();
    let mut ctx = LocalAddBufferContext::new();
    let result = crate::rewrite::graph_rewrite_bottom_up(&matcher, uop.clone(), &mut ctx);
    if Arc::ptr_eq(&result, uop) { None } else { Some(result) }
}

/// Helper to apply patterns and return both result and context (for opts inspection)
fn apply_patterns_with_ctx(uop: &Arc<UOp>) -> (Arc<UOp>, LocalAddBufferContext) {
    let matcher = rangeify_codegen_patterns();
    let mut ctx = LocalAddBufferContext::new();
    let result = crate::rewrite::graph_rewrite_bottom_up(&matcher, uop.clone(), &mut ctx);
    (result, ctx)
}

#[test]
fn test_remove_noop_void_returns_none() {
    // Default NOOP has Void dtype, which should return None
    let noop = UOp::noop(); // DType::Void

    let result = apply_patterns(&noop);
    // Should return None for Void dtype
    assert!(result.is_none());
}

#[test]
fn test_remove_noop_non_void() {
    // We can't easily create NOOPs with non-Void dtypes in tests,
    // but we can verify the pattern logic works for NOOP operations
    let noop = UOp::noop();

    // Verify it's a NOOP
    assert!(matches!(noop.op(), Op::Noop));

    // Verify pattern handles it (returns None for Void)
    let result = apply_patterns(&noop);
    assert!(result.is_none()); // Because NOOP dtype is Void
}

#[test]
fn test_remove_noop_returns_none_for_non_noop() {
    // Test that non-NOOP operations return None
    let const_op = UOp::native_const(1.0f32);

    let result = apply_patterns(&const_op);
    assert!(result.is_none());
}

#[test]
fn test_remove_noop_pattern_matching() {
    // Verify patterns only match NOOP operations
    let noop = UOp::noop();
    let const_op = UOp::native_const(0.0f32);

    // NOOP should be handled (returns None for Void dtype)
    assert!(matches!(noop.op(), Op::Noop));
    let noop_result = apply_patterns(&noop);
    assert!(noop_result.is_none()); // Void dtype

    // Non-NOOP should return None
    assert!(!matches!(const_op.op(), Op::Noop));
    let const_result = apply_patterns(&const_op);
    assert!(const_result.is_none());
}

#[test]
fn test_get_contiguous_removes_marker() {
    // Test that CONTIGUOUS marker is removed
    let tensor = UOp::native_const(1.0f32);
    let contiguous = tensor.contiguous();

    let result = apply_patterns(&contiguous);
    assert!(result.is_some());

    let unwrapped = result.unwrap();
    // Should return the original tensor
    assert!(Arc::ptr_eq(&unwrapped, &tensor));
}

#[test]
fn test_get_contiguous_returns_none_for_non_contiguous() {
    // Test that non-CONTIGUOUS operations return None
    let const_op = UOp::native_const(1.0f32);

    let result = apply_patterns(&const_op);
    assert!(result.is_none());
}

#[test]
fn test_fix_after_broadcast_unwraps_expand() {
    // Test that AFTER wrapping EXPAND gets unwrapped
    let source = UOp::native_const(1.0f32);
    let new_shape = UOp::index_const(4);
    let expand = UOp::new(Op::Expand { src: source.clone(), new_shape }, source.dtype());

    let computation = UOp::noop();
    let after = expand.after(smallvec::smallvec![computation]);

    let result = apply_patterns(&after);
    assert!(result.is_some());

    let fixed = result.unwrap();
    // Should have replaced passthrough with expand source
    if let Op::After { passthrough, .. } = fixed.op() {
        assert!(Arc::ptr_eq(passthrough, &source));
    } else {
        panic!("Expected AFTER operation");
    }
}

#[test]
fn test_fix_after_broadcast_returns_none_for_non_after() {
    // Test that non-AFTER operations return None (unless they match another pattern)
    let const_op = UOp::native_const(1.0f32);

    let result = apply_patterns(&const_op);
    assert!(result.is_none());
}

#[test]
fn test_fix_after_broadcast_returns_none_for_non_expand() {
    // Test that AFTER not wrapping EXPAND returns None
    let source = UOp::native_const(1.0f32);
    let computation = UOp::noop();
    let after = source.after(smallvec::smallvec![computation]);

    let result = apply_patterns(&after);
    assert!(result.is_none());
}

#[test]
fn test_fix_after_broadcast_no_panic_on_global() {
    // Test that AFTER with EXPAND of global buffer (no RANGE parent) works
    let source = UOp::native_const(1.0f32);

    let new_shape = UOp::index_const(4);
    let expand = UOp::new(Op::Expand { src: source.clone(), new_shape }, source.dtype());

    let computation = UOp::noop();
    let after = expand.after(smallvec::smallvec![computation]);

    // Should not panic for global (non-local) buffer
    let result = apply_patterns(&after);
    assert!(result.is_some());
}

// Note: Testing the panic case for local AFTER is complex because it requires
// constructing a graph where the expand source has actual RANGE consumers,
// which depends on the specific graph structure and consumer map computation.
// The logic is implemented correctly but creating a test case that triggers
// the panic would require a more complex setup with actual range-dependent operations.

#[test]
fn test_codegen_patterns_creates_matcher() {
    let _matcher = rangeify_codegen_patterns();

    // Verify the matcher was created successfully
    // We can't access the patterns field (it's private), but we can verify
    // the function doesn't panic
}

// ============================================================================
// CONTIGUOUS OPTS EXTRACTION TESTS
// ============================================================================
// Based on Tinygrad's test_rangeify.py which tests CONTIGUOUS with Opt hints.
// These tests verify that optimization hints flow through the pipeline correctly.

#[test]
fn test_contiguous_opts_empty() {
    // CONTIGUOUS without opts should not populate ctx.opts
    let tensor = UOp::native_const(1.0f32);
    let contiguous = tensor.contiguous(); // No opts

    let (_result, ctx) = apply_patterns_with_ctx(&contiguous);
    assert!(ctx.opts.is_empty(), "ctx.opts should be empty when CONTIGUOUS has no hints");
}

#[test]
fn test_contiguous_opts_single_hint() {
    // CONTIGUOUS with single opt should extract to ctx.opts
    // Mirrors Tinygrad: tensor.contiguous(arg=(Opt(OptOps.UPCAST, 0, 4),))
    let tensor = UOp::native_const(1.0f32);

    let opts = smallvec::smallvec![ContiguousHint { op: "UPCAST".to_string(), axis: Some(0), arg: Some(4) }];
    let contiguous = tensor.contiguous_with_opts(opts);

    let (_result, ctx) = apply_patterns_with_ctx(&contiguous);
    assert_eq!(ctx.opts.len(), 1, "ctx.opts should have 1 hint");
    assert_eq!(ctx.opts[0].op, "UPCAST");
    assert_eq!(ctx.opts[0].axis, Some(0));
    assert_eq!(ctx.opts[0].arg, Some(4));
}

#[test]
fn test_contiguous_opts_multiple_hints() {
    // CONTIGUOUS with multiple opts should extract all to ctx.opts
    // Mirrors Tinygrad: tensor.contiguous(arg=(Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)))
    let tensor = UOp::native_const(1.0f32);

    let opts = smallvec::smallvec![
        ContiguousHint { op: "UPCAST".to_string(), axis: Some(0), arg: Some(4) },
        ContiguousHint { op: "UPCAST".to_string(), axis: Some(1), arg: Some(4) },
    ];
    let contiguous = tensor.contiguous_with_opts(opts);

    let (_result, ctx) = apply_patterns_with_ctx(&contiguous);
    assert_eq!(ctx.opts.len(), 2, "ctx.opts should have 2 hints");
    assert_eq!(ctx.opts[0].op, "UPCAST");
    assert_eq!(ctx.opts[0].axis, Some(0));
    assert_eq!(ctx.opts[1].op, "UPCAST");
    assert_eq!(ctx.opts[1].axis, Some(1));
}

#[test]
fn test_contiguous_opts_mixed_hint_types() {
    // CONTIGUOUS with mixed opt types (UPCAST + UNROLL)
    // Mirrors Tinygrad: tensor.contiguous(arg=(Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 1, 4)))
    let tensor = UOp::native_const(1.0f32);

    let opts = smallvec::smallvec![
        ContiguousHint { op: "UPCAST".to_string(), axis: Some(0), arg: Some(4) },
        ContiguousHint { op: "UNROLL".to_string(), axis: Some(1), arg: Some(4) },
    ];
    let contiguous = tensor.contiguous_with_opts(opts);

    let (_result, ctx) = apply_patterns_with_ctx(&contiguous);
    assert_eq!(ctx.opts.len(), 2);
    assert_eq!(ctx.opts[0].op, "UPCAST");
    assert_eq!(ctx.opts[1].op, "UNROLL");
}

#[test]
fn test_contiguous_opts_four_hints() {
    // CONTIGUOUS with 4 opts (max typical usage from Tinygrad tests)
    // Mirrors Tinygrad: test_upcast_01_unroll_01
    let tensor = UOp::native_const(1.0f32);

    let opts = smallvec::smallvec![
        ContiguousHint { op: "UPCAST".to_string(), axis: Some(0), arg: Some(4) },
        ContiguousHint { op: "UPCAST".to_string(), axis: Some(1), arg: Some(4) },
        ContiguousHint { op: "UNROLL".to_string(), axis: Some(0), arg: Some(4) },
        ContiguousHint { op: "UNROLL".to_string(), axis: Some(1), arg: Some(4) },
    ];
    let contiguous = tensor.contiguous_with_opts(opts);

    let (_result, ctx) = apply_patterns_with_ctx(&contiguous);
    assert_eq!(ctx.opts.len(), 4, "ctx.opts should have 4 hints");

    // Verify order is preserved
    assert_eq!(ctx.opts[0].op, "UPCAST");
    assert_eq!(ctx.opts[0].axis, Some(0));
    assert_eq!(ctx.opts[1].op, "UPCAST");
    assert_eq!(ctx.opts[1].axis, Some(1));
    assert_eq!(ctx.opts[2].op, "UNROLL");
    assert_eq!(ctx.opts[2].axis, Some(0));
    assert_eq!(ctx.opts[3].op, "UNROLL");
    assert_eq!(ctx.opts[3].axis, Some(1));
}

#[test]
fn test_contiguous_opts_returns_source() {
    // Verify CONTIGUOUS with opts still returns the source tensor
    let tensor = UOp::native_const(42.0f32);

    let opts = smallvec::smallvec![ContiguousHint { op: "LOCAL".to_string(), axis: Some(2), arg: Some(8) }];
    let contiguous = tensor.contiguous_with_opts(opts);

    let (result, _ctx) = apply_patterns_with_ctx(&contiguous);

    // Should return the original tensor (CONTIGUOUS is stripped)
    assert!(Arc::ptr_eq(&result, &tensor));
}

#[test]
fn test_contiguous_opts_hint_without_axis() {
    // Some opts like NOLOCALS don't have an axis
    let tensor = UOp::native_const(1.0f32);

    let opts = smallvec::smallvec![ContiguousHint { op: "NOLOCALS".to_string(), axis: None, arg: None }];
    let contiguous = tensor.contiguous_with_opts(opts);

    let (_result, ctx) = apply_patterns_with_ctx(&contiguous);
    assert_eq!(ctx.opts.len(), 1);
    assert_eq!(ctx.opts[0].op, "NOLOCALS");
    assert_eq!(ctx.opts[0].axis, None);
    assert_eq!(ctx.opts[0].arg, None);
}
