//! `rangeify_codegen_patterns`: CONTIGUOUS stripping and the hints it carries
//! into `LocalAddBufferContext::opts`.
//!
//! Mirrors tinygrad's `test_rangeify.py` `Tensor.contiguous(arg=(Opt(...),))`.

use std::sync::Arc;

use svod_ir::{ContiguousHint, UOp};
use test_case::test_case;

use crate::rangeify::kernel::LocalAddBufferContext;
use crate::rangeify::patterns::rangeify_codegen_patterns;

fn apply(uop: Arc<UOp>) -> (Arc<UOp>, LocalAddBufferContext) {
    let mut ctx = LocalAddBufferContext::new();
    let result = crate::rewrite::graph_rewrite(&rangeify_codegen_patterns(), uop, &mut ctx);
    (result, ctx)
}

fn hint(op: &str, axis: Option<usize>, arg: Option<i64>) -> ContiguousHint {
    ContiguousHint { op: op.to_string(), axis, arg }
}

/// The CONTIGUOUS marker is a scheduling instruction, not a value: it is
/// stripped and its source returned.
#[test]
fn contiguous_is_stripped_from_the_graph() {
    let tensor = UOp::native_const(42.0f32);
    let opts = vec![hint("LOCAL", Some(2), Some(8))];

    for wrapped in [tensor.contiguous(), tensor.contiguous_with_opts(opts)] {
        assert!(Arc::ptr_eq(&apply(wrapped).0, &tensor));
    }
}

/// A NOOP carries `Void`, and nothing else in the graph is a codegen pattern's
/// business — both are left untouched.
#[test]
fn void_noops_and_plain_values_are_left_alone() {
    for untouched in [UOp::noop(), UOp::native_const(1.0f32)] {
        assert!(Arc::ptr_eq(&apply(untouched.clone()).0, &untouched));
    }
}

fn no_hints() -> Vec<ContiguousHint> {
    Vec::new()
}

fn one_hint() -> Vec<ContiguousHint> {
    vec![hint("UPCAST", Some(0), Some(4))]
}

/// An opt with no axis, e.g. NOLOCALS.
fn axisless_hint() -> Vec<ContiguousHint> {
    vec![hint("NOLOCALS", None, None)]
}

fn mixed_hints() -> Vec<ContiguousHint> {
    vec![hint("UPCAST", Some(0), Some(4)), hint("UNROLL", Some(1), Some(4))]
}

/// tinygrad's `test_upcast_01_unroll_01`.
fn four_hints() -> Vec<ContiguousHint> {
    vec![
        hint("UPCAST", Some(0), Some(4)),
        hint("UPCAST", Some(1), Some(4)),
        hint("UNROLL", Some(0), Some(4)),
        hint("UNROLL", Some(1), Some(4)),
    ]
}

/// Every hint reaches `ctx.opts` verbatim and in order.
#[test_case(super::no_hints ; "no hints")]
#[test_case(super::one_hint ; "one hint")]
#[test_case(super::axisless_hint ; "hint without an axis")]
#[test_case(super::mixed_hints ; "upcast and unroll")]
#[test_case(super::four_hints ; "four hints")]
fn contiguous_hints_are_collected_in_order(build: fn() -> Vec<ContiguousHint>) {
    let hints = build();
    let (_result, ctx) = apply(UOp::native_const(1.0f32).contiguous_with_opts(hints.clone()));
    assert_eq!(ctx.opts.as_slice(), hints.as_slice());
}
