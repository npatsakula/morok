//! Late decomposition patterns — tinygrad `decompositions.py:321-367`.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::types::ConstValue;
use svod_ir::{BinaryOp, Op, UOp, UnaryOp};
use test_case::test_case;

use crate::rangeify::patterns::{
    pm_comparison_negations, pm_div_to_shr, pm_fdiv_to_mul, pm_mod_to_and, pm_mul_to_shl, pm_neg_from_mul,
};
use crate::rewrite::graph_rewrite;
use crate::symbolic::{pm_fold_cast_const, symbolic_simple};
use svod_ir::ops;

fn x() -> Arc<UOp> {
    UOp::variable("x".into(), 0, 9999, DType::Int32)
}

fn late_rewrite(matcher: &'static crate::TypedPatternMatcher, root: Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(&(symbolic_simple() + pm_fold_cast_const() + matcher), root, &mut ())
}

/// Assert `result` is `op(x, rhs)`.
fn assert_strength_reduced(result: &Arc<UOp>, op: BinaryOp, rhs: i64) {
    let Op::Binary(actual, lhs, actual_rhs) = result.op() else { panic!("expected {op:?}, got {}", result.tree()) };
    assert_eq!(*actual, op, "{}", result.tree());
    assert!(Arc::ptr_eq(lhs, &x()), "LHS must be the original operand");
    assert!(matches!(actual_rhs.op(), Op::Const(c) if c.0 == ConstValue::Int(rhs)), "{}", result.tree());
}

/// Power-of-two `%`, `*` and `//` become bit ops on the same operand.
#[test_case(2, 1 ; "modulo 2")]
#[test_case(8, 7 ; "modulo 8")]
#[test_case(1024, 1023 ; "modulo 1024")]
fn modulo_by_a_power_of_two_becomes_a_mask(divisor: i64, mask: i64) {
    assert_strength_reduced(&late_rewrite(pm_mod_to_and(), x().mod_(&UOp::index_const(divisor))), BinaryOp::And, mask);
}

#[test_case(2, 1 ; "times 2")]
#[test_case(8, 3 ; "times 8")]
#[test_case(256, 8 ; "times 256")]
fn multiply_by_a_power_of_two_becomes_a_left_shift(factor: i64, shift: i64) {
    assert_strength_reduced(&late_rewrite(pm_mul_to_shl(), x().mul(&UOp::index_const(factor))), BinaryOp::Shl, shift);
}

#[test_case(2, 1 ; "over 2")]
#[test_case(8, 3 ; "over 8")]
#[test_case(256, 8 ; "over 256")]
fn divide_by_a_power_of_two_becomes_a_right_shift(divisor: i64, shift: i64) {
    assert_strength_reduced(&late_rewrite(pm_div_to_shr(), x().cdiv(&UOp::index_const(divisor))), BinaryOp::Shr, shift);
}

/// Non-powers of two, and the trivial identities, are left for the backend.
#[test_case(pm_mod_to_and, |c| x().mod_(&c), 7, BinaryOp::FloorMod ; "modulo 7")]
#[test_case(pm_mul_to_shl, |c| x().mul(&c), 7, BinaryOp::Mul ; "times 7")]
#[test_case(pm_div_to_shr, |c| x().cdiv(&c), 7, BinaryOp::CDiv ; "over 7")]
#[test_case(pm_div_to_shr, |c| x().cdiv(&c), 1, BinaryOp::CDiv ; "over 1 is not a zero shift")]
#[test_case(pm_neg_from_mul, |c| x().mul(&c), 1, BinaryOp::Mul ; "times positive one is not a negation")]
fn no_strength_reduction_without_a_power_of_two(
    matcher: fn() -> &'static crate::TypedPatternMatcher,
    build: fn(Arc<UOp>) -> Arc<UOp>,
    operand: i64,
    expected: BinaryOp,
) {
    let root = build(UOp::index_const(operand));
    let result = graph_rewrite(matcher(), root, &mut ());
    assert!(matches!(result.op(), Op::Binary(op, _, _) if *op == expected), "{}", result.tree());
}

#[test]
fn multiply_by_one_is_the_identity() {
    let result = late_rewrite(pm_mul_to_shl(), x().mul(&UOp::index_const(1)));
    assert!(Arc::ptr_eq(&result, &x()));
}

/// `x * -1 → NEG(x)`: the codegen-facing form of the canonical MUL.
#[test]
fn multiply_by_minus_one_becomes_a_negation() {
    let result = late_rewrite(pm_neg_from_mul(), x().mul(&UOp::index_const(-1)));
    let Op::Unary(UnaryOp::Neg, inner) = result.op() else { panic!("expected NEG, got {}", result.tree()) };
    assert!(Arc::ptr_eq(inner, &x()));
}

// ===== FDIV → MUL by reciprocal (decompositions.py:364-366) =====

#[test_case(2.0, 0.5 ; "half")]
#[test_case(4.0, 0.25 ; "quarter")]
#[test_case(5.0, 0.2 ; "fifth")]
#[test_case(0.5, 2.0 ; "reciprocal below one")]
fn dividing_by_a_float_constant_becomes_a_reciprocal_multiply(divisor: f32, reciprocal: f32) {
    let div = UOp::native_const(100.0f32).try_div(&UOp::native_const(divisor)).expect("div");
    let result = graph_rewrite(pm_fdiv_to_mul(), div, &mut ());

    let Op::Binary(BinaryOp::Mul, _, rhs) = result.op() else { panic!("expected MUL, got {}", result.tree()) };
    let Op::Const(c) = rhs.op() else { panic!("expected a constant reciprocal, got {}", result.tree()) };
    let ConstValue::Float(f) = c.0 else { panic!("expected a float reciprocal, got {:?}", c.0) };
    assert!((f - reciprocal as f64).abs() < 1e-6, "expected {reciprocal}, got {f}");
}

#[test]
fn dividing_by_zero_is_rejected_at_construction() {
    assert!(UOp::native_const(10.0f32).try_div(&UOp::native_const(0.0f32)).is_err());
}

// ===== comparison negations (decompositions.py:354-361) =====

/// Negating an integer `<` flips it to the complementary `<` with a shifted
/// bound; a two-sided band collapses to an equality.
#[test]
fn negated_integer_comparisons_become_the_complementary_bound() {
    let five = UOp::index_const(5);

    let not_lt = late_rewrite(pm_comparison_negations(), x().try_cmplt(&five).expect("cmplt").not());
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = not_lt.op() else { panic!("expected LT, got {}", not_lt.tree()) };
    assert!(matches!(lhs.op(), Op::Const(c) if c.0 == ConstValue::Int(4)), "!(x < 5) is 4 < x");
    assert!(Arc::ptr_eq(rhs, &x()));

    let not_gt = late_rewrite(pm_comparison_negations(), five.try_cmplt(&x()).expect("cmplt").not());
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = not_gt.op() else { panic!("expected LT, got {}", not_gt.tree()) };
    assert!(Arc::ptr_eq(lhs, &x()));
    assert!(matches!(rhs.op(), Op::Const(c) if c.0 == ConstValue::Int(6)), "!(5 < x) is x < 6");
}

#[test]
fn a_one_wide_band_collapses_to_an_equality() {
    let above = UOp::index_const(3).try_cmplt(&x()).expect("cmplt");
    let below = x().try_cmplt(&UOp::index_const(5)).expect("cmplt");

    let result = late_rewrite(pm_comparison_negations(), above.try_and_op(&below).expect("and"));

    let Op::Binary(BinaryOp::Eq, lhs, rhs) = result.op() else { panic!("expected EQ, got {}", result.tree()) };
    let (var, konst) = if matches!(lhs.op(), Op::Const(_)) { (rhs, lhs) } else { (lhs, rhs) };
    assert!(Arc::ptr_eq(var, &x()));
    assert!(matches!(konst.op(), Op::Const(c) if c.0 == ConstValue::Int(4)));
}

/// `x * -1 < 5` moves the negation onto the bound: `-5 < x`.
#[test]
fn a_negated_operand_moves_the_bound_instead() {
    let lt = x().mul(&UOp::index_const(-1)).try_cmplt(&UOp::index_const(5)).expect("cmplt");
    let result = late_rewrite(pm_comparison_negations(), lt);

    let Op::Binary(BinaryOp::Lt, lhs, rhs) = result.op() else { panic!("expected LT, got {}", result.tree()) };
    assert!(matches!(lhs.op(), Op::Const(c) if c.0 == ConstValue::Int(-5)), "{}", result.tree());
    assert!(Arc::ptr_eq(rhs, &x()));
}

// ===== renderer-gated late rewrites =====

/// Fast integer division for a non-power-of-two divisor is opt-in; the
/// power-of-two shift is not gated.
#[test]
fn fast_integer_division_is_explicitly_opt_in() {
    let renderer = crate::optimizer::Renderer::cpu().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
    let division = x().cdiv(&UOp::native_const(7i32));
    let modulo = x().cmod(&UOp::native_const(7i32));

    let disabled =
        crate::optimizer::apply_late_rewrites(UOp::sink(vec![division.clone(), modulo.clone()]), &renderer, true);
    assert!(disabled.toposort().iter().any(|node| matches!(node.op(), Op::Binary(BinaryOp::CDiv, ..))));
    assert!(disabled.toposort().iter().any(|node| matches!(node.op(), Op::Binary(BinaryOp::CMod, ..))));

    let enabled = crate::optimizer::apply_late_rewrites(UOp::sink(vec![division, modulo]), &renderer, false);
    assert!(
        !enabled.toposort().iter().any(|node| matches!(node.op(), Op::Binary(BinaryOp::CDiv | BinaryOp::CMod, ..))),
        "{}",
        enabled.tree()
    );

    let power_of_two = crate::optimizer::apply_late_rewrites(x().cdiv(&UOp::native_const(8i32)), &renderer, true);
    assert!(matches!(power_of_two.op(), Op::Binary(BinaryOp::Shr, ..)), "{}", power_of_two.tree());
}

/// Weak lanes must be concretised before the final render, or the renderer mints
/// weak scalar constants it cannot type.
#[test]
fn weak_lowering_concretizes_a_weak_vconst_before_the_final_rewrite() {
    let lanes = UOp::vconst((0..4).map(ConstValue::Int).collect(), DType::WeakInt);

    let lowered = graph_rewrite(
        &crate::symbolic::pm_lower_index_dtype(),
        UOp::sink(vec![lanes]),
        &mut crate::symbolic::WeakMemo::default(),
    );
    let result = graph_rewrite(crate::optimizer::final_rewrite_patterns(), lowered, &mut ());

    assert!(result.toposort().iter().all(|u| !u.dtype().is_weak()), "{}", result.tree());
    let Op::Sink(ops::Sink { sources, .. }) = result.op() else { panic!("expected SINK") };
    assert!(matches!(sources[0].op(), Op::VConst(ops::VConst { values }) if values.len() == 4));
    assert_eq!(sources[0].dtype(), DType::Int32.vec(4).expect("vector dtype"));
}
