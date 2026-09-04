//! Property tests for the symbolic optimizer's algebraic rules: rewrites must preserve
//! semantics and obey the laws they claim.

use std::collections::HashMap;
use std::sync::Arc;

use proptest::prelude::*;

use svod_dtype::{DType, ScalarDType};
use svod_ir::types::{BinaryOp, ConstValue};
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::eval::{eval_binary_op, eval_binary_op_typed, eval_unary_op_typed};
use svod_ir::uop::properties::VminVmaxProperty;
use svod_ir::{Op, UOp};

use crate::rewrite::graph_rewrite;
use crate::symbolic::{symbolic, symbolic_simple};

use svod_ir::ops;
use svod_ir::test::property::generators::*;

fn apply(op: BinaryOp, lhs: &Arc<UOp>, rhs: &Arc<UOp>) -> Arc<UOp> {
    match op {
        BinaryOp::Add => lhs.try_add(rhs),
        BinaryOp::Sub => lhs.try_sub(rhs),
        BinaryOp::Mul => lhs.try_mul(rhs),
        BinaryOp::FloorDiv => lhs.try_div(rhs),
        BinaryOp::And => lhs.try_and_op(rhs),
        BinaryOp::Or => lhs.try_or_op(rhs),
        BinaryOp::Xor => lhs.try_xor_op(rhs),
        other => panic!("{other:?} has no constructor here"),
    }
    .unwrap()
}

/// `(op, identity, also_folds_on_the_left)`: `x op identity == x`.
const IDENTITIES: &[(BinaryOp, i32, bool)] = &[
    (BinaryOp::Add, 0, true),
    (BinaryOp::Sub, 0, false),
    (BinaryOp::Mul, 1, true),
    (BinaryOp::FloorDiv, 1, false),
    (BinaryOp::Or, 0, false),
    (BinaryOp::Xor, 0, false),
];

/// `(op, absorbing element)`: `x op absorbing == absorbing`, either way round.
const ABSORBING: &[(BinaryOp, i32)] = &[(BinaryOp::Mul, 0), (BinaryOp::And, 0)];

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn identity_operand_folds_away(x in arb_simple_uop(DType::Int32)) {
        for &(op, identity, folds_on_the_left) in IDENTITIES {
            let identity = UOp::native_const(identity);
            let right = graph_rewrite(symbolic_simple(), apply(op, &x, &identity), &mut ());
            prop_assert!(Arc::ptr_eq(&right, &x), "x {op:?} identity should be x, got {:?}", right.op());

            if folds_on_the_left {
                let left = graph_rewrite(symbolic_simple(), apply(op, &identity, &x), &mut ());
                prop_assert!(Arc::ptr_eq(&left, &x), "identity {op:?} x should be x, got {:?}", left.op());
            }
        }
    }

    #[test]
    fn absorbing_operand_swallows_the_expression(x in arb_simple_uop(DType::Int32)) {
        for &(op, value) in ABSORBING {
            let absorbing = UOp::native_const(value);
            for expr in [apply(op, &x, &absorbing), apply(op, &absorbing, &x)] {
                let simplified = graph_rewrite(symbolic_simple(), expr, &mut ());
                prop_assert!(Arc::ptr_eq(&simplified, &absorbing), "{op:?} should fold to {value}");
            }
        }
    }

    #[test]
    fn bitwise_self_operation_is_idempotent(x in arb_simple_uop(DType::Int32)) {
        for op in [BinaryOp::And, BinaryOp::Or] {
            let simplified = graph_rewrite(symbolic_simple(), apply(op, &x, &x), &mut ());
            prop_assert!(Arc::ptr_eq(&simplified, &x), "x {op:?} x should be x, got {:?}", simplified.op());
        }
    }

    /// Integer self comparisons are decided without looking at the value.
    #[test]
    fn self_comparison_folds_to_a_constant(x in arb_var_uop(DType::Int32)) {
        for (expr, expected) in [
            (x.try_cmplt(&x).unwrap(), false),
            (x.try_cmpne(&x).unwrap(), false),
            (x.try_cmpeq(&x).unwrap(), true),
        ] {
            let simplified = graph_rewrite(symbolic(), expr, &mut ());
            prop_assert!(
                matches!(simplified.op(), Op::Const(value) if value.0 == ConstValue::Bool(expected)),
                "expected Const({expected}), got {:?}", simplified.op()
            );
        }
    }

    /// `x / x` folds to 1 only when the declared range excludes zero.
    #[test]
    fn self_division_folds_only_away_from_zero(min in 0i64..20, span in 0i64..20) {
        let x = UOp::var("x", DType::Int32, min, min + span);
        let simplified = graph_rewrite(symbolic_simple(), x.try_div(&x).unwrap(), &mut ());

        if min > 0 {
            prop_assert!(
                matches!(simplified.op(), Op::Const(value) if value.0 == ConstValue::Int(1)),
                "nonzero x / x should be 1, got {:?}", simplified.op()
            );
        } else {
            prop_assert!(matches!(simplified.op(), Op::Binary(BinaryOp::FloorDiv, ..)), "got {:?}", simplified.op());
        }
    }

    /// A binary op over two constants folds to the value the evaluator computes.
    #[test]
    fn constant_operands_fold_to_the_evaluated_value(
        a in arb_small_int(),
        b in arb_small_int(),
        divisor in nonzero_int(),
    ) {
        let konst = |value| UOp::const_(DType::Int32, value);
        for (op, rhs) in [(BinaryOp::Add, b), (BinaryOp::Mul, b), (BinaryOp::FloorDiv, divisor)] {
            let simplified = graph_rewrite(symbolic_simple(), apply(op, &konst(a), &konst(rhs)), &mut ());
            match simplified.op() {
                Op::Const(folded) => {
                    // The value is only comparable when both operands are really Int32:
                    // `nonzero_int` also produces divisors that overflow the dtype.
                    let in_range = |v| matches!(v, ConstValue::Int(value) if i32::try_from(value).is_ok());
                    if in_range(a) && in_range(rhs) {
                        let expected = eval_binary_op_typed(op, a, rhs, ScalarDType::Int32);
                        prop_assert_eq!(Some(folded.0), expected, "{:?} {:?} {:?}", a, op, rhs);
                    }
                }
                other => prop_assert!(false, "{op:?} over constants did not fold: {other:?}"),
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// `(a // b) // c` collapses to `a // (b * c)`. Divisors stay small so that `a`'s range
    /// does not let range analysis fold the intermediate division to zero first.
    #[test]
    fn nested_div_collapse(a in arb_var_uop(DType::Int32), b in 2..8i32, c in 2..8i32) {
        let (_, vmax) = VminVmaxProperty::get(&a);
        if let ConstValue::Int(max) = vmax {
            prop_assume!(*max >= (b as i64) * (c as i64));
        }

        let div = a.try_div(&UOp::native_const(b)).unwrap().try_div(&UOp::native_const(c)).unwrap();
        let simplified = graph_rewrite(symbolic(), div, &mut ());

        match simplified.op() {
            Op::Binary(BinaryOp::FloorDiv, var, divisor) => {
                prop_assert!(Arc::ptr_eq(var, &a));
                prop_assert!(matches!(divisor.op(), Op::Const(v) if v.0 == ConstValue::Int((b as i64) * (c as i64))));
            }
            other => prop_assert!(false, "expected a single FloorDiv, got {other:?}"),
        }
    }

    /// `(a * b) * c` collapses to `a * (b * c)`.
    #[test]
    fn nested_mul_collapse(a in arb_var_uop(DType::Int32), b in 2..20i32, c in 2..20i32) {
        let mul = a.try_mul(&UOp::native_const(b)).unwrap().try_mul(&UOp::native_const(c)).unwrap();
        let simplified = graph_rewrite(symbolic(), mul, &mut ());

        match simplified.op() {
            Op::Binary(BinaryOp::Mul, var, factor) => {
                prop_assert!(Arc::ptr_eq(var, &a));
                prop_assert!(matches!(factor.op(), Op::Const(v) if v.0 == ConstValue::Int((b as i64) * (c as i64))));
            }
            other => prop_assert!(false, "expected a single Mul, got {other:?}"),
        }
    }

    /// `(a % b) % b` collapses to `a % b`. Skipped when `a.max < b`, where range analysis
    /// folds the modulo to `a` before the idempotence rule can fire.
    #[test]
    fn mod_idempotence(a in arb_var_uop(DType::Int32), b in 2..100i32) {
        let (_, vmax) = VminVmaxProperty::get(&a);
        if let ConstValue::Int(max) = vmax {
            prop_assume!(*max >= b as i64);
        }

        let divisor = UOp::native_const(b);
        let nested = a.try_mod(&divisor).unwrap().try_mod(&divisor).unwrap();
        let simplified = graph_rewrite(symbolic_simple(), nested, &mut ());

        match simplified.op() {
            Op::Binary(BinaryOp::FloorMod, var, actual) => {
                prop_assert!(Arc::ptr_eq(var, &a));
                prop_assert!(Arc::ptr_eq(actual, &divisor));
            }
            other => prop_assert!(false, "expected FloorMod(a, b), got {other:?}"),
        }
    }

    /// `(a + b) + c` collapses to `a + (b + c)`, or to `a` alone when the sum cancels.
    #[test]
    fn nested_add_collapse(a in arb_var_uop(DType::Int32), b in -100..100i32, c in -100..100i32) {
        let add = a.try_add(&UOp::native_const(b)).unwrap().try_add(&UOp::native_const(c)).unwrap();
        let simplified = graph_rewrite(symbolic(), add, &mut ());

        let sum = (b as i64) + (c as i64);
        match simplified.op() {
            Op::Binary(BinaryOp::Add, var, addend) => {
                prop_assert!(Arc::ptr_eq(var, &a));
                prop_assert!(matches!(addend.op(), Op::Const(v) if v.0 == ConstValue::Int(sum)));
            }
            Op::Binary(BinaryOp::Sub, var, subtrahend) => {
                prop_assert!(Arc::ptr_eq(var, &a));
                prop_assert!(matches!(subtrahend.op(), Op::Const(v) if v.0 == ConstValue::Int(-sum)));
            }
            Op::DefineVar(..) => {
                prop_assert!(Arc::ptr_eq(&simplified, &a));
                prop_assert_eq!(sum, 0, "a bare variable may only come out when the constants cancel");
            }
            other => prop_assert!(false, "expected Add, Sub or the bare variable, got {other:?}"),
        }
    }

    /// `(a - b) - c` collapses to tinygrad's subtraction form, `a + -(b + c)`.
    #[test]
    fn nested_sub_collapse(a in arb_var_uop(DType::Int32), b in 1..100i32, c in 1..100i32) {
        let sub = a.try_sub(&UOp::native_const(b)).unwrap().try_sub(&UOp::native_const(c)).unwrap();
        let simplified = graph_rewrite(symbolic(), sub, &mut ());

        match simplified.op() {
            Op::Binary(BinaryOp::Add, var, addend) => {
                prop_assert!(Arc::ptr_eq(var, &a));
                let expected = -((b as i64) + (c as i64));
                prop_assert!(matches!(addend.op(), Op::Const(v) if v.0 == ConstValue::Int(expected)));
            }
            other => prop_assert!(false, "expected Add with a negative constant, got {other:?}"),
        }
    }

    /// `(a * b) // b` cancels back to `a`.
    #[test]
    fn mul_div_inverse(a in arb_var_uop(DType::Int32), b in 1..100i32) {
        let b = UOp::native_const(b);
        let simplified = graph_rewrite(symbolic_simple(), a.try_mul(&b).unwrap().try_div(&b).unwrap(), &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &a), "got {}", simplified.tree());
    }
}

/// Evaluate an expression tree with variables bound to concrete values. Returns `None` for
/// ops the evaluator does not cover.
fn eval_uop(expr: &Arc<UOp>, vars: &HashMap<String, i64>) -> Option<i64> {
    match expr.op() {
        Op::Const(value) => match value.0 {
            ConstValue::Invalid => Some(i64::MIN),
            ConstValue::Int(v) => Some(v),
            _ => None,
        },
        Op::DefineVar(ops::DefineVar { name, .. }) => vars.get(name.as_str()).copied(),
        Op::Bind(ops::Bind { var, .. }) => eval_uop(var, vars),
        Op::Binary(op, a, b) => {
            let (a, b) = (eval_uop(a, vars)?, eval_uop(b, vars)?);
            match eval_binary_op(*op, ConstValue::Int(a), ConstValue::Int(b))? {
                ConstValue::Int(v) => Some(v),
                ConstValue::Bool(v) => Some(v as i64),
                _ => None,
            }
        }
        Op::Ternary(svod_ir::TernaryOp::Where, cond, t, f) => {
            if eval_uop(cond, vars)? != 0 {
                eval_uop(t, vars)
            } else {
                eval_uop(f, vars)
            }
        }
        Op::Unary(svod_ir::UnaryOp::Not, x) => Some((eval_uop(x, vars)? == 0) as i64),
        _ => None,
    }
}

/// Evaluate at the dtype's own width, so wrapping and narrowing are observable.
fn eval_typed_uop(expr: &Arc<UOp>, variable: &str, value: ConstValue) -> Option<ConstValue> {
    match expr.op() {
        Op::Const(constant) => Some(constant.0),
        Op::DefineVar(ops::DefineVar { name, .. }) if name == variable => Some(value),
        Op::Bind(ops::Bind { var, .. }) => eval_typed_uop(var, variable, value),
        Op::Binary(op, lhs, rhs) => eval_binary_op_typed(
            *op,
            eval_typed_uop(lhs, variable, value)?,
            eval_typed_uop(rhs, variable, value)?,
            expr.dtype().base(),
        ),
        Op::Unary(op, src) => eval_unary_op_typed(*op, eval_typed_uop(src, variable, value)?, expr.dtype().base()),
        _ => None,
    }
}

/// `(a*f + b*g + const)` expressions, with the divisor to apply and the variable ranges.
fn arb_divmod_expr() -> impl Strategy<Value = (Arc<UOp>, i64, Vec<(String, i64, i64)>)> {
    (-20i64..20, -20i64..20, -20i64..20, 2i64..16, 1i64..8, 1i64..8).prop_map(
        |(factor_a, factor_b, offset, divisor, a_max, b_max)| {
            let a = UOp::variable("a".into(), 0, a_max, DType::Int32);
            let b = UOp::variable("b".into(), 0, b_max, DType::Int32);
            let mut expr = UOp::index_const(offset);
            if factor_a != 0 {
                expr = expr.try_add(&UOp::index_const(factor_a).try_mul(&a).unwrap()).unwrap();
            }
            if factor_b != 0 {
                expr = expr.try_add(&UOp::index_const(factor_b).try_mul(&b).unwrap()).unwrap();
            }
            (expr, divisor, vec![("a".into(), 0, a_max), ("b".into(), 0, b_max)])
        },
    )
}

/// The simplification must evaluate identically to the original at every point of the
/// declared variable ranges.
fn prop_assert_same_over_ranges(
    original: &Arc<UOp>,
    simplified: &Arc<UOp>,
    var_ranges: &[(String, i64, i64)],
) -> Result<(), TestCaseError> {
    for a in var_ranges[0].1..=var_ranges[0].2 {
        for b in var_ranges[1].1..=var_ranges[1].2 {
            let vars = HashMap::from([("a".to_string(), a), ("b".to_string(), b)]);
            if let (Some(before), Some(after)) = (eval_uop(original, &vars), eval_uop(simplified, &vars)) {
                prop_assert_eq!(
                    before,
                    after,
                    "mismatch at a={}, b={}.\n  original: {}\n  simplified: {}",
                    a,
                    b,
                    original.tree(),
                    simplified.tree()
                );
            }
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn divmod_mod_soundness((expr, divisor, var_ranges) in arb_divmod_expr()) {
        let modded = expr.try_mod(&UOp::index_const(divisor)).unwrap();
        let simplified = graph_rewrite(symbolic_simple(), modded.clone(), &mut ());
        prop_assert_same_over_ranges(&modded, &simplified, &var_ranges)?;
    }

    #[test]
    fn divmod_idiv_soundness((expr, divisor, var_ranges) in arb_divmod_expr()) {
        let divided = expr.try_div(&UOp::index_const(divisor)).unwrap();
        let simplified = graph_rewrite(symbolic_simple(), divided.clone(), &mut ());
        prop_assert_same_over_ranges(&divided, &simplified, &var_ranges)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// The affine congruence rules must fire and must not change the value at the narrow
    /// dtype they were derived at.
    #[test]
    fn affine_congruence_rewrites_preserve_exact_typed_runtime(
        divisor in 8i64..=16,
        offset in 0i64..=2,
        max in 2i64..=5,
    ) {
        let dtype = DType::Int16;
        let x = UOp::var("affine_x", dtype.clone(), 0, max);
        let divisor_uop = UOp::const_(dtype.clone(), ConstValue::Int(divisor));
        let numerator = x
            .mul(&UOp::const_(dtype.clone(), ConstValue::Int(divisor + 1)))
            .add(&UOp::const_(dtype, ConstValue::Int(offset)));

        for original in [numerator.mod_(&divisor_uop), numerator.floor_div(&divisor_uop)] {
            let rewritten = graph_rewrite(symbolic(), original.clone(), &mut ());
            prop_assert!(!Arc::ptr_eq(&original, &rewritten), "congruence rule did not fire for {}", original.tree());
            for value in 0..=max {
                prop_assert_eq!(
                    eval_typed_uop(&original, "affine_x", ConstValue::Int(value)),
                    eval_typed_uop(&rewritten, "affine_x", ConstValue::Int(value)),
                    "typed affine rewrite mismatch for original {} and replacement {}",
                    original.tree(),
                    rewritten.tree(),
                );
            }
        }
    }

    /// 8-bit divmod rewrites must reproduce the wrapping runtime result, signed or not.
    #[test]
    fn typed_int8_divmod_rewrites_preserve_wrapping_runtime(
        unsigned in any::<bool>(),
        raw_x in any::<u8>(),
        factor in 1u8..=8,
        offset in 0u8..=8,
    ) {
        let (dtype, x_value, factor_value, offset_value, min, max) = if unsigned {
            (
                DType::UInt8,
                ConstValue::UInt(raw_x as u64),
                ConstValue::UInt(factor as u64),
                ConstValue::UInt(offset as u64),
                0,
                u8::MAX as i64,
            )
        } else {
            (
                DType::Int8,
                ConstValue::Int((raw_x as i8) as i64),
                ConstValue::Int(factor as i64),
                ConstValue::Int(offset as i64),
                i8::MIN as i64,
                i8::MAX as i64,
            )
        };
        let x = UOp::var("typed_x", dtype.clone(), min, max);
        let divisor = UOp::const_(dtype.clone(), factor_value);
        let expression = x.mul(&divisor).add(&UOp::const_(dtype, offset_value));

        for original in [expression.floor_div(&divisor), expression.mod_(&divisor)] {
            let rewritten = graph_rewrite(symbolic(), original.clone(), &mut ());
            prop_assert_eq!(
                eval_typed_uop(&original, "typed_x", x_value),
                eval_typed_uop(&rewritten, "typed_x", x_value),
                "typed rewrite mismatch for original {} and replacement {}",
                original.tree(),
                rewritten.tree(),
            );
        }
    }
}
