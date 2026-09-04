use super::*;

use svod_ir::ops;
use test_case::test_case;

/// `magic_unsigned(max, d)` returns `(m, s)` with `(x * m) >> s == x / d` for every
/// `x` in `0..=max`. The larger maxima are the ones a power-of-two factorization
/// leaves behind (`x / 6` becomes `(x >> 1) / 3`, `x / 12` becomes `(x >> 2) / 3`).
#[test_case(100, 3; "small max")]
#[test_case(500, 3; "max left by factoring out 2")]
#[test_case(1000, 7; "odd divisor")]
#[test_case(10000, 10; "even divisor with a wide range")]
fn magic_unsigned_reproduces_integer_division(max: i64, divisor: i64) {
    let (m, s) = magic_unsigned(max, divisor).expect("a magic number exists");
    for x in 0..=max {
        assert_eq!(x / divisor, ((x as i128 * m as i128) >> s) as i64, "{x} / {divisor}");
    }
}

#[test]
fn magic_unsigned_rejects_non_positive_divisors() {
    assert!(magic_unsigned(100, 0).is_none());
    assert!(magic_unsigned(100, -5).is_none());
}

#[test]
fn fast_division_does_not_rewrite_signed_negative_range() {
    let x = UOp::variable("x".into(), -100, 100, svod_ir::DType::Int32);
    let divisor = UOp::const_(svod_ir::DType::Int32, ConstValue::Int(7));
    let div = x.cdiv(&divisor);

    assert!(matches!(
        fast_division_patterns(std::collections::HashSet::new()).rewrite(&div, &mut ()),
        svod_ir::RewriteResult::NoMatch
    ));
}

#[test]
fn fast_division_replacements_are_exhaustive_for_eight_bit_ranges() {
    use std::collections::{HashMap, HashSet};
    use svod_ir::{Op, UOpKey};

    for (dtype, vmax, wider) in [
        (svod_ir::DType::UInt8, u8::MAX as i64, ScalarDType::UInt16),
        (svod_ir::DType::Int8, i8::MAX as i64, ScalarDType::Int16),
    ] {
        let variable = UOp::variable("x".into(), 0, vmax, dtype.clone());
        let supported = HashSet::from([dtype.base(), wider]);
        for divisor in (2..=vmax).filter(|divisor| !(*divisor as u64).is_power_of_two()) {
            let Some(replacement) = fast_idiv(&variable, divisor, false, &supported) else { continue };
            for value in 0..=vmax {
                let substituted = replacement.substitute(&HashMap::from([(
                    UOpKey(variable.clone()),
                    UOp::const_(dtype.clone(), ConstValue::Int(value)),
                )]));
                let folded = svod_ir::rewrite::graph_rewrite(
                    &(crate::symbolic::symbolic() + crate::symbolic::pm_fold_cast_const()),
                    substituted,
                    &mut (),
                );
                let Op::Const(actual) = folded.op() else {
                    panic!("replacement did not fold for {dtype:?} {value}/{divisor}: {}", folded.tree())
                };
                assert_eq!(actual.0.try_int(), Some(value / divisor), "{dtype:?} {value}/{divisor}");
            }
        }
    }
}

#[test_case(8, 64; "byte periods")]
#[test_case(16, 16; "fixed period")]
#[test_case(64, 4096; "wide periods")]
fn symbolic_divisor_factors_out_of_an_affine_numerator(period_min: i64, period_max: i64) {
    // (N*i + j) // N -> i and (N*i + j) % N -> j for a symbolic N, with j in one period.
    let n = UOp::var("n", svod_ir::DType::Index, period_min, period_max);
    let i = UOp::var("i", svod_ir::DType::Index, 0, 7);
    let j = UOp::var("j", svod_ir::DType::Index, 0, period_min - 1);
    let numerator = n.try_mul(&i).unwrap().try_add(&j).unwrap();

    let quotient = svod_ir::rewrite::graph_rewrite(crate::symbolic::symbolic(), numerator.floor_div(&n), &mut ());
    assert!(std::sync::Arc::ptr_eq(&quotient, &i), "expected i, got {}", quotient.tree());

    let remainder = svod_ir::rewrite::graph_rewrite(crate::symbolic::symbolic(), numerator.mod_(&n), &mut ());
    assert!(std::sync::Arc::ptr_eq(&remainder, &j), "expected j, got {}", remainder.tree());
}

/// One expression per rule ported from tinygrad's `uop/divandmod.py`, with the
/// numerator, the divisor and the folded form taken verbatim from tinygrad's
/// `test/null/test_uop_symbolic.py`.
struct DivmodCase {
    vars: Vec<Arc<UOp>>,
    expr: Arc<UOp>,
    expected: Arc<UOp>,
}

fn divmod_case(case: &str) -> DivmodCase {
    let var = |name: &str, min, max| UOp::var(name, svod_ir::DType::Int32, min, max);
    let c = |value: i64| UOp::const_(svod_ir::DType::Int32, ConstValue::Int(value));
    let build = |vars: Vec<Arc<UOp>>, expr: Arc<UOp>, expected: Arc<UOp>| DivmodCase { vars, expr, expected };

    match case {
        // remove_nested_mod (divandmod.py:29-36), test_mod_mod.
        "remove_nested_mod" => {
            let a = var("a", 0, 31);
            build(vec![a.clone()], a.mod_(&c(12)).mod_(&c(4)), a.mod_(&c(4)))
        }
        "remove_nested_mod_to_zero" => {
            let a = var("a", 0, 31);
            build(vec![a.clone()], a.mul(&c(4)).mod_(&c(12)).mod_(&c(4)), c(0))
        }
        // nested_div (divandmod.py:26), test_mod_div_reorder / test_div_into_mod.
        "nested_div_six_over_three" => {
            let x = var("x", 0, 23);
            build(vec![x.clone()], x.mod_(&c(6)).floor_div(&c(3)), x.floor_div(&c(3)).mod_(&c(2)))
        }
        "nested_div_twelve_over_four" => {
            let x = var("x", 0, 23);
            build(vec![x.clone()], x.mod_(&c(12)).floor_div(&c(4)), x.floor_div(&c(4)).mod_(&c(3)))
        }
        "nested_div_into_mod" => {
            let idx = var("idx", 0, 16);
            build(vec![idx.clone()], idx.mul(&c(4)).mod_(&c(8)).floor_div(&c(4)), idx.mod_(&c(2)))
        }
        // gcd_with_remainder (divandmod.py:50-55), test_gcd_with_remainder.
        "gcd_with_remainder_div" => {
            let a = var("a", 0, 2);
            build(vec![a.clone()], a.mul(&c(4)).floor_div(&c(6)), a.mul(&c(2)).floor_div(&c(3)))
        }
        "gcd_with_remainder_div_offset" => {
            let a = var("a", 0, 2);
            build(vec![a.clone()], a.mul(&c(4)).add(&c(2)).floor_div(&c(6)), a.mul(&c(2)).add(&c(1)).floor_div(&c(3)))
        }
        "gcd_with_remainder_mod_offset" => {
            let a = var("a", 0, 2);
            let expected = a.mul(&c(2)).add(&c(1)).mod_(&c(3)).mul(&c(2)).add(&c(1));
            build(vec![a.clone()], a.mul(&c(4)).add(&c(3)).mod_(&c(6)), expected)
        }
        // nest_by_factor (divandmod.py:57-70), test_mod_nest_by_factor.
        "nest_by_factor_mod" => {
            let (gidx0, lidx0) = (var("gidx0", 0, 15), var("lidx0", 0, 3));
            let expr = gidx0.mul(&c(4)).add(&lidx0).mod_(&c(8));
            let expected = gidx0.mod_(&c(2)).mul(&c(4)).add(&lidx0);
            build(vec![gidx0.clone(), lidx0.clone()], expr, expected)
        }
        "nest_by_factor_mod_odd" => {
            let (a, b) = (var("a", 0, 10), var("b", 0, 2));
            let expr = a.mul(&c(3)).add(&b).mod_(&c(9));
            let expected = a.mod_(&c(3)).mul(&c(3)).add(&b);
            build(vec![a.clone(), b.clone()], expr, expected)
        }
        "nest_by_factor_mod_with_const" => {
            let (a, b) = (var("a", 0, 7), var("b", 0, 1));
            let expr = a.mul(&c(4)).add(&b).add(&c(2)).mod_(&c(8));
            let expected = b.add(&a.mod_(&c(2)).mul(&c(4))).add(&c(2));
            build(vec![a.clone(), b.clone()], expr, expected)
        }
        // divide_by_gcd with a negative coefficient (divandmod.py:79-83),
        // test_mod_gcd_factor_neg.
        "divide_by_gcd_negative_coefficient" => {
            let a = var("a", 0, 10);
            let expr = a.mul(&c(-4)).add(&c(4)).mod_(&c(8));
            let expected = a.mul(&c(-1)).add(&c(1)).mod_(&c(2)).mul(&c(4));
            build(vec![a.clone()], expr, expected)
        }
        // factor_remainder must floor the carry, not truncate it: a coefficient
        // of -7 over a divisor of 5 splits as -7 = 5*(-2) + 3, so the carry is
        // -2. Truncating division would give -1 and a numerator that is off by
        // 5 per unit of `a`.
        "factor_remainder_negative_carry" => {
            let (a, b) = (var("a", 0, 10), var("b", 70, 100));
            let expr = a.mul(&c(-7)).add(&b).floor_div(&c(5));
            let expected = a.mul(&c(3)).add(&b).floor_div(&c(5)).add(&a.mul(&c(-2)));
            build(vec![a.clone(), b.clone()], expr, expected)
        }
        // factor_remainder's constant-divisor branch (divandmod.py:88-90),
        // test_div_partial_quotient.
        "factor_remainder_partial_quotient" => {
            let b = var("b", 0, 100);
            let expr = b.mul(&c(31)).add(&c(1)).floor_div(&c(18));
            let expected = b.mul(&c(13)).add(&c(1)).floor_div(&c(18)).add(&b);
            build(vec![b.clone()], expr, expected)
        }
        other => panic!("unknown divmod case {other}"),
    }
}

fn divmod_points(vars: &[Arc<UOp>]) -> Vec<Vec<i64>> {
    let mut points = vec![Vec::new()];
    for var in vars {
        let svod_ir::Op::DefineVar(ops::DefineVar { min_val, max_val, .. }) = var.op() else {
            panic!("expected a variable")
        };
        points = points
            .iter()
            .flat_map(|point| {
                (*min_val..=*max_val).map(|value| point.iter().copied().chain([value]).collect::<Vec<_>>())
            })
            .collect();
    }
    points
}

fn divmod_eval(expr: &Arc<UOp>, vars: &[Arc<UOp>], point: &[i64]) -> i64 {
    let bound = expr.substitute(
        &vars
            .iter()
            .zip(point)
            .map(|(var, value)| (svod_ir::UOpKey(var.clone()), UOp::const_(var.dtype(), ConstValue::Int(*value))))
            .collect(),
    );
    let folded = svod_ir::rewrite::graph_rewrite(crate::symbolic::symbolic(), bound, &mut ());
    match folded.op() {
        svod_ir::Op::Const(value) => value.0.try_int().expect("integer constant"),
        _ => panic!("did not fold to a constant: {}", folded.tree()),
    }
}

#[test_case("remove_nested_mod"; "remove_nested_mod: (a%12)%4 -> a%4")]
#[test_case("remove_nested_mod_to_zero"; "remove_nested_mod: (a*4%12)%4 -> 0")]
#[test_case("nested_div_six_over_three"; "nested_div: x%6//3 -> x//3%2")]
#[test_case("nested_div_twelve_over_four"; "nested_div: x%12//4 -> x//4%3")]
#[test_case("nested_div_into_mod"; "nested_div: idx*4%8//4 -> idx%2")]
#[test_case("gcd_with_remainder_div"; "gcd_with_remainder: a*4//6 -> a*2//3")]
#[test_case("gcd_with_remainder_div_offset"; "gcd_with_remainder: (a*4+2)//6 -> (a*2+1)//3")]
#[test_case("gcd_with_remainder_mod_offset"; "gcd_with_remainder: (a*4+3)%6 -> (a*2+1)%3*2+1")]
#[test_case("nest_by_factor_mod"; "nest_by_factor: (gidx0*4+lidx0)%8 -> lidx0+gidx0%2*4")]
#[test_case("nest_by_factor_mod_odd"; "nest_by_factor: (a*3+b)%9 -> b+a%3*3")]
#[test_case("nest_by_factor_mod_with_const"; "nest_by_factor: (a*4+b+2)%8 -> b+a%2*4+2")]
#[test_case("divide_by_gcd_negative_coefficient"; "divide_by_gcd: (a*-4+4)%8 -> (a*-1+1)%2*4")]
#[test_case("factor_remainder_negative_carry"; "factor_remainder: (a*-7+b)//5 floors the carry")]
#[test_case("factor_remainder_partial_quotient"; "factor_remainder: (b*31+1)//18 -> (b*13+1)//18+b")]
fn tinygrad_divmod_examples_fold_to_the_upstream_form(name: &str) {
    let DivmodCase { vars, expr, expected } = divmod_case(name);
    let folded = svod_ir::rewrite::graph_rewrite(crate::symbolic::symbolic(), expr.clone(), &mut ());
    let expected = svod_ir::rewrite::graph_rewrite(crate::symbolic::symbolic(), expected, &mut ());
    assert!(Arc::ptr_eq(&folded, &expected), "{name}: got {}, want {}", folded.tree(), expected.tree());

    for point in divmod_points(&vars) {
        assert_eq!(divmod_eval(&expr, &vars, &point), divmod_eval(&folded, &vars, &point), "{name} at {point:?}");
    }
}
