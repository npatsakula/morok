mod devectorize_pin;
mod early_reject_pin;
mod index_lowering;

use crate::{
    pattern::RewriteResult,
    rewrite::graph_rewrite,
    symbolic::patterns::{
        advanced_division_dsl_patterns, commutative_canonicalization, comparison_dsl_patterns,
        constant_folding_dsl_patterns, div_mod_recombine_dsl_patterns, division_dsl_patterns,
        identity_and_zero_patterns, pm_remove_invalid, propagate_invalid, range_based_mod_div_patterns,
        sym_phase3_patterns, term_combining_dsl_patterns, vmin_vmax_collapse_patterns, weak_float_values_are_committed,
    },
    symbolic::{
        pm_fold_cast_const, sym, symbolic, symbolic_simple,
        valid_simplification::{parse_valid, pm_drop_and_clauses, simplify_valid, uop_given_valid},
    },
};
use smallvec::smallvec;
use std::{f32::consts::PI, sync::Arc};
use svod_dtype::DType;
use svod_ir::ops;
use svod_ir::pattern::TypedPatternMatcher;
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::properties::HasWeakFloatProperty;
use svod_ir::uop::range_eval::compute_sound_vmin_vmax;
use svod_ir::{BinaryOp, ConstValue, Op, TernaryOp, UOp, UnaryOp};
use test_case::test_case;

fn rewrite(matcher: &TypedPatternMatcher, expr: Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(matcher, expr, &mut ())
}

/// A float the analyses cannot bound: it may be a NaN, an infinity or a signed
/// zero, so no value-sensitive rule may fire on it.
fn unknown_f32() -> Arc<UOp> {
    let buffer = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 1, DType::Float32);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    UOp::load().index(index).call()
}

/// The operands the rewrite tables below are written against.
struct Vars {
    /// Non-negative integers.
    x: Arc<UOp>,
    y: Arc<UOp>,
    /// Integers that straddle zero, for the rules that must survive a sign change.
    a: Arc<UOp>,
    b: Arc<UOp>,
    /// An integer whose range excludes zero, so `n % n` is defined.
    n: Arc<UOp>,
    p: Arc<UOp>,
    q: Arc<UOp>,
    idx: Arc<UOp>,
    bounded: Arc<UOp>,
    unknown: Arc<UOp>,
}

impl Vars {
    fn new() -> Self {
        Self {
            x: UOp::var("x", DType::Int32, 0, 100),
            y: UOp::var("y", DType::Int32, 0, 100),
            a: UOp::var("a", DType::Int32, -100, 100),
            b: UOp::var("b", DType::Int32, -100, 100),
            n: UOp::var("n", DType::Int32, 1, 100),
            p: UOp::var("p", DType::Bool, 0, 1),
            q: UOp::var("q", DType::Bool, 0, 1),
            idx: UOp::define_var("i".to_string(), 0, 1024),
            bounded: UOp::var("bounded", DType::Float32, -1, 1),
            unknown: unknown_f32(),
        }
    }

    fn c(&self, value: i64) -> Arc<UOp> {
        self.x.const_like(value)
    }

    fn ic(&self, value: i64) -> Arc<UOp> {
        UOp::index_const(value)
    }

    fn f(&self, value: f64) -> Arc<UOp> {
        UOp::const_(DType::Float32, ConstValue::Float(value))
    }

    fn t(&self, value: bool) -> Arc<UOp> {
        UOp::const_(DType::Bool, ConstValue::Bool(value))
    }

    fn where_(&self, condition: &Arc<UOp>, true_val: Arc<UOp>, false_val: Arc<UOp>) -> Arc<UOp> {
        UOp::try_where(condition.clone(), true_val, false_val).unwrap()
    }
}

type Term = fn(&Vars) -> Arc<UOp>;

/// The scalar algebra the symbolic tiers must perform. Rules whose whole domain
/// is already swept by a property test — identity and annihilator folding,
/// constant folding, self-application, and the `(a op c1) op c2` collapses —
/// live in `schedule/src/test/property/symbolic_props.rs`; the rows here pin the
/// rules that need a specific shape.
#[test_case(symbolic_simple(), |v| v.n.mod_(&v.n), |v| v.c(0) ; "a value modulo itself")]
#[test_case(symbolic_simple(), |v| v.x.xor(&v.x), |v| v.c(0) ; "a value xored with itself")]
#[test_case(symbolic(), |v| v.c(12).mul(&v.x).floor_div(&v.c(3)), |v| v.c(4).mul(&v.x) ; "division divides an exact coefficient")]
#[test_case(symbolic(), |v| v.c(6).mul(&v.x).add(&v.y).mod_(&v.c(3)), |v| v.y.mod_(&v.c(3)) ; "modulo drops a divisible left term")]
#[test_case(symbolic(), |v| v.x.add(&v.c(9).mul(&v.y)).mod_(&v.c(3)), |v| v.x.mod_(&v.c(3)) ; "modulo drops a divisible right term")]
#[test_case(symbolic(), |v| v.c(6).mul(&v.x).add(&v.c(9).mul(&v.y)).floor_div(&v.c(3)), |v| v.c(2).mul(&v.x).add(&v.c(3).mul(&v.y)) ; "division distributes over a divisible sum")]
#[test_case(symbolic(), |v| v.c(12).mul(&v.x).sub(&v.c(6).mul(&v.y)).floor_div(&v.c(3)), |v| v.c(4).mul(&v.x).add(&v.y.mul(&v.c(-2))) ; "division distributes over a divisible difference")]
#[test_case(symbolic(), |v| v.x.sub(&v.c(3)).add(&v.c(5)), |v| v.x.add(&v.c(2)) ; "a subtraction and an addition fold together")]
#[test_case(symbolic(), |v| v.x.add(&v.c(3)).sub(&v.c(5)), |v| v.x.add(&v.c(-2)) ; "an addition and a subtraction fold together")]
#[test_case(symbolic(), |v| v.x.floor_div(&v.c(2)).add(&v.c(1)).floor_div(&v.c(2)), |v| v.x.add(&v.c(2)).floor_div(&v.c(4)) ; "a nested division absorbs the offset")]
#[test_case(symbolic(), |v| v.ic(1).add(&v.idx).sub(&v.ic(1)), |v| v.idx.clone() ; "an index constant cancels across a sum")]
#[test_case(symbolic(), |v| v.a.add(&v.c(2)).lt(&v.c(5)), |v| v.a.lt(&v.c(3)) ; "a comparison absorbs a constant offset")]
#[test_case(symbolic(), |v| v.a.add(&v.c(10)).lt(&v.c(5)), |v| v.a.lt(&v.c(-5)) ; "a comparison offset may go negative")]
#[test_case(symbolic(), |v| v.a.neg().lt(&v.b.neg()), |v| v.b.lt(&v.a) ; "negating both sides flips a comparison")]
#[test_case(symbolic(), |v| v.a.floor_div(&v.c(-1)), |v| v.a.mul(&v.c(-1)) ; "division by minus one is a negation")]
#[test_case(symbolic(), |v| v.x.neg().neg(), |v| v.x.clone() ; "a doubled integer negation")]
#[test_case(symbolic(), |v| v.bounded.neg().neg(), |v| v.bounded.clone() ; "a doubled float negation")]
#[test_case(symbolic(), |v| v.x.max(&v.x), |v| v.x.clone() ; "max of a value with itself")]
#[test_case(symbolic(), |v| v.unknown.max(&v.unknown), |v| v.unknown.clone() ; "max of an unknown float with itself")]
#[test_case(symbolic_simple(), |v| v.x.try_pow(&v.c(0)).unwrap(), |v| v.c(1) ; "an integer to the power of zero")]
#[test_case(symbolic_simple(), |v| v.x.try_pow(&v.c(1)).unwrap(), |v| v.x.clone() ; "an integer to the power of one")]
#[test_case(symbolic_simple(), |v| v.bounded.try_pow(&v.f(0.0)).unwrap(), |v| v.f(1.0) ; "a float to the power of zero")]
#[test_case(symbolic(), |v| v.bounded.lt(&v.f(2.0)), |v| v.t(true) ; "an explicitly bounded float comparison decides")]
#[test_case(symbolic_simple(), |v| v.unknown.add(&v.f(-0.0)), |v| v.unknown.clone() ; "adding negative zero is the float identity")]
// Term combining.
#[test_case(symbolic(), |v| v.x.add(&v.x), |v| v.x.mul(&v.c(2)) ; "a value added to itself gains a coefficient")]
#[test_case(symbolic(), |v| v.c(3).mul(&v.x).add(&v.c(5).mul(&v.x)), |v| v.x.mul(&v.c(8)) ; "coefficients of a shared factor add")]
#[test_case(symbolic(), |v| v.x.add(&v.x.mul(&v.c(3))), |v| v.x.mul(&v.c(4)) ; "a bare term counts as coefficient one")]
#[test_case(symbolic(), |v| v.y.add(&v.x).add(&v.x), |v| v.y.add(&v.x.mul(&v.c(2))) ; "terms combine across an unrelated addend")]
#[test_case(symbolic(), |v| v.c(-1).mul(&v.x.add(&v.c(3))), |v| v.x.mul(&v.c(-1)).add(&v.c(-3)) ; "negation distributes over a shifted value")]
// Booleans.
#[test_case(symbolic_simple(), |v| v.p.not().not(), |v| v.p.clone() ; "a doubled boolean not")]
#[test_case(symbolic_simple(), |v| v.x.not().not(), |v| v.x.clone() ; "a doubled bitwise not")]
#[test_case(symbolic(), |v| v.p.or_(&v.p.not()), |v| v.t(true) ; "the excluded middle")]
#[test_case(symbolic(), |v| v.p.and_(&v.p.not()), |v| v.t(false) ; "a contradiction")]
#[test_case(symbolic(), |v| v.t(true).or_(&v.p), |v| v.t(true) ; "true absorbs a disjunction")]
#[test_case(symbolic_simple(), |v| v.t(false).and_(&v.p), |v| v.t(false) ; "false absorbs a conjunction")]
#[test_case(symbolic(), |v| v.t(true).and_(&v.p), |v| v.p.clone() ; "true is the conjunction identity")]
#[test_case(symbolic_simple(), |v| v.t(false).or_(&v.p), |v| v.p.clone() ; "false is the disjunction identity")]
#[test_case(symbolic_simple(), |v| v.p.mul(&v.q), |v| v.p.and_(&v.q) ; "a boolean product is a conjunction")]
#[test_case(symbolic_simple(), |v| v.p.add(&v.q), |v| v.p.or_(&v.q) ; "a boolean sum is a disjunction")]
#[test_case(symbolic_simple(), |v| v.p.max(&v.q), |v| v.p.or_(&v.q) ; "a boolean max is a disjunction")]
// Selection.
#[test_case(symbolic_simple(), |v| v.where_(&v.p, v.x.clone(), v.x.clone()), |v| v.x.clone() ; "a selection with equal branches")]
#[test_case(symbolic_simple(), |v| v.where_(&v.p, v.t(true), v.t(false)), |v| v.p.clone() ; "a selection that reproduces its condition")]
#[test_case(symbolic_simple(), |v| v.where_(&v.p, v.t(false), v.t(true)), |v| v.p.not() ; "a selection that negates its condition")]
#[test_case(symbolic_simple(), |v| v.where_(&v.t(true), v.x.clone(), v.y.clone()), |v| v.x.clone() ; "a selection on a true constant")]
#[test_case(symbolic_simple(), |v| v.where_(&v.t(false), v.x.clone(), v.y.clone()), |v| v.y.clone() ; "a selection on a false constant")]
#[test_case(symbolic(), |v| v.where_(&v.p.not(), v.x.clone(), v.y.clone()), |v| v.where_(&v.p, v.y.clone(), v.x.clone()) ; "a negated condition swaps the branches")]
#[test_case(symbolic_simple(), |v| v.where_(&v.p, v.where_(&v.q, v.x.clone(), v.y.clone()), v.y.clone()), |v| v.where_(&v.p.and_(&v.q), v.x.clone(), v.y.clone()) ; "nested selections with a shared false branch merge")]
#[test_case(symbolic(), |v| v.where_(&v.p, v.c(1), v.x.clone()).add(&v.where_(&v.p, v.c(2), v.y.clone())), |v| v.where_(&v.p, v.c(3), v.x.add(&v.y)) ; "an operation hoists through a shared condition")]
#[test_case(sym(), |v| v.where_(&v.p, v.c(1), v.c(0)).cast(DType::Float32), |v| v.where_(&v.p, v.c(1).cast(DType::Float32), v.c(0).cast(DType::Float32)) ; "a cast pushes into both branches")]
fn symbolic_rewrites_to_the_expected_form(matcher: &TypedPatternMatcher, input: Term, expected: Term) {
    let vars = Vars::new();
    let folded = rewrite(matcher, input(&vars));
    let want = expected(&vars);
    assert!(Arc::ptr_eq(&folded, &want), "got {}, want {}", folded.tree(), want.tree());
}

/// Shapes that look like a rewrite but are unsound, and must come back untouched.
#[test_case(symbolic_simple(), |v| v.x.and_(&v.y) ; "a conjunction of two variables")]
#[test_case(symbolic_simple(), |v| v.c(3).mul(&v.x).add(&v.c(5).mul(&v.y)) ; "terms over different variables")]
#[test_case(symbolic_simple(), |v| v.x.add(&v.y).mod_(&v.c(3)) ; "modulo of an indivisible sum")]
#[test_case(symbolic_simple(), |v| v.x.mul(&v.y) ; "an integer product is not a conjunction")]
#[test_case(symbolic_simple(), |v| v.x.cast(DType::Float32) ; "a cast of a variable is not folded")]
#[test_case(symbolic_simple(), |v| v.unknown.ne(&v.unknown) ; "an unknown float may be a NaN")]
#[test_case(symbolic_simple(), |v| v.unknown.lt(&v.unknown) ; "an unknown float compared with itself")]
#[test_case(symbolic_simple(), |v| v.unknown.mul(&v.bounded).floor_div(&v.bounded) ; "float cancellation changes rounding")]
#[test_case(symbolic(), |v| v.unknown.max(&v.f(f32::MAX as f64)) ; "an unknown float against the finite limit")]
#[test_case(symbolic(), |v| v.unknown.lt(&v.f(f32::MAX as f64)) ; "an unknown float below the finite limit")]
#[test_case(symbolic(), |v| v.where_(&v.unknown.lt(&v.f(f32::MAX as f64)), v.f(1.0), v.f(2.0)) ; "a selection on an unknown float comparison")]
#[test_case(symbolic_simple(), |v| v.unknown.add(&v.f(0.0)) ; "adding positive zero keeps a negative zero")]
#[test_case(symbolic(), |v| v.where_(&v.p, v.f(-0.0), v.f(0.0)).max(&v.f(0.0)) ; "a float max tie keeps the sign of zero")]
#[test_case(symbolic_simple(), |v| v.where_(&v.p, v.where_(&v.q, v.x.clone(), v.y.clone()), v.a.clone()) ; "nested selections with different false branches")]
#[test_case(symbolic_simple(), |v| v.where_(&v.p, v.x.clone(), v.y.clone()).add(&v.where_(&v.q, v.x.clone(), v.y.clone())) ; "selections under different conditions")]
fn symbolic_leaves_unsound_shapes_alone(matcher: &TypedPatternMatcher, input: Term) {
    let vars = Vars::new();
    let expr = input(&vars);
    let folded = rewrite(matcher, expr.clone());
    assert!(Arc::ptr_eq(&folded, &expr), "unexpected rewrite to {}", folded.tree());
}

#[test_case(0, 8, 77, Some(true) ; "the range sits entirely below the bound")]
#[test_case(0, 8, 9, Some(true) ; "the range ends one below the bound")]
#[test_case(0, 8, 5, None ; "the bound falls inside the range")]
#[test_case(0, 8, 0, Some(false) ; "the bound is the range minimum")]
#[test_case(3, 8, 3, Some(false) ; "the bound is a shifted range minimum")]
fn comparison_with_a_constant_folds_only_when_the_range_decides(lo: i64, hi: i64, bound: i64, expect: Option<bool>) {
    let a = UOp::var("a", DType::Int32, lo, hi);
    let comparison = a.lt(&a.const_like(bound));
    let folded = rewrite(symbolic(), comparison.clone());
    match expect {
        Some(value) => {
            assert!(matches!(folded.op(), Op::Const(c) if c.0 == ConstValue::Bool(value)), "{}", folded.tree())
        }
        None => assert!(Arc::ptr_eq(&folded, &comparison), "{}", folded.tree()),
    }
}

#[test_case(BinaryOp::Lt, 0, 4, 5, 10, true ; "disjoint ranges decide less-than")]
#[test_case(BinaryOp::Lt, 5, 10, 0, 4, false ; "disjoint ranges decide the reversed less-than")]
#[test_case(BinaryOp::Eq, 0, 4, 10, 20, false ; "disjoint ranges are never equal")]
#[test_case(BinaryOp::Ne, 0, 4, 10, 20, true ; "disjoint ranges always differ")]
fn comparison_between_disjoint_ranges_folds(op: BinaryOp, a_lo: i64, a_hi: i64, b_lo: i64, b_hi: i64, expect: bool) {
    let a = UOp::var("a", DType::Int32, a_lo, a_hi);
    let b = UOp::var("b", DType::Int32, b_lo, b_hi);
    let folded = rewrite(symbolic(), UOp::alu(op, a, b));
    assert!(matches!(folded.op(), Op::Const(c) if c.0 == ConstValue::Bool(expect)), "{}", folded.tree());
}

#[test_case(DType::Int32, ConstValue::Int(42), DType::Float32, ConstValue::Float(42.0) ; "integer to float")]
#[test_case(DType::Float32, ConstValue::Float(PI as f64), DType::Int32, ConstValue::Int(3) ; "float truncates to integer")]
#[test_case(DType::Bool, ConstValue::Bool(true), DType::Int32, ConstValue::Int(1) ; "boolean to integer")]
fn a_cast_constant_folds_at_the_target_dtype(from: DType, value: ConstValue, to: DType, expect: ConstValue) {
    let cast = UOp::const_(from, value).cast(to.clone());
    let RewriteResult::Rewritten(folded) = pm_fold_cast_const().rewrite(&cast, &mut ()) else {
        panic!("cast of a constant did not fold")
    };
    assert_eq!(folded.dtype(), to);
    assert!(matches!(folded.op(), Op::Const(c) if c.0 == expect), "{}", folded.tree());
}

#[test]
fn cast_chains_collapse_only_when_every_hop_is_lossless() {
    let narrow = UOp::var("narrow", DType::Int16, 0, i16::MAX as i64);
    let wide = UOp::var("wide", DType::Int32, 0, i64::MAX);
    let chain =
        |source: &Arc<UOp>, hops: &[DType]| hops.iter().fold(source.clone(), |value, dtype| value.cast(dtype.clone()));

    for (source, hops) in [(&wide, &[DType::Int32][..]), (&narrow, &[DType::Int32, DType::Int16][..])] {
        assert!(Arc::ptr_eq(&rewrite(symbolic_simple(), chain(source, hops)), source));
    }
    // Float32 cannot hold every Int32, so neither the round trip nor a lone widening cast folds.
    for hops in [&[DType::Float32, DType::Int32][..], &[DType::Float32][..]] {
        assert!(matches!(rewrite(symbolic_simple(), chain(&wide, hops)).op(), Op::Cast(..)));
    }
}

/// A trip-count-one RANGE must survive the collapse: folding it away breaks a
/// hand-built kernel's loop carry (tinygrad `uop/symbolic.py:248`).
#[test]
fn single_valued_bounds_collapse_products_but_not_sums() {
    let a = UOp::var("a", DType::Int32, 2, 2);
    let b = UOp::var("b", DType::Int32, 3, 3);

    let product = rewrite(vmin_vmax_collapse_patterns(), a.mul(&b));
    assert!(matches!(product.op(), Op::Const(value) if value.0 == ConstValue::Int(6)));

    for sum in [a.add(&b), a.sub(&b), a.max(&b)] {
        assert!(matches!(rewrite(vmin_vmax_collapse_patterns(), sum).op(), Op::Binary(..)));
    }
}

#[test]
fn a_range_reduced_by_its_own_extent_is_the_range_or_zero() {
    let end = UOp::index_const(8);
    let range = UOp::range(end.clone(), 0);

    assert!(Arc::ptr_eq(&rewrite(symbolic(), range.mod_(&end)), &range));
    let quotient = rewrite(symbolic(), range.floor_div(&end));
    assert!(matches!(quotient.op(), Op::Const(value) if value.0 == ConstValue::Int(0)), "{}", quotient.tree());
}

/// Upstream's reciprocal distribution rules are all IEEE-inexact and are
/// deliberately absent from `sym`; `symbolic/patterns.rs` cites this test by name.
#[test]
fn unknown_float_division_power_and_reciprocal_are_not_algebraically_rewritten() {
    let value = unknown_f32();
    let square = value.try_pow(&value.const_like(2.0)).unwrap();
    let reciprocal = UOp::try_reciprocal(&value.mul(&value)).unwrap();

    assert!(matches!(rewrite(sym(), value.floor_div(&value)).op(), Op::Binary(BinaryOp::Fdiv, ..)));
    assert!(matches!(rewrite(sym(), square).op(), Op::Binary(BinaryOp::Pow, ..)));
    assert!(matches!(rewrite(sym(), reciprocal).op(), Op::Unary(UnaryOp::Reciprocal, ..)));
}

/// Only a range that excludes zero makes `x / x` a constant one.
#[test]
fn finite_nonzero_float_self_division_folds() {
    let x = UOp::var("x", DType::Float32, 1, 10);
    let folded = rewrite(symbolic_simple(), x.floor_div(&x));
    assert!(matches!(folded.op(), Op::Const(value) if value.0 == ConstValue::Float(1.0)), "{}", folded.tree());
}

/// `(-1) * (x + y)` distributes at the `sym` tier, so no product survives at the root.
#[test]
fn negation_distributes_over_a_sum_of_variables() {
    let x = UOp::range_const(10, 0).cast(DType::Int32);
    let y = UOp::range_const(20, 1).cast(DType::Int32);
    let product = UOp::native_const(-1i32).mul(&x.add(&y));

    assert!(!matches!(rewrite(sym(), product).op(), Op::Binary(BinaryOp::Mul, ..)));
}

/// Tinygrad's associative variation of the WHERE/ALU combine (`uop/symbolic.py:207-208`)
/// reaches the pair through an intervening addend, leaving a single WHERE.
#[test]
fn selections_under_one_condition_combine_across_an_addend() {
    let vars = Vars::new();
    let first = vars.where_(&vars.p, vars.c(1), vars.x.clone());
    let second = vars.where_(&vars.p, vars.c(2), vars.y.clone());

    let combined = rewrite(symbolic(), vars.y.add(&first).add(&second));

    let wheres = combined.toposort().iter().filter(|n| matches!(n.op(), Op::Ternary(TernaryOp::Where, ..))).count();
    assert_eq!(wheres, 1, "{}", combined.tree());
}

// ============================================================================
// Structural commutative ordering
// ============================================================================

fn assert_binary_sources(root: &Arc<UOp>, lhs: &Arc<UOp>, rhs: &Arc<UOp>) {
    let Op::Binary(_, actual_lhs, actual_rhs) = root.op() else {
        panic!("expected binary root, got {:?}", root.op());
    };
    assert!(Arc::ptr_eq(actual_lhs, lhs), "unexpected lhs: {:?}", root.op());
    assert!(Arc::ptr_eq(actual_rhs, rhs), "unexpected rhs: {:?}", root.op());
}

#[test]
fn commutative_index_ops_follow_tinygrad_structural_order() {
    let end = UOp::index_const(8);
    let special = UOp::special(end.clone(), "gidx0".to_string());
    let range = UOp::range(end, 0);

    for op in [BinaryOp::Add, BinaryOp::Mul, BinaryOp::And, BinaryOp::Or, BinaryOp::Xor, BinaryOp::Max] {
        let authored = UOp::new(Op::Binary(op, special.clone(), range.clone()), DType::WeakInt);
        let reversed = UOp::new(Op::Binary(op, range.clone(), special.clone()), DType::WeakInt);
        let authored = rewrite(commutative_canonicalization(), authored);
        let reversed = rewrite(commutative_canonicalization(), reversed);

        assert_binary_sources(&authored, &special, &range);
        assert_binary_sources(&reversed, &special, &range);
        assert!(Arc::ptr_eq(&authored, &reversed));
        assert_eq!(authored.content_hash, reversed.content_hash);
    }
}

#[test]
fn commutative_index_order_ranks_constants_ranges_variables_and_stacks() {
    let range0 = UOp::range_const(8, 0);
    let range1 = UOp::range_const(8, 1);
    let special = UOp::special(UOp::index_const(8), "gidx0".to_string());

    let const_first = rewrite(commutative_canonicalization(), UOp::index_const(3).add(&range0));
    assert_binary_sources(&const_first, &range0, &UOp::index_const(3));

    let ranges_reversed = rewrite(commutative_canonicalization(), range1.add(&range0));
    assert_binary_sources(&ranges_reversed, &range0, &range1);

    let var_a = UOp::define_var("a".to_string(), 0, 8);
    let var_b = UOp::define_var("b".to_string(), 0, 8);
    assert_binary_sources(&rewrite(commutative_canonicalization(), var_b.add(&var_a)), &var_a, &var_b);

    // A nested tree is ranked by its own sources, and both authorings converge.
    let nested_first = rewrite(commutative_canonicalization(), range1.add(&range0).add(&special));
    let Op::Binary(BinaryOp::Add, actual_special, actual_nested) = nested_first.op() else {
        panic!("expected nested ADD, got {:?}", nested_first.op());
    };
    assert!(Arc::ptr_eq(actual_special, &special));
    assert_binary_sources(actual_nested, &range0, &range1);
    let other_order = rewrite(commutative_canonicalization(), special.add(&range0.add(&range1)));
    assert!(Arc::ptr_eq(&nested_first, &other_order));

    // A VCONST is projected as tinygrad's STACK of its lanes, so it sorts last.
    let vconst = UOp::vconst(vec![ConstValue::Int(2), ConstValue::Int(1)], DType::WeakInt);
    let stack = UOp::stack(smallvec![range0.clone(), special.clone()]);
    let authored = UOp::new(Op::Binary(BinaryOp::Add, vconst.clone(), stack.clone()), vconst.dtype());
    assert_binary_sources(&rewrite(commutative_canonicalization(), authored), &stack, &vconst);
}

#[test]
fn commutative_order_declines_outside_weak_integers_and_on_ties() {
    for dtype in [DType::Index, DType::Int32, DType::Float32] {
        let lhs = UOp::const_(dtype.clone(), if dtype.is_float() { 4.0.into() } else { 4.into() });
        let rhs = UOp::const_(dtype.clone(), if dtype.is_float() { 3.0.into() } else { 3.into() });
        let authored = UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), dtype);
        assert!(Arc::ptr_eq(&rewrite(commutative_canonicalization(), authored.clone()), &authored));
    }

    // Structurally tied and incomparable operands keep their authored order.
    let base = UOp::index_const(7);
    let (left, right) = (base.with_tag(smallvec![1]), base.with_tag(smallvec![2]));
    let tied = UOp::new(Op::Binary(BinaryOp::Add, right.clone(), left.clone()), DType::WeakInt);
    assert_binary_sources(&rewrite(commutative_canonicalization(), tied), &right, &left);

    let end = UOp::index_const(8);
    let weak = UOp::range_axis(end.clone(), svod_ir::AxisId::Renumbered(0), svod_ir::AxisType::Weak);
    let global = UOp::range_axis(end, svod_ir::AxisId::Renumbered(0), svod_ir::AxisType::Global);
    assert_binary_sources(&rewrite(commutative_canonicalization(), weak.add(&global)), &weak, &global);

    // Reordering preserves the node's tag.
    let range = UOp::range_const(8, 0);
    let special = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let tagged = rewrite(commutative_canonicalization(), range.add(&special).with_tag(smallvec![7]));
    assert_binary_sources(&tagged, &special, &range);
    assert_eq!(tagged.tag().as_deref(), Some(&[7][..]));
}

/// Only the tiers that include the canonicalization reorder; `symbolic_simple`
/// leaves the authored order alone.
#[test]
fn symbolic_boundary_applies_structural_commutative_order() {
    let range = UOp::range_const(8, 0);
    let special = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let reversed = range.add(&special);

    assert_binary_sources(&rewrite(symbolic_simple(), reversed.clone()), &range, &special);
    for matcher in [symbolic(), sym()] {
        assert_binary_sources(&rewrite(matcher, reversed.clone()), &special, &range);
    }
}

// ============================================================================
// Reduced-precision and weak constant folding
// ============================================================================

#[test]
fn reduced_float_folding_commits_result_before_comparison() {
    let one = UOp::const_(DType::FP8E4M3, ConstValue::Float(1.0));
    let half_ulp = UOp::const_(DType::FP8E4M3, ConstValue::Float(0.0625));
    let folded = rewrite(symbolic_simple(), one.add(&half_ulp));
    assert!(matches!(folded.op(), Op::Const(value) if value.0 == ConstValue::Float(1.0)));

    let rounded = UOp::const_(DType::Float32, ConstValue::Float(-3.2));
    let exact_grid_value = UOp::const_(DType::Float32, ConstValue::Float(-3.200000047683716));
    let comparison = rewrite(symbolic(), rounded.eq(&exact_grid_value));
    assert!(matches!(comparison.op(), Op::Const(value) if value.0 == ConstValue::Bool(true)));
}

#[test]
fn reduced_float_vconst_folding_commits_each_result_lane() {
    let values = UOp::vconst(vec![ConstValue::Float(1.0), ConstValue::Float(1.125)], DType::FP8E4M3);
    let increments = UOp::vconst(vec![ConstValue::Float(0.0625), ConstValue::Float(0.0625)], DType::FP8E4M3);
    let folded = rewrite(symbolic_simple(), values.add(&increments));
    assert!(matches!(folded.op(), Op::VConst(ops::VConst { values })
        if values == &vec![ConstValue::Float(1.0), ConstValue::Float(1.25)]));
}

/// The value-sensitive guard runs on every pattern attempt, so it must be a
/// memoised per-node property rather than a graph walk (previously O(n^2)).
#[test_case(DType::Float32, true ; "committed float chain")]
#[test_case(DType::WeakFloat, false ; "weak float leaf")]
fn weak_float_guard_is_memoized_per_node(leaf_dtype: DType, committed: bool) {
    const DEPTH: usize = 64;
    let mut root = UOp::const_(leaf_dtype, ConstValue::Float(0.5));
    for _ in 0..DEPTH {
        root = UOp::new(Op::Unary(UnaryOp::Sqrt, root), DType::Float32);
    }

    assert_eq!(weak_float_values_are_committed(&root), committed);

    let nodes = root.toposort();
    assert_eq!(nodes.len(), DEPTH + 1);
    for node in &nodes {
        assert!(HasWeakFloatProperty::cache(node).get().is_some(), "one evaluation per node, cached in place");
    }
    assert!(std::ptr::eq(HasWeakFloatProperty::get(&root), HasWeakFloatProperty::get(&root)));
}

/// `fold_const_alu` folds `exec_alu(a.op, a.dtype, vals, False)` for every dtype —
/// weak included — and returns `a.const_like(...)` at the node's own dtype.
fn weak_int(value: i64) -> Arc<UOp> {
    UOp::const_(DType::WeakInt, ConstValue::Int(value))
}

fn raw_add(lhs: Arc<UOp>, rhs: Arc<UOp>) -> Arc<UOp> {
    let dtype = lhs.dtype();
    UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), dtype)
}

#[test_case(BinaryOp::Add, 1, 14, 15 ; "add")]
#[test_case(BinaryOp::Mul, 7, 28, 196 ; "mul")]
#[test_case(BinaryOp::Sub, 1, 14, -13 ; "sub")]
fn weak_int_constant_operands_fold(op: BinaryOp, lhs: i64, rhs: i64, expect: i64) {
    let expr = UOp::new(Op::Binary(op, weak_int(lhs), weak_int(rhs)), DType::WeakInt);
    let folded = rewrite(symbolic_simple(), expr);

    assert_eq!(folded.dtype(), DType::WeakInt, "the fold stays at the weak dtype: {}", folded.tree());
    assert_eq!(
        crate::rangeify::indexing::get_const_value(&folded),
        Some(ConstValue::Int(expect)),
        "got {}",
        folded.tree()
    );
}

#[test]
fn weak_int_constants_cancel_across_an_index_sum() {
    // The resnet50 `r_16_32_7_7_512_3_3` index shape: `((1+14) + R*196) + (-15)`.
    let range = UOp::range_const(512, 0);
    let scaled = UOp::new(Op::Binary(BinaryOp::Mul, range, weak_int(196)), DType::WeakInt);
    let expr = raw_add(raw_add(raw_add(weak_int(1), weak_int(14)), scaled.clone()), weak_int(-15));

    let folded = rewrite(symbolic(), expr);

    assert!(Arc::ptr_eq(&folded, &scaled), "the constants must cancel, got {}", folded.tree());
}

/// Distribution over an addition is a weak-dtype-only rule. The tier that carries
/// it depends on the addend: a constant addend folds in term combining, two
/// variables need phase three.
#[test_case(term_combining_dsl_patterns(), false ; "constant addend in term combining")]
#[test_case(sym_phase3_patterns(), true ; "variable addend in phase three")]
fn weak_multiplication_distributes_over_an_addition_in_either_order(matcher: &TypedPatternMatcher, variable: bool) {
    let x = UOp::var("x", DType::WeakInt, 0, i64::MAX);
    let three = UOp::const_(DType::WeakInt, ConstValue::Int(3));
    let addend =
        if variable { UOp::var("y", DType::WeakInt, 0, i64::MAX) } else { UOp::const_(DType::WeakInt, 5.into()) };

    for add in [x.add(&addend), addend.add(&x)] {
        for mul in [add.mul(&three), three.mul(&add)] {
            let RewriteResult::Rewritten(result) = matcher.rewrite(&mul, &mut ()) else {
                panic!("expected weak multiplication distribution for {}", mul.tree());
            };
            assert!(matches!(result.op(), Op::Binary(BinaryOp::Add, ..)), "{}", result.tree());
        }
    }

    // The same shape at a concrete integer dtype is left alone.
    let concrete = UOp::var("c", DType::Int32, 0, i64::MAX);
    let concrete_addend = if variable { UOp::var("d", DType::Int32, 0, i64::MAX) } else { UOp::native_const(5i32) };
    let mul = concrete.add(&concrete_addend).mul(&UOp::native_const(3i32));
    assert!(matches!(matcher.rewrite(&mul, &mut ()), RewriteResult::NoMatch));
}

/// The specific `-1 * (x + c)` rule outranks the general distribution.
#[test]
fn weak_distribution_preserves_negation_rule_priority() {
    let x = UOp::var("x", DType::WeakInt, -100, 100);
    let five = UOp::const_(DType::WeakInt, ConstValue::Int(5));
    let neg_one = UOp::const_(DType::WeakInt, ConstValue::Int(-1));

    let RewriteResult::Rewritten(result) = term_combining_dsl_patterns().rewrite(&neg_one.mul(&x.add(&five)), &mut ())
    else {
        panic!("expected the specific negation distribution");
    };
    let Op::Binary(BinaryOp::Add, lhs, rhs) = result.op() else { panic!("expected Add, got {}", result.tree()) };
    assert!(
        matches!(lhs.op(), Op::Binary(BinaryOp::Mul, value, c) if Arc::ptr_eq(value, &x) && matches!(c.op(), Op::Const(cv) if cv.0 == ConstValue::Int(-1)))
    );
    assert!(matches!(rhs.op(), Op::Const(cv) if cv.0 == ConstValue::Int(-5)));
}

// ============================================================================
// uint64 pack/unpack cancellation (tinygrad uop/symbolic.py:170-173)
// ============================================================================

/// `(hi.cast(u64) << shift) | lo.cast(u64)` — the THREEFRY packing idiom.
fn packed_u64(hi: &Arc<UOp>, lo: &Arc<UOp>, shift: i64) -> Arc<UOp> {
    let amount = UOp::const_(DType::UInt64, ConstValue::Int(shift));
    hi.cast(DType::UInt64).shl(&amount).or_(&lo.cast(DType::UInt64))
}

fn u32_var(name: &str) -> Arc<UOp> {
    UOp::var(name, DType::UInt32, 0, u32::MAX as i64)
}

#[test_case(32, true; "shift of thirty two cancels")]
#[test_case(16, false; "shift of sixteen must not cancel")]
fn uint64_pack_low_half_cancels_only_at_thirty_two(shift: i64, folds: bool) {
    let (hi, lo) = (u32_var("hi"), u32_var("lo"));
    let folded = rewrite(symbolic_simple(), packed_u64(&hi, &lo, shift).cast(DType::UInt32));

    assert_eq!(Arc::ptr_eq(&folded, &lo), folds, "got {}", folded.tree());
}

#[test_case(32, true; "shift of thirty two cancels")]
#[test_case(31, false; "shift of thirty one must not cancel")]
fn uint64_pack_high_half_cancels_only_at_thirty_two(shift: i64, folds: bool) {
    let (hi, lo) = (u32_var("hi"), u32_var("lo"));
    let amount = UOp::const_(DType::UInt64, ConstValue::Int(shift));
    let folded = rewrite(symbolic_simple(), packed_u64(&hi, &lo, shift).shr(&amount));

    assert_eq!(Arc::ptr_eq(&folded, &hi.cast(DType::UInt64)), folds, "got {}", folded.tree());
}

#[test]
fn uint64_pack_high_half_needs_a_narrow_low_arm() {
    // A wide low arm can carry bits into the high half, so `>> 32` is not `hi`.
    let hi = u32_var("hi");
    let wide = UOp::var("wide", DType::UInt64, 0, i64::MAX);
    let amount = UOp::const_(DType::UInt64, ConstValue::Int(32));
    let folded = rewrite(symbolic_simple(), hi.cast(DType::UInt64).shl(&amount).or_(&wide).shr(&amount));

    assert!(!Arc::ptr_eq(&folded, &hi.cast(DType::UInt64)), "must not cancel: {}", folded.tree());
}

// ============================================================================
// Typed division and modulo
// ============================================================================

fn eval_closed_typed(expr: &Arc<UOp>) -> Option<ConstValue> {
    use svod_ir::uop::eval::{eval_binary_op_typed, eval_unary_op_typed};

    match expr.op() {
        Op::Const(value) => Some(value.0),
        Op::DefineVar(ops::DefineVar { min_val, max_val, .. }) if min_val == max_val => {
            ConstValue::Int(*min_val).cast(&expr.dtype().scalar_dtype())
        }
        Op::Binary(op, lhs, rhs) => {
            eval_binary_op_typed(*op, eval_closed_typed(lhs)?, eval_closed_typed(rhs)?, expr.dtype().base())
        }
        Op::Unary(op, src) => eval_unary_op_typed(*op, eval_closed_typed(src)?, expr.dtype().base()),
        _ => None,
    }
}

#[test]
fn typed_divmod_wrap_counterexamples_do_not_misrewrite() {
    let i8_const = |value| UOp::const_(DType::Int8, ConstValue::Int(value));

    let div = i8_const(100).mul(&i8_const(2)).add(&i8_const(1)).floor_div(&i8_const(2));
    let div_result = rewrite(symbolic(), div.clone());
    assert_eq!(eval_closed_typed(&div), Some(ConstValue::Int(-28)));
    assert_eq!(eval_closed_typed(&div_result), eval_closed_typed(&div));
    assert!(!matches!(div_result.op(), Op::Const(value) if value.0 == ConstValue::Int(100)));

    let modulo = i8_const(100).mul(&i8_const(3)).add(&i8_const(1)).mod_(&i8_const(3));
    let mod_result = rewrite(symbolic(), modulo.clone());
    assert_eq!(eval_closed_typed(&modulo), Some(ConstValue::Int(0)));
    assert_eq!(eval_closed_typed(&mod_result), eval_closed_typed(&modulo));
    assert!(!matches!(mod_result.op(), Op::Const(value) if value.0 == ConstValue::Int(1)));

    let x = UOp::var("wrap_x", DType::Int8, 100, 100);
    let y = UOp::var("wrap_y", DType::Int8, 2, 2);
    assert!(matches!(division_dsl_patterns().rewrite(&x.mul(&y).floor_div(&y), &mut ()), RewriteResult::NoMatch));
}

#[test]
fn typed_divmod_guards_cover_zero_and_integer_boundaries() {
    let zero = UOp::var("zero", DType::Int8, 0, 0);
    assert!(matches!(symbolic_simple().rewrite(&zero.floor_div(&zero), &mut ()), RewriteResult::NoMatch));
    assert!(matches!(symbolic_simple().rewrite(&zero.mod_(&zero), &mut ()), RewriteResult::NoMatch));

    let min = UOp::var("min", DType::Int8, i8::MIN as i64, i8::MIN as i64);
    let neg_one = UOp::const_(DType::Int8, ConstValue::Int(-1));
    assert!(matches!(symbolic_simple().rewrite(&min.floor_div(&neg_one), &mut ()), RewriteResult::NoMatch));

    let umax = UOp::var("umax", DType::UInt8, u8::MAX as i64, u8::MAX as i64);
    let two = UOp::const_(DType::UInt8, ConstValue::UInt(2));
    assert!(matches!(
        division_dsl_patterns().rewrite(&umax.mul(&two).floor_div(&two), &mut ()),
        RewriteResult::NoMatch
    ));
}

#[test]
fn typed_division_cancellation_still_fires_when_product_is_exact() {
    for dtype in [
        DType::Int8,
        DType::UInt8,
        DType::WeakInt,
        DType::Index,
        DType::Int8.vec(4).unwrap(),
        DType::UInt8.vec(4).unwrap(),
    ] {
        let x = UOp::var("safe_x", dtype.clone(), 2, 10);
        let y = UOp::var("safe_y", dtype, 2, 3);
        let expression = x.mul(&y).floor_div(&y);
        let RewriteResult::Rewritten(result) = division_dsl_patterns().rewrite(&expression, &mut ()) else {
            panic!("safe typed cancellation did not fire for {}", expression.tree());
        };
        assert!(Arc::ptr_eq(&result, &x));
    }
}

#[test]
fn qr_affine_divmod_congruence_folds_when_typed_arithmetic_is_exact() {
    let x = UOp::var("qr_index", DType::WeakInt, 0, 2);
    let five = UOp::const_(DType::WeakInt, ConstValue::Int(5));
    let numerator = x.mul(&x.const_like(6)).add(&x.const_like(2));

    let modulo = rewrite(symbolic(), numerator.mod_(&five));
    assert!(
        matches!(modulo.op(), Op::Binary(BinaryOp::Add, lhs, rhs)
        if (Arc::ptr_eq(lhs, &x) && matches!(rhs.op(), Op::Const(value) if value.0 == ConstValue::Int(2)))
            || (Arc::ptr_eq(rhs, &x) && matches!(lhs.op(), Op::Const(value) if value.0 == ConstValue::Int(2)))),
        "unexpected modulo replacement: {}",
        modulo.tree()
    );

    let quotient = rewrite(symbolic(), numerator.floor_div(&five));
    assert!(Arc::ptr_eq(&quotient, &x), "unexpected quotient replacement: {}", quotient.tree());
}

/// The congruence fold needs exact host arithmetic over a scalar numerator, so a
/// wrapping dtype, a hardware vector and a broadcast shape all decline — and the
/// vector rows must decline without silently dropping a term.
#[test]
fn affine_divmod_congruence_declines_wrapping_vector_and_broadcast_numerators() {
    let wrapping = UOp::var("qr_wrapping_index", DType::Int8, 20, 21);
    let numerator = wrapping.mul(&wrapping.const_like(6)).add(&wrapping.const_like(2));
    let five = UOp::const_(DType::Int8, ConstValue::Int(5));

    let vector = DType::Int8.vec(4).unwrap();
    let vector_const = |value| UOp::const_(vector.clone(), ConstValue::Int(value));
    let vx = UOp::var("vector_x", vector.clone(), 0, 1);
    let vy = UOp::var("vector_y", vector.clone(), 0, 1);
    let vector_divisor = UOp::const_(vector.clone(), ConstValue::Int(5));

    let scalar = |name| UOp::var(name, DType::Int8, 0, 1);
    let scalar_const = |value| UOp::const_(DType::Int8, ConstValue::Int(value));
    let stacked = UOp::stack(vec![scalar("shape_b0"), scalar("shape_b1")].into());
    let broadcast = scalar("shape_a")
        .mul(&scalar_const(6))
        .add(&stacked.mul(&scalar_const(5)))
        .add(&scalar("shape_d").mul(&scalar_const(11)))
        .mod_(&scalar_const(5));
    assert_eq!(broadcast.shape().unwrap().unwrap().len(), 1, "the broadcast row must stay shaped");

    for expression in [
        numerator.mod_(&five),
        numerator.floor_div(&five),
        vx.mul(&vector_const(6)).add(&vy.mul(&vector_const(2))).mod_(&vector_divisor),
        vx.mul(&vector_const(11)).add(&vy.mul(&vector_const(6))).floor_div(&vector_divisor),
        broadcast,
    ] {
        assert!(
            matches!(advanced_division_dsl_patterns().rewrite(&expression, &mut ()), RewriteResult::NoMatch),
            "congruence fired on {}",
            expression.tree()
        );
    }
}

/// `i64::MAX` coefficients must not overflow the guard's host arithmetic.
#[test]
fn divmod_guards_do_not_overflow_host_arithmetic() {
    let x = UOp::var("x", DType::WeakInt, 0, 1);
    let huge = UOp::const_(DType::WeakInt, ConstValue::Int(i64::MAX));
    let one = UOp::const_(DType::WeakInt, ConstValue::Int(1));
    let expression = x.mod_(&huge).mul(&huge).add(&x.floor_div(&huge).mul(&one));

    let _ = div_mod_recombine_dsl_patterns().rewrite(&expression, &mut ());
}

/// Variables for the recombine cases ported from tinygrad's
/// `test/null/test_uop_symbolic.py` (`test_div_mod_recombine*`,
/// `test_mod_recombine_with_outer_mul`, `test_reshape_index_roundtrip`).
struct RecombineVars {
    x: Arc<UOp>,
    y: Arc<UOp>,
}

impl RecombineVars {
    /// `None` builds the ranged variables the rule rewrites; `Some(point)` pins
    /// them so [`eval_closed_typed`] can check the identity at that point.
    fn new(point: Option<(i64, i64)>) -> Self {
        let ((x_lo, x_hi), (y_lo, y_hi)) = match point {
            Some((x, y)) => ((x, x), (y, y)),
            None => ((0, 150_527), (0, 124)),
        };
        Self { x: UOp::var("x", DType::WeakInt, x_lo, x_hi), y: UOp::var("y", DType::WeakInt, y_lo, y_hi) }
    }

    fn c(&self, value: i64) -> Arc<UOp> {
        self.x.const_like(value)
    }
}

type RecombineTerm = fn(&RecombineVars) -> Arc<UOp>;

// full recombine: q == b//div  ->  b*mul
#[test_case(|v| v.x.mod_(&v.c(4)).add(&v.x.floor_div(&v.c(4)).mul(&v.c(4))), |v| v.x.clone() ; "mod plus scaled quotient")]
#[test_case(|v| v.x.floor_div(&v.c(4)).mul(&v.c(4)).add(&v.x.mod_(&v.c(4))), |v| v.x.clone() ; "scaled quotient plus mod")]
#[test_case(|v| v.y.add(&v.x.mod_(&v.c(4))).add(&v.x.floor_div(&v.c(4)).mul(&v.c(4))), |v| v.x.add(&v.y) ; "trailing quotient after an unrelated term")]
#[test_case(|v| v.y.add(&v.x.floor_div(&v.c(4)).mul(&v.c(4))).add(&v.x.mod_(&v.c(4))), |v| v.x.add(&v.y) ; "trailing mod after an unrelated term")]
#[test_case(|v| v.y.add(&v.x.floor_div(&v.c(4)).mul(&v.c(8))).add(&v.x.mod_(&v.c(4)).mul(&v.c(2))), |v| v.x.mul(&v.c(2)).add(&v.y) ; "scaled pair after an unrelated term")]
#[test_case(|v| v.y.add(&v.x.mod_(&v.c(4)).mul(&v.c(2))).add(&v.x.floor_div(&v.c(4)).mul(&v.c(8))), |v| v.x.mul(&v.c(2)).add(&v.y) ; "scaled mod then quotient after an unrelated term")]
#[test_case(|v| v.x.floor_div(&v.c(2)).mod_(&v.c(4)).add(&v.x.floor_div(&v.c(8)).mul(&v.c(4))), |v| v.x.floor_div(&v.c(2)) ; "merged quotient of a divided base")]
#[test_case(|v| v.x.mul(&v.c(19)).add(&v.c(3)).mod_(&v.c(7)).add(&v.x.mul(&v.c(19)).add(&v.c(3)).floor_div(&v.c(7)).mul(&v.c(7))), |v| v.x.mul(&v.c(19)).add(&v.c(3)) ; "coefficient larger than the divisor")]
#[test_case(|v| v.x.floor_div(&v.c(3)).mod_(&v.c(224)).mul(&v.c(3)).add(&v.x.mod_(&v.c(3))).add(&v.x.floor_div(&v.c(672)).mul(&v.c(672))), |v| v.x.clone() ; "three level ladder")]
#[test_case(|v| v.x.floor_div(&v.c(11)).mod_(&v.c(7)).mul(&v.c(11)).add(&v.x.mod_(&v.c(11))).add(&v.x.floor_div(&v.c(77)).mul(&v.c(77))), |v| v.x.clone() ; "three level ladder other shape")]
#[test_case(|v| v.x.floor_div(&v.c(7)).mod_(&v.c(6)).mul(&v.c(14)).add(&v.x.floor_div(&v.c(42)).mul(&v.c(84))), |v| v.x.floor_div(&v.c(7)).mul(&v.c(14)) ; "three level ladder keeping the outer scale")]
#[test_case(|v| v.x.floor_div(&v.c(3)).add(&v.c(1)).mod_(&v.c(4)).add(&v.x.add(&v.c(3)).floor_div(&v.c(12)).mul(&v.c(4))), |v| v.x.floor_div(&v.c(3)).add(&v.c(1)) ; "offset merged quotient")]
#[test_case(|v| v.x.add(&v.c(1)).mod_(&v.c(3)).add(&v.x.add(&v.c(1)).floor_div(&v.c(3)).add(&v.c(-17)).mul(&v.c(3))), |v| v.x.add(&v.c(1)).add(&v.c(-51)) ; "shifted quotient folds the shift into the result")]
#[test_case(|v| v.x.floor_div(&v.c(8)).mul(&v.c(4)).add(&v.y).add(&v.x.floor_div(&v.c(2)).mod_(&v.c(4))), |v| v.x.floor_div(&v.c(2)).add(&v.y) ; "partners separated inside an additive sum")]
#[test_case(|v| v.x.mul(&v.c(8)).add(&v.y).floor_div(&v.c(4)).mul(&v.c(4)).add(&v.x.mul(&v.c(8)).add(&v.y).mod_(&v.c(4))), |v| v.x.mul(&v.c(8)).add(&v.y) ; "reshape index roundtrip")]
// partial recombine: q == (b//div)%d  ->  (b%(div*d))*mul
#[test_case(|v| v.x.mod_(&v.c(4)).mul(&v.c(3)).add(&v.x.floor_div(&v.c(4)).mod_(&v.c(2)).mul(&v.c(12))), |v| v.x.mod_(&v.c(8)).mul(&v.c(3)) ; "partial widening with an outer mul")]
#[test_case(|v| v.x.mod_(&v.c(4)).mul(&v.c(-2)).add(&v.x.floor_div(&v.c(4)).mod_(&v.c(2)).mul(&v.c(-8))), |v| v.x.mod_(&v.c(8)).mul(&v.c(-2)) ; "partial widening with a negative outer mul")]
#[test_case(|v| v.x.mod_(&v.c(-3)).add(&v.x.floor_div(&v.c(-3)).mod_(&v.c(5)).mul(&v.c(-3))), |v| v.x.mod_(&v.c(-15)) ; "partial widening with a negative divisor")]
#[test_case(|v| v.x.floor_div(&v.c(3)).mod_(&v.c(4)).mul(&v.c(2)).add(&v.x.floor_div(&v.c(12)).mod_(&v.c(5)).mul(&v.c(8))), |v| v.x.floor_div(&v.c(3)).mod_(&v.c(20)).mul(&v.c(2)) ; "partial widening through a merged quotient")]
#[test_case(|v| v.x.floor_div(&v.c(2)).mod_(&v.c(4)).mul(&v.c(2)).add(&v.x.mod_(&v.c(2))), |v| v.x.mod_(&v.c(8)) ; "partial widening recomposing a low order remainder")]
// the padded-conv ladder: the `x//c*c` and `x%c` partners separated by a third term
#[test_case(|v| v.x.floor_div(&v.c(14)).mod_(&v.c(14)).mul(&v.c(14)).add(&v.y).add(&v.x.floor_div(&v.c(196)).mod_(&v.c(512)).mul(&v.c(196))), |v| v.x.floor_div(&v.c(14)).mod_(&v.c(7168)).mul(&v.c(14)).add(&v.y) ; "padded conv ladder with a separating term")]
// declines
#[test_case(|v| v.x.mod_(&v.c(4)).add(&v.x.floor_div(&v.c(5)).mul(&v.c(4))), |v| v.x.mod_(&v.c(4)).add(&v.x.floor_div(&v.c(5)).mul(&v.c(4))) ; "declines when the two divisors differ")]
#[test_case(|v| v.x.floor_div(&v.c(3)).mod_(&v.c(224)).mul(&v.c(3)).add(&v.x.floor_div(&v.c(600)).mul(&v.c(600))), |v| v.x.floor_div(&v.c(3)).mod_(&v.c(224)).mul(&v.c(3)).add(&v.x.floor_div(&v.c(600)).mul(&v.c(600))) ; "declines when the merged divisor mismatches")]
#[test_case(|v| v.x.floor_div(&v.c(3)).mod_(&v.c(224)).mul(&v.c(3)).add(&v.x.floor_div(&v.c(672)).mul(&v.c(700))), |v| v.x.floor_div(&v.c(3)).mod_(&v.c(224)).mul(&v.c(3)).add(&v.x.floor_div(&v.c(672)).mul(&v.c(700))) ; "declines when the partner scale mismatches")]
#[test_case(|v| v.x.floor_div(&v.c(-3)).mod_(&v.c(-2)).add(&v.x.floor_div(&v.c(6)).mul(&v.c(-2))), |v| v.x.floor_div(&v.c(-3)).mod_(&v.c(-2)).add(&v.x.floor_div(&v.c(6)).mul(&v.c(-2))) ; "declines the unsound negative merged quotient")]
fn div_mod_recombine_matches_tinygrad(input: RecombineTerm, expected: RecombineTerm) {
    let ranged = RecombineVars::new(None);
    let rewritten = rewrite(div_mod_recombine_dsl_patterns(), input(&ranged));
    let expected_uop = expected(&ranged);
    assert!(Arc::ptr_eq(&rewritten, &expected_uop), "got {:?}, want {:?}", rewritten.op(), expected_uop.op());

    for point in [(0, 0), (1, 3), (7, 5), (223, 17), (4095, 124), (150_527, 61)] {
        let pinned = RecombineVars::new(Some(point));
        let (lhs, rhs) = (eval_closed_typed(&input(&pinned)), eval_closed_typed(&expected(&pinned)));
        assert!(lhs.is_some(), "input did not evaluate at {point:?}");
        assert_eq!(lhs, rhs, "identity broken at {point:?}");
    }
}

/// Two ranged variables for the congruence rows.
struct CongruenceVars {
    a: Arc<UOp>,
    b: Arc<UOp>,
}

impl CongruenceVars {
    fn new((a_lo, a_hi): (i64, i64), (b_lo, b_hi): (i64, i64)) -> Self {
        Self { a: UOp::var("a", DType::WeakInt, a_lo, a_hi), b: UOp::var("b", DType::WeakInt, b_lo, b_hi) }
    }

    fn c(&self, value: i64) -> Arc<UOp> {
        self.a.const_like(value)
    }
}

type CongruenceTerm = fn(&CongruenceVars) -> Arc<UOp>;

/// `fold_divmod_congruence` (`uop/divandmod.py:38-48`) carries no numerator sign
/// guard and searches both signs of every coefficient's remainder. The first two
/// rows are tinygrad's `test_floordiv_factor_nest_negative_numerator` and
/// `test_floordiv_gcd_with_remainder_negative_numerator`
/// (`test/null/test_uop_symbolic.py:573-582`); the last three need the negative
/// representative, which is only reachable through `rem_choices`.
#[test_case((-10, 10), (0, 3), |v| v.a.mul(&v.c(4)).add(&v.b).floor_div(&v.c(12)), |v| v.a.floor_div(&v.c(3)) ; "factor nest over a negative numerator")]
#[test_case((-1, 5), (0, 0), |v| v.a.mul(&v.c(2)).add(&v.c(7)).floor_div(&v.c(8)), |v| v.a.add(&v.c(3)).floor_div(&v.c(4)) ; "gcd with remainder over a negative numerator")]
#[test_case((2, 3), (0, 0), |v| v.a.mul(&v.c(2)).mod_(&v.c(6)), |v| v.a.mul(&v.c(-4)).add(&v.c(12)) ; "mod that needs the lone term negative remainder")]
#[test_case((2, 3), (0, 0), |v| v.a.mul(&v.c(2)).floor_div(&v.c(6)), |v| v.a.add(&v.c(-2)) ; "quotient that needs the lone term negative remainder")]
#[test_case((1, 2), (0, 1), |v| v.a.mul(&v.c(2)).add(&v.b.mul(&v.c(4))).mod_(&v.c(4)), |v| v.a.mul(&v.c(-2)).add(&v.c(4)) ; "mod that needs the tie break negative remainder")]
fn divmod_congruence_matches_tinygrad(a: (i64, i64), b: (i64, i64), input: CongruenceTerm, expected: CongruenceTerm) {
    let ranged = CongruenceVars::new(a, b);
    let folded = rewrite(symbolic(), input(&ranged));
    let expected_uop = expected(&ranged);
    assert!(Arc::ptr_eq(&folded, &expected_uop), "got {}, want {}", folded.tree(), expected_uop.tree());

    for point_a in a.0..=a.1 {
        for point_b in b.0..=b.1 {
            let pinned = CongruenceVars::new((point_a, point_a), (point_b, point_b));
            let evaluated = eval_closed_typed(&input(&pinned));
            assert!(evaluated.is_some(), "input did not evaluate at ({point_a}, {point_b})");
            assert_eq!(evaluated, eval_closed_typed(&expected(&pinned)), "identity broken at ({point_a}, {point_b})");
        }
    }
}

/// `(x + c)//d -> (x + c%d)//d + c//d` (`uop/divandmod.py:102-105`): "split the
/// multiple of d out of the const, holds for any d!=0". `c` is split with the
/// floor-semantics pair `(c.rem_euclid(d), c.div_euclid(d))` and upstream carries
/// no sign guard, so a negative `c` and a numerator that crosses zero both fold.
/// `None` marks the rows where upstream's `c.val%d.val==c.val` declines because
/// the const is already the reduced representative.
#[test_case(0, 224, -15, 14, Some((13, -2)) ; "negative const over the resnet conv index")]
#[test_case(0, 10, 17, 5, Some((2, 3)) ; "const larger than the divisor")]
#[test_case(-10, 10, -1, 4, Some((3, -1)) ; "numerator that crosses zero")]
#[test_case(-5, 5, -15, 7, Some((6, -3)) ; "negative const and a crossing numerator")]
#[test_case(0, 10, -9, -4, Some((-1, 2)) ; "negative divisor")]
#[test_case(0, 100, 28, 14, Some((0, 2)) ; "const that is an exact multiple of the divisor")]
#[test_case(0, 100, 3, 14, None ; "const already reduced")]
#[test_case(-10, 10, 0, 7, None ; "zero const")]
#[test_case(0, 10, -1, -4, None ; "negative const already reduced for a negative divisor")]
fn const_offset_split_matches_tinygrad(x_min: i64, x_max: i64, c: i64, d: i64, split: Option<(i64, i64)>) {
    let build = |lo, hi| {
        let x = UOp::var("split_x", DType::WeakInt, lo, hi);
        let input = x.add(&x.const_like(c)).floor_div(&x.const_like(d));
        (x, input)
    };

    let (x, input) = build(x_min, x_max);
    let Some((rem, quo)) = split else {
        assert!(
            matches!(range_based_mod_div_patterns().rewrite(&input, &mut ()), RewriteResult::NoMatch),
            "reduced const was split again: {}",
            input.tree()
        );
        return;
    };
    assert_eq!(rem + quo * d, c, "row is not a valid (r, q) split of c");

    let split_of = |x: &Arc<UOp>| x.add(&x.const_like(rem)).floor_div(&x.const_like(d)).add(&x.const_like(quo));
    let RewriteResult::Rewritten(folded) = range_based_mod_div_patterns().rewrite(&input, &mut ()) else {
        panic!("const offset split did not fire for {}", input.tree());
    };
    let expected = split_of(&x);
    assert!(Arc::ptr_eq(&folded, &expected), "got {}, want {}", folded.tree(), expected.tree());

    for point in x_min..=x_max {
        let (pinned_x, pinned_input) = build(point, point);
        let evaluated = eval_closed_typed(&pinned_input);
        assert!(evaluated.is_some(), "input did not evaluate at {point}");
        assert_eq!(evaluated, eval_closed_typed(&split_of(&pinned_x)), "identity broken at {point}");
    }
}

#[test]
fn signed_floor_division_rewrites_keep_negative_cases_exact() {
    let i8_const = |value| UOp::const_(DType::Int8, ConstValue::Int(value));

    let x = UOp::var("comparison_x", DType::Int8, -1, 1);
    let comparison = x.floor_div(&i8_const(3)).lt(&i8_const(0));
    let RewriteResult::Rewritten(lifted) = comparison_dsl_patterns().rewrite(&comparison, &mut ()) else {
        panic!("positive-divisor comparison should lift");
    };
    assert!(matches!(lifted.op(), Op::Binary(BinaryOp::Lt, lhs, rhs)
        if Arc::ptr_eq(lhs, &x) && matches!(rhs.op(), Op::Const(value) if value.0 == ConstValue::Int(0))));

    // `(-128 // -9) // -2` has a single-bucket quotient, so tinygrad's
    // cancel_divmod (`uop/divandmod.py:13`) folds it to the exact constant. The
    // unsound `(a//b)//c -> a//(b*c)` reassociation stays rejected for c < 0.
    let nested = i8_const(-128).floor_div(&i8_const(-9)).floor_div(&i8_const(-2));
    let RewriteResult::Rewritten(folded) = advanced_division_dsl_patterns().rewrite(&nested, &mut ()) else {
        panic!("single-bucket quotient should fold");
    };
    assert_eq!(eval_closed_typed(&nested), Some(ConstValue::Int(-7)));
    assert!(matches!(folded.op(), Op::Const(value) if value.0 == ConstValue::Int(-7)));

    let recombine = i8_const(-20)
        .floor_div(&i8_const(-9))
        .mod_(&i8_const(-2))
        .add(&i8_const(-20).floor_div(&i8_const(18)).mul(&i8_const(-2)));
    assert!(matches!(div_mod_recombine_dsl_patterns().rewrite(&recombine, &mut ()), RewriteResult::NoMatch));
    assert_eq!(eval_closed_typed(&recombine), Some(ConstValue::Int(4)));
}

/// The exactness probe divides by the coefficient, so it must not trap.
#[test]
fn exact_division_probe_declines_signed_min_over_neg_one() {
    let min = UOp::const_(DType::Int64, ConstValue::Int(i64::MIN));
    let zero = UOp::var("zero", DType::Int64, 0, 0);
    let expression = min.add(&zero).floor_div(&UOp::const_(DType::Int64, ConstValue::Int(-1)));
    assert!(matches!(advanced_division_dsl_patterns().rewrite(&expression, &mut ()), RewriteResult::NoMatch));
}

// ============================================================================
// INVALID propagation
// ============================================================================

type InnerCheck = fn(&Arc<UOp>) -> bool;

/// `propagate_invalid` pushes the operation inside the gate and re-types the
/// INVALID lane to the result dtype (tinygrad `uop/symbolic.py:29-38`).
#[test_case(|v| v.where_(&v.p, UOp::var("f16", DType::Float16, 0, 100), UOp::invalid_marker()).cast(DType::Float32),
    |value| matches!(value.op(), Op::Cast(..)) ; "through a cast")]
#[test_case(|v| v.where_(&v.p, v.bounded.clone(), UOp::invalid_marker()).lt(&v.f(1.0)),
    |value| matches!(value.op(), Op::Binary(BinaryOp::Lt, ..)) ; "through a comparison")]
#[test_case(|v| v.where_(&v.p, UOp::var("ix", DType::Index, 0, 100), UOp::invalid_marker()).neg(),
    |value| matches!(value.op(), Op::Binary(BinaryOp::Mul, ..)) ; "through a negation")]
fn propagate_invalid_keeps_the_gate_around_the_operation(build: Term, inner: InnerCheck) {
    let vars = Vars::new();
    let result = rewrite(propagate_invalid(), build(&vars));

    let Op::Ternary(TernaryOp::Where, condition, value, invalid) = result.op() else {
        panic!("expected a gated result, got: {}", result.tree());
    };
    assert!(Arc::ptr_eq(condition, &vars.p));
    assert!(inner(value), "unexpected gated value: {}", result.tree());
    assert!(UOp::is_invalid_marker(invalid));
    assert_eq!(invalid.dtype(), DType::Bool, "the marker keeps its own dtype");
}

/// A bare INVALID poisons a non-comparison binary from either side, but a
/// comparison keeps it as an operand (tinygrad `uop/symbolic.py:75-77`). INVALID
/// only reaches an operand slot through source reconstruction, so the poisoned
/// nodes are built directly rather than through the promoting constructors.
#[test]
fn a_bare_invalid_operand_poisons_arithmetic_but_not_a_comparison() {
    let index = UOp::var("i", DType::Index, 0, 100);
    let marker = UOp::invalid_marker();
    let binary = |op, lhs: &Arc<UOp>, rhs: &Arc<UOp>| UOp::new(Op::Binary(op, lhs.clone(), rhs.clone()), DType::Index);

    for poisoned in [binary(BinaryOp::Sub, &index, &marker), binary(BinaryOp::Sub, &marker, &index)] {
        assert!(UOp::is_invalid_marker(&rewrite(propagate_invalid(), poisoned)));
    }
    let compared = UOp::new(Op::Binary(BinaryOp::Lt, index, marker), DType::Bool);
    assert!(matches!(rewrite(propagate_invalid(), compared).op(), Op::Binary(BinaryOp::Lt, _, _)));
}

#[test]
fn remove_invalid_replaces_a_typed_lane_with_zero() {
    let one = UOp::const_(DType::Float16, ConstValue::Float(1.0));
    let result = rewrite(pm_remove_invalid(), UOp::stack(vec![UOp::invalid_marker(), one].into()));

    assert!(!result.any_in_subtree(UOp::is_invalid_marker));
    let Op::Stack(ops::Stack { sources }) = result.op() else { panic!("expected VECTORIZE, got: {}", result.tree()) };
    assert!(matches!(sources[0].op(), Op::Const(cv) if cv.0 == ConstValue::Float(0.0)));
}

// ============================================================================
// `c0 * x < c1` ceiling division (weak integers only)
// ============================================================================

/// `Some((negate_lhs, bound))` is the lifted `x < bound` — with `x` negated when
/// the coefficient is — and `None` means the rule declines.
#[test_case(3, 10, Some((false, 4)) ; "positive coefficient")]
#[test_case(-3, 10, Some((true, 4)) ; "negative coefficient")]
#[test_case(-3, -10, Some((true, -3)) ; "negative coefficient and bound")]
#[test_case(1, 10, None ; "unit coefficient")]
#[test_case(-1, 10, None ; "negative unit coefficient")]
fn weak_mul_lt_lifts_only_a_nonunit_coefficient(c0: i64, c1: i64, expect: Option<(bool, i64)>) {
    let x = UOp::var("x", DType::WeakInt, -100, 100);
    let weak = |value| UOp::const_(DType::WeakInt, ConstValue::Int(value));
    let lt = weak(c0).mul(&x).lt(&weak(c1));

    let Some((negated, bound)) = expect else {
        assert!(matches!(comparison_dsl_patterns().rewrite(&lt, &mut ()), RewriteResult::NoMatch));
        return;
    };
    let RewriteResult::Rewritten(result) = comparison_dsl_patterns().rewrite(&lt, &mut ()) else {
        panic!("expected a ceil-div comparison simplification");
    };
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = result.op() else { panic!("expected Lt, got {}", result.tree()) };
    if negated {
        assert!(
            matches!(lhs.op(), Op::Binary(BinaryOp::Mul, value, c) if Arc::ptr_eq(value, &x) && matches!(c.op(), Op::Const(cv) if cv.0 == ConstValue::Int(-1)))
        );
    } else {
        assert!(Arc::ptr_eq(lhs, &x));
    }
    assert_eq!(rhs.dtype(), DType::WeakInt);
    assert!(matches!(rhs.op(), Op::Const(cv) if cv.0 == ConstValue::Int(bound)), "{}", result.tree());
}

/// The rule is weak-integer only: a concrete integer keeps the multiplication.
#[test]
fn mul_lt_ceil_div_declines_a_concrete_integer() {
    let x = UOp::var("x", DType::Int32, -100, 100);
    let lt = UOp::native_const(3i32).mul(&x).lt(&UOp::native_const(10i32));
    assert!(matches!(comparison_dsl_patterns().rewrite(&lt, &mut ()), RewriteResult::NoMatch));
}

// ============================================================================
// Validity simplification
// ============================================================================

#[test]
fn lower_bound_clauses_use_the_bounds_minimum() {
    let range = UOp::range_const(20, 0);
    let begin = UOp::var("begin", DType::WeakInt, 2, 9);
    let ne_form = range.lt(&begin).ne(&UOp::native_const(true));
    let not_form = range.lt(&begin).not();

    for clause in [ne_form, not_form] {
        assert_eq!(parse_valid(&clause).map(|(_, upper, bound)| (upper, bound)), Some((false, 2)));
    }
}

#[test]
fn simplify_valid_deduplicates_clauses_without_growing_the_tree() {
    let always = UOp::native_const(true);
    let deduplicated = simplify_valid(&always.and_(&always));
    assert!(deduplicated.is_some_and(|result| Arc::ptr_eq(&result, &always)), "duplicate clauses must collapse");

    let x = UOp::range_const(20, 0);
    let redundant = x.lt(&UOp::index_const(10)).and_(&x.lt(&UOp::index_const(5)));
    if let Some(simplified) = simplify_valid(&redundant) {
        assert!(simplified.node_count() <= redundant.node_count(), "{}", simplified.tree());
    }
}

#[test]
fn uop_given_valid_does_not_leak_fake_params() {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let valid = x.lt(&UOp::native_const(10i32));
    let result = uop_given_valid(&valid, &x.add(&UOp::native_const(1i32)), false);

    assert!(!result.toposort().iter().any(|node| {
        matches!(node.op(), Op::Param(ops::Param { arg, .. }) if arg.name.as_deref().is_some_and(|name| name.starts_with("fake")))
    }));
}

/// `if any(X not in uop.backward_slice_with_self for X,_ in candidate): continue`
/// (tinygrad/uop/symbolic.py:341) — a candidate the uop never mentions is skipped
/// before any substitution, and the uop comes back untouched.
#[test_case(true ; "candidate in the slice rewrites")]
#[test_case(false ; "candidate outside the slice is skipped")]
fn uop_given_valid_only_substitutes_candidates_in_the_slice(in_slice: bool) {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let y = UOp::var("y", DType::Int32, 0, 100);
    let valid = x.lt(&UOp::native_const(10i32));
    // `x < 10` makes `x < 50` true; `y < 50` is not decided by it.
    let expression = if in_slice { &x } else { &y }.lt(&UOp::native_const(50i32));

    let result = uop_given_valid(&valid, &expression, true);

    assert_eq!(!Arc::ptr_eq(&result, &expression), in_slice, "got {}", result.tree());
}

/// An AND clause is dropped only when its ranges do not reach the gated
/// expression; dropping every clause would erase the gate.
#[test_case(1, false ; "a lone clause is left alone")]
#[test_case(2, false ; "two clauses over the same range are both relevant")]
#[test_case(2, true ; "a clause over an unrelated range is dropped")]
fn drop_and_clauses_removes_only_the_irrelevant_ones(clauses: usize, irrelevant: bool) {
    let r0 = UOp::range_const(10, 0);
    let r1 = UOp::range_const(20, 1);
    let mut condition = r0.lt(&UOp::index_const(5));
    if clauses == 2 {
        let second = if irrelevant { r1.lt(&UOp::index_const(15)) } else { r0.lt(&UOp::index_const(8)) };
        condition = condition.and_(&second);
    }
    let gated = UOp::try_where(condition, r0.add(&UOp::index_const(1)), UOp::invalid_marker()).unwrap();

    let result = rewrite(pm_drop_and_clauses(), gated.clone());

    assert_eq!(!Arc::ptr_eq(&result, &gated), irrelevant, "{}", result.tree());
    assert!(matches!(result.op(), Op::Ternary(TernaryOp::Where, ..)), "the gate must survive: {}", result.tree());
}

#[test]
fn substitute_gated_replaces_only_the_mapped_nodes() {
    use std::collections::HashMap;
    use svod_ir::UOpKey;

    let r0 = UOp::range_const(10, 0);
    let r1 = UOp::range_const(20, 1);
    let replacement = UOp::index_const(42);

    let map = HashMap::from([(UOpKey(r0.clone()), replacement.clone())]);
    let result = r0.add(&r1).substitute_gated(&map);
    let Op::Binary(BinaryOp::Add, lhs, rhs) = result.op() else { panic!("expected Add, got {}", result.tree()) };
    assert!(Arc::ptr_eq(lhs, &replacement) || Arc::ptr_eq(rhs, &replacement));
    assert!(Arc::ptr_eq(lhs, &r1) || Arc::ptr_eq(rhs, &r1));

    let empty: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    assert!(Arc::ptr_eq(&r0.substitute_gated(&empty), &r0));
}

/// Sound bounds exist only for nodes whose value the analysis can bound: an AND
/// with a non-constant mask and a LOAD both report nothing.
#[test]
fn sound_vmin_vmax_reports_bounds_only_for_analyzable_nodes() {
    let range = UOp::range_const(10, 0);
    let wide = UOp::range_const(100, 0).cast(DType::Int32);
    let buffer = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();

    let cases = [
        (UOp::native_const(42i32), Some((42, 42))),
        (range.clone(), Some((0, 9))),
        (range.add(&UOp::index_const(5)), Some((5, 14))),
        (UOp::index_const(3).add(&range), Some((3, 12))),
        (wide.and_(&UOp::native_const(7i32)), Some((0, 7))),
        (wide.and_(&UOp::range_const(50, 1).cast(DType::Int32)), None),
        (UOp::load().index(index).call(), None),
    ];
    for (node, expect) in cases {
        let expect = expect.map(|(lo, hi)| (ConstValue::Int(lo), ConstValue::Int(hi)));
        assert_eq!(compute_sound_vmin_vmax(&node), expect, "{}", node.tree());
    }
}

/// The crate's own DSL-authored symbolic matchers compose and fold through `graph_rewrite`.
#[test]
fn symbolic_dsl_matchers_compose() {
    let matcher = constant_folding_dsl_patterns() + identity_and_zero_patterns();
    let x = UOp::var("a", DType::Int32, 0, i64::MAX);
    let add = UOp::new(Op::Binary(BinaryOp::Add, UOp::native_const(0i32), x.clone()), DType::Int32);
    assert!(Arc::ptr_eq(&graph_rewrite(&matcher, add, &mut ()), &x));
}
