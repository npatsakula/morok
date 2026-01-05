//! Symbolic simplification pattern definitions.
//!
//! Defines the core symbolic simplification patterns for algebraic optimization.
//!
//! This module contains:
//! - Constant folding (const op const → const)
//! - Identity element folding (x + 0 → x, x * 1 → x)
//! - Zero propagation (x * 0 → 0, x & 0 → 0)
//!
//! These patterns are separated from rangeify patterns because they apply
//! universally to any UOp graph, not just during schedule transformation.

use morok_dtype::DType;
use morok_ir::types::{BinaryOp, ConstValue};
use morok_ir::uop::cached_property::CachedProperty;
use morok_ir::uop::comparison_analysis::ComparisonAnalyzer;
use morok_ir::uop::eval::{eval_add_typed, eval_binary_op, eval_mul_typed, eval_sub_typed};
use morok_ir::uop::properties::VminVmaxProperty;
use morok_ir::{IntoUOp, Op, UOp};

use crate::pattern::matcher::PatternMatcher;
use crate::patterns;
use crate::rangeify::indexing::get_const_value;
use crate::symbolic::dce::is_empty_range;

use smallvec::SmallVec;
use std::sync::Arc;
use tracing::trace;

/// Constant folding patterns.
///
/// Folds constant expressions at compile time for unary, binary, and ternary operations.
/// Uses dtype-aware evaluation to ensure results respect type boundaries (e.g., Int32 wraps at 32 bits).
pub fn constant_folding_dsl_patterns() -> PatternMatcher {
    use morok_ir::uop::eval::{eval_binary_op_typed, eval_ternary_op_typed, eval_unary_op_typed};

    patterns! {
        // Unary constant folding - 7 operations in one declaration
        for op in unary [Neg, Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc] {
            op(c @const(c_val))
              => eval_unary_op_typed(op, c_val, c.dtype().base()).map(|r| UOp::const_(c.dtype(), r)),
        },

        // Binary constant folding - 13 operations in one declaration
        for op in binary [Add, Mul, Sub, Mod, Max, Pow, Idiv, Fdiv, And, Or, Xor, Shl, Shr] {
            op(a @const(a_val), b @const(b_val))
              => eval_binary_op_typed(op, a_val, b_val, a.dtype().base()).map(|r| UOp::const_(a.dtype(), r)),
        },

        // Ternary constant folding - 2 operations in one declaration
        for op in ternary [Where, MulAcc] {
            // For Where: use second operand's dtype (true branch)
            // For MulAcc: use first operand's dtype (all same dtype)
            op(a @const(a_val), b @const(b_val), c @const(c_val))
              => eval_ternary_op_typed(op, a_val, b_val, c_val, b.dtype().base()).map(|r| UOp::const_(b.dtype(), r)),
        },
    }
}

/// Identity and zero propagation patterns.
///
/// - Identity folding: x + 0 → x, 0 + x → x, x * 1 → x, 1 * x → x, etc.
/// - Zero propagation: x * 0 → 0, 0 * x → 0, x & 0 → 0, 0 & x → 0
pub fn identity_and_zero_patterns() -> PatternMatcher {
    patterns! {
        // ========== Identity folding (commutative) ==========
        Add[x, @zero] ~> x,
        Mul[x, @one] ~> x,
        Or[x, @zero] ~> x,
        Xor[x, @zero] ~> x,

        // ========== Identity folding (non-commutative) ==========
        Sub(x, @zero) ~> x,
        Idiv(x, @one) ~> x,
        Fdiv(x, @one) ~> x,

        // ========== Zero propagation ==========
        Mul[_, zero @ @zero] ~> zero,
        And[_, zero @ @zero] ~> zero,
    }
}

/// Pattern matcher for simple symbolic simplifications.
///
/// Contains algebraic identities and zero propagation rules:
/// - x + 0 → x, 0 + x → x
/// - x - 0 → x
/// - x * 1 → x, 1 * x → x
/// - x / 1 → x (both Idiv and Fdiv)
/// - x | 0 → x, 0 | x → x
/// - x ^ 0 → x, 0 ^ x → x
/// - x * 0 → 0, 0 * x → 0
/// - x & 0 → 0, 0 & x → 0
pub fn symbolic_simple() -> PatternMatcher {
    constant_folding_dsl_patterns()
        + identity_and_zero_patterns()
        + self_folding_dsl_patterns()
        + zero_folding_dsl_patterns()
        + division_dsl_patterns()
        + cast_dsl_patterns()
        + cast_where_dsl_patterns()
        + term_combining_dsl_patterns()
        + alu_folding_dsl_patterns()
        + advanced_division_dsl_patterns()
        + div_mod_recombine_dsl_patterns()
        + comparison_dsl_patterns()
        + boolean_dsl_patterns()
        + minmax_dsl_patterns()
        + power_dsl_patterns()
        + negation_dsl_patterns()
        + range_based_mod_div_patterns()
        + dce_dsl_patterns()
        + dead_loop_patterns()
}

/// Full symbolic simplification matcher.
///
/// Combines all symbolic patterns for comprehensive algebraic optimization:
/// - Constant folding (unary, binary, ternary ops)
/// - Identity folding (x+0→x, x*1→x)
/// - Zero propagation (x*0→0, x&0→0)
/// - Self-folding (x/x→1, x&x→x)
/// - Division simplification
/// - Cast optimization
/// - Term combining
/// - ALU folding
/// - Comparison patterns
/// - Boolean patterns
/// - Dead code elimination
pub fn symbolic() -> PatternMatcher {
    symbolic_simple()
}

/// Self-folding patterns.
///
/// Patterns where an operand appears twice:
/// - x // x → 1
/// - x // -1 → -x
/// - (x % y) % y → x % y
/// - x & x → x
/// - x | x → x
pub fn self_folding_dsl_patterns() -> PatternMatcher {
    patterns! {
        // x // x → 1
        Idiv(x, x) ~> 1.into_uop(x.dtype()),
        // x // -1 → -x
        Idiv(x, c @const(c_val)) if c_val.is_neg_one() ~> UOp::neg(x),
        // (x % y) % y → x % y
        Mod(Mod(x, y), y) => x.try_mod(y).ok(),
        // x & x → x
        And(x, x) ~> Arc::clone(x),
        // x | x → x
        Or(x, x) ~> Arc::clone(x),
    }
}

/// Zero folding patterns.
///
/// Patterns that fold to zero or false:
/// - x < x → False (non-float only)
/// - x % x → 0
/// - x != x → False (int only)
pub fn zero_folding_dsl_patterns() -> PatternMatcher {
    patterns! {
        // x % x → 0
        Mod(x, x) => x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::zero(dt))),
        // x < x → False (non-float only)
        Lt(x, x) if !x.dtype().is_float() ~> false.into_uop(DType::Bool),
        // x != x → False (int only)
        Ne(x, x) if x.dtype().is_int() ~> false.into_uop(DType::Bool),
    }
}

/// Range-based modulo and division simplification patterns.
///
/// Uses vmin/vmax analysis to simplify:
/// - x % n → x when 0 <= vmin(x) && vmax(x) < n
/// - x / n → 0 when 0 <= vmin(x) && vmax(x) < n
///
/// This is critical for RESHAPE range propagation where Range(n) % n should simplify to Range(n).
pub fn range_based_mod_div_patterns() -> PatternMatcher {
    patterns! {
        // x % n → x when 0 <= vmin(x) && vmax(x) < n
        // This handles cases like Range(3) % 3 → Range(3)
        Mod(x, n @const(n_val)) => {
            let (vmin, vmax) = VminVmaxProperty::get(x);
            trace!(
                x.id = x.id,
                vmin = ?vmin,
                vmax = ?vmax,
                n_val = ?n_val,
                "Mod simplification check"
            );
            // Check if x is always non-negative and less than n
            if let (ConstValue::Int(min), ConstValue::Int(max), ConstValue::Int(n_int)) = (vmin, vmax, n_val)
                && *min >= 0 && *max < n_int {
                    trace!(
                        n_int,
                        min = *min,
                        max = *max,
                        "Simplifying x % n → x"
                    );
                    return Some(Arc::clone(x));
                }
            None
        },

        // x / n → 0 when 0 <= vmin(x) && vmax(x) < n
        // This handles cases like Range(3) / 3 → 0 (since Range(3) is 0,1,2 and all /3 = 0)
        Idiv(x, n @const(n_val)) => {
            let (vmin, vmax) = VminVmaxProperty::get(x);
            // Check if x is always non-negative and less than n
            if let (ConstValue::Int(min), ConstValue::Int(max), ConstValue::Int(n_int)) = (vmin, vmax, n_val)
                && *min >= 0 && *max < n_int && n_int > 0 {
                    return Some(UOp::const_(x.dtype(), ConstValue::Int(0)));
                }
            None
        },

        // Idiv(add_chain, n) → Idiv(add_chain_minus_const, n) when the constant doesn't affect floor division
        // This handles: Idiv(Add(Add(base, 1), inner), 64) → Idiv(Add(base, inner), 64)
        // when base is aligned to 64 and 1 < 64
        Idiv(x, n @const(n_val)) if matches!(x.op(), Op::Binary(BinaryOp::Add, ..)) => {
            // Extract total constant offset from Add chain
            fn extract_const_sum(uop: &Arc<UOp>) -> (Arc<UOp>, i64) {
                match uop.op() {
                    Op::Binary(BinaryOp::Add, left, right) => {
                        if let Op::Const(cv) = right.op()
                            && let ConstValue::Int(v) = cv.0 {
                                let (inner, inner_sum) = extract_const_sum(left);
                                return (inner, inner_sum + v);
                            }
                        if let Op::Const(cv) = left.op()
                            && let ConstValue::Int(v) = cv.0 {
                                let (inner, inner_sum) = extract_const_sum(right);
                                return (inner, inner_sum + v);
                        }
                        // Recurse into both sides for nested adds
                        let (left_inner, left_sum) = extract_const_sum(left);
                        let (right_inner, right_sum) = extract_const_sum(right);
                        if left_sum != 0 || right_sum != 0 {
                            // Rebuild Add without the extracted constants
                            let new_add = left_inner.try_add(&right_inner).ok();
                            if let Some(rebuilt) = new_add {
                                return (rebuilt, left_sum + right_sum);
                            }
                        }
                        (Arc::clone(uop), 0)
                    }
                    _ => (Arc::clone(uop), 0),
                }
            }

            let (x_without_const, const_sum) = extract_const_sum(x);
            if const_sum == 0 {
                return None;  // No constant to remove
            }

            let (vmin, vmax) = VminVmaxProperty::get(&x_without_const);
            if let (ConstValue::Int(min), ConstValue::Int(max), ConstValue::Int(n_int)) = (vmin, vmax, n_val)
                && n_int > 0 {
                    // Check if adding const_sum doesn't change floor division result
                    let min_div = *min / n_int;
                    let max_div = *max / n_int;

                    // CRITICAL: All values must be in the same bucket for safe removal
                    // If min and max are in different buckets, intermediate values could cross
                    // bucket boundaries when const_sum is added/removed
                    if min_div != max_div {
                        return None;
                    }

                    // Check that adding const_sum keeps values in the same bucket
                    let min_c_div = (*min + const_sum) / n_int;
                    let max_c_div = (*max + const_sum) / n_int;
                    if min_div == min_c_div && max_div == max_c_div {
                        return x_without_const.try_div(&Arc::clone(n)).ok();
                    }
                }
            None
        },

        // (a + (x // n) * n) // n → x // n  when 0 <= vmin(a) and vmax(a) < n
        // This eliminates redundant idiv chains in address calculations
        // Using [] for both Add and Mul to match all permutations
        Idiv(Add[a, Mul[Idiv(x, n @const(n_val)), n]], n) => {
            let (vmin, vmax) = VminVmaxProperty::get(a);
            if let (ConstValue::Int(min), ConstValue::Int(max), ConstValue::Int(n_int)) = (vmin, vmax, n_val)
                && *min >= 0 && *max < n_int && n_int > 0 {
                    return x.try_div(n).ok();
                }
            None
        },
    }
}

/// Division simplification patterns.
///
/// - x / x → 1.0 (float division)
/// - (x * y) / y → x
/// - (x * y) // y → x
pub fn division_dsl_patterns() -> PatternMatcher {
    patterns! {
        // x / x → 1.0 (float division)
        Fdiv(x, x) => x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::one(dt))),
        // (x * y) / y → x
        Fdiv(Mul(x, y), y) ~> Arc::clone(x),
        // (x * y) // y → x
        Idiv(Mul(x, y), y) ~> Arc::clone(x),
    }
}

/// Cast optimization patterns.
///
/// - cast(const) → const (constant folding)
/// - x.cast(dtype) → x if same dtype (noop cast)
/// - x.cast(a).cast(b) → x.cast(b) (collapse double cast)
pub fn cast_dsl_patterns() -> PatternMatcher {
    patterns! {
        // cast(const) → const
        Cast { src: c @const(c_val), dtype } => c_val.cast(&dtype).map(|v| UOp::const_(dtype.clone(), v)),
        // x.cast(dtype) → x if same dtype
        Cast { src: x, dtype } if x.dtype() == dtype ~> Arc::clone(x),
        // x.cast(a).cast(b) → x.cast(b)
        Cast { src: Cast { src: x, .. }, dtype } ~> UOp::cast(Arc::clone(x), dtype.clone()),
    }
}

/// Term combining patterns.
///
/// - x + x → 2*x
/// - (c1 * x) + (c2 * x) → (c1 + c2) * x
/// - (x * c1) + (x * c2) → x * (c1 + c2)
pub fn term_combining_dsl_patterns() -> PatternMatcher {
    patterns! {
        // x + x → 2*x
        Add(x, x) => 2.into_uop(x.dtype()).try_mul(x).ok(),
        // (c1 * x) + (c2 * x) → (c1 + c2) * x
        Add(Mul(c1 @const(c1_val), x), Mul(c2 @const(c2_val), x))
          => eval_add_typed(c1_val, c2_val, c1.dtype().base())?.into_uop(c1.dtype()).try_mul(x).ok(),
        // (x * c1) + (x * c2) → x * (c1 + c2)
        Add(Mul(x, c1 @const(c1_val)), Mul(x, c2 @const(c2_val)))
          => x.try_mul(&eval_add_typed(c1_val, c2_val, c1.dtype().base())?.into_uop(c1.dtype())).ok(),
    }
}

/// Advanced division and distribution patterns.
///
/// - (a // b) // c → a // (b * c)
/// - expr // divisor → expr.divides(divisor) (generic exact division)
/// - (a + b) % c → simplify when one operand is multiple of c
/// - (a + b) // c → (a // c) + (b // c) when both divide evenly
/// - (a - b) // c → (a // c) - (b // c) when both divide evenly
/// - c * (a + b) → (c * a) + (c * b)
/// - (a + b) * c → (a * c) + (b * c)
pub fn advanced_division_dsl_patterns() -> PatternMatcher {
    patterns! {
        // (a // b) // c → a // (b * c) if b,c non-zero
        Idiv(Idiv(a, b @const(b_val)), c @const(c_val)) if !b_val.is_zero() && !c_val.is_zero() => {
            a.try_div(&UOp::const_(b.dtype(), eval_mul_typed(b_val, c_val, b.dtype().base())?)).ok()
        },
        // expr // divisor → expr.divides(divisor) (generic exact division)
        Idiv(expr, divisor @ @const) => expr.divides(divisor),
        // (a + b) % c → simplify when one operand is multiple of c
        Mod(Add(a, b), c @const(c_val)) => {
            let ConstValue::Int(modulus) = c_val else { return None };
            (modulus > 0 && modulus <= 256).then(|| {
                let (af, bf) = (a.const_factor(), b.const_factor());
                if af % modulus == 0 {
                    b.try_mod(c).ok()
                } else if bf % modulus == 0 {
                    a.try_mod(c).ok()
                } else {
                    None
                }
            }).flatten()
        },
        // (a + b) // c → (a // c) + (b // c) when both divide evenly
        Idiv(Add(a, b), c @ @const) => a.divides(c)?.try_add(&b.divides(c)?).ok(),
        // (a - b) // c → (a // c) - (b // c) when both divide evenly
        Idiv(Sub(a, b), c @ @const) => a.divides(c)?.try_sub(&b.divides(c)?).ok(),
        // c * (a + b) → (c * a) + (c * b)
        Mul(c @ @const, Add(a, b)) => c.try_mul(a).ok()?.try_add(&c.try_mul(b).ok()?).ok(),
        // (a + b) * c → (a * c) + (b * c)
        Mul(Add(a, b), c @ @const) => a.try_mul(c).ok()?.try_add(&b.try_mul(c).ok()?).ok(),
    }
}

/// Two-stage ALU folding patterns.
///
/// Fold constants in associative operation chains:
/// - (x + c1) + c2 → x + (c1 + c2)
/// - (x * c1) * c2 → x * (c1 * c2)
/// - (x - c1) + c2 → x + (c2 - c1) or x - (c1 - c2)
/// - (x + c1) - c2 → x + (c1 - c2) or x - (c2 - c1)
/// - (x - c1) - c2 → x - (c1 + c2)
pub fn alu_folding_dsl_patterns() -> PatternMatcher {
    patterns! {
        // (x + c1) + c2 → x + (c1 + c2) - commutative outer Add
        Add[Add[x, c1 @const(c1_val)], c2 @const(c2_val)]
          => x.try_add(&UOp::const_(c1.dtype(), eval_add_typed(c1_val, c2_val, c1.dtype().base())?)).ok(),

        // (x * c1) * c2 → x * (c1 * c2) - commutative outer Mul
        Mul[Mul[x, c1 @const(c1_val)], c2 @const(c2_val)]
          => x.try_mul(&UOp::const_(c1.dtype(), eval_mul_typed(c1_val, c2_val, c1.dtype().base())?)).ok(),

        // (x - c1) + c2 → x + (c2 - c1) or x - (c1 - c2) - commutative outer Add
        Add[Sub(x, c1 @const(c1_val)), c2 @const(c2_val)] => {
            let diff_val = eval_sub_typed(c2_val, c1_val, c1.dtype().base())?;
            // Normalize: prefer x - |c| over x + (-c)
            if let ConstValue::Int(v) = diff_val && v < 0 {
                x.try_sub(&(-v).into_uop(c1.dtype())).ok()
            } else {
                x.try_add(&UOp::const_(c1.dtype(), diff_val)).ok()
            }
        },

        // (x + c1) - c2 → x + (c1 - c2) or x - (c2 - c1) when result is negative
        Sub(Add(x, c1 @const(c1_val)), c2 @const(c2_val)) => {
            let diff_val = eval_sub_typed(c1_val, c2_val, c1.dtype().base())?;
            // Normalize: prefer x - |c| over x + (-c)
            if let ConstValue::Int(v) = diff_val && v < 0 {
                return x.try_sub(&(-v).into_uop(c1.dtype())).ok();
            }
            x.try_add(&UOp::const_(c1.dtype(), diff_val)).ok()
        },

        // (x - c1) - c2 → x - (c1 + c2)
        Sub(Sub(x, c1 @const(c1_val)), c2 @const(c2_val))
          => x.try_sub(&UOp::const_(c1.dtype(), eval_add_typed(c1_val, c2_val, c1.dtype().base())?)).ok(),
    }
}

/// Dead loop elimination patterns.
///
/// - RANGE with vmax ≤ 0 → Const(0)
/// - END with dead ranges → remove dead ranges
/// - REDUCE with all empty ranges → identity element
pub fn dead_loop_patterns() -> PatternMatcher {
    use crate::symbolic::dce::reduce_identity;

    /// Check if END has any dead ranges (for guard).
    fn has_dead_ranges(end_op: &Arc<UOp>) -> bool {
        if let Op::End { ranges, .. } = end_op.op() { ranges.iter().any(is_empty_range) } else { false }
    }

    /// Check if all REDUCE ranges are empty (for guard).
    fn all_ranges_empty(reduce_op: &Arc<UOp>) -> bool {
        if let Op::Reduce { ranges, .. } = reduce_op.op() { ranges.iter().all(is_empty_range) } else { false }
    }

    /// Filter dead ranges from END, or unwrap if all dead.
    fn filter_dead_ranges(end_op: &Arc<UOp>) -> Arc<UOp> {
        let Op::End { computation, ranges } = end_op.op() else { unreachable!("filter_dead_ranges called on non-End") };

        let live_ranges: SmallVec<[Arc<UOp>; 4]> = ranges.iter().filter(|r| !is_empty_range(r)).cloned().collect();

        if live_ranges.is_empty() {
            // All ranges dead - return computation directly
            Arc::clone(computation)
        } else {
            // Some ranges dead - create new END with only live ranges
            UOp::end(Arc::clone(computation), live_ranges)
        }
    }

    /// Get identity element for REDUCE with all empty ranges.
    fn reduce_to_identity(reduce_op: &Arc<UOp>) -> Arc<UOp> {
        let Op::Reduce { reduce_op: op, .. } = reduce_op.op() else {
            unreachable!("reduce_to_identity called on non-Reduce")
        };
        reduce_identity(*op, reduce_op.dtype())
    }

    patterns! {
        // RANGE with vmax ≤ 0 → Const(0)
        r @ Range(_) if is_empty_range(r) ~> UOp::index_const(0),

        // END with dead ranges → filter or unwrap
        end_op @ End(_, ..) if has_dead_ranges(end_op) ~> filter_dead_ranges(end_op),

        // REDUCE with all empty ranges → identity element
        reduce_op @ Reduce(_, ..) if all_ranges_empty(reduce_op) ~> reduce_to_identity(reduce_op),
    }
}

/// Dead code elimination patterns.
///
/// Handles WHERE optimizations:
/// - WHERE(true, t, f) → t
/// - WHERE(false, t, f) → f
/// - WHERE(_, t, t) → t (same branches)
/// - WHERE(x, true, false) → x (bool)
/// - WHERE(x, false, true) → !x (bool)
/// - WHERE(!cond, t, f) → WHERE(cond, f, t)
pub fn dce_dsl_patterns() -> PatternMatcher {
    patterns! {
        // WHERE with constant condition → select appropriate branch
        Where(cond, true_val, false_val) => {
            match VminVmaxProperty::get(cond) {
                (ConstValue::Bool(true), ConstValue::Bool(true)) => Some(Arc::clone(true_val)),
                (ConstValue::Bool(false), ConstValue::Bool(false)) => Some(Arc::clone(false_val)),
                _ => None,
            }
        },

        // WHERE(_, same, same) → same
        Where(_, t, t) ~> Arc::clone(t),

        // WHERE(x, true, false) → x (for bool x)
        Where(x, t @const(t_val), f @const(f_val))
          if x.dtype() == DType::Bool && t_val == ConstValue::Bool(true) && f_val == ConstValue::Bool(false)
          ~> Arc::clone(x),

        // WHERE(x, false, true) → !x (for bool x)
        Where(x, t @const(t_val), f @const(f_val))
          if x.dtype() == DType::Bool && t_val == ConstValue::Bool(false) && f_val == ConstValue::Bool(true)
          ~> x.not(),

        // WHERE(!cond, t, f) → WHERE(cond, f, t) - negated condition swap
        Where(Not(cond), t, f) => UOp::try_where(Arc::clone(cond), Arc::clone(f), Arc::clone(t)).ok(),

        // WHERE(a, WHERE(b, c, d), d) → WHERE(a & b, c, d) - branch merging
        Where(a, Where(b, c, d), d2) if Arc::ptr_eq(d, d2) => {
            let combined_cond = a.try_and_op(b).ok()?;
            UOp::try_where(combined_cond, Arc::clone(c), Arc::clone(d)).ok()
        },
    }
}

/// Cast pushing through WHERE patterns.
///
/// - where(s, a, b).cast(dtype) → where(s, a.cast(dtype), b.cast(dtype))
pub fn cast_where_dsl_patterns() -> PatternMatcher {
    patterns! {
        // cast(where(s, a, b), dtype) → where(s, cast(a, dtype), cast(b, dtype))
        Cast { src: Where(s, a, b), dtype } => {
            let cast_a = UOp::cast(Arc::clone(a), dtype.clone());
            let cast_b = UOp::cast(Arc::clone(b), dtype.clone());
            UOp::try_where(Arc::clone(s), cast_a, cast_b).ok()
        },
    }
}

/// Comparison patterns.
///
/// Handles Lt, Eq, Ne comparisons with:
/// - Self-comparison fast path (x op x)
/// - Constant folding
/// - Range-based analysis via vmin/vmax
/// - Const offset: (c0 + x) < c1 → x < (c1 - c0)
/// - Negation flip: -x < -y → y < x
pub fn comparison_dsl_patterns() -> PatternMatcher {
    patterns! {
        for op in binary [Lt, Eq, Ne] {
            op(x, y) => {
                // 1. Self-comparison fast path (non-float only)
                if Arc::ptr_eq(x, y) && !x.dtype().is_float() {
                    let result = match op {
                        BinaryOp::Lt => ConstValue::Bool(false),
                        BinaryOp::Eq => ConstValue::Bool(true),
                        BinaryOp::Ne => ConstValue::Bool(false),
                        _ => return None,
                    };
                    return Some(UOp::const_(DType::Bool, result));
                }

                // 2. Constant folding
                if let (Some(a_val), Some(b_val)) = (get_const_value(x), get_const_value(y))
                    && let Some(result) = eval_binary_op(op, a_val, b_val)
                {
                    return Some(UOp::const_(DType::Bool, result));
                }

                // 3. Range-based analysis
                if let Some(result) = ComparisonAnalyzer::analyze(op, x, y) {
                    return Some(result.into_uop(DType::Bool));
                }

                None
            }
        },

        // (c0 + x) < c1 → x < (c1 - c0) for integers - commutative
        Lt(Add[c0 @const(c0_val), x], c1 @const(c1_val)) => {
            let diff = eval_sub_typed(c1_val, c0_val, c0.dtype().base())?;
            x.try_cmplt(&UOp::const_(c0.dtype(), diff)).ok()
        },

        // -x < -y → y < x (negation flip for Lt)
        Lt(Neg(x), Neg(y)) => y.try_cmplt(x).ok(),
    }
}

/// Boolean logic patterns.
///
/// - !!x → x (double negation elimination)
/// - x ^ x → 0 (xor self-cancellation)
/// - x | !x → true (tautology)
/// - x & !x → false (contradiction)
/// - true | x → true, false & x → false
/// - true & x → x, false | x → x (identity)
pub fn boolean_dsl_patterns() -> PatternMatcher {
    patterns! {
        // !!x → x
        Not(Not(x)) ~> Arc::clone(x),
        // x ^ x → 0
        Xor(x, x) => x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::zero(dt))),

        // x | !x → true (tautology) - commutative
        Or[x, Not(y)] if Arc::ptr_eq(x, y) && x.dtype() == DType::Bool
          => Some(UOp::const_(DType::Bool, ConstValue::Bool(true))),

        // x & !x → false (contradiction) - commutative
        And[x, Not(y)] if Arc::ptr_eq(x, y) && x.dtype() == DType::Bool
          => Some(UOp::const_(DType::Bool, ConstValue::Bool(false))),

        // true | x → true (commutative)
        Or[t @const(t_val), _] if t_val == ConstValue::Bool(true) ~> Arc::clone(t),

        // false & x → false (commutative)
        And[f @const(f_val), _] if f_val == ConstValue::Bool(false) ~> Arc::clone(f),

        // true & x → x (identity, commutative)
        And[c @const(c_val), x] if c_val == ConstValue::Bool(true) ~> Arc::clone(x),

        // false | x → x (identity, commutative)
        Or[c @const(c_val), x] if c_val == ConstValue::Bool(false) ~> Arc::clone(x),
    }
}

/// Min/max patterns.
///
/// - max(x, x) → x
/// - min(x, x) → x (via Min = negated Max)
pub fn minmax_dsl_patterns() -> PatternMatcher {
    patterns! {
        // max(x, x) → x
        Max(x, x) ~> Arc::clone(x),
    }
}

/// Power patterns.
///
/// - x ** 0 → 1
/// - x ** 1 → x
pub fn power_dsl_patterns() -> PatternMatcher {
    patterns! {
        // x ** 0 → 1
        Pow(x, c @const(c_val)) if c_val.is_zero() => x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::one(dt))),
        // x ** 1 → x
        Pow(x, c @const(c_val)) if c_val.is_one() ~> Arc::clone(x),
    }
}

/// Negation patterns.
///
/// - -(-x) → x (double negation for arithmetic)
pub fn negation_dsl_patterns() -> PatternMatcher {
    patterns! {
        // Double arithmetic negation: -(-x) → x
        Neg(Neg(x)) ~> Arc::clone(x),
    }
}

/// GEP pushing patterns for devectorize pass.
///
/// Push GEP through ALU operations to simplify vector index extraction.
/// Based on Tinygrad's gep_pushing (symbolic.py:153-176).
///
/// - GEP(GEP(x, inner), outer) → GEP(x, composed)
/// - GEP(VECTORIZE(elements), [i]) → elements[i]
/// - GEP(scalar, [0]) → scalar
/// - GEP(VConst([...]), indices) → extracted element(s)
/// - GEP(Binary(op, a, b), indices) → Binary(op, GEP(a), GEP(b))
/// - GEP(Unary(op, x), indices) → Unary(op, GEP(x))
/// - GEP(UNROLL(x, ...), indices) → GEP(x, indices)
/// - GEP(x, [0,1,2,...,n-1]) → x (identity)
pub fn gep_pushing_patterns() -> PatternMatcher {
    /// Check if VECTORIZE is a broadcast (all elements pointer-equal)
    fn is_broadcast(elements: &SmallVec<[Arc<UOp>; 4]>) -> bool {
        elements.first().is_some_and(|first| elements.iter().all(|e| Arc::ptr_eq(e, first)))
    }

    patterns! {
        // 1. GEP composition: GEP(GEP(x, inner), outer) → GEP(x, inner[outer])
        // Note: nested struct patterns require UOp first field, so we extract inner manually
        Gep { vector, indices } if matches!(vector.op(), Op::Gep { .. }) => {
            let Op::Gep { vector: inner_vec, indices: inner_indices } = vector.op() else { return None };
            let composed: Vec<usize> = indices.iter()
                .map(|&o| inner_indices.get(o).copied())
                .collect::<Option<Vec<_>>>()?;
            Some(UOp::gep(Arc::clone(inner_vec), composed))
        },

        // 2. GEP through BROADCAST: extract scalar (MUST be before general VECTORIZE!)
        // BROADCAST = VECTORIZE with all identical elements → GEP([x,x,x,x], [i]) → x
        Gep { vector, indices } if indices.len() == 1 && matches!(vector.op(), Op::Vectorize { .. }) => {
            let Op::Vectorize { elements } = vector.op() else { return None };
            is_broadcast(elements).then(|| elements.first().cloned()).flatten()
        },

        // 3. GEP through VECTORIZE: extract single element
        Gep { vector, indices } if indices.len() == 1 && matches!(vector.op(), Op::Vectorize { .. }) => {
            let Op::Vectorize { elements } = vector.op() else { return None };
            elements.get(indices[0]).cloned()
        },

        // 4. GEP on scalar: GEP(x, [i]) where x is scalar → x
        Gep { vector, indices } if vector.dtype().vcount() == 1 && indices.len() == 1 ~> Arc::clone(vector),

        // 5. GEP through VConst: extract element(s)
        Gep { vector, indices } if matches!(vector.op(), Op::VConst { .. }) => {
            let Op::VConst { values } = vector.op() else { return None };
            let scalar_dtype = vector.dtype().scalar().map(DType::Scalar).unwrap_or(DType::Index);
            if indices.len() == 1 {
                values.get(indices[0]).map(|v| UOp::const_(scalar_dtype.clone(), *v))
            } else {
                let selected: Vec<_> = indices.iter().filter_map(|&i| values.get(i).cloned()).collect();
                (selected.len() == indices.len()).then(|| UOp::vconst(selected))
            }
        },

        // 6. Push GEP through Binary: GEP(Binary(op, a, b), indices) → Binary(op, GEP(a), GEP(b))
        Gep { vector, indices } if !indices.is_empty() && matches!(vector.op(), Op::Binary(..)) => {
            let Op::Binary(bin_op, a, b) = vector.op() else { return None };
            let gep_a = UOp::gep(Arc::clone(a), indices.clone());
            let gep_b = UOp::gep(Arc::clone(b), indices.clone());
            Some(UOp::new(Op::Binary(*bin_op, gep_a.clone(), gep_b), gep_a.dtype()))
        },

        // 7. Push GEP through Unary: GEP(Unary(op, x), indices) → Unary(op, GEP(x))
        Gep { vector, indices } if !indices.is_empty() && matches!(vector.op(), Op::Unary(..)) => {
            let Op::Unary(un_op, x) = vector.op() else { return None };
            let gep_x = UOp::gep(Arc::clone(x), indices.clone());
            Some(UOp::new(Op::Unary(*un_op, gep_x.clone()), gep_x.dtype()))
        },

        // 8. GEP through UNROLL: GEP(UNROLL(x, ...), indices) → GEP(x, indices)
        Gep { vector, indices } if matches!(vector.op(), Op::Unroll { .. }) => {
            let Op::Unroll { src, .. } = vector.op() else { return None };
            Some(UOp::gep(Arc::clone(src), indices.clone()))
        },

        // 9. Identity GEP removal: GEP(x, [0,1,2,...,n-1]) → x
        Gep { vector, indices }
            if indices.iter().enumerate().all(|(i, &idx)| i == idx)
                && indices.len() == vector.dtype().vcount()
            ~> Arc::clone(vector),

        // 10. GEP through PTRCAT: GEP(PTRCAT([a, b, c, d]), [1, 3]) → PTRCAT([b, d])
        Gep { vector, indices } if matches!(vector.op(), Op::PtrCat { .. }) => {
            let Op::PtrCat { sources } = vector.op() else { return None };
            let reordered: Vec<_> = indices.iter().filter_map(|&idx| sources.get(idx).cloned()).collect();
            (reordered.len() == indices.len()).then(|| UOp::ptrcat(reordered))
        },

        // 11. GEP through CAT: GEP(CAT([a, b, c, d]), [1, 3]) → CAT([b, d])
        Gep { vector, indices } if matches!(vector.op(), Op::Cat { .. }) => {
            let Op::Cat { sources } = vector.op() else { return None };
            let reordered: Vec<_> = indices.iter().filter_map(|&idx| sources.get(idx).cloned()).collect();
            (reordered.len() == indices.len()).then(|| UOp::cat(reordered))
        },
    }
}

/// Div/Mod recombination patterns.
///
/// - x%n + (x//n)*n → x (div-mod identity)
/// - (x//n)*n + x%n → x (commutative form)
/// - (a//c1 + c2) // c3 → (a + c1*c2) // (c1*c3) (nested division)
pub fn div_mod_recombine_dsl_patterns() -> PatternMatcher {
    patterns! {
        // x%n + (x//n)*n → x (div-mod identity)
        Add[Mod(x, n), Mul[Idiv(x2, n2), n3]]
          if Arc::ptr_eq(x, x2) && Arc::ptr_eq(n, n2) && Arc::ptr_eq(n, n3)
          ~> Arc::clone(x),

        // (a//c1 + c2) // c3 → (a + c1*c2) // (c1*c3) (nested division simplification)
        // e.g., (a//2 + 1) // 2 → (a + 2) // 4
        Idiv(Add[Idiv(a, c1 @const(c1_val)), c2 @const(c2_val)], c3 @const(c3_val)) => {
            // Compute c1 * c2 and c1 * c3
            let c1_times_c2 = eval_mul_typed(c1_val, c2_val, c1.dtype().base())?;
            let c1_times_c3 = eval_mul_typed(c1_val, c3_val, c1.dtype().base())?;
            // (a + c1*c2) // (c1*c3)
            a.try_add(&UOp::const_(c1.dtype(), c1_times_c2)).ok()?
             .try_div(&UOp::const_(c1.dtype(), c1_times_c3)).ok()
        },
    }
}
