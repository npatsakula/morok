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
use morok_ir::types::{BinaryOp, ConstValue, TernaryOp};
use morok_ir::uop::cached_property::CachedProperty;
use morok_ir::uop::comparison_analysis::ComparisonAnalyzer;
use morok_ir::uop::eval::{eval_add_typed, eval_binary_op, eval_mul_typed, eval_sub_typed};
use morok_ir::uop::properties::VminVmaxProperty;
use morok_ir::{IntoUOp, Op, UOp};

use crate::TypedPatternMatcher;
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
pub fn constant_folding_dsl_patterns() -> TypedPatternMatcher {
    use morok_ir::uop::eval::{eval_binary_op_typed, eval_ternary_op_typed, eval_unary_op_typed};

    patterns! {
        // Unary constant folding - 7 operations in one declaration
        for op in unary [Neg, Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc] {
            op(c @const(c_val))
              => |c, c_val| eval_unary_op_typed(op, c_val, c.dtype().base()).map(|r| UOp::const_(c.dtype(), r)),
        },

        // Binary constant folding - 13 operations in one declaration
        for op in binary [Add, Mul, Sub, Mod, Max, Pow, Idiv, Fdiv, And, Or, Xor, Shl, Shr] {
            op(a @const(a_val), _b @const(b_val))
              => |a, a_val, b_val| eval_binary_op_typed(op, a_val, b_val, a.dtype().base()).map(|r| UOp::const_(a.dtype(), r)),
        },

        // Ternary constant folding - 2 operations in one declaration
        for op in ternary [Where, MulAcc] {
            // For Where: use second operand's dtype (true branch)
            // For MulAcc: use first operand's dtype (all same dtype)
            op(_a @const(a_val), b @const(b_val), _c @const(c_val))
              => |a_val, b, b_val, c_val| eval_ternary_op_typed(op, a_val, b_val, c_val, b.dtype().base()).map(|r| UOp::const_(b.dtype(), r)),
        },
    }
}

/// Identity and zero propagation patterns.
///
/// - Identity folding: x + 0 → x, 0 + x → x, x * 1 → x, 1 * x → x, etc.
/// - Zero propagation: x * 0 → 0 (non-float only), x & 0 → 0
///
/// NOTE: For floats, x * 0 is NOT simplified because IEEE 754 requires:
/// - NaN * 0 = NaN
/// - Inf * 0 = NaN
pub fn identity_and_zero_patterns() -> TypedPatternMatcher {
    patterns! {
        // ========== Identity folding (commutative) ==========
        Add[x, @zero] ~> |x| x.clone(),
        Mul[x, @one] ~> |x| x.clone(),
        Or[x, @zero] ~> |x| x.clone(),
        Xor[x, @zero] ~> |x| x.clone(),

        // ========== Identity folding (non-commutative) ==========
        Sub(x, @zero) ~> |x| x.clone(),
        Idiv(x, @one) ~> |x| x.clone(),
        Fdiv(x, @one) ~> |x| x.clone(),

        // ========== Zero propagation ==========
        // NOTE: For floats, x * 0 is NOT always 0 due to IEEE 754 special values:
        //   - NaN * 0 = NaN
        //   - Inf * 0 = NaN
        // Therefore we only apply this optimization for non-float types.
        // Integer and boolean types are safe since they have no special values.
        Mul[x, zero @ @zero] if !x.dtype().is_float() ~> |zero| zero.clone(),
        And[_, zero @ @zero] ~> |zero| zero.clone(),
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
pub fn symbolic_simple() -> TypedPatternMatcher {
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
pub fn symbolic() -> TypedPatternMatcher {
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
pub fn self_folding_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // x // x → 1
        Idiv(x, x) ~> |x| 1.into_uop(x.dtype()),
        // x // -1 → -x
        Idiv(x, _c @const(c_val)) if c_val.is_neg_one() ~> |x| UOp::neg(x),
        // (x % y) % y → x % y
        Mod(Mod(x, y), y) => |x, y| x.try_mod(y).ok(),
        // x & x → x
        And(x, x) ~> |x| x.clone(),
        // x | x → x
        Or(x, x) ~> |x| x.clone(),
    }
}

/// Zero folding patterns.
///
/// Patterns that fold to zero or false:
/// - x < x → False (non-float only)
/// - x % x → 0
/// - x != x → False (int only)
pub fn zero_folding_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // x % x → 0
        Mod(x, x) => |x| x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::zero(dt))),
        // x < x → False (non-float only)
        Lt(x, x) if !x.dtype().is_float() ~> |_x| false.into_uop(DType::Bool),
        // x != x → False (int only)
        Ne(x, x) if x.dtype().is_int() ~> |_x| false.into_uop(DType::Bool),
    }
}

/// Range-based modulo and division simplification patterns.
///
/// Uses vmin/vmax analysis to simplify:
/// - x % n → x when 0 <= vmin(x) && vmax(x) < n
/// - x / n → 0 when 0 <= vmin(x) && vmax(x) < n
///
/// This is critical for RESHAPE range propagation where Range(n) % n should simplify to Range(n).
pub fn range_based_mod_div_patterns() -> TypedPatternMatcher {
    patterns! {
        // x % n → x when 0 <= vmin(x) && vmax(x) < n
        // This handles cases like Range(3) % 3 → Range(3)
        Mod(x, _n @const(n_val)) => |x, n_val| {
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

        // (a * m + b) % n → b % n when m == n (factor out multiples of n)
        // This handles matmul index expressions like: (row * 512 + col) % 512 → col % 512
        // Since (a * n) % n = 0, the Mul term can be dropped.
        // Using commutative [] for Add to match both orderings.
        // Note: We compare m_val and n_val by VALUE, not pointer, since they may be separate UOps.
        Mod(Add[Mul[_a, _m @const(m_val)], b], n @const(n_val)) => |b, n, m_val, n_val| {
            trace!(
                ?m_val,
                ?n_val,
                b.id = b.id,
                "Mod factor-out CHECKING: m_val == n_val?"
            );
            if m_val != n_val { return None; }  // Multiplier must equal modulus
            trace!(
                ?n_val,
                b.id = b.id,
                "Mod factor-out SUCCESS: (a * n + b) % n → b % n"
            );
            b.try_mod(n).ok()
        },

        // Nested Add version: ((a * m) + b + c) % n → (b + c) % n when m == n
        // Handles more complex expressions with additional terms.
        Mod(Add[Add[Mul[_a, _m @const(m_val)], b], c], n @const(n_val)) => |b, c, n, m_val, n_val| {
            if m_val != n_val { return None; }  // Multiplier must equal modulus
            trace!(
                ?n_val,
                "Mod factor-out nested: ((a * n) + b + c) % n → (b + c) % n"
            );
            let bc = b.try_add(c).ok()?;
            bc.try_mod(n).ok()
        },

        // (a * m + b) / n → a + b / n when m == n (distribute division over sum)
        // When b is non-negative and small, this can enable further simplification.
        // Specifically: (a * n + b) / n = a when 0 <= b < n
        // Using commutative [] for Add to match both orderings.
        // Note: We compare m_val and n_val by VALUE, not pointer, since they may be separate UOps.
        Idiv(Add[Mul[a, _m @const(m_val)], b], n @const(n_val)) => |a, b, n, m_val, n_val| {
            if m_val != n_val { return None; }  // Multiplier must equal divisor
            let ConstValue::Int(n_int) = n_val else { return None };
            if n_int <= 0 { return None; }

            let (vmin, vmax) = VminVmaxProperty::get(b);
            if let (ConstValue::Int(min), ConstValue::Int(max)) = (vmin, vmax)
                && *min >= 0 && *max < n_int {
                    // b is in [0, n), so (a * n + b) / n = a
                    trace!(
                        ?n_val,
                        a.id = a.id,
                        min = *min,
                        max = *max,
                        "Idiv factor-out: (a * n + b) / n → a (when 0 <= b < n)"
                    );
                    return Some(Arc::clone(a));
                }
            // Fall through: compute a + b / n
            let b_div_n = b.try_div(n).ok()?;
            trace!(
                ?n_val,
                "Idiv factor-out: (a * n + b) / n → a + b / n"
            );
            a.try_add(&b_div_n).ok()
        },

        // x / n → k when all values of x are in the same bucket [k*n, (k+1)*n)
        // This is the "cancel divmod" rule from Tinygrad's fold_divmod_general.
        // Examples:
        //   Range(3) / 3 → 0 (since Range(3) is 0,1,2 and all /3 = 0)
        //   (64 + Range(8)) / 64 → 1 (since 64..71 all /64 = 1)
        Idiv(x, _n @const(n_val)) => |x, n_val| {
            let (vmin, vmax) = VminVmaxProperty::get(x);
            if let (ConstValue::Int(min), ConstValue::Int(max), ConstValue::Int(n_int)) = (vmin, vmax, n_val)
                && n_int > 0 {
                    // Compute floor division for min and max
                    let min_div = if *min >= 0 {
                        *min / n_int
                    } else {
                        // For negative numbers: floor division rounds toward negative infinity
                        (*min - n_int + 1) / n_int
                    };
                    let max_div = if *max >= 0 {
                        *max / n_int
                    } else {
                        (*max - n_int + 1) / n_int
                    };
                    // If both endpoints divide to the same value, all values in range do too
                    if min_div == max_div {
                        trace!(
                            min = *min,
                            max = *max,
                            n_int,
                            result = min_div,
                            "Idiv cancel: x / n → k (all values in same bucket)"
                        );
                        return Some(UOp::const_(x.dtype(), ConstValue::Int(min_div)));
                    }
                }
            None
        },

        // Idiv(add_chain, n) → Idiv(add_chain_minus_const, n) when the constant doesn't affect floor division
        // This handles: Idiv(Add(Add(base, 1), inner), 64) → Idiv(Add(base, inner), 64)
        // when base is aligned to 64 and 1 < 64
        Idiv(x, n @const(n_val)) if matches!(x.op(), Op::Binary(BinaryOp::Add, ..)) => |x, n, n_val| {
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
        Idiv(Add[a, Mul[Idiv(x, n @const(n_val)), n]], n) => |a, x, n, n_val| {
            let (vmin, vmax) = VminVmaxProperty::get(a);
            if let (ConstValue::Int(min), ConstValue::Int(max), ConstValue::Int(n_int)) = (vmin, vmax, n_val)
                && *min >= 0 && *max < n_int && n_int > 0 {
                    return x.try_div(n).ok();
                }
            None
        },

        // Phase 2: (x + c) // d → x // d when small offset c doesn't affect bucket
        // This handles index canonicalization: (base + lane_offset) // n → base // n
        // when ALL values of x + c remain in the same bucket as x.
        // Condition: for all v in [vmin(x), vmax(x)]: (v + c) // d == v // d
        // Using commutative [] to match both Add(x, c) and Add(c, x)
        Idiv(Add[x, _c @const(c_val)], d @const(d_val)) => |x, c_val, d, d_val| {
            let ConstValue::Int(c_int) = c_val else { return None };
            let ConstValue::Int(d_int) = d_val else { return None };
            // Only handle small positive offsets with positive divisor
            if d_int <= 0 || c_int < 0 { return None; }

            let (vmin, vmax) = VminVmaxProperty::get(x);
            if let (ConstValue::Int(min), ConstValue::Int(max)) = (vmin, vmax) {
                // For correctness, we need: (v + c) // d == v // d for ALL v in [min, max]
                // This is true iff adding c doesn't cause any value to cross a bucket boundary.
                // Check at both endpoints (with overflow protection):
                let min_c = min.checked_add(c_int)?;
                let max_c = max.checked_add(c_int)?;
                let min_bucket = *min / d_int;
                let max_bucket = *max / d_int;
                let min_c_bucket = min_c / d_int;
                let max_c_bucket = max_c / d_int;

                if min_bucket == min_c_bucket && max_bucket == max_c_bucket {
                    return x.try_div(d).ok();
                }
            }
            None
        },

        // Phase 1: (x + c) // d → (x + (c % d)) // d + (c // d)
        // When c >= d, split the offset into quotient and remainder parts.
        // This canonicalizes large offsets, allowing further simplification.
        // Based on Tinygrad's divandmod.py:101-104
        Idiv(Add[x, _c @const(c_val)], d @const(d_val)) => |x, c_val, d, d_val| {
            let ConstValue::Int(c_int) = c_val else { return None };
            let ConstValue::Int(d_int) = d_val else { return None };
            if d_int <= 0 { return None; }

            let c_mod_d = c_int % d_int;
            let c_div_d = c_int / d_int;

            // Only apply if remainder differs from original (i.e., c >= d or c < 0)
            if c_mod_d == c_int { return None; }

            // Check x.vmin >= 0 for correctness (negative numerators have different semantics)
            let (vmin, _) = VminVmaxProperty::get(x);
            if let ConstValue::Int(min) = vmin {
                if *min < 0 { return None; }
            } else { return None; }

            // Transform: (x + c) // d → (x + c%d) // d + c//d
            let remainder_const = UOp::const_(d.dtype(), ConstValue::Int(c_mod_d));
            let inner = x.try_add(&remainder_const).ok()?;
            let div_result = inner.try_div(d).ok()?;
            let quotient_const = UOp::const_(d.dtype(), ConstValue::Int(c_div_d));
            div_result.try_add(&quotient_const).ok()
        },
    }
}

/// Division simplification patterns.
///
/// - 0 / 0 → NaN (float division by zero of zero)
/// - (x * 0) / 0 → NaN (any expression that reduces to 0/0)
/// - x / x → 1.0 (float division)
/// - (x * y) / y → x
/// - (x * y) // y → x
pub fn division_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // 0 / 0 → NaN (IEEE 754: 0/0 is indeterminate)
        // NOTE: This must come before x/x → 1 pattern to take priority
        Fdiv(zero1 @ @zero, @zero) if zero1.dtype().is_float()
            => |zero1| Some(UOp::const_(zero1.dtype(), ConstValue::Float(f64::NAN))),
        // (x * 0) / 0 → NaN (anything times zero divided by zero is NaN)
        Fdiv(Mul[_, zero1 @ @zero], @zero) if zero1.dtype().is_float()
            => |zero1| Some(UOp::const_(zero1.dtype(), ConstValue::Float(f64::NAN))),
        // x / x → 1.0 (float division)
        Fdiv(x, x) => |x| x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::one(dt))),
        // (x * y) / y → x
        Fdiv(Mul(x, y), y) ~> |x| x.clone(),
        // (x * y) // y → x
        Idiv(Mul(x, y), y) ~> |x| x.clone(),
    }
}

/// Check if casting from `from` to `to` can safely preserve all values.
///
/// Returns true if all values representable in `to` can be represented in `from`.
/// This is used for double-cast optimization: x.cast(a).cast(b) → x.cast(b)
/// is only safe if `a` can hold all values of `b` (so no truncation occurs in `a`).
fn can_safe_cast(to: &DType, from: &DType) -> bool {
    use morok_dtype::ScalarDType;

    // Get base scalar types for comparison
    let to_scalar = match to {
        DType::Scalar(s) => *s,
        DType::Vector { scalar, .. } => *scalar,
        _ => return false,
    };
    let from_scalar = match from {
        DType::Scalar(s) => *s,
        DType::Vector { scalar, .. } => *scalar,
        _ => return false,
    };

    // Same type is always safe
    if to_scalar == from_scalar {
        return true;
    }

    // Get bit widths and signedness
    let (to_bits, to_signed, to_float) = match to_scalar {
        ScalarDType::Bool => (1, false, false),
        ScalarDType::Int8 => (8, true, false),
        ScalarDType::Int16 => (16, true, false),
        ScalarDType::Int32 => (32, true, false),
        ScalarDType::Int64 => (64, true, false),
        ScalarDType::UInt8 => (8, false, false),
        ScalarDType::UInt16 => (16, false, false),
        ScalarDType::UInt32 => (32, false, false),
        ScalarDType::UInt64 => (64, false, false),
        ScalarDType::Float16 | ScalarDType::BFloat16 => (16, true, true),
        ScalarDType::Float32 => (32, true, true),
        ScalarDType::Float64 => (64, true, true),
        _ => return false,
    };
    let (from_bits, from_signed, from_float) = match from_scalar {
        ScalarDType::Bool => (1, false, false),
        ScalarDType::Int8 => (8, true, false),
        ScalarDType::Int16 => (16, true, false),
        ScalarDType::Int32 => (32, true, false),
        ScalarDType::Int64 => (64, true, false),
        ScalarDType::UInt8 => (8, false, false),
        ScalarDType::UInt16 => (16, false, false),
        ScalarDType::UInt32 => (32, false, false),
        ScalarDType::UInt64 => (64, false, false),
        ScalarDType::Float16 | ScalarDType::BFloat16 => (16, true, true),
        ScalarDType::Float32 => (32, true, true),
        ScalarDType::Float64 => (64, true, true),
        _ => return false,
    };

    // Float <-> int conversions are not safe
    if to_float != from_float {
        return false;
    }

    // For floats: larger precision can hold smaller
    if to_float {
        return from_bits >= to_bits;
    }

    // For integers:
    // - Same signedness: larger width can hold smaller
    // - Unsigned to signed: need one extra bit (e.g., u8 fits in i16)
    // - Signed to unsigned: never safe (negative values lost)
    if to_signed == from_signed {
        return from_bits >= to_bits;
    }

    if !to_signed && from_signed {
        // unsigned → signed: from needs to be at least 1 bit larger
        return from_bits > to_bits;
    }

    // signed → unsigned: never safe
    false
}

/// Cast optimization patterns.
///
/// - cast(const) → const (constant folding)
/// - x.cast(dtype) → x if same dtype (noop cast)
/// - x.cast(a).cast(b) → x.cast(b) when safe (collapse double cast)
///
/// NOTE: Double cast is only safe when the intermediate type `a` can hold all
/// values of the final type `b`. Example of UNSAFE collapse:
///   int64.cast(int8).cast(int64) → int64  // WRONG: loses truncation!
pub fn cast_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // cast(const) → const
        Cast { src: _c @const(c_val), dtype } => |c_val, dtype| c_val.cast(dtype).map(|v| UOp::const_(dtype.clone(), v)),
        // x.cast(dtype) → x if same dtype
        Cast { src: x, dtype } if x.dtype() == *dtype ~> |x| x.clone(),
        // x.cast(a).cast(b) → x when x.dtype == b and a preserves all values of b
        // This handles cases like: bool.cast(int32).cast(bool) → bool
        Cast { src: Cast { src: x, dtype: intermediate }, dtype: outer }
            if x.dtype() == *outer && can_safe_cast(outer, intermediate)
            ~> |x| x.clone(),
        // x.cast(a).cast(b) → x.cast(b) when a doesn't narrow x
        // This handles widening chains: int8.cast(int32).cast(int64) → int8.cast(int64)
        Cast { src: Cast { src: x, dtype: intermediate }, dtype: outer }
            if can_safe_cast(&x.dtype(), intermediate)
            ~> |x, outer| UOp::cast(x.clone(), outer.clone()),
    }
}

/// Term combining patterns.
///
/// - x + x → 2*x
/// - (c1 * x) + (c2 * x) → (c1 + c2) * x
/// - (x * c1) + (x * c2) → x * (c1 + c2)
pub fn term_combining_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // x + x → 2*x
        Add(x, x) => |x| 2.into_uop(x.dtype()).try_mul(x).ok(),
        // (c1 * x) + (c2 * x) → (c1 + c2) * x
        Add(Mul(c1 @const(c1_val), x), Mul(_c2 @const(c2_val), x))
          => |c1, c1_val, c2_val, x| eval_add_typed(c1_val, c2_val, c1.dtype().base())?.into_uop(c1.dtype()).try_mul(x).ok(),
        // (x * c1) + (x * c2) → x * (c1 + c2)
        Add(Mul(x, c1 @const(c1_val)), Mul(x, _c2 @const(c2_val)))
          => |x, c1, c1_val, c2_val| x.try_mul(&eval_add_typed(c1_val, c2_val, c1.dtype().base())?.into_uop(c1.dtype())).ok(),
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
pub fn advanced_division_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // (a // b) // c → a // (b * c) if b,c non-zero
        Idiv(Idiv(a, b @const(b_val)), _c @const(c_val)) if !b_val.is_zero() && !c_val.is_zero() => |a, b, b_val, c_val| {
            a.try_div(&UOp::const_(b.dtype(), eval_mul_typed(b_val, c_val, b.dtype().base())?)).ok()
        },
        // expr // divisor → expr.divides(divisor) (generic exact division)
        Idiv(expr, divisor @ @const) => |expr, divisor| expr.divides(divisor),
        // (a + b) % c → simplify when one operand is multiple of c
        Mod(Add(a, b), c @const(c_val)) => |a, b, c, c_val| {
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
        Idiv(Add(a, b), c @ @const) => |a, b, c| a.divides(c)?.try_add(&b.divides(c)?).ok(),
        // (a - b) // c → (a // c) - (b // c) when both divide evenly
        Idiv(Sub(a, b), c @ @const) => |a, b, c| a.divides(c)?.try_sub(&b.divides(c)?).ok(),
        // c * (a + b) → (c * a) + (c * b)
        Mul(c @ @const, Add(a, b)) => |c, a, b| c.try_mul(a).ok()?.try_add(&c.try_mul(b).ok()?).ok(),
        // (a + b) * c → (a * c) + (b * c)
        Mul(Add(a, b), c @ @const) => |a, b, c| a.try_mul(c).ok()?.try_add(&b.try_mul(c).ok()?).ok(),
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
/// - (x + c) + y → (x + y) + c (constant pushing, for index canonicalization)
pub fn alu_folding_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // (x + c1) + c2 → x + (c1 + c2) - commutative outer Add
        Add[Add[x, c1 @const(c1_val)], _c2 @const(c2_val)]
          => x.try_add(&UOp::const_(c1.dtype(), eval_add_typed(c1_val, c2_val, c1.dtype().base())?)).ok(),

        // Constant pushing: (x + c) + y → (x + y) + c when y is NOT a const
        // Based on Tinygrad's ((UPat.var("x") + UPat.cvar("c1")) + UPat.var("y"), lambda x,c1,y: (x+y)+c1)
        // This ensures constants bubble to the outermost level for index extraction.
        // Using commutative pattern to also match y + (x + c) and (c + x) + y cases.
        // IMPORTANT: Use UOp::new directly instead of try_add() to avoid type promotion
        // which could insert casts and create structurally different expressions.
        Add[inner @ Add[x, c @const(c_val)], y] if !matches!(y.op(), Op::Const(_)) => {
            // Create (x + y) directly preserving the inner Add's dtype
            let xy = UOp::new(Op::Binary(BinaryOp::Add, Arc::clone(x), Arc::clone(y)), inner.dtype());
            // Then add the constant
            Some(UOp::new(Op::Binary(BinaryOp::Add, xy, UOp::const_(c.dtype(), c_val)), inner.dtype()))
        },

        // (x * c1) * c2 → x * (c1 * c2) - commutative outer Mul
        Mul[Mul[x, c1 @const(c1_val)], _c2 @const(c2_val)]
          => x.try_mul(&UOp::const_(c1.dtype(), eval_mul_typed(c1_val, c2_val, c1.dtype().base())?)).ok(),

        // (x - c1) + c2 → x + (c2 - c1) or x - (c1 - c2) - commutative outer Add
        Add[Sub(x, c1 @const(c1_val)), _c2 @const(c2_val)] => {
            let diff_val = eval_sub_typed(c2_val, c1_val, c1.dtype().base())?;
            // Normalize: prefer x - |c| over x + (-c)
            if let ConstValue::Int(v) = diff_val && v < 0 {
                x.try_sub(&(-v).into_uop(c1.dtype())).ok()
            } else {
                x.try_add(&UOp::const_(c1.dtype(), diff_val)).ok()
            }
        },

        // (x + c1) - c2 → x + (c1 - c2) or x - (c2 - c1) when result is negative
        Sub(Add(x, c1 @const(c1_val)), _c2 @const(c2_val)) => {
            let diff_val = eval_sub_typed(c1_val, c2_val, c1.dtype().base())?;
            // Normalize: prefer x - |c| over x + (-c)
            if let ConstValue::Int(v) = diff_val && v < 0 {
                return x.try_sub(&(-v).into_uop(c1.dtype())).ok();
            }
            x.try_add(&UOp::const_(c1.dtype(), diff_val)).ok()
        },

        // (x - c1) - c2 → x - (c1 + c2)
        Sub(Sub(x, c1 @const(c1_val)), _c2 @const(c2_val))
          => x.try_sub(&UOp::const_(c1.dtype(), eval_add_typed(c1_val, c2_val, c1.dtype().base())?)).ok(),
    }
}

/// Dead loop elimination patterns.
///
/// - RANGE with vmax ≤ 0 → Const(0)
/// - END with dead ranges → remove dead ranges
/// - REDUCE with all empty ranges → identity element
pub fn dead_loop_patterns() -> TypedPatternMatcher {
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
pub fn dce_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // WHERE with constant condition → select appropriate branch
        Where(cond, true_val, false_val) => |cond, true_val, false_val| {
            match VminVmaxProperty::get(cond) {
                (ConstValue::Bool(true), ConstValue::Bool(true)) => Some(Arc::clone(true_val)),
                (ConstValue::Bool(false), ConstValue::Bool(false)) => Some(Arc::clone(false_val)),
                _ => None,
            }
        },

        // WHERE(_, same, same) → same
        Where(_, t, t) ~> |t| Arc::clone(t),

        // WHERE(x, true, false) → x (for bool x)
        Where(x, _t @const(t_val), _f @const(f_val))
          if x.dtype() == DType::Bool && t_val == ConstValue::Bool(true) && f_val == ConstValue::Bool(false)
          ~> |x| Arc::clone(x),

        // WHERE(x, false, true) → !x (for bool x)
        Where(x, _t @const(t_val), _f @const(f_val))
          if x.dtype() == DType::Bool && t_val == ConstValue::Bool(false) && f_val == ConstValue::Bool(true)
          ~> |x| x.not(),

        // WHERE(!cond, t, f) → WHERE(cond, f, t) - negated condition swap
        Where(Not(cond), t, f) => |cond, t, f| UOp::try_where(Arc::clone(cond), Arc::clone(f), Arc::clone(t)).ok(),

        // WHERE(a, WHERE(b, c, d), d) → WHERE(a & b, c, d) - branch merging
        Where(a, Where(b, c, d), d2) if Arc::ptr_eq(d, d2) => |a, b, c, d| {
            let combined_cond = a.try_and_op(b).ok()?;
            UOp::try_where(combined_cond, Arc::clone(c), Arc::clone(d)).ok()
        },
    }
}

/// Cast pushing through WHERE patterns.
///
/// - where(s, a, b).cast(dtype) → where(s, a.cast(dtype), b.cast(dtype))
pub fn cast_where_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // cast(where(s, a, b), dtype) → where(s, cast(a, dtype), cast(b, dtype))
        Cast { src: Where(s, a, b), dtype } => |s, a, b, dtype| {
            let cast_a = UOp::cast(a.clone(), dtype.clone());
            let cast_b = UOp::cast(b.clone(), dtype.clone());
            UOp::try_where(s.clone(), cast_a, cast_b).ok()
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
pub fn comparison_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        for op in binary [Lt, Eq, Ne] {
            op(x, y) => |x, y| {
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
            },
        },

        // (c0 + x) < c1 → x < (c1 - c0) for integers - commutative
        Lt(Add[c0 @const(c0_val), x], _c1 @const(c1_val)) => |c0, c0_val, x, c1_val| {
            let diff = eval_sub_typed(c1_val, c0_val, c0.dtype().base())?;
            x.try_cmplt(&UOp::const_(c0.dtype(), diff)).ok()
        },

        // -x < -y → y < x (negation flip for Lt)
        Lt(Neg(x), Neg(y)) => |x, y| y.try_cmplt(x).ok(),

        // Phase 6: (x // d) < c → x < (c * d) when d > 0
        // This lifts division out of comparisons, enabling further simplification.
        // Based on Tinygrad's symbolic.py:229-230
        Lt(Idiv(x, _d @const(d_val)), _c @const(c_val)) => |x, d_val, c_val| {
            let ConstValue::Int(d_int) = d_val else { return None };
            let ConstValue::Int(c_int) = c_val else { return None };
            if d_int <= 0 { return None; }

            // For x // d < c:
            // - If c > 0: equivalent to x < c * d
            // - If c <= 0: equivalent to x < c * d - (d - 1) = c * d - d + 1
            let bound = if c_int > 0 {
                c_int * d_int
            } else {
                c_int * d_int - (d_int - 1)
            };
            x.try_cmplt(&UOp::const_(x.dtype(), ConstValue::Int(bound))).ok()
        },
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
pub fn boolean_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // !!x → x
        Not(Not(x)) ~> |x| x.clone(),
        // x ^ x → 0
        Xor(x, x) => |x| x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::zero(dt))),

        // x | !x → true (tautology) - commutative
        Or[x, Not(y)] if Arc::ptr_eq(x, y) && x.dtype() == DType::Bool
          => |_x| Some(UOp::const_(DType::Bool, ConstValue::Bool(true))),

        // x & !x → false (contradiction) - commutative
        And[x, Not(y)] if Arc::ptr_eq(x, y) && x.dtype() == DType::Bool
          => |_x| Some(UOp::const_(DType::Bool, ConstValue::Bool(false))),

        // true | x → true (commutative)
        Or[t @const(t_val), _] if t_val == ConstValue::Bool(true) ~> |t| t.clone(),

        // false & x → false (commutative)
        And[f @const(f_val), _] if f_val == ConstValue::Bool(false) ~> |f| f.clone(),

        // true & x → x (identity, commutative)
        And[_c @const(c_val), x] if c_val == ConstValue::Bool(true) ~> |x| x.clone(),

        // false | x → x (identity, commutative)
        Or[_c @const(c_val), x] if c_val == ConstValue::Bool(false) ~> |x| x.clone(),
    }
}

/// Min/max patterns.
///
/// - max(x, x) → x
/// - min(x, x) → x (via Min = negated Max)
pub fn minmax_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // max(x, x) → x
        Max(x, x) ~> |x| x.clone(),
    }
}

/// Power patterns.
///
/// - x ** 0 → 1
/// - x ** 1 → x
pub fn power_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // x ** 0 → 1
        Pow(x, _c @const(c_val)) if c_val.is_zero() => |x| x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::one(dt))),
        // x ** 1 → x
        Pow(x, _c @const(c_val)) if c_val.is_one() ~> |x| x.clone(),
    }
}

/// Negation patterns.
///
/// - -(-x) → x (double negation for arithmetic)
pub fn negation_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // Double arithmetic negation: -(-x) → x
        Neg(Neg(x)) ~> |x| x.clone(),
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
pub fn gep_pushing_patterns() -> TypedPatternMatcher {
    /// Check if VECTORIZE is a broadcast (all elements pointer-equal)
    fn is_broadcast(elements: &SmallVec<[Arc<UOp>; 4]>) -> bool {
        elements.first().is_some_and(|first| elements.iter().all(|e| Arc::ptr_eq(e, first)))
    }

    patterns! {
        // 1. GEP composition: GEP(GEP(x, inner), outer) → GEP(x, inner[outer])
        // Note: nested struct patterns require UOp first field, so we extract inner manually
        Gep { vector, indices } if matches!(vector.op(), Op::Gep { .. }) => |vector, indices| {
            let Op::Gep { vector: inner_vec, indices: inner_indices } = vector.op() else { return None };
            let composed: Vec<usize> = indices.iter()
                .map(|&o| inner_indices.get(o).copied())
                .collect::<Option<Vec<_>>>()?;
            Some(UOp::gep(Arc::clone(inner_vec), composed))
        },

        // 2. GEP through BROADCAST: extract scalar (MUST be before general VECTORIZE!)
        // BROADCAST = VECTORIZE with all identical elements → GEP([x,x,x,x], [i]) → x
        Gep { vector, indices } if indices.len() == 1 && matches!(vector.op(), Op::Vectorize { .. }) => |vector| {
            let Op::Vectorize { elements } = vector.op() else { return None };
            is_broadcast(elements).then(|| elements.first().cloned()).flatten()
        },

        // 3. GEP through VECTORIZE: extract single element
        Gep { vector, indices } if indices.len() == 1 && matches!(vector.op(), Op::Vectorize { .. }) => |vector, indices| {
            let Op::Vectorize { elements } = vector.op() else { return None };
            elements.get(indices[0]).cloned()
        },

        // 4. GEP on scalar: GEP(x, [i]) where x is scalar → x
        Gep { vector, indices } if vector.dtype().vcount() == 1 && indices.len() == 1 ~> |vector| Arc::clone(vector),

        // 5. GEP through VConst: extract element(s)
        Gep { vector, indices } if matches!(vector.op(), Op::VConst { .. }) => |vector, indices| {
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
        Gep { vector, indices } if !indices.is_empty() && matches!(vector.op(), Op::Binary(..)) => |vector, indices| {
            let Op::Binary(bin_op, a, b) = vector.op() else { return None };
            let gep_a = UOp::gep(Arc::clone(a), indices.clone());
            let gep_b = UOp::gep(Arc::clone(b), indices.clone());
            Some(UOp::new(Op::Binary(*bin_op, gep_a.clone(), gep_b), gep_a.dtype()))
        },

        // 7. Push GEP through Unary: GEP(Unary(op, x), indices) → Unary(op, GEP(x))
        Gep { vector, indices } if !indices.is_empty() && matches!(vector.op(), Op::Unary(..)) => |vector, indices| {
            let Op::Unary(un_op, x) = vector.op() else { return None };
            let gep_x = UOp::gep(Arc::clone(x), indices.clone());
            Some(UOp::new(Op::Unary(*un_op, gep_x.clone()), gep_x.dtype()))
        },

        // 7b. Push GEP through Ternary: GEP(Ternary(op, a, b, c), indices) → Ternary(op, GEP(a), GEP(b), GEP(c))
        // Required for MulAcc (FMA) and WHERE to work with split_load (which creates CAT of 4-element loads)
        Gep { vector: Where(cond, t, f), indices } if !indices.is_empty() => |cond, t, f, indices| {
            let gep_cond = UOp::gep(Arc::clone(cond), indices.clone());
            let gep_t = UOp::gep(Arc::clone(t), indices.clone());
            let gep_f = UOp::gep(Arc::clone(f), indices.clone());
            Some(UOp::new(Op::Ternary(TernaryOp::Where, gep_cond.clone(), gep_t, gep_f), gep_cond.dtype()))
        },
        Gep { vector: MulAcc(a, b, c), indices } if !indices.is_empty() => |a, b, c, indices| {
            let gep_a = UOp::gep(Arc::clone(a), indices.clone());
            let gep_b = UOp::gep(Arc::clone(b), indices.clone());
            let gep_c = UOp::gep(Arc::clone(c), indices.clone());
            Some(UOp::new(Op::Ternary(TernaryOp::MulAcc, gep_a.clone(), gep_b, gep_c), gep_a.dtype()))
        },

        // 8. GEP through UNROLL: GEP(UNROLL(x, ...), indices) → GEP(x, indices)
        Gep { vector, indices } if matches!(vector.op(), Op::Unroll { .. }) => |vector, indices| {
            let Op::Unroll { src, .. } = vector.op() else { return None };
            Some(UOp::gep(Arc::clone(src), indices.clone()))
        },

        // 9. Identity GEP removal: GEP(x, [0,1,2,...,n-1]) → x
        Gep { vector, indices }
            if indices.iter().enumerate().all(|(i, &idx)| i == idx)
                && indices.len() == vector.dtype().vcount()
            ~> |vector| Arc::clone(vector),

        // 10. GEP through PTRCAT: GEP(PTRCAT([a, b, c, d]), [1, 3]) → PTRCAT([b, d])
        Gep { vector, indices } if matches!(vector.op(), Op::PtrCat { .. }) => |vector, indices| {
            let Op::PtrCat { sources } = vector.op() else { return None };
            let reordered: Vec<_> = indices.iter().filter_map(|&idx| sources.get(idx).cloned()).collect();
            (reordered.len() == indices.len()).then(|| UOp::ptrcat(reordered))
        },

        // 11. GEP through CAT: Extract elements from concatenated vectors
        // GEP(CAT([a<4>, b<4>]), [5]) → GEP(b, [1]) (element 5 = source 1, offset 1)
        // Handles multi-element CAT sources by computing cumulative element offsets.
        Gep { vector, indices } if matches!(vector.op(), Op::Cat { .. }) => |vector, indices| {
            let Op::Cat { sources } = vector.op() else { return None };

            // Build cumulative element counts: [0, count(src0), count(src0)+count(src1), ...]
            let mut cumulative = Vec::with_capacity(sources.len() + 1);
            cumulative.push(0usize);
            for src in sources.iter() {
                let prev = *cumulative.last().unwrap();
                cumulative.push(prev + src.dtype().vcount());
            }

            // Map each element index to (source_idx, element_offset_within_source)
            let extracted: Vec<_> = indices
                .iter()
                .filter_map(|&elem_idx| {
                    // Binary search for the source containing this element
                    let src_idx = cumulative.partition_point(|&c| c <= elem_idx).saturating_sub(1);
                    let src = sources.get(src_idx)?;
                    let offset_in_src = elem_idx - cumulative[src_idx];

                    // Bounds check
                    if offset_in_src >= src.dtype().vcount() {
                        return None;
                    }

                    if src.dtype().vcount() == 1 {
                        Some(src.clone())
                    } else {
                        Some(UOp::gep(src.clone(), vec![offset_in_src]))
                    }
                })
                .collect();

            if extracted.len() != indices.len() {
                return None;
            }

            if extracted.len() == 1 {
                Some(extracted.into_iter().next().unwrap())
            } else {
                Some(UOp::vectorize(extracted.into_iter().collect()))
            }
        },
    }
}

/// Div/Mod recombination patterns.
///
/// Uses automatic ptr_eq checks via duplicate variable names in patterns.
/// - x%n + (x//n)*n → x (div-mod identity)
/// - ((x//a) % c) + (x // b) * c → x // a when a*c == b
/// - (x % c1) * c2 + (x // c1) * c3 → x * c2 when c1*c2 == c3
/// - y + (x % n) + (x // n) * n → y + x
/// - (a//c1 + c2) // c3 → (a + c1*c2) // (c1*c3) (nested division)
pub fn div_mod_recombine_dsl_patterns() -> TypedPatternMatcher {
    patterns! {
        // x%n + (x//n)*n → x (div-mod identity)
        // Note: duplicate variable names (x, n) auto-generate Arc::ptr_eq checks
        Add[Mod(x, n), Mul[Idiv(x, n), n]]
          ~> |x| Arc::clone(x),

        // ((x//a) % c) + (x // b) * c → x // a
        // Condition: a * c == b (divisor composition)
        // Note: x appears twice, c appears twice → auto ptr_eq checks
        Add[Mod(Idiv(x, a @const(a_val)), c @const(c_val)), Mul[Idiv(x, _b @const(b_val)), c]]
          => |x, a, a_val, c_val, b_val| {
            let ConstValue::Int(a_int) = a_val else { return None };
            let ConstValue::Int(c_int) = c_val else { return None };
            let ConstValue::Int(b_int) = b_val else { return None };
            if a_int * c_int == b_int {
                return x.try_div(a).ok();
            }
            None
        },

        // (x % c1) * c2 + (x // c1) * c3 → x * c2
        // Condition: c1 * c2 == c3
        // Note: x appears twice, c1 appears twice → auto ptr_eq checks
        Add[Mul[Mod(x, c1 @const(c1_val)), c2 @const(c2_val)], Mul[Idiv(x, c1), _c3 @const(c3_val)]]
          => |x, c1_val, c2, c2_val, c3_val| {
            let ConstValue::Int(c1_int) = c1_val else { return None };
            let ConstValue::Int(c2_int) = c2_val else { return None };
            let ConstValue::Int(c3_int) = c3_val else { return None };
            if c1_int * c2_int == c3_int {
                return x.try_mul(c2).ok();
            }
            None
        },

        // y + (x % n) + (x // n) * n → y + x
        // Note: x appears twice, n appears 3 times → auto ptr_eq for all
        Add[Add[y, Mod(x, n)], Mul[Idiv(x, n), n]]
          => |y, x| y.try_add(x).ok(),

        // (a//c1 + c2) // c3 → (a + c1*c2) // (c1*c3) (nested division simplification)
        // e.g., (a//2 + 1) // 2 → (a + 2) // 4
        Idiv(Add[Idiv(a, c1 @const(c1_val)), _c2 @const(c2_val)], _c3 @const(c3_val)) => |a, c1, c1_val, c2_val, c3_val| {
            // Compute c1 * c2 and c1 * c3
            let c1_times_c2 = eval_mul_typed(c1_val, c2_val, c1.dtype().base())?;
            let c1_times_c3 = eval_mul_typed(c1_val, c3_val, c1.dtype().base())?;
            // (a + c1*c2) // (c1*c3)
            a.try_add(&UOp::const_(c1.dtype(), c1_times_c2)).ok()?
             .try_div(&UOp::const_(c1.dtype(), c1_times_c3)).ok()
        },
    }
}
