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

use morok_dtype::{DType, ScalarDType};
use morok_ir::types::{BinaryOp, ConstValue, TernaryOp};
use morok_ir::uop::cached_property::CachedProperty;
use morok_ir::uop::comparison_analysis::ComparisonAnalyzer;
use morok_ir::uop::eval::{
    eval_add_typed, eval_binary_op, eval_binary_op_broadcast, eval_binary_op_broadcast_typed, eval_mul_typed,
    eval_sub_typed, eval_unary_op_vec_typed,
};
use morok_ir::uop::properties::VminVmaxProperty;
use morok_ir::{IntoUOp, Op, UOp};

use crate::TypedPatternMatcher;
use crate::rangeify::indexing::get_const_value;
use crate::symbolic::dce::is_empty_range;

use smallvec::SmallVec;
use std::sync::Arc;
use tracing::trace;

/// Constant folding patterns.
///
/// Folds constant expressions at compile time for unary, binary, and ternary operations.
/// Uses dtype-aware evaluation to ensure results respect type boundaries (e.g., Int32 wraps at 32 bits).
pub fn constant_folding_dsl_patterns() -> &'static TypedPatternMatcher {
    use morok_ir::uop::eval::{eval_binary_op_typed, eval_ternary_op_typed, eval_unary_op_typed};

    crate::cached_patterns! {
        // Unary constant folding - 7 operations in one declaration
        for op in unary [Neg, Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc] {
            op(c @const(c_val))
              => eval_unary_op_typed(op, c_val, c.dtype().base()).map(|r| UOp::const_(c.dtype(), r)),
        },

        // Binary constant folding - 13 operations in one declaration
        for op in binary [Add, Mul, Sub, Mod, Max, Pow, Idiv, Fdiv, And, Or, Xor, Shl, Shr] {
            op(a @const(a_val), _b @const(b_val))
              => eval_binary_op_typed(op, a_val, b_val, a.dtype().base()).map(|r| UOp::const_(a.dtype(), r)),
        },

        // Ternary constant folding - 2 operations in one declaration
        for op in ternary [Where, MulAcc] {
            // For Where: use second operand's dtype (true branch)
            // For MulAcc: use first operand's dtype (all same dtype)
            op(_a @const(a_val), b @const(b_val), _c @const(c_val))
              => eval_ternary_op_typed(op, a_val, b_val, c_val, b.dtype().base()).map(|r| UOp::const_(b.dtype(), r)),
        },
    }
}

/// VConst constant folding patterns.
///
/// Folds VConst expressions at compile time:
/// - Binary operations on VConst pairs: VConst op VConst → VConst
/// - Binary operations mixing Const and VConst (with broadcast)
/// - Unary operations on VConst
///
/// Based on Tinygrad's exec_alu for VCONST handling.
pub fn vconst_folding_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Binary VConst folding: VConst op VConst → VConst
        for op in binary [Add, Mul, Sub, Mod, Max, Idiv, And, Or, Xor, Shl, Shr] {
            op(a @vconst(vals_a), _b @vconst(vals_b))
              => {
                  eval_binary_op_broadcast_typed(op, &vals_a, &vals_b, a.dtype().base())
                      .map(UOp::vconst)
              },
        },

        // Comparison VConst folding: VConst cmp VConst → VConst(Bool)
        // Comparisons return Bool regardless of input dtype. UOp::vconst infers Bool
        // from ConstValue::Bool values. Critical for sort correctness: enables
        // Lt(VCONST([0,1]), VECTORIZE([1,1])) → VCONST([true, false]) before devectorize,
        // allowing WHERE(cond, val, 0.0) to constant-fold per lane.
        for op in binary [Lt, Le, Eq, Ne, Gt, Ge] {
            op(_a @vconst(vals_a), _b @vconst(vals_b))
              => {
                  eval_binary_op_broadcast(op, &vals_a, &vals_b)
                      .map(UOp::vconst)
              },
        },

        // Mixed Const + VConst folding (broadcast): Const op VConst → VConst
        for op in binary [Add, Mul, Sub, Mod, Max, Idiv, And, Or, Xor, Shl, Shr] {
            op(a @anyconst(vals_a), _b @anyconst(vals_b))
              if vals_a.len() != vals_b.len() // Only when broadcasting needed
              => eval_binary_op_broadcast_typed(op, &vals_a, &vals_b, a.dtype().base()).map(UOp::vconst),
        },

        // Comparison mixed Const + VConst folding (broadcast)
        for op in binary [Lt, Le, Eq, Ne, Gt, Ge] {
            op(_a @anyconst(vals_a), _b @anyconst(vals_b))
              if vals_a.len() != vals_b.len()
              => eval_binary_op_broadcast(op, &vals_a, &vals_b).map(UOp::vconst),
        },

        // Unary VConst folding
        for op in unary [Neg, Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc] {
            op(a @vconst(vals))
              => eval_unary_op_vec_typed(op, &vals, a.dtype().base()).map(UOp::vconst),
        },
    }
}

/// Bool arithmetic patterns.
pub fn bool_arithmetic_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Bool * Bool → AND
        Mul[x, y] if x.dtype() == DType::Bool && y.dtype() == DType::Bool ~> x.and_(y),
        // Bool + Bool → OR
        Add[x, y] if x.dtype() == DType::Bool && y.dtype() == DType::Bool ~> x.or_(y),
        // Bool max Bool → OR
        Max(x, y) if x.dtype() == DType::Bool && y.dtype() == DType::Bool ~> x.or_(y),
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
pub fn identity_and_zero_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // ========== Identity folding (commutative) ==========
        Add[x, @zero] ~> x.clone(),
        Mul[x, @one] ~> x.clone(),
        Or[x, @zero] ~> x.clone(),
        Xor[x, @zero] ~> x.clone(),

        // ========== Identity folding (non-commutative) ==========
        Sub(x, @zero) ~> x.clone(),
        Idiv(x, @one) ~> x.clone(),
        Fdiv(x, @one) ~> x.clone(),
        // x % 1 → 0 (anything mod 1 is 0)
        Mod(x, @one) => x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::zero(dt))),

        // ========== Rounding identity for integer types ==========
        // Floor/Ceil/Trunc/Round on integers is identity — rounding is a no-op.
        for op in unary [Floor, Ceil, Trunc, Round] {
            op(x) if !x.dtype().is_float() ~> { let _ = op; x.clone() }
        },

        // ========== Zero propagation ==========
        // NOTE: For floats, x * 0 is NOT always 0 due to IEEE 754 special values:
        //   - NaN * 0 = NaN
        //   - Inf * 0 = NaN
        // Therefore we only apply this optimization for non-float types.
        // Integer and boolean types are safe since they have no special values.
        Mul[x, zero @ @zero] if !x.dtype().is_float() ~> zero.clone(),
        And[_, zero @ @zero] ~> zero.clone(),
    }
}

/// Invalid propagation patterns (Tinygrad symbolic.py:29-38).
///
/// Push arithmetic through WHERE-encoded gates to preserve validity tracking:
/// - CAST(WHERE(cond, x, Invalid)) → WHERE(cond, CAST(x), Invalid)
/// - ALU(WHERE(cond, x, Invalid), y) → WHERE(cond, ALU(x, y), Invalid)
/// - ALU(y, WHERE(cond, x, Invalid)) → WHERE(cond, ALU(y, x), Invalid)
/// - ALU(Invalid, y) → Invalid  (only when y is Index dtype, left position only)
///
/// Note: Tinygrad only propagates bare Invalid from the LEFT position and requires
/// the right operand to be Index dtype (symbolic.py:37). Right-position bare Invalid
/// is NOT propagated to avoid contaminating non-index computations.
///
/// MUST be first in `symbolic_simple()` — before `x*0→0` which would eat
/// `MUL(0, WHERE(cond, x, Invalid))` → `0`, losing validity tracking.
pub fn propagate_invalid() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Canonicalize: WHERE(cond, INVALID, x) → WHERE(NOT(cond), x, INVALID)
        // INVALID must be in the false branch for downstream patterns to match.
        Where(cond, inv, x) if matches!(inv.op(), Op::Invalid) ~> {
            let invalid = if inv.dtype() == x.dtype() { inv.clone() } else { UOp::new(Op::Invalid, x.dtype()) };
            UOp::try_where(cond.not(), x.clone(), invalid).expect("failes to create WHERE")
        },

        // Merge nested WHERE-Invalid: WHERE(c1, WHERE(c2, x, Inv), Inv) → WHERE(AND(c1, c2), x, Inv)
        // Multi-dimensional padding creates nested WHERE-Invalid after propagation through
        // linearized index arithmetic (e.g., WHERE(valid_h, idx_h, Inv)*W + WHERE(valid_w, idx_w, Inv)
        // → WHERE(valid_h, WHERE(valid_w, linear_idx, Inv), Inv)). Merging to a single level
        // ensures pm_lower_index_dtype's INDEX pattern can consume it in one step.
        Where(c1, Where(c2, x, inner_inv), outer_inv) if matches!(inner_inv.op(), Op::Invalid) && matches!(outer_inv.op(), Op::Invalid) ~> {
            let combined = c1.and_(c2);
            UOp::try_where(combined, x.clone(), inner_inv.clone()).expect("failed to create WHERE")
        },

        // Safety net: Eliminate WHERE-Invalid from data path.
        // If absorb_invalid_into_index_gate didn't catch it (e.g. multi-index INDEX),
        // WHERE(c1, WHERE(c2, x, Inv), y) → WHERE(AND(c1,c2), x, y) remains correct.
        Where(c1, Where(c2, x, inner_inv), y) if matches!(inner_inv.op(), Op::Invalid) ~> {
            let combined = c1.and_(c2);
            UOp::try_where(combined, x.clone(), y.clone()).expect("failed to create WHERE")
        },

        // Drop WHERE-Invalid through CAST (Tinygrad symbolic.py:31)
        // CAST(WHERE(cond, x, Invalid)) → CAST(x)
        // INVALID only lives in Index dtype for INDEX addresses. After CAST to a
        // different type, the protection is irrelevant — the outer value-level WHERE
        // on the PAD/Concat handles correctness.
        Cast { src: Where(_cond, x, invalid), dtype } if matches!(invalid.op(), Op::Invalid) ~> x.cast(dtype.clone()),

        // Push binary ALU (non-comparison) through WHERE-with-Invalid (left operand)
        // ALU(WHERE(cond, x, Invalid), y) → WHERE(cond, ALU(x, y), Invalid)
        for op in binary [Add, Mul, Sub, Mod, Max, Idiv, Fdiv, Pow, And, Or, Xor, Shl, Shr] {
            r @ op(Where(cond, x, invalid), y)
                if matches!(invalid.op(), Op::Invalid)
                ~> {
                    let inner = UOp::new(Op::Binary(op, x.clone(), y.clone()), r.dtype());
                    UOp::try_where(cond.clone(), inner, invalid.clone()).expect("failed to create WHERE")
                },
        },

        // Push binary ALU (non-comparison) through WHERE-with-Invalid (right operand)
        // ALU(y, WHERE(cond, x, Invalid)) → WHERE(cond, ALU(y, x), Invalid)
        for op in binary [Add, Mul, Sub, Mod, Max, Idiv, Fdiv, Pow, And, Or, Xor, Shl, Shr] {
            r @ op(y, Where(cond, x, invalid))
                if matches!(invalid.op(), Op::Invalid)
                ~> {
                    let inner = UOp::new(Op::Binary(op, y.clone(), x.clone()), r.dtype());
                    UOp::try_where(cond.clone(), inner, invalid.clone()).expect("failed to create WHERE")
                },
        },

        // Strip WHERE-Invalid from comparison inputs (Tinygrad symbolic.py:35)
        // CMP(WHERE(cond, x, Invalid), y) → CMP(x, y)
        // When comparing padded values, the Invalid region is already gated downstream,
        // so we can safely compare just the valid part.
        for op in binary [Lt, Le, Eq, Ne, Gt, Ge] {
            r @ op(Where(_cond, x, invalid), y)
                if matches!(invalid.op(), Op::Invalid)
                ~> UOp::new(Op::Binary(op, x.clone(), y.clone()), r.dtype()),
        },

        // CMP(y, WHERE(cond, x, Invalid)) → CMP(y, x) (right operand variant)
        for op in binary [Lt, Le, Eq, Ne, Gt, Ge] {
            r @ op(y, Where(_cond, x, invalid))
                if matches!(invalid.op(), Op::Invalid)
                ~> UOp::new(Op::Binary(op, y.clone(), x.clone()), r.dtype()),
        },

        // ALU with bare Invalid → Invalid (Tinygrad symbolic.py:37)
        // Tinygrad: `invalid_pat.alu(op, UPat(dtype=dtypes.index))` with auto-commutation.
        // Left position (all ops):
        for op in binary [Add, Mul, Sub, Mod, Max, Idiv, Fdiv, Pow, And, Or, Xor, Shl, Shr] {
            op(invalid, y) if matches!(invalid.op(), Op::Invalid) && y.dtype() == DType::Index
                ~> { let _ = op; invalid.clone() },
        },
        // Right position (commutative ops only — Tinygrad auto-commutation):
        for op in binary [Add, Mul, Max, And, Or, Xor] {
            op(y, invalid) if matches!(invalid.op(), Op::Invalid) && y.dtype() == DType::Index
                ~> { let _ = op; invalid.clone() },
        },
    }
}

/// Fold LOAD/STORE with fully-Invalid INDEX (Tinygrad symbolic.py:408-409).
///
/// When an INDEX has an Invalid marker as its index, the entire access is out-of-bounds:
/// - LOAD(INDEX(buf, Invalid)) → const 0 (invalid load produces zero)
/// - STORE(INDEX(buf, Invalid), value) → NOOP (invalid store does nothing)
///
/// Also handles CAST-wrapped variants:
/// - LOAD(CAST(INDEX(buf, Invalid))) → const 0
/// - STORE(CAST(INDEX(buf, Invalid)), value) → NOOP
///
/// This occurs when padding creates regions entirely outside the original tensor bounds,
/// causing WHERE(valid, idx, Invalid) to simplify to just Invalid when valid is always false.
pub fn fold_invalid_load_store() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // LOAD(INDEX(buf, Invalid)) → const 0 (dtype-appropriate)
        load @ Load { index: Index { indices, .. }, .. }
            if indices.len() == 1 && matches!(indices[0].op(), Op::Invalid)
            => {
                let zero = ConstValue::zero(load.dtype().scalar()?);
                Some(load.const_like(zero))
            },

        // LOAD(CAST(INDEX(buf, Invalid))) → const 0 (dtype-appropriate)
        load @ Load { index: Cast { src: Index { indices, .. }, .. }, .. }
            if indices.len() == 1 && matches!(indices[0].op(), Op::Invalid)
            => {
                let zero = ConstValue::zero(load.dtype().scalar()?);
                Some(load.const_like(zero))
            },

        // STORE(INDEX(buf, Invalid), value) → NOOP
        Store { index: Index { indices, .. }, .. }
            if indices.len() == 1 && matches!(indices[0].op(), Op::Invalid)
            ~> UOp::new(Op::Noop, DType::Void),

        // STORE(CAST(INDEX(buf, Invalid)), value) → NOOP
        Store { index: Cast { src: Index { indices, .. }, .. }, .. }
            if indices.len() == 1 && matches!(indices[0].op(), Op::Invalid)
            ~> UOp::new(Op::Noop, DType::Void),
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
pub fn symbolic_simple() -> &'static TypedPatternMatcher {
    static CACHED: std::sync::LazyLock<TypedPatternMatcher> = std::sync::LazyLock::new(|| {
        propagate_invalid()
            + fold_invalid_load_store()
            + constant_folding_dsl_patterns()
            + vconst_folding_patterns()
            + bool_arithmetic_patterns()
            + identity_and_zero_patterns()
            + commutative_canonicalization()
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
            + where_bound_patterns()
            + power_dsl_patterns()
            + negation_dsl_patterns()
            + range_based_mod_div_patterns()
            + dce_dsl_patterns()
            + dead_loop_patterns()
            + after_simplification_patterns()
            + pm_move_where_on_load()
    });
    &CACHED
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
pub fn symbolic() -> &'static TypedPatternMatcher {
    static CACHED: std::sync::LazyLock<TypedPatternMatcher> = std::sync::LazyLock::new(|| {
        symbolic_simple()
            + gep_pushing_patterns()
            + vmin_vmax_collapse_patterns()
            + where_alu_combining_patterns()
            + long_to_int_narrowing_patterns()
            + super::valid_simplification::pm_simplify_valid()
            + sym_phase3_patterns()
            + store_load_folding_patterns()
    });
    &CACHED
}

/// Commutative operand canonicalization for index-type operations.
///
/// Ensures commutative binary ops have operands in canonical order (smaller
/// id on the left). Without this, mathematically equivalent expressions like
/// `R1*8000 + R2*16` and `R2*16 + R1*8000` won't be deduplicated by hash
/// consing, breaking grouping in `expand_vector_index`.
///
/// Follows Tinygrad's approach (symbolic.py:178-182): only applies to
/// index-type operations to avoid breaking vector math merging.
fn commutative_canonicalization() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        for op in binary [Add, Mul, Max, Eq, Ne, And, Or, Xor] {
            r @ op(a, b)
                if r.dtype() == DType::Index && b.id < a.id
                ~> UOp::new(Op::Binary(op, b.clone(), a.clone()), r.dtype()),
        },
    }
}

/// Self-folding patterns.
///
/// Patterns where an operand appears twice:
/// - x // x → 1
/// - x // -1 → -x
/// - (x % y) % y → x % y
/// - x & x → x
/// - x | x → x
pub fn self_folding_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // x // x → 1
        Idiv(x, x) ~> 1.into_uop(x.dtype()),
        // x // -1 → -x
        Idiv(x, _c @const(c_val)) if c_val.is_neg_one() ~> x.neg(),
        // (x % y) % y → x % y
        Mod(Mod(x, y), y) ~> x.mod_(y),
        // x & x → x
        And(x, x) ~> x.clone(),
        // x | x → x
        Or(x, x) ~> x.clone(),
    }
}

/// Zero folding patterns.
///
/// Patterns that fold to zero or false:
/// - x < x → False (non-float only)
/// - x % x → 0
/// - x != x → False (int only)
pub fn zero_folding_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
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
pub fn range_based_mod_div_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Range(end) % end → Range(end) (identity: range values are [0, end), always < end)
        Mod(range @ Range { end, .. }, end) ~> range,
        // Range(end) // end → 0 (all values in [0, end) divide by end to 0)
        Idiv(Range { end, .. }, end)  ~> UOp::index_const(0),

        // x % n → x when 0 <= vmin(x) && vmax(x) < n
        // This handles cases like Range(3) % 3 → Range(3)
        Mod(x, _n @const(n_val)) => {
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
        // (a * n + b) % n → b % n when b >= 0 (truncated mod is correct only for non-negative).
        Mod(Add[Mul[_a, n @const(_m_val)], b], n @const(_n_val)) => {
            if !matches!(b.vmin(), ConstValue::Int(v) if *v >= 0) { return None; }
            Some(b.mod_(n))
        },

        // ((a * n) + b + c) % n → (b + c) % n when (b + c) >= 0.
        Mod(Add[Add[Mul[_a, n @const(_m_val)], b], c], n @const(_n_val)) => {
            let bc = b.add(c);
            if !matches!(bc.vmin(), ConstValue::Int(v) if *v >= 0) { return None; }
            Some(bc.mod_(n))
        },

        // (a * m + b) / n → a + b / n when m == n (distribute division over sum)
        // When b is non-negative and small, this can enable further simplification.
        // Specifically: (a * n + b) / n = a when 0 <= b < n
        // Using commutative [] for Add to match both orderings.
        // Note: We compare m_val and n_val by VALUE, not pointer, since they may be separate UOps.
        Idiv(Add[Mul[a, n @const(_m_val)], b], n @const(n_val)) => {
            let n_int = n_val.try_int()?;
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
            // Fall through: compute a + b / n (only valid when b >= 0 for truncated division)
            if !matches!(b.vmin(), ConstValue::Int(v) if *v >= 0) { return None; }
            let b_div_n = b.idiv(n);
            Some(a.add(&b_div_n))
        },

        // x / n → k when all values of x are in the same bucket [k*n, (k+1)*n)
        // This is the "cancel divmod" rule from Tinygrad's fold_divmod_general.
        // Examples:
        //   Range(3) / 3 → 0 (since Range(3) is 0,1,2 and all /3 = 0)
        //   (64 + Range(8)) / 64 → 1 (since 64..71 all /64 = 1)
        Idiv(x, _n @const(n_val)) => {
            let (vmin, vmax) = VminVmaxProperty::get(x);
            if let (ConstValue::Int(min), ConstValue::Int(max), ConstValue::Int(n_int)) = (vmin, vmax, n_val)
                && n_int > 0 {
                    // Truncation division (Rust's `/` rounds toward zero) matches Morok's
                    // IDIV semantics. n_int > 0 is already guarded above.
                    let min_div = *min / n_int;
                    let max_div = *max / n_int;
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
        Idiv(x @ Add(_, _), n @const(n_val)) => {
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

                    // CRITICAL: All values must be in the same bucket for safe removal.
                    if min_div != max_div {
                        return None;
                    }

                    // Check that adding const_sum keeps values in the same bucket
                    let min_c_div = (*min + const_sum).div_euclid(n_int);
                    let max_c_div = (*max + const_sum).div_euclid(n_int);
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
                    return Some(x.idiv(n));
                }
            None
        },

        // (x + c) // d → x // d when adding c never crosses a bucket boundary
        // Condition: for ALL v in [vmin(x), vmax(x)], (v+c)//d == v//d.
        // A value v crosses a boundary when v%d + c >= d. So the rule is safe iff
        // the maximum remainder in [min, max] satisfies max_rem + c < d.
        Idiv(Add[x, _c @const(c_val)], d @const(d_val)) => {
            let c_int = c_val.try_int()?;
            let d_int = d_val.try_int()?;
            if d_int <= 0 || c_int <= 0 { return None; }

            let (vmin, vmax) = VminVmaxProperty::get(x);
            if let (ConstValue::Int(min), ConstValue::Int(max)) = (vmin, vmax)
                && *min >= 0
            {
                // Max remainder of v%d for v in [min, max]:
                // - if range spans a full cycle (max - min >= d - 1), max_rem = d - 1
                // - if min%d > max%d (modular wrap), max_rem = d - 1
                // - otherwise, max_rem = max%d
                let max_rem = if max - min >= d_int - 1 || *min % d_int > *max % d_int {
                    d_int - 1
                } else {
                    *max % d_int
                };

                if max_rem + c_int < d_int {
                    return Some(x.idiv(d));
                }
            }
            None
        },

        // Phase 1: (x + c) // d → (x + (c % d)) // d + (c // d)
        // When c >= d, split the offset into quotient and remainder parts.
        // This canonicalizes large offsets, allowing further simplification.
        // Based on Tinygrad's divandmod.py:101-104
        Idiv(Add[x, _c @const(c_val)], d @const(d_val)) => {
            let c_int = c_val.try_int()?;
            let d_int = d_val.try_int()?;
            if d_int <= 0 { return None; }

            let c_mod_d = c_int % d_int;
            let c_div_d = c_int / d_int;

            // Only apply if remainder differs from original (i.e., c >= d or c < 0)
            if c_mod_d == c_int { return None; }

            // Guard: BOTH (x + c) AND (x + c%d) must be non-negative.
            // The transform splits: (x+c)//d = (x+c%d)//d + c//d
            // This is only correct when truncated div == floor div (non-negative numerator).
            let (vmin, _) = VminVmaxProperty::get(x);
            if let ConstValue::Int(min) = vmin {
                if min + c_int < 0 || min + c_mod_d < 0 { return None; }
            } else { return None; }

            // Transform: (x + c) // d → (x + c%d) // d + c//d
            let remainder_const = UOp::const_(d.dtype(), ConstValue::Int(c_mod_d));
            let inner = x.add(&remainder_const);
            let div_result = inner.idiv(d);
            let quotient_const = UOp::const_(d.dtype(), ConstValue::Int(c_div_d));
            Some(div_result.add(&quotient_const))
        },

        // Unified divmod simplification (catch-all for IDIV/MOD on Index dtype).
        // Based on Tinygrad's fold_divmod_general (divandmod.py:8-93).
        // Tries rules in priority order: cancel_divmod → remove_nested_mod →
        // fold_binary_numerator → fold_divmod_congruence → gcd_with_remainder →
        // divide_by_gcd → factor_remainder.
        for op in binary [Idiv, Mod] {
            d @ op(x, y) if d.dtype() == DType::Index => crate::symbolic::divmod::fold_divmod_general(op, x, y),
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
pub fn division_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // 0 / 0 → NaN (IEEE 754: 0/0 is indeterminate)
        // NOTE: This must come before x/x → 1 pattern to take priority
        Fdiv(zero1 @ @zero, @zero) if zero1.dtype().is_float()
            ~> UOp::const_(zero1.dtype(), ConstValue::Float(f64::NAN)),
        // (x * 0) / 0 → NaN (anything times zero divided by zero is NaN)
        Fdiv(Mul[_, zero1 @ @zero], @zero) if zero1.dtype().is_float()
            ~> UOp::const_(zero1.dtype(), ConstValue::Float(f64::NAN)),
        // x / x → 1.0 (float division)
        Fdiv(x, x) => x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::one(dt))),
        // (x * y) / y → x
        Fdiv(Mul(x, y), y) ~> x.clone(),
        // (x * y) // y → x
        Idiv(Mul(x, y), y) ~> x.clone(),
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
pub fn cast_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // cast(const) → const
        Cast { src: _c @const(c_val), dtype } => c_val.cast(dtype).map(|v| UOp::const_(dtype.clone(), v)),
        // x.cast(dtype) → x if same dtype
        Cast { src: x, dtype } if x.dtype() == *dtype ~> x.clone(),
        // x.cast(a).cast(b) → x when x.dtype == b and a preserves all values of b
        // This handles cases like: bool.cast(int32).cast(bool) → bool
        Cast { src: Cast { src: x, dtype: intermediate }, dtype: outer }
            if x.dtype() == *outer && can_safe_cast(outer, intermediate)
            ~> x.clone(),
        // x.cast(a).cast(b) → x.cast(b) when a doesn't narrow x
        // This handles widening chains: int8.cast(int32).cast(int64) → int8.cast(int64)
        Cast { src: Cast { src: x, dtype: intermediate }, dtype: outer }
            if can_safe_cast(&x.dtype(), intermediate)
            ~> |x, outer| x.cast(outer.clone()),
    }
}

/// Term combining patterns.
pub fn term_combining_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // x + x → 2*x
        Add(x, x) ~> 2.into_uop(x.dtype()).mul(x),
        // (x * c1) + (x * c2) → x * (c1 + c2)  (Mul[] is commutative, covers c*x too)
        Add(Mul[x, c1 @const(c1_val)], Mul[x, _c2 @const(c2_val)])
            ~> x.mul(&eval_add_typed(c1_val, c2_val, c1.dtype().base())
                .expect("failed to add constants")
                .into_uop(c1.dtype())),
        // x + x*c → x*(c+1) — commutative outer Add
        Add[x, Mul[x, c @const(c_val)]] ~> {
            let one = ConstValue::one(c.dtype().base());
            let new_c = eval_add_typed(c_val, one, c.dtype().base()).expect("failed to add constants");
            x.mul(&UOp::const_(c.dtype(), new_c))
        },
        // (y + x*c0) + x*c1 → y + x*(c0+c1) — commutative outer Add
        Add[Add[y, Mul[x, c0 @const(c0_val)]], Mul[x, _c1 @const(c1_val)]] ~> {
            let new_c = eval_add_typed(c0_val, c1_val, c0.dtype().base()).expect("failed to add constants");
            let xc = x.mul(&UOp::const_(c0.dtype(), new_c));
            y.add(&xc)
        },
        // (y + x) + x*c → y + x*(c+1) — commutative outer Add
        Add[Add[y, x], Mul[x, c @const(c_val)]] ~> {
            let one = ConstValue::one(c.dtype().base());
            let new_c = eval_add_typed(c_val, one, c.dtype().base()).expect("failed to add constants");
            let xc = x.mul(&UOp::const_(c.dtype(), new_c));
            y.add(&xc)
        },
        // (y + x*c) + x → y + x*(c+1) — commutative outer Add
        Add[Add[y, Mul[x, c @const(c_val)]], x] ~> {
            let one = ConstValue::one(c.dtype().base());
            let new_c = eval_add_typed(c_val, one, c.dtype().base()).expect("failed to add constants");
            let xc = x.mul(&UOp::const_(c.dtype(), new_c));
            y.add(&xc)
        },
        // (y + x) + x → y + x*2 — commutative outer Add
        Add[Add[y, x], x] ~> {
            let x2 = 2.into_uop(x.dtype()).mul(x);
            y.add(&x2)
        },
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
pub fn advanced_division_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // (a // b) // c → a // (b * c) if b,c non-zero
        Idiv(Idiv(a, b @const(b_val)), _c @const(c_val)) if !b_val.is_zero() && !c_val.is_zero() ~> {
            let mul = eval_mul_typed(b_val, c_val, b.dtype().base()).expect("failed to multiply constants");
            a.idiv(&UOp::const_(b.dtype(), mul))
        },
        // expr // divisor → expr.divides(divisor) (generic exact division)
        Idiv(expr, divisor @ @const) => expr.divides(divisor),
        // Decomposes x into sum(factor_i * term_i) + const, computes centered remainders,
        // and folds if the remainder range fits in one bucket (rem.vmin//c == rem.vmax//c).
        Mod(x, c @const(c_val)) => crate::symbolic::divmod::fold_divmod_congruence(x, c, c_val, true),
        Idiv(x, c @const(c_val)) => crate::symbolic::divmod::fold_divmod_congruence(x, c, c_val, false),
        // (a + b) // c → (a // c) + (b // c) when both divide evenly
        Idiv(Add(a, b), c @ @const) => Some(a.divides(c)?.add(&b.divides(c)?)),
        // (a - b) // c → (a // c) - (b // c) when both divide evenly
        Idiv(Sub(a, b), c @ @const) => Some(a.divides(c)?.sub(&b.divides(c)?)),
        // c * (a + b) → (c * a) + (c * b)
        Mul(c @ @const, Add(a, b)) ~> c.mul(a).add(&c.mul(b)),
        // (a + b) * c → (a * c) + (b * c)
        Mul(Add(a, b), c @ @const) ~> a.mul(c).add(&b.mul(c)),
    }
}

/// Two-stage ALU folding patterns.
pub fn alu_folding_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // (x + c1) + c2 → x + (c1 + c2) - commutative outer Add
        Add[Add[x, c1 @const(c1_val)], _c2 @const(c2_val)] ~> {
            let csum = eval_add_typed(c1_val, c2_val, c1.dtype().base()).expect("failed to add constants");
            x.add(&UOp::const_(c1.dtype(), csum))
        },
        // Constant pushing: (x + c) + y → (x + y) + c when y is NOT a const
        Add[Add[x, c @const(_c_val)], y] if !matches!(y.op(), Op::Const(_)) ~> x.add(y).add(c),
        // (x * c1) * c2 → x * (c1 * c2) - commutative outer Mul
        Mul[Mul[x, c1 @const(c1_val)], _c2 @const(c2_val)] ~> {
            let cmul = eval_mul_typed(c1_val, c2_val, c1.dtype().base()).expect("failed to multiply constants");
            x.mul(&UOp::const_(c1.dtype(), cmul))
        },
        // (x - c1) + c2 → x + (c2 - c1) or x - (c1 - c2) - commutative outer Add
        Add[Sub(x, c1 @const(c1_val)), _c2 @const(c2_val)] ~> {
            let diff_val = eval_sub_typed(c2_val, c1_val, c1.dtype().base()).expect("failed to subtract constants");
            // Normalize: prefer x - |c| over x + (-c)
            if let ConstValue::Int(v) = diff_val && v < 0 {
                x.sub(&(-v).into_uop(c1.dtype()))
            } else {
                x.add(&UOp::const_(c1.dtype(), diff_val))
            }
        },
        // (x + c1) - c2 → x + (c1 - c2) or x - (c2 - c1) when result is negative
        Sub(Add(x, c1 @const(c1_val)), _c2 @const(c2_val)) ~> {
            let diff_val = eval_sub_typed(c1_val, c2_val, c1.dtype().base()).expect("failed to subtract constants");
            // Normalize: prefer x - |c| over x + (-c)
            if let Some(v) = diff_val.try_int() && v < 0 {
                x.sub(&(-v).into_uop(c1.dtype()))
            } else {
                x.add(&UOp::const_(c1.dtype(), diff_val))
            }
        },
        // (x - c1) - c2 → x - (c1 + c2)
        Sub(Sub(x, c1 @const(c1_val)), _c2 @const(c2_val)) ~> {
            let csum = eval_add_typed(c1_val, c2_val, c1.dtype().base()).expect("failed to add constants");
            x.sub(&UOp::const_(c1.dtype(), csum))
        },
        // SUB canonicalization: a - (b - x) → x + (a - b)
        Sub(a, Sub(b, x)) ~> x.add(&a.sub(b)),
        // Const negation distribution: (-1) * (x + c) → x.neg() + (-c)
        // Only when the Add operand is a const — avoids infinite loop with sym_phase3.
        Mul[_neg @const(nv), Add[x, c @const(cv)]] if nv.is_neg_one() => {
            let neg_one = ConstValue::neg_one(c.dtype().base())?;
            let neg_cv = eval_mul_typed(cv, neg_one, c.dtype().base()).expect("failed to negate constant");
            Some(UOp::neg(x).add(&UOp::const_(c.dtype(), neg_cv)))
        },
    }
}

/// Dead loop elimination patterns.
///
/// - RANGE with vmax ≤ 0 → Const(0)
/// - END with dead ranges → remove dead ranges
/// - REDUCE with all empty ranges → identity element
pub fn dead_loop_patterns() -> &'static TypedPatternMatcher {
    use crate::symbolic::dce::reduce_identity;

    /// Filter dead ranges from END, or unwrap if all dead.
    fn filter_dead_ranges(end_op: &Arc<UOp>) -> Arc<UOp> {
        let Op::End { computation, ranges } = end_op.op() else { unreachable!("filter_dead_ranges called on non-End") };

        let live_ranges: SmallVec<[Arc<UOp>; 4]> = ranges.iter().filter(|r| !is_empty_range(r)).cloned().collect();

        if live_ranges.is_empty() {
            // All ranges dead - return computation directly
            Arc::clone(computation)
        } else {
            // Some ranges dead - create new END with only live ranges
            computation.end(live_ranges)
        }
    }

    /// Check if a Range is trivial (vmin == vmax), meaning only one value.
    /// This matches Tinygrad's simplification: Range(Const) → Const when vmin == vmax.
    fn is_trivial_range(uop: &Arc<UOp>) -> bool {
        let (vmin, vmax) = VminVmaxProperty::get(uop);
        vmin == vmax
    }

    /// Get the constant value for a trivial range (vmin which equals vmax).
    fn trivial_range_value(uop: &Arc<UOp>) -> Arc<UOp> {
        let (vmin, _) = VminVmaxProperty::get(uop);
        UOp::const_(uop.dtype(), *vmin)
    }

    crate::cached_patterns! {
        // RANGE with vmax < 0 (empty/dead) → Const(0)
        r @ Range(_) if is_empty_range(r) ~> UOp::index_const(0),

        // RANGE(Const) with vmin == vmax (trivial, e.g., end=1) → Const(vmin)
        // Matches Tinygrad symbolic.py:211
        r @ Range { end: Const(_) } if is_trivial_range(r) ~> trivial_range_value(r),

        // END with dead ranges → filter or unwrap
        end_op @ End { ranges, .. } if ranges.iter().any(is_empty_range) ~> filter_dead_ranges(end_op),

        // REDUCE with all empty ranges → identity element
        rop @ Reduce { ranges, reduce_op: op, .. } if !ranges.is_empty() && ranges.iter().all(is_empty_range)
          ~> reduce_identity(*op, rop.dtype()),
    }
}

/// Vmin==Vmax collapse patterns.
///
/// When a node's vmin equals vmax, it's provably constant.
/// Only applies to computation nodes (Binary, Unary, Ternary, DefineVar, Special)
/// to avoid collapsing structural nodes like Range, Buffer, etc.
pub fn vmin_vmax_collapse_patterns() -> &'static TypedPatternMatcher {
    use morok_ir::uop::properties::SoundVminVmaxProperty;

    fn is_collapsible(uop: &Arc<UOp>) -> bool {
        matches!(uop.op(), Op::Binary(..) | Op::Unary(..) | Op::Ternary(..) | Op::DefineVar { .. } | Op::Special { .. })
    }

    fn try_collapse(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
        let (vmin, vmax) = SoundVminVmaxProperty::get(uop).as_ref()?;
        if vmin == vmax { Some(uop.const_like(*vmin)) } else { None }
    }

    crate::cached_patterns! {
        // ALU/DefineVar/Special with sound vmin == vmax → const (Tinygrad symbolic.py:210)
        // Uses SoundVminVmaxProperty: returns None for ops with unsound range analysis
        // (LOAD, Pow, Fdiv, etc.), preventing incorrect collapse.
        for op in binary [Add, Mul, Sub, Mod, Max, Pow, Idiv, Fdiv, And, Or, Xor, Shl, Shr, Lt, Le, Eq, Ne, Gt, Ge] {
            r @ op(_, _) if is_collapsible(r) => { let _ = op; try_collapse(r) },
        },
        for op in unary [Neg, Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc, Not, Floor, Ceil, Round] {
            r @ op(_) if is_collapsible(r) => { let _ = op; try_collapse(r) },
        },
        for op in ternary [Where, MulAcc] {
            r @ op(_, _, _) if is_collapsible(r) => { let _ = op; try_collapse(r) },
        },
        // DefineVar/Special with vmin == vmax → const (e.g., Variable with min==max after binding)
        r @ DefineVar { name: _, min_val: _, max_val: _ } => try_collapse(r),
        r @ Special { end: _, name: _ } => try_collapse(r),
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
pub fn dce_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // WHERE with constant condition → select appropriate branch
        Where(cond, true_val, false_val) => {
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
          ~> Arc::clone(x),

        // WHERE(x, false, true) → !x (for bool x)
        Where(x, _t @const(t_val), _f @const(f_val))
          if x.dtype() == DType::Bool && t_val == ConstValue::Bool(false) && f_val == ConstValue::Bool(true)
          ~> x.not(),

        // WHERE(!cond, t, f) → WHERE(cond, f, t) - negated condition swap
        // Guard: don't swap when f contains Invalid — PAD creates WHERE(valid, idx, Invalid),
        // and swapping would move Invalid to the true branch where downstream patterns can't match it.
        // Handles both scalar Invalid and vectorized VECTORIZE(Invalid, ...) from expansion.
        // Tinygrad symbolic.py:201-202 has this same guard.
        Where(Not(cond), t, f)
            if !has_invalid(f)
            => UOp::try_where(Arc::clone(cond), Arc::clone(f), Arc::clone(t)).ok(),

        // WHERE(a, WHERE(b, c, d), d) → WHERE(a & b, c, d) - branch merging
        Where(a, Where(b, c, d), d) => {
            let combined_cond = a.and_(b);
            UOp::try_where(combined_cond, Arc::clone(c), Arc::clone(d)).ok()
        },
    }
}

/// AFTER simplification patterns (Tinygrad symbolic.py:256).
///
/// - AFTER(x, []) → x (empty deps, just passthrough)
pub fn after_simplification_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // AFTER(x, []) → x: empty dependencies means no ordering constraint
        After { passthrough, deps } if deps.is_empty() ~> Arc::clone(passthrough),
    }
}

/// Move WHERE condition to LOAD gate patterns (Tinygrad symbolic.py:360).
///
/// Transforms `WHERE(cond, LOAD(INDEX(buf, idx)), 0)` to `LOAD(INDEX(buf, idx, gate=cond), alt=0)`
/// when the condition can be safely moved into the INDEX's gate field.
///
/// This optimization:
/// 1. Eliminates the WHERE operation overhead
/// 2. Enables hardware predication for masked loads
/// 3. Allows the backend to generate efficient conditional load instructions
///
/// **Critical**: This pattern runs at Stage 8 (Post-Opt Symbolic), BEFORE LOADs are added
/// at Stage 13. Therefore, it matches INDEX directly, not LOAD(INDEX).
///
/// Matches Tinygrad's `pm_move_where_on_load` pattern:
/// ```python
/// (UPat.var("c1").where(UPat.var("buf").index(UPat.var("x")), 0), where_on_load),
/// ```
///
/// Moved clauses are embedded as WHERE(cond, idx, Invalid) in indices[0] instead of
/// the gate field. This prevents gate vectorization during expansion — pm_lower_index_dtype
/// extracts the scalar gate after devectorize.
///
/// The condition can be moved if:
/// - All RANGE dependencies in the condition are within the INDEX's range scope
/// - The condition doesn't depend on other INDEX operations (avoids speculative loads)
pub fn pm_move_where_on_load() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Pattern 1: WHERE(cond, INDEX(buf, idx, None), 0)
        // Embed cond clauses as WHERE-Invalid in INDEX indices[0]
        // Note: Matches INDEX directly (no LOAD), since this runs at Stage 8
        Where(cond, idx @ Index { buffer, indices, gate: None }, f @ const(false_val)) if false_val.is_zero() => {
            where_on_load_index_transform(cond, buffer, indices, f, idx.dtype())
        },

        // Pattern 2: WHERE(cond, 0, INDEX(buf, idx, None)) - inverted pattern
        // Use !cond embedded as WHERE-Invalid
        Where(cond, f @ const(false_val), idx @ Index { buffer, indices, gate: None }) if false_val.is_zero() => {
            let not_cond = cond.not();
            where_on_load_index_transform(&not_cond, buffer, indices, f, idx.dtype())
        },
    }
}

/// Check if a UOp is or contains Invalid (scalar or vectorized).
fn has_invalid(uop: &Arc<UOp>) -> bool {
    match uop.op() {
        Op::Invalid => true,
        Op::Vectorize { elements } => elements.iter().any(|e| matches!(e.op(), Op::Invalid)),
        _ => false,
    }
}

/// Transform WHERE(cond, INDEX(buf, idx), 0) by embedding moveable clauses as WHERE-Invalid in indices[0].
///
/// This is the Stage 8 version that works directly with INDEX, matching Tinygrad's approach.
/// LOADs are added later at Stage 13.
///
/// Supports **partial clause movement** (Tinygrad: where_on_load in symbolic.py):
/// - Splits condition into AND clauses
/// - Moves only clauses where ALL ranges are within index scope AND no load dependencies
/// - Keeps remaining clauses in outer WHERE
/// - Deduplicates clauses already present in indices[0]'s existing WHERE-Invalid validity
///
/// Instead of setting the INDEX gate field (which gets vectorized by the expander),
/// embeds moved clauses as WHERE(combined_cond, clean_idx, Invalid) in indices[0].
/// pm_lower_index_dtype extracts this after devectorize when the gate is always scalar.
fn where_on_load_index_transform(
    cond: &Arc<UOp>,
    idx_buf: &Arc<UOp>,
    indices: &SmallVec<[Arc<UOp>; 4]>,
    false_val: &Arc<UOp>,
    index_dtype: DType,
) -> Option<Arc<UOp>> {
    // Step 1: Split condition into AND clauses
    let c1_clauses = split_and_clauses(cond);

    // Step 2: Get existing validity clauses from indices[0] (handles re-application)
    let existing_valid = indices.first()?.get_valid();
    let c2_clauses: Vec<Arc<UOp>> = if matches!(existing_valid.op(), Op::Const(cv) if cv.0 == ConstValue::Bool(true)) {
        vec![]
    } else {
        split_and_clauses(&existing_valid)
    };

    // Step 3: Find duplicate clauses (already in existing validity)
    let duplicate_ids: std::collections::HashSet<u64> =
        c1_clauses.iter().filter(|c| c2_clauses.iter().any(|c2| c.id == c2.id)).map(|c| c.id).collect();

    // Step 4: Collect RANGE and INDEX ids reachable from indices (index scope)
    // Tinygrad: idx_index = {u for u in idx.backward_slice_with_self if u.op is Ops.INDEX}
    let mut index_ranges = std::collections::HashSet::new();
    let mut idx_indices = std::collections::HashSet::new();
    for idx in indices {
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![idx.clone()];
        while let Some(node) = stack.pop() {
            if !visited.insert(Arc::as_ptr(&node)) {
                continue;
            }
            match node.op() {
                Op::Range { .. } => {
                    index_ranges.insert(node.id);
                }
                Op::Index { .. } => {
                    idx_indices.insert(node.id);
                }
                _ => {}
            }
            node.op().map_child(|child| {
                if !visited.contains(&Arc::as_ptr(child)) {
                    stack.push(child.clone());
                }
            });
        }
    }

    // Step 5: Partition clauses into moveable vs remaining
    // Single DFS per clause: check range scope + index deps simultaneously
    // Tinygrad: can_move checks c.ranges <= idx.ranges AND all INDEX ops are in idx_index
    let (moved_clauses, remaining_clauses): (Vec<_>, Vec<_>) = c1_clauses.iter().cloned().partition(|clause| {
        if duplicate_ids.contains(&clause.id) {
            return true; // Treat as "moved" (but won't add to validity)
        }

        let mut ranges_in_scope = true;
        let mut has_index_deps = false;
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![clause.clone()];
        while let Some(node) = stack.pop() {
            if !visited.insert(Arc::as_ptr(&node)) {
                continue;
            }
            match node.op() {
                Op::Range { .. } if !index_ranges.contains(&node.id) => {
                    ranges_in_scope = false;
                    break; // Out-of-scope range found, can't move
                }
                Op::Index { .. } if !idx_indices.contains(&node.id) => {
                    has_index_deps = true;
                    break; // External INDEX dep found, can't move
                }
                _ => {}
            }
            node.op().map_child(|child| {
                if !visited.contains(&Arc::as_ptr(child)) {
                    stack.push(child.clone());
                }
            });
        }

        ranges_in_scope && !has_index_deps
    });

    // Step 6: If no movement possible and no duplicates removed, return None
    let actually_moved: Vec<_> = moved_clauses.into_iter().filter(|c| !duplicate_ids.contains(&c.id)).collect();

    if actually_moved.is_empty() && duplicate_ids.is_empty() {
        return None; // Nothing to move or deduplicate
    }

    // Step 7: Build combined validity (moved clauses + existing validity)
    let mut validity_clauses: Vec<Arc<UOp>> = actually_moved;
    validity_clauses.extend(c2_clauses);

    // Step 8: Create INDEX with WHERE-Invalid in indices[0], NO gate field
    let clean_idx = indices.first()?.get_idx();
    let new_idx = if validity_clauses.is_empty() {
        clean_idx
    } else {
        let combined_valid = validity_clauses.into_iter().reduce(|a, b| a.and_(&b)).unwrap();
        clean_idx.valid(combined_valid)
    };
    let mut new_indices = indices.clone();
    new_indices[0] = new_idx;

    let new_index = UOp::index()
        .buffer(idx_buf.clone())
        .indices(new_indices)
        .call()
        .expect("where_on_load_index_transform: INDEX construction failed")
        .with_dtype(index_dtype);

    // Step 9: Wrap in remaining WHERE if there are non-moved clauses
    if remaining_clauses.is_empty() {
        Some(new_index)
    } else {
        let remaining_cond = remaining_clauses.into_iter().reduce(|a, b| a.and_(&b)).unwrap();
        UOp::try_where(remaining_cond, new_index, false_val.clone()).ok()
    }
}

/// Split a condition into its AND clauses recursively.
fn split_and_clauses(cond: &Arc<UOp>) -> Vec<Arc<UOp>> {
    match cond.op() {
        Op::Binary(BinaryOp::And, left, right) => {
            let mut result = split_and_clauses(left);
            result.extend(split_and_clauses(right));
            result
        }
        _ => vec![cond.clone()],
    }
}

/// Cast pushing through WHERE patterns.
///
/// - where(s, a, b).cast(dtype) → where(s, a.cast(dtype), b.cast(dtype))
pub fn cast_where_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // cast(where(s, a, b), dtype) → where(s, cast(a, dtype), cast(b, dtype))
        Cast { src: Where(s, a, b), dtype } ~> {
            let cast_a = a.cast(dtype.clone());
            let cast_b = b.cast(dtype.clone());
            UOp::try_where(s.clone(), cast_a, cast_b).expect("failed to create WHERE")
        },
    }
}

/// Comparison patterns.
///
/// Handles all comparison operations with:
/// - Self-comparison fast path (x op x)
/// - Constant folding
/// - Range-based analysis via vmin/vmax
/// - Const offset: (c0 + x) < c1 → x < (c1 - c0)
/// - Negation flip: -x < -y → y < x
pub fn comparison_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        for op in binary [Lt, Le, Eq, Ne, Gt, Ge] {
            op(x, y) => {
                // 1. Self-comparison fast path (non-float only)
                if Arc::ptr_eq(x, y) && !x.dtype().is_float() {
                    let result = match op {
                        BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Ne => ConstValue::Bool(false),
                        BinaryOp::Le | BinaryOp::Ge | BinaryOp::Eq => ConstValue::Bool(true),
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
        Lt(Add[c0 @const(c0_val), x], _c1 @const(c1_val)) ~> {
            let diff = eval_sub_typed(c1_val, c0_val, c0.dtype().base()).expect("failed to evaluate sub");
            x.try_cmplt(&UOp::const_(c0.dtype(), diff)).expect("failed to create cmplt")
        },

        // -x < -y → y < x (negation flip for Lt)
        Lt(Neg(x), Neg(y)) ~> y.try_cmplt(x).expect("failed to create cmplt"),

        // Phase 6: (x // d) < c → x < (c * d) when d > 0
        // This lifts division out of comparisons, enabling further simplification.
        // Based on Tinygrad's symbolic.py:229-230
        Lt(Idiv(x, _d @const(d_val)), _c @const(c_val)) => {
            let d_int = d_val.try_int()?;
            let c_int = c_val.try_int()?;
            if d_int <= 0 { return None; }

            // For x // d < c:
            // - If c > 0: equivalent to x < c * d
            // - If c <= 0: equivalent to x < c * d - (d - 1) = c * d - d + 1
            let bound = if c_int > 0 {
                c_int * d_int
            } else {
                c_int * d_int - (d_int - 1)
            };

            Some(x.try_cmplt(&UOp::const_(x.dtype(), ConstValue::Int(bound))).expect("failed to create cmplt"))
        },

        // c0*x < c1 → x < ceil(c1/c0) for positive c0, c1 with Index dtype
        Lt(Mul[_c0 @const(c0_val), x], _c1 @const(c1_val))
          if x.dtype() == DType::Index
          => {
            let c0 = c0_val.try_int()?;
            let c1 = c1_val.try_int()?;
            if c0 <= 0 || c1 <= 0 { return None; }
            let ceil_div = (c1 + c0 - 1) / c0;
            Some(x.try_cmplt(&UOp::index_const(ceil_div)).expect("failed to create cmplt"))
          },

        // Lt(x, c) with GCD-based folding for Index dtype (Tinygrad symbolic.py:122-126, 236)
        Lt(x, _c @const(cv)) if x.dtype() == DType::Index => {
            let c_int = cv.try_int()?;
            if c_int <= 0 { return None; }
            lt_folding(x, c_int)
          },
    }
}

/// GCD-based Lt folding (Tinygrad symbolic.py:122-126, 236).
///
/// Split x into add terms, partition into unit-factor (|const_factor| <= 1)
/// and non-unit terms. Compute d = gcd(non-unit factors, c). If d > 1 and
/// the unit-factor sum is bounded in [0, d), then x = d*q + r with r in [0, d),
/// so (x < c) iff (q < c/d) since d divides c.
fn lt_folding(x: &Arc<UOp>, c_int: i64) -> Option<Arc<UOp>> {
    let terms = x.split_uop(BinaryOp::Add);
    if terms.len() < 2 {
        return None;
    }

    // Partition terms by const_factor: exactly 1 → unit, otherwise → non-unit
    // Matches Tinygrad: partition(x.split_uop(Ops.ADD), lambda u: u.const_factor() == 1)
    let mut unit_terms = Vec::new();
    let mut non_unit_factors = Vec::new();
    for t in &terms {
        let f = t.const_factor();
        if f == 1 {
            unit_terms.push(Arc::clone(t));
        } else {
            non_unit_factors.push(f);
        }
    }

    if non_unit_factors.is_empty() || unit_terms.is_empty() {
        return None;
    }

    // Compute GCD of non-unit factors AND c (Tinygrad: d = gcd(*factors, c))
    let mut d = c_int.unsigned_abs() as i64;
    for &f in &non_unit_factors {
        d = gcd(d, f);
    }
    if d <= 1 {
        return None;
    }

    // Check that unit sum is in [0, d)
    let unit_sum = super::divmod::uop_sum(&unit_terms, x);
    let (us_vmin, us_vmax) = VminVmaxProperty::get(&unit_sum);
    let us_min = us_vmin.try_int()?;
    let us_max = us_vmax.try_int()?;
    if us_min < 0 || us_max >= d {
        return None;
    }

    // Build the non-unit sum divided by d (Tinygrad: UOp.sum(*np).divides(d))
    let non_unit_terms: Vec<Arc<UOp>> = terms.iter().filter(|t| t.const_factor() != 1).cloned().collect();
    let non_unit_sum = super::divmod::uop_sum(&non_unit_terms, x);
    let q = non_unit_sum.divides_int(d)?;

    // Since d | c, use exact division (no ceiling needed)
    q.try_cmplt(&UOp::index_const(c_int / d)).ok()
}

/// Compute GCD of two positive integers.
fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a as i64
}

/// Boolean logic patterns.
///
/// - !!x → x (double negation elimination)
/// - x ^ x → 0 (xor self-cancellation)
/// - x | !x → true (tautology)
/// - x & !x → false (contradiction)
/// - true | x → true, false & x → false
/// - true & x → x, false | x → x (identity)
/// - (!x) & (!y) → !(x | y) (De Morgan's law)
/// - (!x) | (!y) → !(x & y) (De Morgan's law)
pub fn boolean_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // !!x → x
        Not(Not(x)) ~> x.clone(),
        // x ^ x → 0
        Xor(x, x) => x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::zero(dt))),

        // x | !x → true (tautology) - commutative
        Or[x, Not(x)] if x.dtype() == DType::Bool ~> UOp::const_(DType::Bool, ConstValue::Bool(true)),

        // x & !x → false (contradiction) - commutative
        And[x, Not(x)] if x.dtype() == DType::Bool ~> UOp::const_(DType::Bool, ConstValue::Bool(false)),

        // true | x → true (commutative)
        Or[t @const(t_val), _] if t_val == ConstValue::Bool(true) ~> t.clone(),

        // false & x → false (commutative)
        And[f @const(f_val), _] if f_val == ConstValue::Bool(false) ~> f.clone(),

        // true & x → x (identity, commutative)
        And[_c @const(c_val), x] if c_val == ConstValue::Bool(true) ~> x.clone(),

        // false | x → x (identity, commutative)
        Or[_c @const(c_val), x] if c_val == ConstValue::Bool(false) ~> x.clone(),

        // De Morgan's laws (Tinygrad: decompositions.py)
        // (!x) & (!y) → !(x | y)
        And[Not(x), Not(y)] ~> x.or_(y).not(),

        // (!x) | (!y) → !(x & y)
        Or[Not(x), Not(y)] ~> x.and_(y).not(),
    }
}

/// Min/max elimination via bounds analysis.
///
/// Based on Tinygrad symbolic.py:213:
///   `max(x, y) → x if x.vmin >= y.vmax else y if x.vmax <= y.vmin`
pub fn minmax_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        Max(x, x) ~> x.clone(),
        Max(x, y) => {
            let (x_vmin, x_vmax) = VminVmaxProperty::get(x);
            let (y_vmin, y_vmax) = VminVmaxProperty::get(y);
            if cv_ge(x_vmin, y_vmax) {
                return Some(Arc::clone(x));
            }
            if cv_ge(y_vmin, x_vmax) {
                return Some(Arc::clone(y));
            }
            None
        },
    }
}

/// WHERE condition elimination via bounds analysis.
///
/// Eliminates WHERE(Lt) when the condition is provably always true or false.
/// Uses vmin/vmax to determine if x < c holds for all possible values.
pub fn where_bound_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        Where(Lt(x, c), t, f) => {
            let (x_vmin, x_vmax) = VminVmaxProperty::get(x);
            let (c_vmin, c_vmax) = VminVmaxProperty::get(c);
            // Always true: x.vmax < c.vmin → take true branch
            if cv_lt(x_vmax, c_vmin) { return Some(Arc::clone(t)); }
            // Always false: x.vmin >= c.vmax → take false branch
            if cv_ge(x_vmin, c_vmax) { return Some(Arc::clone(f)); }
            None
        },
    }
}

/// Compare ConstValue: a >= b
fn cv_ge(a: &ConstValue, b: &ConstValue) -> bool {
    match (a, b) {
        (ConstValue::Int(a), ConstValue::Int(b)) => a >= b,
        (ConstValue::UInt(a), ConstValue::UInt(b)) => a >= b,
        (ConstValue::Float(a), ConstValue::Float(b)) => a >= b,
        _ => false,
    }
}

/// Compare ConstValue: a < b
fn cv_lt(a: &ConstValue, b: &ConstValue) -> bool {
    match (a, b) {
        (ConstValue::Int(a), ConstValue::Int(b)) => a < b,
        (ConstValue::UInt(a), ConstValue::UInt(b)) => a < b,
        (ConstValue::Float(a), ConstValue::Float(b)) => a < b,
        _ => false,
    }
}

/// Power patterns.
///
/// - x ** 0 → 1
/// - x ** 1 → x
pub fn power_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // x ** 0 → 1
        Pow(x, _c @const(c_val)) if c_val.is_zero() => x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::one(dt))),
        // x ** 1 → x
        Pow(x, _c @const(c_val)) if c_val.is_one() ~> x.clone(),
    }
}

/// Negation patterns.
///
/// - -(-x) → x (double negation for arithmetic)
pub fn negation_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Double arithmetic negation: -(-x) → x
        Neg(Neg(x)) ~> |x| x.clone(),
    }
}

/// Phase 3 symbolic patterns (full symbolic() only, not symbolic_simple()).
///
/// General negation distribution (Tinygrad symbolic.py:428):
/// - (-1) * (x + y) → x.neg() + y.neg()
pub fn sym_phase3_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // General negation distribution: (-1) * (x + y) → neg(x) + neg(y)
        Mul[_neg @const(nv), Add(x, y)] if nv.is_neg_one() ~> x.neg().add(&y.neg()),
    }
}

/// Store/load folding patterns.
///
/// - STORE(idx, LOAD(idx)) → NOOP (storing what was just loaded is a no-op)
pub fn store_load_folding_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // STORE(idx, LOAD(idx)) → NOOP when the INDEX nodes are ptr_eq
        Store { index, value: Load { index, .. } } ~> UOp::new(Op::Noop, DType::Void),
    }
}

/// WHERE ALU combining patterns.
///
/// When both operands of a binary ALU are WHERE nodes with the same condition,
/// push the ALU inside the WHERE:
/// - ALU(WHERE(c, a, b), WHERE(c, d, e)) → WHERE(c, ALU(a, d), ALU(b, e))
pub fn where_alu_combining_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Tinygrad: only combine when both true branches or both false branches are const
        // Variant 1: both true branches are const
        for op in binary [Add, Mul, Sub, Max, And, Or, Xor] {
            r @ op(Where(c, a @const(_a), b), Where(c, d @const(_d), e)) ~> {
                let true_branch = UOp::new(Op::Binary(op, Arc::clone(a), Arc::clone(d)), r.dtype());
                let false_branch = UOp::new(Op::Binary(op, Arc::clone(b), Arc::clone(e)), r.dtype());
                UOp::try_where(Arc::clone(c), true_branch, false_branch).expect("failed to construct WHERE")
            },
        },
        // Variant 2: both false branches are const
        for op in binary [Add, Mul, Sub, Max, And, Or, Xor] {
            r @ op(Where(c, a, b @const(_b)), Where(c, d, e @const(_e))) ~> {
                let true_branch = UOp::new(Op::Binary(op, Arc::clone(a), Arc::clone(d)), r.dtype());
                let false_branch = UOp::new(Op::Binary(op, Arc::clone(b), Arc::clone(e)), r.dtype());
                UOp::try_where(Arc::clone(c), true_branch, false_branch).expect("failed to construct WHERE")
            },
        },

        // Variant 3: Associative Add — (y + WHERE(c,t,f)) + WHERE(c,tt,ff) → y + WHERE(c,t+tt,f+ff)
        // Tinygrad symbolic.py:207-208: handles WHERE-gates at different nesting levels in Add chains.
        // Both true branches const:
        Add(Add(y, Where(c, t @const(_t), f)), Where(c, tt @const(_tt), ff)) ~> {
            let true_sum = t.add(tt);
            let false_sum = f.add(ff);
            let combined = UOp::try_where(c.clone(), true_sum, false_sum).expect("failed to construct WHERE");
            y.add(&combined)
          },
        // Both false branches const:
        Add(Add(y, Where(c, t, f @const(_f))), Where(c, tt, ff @const(_ff))) ~> {
            let true_sum = t.add(tt);
            let false_sum = f.add(ff);
            let combined = UOp::try_where(c.clone(), true_sum, false_sum).expect("failed to construct WHERE");
            y.add(&combined)
          },
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
pub fn gep_pushing_patterns() -> &'static TypedPatternMatcher {
    /// Check if VECTORIZE is a broadcast (all elements pointer-equal)
    fn is_broadcast(elements: &SmallVec<[Arc<UOp>; 4]>) -> bool {
        elements.first().is_some_and(|first| elements.iter().all(|e| Arc::ptr_eq(e, first)))
    }

    crate::cached_patterns! {
        // 0. GEP on void: GEP(x, _) where x is void → x
        // Removes GEP on GROUP, STORE, and other void-typed nodes.
        // Matches Tinygrad: (UPat(Ops.GEP, src=(UPat(dtype=dtypes.void),)), lambda x: x)
        Gep { vector, .. } if vector.dtype() == DType::Void ~> Arc::clone(vector),

        // 1. GEP composition: GEP(GEP(x, inner), outer) → GEP(x, inner[outer])
        // Note: nested struct patterns require UOp first field, so we extract inner manually
        Gep { vector: Gep { vector: inner_vec, indices: inner_indices }, indices } => {
            let composed: Vec<usize> = indices.iter().map(|&o| inner_indices.get(o).copied()).collect::<Option<Vec<_>>>()?;
            Some(inner_vec.gep(composed))
        },

        // 2. GEP through BROADCAST: extract scalar (MUST be before general VECTORIZE!)
        // BROADCAST = VECTORIZE with all identical elements → GEP([x,x,x,x], [i]) → x
        Gep { vector: Vectorize { elements, .. }, indices } if indices.len() == 1 => {
            is_broadcast(elements).then(|| elements.first().cloned()).flatten()
        },

        // 3. GEP through VECTORIZE: extract single element
        Gep { vector: Vectorize { elements, .. }, indices } if indices.len() == 1 => elements.get(indices[0]).cloned(),

        // 4. GEP on scalar: GEP(x, [i]) where x is scalar → x
        Gep { vector, indices } if vector.dtype().vcount() == 1 && indices.len() == 1 ~> Arc::clone(vector),

        // 5. GEP through VConst: extract element(s)
        Gep { vector: v @ VConst { values }, indices } => {
            let scalar_dtype = v.dtype().scalar().map(DType::Scalar).unwrap_or(DType::Index);
            if indices.len() == 1 {
                values.get(indices[0]).map(|v| UOp::const_(scalar_dtype.clone(), *v))
            } else {
                let selected: Vec<_> = indices.iter().filter_map(|&i| values.get(i).cloned()).collect();
                (selected.len() == indices.len()).then(|| UOp::vconst(selected))
            }
        },

        // 6. Push GEP through ALU ops (Binary, Unary, Ternary).
        // Tinygrad (symbolic.py:166-168): fires for all ALU ops, guarded only by PtrDType.
        // We additionally restrict to Index and Bool dtypes to avoid graph explosion
        // when combining gep_pushing with no_vectorized_alu on high-dimensional kernels.
        // Bool is needed for comparisons: GEP(Lt(VCONST, VECTORIZE), [i]) must push through
        // to produce Lt(CONST, CONST) which then constant-folds (critical for sort correctness).
        Gep { vector, indices }
            if !indices.is_empty()
            && matches!(vector.op(), Op::Binary(..))
            && matches!(vector.dtype().base(), ScalarDType::Index | ScalarDType::Bool)
            && !matches!(vector.dtype(), DType::Ptr { .. })
            => {
                let Op::Binary(bin_op, a, b) = vector.op() else { return None };
                let gep_a = a.gep(indices.clone());
                let gep_b = b.gep(indices.clone());
                if matches!(gep_a.dtype(), DType::Ptr { .. }) { return None; }
                let gep_count = indices.len();
                let scalar_base = vector.dtype().base();
                let result_dtype = if gep_count > 1 {
                    DType::Scalar(scalar_base).vec(gep_count)
                } else {
                    DType::Scalar(scalar_base)
                };
                Some(UOp::new(Op::Binary(*bin_op, gep_a, gep_b), result_dtype))
            },

        // 7. Push GEP through Unary for index/bool types (Tinygrad: no dtype guard)
        Gep { vector, indices }
            if !indices.is_empty()
            && matches!(vector.op(), Op::Unary(..))
            && matches!(vector.dtype().base(), ScalarDType::Index | ScalarDType::Bool)
            && !matches!(vector.dtype(), DType::Ptr { .. })
            => {
                let Op::Unary(un_op, x) = vector.op() else { return None };
                let gep_x = x.gep(indices.clone());
                if matches!(gep_x.dtype(), DType::Ptr { .. }) { return None; }
                Some(UOp::new(Op::Unary(*un_op, gep_x.clone()), gep_x.dtype()))
            },

        // 7b. Push GEP through Ternary for index types only (Tinygrad: dtype=dtypes.index guard)
        Gep { vector: Where(cond, t, f), indices }
            if !indices.is_empty()
            && t.dtype().base() == ScalarDType::Index
            && !matches!(cond.dtype(), DType::Ptr { .. })
            => {
                let gep_cond = cond.gep(indices.clone());
                let gep_t = t.gep(indices.clone());
                let gep_f = f.gep(indices.clone());
                if matches!(gep_t.dtype(), DType::Ptr { .. }) { return None; }
                Some(UOp::new(Op::Ternary(TernaryOp::Where, gep_cond, gep_t.clone(), gep_f), gep_t.dtype()))
            },
        Gep { vector: MulAcc(a, b, c), indices }
            if !indices.is_empty()
            && a.dtype().base() == ScalarDType::Index
            && !matches!(a.dtype(), DType::Ptr { .. })
            => {
                let gep_a = a.gep(indices.clone());
                let gep_b = b.gep(indices.clone());
                let gep_c = c.gep(indices.clone());
                if matches!(gep_a.dtype(), DType::Ptr { .. }) { return None; }
                UOp::try_mulacc(gep_a, gep_b, gep_c).ok()
            },

        // 8. GEP through UNROLL: GEP(UNROLL(x, ...), indices) → GEP(x, indices)
        Gep { vector: Unroll { src, .. }, indices } ~> src.gep(indices.clone()),

        // 9. Identity GEP removal: GEP(x, [0,1,2,...,n-1]) → x
        Gep { vector, indices } if indices.iter().enumerate().all(|(i, &idx)| i == idx) && indices.len() == vector.dtype().vcount()
            ~> Arc::clone(vector),

        // 9b. VECTORIZE of same-source GEPs → single multi-index GEP
        // VECTORIZE(GEP(x, [0]), GEP(x, [1]), ..., GEP(x, [N-1])) → GEP(x, [0, 1, ..., N-1])
        // This collapses the VECTORIZE(scalar_op_0, ..., scalar_op_N-1) structure created by
        // no_vectorized_alu back into a single GEP, which the identity pattern (#9) then removes.
        // Based on Tinygrad's gep_pushing (symbolic.py:173).
        Vectorize { elements }
            if elements.len() > 1 && matches!(elements[0].op(), Op::Gep { .. })
            => {
                // All elements must be single-index GEPs from the same source
                let Op::Gep { vector: first_src, indices: first_idx } = elements[0].op() else { return None };
                if first_idx.len() != 1 { return None; }
                let mut combined = Vec::with_capacity(elements.len());
                combined.push(first_idx[0]);
                for elem in elements.iter().skip(1) {
                    let Op::Gep { vector, indices } = elem.op() else { return None };
                    if indices.len() != 1 || vector.id != first_src.id { return None; }
                    combined.push(indices[0]);
                }
                Some(first_src.gep(combined))
            },

        // 10. GEP through PTRCAT: GEP(PTRCAT([a, b, c, d]), [1, 3]) → PTRCAT([b, d])
        Gep { vector: PtrCat { sources }, indices } => {
            let reordered: Vec<_> = indices.iter().filter_map(|&idx| sources.get(idx).cloned()).collect();
            (reordered.len() == indices.len()).then(|| UOp::ptrcat().sources(reordered).call())
        },

        // 11. GEP through CAT: Extract elements from concatenated vectors
        // GEP(CAT([a<4>, b<4>]), [5]) → GEP(b, [1]) (element 5 = source 1, offset 1)
        // Handles multi-element CAT sources by computing cumulative element offsets.
        Gep { vector: Cat { sources }, indices } => {
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
                        Some(src.gep(vec![offset_in_src]))
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

        // 12. GEP through CAST: GEP(CAST(x, dtype), indices) → CAST(GEP(x, indices), scalar_dtype)
        // Allows GEP to propagate through type casts for better optimization.
        Gep { vector: Cast { src, dtype }, indices } ~> {
            let inner_gep = src.gep(indices.clone());
            // Compute result dtype: use scalar version of target dtype
            // (scalar_dtype() works for both Scalar and Vector, unlike scalar())
            let result_scalar = dtype.scalar_dtype();
            inner_gep.cast(result_scalar)
        },

        // 13. GEP through BITCAST: GEP(BITCAST(x, dtype), indices) → BITCAST(GEP(x, indices), scalar_dtype)
        // Allows GEP to propagate through bitcasts for better optimization.
        Gep { vector: BitCast { src, dtype }, indices } ~> {
            let inner_gep = src.gep(indices.clone());
            // Compute result dtype: use scalar version of target dtype
            let result_scalar = dtype.scalar_dtype();
            inner_gep.bitcast(result_scalar)
        },

        // 14. GEP through WMMA: push GEP through WMMA by selecting corresponding input subgroups.
        // Based on Tinygrad's gep_through_wmma (symbolic.py:140-151).
        //
        // GEP(WMMA(a, b, c), indices) → WMMA(GEP(a, ...), GEP(b, ...), GEP(c, ...))
        //
        // The GEP indices must form regular groups of `out_sz` consecutive elements,
        // where `out_sz` is the product of the output upcast_axes sizes. Each group
        // maps to a "tile" of the WMMA output. The pattern remaps these tile indices
        // to the corresponding input tiles for each source, scaled by that source's
        // vector count.
        Gep { vector: Wmma { a, b, c, metadata }, indices } if !indices.is_empty() => {
            // out_sz: number of output elements per tile (from C/accumulator upcast axes)
            let out_sz: usize = metadata.upcast_axes.c.iter().map(|(_, s)| s).product();
            if out_sz == 0 || indices.len() % out_sz != 0 { return None; }

            // Extract tile base indices: every out_sz-th element from GEP indices
            let tile_idxs: Vec<usize> = indices.iter().step_by(out_sz).copied().collect();

            // Validate: GEP indices must form regular groups of out_sz consecutive elements.
            // For each offset i within a tile, indices[i::out_sz] - i must equal tile_idxs.
            for i in 1..out_sz {
                let adjusted: Option<Vec<usize>> = indices
                    .iter()
                    .skip(i)
                    .step_by(out_sz)
                    .map(|&x| x.checked_sub(i))
                    .collect();
                if adjusted.as_deref() != Some(tile_idxs.as_slice()) { return None; }
            }

            // For each WMMA source, compute the corresponding input GEP indices.
            // Each source has its own per-source size from upcast_axes (A, B, C differ).
            let map_source = |src: &Arc<UOp>, src_idx: usize| -> Arc<UOp> {
                let ssz = metadata.upcast_axes.source_size(src_idx);
                let mut src_indices = Vec::with_capacity(tile_idxs.len() * ssz);
                for &w in &tile_idxs {
                    let group = w / out_sz;
                    let start = group * ssz;
                    src_indices.extend(start..start + ssz);
                }
                src.gep(src_indices)
            };

            // Result dtype matches the GEP output: scalar base × number of extracted elements
            let scalar_base = metadata.dtype_out.base();
            let result_dtype = DType::Scalar(scalar_base).vec(indices.len());

            Some(UOp::new(
                Op::Wmma {
                    a: map_source(a, 0),
                    b: map_source(b, 1),
                    c: map_source(c, 2),
                    metadata: metadata.clone(),
                },
                result_dtype,
            ))
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
pub fn div_mod_recombine_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // x%n + (x//n)*n → x (div-mod identity)
        // Note: duplicate variable names (x, n) auto-generate Arc::ptr_eq checks
        Add[Mod(x, n), Mul[Idiv(x, n), n]]
          ~> |x| Arc::clone(x),

        // ((x//a) % c) + (x // b) * c → x // a
        // Condition: a * c == b (divisor composition)
        // Note: x appears twice, c appears twice → auto ptr_eq checks
        Add[Mod(Idiv(x, a @const(a_val)), c @const(c_val)), Mul[Idiv(x, _b @const(b_val)), c]] => {
            let a_int = a_val.try_int()?;
            let c_int = c_val.try_int()?;
            let b_int = b_val.try_int()?;
            if a_int * c_int == b_int {
                return x.try_div(a).ok();
            }
            None
        },

        // (x % c1) * c2 + (x // c1) * c3 → x * c2
        // Condition: c1 * c2 == c3
        // Note: x appears twice, c1 appears twice → auto ptr_eq checks
        Add[Mul[Mod(x, c1 @const(c1_val)), c2 @const(c2_val)], Mul[Idiv(x, c1), _c3 @const(c3_val)]] => {
            let c1_int = c1_val.try_int()?;
            let c2_int = c2_val.try_int()?;
            let c3_int = c3_val.try_int()?;
            if c1_int * c2_int == c3_int {
                return Some(x.mul(c2));
            }
            None
        },

        // y + (x % n) + (x // n) * n → y + x
        // Note: x appears twice, n appears 3 times → auto ptr_eq for all
        Add[Add[y, Mod(x, n)], Mul[Idiv(x, n), n]] ~> y.add(x),

        // (a//c1 + c2) // c3 → (a + c1*c2) // (c1*c3) (nested division simplification)
        // e.g., (a//2 + 1) // 2 → (a + 2) // 4
        // Guards: c1>0, c3>0, and (a>=0 && c2>=0) or (a<=0 && c2<=0) (same-sign requirement)
        Idiv(Add[Idiv(a, c1 @const(c1_val)), _c2 @const(c2_val)], _c3 @const(c3_val)) => {
            let c1_int = c1_val.try_int().expect("failed to extract int");
            let c2_int = c2_val.try_int().expect("failed to extract int");
            let c3_int = c3_val.try_int().expect("failed to extract int");
            if c1_int <= 0 || c3_int <= 0 { return None; }
            let a_vmin = a.vmin().try_int().expect("failed to extract int from vmin");
            let a_vmax = a.vmax().try_int().expect("failed to extract int from vmax");
            if !((a_vmin >= 0 && c2_int >= 0) || (a_vmax <= 0 && c2_int <= 0)) { return None; }
            let c1_times_c2 = eval_mul_typed(c1_val, c2_val, c1.dtype().base()).expect("failed to evaluate cprod");
            let c1_times_c3 = eval_mul_typed(c1_val, c3_val, c1.dtype().base()).expect("failed to evaluate cprod");
            // (a + c1*c2) // (c1*c3)
            Some(a.add(&UOp::const_(c1.dtype(), c1_times_c2))
             .idiv(&UOp::const_(c1.dtype(), c1_times_c3)))
        },
    }
}

/// Long->Int narrowing patterns.
///
/// Narrows Int64 binary operations to Int32 when both operands and the result
/// fit in i32 range, reducing register pressure and enabling 32-bit ALU usage.
pub fn long_to_int_narrowing_patterns() -> &'static TypedPatternMatcher {
    use morok_ir::uop::properties::SoundVminVmaxProperty;

    fn fits_i32(uop: &Arc<UOp>) -> bool {
        let Some((vmin, vmax)) = SoundVminVmaxProperty::get(uop) else { return false };
        matches!(
            (vmin, vmax),
            (ConstValue::Int(min), ConstValue::Int(max))
                if *min >= i32::MIN as i64 && *max <= i32::MAX as i64
        )
    }

    crate::cached_patterns! {
        for op in binary [Add, Mul, Sub, Mod, Max, Idiv, And, Or, Xor, Shl, Shr] {
            result @ op(x, y)
                if x.dtype() == DType::Scalar(ScalarDType::Int64)
                && fits_i32(x) && fits_i32(y) && fits_i32(result)
                => {
                    let i32_dt = DType::Scalar(ScalarDType::Int32);
                    let i64_dt = DType::Scalar(ScalarDType::Int64);
                    let x32 = x.cast(i32_dt.clone());
                    let y32 = y.cast(i32_dt.clone());
                    let r32 = UOp::new(Op::Binary(op, x32, y32), i32_dt);
                    Some(r32.cast(i64_dt))
                },
        },
    }
}
