//! Range analysis (vmin/vmax) evaluation for UOp operations.
//!
//! This module computes minimum and maximum possible values for operations
//! based on their semantics and input ranges. The analysis is conservative -
//! when in doubt, it returns the full dtype bounds to avoid incorrect optimizations.

use crate::types::{BinaryOp, ConstValue, TernaryOp, UnaryOp};
use crate::{Op, UOp};
use morok_dtype::DType;
use std::cmp::Ordering;
use std::sync::Arc;

/// Compute the minimum and maximum possible values for a UOp.
///
/// Returns a tuple (vmin, vmax) where both values are ConstValue types.
/// The analysis propagates ranges bottom-up through the computation graph.
pub fn compute_vmin_vmax(uop: &Arc<UOp>) -> (ConstValue, ConstValue) {
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::VminVmaxProperty;

    match &uop.op {
        Op::Const(c) => (c.0, c.0),
        Op::VConst { values } => sources_range_values(values, &uop.dtype),
        Op::DefineVar { min_val, max_val, .. } => (ConstValue::Int(*min_val), ConstValue::Int(*max_val)),

        // [0, end-1] ranges: Range, Special (Tinygrad ops.py:763)
        Op::Range { end, .. } | Op::Special { end, .. } => zero_to_end_minus_one(end, &uop.dtype),

        // Propagate source range: Unroll, Gep, Bind (Tinygrad ops.py:764-768)
        Op::Unroll { src, .. } | Op::Bind { var: src, .. } | Op::Gep { vector: src, .. } => {
            let (vmin, vmax) = VminVmaxProperty::get(src);
            (*vmin, *vmax)
        }

        // Union of element ranges: Vectorize, Cat (Tinygrad ops.py:765)
        Op::Vectorize { elements } => sources_range(elements, &uop.dtype),
        Op::Cat { sources } => sources_range(sources, &uop.dtype),

        Op::Unary(op, src) => {
            let (src_min, src_max) = VminVmaxProperty::get(src);
            compute_unary_range(*op, *src_min, *src_max, &uop.dtype)
        }
        Op::Binary(op, a, b) => {
            let (a_min, a_max) = VminVmaxProperty::get(a);
            let (b_min, b_max) = VminVmaxProperty::get(b);
            compute_binary_range(*op, *a_min, *a_max, *b_min, *b_max, &uop.dtype)
        }
        Op::Ternary(op, a, b, c) => {
            let (cond_min, cond_max) = VminVmaxProperty::get(a);
            let (true_min, true_max) = VminVmaxProperty::get(b);
            let (false_min, false_max) = VminVmaxProperty::get(c);
            compute_ternary_range(*op, *cond_min, *cond_max, *true_min, *true_max, *false_min, *false_max, &uop.dtype)
        }

        // Cast: only narrow for monotone targets (float/signed/index). Tinygrad ops.py:769.
        Op::Cast { src, .. } => {
            let dt = &uop.dtype;
            if !(dt.is_float() || dt.is_signed() || *dt == DType::Index) {
                return dtype_bounds(dt);
            }
            let (src_min, src_max) = VminVmaxProperty::get(src);
            let has_special = matches!(src_min, ConstValue::Float(f) if f.is_nan() || f.is_infinite())
                || matches!(src_max, ConstValue::Float(f) if f.is_nan() || f.is_infinite());
            if has_special {
                return dtype_bounds(dt);
            }
            let (target_min, target_max) = dtype_bounds(dt);
            let clamped_min = clamp_value(*src_min, target_min, target_max);
            let clamped_max = clamp_value(*src_max, target_min, target_max);
            (clamped_min.cast(dt).unwrap_or(target_min), clamped_max.cast(dt).unwrap_or(target_max))
        }

        _ => dtype_bounds(&uop.dtype),
    }
}

/// Range [0, end-1] for Range and Special ops.
fn zero_to_end_minus_one(end: &Arc<UOp>, dtype: &DType) -> (ConstValue, ConstValue) {
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::VminVmaxProperty;
    let (_, end_max) = VminVmaxProperty::get(end);
    let max = match end_max {
        ConstValue::Int(v) => ConstValue::Int(v - 1),
        ConstValue::UInt(v) => ConstValue::UInt(v - 1),
        _ => dtype_bounds(dtype).1,
    };
    (ConstValue::Int(0), max)
}

/// Union of ranges across multiple UOp sources (Vectorize, Cat).
fn sources_range(sources: &[Arc<UOp>], dtype: &DType) -> (ConstValue, ConstValue) {
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::VminVmaxProperty;
    if sources.is_empty() {
        return dtype_bounds(dtype);
    }
    let (first_min, first_max) = VminVmaxProperty::get(&sources[0]);
    sources.iter().skip(1).fold((*first_min, *first_max), |(vmin, vmax), src| {
        let (s_min, s_max) = VminVmaxProperty::get(src);
        (min_value(vmin, *s_min), max_value(vmax, *s_max))
    })
}

/// Union of ranges across ConstValue slice (VConst).
fn sources_range_values(values: &[ConstValue], dtype: &DType) -> (ConstValue, ConstValue) {
    if values.is_empty() {
        return dtype_bounds(dtype);
    }
    values.iter().skip(1).fold((values[0], values[0]), |(vmin, vmax), &v| (min_value(vmin, v), max_value(vmax, v)))
}

// ============================================================================
// Unary Operations
// ============================================================================

/// Compute range for unary operations.
fn compute_unary_range(op: UnaryOp, vmin: ConstValue, vmax: ConstValue, dtype: &DType) -> (ConstValue, ConstValue) {
    use crate::uop::eval::eval_unary_op;

    match op {
        UnaryOp::Neg => {
            // Negation flips the range
            let new_min = eval_unary_op(UnaryOp::Neg, vmax).unwrap_or_else(|| dtype_bounds(dtype).0);
            let new_max = eval_unary_op(UnaryOp::Neg, vmin).unwrap_or_else(|| dtype_bounds(dtype).1);
            (new_min, new_max)
        }
        UnaryOp::Abs => {
            // Absolute value: if range crosses zero, min becomes 0
            // Otherwise, we need to take abs of both endpoints and find min/max
            let crosses_zero = match (vmin, vmax) {
                (ConstValue::Int(min), ConstValue::Int(max)) => min <= 0 && max >= 0,
                (ConstValue::Float(min), ConstValue::Float(max)) => min <= 0.0 && max >= 0.0,
                _ => false,
            };

            if crosses_zero {
                // Range includes zero, so min is 0
                let zero = match vmin {
                    ConstValue::Int(_) => ConstValue::Int(0),
                    ConstValue::UInt(_) => ConstValue::UInt(0),
                    ConstValue::Float(_) => ConstValue::Float(0.0),
                    _ => dtype_bounds(dtype).0,
                };

                let abs_min = eval_unary_op(UnaryOp::Abs, vmin);
                let abs_max = eval_unary_op(UnaryOp::Abs, vmax);
                let max_val = match (abs_min, abs_max) {
                    (Some(a), Some(b)) => {
                        if compare_const_values(&a, &b) == Ordering::Greater {
                            a
                        } else {
                            b
                        }
                    }
                    _ => dtype_bounds(dtype).1,
                };
                (zero, max_val)
            } else {
                // Range doesn't cross zero, evaluate at endpoints
                let val_min = eval_unary_op(op, vmin);
                let val_max = eval_unary_op(op, vmax);
                match (val_min, val_max) {
                    (Some(min), Some(max)) => {
                        if compare_const_values(&min, &max) == Ordering::Greater {
                            (max, min)
                        } else {
                            (min, max)
                        }
                    }
                    _ => dtype_bounds(dtype),
                }
            }
        }
        UnaryOp::Sin | UnaryOp::Cos => {
            // Sin and Cos are bounded in [-1, 1] for any input
            // TODO: Could be more precise for small ranges
            (ConstValue::Float(-1.0), ConstValue::Float(1.0))
        }
        UnaryOp::Tan => {
            // Tan is unbounded, so use dtype bounds
            // TODO: Could be more precise for small ranges avoiding discontinuities
            dtype_bounds(dtype)
        }
        UnaryOp::Erf => {
            // Erf is bounded in [-1, 1] for all inputs
            (ConstValue::Float(-1.0), ConstValue::Float(1.0))
        }
        UnaryOp::Sign => {
            // Sign returns -1, 0, or 1
            match vmin {
                ConstValue::Int(_) => (ConstValue::Int(-1), ConstValue::Int(1)),
                ConstValue::Float(_) => (ConstValue::Float(-1.0), ConstValue::Float(1.0)),
                ConstValue::UInt(_) => (ConstValue::UInt(0), ConstValue::UInt(1)),
                _ => dtype_bounds(dtype),
            }
        }
        UnaryOp::Square => {
            // Square: x² - similar to Abs, if range crosses zero, min becomes 0
            let crosses_zero = match (vmin, vmax) {
                (ConstValue::Int(min), ConstValue::Int(max)) => min <= 0 && max >= 0,
                (ConstValue::Float(min), ConstValue::Float(max)) => min <= 0.0 && max >= 0.0,
                _ => false,
            };

            if crosses_zero {
                // Range includes zero, so min is 0
                let zero = match vmin {
                    ConstValue::Int(_) => ConstValue::Int(0),
                    ConstValue::UInt(_) => ConstValue::UInt(0),
                    ConstValue::Float(_) => ConstValue::Float(0.0),
                    _ => dtype_bounds(dtype).0,
                };

                let sq_min = eval_unary_op(UnaryOp::Square, vmin);
                let sq_max = eval_unary_op(UnaryOp::Square, vmax);
                let max_val = match (sq_min, sq_max) {
                    (Some(a), Some(b)) => {
                        if compare_const_values(&a, &b) == Ordering::Greater {
                            a
                        } else {
                            b
                        }
                    }
                    _ => dtype_bounds(dtype).1,
                };
                (zero, max_val)
            } else {
                // Range doesn't cross zero, evaluate at endpoints
                let val_min = eval_unary_op(op, vmin);
                let val_max = eval_unary_op(op, vmax);
                match (val_min, val_max) {
                    (Some(min), Some(max)) => {
                        if compare_const_values(&min, &max) == Ordering::Greater {
                            (max, min)
                        } else {
                            (min, max)
                        }
                    }
                    _ => dtype_bounds(dtype),
                }
            }
        }
        UnaryOp::Not => {
            // Not flips bits/booleans - evaluate at endpoints and swap
            let new_min = eval_unary_op(UnaryOp::Not, vmax).unwrap_or_else(|| dtype_bounds(dtype).0);
            let new_max = eval_unary_op(UnaryOp::Not, vmin).unwrap_or_else(|| dtype_bounds(dtype).1);
            (new_min, new_max)
        }
        UnaryOp::Sqrt
        | UnaryOp::Rsqrt
        | UnaryOp::Exp
        | UnaryOp::Exp2
        | UnaryOp::Log
        | UnaryOp::Log2
        | UnaryOp::Reciprocal
        | UnaryOp::Trunc
        | UnaryOp::Floor
        | UnaryOp::Ceil
        | UnaryOp::Round => {
            // For monotonic or simple functions, evaluate at endpoints
            let val_min = eval_unary_op(op, vmin);
            let val_max = eval_unary_op(op, vmax);

            match (val_min, val_max) {
                (Some(min), Some(max)) => {
                    // Ensure min <= max (for non-monotonic functions)
                    if compare_const_values(&min, &max) == Ordering::Greater { (max, min) } else { (min, max) }
                }
                _ => dtype_bounds(dtype),
            }
        }
    }
}

// ============================================================================
// Binary Operations
// ============================================================================

/// Compute range for binary operations.
fn compute_binary_range(
    op: BinaryOp,
    a_min: ConstValue,
    a_max: ConstValue,
    b_min: ConstValue,
    b_max: ConstValue,
    dtype: &DType,
) -> (ConstValue, ConstValue) {
    use crate::uop::eval::eval_binary_op;

    // Fast path: if both operands are constants, evaluate exactly
    // (except for comparisons which always return full bool range for consistency)
    if a_min == a_max
        && b_min == b_max
        && !matches!(op, BinaryOp::Lt | BinaryOp::Le | BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Gt | BinaryOp::Ge)
    {
        if let Some(val) = eval_binary_op(op, a_min, b_min) {
            return (val, val);
        }
        return dtype_bounds(dtype);
    }

    match op {
        // Arithmetic operations with overflow checking
        BinaryOp::Add => {
            match (a_min, a_max, b_min, b_max) {
                (ConstValue::Int(amin), ConstValue::Int(amax), ConstValue::Int(bmin), ConstValue::Int(bmax)) => {
                    match (amin.checked_add(bmin), amax.checked_add(bmax)) {
                        (Some(min), Some(max)) => (ConstValue::Int(min), ConstValue::Int(max)),
                        _ => dtype_bounds(dtype), // Overflow - return conservative bounds
                    }
                }
                (ConstValue::UInt(amin), ConstValue::UInt(amax), ConstValue::UInt(bmin), ConstValue::UInt(bmax)) => {
                    match (amin.checked_add(bmin), amax.checked_add(bmax)) {
                        (Some(min), Some(max)) => (ConstValue::UInt(min), ConstValue::UInt(max)),
                        _ => dtype_bounds(dtype), // Overflow - return conservative bounds
                    }
                }
                _ => {
                    // Float or fallback - use eval_binary_op (floats don't overflow to wrong values)
                    let min = eval_binary_op(BinaryOp::Add, a_min, b_min).unwrap_or_else(|| dtype_bounds(dtype).0);
                    let max = eval_binary_op(BinaryOp::Add, a_max, b_max).unwrap_or_else(|| dtype_bounds(dtype).1);
                    (min, max)
                }
            }
        }
        BinaryOp::Sub => {
            match (a_min, a_max, b_min, b_max) {
                (ConstValue::Int(amin), ConstValue::Int(amax), ConstValue::Int(bmin), ConstValue::Int(bmax)) => {
                    match (amin.checked_sub(bmax), amax.checked_sub(bmin)) {
                        (Some(min), Some(max)) => (ConstValue::Int(min), ConstValue::Int(max)),
                        _ => dtype_bounds(dtype), // Overflow - return conservative bounds
                    }
                }
                (ConstValue::UInt(amin), ConstValue::UInt(amax), ConstValue::UInt(bmin), ConstValue::UInt(bmax)) => {
                    match (amin.checked_sub(bmax), amax.checked_sub(bmin)) {
                        (Some(min), Some(max)) => (ConstValue::UInt(min), ConstValue::UInt(max)),
                        _ => dtype_bounds(dtype), // Overflow - return conservative bounds
                    }
                }
                _ => {
                    // Float or fallback
                    let min = eval_binary_op(BinaryOp::Sub, a_min, b_max).unwrap_or_else(|| dtype_bounds(dtype).0);
                    let max = eval_binary_op(BinaryOp::Sub, a_max, b_min).unwrap_or_else(|| dtype_bounds(dtype).1);
                    (min, max)
                }
            }
        }
        BinaryOp::Max => {
            let min = eval_binary_op(BinaryOp::Max, a_min, b_min).unwrap_or_else(|| dtype_bounds(dtype).0);
            let max = eval_binary_op(BinaryOp::Max, a_max, b_max).unwrap_or_else(|| dtype_bounds(dtype).1);
            (min, max)
        }

        // Operations requiring all four corners
        BinaryOp::Mul | BinaryOp::Pow => eval_four_corners(op, a_min, a_max, b_min, b_max, dtype),

        // Division operations
        BinaryOp::Idiv | BinaryOp::Fdiv => {
            if contains_zero(b_min, b_max) {
                dtype_bounds(dtype)
            } else {
                eval_four_corners(op, a_min, a_max, b_min, b_max, dtype)
            }
        }

        // Modulo operation — NOT monotonic, four-corner evaluation is unsound.
        // (Tinygrad: ops.py _min_max, Ops.MOD handler)
        BinaryOp::Mod => {
            match (a_min, a_max, b_min, b_max) {
                // Non-negative dividend, positive modulus: a % m ∈ [0, min(a_max, m_max - 1)]
                (ConstValue::Int(a_lo), ConstValue::Int(a_hi), ConstValue::Int(b_lo), ConstValue::Int(b_hi))
                    if a_lo >= 0 && b_lo > 0 =>
                {
                    (ConstValue::Int(0), ConstValue::Int(a_hi.min(b_hi - 1)))
                }
                // Non-positive dividend, positive modulus: result ∈ [-(m_max-1), 0]
                (ConstValue::Int(_a_lo), ConstValue::Int(a_hi), ConstValue::Int(b_lo), ConstValue::Int(b_hi))
                    if a_hi <= 0 && b_lo > 0 =>
                {
                    (ConstValue::Int(-(b_hi - 1)), ConstValue::Int(0))
                }
                // Mixed-sign dividend, positive modulus: result ∈ [-(m_max-1), m_max-1]
                (ConstValue::Int(_), ConstValue::Int(_), ConstValue::Int(b_lo), ConstValue::Int(b_hi)) if b_lo > 0 => {
                    (ConstValue::Int(-(b_hi - 1)), ConstValue::Int(b_hi - 1))
                }
                // Unsigned: always non-negative
                (ConstValue::UInt(_), ConstValue::UInt(a_hi), ConstValue::UInt(b_lo), ConstValue::UInt(b_hi))
                    if b_lo > 0 =>
                {
                    (ConstValue::UInt(0), ConstValue::UInt(a_hi.min(b_hi - 1)))
                }
                _ => dtype_bounds(dtype),
            }
        }

        // Comparison operations - use unified ComparisonAnalyzer
        BinaryOp::Lt | BinaryOp::Le | BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Gt | BinaryOp::Ge => {
            use crate::uop::comparison_analysis::ComparisonAnalyzer;
            ComparisonAnalyzer::get_comparison_range(op, a_min, a_max, b_min, b_max)
        }

        // Bitwise operations
        BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => compute_bitwise_range(op, a_min, a_max, b_min, b_max, dtype),

        // Shift operations
        BinaryOp::Shl | BinaryOp::Shr => compute_shift_range(op, a_min, a_max, b_min, b_max, dtype),

        // PRNG - unpredictable
        BinaryOp::Threefry => dtype_bounds(dtype),
    }
}

/// Compute range for bitwise operations.
fn compute_bitwise_range(
    op: BinaryOp,
    a_min: ConstValue,
    a_max: ConstValue,
    b_min: ConstValue,
    b_max: ConstValue,
    dtype: &DType,
) -> (ConstValue, ConstValue) {
    if dtype == &DType::Bool {
        // For bool, evaluate all combinations
        eval_four_corners(op, a_min, a_max, b_min, b_max, dtype)
    } else {
        match op {
            BinaryOp::And => {
                // AND result is bounded by the smaller operand
                // For positive values: 0 <= result <= min(a_max, b_max)
                if let (ConstValue::Int(a), ConstValue::Int(b)) = (a_max, b_max)
                    && a >= 0
                    && b >= 0
                {
                    return (ConstValue::Int(0), ConstValue::Int(a.min(b)));
                }
                // Conservative for mixed signs or unknowns
                dtype_bounds(dtype)
            }
            _ => dtype_bounds(dtype), // OR, XOR are harder to bound
        }
    }
}

/// Compute range for shift operations.
fn compute_shift_range(
    op: BinaryOp,
    a_min: ConstValue,
    a_max: ConstValue,
    b_min: ConstValue,
    b_max: ConstValue,
    dtype: &DType,
) -> (ConstValue, ConstValue) {
    // Get the bit width of the dtype
    let bit_width = if dtype == &DType::Int8 || dtype == &DType::UInt8 {
        8
    } else if dtype == &DType::Int16 || dtype == &DType::UInt16 {
        16
    } else if dtype == &DType::Int32 || dtype == &DType::UInt32 {
        32
    } else if dtype == &DType::Int64 || dtype == &DType::UInt64 {
        64
    } else {
        return dtype_bounds(dtype); // Unsupported type for shifts
    };

    // Check if shift amount is valid (0 to bit_width-1)
    match (b_min, b_max) {
        (ConstValue::Int(shift_min), ConstValue::Int(shift_max)) if shift_min >= 0 && shift_max < bit_width as i64 => {
            eval_four_corners(op, a_min, a_max, b_min, b_max, dtype)
        }
        (ConstValue::UInt(shift_min), ConstValue::UInt(shift_max))
            if shift_min == 0 && shift_max < bit_width as u64 =>
        {
            eval_four_corners(op, a_min, a_max, b_min, b_max, dtype)
        }
        _ => dtype_bounds(dtype), // Invalid shift amount or range crosses zero
    }
}

// ============================================================================
// Ternary Operations
// ============================================================================

/// Compute range for ternary operations.
#[allow(clippy::too_many_arguments)]
fn compute_ternary_range(
    op: TernaryOp,
    cond_min: ConstValue,
    cond_max: ConstValue,
    true_min: ConstValue,
    true_max: ConstValue,
    false_min: ConstValue,
    false_max: ConstValue,
    dtype: &DType,
) -> (ConstValue, ConstValue) {
    match op {
        TernaryOp::Where => {
            // WHERE: if cond then true_val else false_val
            match (cond_min, cond_max) {
                (ConstValue::Bool(true), ConstValue::Bool(true)) => (true_min, true_max),
                (ConstValue::Bool(false), ConstValue::Bool(false)) => (false_min, false_max),
                _ => {
                    // Could be either branch - take union of ranges
                    let candidates = [true_min, true_max, false_min, false_max];
                    range_union(&candidates)
                }
            }
        }
        TernaryOp::MulAcc => {
            // MulAcc: a * b + c
            // Conservative: evaluate all 8 corners
            use crate::uop::eval::eval_ternary_op;

            let corners = [
                (cond_min, true_min, false_min),
                (cond_min, true_min, false_max),
                (cond_min, true_max, false_min),
                (cond_min, true_max, false_max),
                (cond_max, true_min, false_min),
                (cond_max, true_min, false_max),
                (cond_max, true_max, false_min),
                (cond_max, true_max, false_max),
            ];

            let mut min = None;
            let mut max = None;

            for &(a, b, c) in &corners {
                if let Some(val) = eval_ternary_op(TernaryOp::MulAcc, a, b, c) {
                    min = Some(min.map_or(val, |m| min_value(m, val)));
                    max = Some(max.map_or(val, |m| max_value(m, val)));
                }
            }

            min.zip(max).unwrap_or_else(|| dtype_bounds(dtype))
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Evaluate binary operation at all four corners of input ranges.
fn eval_four_corners(
    op: BinaryOp,
    a_min: ConstValue,
    a_max: ConstValue,
    b_min: ConstValue,
    b_max: ConstValue,
    dtype: &DType,
) -> (ConstValue, ConstValue) {
    use crate::uop::eval::eval_binary_op;

    let corners = [(a_min, b_min), (a_min, b_max), (a_max, b_min), (a_max, b_max)];

    let mut min = None;
    let mut max = None;

    for &(a, b) in &corners {
        if let Some(val) = eval_binary_op(op, a, b) {
            min = Some(min.map_or(val, |m| min_value(m, val)));
            max = Some(max.map_or(val, |m| max_value(m, val)));
        }
    }

    min.zip(max).unwrap_or_else(|| dtype_bounds(dtype))
}

/// Get the minimum and maximum values for a dtype.
fn dtype_bounds(dtype: &DType) -> (ConstValue, ConstValue) {
    let s = dtype.base();
    (ConstValue::min(s), ConstValue::max(s))
}

/// Compare two ConstValues and return the minimum.
fn min_value(a: ConstValue, b: ConstValue) -> ConstValue {
    if compare_const_values(&a, &b) == Ordering::Less { a } else { b }
}

/// Compare two ConstValues and return the maximum.
fn max_value(a: ConstValue, b: ConstValue) -> ConstValue {
    if compare_const_values(&a, &b) == Ordering::Greater { a } else { b }
}

/// Get the union of ranges (min of mins, max of maxes).
fn range_union(values: &[ConstValue]) -> (ConstValue, ConstValue) {
    let min = values.iter().copied().reduce(min_value).unwrap();
    let max = values.iter().copied().reduce(max_value).unwrap();
    (min, max)
}

/// Compare two ConstValues for ordering.
fn compare_const_values(a: &ConstValue, b: &ConstValue) -> Ordering {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => x.cmp(y),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => x.cmp(y),
        (ConstValue::Float(x), ConstValue::Float(y)) => {
            // Handle NaN properly
            if x.is_nan() && y.is_nan() {
                Ordering::Equal
            } else if x.is_nan() {
                Ordering::Greater // NaN is "greater" for consistency
            } else if y.is_nan() {
                Ordering::Less
            } else {
                x.partial_cmp(y).unwrap_or(Ordering::Equal)
            }
        }
        (ConstValue::Bool(x), ConstValue::Bool(y)) => x.cmp(y),
        _ => Ordering::Equal, // Mixed types shouldn't happen
    }
}

/// Check if a range contains zero.
fn contains_zero(min: ConstValue, max: ConstValue) -> bool {
    match (min, max) {
        (ConstValue::Int(min_v), ConstValue::Int(max_v)) => min_v <= 0 && max_v >= 0,
        (ConstValue::UInt(min_v), _) => min_v == 0, // UInt range contains zero iff min is zero
        (ConstValue::Float(min_v), ConstValue::Float(max_v)) => min_v <= 0.0 && max_v >= 0.0,
        _ => false,
    }
}

/// Clamp a value to a range.
fn clamp_value(v: ConstValue, min: ConstValue, max: ConstValue) -> ConstValue {
    if compare_const_values(&v, &min) == Ordering::Less {
        min
    } else if compare_const_values(&v, &max) == Ordering::Greater {
        max
    } else {
        v
    }
}
