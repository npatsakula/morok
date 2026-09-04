//! Range analysis (vmin/vmax) evaluation for UOp operations.
//!
//! This module computes minimum and maximum possible values for operations
//! based on their semantics and input ranges. The analysis is conservative -
//! when in doubt, it returns the full dtype bounds to avoid incorrect optimizations.

use crate::ops;
use crate::types::{BinaryOp, ConstValue, TernaryOp, UnaryOp};
use crate::{Op, UOp};
use std::cmp::Ordering;
use std::sync::Arc;
use svod_dtype::{DType, ScalarDType};

/// Best-effort Tinygrad-style range analysis.
///
/// `None` means the caller must use the dtype's conservative analysis bounds.
/// In particular, non-constant floating ALU falls back to `[-inf, +inf]` rather
/// than finite format extrema.
pub fn compute_vmin_vmax(uop: &Arc<UOp>) -> Option<(ConstValue, ConstValue)> {
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::VminVmaxProperty;

    match &uop.op {
        Op::Const(c) => Some((c.0, c.0)),
        Op::VConst(ops::VConst { values }) => sources_range_values(values, &uop.dtype),
        Op::DefineVar(ops::DefineVar { min_val, max_val, .. }) => {
            Some(declared_bounds(*min_val, *max_val, &uop.dtype)?)
        }
        Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => {
            arg.vmin_vmax.as_ref().and_then(|(min, max)| normalize_declared_bounds(min.0, max.0, &uop.dtype))
        }

        // [0, end-1] ranges: Range, Special (Tinygrad ops.py:763)
        Op::Range(ops::Range { end, .. }) | Op::Special(ops::Special { end, .. }) => {
            Some(zero_to_end_minus_one(end, &uop.dtype))
        }

        Op::Bind(ops::Bind { var: src, .. }) => Some(*VminVmaxProperty::get(src)),

        // Union of element ranges: Stack is sound only if all sources are sound.
        Op::Stack(ops::Stack { sources }) => sources_range(sources),

        // Unary: Tinygrad has no explicit unary rules — all fall through to dtype bounds.
        // Our analysis is more aggressive but some ops (Exp2, Log2, Reciprocal on floats)
        // can produce NaN/Inf that breaks monotonicity assumptions. Be conservative.
        Op::Unary(op, src) => {
            let (src_min, src_max) = *VminVmaxProperty::get(src);
            if uop.dtype.is_float() && src_min != src_max {
                return None;
            }
            // Only Neg and Not are truly monotone/anti-monotone for all inputs
            match op {
                UnaryOp::Neg | UnaryOp::Not => Some(compute_unary_range(*op, src_min, src_max, &uop.dtype)),
                _ => None,
            }
        }

        // Binary: match Tinygrad's explicit rules
        Op::Binary(op, a, b) => {
            let (a_min, a_max) = *VminVmaxProperty::get(a);
            let (b_min, b_max) = *VminVmaxProperty::get(b);
            if op.is_comparison() && (a.dtype().is_float() || b.dtype().is_float()) {
                use crate::uop::properties::SoundVminVmaxProperty;

                let Some((a_min, a_max)) = *SoundVminVmaxProperty::get(a) else {
                    return Some(bool_bounds());
                };
                let Some((b_min, b_max)) = *SoundVminVmaxProperty::get(b) else {
                    return Some(bool_bounds());
                };
                if Arc::ptr_eq(a, b) {
                    return Some(reflexive_comparison_range(*op));
                }
                return Some(compute_binary_range(*op, a_min, a_max, b_min, b_max, &uop.dtype));
            }
            // Const-const fast path: any op on constants is sound
            if a_min == a_max && b_min == b_max {
                return Some(compute_binary_range(*op, a_min, a_max, b_min, b_max, &uop.dtype));
            }
            if uop.dtype.is_float() {
                return None;
            }
            match op {
                // Tinygrad has explicit rules for these
                BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Max
                | BinaryOp::FloorMod
                | BinaryOp::FloorDiv
                | BinaryOp::Shl
                | BinaryOp::Shr
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::Gt
                | BinaryOp::Ge => Some(compute_binary_range(*op, a_min, a_max, b_min, b_max, &uop.dtype)),
                // Bool AND/OR have rules in Tinygrad
                BinaryOp::And | BinaryOp::Or if uop.dtype == DType::Bool => {
                    Some(compute_binary_range(*op, a_min, a_max, b_min, b_max, &uop.dtype))
                }
                // AND with constant non-negative mask (Tinygrad ops.py:739-740)
                // Only sound when one operand is constant non-negative.
                BinaryOp::And
                    if uop.dtype.is_int() && b_min == b_max && matches!(b_min, ConstValue::Int(v) if v >= 0) =>
                {
                    Some(compute_binary_range(*op, a_min, a_max, b_min, b_max, &uop.dtype))
                }
                // Pow, Fdiv, XOR, OR/AND (non-bool non-int): unsound for variable ranges
                _ => None,
            }
        }

        // Ternary
        Op::Ternary(op, a, b, c) => {
            let (a_min, a_max) = *VminVmaxProperty::get(a);
            let (b_min, b_max) = *VminVmaxProperty::get(b);
            let (c_min, c_max) = *VminVmaxProperty::get(c);
            // Const-const-const fast path
            if a_min == a_max && b_min == b_max && c_min == c_max {
                return Some(compute_ternary_range(*op, a_min, a_max, b_min, b_max, c_min, c_max, &uop.dtype));
            }
            // WHERE for int only (Tinygrad ops.py:1100).
            match op {
                TernaryOp::Where if uop.dtype.is_int() || uop.dtype == DType::Index => {
                    Some(compute_ternary_range(*op, a_min, a_max, b_min, b_max, c_min, c_max, &uop.dtype))
                }
                _ => None,
            }
        }

        // Cast: only for monotone targets (Tinygrad ops.py:770-771)
        Op::Cast(ops::Cast { src, .. }) => {
            let (src_min, src_max) = *VminVmaxProperty::get(src);
            cast_range(src_min, src_max, &src.dtype(), &uop.dtype)
        }

        // Everything else: LOAD, STORE, INDEX, REDUCE, NOOP, etc.
        _ => None,
    }
}

/// Bounds that are safe to consume as compiler proofs.
///
/// `Some` guarantees that every runtime value is ordered and enclosed by the
/// returned interval. A possible NaN therefore returns `None`, as do floating
/// operations whose overflow, domain, or signed-zero behavior cannot be proven
/// from endpoint bounds alone.
pub fn compute_sound_vmin_vmax(uop: &Arc<UOp>) -> Option<(ConstValue, ConstValue)> {
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::eval::{eval_binary_op, eval_binary_op_typed, eval_ternary_op_typed, eval_unary_op_typed};
    use crate::uop::properties::SoundVminVmaxProperty;

    let sound = |src: &Arc<UOp>| *SoundVminVmaxProperty::get(src);
    match &uop.op {
        Op::Const(c) => ordered_value(c.0).then_some((c.0, c.0)),
        Op::VConst(ops::VConst { values }) => sound_values_range(values),
        Op::DefineVar(ops::DefineVar { min_val, max_val, .. }) => {
            let bounds = declared_bounds(*min_val, *max_val, &uop.dtype)?;
            ordered_range(bounds.0, bounds.1).then_some(bounds)
        }
        Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => {
            let (min, max) = arg.vmin_vmax.as_ref()?;
            let bounds = normalize_declared_bounds(min.0, max.0, &uop.dtype)?;
            ordered_range(bounds.0, bounds.1).then_some(bounds)
        }
        Op::Range(ops::Range { end, .. }) | Op::Special(ops::Special { end, .. }) => {
            let (_, end_max) = sound(end)?;
            let bounds = zero_to_typed_end_minus_one(end_max, &uop.dtype)?;
            ordered_range(bounds.0, bounds.1).then_some(bounds)
        }
        Op::Bind(ops::Bind { var, .. }) => sound(var),
        Op::Stack(ops::Stack { sources }) => sound_sources_range(sources),
        Op::Unary(op, src) => {
            let (min, max) = sound(src)?;
            if uop.dtype.is_float() {
                if min != max {
                    return None;
                }
                let value = eval_unary_op_typed(*op, min, uop.dtype.base())?;
                return ordered_value(value).then_some((value, value));
            }

            matches!(op, UnaryOp::Neg | UnaryOp::Not).then(|| compute_unary_range(*op, min, max, &uop.dtype))
        }
        Op::Binary(op, a, b) => {
            let (a_min, a_max) = sound(a)?;
            let (b_min, b_max) = sound(b)?;
            if a_min == a_max && b_min == b_max {
                let value = if op.is_comparison() {
                    eval_binary_op(*op, a_min, b_min)?
                } else {
                    eval_binary_op_typed(*op, a_min, b_min, uop.dtype.base())?
                };
                return ordered_value(value).then_some((value, value));
            }

            if op.is_comparison() && Arc::ptr_eq(a, b) {
                return Some(reflexive_comparison_range(*op));
            }

            if uop.dtype.is_float() {
                return (*op == BinaryOp::Max)
                    .then(|| compute_binary_range(*op, a_min, a_max, b_min, b_max, &uop.dtype));
            }
            match op {
                BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Max
                | BinaryOp::FloorMod
                | BinaryOp::CMod
                | BinaryOp::FloorDiv
                | BinaryOp::CDiv
                | BinaryOp::Shl
                | BinaryOp::Shr
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::Gt
                | BinaryOp::Ge => Some(compute_binary_range(*op, a_min, a_max, b_min, b_max, &uop.dtype)),
                BinaryOp::And | BinaryOp::Or if uop.dtype == DType::Bool => {
                    Some(compute_binary_range(*op, a_min, a_max, b_min, b_max, &uop.dtype))
                }
                BinaryOp::And
                    if uop.dtype.is_int() && b_min == b_max && matches!(b_min, ConstValue::Int(v) if v >= 0) =>
                {
                    Some(compute_binary_range(*op, a_min, a_max, b_min, b_max, &uop.dtype))
                }
                _ => None,
            }
        }
        Op::Ternary(op, cond, true_value, false_value) => {
            let (cond_min, cond_max) = sound(cond)?;
            let (true_min, true_max) = sound(true_value)?;
            let (false_min, false_max) = sound(false_value)?;
            if cond_min == cond_max && true_min == true_max && false_min == false_max {
                let value = eval_ternary_op_typed(*op, cond_min, true_min, false_min, uop.dtype.base())?;
                return ordered_value(value).then_some((value, value));
            }
            match op {
                TernaryOp::Where => Some(compute_ternary_range(
                    *op, cond_min, cond_max, true_min, true_max, false_min, false_max, &uop.dtype,
                )),
                TernaryOp::MulAcc => None,
            }
        }
        Op::Cast(ops::Cast { src, .. }) => sound_cast_range(src, &uop.dtype),
        _ => None,
    }
}

fn declared_bounds(min: i64, max: i64, dtype: &DType) -> Option<(ConstValue, ConstValue)> {
    normalize_declared_bounds(ConstValue::Int(min), ConstValue::Int(max), dtype)
}

fn normalize_declared_bounds(min: ConstValue, max: ConstValue, dtype: &DType) -> Option<(ConstValue, ConstValue)> {
    if dtype.is_float() {
        let scalar = dtype.scalar_dtype();
        Some((min.cast(&scalar)?, max.cast(&scalar)?))
    } else if dtype.base().is_int() || dtype.is_bool() {
        normalize_declared_integer_bounds(min, max, dtype)
    } else {
        Some((min, max))
    }
}

fn sound_cast_range(src: &Arc<UOp>, target: &DType) -> Option<(ConstValue, ConstValue)> {
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::SoundVminVmaxProperty;

    let (src_min, src_max) = (*SoundVminVmaxProperty::get(src))?;
    if target.base().is_int() && !src.dtype().is_float() && src_min != src_max {
        let (target_min, target_max) = integer_dtype_bounds(target.base())?;
        if integer_value(src_min)? < target_min || integer_value(src_max)? > target_max {
            return None;
        }
    }
    cast_range(src_min, src_max, &src.dtype(), target)
}

fn cast_range(
    src_min: ConstValue,
    src_max: ConstValue,
    src_dtype: &DType,
    target: &DType,
) -> Option<(ConstValue, ConstValue)> {
    let scalar_target = target.scalar_dtype();
    if src_min == src_max {
        let value = src_min.cast(&scalar_target)?;
        return ordered_value(value).then_some((value, value));
    }

    if target.is_bool() {
        let can_be_zero = contains_zero(src_min, src_max);
        return Some(if can_be_zero { bool_bounds() } else { (ConstValue::Bool(true), ConstValue::Bool(true)) });
    }
    if target.is_float() {
        let min = src_min.cast(&scalar_target)?;
        let max = src_max.cast(&scalar_target)?;
        return ordered_range(min, max).then_some((min, max));
    }
    if src_dtype.is_float() {
        return None;
    }
    if target.base().is_int() {
        return cast_integer_or_bool_range(src_min, src_max, target);
    }
    None
}

fn cast_integer_or_bool_range(min: ConstValue, max: ConstValue, target: &DType) -> Option<(ConstValue, ConstValue)> {
    if target.is_bool() {
        if min == max {
            let value = min.cast(&DType::Bool)?;
            return Some((value, value));
        }
        return Some(if contains_zero(min, max) {
            bool_bounds()
        } else {
            (ConstValue::Bool(true), ConstValue::Bool(true))
        });
    }

    let min = integer_value(min)?;
    let max = integer_value(max)?;
    let (target_min, target_max) = integer_dtype_bounds(target.base())?;
    if min < target_min || max > target_max {
        return Some(dtype_bounds(target));
    }
    commit_integer_math_range(min, max, target)
}

fn normalize_declared_integer_bounds(
    min: ConstValue,
    max: ConstValue,
    dtype: &DType,
) -> Option<(ConstValue, ConstValue)> {
    if dtype.is_bool() {
        return cast_integer_or_bool_range(min, max, dtype);
    }
    let min = integer_value(min)?;
    let max = integer_value(max)?;
    let (dtype_min, dtype_max) = integer_dtype_bounds(dtype.base())?;
    commit_integer_math_range(min.max(dtype_min), max.min(dtype_max), dtype)
}

fn bool_bounds() -> (ConstValue, ConstValue) {
    (ConstValue::Bool(false), ConstValue::Bool(true))
}

fn reflexive_comparison_range(op: BinaryOp) -> (ConstValue, ConstValue) {
    let value = match op {
        BinaryOp::Lt | BinaryOp::Ne | BinaryOp::Gt => false,
        BinaryOp::Le | BinaryOp::Eq | BinaryOp::Ge => true,
        _ => return bool_bounds(),
    };
    (ConstValue::Bool(value), ConstValue::Bool(value))
}

fn ordered_value(value: ConstValue) -> bool {
    !matches!(value, ConstValue::Float(v) if v.is_nan()) && value != ConstValue::Invalid
}

fn ordered_range(min: ConstValue, max: ConstValue) -> bool {
    ordered_value(min) && ordered_value(max) && compare_const_values(&min, &max) != Ordering::Greater
}

/// Range [0, end-1] for Range and Special ops.
fn zero_to_end_minus_one(end: &Arc<UOp>, dtype: &DType) -> (ConstValue, ConstValue) {
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::VminVmaxProperty;
    let (_, end_max) = VminVmaxProperty::get(end);
    zero_to_typed_end_minus_one(*end_max, dtype).unwrap_or_else(|| dtype_bounds(dtype))
}

fn zero_to_typed_end_minus_one(end_max: ConstValue, dtype: &DType) -> Option<(ConstValue, ConstValue)> {
    let end = integer_value(end_max)?;
    let max = end.checked_sub(1)?;
    let (dtype_min, dtype_max) = integer_dtype_bounds(dtype.base())?;
    if max < dtype_min || max > dtype_max {
        return None;
    }
    let zero = ConstValue::zero(dtype.base()).cast(&dtype.scalar_dtype())?;
    let max = if dtype.base().is_unsigned() {
        ConstValue::UInt(u64::try_from(max).ok()?)
    } else {
        ConstValue::Int(i64::try_from(max).ok()?)
    }
    .cast(&dtype.scalar_dtype())?;
    Some((zero, max))
}

/// Sound union of ranges — returns None if any source is unsound.
fn sound_sources_range(sources: &[Arc<UOp>]) -> Option<(ConstValue, ConstValue)> {
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::SoundVminVmaxProperty;
    if sources.is_empty() {
        return None;
    }
    let (first_min, first_max) = (*SoundVminVmaxProperty::get(&sources[0]))?;
    sources.iter().skip(1).try_fold((first_min, first_max), |(vmin, vmax), src| {
        let (s_min, s_max) = (*SoundVminVmaxProperty::get(src))?;
        Some((min_value(vmin, s_min), max_value(vmax, s_max)))
    })
}

fn sources_range(sources: &[Arc<UOp>]) -> Option<(ConstValue, ConstValue)> {
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::VminVmaxProperty;
    let first = sources.first()?;
    let (first_min, first_max) = *VminVmaxProperty::get(first);
    let range = sources.iter().skip(1).fold((first_min, first_max), |(vmin, vmax), src| {
        let (src_min, src_max) = *VminVmaxProperty::get(src);
        (min_value(vmin, src_min), max_value(vmax, src_max))
    });
    ordered_range(range.0, range.1).then_some(range)
}

/// Union of ranges across ConstValue slice (VConst).
fn sources_range_values(values: &[ConstValue], _dtype: &DType) -> Option<(ConstValue, ConstValue)> {
    if values.is_empty() {
        return None;
    }
    let range =
        values.iter().skip(1).fold((values[0], values[0]), |(vmin, vmax), &v| (min_value(vmin, v), max_value(vmax, v)));
    ordered_range(range.0, range.1).then_some(range)
}

fn sound_values_range(values: &[ConstValue]) -> Option<(ConstValue, ConstValue)> {
    let range = sources_range_values(values, &DType::Void)?;
    values.iter().all(|value| ordered_value(*value)).then_some(range)
}

// ============================================================================
// Unary Operations
// ============================================================================

/// Compute range for unary operations.
fn compute_unary_range(op: UnaryOp, vmin: ConstValue, vmax: ConstValue, dtype: &DType) -> (ConstValue, ConstValue) {
    use crate::uop::eval::{eval_unary_op, eval_unary_op_typed};

    if vmin == vmax
        && let Some(value) = eval_unary_op_typed(op, vmin, dtype.base())
    {
        return (value, value);
    }

    if dtype.base().is_int() {
        return match op {
            UnaryOp::Neg => {
                let Some(min) = integer_value(vmax).and_then(i128::checked_neg) else {
                    return dtype_bounds(dtype);
                };
                let Some(max) = integer_value(vmin).and_then(i128::checked_neg) else {
                    return dtype_bounds(dtype);
                };
                commit_integer_math_range(min, max, dtype).unwrap_or_else(|| dtype_bounds(dtype))
            }
            UnaryOp::Not => {
                let min = eval_unary_op_typed(op, vmax, dtype.base());
                let max = eval_unary_op_typed(op, vmin, dtype.base());
                min.zip(max).filter(|&(min, max)| ordered_range(min, max)).unwrap_or_else(|| dtype_bounds(dtype))
            }
            _ => dtype_bounds(dtype),
        };
    }

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
    use crate::uop::eval::{eval_binary_op, eval_binary_op_typed};

    // Fast path: if both operands are constants, evaluate exactly
    // (except for comparisons which always return full bool range for consistency)
    if a_min == a_max
        && b_min == b_max
        && !matches!(op, BinaryOp::Lt | BinaryOp::Le | BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Gt | BinaryOp::Ge)
    {
        if let Some(val) = eval_binary_op_typed(op, a_min, b_min, dtype.base()) {
            return (val, val);
        }
        return dtype_bounds(dtype);
    }

    if dtype.base().is_int()
        && let Some(range) = compute_typed_integer_binary_range(op, a_min, a_max, b_min, b_max, dtype)
    {
        return range;
    }
    if dtype.base().is_int() && op == BinaryOp::Pow {
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
        BinaryOp::FloorDiv | BinaryOp::CDiv | BinaryOp::Fdiv => {
            if contains_zero(b_min, b_max) {
                dtype_bounds(dtype)
            } else {
                eval_four_corners(op, a_min, a_max, b_min, b_max, dtype)
            }
        }

        // Floor modulo has the sign of the divisor.
        BinaryOp::FloorMod => match (a_min, a_max, b_min, b_max) {
            (ConstValue::Int(_), ConstValue::Int(_), ConstValue::Int(b_lo), ConstValue::Int(b_hi)) if b_lo > 0 => {
                (ConstValue::Int(0), ConstValue::Int(b_hi - 1))
            }
            (ConstValue::Int(_), ConstValue::Int(_), ConstValue::Int(b_lo), ConstValue::Int(b_hi)) if b_hi < 0 => {
                (ConstValue::Int(b_lo + 1), ConstValue::Int(0))
            }
            (ConstValue::UInt(_), ConstValue::UInt(_), ConstValue::UInt(b_lo), ConstValue::UInt(b_hi)) if b_lo > 0 => {
                (ConstValue::UInt(0), ConstValue::UInt(b_hi - 1))
            }
            _ => dtype_bounds(dtype),
        },

        // C remainder is not monotonic; its sign follows the dividend.
        BinaryOp::CMod => {
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

fn compute_typed_integer_binary_range(
    op: BinaryOp,
    a_min: ConstValue,
    a_max: ConstValue,
    b_min: ConstValue,
    b_max: ConstValue,
    dtype: &DType,
) -> Option<(ConstValue, ConstValue)> {
    let amin = integer_value(a_min)?;
    let amax = integer_value(a_max)?;
    let bmin = integer_value(b_min)?;
    let bmax = integer_value(b_max)?;
    let full = || dtype_bounds(dtype);
    let (dtype_min, _) = integer_dtype_bounds(dtype.base())?;
    if matches!(op, BinaryOp::FloorDiv | BinaryOp::CDiv | BinaryOp::FloorMod | BinaryOp::CMod)
        && amin <= dtype_min
        && dtype_min <= amax
        && bmin <= -1
        && -1 <= bmax
    {
        return Some(full());
    }

    let math_range = match op {
        BinaryOp::Add => amin.checked_add(bmin).zip(amax.checked_add(bmax)),
        BinaryOp::Sub => amin.checked_sub(bmax).zip(amax.checked_sub(bmin)),
        BinaryOp::Mul => integer_corner_range(amin, amax, bmin, bmax, i128::checked_mul),
        BinaryOp::Max => Some((amin.max(bmin), amax.max(bmax))),
        BinaryOp::FloorDiv | BinaryOp::CDiv => {
            if bmin <= 0 && bmax >= 0 {
                return Some(full());
            }
            integer_corner_range(amin, amax, bmin, bmax, |a, b| match op {
                BinaryOp::FloorDiv => floor_div_i128(a, b),
                BinaryOp::CDiv => a.checked_div(b),
                _ => unreachable!(),
            })
        }
        BinaryOp::FloorMod | BinaryOp::CMod => {
            if bmin <= 0 && bmax >= 0 {
                return Some(full());
            }
            let limit = bmin.checked_abs()?.max(bmax.checked_abs()?).checked_sub(1)?;
            if op == BinaryOp::FloorMod {
                if bmin > 0 { Some((0, limit)) } else { Some((-limit, 0)) }
            } else if amin >= 0 {
                Some((0, amax.min(limit)))
            } else if amax <= 0 {
                Some((amin.max(-limit), 0))
            } else {
                Some((amin.max(-limit), amax.min(limit)))
            }
        }
        BinaryOp::Shl | BinaryOp::Shr => {
            let width = integer_bit_width(dtype.base())?;
            if bmin < 0 || bmax >= i128::from(width) {
                return Some(full());
            }
            integer_shift_range(op, amin, amax, bmin as u32, bmax as u32)
        }
        // Variable integer powers and bitwise operations are not monotone. The
        // caller's existing conservative rules handle them.
        BinaryOp::Pow | BinaryOp::And | BinaryOp::Or | BinaryOp::Xor | BinaryOp::Threefry => return None,
        BinaryOp::Fdiv | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Gt | BinaryOp::Ge => {
            return None;
        }
    };

    Some(math_range.and_then(|(min, max)| commit_integer_math_range(min, max, dtype)).unwrap_or_else(full))
}

fn integer_corner_range(
    amin: i128,
    amax: i128,
    bmin: i128,
    bmax: i128,
    eval: impl Fn(i128, i128) -> Option<i128>,
) -> Option<(i128, i128)> {
    [(amin, bmin), (amin, bmax), (amax, bmin), (amax, bmax)].into_iter().map(|(a, b)| eval(a, b)).try_fold(
        None,
        |range, value| {
            let value = value?;
            Some(Some(range.map_or((value, value), |(min, max): (i128, i128)| (min.min(value), max.max(value)))))
        },
    )?
}

fn integer_shift_range(op: BinaryOp, amin: i128, amax: i128, bmin: u32, bmax: u32) -> Option<(i128, i128)> {
    integer_corner_range(amin, amax, i128::from(bmin), i128::from(bmax), |value, shift| {
        let shift = u32::try_from(shift).ok()?;
        match op {
            BinaryOp::Shl => value.checked_mul(1i128.checked_shl(shift)?),
            BinaryOp::Shr => value.checked_shr(shift),
            _ => None,
        }
    })
}

fn floor_div_i128(a: i128, b: i128) -> Option<i128> {
    let quotient = a.checked_div(b)?;
    let remainder = a.checked_rem(b)?;
    if remainder != 0 && (a < 0) != (b < 0) { quotient.checked_sub(1) } else { Some(quotient) }
}

fn integer_value(value: ConstValue) -> Option<i128> {
    match value {
        ConstValue::Int(value) => Some(i128::from(value)),
        ConstValue::UInt(value) => Some(i128::from(value)),
        ConstValue::Bool(value) => Some(i128::from(value)),
        _ => None,
    }
}

fn integer_dtype_bounds(dtype: ScalarDType) -> Option<(i128, i128)> {
    let min = integer_value(ConstValue::min(dtype))?;
    let max = integer_value(ConstValue::max(dtype))?;
    Some((min, max))
}

fn integer_bit_width(dtype: ScalarDType) -> Option<u32> {
    use ScalarDType::*;
    match dtype {
        Int8 | UInt8 => Some(8),
        Int16 | UInt16 => Some(16),
        Int32 | UInt32 => Some(32),
        WeakInt | Int64 | UInt64 | Index => Some(64),
        _ => None,
    }
}

fn commit_integer_math_range(min: i128, max: i128, dtype: &DType) -> Option<(ConstValue, ConstValue)> {
    let scalar = dtype.base();
    let (dtype_min, dtype_max) = integer_dtype_bounds(scalar)?;
    if min > max || min < dtype_min || max > dtype_max {
        return None;
    }
    let make_value = |value: i128| {
        let uncommitted = if scalar.is_unsigned() {
            ConstValue::UInt(u64::try_from(value).ok()?)
        } else {
            ConstValue::Int(i64::try_from(value).ok()?)
        };
        uncommitted.cast(&DType::Scalar(scalar))
    };
    Some((make_value(min)?, make_value(max)?))
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
                // Sound only with constant non-negative mask: 0 <= (x & mask) <= mask
                // (compute_sound_vmin_vmax ensures b_min == b_max >= 0)
                if let (ConstValue::Int(bmin), ConstValue::Int(bmax)) = (b_min, b_max)
                    && bmin == bmax
                    && bmin >= 0
                {
                    return (ConstValue::Int(0), ConstValue::Int(bmax));
                }
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
            if dtype.base().is_int() {
                return compute_typed_integer_mulacc_range(
                    cond_min, cond_max, true_min, true_max, false_min, false_max, dtype,
                );
            }

            // MulAcc: a * b + c. Floating corners are only used by ordinary
            // best-effort analysis; sound non-constant float MULACC declines.
            use crate::uop::eval::eval_ternary_op_typed;

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
                if let Some(val) = eval_ternary_op_typed(TernaryOp::MulAcc, a, b, c, dtype.base()) {
                    min = Some(min.map_or(val, |m| min_value(m, val)));
                    max = Some(max.map_or(val, |m| max_value(m, val)));
                }
            }

            min.zip(max).unwrap_or_else(|| dtype_bounds(dtype))
        }
    }
}

fn compute_typed_integer_mulacc_range(
    a_min: ConstValue,
    a_max: ConstValue,
    b_min: ConstValue,
    b_max: ConstValue,
    c_min: ConstValue,
    c_max: ConstValue,
    dtype: &DType,
) -> (ConstValue, ConstValue) {
    let Some((amin, amax, bmin, bmax, cmin, cmax)) = integer_value(a_min)
        .zip(integer_value(a_max))
        .zip(integer_value(b_min))
        .zip(integer_value(b_max))
        .zip(integer_value(c_min))
        .zip(integer_value(c_max))
        .map(|(((((amin, amax), bmin), bmax), cmin), cmax)| (amin, amax, bmin, bmax, cmin, cmax))
    else {
        return dtype_bounds(dtype);
    };
    let range = [amin, amax]
        .into_iter()
        .flat_map(|a| [bmin, bmax].into_iter().map(move |b| (a, b)))
        .flat_map(|(a, b)| [cmin, cmax].into_iter().map(move |c| (a, b, c)))
        .map(|(a, b, c)| a.checked_mul(b).and_then(|product| product.checked_add(c)))
        .try_fold(None, |range, value| {
            let value = value?;
            Some(Some(range.map_or((value, value), |(min, max): (i128, i128)| (min.min(value), max.max(value)))))
        })
        .flatten();
    range.and_then(|(min, max)| commit_integer_math_range(min, max, dtype)).unwrap_or_else(|| dtype_bounds(dtype))
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

/// Conservative compiler-analysis bounds for a dtype.
///
/// Unlike finite numeric format limits, every floating dtype uses
/// `[-inf, +inf]`. This includes vector dtypes through their scalar base.
pub fn analysis_bounds(dtype: &DType) -> (ConstValue, ConstValue) {
    let s = dtype.base();
    if s.is_float() {
        let (min, max) = dtype.analysis_bounds();
        (ConstValue::Float(min), ConstValue::Float(max))
    } else {
        (ConstValue::min(s), ConstValue::max(s))
    }
}

#[doc(hidden)]
pub fn dtype_bounds(dtype: &DType) -> (ConstValue, ConstValue) {
    analysis_bounds(dtype)
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
            } else if *x == 0.0 && *y == 0.0 {
                // Preserve the signed-zero span. Numerically -0.0 == +0.0,
                // but collapsing their union to one endpoint can make a later
                // reciprocal look exact even though it may be either infinity.
                x.is_sign_negative().cmp(&y.is_sign_negative()).reverse()
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
