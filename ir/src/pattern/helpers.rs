//! Helper functions for pattern matching.
//!
//! These functions are used by the generated pattern matching code to check
//! common conditions like zero/one constants.

use std::sync::Arc;

use crate::ConstValue;

use crate::{Op, UOp};

/// Check if a UOp is a zero constant.
#[inline]
pub fn is_zero(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Const(cv) if cv.0.is_zero())
}

/// Check if a UOp is a one constant.
#[inline]
pub fn is_one(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Const(cv) if cv.0.is_one())
}

/// Check if a UOp is a negative one constant.
#[inline]
pub fn is_neg_one(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Const(cv) if cv.0.is_neg_one())
}

/// Check if a UOp is a non-zero constant.
#[inline]
pub fn is_nonzero(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Const(cv) if !cv.0.is_zero())
}

/// Extract const value if present.
#[inline]
pub fn try_const(uop: &Arc<UOp>) -> Option<&ConstValue> {
    match uop.op() {
        Op::Const(cv) => Some(&cv.0),
        _ => None,
    }
}

/// Check if a UOp is a VConst (vector constant).
#[inline]
pub fn is_vconst(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::VConst { .. })
}

/// Check if a UOp is a pure constant tree (no buffer references).
///
/// Returns true for bare CONST/VCONST, and also for unary transformations
/// of constants (e.g., CAST(CONST), BITCAST(CONST), RESHAPE(CONST),
/// EXPAND(CONST)). These trees have no buffer backing and need
/// `.contiguous()` wrapping before realization.
#[inline]
pub fn is_any_const(uop: &Arc<UOp>) -> bool {
    match uop.op() {
        Op::Const(_) | Op::VConst { .. } => true,
        Op::Cast { src, .. }
        | Op::BitCast { src, .. }
        | Op::Reshape { src, .. }
        | Op::Expand { src, .. }
        | Op::Shrink { src, .. }
        | Op::Pad { src, .. }
        | Op::Permute { src, .. }
        | Op::Flip { src, .. } => is_any_const(src),
        _ => false,
    }
}

/// Extract VConst values if present.
#[inline]
pub fn try_vconst(uop: &Arc<UOp>) -> Option<&Vec<ConstValue>> {
    match uop.op() {
        Op::VConst { values } => Some(values),
        _ => None,
    }
}

/// Extract values from any constant (Const returns single-element slice, VConst returns full slice).
#[inline]
pub fn try_any_const_values(uop: &Arc<UOp>) -> Option<Vec<ConstValue>> {
    match uop.op() {
        Op::Const(cv) => Some(vec![cv.0]),
        Op::VConst { values } => Some(values.clone()),
        _ => None,
    }
}

/// Check if a UOp matches a constant predicate.
#[inline]
pub fn const_matches<F>(uop: &Arc<UOp>, predicate: F) -> bool
where
    F: FnOnce(&ConstValue) -> bool,
{
    match uop.op() {
        Op::Const(cv) => predicate(&cv.0),
        _ => false,
    }
}

#[cfg(test)]
#[path = "../test/unit/pattern/helpers_internal.rs"]
mod tests;
