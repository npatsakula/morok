//! Dead Code Elimination (DCE) helpers for symbolic optimization.

use morok_ir::{Op, UOp};
use std::sync::Arc;

/// Check if a range is provably empty (no iterations).
///
/// A range is empty if its maximum iteration count is < 0 (negative).
/// A range with vmax = 0 means ONE iteration (0..=0), NOT zero iterations.
///
/// IMPORTANT: vmax <= 0 would be incorrect because:
/// - vmax = 0: 1 iteration (valid, e.g., after full unroll splits REDUCE axis)
/// - vmax = -1: 0 iterations (truly empty/dead)
/// - vmax < 0: unreachable/dead code
///
/// Also recognizes `Const(0)` with Index dtype as a dead range marker.
/// This happens after the rewrite engine transforms dead RANGE → Const(0).
pub fn is_empty_range(uop: &Arc<UOp>) -> bool {
    use morok_dtype::DType;
    use morok_ir::types::ConstValue;
    use morok_ir::uop::cached_property::CachedProperty;
    use morok_ir::uop::properties::VminVmaxProperty;

    match uop.op() {
        Op::Range { .. } => {
            // Get the RANGE's vmin_vmax (not the end's!)
            // RANGE vmax = end_max - 1, so:
            // - end=0 → vmax=-1 (empty, 0 iterations)
            // - end=1 → vmax=0 (one iteration: [0])
            // - end=2 → vmax=1 (two iterations: [0, 1])
            let (_, vmax) = VminVmaxProperty::get(uop);
            // Only treat as empty if vmax < 0 (truly unreachable)
            // NOT vmax == 0 (which is valid single iteration)
            matches!(vmax, ConstValue::Int(v) if *v < 0)
            // Note: UInt cannot be negative, so no UInt case needed for "empty"
        }
        Op::Const(cv) if uop.dtype() == DType::Index => {
            // Dead ranges become Const(0) after rewrite engine processes them.
            // Recognize this pattern in END/REDUCE ranges.
            matches!(cv.0, ConstValue::Int(0) | ConstValue::UInt(0))
        }
        _ => false,
    }
}

/// Get the identity element for a reduce operation.
///
/// The identity element is the value that has no effect in the reduction:
/// - Add: 0 (x + 0 = x)
/// - Mul: 1 (x * 1 = x)
/// - Max: minimum value for dtype (max(x, -∞) = x)
/// - Min: maximum value for dtype (min(x, +∞) = x)
///
/// Follows Tinygrad's approach (dtype.py:134-141): floats use ±inf,
/// integers use type bounds, bools use false/true.
pub fn reduce_identity(op: morok_ir::types::ReduceOp, dtype: morok_dtype::DType) -> Arc<UOp> {
    use morok_ir::types::ConstValue::{Float, Int};
    use morok_ir::types::ReduceOp;

    let val = match op {
        ReduceOp::Add => {
            if dtype.is_float() {
                Float(0.0)
            } else {
                Int(0)
            }
        }
        ReduceOp::Mul => {
            if dtype.is_float() {
                Float(1.0)
            } else {
                Int(1)
            }
        }
        ReduceOp::Max => dtype_min(&dtype),
        ReduceOp::Min => dtype_max(&dtype),
    };
    UOp::const_(dtype, val)
}

/// Return the minimum value for a dtype (Tinygrad: dtypes.min).
fn dtype_min(dtype: &morok_dtype::DType) -> morok_ir::types::ConstValue {
    use morok_dtype::ScalarDType;
    use morok_ir::types::ConstValue::{Bool, Float, Int, UInt};

    if dtype.is_float() {
        return Float(f64::NEG_INFINITY);
    }
    if dtype.is_bool() {
        return Bool(false);
    }
    // Integer types: signed use MIN, unsigned use 0
    match dtype.base() {
        ScalarDType::Int8 => Int(i8::MIN as i64),
        ScalarDType::Int16 => Int(i16::MIN as i64),
        ScalarDType::Int32 => Int(i32::MIN as i64),
        ScalarDType::Int64 | ScalarDType::Index => Int(i64::MIN),
        ScalarDType::UInt8 => UInt(0),
        ScalarDType::UInt16 => UInt(0),
        ScalarDType::UInt32 => UInt(0),
        ScalarDType::UInt64 => UInt(0),
        _ => Int(0),
    }
}

/// Return the maximum value for a dtype (Tinygrad: dtypes.max).
fn dtype_max(dtype: &morok_dtype::DType) -> morok_ir::types::ConstValue {
    use morok_dtype::ScalarDType;
    use morok_ir::types::ConstValue::{Bool, Float, Int, UInt};

    if dtype.is_float() {
        return Float(f64::INFINITY);
    }
    if dtype.is_bool() {
        return Bool(true);
    }
    // Integer types
    match dtype.base() {
        ScalarDType::Int8 => Int(i8::MAX as i64),
        ScalarDType::Int16 => Int(i16::MAX as i64),
        ScalarDType::Int32 => Int(i32::MAX as i64),
        ScalarDType::Int64 | ScalarDType::Index => Int(i64::MAX),
        ScalarDType::UInt8 => UInt(u8::MAX as u64),
        ScalarDType::UInt16 => UInt(u16::MAX as u64),
        ScalarDType::UInt32 => UInt(u32::MAX as u64),
        ScalarDType::UInt64 => UInt(u64::MAX),
        _ => Int(0),
    }
}
