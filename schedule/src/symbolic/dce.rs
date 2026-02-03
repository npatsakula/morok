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
/// - Max: minimum value for dtype (max(x, MIN) = x)
pub fn reduce_identity(op: morok_ir::types::ReduceOp, dtype: morok_dtype::DType) -> Arc<UOp> {
    use morok_dtype::DType;
    use morok_ir::types::ConstValue::*;
    use morok_ir::types::ReduceOp;

    match op {
        ReduceOp::Add => {
            // 0 is identity for addition (x + 0 = x)
            let zero = if dtype.is_float() { Float(0.0) } else { Int(0) };
            UOp::const_(dtype, zero)
        }
        ReduceOp::Mul => {
            // 1 is identity for multiplication (x * 1 = x)
            let one = if dtype.is_float() { Float(1.0) } else { Int(1) };
            UOp::const_(dtype, one)
        }
        ReduceOp::Max => {
            // Return dtype minimum value
            let min_val = if dtype == DType::Int8 {
                Int(i8::MIN as i64)
            } else if dtype == DType::Int16 {
                Int(i16::MIN as i64)
            } else if dtype == DType::Int32 {
                Int(i32::MIN as i64)
            } else if dtype == DType::Int64 {
                Int(i64::MIN)
            } else if dtype == DType::UInt8 {
                UInt(u8::MIN as u64)
            } else if dtype == DType::UInt16 {
                UInt(u16::MIN as u64)
            } else if dtype == DType::UInt32 {
                UInt(u32::MIN as u64)
            } else if dtype == DType::UInt64 {
                UInt(u64::MIN)
            } else if dtype == DType::Float16 {
                Float(-65504.0)
            } else if dtype == DType::BFloat16 {
                Float(-3.38953e38)
            } else if dtype == DType::Float32 {
                Float(f32::MIN as f64)
            } else if dtype == DType::Float64 {
                Float(f64::MIN)
            } else {
                Int(0) // Fallback for unsupported types
            };
            UOp::const_(dtype, min_val)
        }
        ReduceOp::Min => {
            // Return dtype maximum value
            let max_val = if dtype == DType::Int8 {
                Int(i8::MAX as i64)
            } else if dtype == DType::Int16 {
                Int(i16::MAX as i64)
            } else if dtype == DType::Int32 {
                Int(i32::MAX as i64)
            } else if dtype == DType::Int64 {
                Int(i64::MAX)
            } else if dtype == DType::UInt8 {
                UInt(u8::MAX as u64)
            } else if dtype == DType::UInt16 {
                UInt(u16::MAX as u64)
            } else if dtype == DType::UInt32 {
                UInt(u32::MAX as u64)
            } else if dtype == DType::UInt64 {
                UInt(u64::MAX)
            } else if dtype == DType::Float16 {
                Float(65504.0)
            } else if dtype == DType::BFloat16 {
                Float(3.38953e38)
            } else if dtype == DType::Float32 {
                Float(f32::MAX as f64)
            } else if dtype == DType::Float64 {
                Float(f64::MAX)
            } else {
                Int(0) // Fallback for unsupported types
            };
            UOp::const_(dtype, max_val)
        }
    }
}
