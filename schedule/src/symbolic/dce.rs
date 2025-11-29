//! Dead Code Elimination (DCE) helpers for symbolic optimization.

use morok_ir::{Op, UOp};
use std::rc::Rc;

/// Check if a range is provably empty (no iterations).
pub fn is_empty_range(uop: &Rc<UOp>) -> bool {
    use morok_ir::types::ConstValue;
    use morok_ir::uop::cached_property::CachedProperty;
    use morok_ir::uop::properties::VminVmaxProperty;

    if let Op::Range { end, .. } = uop.op() {
        let (_, vmax) = VminVmaxProperty::get(end);
        matches!(vmax, ConstValue::Int(v) if *v <= 0) || matches!(vmax, ConstValue::UInt(0))
    } else {
        false
    }
}

/// Get the identity element for a reduce operation.
///
/// The identity element is the value that has no effect in the reduction:
/// - Add: 0 (x + 0 = x)
/// - Mul: 1 (x * 1 = x)
/// - Max: minimum value for dtype (max(x, MIN) = x)
pub fn reduce_identity(op: morok_ir::types::ReduceOp, dtype: morok_dtype::DType) -> Rc<UOp> {
    use morok_dtype::DType;
    use morok_ir::types::ConstValue::*;
    use morok_ir::types::ReduceOp;

    match op {
        ReduceOp::Add => UOp::const_(dtype, Int(0)),
        ReduceOp::Mul => UOp::const_(dtype, Int(1)),
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
