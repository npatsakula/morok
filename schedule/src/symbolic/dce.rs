//! Dead Code Elimination (DCE) helpers for symbolic optimization.

use morok_ir::types::ConstValue;
use morok_ir::{Op, UOp};
use std::cmp::Ordering;
use std::rc::Rc;

/// Compare two constant values for ordering.
pub fn compare_const_values(a: &ConstValue, b: &ConstValue) -> Option<Ordering> {
    use ConstValue::*;
    match (a, b) {
        (Int(x), Int(y)) => Some(x.cmp(y)),
        (UInt(x), UInt(y)) => Some(x.cmp(y)),
        (Bool(x), Bool(y)) => Some(x.cmp(y)),
        (Float(x), Float(y)) if !x.is_nan() && !y.is_nan() => x.partial_cmp(y),
        _ => None,
    }
}

/// Check if an operation is pure (has no side effects).
pub fn is_pure(uop: &Rc<UOp>) -> bool {
    !matches!(
        uop.op(),
        Op::Store { .. }
            | Op::StoreGated { .. }
            | Op::Barrier { .. }
            | Op::Custom { .. }
            | Op::CustomI { .. }
            | Op::AllReduce { .. }
            | Op::Assign { .. }
            | Op::Sink { .. }
            | Op::Kernel { .. }
    )
}

/// Check if a range is provably empty (no iterations).
pub fn is_empty_range(uop: &Rc<UOp>) -> bool {
    use morok_ir::uop::cached_property::CachedProperty;
    use morok_ir::uop::properties::VminVmaxProperty;

    if let Op::Range { end, .. } = uop.op() {
        let (_, vmax) = VminVmaxProperty::get(end);
        matches!(vmax, ConstValue::Int(v) if *v <= 0) || matches!(vmax, ConstValue::UInt(0))
    } else {
        false
    }
}

/// Check if a value equals a specific constant.
pub fn is_const_value(v: &ConstValue, target: i64) -> bool {
    use ConstValue::*;
    match (v, target) {
        (Int(x), t) => *x == t,
        (UInt(x), t) if t >= 0 => *x == t as u64,
        (Float(x), t) => *x == t as f64,
        (Bool(x), 0) => !x,
        (Bool(x), 1) => *x,
        _ => false,
    }
}

/// Check if a value is provably zero.
pub fn is_zero(v: &ConstValue) -> bool {
    is_const_value(v, 0)
}

/// Check if a value is provably one.
pub fn is_one(v: &ConstValue) -> bool {
    is_const_value(v, 1)
}

/// Helper for range-based comparison analysis.
///
/// Returns (lt_result, eq_result, ne_result) for compatibility with existing code.
pub fn analyze_comparison(x: &Rc<UOp>, y: &Rc<UOp>) -> (Option<bool>, Option<bool>, Option<bool>) {
    use morok_ir::types::BinaryOp;
    use morok_ir::uop::comparison_analysis::ComparisonAnalyzer;

    let lt_result = ComparisonAnalyzer::analyze(BinaryOp::Lt, x, y);
    let eq_result = ComparisonAnalyzer::analyze(BinaryOp::Eq, x, y);
    let ne_result = ComparisonAnalyzer::analyze(BinaryOp::Ne, x, y);

    (lt_result, eq_result, ne_result)
}

/// Get the identity element for a reduce operation.
///
/// The identity element is the value that has no effect in the reduction:
/// - Add: 0 (x + 0 = x)
/// - Mul: 1 (x * 1 = x)
/// - Max: minimum value for dtype (max(x, MIN) = x)
pub fn reduce_identity(op: morok_ir::types::ReduceOp, dtype: morok_dtype::DType) -> Rc<UOp> {
    use ConstValue::*;
    use morok_dtype::DType;
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_const_values() {
        assert_eq!(compare_const_values(&ConstValue::Int(5), &ConstValue::Int(10)), Some(Ordering::Less));
        assert_eq!(compare_const_values(&ConstValue::Float(f64::NAN), &ConstValue::Float(1.0)), None);
    }

    #[test]
    fn test_is_pure() {
        use morok_dtype::DType;
        use morok_ir::types::BinaryOp;

        let a = UOp::const_(DType::Int32, ConstValue::Int(1));
        let b = UOp::const_(DType::Int32, ConstValue::Int(2));
        assert!(is_pure(&UOp::new(Op::Binary(BinaryOp::Add, a, b), DType::Int32)));
    }

    #[test]
    fn test_is_const_value() {
        assert!(is_zero(&ConstValue::Int(0)));
        assert!(is_one(&ConstValue::Int(1)));
        assert!(!is_zero(&ConstValue::Int(1)));
        assert!(!is_one(&ConstValue::Int(0)));
    }
}
