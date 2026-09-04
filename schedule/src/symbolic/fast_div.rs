//! Integer division strength reduction.
//!
//! This is a direct port of pinned Tinygrad's `codegen/decomp/op.py` integer
//! division helpers. Keep the operation order aligned with that source.

use std::collections::HashSet;
use std::sync::Arc;

use svod_dtype::ScalarDType;
use svod_ir::types::ConstValue;
use svod_ir::uop::range_eval::dtype_bounds;
use svod_ir::{DType, UOp};

use crate::{TypedPatternMatcher, patterns};

fn magic_unsigned(vmax: i64, divisor: i64) -> Option<(i64, u32)> {
    if divisor <= 0 || vmax < 0 {
        return None;
    }
    let divisor = divisor as i128;
    let nc = (vmax as i128 + 1) / divisor * divisor - 1;
    let nbits = 64 - vmax.leading_zeros();
    for shift in 0..=2 * nbits {
        let power = 1i128 << shift;
        if power > nc * (divisor - 1 - (power - 1) % divisor) {
            let multiplier = (power + divisor - 1 - (power - 1) % divisor) / divisor;
            return Some((i64::try_from(multiplier).ok()?, shift));
        }
    }
    None
}

fn value_as_i64(value: &ConstValue) -> Option<i64> {
    match value {
        ConstValue::Int(value) => Some(*value),
        ConstValue::UInt(value) => i64::try_from(*value).ok(),
        _ => None,
    }
}

fn dtype_bounds_i128(dtype: &DType) -> Option<(i128, i128)> {
    let convert = |value| match value {
        ConstValue::Int(value) => Some(value as i128),
        ConstValue::UInt(value) => Some(value as i128),
        _ => None,
    };
    let (min, max) = dtype_bounds(dtype);
    Some((convert(min)?, convert(max)?))
}

fn multiplication_fits(multiplier: i64, vmin: i64, vmax: i64, dtype: &DType) -> bool {
    let Some((dtype_min, dtype_max)) = dtype_bounds_i128(dtype) else { return false };
    multiplier as i128 * vmin as i128 >= dtype_min && multiplier as i128 * vmax as i128 <= dtype_max
}

fn next_integer_dtype(dtype: &DType) -> Option<DType> {
    Some(match dtype.base() {
        ScalarDType::Int8 => DType::Int16,
        ScalarDType::Int16 => DType::Int32,
        ScalarDType::Int32 => DType::Int64,
        ScalarDType::Int64 => DType::UInt64,
        ScalarDType::UInt8 => DType::UInt16,
        ScalarDType::UInt16 => DType::UInt32,
        ScalarDType::UInt32 => DType::UInt64,
        _ => return None,
    })
}

fn fast_idiv(x: &Arc<UOp>, divisor: i64, dont_cast: bool, supported_dtypes: &HashSet<ScalarDType>) -> Option<Arc<UOp>> {
    let is_unsigned = value_as_i64(x.vmin())? >= 0 || x.dtype().is_unsigned();
    assert!(divisor > 0, "sign should have been taken out of divisor");
    let (dtype_min, dtype_max) = dtype_bounds_i128(&x.dtype())?;
    let vmin = (value_as_i64(x.vmin())? as i128).max(dtype_min) as i64;
    let vmax = (value_as_i64(x.vmax())? as i128).min(dtype_max) as i64;
    if vmin > -divisor && vmax < divisor {
        return Some(x.const_like(0));
    }
    let (multiplier, shift) = magic_unsigned(vmax.max(vmin.saturating_abs()), divisor)?;
    let multiply_shift = |value: &Arc<UOp>| {
        value.try_mul(&value.const_like(multiplier)).ok()?.try_shr_op(&value.const_like(shift as i64)).ok()
    };
    let signed_adjustment =
        || UOp::try_where(x.try_cmplt(&x.const_like(0)).ok()?, x.const_like(1), x.const_like(0)).ok();
    if multiplication_fits(multiplier, vmin, vmax, &x.dtype()) {
        let result = multiply_shift(x)?;
        return if is_unsigned { Some(result) } else { result.try_add(&signed_adjustment()?).ok() };
    }
    let factor = divisor.isolate_lowest_one();
    if factor > 1 {
        let reduced = x.cdiv(&x.const_like(factor));
        if let Some(result) = fast_idiv(&reduced, divisor / factor, true, supported_dtypes) {
            return Some(result);
        }
    }
    if dont_cast {
        return None;
    }
    let next_dtype = next_integer_dtype(&x.dtype())?;
    if supported_dtypes.contains(&next_dtype.base()) && multiplication_fits(multiplier, vmin, vmax, &next_dtype) {
        let result = multiply_shift(&x.cast(next_dtype))?.cast(x.dtype());
        return if is_unsigned { Some(result) } else { result.try_add(&signed_adjustment()?).ok() };
    }
    None
}

pub fn fast_division_patterns(supported_dtypes: HashSet<ScalarDType>) -> TypedPatternMatcher {
    patterns! {
        CDiv(x, _d @const(d_val))
            if x.dtype().is_int() && (x.dtype().is_unsigned() || value_as_i64(x.vmin()).is_some_and(|v| v >= 0))
            => {
                let divisor = value_as_i64(&d_val)?;
                (divisor > 0 && !(divisor as u64).is_power_of_two()).then_some(())?;
                fast_idiv(x, divisor, false, &supported_dtypes)
            },
    }
}

#[cfg(test)]
#[path = "../test/unit/symbolic/fast_div_internal.rs"]
mod tests;
