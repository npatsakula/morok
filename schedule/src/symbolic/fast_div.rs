//! Fast integer division using magic number multiplication.
//!
//! Implements the "magic number" method from Hacker's Delight for replacing
//! integer division by constant with multiply-and-shift operations.
//!
//! For division by constant d:
//!   x / d ≈ (x * M) >> S
//!
//! where M (magic multiplier) and S (shift amount) are computed such that
//! the multiply-shift gives exact results for all values in the expected range.

use std::sync::Arc;

use morok_ir::UOp;
use morok_ir::types::ConstValue;

use crate::TypedPatternMatcher;
use crate::patterns;

/// Compute magic number M and shift S for unsigned division.
///
/// Given max_val (maximum value the dividend can take) and divisor d,
/// computes M and S such that: x/d = (x*M) >> S for all 0 <= x <= max_val.
///
/// Matches Tinygrad's `magicgu` (decompositions.py:272-280). Finds the smallest
/// shift S, producing the smallest magic number — critical for fitting the
/// intermediate multiply in narrow types (e.g. Int32).
///
/// # Returns
/// `(magic_multiplier, shift_amount)` or `None` if no valid pair found.
fn magic_unsigned(max_val: i64, divisor: i64) -> Option<(i64, u32)> {
    if divisor <= 0 || max_val <= 0 {
        return None;
    }

    let d = divisor as i128;
    let nc = ((max_val as i128 + 1) / d * d - 1).max(0);
    let nbits = 64 - max_val.leading_zeros(); // = bit_length

    for s in 0..=(2 * nbits) {
        let two_s: i128 = 1 << s;
        if two_s > nc * (d - 1 - (two_s - 1) % d) {
            let m = (two_s + d - 1 - (two_s - 1) % d) / d;
            if m > i64::MAX as i128 {
                return None;
            }
            return Some((m as i64, s));
        }
    }
    None
}

/// Check if a value is a power of two.
#[inline]
fn is_power_of_two(n: i64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Check if dtype is an integer type.
fn is_int_dtype(uop: &Arc<UOp>) -> bool {
    uop.dtype().is_int()
}

/// Get vmin value as i64 from a UOp.
fn vmin_as_i64(uop: &Arc<UOp>) -> Option<i64> {
    match uop.vmin() {
        ConstValue::Int(v) => Some(*v),
        ConstValue::UInt(v) => i64::try_from(*v).ok(),
        _ => None,
    }
}

/// Get vmax value as i64 from a UOp.
fn vmax_as_i64(uop: &Arc<UOp>) -> Option<i64> {
    match uop.vmax() {
        ConstValue::Int(v) => Some(*v),
        ConstValue::UInt(v) => i64::try_from(*v).ok(),
        _ => None,
    }
}

/// Emit `(x * m) >> s`, with signed adjustment if needed.
/// Matches Tinygrad decompositions.py:291.
fn emit_fast_div(x: &Arc<UOp>, m: i64, s: u32, is_unsigned: bool, dtype: &morok_ir::DType) -> Option<Arc<UOp>> {
    let m_const = UOp::const_(dtype.clone(), ConstValue::Int(m));
    let s_const = UOp::const_(dtype.clone(), ConstValue::Int(s as i64));
    let mul_result = x.mul(&m_const);
    if is_unsigned {
        Some(mul_result.shr(&s_const))
    } else {
        let base = mul_result.shr(&s_const);
        let zero = UOp::const_(dtype.clone(), ConstValue::Int(0));
        let one = UOp::const_(dtype.clone(), ConstValue::Int(1));
        let is_negative = x.try_cmplt(&zero).ok()?;
        let adjustment = UOp::try_where(is_negative, one, zero).ok()?;
        Some(base.add(&adjustment))
    }
}

/// Check if m*vmin and m*vmax fit within a dtype's representable range.
fn fits_in_dtype(m: i64, vmin: i64, vmax: i64, dtype: &morok_ir::DType) -> bool {
    use morok_ir::uop::range_eval::dtype_bounds;
    let (dt_min, dt_max) = dtype_bounds(dtype);
    let dt_min_i = match dt_min {
        ConstValue::Int(v) => v,
        _ => return false,
    };
    let dt_max_i = match dt_max {
        ConstValue::Int(v) => v,
        _ => return false,
    };
    match (m.checked_mul(vmin), m.checked_mul(vmax)) {
        (Some(lo), Some(hi)) => lo >= dt_min_i && hi <= dt_max_i,
        _ => false,
    }
}

/// Pattern matcher for fast integer division.
///
/// Transforms `x // d` where d is a non-power-of-2 constant into:
/// `(x * M) >> S` (unsigned) or `((x * M) >> S) + (x < 0)` (signed).
///
/// Matches Tinygrad's `fast_idiv` (decompositions.py:282-300):
/// 1. Try same-dtype multiply if m*x fits
/// 2. Factor out powers of two in d to reduce magnitude
/// 3. Widen to Int64 if needed (for Int32 inputs)
pub fn fast_division_patterns() -> TypedPatternMatcher {
    patterns! {
        Idiv(x, _d @const(d_val)) if is_int_dtype(x) => |x, d_val| {
            let d_int = match d_val {
                ConstValue::Int(v) => v,
                ConstValue::UInt(v) => i64::try_from(v).ok()?,
                _ => return None,
            };
            if d_int <= 0 || is_power_of_two(d_int) {
                return None;
            }

            let dtype = x.dtype();
            let vmin = vmin_as_i64(x)?;
            let vmax = vmax_as_i64(x)?;
            let is_unsigned = vmin >= 0;
            let max_abs = vmax.max(vmin.saturating_abs());
            let (m, s) = magic_unsigned(max_abs, d_int)?;

            // 1. Try same-dtype if m*x fits (decompositions.py:290-291)
            if fits_in_dtype(m, vmin, vmax, &dtype) {
                return emit_fast_div(x, m, s, is_unsigned, &dtype);
            }

            // 2. Factor out powers of two in d (decompositions.py:293-294)
            let pow2_factor = d_int & (-d_int);
            if pow2_factor > 1 {
                let reduced_d = d_int / pow2_factor;
                if reduced_d > 1 && !is_power_of_two(reduced_d) {
                    let shift_bits = (pow2_factor as u64).trailing_zeros() as i64;
                    let shift_const = UOp::const_(dtype.clone(), ConstValue::Int(shift_bits));
                    let shifted = x.shr(&shift_const);
                    let rv_min = vmin_as_i64(&shifted).unwrap_or(vmin >> shift_bits);
                    let rv_max = vmax_as_i64(&shifted).unwrap_or(vmax >> shift_bits);
                    let r_max_abs = rv_max.max(rv_min.saturating_abs());
                    if let Some((rm, rs)) = magic_unsigned(r_max_abs, reduced_d)
                        && fits_in_dtype(rm, rv_min, rv_max, &dtype) {
                            return emit_fast_div(&shifted, rm, rs, rv_min >= 0, &dtype);
                        }
                } else if reduced_d == 1 {
                    let shift_bits = (pow2_factor as u64).trailing_zeros() as i64;
                    let shift_const = UOp::const_(dtype.clone(), ConstValue::Int(shift_bits));
                    return Some(x.shr(&shift_const));
                }
            }

            // 3. Widen to Int64 if current dtype is narrower (decompositions.py:297-299)
            if dtype.bytes() < 8 {
                let wide = morok_ir::DType::Int64;
                if fits_in_dtype(m, vmin, vmax, &wide) {
                    let wide_x = x.cast(wide.clone());
                    let result = emit_fast_div(&wide_x, m, s, is_unsigned, &wide)?;
                    return Some(result.cast(dtype));
                }
            }

            None
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_unsigned_div_3() {
        // x / 3 for x in 0..=100
        let result = magic_unsigned(100, 3);
        assert!(result.is_some());
        let (m, s) = result.unwrap();

        // Verify for some values
        for x in 0..=100 {
            let expected = x / 3;
            let actual = ((x as i128 * m as i128) >> s) as i64;
            assert_eq!(expected, actual, "Failed for x = {}", x);
        }
    }

    #[test]
    fn test_magic_unsigned_div_7() {
        // x / 7 for x in 0..=1000
        let result = magic_unsigned(1000, 7);
        assert!(result.is_some());
        let (m, s) = result.unwrap();

        for x in 0..=1000 {
            let expected = x / 7;
            let actual = ((x as i128 * m as i128) >> s) as i64;
            assert_eq!(expected, actual, "Failed for x = {}", x);
        }
    }

    #[test]
    fn test_magic_unsigned_div_10() {
        // x / 10 for x in 0..=10000
        let result = magic_unsigned(10000, 10);
        assert!(result.is_some());
        let (m, s) = result.unwrap();

        for x in (0..=10000).step_by(100) {
            let expected = x / 10;
            let actual = ((x as i128 * m as i128) >> s) as i64;
            assert_eq!(expected, actual, "Failed for x = {}", x);
        }
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(8));
        assert!(is_power_of_two(1024));

        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(-1));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(6));
        assert!(!is_power_of_two(7));
    }

    #[test]
    fn test_magic_unsigned_invalid() {
        // Zero divisor
        assert!(magic_unsigned(100, 0).is_none());

        // Negative divisor
        assert!(magic_unsigned(100, -5).is_none());
    }

    #[test]
    fn test_magic_unsigned_div_6_factorization() {
        // x / 6 for x in 0..=1000
        // Tests power-of-two factorization: 6 = 2 * 3
        // Division by 6 should become: (x >> 1) / 3
        let result = magic_unsigned(500, 3); // After shift, max is 500
        assert!(result.is_some());
        let (m, s) = result.unwrap();

        for x in 0..=1000 {
            let expected = x / 6;
            // Simulate factorization: (x >> 1) then magic divide by 3
            let shifted = x >> 1;
            let actual = ((shifted as i128 * m as i128) >> s) as i64;
            assert_eq!(expected, actual, "Failed for x = {}", x);
        }
    }

    #[test]
    fn test_magic_unsigned_div_12_factorization() {
        // x / 12 for x in 0..=1200
        // Tests power-of-two factorization: 12 = 4 * 3
        // Division by 12 should become: (x >> 2) / 3
        let result = magic_unsigned(300, 3); // After shift by 2, max is 300
        assert!(result.is_some());
        let (m, s) = result.unwrap();

        for x in 0..=1200 {
            let expected = x / 12;
            // Simulate factorization: (x >> 2) then magic divide by 3
            let shifted = x >> 2;
            let actual = ((shifted as i128 * m as i128) >> s) as i64;
            assert_eq!(expected, actual, "Failed for x = {}", x);
        }
    }
}
