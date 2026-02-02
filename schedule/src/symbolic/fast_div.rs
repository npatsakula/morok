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
/// Algorithm from Hacker's Delight, "magicgu" function.
///
/// # Returns
/// `(magic_multiplier, shift_amount)` or `None` if overflow would occur.
fn magic_unsigned(max_val: i64, divisor: i64) -> Option<(i64, u32)> {
    if divisor <= 0 || max_val <= 0 {
        return None;
    }

    // nc = (max_val + 1) - (max_val + 1) % d - 1
    // This is the largest value < (max_val + 1) that is divisible by d, minus 1
    let nc = max_val - max_val % divisor;

    // Find smallest p such that 2^p > nc * (d - 2^p mod d)
    // Start from p = 32 (for 32-bit arithmetic)
    let mut p = 32u32;

    // Limit search to prevent infinite loop
    const MAX_P: u32 = 64;

    while p < MAX_P {
        // Check if 2^p > nc * (d - (2^p mod d))
        // Using 128-bit to avoid overflow
        let two_p = 1i128 << p;
        let two_p_mod_d = (two_p % (divisor as i128)) as i64;
        let rhs = (nc as i128) * ((divisor - two_p_mod_d) as i128);

        if two_p > rhs {
            break;
        }
        p += 1;
    }

    if p >= MAX_P {
        return None;
    }

    // m = (2^p + d - 1 - (2^p - 1) % d) / d
    // Simplifies to: m = ceil(2^p / d)
    // Using 128-bit arithmetic to avoid overflow
    let two_p = 1i128 << p;
    let m = (two_p + (divisor as i128) - 1 - (two_p - 1) % (divisor as i128)) / (divisor as i128);

    // Check if m fits in i64
    if m > i64::MAX as i128 {
        return None;
    }

    Some((m as i64, p))
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

/// Pattern matcher for fast integer division.
///
/// Transforms `x // d` where d is a non-power-of-2 constant into:
/// - Unsigned case (x >= 0): `(x * M) >> S`
/// - Signed case (x may be negative): `((x * M) >> S) + (x < 0 ? 1 : 0)`
///
/// The magic number M and shift S are computed such that the multiply-shift
/// gives exact division results for all values in the range [vmin, vmax].
///
/// # Performance
///
/// Multiplications are typically 5-20x faster than divisions on modern CPUs
/// and GPUs. This transformation is particularly beneficial for:
/// - Index calculations in tensor operations
/// - Modulo decomposition in range splitting
/// - Array indexing with non-power-of-2 strides
pub fn fast_division_patterns() -> TypedPatternMatcher {
    patterns! {
        // x // d where d is constant non-power-of-2
        Idiv(x, _d @const(d_val)) if is_int_dtype(x) => |x, d_val| {
            // Extract divisor value
            let d_int = match d_val {
                ConstValue::Int(v) => v,
                ConstValue::UInt(v) => match i64::try_from(v) {
                    Ok(v) => v,
                    Err(_) => return None,
                },
                _ => return None,
            };

            // Guard: divisor must be positive
            if d_int <= 0 {
                return None;
            }

            // Guard: skip power-of-2 (already handled by shift optimization)
            if is_power_of_two(d_int) {
                return None;
            }

            // Get value bounds
            let vmin = vmin_as_i64(x)?;
            let vmax = vmax_as_i64(x)?;

            // Compute the maximum absolute value for magic number calculation
            let max_abs = vmax.max(vmin.abs());

            // Compute magic number and shift
            let (m, s) = magic_unsigned(max_abs, d_int)?;

            // Check for potential overflow in the multiplication
            // m * vmin and m * vmax must not overflow
            if m.checked_mul(vmin).is_none() || m.checked_mul(vmax).is_none() {
                return None;
            }

            // Create constants
            let m_const = UOp::index_const(m);
            let s_const = UOp::index_const(s as i64);

            if vmin >= 0 {
                // Unsigned case: (x * m) >> s
                let mul_result = x.mul(&m_const);
                Some(mul_result.shr(&s_const))
            } else {
                // Signed case: ((x * m) >> s) + (x < 0 ? 1 : 0)
                // This handles rounding towards zero for negative dividends
                let mul_result = x.mul(&m_const);
                let base = mul_result.shr(&s_const);

                // Compute adjustment: (x < 0) ? 1 : 0
                let zero = UOp::index_const(0);
                let one = UOp::index_const(1);
                let is_negative = x.try_cmplt(&zero).ok()?;
                let adjustment = UOp::try_where(is_negative, one, zero).ok()?;

                Some(base.add(&adjustment))
            }
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
}
