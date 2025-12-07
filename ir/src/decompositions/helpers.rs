//! Helper functions for building decomposition UOp trees.
//!
//! These helpers mirror tinygrad's decomposition utilities:
//! - `poly_n`: Horner's method for polynomial evaluation
//! - `rintk`: Round to nearest integer
//! - `pow2if`: 2^q for integer q
//! - `ldexp2k`: d * 2^e
//! - `ilogb2k`: Integer part of log2
//! - `frexp`: Extract mantissa and exponent
//! - Bit manipulation utilities

use std::sync::Arc;

use morok_dtype::{DType, ScalarDType};

use crate::types::ConstValue;
use crate::uop::UOp;

// ============================================================================
// Dtype helpers
// ============================================================================

/// Get the scalar element type, extracting from Ptr if needed.
///
/// Decomposition functions operate on scalar types, but during rangeify
/// the IR may contain Ptr-typed operands (from INDEX operations).
/// This helper extracts the underlying scalar type so decompositions
/// can create correctly-typed constants and operations.
///
/// # Examples
///
/// - `Ptr<Float32>` → `Float32`
/// - `Float32` → `Float32`
/// - `Ptr<Ptr<Int32>>` → `Int32` (recursive)
pub fn scalar_dtype(dtype: &DType) -> DType {
    match dtype {
        DType::Ptr { base, .. } => scalar_dtype(base),
        other => other.clone(),
    }
}

/// Ensure operand has scalar type by casting Ptr to its base scalar.
///
/// During rangeify, INDEX operations produce Ptr-typed values that will
/// be auto-loaded by codegen. However, decomposition runs before codegen
/// and needs scalar-typed operands for arithmetic/comparison operations.
///
/// This helper:
/// - If operand has Ptr dtype: casts to base scalar type
/// - If operand already has scalar dtype: returns as-is
///
/// The cast is a no-op at runtime (codegen will auto-load the pointer),
/// but ensures the IR type system is satisfied.
pub fn ensure_scalar(d: &Arc<UOp>) -> Arc<UOp> {
    let dtype = d.dtype();
    let scalar = scalar_dtype(&dtype);
    if dtype != scalar { UOp::cast(d.clone(), scalar) } else { d.clone() }
}

// ============================================================================
// Dtype-dependent constants
// ============================================================================

/// Number of mantissa bits for a float dtype.
///
/// Extracts scalar type from Ptr if needed.
pub fn mantissa_bits(dtype: &DType) -> i64 {
    let scalar = scalar_dtype(dtype);
    match scalar.scalar() {
        Some(ScalarDType::Float64) => 52,
        Some(ScalarDType::Float32) => 23,
        Some(ScalarDType::Float16) => 10,
        _ => panic!("mantissa_bits: unsupported dtype {:?}", dtype),
    }
}

/// Exponent bias for a float dtype.
///
/// Extracts scalar type from Ptr if needed.
pub fn exponent_bias(dtype: &DType) -> i64 {
    let scalar = scalar_dtype(dtype);
    match scalar.scalar() {
        Some(ScalarDType::Float64) => 1023,
        Some(ScalarDType::Float32) => 127,
        Some(ScalarDType::Float16) => 15,
        _ => panic!("exponent_bias: unsupported dtype {:?}", dtype),
    }
}

/// Exponent mask for a float dtype.
///
/// Extracts scalar type from Ptr if needed.
pub fn exponent_mask(dtype: &DType) -> i64 {
    let scalar = scalar_dtype(dtype);
    match scalar.scalar() {
        Some(ScalarDType::Float64) => 2047,
        Some(ScalarDType::Float32) => 255,
        Some(ScalarDType::Float16) => 31,
        _ => panic!("exponent_mask: unsupported dtype {:?}", dtype),
    }
}

/// Get the integer dtype for a float dtype (for bit manipulation).
///
/// Extracts scalar type from Ptr if needed.
pub fn float_to_int_dtype(dtype: &DType) -> DType {
    let scalar = scalar_dtype(dtype);
    match scalar.scalar() {
        Some(ScalarDType::Float64) => DType::Int64,
        Some(ScalarDType::Float32) => DType::Int32,
        Some(ScalarDType::Float16) => DType::Int16,
        _ => panic!("float_to_int_dtype: unsupported dtype {:?}", dtype),
    }
}

/// Get the integer dtype for a float dtype (for bit manipulation).
///
/// Extracts scalar type from Ptr if needed.
pub fn int_to_float_dtype(dtype: &DType) -> DType {
    let scalar = scalar_dtype(dtype);
    match scalar.scalar() {
        Some(ScalarDType::Int64) => DType::Float64,
        Some(ScalarDType::Int32) => DType::Float32,
        Some(ScalarDType::Int16) => DType::Float16,
        _ => panic!("int_to_float_dtype: unsupported dtype {:?}", dtype),
    }
}

// ============================================================================
// Bit manipulation helpers
// ============================================================================

/// Shift right: x >> y (arithmetic)
pub fn shr(x: &Arc<UOp>, y: i64) -> Arc<UOp> {
    if y == 0 {
        return x.clone();
    }
    let shift = UOp::const_(x.dtype(), ConstValue::Int(y));
    x.try_shr_op(&shift).expect("shr: shift failed")
}

/// Shift left: x << y
pub fn shl(x: &Arc<UOp>, y: i64) -> Arc<UOp> {
    if y == 0 {
        return x.clone();
    }
    let shift = UOp::const_(x.dtype(), ConstValue::Int(y));
    x.try_shl_op(&shift).expect("shl: shift failed")
}

/// Bitwise AND with constant.
pub fn and_const(x: &Arc<UOp>, mask: i64) -> Arc<UOp> {
    let mask_uop = UOp::const_(x.dtype(), ConstValue::Int(mask));
    x.try_and_op(&mask_uop).expect("and_const: failed")
}

// ============================================================================
// Core decomposition helpers (following tinygrad)
// ============================================================================

/// Horner's method for polynomial evaluation (tinygrad-style).
///
/// Coefficients are in DESCENDING power order: [c0, c1, ..., c_{n-1}] evaluates as:
/// c0*x^{n-1} + c1*x^{n-2} + ... + c_{n-2}*x + c_{n-1}
///
/// This matches tinygrad's `polyN(x, p) = reduce(lambda acc,c: acc*x+c, p, 0.0)`:
/// - Start with 0.0
/// - For each c in coeffs: acc = acc * x + c
/// - This computes: c0*x^{n-1} + c1*x^{n-2} + ... + c_{n-1}
///
/// Example: polyN(x, [a, b, c, d]) = a*x^3 + b*x^2 + c*x + d
/// At x=0: result = d (the last coefficient)
pub fn poly_n(x: &Arc<UOp>, coeffs: &[f64]) -> Arc<UOp> {
    assert!(!coeffs.is_empty(), "poly_n: need at least one coefficient");
    let dtype = x.dtype();

    // Start with 0.0
    let mut result = float_const(&dtype, 0.0);

    // For each coefficient: result = result * x + coeff
    for &coeff in coeffs {
        let c = float_const(&dtype, coeff);
        let mul = result.try_mul(x).expect("poly_n: mul failed");
        result = mul.try_add(&c).expect("poly_n: add failed");
    }

    result
}

/// Create a float constant with the appropriate dtype.
///
/// Extracts scalar type from Ptr if needed, ensuring decompositions
/// work correctly when operands have Ptr dtypes (from INDEX ops).
pub fn float_const(dtype: &DType, value: f64) -> Arc<UOp> {
    let scalar = scalar_dtype(dtype);
    UOp::const_(scalar, ConstValue::Float(value))
}

/// Create an integer constant with the appropriate dtype.
///
/// Extracts scalar type from Ptr if needed, ensuring decompositions
/// work correctly when operands have Ptr dtypes (from INDEX ops).
pub fn int_const(dtype: &DType, value: i64) -> Arc<UOp> {
    let scalar = scalar_dtype(dtype);
    UOp::const_(scalar, ConstValue::Int(value))
}

/// Create a boolean constant.
pub fn bool_const(value: bool) -> Arc<UOp> {
    UOp::const_(DType::Bool, ConstValue::Bool(value))
}

/// Round d:float to int away from 0: rintk(d)
///
/// Equivalent to: cast(d + (d < 0 ? -0.5 : 0.5), int_dtype)
pub fn rintk(d: &Arc<UOp>) -> Arc<UOp> {
    let dtype = d.dtype();
    let int_dtype = float_to_int_dtype(&dtype);

    let zero = float_const(&dtype, 0.0);
    let half = float_const(&dtype, 0.5);
    let neg_half = float_const(&dtype, -0.5);

    // d < 0.0
    let is_neg = d.try_cmplt(&zero).expect("rintk: cmplt failed");

    // (d < 0) ? -0.5 : 0.5
    let adjustment = UOp::try_where(is_neg, neg_half, half).expect("rintk: where failed");

    // d + adjustment
    let adjusted = d.try_add(&adjustment).expect("rintk: add failed");

    // cast to int
    UOp::cast(adjusted, int_dtype)
}

/// pow2if: cast(2^q, float_dtype) where q is an integer in [-126, 127] range.
///
/// Implements by constructing the float bit pattern directly:
/// 2^q has mantissa=0 and exponent=(q + bias), so bits = (q + bias) << mantissa_bits
pub fn pow2if(q: &Arc<UOp>, float_dtype: &DType) -> Arc<UOp> {
    let int_dtype = float_to_int_dtype(float_dtype);
    let bias = exponent_bias(float_dtype);
    let mantissa = mantissa_bits(float_dtype);

    // q + bias
    let bias_const = int_const(&int_dtype, bias);
    let q_int = if q.dtype() == int_dtype { q.clone() } else { UOp::cast(q.clone(), int_dtype.clone()) };
    let biased = q_int.try_add(&bias_const).expect("pow2if: add failed");

    // (q + bias) << mantissa_bits
    let shifted = shl(&biased, mantissa);

    // bitcast to float
    UOp::bitcast(shifted, float_dtype.clone())
}

/// ldexp2k: d * 2^e where d > 0 and d is not denormal.
///
/// Faster than ldexp3k but assumes d is normalized positive.
/// Implements as: (d * pow2if(e >> 1)) * pow2if(e - (e >> 1))
pub fn ldexp2k(d: &Arc<UOp>, e: &Arc<UOp>) -> Arc<UOp> {
    let float_dtype = d.dtype();

    // e >> 1 (half of exponent)
    let e_half = shr(e, 1);

    // e - (e >> 1) (other half)
    let e_other = e.try_sub(&e_half).expect("ldexp2k: sub failed");

    // pow2if(e >> 1)
    let pow_half = pow2if(&e_half, &float_dtype);

    // pow2if(e - (e >> 1))
    let pow_other = pow2if(&e_other, &float_dtype);

    // d * pow2if(e >> 1)
    let step1 = d.try_mul(&pow_half).expect("ldexp2k: mul1 failed");

    // result * pow2if(e - (e >> 1))
    step1.try_mul(&pow_other).expect("ldexp2k: mul2 failed")
}

/// ldexp3k: d * 2^e where e is a float that was cast from int in [-127, 127].
///
/// More general than ldexp2k but slower.
/// Implements by adding e (as int) to the exponent bits of d directly.
pub fn ldexp3k(d: &Arc<UOp>, e: &Arc<UOp>) -> Arc<UOp> {
    let float_dtype = d.dtype();
    let int_dtype = float_to_int_dtype(&float_dtype);
    let mantissa = mantissa_bits(&float_dtype);

    // Bitcast d to int
    let d_bits = UOp::bitcast(d.clone(), int_dtype.clone());

    // Cast e to int and shift
    let e_int = UOp::cast(e.clone(), int_dtype.clone());
    let e_shifted = shl(&e_int, mantissa);

    // Add exponent bits
    let result_bits = d_bits.try_add(&e_shifted).expect("ldexp3k: add failed");

    // Bitcast back to float - no need for cast since bitcast already produces correct dtype
    UOp::bitcast(result_bits, float_dtype)
}

/// ilogb2k: Integer part of log2(d), where d is normalized fp in [0, +inf).
///
/// Extracts the exponent from the float bit representation:
/// exponent_bits = (bits >> mantissa_bits) & exponent_mask
/// ilogb2k = exponent_bits - bias
pub fn ilogb2k(d: &Arc<UOp>) -> Arc<UOp> {
    let float_dtype = d.dtype();
    let int_dtype = float_to_int_dtype(&float_dtype);
    let mantissa = mantissa_bits(&float_dtype);
    let mask = exponent_mask(&float_dtype);
    let bias = exponent_bias(&float_dtype);

    // Bitcast to int
    let d_bits = UOp::bitcast(d.clone(), int_dtype.clone());

    // (bits >> mantissa_bits) & exponent_mask
    let shifted = shr(&d_bits, mantissa);
    let masked = and_const(&shifted, mask);

    // - bias
    let bias_const = int_const(&int_dtype, bias);
    masked.try_sub(&bias_const).expect("ilogb2k: sub failed")
}

/// Conditional value replacement for special cases (inf, -inf, nan).
///
/// lazy_map_numbers(x, inf, _inf, nan, ratio):
///   - if x == inf -> return inf
///   - if x == -inf -> return _inf
///   - if x != x (nan) -> return nan
///   - otherwise -> return ratio
pub fn lazy_map_numbers(
    x: &Arc<UOp>,
    inf_val: &Arc<UOp>,
    neg_inf_val: &Arc<UOp>,
    nan_val: &Arc<UOp>,
    ratio: &Arc<UOp>,
) -> Arc<UOp> {
    let dtype = x.dtype();
    let pos_inf = float_const(&dtype, f64::INFINITY);
    let neg_inf = float_const(&dtype, f64::NEG_INFINITY);

    // x != inf
    let not_pos_inf = x.try_cmpne(&pos_inf).expect("lazy_map: cmpne pos_inf");

    // x != -inf
    let not_neg_inf = x.try_cmpne(&neg_inf).expect("lazy_map: cmpne neg_inf");

    // x != x (NaN check)
    let is_nan = x.try_cmpne(x).expect("lazy_map: cmpne nan");

    // Build from inside out:
    // x.ne(-inf).where(ratio, neg_inf_val)
    let inner = UOp::try_where(not_neg_inf, ratio.clone(), neg_inf_val.clone()).expect("lazy_map: where1");

    // x.ne(x).where(nan_val, inner) - NaN check
    let with_nan = UOp::try_where(is_nan, nan_val.clone(), inner).expect("lazy_map: where2");

    // x.ne(inf).where(with_nan, inf_val) - +inf check
    UOp::try_where(not_pos_inf, with_nan, inf_val.clone()).expect("lazy_map: where3")
}
