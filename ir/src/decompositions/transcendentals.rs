//! Transcendental function decompositions.
//!
//! Implements ~1 ULP approximations for exp2, log2, sin, cos, pow.
//! Based on tinygrad's decompositions.py and SLEEF library.
//!
//! References:
//! - Paper: https://arxiv.org/pdf/2001.09258
//! - SLEEF: https://github.com/shibatch/sleef

use std::{
    f64::consts::{FRAC_1_PI, LN_2},
    sync::Arc,
};

use morok_dtype::{DType, ScalarDType};

use crate::uop::UOp;

use super::helpers::*;

// ============================================================================
// Constants for polynomial approximations
// ============================================================================

/// exp2 coefficients for float32 (7 terms)
const EXP2_COEFFS_F32: &[f64] =
    &[0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0];

/// exp2 coefficients for float64 (12 terms)
const EXP2_COEFFS_F64: &[f64] = &[
    4.434_359_082_926_529_5e-10,
    7.073_164_598_085_707_4e-9,
    1.017_819_260_921_760_5e-7,
    1.321_543_872_511_327_6e-6,
    1.525_273_353_517_584_7e-5,
    1.540_353_045_101_147_8e-4,
    1.333_355_814_670_499e-3,
    9.618_129_107_597_6e-3,
    5.550_410_866_482_046_6e-2,
    2.402_265_069_591_012_2e-1,
    LN_2,
    1.0,
];

/// log2 coefficients for float32 (3 terms for x^2 polynomial)
const LOG2_COEFFS_F32: &[f64] = &[0.4374550283e+0, 0.5764790177e+0, 0.9618012905120];

/// log2 coefficients for float64 (7 terms for x^2 polynomial)
const LOG2_COEFFS_F64: &[f64] = &[
    2.211_941_750_456_081_5e-1,
    2.200_768_693_152_277_7e-1,
    2.623_708_057_488_514_7e-1,
    3.205_977_477_944_495_5e-1,
    4.121_985_945_485_324_7e-1,
    5.770_780_162_997_059e-1,
    0.961_796_693_926_080_9,
];

/// sin polynomial coefficients for float32 (5 terms)
const SIN_COEFFS_F32: &[f64] = &[
    2.608_315_980_978_659_4e-6,
    -0.000_198_106_907_191_686_33,
    0.008_333_078_585_565_09,
    -0.166_666_597_127_914_43,
    1.0,
];

/// sin polynomial coefficients for float64 (10 terms)
const SIN_COEFFS_F64: &[f64] = &[
    -7.972_559_550_090_379e-18,
    2.810_099_727_108_632e-15,
    -7.647_122_191_181_588e-13,
    1.605_904_306_056_645e-10,
    -2.505_210_837_635_020_5e-8,
    2.755_731_922_391_987_5e-6,
    -0.000_198_412_698_412_696_16,
    0.008_333_333_333_333_33,
    -0.166_666_666_666_666_66,
    1.0,
];

// ============================================================================
// Core transcendental decompositions
// ============================================================================

/// xexp2: 2^d with ~1 ULP precision.
///
/// Algorithm:
/// 1. Mask special values (inf, nan) as 0
/// 2. q = round(d)
/// 3. s = d - q (fractional part in [-0.5, 0.5])
/// 4. u = poly(s) (polynomial approximation of 2^s)
/// 5. result = ldexp2k(u, q) = u * 2^q
/// 6. Handle edge cases: large positive → inf, large negative → 0
pub fn xexp2(d: &Arc<UOp>) -> Arc<UOp> {
    // Ensure input is scalar (cast Ptr to base scalar if needed)
    let d = &ensure_scalar(d);
    let dtype = d.dtype();

    // Mask inf/nan as zero for computation
    let zero = float_const(&dtype, 0.0);
    let x = lazy_map_numbers(d, &zero, &zero, &zero, d);

    // q = round(x) as integer
    let q = rintk(&x);

    // s = x - q (fractional part)
    let q_float = q.cast(dtype.clone());
    let s = x.try_sub(&q_float).expect("xexp2: sub failed");

    // Polynomial approximation of 2^s
    let coeffs = match dtype.scalar() {
        Some(ScalarDType::Float64) => EXP2_COEFFS_F64,
        _ => EXP2_COEFFS_F32,
    };
    let u = poly_n(&s, coeffs);

    // u * 2^q
    let result = ldexp2k(&u, &q);

    // Handle overflow/underflow bounds
    let (upper, lower) = match dtype.scalar() {
        Some(ScalarDType::Float64) => (1024.0, -2000.0),
        Some(ScalarDType::Float32) => (128.0, -150.0),
        Some(ScalarDType::Float16) => (23.0, -22.0),
        _ => (128.0, -150.0),
    };

    let upper_const = float_const(&dtype, upper);
    let lower_const = float_const(&dtype, lower);
    let inf = float_const(&dtype, f64::INFINITY);
    let nan = float_const(&dtype, f64::NAN);

    // d >= upper → inf
    let is_overflow = d.try_cmpge(&upper_const).expect("xexp2: cmpge overflow");
    let result = UOp::try_where(is_overflow, inf, result).expect("xexp2: where overflow");

    // d < lower → 0
    let is_underflow = d.try_cmplt(&lower_const).expect("xexp2: cmplt underflow");
    let result = UOp::try_where(is_underflow, zero.clone(), result).expect("xexp2: where underflow");

    // d != d (nan) → nan
    let is_nan = d.try_cmpne(d).expect("xexp2: cmpne nan");
    UOp::try_where(is_nan, nan, result).expect("xexp2: where nan")
}

/// xlog2: log2(d) with ~1 ULP precision.
///
/// Algorithm (from SLEEF/tinygrad):
/// 1. Handle denormals by scaling up
/// 2. Extract exponent e = ilogb2k(d * (1/0.75))
/// 3. Compute mantissa m = ldexp3k(d, -e) in [0.75, 1.5)
/// 4. Transform: x = (m-1)/(m+1)
/// 5. Polynomial approximation on x^2
/// 6. Handle edge cases
pub fn xlog2(d: &Arc<UOp>) -> Arc<UOp> {
    // Ensure input is scalar (cast Ptr to base scalar if needed)
    let d = &ensure_scalar(d);
    let dtype = d.dtype();

    // For float16, upcast to float32 for precision
    if dtype.scalar() == Some(ScalarDType::Float16) {
        let d_f32 = d.cast(DType::Float32);
        let result_f32 = xlog2(&d_f32);
        return result_f32.cast(dtype);
    }

    let flt_min = match dtype.scalar() {
        Some(ScalarDType::Float64) => 1e-4,
        _ => 1e-4,
    };
    let flt_min_const = float_const(&dtype, flt_min);

    // Check for denormals
    let is_denormal = d.try_cmplt(&flt_min_const).expect("xlog2: cmplt denormal");

    // Scale up denormals by 2^64
    let scale_up = float_const(&dtype, (2.0_f64).powi(64));
    let scaled = d.try_mul(&scale_up).expect("xlog2: mul scale");
    let a = UOp::try_where(is_denormal.clone(), scaled, d.clone()).expect("xlog2: where denormal");

    // Extract exponent: e = ilogb2k(a * (1/0.75))
    let inv_0_75 = float_const(&dtype, 1.0 / 0.75);
    let a_scaled = a.try_mul(&inv_0_75).expect("xlog2: mul inv_0_75");
    let e = ilogb2k(&a_scaled);
    let e_float = e.cast(dtype.clone());

    // Mantissa: m = ldexp3k(a, -e)
    let neg_e = e_float.neg();
    let m = ldexp3k(&a, &neg_e);

    // Adjust exponent for denormals
    let int_dtype = float_to_int_dtype(&dtype);
    let sixty_four = int_const(&int_dtype, 64);
    let e_adjusted = e.try_sub(&sixty_four).expect("xlog2: sub 64");
    let e = UOp::try_where(is_denormal, e_adjusted, e).expect("xlog2: where e adjust");
    let e_float = e.cast(dtype.clone());

    // Transform: x = (m - 1) / (m + 1)
    let one = float_const(&dtype, 1.0);
    let m_minus_1 = m.try_sub(&one).expect("xlog2: m - 1");
    let m_plus_1 = m.try_add(&one).expect("xlog2: m + 1");
    let x = m_minus_1.try_div(&m_plus_1).expect("xlog2: div");

    // x^2 for polynomial
    let x2 = x.try_mul(&x).expect("xlog2: x^2");

    // Polynomial on x^2
    let coeffs = match dtype.scalar() {
        Some(ScalarDType::Float64) => LOG2_COEFFS_F64,
        _ => LOG2_COEFFS_F32,
    };
    let t = poly_n(&x2, coeffs);

    // Build result: t * (x * x^2) + (e + x * 2.885...)
    let x_x2 = x.try_mul(&x2).expect("xlog2: x*x2");
    let t_term = t.try_mul(&x_x2).expect("xlog2: t*x*x2");

    // s_hi = e + x * 2.885...
    let log2_e = match dtype.scalar() {
        Some(ScalarDType::Float64) => 2.885_390_081_777_926_8,
        _ => 2.885_390_043_258_667,
    };
    let log2_e_const = float_const(&dtype, log2_e);
    let x_log2e = x.try_mul(&log2_e_const).expect("xlog2: x*log2e");
    let s_hi = e_float.try_add(&x_log2e).expect("xlog2: e + x*log2e");

    // For float32, add correction term
    let s = if dtype.scalar() == Some(ScalarDType::Float64) {
        s_hi
    } else {
        let s_lo_coeff = float_const(&dtype, 3.273_447_448_356_849e-8);
        let s_lo = x.try_mul(&s_lo_coeff).expect("xlog2: s_lo");
        s_hi.try_add(&s_lo).expect("xlog2: s_hi + s_lo")
    };

    let r = t_term.try_add(&s).expect("xlog2: final add");

    // Handle special cases
    let inf = float_const(&dtype, f64::INFINITY);
    let neg_inf = float_const(&dtype, f64::NEG_INFINITY);
    let nan = float_const(&dtype, f64::NAN);
    let neg_zero = float_const(&dtype, -0.0);

    // log2(inf) = inf
    let is_inf = d.try_cmpeq(&inf).expect("xlog2: cmpeq inf");
    let r = UOp::try_where(is_inf, inf.clone(), r).expect("xlog2: where inf");

    // log2(x < 0) = nan
    let is_neg = d.try_cmplt(&neg_zero).expect("xlog2: cmplt neg");
    let r = UOp::try_where(is_neg, nan.clone(), r).expect("xlog2: where neg");

    // log2(0) = -inf (check by looking at result being very negative)
    let log2_zero = match dtype.scalar() {
        Some(ScalarDType::Float64) => -1087.0,
        Some(ScalarDType::Float32) => -191.0,
        _ => -79.0,
    };
    let log2_zero_const = float_const(&dtype, log2_zero);
    let is_zero = r.try_cmpeq(&log2_zero_const).expect("xlog2: cmpeq zero");
    let r = UOp::try_where(is_zero, neg_inf.clone(), r).expect("xlog2: where zero");

    // log2(nan) = nan
    let is_nan = d.try_cmpne(d).expect("xlog2: cmpne nan");
    let r = UOp::try_where(is_nan, nan.clone(), r).expect("xlog2: where nan");

    // log2(-0.0) = -inf (check via reciprocal)
    let d_recip = UOp::try_reciprocal(d).expect("xlog2: reciprocal");
    let is_neg_zero = d_recip.try_cmpeq(&neg_inf).expect("xlog2: cmpeq neg_zero");
    UOp::try_where(is_neg_zero, neg_inf, r).expect("xlog2: where neg_zero")
}

/// xsin: sin(d) with ~1 ULP precision using Cody-Waite reduction.
///
/// Algorithm:
/// 1. Mask special values
/// 2. Cody-Waite reduction to [-π/2, π/2]
/// 3. Polynomial approximation
/// 4. Sign adjustment based on quadrant
pub fn xsin(d: &Arc<UOp>) -> Arc<UOp> {
    // Ensure input is scalar (cast Ptr to base scalar if needed)
    let d = &ensure_scalar(d);
    let dtype = d.dtype();

    // Mask inf/nan as 0 for computation
    let zero = float_const(&dtype, 0.0);
    let x = lazy_map_numbers(d, &zero, &zero, &zero, d);

    // Get sign of x
    let one = float_const(&dtype, 1.0);
    let neg_one = float_const(&dtype, -1.0);
    let is_zero = x.try_cmpeq(&zero).expect("xsin: cmpeq zero");
    let is_neg = x.try_cmplt(&zero).expect("xsin: cmplt zero");
    let sign = UOp::try_where(is_neg.clone(), neg_one.clone(), one.clone()).expect("xsin: where sign");
    let x_sign = UOp::try_where(is_zero, zero.clone(), sign).expect("xsin: where x_sign");

    // x_abs = |x|
    let x_abs = x.try_mul(&x_sign).expect("xsin: abs");

    // Cody-Waite reduction
    let (r, q) = cody_waite_reduction(&x_abs);

    // sin polynomial on reduced argument
    let result = sin_poly_small(&r, &q);

    // Adjust sign
    let result = result.try_mul(&x_sign).expect("xsin: sign adjust");

    // sin(inf) = nan, sin(-inf) = nan, sin(nan) = nan
    let nan = float_const(&dtype, f64::NAN);
    lazy_map_numbers(d, &nan, &nan, &nan, &result)
}

/// xcos: cos(d) = sin(d + π/2)
pub fn xcos(d: &Arc<UOp>) -> Arc<UOp> {
    // Ensure input is scalar (cast Ptr to base scalar if needed)
    let d = &ensure_scalar(d);
    let dtype = d.dtype();
    let pi_2 = float_const(&dtype, std::f64::consts::FRAC_PI_2);
    let shifted = d.try_add(&pi_2).expect("xcos: add pi/2");
    xsin(&shifted)
}

/// xexp: e^d = exp2(d * log2(e))
pub fn xexp(d: &Arc<UOp>) -> Arc<UOp> {
    // Ensure input is scalar (cast Ptr to base scalar if needed)
    let d = &ensure_scalar(d);
    let dtype = d.dtype();
    let log2_e = float_const(&dtype, std::f64::consts::LOG2_E);
    let scaled = d.try_mul(&log2_e).expect("xexp: mul log2e");
    xexp2(&scaled)
}

/// xlog: ln(d) = log2(d) / log2(e) = log2(d) * ln(2)
pub fn xlog(d: &Arc<UOp>) -> Arc<UOp> {
    // Ensure input is scalar (cast Ptr to base scalar if needed)
    let d = &ensure_scalar(d);
    let dtype = d.dtype();
    let ln_2 = float_const(&dtype, std::f64::consts::LN_2);
    let log2_d = xlog2(d);
    log2_d.try_mul(&ln_2).expect("xlog: mul ln2")
}

/// xtan: tan(d) = sin(d) / cos(d)
pub fn xtan(d: &Arc<UOp>) -> Arc<UOp> {
    // Ensure input is scalar (cast Ptr to base scalar if needed)
    let d = &ensure_scalar(d);
    let sin_d = xsin(d);
    let cos_d = xcos(d);
    sin_d.try_div(&cos_d).expect("xtan: div")
}

/// xsqrt: sqrt(d) = d^0.5
///
/// This decomposes sqrt into pow(d, 0.5) which then uses exp2/log2.
/// For backends that have native sqrt, this decomposition won't be used.
pub fn xsqrt(d: &Arc<UOp>) -> Arc<UOp> {
    // Ensure input is scalar (cast Ptr to base scalar if needed)
    let d = &ensure_scalar(d);
    let dtype = d.dtype();
    let half = float_const(&dtype, 0.5);
    xpow(d, &half)
}

/// xrsqrt: 1/sqrt(d) = d^(-0.5)
///
/// This decomposes rsqrt into pow(d, -0.5) which then uses exp2/log2.
/// For backends that have native rsqrt, this decomposition won't be used.
pub fn xrsqrt(d: &Arc<UOp>) -> Arc<UOp> {
    // Ensure input is scalar (cast Ptr to base scalar if needed)
    let d = &ensure_scalar(d);
    let dtype = d.dtype();
    let neg_half = float_const(&dtype, -0.5);
    xpow(d, &neg_half)
}

/// xerf: Error function approximation.
///
/// Uses Horner's method with coefficients from Abramowitz & Stegun.
/// Approximation: erf(x) ≈ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
/// where t = 1 / (1 + p*|x|) and p = 0.3275911
///
/// This gives ~1.5e-7 maximum error for float32.
pub fn xerf(d: &Arc<UOp>) -> Arc<UOp> {
    // Ensure input is scalar (cast Ptr to base scalar if needed)
    let d = &ensure_scalar(d);
    let dtype = d.dtype();

    // Constants for the approximation
    let p = float_const(&dtype, 0.3275911);
    let a1 = float_const(&dtype, 0.254829592);
    let a2 = float_const(&dtype, -0.284496736);
    let a3 = float_const(&dtype, 1.421413741);
    let a4 = float_const(&dtype, -1.453152027);
    let a5 = float_const(&dtype, 1.061405429);

    // Get sign of x
    let zero = float_const(&dtype, 0.0);
    let one = float_const(&dtype, 1.0);
    let neg_one = float_const(&dtype, -1.0);
    let is_neg = d.try_cmplt(&zero).expect("xerf: cmplt");
    let sign = UOp::try_where(is_neg.clone(), neg_one, one.clone()).expect("xerf: where sign");

    // x_abs = |x|
    let x_abs = d.abs();

    // t = 1 / (1 + p * |x|)
    let p_x = p.try_mul(&x_abs).expect("xerf: p*x");
    let one_plus_px = one.try_add(&p_x).expect("xerf: 1+px");
    let t = UOp::try_reciprocal(&one_plus_px).expect("xerf: reciprocal");

    // Polynomial: a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    // Using Horner's method: t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    let poly = {
        let inner = a5.try_mul(&t).expect("xerf: a5*t");
        let inner = inner.try_add(&a4).expect("xerf: +a4");
        let inner = inner.try_mul(&t).expect("xerf: *t");
        let inner = inner.try_add(&a3).expect("xerf: +a3");
        let inner = inner.try_mul(&t).expect("xerf: *t");
        let inner = inner.try_add(&a2).expect("xerf: +a2");
        let inner = inner.try_mul(&t).expect("xerf: *t");
        let inner = inner.try_add(&a1).expect("xerf: +a1");
        inner.try_mul(&t).expect("xerf: *t final")
    };

    // exp(-x^2)
    let x2 = x_abs.try_mul(&x_abs).expect("xerf: x^2");
    let neg_x2 = x2.neg();
    let exp_neg_x2 = xexp(&neg_x2);

    // result = 1 - poly * exp(-x^2)
    let poly_exp = poly.try_mul(&exp_neg_x2).expect("xerf: poly*exp");
    let result = one.try_sub(&poly_exp).expect("xerf: 1-poly*exp");

    // Apply sign
    result.try_mul(&sign).expect("xerf: *sign")
}

/// xpow: base^exponent = exp2(exponent * log2(base))
///
/// Handles special cases:
/// - Negative base with non-integer exponent → NaN
/// - Negative base with odd integer exponent → negate result
/// - 0^0 → 1
pub fn xpow(base: &Arc<UOp>, exponent: &Arc<UOp>) -> Arc<UOp> {
    // Ensure inputs are scalar (cast Ptr to base scalar if needed)
    let base = &ensure_scalar(base);
    let exponent = &ensure_scalar(exponent);
    let dtype = base.dtype();

    // |base|^exponent = exp2(exponent * log2(|base|))
    let abs_base = base.abs();
    let log2_abs = xlog2(&abs_base);
    let scaled = exponent.try_mul(&log2_abs).expect("xpow: mul");
    let ret = xexp2(&scaled);

    // Handle negative base
    let zero = float_const(&dtype, 0.0);
    let one = float_const(&dtype, 1.0);
    let neg_one = float_const(&dtype, -1.0);
    let nan = float_const(&dtype, f64::NAN);

    // base < 0
    let base_neg = base.try_cmplt(&zero).expect("xpow: cmplt base");

    // Check if exponent is integer: exp != cast(cast(exp, int), float)
    let int_dtype = float_to_int_dtype(&dtype);
    let exp_int = exponent.cast(int_dtype.clone());
    let exp_back = exp_int.cast(dtype.clone());
    let non_int = exponent.try_cmpne(&exp_back).expect("xpow: cmpne int");

    // For negative base: nan if non-integer exponent, else check odd/even
    // |exp| as int
    let exp_abs = exponent.abs();
    let exp_abs_int = exp_abs.cast(int_dtype.clone());

    // exp % 2 (check if odd)
    let two = int_const(&int_dtype, 2);
    let exp_mod_2 = exp_abs_int.try_mod(&two).expect("xpow: mod 2");
    let zero_int = int_const(&int_dtype, 0);
    let is_odd = exp_mod_2.try_cmpne(&zero_int).expect("xpow: cmpne odd");
    let is_odd_bool = is_odd.cast(DType::Bool);

    // Adjustment for negative base: -1 if odd exponent, 1 otherwise
    let odd_adj = UOp::try_where(is_odd_bool, neg_one, one.clone()).expect("xpow: where odd");

    // non_int → nan, else odd_adj
    let adj = UOp::try_where(non_int, nan, odd_adj).expect("xpow: where non_int");

    // Apply adjustment only for negative base
    let result = UOp::try_where(base_neg, ret.try_mul(&adj).expect("xpow: mul adj"), ret).expect("xpow: where neg");

    // 0^0 = 1
    let base_zero = base.try_cmpeq(&zero).expect("xpow: cmpeq base zero");
    let exp_zero = exponent.try_cmpeq(&zero).expect("xpow: cmpeq exp zero");
    let both_zero = base_zero.try_and_op(&exp_zero).expect("xpow: and zeros");
    UOp::try_where(both_zero, one, result).expect("xpow: where 0^0")
}

// ============================================================================
// Internal helper functions
// ============================================================================

/// Cody-Waite reduction: reduce d to [-π/2, π/2] range.
///
/// Returns (reduced_value, quadrant)
fn cody_waite_reduction(d: &Arc<UOp>) -> (Arc<UOp>, Arc<UOp>) {
    let dtype = scalar_dtype(&d.dtype());
    let m_1_pi = FRAC_1_PI; // 1/π

    // quadrant = round(d / π)
    let m_1_pi_const = float_const(&dtype, m_1_pi);
    let d_over_pi = d.try_mul(&m_1_pi_const).expect("cody_waite: d/pi");
    let quadrant = rintk(&d_over_pi);
    let q_float = quadrant.cast(dtype.clone());

    // Reduce: d - quadrant * π (using extended precision constants)
    let reduced = if dtype.scalar() == Some(ScalarDType::Float64) {
        // High precision reduction for float64
        let pi_a = float_const(&dtype, 3.141_592_621_803_283_7);
        let pi_b = float_const(&dtype, 3.178_650_942_459_171_3e-8);
        let pi_c = float_const(&dtype, 1.224_646_786_410_718_9e-16);
        let pi_d = float_const(&dtype, 1.273_663_432_702_19e-24);

        let mut r = d.clone();
        r = r.try_sub(&q_float.try_mul(&pi_a).expect("cw: mul pi_a")).expect("cw: sub pi_a");
        r = r.try_sub(&q_float.try_mul(&pi_b).expect("cw: mul pi_b")).expect("cw: sub pi_b");
        r = r.try_sub(&q_float.try_mul(&pi_c).expect("cw: mul pi_c")).expect("cw: sub pi_c");
        r = r.try_sub(&q_float.try_mul(&pi_d).expect("cw: mul pi_d")).expect("cw: sub pi_d");
        r
    } else if dtype.scalar() == Some(ScalarDType::Float16) {
        // Float16 needs float32 precision
        let d_f32 = d.cast(DType::Float32);
        let q_f32 = q_float.cast(DType::Float32);
        let (r_f32, _) = cody_waite_reduction_f32(&d_f32, &q_f32);
        r_f32.cast(dtype.clone())
    } else {
        // Float32 reduction
        let (r, _) = cody_waite_reduction_f32(d, &q_float);
        r
    };

    (reduced, quadrant)
}

/// Float32 Cody-Waite reduction helper.
fn cody_waite_reduction_f32(d: &Arc<UOp>, q: &Arc<UOp>) -> (Arc<UOp>, Arc<UOp>) {
    let dtype = scalar_dtype(&d.dtype());

    let pi_1 = float_const(&dtype, 3.1414794921875);
    let pi_2 = float_const(&dtype, 0.000_113_159_418_106_079_1);
    let pi_3 = float_const(&dtype, 1.984_187_258_941_006e-9);
    let pi_4 = float_const(&dtype, 1.215_420_125_655_342e-10);

    let mut r = d.clone();
    r = r.try_sub(&q.try_mul(&pi_1).expect("cw32: mul pi1")).expect("cw32: sub pi1");
    r = r.try_sub(&q.try_mul(&pi_2).expect("cw32: mul pi2")).expect("cw32: sub pi2");
    r = r.try_sub(&q.try_mul(&pi_3).expect("cw32: mul pi3")).expect("cw32: sub pi3");
    r = r.try_sub(&q.try_mul(&pi_4).expect("cw32: mul pi4")).expect("cw32: sub pi4");

    (r, q.clone())
}

/// sin polynomial approximation: sin(d) ≈ d * poly(d^2)
fn sin_poly(d: &Arc<UOp>) -> Arc<UOp> {
    let dtype = scalar_dtype(&d.dtype());
    let d2 = d.try_mul(d).expect("sin_poly: d^2");

    let coeffs = match dtype.scalar() {
        Some(ScalarDType::Float64) => SIN_COEFFS_F64,
        _ => SIN_COEFFS_F32,
    };

    let poly_result = poly_n(&d2, coeffs);
    d.try_mul(&poly_result).expect("sin_poly: d * poly")
}

/// sin polynomial with quadrant adjustment (small angle version).
///
/// For quadrant q:
/// - q % 4 == 0: sin(r)
/// - q % 4 == 1: sin(r + π/2) = cos(r)
/// - q % 4 == 2: -sin(r)
/// - q % 4 == 3: -cos(r)
fn sin_poly_small(r: &Arc<UOp>, q: &Arc<UOp>) -> Arc<UOp> {
    let dtype = scalar_dtype(&r.dtype());
    let result = sin_poly(r);

    // q & 1 != 0 → negate
    let int_dtype = float_to_int_dtype(&dtype);
    let one_int = int_const(&int_dtype, 1);
    let q_and_1 = q.try_and_op(&one_int).expect("sin_small: q & 1");
    let zero_int = int_const(&int_dtype, 0);
    let is_odd = q_and_1.try_cmpne(&zero_int).expect("sin_small: cmpne");

    let neg_one = float_const(&dtype, -1.0);
    let one = float_const(&dtype, 1.0);
    let sign = UOp::try_where(is_odd, neg_one, one).expect("sin_small: where sign");

    result.try_mul(&sign).expect("sin_small: mul sign")
}
