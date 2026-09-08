//! Scalar operands, the `std::ops` operators and the ops that shed their `Result`.

use crate::*;
use svod_dtype::DType;
use test_case::test_case;

fn f32v(t: &Tensor) -> Vec<f32> {
    t.to_vec::<f32>().unwrap()
}

fn floats() -> Tensor {
    Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0])
}

fn ints() -> Tensor {
    Tensor::from_slice([1i32, 2, 3, 4])
}

// =============================================================================
// Scalar right-hand sides
// =============================================================================

/// A scalar rhs computes what the same value spelled as a tensor computes.
#[test_case(Tensor::try_add, 2.0, &[3.0, 4.0, 5.0, 6.0]; "add")]
#[test_case(Tensor::try_sub, 2.0, &[-1.0, 0.0, 1.0, 2.0]; "sub")]
#[test_case(Tensor::try_mul, 2.0, &[2.0, 4.0, 6.0, 8.0]; "mul")]
#[test_case(Tensor::try_div, 2.0, &[0.5, 1.0, 1.5, 2.0]; "div")]
#[test_case(Tensor::try_pow, 2.0, &[1.0, 4.0, 9.0, 16.0]; "pow")]
#[test_case(Tensor::maximum, 2.0, &[2.0, 2.0, 3.0, 4.0]; "maximum")]
#[test_case(Tensor::minimum, 2.0, &[1.0, 2.0, 2.0, 2.0]; "minimum")]
fn float_scalar_rhs(op: fn(&Tensor, f64) -> Result<Tensor>, scalar: f64, expected: &[f32]) {
    assert_eq!(f32v(&op(&floats(), scalar).unwrap()), expected);
}

/// A scalar rhs computes exactly what the same constant spelled as a tensor does.
#[test]
fn scalar_rhs_matches_tensor_rhs() {
    let x = floats();
    let two = Tensor::const_(2.0f32, DType::Float32);
    let pairs: [(Result<Tensor>, Result<Tensor>); 5] = [
        (x.try_sub(2.0), x.try_sub(&two)),
        (x.try_div(2.0), x.try_div(&two)),
        (x.try_pow(2.0), x.try_pow(&two)),
        (x.maximum(2.0), x.maximum(&two)),
        (x.minimum(2.0), x.minimum(&two)),
    ];
    for (scalar, tensor) in pairs {
        assert_eq!(f32v(&scalar.unwrap()), f32v(&tensor.unwrap()));
    }
}

/// Integer tensors take integer scalars, staying in their own dtype.
#[test_case(Tensor::try_add, 3, &[4, 5, 6, 7]; "add")]
#[test_case(Tensor::try_mul, 3, &[3, 6, 9, 12]; "mul")]
#[test_case(Tensor::try_div, 2, &[0, 1, 1, 2]; "div")]
#[test_case(Tensor::try_mod, 2, &[1, 0, 1, 0]; "modulo")]
#[test_case(Tensor::try_bitand, 2, &[0, 2, 2, 0]; "bitand")]
#[test_case(Tensor::try_bitor, 2, &[3, 2, 3, 6]; "bitor")]
#[test_case(Tensor::try_bitxor, 2, &[3, 0, 1, 6]; "bitxor")]
#[test_case(Tensor::try_shl, 1, &[2, 4, 6, 8]; "shl")]
#[test_case(Tensor::try_shr, 1, &[0, 1, 1, 2]; "shr")]
fn int_scalar_rhs(op: fn(&Tensor, i32) -> Result<Tensor>, scalar: i32, expected: &[i32]) {
    let out = op(&ints(), scalar).unwrap();
    assert_eq!(out.dtype(), DType::Int32);
    assert_eq!(out.to_vec::<i32>().unwrap(), expected);
}

/// Comparisons against a scalar produce bool regardless of the input dtype.
#[test_case(Tensor::try_eq, &[false, true, false, false]; "eq")]
#[test_case(Tensor::try_ne, &[true, false, true, true]; "ne")]
#[test_case(Tensor::try_lt, &[true, false, false, false]; "lt")]
#[test_case(Tensor::try_le, &[true, true, false, false]; "le")]
#[test_case(Tensor::try_gt, &[false, false, true, true]; "gt")]
#[test_case(Tensor::try_ge, &[false, true, true, true]; "ge")]
fn float_scalar_comparison(op: fn(&Tensor, f64) -> Result<Tensor>, expected: &[bool]) {
    let out = op(&floats(), 2.0).unwrap();
    assert_eq!(out.dtype(), DType::Bool);
    assert_eq!(out.to_vec::<bool>().unwrap(), expected);
}

/// A bool tensor takes `true`/`false` as a scalar operand.
#[test]
fn bool_scalar_rhs() {
    let mask = Tensor::from_slice([true, false, true, false]);
    assert_eq!(mask.try_bitand(true).unwrap().to_vec::<bool>().unwrap(), [true, false, true, false]);
    assert_eq!(mask.try_bitor(true).unwrap().to_vec::<bool>().unwrap(), [true; 4]);
    assert_eq!(mask.try_ne(true).unwrap().to_vec::<bool>().unwrap(), [false, true, false, true]);
}

/// `&t`, `t` and `&&t` are all valid operands, alongside the scalar.
#[test]
// The `&&Tensor` conversion is exactly what this test pins down.
#[allow(clippy::needless_borrows_for_generic_args)]
fn operand_conversions() {
    let x = floats();
    let rhs = Tensor::const_(2.0f32, DType::Float32);
    let borrowed: &Tensor = &rhs;
    let expected = [3.0f32, 4.0, 5.0, 6.0];
    assert_eq!(f32v(&x.try_add(&rhs).unwrap()), expected);
    assert_eq!(f32v(&x.try_add(borrowed).unwrap()), expected);
    assert_eq!(f32v(&x.try_add(&borrowed).unwrap()), expected);
    assert_eq!(f32v(&x.try_add(rhs.clone()).unwrap()), expected);
    assert_eq!(f32v(&x.try_add(2.0f32).unwrap()), expected);
    assert_eq!(f32v(&x.try_add(svod_ir::ConstValue::Float(2.0)).unwrap()), expected);
}

// =============================================================================
// Operators
// =============================================================================

#[test_case(&(&floats() + 2.0), &[3.0, 4.0, 5.0, 6.0]; "add scalar")]
#[test_case(&(&floats() - 1.0), &[0.0, 1.0, 2.0, 3.0]; "sub scalar")]
#[test_case(&(&floats() * 3.0), &[3.0, 6.0, 9.0, 12.0]; "mul scalar")]
#[test_case(&(&floats() / 2.0), &[0.5, 1.0, 1.5, 2.0]; "div scalar")]
#[test_case(&(2.0f64 + &floats()), &[3.0, 4.0, 5.0, 6.0]; "scalar add")]
#[test_case(&(10.0f64 - &floats()), &[9.0, 8.0, 7.0, 6.0]; "scalar sub")]
#[test_case(&(3.0f32 * &floats()), &[3.0, 6.0, 9.0, 12.0]; "scalar mul")]
#[test_case(&(12.0f64 / &floats()), &[12.0, 6.0, 4.0, 3.0]; "scalar div")]
#[test_case(&(floats() + floats()), &[2.0, 4.0, 6.0, 8.0]; "owned both sides")]
#[test_case(&(&floats() + floats()), &[2.0, 4.0, 6.0, 8.0]; "owned rhs")]
fn operator_results(actual: &Result<Tensor>, expected: &[f32]) {
    assert_eq!(f32v(actual.as_ref().unwrap()), expected);
}

/// The integer-only operators, including the `Rem` → `try_mod` mapping.
#[test]
fn integer_operators() {
    assert_eq!((ints() % 2).unwrap().to_vec::<i32>().unwrap(), [1, 0, 1, 0]);
    assert_eq!((ints() & 2).unwrap().to_vec::<i32>().unwrap(), [0, 2, 2, 0]);
    assert_eq!((ints() | 2).unwrap().to_vec::<i32>().unwrap(), [3, 2, 3, 6]);
    assert_eq!((ints() ^ 2).unwrap().to_vec::<i32>().unwrap(), [3, 0, 1, 6]);
    assert_eq!((ints() << 1).unwrap().to_vec::<i32>().unwrap(), [2, 4, 6, 8]);
    assert_eq!((ints() >> 1).unwrap().to_vec::<i32>().unwrap(), [0, 1, 1, 2]);
}

/// Chained operators: each stage is a `Result` the next one unwraps.
#[test]
fn operator_chain() {
    let a = floats();
    let b = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);
    let chained = ((&a + &b).unwrap() * 2.0).unwrap();
    assert_eq!(f32v(&chained), [22.0, 44.0, 66.0, 88.0]);

    let mixed = (((&a * &b).unwrap() - 1.0).unwrap() / 3.0).unwrap();
    crate::test::helpers::assert_close_f32(&f32v(&mixed), &[3.0, 13.0, 89.0 / 3.0, 53.0], 1e-5);
}

/// A mismatched shape surfaces as `Err`, not a panic.
#[test]
fn operator_shape_mismatch_is_err() {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([1.0f32, 2.0]);
    assert!((&a + &b).is_err());
}

/// Negation stays infallible.
#[test]
fn neg_is_infallible() {
    let x = floats();
    assert_eq!(f32v(&-&x), [-1.0, -2.0, -3.0, -4.0]);
    assert_eq!(f32v(&-x.clone()), [-1.0, -2.0, -3.0, -4.0]);
    assert_eq!(f32v(&x.neg()), [-1.0, -2.0, -3.0, -4.0]);
}

// =============================================================================
// select / clamp / masked_fill
// =============================================================================

#[test]
fn select_tensor_and_scalar_branches() {
    let x = floats();
    let big = x.try_gt(2.0).unwrap();

    // Tensor / tensor.
    let zeros = Tensor::zeros(&[4], DType::Float32);
    assert_eq!(f32v(&big.select(&x, &zeros).unwrap()), [0.0, 0.0, 3.0, 4.0]);
    // Tensor / scalar.
    assert_eq!(f32v(&big.select(&x, 0.0).unwrap()), [0.0, 0.0, 3.0, 4.0]);
    // Scalar / tensor.
    assert_eq!(f32v(&big.select(-1.0, &x).unwrap()), [1.0, 2.0, -1.0, -1.0]);
    // Scalar / scalar — the dtype falls back to the scalars' own.
    let picked = big.select(1.0, 0.0).unwrap();
    assert_eq!(picked.dtype(), DType::Float32);
    assert_eq!(f32v(&picked), [0.0, 0.0, 1.0, 1.0]);
}

/// `select` is `where_` with the branches swapped around the condition.
#[test]
fn select_matches_where() {
    let x = floats();
    let y = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);
    let cond = x.try_lt(3.0).unwrap();
    assert_eq!(f32v(&cond.select(&x, &y).unwrap()), f32v(&x.where_(&cond, &y).unwrap()));
}

#[test_case(Some(2.0), Some(3.0), &[2.0, 2.0, 3.0, 3.0]; "both bounds")]
#[test_case(Some(2.0), None, &[2.0, 2.0, 3.0, 4.0]; "lower only")]
#[test_case(None, Some(3.0), &[1.0, 2.0, 3.0, 3.0]; "upper only")]
fn clamp_scalar_bounds(min: Option<f64>, max: Option<f64>, expected: &[f32]) {
    let out = floats().clamp().maybe_min(min).maybe_max(max).call().unwrap();
    assert_eq!(f32v(&out), expected);
}

/// The scalar and tensor spellings of the bounds agree, through both names.
#[test]
fn clamp_and_clip_accept_both_spellings() {
    let x = floats();
    let lo = Tensor::const_(2.0f32, DType::Float32);
    let expected = [2.0f32, 2.0, 3.0, 3.0];
    assert_eq!(f32v(&x.clamp().min(2.0).max(3.0).call().unwrap()), expected);
    assert_eq!(f32v(&x.clamp().min(&lo).max(3.0).call().unwrap()), expected);
    assert_eq!(f32v(&x.clip().min(2.0).max(3.0).call().unwrap()), expected);
}

#[test]
fn masked_fill_scalar_and_tensor() {
    let x = floats();
    let mask = Tensor::from_slice([true, false, true, false]);
    assert_eq!(f32v(&x.masked_fill(&mask, 0.0f32).unwrap()), [0.0, 2.0, 0.0, 4.0]);
    let fill = Tensor::from_slice([-1.0f32, -2.0, -3.0, -4.0]);
    assert_eq!(f32v(&x.masked_fill(&mask, &fill).unwrap()), [-1.0, 2.0, -3.0, 4.0]);
}

// =============================================================================
// Ops that no longer return `Result`
// =============================================================================

#[test_case(Tensor::abs, &[1.0, 2.0, 3.0]; "abs")]
#[test_case(Tensor::neg, &[-1.0, 2.0, -3.0]; "neg")]
#[test_case(Tensor::square, &[1.0, 4.0, 9.0]; "square")]
#[test_case(Tensor::sign, &[1.0, -1.0, 1.0]; "sign")]
fn infallible_unary(op: fn(&Tensor) -> Tensor, expected: &[f32]) {
    let x = Tensor::from_slice([1.0f32, -2.0, 3.0]);
    assert_eq!(f32v(&op(&x)), expected);
}

#[test_case(Tensor::floor, &[1.0, -2.0, 2.0]; "floor")]
#[test_case(Tensor::ceil, &[2.0, -1.0, 3.0]; "ceil")]
#[test_case(Tensor::round, &[1.0, -1.0, 2.0]; "round")] // half to even
#[test_case(Tensor::trunc, &[1.0, -1.0, 2.0]; "trunc")]
fn infallible_rounding(op: fn(&Tensor) -> Tensor, expected: &[f32]) {
    let x = Tensor::from_slice([1.2f32, -1.2, 2.5]);
    assert_eq!(f32v(&op(&x)), expected);
}

#[test]
fn infallible_constructors_and_cast() {
    assert_eq!(f32v(&Tensor::zeros(&[3], DType::Float32)), [0.0; 3]);
    assert_eq!(f32v(&Tensor::ones(&[3], DType::Float32)), [1.0; 3]);
    assert_eq!(f32v(&Tensor::full(&[3], 7.0f32, DType::Float32)), [7.0; 3]);
    // A rank-0 request yields the bare scalar.
    assert!(Tensor::zeros(&[], DType::Float32).shape().unwrap().is_empty());

    let casted = Tensor::from_slice([1.7f32, -2.3]).cast(DType::Int32);
    assert_eq!(casted.dtype(), DType::Int32);
    assert_eq!(casted.to_vec::<i32>().unwrap(), [1, -2]);
}

#[test]
fn logical_and_bitwise_not() {
    let flags = Tensor::from_slice([true, false, true]);
    assert_eq!(flags.logical_not().unwrap().to_vec::<bool>().unwrap(), [false, true, false]);

    let ints = Tensor::from_slice([0i32, 1, 2, -1]);
    assert_eq!(ints.bitwise_not().unwrap().to_vec::<i32>().unwrap(), [-1, -2, -3, 0]);
    assert!(Tensor::from_slice([1.0f32]).bitwise_not().is_err());
}
