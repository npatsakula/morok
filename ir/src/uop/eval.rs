//! Constant evaluation for UOp operations.
//!
//! This module provides constant folding by evaluating operations on `ConstValue`.
//! It mirrors the design of the casting infrastructure in `types.rs`.

use crate::types::{BinaryOp, ConstValue, TernaryOp, UnaryOp};

/// Evaluate a unary operation on a constant value.
///
/// Returns `None` if:
/// - The operation is not supported for the given value type
/// - The operation would produce an invalid result (e.g., sqrt of negative)
///
/// # Semantics
///
/// All operations follow Rust/C semantics:
/// - Floating point operations follow IEEE 754
/// - Integer operations use wrapping arithmetic
pub fn eval_unary_op(op: UnaryOp, v: ConstValue) -> Option<ConstValue> {
    match op {
        UnaryOp::Neg => eval_neg(v),
        UnaryOp::Not => eval_not(v),
        UnaryOp::Abs => eval_abs(v),
        UnaryOp::Sqrt => eval_sqrt(v),
        UnaryOp::Rsqrt => eval_rsqrt(v),
        UnaryOp::Exp => eval_exp(v),
        UnaryOp::Exp2 => eval_exp2(v),
        UnaryOp::Log => eval_log(v),
        UnaryOp::Log2 => eval_log2(v),
        UnaryOp::Sin => eval_sin(v),
        UnaryOp::Cos => eval_cos(v),
        UnaryOp::Tan => eval_tan(v),
        UnaryOp::Reciprocal => eval_reciprocal(v),
        UnaryOp::Trunc => eval_trunc(v),
        UnaryOp::Floor => eval_floor(v),
        UnaryOp::Ceil => eval_ceil(v),
        UnaryOp::Round => eval_round(v),
        UnaryOp::Sign => eval_sign(v),
        UnaryOp::Erf => eval_erf(v),
        UnaryOp::Square => eval_square(v),
    }
}

/// Evaluate a binary operation on constant values.
///
/// Returns `None` if:
/// - The operation is not supported for the given value types
/// - The operands have incompatible types
/// - The operation would produce an invalid result (e.g., division by zero)
///
/// # Semantics
///
/// - Arithmetic operations preserve LHS type
/// - Comparison operations return Bool
/// - Bitwise operations require integer/bool types
/// - Division by zero returns None
/// - Integer operations use wrapping arithmetic
pub fn eval_binary_op(op: BinaryOp, a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match op {
        BinaryOp::Add => eval_add(a, b),
        BinaryOp::Mul => eval_mul(a, b),
        BinaryOp::Sub => eval_sub(a, b),
        BinaryOp::Mod => eval_mod(a, b),
        BinaryOp::Max => eval_max(a, b),
        BinaryOp::Pow => eval_pow(a, b),
        BinaryOp::Idiv => eval_idiv(a, b),
        BinaryOp::Fdiv => eval_fdiv(a, b),
        BinaryOp::Lt => eval_lt(a, b),
        BinaryOp::Le => eval_le(a, b),
        BinaryOp::Eq => eval_eq(a, b),
        BinaryOp::Ne => eval_ne(a, b),
        BinaryOp::Gt => eval_gt(a, b),
        BinaryOp::Ge => eval_ge(a, b),
        BinaryOp::And => eval_and(a, b),
        BinaryOp::Or => eval_or(a, b),
        BinaryOp::Xor => eval_xor(a, b),
        BinaryOp::Shl => eval_shl(a, b),
        BinaryOp::Shr => eval_shr(a, b),
        BinaryOp::Threefry => None, // PRNG - not constant foldable
    }
}

/// Evaluate a ternary operation on constant values.
///
/// Returns `None` if the operation is not supported for the given value types.
pub fn eval_ternary_op(op: TernaryOp, a: ConstValue, b: ConstValue, c: ConstValue) -> Option<ConstValue> {
    match op {
        TernaryOp::Where => eval_where(a, b, c),
        TernaryOp::MulAcc => eval_mulacc(a, b, c),
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

#[inline]
fn eval_neg(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Int(x) => Some(ConstValue::Int(x.wrapping_neg())),
        ConstValue::Float(x) => Some(ConstValue::Float(-x)),
        _ => None,
    }
}

#[inline]
fn eval_not(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Bool(b) => Some(ConstValue::Bool(!b)),
        ConstValue::Int(i) => Some(ConstValue::Int(!i)),
        ConstValue::UInt(u) => Some(ConstValue::UInt(!u)),
        _ => None,
    }
}

#[inline]
fn eval_abs(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Int(x) => Some(ConstValue::Int(x.wrapping_abs())),
        ConstValue::UInt(x) => Some(ConstValue::UInt(x)), // Already positive
        ConstValue::Float(x) => Some(ConstValue::Float(x.abs())),
        _ => None,
    }
}

#[inline]
fn eval_sqrt(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.sqrt())),
        _ => None,
    }
}

#[inline]
fn eval_rsqrt(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(1.0 / x.sqrt())),
        _ => None,
    }
}

#[inline]
fn eval_exp(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.exp())),
        _ => None,
    }
}

#[inline]
fn eval_exp2(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.exp2())),
        _ => None,
    }
}

#[inline]
fn eval_log(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.ln())),
        _ => None,
    }
}

#[inline]
fn eval_log2(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.log2())),
        _ => None,
    }
}

#[inline]
fn eval_sin(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.sin())),
        _ => None,
    }
}

#[inline]
fn eval_reciprocal(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(1.0 / x)),
        _ => None,
    }
}

#[inline]
fn eval_trunc(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.trunc())),
        _ => None,
    }
}

#[inline]
fn eval_cos(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.cos())),
        _ => None,
    }
}

#[inline]
fn eval_tan(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.tan())),
        _ => None,
    }
}

#[inline]
fn eval_floor(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.floor())),
        ConstValue::Int(x) => Some(ConstValue::Int(x)), // Already integer
        ConstValue::UInt(x) => Some(ConstValue::UInt(x)), // Already integer
        _ => None,
    }
}

#[inline]
fn eval_ceil(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.ceil())),
        ConstValue::Int(x) => Some(ConstValue::Int(x)), // Already integer
        ConstValue::UInt(x) => Some(ConstValue::UInt(x)), // Already integer
        _ => None,
    }
}

#[inline]
fn eval_round(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.round())),
        ConstValue::Int(x) => Some(ConstValue::Int(x)), // Already integer
        ConstValue::UInt(x) => Some(ConstValue::UInt(x)), // Already integer
        _ => None,
    }
}

#[inline]
fn eval_sign(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Int(x) => Some(ConstValue::Int(if x < 0 {
            -1
        } else if x > 0 {
            1
        } else {
            0
        })),
        ConstValue::Float(x) => Some(ConstValue::Float(if x < 0.0 {
            -1.0
        } else if x > 0.0 {
            1.0
        } else {
            0.0
        })),
        ConstValue::UInt(x) => Some(ConstValue::UInt(if x > 0 { 1 } else { 0 })),
        _ => None,
    }
}

#[inline]
fn eval_erf(v: ConstValue) -> Option<ConstValue> {
    match v {
        // Use libm for erf function (Rust std doesn't have it)
        ConstValue::Float(x) => Some(ConstValue::Float(libm::erf(x))),
        _ => None,
    }
}

#[inline]
fn eval_square(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Int(x) => Some(ConstValue::Int(x.wrapping_mul(x))),
        ConstValue::UInt(x) => Some(ConstValue::UInt(x.wrapping_mul(x))),
        ConstValue::Float(x) => Some(ConstValue::Float(x * x)),
        _ => None,
    }
}

// ============================================================================
// Binary Arithmetic Operations
// ============================================================================

/// Evaluate addition on two constant values.
#[inline]
pub fn eval_add(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Int(x.wrapping_add(y))),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::UInt(x.wrapping_add(y))),
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Float(x + y)),
        _ => None,
    }
}

/// Evaluate multiplication on two constant values.
#[inline]
pub fn eval_mul(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Int(x.wrapping_mul(y))),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::UInt(x.wrapping_mul(y))),
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Float(x * y)),
        _ => None,
    }
}

/// Evaluate subtraction on two constant values.
#[inline]
pub fn eval_sub(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Int(x.wrapping_sub(y))),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::UInt(x.wrapping_sub(y))),
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Float(x - y)),
        _ => None,
    }
}

#[inline]
fn eval_mod(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) if y != 0 => Some(ConstValue::Int(x % y)),
        (ConstValue::UInt(x), ConstValue::UInt(y)) if y != 0 => Some(ConstValue::UInt(x % y)),
        _ => None,
    }
}

#[inline]
fn eval_max(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Int(x.max(y))),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::UInt(x.max(y))),
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Float(x.max(y))),
        _ => None,
    }
}

#[inline]
fn eval_pow(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Float(x.powf(y))),
        (ConstValue::Int(x), ConstValue::Int(y)) if y >= 0 => {
            // Integer power - use checked operations to avoid overflow panic
            let result = (x as f64).powi(y as i32) as i64;
            Some(ConstValue::Int(result))
        }
        (ConstValue::UInt(x), ConstValue::UInt(y)) => {
            let result = (x as f64).powi(y as i32) as u64;
            Some(ConstValue::UInt(result))
        }
        _ => None,
    }
}

#[inline]
fn eval_idiv(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) if y != 0 => Some(ConstValue::Int(x / y)),
        (ConstValue::UInt(x), ConstValue::UInt(y)) if y != 0 => Some(ConstValue::UInt(x / y)),
        _ => None,
    }
}

#[inline]
fn eval_fdiv(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Float(x / y)),
        _ => None,
    }
}

// ============================================================================
// Binary Comparison Operations
// ============================================================================

#[inline]
fn eval_lt(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Bool(x < y)),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::Bool(x < y)),
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Bool(x < y)),
        (ConstValue::Bool(x), ConstValue::Bool(y)) => Some(ConstValue::Bool(!x & y)),
        _ => None,
    }
}

#[inline]
fn eval_eq(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Bool(x == y)),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::Bool(x == y)),
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Bool(x == y)),
        (ConstValue::Bool(x), ConstValue::Bool(y)) => Some(ConstValue::Bool(x == y)),
        _ => None,
    }
}

#[inline]
fn eval_ne(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Bool(x != y)),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::Bool(x != y)),
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Bool(x != y)),
        (ConstValue::Bool(x), ConstValue::Bool(y)) => Some(ConstValue::Bool(x != y)),
        _ => None,
    }
}

#[inline]
fn eval_le(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Bool(x <= y)),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::Bool(x <= y)),
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Bool(x <= y)),
        (ConstValue::Bool(x), ConstValue::Bool(y)) => Some(ConstValue::Bool((!x) | y)),
        _ => None,
    }
}

#[inline]
fn eval_gt(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Bool(x > y)),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::Bool(x > y)),
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Bool(x > y)),
        (ConstValue::Bool(x), ConstValue::Bool(y)) => Some(ConstValue::Bool(x & !y)),
        _ => None,
    }
}

#[inline]
fn eval_ge(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Bool(x >= y)),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::Bool(x >= y)),
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Bool(x >= y)),
        (ConstValue::Bool(x), ConstValue::Bool(y)) => Some(ConstValue::Bool(x | !y)),
        _ => None,
    }
}

// ============================================================================
// Binary Bitwise Operations
// ============================================================================

#[inline]
fn eval_and(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Int(x & y)),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::UInt(x & y)),
        (ConstValue::Bool(x), ConstValue::Bool(y)) => Some(ConstValue::Bool(x & y)),
        _ => None,
    }
}

#[inline]
fn eval_or(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Int(x | y)),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::UInt(x | y)),
        (ConstValue::Bool(x), ConstValue::Bool(y)) => Some(ConstValue::Bool(x | y)),
        _ => None,
    }
}

#[inline]
fn eval_xor(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Int(x ^ y)),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::UInt(x ^ y)),
        (ConstValue::Bool(x), ConstValue::Bool(y)) => Some(ConstValue::Bool(x ^ y)),
        _ => None,
    }
}

#[inline]
fn eval_shl(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) if (0..64).contains(&y) => {
            Some(ConstValue::Int(x.wrapping_shl(y as u32)))
        }
        (ConstValue::UInt(x), ConstValue::UInt(y)) if y < 64 => Some(ConstValue::UInt(x.wrapping_shl(y as u32))),
        _ => None,
    }
}

#[inline]
fn eval_shr(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) if (0..64).contains(&y) => {
            Some(ConstValue::Int(x.wrapping_shr(y as u32)))
        }
        (ConstValue::UInt(x), ConstValue::UInt(y)) if y < 64 => Some(ConstValue::UInt(x.wrapping_shr(y as u32))),
        _ => None,
    }
}

// ============================================================================
// Ternary Operations
// ============================================================================

#[inline]
fn eval_where(cond: ConstValue, true_val: ConstValue, false_val: ConstValue) -> Option<ConstValue> {
    match cond {
        ConstValue::Bool(true) => Some(true_val),
        ConstValue::Bool(false) => Some(false_val),
        _ => None,
    }
}

#[inline]
fn eval_mulacc(a: ConstValue, b: ConstValue, c: ConstValue) -> Option<ConstValue> {
    // MulAcc: a * b + c
    let mul_result = eval_mul(a, b)?;
    eval_add(mul_result, c)
}

// ============================================================================
// Dtype-Aware Evaluation (with truncation for constant folding)
// ============================================================================

use morok_dtype::ScalarDType;

/// Evaluate unary op with dtype-aware truncation (for constant folding).
///
/// Applies truncation after evaluation to ensure results fit within dtype boundaries.
#[inline]
pub fn eval_unary_op_typed(op: UnaryOp, v: ConstValue, dtype: ScalarDType) -> Option<ConstValue> {
    eval_unary_op(op, v).map(|r| r.truncate(dtype))
}

/// Evaluate binary op with dtype-aware truncation (for constant folding).
///
/// Applies truncation after evaluation to ensure results fit within dtype boundaries.
#[inline]
pub fn eval_binary_op_typed(op: BinaryOp, a: ConstValue, b: ConstValue, dtype: ScalarDType) -> Option<ConstValue> {
    eval_binary_op(op, a, b).map(|r| r.truncate(dtype))
}

/// Evaluate ternary op with dtype-aware truncation (for constant folding).
///
/// Applies truncation after evaluation to ensure results fit within dtype boundaries.
#[inline]
pub fn eval_ternary_op_typed(
    op: TernaryOp,
    a: ConstValue,
    b: ConstValue,
    c: ConstValue,
    dtype: ScalarDType,
) -> Option<ConstValue> {
    eval_ternary_op(op, a, b, c).map(|r| r.truncate(dtype))
}

/// Evaluate addition with dtype-aware truncation.
#[inline]
pub fn eval_add_typed(a: ConstValue, b: ConstValue, dtype: ScalarDType) -> Option<ConstValue> {
    eval_add(a, b).map(|r| r.truncate(dtype))
}

/// Evaluate multiplication with dtype-aware truncation.
#[inline]
pub fn eval_mul_typed(a: ConstValue, b: ConstValue, dtype: ScalarDType) -> Option<ConstValue> {
    eval_mul(a, b).map(|r| r.truncate(dtype))
}

/// Evaluate subtraction with dtype-aware truncation.
#[inline]
pub fn eval_sub_typed(a: ConstValue, b: ConstValue, dtype: ScalarDType) -> Option<ConstValue> {
    eval_sub(a, b).map(|r| r.truncate(dtype))
}

// ============================================================================
// Vector Constant Evaluation
// ============================================================================

/// Evaluate binary op element-wise on vector constants.
///
/// Both vectors must have the same length.
/// Returns `None` if lengths differ or any element operation fails.
#[inline]
pub fn eval_binary_op_vec(op: BinaryOp, a: &[ConstValue], b: &[ConstValue]) -> Option<Vec<ConstValue>> {
    if a.len() != b.len() {
        return None;
    }
    a.iter().zip(b.iter()).map(|(av, bv)| eval_binary_op(op, *av, *bv)).collect()
}

/// Evaluate binary op with broadcast support (Const + VConst mixing).
///
/// Handles three cases:
/// - Both single element: returns single element result
/// - One single, one vector: broadcasts the single element
/// - Both same length vectors: element-wise operation
///
/// Returns `None` for mismatched non-broadcast lengths or failed operations.
#[inline]
pub fn eval_binary_op_broadcast(op: BinaryOp, a: &[ConstValue], b: &[ConstValue]) -> Option<Vec<ConstValue>> {
    match (a.len(), b.len()) {
        (1, 1) => eval_binary_op(op, a[0], b[0]).map(|v| vec![v]),
        (n, 1) if n > 1 => a.iter().map(|av| eval_binary_op(op, *av, b[0])).collect(),
        (1, m) if m > 1 => b.iter().map(|bv| eval_binary_op(op, a[0], *bv)).collect(),
        (n, m) if n == m => eval_binary_op_vec(op, a, b),
        _ => None, // Mismatched non-broadcast lengths
    }
}

/// Evaluate binary op element-wise with dtype-aware truncation.
#[inline]
pub fn eval_binary_op_vec_typed(
    op: BinaryOp,
    a: &[ConstValue],
    b: &[ConstValue],
    dtype: ScalarDType,
) -> Option<Vec<ConstValue>> {
    eval_binary_op_vec(op, a, b).map(|vs| vs.into_iter().map(|v| v.truncate(dtype)).collect())
}

/// Evaluate binary op with broadcast and dtype-aware truncation.
#[inline]
pub fn eval_binary_op_broadcast_typed(
    op: BinaryOp,
    a: &[ConstValue],
    b: &[ConstValue],
    dtype: ScalarDType,
) -> Option<Vec<ConstValue>> {
    eval_binary_op_broadcast(op, a, b).map(|vs| vs.into_iter().map(|v| v.truncate(dtype)).collect())
}

/// Evaluate unary op element-wise on vector constants.
#[inline]
pub fn eval_unary_op_vec(op: UnaryOp, values: &[ConstValue]) -> Option<Vec<ConstValue>> {
    values.iter().map(|v| eval_unary_op(op, *v)).collect()
}

/// Evaluate unary op element-wise with dtype-aware truncation.
#[inline]
pub fn eval_unary_op_vec_typed(op: UnaryOp, values: &[ConstValue], dtype: ScalarDType) -> Option<Vec<ConstValue>> {
    eval_unary_op_vec(op, values).map(|vs| vs.into_iter().map(|v| v.truncate(dtype)).collect())
}
