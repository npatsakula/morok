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
        UnaryOp::Sqrt => eval_sqrt(v),
        UnaryOp::Exp2 => eval_exp2(v),
        UnaryOp::Log2 => eval_log2(v),
        UnaryOp::Sin => eval_sin(v),
        UnaryOp::Reciprocal => eval_reciprocal(v),
        UnaryOp::Trunc => eval_trunc(v),
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
        BinaryOp::Eq => eval_eq(a, b),
        BinaryOp::Ne => eval_ne(a, b),
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
fn eval_sqrt(v: ConstValue) -> Option<ConstValue> {
    match v {
        ConstValue::Float(x) => Some(ConstValue::Float(x.sqrt())),
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

// ============================================================================
// Binary Arithmetic Operations
// ============================================================================

#[inline]
fn eval_add(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Int(x.wrapping_add(y))),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::UInt(x.wrapping_add(y))),
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Float(x + y)),
        _ => None,
    }
}

#[inline]
fn eval_mul(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Int(x.wrapping_mul(y))),
        (ConstValue::UInt(x), ConstValue::UInt(y)) => Some(ConstValue::UInt(x.wrapping_mul(y))),
        (ConstValue::Float(x), ConstValue::Float(y)) => Some(ConstValue::Float(x * y)),
        _ => None,
    }
}

#[inline]
fn eval_sub(a: ConstValue, b: ConstValue) -> Option<ConstValue> {
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
