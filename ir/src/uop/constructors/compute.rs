//! Mathematical operations: arithmetic, transcendental, bitwise, comparison.
//!
//! This module contains all computational operations:
//! - Arithmetic: add, sub, mul, div, mod, pow, max, neg, abs, square, sign
//! - Transcendental: sqrt, rsqrt, exp, exp2, log, log2, sin, cos, tan, erf, reciprocal
//! - Rounding: trunc, floor, ceil, round
//! - Bitwise: and, or, xor, shl, shr, not
//! - Comparison: lt, le, eq, ne, gt, ge
//! - Ternary: where, mulacc
//! - Random: threefry
//! - Scalar convenience: add_scalar, sub_scalar, mul_scalar, mod_scalar

use std::sync::Arc;

use morok_dtype::DType;
use snafu::ensure;

use crate::error::InvalidDTypeForUnaryOpSnafu;
use crate::op::Op;
use crate::types::{BinaryOp, TernaryOp, UnaryOp};
use crate::uop::UOp;
use crate::{IntoUOp, Result};

// =========================================================================
// Macro Definitions
// =========================================================================

/// Macro for simple binary arithmetic operations with type promotion.
macro_rules! binary_arith_ops {
    ($($method:ident => $op:ident),+ $(,)?) => {
        $(
            #[track_caller]
            pub fn $method(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
                let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
                Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::$op)?;
                Ok(Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), dtype))
            }
        )+
    };
}

/// Macro for division-like operations that check for division by zero.
macro_rules! division_ops {
    ($($method:ident => $op:ident),+ $(,)?) => {
        $(
            #[track_caller]
            pub fn $method(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
                Self::check_division_by_zero(rhs)?;
                let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
                Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::$op)?;
                Ok(Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), dtype))
            }
        )+
    };
}

/// Macro for bitwise binary operations with type promotion and dtype validation.
macro_rules! bitwise_binary_ops {
    ($($method:ident => $op:ident),+ $(,)?) => {
        $(
            pub fn $method(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
                let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
                Self::check_bitwise_dtype(dtype.clone(), BinaryOp::$op)?;
                Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::$op)?;
                Ok(Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), dtype))
            }
        )+
    };
}

/// Macro for shift operations that only check LHS dtype.
macro_rules! shift_ops {
    ($($method:ident => $op:ident),+ $(,)?) => {
        $(
            pub fn $method(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
                let dtype = self.dtype();
                Self::check_bitwise_dtype(dtype.clone(), BinaryOp::$op)?;
                Self::validate_binary_shapes(self, rhs, BinaryOp::$op)?;
                Ok(Self::new(Op::Binary(BinaryOp::$op, self.clone(), rhs.clone()), dtype))
            }
        )+
    };
}

/// Macro for comparison operations.
/// Preserves vectorization: <N x T> cmp <N x T> → <N x bool>
macro_rules! cmp_ops {
    ($($method:ident => $op:ident),+ $(,)?) => {
        $(
            #[track_caller]
            pub fn $method(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
                // Use type promotion to validate types and find common type
                let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
                Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::$op)?;
                // Preserve vectorization: <N x T> cmp <N x T> → <N x bool>
                let vcount = dtype.vcount();
                let result_dtype = if vcount > 1 { DType::Bool.vec(vcount) } else { DType::Bool };
                Ok(Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), result_dtype))
            }
        )+
    };
}

/// Macro for transcendental functions that require float dtype.
macro_rules! transcendental_ops {
    ($($method:ident => $op:ident),+ $(,)?) => {
        $(
            #[track_caller]
            pub fn $method(self: &Arc<Self>) -> Result<Arc<Self>> {
                let dtype = self.dtype();
                ensure!(dtype.is_float(), InvalidDTypeForUnaryOpSnafu { operation: UnaryOp::$op, dtype });
                Ok(Self::new(Op::Unary(UnaryOp::$op, self.clone()), dtype))
            }
        )+
    };
}

/// Macro for scalar convenience wrappers.
macro_rules! scalar_ops {
    ($($method:ident => $op_method:ident),+ $(,)?) => {
        $(
            pub fn $method<T: IntoUOp>(lhs: Arc<Self>, rhs: T) -> Result<Arc<Self>> {
                let rhs_uop = rhs.into_uop(lhs.dtype());
                lhs.$op_method(&rhs_uop)
            }
        )+
    };
}

impl UOp {
    // =========================================================================
    // Arithmetic Operations
    // =========================================================================

    binary_arith_ops! {
        try_add => Add,
        try_sub => Sub,
        try_mul => Mul,
    }

    division_ops! {
        try_mod => Mod,
    }

    /// Division with automatic type-based operator selection.
    ///
    /// Uses Idiv for integer types and Fdiv for float types.
    #[track_caller]
    pub fn try_div(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
        Self::check_division_by_zero(rhs)?;
        let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;

        // Choose division operator based on dtype
        let op = if dtype.is_float() { BinaryOp::Fdiv } else { BinaryOp::Idiv };

        Self::validate_binary_shapes(&lhs, &rhs, op)?;
        Ok(Self::new(Op::Binary(op, lhs, rhs), dtype))
    }

    /// Maximum of two values: max(a, b).
    pub fn try_max(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
        let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
        Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::Max)?;
        Ok(Self::new(Op::Binary(BinaryOp::Max, lhs, rhs), dtype))
    }

    /// Power: a^b.
    pub fn try_pow(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
        let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
        Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::Pow)?;
        Ok(Self::new(Op::Binary(BinaryOp::Pow, lhs, rhs), dtype))
    }

    /// Negation: -x.
    #[track_caller]
    pub fn neg(self: &Arc<Self>) -> Arc<Self> {
        let dtype = self.dtype.clone();
        Self::new(Op::Unary(UnaryOp::Neg, self.clone()), dtype)
    }

    /// Absolute value: |x|.
    #[track_caller]
    pub fn abs(self: &Arc<Self>) -> Arc<Self> {
        let dtype = self.dtype.clone();
        Self::new(Op::Unary(UnaryOp::Abs, self.clone()), dtype)
    }

    /// Square: x².
    #[track_caller]
    pub fn square(operand: Arc<Self>) -> Arc<Self> {
        let dtype = operand.dtype();
        Self::new(Op::Unary(UnaryOp::Square, operand), dtype)
    }

    /// Sign: -1 for negative, 0 for zero, 1 for positive.
    pub fn sign(operand: Arc<Self>) -> Arc<Self> {
        let dtype = operand.dtype();
        Self::new(Op::Unary(UnaryOp::Sign, operand), dtype)
    }

    // =========================================================================
    // Scalar Convenience Methods
    // =========================================================================

    scalar_ops! {
        try_add_scalar => try_add,
        try_sub_scalar => try_sub,
        try_mul_scalar => try_mul,
        try_mod_scalar => try_mod,
    }

    // =========================================================================
    // Transcendental Operations
    // =========================================================================

    transcendental_ops! {
        try_sqrt => Sqrt,
        try_rsqrt => Rsqrt,
        try_exp => Exp,
        try_exp2 => Exp2,
        try_log => Log,
        try_log2 => Log2,
        try_sin => Sin,
        try_cos => Cos,
        try_tan => Tan,
    }

    /// Error function: erf(x) - requires float dtype.
    #[track_caller]
    pub fn erf(operand: Arc<Self>) -> Result<Arc<Self>> {
        let dtype = operand.dtype();
        ensure!(dtype.is_float(), InvalidDTypeForUnaryOpSnafu { operation: UnaryOp::Erf, dtype });
        Ok(Self::new(Op::Unary(UnaryOp::Erf, operand), dtype))
    }

    /// Reciprocal: 1/x - requires float dtype.
    #[track_caller]
    pub fn try_reciprocal(operand: &Arc<Self>) -> Result<Arc<Self>> {
        let dtype = operand.dtype();
        ensure!(dtype.is_float(), InvalidDTypeForUnaryOpSnafu { operation: UnaryOp::Reciprocal, dtype });
        Ok(Self::new(Op::Unary(UnaryOp::Reciprocal, operand.clone()), dtype))
    }

    // =========================================================================
    // Rounding Operations
    // =========================================================================

    /// Truncate towards zero.
    #[track_caller]
    pub fn trunc(operand: Arc<Self>) -> Arc<Self> {
        let dtype = operand.dtype();
        Self::new(Op::Unary(UnaryOp::Trunc, operand), dtype)
    }

    /// Floor: round towards -∞.
    #[track_caller]
    pub fn floor(operand: Arc<Self>) -> Arc<Self> {
        let dtype = operand.dtype();
        Self::new(Op::Unary(UnaryOp::Floor, operand), dtype)
    }

    /// Ceiling: round towards +∞.
    #[track_caller]
    pub fn ceil(operand: Arc<Self>) -> Arc<Self> {
        let dtype = operand.dtype();
        Self::new(Op::Unary(UnaryOp::Ceil, operand), dtype)
    }

    /// Round: round to nearest integer (half to even).
    pub fn round(operand: Arc<Self>) -> Arc<Self> {
        let dtype = operand.dtype();
        Self::new(Op::Unary(UnaryOp::Round, operand), dtype)
    }

    // =========================================================================
    // Bitwise Operations
    // =========================================================================

    bitwise_binary_ops! {
        try_and_op => And,
        try_or_op => Or,
        try_xor_op => Xor,
    }

    shift_ops! {
        try_shl_op => Shl,
        try_shr_op => Shr,
    }

    /// Logical not: !x.
    #[track_caller]
    pub fn not(self: &Arc<Self>) -> Arc<Self> {
        let dtype = self.dtype.clone();
        Self::new(Op::Unary(UnaryOp::Not, self.clone()), dtype)
    }

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    cmp_ops! {
        try_cmplt => Lt,
        try_cmple => Le,
        try_cmpeq => Eq,
        try_cmpne => Ne,
        try_cmpgt => Gt,
        try_cmpge => Ge,
    }

    // =========================================================================
    // Ternary Operations
    // =========================================================================

    /// Conditional selection: condition ? true_val : false_val.
    #[track_caller]
    pub fn try_where(condition: Arc<Self>, true_val: Arc<Self>, false_val: Arc<Self>) -> Result<Arc<Self>> {
        let dtype = true_val.dtype(); // Result has same dtype as branches
        Self::validate_ternary_shapes(&true_val, &false_val)?;
        Ok(Self::new(Op::Ternary(TernaryOp::Where, condition, true_val, false_val), dtype))
    }

    /// Multiply-accumulate: a * b + c (fused operation).
    pub fn try_mulacc(a: Arc<Self>, b: Arc<Self>, c: Arc<Self>) -> Result<Arc<Self>> {
        let dtype = a.dtype(); // Preserve first operand dtype
        // Validate all three operands have matching shapes
        Self::validate_ternary_shapes(&a, &b)?;
        Self::validate_ternary_shapes(&a, &c)?;
        Ok(Self::new(Op::Ternary(TernaryOp::MulAcc, a, b, c), dtype))
    }

    // =========================================================================
    // Random Operations
    // =========================================================================

    /// Threefry PRNG: threefry(x, key).
    pub fn threefry(lhs: Arc<Self>, rhs: Arc<Self>) -> Result<Arc<Self>> {
        let dtype = DType::UInt64; // Threefry always returns uint64
        Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::Threefry)?;
        Ok(Self::new(Op::Binary(BinaryOp::Threefry, lhs, rhs), dtype))
    }
}
