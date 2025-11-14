//! Arithmetic operations (add, sub, mul, div, rem, neg).

use std::rc::Rc;

use super::super::{BinaryOp, Op, Result, UOp, UnaryOp};

/// Macro for simple binary arithmetic operations with type promotion.
///
/// Generates methods that:
/// 1. Promote operands to common type
/// 2. Create binary operation node
/// 3. Return result with promoted dtype
macro_rules! binary_arith_ops {
    ($($method:ident => $op:ident),+ $(,)?) => {
        $(
            #[doc = concat!(" ", stringify!($op), " two UOps with automatic type promotion.")]
            ///
            /// # Errors
            /// Returns error if type promotion fails or void type is used.
            pub fn $method(lhs: Rc<Self>, rhs: Rc<Self>) -> Result<Rc<Self>> {
                let (lhs, rhs, dtype) = Self::promote_and_cast(lhs, rhs)?;
                Ok(Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), dtype))
            }
        )+
    };
}

/// Macro for division-like operations (div, rem) that check for division by zero.
///
/// Generates methods that:
/// 1. Check divisor is not constant zero
/// 2. Promote operands to common type
/// 3. Create binary operation node
/// 4. Return result with promoted dtype
macro_rules! division_ops {
    ($($method:ident => $op:ident),+ $(,)?) => {
        $(
            #[doc = concat!(" ", stringify!($op), " two UOps with automatic type promotion and zero check.")]
            ///
            /// # Errors
            /// Returns error if type promotion fails, void type is used, or division by zero is detected.
            pub fn $method(lhs: Rc<Self>, rhs: Rc<Self>) -> Result<Rc<Self>> {
                Self::check_division_by_zero(&rhs)?;
                let (lhs, rhs, dtype) = Self::promote_and_cast(lhs, rhs)?;
                Ok(Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), dtype))
            }
        )+
    };
}

impl UOp {
    // Simple arithmetic operations
    binary_arith_ops! {
        try_add_op => Add,
        try_sub_op => Sub,
        try_mul_op => Mul,
    }

    // Division operations with zero check
    division_ops! {
        try_div_op => Div,
        try_rem_op => Rem,
    }

    /// Negate a UOp (unary minus).
    ///
    /// Preserves the dtype of the operand.
    pub fn neg_op(operand: Rc<Self>) -> Rc<Self> {
        let dtype = operand.dtype();
        Self::new(Op::Unary(UnaryOp::Neg, operand), dtype)
    }
}
