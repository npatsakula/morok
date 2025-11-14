//! Bitwise and shift operations (and, or, xor, not, shl, shr).

use std::rc::Rc;

use super::super::{BinaryOp, Op, Result, UOp, UnaryOp};

/// Macro for bitwise binary operations with type promotion and dtype validation.
///
/// Generates methods that:
/// 1. Promote operands to common type
/// 2. Validate dtype is int or bool
/// 3. Create binary operation node
macro_rules! bitwise_binary_ops {
    ($($method:ident => $op:ident),+ $(,)?) => {
        $(
            #[doc = concat!(" Bitwise ", stringify!($op), " with type promotion and dtype checking.")]
            ///
            /// # Errors
            /// Returns error if type promotion fails, void type is used, or dtype is not int/bool.
            pub fn $method(lhs: Rc<Self>, rhs: Rc<Self>) -> Result<Rc<Self>> {
                let (lhs, rhs, dtype) = Self::promote_and_cast(lhs, rhs)?;
                Self::check_bitwise_dtype(dtype.clone(), stringify!($method))?;
                Ok(Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), dtype))
            }
        )+
    };
}

/// Macro for shift operations that only check LHS dtype.
///
/// Shift operations don't promote types - they preserve the LHS dtype
/// and only validate that it's a valid bitwise type.
macro_rules! shift_ops {
    ($($method:ident => $op:ident),+ $(,)?) => {
        $(
            #[doc = concat!(" ", stringify!($op), " operation (preserves LHS dtype).")]
            ///
            /// # Errors
            /// Returns error if LHS dtype is not int/bool.
            pub fn $method(lhs: Rc<Self>, rhs: Rc<Self>) -> Result<Rc<Self>> {
                let dtype = lhs.dtype();
                Self::check_bitwise_dtype(dtype.clone(), stringify!($method))?;
                Ok(Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), dtype))
            }
        )+
    };
}

impl UOp {
    // Bitwise binary operations
    bitwise_binary_ops! {
        try_and_op => And,
        try_or_op => Or,
        try_xor_op => Xor,
    }

    // Shift operations
    shift_ops! {
        try_shl_op => Shl,
        try_shr_op => Shr,
    }

    /// Bitwise NOT (unary).
    ///
    /// # Errors
    /// Returns error if dtype is not int/bool.
    pub fn try_not_op(operand: Rc<Self>) -> Result<Rc<Self>> {
        let dtype = operand.dtype();
        Self::check_bitwise_dtype(dtype.clone(), "not_op")?;
        Ok(Self::new(Op::Unary(UnaryOp::Not, operand), dtype))
    }
}
