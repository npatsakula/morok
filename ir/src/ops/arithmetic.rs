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
    ($($method:ident => $op:ident, $desc:expr),+ $(,)?) => {
        $(
            #[doc = concat!($desc, " of two UOps with automatic type promotion.")]
            ///
            /// # Errors
            /// Returns error if type promotion fails or void type is used.
            /// ```rust
            /// # use morok_ir::{UOp, ConstValue, error::Error};
            /// # use morok_dtype::DType;
            /// let a = UOp::const_(DType::Int32, ConstValue::Int(5));
            /// let b = UOp::const_(DType::Void, ConstValue::Int(0));
            #[doc = concat!("let result = UOp::", stringify!($method), "(a, b);")]
            /// assert!(result.is_err(), "Expected error for void type");
            /// ```
            /// # Examples
            /// Basic usage:
            /// ```rust
            /// # use morok_ir::{UOp, ConstValue, error::Error};
            /// # use morok_dtype::DType;
            /// let a = UOp::const_(DType::Float32, ConstValue::Float(5.0));
            /// let b = UOp::const_(DType::Float32, ConstValue::Float(3.0));
            #[doc = concat!("let result = UOp::", stringify!($method), "(a, b)?;")]
            /// assert_eq!(result.dtype(), DType::Float32);
            /// # Ok::<(), Error>(())
            /// ```
            /// Type promotion:
            /// ```rust
            /// # use morok_ir::{UOp, ConstValue, error::Error};
            /// # use morok_dtype::DType;
            /// // Int32 + Float32 automatically promotes to Float32
            /// let int_val = UOp::const_(DType::Int32, ConstValue::Int(5));
            /// let float_val = UOp::const_(DType::Float32, ConstValue::Float(3.0));
            #[doc = concat!("let result = UOp::", stringify!($method), "(int_val, float_val)?;")]
            /// assert_eq!(result.dtype(), DType::Float32);
            /// # Ok::<(), Error>(())
            /// ```
            pub fn $method(self: &Rc<Self>, rhs: &Rc<Self>) -> Result<Rc<Self>> {
                let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
                Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::$op)?;
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
    ($($method:ident => $op:ident, $desc:expr),+ $(,)?) => {
        $(
            #[doc = concat!($desc, " of two UOps with automatic type promotion and zero check.")]
            ///
            /// # Errors
            /// Returns error if type promotion fails, void type is used, or division by zero is detected.
            /// ```rust
            /// # use morok_ir::{UOp, ConstValue, error::Error};
            /// # use morok_dtype::DType;
            /// let a = UOp::const_(DType::Float32, ConstValue::Float(10.0));
            /// let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
            #[doc = concat!("let result = UOp::", stringify!($method), "(a, zero);")]
            /// assert!(result.is_err(), "Expected error for division by zero");
            /// ```
            /// # Examples
            /// ```rust
            /// # use morok_ir::{UOp, ConstValue, error::Error};
            /// # use morok_dtype::DType;
            /// let a = UOp::const_(DType::Float32, ConstValue::Float(10.0));
            /// let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
            #[doc = concat!("let result = UOp::", stringify!($method), "(a, b)?;")]
            /// assert_eq!(result.dtype(), DType::Float32);
            /// # Ok::<(), Error>(())
            /// ```
            pub fn $method(self: &Rc<Self>, rhs: &Rc<Self>) -> Result<Rc<Self>> {
                Self::check_division_by_zero(rhs)?;
                let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
                Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::$op)?;
                Ok(Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), dtype))
            }
        )+
    };
}

impl UOp {
    // Simple arithmetic operations
    binary_arith_ops! {
        try_add_op => Add, "Addition",
        try_sub_op => Sub, "Subtraction",
        try_mul_op => Mul, "Multiplication",
    }

    // Division operations with zero check
    division_ops! {
        try_mod_op => Mod, "Modulo",
    }

    /// Negate a UOp (unary minus).
    ///
    /// Preserves the dtype of the operand.
    ///
    /// # Examples
    /// ```rust
    /// # use morok_ir::{UOp, ConstValue};
    /// # use morok_dtype::DType;
    /// let val = UOp::const_(DType::Float32, ConstValue::Float(5.0));
    /// let result = UOp::neg_op(val);
    /// assert_eq!(result.dtype(), DType::Float32);
    /// ```
    pub fn neg_op(operand: Rc<Self>) -> Rc<Self> {
        let dtype = operand.dtype();
        Self::new(Op::Unary(UnaryOp::Neg, operand), dtype)
    }
}
