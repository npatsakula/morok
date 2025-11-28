//! Bitwise and shift operations (and, or, xor, not, shl, shr).

use std::rc::Rc;

use super::super::{BinaryOp, Op, Result, UOp};

/// Macro for bitwise binary operations with type promotion and dtype validation.
///
/// Generates methods that:
/// 1. Promote operands to common type
/// 2. Validate dtype is int or bool
/// 3. Create binary operation node
macro_rules! bitwise_binary_ops {
    ($($method:ident => $op:ident, $desc:expr),+ $(,)?) => {
        $(
            #[doc = concat!("Bitwise ", $desc, " operation with type promotion and dtype checking.")]
            ///
            /// # Errors
            /// Returns error if type promotion fails, void type is used, or dtype is not int/bool.
            /// ```rust
            /// # use morok_ir::{UOp, ConstValue, error::Error};
            /// # use morok_dtype::DType;
            /// let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
            /// let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
            #[doc = concat!("let result = a.", stringify!($method), "(&b);")]
            /// assert!(result.is_err(), "Expected error for float dtype");
            /// ```
            /// # Examples
            /// ```rust
            /// # use morok_ir::{UOp, ConstValue, error::Error};
            /// # use morok_dtype::DType;
            /// let a = UOp::const_(DType::Int32, ConstValue::Int(0b1010));
            /// let b = UOp::const_(DType::Int32, ConstValue::Int(0b1100));
            #[doc = concat!("let result = a.", stringify!($method), "(&b)?;")]
            /// assert_eq!(result.dtype(), DType::Int32);
            /// # Ok::<(), Error>(())
            /// ```
            pub fn $method(self: &Rc<Self>, rhs: &Rc<Self>) -> Result<Rc<Self>> {
                let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
                Self::check_bitwise_dtype(dtype.clone(), stringify!($method))?;
                Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::$op)?;
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
    ($($method:ident => $op:ident, $desc:expr),+ $(,)?) => {
        $(
            #[doc = concat!($desc, " operation (preserves LHS dtype).")]
            ///
            /// # Errors
            /// Returns error if LHS dtype is not int/bool.
            /// ```rust
            /// # use morok_ir::{UOp, ConstValue, error::Error};
            /// # use morok_dtype::DType;
            /// let value = UOp::const_(DType::Float32, ConstValue::Float(8.0));
            /// let shift_amount = UOp::const_(DType::Int32, ConstValue::Int(2));
            #[doc = concat!("let result = value.", stringify!($method), "(&shift_amount);")]
            /// assert!(result.is_err(), "Expected error for float LHS")
            /// ```
            /// # Examples
            /// ```rust
            /// # use morok_ir::{UOp, ConstValue, error::Error};
            /// # use morok_dtype::DType;
            /// let value = UOp::const_(DType::Int32, ConstValue::Int(8));
            /// let shift_amount = UOp::const_(DType::Int32, ConstValue::Int(2));
            #[doc = concat!("let result = value.", stringify!($method), "(&shift_amount)?;")]
            /// // Result preserves LHS dtype
            /// assert_eq!(result.dtype(), DType::Int32);
            /// # Ok::<(), Error>(())
            /// ```
            pub fn $method(self: &Rc<Self>, rhs: &Rc<Self>) -> Result<Rc<Self>> {
                let dtype = self.dtype();
                Self::check_bitwise_dtype(dtype.clone(), stringify!($method))?;
                Self::validate_binary_shapes(self, rhs, BinaryOp::$op)?;
                Ok(Self::new(Op::Binary(BinaryOp::$op, self.clone(), rhs.clone()), dtype))
            }
        )+
    };
}

impl UOp {
    // Bitwise binary operations
    bitwise_binary_ops! {
        try_and_op => And, "AND",
        try_or_op => Or, "OR",
        try_xor_op => Xor, "XOR",
    }

    // Shift operations
    shift_ops! {
        try_shl_op => Shl, "Left shift",
        try_shr_op => Shr, "Right shift",
    }
}
