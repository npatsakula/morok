//! Scalar convenience methods for arithmetic operations.
//!
//! These methods allow users to perform operations with scalar values
//! directly without manually converting them to UOps first.

use std::rc::Rc;

use super::super::{IntoUOp, Result, UOp};

/// Macro for scalar convenience wrappers.
///
/// Generates methods that:
/// 1. Convert scalar to UOp using IntoUOp trait
/// 2. Delegate to the corresponding operation method
macro_rules! scalar_ops {
    ($($method:ident => $op_method:ident, $desc:expr),+ $(,)?) => {
        $(
            #[doc = concat!($desc, " with a scalar value.")]
            ///
            /// Convenience method that converts the scalar to a UOp and performs the operation.
            ///
            /// # Errors
            /// Returns error if type promotion fails or void type is used.
            /// # Examples
            /// ```rust
            /// # use morok_ir::{UOp, ConstValue, error::Error};
            /// # use morok_dtype::DType;
            /// let val = UOp::const_(DType::Float32, ConstValue::Float(5.0));
            /// // Add scalar directly without manual conversion
            #[doc = concat!("let result = UOp::", stringify!($method), "(val, 3.0)?;")]
            /// assert_eq!(result.dtype(), DType::Float32);
            /// # Ok::<(), Error>(())
            /// ```
            pub fn $method<T: IntoUOp>(lhs: Rc<Self>, rhs: T) -> Result<Rc<Self>> {
                let rhs_uop = rhs.into_uop(lhs.dtype());
                Self::$op_method(lhs, rhs_uop)
            }
        )+
    };
}

impl UOp {
    scalar_ops! {
        try_add_scalar => try_add_op, "Addition",
        try_sub_scalar => try_sub_op, "Subtraction",
        try_mul_scalar => try_mul_op, "Multiplication",
        try_mod_scalar => try_mod_op, "Modulo",
    }
}
