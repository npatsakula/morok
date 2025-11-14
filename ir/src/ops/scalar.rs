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
    ($($method:ident => $op_method:ident),+ $(,)?) => {
        $(
            #[doc = concat!(" ", stringify!($op_method), " with a scalar value.")]
            ///
            /// Convenience method that converts the scalar to a UOp and performs the operation.
            ///
            /// # Errors
            /// Returns error if type promotion fails or void type is used.
            pub fn $method<T: IntoUOp>(lhs: Rc<Self>, rhs: T) -> Result<Rc<Self>> {
                let rhs_uop = rhs.into_uop(lhs.dtype());
                Self::$op_method(lhs, rhs_uop)
            }
        )+
    };
}

impl UOp {
    scalar_ops! {
        try_add_scalar => try_add_op,
        try_sub_scalar => try_sub_op,
        try_mul_scalar => try_mul_op,
        try_div_scalar => try_div_op,
        try_rem_scalar => try_rem_op,
    }
}
