use std::panic::Location;

use snafu::ResultExt;
use svod_ir::origin::OriginScope;

use super::*;

/// The op name behind a `try_*` entry point, resolved at compile time so a call
/// frame reads `add` rather than `try_add`.
const fn op_name(name: &'static str) -> &'static str {
    match name.as_bytes() {
        [b't', b'r', b'y', b'_', rest @ ..] => match std::str::from_utf8(rest) {
            Ok(rest) => rest,
            Err(_) => name,
        },
        _ => name,
    }
}

/// Unified macro for implementing Tensor operations.
///
/// Automatically handles:
/// - Binary operations: scalar-or-tensor rhs, broadcasting, `Result` path
/// - `unary_infallible`: plain `Tensor` return
/// - `unary_fallible`: `Result` path
macro_rules! impl_tensor_ops {
    (
        binary { $($bin_method:ident => $bin_uop:ident),* $(,)? }
        unary_infallible { $($inf_method:ident => $inf_uop:ident),* $(,)? }
        unary_fallible { $($fall_method:ident => $fall_uop:ident),* $(,)? }
    ) => {
        // Binary operations (scalar or tensor rhs, with automatic broadcasting)
        $(
            #[track_caller]
            pub fn $bin_method<'o>(&self, other: impl Into<Operand<'o>>) -> Result<Tensor> {
                let _origin = OriginScope::outer_call(op_name(stringify!($bin_method)), Location::caller());

                // Materialize a scalar rhs in self's dtype, then broadcast to a common shape
                let other = self.operand(other);
                let (lhs, rhs) = self.broadcast_for_binop(&other)?;

                // Now call UOp operation with matching shapes
                lhs.uop().$bin_uop(&rhs.uop()).map(Self::new).context(UOpSnafu).map_err(Into::into)
            }
        )*

        // Unary infallible operations
        $(
            #[track_caller]
            pub fn $inf_method(&self) -> Tensor {
                let _origin = OriginScope::outer_call(op_name(stringify!($inf_method)), Location::caller());
                Self::new(self.uop().$inf_uop())
            }
        )*

        // Unary fallible operations
        $(
            #[track_caller]
            pub fn $fall_method(&self) -> Result<Tensor> {
                let _origin = OriginScope::outer_call(op_name(stringify!($fall_method)), Location::caller());
                self.uop().$fall_uop().map(Self::new).context(UOpSnafu).map_err(Into::into)
            }
        )*
    };
}

impl Tensor {
    impl_tensor_ops! {
        binary {
            try_add => try_add,
            try_sub => try_sub,
            try_mul => try_mul,
            try_div => try_div,
            try_cdiv => try_cdiv,
            try_mod => try_mod,
            try_cmod => try_cmod,
            try_pow => try_pow,
            try_eq => try_cmpeq,
            try_ne => try_cmpne,
            try_lt => try_cmplt,
            try_le => try_cmple,
            try_gt => try_cmpgt,
            try_ge => try_cmpge,
            try_bitor => try_or_op,
            try_bitand => try_and_op,
            try_bitxor => try_xor_op,
            try_shl => try_shl_op,
            try_shr => try_shr_op,
        }
        unary_infallible {
            neg => neg,
            abs => abs,
        }
        unary_fallible {
            try_sqrt => try_sqrt,
            try_rsqrt => try_rsqrt,
            try_exp => try_exp,
            try_exp2 => try_exp2,
            try_log => try_log,
            try_log2 => try_log2,
        }
    }

    /// Logical NOT for boolean tensors.
    ///
    /// Converts to boolean dtype and applies logical negation.
    /// For non-boolean tensors, treats zero as false, non-zero as true.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[true, false, true]);
    /// let result = t.logical_not()?;  // [false, true, false]
    ///
    /// let nums = Tensor::from_slice(&[0.0f32, 1.0, 2.0]);
    /// let result = nums.logical_not()?;  // [true, false, false]
    /// ```
    #[track_caller]
    pub fn logical_not(&self) -> Result<Tensor> {
        origin_call!("logical_not");
        // !x ≡ (x != true), with the constant broadcast by the binop itself.
        self.cast(svod_dtype::DType::Bool).try_ne(true)
    }

    /// Bitwise NOT for integer tensors.
    ///
    /// Applies bitwise NOT operation using two's complement: `~x = -x - 1`.
    /// Only works for integer dtypes.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[0i32, 1, 2, -1]);
    /// let result = t.bitwise_not()?;  // [-1, -2, -3, 0]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if called on non-integer dtype.
    #[track_caller]
    pub fn bitwise_not(&self) -> Result<Tensor> {
        origin_call!("bitwise_not");
        let dtype = self.uop().dtype();
        snafu::ensure!(
            dtype.is_int(),
            SymbolicShapeUnsupportedSnafu { operation: format!("bitwise_not on non-integer dtype {dtype:?}") }
        );
        // Two's complement: ~x = -x - 1.
        self.neg().try_sub(1)
    }
}
