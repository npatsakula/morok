use super::*;

/// Unified macro for implementing Tensor operations.
///
/// Automatically handles:
/// - Binary operations (with `other` parameter): Always use Result path
/// - Unary operations with `@infallible` marker: Wrap in Ok()
/// - Unary operations without marker: Use Result path
macro_rules! impl_tensor_ops {
    (
        binary { $($bin_method:ident => $bin_uop:ident),* $(,)? }
        unary_infallible { $($inf_method:ident => $inf_uop:ident),* $(,)? }
        unary_fallible { $($fall_method:ident => $fall_uop:ident),* $(,)? }
    ) => {
        // Binary operations (with automatic broadcasting)
        $(
            #[track_caller]
            pub fn $bin_method(&self, other: &Tensor) -> Result<Tensor> {
                // Broadcast tensors to common shape
                let (lhs, rhs) = self.broadcast_for_binop(other)?;

                // Now call UOp operation with matching shapes
                lhs.uop.$bin_uop(&rhs.uop).map(Self::new).context(UOpSnafu)
            }
        )*

        // Unary infallible operations
        $(
            #[track_caller]
            pub fn $inf_method(&self) -> Result<Tensor> {
                Ok(Self::new(self.uop.$inf_uop()))
            }
        )*

        // Unary fallible operations
        $(
            #[track_caller]
            pub fn $fall_method(&self) -> Result<Tensor> {
                self.uop.$fall_uop().map(Self::new).context(UOpSnafu)
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
            try_pow => try_pow,
            try_eq => try_cmpeq,
            try_ne => try_cmpne,
            try_lt => try_cmplt,
            try_le => try_cmple,
            try_gt => try_cmpgt,
            try_ge => try_cmpge,
        }
        unary_infallible {
            try_neg => neg,
            try_abs => abs,
        }
        unary_fallible {
            try_sqrt => try_sqrt,
            try_rsqrt => try_rsqrt,
            try_exp => try_exp,
            try_log => try_log,
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
    pub fn logical_not(&self) -> Result<Tensor> {
        use morok_dtype::DType;

        // Cast to bool (non-zero becomes true)
        let as_bool = self.cast(DType::Bool)?;

        // Create true constant tensor and broadcast to match shape
        let true_scalar = Self::from_slice([true]);
        let self_shape = as_bool.shape()?;

        let true_broadcast = if self_shape.is_empty() {
            // Input is scalar - reshape [1] to []
            true_scalar.try_reshape(&[])?
        } else {
            // Broadcast to match non-scalar shape
            true_scalar.broadcast_to(&self_shape)?
        };

        // Compare: !x ≡ (x != true)
        as_bool.try_ne(&true_broadcast)
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
    pub fn bitwise_not(&self) -> Result<Tensor> {
        // Verify dtype is integer
        let dtype = self.uop.dtype();
        if !dtype.is_int() {
            return Err(Error::SymbolicShapeUnsupported {
                operation: format!("bitwise_not on non-integer dtype {:?}", dtype),
            });
        }

        // Bitwise NOT using two's complement: ~x = -x - 1
        let negated = self.try_neg()?;
        let one_scalar = Self::from_slice([1i32]).cast(dtype)?;

        // Broadcast one to match self shape
        let self_shape = self.shape()?;

        let one_broadcast = if self_shape.is_empty() {
            // Input is scalar - reshape [1] to []
            one_scalar.try_reshape(&[])?
        } else {
            // Broadcast to match non-scalar shape
            one_scalar.broadcast_to(&self_shape)?
        };

        negated.try_sub(&one_broadcast)
    }
}
