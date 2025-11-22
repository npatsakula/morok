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
        // Binary operations
        $(
            #[track_caller]
            pub fn $bin_method(&self, other: &Tensor) -> Result<Tensor> {
                self.uop.$bin_uop(&other.uop).map(Self::new).context(UOpSnafu)
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
            try_add => try_add_op,
            try_sub => try_sub_op,
            try_mul => try_mul_op,
            try_div => try_div_op,
            try_pow => try_pow_op,
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

        let true_broadcast = if !self_shape.is_empty() {
            let shape_vec: Vec<isize> = self_shape.iter().map(|s| s.as_const().unwrap() as isize).collect();
            let ones_shape = vec![1isize; self_shape.len()];
            let true_reshaped = true_scalar.try_reshape(&ones_shape)?;
            true_reshaped.try_expand(&shape_vec)?
        } else {
            true_scalar.try_reshape(&[])?
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
        let one_broadcast = if !self_shape.is_empty() {
            let shape_vec: Vec<isize> = self_shape.iter().map(|s| s.as_const().unwrap() as isize).collect();
            let ones_shape = vec![1isize; self_shape.len()];
            let one_reshaped = one_scalar.try_reshape(&ones_shape)?;
            one_reshaped.try_expand(&shape_vec)?
        } else {
            one_scalar.try_reshape(&[])?
        };

        negated.try_sub(&one_broadcast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_ir::Op;

    #[test]
    fn simple() {
        let tensor = Tensor::from_slice([1i32, 2, 3]);
        assert_eq!(tensor.buffer().unwrap().size(), 3 * 4);
    }

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = &a + &b;

        // Verify it created an Add operation
        if let Op::Binary(op, _, _) = c.uop.op() {
            assert_eq!(format!("{:?}", op), "Add");
        } else {
            panic!("Expected Binary Add operation");
        }

        // Verify shape is preserved
        assert_eq!(c.uop.shape().unwrap().as_ref().map(|s| s.len()), Some(1));
    }

    #[test]
    fn test_mul_same_shape() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = &a * &b;

        // Verify it created a Mul operation
        if let Op::Binary(op, _, _) = c.uop.op() {
            assert_eq!(format!("{:?}", op), "Mul");
        } else {
            panic!("Expected Binary Mul operation");
        }

        // Verify shape is preserved
        assert_eq!(c.uop.shape().unwrap().as_ref().map(|s| s.len()), Some(1));
    }

    #[test]
    fn test_add_type_promotion() {
        let a = Tensor::from_slice([1i32, 2, 3]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = &a + &b;

        // Result should be promoted to Float32
        assert_eq!(c.uop.dtype(), morok_dtype::DType::Float32);
    }

    #[test]
    fn test_mul_type_promotion() {
        let a = Tensor::from_slice([1i32, 2, 3]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = &a * &b;

        // Result should be promoted to Float32
        assert_eq!(c.uop.dtype(), morok_dtype::DType::Float32);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0]);

        let result = a.try_add(&b);
        assert!(result.is_err());

        if let Err(Error::UOp { source: morok_ir::Error::BinaryShapeMismatch { op: _, lhs, rhs } }) = result {
            assert_eq!(lhs.len(), 1);
            assert_eq!(rhs.len(), 1);
        } else {
            panic!("Expected BinaryShapeMismatch error");
        }
    }

    #[test]
    fn test_operator_variants_add() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);

        // Test all 4 variants
        let _c1 = &a + &b; // &Tensor + &Tensor
        let _c2 = a.clone() + b.clone(); // Tensor + Tensor (need clone since Tensor doesn't impl Copy)
        let _c3 = &a + b.clone(); // &Tensor + Tensor
        let _c4 = a.clone() + &b; // Tensor + &Tensor

        // All should succeed
    }

    #[test]
    fn test_operator_variants_mul() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);

        // Test all 4 variants
        let _c1 = &a * &b; // &Tensor + &Tensor
        let _c2 = a.clone() * b.clone(); // Tensor * Tensor
        let _c3 = &a * b.clone(); // &Tensor * Tensor
        let _c4 = a.clone() * &b; // Tensor * &Tensor

        // All should succeed
    }

    #[test]
    fn test_chained_operations() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = Tensor::from_slice([7.0f32, 8.0, 9.0]);

        // Test (a + b) * c
        let result = (&a + &b) * &c;

        // Verify it creates the correct UOp graph
        if let Op::Binary(op, _, _) = result.uop.op() {
            assert_eq!(format!("{:?}", op), "Mul");
        } else {
            panic!("Expected Binary Mul operation at top level");
        }
    }

    #[test]
    fn test_lazy_evaluation() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);

        // Perform operation
        let c = &a + &b;

        // Result should NOT have a buffer (lazy evaluation)
        assert!(c.buffer().is_none());

        // Only input tensors should have buffers
        assert!(a.buffer().is_some());
        assert!(b.buffer().is_some());
    }

    #[test]
    fn test_sub_same_shape() {
        let a = Tensor::from_slice([5.0f32, 6.0, 7.0]);
        let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let c = &a - &b;

        if let Op::Binary(op, _, _) = c.uop.op() {
            assert_eq!(format!("{:?}", op), "Sub");
        } else {
            panic!("Expected Binary Sub operation");
        }
    }

    #[test]
    fn test_div_same_shape() {
        let a = Tensor::from_slice([10.0f32, 20.0, 30.0]);
        let b = Tensor::from_slice([2.0f32, 4.0, 5.0]);
        let c = &a / &b;

        if let Op::Binary(op, _, _) = c.uop.op() {
            assert_eq!(format!("{:?}", op), "Fdiv");
        } else {
            panic!("Expected Binary Fdiv operation");
        }
    }

    #[test]
    fn test_pow_same_shape() {
        let a = Tensor::from_slice([2.0f32, 3.0, 4.0]);
        let b = Tensor::from_slice([2.0f32, 2.0, 2.0]);
        let c = a.try_pow(&b).unwrap();

        if let Op::Binary(op, _, _) = c.uop.op() {
            assert_eq!(format!("{:?}", op), "Pow");
        } else {
            panic!("Expected Binary Pow operation");
        }
    }

    #[test]
    fn test_operator_variants_sub() {
        let a = Tensor::from_slice([5.0f32, 6.0, 7.0]);
        let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);

        let _c1 = &a - &b;
        let _c2 = a.clone() - b.clone();
        let _c3 = &a - b.clone();
        let _c4 = a.clone() - &b;
    }

    #[test]
    fn test_operator_variants_div() {
        let a = Tensor::from_slice([10.0f32, 20.0, 30.0]);
        let b = Tensor::from_slice([2.0f32, 4.0, 5.0]);

        let _c1 = &a / &b;
        let _c2 = a.clone() / b.clone();
        let _c3 = &a / b.clone();
        let _c4 = a.clone() / &b;
    }

    #[test]
    fn test_eq_comparison() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let c = a.try_eq(&b).unwrap();

        assert_eq!(c.uop.dtype(), morok_dtype::DType::Bool);
    }

    #[test]
    fn test_ne_comparison() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([1.0f32, 2.0, 4.0]);
        let c = a.try_ne(&b).unwrap();

        assert_eq!(c.uop.dtype(), morok_dtype::DType::Bool);
    }

    #[test]
    fn test_lt_comparison() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([2.0f32, 3.0, 4.0]);
        let c = a.try_lt(&b).unwrap();

        assert_eq!(c.uop.dtype(), morok_dtype::DType::Bool);
    }

    #[test]
    fn test_le_comparison() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let c = a.try_le(&b).unwrap();

        assert_eq!(c.uop.dtype(), morok_dtype::DType::Bool);
    }

    #[test]
    fn test_gt_comparison() {
        let a = Tensor::from_slice([2.0f32, 3.0, 4.0]);
        let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let c = a.try_gt(&b).unwrap();

        assert_eq!(c.uop.dtype(), morok_dtype::DType::Bool);
    }

    #[test]
    fn test_ge_comparison() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let c = a.try_ge(&b).unwrap();

        assert_eq!(c.uop.dtype(), morok_dtype::DType::Bool);
    }

    #[test]
    fn test_provenance_tracking() {
        use morok_dtype::DType;
        use morok_ir::{ConstValue, UOp, provenance::PROVENANCE_TRACKER};

        // Clear any existing provenance
        PROVENANCE_TRACKER.with(|t| t.borrow_mut().clear());

        // Test at UOp level first to verify #[track_caller] works
        let uop_a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let uop_b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
        let uop_c = uop_a.try_add_op(&uop_b).unwrap(); // Line 248: Track this

        PROVENANCE_TRACKER.with(|tracker| {
            let t = tracker.borrow();

            eprintln!("\n=== UOp Level Provenance ===");
            let events = t.get_events(uop_c.id);
            assert!(events.is_some(), "Expected provenance for UOp");

            for (i, event) in events.unwrap().iter().enumerate() {
                eprintln!("  [{}] {}", i, event);
            }
        });

        // Now test at Tensor level
        PROVENANCE_TRACKER.with(|t| t.borrow_mut().clear());

        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = &a + &b; // Line 266: Track this

        PROVENANCE_TRACKER.with(|tracker| {
            let t = tracker.borrow();

            eprintln!("\n=== Tensor Level Provenance ===");
            let events = t.get_events(c.uop.id);
            assert!(events.is_some(), "Expected provenance for Tensor");

            for (i, event) in events.unwrap().iter().enumerate() {
                eprintln!("  [{}] {}", i, event);
            }

            // Just verify provenance exists - the exact location may vary due to inlining
            assert!(!events.unwrap().is_empty(), "Expected at least one provenance event");
        });
    }

    #[test]
    fn test_neg_basic() {
        let a = Tensor::from_slice([1.0f32, -2.0, 3.0]);
        let b = -&a;

        if let Op::Unary(op, _) = b.uop.op() {
            assert_eq!(format!("{:?}", op), "Neg");
        } else {
            panic!("Expected Unary Neg operation");
        }
    }

    #[test]
    fn test_neg_trait_variants() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);

        let _b1 = -&a; // &Tensor
        let _b2 = -a.clone(); // Tensor
    }

    #[test]
    fn test_abs_basic() {
        let a = Tensor::from_slice([-1.0f32, 2.0, -3.0]);
        let b = a.try_abs().unwrap();

        if let Op::Unary(op, _) = b.uop.op() {
            assert_eq!(format!("{:?}", op), "Abs");
        } else {
            panic!("Expected Unary Abs operation");
        }
    }

    #[test]
    fn test_abs_int() {
        let a = Tensor::from_slice([-1i32, 2, -3]);
        let b = a.try_abs().unwrap();

        if let Op::Unary(op, _) = b.uop.op() {
            assert_eq!(format!("{:?}", op), "Abs");
        } else {
            panic!("Expected Unary Abs operation");
        }
    }

    #[test]
    fn test_sqrt_basic() {
        let a = Tensor::from_slice([1.0f32, 4.0, 9.0]);
        let b = a.try_sqrt().unwrap();

        if let Op::Unary(op, _) = b.uop.op() {
            assert_eq!(format!("{:?}", op), "Sqrt");
        } else {
            panic!("Expected Unary Sqrt operation");
        }

        // Verify dtype is preserved
        assert_eq!(b.uop.dtype(), morok_dtype::DType::Float32);
    }

    #[test]
    fn test_sqrt_error_on_int() {
        let a = Tensor::from_slice([1i32, 4, 9]);
        let result = a.try_sqrt();
        assert!(result.is_err());
    }

    #[test]
    fn test_rsqrt_basic() {
        let a = Tensor::from_slice([1.0f32, 4.0, 9.0]);
        let b = a.try_rsqrt().unwrap();

        if let Op::Unary(op, _) = b.uop.op() {
            assert_eq!(format!("{:?}", op), "Rsqrt");
        } else {
            panic!("Expected Unary Rsqrt operation");
        }

        assert_eq!(b.uop.dtype(), morok_dtype::DType::Float32);
    }

    #[test]
    fn test_exp_basic() {
        let a = Tensor::from_slice([0.0f32, 1.0, 2.0]);
        let b = a.try_exp().unwrap();

        if let Op::Unary(op, _) = b.uop.op() {
            assert_eq!(format!("{:?}", op), "Exp");
        } else {
            panic!("Expected Unary Exp operation");
        }

        assert_eq!(b.uop.dtype(), morok_dtype::DType::Float32);
    }

    #[test]
    fn test_log_basic() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = a.try_log().unwrap();

        if let Op::Unary(op, _) = b.uop.op() {
            assert_eq!(format!("{:?}", op), "Log");
        } else {
            panic!("Expected Unary Log operation");
        }

        assert_eq!(b.uop.dtype(), morok_dtype::DType::Float32);
    }

    #[test]
    fn test_transcendental_error_on_int() {
        let a = Tensor::from_slice([1i32, 2, 3]);

        assert!(a.try_exp().is_err(), "Exp should fail on int");
        assert!(a.try_log().is_err(), "Log should fail on int");
        assert!(a.try_rsqrt().is_err(), "Rsqrt should fail on int");
    }

    #[test]
    fn test_unary_lazy_evaluation() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = -&a;
        let c = a.try_abs().unwrap();
        let d = a.try_sqrt().unwrap();

        // Results should NOT have buffers (lazy evaluation)
        assert!(b.buffer().is_none());
        assert!(c.buffer().is_none());
        assert!(d.buffer().is_none());

        // Only input should have buffer
        assert!(a.buffer().is_some());
    }
}
