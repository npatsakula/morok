//! Advanced and specialized operations.
//!
//! This module contains complex operations like WMMA (tensor cores),
//! contract/unroll for vectorization, and various other specialized ops.

use std::rc::Rc;

use morok_dtype::DType;
use smallvec::SmallVec;

use super::super::{BinaryOp, Op, Result, TernaryOp, UOp, UnaryOp, WmmaMetadata};
use crate::error::InvalidDTypeForOpSnafu;

impl UOp {
    // =========================================================================
    // Extended Unary Operations
    // =========================================================================

    /// Sine: sin(x) - requires float dtype.
    pub fn sin_op(operand: Rc<Self>) -> Result<Rc<Self>> {
        let dtype = operand.dtype();
        if !dtype.is_float() {
            return InvalidDTypeForOpSnafu { operation: "sin", dtype }.fail();
        }
        Ok(Self::new(Op::Unary(UnaryOp::Sin, operand), dtype))
    }

    /// Reciprocal: 1/x.
    pub fn reciprocal_op(operand: Rc<Self>) -> Rc<Self> {
        let dtype = operand.dtype();
        Self::new(Op::Unary(UnaryOp::Reciprocal, operand), dtype)
    }

    /// Truncate towards zero.
    pub fn trunc_op(operand: Rc<Self>) -> Rc<Self> {
        let dtype = operand.dtype();
        Self::new(Op::Unary(UnaryOp::Trunc, operand), dtype)
    }

    // =========================================================================
    // Extended Binary Operations
    // =========================================================================

    /// Maximum of two values: max(a, b).
    ///
    /// # Errors
    /// Returns error if type promotion fails or void type is used.
    pub fn try_max_op(self: &Rc<Self>, rhs: &Rc<Self>) -> Result<Rc<Self>> {
        let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
        Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::Max)?;
        Ok(Self::new(Op::Binary(BinaryOp::Max, lhs, rhs), dtype))
    }

    /// Power: a^b.
    ///
    /// # Errors
    /// Returns error if type promotion fails or void type is used.
    pub fn try_pow_op(self: &Rc<Self>, rhs: &Rc<Self>) -> Result<Rc<Self>> {
        let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
        Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::Pow)?;
        Ok(Self::new(Op::Binary(BinaryOp::Pow, lhs, rhs), dtype))
    }

    /// Integer division: a // b (truncated).
    ///
    /// # Errors
    /// Returns error if type promotion fails, void type is used, or division by zero is detected.
    pub fn try_idiv_op(self: &Rc<Self>, rhs: &Rc<Self>) -> Result<Rc<Self>> {
        Self::check_division_by_zero(rhs)?;
        let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
        Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::Idiv)?;
        Ok(Self::new(Op::Binary(BinaryOp::Idiv, lhs, rhs), dtype))
    }

    /// Float division: a / b.
    ///
    /// # Errors
    /// Returns error if type promotion fails, void type is used, or division by zero is detected.
    pub fn try_fdiv_op(self: &Rc<Self>, rhs: &Rc<Self>) -> Result<Rc<Self>> {
        Self::check_division_by_zero(rhs)?;
        let (lhs, rhs, dtype) = Self::promote_and_cast(self.clone(), rhs.clone())?;
        Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::Fdiv)?;
        Ok(Self::new(Op::Binary(BinaryOp::Fdiv, lhs, rhs), dtype))
    }

    /// Threefry PRNG: threefry(x, key).
    ///
    /// # Errors
    /// Returns error if shapes don't match.
    pub fn threefry_op(lhs: Rc<Self>, rhs: Rc<Self>) -> Result<Rc<Self>> {
        let dtype = DType::UInt64; // Threefry always returns uint64
        Self::validate_binary_shapes(&lhs, &rhs, BinaryOp::Threefry)?;
        Ok(Self::new(Op::Binary(BinaryOp::Threefry, lhs, rhs), dtype))
    }

    // =========================================================================
    // Ternary Operations
    // =========================================================================

    /// Conditional selection: condition ? true_val : false_val.
    ///
    /// # Errors
    /// Returns error if true_val and false_val have mismatched shapes.
    pub fn where_op(condition: Rc<Self>, true_val: Rc<Self>, false_val: Rc<Self>) -> Result<Rc<Self>> {
        let dtype = true_val.dtype(); // Result has same dtype as branches
        Self::validate_ternary_shapes(&true_val, &false_val)?;
        Ok(Self::new(Op::Ternary(TernaryOp::Where, condition, true_val, false_val), dtype))
    }

    /// Multiply-accumulate: a * b + c (fused operation).
    ///
    /// # Errors
    /// Returns error if operands have mismatched shapes.
    pub fn mulacc_op(a: Rc<Self>, b: Rc<Self>, c: Rc<Self>) -> Result<Rc<Self>> {
        let dtype = a.dtype(); // Preserve first operand dtype
        // Validate all three operands have matching shapes
        Self::validate_ternary_shapes(&a, &b)?;
        Self::validate_ternary_shapes(&a, &c)?;
        Ok(Self::new(Op::Ternary(TernaryOp::MulAcc, a, b, c), dtype))
    }

    // =========================================================================
    // Type Operations
    // =========================================================================

    /// Bitcast: reinterpret bits as different type.
    pub fn bitcast(src: Rc<Self>, dtype: DType) -> Rc<Self> {
        Self::new(Op::BitCast { src, dtype: dtype.clone() }, dtype)
    }

    // =========================================================================
    // Tensor Core and Vectorization Operations
    // =========================================================================

    /// Warp Matrix Multiply-Accumulate (tensor cores).
    pub fn wmma(a: Rc<Self>, b: Rc<Self>, c: Rc<Self>, metadata: WmmaMetadata) -> Rc<Self> {
        let base_dtype = metadata.dtype_out.clone();

        // Calculate vector size from upcast axes (product of all axis sizes)
        let vec_size = metadata.upcast_axes.iter().map(|(_, size)| size).product::<usize>();

        let dtype = if vec_size > 1 { base_dtype.vec(vec_size) } else { base_dtype };

        Self::new(Op::Wmma { a, b, c, metadata }, dtype)
    }

    /// Contract scalar operations into vectorized form.
    pub fn contract(src: Rc<Self>, upcast_ranges: Vec<(usize, usize)>) -> Rc<Self> {
        let base_dtype = src.dtype();

        // Calculate vector size from upcast ranges (product of all range sizes)
        let vec_size = upcast_ranges.iter().map(|(_, size)| size).product::<usize>();

        let dtype = if vec_size > 1 { base_dtype.vec(vec_size) } else { base_dtype };

        Self::new(Op::Contract { src, upcast_ranges }, dtype)
    }

    /// Unroll loops for vectorization.
    pub fn unroll(src: Rc<Self>, unroll_axes: Vec<(usize, usize)>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Unroll { src, unroll_axes }, dtype)
    }

    // =========================================================================
    // Kernel and Optimization Operations
    // =========================================================================

    /// Kernel wrapper.
    ///
    /// Creates a KERNEL operation with the given sources (kernel arguments) and AST (computation).
    ///
    /// # Arguments
    ///
    /// * `sources` - Kernel arguments (buffers and variables)
    /// * `ast` - The computation graph (usually SINK, COPY, or BUFFER_VIEW)
    pub fn kernel(sources: SmallVec<[Rc<Self>; 4]>, ast: Rc<Self>) -> Rc<Self> {
        Self::new(Op::Kernel { sources, ast }, DType::Void)
    }

    /// In-place assignment.
    pub fn assign(target: Rc<Self>, value: Rc<Self>) -> Rc<Self> {
        let dtype = target.dtype();
        Self::new(Op::Assign { target, value }, dtype)
    }

    /// Detach from gradient flow / force materialization.
    pub fn detach(src: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Detach { src }, dtype)
    }

    /// Ensure contiguous memory layout.
    pub fn contiguous(src: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Contiguous { src }, dtype)
    }

    /// Contiguous backward pass.
    pub fn contiguous_backward(src: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::ContiguousBackward { src }, dtype)
    }

    /// Ordering constraint: passthrough depends on deps.
    pub fn after(passthrough: Rc<Self>, deps: SmallVec<[Rc<Self>; 4]>) -> Rc<Self> {
        let dtype = passthrough.dtype();
        Self::new(Op::After { passthrough, deps }, dtype)
    }

    /// Precast optimizer hint.
    pub fn precast(src: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Precast { src }, dtype)
    }

    /// Custom code (statement).
    pub fn custom(deps: SmallVec<[Rc<Self>; 4]>, code: String, dtype: DType) -> Rc<Self> {
        Self::new(Op::Custom { deps, code }, dtype)
    }

    /// Custom inline code (expression).
    pub fn customi(deps: SmallVec<[Rc<Self>; 4]>, code: String, dtype: DType) -> Rc<Self> {
        Self::new(Op::CustomI { deps, code }, dtype)
    }

    /// Special GPU dimension index.
    pub fn special(end: Rc<Self>, name: String) -> Rc<Self> {
        Self::new(Op::Special { end, name }, DType::Index)
    }
}
