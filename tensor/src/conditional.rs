//! Conditional and selection operations for tensors.
//!
//! This module provides element-wise conditional operations like where, maximum,
//! minimum, and clamp that are fundamental for many ML operations.

use bon::bon;
use snafu::ResultExt;
use svod_ir::UOp;

use crate::{Operand, Result, Tensor, error::UOpSnafu, operand::common_dtype};

#[bon]
impl Tensor {
    /// Element-wise conditional selection: `condition ? self : other`.
    ///
    /// For each element, returns `self[i]` if `condition[i]` is true, else `other[i]`.
    ///
    /// # Arguments
    /// * `condition` - Boolean tensor (dtype should be Bool or will be treated as boolean)
    /// * `other` - Alternative value tensor
    ///
    /// # Shape Requirements
    /// All three tensors (self, condition, other) must be broadcastable to the same shape.
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let condition = &x.gt(&Tensor::from_slice(&[2.0f32]))?; // [false, false, true, true]
    /// let zeros = Tensor::from_slice(&[0.0f32]);
    ///
    /// // Replace values > 2.0 with the original value, else 0
    /// let result = x.where_(condition, &zeros)?;
    /// // result = [0.0, 0.0, 3.0, 4.0]
    /// ```
    #[track_caller]
    pub fn where_<'o>(&self, condition: &Tensor, other: impl Into<Operand<'o>>) -> Result<Self> {
        origin_call!("where");
        use svod_ir::shape::{align_shapes_left, broadcast_shapes};

        let other = self.operand(other);
        let cond_shape = condition.shape()?;
        let self_shape = self.shape()?;
        let other_shape = other.shape()?;

        // Broadcast all three to a common shape
        let aligned = align_shapes_left(&[cond_shape.clone(), self_shape.clone(), other_shape.clone()]);
        let target = broadcast_shapes(&aligned).context(UOpSnafu)?;

        let cond_bc = condition.broadcast_to(&target)?;
        let self_bc = self.broadcast_to(&target)?;
        let other_bc = other.broadcast_to(&target)?;

        let result = UOp::try_where(cond_bc.uop(), self_bc.uop(), other_bc.uop()).context(UOpSnafu)?;
        Ok(Self::new(result))
    }

    /// Element-wise maximum: `max(self, other)`.
    ///
    /// Returns the element-wise maximum of two tensors.
    /// This is NOT a reduction - it returns a tensor of the same shape.
    ///
    /// # Shape Requirements
    /// Both tensors must be broadcastable to the same shape.
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 5.0, 3.0]);
    /// let b = Tensor::from_slice(&[2.0f32, 3.0, 4.0]);
    /// let result = a.maximum(&b)?;
    /// // result = [2.0, 5.0, 4.0]
    /// ```
    #[track_caller]
    pub fn maximum<'o>(&self, other: impl Into<Operand<'o>>) -> Result<Self> {
        origin_call!("maximum");
        let other = self.operand(other);
        let (lhs, rhs) = self.broadcast_for_binop(&other)?;
        let result = lhs.uop().try_max(&rhs.uop()).context(UOpSnafu)?;
        Ok(Self::new(result))
    }

    /// Element-wise minimum: `min(self, other)`.
    ///
    /// Returns the element-wise minimum of two tensors.
    /// This is NOT a reduction - it returns a tensor of the same shape.
    ///
    /// # Shape Requirements
    /// Both tensors must be broadcastable to the same shape.
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 5.0, 3.0]);
    /// let b = Tensor::from_slice(&[2.0f32, 3.0, 4.0]);
    /// let result = a.minimum(&b)?;
    /// // result = [1.0, 3.0, 3.0]
    /// ```
    #[track_caller]
    pub fn minimum<'o>(&self, other: impl Into<Operand<'o>>) -> Result<Self> {
        origin_call!("minimum");
        // Minimum is not a primitive, we implement it as: where(a < b, a, b)
        let other = self.operand(other);
        let condition = self.try_lt(&other)?;
        self.where_(&condition, &other)
    }

    /// Clamp values to a range: `max(min_val, min(self, max_val))`.
    ///
    /// Constrains all elements to be within [min_val, max_val].
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0, 3.0]);
    /// let min = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 0.0]);
    /// let max = Tensor::from_slice(&[2.0f32, 2.0, 2.0, 2.0, 2.0]);
    ///
    /// // Clamp to [0, 2]
    /// let result = x.clamp().min(&min).max(&max).call()?;
    /// // result = [0.0, 0.0, 1.0, 2.0, 2.0]
    ///
    /// // Clamp only lower bound
    /// let result = x.clamp().min(&min).call()?;
    /// // result = [0.0, 0.0, 1.0, 2.0, 3.0]
    ///
    /// // Clamp only upper bound
    /// let result = x.clamp().max(&max).call()?;
    /// // result = [-1.0, 0.0, 1.0, 2.0, 2.0]
    /// ```
    #[builder]
    #[track_caller]
    pub fn clamp<'lo, 'hi>(
        &self,
        #[builder(into)] min: Option<Operand<'lo>>,
        #[builder(into)] max: Option<Operand<'hi>>,
    ) -> Result<Self> {
        origin_call!("clamp");
        let mut result = self.clone();

        if let Some(min_val) = min {
            result = result.maximum(min_val)?;
        }

        if let Some(max_val) = max {
            result = result.minimum(max_val)?;
        }

        Ok(result)
    }

    /// Alias for `clamp` (matches NumPy/PyTorch naming).
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0, 3.0]);
    /// let min = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 0.0]);
    /// let max = Tensor::from_slice(&[2.0f32, 2.0, 2.0, 2.0, 2.0]);
    ///
    /// // Clip to [0, 2]
    /// let result = x.clip().min(&min).max(&max).call()?;
    /// ```
    #[builder]
    #[track_caller]
    pub fn clip<'lo, 'hi>(
        &self,
        #[builder(into)] min: Option<Operand<'lo>>,
        #[builder(into)] max: Option<Operand<'hi>>,
    ) -> Result<Self> {
        origin_call!("clip");
        self.clamp().maybe_min(min).maybe_max(max).call()
    }

    /// Fill elements where `mask` is true with `value`.
    ///
    /// `mask` must be broadcastable to `self`'s shape. `value` is either a
    /// scalar convertible to `ConstValue` (`i8`/`i16`/`i32`/`i64`/`u8`/`u16`/
    /// `u32`/`u64`/`f32`/`f64`/`bool`) or a `&Tensor`.
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let mask = Tensor::from_slice(&[true, false, true, false]);
    /// // Scalar value:
    /// let r1 = x.masked_fill(&mask, 0.0f32)?;            // [0.0, 2.0, 0.0, 4.0]
    /// // Tensor value:
    /// let fill = Tensor::from_slice(&[-1.0f32, -2.0, -3.0, -4.0]);
    /// let r2 = x.masked_fill(&mask, &fill)?;             // [-1.0, 2.0, -3.0, 4.0]
    /// ```
    #[track_caller]
    pub fn masked_fill<'v>(&self, mask: &Tensor, value: impl Into<Operand<'v>>) -> Result<Self> {
        origin_call!("masked_fill");
        self.operand(value).where_(mask, self)
    }

    /// Branch on this boolean tensor: `self ? when_true : when_false`.
    ///
    /// The mirror image of [`where_`](Self::where_), which is called on the
    /// *true* branch. Either branch may be a tensor or a scalar; a scalar takes
    /// the dtype of the other branch.
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let big = x.try_gt(2.0)?;
    /// let r = big.select(&x, 0.0)?;   // [0.0, 0.0, 3.0, 4.0]
    /// ```
    #[track_caller]
    pub fn select<'t, 'f>(
        &self,
        when_true: impl Into<Operand<'t>>,
        when_false: impl Into<Operand<'f>>,
    ) -> Result<Self> {
        origin_call!("select");
        let (t, f) = (when_true.into(), when_false.into());
        let dtype = common_dtype(&t, &f);
        let when_false = f.materialize(dtype.clone());
        t.materialize(dtype).where_(self, &when_false)
    }
}
