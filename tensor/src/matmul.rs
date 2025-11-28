//! Matrix multiplication and linear transformations.
//!
//! This module provides dot product and matrix multiplication operations
//! following Tinygrad's implementation strategy.

use std::iter;

use bon::bon;
use morok_dtype::DType;
use morok_ir::{SInt, shape::Shape};
use snafu::{ResultExt, ensure};

use crate::{Result, Tensor, UOpSnafu, error::*};

impl Tensor {
    /// Dot product / matrix multiplication.
    ///
    /// Core method following Tinygrad's API:
    /// - 1D @ 1D: dot product (scalar)
    /// - 2D @ 2D: matrix multiplication
    /// - 1D @ 2D: vector @ matrix
    /// - 2D @ 1D: matrix @ vector
    /// - 3D+: batched matmul (batch dims broadcast)
    ///
    /// # Arguments
    /// * `other` - Right-hand tensor
    ///
    /// # Examples
    /// ```ignore
    /// // Vector dot product
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    /// let result = a.dot(&b)?; // scalar: 32.0
    ///
    /// // Matrix multiplication
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2])?;
    /// let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2])?;
    /// let result = a.dot(&b)?; // [2, 2]
    /// ```
    pub fn dot(&self, other: &Tensor) -> Result<Tensor> {
        self.matmul_with().other(other).call()
    }

    /// Matrix multiplication (alias for dot).
    ///
    /// Matches PyTorch API. Equivalent to `self.dot(other)`.
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2])?;
    /// let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2])?;
    /// let result = a.matmul(&b)?;
    /// ```
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        self.matmul_with().other(other).call()
    }
}

/// Build matmul broadcast shape by inserting broadcast dimensions.
///
/// Constructs: shape[..prefix_len] + [1; broadcast_dims] + shape[tail_start..]
fn build_matmul_broadcast_shape(shape: &Shape, prefix_len: usize, broadcast_dims: usize, tail_start: usize) -> Shape {
    shape[..prefix_len]
        .iter()
        .cloned()
        .chain(iter::repeat_n(SInt::Const(1), broadcast_dims))
        .chain(shape[tail_start..].iter().cloned())
        .collect()
}

#[bon]
impl Tensor {
    /// Matrix multiplication with optional dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2])?;
    /// let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2])?;
    /// let result = a.matmul_with(&b).dtype(DType::Float64).call()?;
    /// ```
    #[builder]
    pub fn matmul_with(&self, other: &Tensor, dtype: Option<DType>) -> Result<Tensor> {
        // Step 1: Check dimensions
        let (dx, dw) = (self.ndim()?, other.ndim()?);
        ensure!(dx != 0 && dw != 0, DotDimensionSnafu { lhs_dims: dx, rhs_dims: dw });

        let x_shape = self.shape()?;
        let w_shape = other.shape()?;

        // Step 2: Determine contraction axis and validate
        let axis_w = -(dw.min(2) as isize);
        ensure!(self.dim(-1)? == other.dim(axis_w)?, DotShapeMismatchSnafu { lhs_shape: x_shape, rhs_shape: w_shape });

        // Step 3: Reshape for broadcasting
        let broadcast_dims = (dx - 1).min(dw - 1).min(1);

        // Reshape x: [..., K] → [..., 1, K]
        let x_new_shape = build_matmul_broadcast_shape(&x_shape, dx - 1, broadcast_dims, dx - 1);
        let x_reshaped = self.uop.try_reshape(&x_new_shape).map(Self::new).context(UOpSnafu)?;

        // Reshape w: [..., K, N] → [..., 1, K, N]
        let axis_w_pos = Tensor::normalize_axis(axis_w, dw)?;
        let w_new_shape = build_matmul_broadcast_shape(&w_shape, dw.saturating_sub(2), broadcast_dims, axis_w_pos);
        let w_reshaped = other.uop.try_reshape(&w_new_shape).map(Self::new).context(UOpSnafu)?;

        // Step 4: Transpose, multiply, and sum
        let product = x_reshaped.try_mul(&w_reshaped.try_transpose(-1, axis_w)?)?;

        if let Some(dt) = dtype { product.sum_with().axes(-1).dtype(dt).call() } else { product.sum(-1) }
    }

    /// Linear transformation: `self @ weight.T + bias`.
    ///
    /// Common operation in neural networks (fully connected layers).
    /// Follows PyTorch convention where weight has shape `[out_features, in_features]`
    /// and is transposed before multiplication.
    ///
    /// # Arguments
    /// * `weight` - Weight matrix (shape: [out_features, in_features])
    /// * `bias` - Optional bias vector (shape: [out_features])
    ///
    /// # Shape Requirements
    /// - self: [..., in_features]
    /// - weight: [out_features, in_features]
    /// - bias: [out_features] or None
    /// - result: [..., out_features]
    ///
    /// # Examples
    /// ```ignore
    /// let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0]).try_reshape(&[1, 3])?;
    /// let weight = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3])?;
    /// let bias = Tensor::from_slice(&[0.1f32, 0.2f32]);
    /// let result = input.linear().weight(&weight).bias(&bias).call()?;
    /// // result shape: [1, 2]
    /// ```
    #[builder]
    pub fn linear(&self, weight: &Tensor, bias: Option<&Tensor>, dtype: Option<DType>) -> Result<Tensor> {
        let weight_shape = weight.shape()?;

        // For 1D weight, use element-wise multiply (broadcast)
        let result = if weight_shape.len() == 1 {
            if let Some(dt) = dtype {
                let casted = self.cast(dt)?;
                casted.try_mul(weight)?
            } else {
                self.try_mul(weight)?
            }
        } else {
            // For 2D+ weight, transpose it first (PyTorch convention)
            // PyTorch Linear layer: x @ weight.T
            let weight_t = weight.try_transpose(-1, -2)?;
            self.matmul_with().other(&weight_t).maybe_dtype(dtype).call()?
        };

        // Add bias if provided
        if let Some(bias_tensor) = bias {
            let result_shape = result.shape()?;
            let bias_broadcasted = bias_tensor.broadcast_to(&result_shape)?;
            result.try_add(&bias_broadcasted)
        } else {
            Ok(result)
        }
    }
}
