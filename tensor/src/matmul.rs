//! Matrix multiplication and linear transformations.
//!
//! This module provides dot product and matrix multiplication operations
//! following Tinygrad's implementation strategy.

use bon::bon;
use morok_dtype::DType;

use crate::{Error, Result, Tensor};

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
        self.dot_impl(other, None)
    }

    /// Implementation of dot product with optional dtype.
    fn dot_impl(&self, other: &Tensor, dtype: Option<DType>) -> Result<Tensor> {
        let x_shape = self.shape()?;
        let w_shape = other.shape()?;

        let dx = x_shape.len();
        let dw = w_shape.len();

        // Step 1: Validate dimensions - both must be at least 1D
        if dx == 0 || dw == 0 {
            return Err(Error::DotDimensionError { lhs_dims: dx, rhs_dims: dw });
        }

        // Step 2: Determine contraction axis for RHS
        // In Python: axis_w = -min(w.ndim, 2)
        // For 1D: axis is -1, for 2D+: axis is -2
        // We'll use negative indexing to match Tinygrad's behavior
        let axis_w = -(std::cmp::min(dw, 2) as isize);

        // Step 3: Validate contraction dimension
        // x's last dim must match w's contraction dim
        let x_shape_vec = require_concrete_shape(self)?;
        let w_shape_vec = require_concrete_shape(other)?;

        let x_contract = x_shape_vec[dx - 1];
        // Convert negative axis_w to positive index for accessing the vector
        let axis_w_pos = if axis_w < 0 { (dw as isize + axis_w) as usize } else { axis_w as usize };
        let w_contract = w_shape_vec[axis_w_pos];

        if x_contract != w_contract {
            return Err(Error::DotShapeMismatch { lhs_shape: x_shape_vec, rhs_shape: w_shape_vec });
        }

        // Step 4: Reshape for broadcasting
        // broadcast_dims = min(min(dx-1, dw-1), 1)
        let broadcast_dims = std::cmp::min(std::cmp::min(dx.saturating_sub(1), dw.saturating_sub(1)), 1);

        // Reshape x: [..., K] → [..., 1, K] (insert broadcast_dims before last)
        let mut x_new_shape: Vec<isize> = Vec::new();
        x_new_shape.extend(x_shape_vec[..dx - 1].iter().map(|&s| s as isize));
        x_new_shape.extend(vec![1; broadcast_dims]);
        x_new_shape.push(x_shape_vec[dx - 1] as isize);

        let x_reshaped = self.try_reshape(&x_new_shape)?;

        // Reshape w: [..., K, N] → [..., 1, K, N] (insert broadcast_dims after batch dims)
        // In Python: w.reshape(*w.shape[0:-2], *[1]*min(dx-1, dw-1, 1), *w.shape[axis_w:])
        // w.shape[0:-2] takes all but last 2 dims
        // w.shape[axis_w:] takes from axis_w to end
        let mut w_new_shape: Vec<isize> = Vec::new();
        // For dw >= 2: take w.shape[0:-2] (all but last 2)
        // For dw == 1: take nothing (0:-2 is empty for 1D)
        if dw >= 2 {
            w_new_shape.extend(w_shape_vec[..dw - 2].iter().map(|&s| s as isize));
        }
        w_new_shape.extend(vec![1; broadcast_dims]);
        w_new_shape.extend(w_shape_vec[axis_w_pos..].iter().map(|&s| s as isize));

        let w_reshaped = other.try_reshape(&w_new_shape)?;

        // Step 5: Transpose w to move contraction dim to last
        // Tinygrad: w.reshape(...).transpose(-1, axis_w)
        // axis_w is negative: -1 for 1D, -2 for 2D+
        let w_transposed = w_reshaped.try_transpose(-1, axis_w)?;

        // Step 6: Broadcast and element-wise multiply
        // After reshape and transpose, shapes should be broadcastable
        // For 2D @ 2D: x=[2,1,2], w=[1,2,2] → need to broadcast to [2,2,2]
        // For 2D @ 1D: x=[2,3], w=[3] → shapes already align on last dim
        let x_shape_after = x_reshaped.shape()?;
        let w_shape_after = w_transposed.shape()?;

        // Compute broadcast shape (element-wise max of each dimension)
        // Handle different ranks by padding with 1s on the left
        let max_rank = std::cmp::max(x_shape_after.len(), w_shape_after.len());
        let mut broadcast_shape: Vec<isize> = Vec::new();

        for i in 0..max_rank {
            // Index from the right (negative indexing)
            let idx = i as isize - max_rank as isize;

            let x_dim = if (idx + x_shape_after.len() as isize) >= 0 {
                let pos = (idx + x_shape_after.len() as isize) as usize;
                x_shape_after[pos]
                    .as_const()
                    .ok_or(Error::SymbolicShapeUnsupported { operation: "matmul".to_string() })?
            } else {
                1 // Implicit dimension of size 1
            };

            let w_dim = if (idx + w_shape_after.len() as isize) >= 0 {
                let pos = (idx + w_shape_after.len() as isize) as usize;
                w_shape_after[pos]
                    .as_const()
                    .ok_or(Error::SymbolicShapeUnsupported { operation: "matmul".to_string() })?
            } else {
                1 // Implicit dimension of size 1
            };

            broadcast_shape.push(std::cmp::max(x_dim, w_dim) as isize);
        }

        // Expand both tensors to broadcast shape
        // First reshape to add missing dimensions (as 1s), then expand
        let x_with_dims = if x_shape_after.len() < max_rank {
            // Pad on left with 1s
            let mut new_shape = vec![1isize; max_rank - x_shape_after.len()];
            new_shape.extend(x_shape_after.iter().map(|s| s.as_const().unwrap() as isize));
            x_reshaped.try_reshape(&new_shape)?
        } else {
            x_reshaped
        };

        let w_with_dims = if w_shape_after.len() < max_rank {
            // Pad on left with 1s
            let mut new_shape = vec![1isize; max_rank - w_shape_after.len()];
            new_shape.extend(w_shape_after.iter().map(|s| s.as_const().unwrap() as isize));
            w_transposed.try_reshape(&new_shape)?
        } else {
            w_transposed
        };

        let x_expanded = x_with_dims.try_expand(&broadcast_shape)?;
        let w_expanded = w_with_dims.try_expand(&broadcast_shape)?;

        let product = x_expanded.try_mul(&w_expanded)?;

        // Step 7: Sum over last axis (the contraction dimension)
        let result =
            if let Some(dt) = dtype { product.sum_with().axes(-1).dtype(dt).call()? } else { product.sum(-1)? };

        Ok(result)
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
        self.dot(other)
    }
}

#[bon]
impl Tensor {
    /// Dot product with optional accumulation dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    ///
    /// // Use float64 accumulation for better precision
    /// let result = a.dot_with(&b).dtype(DType::Float64).call()?;
    /// ```
    #[builder]
    pub fn dot_with(&self, other: &Tensor, dtype: Option<DType>) -> Result<Tensor> {
        self.dot_impl(other, dtype)
    }

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
        self.dot_impl(other, dtype)
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
            self.dot_impl(&weight_t, dtype)?
        };

        // Add bias if provided
        if let Some(bias_tensor) = bias {
            // Bias might need broadcasting if result has more dimensions
            // e.g., result=[1,2], bias=[2] needs bias to be reshaped to [1,2]
            let result_shape = result.shape()?;
            let bias_shape = bias_tensor.shape()?;

            if result_shape.len() == bias_shape.len() {
                // Same rank, direct add
                result.try_add(bias_tensor)
            } else if result_shape.len() > bias_shape.len() {
                // Need to add leading dimensions to bias
                let mut new_bias_shape: Vec<isize> = vec![1; result_shape.len() - bias_shape.len()];
                new_bias_shape.extend(bias_shape.iter().map(|s| s.as_const().unwrap() as isize));

                // Reshape bias to match rank
                let bias_reshaped = bias_tensor.try_reshape(&new_bias_shape)?;

                // Expand to match result shape
                let result_shape_vec: Vec<isize> =
                    result_shape.iter().map(|s| s.as_const().unwrap() as isize).collect();
                let bias_expanded = bias_reshaped.try_expand(&result_shape_vec)?;

                result.try_add(&bias_expanded)
            } else {
                // Bias has more dimensions than result - shouldn't happen in practice
                result.try_add(bias_tensor)
            }
        } else {
            Ok(result)
        }
    }
}

/// Validate and extract concrete shape (no symbolic dims allowed for matmul).
fn require_concrete_shape(tensor: &Tensor) -> Result<Vec<usize>> {
    let shape = tensor.shape()?;
    shape
        .iter()
        .map(|dim| dim.as_const().ok_or(Error::SymbolicShapeUnsupported { operation: "matmul".to_string() }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Basic 2D x 2D Tests ==========

    #[test]
    fn test_matmul_2d_basic() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
        let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2]).unwrap();
        let c = a.dot(&b).unwrap();

        let c_shape = c.shape().unwrap();
        assert_eq!(c_shape.len(), 2);
        assert_eq!(c_shape[0].as_const().unwrap(), 2);
        assert_eq!(c_shape[1].as_const().unwrap(), 2);
    }

    #[test]
    fn test_matmul_2d_non_square() {
        // [2, 3] @ [3, 4] → [2, 4]
        let a = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
        let b = Tensor::from_slice([1.0f32; 12]).try_reshape(&[3, 4]).unwrap();
        let c = a.dot(&b).unwrap();

        let c_shape = c.shape().unwrap();
        assert_eq!(c_shape.len(), 2);
        assert_eq!(c_shape[0].as_const().unwrap(), 2);
        assert_eq!(c_shape[1].as_const().unwrap(), 4);
    }

    #[test]
    fn test_matmul_alias() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
        let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2]).unwrap();

        // Test that matmul is an alias for dot
        let c1 = a.dot(&b).unwrap();
        let c2 = a.matmul(&b).unwrap();

        assert_eq!(c1.shape().unwrap().len(), c2.shape().unwrap().len());
    }

    // ========== 1D x 1D Tests (Dot Product) ==========

    #[test]
    fn test_dot_product_1d() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = a.dot(&b).unwrap();

        // Result should be scalar (0D tensor)
        let c_shape = c.shape().unwrap();
        assert_eq!(c_shape.len(), 0);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = Tensor::from_slice([1.0f32, 0.0, 0.0]);
        let b = Tensor::from_slice([0.0f32, 1.0, 0.0]);
        let c = a.dot(&b).unwrap();

        // Orthogonal vectors → dot product = 0
        let c_shape = c.shape().unwrap();
        assert_eq!(c_shape.len(), 0);
    }

    // ========== 1D x 2D and 2D x 1D Tests ==========

    #[test]
    fn test_vector_matrix() {
        // [3] @ [3, 4] → [4]
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([1.0f32; 12]).try_reshape(&[3, 4]).unwrap();
        let c = a.dot(&b).unwrap();

        let c_shape = c.shape().unwrap();
        assert_eq!(c_shape.len(), 1);
        assert_eq!(c_shape[0].as_const().unwrap(), 4);
    }

    #[test]
    fn test_matrix_vector() {
        // [2, 3] @ [3] → [2]
        let a = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
        let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let c = a.dot(&b).unwrap();

        let c_shape = c.shape().unwrap();
        assert_eq!(c_shape.len(), 1);
        assert_eq!(c_shape[0].as_const().unwrap(), 2);
    }

    // ========== Batched Matmul Tests ==========

    #[test]
    fn test_batched_matmul_3d() {
        // [2, 3, 4] @ [2, 4, 5] → [2, 3, 5]
        let a = Tensor::from_slice([1.0f32; 24]).try_reshape(&[2, 3, 4]).unwrap();
        let b = Tensor::from_slice([1.0f32; 40]).try_reshape(&[2, 4, 5]).unwrap();
        let c = a.dot(&b).unwrap();

        let c_shape = c.shape().unwrap();
        assert_eq!(c_shape.len(), 3);
        assert_eq!(c_shape[0].as_const().unwrap(), 2);
        assert_eq!(c_shape[1].as_const().unwrap(), 3);
        assert_eq!(c_shape[2].as_const().unwrap(), 5);
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_matmul_error_0d() {
        let scalar = Tensor::from_slice([1.0f32]).try_reshape(&[]).unwrap();
        let vector = Tensor::from_slice([1.0f32, 2.0, 3.0]);

        // 0D tensors not supported
        assert!(scalar.dot(&vector).is_err());
        assert!(vector.dot(&scalar).is_err());
    }

    #[test]
    fn test_matmul_error_shape_mismatch() {
        // [2, 3] @ [4, 5] - inner dimensions don't match
        let a = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
        let b = Tensor::from_slice([1.0f32; 20]).try_reshape(&[4, 5]).unwrap();

        let result = a.dot(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_identity() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
        let identity = Tensor::from_slice([1.0f32, 0.0, 0.0, 1.0]).try_reshape(&[2, 2]).unwrap();

        let result = a.dot(&identity).unwrap();

        // Result shape should match input
        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 2);
        assert_eq!(result_shape[0].as_const().unwrap(), 2);
        assert_eq!(result_shape[1].as_const().unwrap(), 2);
    }

    // ========== Dtype Tests ==========

    #[test]
    fn test_matmul_dtype_promotion() {
        let a = Tensor::from_slice([1i32, 2, 3, 4]).try_reshape(&[2, 2]).unwrap();
        let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2]).unwrap();

        let c = a.dot(&b).unwrap();
        // Result should be promoted to float32
        assert_eq!(c.uop.dtype(), DType::Float32);
    }

    #[test]
    fn test_matmul_explicit_dtype() {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
        let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2]).unwrap();

        // Use float64 accumulation
        let c = a.dot_with().other(&b).dtype(DType::Float64).call().unwrap();
        assert_eq!(c.uop.dtype(), DType::Float64);
    }

    // ========== Linear Layer Tests ==========

    #[test]
    fn test_linear_basic() {
        // input: [1, 3], weight: [2, 3], bias: [2]
        let input = Tensor::from_slice([1.0f32, 2.0, 3.0]).try_reshape(&[1, 3]).unwrap();
        let weight = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
        let bias = Tensor::from_slice([0.1f32, 0.2]);

        let result = input.linear().weight(&weight).bias(&bias).call().unwrap();

        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 2);
        assert_eq!(result_shape[0].as_const().unwrap(), 1);
        assert_eq!(result_shape[1].as_const().unwrap(), 2);
    }

    #[test]
    fn test_linear_no_bias() {
        let input = Tensor::from_slice([1.0f32, 2.0, 3.0]).try_reshape(&[1, 3]).unwrap();
        let weight = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();

        let result = input.linear().weight(&weight).call().unwrap();

        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 2);
        assert_eq!(result_shape[0].as_const().unwrap(), 1);
        assert_eq!(result_shape[1].as_const().unwrap(), 2);
    }

    #[test]
    fn test_linear_batched() {
        // input: [4, 3], weight: [2, 3] → output: [4, 2]
        let input = Tensor::from_slice([1.0f32; 12]).try_reshape(&[4, 3]).unwrap();
        let weight = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();

        let result = input.linear().weight(&weight).call().unwrap();

        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 2);
        assert_eq!(result_shape[0].as_const().unwrap(), 4);
        assert_eq!(result_shape[1].as_const().unwrap(), 2);
    }

    #[test]
    fn test_linear_1d_weight() {
        // Test 1D weight case (element-wise multiply)
        let input = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let weight = Tensor::from_slice([2.0f32, 3.0, 4.0]);

        let result = input.linear().weight(&weight).call().unwrap();

        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 1);
        assert_eq!(result_shape[0].as_const().unwrap(), 3);
    }
}
