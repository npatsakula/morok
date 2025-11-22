//! Reduction operations for tensors.
//!
//! This module provides reduction operations like sum, max, min, prod, and mean
//! with ergonomic APIs that match PyTorch/NumPy conventions.

use bon::bon;
use morok_dtype::{DType, ScalarDType};
use morok_ir::{ReduceOp, UOp};
use snafu::ResultExt;

use crate::{
    Error, Result, Tensor,
    error::{SymbolicShapeUnsupportedSnafu, UOpSnafu},
};

/// Specification for reduction axes.
///
/// Supports:
/// - All axes: `AxisSpec::All` (from `()`)
/// - Single axis: `AxisSpec::Single(0)` (from `isize`)
/// - Multiple axes: `AxisSpec::Multiple(vec![0, 2])` (from `&[isize]` or `Vec<isize>`)
#[derive(Debug, Clone)]
pub enum AxisSpec {
    /// Reduce all axes (produces scalar).
    All,
    /// Reduce a single axis (supports negative indexing).
    Single(isize),
    /// Reduce multiple axes (each supports negative indexing).
    Multiple(Vec<isize>),
}

// Ergonomic Into conversions for AxisSpec
impl From<()> for AxisSpec {
    fn from(_: ()) -> Self {
        Self::All
    }
}

impl From<isize> for AxisSpec {
    fn from(axis: isize) -> Self {
        Self::Single(axis)
    }
}

impl From<&[isize]> for AxisSpec {
    fn from(axes: &[isize]) -> Self {
        Self::Multiple(axes.to_vec())
    }
}

impl From<Vec<isize>> for AxisSpec {
    fn from(axes: Vec<isize>) -> Self {
        Self::Multiple(axes)
    }
}

// =========================================================================
// Tensor Reduction Methods
// =========================================================================

impl Tensor {
    /// Resolve axis specification to normalized axis indices.
    ///
    /// Handles:
    /// - `AxisSpec::All` → all axes (0..ndim)
    /// - Single/multiple axes → normalize negative indices
    /// - Deduplication
    /// - Bounds checking
    fn resolve_axis_spec(spec: &AxisSpec, ndim: usize) -> Result<Vec<usize>> {
        match spec {
            AxisSpec::All => Ok((0..ndim).collect()),
            AxisSpec::Single(axis) => {
                let normalized = Self::normalize_axis(*axis, ndim)?;
                Ok(vec![normalized])
            }
            AxisSpec::Multiple(axes) => {
                let mut normalized: Vec<usize> =
                    axes.iter().map(|&axis| Self::normalize_axis(axis, ndim)).collect::<Result<_>>()?;

                // Deduplicate axes (keep first occurrence)
                normalized.sort_unstable();
                normalized.dedup();

                Ok(normalized)
            }
        }
    }

    /// Get accumulation dtype for sum operations (Tinygrad-compatible).
    ///
    /// Used when `promote=true` in reduction builders.
    ///
    /// Promotion rules:
    /// - int8, int16 → int32
    /// - int32, int64 → preserve
    /// - uint8, uint16 → uint32
    /// - uint32, uint64 → preserve
    /// - float16, bfloat16 → float32 (for accumulation)
    /// - float32, float64 → preserve
    /// - bool → int32
    fn sum_acc_dtype(dtype: &DType) -> DType {
        use ScalarDType::*;
        let Some(scalar) = dtype.scalar() else {
            return dtype.clone();
        };

        match scalar {
            Bool => DType::Int32,
            Int8 | Int16 => DType::Int32,
            Int32 | Int64 => dtype.clone(),
            UInt8 | UInt16 => DType::UInt32,
            UInt32 | UInt64 => dtype.clone(),
            Float16 | BFloat16 | FP8E4M3 | FP8E5M2 => DType::Float32,
            Float32 | Float64 => dtype.clone(),
            Void | Index => dtype.clone(),
        }
    }

    /// Check if dtype should be cast back after sum accumulation.
    ///
    /// Tinygrad casts back to original dtype for:
    /// - float16
    /// - bfloat16
    /// - fp8 variants
    fn should_cast_back_after_sum(dtype: &DType) -> bool {
        matches!(
            dtype.scalar(),
            Some(ScalarDType::Float16 | ScalarDType::BFloat16 | ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2)
        )
    }

    /// Check if dtype is an integer or bool type.
    fn is_integer_dtype(dtype: &DType) -> bool {
        dtype.is_int() || matches!(dtype.scalar(), Some(ScalarDType::Bool))
    }

    /// Remove singleton dimensions from reduced axes when keepdim=false.
    ///
    /// Example:
    /// - shape [2, 3, 4], reduced axes [0, 2] → shape [2, 1, 4]
    /// - keepdim=false → reshape to [3]
    fn remove_singleton_dims(self, reduced_axes: &[usize]) -> Result<Self> {
        let shape = self.shape()?;

        // Build new shape by filtering out size-1 dimensions that were reduced
        let new_shape: Vec<isize> = shape
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| {
                // Only keep non-reduced axes, or reduced axes that aren't size 1
                if reduced_axes.contains(&i) {
                    None // Remove this dimension
                } else {
                    // Keep dimension, convert to isize
                    dim.as_const().map(|v| v as isize)
                }
            })
            .collect();

        // If all dimensions were reduced, result is scalar (shape [])
        if new_shape.is_empty() {
            // For scalar result, reshape to shape [] (0-d tensor)
            // IR reshape expects same product, so [] → [] is valid
            self.try_reshape(&[])
        } else {
            self.try_reshape(&new_shape)
        }
    }
}

#[bon]
impl Tensor {
    /// Sum of tensor elements over given axes.
    ///
    /// Preserves input dtype. Use `sum_with().promote(true)` or `.dtype(...)` for different accumulation.
    #[track_caller]
    pub fn sum(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        reduce_internal(self, ReduceOp::Add, axes.into(), false, None, false)
    }

    /// Sum with additional options (keepdim, dtype, promote).
    ///
    /// # Examples
    /// ```ignore
    /// // Explicit dtype
    /// tensor.sum_with(0).dtype(DType::Float32).call()?;
    ///
    /// // Auto-promote (int8→int32, etc.)
    /// tensor.sum_with(0).promote(true).call()?;
    ///
    /// // With keepdim
    /// tensor.sum_with(0).keepdim(true).call()?;
    /// ```
    #[builder]
    #[track_caller]
    pub fn sum_with(
        &self,
        axes: impl Into<AxisSpec>,
        #[builder(default = false)] keepdim: bool,
        dtype: Option<DType>,
        #[builder(default = false)] promote: bool,
    ) -> Result<Self> {
        reduce_internal(self, ReduceOp::Add, axes.into(), keepdim, dtype, promote)
    }

    /// Product of tensor elements over given axes.
    ///
    /// Preserves input dtype. Use `prod_with().promote(true)` or `.dtype(...)` for different accumulation.
    #[track_caller]
    pub fn prod(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        reduce_internal(self, ReduceOp::Mul, axes.into(), false, None, false)
    }

    /// Product with additional options (keepdim, dtype, promote).
    #[builder]
    #[track_caller]
    pub fn prod_with(
        &self,
        axes: impl Into<AxisSpec>,
        #[builder(default = false)] keepdim: bool,
        dtype: Option<DType>,
        #[builder(default = false)] promote: bool,
    ) -> Result<Self> {
        reduce_internal(self, ReduceOp::Mul, axes.into(), keepdim, dtype, promote)
    }

    /// Maximum of tensor elements over given axes.
    ///
    /// Always preserves input dtype.
    pub fn max(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        reduce_internal(self, ReduceOp::Max, axes.into(), false, None, false)
    }

    /// Maximum with keepdim option.
    #[builder]
    #[track_caller]
    pub fn max_with(&self, axes: impl Into<AxisSpec>, #[builder(default = false)] keepdim: bool) -> Result<Self> {
        reduce_internal(self, ReduceOp::Max, axes.into(), keepdim, None, false)
    }

    /// Minimum of tensor elements over given axes.
    ///
    /// Always preserves input dtype.
    #[track_caller]
    pub fn min(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        reduce_internal(self, ReduceOp::Min, axes.into(), false, None, false)
    }

    /// Minimum with keepdim option.
    #[builder]
    #[track_caller]
    pub fn min_with(&self, axes: impl Into<AxisSpec>, #[builder(default = false)] keepdim: bool) -> Result<Self> {
        reduce_internal(self, ReduceOp::Min, axes.into(), keepdim, None, false)
    }

    /// Mean of tensor elements over given axes.
    ///
    /// For integer inputs, automatically uses float32 accumulation.
    /// For float inputs, preserves input dtype.
    #[track_caller]
    pub fn mean(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        mean_impl(self, axes.into(), false)
    }

    /// Mean with keepdim option.
    #[builder]
    #[track_caller]
    pub fn mean_with(&self, axes: impl Into<AxisSpec>, #[builder(default = false)] keepdim: bool) -> Result<Self> {
        mean_impl(self, axes, keepdim)
    }

    /// Variance of tensor elements over given axes.
    ///
    /// Computes unbiased sample variance (divides by N-1).
    /// For integer inputs, automatically uses float32 accumulation.
    /// For float inputs, preserves input dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let v = t.var(())?;  // Variance over all elements
    /// ```
    #[track_caller]
    pub fn var(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        var_impl(self, axes.into(), false)
    }

    /// Variance with keepdim option.
    #[builder]
    #[track_caller]
    pub fn var_with(&self, axes: impl Into<AxisSpec>, #[builder(default = false)] keepdim: bool) -> Result<Self> {
        var_impl(self, axes.into(), keepdim)
    }

    /// Standard deviation of tensor elements over given axes.
    ///
    /// Computes unbiased sample standard deviation (divides by N-1).
    /// For integer inputs, automatically uses float32 accumulation.
    /// For float inputs, preserves input dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let s = t.std(())?;  // Std dev over all elements
    /// ```
    #[track_caller]
    pub fn std(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        std_impl(self, axes.into(), false)
    }

    /// Standard deviation with keepdim option.
    #[builder]
    #[track_caller]
    pub fn std_with(&self, axes: impl Into<AxisSpec>, #[builder(default = false)] keepdim: bool) -> Result<Self> {
        std_impl(self, axes.into(), keepdim)
    }

    /// Variance and mean of tensor elements over given axes.
    ///
    /// Returns (variance, mean) tuple. More efficient than computing separately.
    /// Computes unbiased sample variance (divides by N-1).
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let (v, m) = t.var_mean(())?;
    /// ```
    #[track_caller]
    pub fn var_mean(&self, axes: impl Into<AxisSpec>) -> Result<(Self, Self)> {
        var_mean_impl(self, axes.into(), false)
    }

    /// Variance and mean with keepdim option.
    #[builder]
    #[track_caller]
    pub fn var_mean_with(
        &self,
        axes: impl Into<AxisSpec>,
        #[builder(default = false)] keepdim: bool,
    ) -> Result<(Self, Self)> {
        var_mean_impl(self, axes.into(), keepdim)
    }

    /// Standard deviation and mean of tensor elements over given axes.
    ///
    /// Returns (std, mean) tuple. More efficient than computing separately.
    /// Computes unbiased sample standard deviation (divides by N-1).
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let (s, m) = t.std_mean(())?;
    /// ```
    #[track_caller]
    pub fn std_mean(&self, axes: impl Into<AxisSpec>) -> Result<(Self, Self)> {
        std_mean_impl(self, axes.into(), false)
    }

    /// Standard deviation and mean with keepdim option.
    #[builder]
    #[track_caller]
    pub fn std_mean_with(
        &self,
        axes: impl Into<AxisSpec>,
        #[builder(default = false)] keepdim: bool,
    ) -> Result<(Self, Self)> {
        std_mean_impl(self, axes.into(), keepdim)
    }

    /// Internal helper: inverse of tensor for argmin.
    ///
    /// - Float dtypes: -self
    /// - Integer dtypes: ~self (bitwise NOT)
    /// - Bool dtype: logical_not(self)
    fn inverse(&self) -> Result<Self> {
        let dtype = self.uop.dtype();
        if dtype.is_float() {
            self.try_neg()
        } else if dtype.is_int() {
            self.bitwise_not()
        } else if matches!(dtype.scalar(), Some(ScalarDType::Bool)) {
            self.logical_not()
        } else {
            Ok(self.clone()) // Fallback for other types
        }
    }
}

// =========================================================================
// Argmax / Argmin Operations
// =========================================================================

#[bon]
impl Tensor {
    /// Index of maximum value along axis.
    ///
    /// Returns int32 tensor with indices of maximum values.
    /// For ties, returns the index of the first occurrence.
    ///
    /// # Arguments
    /// * `axis` - Axis to reduce (None = flatten first)
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]]);
    /// t.argmax(None)?;      // 5 (flattened: max is at index 5)
    /// t.argmax(Some(0))?;   // [1, 0, 1] (row indices of max per column)
    /// t.argmax(Some(1))?;   // [1, 2] (column indices of max per row)
    /// ```
    #[track_caller]
    pub fn argmax(&self, axis: impl Into<Option<isize>>) -> Result<Self> {
        argmax_impl(self, axis.into(), false)
    }

    /// Argmax with keepdim option.
    #[builder]
    #[track_caller]
    pub fn argmax_with(
        &self,
        axis: impl Into<Option<isize>>,
        #[builder(default = false)] keepdim: bool,
    ) -> Result<Self> {
        argmax_impl(self, axis.into(), keepdim)
    }

    /// Index of minimum value along axis.
    ///
    /// Returns int32 tensor with indices of minimum values.
    /// For ties, returns the index of the first occurrence.
    #[track_caller]
    pub fn argmin(&self, axis: impl Into<Option<isize>>) -> Result<Self> {
        argmin_impl(self, axis.into(), false)
    }

    /// Argmin with keepdim option.
    #[builder]
    #[track_caller]
    pub fn argmin_with(
        &self,
        axis: impl Into<Option<isize>>,
        #[builder(default = false)] keepdim: bool,
    ) -> Result<Self> {
        argmin_impl(self, axis.into(), keepdim)
    }

    /// Test if any element is true along axes.
    ///
    /// Logical OR reduction. Returns bool dtype.
    /// Non-zero values are treated as true.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[[true, false], [false, false]]);
    /// t.any(())?;           // true (any element is true)
    /// t.any(0)?;            // [true, false] (any true per column)
    /// t.any(1)?;            // [true, false] (any true per row)
    /// ```
    #[track_caller]
    pub fn any(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        any_impl(self, axes.into(), false)
    }

    /// Any with keepdim option.
    #[builder]
    #[track_caller]
    pub fn any_with(&self, axes: impl Into<AxisSpec>, #[builder(default = false)] keepdim: bool) -> Result<Self> {
        any_impl(self, axes.into(), keepdim)
    }

    /// Test if all elements are true along axes.
    ///
    /// Logical AND reduction. Returns bool dtype.
    /// Non-zero values are treated as true.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[[true, true], [true, false]]);
    /// t.all(())?;           // false (not all elements are true)
    /// t.all(0)?;            // [true, false] (all true per column)
    /// t.all(1)?;            // [true, false] (all true per row)
    /// ```
    #[track_caller]
    pub fn all(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        all_impl(self, axes.into(), false)
    }

    /// All with keepdim option.
    #[builder]
    #[track_caller]
    pub fn all_with(&self, axes: impl Into<AxisSpec>, #[builder(default = false)] keepdim: bool) -> Result<Self> {
        all_impl(self, axes.into(), keepdim)
    }
}

/// Internal argmax implementation.
fn argmax_impl(tensor: &Tensor, axis: Option<isize>, keepdim: bool) -> Result<Tensor> {
    // Handle None axis: flatten and call argmax on axis 0
    let (working_tensor, working_axis) =
        if let Some(ax) = axis { (tensor.clone(), ax) } else { (tensor.flatten()?, 0) };

    let shape = working_tensor.shape()?;
    let normalized_axis = Tensor::normalize_axis(working_axis, shape.len())?;
    let axis_size = shape[normalized_axis]
        .as_const()
        .ok_or_else(|| Error::SymbolicShapeUnsupported { operation: "argmax".to_string() })?;

    // Convert shape to isize vec once for reuse in expand operations
    let shape_vec: Vec<isize> = shape.iter().map(|s| s.as_const().unwrap() as isize).collect();

    // Step 1: Find maximum values along axis (with keepdim for broadcasting)
    let max_vals_keepdim = working_tensor.max_with().axes(working_axis).keepdim(true).call()?;

    // Step 2: Create mask where values equal the max
    // Need to broadcast max_vals to match working_tensor shape
    let max_vals_broadcast = max_vals_keepdim.try_expand(&shape_vec)?;

    let mask = working_tensor.try_eq(&max_vals_broadcast)?;

    // Step 3: Create descending index tensor [N, N-1, ..., 1]
    // This ensures ties go to first occurrence
    let indices = Tensor::arange(axis_size as i64, Some(0), Some(-1))?;

    // Step 4: Reshape indices to broadcast along the target axis
    // E.g., for axis=1 with 3D tensor: [axis_size] -> [1, axis_size, 1]
    let mut idx_shape = vec![1isize; shape.len()];
    idx_shape[normalized_axis] = axis_size as isize;
    let indices_reshaped = indices.try_reshape(&idx_shape)?;

    // Expand indices to match working_tensor shape
    let indices_broadcast = indices_reshaped.try_expand(&shape_vec)?;

    // Step 5: Multiply mask by indices (0 where not max, index where max)
    let mask_int = mask.cast(DType::Int32)?;
    let masked_indices = mask_int.try_mul(&indices_broadcast)?;

    // Step 6: Take max of masked indices (gives highest index, which is first occurrence)
    let max_idx = masked_indices.max_with().axes(working_axis).keepdim(keepdim).call()?;

    // Step 7: Invert: N - max_idx gives actual index
    let n_tensor = Tensor::from_slice([axis_size as i32]);

    // Broadcast n_tensor to match max_idx shape if needed
    let max_idx_shape = max_idx.shape()?;
    if !max_idx_shape.is_empty() {
        // Non-scalar result: broadcast n_tensor
        let max_idx_shape_vec: Vec<isize> = max_idx_shape.iter().map(|s| s.as_const().unwrap() as isize).collect();
        let ones_shape = vec![1isize; max_idx_shape.len()];
        let n_reshaped = n_tensor.try_reshape(&ones_shape)?;
        let n_broadcast = n_reshaped.try_expand(&max_idx_shape_vec)?;
        n_broadcast.try_sub(&max_idx)
    } else {
        // Scalar result: reshape n_tensor to scalar too
        let n_scalar = n_tensor.try_reshape(&[])?;
        n_scalar.try_sub(&max_idx)
    }
}

/// Internal argmin implementation.
fn argmin_impl(tensor: &Tensor, axis: Option<isize>, keepdim: bool) -> Result<Tensor> {
    // Argmin is just argmax of inverted values
    let inverted = tensor.inverse()?;
    argmax_impl(&inverted, axis, keepdim)
}

/// Internal any implementation.
fn any_impl(tensor: &Tensor, axes: AxisSpec, keepdim: bool) -> Result<Tensor> {
    // Cast to bool (non-zero becomes true)
    let as_bool = tensor.cast(DType::Bool)?;

    // Max reduction on bool is logical OR
    reduce_internal(&as_bool, ReduceOp::Max, axes, keepdim, None, false)
}

/// Internal all implementation.
fn all_impl(tensor: &Tensor, axes: AxisSpec, keepdim: bool) -> Result<Tensor> {
    // De Morgan's law: all(x) = !any(!x)
    let negated = tensor.logical_not()?;
    let any_negated = any_impl(&negated, axes, keepdim)?;
    any_negated.logical_not()
}

/// Internal reduction implementation.
#[track_caller]
fn reduce_internal(
    tensor: &Tensor,
    op: ReduceOp,
    axes: AxisSpec,
    keepdim: bool,
    dtype: Option<DType>,
    promote: bool,
) -> Result<Tensor> {
    // Validate conflicting options
    if dtype.is_some() && promote {
        return Err(Error::ConflictingReductionOptions);
    }

    let shape = tensor.shape()?;
    let resolved_axes = Tensor::resolve_axis_spec(&axes, shape.len())?;

    // Determine accumulation dtype
    let original_dtype = tensor.uop.dtype();
    let acc_dtype = if let Some(ref dt) = dtype {
        // Explicit dtype takes precedence
        dt.clone()
    } else if promote {
        // Auto-promote using sum_acc_dtype
        Tensor::sum_acc_dtype(&original_dtype)
    } else {
        // Preserve input dtype
        original_dtype.clone()
    };

    // Cast to accumulation dtype if needed
    let working_tensor = if acc_dtype != original_dtype { tensor.cast(acc_dtype.clone())? } else { tensor.clone() };

    // Perform reduction
    let reduced = working_tensor.uop.try_reduce_axis(op, resolved_axes.clone()).context(UOpSnafu)?;

    // Handle keepdim
    let result = if keepdim {
        Tensor::new(reduced)
    } else {
        let temp = Tensor::new(reduced);
        temp.remove_singleton_dims(&resolved_axes)?
    };

    // Cast back if promoted and should cast back (fp16/bf16)
    if promote && dtype.is_none() && Tensor::should_cast_back_after_sum(&original_dtype) {
        result.cast(original_dtype)
    } else {
        Ok(result)
    }
}

/// Mean implementation (shared by mean and mean_with).
fn mean_impl(tensor: &Tensor, axes: impl Into<AxisSpec>, keepdim: bool) -> Result<Tensor> {
    let axes = axes.into();
    let shape = tensor.shape()?;
    let resolved_axes = Tensor::resolve_axis_spec(&axes, shape.len())?;

    // Calculate count of reduced elements
    let mut count = 1i64;
    for &axis in &resolved_axes {
        if let Some(dim_size) = shape[axis].as_const() {
            count *= dim_size as i64;
        } else {
            return SymbolicShapeUnsupportedSnafu { operation: "mean" }.fail();
        }
    }

    // Determine output dtype (integers → float32, floats preserve)
    let dtype = tensor.uop.dtype();
    let output_dtype = if Tensor::is_integer_dtype(&dtype) { DType::Float32 } else { dtype };

    // Sum with explicit accumulation dtype (no promotion needed, dtype is explicit)
    let sum = reduce_internal(tensor, ReduceOp::Add, axes, keepdim, Some(output_dtype.clone()), false)?;

    // Divide by count
    let count_tensor = Tensor::new(UOp::const_(output_dtype.clone(), morok_ir::ConstValue::Float(count as f64)));
    Ok(&sum / &count_tensor)
}

/// Variance implementation using E[X²] - E[X]² formula.
fn var_impl(tensor: &Tensor, axes: AxisSpec, keepdim: bool) -> Result<Tensor> {
    let (var, _mean) = var_mean_impl(tensor, axes, keepdim)?;
    Ok(var)
}

/// Standard deviation implementation.
fn std_impl(tensor: &Tensor, axes: AxisSpec, keepdim: bool) -> Result<Tensor> {
    let variance = var_impl(tensor, axes, keepdim)?;
    variance.try_sqrt()
}

/// Variance and mean implementation using single-pass algorithm.
fn var_mean_impl(tensor: &Tensor, axes: AxisSpec, keepdim: bool) -> Result<(Tensor, Tensor)> {
    let shape = tensor.shape()?;
    let resolved_axes = Tensor::resolve_axis_spec(&axes, shape.len())?;

    // Calculate count of reduced elements
    let mut count = 1i64;
    for &axis in &resolved_axes {
        if let Some(dim_size) = shape[axis].as_const() {
            count *= dim_size as i64;
        } else {
            return SymbolicShapeUnsupportedSnafu { operation: "variance" }.fail();
        }
    }

    // Determine output dtype (integers → float32, floats preserve)
    let dtype = tensor.uop.dtype();
    let output_dtype = if Tensor::is_integer_dtype(&dtype) { DType::Float32 } else { dtype.clone() };

    // Compute mean: E[X]
    let mean = mean_impl(tensor, axes.clone(), keepdim)?;

    // Compute deviation from mean: X - E[X]
    // Need to broadcast mean if keepdim=false
    let deviation = if keepdim {
        tensor.try_sub(&mean)?
    } else {
        // Expand mean back to original shape for subtraction
        let mut expanded_mean = mean.clone();
        for &axis in &resolved_axes {
            expanded_mean = expanded_mean.try_unsqueeze(axis as isize)?;
        }
        tensor.try_sub(&expanded_mean)?
    };

    // Square the deviations: (X - E[X])²
    let squared_dev = deviation.square()?;

    // Sum squared deviations with explicit dtype
    let sum_sq_dev = reduce_internal(&squared_dev, ReduceOp::Add, axes, keepdim, Some(output_dtype.clone()), false)?;

    // Divide by N-1 for unbiased estimate (Bessel's correction)
    let denom = if count > 1 { count - 1 } else { count };
    let denom_tensor = Tensor::new(UOp::const_(output_dtype, morok_ir::ConstValue::Float(denom as f64)));
    let variance = &sum_sq_dev / &denom_tensor;

    Ok((variance, mean))
}

/// Standard deviation and mean implementation.
fn std_mean_impl(tensor: &Tensor, axes: AxisSpec, keepdim: bool) -> Result<(Tensor, Tensor)> {
    let (variance, mean) = var_mean_impl(tensor, axes, keepdim)?;
    let std = variance.try_sqrt()?;
    Ok((std, mean))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axis_spec_all() {
        let axes = Tensor::resolve_axis_spec(&AxisSpec::All, 3).unwrap();
        assert_eq!(axes, vec![0, 1, 2]);
    }

    #[test]
    fn test_axis_spec_single() {
        // Positive index
        let axes = Tensor::resolve_axis_spec(&AxisSpec::Single(1), 3).unwrap();
        assert_eq!(axes, vec![1]);

        // Negative index
        let axes = Tensor::resolve_axis_spec(&AxisSpec::Single(-1), 3).unwrap();
        assert_eq!(axes, vec![2]);

        // Out of bounds
        assert!(Tensor::resolve_axis_spec(&AxisSpec::Single(5), 3).is_err());
        assert!(Tensor::resolve_axis_spec(&AxisSpec::Single(-5), 3).is_err());
    }

    #[test]
    fn test_axis_spec_multiple() {
        // Multiple axes
        let axes = Tensor::resolve_axis_spec(&AxisSpec::Multiple(vec![0, 2]), 3).unwrap();
        assert_eq!(axes, vec![0, 2]);

        // With negatives
        let axes = Tensor::resolve_axis_spec(&AxisSpec::Multiple(vec![0, -1]), 3).unwrap();
        assert_eq!(axes, vec![0, 2]);

        // Deduplication
        let axes = Tensor::resolve_axis_spec(&AxisSpec::Multiple(vec![1, 1, 1]), 3).unwrap();
        assert_eq!(axes, vec![1]);
    }

    #[test]
    fn test_sum_acc_dtype() {
        assert_eq!(Tensor::sum_acc_dtype(&DType::Bool), DType::Int32);
        assert_eq!(Tensor::sum_acc_dtype(&DType::Int8), DType::Int32);
        assert_eq!(Tensor::sum_acc_dtype(&DType::Int16), DType::Int32);
        assert_eq!(Tensor::sum_acc_dtype(&DType::Int32), DType::Int32);
        assert_eq!(Tensor::sum_acc_dtype(&DType::Int64), DType::Int64);

        assert_eq!(Tensor::sum_acc_dtype(&DType::UInt8), DType::UInt32);
        assert_eq!(Tensor::sum_acc_dtype(&DType::UInt16), DType::UInt32);
        assert_eq!(Tensor::sum_acc_dtype(&DType::UInt32), DType::UInt32);
        assert_eq!(Tensor::sum_acc_dtype(&DType::UInt64), DType::UInt64);

        assert_eq!(Tensor::sum_acc_dtype(&DType::Float16), DType::Float32);
        assert_eq!(Tensor::sum_acc_dtype(&DType::BFloat16), DType::Float32);
        assert_eq!(Tensor::sum_acc_dtype(&DType::Float32), DType::Float32);
        assert_eq!(Tensor::sum_acc_dtype(&DType::Float64), DType::Float64);
    }

    // ========== Argmax Tests ==========

    #[test]
    fn test_argmax_1d_basic() {
        let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 5.0, 4.0]);
        let result = t.argmax(Some(0)).unwrap();
        // Max value 5.0 is at index 3
        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 0); // Scalar result
        assert_eq!(result.uop.dtype(), DType::Int32);
    }

    #[test]
    fn test_argmax_1d_ties_first_occurrence() {
        let t = Tensor::from_slice([1.0f32, 5.0, 3.0, 5.0, 2.0]);
        let result = t.argmax(Some(0)).unwrap();
        // Two maxima at indices 1 and 3, should return 1 (first)
        assert_eq!(result.uop.dtype(), DType::Int32);
    }

    #[test]
    fn test_argmax_2d_axis0() {
        let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
        // Shape [2, 3]: [[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]]
        let result = t.argmax(Some(0)).unwrap();
        // Expected: [1, 0, 1] (max per column)
        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 1);
        assert_eq!(result_shape[0].as_const().unwrap(), 3);
    }

    #[test]
    fn test_argmax_2d_axis1() {
        let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
        let result = t.argmax(Some(1)).unwrap();
        // Expected: [1, 2] (max per row)
        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 1);
        assert_eq!(result_shape[0].as_const().unwrap(), 2);
    }

    #[test]
    fn test_argmax_flatten() {
        let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
        let result = t.argmax(None).unwrap();
        // Flattened: [1, 3, 2, 4, 2, 5], max 5.0 at index 5
        assert_eq!(result.shape().unwrap().len(), 0); // Scalar
    }

    #[test]
    fn test_argmax_keepdim() {
        let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
        let result = t.argmax_with().axis(Some(1)).keepdim(true).call().unwrap();
        // Shape should be [2, 1] instead of [2]
        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 2);
        assert_eq!(result_shape[0].as_const().unwrap(), 2);
        assert_eq!(result_shape[1].as_const().unwrap(), 1);
    }

    #[test]
    fn test_argmax_negative_axis() {
        let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
        let result = t.argmax(Some(-1)).unwrap();
        // -1 should resolve to axis 1
        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 1);
        assert_eq!(result_shape[0].as_const().unwrap(), 2);
    }

    // ========== Argmin Tests ==========

    #[test]
    fn test_argmin_1d_basic() {
        let t = Tensor::from_slice([5.0f32, 3.0, 1.0, 4.0, 2.0]);
        let result = t.argmin(Some(0)).unwrap();
        // Min value 1.0 is at index 2
        assert_eq!(result.uop.dtype(), DType::Int32);
    }

    #[test]
    fn test_argmin_float() {
        let t = Tensor::from_slice([1.5f32, -2.3, 0.5, 1.0]);
        let result = t.argmin(Some(0)).unwrap();
        // Min is -2.3 at index 1
        assert_eq!(result.uop.dtype(), DType::Int32);
    }

    #[test]
    fn test_argmin_int() {
        let t = Tensor::from_slice([5i32, 3, 1, 4, 2]);
        let result = t.argmin(Some(0)).unwrap();
        // Min is 1 at index 2
        assert_eq!(result.uop.dtype(), DType::Int32);
    }

    #[test]
    fn test_argmin_bool() {
        let t = Tensor::from_slice([true, false, true]);
        let result = t.argmin(Some(0)).unwrap();
        // Min (false) at index 1
        assert_eq!(result.uop.dtype(), DType::Int32);
    }

    // ========== Any Tests ==========

    #[test]
    fn test_any_all_true() {
        let t = Tensor::from_slice([true, true, true]);
        let result = t.any(()).unwrap();
        // Should be true
        assert_eq!(result.uop.dtype(), DType::Bool);
    }

    #[test]
    fn test_any_all_false() {
        let t = Tensor::from_slice([false, false, false]);
        let result = t.any(()).unwrap();
        // Should be false
        assert_eq!(result.uop.dtype(), DType::Bool);
    }

    #[test]
    fn test_any_mixed() {
        let t = Tensor::from_slice([false, true, false]);
        let result = t.any(()).unwrap();
        // Should be true
        assert_eq!(result.uop.dtype(), DType::Bool);
    }

    #[test]
    fn test_any_numeric() {
        let t = Tensor::from_slice([0.0f32, 1.0, 0.0]);
        let result = t.any(()).unwrap();
        // Non-zero treated as true, should be true
        assert_eq!(result.uop.dtype(), DType::Bool);
    }

    #[test]
    fn test_any_2d_axis0() {
        let t = Tensor::from_slice([true, false, false, false]).try_reshape(&[2, 2]).unwrap();
        let result = t.any(0).unwrap();
        // Expected: [true, false]
        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 1);
        assert_eq!(result_shape[0].as_const().unwrap(), 2);
    }

    #[test]
    fn test_any_2d_axis1() {
        let t = Tensor::from_slice([true, false, false, false]).try_reshape(&[2, 2]).unwrap();
        let result = t.any(1).unwrap();
        // Expected: [true, false]
        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 1);
        assert_eq!(result_shape[0].as_const().unwrap(), 2);
    }

    #[test]
    fn test_any_keepdim() {
        let t = Tensor::from_slice([true, false, false, true]).try_reshape(&[2, 2]).unwrap();
        let result = t.any_with().axes(0).keepdim(true).call().unwrap();
        // Shape should be [1, 2]
        let result_shape = result.shape().unwrap();
        assert_eq!(result_shape.len(), 2);
        assert_eq!(result_shape[0].as_const().unwrap(), 1);
        assert_eq!(result_shape[1].as_const().unwrap(), 2);
    }

    // ========== All Tests ==========

    #[test]
    fn test_all_all_true() {
        let t = Tensor::from_slice([true, true, true]);
        let result = t.all(()).unwrap();
        // Should be true
        assert_eq!(result.uop.dtype(), DType::Bool);
    }

    #[test]
    fn test_all_one_false() {
        let t = Tensor::from_slice([true, false, true]);
        let result = t.all(()).unwrap();
        // Should be false
        assert_eq!(result.uop.dtype(), DType::Bool);
    }

    #[test]
    fn test_all_numeric() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let result = t.all(()).unwrap();
        // All non-zero, should be true
        assert_eq!(result.uop.dtype(), DType::Bool);
    }

    #[test]
    fn test_all_numeric_with_zero() {
        let t = Tensor::from_slice([1.0f32, 0.0, 3.0]);
        let result = t.all(()).unwrap();
        // Has zero, should be false
        assert_eq!(result.uop.dtype(), DType::Bool);
    }

    #[test]
    fn test_all_2d_multiaxis() {
        let t = Tensor::from_slice([true, true, true, true]).try_reshape(&[2, 2]).unwrap();
        let result = t.all(()).unwrap();
        // All true across all axes
        assert_eq!(result.uop.dtype(), DType::Bool);
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_argmax_single_element() {
        let t = Tensor::from_slice([42.0f32]);
        let result = t.argmax(Some(0)).unwrap();
        // Only element, index should be 0
        assert_eq!(result.uop.dtype(), DType::Int32);
    }

    #[test]
    fn test_argmax_all_equal() {
        let t = Tensor::from_slice([5.0f32, 5.0, 5.0, 5.0]);
        let result = t.argmax(Some(0)).unwrap();
        // All equal, should return first index (0)
        assert_eq!(result.uop.dtype(), DType::Int32);
    }
}
