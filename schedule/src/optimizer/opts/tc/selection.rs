//! Tensor core selection logic.
//!
//! Filters available tensor cores by dtype compatibility and selects the best match.

use super::matching::MatmulPattern;
use crate::optimizer::{Renderer, error::*};
use morok_ir::{Op, UOp};
use std::sync::Arc;

/// Result of tensor core selection.
#[derive(Debug, Clone)]
pub struct TcSelection {
    /// Index of selected tensor core in renderer's tensor_cores list.
    pub tc_index: usize,
    /// Selected axis triple (N_range, M_range, K_range).
    pub axes: (Arc<UOp>, Arc<UOp>, Arc<UOp>),
}

/// Select appropriate tensor core for the given matmul pattern.
///
/// # Arguments
///
/// * `pattern` - Detected matmul pattern
/// * `renderer` - Backend renderer with available tensor cores
/// * `tc_select` - Tensor core selection index (-1 for auto, >= 0 for specific TC)
/// * `axis_choice` - Which axis combination to try (index into pattern.axis_choices)
///
/// # Returns
///
/// - `Ok(Some(selection))` if a compatible tensor core was found
/// - `Ok(None)` if no compatible tensor core available
/// - `Err(_)` on validation errors
pub fn select_tensor_core(
    pattern: &MatmulPattern,
    renderer: &Renderer,
    tc_select: i32,
    axis_choice: usize,
) -> Result<Option<TcSelection>, OptError> {
    // 1. Determine which tensor cores to try
    let tensor_cores = if tc_select == -1 {
        // Auto-select: try all tensor cores
        &renderer.tensor_cores[..]
    } else {
        // Specific selection
        let idx = tc_select as usize;
        if idx >= renderer.tensor_cores.len() {
            return ValidationFailedSnafu { op: "TC", reason: "tc_select index out of bounds" }.fail();
        }
        &renderer.tensor_cores[idx..idx + 1]
    };

    // 2. Get scalar dtypes from pattern
    let in0_scalar = pattern.in0.dtype().scalar();
    let in1_scalar = pattern.in1.dtype().scalar();
    let out_scalar = pattern.reduce_op.dtype().scalar();

    // 3. Try each tensor core
    for (tc_idx, tc) in tensor_cores.iter().enumerate() {
        // Check dtype compatibility (compare scalar types)
        let tc_in_scalar = tc.dtype_in.scalar();
        let tc_out_scalar = tc.dtype_out.scalar();

        if in0_scalar != tc_in_scalar || in1_scalar != tc_in_scalar {
            continue; // Input dtypes don't match
        }
        if out_scalar != tc_out_scalar {
            continue; // Output dtype doesn't match
        }

        // 4. Try the specified axis choice
        if axis_choice >= pattern.axis_choices.len() {
            continue; // Invalid axis choice
        }

        let axes = pattern.axis_choices[axis_choice].clone();

        // 5. Validate dimensions are compatible
        if !check_dimension_compatibility(&axes, tc) {
            continue; // Dimensions don't match or aren't divisible
        }

        // Found a match!
        let actual_tc_idx = if tc_select == -1 {
            // Find actual index in renderer.tensor_cores
            renderer.tensor_cores.iter().position(|t| std::ptr::eq(t, tc)).unwrap_or(tc_idx)
        } else {
            tc_select as usize
        };

        return Ok(Some(TcSelection { tc_index: actual_tc_idx, axes }));
    }

    // No compatible tensor core found
    Ok(None)
}

/// Check if the given axes are compatible with the tensor core dimensions.
///
/// Axes must be divisible by the tensor core's dimension requirements,
/// or be paddable (checked later during application).
fn check_dimension_compatibility(
    axes: &(Arc<UOp>, Arc<UOp>, Arc<UOp>),
    _tc: &crate::optimizer::renderer::TensorCore,
) -> bool {
    // Extract sizes from ranges
    let (n_range, m_range, k_range) = axes;

    // Get sizes (if constant)
    let _n_size = get_range_size(n_range);
    let _m_size = get_range_size(m_range);
    let _k_size = get_range_size(k_range);

    // For now, accept all ranges - divisibility will be checked during application
    // and PADTO will be applied if tc_opt >= 2
    // TODO: Could add stricter checking here if needed
    true
}

/// Get the size of a RANGE UOp (if constant).
fn get_range_size(range: &Arc<UOp>) -> Option<i64> {
    if let Op::Range { end, .. } = range.op()
        && let Op::Const(cv) = end.op()
        && let morok_ir::ConstValue::Int(size) = cv.0
    {
        return Some(size);
    }
    None
}
