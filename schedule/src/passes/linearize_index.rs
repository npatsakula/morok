//! Multi-index linearization pass.
//!
//! Transforms `INDEX(buffer, [i, j, k])` → `INDEX(buffer, [linear_offset])`
//! using row-major linearization.
//!
//! This moves the multi-index → linear offset computation from codegen
//! to schedule, eliminating duplicated logic in LLVM and Cranelift backends.
//!
//! # Row-Major Linearization
//!
//! For a 3D index `[i, j, k]` with dimensions `[D0, D1, D2]`:
//! - Strides: `[D1*D2, D2, 1]`
//! - Linear offset: `i*(D1*D2) + j*D2 + k`
//!
//! # Vectorized Indices
//!
//! When any index is vectorized (from UPCAST), the linearization is applied
//! element-wise, producing a vector of linear offsets for gather/scatter.

use std::sync::Arc;

use morok_ir::{BinaryOp, ConstValue, DType, Op, UOp};
use smallvec::SmallVec;
use tracing::trace;

use crate::TypedPatternMatcher;

/// Extract the dimension (iteration count) from an index UOp.
///
/// Uses vmin/vmax properties to determine the range of values an index can take.
/// This is more robust than pattern matching on specific op types.
pub(crate) fn extract_index_dimension(idx_uop: &Arc<UOp>) -> i64 {
    use morok_ir::uop::cached_property::CachedProperty;
    use morok_ir::uop::properties::VminVmaxProperty;

    let (vmin, vmax) = VminVmaxProperty::get(idx_uop);

    match (vmin, vmax) {
        (ConstValue::Int(min), ConstValue::Int(max)) => {
            // Dimension is the number of distinct values: max - min + 1
            // But for strides, we need the "size" which is typically max + 1 for 0-based indices
            // Since most indices are 0-based (vmin=0), this gives max + 1
            (max - min + 1).max(1)
        }
        _ => {
            // Fallback for non-integer types or unknown bounds.
            // This mirrors Tinygrad which throws RuntimeError for missing shapes,
            // but we warn instead to avoid breaking compilation.
            // TODO: Consider making this an ICE once all index sources have proper bounds.
            trace!(
                uop_id = idx_uop.id,
                vmin = ?vmin,
                vmax = ?vmax,
                "extract_index_dimension: fallback to 1 for unknown bounds"
            );
            1
        }
    }
}

/// Compute row-major strides from dimensions.
///
/// For dims `[D0, D1, D2]`, strides are `[D1*D2, D2, 1]`.
pub(crate) fn compute_row_major_strides(dims: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Check if any index in the list is vectorized.
fn any_index_vectorized(indices: &[Arc<UOp>]) -> bool {
    indices.iter().any(|idx| idx.dtype().vcount() > 1)
}

/// Get the vector count from the first vectorized index, or 1 if none.
fn get_vector_count(indices: &[Arc<UOp>]) -> usize {
    indices
        .iter()
        .find_map(|idx| {
            let vc = idx.dtype().vcount();
            if vc > 1 { Some(vc) } else { None }
        })
        .unwrap_or(1)
}

/// Build a linear index expression from multi-dimensional indices and strides.
///
/// Computes: `indices[0] * strides[0] + indices[1] * strides[1] + ...`
pub(crate) fn build_linear_index(indices: &[Arc<UOp>], strides: &[i64]) -> Arc<UOp> {
    // Start with zero
    let mut linear = UOp::index_const(0);

    for (idx, &stride) in indices.iter().zip(strides.iter()) {
        if stride == 0 {
            // Skip zero-stride dimensions (scalar broadcast)
            continue;
        }

        let term = if stride == 1 {
            // Optimization: avoid multiplication by 1
            idx.clone()
        } else {
            let stride_const = UOp::index_const(stride);
            UOp::new(Op::Binary(BinaryOp::Mul, idx.clone(), stride_const), DType::Index)
        };

        // Check if linear is still zero (first iteration)
        if let Op::Const(cv) = linear.op()
            && matches!(cv.0, ConstValue::Int(0))
        {
            linear = term;
        } else {
            linear = UOp::new(Op::Binary(BinaryOp::Add, linear, term), DType::Index);
        }
    }

    linear
}

/// Build a vectorized linear index for UPCAST patterns.
///
/// When indices are vectorized, extracts each lane and computes
/// linearization per-lane, then assembles into a vector result.
fn build_vectorized_linear_index(indices: &[Arc<UOp>], strides: &[i64], vcount: usize) -> Arc<UOp> {
    // For each lane, extract scalar indices and compute linear offset
    let lane_indices: SmallVec<[Arc<UOp>; 4]> = (0..vcount)
        .map(|lane| {
            let scalar_indices: Vec<Arc<UOp>> = indices
                .iter()
                .map(|idx| {
                    if idx.dtype().vcount() > 1 {
                        // Extract scalar from vector
                        UOp::gep(idx.clone(), vec![lane])
                    } else {
                        // Scalar index, use directly
                        idx.clone()
                    }
                })
                .collect();
            build_linear_index(&scalar_indices, strides)
        })
        .collect();

    // Assemble into vector using VECTORIZE
    UOp::vectorize(lane_indices)
}

/// Pattern matcher to linearize multi-index INDEX operations.
///
/// Transforms:
/// - `INDEX(buffer, [i, j, k])` → `INDEX(buffer, [linear])`
///
/// Where `linear = i * (D1*D2) + j * D2 + k` for row-major layout.
///
/// This eliminates backend-specific linearization in LLVM/Cranelift codegen.
pub fn pm_linearize_multi_index() -> TypedPatternMatcher<()> {
    crate::patterns! {
        // Match INDEX with multiple indices
        idx @ Index { buffer, indices, gate } if indices.len() > 1 => |idx, buffer, indices, gate| {
            // Extract dimensions from each index
            let dims: Vec<i64> = indices.iter().map(extract_index_dimension).collect();

            trace!(
                uop_id = idx.id,
                dims = ?dims,
                "linearize_multi_index: linearizing {}-dimensional index",
                indices.len()
            );

            // Compute row-major strides
            let strides = compute_row_major_strides(&dims);

            // Check if any index is vectorized
            let is_vectorized = any_index_vectorized(indices);

            let linear_index = if is_vectorized {
                let vcount = get_vector_count(indices);
                trace!(uop_id = idx.id, vcount, "linearize_multi_index: vectorized indices");
                build_vectorized_linear_index(indices, &strides, vcount)
            } else {
                build_linear_index(indices, &strides)
            };

            // Create new INDEX with single linear index, preserving gate and dtype
            let new_op = Op::Index {
                buffer: buffer.clone(),
                indices: smallvec::smallvec![linear_index],
                gate: gate.clone(),
            };

            Some(UOp::new(new_op, idx.dtype().clone()))
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_dimension_from_range() {
        use morok_ir::{AxisId, AxisType};
        let end = UOp::index_const(10);
        let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);
        assert_eq!(extract_index_dimension(&range), 10);
    }

    #[test]
    fn test_extract_dimension_from_define_var() {
        // DefineVar with min=1, max=99 has 99 distinct values (1..=99)
        let var = UOp::new(Op::DefineVar { name: "n".to_string(), min_val: 1, max_val: 99 }, DType::Index);
        assert_eq!(extract_index_dimension(&var), 99);

        // DefineVar with min=0, max=9 has 10 distinct values (0..=9)
        let var_zero_based = UOp::new(Op::DefineVar { name: "m".to_string(), min_val: 0, max_val: 9 }, DType::Index);
        assert_eq!(extract_index_dimension(&var_zero_based), 10);
    }

    #[test]
    fn test_compute_row_major_strides() {
        // 3D tensor [2, 3, 4]: strides should be [12, 4, 1]
        assert_eq!(compute_row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);

        // 2D matrix [5, 10]: strides should be [10, 1]
        assert_eq!(compute_row_major_strides(&[5, 10]), vec![10, 1]);

        // 1D: stride is [1]
        assert_eq!(compute_row_major_strides(&[100]), vec![1]);
    }

    #[test]
    fn test_build_linear_index() {
        let i = UOp::index_const(2);
        let j = UOp::index_const(3);
        let linear = build_linear_index(&[i, j], &[10, 1]);

        // Should produce: 2*10 + 3 = Add(Mul(2, 10), 3)
        assert!(matches!(linear.op(), Op::Binary(BinaryOp::Add, _, _)));
    }
}
