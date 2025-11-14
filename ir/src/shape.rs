use crate::{Op, UOp};
use std::rc::Rc;

/// Infer the shape of a UOp if possible.
///
/// Returns `Some(shape)` if the shape can be determined from the operation
/// and its inputs, `None` otherwise.
///
/// Shape inference rules:
/// - Const: scalar (shape `[]`)
/// - VConst: vector with count elements (shape `[count]`)
/// - Unary ops: preserve input shape
/// - Binary ops: preserve input shape (assumed matching)
/// - Cast/BitCast: preserve input shape
/// - Vectorize: create shape `[count]`
/// - Gep: scalar (extract element from vector)
/// - Movement ops: compute shape from operation arguments
/// - Reduce/ReduceAxis: compute reduced shape
/// - Others: unknown (returns `None`)
pub fn infer_shape(uop: &Rc<UOp>) -> Option<Vec<usize>> {
    match uop.op() {
        // Nullary operations
        Op::Const(_) => Some(vec![]), // Scalar
        Op::Unique(_) | Op::Device(_) | Op::Noop => None,
        Op::DefineGlobal(_) | Op::DefineLocal(_) => None,

        // Special operations
        Op::MSelect { .. } => None,
        Op::Special { .. } => Some(vec![]), // Special returns index (scalar)

        // VConst creates a vector
        Op::VConst { values } => Some(vec![values.len()]),

        // Unary operations preserve shape
        Op::Unary(_, input) => infer_shape(input),

        // Binary operations preserve shape (we assume inputs have matching shapes)
        Op::Binary(_, lhs, _) => infer_shape(lhs),

        // Ternary operations preserve shape
        Op::Ternary(_, _, true_val, _) => infer_shape(true_val),

        // Type operations preserve shape
        Op::Cast { src, .. } | Op::BitCast { src, .. } => infer_shape(src),

        // Vectorize creates a vector
        Op::Vectorize { elements } => {
            // Vectorize creates a vector from elements
            Some(vec![elements.len()])
        }

        // Gep extracts element from vector
        Op::Gep { .. } => Some(vec![]), // Extracts scalar from vector

        // Movement operations with dynamic shapes
        // These now use Rc<UOp> for shapes, so we can't statically infer them
        Op::Reshape { .. } => None,
        Op::Expand { .. } => None,
        Op::Pad { .. } => None,
        Op::Shrink { .. } => None,

        Op::Permute { axes, src } => {
            let src_shape = infer_shape(src)?;
            if axes.len() != src_shape.len() {
                return None;
            }
            // Reorder dimensions according to permutation
            Some(axes.iter().map(|&i| src_shape[i]).collect())
        }

        Op::Flip { src, .. } => {
            // Flip preserves shape
            infer_shape(src)
        }

        // Reduce operations
        Op::ReduceAxis { axes, src, .. } => {
            let src_shape = infer_shape(src)?;
            // Remove reduced axes
            Some(src_shape.iter().enumerate().filter(|(i, _)| !axes.contains(i)).map(|(_, &dim)| dim).collect())
        }

        Op::Reduce { .. } => {
            // Reduce collapses to scalar or smaller shape - context dependent
            None
        }

        // Buffer operations - shapes are dynamic
        Op::Buffer { .. }
        | Op::BufferView { .. }
        | Op::Bufferize { .. }
        | Op::Index { .. }
        | Op::Copy { .. }
        | Op::MStack { .. } => None,

        // Control flow - no static shape
        Op::If { .. } | Op::EndIf { .. } | Op::Range { .. } | Op::End { .. } | Op::Barrier { .. } => None,

        // Memory operations - shape depends on buffer
        Op::Load { .. } | Op::LoadGated { .. } | Op::Store { .. } | Op::StoreGated { .. } => None,

        // Advanced operations - shape depends on context
        Op::Wmma { .. }
        | Op::Contract { .. }
        | Op::Unroll { .. }
        | Op::Kernel { .. }
        | Op::Assign { .. }
        | Op::Detach { .. }
        | Op::Contiguous { .. }
        | Op::ContiguousBackward { .. }
        | Op::After { .. }
        | Op::Precast { .. }
        | Op::Custom { .. }
        | Op::CustomI { .. }
        | Op::AllReduce { .. }
        | Op::DefineVar { .. }
        | Op::Bind { .. }
        | Op::DefineReg { .. }
        | Op::Multi { .. } => None,
    }
}

/// Check if two shapes are compatible for elementwise operations.
///
/// Shapes are compatible if they are either:
/// 1. Exactly equal
/// 2. One or both are unknown (None)
/// 3. Broadcasting rules apply (not implemented yet)
pub fn shapes_compatible(lhs: Option<&[usize]>, rhs: Option<&[usize]>) -> bool {
    match (lhs, rhs) {
        (None, _) | (_, None) => true, // Unknown shapes are assumed compatible
        (Some(l), Some(r)) => l == r,  // Exact match required for now
    }
}

/// Validate that a shape specification is valid (all positive, no zeros).
pub fn validate_shape(shape: &[isize]) -> Result<Vec<usize>, crate::Error> {
    use crate::error::ReshapeNegativeDimensionSnafu;
    use snafu::ensure;

    ensure!(shape.iter().all(|&s| s > 0), ReshapeNegativeDimensionSnafu { shape: shape.to_vec() });

    Ok(shape.iter().map(|&s| s as usize).collect())
}

/// Compute product of shape dimensions.
pub fn shape_product(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstValue, UOp};
    use morok_dtype::DType;

    #[test]
    fn test_infer_const_shape() {
        let uop = UOp::const_(DType::float32(), ConstValue::Float(42.0));
        assert_eq!(infer_shape(&uop), Some(vec![]));
    }

    #[test]
    fn test_infer_unary_shape() {
        let uop = UOp::const_(DType::float32(), ConstValue::Float(42.0));
        let neg = UOp::new(Op::Unary(crate::UnaryOp::Neg, uop), DType::float32());
        assert_eq!(infer_shape(&neg), Some(vec![]));
    }

    #[test]
    fn test_infer_vectorize_shape() {
        let uop1 = UOp::const_(DType::float32(), ConstValue::Float(1.0));
        let uop2 = UOp::const_(DType::float32(), ConstValue::Float(2.0));
        let uop3 = UOp::const_(DType::float32(), ConstValue::Float(3.0));
        let uop4 = UOp::const_(DType::float32(), ConstValue::Float(4.0));
        let vec = UOp::new(
            Op::Vectorize { elements: smallvec::SmallVec::from_vec(vec![uop1, uop2, uop3, uop4]) },
            DType::float32().vec(4),
        );
        assert_eq!(infer_shape(&vec), Some(vec![4]));
    }

    #[test]
    fn test_shape_product() {
        assert_eq!(shape_product(&[2, 3, 4]), 24);
        assert_eq!(shape_product(&[]), 1);
        assert_eq!(shape_product(&[5]), 5);
    }

    #[test]
    fn test_shapes_compatible() {
        assert!(shapes_compatible(Some(&[2, 3]), Some(&[2, 3])));
        assert!(!shapes_compatible(Some(&[2, 3]), Some(&[2, 4])));
        assert!(shapes_compatible(None, Some(&[2, 3])));
        assert!(shapes_compatible(Some(&[2, 3]), None));
    }
}
