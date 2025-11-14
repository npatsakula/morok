//! Shape utilities for UOps with symbolic shape support.
//!
//! This module provides shape-related types and functions following Tinygrad's approach:
//! - Shapes can contain both concrete integers and symbolic UOp expressions
//! - Shape inference with validation
//! - Broadcasting utilities (explicit, non-automatic)
//!
//! Key differences from Tinygrad:
//! - Uses Rust's type system (SInt enum vs Python Union)
//! - Explicit Result types instead of exceptions
//! - Non-automatic broadcasting (must be explicit)

use std::rc::Rc;

use smallvec::{SmallVec, smallvec};

use crate::{Error, Op, Result, SInt, UOp};

/// Shape type - sequence of symbolic integers.
///
/// Uses SmallVec with inline capacity of 4 to avoid heap allocation for
/// common tensor ranks (1D-4D), which covers 99% of ML workloads.
///
/// Can contain mix of concrete and symbolic dimensions.
pub type Shape = SmallVec<[SInt; 4]>;

// =========================================================================
// Shape Utilities
// =========================================================================

/// Check if shape is fully concrete (all dimensions are constants).
///
/// # Examples
///
/// ```rust
/// # use morok_ir::{SInt, shape::is_static};
/// let shape = vec![SInt::from(3), SInt::from(4), SInt::from(5)];
/// assert!(is_static(&shape));
/// ```
pub fn is_static(shape: &Shape) -> bool {
    shape.iter().all(|dim| dim.is_const())
}

/// Convert shape to concrete Vec<usize> if fully static, None otherwise.
///
/// # Examples
///
/// ```rust
/// # use morok_ir::{SInt, shape::to_static};
/// let shape = vec![SInt::from(3), SInt::from(4)];
/// assert_eq!(to_static(&shape), Some(vec![3, 4]));
/// ```
pub fn to_static(shape: &Shape) -> Option<Vec<usize>> {
    if is_static(shape) { Some(shape.iter().map(|dim| dim.as_const().unwrap()).collect()) } else { None }
}

// =========================================================================
// Shape Validation
// =========================================================================

/// Validate that a shape specification is valid (all positive, no zeros).
///
/// # Errors
/// Returns error if any dimension is negative or zero.
///
/// # Examples
///
/// ```rust
/// # use morok_ir::shape::validate_shape;
/// let valid = vec![1, 2, 3];
/// assert!(validate_shape(&valid).is_ok());
///
/// let invalid = vec![1, -2, 3];
/// assert!(validate_shape(&invalid).is_err());
/// ```
pub fn validate_shape(shape: &[isize]) -> Result<Vec<usize>> {
    use crate::error::ReshapeNegativeDimensionSnafu;
    use snafu::ensure;

    ensure!(shape.iter().all(|&s| s > 0), ReshapeNegativeDimensionSnafu { shape: shape.to_vec() });

    Ok(shape.iter().map(|&s| s as usize).collect())
}

/// Check if two shapes are equal.
///
/// Uses pointer equality for symbolic dimensions (consistent with hash consing).
pub fn shapes_equal(lhs: &Shape, rhs: &Shape) -> bool {
    lhs.len() == rhs.len() && lhs.iter().zip(rhs.iter()).all(|(l, r)| l == r)
}

/// Check if all shapes in a slice are equal.
///
/// # Examples
///
/// ```rust
/// # use morok_ir::{SInt, shape::all_shapes_equal};
/// let shape1 = vec![SInt::from(3), SInt::from(4)];
/// let shape2 = vec![SInt::from(3), SInt::from(4)];
/// let shape3 = vec![SInt::from(3), SInt::from(4)];
/// assert!(all_shapes_equal(&[shape1, shape2, shape3]));
/// ```
pub fn all_shapes_equal(shapes: &[Shape]) -> bool {
    if shapes.is_empty() {
        return true;
    }
    shapes.iter().all(|s| shapes_equal(s, &shapes[0]))
}

// =========================================================================
// Broadcasting Utilities (Explicit, Non-automatic)
// =========================================================================

/// Align shapes to the left by prepending 1s.
///
/// Makes all shapes have the same number of dimensions by adding dimensions
/// of size 1 on the left.
///
/// # Examples
///
/// ```rust
/// # use morok_ir::{SInt, shape::align_shapes_left};
/// let shape1 = vec![SInt::from(5)];
/// let shape2 = vec![SInt::from(3), SInt::from(5)];
/// let aligned = align_shapes_left(&[shape1, shape2]);
/// assert_eq!(aligned.len(), 2);
/// assert_eq!(aligned[0].len(), 2); // [1, 5]
/// assert_eq!(aligned[1].len(), 2); // [3, 5]
/// ```
pub fn align_shapes_left(shapes: &[Shape]) -> Vec<Shape> {
    if shapes.is_empty() {
        return Vec::new();
    }

    let max_dims = shapes.iter().map(|s| s.len()).max().unwrap();

    shapes
        .iter()
        .map(|shape| {
            let padding = max_dims - shape.len();
            let mut aligned = SmallVec::with_capacity(max_dims);
            aligned.extend(std::iter::repeat_n(SInt::from(1), padding));
            aligned.extend(shape.iter().cloned());
            aligned
        })
        .collect()
}

/// Check if two shapes can be broadcast together (NumPy-style broadcasting).
///
/// Two shapes are broadcastable if:
/// - They have the same number of dimensions
/// - For each dimension, either the dimensions match or one of them is 1
///
/// # Examples
///
/// ```rust
/// # use morok_ir::{SInt, shape::can_broadcast};
/// let shape1 = vec![SInt::from(1), SInt::from(5)];
/// let shape2 = vec![SInt::from(3), SInt::from(5)];
/// assert!(can_broadcast(&shape1, &shape2));
///
/// let shape3 = vec![SInt::from(3), SInt::from(4)];
/// assert!(!can_broadcast(&shape1, &shape3));
/// ```
pub fn can_broadcast(lhs: &Shape, rhs: &Shape) -> bool {
    if lhs.len() != rhs.len() {
        return false;
    }

    lhs.iter().zip(rhs.iter()).all(|(l, r)| {
        // If both are concrete, check broadcast rule
        if let (Some(lv), Some(rv)) = (l.as_const(), r.as_const()) {
            lv == rv || lv == 1 || rv == 1
        } else if l == r {
            // Same symbolic expression
            true
        } else {
            // Different symbolic expressions - conservatively assume compatible
            // (runtime check would be needed)
            true
        }
    })
}

/// Compute the broadcast result shape for two shapes.
///
/// Returns the shape that results from broadcasting the two input shapes.
/// Both shapes must be broadcastable (checked with can_broadcast).
///
/// # Errors
/// Returns error if shapes are not broadcastable.
///
/// # Examples
///
/// ```rust
/// # use morok_ir::{SInt, shape::broadcast_shape};
/// let shape1 = vec![SInt::from(1), SInt::from(5)];
/// let shape2 = vec![SInt::from(3), SInt::from(5)];
/// let result = broadcast_shape(&shape1, &shape2).unwrap();
/// assert_eq!(result[0].as_const(), Some(3));
/// assert_eq!(result[1].as_const(), Some(5));
/// ```
pub fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    use crate::error::BroadcastShapeMismatchSnafu;
    use snafu::ensure;

    ensure!(
        lhs.len() == rhs.len(),
        BroadcastShapeMismatchSnafu { lhs: format!("{:?}", lhs), rhs: format!("{:?}", rhs) }
    );

    let mut result = SmallVec::with_capacity(lhs.len());

    for (l, r) in lhs.iter().zip(rhs.iter()) {
        if l == r {
            // Same dimension (concrete value or symbolic expression)
            result.push(l.clone());
        } else if let (Some(lv), Some(rv)) = (l.as_const(), r.as_const()) {
            // Both concrete - apply broadcast rule
            if lv == 1 {
                result.push(r.clone());
            } else if rv == 1 || lv == rv {
                result.push(l.clone());
            } else {
                return Err(Error::BroadcastShapeMismatch { lhs: format!("{:?}", lhs), rhs: format!("{:?}", rhs) });
            }
        } else {
            // At least one is symbolic - use max (conservatively)
            result.push(crate::sint_max(&[l.clone(), r.clone()]));
        }
    }

    Ok(result)
}

/// Compute broadcast result for multiple shapes.
///
/// # Errors
/// Returns error if any pair of shapes is not broadcastable.
pub fn broadcast_shapes(shapes: &[Shape]) -> Result<Shape> {
    if shapes.is_empty() {
        return Ok(SmallVec::new());
    }

    // Align all shapes to same number of dimensions
    let aligned = align_shapes_left(shapes);

    // Successively broadcast pairs
    let mut result = aligned[0].clone();
    for shape in &aligned[1..] {
        result = broadcast_shape(&result, shape)?;
    }

    Ok(result)
}

// =========================================================================
// Movement Op Argument Extraction (marg equivalent)
// =========================================================================

/// Extract shape dimensions from a VECTORIZE or CONST UOp.
///
/// Following Tinygrad's `marg` pattern, this extracts concrete or symbolic
/// dimensions from the UOp used to store shape information.
///
/// Returns None if the UOp is not in the expected format.
fn extract_shape_from_uop(shape_uop: &Rc<UOp>) -> Option<Shape> {
    match shape_uop.op() {
        // VECTORIZE with Index-typed elements
        Op::Vectorize { elements } => {
            let mut dims = SmallVec::with_capacity(elements.len());
            for elem in elements {
                // Each element should be an Index UOp (const or symbolic)
                dims.push(SInt::from(elem.clone()));
            }
            Some(dims)
        }

        // Single CONST value (for 1D shapes)
        Op::Const(const_hash) => match const_hash.0 {
            crate::ConstValue::Int(v) if v >= 0 => Some(smallvec![SInt::from(v as usize)]),
            crate::ConstValue::UInt(v) => Some(smallvec![SInt::from(v as usize)]),
            _ => None,
        },

        // VConst for multiple concrete dimensions
        Op::VConst { values } => {
            let mut dims = SmallVec::with_capacity(values.len());
            for val in values {
                match val {
                    crate::ConstValue::Int(v) if *v >= 0 => dims.push(SInt::from(*v as usize)),
                    crate::ConstValue::UInt(v) => dims.push(SInt::from(*v as usize)),
                    _ => return None,
                }
            }
            Some(dims)
        }

        _ => None,
    }
}

/// Extract padding/shrink ranges from UOps.
///
/// Returns pairs of (begin, end) for each dimension.
fn extract_ranges_from_uops(begins_uop: &Rc<UOp>, ends_uop: &Rc<UOp>) -> Option<Vec<(SInt, SInt)>> {
    let begins = extract_shape_from_uop(begins_uop)?;
    let ends = extract_shape_from_uop(ends_uop)?;

    if begins.len() != ends.len() {
        return None;
    }

    Some(begins.into_iter().zip(ends).collect())
}

/// Convert a Shape to a VECTORIZE UOp for use in movement operations.
///
/// This creates a UOp that encodes the shape dimensions, suitable for
/// passing to Reshape, Expand, etc.
///
/// # Examples
///
/// ```rust
/// # use morok_ir::{SInt, shape::shape_to_uop};
/// # use morok_dtype::DType;
/// let shape = vec![SInt::from(3), SInt::from(4), SInt::from(5)];
/// let shape_uop = shape_to_uop(&shape);
/// assert_eq!(shape_uop.dtype(), DType::Index.vec(3));
/// ```
pub fn shape_to_uop(shape: &Shape) -> Rc<UOp> {
    use morok_dtype::DType;
    use smallvec::SmallVec;

    if shape.is_empty() {
        // Empty shape - return a const 0 or empty vectorize?
        // Following Tinygrad, empty shape might use VConst with empty vec
        return UOp::new(Op::VConst { values: vec![] }, DType::Index);
    }

    // Convert each SInt to a UOp
    let elements: SmallVec<[Rc<UOp>; 4]> = shape.iter().map(|dim| dim.to_uop(DType::Index)).collect();

    let vec_dtype = DType::Index.vec(elements.len());
    UOp::new(Op::Vectorize { elements }, vec_dtype)
}

/// Convert a vector of (begin, end) ranges to two UOps for Pad/Shrink operations.
///
/// Returns (begins_uop, ends_uop) as VECTORIZE UOps.
pub fn ranges_to_uops(ranges: &[(SInt, SInt)]) -> (Rc<UOp>, Rc<UOp>) {
    use morok_dtype::DType;
    use smallvec::SmallVec;

    let begins: SmallVec<[Rc<UOp>; 4]> = ranges.iter().map(|(begin, _)| begin.to_uop(DType::Index)).collect();

    let ends: SmallVec<[Rc<UOp>; 4]> = ranges.iter().map(|(_, end)| end.to_uop(DType::Index)).collect();

    let vec_dtype = DType::Index.vec(ranges.len());

    (
        UOp::new(Op::Vectorize { elements: begins }, vec_dtype.clone()),
        UOp::new(Op::Vectorize { elements: ends }, vec_dtype),
    )
}

// =========================================================================
// Shape Inference (Tinygrad-style)
// =========================================================================

/// Infer shape from a UOp's operation.
///
/// This is the core shape inference function, following Tinygrad's approach.
/// Returns None for operations without a well-defined shape (control flow, etc.).
///
/// # Shape Inference Rules
///
/// - **Nullary ops** (Const, VConst): Return concrete shape
/// - **Unary ops**: Preserve input shape
/// - **Binary ops**: Validate inputs match, return common shape
/// - **Ternary ops**: Return shape of value branches
/// - **Movement ops**: Compute shape from operation arguments
/// - **Reduce ops**: Compute reduced shape
/// - **Late/control flow ops**: Return None
pub fn infer_shape_from_op(uop: &UOp) -> Option<Shape> {
    match uop.op() {
        // =====================================================================
        // Nullary operations
        // =====================================================================
        Op::Const(_) => Some(SmallVec::new()), // Scalar has empty shape

        Op::VConst { values } => Some(smallvec![SInt::from(values.len())]),

        Op::Unique(_) | Op::Device(_) | Op::Noop | Op::DefineGlobal(_) | Op::DefineLocal(_) => None,

        // =====================================================================
        // Unary operations - preserve shape
        // =====================================================================
        Op::Unary(_, input) => input.shape().cloned(),

        // =====================================================================
        // Binary operations - validate shapes match
        // =====================================================================
        Op::Binary(_, lhs, rhs) => {
            let lhs_shape = lhs.shape()?;
            let rhs_shape = rhs.shape()?;

            // Shapes must match for binary ops (no automatic broadcasting)
            if shapes_equal(lhs_shape, rhs_shape) {
                Some(lhs_shape.clone())
            } else {
                // Shape mismatch - in strict mode this would error
                // For now, return None (unknown shape)
                None
            }
        }

        // =====================================================================
        // Ternary operations
        // =====================================================================
        Op::Ternary(_, _condition, true_val, _false_val) => {
            // Result has shape of value branches (should match)
            true_val.shape().cloned()
        }

        // =====================================================================
        // Type operations - preserve shape
        // =====================================================================
        Op::Cast { src, .. } | Op::BitCast { src, .. } => src.shape().cloned(),

        // =====================================================================
        // Vector operations
        // =====================================================================
        Op::Vectorize { elements } => Some(smallvec![SInt::from(elements.len())]),

        Op::Gep { .. } => Some(SmallVec::new()), // Extract element from vector -> scalar

        // =====================================================================
        // Movement operations
        // =====================================================================
        Op::Reshape { new_shape, .. } => {
            // Extract shape from VECTORIZE/CONST UOp
            extract_shape_from_uop(new_shape)
        }

        Op::Permute { axes, src } => {
            let src_shape = src.shape()?;
            // Reorder dimensions according to permutation
            Some(axes.iter().map(|&i| src_shape[i].clone()).collect())
        }

        Op::Expand { new_shape, .. } => {
            // Extract shape from VECTORIZE/CONST UOp
            extract_shape_from_uop(new_shape)
        }

        Op::Pad { src, begin_pads, end_pads } => {
            let src_shape = src.shape()?;
            let ranges = extract_ranges_from_uops(begin_pads, end_pads)?;

            if src_shape.len() != ranges.len() {
                return None;
            }

            // New shape = src_shape + begin_pads + end_pads for each dimension
            Some(
                src_shape
                    .iter()
                    .zip(ranges.iter())
                    .map(|(dim, (begin, end))| {
                        // dim + begin + end
                        // For now, if all are const we can compute, otherwise create symbolic add
                        if let (Some(d), Some(b), Some(e)) = (dim.as_const(), begin.as_const(), end.as_const()) {
                            SInt::from(d + b + e)
                        } else {
                            // Would need to create Add UOps - for now approximate
                            dim.clone()
                        }
                    })
                    .collect(),
            )
        }

        Op::Shrink { src, begins, ends } => {
            let src_shape = src.shape()?;
            let ranges = extract_ranges_from_uops(begins, ends)?;

            if src_shape.len() != ranges.len() {
                return None;
            }

            // New shape = end - begin for each dimension
            Some(
                ranges
                    .iter()
                    .map(|(begin, end)| {
                        // end - begin
                        if let (Some(b), Some(e)) = (begin.as_const(), end.as_const()) {
                            if e >= b {
                                SInt::from(e - b)
                            } else {
                                SInt::from(0) // Invalid shrink
                            }
                        } else {
                            // Would need to create Sub UOps - approximate for now
                            SInt::from(1)
                        }
                    })
                    .collect(),
            )
        }

        Op::Flip { src, .. } => {
            // Flip preserves shape
            src.shape().cloned()
        }

        Op::Multi { src, .. } => {
            // Multi preserves shape
            src.shape().cloned()
        }

        // =====================================================================
        // Reduction operations
        // =====================================================================
        Op::ReduceAxis { axes, src, .. } => {
            let src_shape = src.shape()?;
            // Remove reduced axes
            Some(src_shape.iter().enumerate().filter(|(i, _)| !axes.contains(i)).map(|(_, dim)| dim.clone()).collect())
        }

        Op::Reduce { .. } => {
            // Reduce with ranges - context dependent
            None
        }

        Op::AllReduce { src, .. } => {
            // AllReduce preserves shape
            src.shape().cloned()
        }

        // =====================================================================
        // Buffer and memory operations - shape depends on buffer
        // =====================================================================
        Op::Buffer { .. }
        | Op::BufferView { .. }
        | Op::Bufferize { .. }
        | Op::Index { .. }
        | Op::Copy { .. }
        | Op::MStack { .. }
        | Op::Load { .. }
        | Op::LoadGated { .. }
        | Op::Store { .. }
        | Op::StoreGated { .. } => None,

        // =====================================================================
        // Control flow - no static shape
        // =====================================================================
        Op::If { .. } | Op::EndIf { .. } | Op::Range { .. } | Op::End { .. } | Op::Barrier { .. } => None,

        // =====================================================================
        // Special operations
        // =====================================================================
        Op::MSelect { .. } => None,

        Op::Special { .. } => Some(SmallVec::new()), // Special returns index (scalar)

        Op::DefineVar { .. } => Some(SmallVec::new()), // Variable is scalar

        Op::Bind { value, .. } => value.shape().cloned(),

        Op::DefineReg { .. } => None,

        // =====================================================================
        // Advanced operations
        // =====================================================================
        Op::Wmma { .. } | Op::Contract { .. } | Op::Unroll { .. } => {
            // These require more complex shape computation
            None
        }

        Op::Kernel { .. } => None,

        Op::Assign { target, .. } => target.shape().cloned(),

        Op::Detach { src } | Op::Contiguous { src } | Op::ContiguousBackward { src } | Op::Precast { src } => {
            src.shape().cloned()
        }

        Op::After { passthrough, .. } => passthrough.shape().cloned(),

        Op::Custom { .. } | Op::CustomI { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstValue, UOp};
    use morok_dtype::DType;

    #[test]
    fn test_is_static() {
        let static_shape = smallvec![SInt::from(3), SInt::from(4), SInt::from(5)];
        assert!(is_static(&static_shape));

        // Note: Const UOps are automatically simplified to SInt::Const
        // To get a truly symbolic dimension, we need a non-const UOp
        // For now, just test with concrete values
        let also_static = smallvec![SInt::from(3), SInt::from(10)];
        assert!(is_static(&also_static));
    }

    #[test]
    fn test_to_static() {
        let shape = smallvec![SInt::from(3), SInt::from(4)];
        assert_eq!(to_static(&shape), Some(vec![3, 4]));

        // Note: Const UOps are automatically simplified, so we'd need
        // a truly symbolic UOp (like from an operation) to test dynamic shapes
        // For now, just verify static conversion works
        let shape2 = smallvec![SInt::from(5), SInt::from(6), SInt::from(7)];
        assert_eq!(to_static(&shape2), Some(vec![5, 6, 7]));
    }

    #[test]
    fn test_ndim() {
        let shape: Shape = smallvec![SInt::from(3), SInt::from(4), SInt::from(5)];
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_shape_product() {
        let shape: Shape = smallvec![SInt::from(2), SInt::from(3), SInt::from(4)];
        let product = crate::sint_prod(&shape);
        assert_eq!(product.as_const(), Some(24));
    }

    #[test]
    fn test_validate_shape() {
        assert!(validate_shape(&[1, 2, 3]).is_ok());
        assert!(validate_shape(&[1, -2, 3]).is_err());
        assert!(validate_shape(&[1, 0, 3]).is_err());
    }

    #[test]
    fn test_shapes_equal() {
        let shape1 = smallvec![SInt::from(3), SInt::from(4)];
        let shape2 = smallvec![SInt::from(3), SInt::from(4)];
        assert!(shapes_equal(&shape1, &shape2));

        let shape3 = smallvec![SInt::from(3), SInt::from(5)];
        assert!(!shapes_equal(&shape1, &shape3));
    }

    #[test]
    fn test_all_shapes_equal() {
        let shape1 = smallvec![SInt::from(3), SInt::from(4)];
        let shape2 = smallvec![SInt::from(3), SInt::from(4)];
        let shape3 = smallvec![SInt::from(3), SInt::from(4)];
        assert!(all_shapes_equal(&[shape1, shape2, shape3]));
    }

    #[test]
    fn test_align_shapes_left() {
        let shape1 = smallvec![SInt::from(5)];
        let shape2 = smallvec![SInt::from(3), SInt::from(5)];
        let aligned = align_shapes_left(&[shape1, shape2]);

        assert_eq!(aligned.len(), 2);
        assert_eq!(aligned[0].len(), 2);
        assert_eq!(aligned[0][0].as_const(), Some(1));
        assert_eq!(aligned[0][1].as_const(), Some(5));
    }

    #[test]
    fn test_can_broadcast() {
        let shape1 = smallvec![SInt::from(1), SInt::from(5)];
        let shape2 = smallvec![SInt::from(3), SInt::from(5)];
        assert!(can_broadcast(&shape1, &shape2));

        let shape3 = smallvec![SInt::from(3), SInt::from(4)];
        assert!(!can_broadcast(&shape1, &shape3));
    }

    #[test]
    fn test_broadcast_shape() {
        let shape1 = smallvec![SInt::from(1), SInt::from(5)];
        let shape2 = smallvec![SInt::from(3), SInt::from(5)];
        let result = broadcast_shape(&shape1, &shape2).unwrap();

        assert_eq!(result[0].as_const(), Some(3));
        assert_eq!(result[1].as_const(), Some(5));
    }

    #[test]
    fn test_broadcast_shape_error() {
        let shape1 = smallvec![SInt::from(3), SInt::from(4)];
        let shape2 = smallvec![SInt::from(3), SInt::from(5)];
        assert!(broadcast_shape(&shape1, &shape2).is_err());
    }

    #[test]
    fn test_broadcast_shapes_multiple() {
        let shape1 = smallvec![SInt::from(1), SInt::from(5)];
        let shape2 = smallvec![SInt::from(3), SInt::from(1)];
        let shape3 = smallvec![SInt::from(3), SInt::from(5)];

        let result = broadcast_shapes(&[shape1, shape2, shape3]).unwrap();
        assert_eq!(result[0].as_const(), Some(3));
        assert_eq!(result[1].as_const(), Some(5));
    }

    // =====================================================================
    // Shape Inference Tests
    // =====================================================================

    #[test]
    fn test_infer_const_shape() {
        let scalar = UOp::const_(DType::Float32, ConstValue::Float(42.0));
        let shape = scalar.shape().expect("Const should have shape");
        assert_eq!(shape.len(), 0); // Scalar has empty shape
    }

    #[test]
    fn test_infer_vconst_shape() {
        let values =
            vec![ConstValue::Float(1.0), ConstValue::Float(2.0), ConstValue::Float(3.0), ConstValue::Float(4.0)];
        let vec = UOp::new(crate::Op::VConst { values: values.clone() }, DType::Float32.vec(4));
        let shape = vec.shape().expect("VConst should have shape");
        assert_eq!(shape.len(), 1);
        assert_eq!(shape[0].as_const(), Some(4));
    }

    #[test]
    fn test_infer_unary_shape() {
        let val = UOp::const_(DType::Float32, ConstValue::Float(5.0));
        let neg = UOp::neg_op(val);
        let shape = neg.shape().expect("Unary should have shape");
        assert_eq!(shape.len(), 0); // Preserves scalar shape
    }

    #[test]
    fn test_infer_binary_shape() {
        let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
        let add = UOp::try_add_op(a, b).unwrap();
        let shape = add.shape().expect("Binary should have shape");
        assert_eq!(shape.len(), 0); // Both scalars -> scalar result
    }

    #[test]
    fn test_infer_cast_shape() {
        let val = UOp::const_(DType::Float32, ConstValue::Float(1.5));
        let cast = UOp::cast(val, DType::Int32);
        let shape = cast.shape().expect("Cast should preserve shape");
        assert_eq!(shape.len(), 0);
    }

    #[test]
    fn test_shape_caching() {
        let val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        // First access computes shape
        let shape1 = val.shape().expect("Should have shape");
        // Second access uses cached value (same pointer)
        let shape2 = val.shape().expect("Should have cached shape");
        assert!(std::ptr::eq(shape1, shape2), "Shape should be cached");
    }
}
