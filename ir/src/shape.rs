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
use snafu::ensure;

use crate::{ConstValue, Op, Result, SInt, UOp, error::*};

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
/// # use smallvec::smallvec;
/// let shape = smallvec![SInt::from(3), SInt::from(4), SInt::from(5)];
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
/// # use smallvec::smallvec;
/// let shape = smallvec![SInt::from(3), SInt::from(4)];
/// assert_eq!(to_static(&shape), Some(smallvec![3, 4]));
/// ```
pub fn to_static(shape: &Shape) -> Option<SmallVec<[usize; 4]>> {
    is_static(shape).then_some(shape.iter().map(|dim| dim.as_const().unwrap()).collect())
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
/// ```rust
/// # use morok_ir::shape::validate_shape;
/// let valid = vec![1, 2, 3];
/// assert!(validate_shape(&valid).is_ok());
/// let invalid = vec![1, -2, 3];
/// assert!(validate_shape(&invalid).is_err());
/// ```
pub fn validate_shape(shape: &[isize]) -> Result<SmallVec<[usize; 4]>> {
    ensure!(shape.iter().all(|&s| s > 0), ReshapeNegativeDimensionSnafu { shape });
    Ok(shape.iter().map(|&s| s as usize).collect())
}

/// Check if two shapes are equal.
///
/// Uses pointer equality for symbolic dimensions (consistent with hash consing).
pub fn shapes_equal(lhs: &Shape, rhs: &Shape) -> bool {
    lhs == rhs
}

/// Check if all shapes in a slice are equal.
///
/// # Examples
///
/// ```rust
/// # use morok_ir::{SInt, shape::all_shapes_equal};
/// # use smallvec::smallvec;
/// let shape1 = smallvec![SInt::from(3), SInt::from(4)];
/// let shape2 = smallvec![SInt::from(3), SInt::from(4)];
/// let shape3 = smallvec![SInt::from(3), SInt::from(4)];
/// assert!(all_shapes_equal(&[shape1, shape2, shape3]));
/// ```
pub fn all_shapes_equal(shapes: &[Shape]) -> bool {
    (!shapes.is_empty()) && shapes.iter().all(|s| shapes_equal(s, &shapes[0]))
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
/// # use smallvec::smallvec;
/// let shape1 = smallvec![SInt::from(5)];
/// let shape2 = smallvec![SInt::from(3), SInt::from(5)];
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
/// # use smallvec::smallvec;
/// let shape1 = smallvec![SInt::from(1), SInt::from(5)];
/// let shape2 = smallvec![SInt::from(3), SInt::from(5)];
/// assert!(can_broadcast(&shape1, &shape2));
///
/// let shape3 = smallvec![SInt::from(3), SInt::from(4)];
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
/// # use smallvec::smallvec;
/// let shape1 = smallvec![SInt::from(1), SInt::from(5)];
/// let shape2 = smallvec![SInt::from(3), SInt::from(5)];
/// let result = broadcast_shape(&shape1, &shape2).unwrap();
/// assert_eq!(result[0].as_const(), Some(3));
/// assert_eq!(result[1].as_const(), Some(5));
/// ```
pub fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    use crate::error::BroadcastShapeMismatchSnafu;
    use snafu::ensure;

    ensure!(lhs.len() == rhs.len(), BroadcastShapeMismatchSnafu { lhs: lhs.clone(), rhs: rhs.clone() });

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
                return BroadcastShapeMismatchSnafu { lhs: lhs.clone(), rhs: rhs.clone() }.fail();
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

/// Convert shape to Vec<usize>, ensuring all dimensions are concrete.
///
/// This is a helper function to reduce boilerplate when converting shapes
/// for operations that require concrete (non-symbolic) dimensions.
///
/// # Errors
///
/// Returns error if any dimension contains a symbolic (non-const) value.
pub fn to_vec_usize(shape: &Shape) -> Result<Vec<usize>> {
    shape
        .iter()
        .map(|dim| {
            dim.as_const().ok_or_else(|| Error::SymbolicShapeUnsupported { operation: "shape conversion".to_string() })
        })
        .collect()
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
        Op::Vectorize { elements } => Some(elements.into_iter().cloned().map(SInt::from).collect()),

        // Single CONST value (for 1D shapes)
        Op::Const(const_hash) => match const_hash.0 {
            ConstValue::Int(v) if v >= 0 => Some(smallvec![SInt::from(v as usize)]),
            ConstValue::UInt(v) => Some(smallvec![SInt::from(v as usize)]),
            _ => None,
        },

        // VConst for multiple concrete dimensions
        Op::VConst { values } => {
            let mut dims = SmallVec::with_capacity(values.len());
            for val in values {
                match val {
                    ConstValue::Int(v) if *v >= 0 => dims.push(SInt::from(*v as usize)),
                    ConstValue::UInt(v) => dims.push(SInt::from(*v as usize)),
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
/// # use smallvec::smallvec;
/// let shape = smallvec![SInt::from(3), SInt::from(4), SInt::from(5)];
/// let shape_uop = shape_to_uop(&shape);
/// assert_eq!(shape_uop.dtype(), DType::Index.vec(3));
/// ```
pub fn shape_to_uop(shape: &Shape) -> Rc<UOp> {
    use morok_dtype::DType;
    use smallvec::SmallVec;

    // Use Vectorize for all shapes (including empty)
    // Empty shape → Vectorize { elements: [] }
    // This is consistent with non-empty path and compatible with extract_shape_from_uop
    let elements: SmallVec<[Rc<UOp>; 4]> = shape.iter().map(|dim| dim.to_uop(DType::Index)).collect();

    UOp::vectorize(elements)
}

/// Convert a vector of (begin, end) ranges to two UOps for Pad/Shrink operations.
///
/// Returns (begins_uop, ends_uop) as VECTORIZE UOps.
pub fn ranges_to_uops(ranges: &[(SInt, SInt)]) -> (Rc<UOp>, Rc<UOp>) {
    use morok_dtype::DType;
    use smallvec::SmallVec;

    let begins: SmallVec<[Rc<UOp>; 4]> = ranges.iter().map(|(begin, _)| begin.to_uop(DType::Index)).collect();

    let ends: SmallVec<[Rc<UOp>; 4]> = ranges.iter().map(|(_, end)| end.to_uop(DType::Index)).collect();

    (UOp::vectorize(begins), UOp::vectorize(ends))
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
pub fn infer_shape_from_op(uop: &UOp) -> crate::Result<Option<Shape>> {
    Ok(match uop.op() {
        // =====================================================================
        // Nullary operations
        // =====================================================================
        Op::Const(_) => Some(SmallVec::new()), // Scalar has empty shape

        Op::VConst { .. } => None,

        Op::Unique(_) | Op::Device(_) | Op::Noop | Op::Invalid => None,

        // Define operations have shape from dtype.size (pointer types)
        // Following Tinygrad: shape comes from PtrDType.size, not from the id parameter
        Op::DefineGlobal(_id) | Op::DefineLocal(_id) => {
            use morok_dtype::DType;
            match uop.dtype() {
                DType::Ptr { size: Some(s), .. } => Some(smallvec![SInt::from(s)]),
                DType::Ptr { size: None, .. } => {
                    // Unlimited size - represented as -1 following Tinygrad
                    let neg_one = UOp::index_const(-1);
                    Some(smallvec![SInt::from(neg_one)])
                }
                dtype => {
                    // DefineGlobal/Local must have Ptr dtype (following Tinygrad spec)
                    return crate::error::DefineGlobalRequiresPtrDTypeSnafu {
                        op: "DefineGlobal/DefineLocal",
                        dtype: dtype.clone(),
                    }
                    .fail();
                }
            }
        }

        // =====================================================================
        // Unary operations - preserve shape
        // =====================================================================
        Op::Unary(_, input) => input.shape()?.cloned(),

        // =====================================================================
        // Binary operations - validate shapes match
        // =====================================================================
        Op::Binary(op, lhs, rhs) => {
            match (lhs.shape()?, rhs.shape()?) {
                (Some(lhs_shape), Some(rhs_shape)) if !shapes_equal(lhs_shape, rhs_shape) => {
                    // Both have shapes but they differ - ERROR
                    return BinaryShapeMismatchSnafu {
                        op: *op,
                        lhs: Box::new(lhs_shape.clone()),
                        rhs: Box::new(rhs_shape.clone()),
                    }
                    .fail();
                }
                (Some(s), _) | (_, Some(s)) => Some(s.clone()),
                (None, None) => None, // Both shapeless - valid (RANGE + RANGE)
            }
        }

        // =====================================================================
        // Ternary operations
        // =====================================================================
        Op::Ternary(_, _condition, true_val, false_val) => {
            // Result has shape of value branches - they must match
            let true_shape = true_val.shape()?;
            let false_shape = false_val.shape()?;

            match (true_shape, false_shape) {
                (Some(ts), Some(fs)) if !shapes_equal(ts, fs) => {
                    return crate::error::TernaryBranchShapeMismatchSnafu {
                        true_branch: Box::new(ts.clone()),
                        false_branch: Box::new(fs.clone()),
                    }
                    .fail();
                }
                (Some(s), _) | (_, Some(s)) => Some(s.clone()),
                (None, None) => None,
            }
        }

        // =====================================================================
        // Type operations - preserve shape
        // =====================================================================
        Op::Cast { src, .. } | Op::BitCast { src, .. } => src.shape()?.cloned(),

        // =====================================================================
        // Vector operations (kernel-level, no tensor shape)
        // =====================================================================
        Op::Vectorize { .. } => None,

        Op::Gep { .. } => Some(SmallVec::new()), // Extract element from vector -> scalar

        // =====================================================================
        // Movement operations
        // =====================================================================
        Op::Reshape { new_shape, .. } => {
            // Extract shape from VECTORIZE/CONST UOp
            extract_shape_from_uop(new_shape)
        }

        Op::Permute { axes, src } => {
            let src_shape = src.shape()?.ok_or_else(|| crate::Error::VoidTypeInOp)?;
            // Reorder dimensions according to permutation
            Some(axes.iter().map(|&i| src_shape[i].clone()).collect())
        }

        Op::Expand { new_shape, .. } => {
            // Extract shape from VECTORIZE/CONST UOp
            extract_shape_from_uop(new_shape)
        }

        Op::Pad { src, begin_pads, end_pads } => {
            let src_shape = src.shape()?.ok_or_else(|| crate::Error::VoidTypeInOp)?;
            let ranges = extract_ranges_from_uops(begin_pads, end_pads).ok_or_else(|| crate::Error::VoidTypeInOp)?;

            if src_shape.len() != ranges.len() {
                return Ok(None);
            }

            // New shape = src_shape + begin_pads + end_pads for each dimension
            // All padding values must be concrete (checked during construction)
            Some(
                src_shape
                    .iter()
                    .zip(ranges.iter())
                    .map(|(dim, (begin, end))| {
                        // dim + begin + end
                        if let (Some(d), Some(b), Some(e)) = (dim.as_const(), begin.as_const(), end.as_const()) {
                            Ok(SInt::from(d + b + e))
                        } else {
                            // Symbolic padding should have been rejected at construction time
                            // This case should not be reachable if try_pad validates properly
                            crate::error::SymbolicPaddingUnsupportedSnafu.fail()
                        }
                    })
                    .collect::<crate::Result<Shape>>()?,
            )
        }

        Op::Shrink { src, begins, ends } => {
            let src_shape = src.shape()?.ok_or_else(|| crate::Error::VoidTypeInOp)?;
            let ranges = extract_ranges_from_uops(begins, ends).ok_or_else(|| crate::Error::VoidTypeInOp)?;

            if src_shape.len() != ranges.len() {
                return Ok(None);
            }

            // New shape = end - begin for each dimension
            // All shrink values must be concrete (checked during construction)
            Some(
                ranges
                    .iter()
                    .map(|(begin, end)| {
                        // end - begin
                        if let (Some(b), Some(e)) = (begin.as_const(), end.as_const()) {
                            Ok(if e >= b {
                                SInt::from(e - b)
                            } else {
                                SInt::from(0) // Invalid shrink
                            })
                        } else {
                            // Symbolic shrinking should have been rejected at construction time
                            // This case should not be reachable if try_shrink validates properly
                            crate::error::SymbolicShrinkingUnsupportedSnafu.fail()
                        }
                    })
                    .collect::<crate::Result<Shape>>()?,
            )
        }

        Op::Flip { src, .. } => {
            // Flip preserves shape
            src.shape()?.cloned()
        }

        Op::Multi { src, .. } => {
            // Multi scales the specified axis by device count
            // TODO: Need device count from somewhere - for now preserve shape
            // Tinygrad: tuple(s*len(self.device) if a == self.axis else s for a,s in enumerate(ps))
            src.shape()?.cloned()
        }

        // =====================================================================
        // Reduction operations
        // =====================================================================
        Op::ReduceAxis { axes, src, .. } => {
            let src_shape = src.shape()?.ok_or_else(|| crate::Error::VoidTypeInOp)?;
            // Set reduced axes to 1 (don't remove them - matches Tinygrad)
            Some(
                src_shape
                    .iter()
                    .enumerate()
                    .map(|(i, dim)| if axes.contains(&i) { SInt::from(1) } else { dim.clone() })
                    .collect(),
            )
        }

        Op::Reduce { .. } => {
            // Reduce with ranges - context dependent
            None
        }

        Op::AllReduce { src, .. } => {
            // AllReduce preserves shape
            src.shape()?.cloned()
        }

        // =====================================================================
        // Buffer and memory operations - shape depends on buffer
        // =====================================================================
        // Buffer operations have shape (size,)
        Op::Buffer { size, .. } => Some(smallvec![SInt::from(*size)]),
        Op::BufferView { size, .. } => Some(smallvec![SInt::from(*size)]),

        // Passthrough operations
        Op::Copy { src, .. } => src.shape()?.cloned(),
        Op::MStack { buffers } => match buffers.first() {
            Some(b) => b.shape()?.cloned(),
            None => None,
        },

        // These have no shape
        Op::Bufferize { .. }
        | Op::Index { .. }
        | Op::Load { .. }
        | Op::LoadGated { .. }
        | Op::Store { .. }
        | Op::StoreGated { .. } => None,

        // =====================================================================
        // Control flow - no static shape
        // =====================================================================
        Op::If { .. } | Op::EndIf { .. } | Op::Range { .. } | Op::Barrier { .. } => None,

        // End passes through the computation shape
        Op::End { computation, .. } => computation.shape()?.cloned(),

        // =====================================================================
        // Special operations
        // =====================================================================
        // MSelect passes through buffer shape
        Op::MSelect { buffer, .. } => buffer.shape()?.cloned(),

        Op::Special { .. } => None,

        Op::DefineVar { .. } => Some(SmallVec::new()), // Variable is scalar

        Op::Bind { value, .. } => value.shape()?.cloned(),

        Op::DefineReg { size } => Some(smallvec![SInt::from(*size)]),

        // =====================================================================
        // Advanced operations
        // =====================================================================
        Op::Wmma { .. } | Op::Contract { .. } | Op::Unroll { .. } => {
            // These require more complex shape computation
            None
        }

        Op::Kernel { .. } => None,

        Op::Assign { target, .. } => target.shape()?.cloned(),

        Op::Detach { src } | Op::Contiguous { src } | Op::ContiguousBackward { src } | Op::Precast { src } => {
            src.shape()?.cloned()
        }

        Op::After { passthrough, .. } => passthrough.shape()?.cloned(),

        Op::Custom { .. } | Op::CustomI { .. } => None,

        // Graph organization operations have no shape
        Op::Sink { .. } => None,
        Op::Group { sources } => match sources.first() {
            Some(src) => src.shape()?.cloned(),
            None => None,
        },

        // PointerIndex is a scalar index operation (no shape)
        Op::PointerIndex { .. } => Some(smallvec![]),

        // Cat and PtrCat are kernel-level vector ops (no tensor shape)
        Op::Cat { .. } | Op::PtrCat { .. } => None,
    })
}
