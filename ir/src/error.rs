use morok_dtype::DType;
use smallvec::SmallVec;
use snafu::Snafu;

use crate::shape::Shape;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Clone, PartialEq, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    /// DType mismatch in binary operation.
    #[snafu(display("dtype mismatch: cannot perform operation on {lhs:?} and {rhs:?}"))]
    DTypeMismatch { lhs: DType, rhs: DType },

    /// Type promotion failed - no common type.
    #[snafu(display("type promotion failed: no common type for {lhs:?} and {rhs:?}"))]
    TypePromotionFailed { lhs: DType, rhs: DType },

    /// Invalid dtype for operation (e.g., bitwise on float).
    #[snafu(display("invalid dtype for operation: {operation:?} requires int or bool dtype, got {dtype:?}"))]
    InvalidDTypeForOp { operation: &'static str, dtype: DType },

    /// Void dtype cannot be used in operations.
    #[snafu(display("void dtype cannot be used in operations"))]
    VoidTypeInOp,

    /// Index parameter must have Index dtype.
    #[snafu(display("index parameter must have Index dtype, got {actual:?}"))]
    IndexTypeMismatch { actual: DType },

    /// Division by zero.
    #[snafu(display("division by zero"))]
    DivisionByZero,

    /// Reshape size mismatch.
    #[snafu(display("reshape size mismatch: input size {input_size} != output size {output_size}"))]
    ReshapeSizeMismatch { input_size: usize, output_size: usize },

    /// Shrink bounds violation.
    #[snafu(display(
        "shrink bounds violation: dimension {dim} has range [{begin}, {end}) but shape size is {shape_size}",
    ))]
    ShrinkBoundsViolation { dim: usize, begin: usize, end: usize, shape_size: usize },

    /// Bind value out of range.
    #[snafu(display("bind value {value} is outside valid range [{min}, {max}]"))]
    BindValueOutOfRange { value: i64, min: i64, max: i64 },

    /// Index out of bounds.
    #[snafu(display("index out of bounds"))]
    IndexOutOfBounds,

    /// Expand dimension count mismatch.
    #[snafu(display("expand dimension mismatch: input has {input_dims} dimensions, output has {output_dims}"))]
    ExpandDimensionMismatch { input_dims: usize, output_dims: usize },

    /// Expand invalid dimension (can only expand dimensions of size 1).
    #[snafu(display(
        "expand invalid: dimension {dim} has size {input} but needs to expand to {output} (can only expand from 1)",
    ))]
    ExpandInvalidDimension { dim: usize, input: usize, output: usize },

    /// Permute has invalid permutation.
    #[snafu(display("invalid permutation {permutation:?}: expected permutation of 0..{expected_dims}"))]
    PermuteInvalidPermutation { permutation: Vec<usize>, expected_dims: usize },

    /// Pad has negative padding value.
    #[snafu(display(
        "pad has negative value: dimension {dim} has padding ({begin}, {end}) but padding must be non-negative",
    ))]
    PadNegativeValue { dim: usize, begin: isize, end: isize },

    /// Pad dimension count mismatch.
    #[snafu(display("pad dimension mismatch: padding has {padding_dims} dimensions but shape has {shape_dims}"))]
    PadDimensionMismatch { padding_dims: usize, shape_dims: usize },

    /// Flip specification invalid.
    #[snafu(display("flip specification invalid: expected {expected_dims} dimensions, got {got_dims}"))]
    FlipInvalidSpec { expected_dims: usize, got_dims: usize },

    /// Reduce axis invalid.
    #[snafu(display("reduce axis {axis} is invalid for shape with {shape_dims} dimensions"))]
    ReduceAxisInvalid { axis: i32, shape_dims: usize },

    /// Shape mismatch in elementwise operation.
    #[snafu(display("shape mismatch: cannot perform elementwise operation on shapes {lhs_shape:?} and {rhs_shape:?}"))]
    ShapeMismatch { lhs_shape: Vec<usize>, rhs_shape: Vec<usize> },

    /// Shape mismatch in binary operation.
    #[snafu(display("Shape mismatch in {op:?}: {lhs:?} vs {rhs:?}"))]
    BinaryShapeMismatch { op: crate::types::BinaryOp, lhs: Box<Shape>, rhs: Box<Shape> },

    /// Reshape contains negative dimension.
    #[snafu(display("reshape contains negative dimension in {shape:?}"))]
    ReshapeNegativeDimension { shape: SmallVec<[isize; 4]> },

    /// Broadcasting shape mismatch.
    #[snafu(display("cannot broadcast shapes {lhs:?} and {rhs:?}"))]
    BroadcastShapeMismatch { lhs: Box<Shape>, rhs: Box<Shape> },

    /// Symbolic padding unsupported.
    #[snafu(display("symbolic padding is not supported: padding dimensions must be concrete values"))]
    SymbolicPaddingUnsupported,

    /// Symbolic shrinking unsupported.
    #[snafu(display("symbolic shrinking is not supported: shrink ranges must be concrete values"))]
    SymbolicShrinkingUnsupported,

    /// Ternary branch shape mismatch.
    #[snafu(display(
        "ternary operation branches have mismatched shapes: true branch {true_branch:?} vs false branch {false_branch:?}"
    ))]
    TernaryBranchShapeMismatch { true_branch: Box<Shape>, false_branch: Box<Shape> },

    /// DefineGlobal/DefineLocal must have Ptr dtype.
    #[snafu(display(
        "{op} must have Ptr dtype (following Tinygrad spec), got {dtype:?}. Use DefineVar for scalar variables."
    ))]
    DefineGlobalRequiresPtrDType { op: &'static str, dtype: DType },
}
