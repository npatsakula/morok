use morok_dtype::DType;
use smallvec::SmallVec;
use snafu::Snafu;

use crate::{BinaryOp, UnaryOp, shape::Shape};

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
    #[snafu(display("invalid dtype for operation: operation {operation:?}; dtype {dtype:?}"))]
    InvalidDTypeForUnaryOp { operation: UnaryOp, dtype: DType },

    /// Invalid dtype for operation (e.g., bitwise on float).
    #[snafu(display("invalid dtype for operation: operation {operation:?}; dtypes {dtypes:?}"))]
    InvalidDTypeForBinaryOp { operation: BinaryOp, dtypes: SmallVec<[DType; 2]> },

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

    /// Symbolic shape unsupported.
    #[snafu(display("symbolic shape is not supported for {operation}: shape dimensions must be concrete values"))]
    SymbolicShapeUnsupported { operation: String },

    /// Symbolic buffer size unsupported.
    #[snafu(display("cannot allocate buffer with symbolic size: range bound resolved to {bound:?}"))]
    SymbolicBufferSize { bound: crate::ConstValue },

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

    // =========================================================================
    // UOp Builder Guards (user-facing API for kernel implementation)
    // =========================================================================
    /// VECTORIZE requires at least one element.
    #[snafu(display("VECTORIZE requires at least one element"))]
    VectorizeEmpty,

    /// VECTORIZE elements have mismatched dtypes.
    #[snafu(display("VECTORIZE elements have mismatched dtypes: expected {expected:?}, got {actual:?}"))]
    VectorizeDTypeMismatch { expected: DType, actual: DType },

    /// GEP index out of bounds.
    #[snafu(display("GEP index {index} out of bounds for vector with {vcount} elements"))]
    GepIndexOutOfBounds { index: usize, vcount: usize },

    /// GEP requires vector source.
    #[snafu(display("GEP requires vector source (vcount > 1), got {dtype:?}"))]
    GepRequiresVector { dtype: DType },

    /// CONTRACT dtype count != axis product.
    #[snafu(display("CONTRACT dtype count {dtype_count} != axis product {axis_product}"))]
    ContractCountMismatch { dtype_count: usize, axis_product: usize },

    /// UNROLL src dtype count != axis product.
    #[snafu(display("UNROLL src dtype count {dtype_count} != axis product {axis_product}"))]
    UnrollCountMismatch { dtype_count: usize, axis_product: usize },

    /// WHERE condition must be bool.
    #[snafu(display("WHERE condition must be bool, got {actual:?}"))]
    WhereConditionNotBool { actual: DType },

    /// BROADCAST requires scalar source.
    #[snafu(display("BROADCAST requires scalar source (vcount=1), got {dtype:?}"))]
    BroadcastRequiresScalar { dtype: DType },
}

/// Enhance an error with provenance information for a UOp.
///
/// This function retrieves the provenance chain for a UOp and logs it,
/// providing detailed debugging information about the operation's origin and
/// transformation history.
pub fn log_provenance(uop_id: u64, error: &Error) {
    use crate::provenance::{PROVENANCE_TRACKER, format_chain};

    PROVENANCE_TRACKER.with(|tracker| {
        let chain = tracker.borrow().get_chain(uop_id);
        if !chain.is_empty() {
            tracing::error!(
                uop.id = uop_id,
                error = %error,
                provenance_chain = %format_chain(&chain),
                "uop error with provenance"
            );
        }
    });
}
