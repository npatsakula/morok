use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("IR operation error"))]
    UOp { source: morok_ir::Error },

    #[snafu(display("Tensor shape is unknown (symbolic or not yet inferred)"))]
    ShapeUnknown,

    #[snafu(display("Operation '{operation}' does not support symbolic shapes"))]
    SymbolicShapeUnsupported { operation: String },

    #[snafu(display("Axis {axis} is out of range for tensor with {ndim} dimensions"))]
    AxisOutOfRange { axis: isize, ndim: usize },

    #[snafu(display("Permutation length mismatch: expected {expected} axes, got {got}"))]
    PermutationLengthMismatch { expected: usize, got: usize },

    #[snafu(display("Invalid permutation: axes {axes:?} is not a valid permutation"))]
    InvalidPermutation { axes: Vec<isize> },

    #[snafu(display("Multiple -1 dimensions in reshape are not allowed"))]
    MultipleInferDimensions,

    #[snafu(display("Negative dimension {dim} is not allowed (except -1 for inference)"))]
    NegativeDimension { dim: isize },

    #[snafu(display("Reshape size mismatch during {operation}"))]
    ReshapeSizeMismatch { operation: String },

    #[snafu(display(
        "Expand dimension mismatch: current shape has {current_dims} dims, target has {target_dims} dims"
    ))]
    ExpandDimensionMismatch { current_dims: usize, target_dims: usize },

    #[snafu(display("Cannot squeeze dimension {dim}: size is {size}, not 1"))]
    SqueezeDimensionNotOne { dim: usize, size: usize },

    #[snafu(display("Cannot specify both 'dtype' and 'promote=true' in reduction operation"))]
    ConflictingReductionOptions,

    #[snafu(display(
        "Matrix multiplication requires tensors with at least 1 dimension, got lhs: {lhs_dims}D, rhs: {rhs_dims}D"
    ))]
    DotDimensionError { lhs_dims: usize, rhs_dims: usize },

    #[snafu(display(
        "Matrix multiplication shape mismatch: cannot multiply shapes {lhs_shape:?} and {rhs_shape:?} (contraction dimension mismatch)"
    ))]
    DotShapeMismatch { lhs_shape: Vec<usize>, rhs_shape: Vec<usize> },
}

pub type Result<T> = std::result::Result<T, Error>;
