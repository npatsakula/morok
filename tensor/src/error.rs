use morok_ir::shape::Shape;
use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    // =========================================================================
    // IR Layer Errors
    // =========================================================================
    #[snafu(display("IR operation error: {source}"))]
    UOp { source: morok_ir::Error },

    // =========================================================================
    // Shape Errors
    // =========================================================================
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

    // =========================================================================
    // NN Input Validation Errors
    // =========================================================================
    #[snafu(display("{op} requires exactly {expected}D input, got {actual}D"))]
    NdimExact { op: &'static str, expected: usize, actual: usize },

    #[snafu(display("{op} requires at least {min}D input, got {actual}D"))]
    NdimMinimum { op: &'static str, min: usize, actual: usize },

    #[snafu(display("{op}: {lhs_name} ({lhs}) must be divisible by {rhs_name} ({rhs})"))]
    Divisibility { op: &'static str, lhs_name: &'static str, lhs: usize, rhs_name: &'static str, rhs: usize },

    #[snafu(display("{op}: {param} = {value} is invalid, expected {constraint}"))]
    ParamRange { op: &'static str, param: &'static str, value: String, constraint: &'static str },

    // =========================================================================
    // Reduction Errors
    // =========================================================================
    #[snafu(display("Cannot specify both 'dtype' and 'promote=true' in reduction operation"))]
    ConflictingReductionOptions,

    // =========================================================================
    // Matrix Multiplication Errors
    // =========================================================================
    #[snafu(display(
        "Matrix multiplication requires tensors with at least 1 dimension, got lhs: {lhs_dims}D, rhs: {rhs_dims}D"
    ))]
    DotDimensionError { lhs_dims: usize, rhs_dims: usize },

    #[snafu(display(
        "Matrix multiplication shape mismatch: cannot multiply shapes {lhs_shape:?} and {rhs_shape:?} (contraction dimension mismatch)"
    ))]
    DotShapeMismatch { lhs_shape: Box<Shape>, rhs_shape: Box<Shape> },

    // =========================================================================
    // Broadcasting Errors
    // =========================================================================
    #[snafu(display(
        "Cannot broadcast to fewer dimensions: tensor has {from_dims} dimensions, target has {to_dims} dimensions"
    ))]
    BroadcastFewerDimensions { from_dims: usize, to_dims: usize },

    #[snafu(display(
        "Incompatible dimension {dim} for broadcasting: cannot broadcast size {from_size} to size {to_size}"
    ))]
    BroadcastIncompatible { dim: usize, from_size: usize, to_size: usize },

    // =========================================================================
    // Codegen Errors (from device traits that wrap codegen)
    // =========================================================================
    #[snafu(display("Failed to render kernel: {source}"))]
    RenderKernel { source: morok_device::Error },

    #[snafu(display("Failed to compile kernel: {source}"))]
    CompileKernel { source: morok_device::Error },

    // =========================================================================
    // Schedule/Pipeline Errors
    // =========================================================================
    #[snafu(display("Rangeify failed: {source}"))]
    Rangeify { source: morok_ir::Error },

    #[snafu(display("Optimization error: {source}"))]
    Optimize { source: morok_schedule::OptError },

    #[snafu(display("No kernels found after scheduling pipeline"))]
    NoKernelsFound,

    #[snafu(display("Schedule contains dependency cycles"))]
    DependencyCycles,

    #[snafu(display("Empty schedule"))]
    EmptySchedule,

    #[snafu(display("Batch output count mismatch: expected {expected}, got {actual}"))]
    BatchOutputMismatch { expected: usize, actual: usize },

    #[snafu(display("Expected KERNEL operation"))]
    ExpectedKernelOp,

    // =========================================================================
    // Runtime Errors
    // =========================================================================
    #[snafu(display("Execution failed: {source}"))]
    Execution { source: morok_runtime::Error },

    #[snafu(display("Failed to create program: {source}"))]
    CreateProgram { source: morok_device::Error },

    #[snafu(display("Failed to get device: {source}"))]
    DeviceFactory { source: morok_runtime::Error },

    #[snafu(display("Buffer for UOp {} not found in registry", uop_id))]
    BufferNotFound { uop_id: u64 },

    #[snafu(display("Device error: {source}"))]
    Device { source: morok_device::Error },

    // =========================================================================
    // Type Errors
    // =========================================================================
    #[snafu(display("Expected Ptr dtype for {context}, got {actual:?}"))]
    ExpectedPtrDtype { context: &'static str, actual: morok_dtype::DType },

    #[snafu(display("Buffer Ptr dtype has no size"))]
    BufferPtrNoSize,

    #[snafu(display("Tensor has no buffer (unrealized tensor?)"))]
    NoBuffer,

    #[snafu(display("Tensor has no shape"))]
    NoShape,

    #[snafu(display("Shape mismatch for '{context}': expected {expected}, got {actual}"))]
    ShapeMismatch { context: String, expected: String, actual: String },

    #[snafu(display("IR construction error: {details}"))]
    IrConstruction { details: String },

    #[snafu(display("Type mismatch: expected {expected:?}, got {actual:?}"))]
    TypeMismatch { expected: morok_dtype::DType, actual: morok_dtype::DType },

    #[snafu(display("{op} requires floating-point dtype for {arg}, got {dtype:?}"))]
    FloatDTypeRequired { op: &'static str, arg: &'static str, dtype: morok_dtype::DType },

    #[snafu(display("Failed to create ndarray: {source}"))]
    NdarrayShape { source: ndarray::ShapeError },

    // =========================================================================
    // Variable Errors
    // =========================================================================
    #[snafu(display("Variable '{name}' value {val} out of range [{min}, {max}]"))]
    VariableOutOfRange { name: String, val: i64, min: i64, max: i64 },

    #[snafu(display("Cannot read data from tensor with symbolic shape — reduce or slice to concrete shape first"))]
    SymbolicShape,
}

pub type Result<T> = std::result::Result<T, Error>;
