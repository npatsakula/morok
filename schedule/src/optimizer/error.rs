use snafu::Snafu;

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
#[snafu(visibility(pub))]
pub enum OptError {
    #[snafu(display("invalid argument type for operation; expected {expected}, found {found}"))]
    InvalidArgType { expected: &'static str, found: &'static str },
    #[snafu(display("operation validation failed for {op}: {reason}"))]
    ValidationFailed { op: &'static str, reason: &'static str },
    #[snafu(display("axis out of bounds: axis {axis} > max {max}"))]
    AxisOutOfBounds { axis: usize, max: usize },
    #[snafu(display("division constraint violated: {size} is not divisible by {amount}"))]
    DivisionError { size: usize, amount: usize },
    #[snafu(display("symbolic size cannot be verified for divisibility by {amount}"))]
    SymbolicDivisionError { amount: usize },
    #[snafu(display("expected Range operation, found other operation type"))]
    ExpectedRangeOperation,
    #[snafu(display("missing axis parameter for operation"))]
    MissingAxisParameter,
    #[snafu(display("backend doesn't support required feature: {feature}"))]
    UnsupportedFeature { feature: &'static str },
    #[snafu(display("optimization would exceed device limit: {limit_type} {value} > max {max}"))]
    DeviceLimitExceeded { limit_type: &'static str, value: usize, max: usize },
    // /// Tensor core pattern not matched.
    // TensorCoreNotMatched { reason: String },
}
