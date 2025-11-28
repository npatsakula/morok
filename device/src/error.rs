use snafu::Snafu;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    /// Shape of target tensor does not match expected shape.
    #[snafu(display("shape mismatch: expected {expected:?}, got {actual:?}"))]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[snafu(display("size mismatch: expected {expected}, got {actual}"))]
    SizeMismatch { expected: usize, actual: usize },

    /// Failed to copy data between host and device.
    #[snafu(display("copy operation failed: {reason}"))]
    CopyFailed { reason: String },

    /// Invalid device specification.
    #[snafu(display("invalid device: {device}"))]
    InvalidDevice { device: String },

    /// Buffer is not allocated.
    #[snafu(display("buffer not allocated"))]
    NotAllocated,

    /// Invalid buffer view parameters.
    #[snafu(display("invalid view: offset {offset} + size {size} exceeds buffer size {buffer_size}"))]
    InvalidView { offset: usize, size: usize, buffer_size: usize },

    #[cfg(feature = "cuda")]
    /// CUDA-specific errors.
    #[snafu(display("CUDA error: {source}"))]
    CudaError { source: cudarc::driver::DriverError },
}
