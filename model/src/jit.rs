use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum JitError {
    #[snafu(display("JIT not prepared: call prepare() first"))]
    NotPrepared,

    #[snafu(display("input buffer not found: {name}"))]
    InputBufferNotFound { name: &'static str },

    #[snafu(display("duplicate JIT input buffer: {name} aliases {duplicate_of} with {buffer_id:?}"))]
    DuplicateInputBuffer { name: &'static str, duplicate_of: &'static str, buffer_id: morok_device::BufferId },

    #[snafu(display("{source}"))]
    Build { source: Box<dyn std::error::Error + Send + Sync> },

    #[snafu(display("{source}"))]
    Tensor { source: Box<morok_tensor::error::Error> },

    #[snafu(display("{source}"))]
    Runtime { source: morok_runtime::Error },
}

pub type Result<T> = std::result::Result<T, JitError>;
