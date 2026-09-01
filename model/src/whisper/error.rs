use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("{source}"))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("{source}"))]
    Jit {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("{source}"))]
    Device {
        #[snafu(source(from(svod_device::error::Error, Box::new)))]
        source: Box<svod_device::error::Error>,
    },
    #[snafu(display("{source}"))]
    State { source: crate::state::Error },
    #[snafu(display("model not found: {name}"))]
    ModelNotFound { name: String },
    #[snafu(display("download failed: {source}"))]
    Download { source: std::io::Error },
    #[snafu(display("{source}"))]
    Io { source: std::io::Error },
    #[snafu(display("checkpoint error: {msg}"))]
    Checkpoint { msg: String },
    #[snafu(display("tokenizer error: {msg}"))]
    Tokenizer { msg: String },
    #[snafu(display("decode error: {msg}"))]
    Decode { msg: String },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Bridge a `svod-tk` launch error into the tensor error domain. tk's launch
/// `Err` means a structurally invalid request (a caller bug — fallback-worthy
/// conditions come back as `Ok(None)` instead), so it surfaces as an IR
/// construction failure. One definition so the launch-error contract has a
/// single point of change across the attention call sites.
pub(crate) fn tk_launch_error(e: impl std::fmt::Display) -> svod_tensor::error::Error {
    svod_tensor::error::Error::IrConstruction { details: e.to_string() }
}
