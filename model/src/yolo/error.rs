use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("{source}"))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("{source}"))]
    State {
        #[snafu(source(from(crate::state::Error, Box::new)))]
        source: Box<crate::state::Error>,
    },
    #[snafu(display("hub error: {source}"))]
    Hub { source: hf_hub::HFError },
    #[snafu(display("invalid yolo config: {message}"))]
    Config { message: String },
    #[snafu(display("{what} requires a concrete shape, got a symbolic dim"))]
    SymbolicShape { what: &'static str },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<crate::blocks::Error> for Error {
    fn from(e: crate::blocks::Error) -> Self {
        match e {
            crate::blocks::Error::Tensor { source } => Error::Tensor { source },
        }
    }
}

impl From<svod_tensor::error::Error> for Error {
    fn from(e: svod_tensor::error::Error) -> Self {
        Error::Tensor { source: Box::new(e) }
    }
}
