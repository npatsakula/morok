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
    #[snafu(display("invalid resnet config: {message}"))]
    Config { message: String },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<crate::blocks::Error> for Error {
    fn from(e: crate::blocks::Error) -> Self {
        match e {
            crate::blocks::Error::Tensor { source } => Error::Tensor { source },
        }
    }
}
