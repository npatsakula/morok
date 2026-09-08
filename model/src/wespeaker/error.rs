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
    #[snafu(display("{source}"))]
    Blocks {
        #[snafu(source(from(crate::blocks::Error, Box::new)))]
        source: Box<crate::blocks::Error>,
    },
    #[snafu(display("hub error: {source}"))]
    Hub { source: hf_hub::HFError },
    #[snafu(display("{source}"))]
    Pickle {
        #[snafu(source(from(super::pickle::Error, Box::new)))]
        source: Box<super::pickle::Error>,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<crate::blocks::Error> for Error {
    fn from(e: crate::blocks::Error) -> Self {
        Error::Blocks { source: Box::new(e) }
    }
}
