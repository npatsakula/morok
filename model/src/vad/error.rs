use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("{source}"))]
    Tensor {
        #[snafu(source(from(morok_tensor::error::Error, Box::new)))]
        source: Box<morok_tensor::error::Error>,
    },
    #[snafu(display("{source}"))]
    State {
        #[snafu(source(from(crate::state::Error, Box::new)))]
        source: Box<crate::state::Error>,
    },
    #[snafu(display("hub error: {source}"))]
    Hub { source: hf_hub::api::sync::ApiError },
}

pub type Result<T> = std::result::Result<T, Error>;
