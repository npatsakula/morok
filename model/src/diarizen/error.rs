use snafu::Snafu;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("tensor op failed"))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("state-dict op failed"))]
    State {
        #[snafu(source(from(crate::state::Error, Box::new)))]
        source: Box<crate::state::Error>,
    },
    #[snafu(display("WavLM error"))]
    WavLm {
        #[snafu(source(from(crate::wavlm::Error, Box::new)))]
        source: Box<crate::wavlm::Error>,
    },
    #[snafu(display("HF Hub op failed"))]
    Hub { source: hf_hub::HFError },
    #[snafu(display("pickle loader failed"))]
    Pickle {
        #[snafu(source(from(crate::wespeaker::pickle::Error, Box::new)))]
        source: Box<crate::wespeaker::pickle::Error>,
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
    #[snafu(display("waveform is {wav_sr} Hz, model expects {model_sr} Hz (resample first)"))]
    SampleRateMismatch { wav_sr: u32, model_sr: u32 },
}

impl From<crate::wavlm::Error> for Error {
    fn from(e: crate::wavlm::Error) -> Self {
        use crate::wavlm::Error as W;
        match e {
            W::Tensor { source } => Error::Tensor { source },
            W::State { source } => Error::State { source },
            W::Pickle { source } => Error::Pickle { source },
            W::Hub { source } => Error::Hub { source },
            other => Error::WavLm { source: Box::new(other) },
        }
    }
}

impl From<crate::blocks::Error> for Error {
    fn from(e: crate::blocks::Error) -> Self {
        match e {
            crate::blocks::Error::Tensor { source } => Error::Tensor { source },
        }
    }
}
