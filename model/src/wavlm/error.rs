use snafu::Snafu;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("{source}"), context(false))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("state-dict op failed"), context(false))]
    State {
        #[snafu(source(from(crate::state::Error, Box::new)))]
        source: Box<crate::state::Error>,
    },
    #[snafu(display("HF Hub op failed"), context(false))]
    Hub { source: hf_hub::HFError },
    #[snafu(display("pickle loader failed"))]
    Pickle {
        #[snafu(source(from(crate::wespeaker::pickle::Error, Box::new)))]
        source: Box<crate::wespeaker::pickle::Error>,
    },
    #[snafu(display(
        "weight-norm reconstruction failed: missing or mismatched parametrizations.weight.original* (g={g_shape:?}, v={v_shape:?})"
    ))]
    WeightNormShape { g_shape: Vec<usize>, v_shape: Vec<usize> },
}
