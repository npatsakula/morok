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
    #[snafu(display("HF Hub op failed"))]
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
    #[snafu(display("{what} requires a concrete sequence length, got a symbolic dim"))]
    SymbolicShape { what: &'static str },
}
