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
    Hub { source: hf_hub::api::sync::ApiError },
    #[snafu(display("reading config.json failed: {message}"))]
    Config { message: String },
    #[snafu(display("{what} requires a concrete sequence length, got a symbolic dim"))]
    SymbolicShape { what: &'static str },
    #[snafu(display("{what} requires an attention mask"))]
    MissingMask { what: &'static str },
    #[snafu(display("loading tokenizer failed: {source}"))]
    Tokenizer { source: svod_arch::pipelines::text::HfTokenizerError },
    #[snafu(display("building encoder head failed: {source}"))]
    Head { source: super::head_jit::HeadError },
}
