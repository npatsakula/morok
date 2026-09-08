//! OpenAI Whisper: encoder-decoder transformer for speech recognition.
//!
//! Architecture: convolutional frontend, sinusoidal encoder positions, and a
//! pre-norm transformer encoder; the learned-position decoder uses cached
//! self-attention and cross-attention over encoder features. Selected
//! cross-attention heads provide DTW word alignment.
//!
//! # Quick start
//!
//! ```no_run
//! use svod_model::whisper::{Whisper, WhisperSize, ModelDimensions};
//!
//! let dims = ModelDimensions::for_size(WhisperSize::Tiny);
//! let model = Whisper::empty(dims);
//! ```

pub mod aligner;
pub mod attention;
pub mod blocks;
pub mod config;
pub mod decode;
pub mod decoder;
pub mod dtw;
pub mod encoder;
pub mod error;
pub mod jit;
pub mod mel;
pub mod model;
pub mod plan;
pub mod tokenizer;
pub mod transcribe;

mod loader;
pub(crate) mod profile;

pub use aligner::{WhisperAligner, WhisperAlignmentInput};
pub use attention::MultiHeadAttention;
pub use blocks::sinusoids;
pub use config::{ModelDimensions, WhisperSize};
pub use decode::{
    DecodeOptions, DecodeResult, DecodeStrategy, FallbackPolicy, LanguageDetection, WhisperTask, detect_language,
    split_into_segments,
};
pub use decoder::{DecoderBlock, TextDecoder};
pub use dtw::{
    WordTiming, dtw, find_alignment_path, find_alignment_path_selected, median_filter, path_to_word_timings,
};
pub use encoder::{AudioEncoder, EncoderBlock};
pub use error::{Error, Result};
pub use jit::{
    WhisperAlignmentJit, WhisperAlignmentModel, WhisperCrossKvJit, WhisperDecoderJit, WhisperDecoderStepJit,
    WhisperEncoderJit, WhisperPrefillJit,
};
pub use mel::WhisperMel;
pub use model::Whisper;
pub use plan::WhisperPlan;
pub use tokenizer::WhisperTokenizer;
pub use transcribe::{TranscribeError, WhisperAlignedTranscriber, WhisperRecognizer};

// Re-export audio constants
pub use config::{
    CHUNK_LENGTH, FRAMES_PER_SECOND, HOP_LENGTH, N_AUDIO_CTX, N_FFT, N_FRAMES, N_SAMPLES, N_SAMPLES_PER_TOKEN,
    N_TEXT_CTX, SAMPLE_RATE, TOKENS_PER_SECOND,
};
