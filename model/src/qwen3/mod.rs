//! Qwen3 — decoder-only LLM backbone and embedding model.
//!
//! Pure-Rust port of the Qwen3 architecture (as used by
//! `Qwen/Qwen3-Embedding-0.6B`): token embeddings → causal RoPE decoder
//! stack (GQA, per-head Q/K RMSNorm, SwiGLU) → final RMSNorm.
//!
//! The embedding model adds last-token pooling + L2 normalization.
//! Computes in bf16 on the AMD target; f32 for CPU parity. Tokenization is
//! out-of-band — the API takes pre-computed `input_ids`.

mod attention;
mod config;
mod decoder_layer;
mod embedder;
mod embeddings;
mod error;
mod feed_forward;
mod jit;
mod model;
mod reranker;
mod rms_norm;
mod rotary;

pub use attention::Qwen3Attention;
pub use config::{Qwen3Config, qwen3_embedding_0_6b};
pub use decoder_layer::Qwen3DecoderLayer;
pub use embedder::Qwen3Embedding;
pub use embeddings::Qwen3Embeddings;
pub use error::{Error, Result};
pub use feed_forward::Qwen3MLP;
pub use jit::{Qwen3EmbeddingJit, Qwen3RerankerJit};
pub use model::Qwen3Model;
pub use reranker::Qwen3Reranker;
pub use rms_norm::RmsNormWeights;
pub use rotary::RotaryTable;

use std::path::PathBuf;

use crate::hub::HubRepo;
use snafu::ResultExt;

/// Download all safetensors weight files from a HuggingFace repo into the
/// hub cache and return the cache directory containing them.
///
/// Handles both single-file (`model.safetensors`) and multi-shard
/// (`model-NNNNN-of-NNNNN.safetensors` + `model.safetensors.index.json`).
pub(crate) fn download_safetensors(repo: &HubRepo) -> Result<PathBuf> {
    // Try single-file first.
    if let Ok(path) = repo.get("model.safetensors") {
        return Ok(path.parent().expect("non-root cache dir").to_path_buf());
    }

    // Multi-shard: parse the index to discover shard filenames.
    let index_path = repo.get("model.safetensors.index.json").context(error::HubSnafu)?;
    let dir = index_path.parent().expect("non-root cache dir").to_path_buf();

    // The index file is already downloaded. But we need to ensure all shard
    // files are too — `repo.get` downloads on demand.
    let index_data = std::fs::read_to_string(&index_path)
        .map_err(|e| Error::Config { message: format!("reading safetensors index: {e}") })?;
    let index: crate::state::SafetensorsIndex = serde_json::from_str(&index_data)
        .map_err(|e| Error::Config { message: format!("parsing safetensors index: {e}") })?;

    for shard in index.unique_shards() {
        repo.get(&shard).context(error::HubSnafu)?;
    }

    Ok(dir)
}
