//! One-call hub loaders for ModernBERT: fetch `config.json`,
//! `model.safetensors`, and `tokenizer.json` from a pinned revision, then
//! assemble the tokenizer + model ready for an
//! [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline). The three
//! loaders share one [`fetch_hub`] prelude; each supplies only its model-build
//! tail.

use snafu::ResultExt;
use svod_arch::pipelines::text::HfTokenizer;
use svod_dtype::DType;

use super::classifier::{ModernBertClassificationModel, ModernBertClassifier};
use super::config::ModernBertConfig;
use super::embedder::ModernBertEmbedder;
use super::error::{HeadSnafu, HubSnafu, Result, StateSnafu, TokenizerSnafu};
use super::model::ModernBert;
use super::token_classifier::{ModernBertTokenClassificationModel, ModernBertTokenClassifier};
use crate::state::StateDict;

// ── shared hub-fetch prelude ───────────────────────────────────────────────

/// Fetch `config.json` + `model.safetensors` + `tokenizer.json` from a pinned
/// hub revision. Seeds the config from [`ModernBertConfig::default`] with the caller-chosen
/// `dtype` + `max_batch`, then splices every structural field (including
/// `id2label`) from the downloaded `config.json` via
/// [`ModernBertConfig::apply_checkpoint`]. Returns the resolved config, the
/// loaded state dict, and the tokenizer — everything the three head loaders need
/// to build their model + head.
fn fetch_hub(
    model_id: &str,
    revision: &str,
    max_batch: usize,
    dtype: DType,
) -> Result<(ModernBertConfig, StateDict, HfTokenizer)> {
    let mut config = ModernBertConfig { dtype, max_batch_size: max_batch, ..Default::default() };

    let api = hf_hub::api::sync::Api::new().context(HubSnafu)?;
    let repo =
        api.repo(hf_hub::Repo::with_revision(model_id.to_string(), hf_hub::RepoType::Model, revision.to_string()));

    let cfg_path = repo.get("config.json").context(HubSnafu)?;
    let parsed = ModernBertConfig::from_json(&cfg_path)?;
    config.apply_checkpoint(&parsed);

    let weights_path = repo.get("model.safetensors").context(HubSnafu)?;
    let sd = crate::state::load_safetensors(&weights_path).context(StateSnafu)?;

    let tok_path = repo.get("tokenizer.json").context(HubSnafu)?;
    let tokenizer = HfTokenizer::from_path(&tok_path).context(TokenizerSnafu)?;

    Ok((config, sd, tokenizer))
}

// ── embedder ───────────────────────────────────────────────────────────────

/// Download `config.json` + `model.safetensors` + `tokenizer.json` from a
/// HuggingFace Hub repository (default revision `"main"`) and assemble the
/// `(HfTokenizer, ModernBertEmbedder)` pair ready for an
/// [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).
///
/// `max_seq` is derived from the checkpoint's `max_position_embeddings`;
/// `max_batch` is caller-chosen (not in `config.json`). `dtype` selects the
/// compute precision (bf16 for GPU, f32 for CPU parity).
///
/// See [`from_hub_with_revision`] for the per-revision form.
pub fn from_hub(model_id: &str, max_batch: usize, dtype: DType) -> Result<(HfTokenizer, ModernBertEmbedder)> {
    from_hub_with_revision(model_id, "main", max_batch, dtype)
}

/// Per-revision form of [`from_hub`].
pub fn from_hub_with_revision(
    model_id: &str,
    revision: &str,
    max_batch: usize,
    dtype: DType,
) -> Result<(HfTokenizer, ModernBertEmbedder)> {
    let (config, sd, tokenizer) = fetch_hub(model_id, revision, max_batch, dtype)?;
    let max_seq = config.max_position_embeddings;
    let model = ModernBert::from_state_dict(&sd, config)?;
    let embedder = ModernBertEmbedder::new(model, max_batch, max_seq).context(HeadSnafu)?;
    Ok((tokenizer, embedder))
}

// ── classifier ─────────────────────────────────────────────────────────────

/// Loaded sequence-classifier pair: the tokenizer, the head, and the dense
/// label-name table from `config.json` (`id2label`), so a caller can decode
/// predicted class ids to names without re-fetching the config.
pub struct ModernBertClassifierLoad {
    pub tokenizer: HfTokenizer,
    pub classifier: ModernBertClassifier,
    pub id2label: Vec<String>,
}

/// Download `config.json` + `model.safetensors` + `tokenizer.json` from a
/// HuggingFace Hub repository (default revision `"main"`) and assemble a
/// [`ModernBertClassifierLoad`] ready for an
/// [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline). `id2label`
/// is the dense label-name vec parsed from `config.json`.
pub fn from_hub_classifier(model_id: &str, max_batch: usize, dtype: DType) -> Result<ModernBertClassifierLoad> {
    from_hub_classifier_with_revision(model_id, "main", max_batch, dtype)
}

/// Per-revision form of [`from_hub_classifier`].
pub fn from_hub_classifier_with_revision(
    model_id: &str,
    revision: &str,
    max_batch: usize,
    dtype: DType,
) -> Result<ModernBertClassifierLoad> {
    let (config, sd, tokenizer) = fetch_hub(model_id, revision, max_batch, dtype)?;
    let model = ModernBertClassificationModel::from_state_dict(&sd, &config)?;
    let max_seq = config.max_position_embeddings;
    let classifier = ModernBertClassifier::new(model, max_batch, max_seq).context(HeadSnafu)?;
    Ok(ModernBertClassifierLoad { tokenizer, classifier, id2label: config.id2label })
}

// ── token classification ───────────────────────────────────────────────────

/// Loaded token-classifier pair: the tokenizer, the head, and the dense
/// label-name table from `config.json` (`id2label`), so a caller can decode
/// predicted token-label ids to names without re-fetching the config.
pub struct ModernBertTokenClassifierLoad {
    pub tokenizer: HfTokenizer,
    pub classifier: ModernBertTokenClassifier,
    pub id2label: Vec<String>,
}

/// Download `config.json` + `model.safetensors` + `tokenizer.json` from a
/// HuggingFace Hub repository (default revision `"main"`) and assemble a
/// [`ModernBertTokenClassifierLoad`] ready for an
/// [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline). `id2label`
/// is the dense label-name vec parsed from `config.json`.
pub fn from_hub_token_classification(
    model_id: &str,
    max_batch: usize,
    dtype: DType,
) -> Result<ModernBertTokenClassifierLoad> {
    from_hub_token_classification_with_revision(model_id, "main", max_batch, dtype)
}

/// Per-revision form of [`from_hub_token_classification`].
pub fn from_hub_token_classification_with_revision(
    model_id: &str,
    revision: &str,
    max_batch: usize,
    dtype: DType,
) -> Result<ModernBertTokenClassifierLoad> {
    let (config, sd, tokenizer) = fetch_hub(model_id, revision, max_batch, dtype)?;
    let model = ModernBertTokenClassificationModel::from_state_dict(&sd, &config)?;
    let max_seq = config.max_position_embeddings;
    let classifier = ModernBertTokenClassifier::new(model, max_batch, max_seq).context(HeadSnafu)?;
    Ok(ModernBertTokenClassifierLoad { tokenizer, classifier, id2label: config.id2label })
}
