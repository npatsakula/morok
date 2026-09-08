//! BGE-M3 composite model: XLM-RoBERTa backbone + dense / sparse / ColBERT heads.
//!
//! `BAAI/bge-m3` produces three types of embeddings simultaneously:
//! - **Dense**: CLS pooling + L2 normalize → `(B, D)`
//! - **Sparse**: Linear → ReLU → scatter-to-vocab → `(B, vocab_size)`
//! - **ColBERT**: per-token Linear (skip CLS) + L2 normalize → `(B, L-1, Dc)`
//!
//! Loads from HuggingFace Hub: `config.json` + `pytorch_model.bin` (backbone)
//! + `sparse_linear.pt` + `colbert_linear.pt` (heads).

use snafu::OptionExt;
use svod_tensor::Tensor;

use crate::xlm_roberta::config::XlmRobertaConfig;
use crate::xlm_roberta::error::{MissingHeadSnafu, Result};

use crate::xlm_roberta::model::XlmRobertaModel;

use super::colbert_head::ColbertHead;
use super::sparse_head::SparseHead;

/// Output of [`BgeM3::encode`]: any combination of the three embedding types.
#[derive(Clone, Default)]
pub struct BgeM3Output {
    pub dense_vecs: Option<Tensor>,
    pub sparse_vecs: Option<Tensor>,
    pub colbert_vecs: Option<Tensor>,
}

/// Flags controlling which embedding types to compute.
#[derive(Clone, Copy, Debug, Default)]
pub struct EncodeOpts {
    pub return_dense: bool,
    pub return_sparse: bool,
    pub return_colbert: bool,
}

impl EncodeOpts {
    pub fn dense() -> Self {
        Self { return_dense: true, return_sparse: false, return_colbert: false }
    }
    pub fn all() -> Self {
        Self { return_dense: true, return_sparse: true, return_colbert: true }
    }
}

#[derive(Clone)]
pub struct BgeM3 {
    pub model: XlmRobertaModel,
    pub sparse_head: Option<SparseHead>,
    pub colbert_head: Option<ColbertHead>,
    pub normalize_dense: bool,
}

impl BgeM3 {
    pub fn empty(config: XlmRobertaConfig) -> Self {
        let dtype = config.dtype.clone();
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let model = XlmRobertaModel::empty(config);
        Self {
            model,
            sparse_head: Some(SparseHead::empty(hidden, vocab, dtype.clone())),
            colbert_head: Some(ColbertHead::empty(hidden, hidden, dtype)),
            normalize_dense: true,
        }
    }

    /// Eager forward computing all requested embedding types.
    pub fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor, opts: EncodeOpts) -> Result<BgeM3Output> {
        let hidden = self.model.forward(input_ids, Some(attention_mask))?;
        let mut out = BgeM3Output::default();

        if opts.return_dense {
            let dense = hidden.take_index(1, 0)?;
            out.dense_vecs = Some(if self.normalize_dense { dense.lp_normalize(-1, 2)? } else { dense });
        }
        if opts.return_sparse {
            let head = self.sparse_head.as_ref().context(MissingHeadSnafu { head: "sparse" })?;
            out.sparse_vecs = Some(head.forward(&hidden, input_ids)?);
        }
        if opts.return_colbert {
            let head = self.colbert_head.as_ref().context(MissingHeadSnafu { head: "colbert" })?;
            out.colbert_vecs = Some(head.forward(&hidden, Some(attention_mask))?);
        }
        Ok(out)
    }

    /// Dense-only forward (most common path). Returns `(B, D)`.
    pub fn encode_dense(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let hidden = self.model.forward(input_ids, Some(attention_mask))?;
        let dense = hidden.take_index(1, 0)?;
        if self.normalize_dense { Ok(dense.lp_normalize(-1, 2)?) } else { Ok(dense) }
    }

    /// ColBERT-only forward. Returns `(B, L-1, Dc)`.
    pub fn encode_colbert(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let hidden = self.model.forward(input_ids, Some(attention_mask))?;
        let head = self.colbert_head.as_ref().context(MissingHeadSnafu { head: "colbert" })?;
        head.forward(&hidden, Some(attention_mask))
    }

    /// Download all files from HuggingFace Hub and load the full BGE-M3 model.
    pub fn from_hub(model_id: &str, mut config: XlmRobertaConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", &mut config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: &mut XlmRobertaConfig) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;

        let cfg_path = repo.get("config.json")?;
        let parsed = XlmRobertaConfig::from_json(&cfg_path)?;
        config.merge_structural_from(&parsed);

        let weights_path = repo.get("pytorch_model.bin")?;
        let dtype = config.dtype.clone();
        let model = XlmRobertaModel::from_pytorch_bin(&weights_path, config.clone())?;

        let sparse_head = match repo.get("sparse_linear.pt") {
            Ok(path) => Some(SparseHead::from_pytorch_bin(&path, config.vocab_size, dtype.clone())?),
            Err(_) => None,
        };
        let colbert_head = match repo.get("colbert_linear.pt") {
            Ok(path) => Some(ColbertHead::from_pytorch_bin(&path, config.hidden_size, dtype)?),
            Err(_) => None,
        };

        Ok(Self { model, sparse_head, colbert_head, normalize_dense: true })
    }
}
