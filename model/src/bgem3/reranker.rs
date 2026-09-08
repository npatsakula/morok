//! BGE-reranker-v2-m3: XLM-RoBERTa backbone + sequence classification head.
//!
//! Cross-encoder reranker fine-tuned from `BAAI/bge-m3`. The tokenizer
//! concatenates `(query, passage)` into a single input sequence with
//! `<s> query </s></s> passage </s>`; the model runs the backbone, takes the
//! CLS token, applies the two-layer classification head (`dense → tanh →
//! out_proj`), and optionally sigmoid-normalizes.
//!
//! Same backbone as [`crate::xlm_roberta::XlmRobertaModel`] — only the head
//! differs. Loads from `model.safetensors` with `roberta.` prefix on backbone
//! keys.

use svod_dtype::DType;
use svod_tensor::{BoundVariable, Tensor};

use crate::init::{fan_in_uniform, zeros};
use crate::state::{self, HasStateDict, StateDict, cast_all, get_tensor, prefixed};
use crate::xlm_roberta::config::XlmRobertaConfig;
use crate::xlm_roberta::error::Result;

use crate::xlm_roberta::model::XlmRobertaModel;
use crate::xlm_roberta::pooling::cls;

/// `RobertaClassificationHead`: `dense(D, D) → tanh → out_proj(D, num_labels)`.
#[derive(Clone)]
pub struct ClassificationHead {
    pub dense_weight: Tensor,
    pub dense_bias: Tensor,
    pub out_proj_weight: Tensor,
    pub out_proj_bias: Tensor,
}

impl ClassificationHead {
    pub fn empty(hidden_size: usize, num_labels: usize, dtype: DType) -> Self {
        Self {
            dense_weight: fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype.clone()),
            dense_bias: zeros(&[hidden_size], dtype.clone()),
            out_proj_weight: fan_in_uniform(&[num_labels, hidden_size], hidden_size, dtype.clone()),
            out_proj_bias: zeros(&[num_labels], dtype),
        }
    }

    /// Forward: `CLS → dense → tanh → out_proj → (B, num_labels)`.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let cls_emb = cls(hidden)?;
        let h = cls_emb.linear().weight(&self.dense_weight).bias(&self.dense_bias).call()?;
        let h = h.tanh()?;
        Ok(h.linear().weight(&self.out_proj_weight).bias(&self.out_proj_bias).call()?)
    }
}

impl HasStateDict for ClassificationHead {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "dense.weight"), self.dense_weight.clone());
        sd.insert(prefixed(prefix, "dense.bias"), self.dense_bias.clone());
        sd.insert(prefixed(prefix, "out_proj.weight"), self.out_proj_weight.clone());
        sd.insert(prefixed(prefix, "out_proj.bias"), self.out_proj_bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.dense_weight = get_tensor(sd, &prefixed(prefix, "dense.weight"))?;
        self.dense_bias = get_tensor(sd, &prefixed(prefix, "dense.bias"))?;
        self.out_proj_weight = get_tensor(sd, &prefixed(prefix, "out_proj.weight"))?;
        self.out_proj_bias = get_tensor(sd, &prefixed(prefix, "out_proj.bias"))?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct BgeRerankerV2M3 {
    pub model: XlmRobertaModel,
    pub classifier: ClassificationHead,
}

impl BgeRerankerV2M3 {
    pub fn empty(config: XlmRobertaConfig) -> Self {
        let dtype = config.dtype.clone();
        let hidden = config.hidden_size;
        Self { model: XlmRobertaModel::empty(config), classifier: ClassificationHead::empty(hidden, 1, dtype) }
    }

    /// Forward: backbone → classification head → logits `(B, 1)`.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let hidden = self.model.forward(input_ids, attention_mask)?;
        self.classifier.forward(&hidden)
    }

    /// Score with optional sigmoid normalization. Returns `(B, 1)`.
    pub fn compute_score(&self, input_ids: &Tensor, attention_mask: &Tensor, normalize: bool) -> Result<Tensor> {
        let logits = self.forward(input_ids, Some(attention_mask))?;
        if normalize { Ok(logits.sigmoid()?) } else { Ok(logits) }
    }

    /// JIT-path forward with rebindable batch. Returns `(B, 1)`.
    pub fn forward_batch(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        b: &BoundVariable,
    ) -> Result<Tensor> {
        let hidden = self.model.forward_batch(input_ids, attention_mask, b)?;
        self.classifier.forward(&hidden)
    }

    /// Download from HuggingFace Hub and load.
    pub fn from_hub(model_id: &str, mut config: XlmRobertaConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", &mut config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: &mut XlmRobertaConfig) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;

        let cfg_path = repo.get("config.json")?;
        let parsed = XlmRobertaConfig::from_json(&cfg_path)?;
        config.merge_structural_from(&parsed);

        let weights_path = repo.get("model.safetensors")?;
        Self::from_safetensors(&weights_path, config.clone())
    }

    /// Load from a `model.safetensors` checkpoint. Strips the `roberta.`
    /// prefix from backbone keys.
    pub fn from_safetensors(path: &std::path::Path, config: XlmRobertaConfig) -> Result<Self> {
        let sd = crate::state::load_safetensors(path)?;
        Self::from_state_dict(&sd, config)
    }

    /// Build from a preloaded state dict. Strips `roberta.` prefix.
    pub fn from_state_dict(sd: &StateDict, config: XlmRobertaConfig) -> Result<Self> {
        let dtype = config.dtype.clone();
        let stripped = strip_roberta_prefix(sd);
        let sd_cast = cast_all(&stripped, dtype);
        let mut model = Self::empty(config);
        model.load_state_dict(&sd_cast, "")?;
        Ok(model)
    }
}

impl HasStateDict for BgeRerankerV2M3 {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.model.state_dict(prefix);
        sd.extend(self.classifier.state_dict(&prefixed(prefix, "classifier")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.model.load_state_dict(sd, prefix)?;
        self.classifier.load_state_dict(sd, &prefixed(prefix, "classifier"))?;
        Ok(())
    }
}

fn strip_roberta_prefix(sd: &StateDict) -> StateDict {
    sd.iter()
        .map(|(k, v)| {
            let stripped = k.strip_prefix("roberta.").unwrap_or(k);
            (stripped.to_string(), v.clone())
        })
        .collect()
}
