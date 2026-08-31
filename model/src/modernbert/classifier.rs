//! [`ModernBertClassifier`] — sequence classification over the ModernBERT
//! backbone: `input_ids` + `attention_mask` → raw class logits `(B, num_labels)`.
//!
//! Implements `svod_arch::pipelines::text::Classify` so it drops straight into an
//! [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline). The model
//! owns the forward + fused classification head (via
//! [`ModernBertClassifierJit`]); the pipeline owns chunking and profile
//! assembly.
//!
//! The classification head mirrors HF's `ModernBertForSequenceClassification`:
//! pool (cls or mean) → `head.dense` → GELU → `head.norm` → `classifier` linear.
//! Fused into one JIT plan so the `(B, L, D)` activations stay on-device.

use snafu::{OptionExt, ResultExt};
use svod_arch::pipelines::text::{Classification, Classify, EncoderHead, Encoding, RunProfile};
use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::{BoundVariable, PrepareConfig, Tensor};

use crate::init::fan_in_uniform;
use crate::jit::InputSpec;
use crate::modernbert::config::{ClassifierPooling, ModernBertConfig};
use crate::modernbert::error::{MissingMaskSnafu, Result, StateSnafu, TensorSnafu};
use crate::modernbert::head_jit::{HeadError, JitSnafu, execute_head};
use crate::modernbert::masked_mean;
use crate::modernbert::model::ModernBert;
use crate::modernbert::normalization::LayerNormWeights;
use crate::state::{self, HasStateDict, StateDict, get_tensor};

// ─── head weights ──────────────────────────────────────────────────────────

/// Classification head weights: HF `head.dense` + `head.norm` + `classifier`.
/// `dense_bias` is `None` when `classifier_bias = false` (ModernBERT default),
/// gating `head.dense` only. The final `classifier` Linear's bias is **always
/// present** (HF uses `nn.Linear` with the PyTorch default `bias = True`),
/// independent of `classifier_bias`. Shared by the sequence-classification and
/// token-classification heads — both are HF's `ModernBertPredictionHead` +
/// `classifier`; the sequence head pools first, the token head does not.
#[derive(Clone)]
pub(crate) struct ClassifierHead {
    dense_weight: Tensor,
    dense_bias: Option<Tensor>,
    norm: LayerNormWeights,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
}

impl ClassifierHead {
    pub(crate) fn empty(config: &ModernBertConfig) -> Self {
        let d = config.hidden_size;
        let n = config.num_labels;
        let dt = config.dtype.clone();
        Self {
            dense_weight: fan_in_uniform(&[d, d], d, dt.clone()),
            dense_bias: config.classifier_bias.then(|| fan_in_uniform(&[d], d, dt.clone())),
            norm: LayerNormWeights::with_eps(d, config.layer_norm_eps, dt.clone()),
            classifier_weight: fan_in_uniform(&[n, d], d, dt.clone()),
            classifier_bias: fan_in_uniform(&[n], d, dt.clone()),
        }
    }

    /// Number of output labels (rows of the `classifier` weight).
    pub(crate) fn num_labels(&self) -> usize {
        self.classifier_weight.shape().expect("classifier weight shape")[0]
            .as_const()
            .expect("classifier weight row count must be concrete")
    }
}

impl HasStateDict for ClassifierHead {
    fn state_dict(&self, _prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert("head.dense.weight".to_string(), self.dense_weight.clone());
        if let Some(b) = &self.dense_bias {
            sd.insert("head.dense.bias".to_string(), b.clone());
        }
        sd.extend(self.norm.state_dict("head.norm"));
        sd.insert("classifier.weight".to_string(), self.classifier_weight.clone());
        sd.insert("classifier.bias".to_string(), self.classifier_bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, _prefix: &str) -> std::result::Result<(), state::Error> {
        self.dense_weight = get_tensor(sd, "head.dense.weight")?;
        self.dense_bias = sd.get("head.dense.bias").cloned();
        self.norm.load_state_dict(sd, "head.norm")?;
        self.classifier_weight = get_tensor(sd, "classifier.weight")?;
        // `classifier.bias` is always present in real HF checkpoints (PyTorch
        // `nn.Linear` default); tolerate its absence by keeping the empty-init.
        if let Some(b) = sd.get("classifier.bias") {
            self.classifier_bias = b.clone();
        }
        Ok(())
    }
}

// ─── composite model (backbone + head) ─────────────────────────────────────

/// Backbone + classification head — the model type wrapped by the JIT.
/// `forward_batch` fuses backbone → pool → dense → GELU → norm → classifier
/// into a single graph.
#[derive(Clone)]
pub(crate) struct ModernBertClassificationModel {
    pub(crate) backbone: ModernBert,
    head: ClassifierHead,
    pooling: ClassifierPooling,
}

impl ModernBertClassificationModel {
    /// Deterministic-init model for testing (mirrors `ModernBert::empty`).
    #[cfg(test)]
    pub(crate) fn empty(config: &ModernBertConfig) -> Self {
        Self {
            backbone: ModernBert::empty(config.clone()),
            head: ClassifierHead::empty(config),
            pooling: config.classifier_pooling,
        }
    }

    pub(crate) fn from_state_dict(sd: &StateDict, config: &ModernBertConfig) -> Result<Self> {
        let casted = crate::state::cast_all(sd, config.dtype.clone());

        let mut backbone = ModernBert::empty(config.clone());
        backbone.load_state_dict(&casted, "").context(StateSnafu)?;

        let mut head = ClassifierHead::empty(config);
        head.load_state_dict(&casted, "").context(StateSnafu)?;

        Ok(Self { backbone, head, pooling: config.classifier_pooling })
    }

    /// Fused forward: backbone → pool → head → classifier → logits `(B, num_labels)`.
    pub(crate) fn forward_batch(
        &self,
        input_ids: &Tensor,
        padding_mask: Option<&Tensor>,
        b: &BoundVariable,
    ) -> Result<Tensor> {
        let hidden = self.backbone.forward_batch(input_ids, padding_mask, b)?;
        let mask = padding_mask.context(MissingMaskSnafu { what: "classification" })?;
        classify_head(&hidden, mask, &self.head, self.pooling)
    }
}

// ─── classify_head IR builder ──────────────────────────────────────────────

/// Pool → `prediction_head_tail`. Pure IR builder — fused into the JIT plan by
/// the `build` closure in [`ModernBertClassifierJit`].
fn classify_head(hidden: &Tensor, mask: &Tensor, head: &ClassifierHead, pooling: ClassifierPooling) -> Result<Tensor> {
    let pooled = match pooling {
        ClassifierPooling::Cls => {
            let slice = hidden.try_shrink([None, Some((SInt::Const(0), SInt::Const(1))), None]).context(TensorSnafu)?;
            slice.try_squeeze(Some(1)).context(TensorSnafu)?
        }
        ClassifierPooling::Mean => masked_mean(hidden, mask)?,
    };
    prediction_head_tail(&pooled, head)
}

/// HF `ModernBertPredictionHead` + the `classifier` Linear: `dense → GELU →
/// LayerNorm → classifier → f32`. Shared by the sequence-classification head
/// (applied to the pooled `(B, D)` state) and the token-classification head
/// (applied to the full `(B, L, D)` state — the `Linear`s broadcast over the
/// leading axes). Returns logits in the model's `num_labels`.
pub(crate) fn prediction_head_tail(hidden: &Tensor, head: &ClassifierHead) -> Result<Tensor> {
    // head.dense → GELU → head.norm
    let dense = if let Some(b) = &head.dense_bias {
        hidden.linear().weight(&head.dense_weight).bias(b).call().context(TensorSnafu)?
    } else {
        hidden.linear().weight(&head.dense_weight).call().context(TensorSnafu)?
    };
    let activated = dense.gelu_exact().context(TensorSnafu)?;
    let normed = head.norm.apply(&activated)?;

    // classifier: Linear(hidden → num_labels), bias always present.
    let logits =
        normed.linear().weight(&head.classifier_weight).bias(&head.classifier_bias).call().context(TensorSnafu)?;

    logits.cast(DType::Float32).context(TensorSnafu)
}

// ─── runtime (owns JIT, impl EncoderHead + Classify) ───────────────────────────

/// Finished-classifier model. Build once (eager JIT prepare) and reuse across
/// calls. Implements [`EncoderHead`] (with [`Classify`] fixing the output kinds)
/// for drop-in use with
/// [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).
pub struct ModernBertClassifier {
    jit: crate::modernbert::classifier_jit::ModernBertClassifierJit,
    max_batch: usize,
    max_seq: usize,
    num_classes: usize,
}

impl ModernBertClassifier {
    /// Prepare the classifier JIT at `[max_batch, max_seq]`.
    pub(crate) fn new(
        model: ModernBertClassificationModel,
        max_batch: usize,
        max_seq: usize,
    ) -> std::result::Result<Self, HeadError> {
        let num_classes = model.head.num_labels();
        let mut jit = crate::modernbert::classifier_jit::ModernBertClassifierJit::new(model).with_b_bound(max_batch);
        let ids_spec = InputSpec::i64(&[max_batch, max_seq]);
        let mask_spec = InputSpec::i64(&[max_batch, max_seq]);
        jit.prepare_with_config(ids_spec, mask_spec, &PrepareConfig::from_env()).context(JitSnafu)?;
        Ok(Self { jit, max_batch, max_seq, num_classes })
    }
}

impl EncoderHead for ModernBertClassifier {
    type Output = Classification;
    type Error = HeadError;

    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.max_seq)
    }

    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> std::result::Result<(Vec<Classification>, Option<RunProfile>), HeadError> {
        let (b, flat, prof) = execute_head(&mut self.jit, batch, self.max_batch, self.max_seq, profile, "classify")?;
        let nc = self.num_classes;
        let classifications: Vec<Classification> =
            (0..b).map(|i| Classification { logits: flat[i * nc..i * nc + nc].to_vec() }).collect();
        Ok((classifications, prof))
    }
}

impl Classify for ModernBertClassifier {
    fn num_labels(&self) -> usize {
        self.num_classes
    }
}
