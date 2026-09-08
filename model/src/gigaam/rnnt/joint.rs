//! RNN-T joint network: encoder + predictor projections combined into per-step
//! log-probabilities.

use svod_dtype::DType;
use svod_tensor::Tensor;

use svod_tensor::nn::Module;

use crate::init::fan_in_uniform;

use crate::gigaam::Result;

/// RNN-T joint: `log_softmax(out_w · ReLU(enc_w · enc_t + enc_b + pred_w · g + pred_b) + out_b)`.
///
/// All Linear weights stored PyTorch-style `[out_features, in_features]` so
/// they plug straight into the `linear()` builder (which transposes
/// internally).
#[derive(Clone, Module)]
pub struct RnntJoint {
    pub enc_w: Tensor,
    pub enc_b: Tensor,
    pub pred_w: Tensor,
    pub pred_b: Tensor,
    pub out_w: Tensor,
    pub out_b: Tensor,
}

impl RnntJoint {
    pub fn empty(enc_hidden: usize, pred_hidden: usize, joint_hidden: usize, num_classes: usize) -> Self {
        Self {
            enc_w: fan_in_uniform(&[joint_hidden, enc_hidden], enc_hidden, DType::Float32),
            enc_b: fan_in_uniform(&[joint_hidden], enc_hidden, DType::Float32),
            pred_w: fan_in_uniform(&[joint_hidden, pred_hidden], pred_hidden, DType::Float32),
            pred_b: fan_in_uniform(&[joint_hidden], pred_hidden, DType::Float32),
            out_w: fan_in_uniform(&[num_classes, joint_hidden], joint_hidden, DType::Float32),
            out_b: fan_in_uniform(&[num_classes], joint_hidden, DType::Float32),
        }
    }

    /// `enc_t [1, 1, enc_hidden]`, `g [1, 1, pred_hidden]` → raw logits
    /// `[1, 1, num_classes]` (pre-softmax).
    fn logits(&self, enc_t: &Tensor, g: &Tensor) -> Result<Tensor> {
        let enc_proj = enc_t.linear().weight(&self.enc_w).bias(&self.enc_b).call()?;
        let pred_proj = g.linear().weight(&self.pred_w).bias(&self.pred_b).call()?;
        let summed = enc_proj.try_add(&pred_proj)?;
        let activated = summed.relu()?;
        Ok(activated.linear().weight(&self.out_w).bias(&self.out_b).call()?)
    }

    /// `enc_t [1, 1, enc_hidden]`, `g [1, 1, pred_hidden]` → log-probs
    /// `[1, 1, num_classes]`.
    pub fn forward(&self, enc_t: &Tensor, g: &Tensor) -> Result<Tensor> {
        Ok(self.logits(enc_t, g)?.log_softmax(-1isize)?)
    }

    /// Greedy variant: the device-side argmax token index `[1, 1]` (int32)
    /// over the vocab. `log_softmax` is omitted — argmax is invariant under the
    /// monotonic log-softmax, so the chosen index is identical while the host
    /// reads back a single int instead of the full vocab logit vector.
    pub fn forward_argmax(&self, enc_t: &Tensor, g: &Tensor) -> Result<Tensor> {
        Ok(self.logits(enc_t, g)?.argmax(-1isize)?)
    }

    /// Encoder projection `enc_w · enc + enc_b` over a whole frame axis —
    /// hoisted out of the decode loop (`[B, T, E] → [B, T, J]`, one MFMA
    /// matmul per wave instead of a per-step row projection).
    pub fn project_encoder(&self, enc: &Tensor) -> Result<Tensor> {
        Ok(enc.linear().weight(&self.enc_w).bias(&self.enc_b).call()?)
    }

    /// Greedy argmax over PRE-PROJECTED encoder rows ([`Self::project_encoder`]).
    pub fn argmax_preproj(&self, enc_proj_t: &Tensor, g: &Tensor) -> Result<Tensor> {
        let pred_proj = g.linear().weight(&self.pred_w).bias(&self.pred_b).call()?;
        let activated = enc_proj_t.try_add(&pred_proj)?.relu()?;
        let logits = activated.linear().weight(&self.out_w).bias(&self.out_b).call()?;
        Ok(logits.argmax(-1isize)?)
    }
}
