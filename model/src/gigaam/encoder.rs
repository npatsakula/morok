use morok_dtype::DType;
use morok_ir::SInt;
use morok_tensor::Tensor;
use snafu::ResultExt;

use crate::state::{HasStateDict, StateDict, get_tensor, prefixed};

use super::error::TensorSnafu;
use super::{ConvNormType, GigaAmConfig, SubsamplingMode};

fn zeros(shape: &[usize]) -> Tensor {
    Tensor::full(shape, 0.0, DType::Float32).unwrap()
}

fn ones(shape: &[usize]) -> Tensor {
    Tensor::full(shape, 1.0, DType::Float32).unwrap()
}

type Result<T> = super::Result<T>;

// ---------------------------------------------------------------------------
// LayerNormWeights
// ---------------------------------------------------------------------------

/// Affine layer normalization: `layernorm(x) * weight + bias`.
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl LayerNormWeights {
    pub fn empty(size: usize) -> Self {
        Self { weight: ones(&[size]), bias: zeros(&[size]), eps: 1e-5 }
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let normed = x.layernorm(-1, self.eps).context(TensorSnafu)?;
        normed.try_mul(&self.weight).context(TensorSnafu)?.try_add(&self.bias).context(TensorSnafu)
    }
}

impl HasStateDict for LayerNormWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        sd.insert(prefixed(prefix, "bias"), self.bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        self.bias = get_tensor(sd, &prefixed(prefix, "bias"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FeedForward
// ---------------------------------------------------------------------------

/// Conformer FFN: LayerNorm -> Linear(d->4d) -> SiLU -> Linear(4d->d).
///
/// Does NOT apply residual or 0.5 scaling — caller handles that.
pub struct FeedForward {
    pub norm: LayerNormWeights,
    pub linear1_weight: Tensor,
    pub linear1_bias: Tensor,
    pub linear2_weight: Tensor,
    pub linear2_bias: Tensor,
}

impl FeedForward {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let (d, d_ff) = (config.d_model, config.d_ff);
        Self {
            norm: LayerNormWeights::empty(d),
            linear1_weight: zeros(&[d_ff, d]),
            linear1_bias: zeros(&[d_ff]),
            linear2_weight: zeros(&[d, d_ff]),
            linear2_bias: zeros(&[d]),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.norm.apply(x)?;
        let y = y.linear().weight(&self.linear1_weight).bias(&self.linear1_bias).call().context(TensorSnafu)?;
        let y = y.silu().context(TensorSnafu)?;
        y.linear().weight(&self.linear2_weight).bias(&self.linear2_bias).call().context(TensorSnafu)
    }
}

impl HasStateDict for FeedForward {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.norm.state_dict(&prefixed(prefix, "norm"));
        sd.insert(prefixed(prefix, "linear1.weight"), self.linear1_weight.clone());
        sd.insert(prefixed(prefix, "linear1.bias"), self.linear1_bias.clone());
        sd.insert(prefixed(prefix, "linear2.weight"), self.linear2_weight.clone());
        sd.insert(prefixed(prefix, "linear2.bias"), self.linear2_bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.norm.load_state_dict(sd, &prefixed(prefix, "norm"))?;
        self.linear1_weight = get_tensor(sd, &prefixed(prefix, "linear1.weight"))?;
        self.linear1_bias = get_tensor(sd, &prefixed(prefix, "linear1.bias"))?;
        self.linear2_weight = get_tensor(sd, &prefixed(prefix, "linear2.weight"))?;
        self.linear2_bias = get_tensor(sd, &prefixed(prefix, "linear2.bias"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MultiHeadSelfAttention
// ---------------------------------------------------------------------------

/// Multi-head self-attention with rotary position embeddings.
pub struct MultiHeadSelfAttention {
    pub norm: LayerNormWeights,
    pub q_proj: Tensor,
    pub q_bias: Tensor,
    pub k_proj: Tensor,
    pub k_bias: Tensor,
    pub v_proj: Tensor,
    pub v_bias: Tensor,
    pub out_proj: Tensor,
    pub out_bias: Tensor,
    pub n_heads: usize,
}

impl MultiHeadSelfAttention {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let d = config.d_model;
        Self {
            norm: LayerNormWeights::empty(d),
            q_proj: zeros(&[d, d]),
            q_bias: zeros(&[d]),
            k_proj: zeros(&[d, d]),
            k_bias: zeros(&[d]),
            v_proj: zeros(&[d, d]),
            v_bias: zeros(&[d]),
            out_proj: zeros(&[d, d]),
            out_bias: zeros(&[d]),
            n_heads: config.n_heads,
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        _batch: SInt,
        _seq_len: SInt,
    ) -> Result<Tensor> {
        let shape = x.shape().context(TensorSnafu)?;
        let b = shape[0].clone();
        let t = shape[1].clone();
        let d_model = self.norm.weight.shape().context(TensorSnafu)?[0].as_const().unwrap();
        let d_k = d_model / self.n_heads;
        let h = self.n_heads;

        let y = self.norm.apply(x)?;

        // Reshape to [T, B, H, d_k] for RoPE (matches PyTorch ordering)
        let y_heads = y
            .try_transpose(0, 1)
            .context(TensorSnafu)?
            .try_reshape([t.clone(), b.clone(), SInt::Const(h), SInt::Const(d_k)])
            .context(TensorSnafu)?;

        // Apply RoPE BEFORE linear projections
        let q_rot = y_heads.apply_rotary_emb(cos, sin, false).context(TensorSnafu)?;
        let k_rot = y_heads.apply_rotary_emb(cos, sin, false).context(TensorSnafu)?;

        // Reshape back to [B, T, d_model]
        let q_input = q_rot
            .try_reshape([t.clone(), b.clone(), SInt::Const(d_model)])
            .context(TensorSnafu)?
            .try_transpose(0, 1)
            .context(TensorSnafu)?;
        let k_input = k_rot
            .try_reshape([t.clone(), b.clone(), SInt::Const(d_model)])
            .context(TensorSnafu)?
            .try_transpose(0, 1)
            .context(TensorSnafu)?;

        // Project through Q, K, V linear layers
        let q = q_input.linear().weight(&self.q_proj).bias(&self.q_bias).call().context(TensorSnafu)?;
        let k = k_input.linear().weight(&self.k_proj).bias(&self.k_bias).call().context(TensorSnafu)?;
        let v = y.linear().weight(&self.v_proj).bias(&self.v_bias).call().context(TensorSnafu)?;

        // Reshape to [B, H, T, d_k]
        let q = q
            .try_reshape([b.clone(), t.clone(), SInt::Const(h), SInt::Const(d_k)])
            .context(TensorSnafu)?
            .try_transpose(1, 2)
            .context(TensorSnafu)?;
        let k = k
            .try_reshape([b.clone(), t.clone(), SInt::Const(h), SInt::Const(d_k)])
            .context(TensorSnafu)?
            .try_transpose(1, 2)
            .context(TensorSnafu)?;
        let v = v
            .try_reshape([b.clone(), t.clone(), SInt::Const(h), SInt::Const(d_k)])
            .context(TensorSnafu)?
            .try_transpose(1, 2)
            .context(TensorSnafu)?;

        // Scaled dot-product attention
        let attn =
            q.scaled_dot_product_attention().key(&k).value(&v).maybe_attn_mask(mask).call().context(TensorSnafu)?;

        // [B, H, T, d_k] -> [B, T, H, d_k] -> [B, T, d_model]
        let out = attn
            .try_transpose(1, 2)
            .context(TensorSnafu)?
            .try_reshape([b, t, SInt::Const(d_model)])
            .context(TensorSnafu)?;

        out.linear().weight(&self.out_proj).bias(&self.out_bias).call().context(TensorSnafu)
    }
}

impl HasStateDict for MultiHeadSelfAttention {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.norm.state_dict(&prefixed(prefix, "norm"));
        for (name, t) in [
            ("q_proj", &self.q_proj),
            ("q_bias", &self.q_bias),
            ("k_proj", &self.k_proj),
            ("k_bias", &self.k_bias),
            ("v_proj", &self.v_proj),
            ("v_bias", &self.v_bias),
            ("out_proj", &self.out_proj),
            ("out_bias", &self.out_bias),
        ] {
            sd.insert(prefixed(prefix, name), t.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.norm.load_state_dict(sd, &prefixed(prefix, "norm"))?;
        self.q_proj = get_tensor(sd, &prefixed(prefix, "q_proj"))?;
        self.q_bias = get_tensor(sd, &prefixed(prefix, "q_bias"))?;
        self.k_proj = get_tensor(sd, &prefixed(prefix, "k_proj"))?;
        self.k_bias = get_tensor(sd, &prefixed(prefix, "k_bias"))?;
        self.v_proj = get_tensor(sd, &prefixed(prefix, "v_proj"))?;
        self.v_bias = get_tensor(sd, &prefixed(prefix, "v_bias"))?;
        self.out_proj = get_tensor(sd, &prefixed(prefix, "out_proj"))?;
        self.out_bias = get_tensor(sd, &prefixed(prefix, "out_bias"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConvModule
// ---------------------------------------------------------------------------

pub enum ConvNorm {
    LayerNorm(LayerNormWeights),
    BatchNorm { scale: Tensor, bias: Tensor, mean: Tensor, invstd: Tensor },
}

/// Conformer convolution module:
/// LayerNorm -> Conv1d(d,2d,k=1) -> GLU -> DepthwiseConv1d -> Norm -> SiLU -> Conv1d(d,d,k=1)
pub struct ConvModule {
    pub norm: LayerNormWeights,
    pub pw1_weight: Tensor,
    pub pw1_bias: Tensor,
    pub dw_weight: Tensor,
    pub dw_bias: Tensor,
    pub conv_norm: ConvNorm,
    pub pw2_weight: Tensor,
    pub pw2_bias: Tensor,
    d_model: usize,
    conv_kernel: usize,
}

impl ConvModule {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let (d, k) = (config.d_model, config.conv_kernel);
        let conv_norm = match &config.conv_norm_type {
            ConvNormType::LayerNorm => ConvNorm::LayerNorm(LayerNormWeights::empty(d)),
            ConvNormType::BatchNorm => {
                ConvNorm::BatchNorm { scale: ones(&[d]), bias: zeros(&[d]), mean: zeros(&[d]), invstd: ones(&[d]) }
            }
        };
        Self {
            norm: LayerNormWeights::empty(d),
            pw1_weight: zeros(&[2 * d, d, 1]),
            pw1_bias: zeros(&[2 * d]),
            dw_weight: zeros(&[d, 1, k]),
            dw_bias: zeros(&[d]),
            conv_norm,
            pw2_weight: zeros(&[d, d, 1]),
            pw2_bias: zeros(&[d]),
            d_model: d,
            conv_kernel: k,
        }
    }

    pub fn forward(&self, x: &Tensor, pad_mask: Option<&Tensor>) -> Result<Tensor> {
        let y = self.norm.apply(x)?;

        let y = y.try_transpose(-1, -2).context(TensorSnafu)?.contiguous();

        let y = y.conv2d().weight(&self.pw1_weight).bias(&self.pw1_bias).call().context(TensorSnafu)?;

        let mut y = y.glu(1).context(TensorSnafu)?;

        if let Some(mask) = pad_mask {
            let valid = mask.logical_not().context(TensorSnafu)?;
            let valid = valid.try_unsqueeze(1).context(TensorSnafu)?;
            let zeros = y.zero().context(TensorSnafu)?;
            y = y.where_(&valid, &zeros).context(TensorSnafu)?;
        }

        let pad = ((self.conv_kernel - 1) / 2) as isize;
        let y = y
            .conv2d()
            .weight(&self.dw_weight)
            .bias(&self.dw_bias)
            .groups(self.d_model)
            .padding(&[(pad, pad)])
            .call()
            .context(TensorSnafu)?;

        let y = match &self.conv_norm {
            ConvNorm::LayerNorm(ln) => {
                let y = y.try_transpose(-1, -2).context(TensorSnafu)?;
                let y = ln.apply(&y)?;
                y.try_transpose(-1, -2).context(TensorSnafu)?
            }
            ConvNorm::BatchNorm { scale, bias, mean, invstd } => {
                y.batchnorm().scale(scale).bias(bias).mean(mean).invstd(invstd).call().context(TensorSnafu)?
            }
        };

        let y = y.silu().context(TensorSnafu)?;

        let y = y.conv2d().weight(&self.pw2_weight).bias(&self.pw2_bias).call().context(TensorSnafu)?;

        y.try_transpose(-1, -2).context(TensorSnafu)
    }
}

impl HasStateDict for ConvModule {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.norm.state_dict(&prefixed(prefix, "norm"));
        for (name, t) in [
            ("pw1_weight", &self.pw1_weight),
            ("pw1_bias", &self.pw1_bias),
            ("dw_weight", &self.dw_weight),
            ("dw_bias", &self.dw_bias),
            ("pw2_weight", &self.pw2_weight),
            ("pw2_bias", &self.pw2_bias),
        ] {
            sd.insert(prefixed(prefix, name), t.clone());
        }
        match &self.conv_norm {
            ConvNorm::LayerNorm(ln) => sd.extend(ln.state_dict(&prefixed(prefix, "conv_norm"))),
            ConvNorm::BatchNorm { scale, bias, mean, invstd } => {
                for (name, t) in [("bn_scale", scale), ("bn_bias", bias), ("bn_mean", mean), ("bn_invstd", invstd)] {
                    sd.insert(prefixed(prefix, name), t.clone());
                }
            }
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.norm.load_state_dict(sd, &prefixed(prefix, "norm"))?;
        self.pw1_weight = get_tensor(sd, &prefixed(prefix, "pw1_weight"))?;
        self.pw1_bias = get_tensor(sd, &prefixed(prefix, "pw1_bias"))?;
        self.dw_weight = get_tensor(sd, &prefixed(prefix, "dw_weight"))?;
        self.dw_bias = get_tensor(sd, &prefixed(prefix, "dw_bias"))?;
        self.pw2_weight = get_tensor(sd, &prefixed(prefix, "pw2_weight"))?;
        self.pw2_bias = get_tensor(sd, &prefixed(prefix, "pw2_bias"))?;
        match &mut self.conv_norm {
            ConvNorm::LayerNorm(ln) => ln.load_state_dict(sd, &prefixed(prefix, "conv_norm"))?,
            ConvNorm::BatchNorm { scale, bias, mean, invstd } => {
                *scale = get_tensor(sd, &prefixed(prefix, "bn_scale"))?;
                *bias = get_tensor(sd, &prefixed(prefix, "bn_bias"))?;
                *mean = get_tensor(sd, &prefixed(prefix, "bn_mean"))?;
                *invstd = get_tensor(sd, &prefixed(prefix, "bn_invstd"))?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// StridingSubsampling
// ---------------------------------------------------------------------------

/// Striding subsampling: 2x (Conv stride-2 + ReLU), optionally followed by Linear.
///
/// Supports two modes:
/// - **conv1d**: `Conv1d(n_mels→d, k, stride=2)` x2, no linear projection.
/// - **conv2d**: `Conv2d(1→d, 3x3, stride=2)` x2 + `Linear(d * n_mels/4, d)`.
///
/// Input: `[B, T, n_mels]` -> Output: `[B, T/4, d_model]`.
pub struct StridingSubsampling {
    pub conv1_weight: Tensor,
    pub conv1_bias: Tensor,
    pub conv2_weight: Tensor,
    pub conv2_bias: Tensor,
    pub linear_weight: Option<Tensor>,
    pub linear_bias: Option<Tensor>,
    n_mels: usize,
    d_model: usize,
    mode: SubsamplingMode,
    kernel_size: usize,
}

impl StridingSubsampling {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let d = config.d_model;
        let k = config.subs_kernel_size;
        match &config.subsampling_mode {
            SubsamplingMode::Conv1d => Self {
                conv1_weight: zeros(&[d, config.n_mels, k]),
                conv1_bias: zeros(&[d]),
                conv2_weight: zeros(&[d, d, k]),
                conv2_bias: zeros(&[d]),
                linear_weight: None,
                linear_bias: None,
                n_mels: config.n_mels,
                d_model: d,
                mode: SubsamplingMode::Conv1d,
                kernel_size: k,
            },
            SubsamplingMode::Conv2d => Self {
                conv1_weight: zeros(&[d, 1, 3, 3]),
                conv1_bias: zeros(&[d]),
                conv2_weight: zeros(&[d, d, 3, 3]),
                conv2_bias: zeros(&[d]),
                linear_weight: Some(zeros(&[d, d * (config.n_mels / 4)])),
                linear_bias: Some(zeros(&[d])),
                n_mels: config.n_mels,
                d_model: d,
                mode: SubsamplingMode::Conv2d,
                kernel_size: 3,
            },
        }
    }

    pub fn output_length(&self, input_length: usize) -> usize {
        let pad = (self.kernel_size - 1) / 2;
        let mut len = input_length;
        for _ in 0..2 {
            len = (len + 2 * pad - self.kernel_size) / 2 + 1;
        }
        len
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.mode {
            SubsamplingMode::Conv1d => self.forward_conv1d(x),
            SubsamplingMode::Conv2d => self.forward_conv2d(x),
        }
    }

    fn forward_conv1d(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.try_transpose(-1, -2).context(TensorSnafu)?;

        let pad = (self.kernel_size / 2) as isize;
        let x = x
            .conv2d()
            .weight(&self.conv1_weight)
            .bias(&self.conv1_bias)
            .stride(&[2])
            .padding(&[(pad, pad)])
            .call()
            .context(TensorSnafu)?;
        let x = x.relu().context(TensorSnafu)?;

        let x = x
            .conv2d()
            .weight(&self.conv2_weight)
            .bias(&self.conv2_bias)
            .stride(&[2])
            .padding(&[(pad, pad)])
            .call()
            .context(TensorSnafu)?;
        let x = x.relu().context(TensorSnafu)?;

        x.try_transpose(-1, -2).context(TensorSnafu)
    }

    fn forward_conv2d(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape().context(TensorSnafu)?;
        let b = shape[0].clone();

        let x = x.try_unsqueeze(1).context(TensorSnafu)?;

        let x = x
            .conv2d()
            .weight(&self.conv1_weight)
            .bias(&self.conv1_bias)
            .stride(&[2, 2])
            .padding(&[(1, 1), (1, 1)])
            .call()
            .context(TensorSnafu)?;
        let x = x.relu().context(TensorSnafu)?;

        let x = x
            .conv2d()
            .weight(&self.conv2_weight)
            .bias(&self.conv2_bias)
            .stride(&[2, 2])
            .padding(&[(1, 1), (1, 1)])
            .call()
            .context(TensorSnafu)?;
        let x = x.relu().context(TensorSnafu)?;

        let x = x.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)?;
        let x = x.try_reshape([b, SInt::Infer, SInt::Const(self.d_model * self.n_mels / 4)]).context(TensorSnafu)?;

        let lw = self.linear_weight.as_ref().expect("conv2d mode requires linear_weight");
        let lb = self.linear_bias.as_ref().expect("conv2d mode requires linear_bias");
        x.linear().weight(lw).bias(lb).call().context(TensorSnafu)
    }
}

impl HasStateDict for StridingSubsampling {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        for (name, t) in [
            ("conv1_weight", &self.conv1_weight),
            ("conv1_bias", &self.conv1_bias),
            ("conv2_weight", &self.conv2_weight),
            ("conv2_bias", &self.conv2_bias),
        ] {
            sd.insert(prefixed(prefix, name), t.clone());
        }
        if let (Some(lw), Some(lb)) = (&self.linear_weight, &self.linear_bias) {
            sd.insert(prefixed(prefix, "linear_weight"), lw.clone());
            sd.insert(prefixed(prefix, "linear_bias"), lb.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.conv1_weight = get_tensor(sd, &prefixed(prefix, "conv1_weight"))?;
        self.conv1_bias = get_tensor(sd, &prefixed(prefix, "conv1_bias"))?;
        self.conv2_weight = get_tensor(sd, &prefixed(prefix, "conv2_weight"))?;
        self.conv2_bias = get_tensor(sd, &prefixed(prefix, "conv2_bias"))?;
        if matches!(self.mode, SubsamplingMode::Conv2d) {
            self.linear_weight = Some(get_tensor(sd, &prefixed(prefix, "linear_weight"))?);
            self.linear_bias = Some(get_tensor(sd, &prefixed(prefix, "linear_bias"))?);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConformerLayer
// ---------------------------------------------------------------------------

/// One Conformer layer (Macaron-style):
/// FFN1(x0.5) + MHSA + Conv + FFN2(x0.5) + LayerNorm
pub struct ConformerLayer {
    pub ffn1: FeedForward,
    pub mhsa: MultiHeadSelfAttention,
    pub conv: ConvModule,
    pub ffn2: FeedForward,
    pub final_norm: LayerNormWeights,
}

impl ConformerLayer {
    pub fn empty(config: &GigaAmConfig) -> Self {
        Self {
            ffn1: FeedForward::empty(config),
            mhsa: MultiHeadSelfAttention::empty(config),
            conv: ConvModule::empty(config),
            ffn2: FeedForward::empty(config),
            final_norm: LayerNormWeights::empty(config.d_model),
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        att_mask: Option<&Tensor>,
        pad_mask: Option<&Tensor>,
        batch: SInt,
        seq_len: SInt,
    ) -> Result<Tensor> {
        let half = Tensor::from_const(0.5f64).cast(x.uop().dtype()).context(TensorSnafu)?;

        // FFN1 half-step
        let x = x.try_add(&self.ffn1.forward(x)?.try_mul(&half).context(TensorSnafu)?).context(TensorSnafu)?;

        // MHSA
        let x = x
            .try_add(&self.mhsa.forward(&x, cos, sin, att_mask, batch.clone(), seq_len.clone())?)
            .context(TensorSnafu)?;

        // Convolution
        let x = x.try_add(&self.conv.forward(&x, pad_mask)?).context(TensorSnafu)?;

        // FFN2 half-step
        let x = x.try_add(&self.ffn2.forward(&x)?.try_mul(&half).context(TensorSnafu)?).context(TensorSnafu)?;

        // Final layer norm
        self.final_norm.apply(&x)
    }
}

impl HasStateDict for ConformerLayer {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.ffn1.state_dict(&prefixed(prefix, "ffn1")));
        sd.extend(self.mhsa.state_dict(&prefixed(prefix, "mhsa")));
        sd.extend(self.conv.state_dict(&prefixed(prefix, "conv")));
        sd.extend(self.ffn2.state_dict(&prefixed(prefix, "ffn2")));
        sd.extend(self.final_norm.state_dict(&prefixed(prefix, "final_norm")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.ffn1.load_state_dict(sd, &prefixed(prefix, "ffn1"))?;
        self.mhsa.load_state_dict(sd, &prefixed(prefix, "mhsa"))?;
        self.conv.load_state_dict(sd, &prefixed(prefix, "conv"))?;
        self.ffn2.load_state_dict(sd, &prefixed(prefix, "ffn2"))?;
        self.final_norm.load_state_dict(sd, &prefixed(prefix, "final_norm"))?;
        Ok(())
    }
}
