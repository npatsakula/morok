//! Conformer encoder used as DiariZen's segmentation head.
//!
//! Direct port of `submodules/DiariZen/diarizen/models/module/conformer.py:8-325`.
//! With `use_posi=False` (the default for `s80-md-v2`) the MHSA reduces to
//! standard scaled-dot-product attention; no relative-position embeddings.
//! Each block applies (in order, all with internal residuals):
//!
//! ```text
//! ffn1(x)  = x + 0.5 · Dropout(W2 · Swish(W1 · LN(x)))
//! mha(x)   = x + Dropout(MHSA(LN(x)))                          // no rel-pos
//! conv(x)  = x + transpose(pw2(Dropout(Swish(BN(DepthwiseConv(GLU(pw1(LN(x).T))))))))
//! ffn2(x)  = x + 0.5 · Dropout(W2 · Swish(W1 · LN(x)))
//! out      = LN(ffn2(conv(mha(ffn1(x)))))
//! ```
//!
//! State-dict keys follow the upstream Python module names:
//! `ffn1.{ln_norm, w_1, w_2}`, `mha.{ln_norm, mha.{linearQ, linearK, linearV, linearO}}`,
//! `conv.{ln_norm, pointwise_conv1, depthwise_conv, bn_norm, pointwise_conv2}`,
//! `ffn2.{ln_norm, w_1, w_2}`, and a final `ln_norm` on the block.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::blocks::BatchNormWeights;
use crate::init::{fan_in_uniform, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};
use crate::wavlm::LayerNormWeights;

use super::error::Result;

// ---------------------------------------------------------------------------
// PositionwiseFeedForward
// ---------------------------------------------------------------------------

/// Pre-norm FFN with 0.5-scaled residual — Conformer convention.
#[derive(Clone)]
pub struct PositionwiseFeedForward {
    pub ln_norm: LayerNormWeights,
    pub w1_weight: Tensor,
    pub w1_bias: Tensor,
    pub w2_weight: Tensor,
    pub w2_bias: Tensor,
}

impl PositionwiseFeedForward {
    pub fn empty(in_size: usize, ffn_hidden: usize) -> Self {
        Self {
            ln_norm: LayerNormWeights::empty(in_size),
            w1_weight: fan_in_uniform(&[ffn_hidden, in_size], in_size, DType::Float32),
            w1_bias: zeros(&[ffn_hidden], DType::Float32),
            w2_weight: fan_in_uniform(&[in_size, ffn_hidden], ffn_hidden, DType::Float32),
            w2_bias: zeros(&[in_size], DType::Float32),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.ln_norm.apply(x)?;
        let y = normed.linear().weight(&self.w1_weight).bias(&self.w1_bias).call()?;
        let y = y.silu()?; // Python `Swish` == SiLU.
        let y = y.linear().weight(&self.w2_weight).bias(&self.w2_bias).call()?;
        let half = Tensor::const_(0.5f32, y.dtype());
        let scaled = y.try_mul(&half)?;
        Ok(x.try_add(&scaled)?)
    }
}

impl HasStateDict for PositionwiseFeedForward {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.ln_norm.state_dict(&prefixed(prefix, "ln_norm"));
        sd.insert(prefixed(prefix, "w_1.weight"), self.w1_weight.clone());
        sd.insert(prefixed(prefix, "w_1.bias"), self.w1_bias.clone());
        sd.insert(prefixed(prefix, "w_2.weight"), self.w2_weight.clone());
        sd.insert(prefixed(prefix, "w_2.bias"), self.w2_bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.ln_norm.load_state_dict(sd, &prefixed(prefix, "ln_norm"))?;
        self.w1_weight = get_tensor(sd, &prefixed(prefix, "w_1.weight"))?;
        self.w1_bias = get_tensor(sd, &prefixed(prefix, "w_1.bias"))?;
        self.w2_weight = get_tensor(sd, &prefixed(prefix, "w_2.weight"))?;
        self.w2_bias = get_tensor(sd, &prefixed(prefix, "w_2.bias"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Plain MultiHeadSelfAttention (no rel-pos; use_posi=False)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct PlainMultiHeadSelfAttention {
    pub n_heads: usize,
    pub d_k: usize,
    pub n_units: usize,
    pub linear_q_weight: Tensor,
    pub linear_q_bias: Tensor,
    pub linear_k_weight: Tensor,
    pub linear_k_bias: Tensor,
    pub linear_v_weight: Tensor,
    pub linear_v_bias: Tensor,
    pub linear_o_weight: Tensor,
    pub linear_o_bias: Tensor,
}

impl PlainMultiHeadSelfAttention {
    pub fn empty(n_units: usize, n_heads: usize) -> Self {
        assert!(n_units.is_multiple_of(n_heads), "n_units must be divisible by n_heads");
        let d_k = n_units / n_heads;
        Self {
            n_heads,
            d_k,
            n_units,
            linear_q_weight: fan_in_uniform(&[n_units, n_units], n_units, DType::Float32),
            linear_q_bias: zeros(&[n_units], DType::Float32),
            linear_k_weight: fan_in_uniform(&[n_units, n_units], n_units, DType::Float32),
            linear_k_bias: zeros(&[n_units], DType::Float32),
            linear_v_weight: fan_in_uniform(&[n_units, n_units], n_units, DType::Float32),
            linear_v_bias: zeros(&[n_units], DType::Float32),
            linear_o_weight: fan_in_uniform(&[n_units, n_units], n_units, DType::Float32),
            linear_o_bias: zeros(&[n_units], DType::Float32),
        }
    }

    /// Forward on `(B, L, n_units)` → `(B, L, n_units)`. Plain scaled-dot-product.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.dim(0)?;
        let l = x.dim_const(1)?;

        let project_and_split = |w: &Tensor, bias: &Tensor| -> Result<Tensor> {
            let y = x.linear().weight(w).bias(bias).call()?;
            Ok(y.try_reshape(&[b.clone(), l.into(), self.n_heads.into(), self.d_k.into()])?
                .try_permute(&[0, 2, 1, 3])?)
        };

        let q = project_and_split(&self.linear_q_weight, &self.linear_q_bias)?; // (B, h, L, d_k)
        let k = project_and_split(&self.linear_k_weight, &self.linear_k_bias)?;
        let v = project_and_split(&self.linear_v_weight, &self.linear_v_bias)?;

        let scaling = (self.d_k as f32).powf(-0.5);
        let scaling_t = Tensor::full(&[1], scaling, DType::Float32)?;
        let q_scaled = q.try_mul(&scaling_t)?;
        let k_t = k.try_transpose(-2, -1)?; // (B, h, d_k, L)
        let scores = q_scaled.matmul(&k_t)?; // (B, h, L, L)
        let weights = scores.softmax(-1)?;
        let attn_out = weights.matmul(&v)?; // (B, h, L, d_k)

        let attn_out = attn_out.try_permute(&[0, 2, 1, 3])?.try_reshape(&[b, l.into(), self.n_units.into()])?;

        Ok(attn_out.linear().weight(&self.linear_o_weight).bias(&self.linear_o_bias).call()?)
    }
}

impl HasStateDict for PlainMultiHeadSelfAttention {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "linearQ.weight"), self.linear_q_weight.clone());
        sd.insert(prefixed(prefix, "linearQ.bias"), self.linear_q_bias.clone());
        sd.insert(prefixed(prefix, "linearK.weight"), self.linear_k_weight.clone());
        sd.insert(prefixed(prefix, "linearK.bias"), self.linear_k_bias.clone());
        sd.insert(prefixed(prefix, "linearV.weight"), self.linear_v_weight.clone());
        sd.insert(prefixed(prefix, "linearV.bias"), self.linear_v_bias.clone());
        sd.insert(prefixed(prefix, "linearO.weight"), self.linear_o_weight.clone());
        sd.insert(prefixed(prefix, "linearO.bias"), self.linear_o_bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.linear_q_weight = get_tensor(sd, &prefixed(prefix, "linearQ.weight"))?;
        self.linear_q_bias = get_tensor(sd, &prefixed(prefix, "linearQ.bias"))?;
        self.linear_k_weight = get_tensor(sd, &prefixed(prefix, "linearK.weight"))?;
        self.linear_k_bias = get_tensor(sd, &prefixed(prefix, "linearK.bias"))?;
        self.linear_v_weight = get_tensor(sd, &prefixed(prefix, "linearV.weight"))?;
        self.linear_v_bias = get_tensor(sd, &prefixed(prefix, "linearV.bias"))?;
        self.linear_o_weight = get_tensor(sd, &prefixed(prefix, "linearO.weight"))?;
        self.linear_o_bias = get_tensor(sd, &prefixed(prefix, "linearO.bias"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConformerMHA — pre-norm wrap + residual + dropout
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ConformerMHA {
    pub ln_norm: LayerNormWeights,
    pub mha: PlainMultiHeadSelfAttention,
}

impl ConformerMHA {
    pub fn empty(in_size: usize, num_head: usize) -> Self {
        Self { ln_norm: LayerNormWeights::empty(in_size), mha: PlainMultiHeadSelfAttention::empty(in_size, num_head) }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.ln_norm.apply(x)?;
        let delta = self.mha.forward(&normed)?;
        Ok(x.try_add(&delta)?)
    }
}

impl HasStateDict for ConformerMHA {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.ln_norm.state_dict(&prefixed(prefix, "ln_norm"));
        sd.extend(self.mha.state_dict(&prefixed(prefix, "mha")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.ln_norm.load_state_dict(sd, &prefixed(prefix, "ln_norm"))?;
        self.mha.load_state_dict(sd, &prefixed(prefix, "mha"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConvolutionModule
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ConvolutionModule {
    pub channels: usize,
    pub kernel_size: usize,
    pub ln_norm: LayerNormWeights,
    /// `(2C, C, 1)` 1×1 conv to 2× channels (split into GLU halves).
    pub pointwise1_weight: Tensor,
    pub pointwise1_bias: Tensor,
    /// `(C, 1, k)` depthwise conv (groups = channels).
    pub depthwise_weight: Tensor,
    pub depthwise_bias: Tensor,
    pub bn_norm: BatchNormWeights,
    /// `(C, C, 1)` 1×1 conv back to channel count.
    pub pointwise2_weight: Tensor,
    pub pointwise2_bias: Tensor,
}

impl ConvolutionModule {
    pub fn empty(channels: usize, kernel_size: usize) -> Self {
        assert!(!kernel_size.is_multiple_of(2), "kernel_size must be odd for SAME padding");
        Self {
            channels,
            kernel_size,
            ln_norm: LayerNormWeights::empty(channels),
            pointwise1_weight: fan_in_uniform(&[2 * channels, channels, 1], channels, DType::Float32),
            pointwise1_bias: zeros(&[2 * channels], DType::Float32),
            depthwise_weight: fan_in_uniform(&[channels, 1, kernel_size], kernel_size, DType::Float32),
            depthwise_bias: zeros(&[channels], DType::Float32),
            bn_norm: BatchNormWeights::empty(channels),
            pointwise2_weight: fan_in_uniform(&[channels, channels, 1], channels, DType::Float32),
            pointwise2_bias: zeros(&[channels], DType::Float32),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, T, C)
        let normed = self.ln_norm.apply(x)?;
        // → (B, C, T)
        let y = normed.try_permute(&[0, 2, 1])?;

        // Pointwise conv1: (B, C, T) → (B, 2C, T)
        let y = y
            .conv2d()
            .weight(&self.pointwise1_weight)
            .bias(&self.pointwise1_bias)
            .stride(&[1])
            .padding(&[(0, 0)])
            .call()?;

        // GLU over channel dim (dim 1).
        let y = y.glu(1)?; // (B, C, T)

        // Depthwise conv with groups=channels and SAME padding.
        let p = ((self.kernel_size - 1) / 2) as isize;
        let y = y
            .conv2d()
            .weight(&self.depthwise_weight)
            .bias(&self.depthwise_bias)
            .groups(self.channels)
            .stride(&[1])
            .padding(&[(p, p)])
            .call()?;

        // BN over channel axis (axis 1 default works for NCT).
        let y = self.bn_norm.forward(&y)?;
        let y = y.silu()?;

        // Pointwise conv2: (B, C, T) → (B, C, T)
        let y = y
            .conv2d()
            .weight(&self.pointwise2_weight)
            .bias(&self.pointwise2_bias)
            .stride(&[1])
            .padding(&[(0, 0)])
            .call()?;

        // Back to (B, T, C) and residual.
        let y = y.try_permute(&[0, 2, 1])?;
        Ok(x.try_add(&y)?)
    }
}

impl HasStateDict for ConvolutionModule {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.ln_norm.state_dict(&prefixed(prefix, "ln_norm"));
        sd.insert(prefixed(prefix, "pointwise_conv1.weight"), self.pointwise1_weight.clone());
        sd.insert(prefixed(prefix, "pointwise_conv1.bias"), self.pointwise1_bias.clone());
        sd.insert(prefixed(prefix, "depthwise_conv.weight"), self.depthwise_weight.clone());
        sd.insert(prefixed(prefix, "depthwise_conv.bias"), self.depthwise_bias.clone());
        sd.extend(self.bn_norm.state_dict(&prefixed(prefix, "bn_norm")));
        sd.insert(prefixed(prefix, "pointwise_conv2.weight"), self.pointwise2_weight.clone());
        sd.insert(prefixed(prefix, "pointwise_conv2.bias"), self.pointwise2_bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.ln_norm.load_state_dict(sd, &prefixed(prefix, "ln_norm"))?;
        self.pointwise1_weight = get_tensor(sd, &prefixed(prefix, "pointwise_conv1.weight"))?;
        self.pointwise1_bias = get_tensor(sd, &prefixed(prefix, "pointwise_conv1.bias"))?;
        self.depthwise_weight = get_tensor(sd, &prefixed(prefix, "depthwise_conv.weight"))?;
        self.depthwise_bias = get_tensor(sd, &prefixed(prefix, "depthwise_conv.bias"))?;
        self.bn_norm.load_state_dict(sd, &prefixed(prefix, "bn_norm"))?;
        self.pointwise2_weight = get_tensor(sd, &prefixed(prefix, "pointwise_conv2.weight"))?;
        self.pointwise2_bias = get_tensor(sd, &prefixed(prefix, "pointwise_conv2.bias"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConformerBlock and ConformerEncoder
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ConformerBlock {
    pub ffn1: PositionwiseFeedForward,
    pub mha: ConformerMHA,
    pub conv: ConvolutionModule,
    pub ffn2: PositionwiseFeedForward,
    pub ln_norm: LayerNormWeights,
}

impl ConformerBlock {
    pub fn empty(in_size: usize, ffn_hidden: usize, num_head: usize, kernel_size: usize) -> Self {
        Self {
            ffn1: PositionwiseFeedForward::empty(in_size, ffn_hidden),
            mha: ConformerMHA::empty(in_size, num_head),
            conv: ConvolutionModule::empty(in_size, kernel_size),
            ffn2: PositionwiseFeedForward::empty(in_size, ffn_hidden),
            ln_norm: LayerNormWeights::empty(in_size),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ffn1.forward(x)?;
        let x = self.mha.forward(&x)?;
        let x = self.conv.forward(&x)?;
        let x = self.ffn2.forward(&x)?;
        Ok(self.ln_norm.apply(&x)?)
    }
}

impl HasStateDict for ConformerBlock {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.ffn1.state_dict(&prefixed(prefix, "ffn1"));
        sd.extend(self.mha.state_dict(&prefixed(prefix, "mha")));
        sd.extend(self.conv.state_dict(&prefixed(prefix, "conv")));
        sd.extend(self.ffn2.state_dict(&prefixed(prefix, "ffn2")));
        sd.extend(self.ln_norm.state_dict(&prefixed(prefix, "ln_norm")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.ffn1.load_state_dict(sd, &prefixed(prefix, "ffn1"))?;
        self.mha.load_state_dict(sd, &prefixed(prefix, "mha"))?;
        self.conv.load_state_dict(sd, &prefixed(prefix, "conv"))?;
        self.ffn2.load_state_dict(sd, &prefixed(prefix, "ffn2"))?;
        self.ln_norm.load_state_dict(sd, &prefixed(prefix, "ln_norm"))?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct ConformerEncoder {
    pub layers: Vec<ConformerBlock>,
}

impl ConformerEncoder {
    pub fn empty(
        attention_in: usize,
        ffn_hidden: usize,
        num_head: usize,
        num_layer: usize,
        kernel_size: usize,
    ) -> Self {
        Self {
            layers: (0..num_layer)
                .map(|_| ConformerBlock::empty(attention_in, ffn_hidden, num_head, kernel_size))
                .collect(),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    /// Same as [`forward`](Self::forward) but also returns each block's
    /// output. Useful for stage-by-stage parity testing.
    pub fn forward_with_block_outputs(&self, x: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let mut x = x.clone();
        let mut outputs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            x = layer.forward(&x)?;
            outputs.push(x.clone());
        }
        Ok((x, outputs))
    }
}

impl HasStateDict for ConformerEncoder {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        for (i, layer) in self.layers.iter().enumerate() {
            sd.extend(layer.state_dict(&format!("{prefix}.conformer_layer.{i}")));
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(sd, &format!("{prefix}.conformer_layer.{i}"))?;
        }
        Ok(())
    }
}
