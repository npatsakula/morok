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
use svod_tensor::nn::{BatchNorm2d, Conv1d, Layer, LayerNorm, Linear, Module};

use crate::init::{Bias, layer_norm};

use super::error::Result;

/// PyTorch's `nn.BatchNorm1d` default epsilon.
const EPS: f64 = 1e-5;

/// Every `Linear` and `Conv1d` here is zero-biased, as upstream initializes them.
fn linear(inp: usize, out: usize) -> Linear {
    crate::init::linear(inp, out, Bias::Zero, DType::Float32)
}

fn conv1d(in_per_group: usize, out: usize, kernel: usize) -> Conv1d {
    crate::init::conv1d(in_per_group, out, kernel, Bias::Zero, DType::Float32)
}

// ---------------------------------------------------------------------------
// PositionwiseFeedForward
// ---------------------------------------------------------------------------

/// Pre-norm FFN with 0.5-scaled residual — Conformer convention.
#[derive(Clone, Module)]
pub struct PositionwiseFeedForward {
    pub ln_norm: LayerNorm,
    #[module(key = "w_1")]
    pub w1: Linear,
    #[module(key = "w_2")]
    pub w2: Linear,
}

impl PositionwiseFeedForward {
    pub fn empty(in_size: usize, ffn_hidden: usize) -> Self {
        Self {
            ln_norm: layer_norm(in_size, DType::Float32),
            w1: linear(in_size, ffn_hidden),
            w2: linear(ffn_hidden, in_size),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Python `Swish` == SiLU.
        let y = self.w1.forward(&self.ln_norm.forward(x)?)?.silu()?;
        Ok(x.try_add(&self.w2.forward(&y)?.try_mul(0.5)?)?)
    }
}

// ---------------------------------------------------------------------------
// Plain MultiHeadSelfAttention (no rel-pos; use_posi=False)
// ---------------------------------------------------------------------------

#[derive(Clone, Module)]
pub struct PlainMultiHeadSelfAttention {
    pub n_heads: usize,
    #[module(key = "linearQ")]
    pub q: Linear,
    #[module(key = "linearK")]
    pub k: Linear,
    #[module(key = "linearV")]
    pub v: Linear,
    #[module(key = "linearO")]
    pub o: Linear,
}

impl PlainMultiHeadSelfAttention {
    pub fn empty(n_units: usize, n_heads: usize) -> Self {
        assert!(n_units.is_multiple_of(n_heads), "n_units must be divisible by n_heads");
        Self {
            n_heads,
            q: linear(n_units, n_units),
            k: linear(n_units, n_units),
            v: linear(n_units, n_units),
            o: linear(n_units, n_units),
        }
    }

    /// Forward on `(B, L, n_units)` → `(B, L, n_units)`. Plain scaled-dot-product.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let project = |l: &Linear| l.forward(x)?.split_heads(self.n_heads);
        let (q, k, v) = (project(&self.q)?, project(&self.k)?, project(&self.v)?);
        let attended = q.scaled_dot_product_attention().key(&k).value(&v).call()?;
        Ok(self.o.forward(&attended.merge_heads()?)?)
    }
}

// ---------------------------------------------------------------------------
// ConformerMHA — pre-norm wrap + residual + dropout
// ---------------------------------------------------------------------------

#[derive(Clone, Module)]
pub struct ConformerMHA {
    pub ln_norm: LayerNorm,
    pub mha: PlainMultiHeadSelfAttention,
}

impl ConformerMHA {
    pub fn empty(in_size: usize, num_head: usize) -> Self {
        Self {
            ln_norm: layer_norm(in_size, DType::Float32),
            mha: PlainMultiHeadSelfAttention::empty(in_size, num_head),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let delta = self.mha.forward(&self.ln_norm.forward(x)?)?;
        Ok(x.try_add(&delta)?)
    }
}

// ---------------------------------------------------------------------------
// ConvolutionModule
// ---------------------------------------------------------------------------

#[derive(Clone, Module)]
pub struct ConvolutionModule {
    pub ln_norm: LayerNorm,
    /// `(2C, C, 1)` 1×1 conv to 2× channels (split into GLU halves).
    #[module(key = "pointwise_conv1")]
    pub pointwise1: Conv1d,
    /// `(C, 1, k)` depthwise conv (groups = channels) with SAME padding.
    #[module(key = "depthwise_conv")]
    pub depthwise: Conv1d,
    pub bn_norm: BatchNorm2d,
    /// `(C, C, 1)` 1×1 conv back to the channel count.
    #[module(key = "pointwise_conv2")]
    pub pointwise2: Conv1d,
}

impl ConvolutionModule {
    pub fn empty(channels: usize, kernel_size: usize) -> Self {
        assert!(!kernel_size.is_multiple_of(2), "kernel_size must be odd for SAME padding");
        let pad = ((kernel_size - 1) / 2) as isize;
        Self {
            ln_norm: layer_norm(channels, DType::Float32),
            pointwise1: conv1d(channels, 2 * channels, 1),
            depthwise: conv1d(1, channels, kernel_size).with_groups(channels).with_padding((pad, pad)),
            bn_norm: BatchNorm2d::with_dims(channels, EPS, DType::Float32),
            pointwise2: conv1d(channels, channels, 1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // (B, T, C) → (B, C, T), pointwise expand, GLU over the channel dim.
        let y = self.ln_norm.forward(x)?.try_permute(&[0, 2, 1])?;
        let y = self.pointwise1.forward(&y)?.glu(1)?;
        // Depthwise + BN over the channel axis (axis 1, the NCT default) + Swish.
        let y = self.bn_norm.forward(&self.depthwise.forward(&y)?)?.silu()?;
        // Back to (B, T, C) and residual.
        Ok(x.try_add(&self.pointwise2.forward(&y)?.try_permute(&[0, 2, 1])?)?)
    }
}

// ---------------------------------------------------------------------------
// ConformerBlock and ConformerEncoder
// ---------------------------------------------------------------------------

#[derive(Clone, Module)]
pub struct ConformerBlock {
    pub ffn1: PositionwiseFeedForward,
    pub mha: ConformerMHA,
    pub conv: ConvolutionModule,
    pub ffn2: PositionwiseFeedForward,
    pub ln_norm: LayerNorm,
}

impl ConformerBlock {
    pub fn empty(in_size: usize, ffn_hidden: usize, num_head: usize, kernel_size: usize) -> Self {
        Self {
            ffn1: PositionwiseFeedForward::empty(in_size, ffn_hidden),
            mha: ConformerMHA::empty(in_size, num_head),
            conv: ConvolutionModule::empty(in_size, kernel_size),
            ffn2: PositionwiseFeedForward::empty(in_size, ffn_hidden),
            ln_norm: layer_norm(in_size, DType::Float32),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ffn1.forward(x)?;
        let x = self.mha.forward(&x)?;
        let x = self.conv.forward(&x)?;
        let x = self.ffn2.forward(&x)?;
        Ok(self.ln_norm.forward(&x)?)
    }
}

#[derive(Clone, Module)]
pub struct ConformerEncoder {
    #[module(key = "conformer_layer")]
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
        self.layers.iter().try_fold(x.clone(), |x, layer| layer.forward(&x))
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
