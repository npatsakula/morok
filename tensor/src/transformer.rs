//! Transformer building blocks: embedding, attention, rotary position embeddings.

use crate::Tensor;
use bon::bon;
use snafu::{OptionExt, ensure};
use svod_dtype::DType;
use svod_ir::ConstValue;

use crate::error::{FloatDTypeRequiredSnafu, NdimMinimumSnafu, SymbolicShapeUnsupportedSnafu};

type Result<T> = crate::Result<T>;

impl Tensor {
    /// Embedding lookup: `self` is the weight table `[vocab_size, embed_dim]`.
    /// Returns `self[indices]` with shape `[*indices.shape, embed_dim]`.
    ///
    /// Mirrors tinygrad's `_embedding_fwd`: a one-hot mask
    /// (`indices.unsqueeze(-1) == arange(vocab)`) selects rows via
    /// `where(weight, 0).sum(vocab)`. It operates on the index's natural shape
    /// — no flatten-to-`[-1]` — so a symbolic dim on `indices` (e.g. a JIT
    /// batch bound to a `Variable`) passes through. Only the vocab axis
    /// (weight dim 0) must be concrete.
    #[track_caller]
    pub fn embedding(&self, indices: &Tensor) -> Result<Tensor> {
        origin_call!("embedding");
        let weight_shape = self.shape()?;
        let vocab_size =
            weight_shape[0].as_const().context(SymbolicShapeUnsupportedSnafu { operation: "embedding" })?;

        // one-hot mask: [*idx_shape, vocab] bool — True where the vocab row
        // matches the index value.
        let vocab_arange = Tensor::arange(0, Some(vocab_size as i64), None)?.cast(indices.uop().dtype())?;
        let mask = indices.try_unsqueeze(-1)?.try_eq(&vocab_arange)?;

        // Reshape weight to [1... (per idx dim), vocab, embed] so its trailing
        // (vocab, embed) rows broadcast under the mask in the where_. Binary
        // ops broadcast aligning trailing dims, so no explicit broadcast_to.
        let idx_ndim = indices.shape()?.len();
        let mut leading_ones: Vec<isize> = (0..idx_ndim).map(|_| 1).collect();
        leading_ones.push(vocab_size as isize);
        leading_ones.push(-1); // embed_dim inferred from weight.
        let weight_bc = self.try_reshape(&leading_ones)?;

        // Select + collapse the vocab axis. mask.unsqueeze(-1) broadcasts over
        // embed; summing the vocab axis (now -2) leaves [*idx_shape, embed].
        weight_bc
            .where_(
                &mask.try_unsqueeze(-1)?,
                &Tensor::new(self.uop().const_like(ConstValue::zero(self.uop().dtype().base()))),
            )?
            .sum_with()
            .axes(-2isize)
            .dtype(self.uop().dtype())
            .call()
    }

    /// Apply rotary position embedding rotation.
    /// `self`: `[..., rot_dim]` tensor to rotate.
    /// `cos`, `sin`: broadcastable to `self`'s shape `[..., rot_dim/2]`.
    /// If interleaved: pairs are (even, odd) indices.
    /// If not interleaved: pairs are (first_half, second_half).
    #[track_caller]
    pub fn apply_rotary_emb(&self, cos: &Tensor, sin: &Tensor, interleaved: bool) -> Result<Tensor> {
        origin_call!("apply_rotary_emb");
        let shape = self.shape()?;
        let last_dim = shape
            .last()
            .context(NdimMinimumSnafu { op: "apply_rotary_emb", min: 1usize, actual: 0usize })?
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "apply_rotary_emb" })?;
        let half = last_dim / 2;

        let (x1, x2) = if interleaved {
            let mut rs: Vec<isize> = shape
                .iter()
                .take(shape.len() - 1)
                .map(|d| {
                    Ok(d.as_const().context(SymbolicShapeUnsupportedSnafu { operation: "apply_rotary_emb" })? as isize)
                })
                .collect::<Result<_>>()?;
            rs.push(half as isize);
            rs.push(2);
            let r = self.try_reshape(&rs)?;
            let p = r.split(&[1, 1], -1)?;
            (p[0].try_squeeze(Some(-1))?, p[1].try_squeeze(Some(-1))?)
        } else {
            let p = self.split(&[half, half], -1)?;
            (p[0].clone(), p[1].clone())
        };

        let real = x1.try_mul(cos)?.try_sub(&x2.try_mul(sin)?)?;
        let imag = x1.try_mul(sin)?.try_add(&x2.try_mul(cos)?)?;

        if interleaved {
            let stacked = Tensor::stack(&[&real, &imag], -1)?;
            let mut fs: Vec<isize> = shape
                .iter()
                .map(|d| {
                    Ok(d.as_const().context(SymbolicShapeUnsupportedSnafu { operation: "apply_rotary_emb" })? as isize)
                })
                .collect::<Result<_>>()?;
            // Last dim already correct from original shape
            let _ = fs.last_mut().map(|d| *d = last_dim as isize);
            stacked.try_reshape(&fs)
        } else {
            Tensor::cat(&[&real, &imag], -1)
        }
    }
}

#[bon]
impl Tensor {
    /// Scaled dot-product attention.
    /// `self` (Q): `[B, H, Sq, D]`, `key` (K): `[B, H, Sk, D]`, `value` (V): `[B, H, Sk, Dv]`.
    /// Returns `[B, H, Sq, Dv]`.
    ///
    /// `window = Some((left, right))` restricts each query `q` to keys in
    /// `[q - left, q + right]` (sliding-window / banded attention, as in
    /// ModernBERT's local layers). `None` = full (global) attention. The band is
    /// intersected with any causal mask and the boolean `attn_mask` (when the
    /// latter encodes padding).
    #[builder]
    #[track_caller]
    pub fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        scale: Option<f64>,
        #[builder(default)] is_causal: bool,
        window: Option<(usize, usize)>,
        softcap: Option<f64>,
    ) -> Result<Tensor> {
        origin_call!("scaled_dot_product_attention");
        let q_dtype = self.uop().dtype();
        ensure!(
            q_dtype.is_float(),
            FloatDTypeRequiredSnafu { op: "scaled_dot_product_attention", arg: "query", dtype: q_dtype.clone() }
        );
        let k_dtype = key.uop().dtype();
        ensure!(
            k_dtype.is_float(),
            FloatDTypeRequiredSnafu { op: "scaled_dot_product_attention", arg: "key", dtype: k_dtype.clone() }
        );
        let v_dtype = value.uop().dtype();
        ensure!(
            v_dtype.is_float(),
            FloatDTypeRequiredSnafu { op: "scaled_dot_product_attention", arg: "value", dtype: v_dtype.clone() }
        );

        let q_shape = self.shape()?;
        let k_shape = key.shape()?;
        let head_dim = q_shape[q_shape.len() - 1]
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "scaled_dot_product_attention" })?;
        let scale_val = scale.unwrap_or(1.0 / (head_dim as f64).sqrt());

        // Scores are formed, masked and softmaxed in the sum accumulator dtype
        // (float32 for fp16/bf16/fp8): the fallback softmax otherwise runs in
        // float16, where the additive mask constant is only -65504.
        let scores_dtype = Tensor::sum_acc_dtype(&q_dtype);

        // Q @ K^T
        let kt = key.try_transpose(-1, -2)?;
        let mut scores = self.matmul_with().other(&kt).dtype(scores_dtype.clone()).call()?;

        // Scale
        let scale_t = Tensor::const_(scale_val, scores_dtype.clone());
        scores = scores.try_mul(&scale_t)?;

        // Softcap the raw scaled scores, before any mask: capping afterwards
        // squashes a masked `dtype::min` to `-cap`, leaving it softmax weight.
        if let Some(cap) = softcap
            && cap > 0.0
        {
            let cap_t = Tensor::const_(cap, scores_dtype.clone());
            scores = scores.try_div(&cap_t)?.tanh()?.try_mul(&cap_t)?;
        }

        // Build a boolean "keep" mask that ANDs together the causal constraint,
        // the optional sliding-window band, and the user-supplied `attn_mask`.
        // True = attend, False = masked out. The mask is applied additively
        // (mask_out → -large) before softmax, and the weights are also zeroed
        // post-softmax to guarantee exact-zero out-of-band columns even when a
        // full row is masked (softmax-of-all-equal → uniform, not zero).
        let q_len = q_shape[q_shape.len() - 2]
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "scaled_dot_product_attention" })?;
        let k_len = k_shape[k_shape.len() - 2]
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "scaled_dot_product_attention" })?;

        let mut keep_mask: Option<Tensor> = None;

        // Causal: keep k ≤ q.
        if is_causal {
            let q_idx = Tensor::arange(0, Some(q_len as i64), None)?.try_unsqueeze(-1)?; // (Q, 1)
            let k_idx = Tensor::arange(0, Some(k_len as i64), None)?; // (K,)
            let causal = k_idx.try_le(&q_idx)?; // k <= q
            keep_mask = Some(causal);
        }

        // Sliding-window band: keep q - left ≤ k ≤ q + right.
        if let Some((left, right)) = window {
            let q_idx = Tensor::arange(0, Some(q_len as i64), None)?.try_unsqueeze(-1)?; // (Q, 1)
            let k_idx = Tensor::arange(0, Some(k_len as i64), None)?; // (K,)
            let lo = Tensor::const_(ConstValue::Int(left as i64), DType::Int32);
            let hi = Tensor::const_(ConstValue::Int(right as i64), DType::Int32);
            // q - left <= k  AND  k <= q + right
            let lower = q_idx.try_sub(&lo)?.try_le(&k_idx)?;
            let upper = k_idx.try_le(&q_idx.try_add(&hi)?)?;
            let band = lower.try_bitand(&upper)?;
            keep_mask = Some(match keep_mask {
                Some(prev) => prev.try_bitand(&band)?,
                None => band,
            });
        }

        // User-supplied attention mask. Bool: True = mask OUT (False = keep) —
        // invert before ANDing. Float: additive, applied separately below.
        let mut float_additive_mask: Option<&Tensor> = None;
        if let Some(mask) = attn_mask {
            if mask.uop().dtype() == DType::Bool {
                let keep = mask.logical_not()?; // True = keep
                keep_mask = Some(match keep_mask {
                    Some(prev) => prev.try_bitand(&keep)?,
                    None => keep,
                });
            } else {
                float_additive_mask = Some(mask);
            }
        }

        // Apply the boolean keep mask additively (out-of-band → -large).
        if let Some(keep) = keep_mask.as_ref() {
            let neg_large = Tensor::const_(ConstValue::min(scores_dtype.base()), scores_dtype.clone());
            scores = scores.where_(keep, &neg_large)?;
        }
        // Apply a float additive mask (e.g. a pre-computed -inf padding mask).
        if let Some(additive) = float_additive_mask {
            scores = scores.try_add(additive)?;
        }

        // Softmax + output. Re-zero out-of-band weights so a fully-masked row
        // (whose softmax would otherwise be uniform over the masked keys)
        // produces exact zeros rather than `1/k_len` leakage.
        let mut attn_weights = scores.softmax(-1isize)?;
        if let Some(keep) = keep_mask.as_ref() {
            let zero = Tensor::const_(ConstValue::zero(scores_dtype.base()), scores_dtype.clone());
            let masked_out = keep.logical_not()?;
            attn_weights = zero.where_(&masked_out, &attn_weights)?;
        }
        // Back to the query dtype for `@ V`, as tinygrad does.
        if scores_dtype != q_dtype {
            attn_weights = attn_weights.cast(q_dtype)?;
        }
        attn_weights.matmul(value)
    }
}
