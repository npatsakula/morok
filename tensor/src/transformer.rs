//! Transformer building blocks: embedding, attention, rotary position embeddings.

use crate::Tensor;
use bon::bon;
use morok_dtype::DType;
use morok_ir::ConstValue;
use snafu::ensure;

use crate::error::FloatDTypeRequiredSnafu;

type Result<T> = crate::Result<T>;

impl Tensor {
    /// Embedding lookup: `self` is the weight table `[vocab_size, embed_dim]`.
    /// Returns `self[indices]` with shape `[*indices.shape, embed_dim]`.
    pub fn embedding(&self, indices: &Tensor) -> Result<Tensor> {
        let weight_shape = self.shape()?;
        let embed_dim = weight_shape[1].as_const().expect("embedding weight dim 1 must be concrete") as isize;
        let idx_shape = indices.shape()?;

        let flat = indices.try_reshape([-1])?;
        let expanded = flat.try_unsqueeze(-1)?.try_expand([-1, embed_dim])?;
        let gathered = self.gather(0, &expanded)?;

        let mut out_shape: Vec<isize> =
            idx_shape.iter().map(|d| d.as_const().expect("embedding index dims must be concrete") as isize).collect();
        out_shape.push(embed_dim);
        gathered.try_reshape(&out_shape)
    }

    /// Apply rotary position embedding rotation.
    /// `self`: `[..., rot_dim]` tensor to rotate.
    /// `cos`, `sin`: broadcastable to `self`'s shape `[..., rot_dim/2]`.
    /// If interleaved: pairs are (even, odd) indices.
    /// If not interleaved: pairs are (first_half, second_half).
    pub fn apply_rotary_emb(&self, cos: &Tensor, sin: &Tensor, interleaved: bool) -> Result<Tensor> {
        let shape = self.shape()?;
        let last_dim = shape
            .last()
            .expect("apply_rotary_emb requires non-scalar input")
            .as_const()
            .expect("last dim must be concrete");
        let half = last_dim / 2;

        let (x1, x2) = if interleaved {
            let mut rs: Vec<isize> = shape
                .iter()
                .take(shape.len() - 1)
                .map(|d| d.as_const().expect("dims must be concrete") as isize)
                .collect();
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
            let mut fs: Vec<isize> = shape.iter().map(|d| d.as_const().unwrap() as isize).collect();
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
    #[builder]
    pub fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        scale: Option<f64>,
        #[builder(default)] is_causal: bool,
        softcap: Option<f64>,
    ) -> Result<Tensor> {
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
        let head_dim = q_shape[q_shape.len() - 1].as_const().expect("Q head_dim must be concrete");
        let scale_val = scale.unwrap_or(1.0 / (head_dim as f64).sqrt());

        let scores_dtype = self.uop().dtype();

        // Q @ K^T
        let kt = key.try_transpose(-1, -2)?;
        let mut scores = self.matmul(&kt)?;

        // Scale
        let scale_t = Tensor::const_(scale_val, scores_dtype.clone());
        scores = scores.try_mul(&scale_t)?;

        // Causal mask
        if is_causal {
            let q_len = q_shape[q_shape.len() - 2].as_const().expect("Q seq_len must be concrete");
            let k_len = k_shape[k_shape.len() - 2].as_const().expect("K seq_len must be concrete");
            let causal = Tensor::full(&[q_len, k_len], true, DType::Bool)?.tril(0)?;
            let neg_large = Tensor::const_(ConstValue::min(scores_dtype.base()), scores_dtype.clone());
            scores = scores.where_(&causal, &neg_large)?;
        }

        // Attention mask
        let mut bool_mask: Option<Tensor> = None;
        if let Some(mask) = attn_mask {
            let mask_dtype = mask.uop().dtype();
            if mask_dtype == DType::Bool {
                // Bool mask: True = mask out, False = keep.
                let neg_large = Tensor::const_(ConstValue::min(scores_dtype.base()), scores_dtype.clone());
                let zero = Tensor::const_(ConstValue::zero(scores_dtype.base()), scores_dtype.clone());
                let additive = neg_large.where_(mask, &zero)?;
                scores = scores.try_add(&additive)?;
                bool_mask = Some(mask.clone());
            } else {
                // Float additive mask
                scores = scores.try_add(mask)?;
            }
        }

        // Softcap
        if let Some(cap) = softcap
            && cap > 0.0
        {
            let cap_t = Tensor::const_(cap, scores_dtype.clone());
            scores = scores.try_div(&cap_t)?.tanh()?.try_mul(&cap_t)?;
        }

        // Softmax + output
        let mut attn_weights = scores.softmax(-1isize)?;
        if let Some(mask) = bool_mask.as_ref() {
            let zero = Tensor::const_(ConstValue::zero(scores_dtype.base()), scores_dtype);
            attn_weights = zero.where_(mask, &attn_weights)?;
        }
        attn_weights.matmul(value)
    }
}
