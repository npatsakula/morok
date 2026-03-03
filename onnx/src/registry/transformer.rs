use morok_dtype::DType;
use morok_tensor::Tensor;

use crate::error::Result;
use crate::parser::onnx::NodeProto;

use super::*;

// =========================================================================
// Standard ONNX ops
// =========================================================================

/// RMSNormalization: `x * rsqrt(mean(x^2) + eps) * scale`
pub(crate) fn op_rms_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let scale = inp(inputs, 1);
    let axis = get_attr_int(node, "axis", -1) as isize;
    let epsilon = get_attr_float(node, "epsilon", 1e-5) as f64;
    Ok(x.rms_norm(axis, epsilon)?.try_mul(scale)?)
}

/// Standard ONNX Attention (pre-projected Q, K, V).
pub(crate) fn op_attention_onnx(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    let q = inp(inputs, 0);
    let k = inp(inputs, 1);
    let v = inp(inputs, 2);
    let attn_mask = inputs.get(3).and_then(|o| o.as_ref());
    let past_key = inputs.get(4).and_then(|o| o.as_ref());
    let past_value = inputs.get(5).and_then(|o| o.as_ref());
    let nonpad_kv_seqlen = inputs.get(6).and_then(|o| o.as_ref());

    let is_causal = get_attr_int(node, "is_causal", 0) != 0;
    let q_num_heads = get_attr_int(node, "q_num_heads", 0) as usize;
    let kv_num_heads = get_attr_int(node, "kv_num_heads", 0) as usize;
    let qk_matmul_output_mode = get_attr_int(node, "qk_matmul_output_mode", 0);
    let scale_attr = get_attr_float(node, "scale", 0.0);
    let scale = if scale_attr != 0.0 { Some(scale_attr as f64) } else { None };
    let softcap_val = get_attr_float(node, "softcap", 0.0) as f64;
    let softcap = if softcap_val > 0.0 { Some(softcap_val) } else { None };
    let softmax_precision = get_attr_int(node, "softmax_precision", 0);

    let q_shape = q.shape()?;
    let is_3d = q_shape.len() == 3;

    // Reshape 3D → 4D [B, S, hidden] → [B, H, S, D]
    let (q, k, v) = if is_3d {
        if q_num_heads == 0 {
            return Err(Error::IrConstruction { details: "q_num_heads required for 3D input".into() });
        }
        if kv_num_heads == 0 {
            return Err(Error::IrConstruction { details: "kv_num_heads required for 3D input".into() });
        }
        let q_hidden = q_shape[2].as_const().unwrap();
        let q_head_dim = q_hidden / q_num_heads;
        let k_shape = k.shape()?;
        let k_head_dim = k_shape[2].as_const().unwrap() / kv_num_heads;
        let v_shape = v.shape()?;
        let v_head_dim = v_shape[2].as_const().unwrap() / kv_num_heads;
        let batch = q_shape[0].as_const().unwrap() as isize;
        let q_seq = q_shape[1].as_const().unwrap() as isize;
        let k_seq = k_shape[1].as_const().unwrap() as isize;

        let q =
            q.try_reshape(&[batch, q_seq, q_num_heads as isize, q_head_dim as isize])?.try_permute(&[0, 2, 1, 3])?;
        let k =
            k.try_reshape(&[batch, k_seq, kv_num_heads as isize, k_head_dim as isize])?.try_permute(&[0, 2, 1, 3])?;
        let v =
            v.try_reshape(&[batch, k_seq, kv_num_heads as isize, v_head_dim as isize])?.try_permute(&[0, 2, 1, 3])?;
        (q, k, v)
    } else {
        (q.clone(), k.clone(), v.clone())
    };

    // Past KV concatenation
    let k = if let Some(pk) = past_key { Tensor::cat(&[pk, &k], -2)? } else { k };
    let v = if let Some(pv) = past_value { Tensor::cat(&[pv, &v], -2)? } else { v };

    let present_key = k.clone();
    let present_value = v.clone();

    // GQA: repeat-interleave K/V heads to match Q head count
    // For 4D input, head counts come from shape dim 1 when attributes are unset
    let q_s = q.shape()?;
    let eff_q_heads = if q_num_heads > 0 { q_num_heads } else { q_s[1].as_const().unwrap() };
    let eff_kv_heads = if kv_num_heads > 0 { kv_num_heads } else { k.shape()?[1].as_const().unwrap() };
    let (k, v) = if eff_q_heads != eff_kv_heads {
        let ratio = eff_q_heads / eff_kv_heads;
        let k_s = k.shape()?;
        let b = k_s[0].as_const().unwrap() as isize;
        let kv_h = eff_kv_heads as isize;
        let r = ratio as isize;
        let s_k = k_s[2].as_const().unwrap() as isize;
        let d_k = k_s[3].as_const().unwrap() as isize;
        let v_s = v.shape()?;
        let d_v = v_s[3].as_const().unwrap() as isize;
        // [B, kv_h, S, D] → [B, kv_h, 1, S, D] → expand → [B, q_h, S, D]
        let k = k.try_unsqueeze(2)?.try_expand(&[b, kv_h, r, s_k, d_k])?.try_reshape(&[
            b,
            eff_q_heads as isize,
            s_k,
            d_k,
        ])?;
        let v = v.try_unsqueeze(2)?.try_expand(&[b, kv_h, r, s_k, d_v])?.try_reshape(&[
            b,
            eff_q_heads as isize,
            s_k,
            d_v,
        ])?;
        (k, v)
    } else {
        (k, v)
    };

    let q_dtype = q.uop().dtype();
    let q_s = q.shape()?;
    let k_s = k.shape()?;
    let head_dim = q_s[q_s.len() - 1].as_const().unwrap();
    let scale_val = scale.unwrap_or(1.0 / (head_dim as f64).sqrt());

    // Handle nonpad_kv_seqlen: pad attn_mask to full K length and create padding mask
    let full_k_len = k_s[k_s.len() - 2].as_const().unwrap();
    let attn_mask = if let Some(seqlen) = nonpad_kv_seqlen {
        // Padding mask: position >= seqlen[b] → -inf, else 0. Shape: [B, 1, 1, full_k]
        let range = Tensor::arange(full_k_len as i64, None, None)?;
        let valid = range.try_lt(&seqlen.try_unsqueeze(-1)?)?;
        let neg_inf = Tensor::const_(f64::NEG_INFINITY, q_dtype.clone());
        let zero = Tensor::const_(0.0f64, q_dtype.clone());
        let pad_mask = zero.where_(&valid, &neg_inf)?.try_unsqueeze(1)?.try_unsqueeze(1)?;

        // Pad existing attn_mask to full K length, then combine with padding mask
        if let Some(mask) = attn_mask {
            let mask_k = mask.shape()?[mask.ndim()? - 1].as_const().unwrap();
            let padded_mask = if mask_k < full_k_len {
                let mut pad_shape: Vec<isize> = mask.shape()?.iter().map(|d| d.as_const().unwrap() as isize).collect();
                let pad_cols = full_k_len - mask_k;
                *pad_shape.last_mut().unwrap() = pad_cols as isize;
                let pad_fill = Tensor::full(
                    &pad_shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                    0.0f64,
                    mask.uop().dtype(),
                )?;
                Tensor::cat(&[mask, &pad_fill], -1)?
            } else {
                mask.clone()
            };
            Some(padded_mask.try_add(&pad_mask)?)
        } else {
            Some(pad_mask)
        }
    } else {
        attn_mask.cloned()
    };
    let attn_mask = attn_mask.as_ref();

    // Always compute attention manually to capture QK intermediates for all modes
    let kt = k.try_transpose(-1, -2)?;
    let mut scores = q.matmul(&kt)?;
    let scale_t = Tensor::const_(scale_val, q_dtype.clone());
    scores = scores.try_mul(&scale_t)?;

    // Mode 0: raw Q@K^T * scale
    let qk_mode0 = scores.clone();

    // Causal mask: only restrict CURRENT K positions, past positions always attendable
    if is_causal {
        let past_seq_len = past_key.map(|pk| pk.shape().unwrap()[2].as_const().unwrap()).unwrap_or(0);
        let q_len = q_s[q_s.len() - 2].as_const().unwrap();
        let causal = Tensor::full(&[q_len, full_k_len], true, DType::Bool)?.tril(past_seq_len as i64)?;
        let neg_inf = Tensor::const_(f64::NEG_INFINITY, q_dtype.clone());
        scores = scores.where_(&causal, &neg_inf)?;
    }

    // Attention mask
    if let Some(mask) = attn_mask {
        let mask_dtype = mask.uop().dtype();
        if mask_dtype == DType::Bool {
            let neg_inf = Tensor::const_(f64::NEG_INFINITY, q_dtype.clone());
            let zero = Tensor::const_(0.0f64, q_dtype.clone());
            let additive = zero.where_(mask, &neg_inf)?;
            scores = scores.try_add(&additive)?;
        } else {
            scores = scores.try_add(mask)?;
        }
    }

    // Mode 1: after mask
    let qk_mode1 = scores.clone();

    // Softcap
    if let Some(cap) = softcap {
        let cap_t = Tensor::const_(cap, q_dtype.clone());
        scores = scores.try_div(&cap_t)?.tanh()?.try_mul(&cap_t)?;
    }

    // Mode 2: after softcap
    let qk_mode2 = scores.clone();

    // Softmax precision casting
    let scores = if softmax_precision > 0 {
        let sm_dtype = match softmax_precision {
            1 => DType::Float32,
            10 => DType::Float16,
            16 => DType::BFloat16,
            _ => DType::Float32,
        };
        scores.cast(sm_dtype)?
    } else {
        scores
    };

    let attn_weights = scores.softmax(-1isize)?.cast(q_dtype.clone())?;

    // Mode 3: after softmax
    let qk_mode3 = attn_weights.clone();

    let output = attn_weights.matmul(&v)?.cast(q_dtype)?;

    let qk_return = match qk_matmul_output_mode {
        1 => qk_mode1,
        2 => qk_mode2,
        3 => qk_mode3,
        _ => qk_mode0,
    };

    // Reshape back to 3D if input was 3D
    let output = if is_3d {
        let out_shape = output.shape()?;
        let batch = out_shape[0].as_const().unwrap() as isize;
        let seq = out_shape[2].as_const().unwrap() as isize;
        output.try_permute(&[0, 2, 1, 3])?.try_reshape(&[batch, seq, -1])?
    } else {
        output
    };

    Ok(vec![output, present_key, present_value, qk_return])
}

// =========================================================================
// Microsoft contrib ops
// =========================================================================

/// SkipLayerNormalization: `x + skip [+ bias] → layernorm → * gamma [+ beta]`
pub(crate) fn op_skip_layer_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    let x = inp(inputs, 0);
    let skip = inp(inputs, 1);
    let gamma = inp(inputs, 2);
    let beta = inputs.get(3).and_then(|o| o.as_ref());
    let bias = inputs.get(4).and_then(|o| o.as_ref());
    let epsilon = get_attr_float(node, "epsilon", 1e-12) as f64;

    let mut x_sum = x.try_add(skip)?;
    if let Some(b) = bias {
        x_sum = x_sum.try_add(b)?;
    }
    let mut out = x_sum.layernorm(-1, epsilon)?.try_mul(gamma)?;
    if let Some(b) = beta {
        out = out.try_add(b)?;
    }
    let dummy = Tensor::from_slice([0.0f32]);
    Ok(vec![out, dummy.clone(), dummy, x_sum])
}

/// EmbedLayerNormalization: word + position [+ segment] embedding → layernorm → * gamma + beta
pub(crate) fn op_embed_layer_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    let input_ids = inp(inputs, 0);
    let segment_ids = inputs.get(1).and_then(|o| o.as_ref());
    let word_emb = inp(inputs, 2);
    let pos_emb = inp(inputs, 3);
    let seg_emb = inputs.get(4).and_then(|o| o.as_ref());
    let gamma = inp(inputs, 5);
    let beta = inp(inputs, 6);
    let position_ids = inputs.get(8).and_then(|o| o.as_ref());
    let epsilon = get_attr_float(node, "epsilon", 1e-12) as f64;

    let w = word_emb.embedding(input_ids)?;

    let pos_ids = match position_ids {
        Some(ids) => ids.clone(),
        None => {
            let id_shape = input_ids.shape()?;
            let seq_len = id_shape[1].as_const().unwrap() as i64;
            let batch = id_shape[0].as_const().unwrap() as isize;
            let pos = Tensor::arange(seq_len, None, None)?;
            pos.try_unsqueeze(0)?.try_expand(&[batch, seq_len as isize])?
        }
    };
    let p = pos_emb.embedding(&pos_ids)?;

    let mut sum = w.try_add(&p)?;
    if let (Some(sid), Some(se)) = (segment_ids, seg_emb) {
        sum = sum.try_add(&se.embedding(sid)?)?;
    }

    let out = sum.layernorm(-1, epsilon)?.try_mul(gamma)?.try_add(beta)?;
    let dummy = Tensor::from_slice([0.0f32]);
    Ok(vec![out, dummy, sum])
}

/// RotaryEmbedding (standard ONNX opset 23): inputs are (input, cos_cache, sin_cache, position_ids?)
pub(crate) fn op_rotary_embedding(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    rotary_embedding_impl(
        inp(inputs, 0),
        inp(inputs, 1),
        inp(inputs, 2),
        inputs.get(3).and_then(|o| o.as_ref()),
        node,
    )
}

/// RotaryEmbedding (Microsoft contrib): inputs are (input, position_ids?, cos_cache, sin_cache)
pub(crate) fn op_rotary_embedding_contrib(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    rotary_embedding_impl(
        inp(inputs, 0),
        inp(inputs, 2),
        inp(inputs, 3),
        inputs.get(1).and_then(|o| o.as_ref()),
        node,
    )
}

/// Shared RotaryEmbedding implementation.
///
/// reshape → split rotate/pass → lookup cos/sin → apply rotation → concat
fn rotary_embedding_impl(
    x: &Tensor,
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    position_ids: Option<&Tensor>,
    node: &NodeProto,
) -> Result<Vec<Tensor>> {
    let interleaved = get_attr_int(node, "interleaved", 0) != 0;
    let num_heads = get_attr_int(node, "num_heads", 0) as usize;
    let rotary_embedding_dim = get_attr_int(node, "rotary_embedding_dim", 0) as usize;

    let x_shape = x.shape()?;
    let x_ndim = x_shape.len();

    // Normalize shape to [B, S, H, D]
    let x_work = if x_ndim == 4 {
        // [B, H, S, D] -> [B, S, H, D]
        x.try_permute(&[0, 2, 1, 3])?
    } else if x_ndim == 3 {
        if num_heads == 0 {
            return Err(Error::IrConstruction { details: "num_heads must be provided for 3D input".into() });
        }
        let hidden = x_shape[2].as_const().unwrap();
        let head_dim = hidden / num_heads;
        x.unflatten(-1, &[num_heads as isize, head_dim as isize])?
    } else {
        x.clone()
    };

    let work_shape = x_work.shape()?;
    let head_size = work_shape.last().unwrap().as_const().unwrap();
    let rot_dim = if rotary_embedding_dim > 0 { rotary_embedding_dim } else { head_size };

    // Split into x_rotate and x_pass
    let (x_rotate, x_pass) = if rot_dim < head_size {
        let parts = x_work.split(&[rot_dim, head_size - rot_dim], -1)?;
        (parts[0].clone(), Some(parts[1].clone()))
    } else {
        (x_work.clone(), None)
    };

    // Lookup cos/sin from cache
    let (cos, sin) = if let Some(pos_ids) = position_ids {
        // cache is [max_seq_len, D/2]; index with position_ids [B, S] → [B, S, D/2]
        (cos_cache.embedding(pos_ids)?, sin_cache.embedding(pos_ids)?)
    } else {
        // cache is already [B, S, D/2] (pre-indexed); use directly
        (cos_cache.clone(), sin_cache.clone())
    };

    // Slice to rot_dim/2 if cache has more columns than needed
    let half_rot = rot_dim / 2;
    let cos = slice_last_dim_if_needed(&cos, half_rot)?;
    let sin = slice_last_dim_if_needed(&sin, half_rot)?;

    // Unsqueeze for head dimension broadcast: [B, S, D/2] → [B, S, 1, D/2]
    let cos = if cos.ndim()? < x_rotate.ndim()? { cos.try_unsqueeze(-2)? } else { cos };
    let sin = if sin.ndim()? < x_rotate.ndim()? { sin.try_unsqueeze(-2)? } else { sin };

    let x_rotated = x_rotate.apply_rotary_emb(&cos, &sin, interleaved)?;

    // Concat with x_pass
    let output = if let Some(pass) = x_pass { Tensor::cat(&[&x_rotated, &pass], -1)? } else { x_rotated };

    // Restore original shape
    let output = if x_ndim == 3 {
        let out_shape = output.shape()?;
        let batch = out_shape[0].as_const().unwrap() as isize;
        let seq = out_shape[1].as_const().unwrap() as isize;
        output.try_reshape(&[batch, seq, -1])?
    } else {
        // [B, S, H, D] -> [B, H, S, D]
        output.try_permute(&[0, 2, 1, 3])?
    };

    Ok(vec![output])
}

/// Slice last dimension to `target` if it's larger, otherwise return as-is.
fn slice_last_dim_if_needed(t: &Tensor, target: usize) -> Result<Tensor> {
    let shape = t.shape()?;
    let last = shape[shape.len() - 1].as_const().unwrap();
    if last > target {
        let parts = t.split(&[target, last - target], -1)?;
        Ok(parts[0].clone())
    } else {
        Ok(t.clone())
    }
}

/// Microsoft contrib Attention: packed QKV projection, mask handling, SDPA.
pub(crate) fn op_attention_contrib(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    let x = inp(inputs, 0);
    let weights = inp(inputs, 1);
    let bias = inputs.get(2).and_then(|o| o.as_ref());
    let mask_index = inputs.get(3).and_then(|o| o.as_ref());
    let past = inputs.get(4).and_then(|o| o.as_ref());

    let num_heads = get_attr_int(node, "num_heads", 0) as usize;
    if num_heads == 0 {
        return Err(Error::IrConstruction { details: "num_heads is required for Attention".into() });
    }
    let mask_filter_value = get_attr_float(node, "mask_filter_value", -10000.0) as f64;
    let scale_attr = get_attr_float(node, "scale", 0.0);
    let unidirectional = get_attr_int(node, "unidirectional", 0) != 0;

    let qkv_hidden_sizes = get_attr_ints(node, "qkv_hidden_sizes");
    let w_shape = weights.shape()?;
    let total_hidden = w_shape[1].as_const().unwrap();
    let (q_hidden, k_hidden, v_hidden) = if qkv_hidden_sizes.is_empty() {
        let h = total_hidden / 3;
        (h, h, h)
    } else {
        (qkv_hidden_sizes[0] as usize, qkv_hidden_sizes[1] as usize, qkv_hidden_sizes[2] as usize)
    };

    let q_head_dim = q_hidden / num_heads;
    let scale_val = if scale_attr != 0.0 { scale_attr as f64 } else { 1.0 / (q_head_dim as f64).sqrt() };

    // QKV projection: ONNX weight is [input_hidden, 3*hidden], NOT [out, in]
    let mut qkv = x.matmul(weights)?;
    if let Some(b) = bias {
        qkv = qkv.try_add(b)?;
    }

    // Split into Q, K, V
    let parts = qkv.split(&[q_hidden, k_hidden, v_hidden], -1)?;
    let x_shape = x.shape()?;
    let batch = x_shape[0].as_const().unwrap() as isize;
    let seq_len = x_shape[1].as_const().unwrap();

    // Reshape [B, S, hidden] -> [B, H, S, D]
    let q = parts[0]
        .try_reshape(&[batch, seq_len as isize, num_heads as isize, q_head_dim as isize])?
        .try_permute(&[0, 2, 1, 3])?;
    let k_head_dim = k_hidden / num_heads;
    let v_head_dim = v_hidden / num_heads;
    let mut k = parts[1]
        .try_reshape(&[batch, seq_len as isize, num_heads as isize, k_head_dim as isize])?
        .try_permute(&[0, 2, 1, 3])?;
    let mut v = parts[2]
        .try_reshape(&[batch, seq_len as isize, num_heads as isize, v_head_dim as isize])?
        .try_permute(&[0, 2, 1, 3])?;

    // Past KV
    let has_past = past.is_some();
    if let Some(past_kv) = past {
        let past_parts = past_kv.split(&[1, 1], 0)?;
        let pk = past_parts[0].try_squeeze(Some(0))?;
        let pv = past_parts[1].try_squeeze(Some(0))?;
        k = Tensor::cat(&[&pk, &k], -2)?;
        v = Tensor::cat(&[&pv, &v], -2)?;
    }

    let k_shape = k.shape()?;
    let total_seq = k_shape[k_shape.len() - 2].as_const().unwrap();

    // Build attention mask
    let q_dtype = q.uop().dtype();
    let mut attn_mask: Option<Tensor> = None;

    if let Some(mi) = mask_index {
        let mi_shape = mi.shape()?;
        let mi_ndim = mi_shape.len();
        if mi_ndim > 1 {
            // nD mask: broadcast to [B, 1, Sq, Sk] or similar
            let mask_dtype = mi.uop().dtype();
            if mask_dtype == DType::Bool {
                let filter = Tensor::const_(mask_filter_value, q_dtype.clone());
                let zero = Tensor::const_(0.0f64, q_dtype.clone());
                attn_mask = Some(zero.where_(mi, &filter)?);
            } else {
                attn_mask = Some(mi.clone());
            }
        } else {
            // 1D mask: per-sample end positions
            let mi_len = mi_shape[0].as_const().unwrap();
            if mi_len == batch as usize {
                // mask_index[b] = end position for sample b
                let range = Tensor::arange(total_seq as i64, None, None)?.try_reshape(&[1, total_seq as isize])?;
                let ends = mi.try_reshape(&[batch, 1])?;
                let mask = range.try_lt(&ends)?;
                let filter = Tensor::const_(mask_filter_value, q_dtype.clone());
                let zero = Tensor::const_(0.0f64, q_dtype.clone());
                let additive = zero.where_(&mask, &filter)?;
                attn_mask = Some(additive.try_reshape(&[batch, 1, 1, total_seq as isize])?);
            } else if mi_len == 2 * batch as usize {
                // [end_0..end_B, start_0..start_B]
                let end_parts = mi.split(&[batch as usize, batch as usize], 0)?;
                let ends = end_parts[0].try_reshape(&[batch, 1])?;
                let starts = end_parts[1].try_reshape(&[batch, 1])?;
                let range = Tensor::arange(total_seq as i64, None, None)?.try_reshape(&[1, total_seq as isize])?;
                let mask_end = range.try_lt(&ends)?;
                let mask_start = range.try_ge(&starts)?;
                // Combined: position >= start AND position < end
                let combined = mask_end.try_mul(&mask_start)?;
                let filter = Tensor::const_(mask_filter_value, q_dtype.clone());
                let zero = Tensor::const_(0.0f64, q_dtype.clone());
                let additive = zero.where_(&combined, &filter)?;
                attn_mask = Some(additive.try_reshape(&[batch, 1, 1, total_seq as isize])?);
            }
        }
    }

    // Unidirectional causal mask
    if unidirectional {
        let causal =
            Tensor::full(&[seq_len, total_seq], true, DType::Bool)?.tril((total_seq as i64) - (seq_len as i64))?;
        let filter = Tensor::const_(mask_filter_value, q_dtype.clone());
        let zero = Tensor::const_(0.0f64, q_dtype.clone());
        let causal_additive = zero.where_(&causal, &filter)?;
        attn_mask = Some(match attn_mask {
            Some(existing) => existing.try_add(&causal_additive)?,
            None => causal_additive,
        });
    }

    // Attention computation
    let output = q
        .scaled_dot_product_attention()
        .key(&k)
        .value(&v)
        .maybe_attn_mask(attn_mask.as_ref())
        .scale(scale_val)
        .call()?;

    // Reshape [B, H, S, D] -> [B, S, H*D]
    let output = output.try_permute(&[0, 2, 1, 3])?.try_reshape(&[batch, seq_len as isize, -1])?;

    let present = if has_past || past.is_some() { Tensor::stack(&[&k, &v], 0)? } else { Tensor::from_slice([0.0f32]) };

    Ok(vec![output, present])
}
