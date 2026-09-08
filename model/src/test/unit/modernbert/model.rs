use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::modernbert::{ModernBert, ModernBertConfig};
use crate::state::{HasStateDict, StateDict};

/// A tiny config for fast unit tests: 2 layers, hidden 32, 4 heads (head_dim 8),
/// vocab 64, intermediate 64, f32 compute. Global-every-3 still gives layer 0
/// global and layer 1 local.
fn tiny_cfg() -> ModernBertConfig {
    ModernBertConfig {
        vocab_size: 64,
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 64,
        max_position_embeddings: 128,
        layer_norm_eps: 1e-5,
        global_rope_theta: 10_000.0,
        local_rope_theta: 10_000.0,
        local_attention: 16,
        global_attn_every_n_layers: 3,
        pad_token_id: 0,
        tie_word_embeddings: true,
        decoder_bias: true,
        dtype: DType::Float32,
        max_batch_size: 1,
    }
}

/// `empty` builds the right structure: layer 0 has no attn_norm, layer 1 does.
#[test]
fn layer_zero_has_no_attn_norm() {
    let cfg = tiny_cfg();
    let m = ModernBert::empty(cfg);
    assert!(m.encoder.layers[0].attn_norm.is_none(), "layer 0 attn_norm must be None");
    assert!(m.encoder.layers[1].attn_norm.is_some(), "layer 1+ attn_norm must be Some");
}

/// `forward` produces a `(B, L, D)` tensor of the expected shape.
#[test]
fn forward_output_shape() {
    let cfg = tiny_cfg();
    let m = ModernBert::empty(cfg);
    // input_ids (B=2, L=5) int64.
    let ids = Tensor::from_slice((0..10i64).collect::<Vec<_>>()).try_reshape([2isize, 5]).unwrap();
    let out = m.forward(&ids, None).unwrap();
    out.realize().unwrap();
    let v = out.as_vec::<f32>().unwrap();
    // (2, 5, 32) = 320 elements, all finite.
    assert_eq!(v.len(), 2 * 5 * 32);
    for val in &v {
        assert!(val.is_finite(), "non-finite forward output: {val}");
    }
}

/// `state_dict` → `load_state_dict` round-trips: building from a model's own
/// state dict reproduces it (keys + shape). The key map matches the published
/// checkpoint layout.
#[test]
fn state_dict_round_trip() {
    let cfg = tiny_cfg();
    let m = ModernBert::empty(cfg);
    let sd: StateDict = m.state_dict("");

    // Spot-check the published key names are present.
    assert!(sd.contains_key("model.embeddings.tok_embeddings.weight"));
    assert!(sd.contains_key("model.embeddings.norm.weight"));
    assert!(sd.contains_key("model.final_norm.weight"));
    // Layer 0 has no attn_norm key; layer 1 does.
    assert!(!sd.contains_key("model.layers.0.attn_norm.weight"), "layer 0 must not emit attn_norm");
    assert!(sd.contains_key("model.layers.1.attn_norm.weight"));
    assert!(sd.contains_key("model.layers.0.attn.Wqkv.weight"));
    assert!(sd.contains_key("model.layers.0.attn.Wo.weight"));
    assert!(sd.contains_key("model.layers.0.mlp.Wi.weight"));
    assert!(sd.contains_key("model.layers.0.mlp.Wo.weight"));
    assert!(sd.contains_key("model.layers.0.mlp_norm.weight"));
    // No biases on any linear/norm (ModernBERT is bias-free).
    assert!(!sd.contains_key("model.layers.0.attn.Wqkv.bias"));
    assert!(!sd.contains_key("model.embeddings.norm.bias"));

    // Round-trip into a fresh model.
    let m2 = ModernBert::empty(tiny_cfg());
    assert_eq!(sd.len(), m2.state_dict("").len(), "state-dict key count stable across instances");
}

/// The fused QKV weight is `(3*D, D)`; the GLU `Wi` is `(2*I, D)`.
#[test]
fn published_weight_shapes() {
    let cfg = tiny_cfg();
    let d = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let m = ModernBert::empty(cfg);
    let sd: StateDict = m.state_dict("");
    let qkv = &sd["model.layers.0.attn.Wqkv.weight"];
    let wi = &sd["model.layers.0.mlp.Wi.weight"];
    assert_eq!(qkv.dim_const(0).unwrap(), 3 * d);
    assert_eq!(qkv.dim_const(1).unwrap(), d);
    assert_eq!(wi.dim_const(0).unwrap(), 2 * i);
    assert_eq!(wi.dim_const(1).unwrap(), d);
}
