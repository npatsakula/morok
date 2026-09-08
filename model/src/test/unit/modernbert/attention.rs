use crate::modernbert::{EncoderLayer, ModernBertAttention, ModernBertConfig};

fn test_cfg() -> ModernBertConfig {
    ModernBertConfig {
        hidden_size: 768,
        num_attention_heads: 12,
        local_attention: 128,
        global_attn_every_n_layers: 3,
        ..ModernBertConfig::default()
    }
}

/// Global layers carry `window = None`; local layers carry the configured
/// sliding window. This is the structural contract the encoder relies on.
#[test]
fn window_set_per_layer_kind() {
    let cfg = test_cfg();
    // Layer 0, 3 are global (id % 3 == 0) → no window.
    let l0 = EncoderLayer::empty(&cfg, 0);
    assert!(l0.attention.window.is_none(), "layer 0 (global) must not have a window");
    let l3 = EncoderLayer::empty(&cfg, 3);
    assert!(l3.attention.window.is_none(), "layer 3 (global) must not have a window");
    // Layer 1, 2 are local → window (64, 64).
    let l1 = EncoderLayer::empty(&cfg, 1);
    assert_eq!(l1.attention.window, Some((64, 64)), "layer 1 (local) window");
    let l2 = EncoderLayer::empty(&cfg, 2);
    assert_eq!(l2.attention.window, Some((64, 64)), "layer 2 (local) window");
}

/// The fused QKV weight has shape `(3*hidden, hidden)`; output proj `(hidden, hidden)`.
#[test]
fn attention_weight_shapes() {
    let cfg = test_cfg();
    let attn = ModernBertAttention::empty(cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim(), None, cfg.dtype);
    let qkv = attn.qkv_weight.dims().unwrap();
    let out = attn.out_weight.dims().unwrap();
    assert_eq!(qkv[0], 3 * cfg.hidden_size);
    assert_eq!(qkv[1], cfg.hidden_size);
    assert_eq!(out[0], cfg.hidden_size);
    assert_eq!(out[1], cfg.hidden_size);
}
