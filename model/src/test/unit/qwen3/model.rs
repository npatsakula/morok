use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::qwen3::{Qwen3Config, Qwen3Embedding, Qwen3Model};
use crate::state::HasStateDict;

fn tiny_cfg() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 100,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        intermediate_size: 128,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_bias: false,
        tie_word_embeddings: true,
        pad_token_id: 0,
        dtype: DType::Float32,
        max_batch_size: 1,
    }
}

#[test]
fn forward_output_shape() {
    let model = Qwen3Model::empty(tiny_cfg());
    let ids = Tensor::from_slice([0i64, 1, 2, 3, 4, 5, 6, 7]).try_reshape([1isize, 8]).unwrap();

    let out = model.forward(&ids, None).unwrap();
    out.realize().unwrap();
    let s = out.dims().unwrap();
    assert_eq!(s[0], 1);
    assert_eq!(s[1], 8);
    assert_eq!(s[2], 64);

    let v = out.as_vec::<f32>().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn embedding_output_shape() {
    let emb = Qwen3Embedding::empty(tiny_cfg());
    let ids = Tensor::from_slice([0i64, 1, 2, 3, 4, 5, 6, 7]).try_reshape([1isize, 8]).unwrap();
    let mask = Tensor::from_slice([1i64, 1, 1, 1, 1, 1, 1, 1]).try_reshape([1isize, 8]).unwrap();

    let out = emb.encode(&ids, &mask).unwrap();
    out.realize().unwrap();
    let s = out.dims().unwrap();
    assert_eq!(s[0], 1);
    assert_eq!(s[1], 64);

    let v = out.as_vec::<f32>().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn state_dict_round_trip() {
    let model = Qwen3Model::empty(tiny_cfg());
    let sd = model.state_dict("");

    assert!(sd.contains_key("embed_tokens.weight"));
    assert!(sd.contains_key("norm.weight"));
    assert!(sd.contains_key("layers.0.input_layernorm.weight"));
    assert!(sd.contains_key("layers.0.self_attn.q_proj.weight"));
    assert!(sd.contains_key("layers.0.self_attn.q_norm.weight"));
    assert!(sd.contains_key("layers.0.mlp.gate_proj.weight"));

    let mut model2 = Qwen3Model::empty(tiny_cfg());
    model2.load_state_dict(&sd, "").unwrap();

    let mut k1: Vec<String> = sd.keys().cloned().collect();
    let mut k2: Vec<String> = model2.state_dict("").keys().cloned().collect();
    k1.sort();
    k2.sort();
    assert_eq!(k1, k2);
}

#[test]
fn key_count_matches_checkpoint_layout() {
    let cfg = qwen3_0_6b_structural();
    let model = Qwen3Model::empty(cfg);
    let sd = model.state_dict("");
    // 1 embed + 28 layers × 13 + 1 norm = 366
    // 13 per layer: input_layernorm, q/k/v/o_proj, q/k_norm, post_attention_layernorm,
    //               gate/up/down_proj = 3+2+2+1+3 = 11... wait: input_ln(1) + q(1)+k(1)+v(1)+o(1) + qn(1)+kn(1) + post_ln(1) + gate(1)+up(1)+down(1) = 11
    // 1 + 28*11 + 1 = 310
    let expected = 1 + 28 * 11 + 1;
    assert_eq!(sd.len(), expected, "expected {expected} keys, got {}", sd.len());
}

fn qwen3_0_6b_structural() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 151_669,
        hidden_size: 1024,
        num_hidden_layers: 28,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        head_dim: 128,
        intermediate_size: 3072,
        max_position_embeddings: 32_768,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        attention_bias: false,
        tie_word_embeddings: true,
        pad_token_id: 151_643,
        dtype: DType::Float32,
        max_batch_size: 1,
    }
}
