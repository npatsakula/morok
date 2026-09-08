use svod_dtype::DType;
use svod_tensor::Tensor;

use svod_tensor::nn::{Module, StateDict};

use crate::qwen3::{Qwen3Config, Qwen3Embedding, Qwen3Model};

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

/// Every key a 2-layer tiny backbone emits, in the published `Qwen3Model`
/// naming — captured from the hand-written `HasStateDict` impl this model was
/// migrated from. Drift here is a checkpoint-compatibility break.
fn expected_keys() -> Vec<String> {
    let mut keys = vec!["embed_tokens.weight".to_string(), "norm.weight".to_string()];
    for i in 0..2 {
        for k in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            keys.push(format!("layers.{i}.{k}"));
        }
    }
    keys.sort();
    keys
}

fn sorted_keys(sd: &StateDict) -> Vec<String> {
    let mut keys: Vec<String> = sd.keys().cloned().collect();
    keys.sort();
    keys
}

/// The emitted key set is exactly the published layout, at the root and nested
/// under a prefix (which must not grow a leading dot).
#[test]
fn state_dict_keys_match_published_layout() {
    let model = Qwen3Model::empty(tiny_cfg());
    let want = expected_keys();
    assert_eq!(sorted_keys(&model.state_dict("")), want);

    let want_nested: Vec<String> = want.iter().map(|k| format!("m.{k}")).collect();
    assert_eq!(sorted_keys(&model.state_dict("m")), want_nested);
}

/// The embedding wrapper is transparent: it emits the backbone's keys verbatim.
#[test]
fn embedding_state_dict_is_transparent() {
    let emb = Qwen3Embedding::empty(tiny_cfg());
    assert_eq!(sorted_keys(&emb.state_dict("")), expected_keys());
}

#[test]
fn state_dict_round_trip() {
    let model = Qwen3Model::empty(tiny_cfg());
    let sd = model.state_dict("");

    let mut model2 = Qwen3Model::empty(tiny_cfg());
    model2.load_state_dict(&sd, "").unwrap();
    assert_eq!(sorted_keys(&sd), sorted_keys(&model2.state_dict("")));
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
