use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::qwen3::{Qwen3Config, Qwen3Reranker};

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
    let mut reranker = Qwen3Reranker::empty(tiny_cfg());
    reranker.yes_loc = 5; // within tiny vocab_size=100
    let ids = Tensor::from_slice([0i64, 1, 2, 3, 4, 5, 6, 7]).try_reshape([1isize, 8]).unwrap();
    let mask = Tensor::from_slice([1i64, 1, 1, 1, 1, 1, 1, 1]).try_reshape([1isize, 8]).unwrap();

    let out = reranker.forward(&ids, &mask).unwrap();
    out.realize().unwrap();
    let s = out.dims().unwrap();
    // (B,) scalar score per row (after sigmoid)
    assert_eq!(s.len(), 1);
    assert_eq!(s[0], 1);

    let v = out.as_vec::<f32>().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
    // sigmoid output must be in (0, 1)
    assert!(v[0] > 0.0 && v[0] < 1.0);
}

#[test]
fn lm_head_tied_to_embeddings() {
    let reranker = Qwen3Reranker::empty(tiny_cfg());
    // The lm_head_weight should have the same shape as embed_tokens.weight
    let lm_shape = reranker.lm_head_weight.dims().unwrap();
    assert_eq!(lm_shape[0], 100); // vocab_size
    assert_eq!(lm_shape[1], 64); // hidden_size
}
