use crate::bgem3::{BgeM3, BgeRerankerV2M3, EncodeOpts};
use crate::state::HasStateDict;
use crate::xlm_roberta::XlmRobertaConfig;
use svod_dtype::DType;

fn tiny_cfg() -> XlmRobertaConfig {
    XlmRobertaConfig {
        vocab_size: 100,
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 64,
        max_position_embeddings: 20,
        type_vocab_size: 1,
        layer_norm_eps: 1e-5,
        pad_token_id: 1,
        dtype: DType::Float32,
        max_batch_size: 1,
    }
}

#[test]
fn bgem3_encode_dense_shape() {
    let model = BgeM3::empty(tiny_cfg());
    let ids = svod_tensor::Tensor::from_slice([0i64, 10, 20, 2]).try_reshape([1isize, 4]).unwrap();
    let mask = svod_tensor::Tensor::from_slice([1i64, 1, 1, 1]).try_reshape([1isize, 4]).unwrap();
    let dense = model.encode_dense(&ids, &mask).unwrap();
    dense.realize().unwrap();
    let s = dense.dims().unwrap();
    assert_eq!(s[0], 1);
    assert_eq!(s[1], 32);
}

#[test]
fn bgem3_encode_all_modalities() {
    let model = BgeM3::empty(tiny_cfg());
    let ids = svod_tensor::Tensor::from_slice([0i64, 10, 20, 2]).try_reshape([1isize, 4]).unwrap();
    let mask = svod_tensor::Tensor::from_slice([1i64, 1, 1, 1]).try_reshape([1isize, 4]).unwrap();
    let out = model.encode(&ids, &mask, EncodeOpts::all()).unwrap();
    assert!(out.dense_vecs.is_some());
    assert!(out.sparse_vecs.is_some());
    assert!(out.colbert_vecs.is_some());
}

#[test]
fn reranker_forward_shape() {
    let model = BgeRerankerV2M3::empty(tiny_cfg());
    let ids = svod_tensor::Tensor::from_slice([0i64, 10, 20, 2, 1]).try_reshape([1isize, 5]).unwrap();
    let mask = svod_tensor::Tensor::from_slice([1i64, 1, 1, 1, 0]).try_reshape([1isize, 5]).unwrap();
    let out = model.forward(&ids, Some(&mask)).unwrap();
    out.realize().unwrap();
    let s = out.dims().unwrap();
    assert_eq!(s[0], 1);
    assert_eq!(s[1], 1);
}

#[test]
fn reranker_state_dict_round_trip() {
    let model = BgeRerankerV2M3::empty(tiny_cfg());
    let sd = model.state_dict("");
    assert!(sd.contains_key("classifier.dense.weight"));
    assert!(sd.contains_key("classifier.out_proj.weight"));
    assert!(sd.contains_key("embeddings.word_embeddings.weight"));
    let mut model2 = BgeRerankerV2M3::empty(tiny_cfg());
    model2.load_state_dict(&sd, "").unwrap();
    let mut k1: Vec<String> = sd.keys().cloned().collect();
    let mut k2: Vec<String> = model2.state_dict("").keys().cloned().collect();
    k1.sort();
    k2.sort();
    assert_eq!(k1, k2);
}
