use crate::state::HasStateDict;
use crate::xlm_roberta::{XlmRobertaConfig, XlmRobertaModel};
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
fn forward_output_shape() {
    let model = XlmRobertaModel::empty(tiny_cfg());
    let ids = svod_tensor::Tensor::from_slice([0i64, 10, 20, 2, 1]).try_reshape([1isize, 5]).unwrap();
    let out = model.forward(&ids, None).unwrap();
    out.realize().unwrap();
    let v = out.as_vec::<f32>().unwrap();
    assert_eq!(v.len(), 5 * 32);
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn state_dict_round_trip() {
    let model = XlmRobertaModel::empty(tiny_cfg());
    let sd = model.state_dict("");
    assert!(sd.contains_key("embeddings.word_embeddings.weight"));
    assert!(sd.contains_key("encoder.layer.0.attention.self.query.weight"));
    let mut model2 = XlmRobertaModel::empty(tiny_cfg());
    model2.load_state_dict(&sd, "").unwrap();
    let mut k1: Vec<String> = sd.keys().cloned().collect();
    let mut k2: Vec<String> = model2.state_dict("").keys().cloned().collect();
    k1.sort();
    k2.sort();
    assert_eq!(k1, k2);
}
