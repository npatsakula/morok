use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::qwen3::{Qwen3Attention, qwen3_embedding_0_6b};

fn tiny_attn() -> Qwen3Attention {
    Qwen3Attention::empty(64, 4, 2, 32, 1e-5, DType::Float32)
}

#[test]
fn gqa_projection_shapes() {
    let attn = tiny_attn();
    let q_shape = attn.q_proj_weight.dims().unwrap();
    let k_shape = attn.k_proj_weight.dims().unwrap();
    let v_shape = attn.v_proj_weight.dims().unwrap();
    let o_shape = attn.o_proj_weight.dims().unwrap();

    // q: (4*32, 64) = (128, 64), k/v: (2*32, 64) = (64, 64), o: (64, 128)
    assert_eq!(q_shape[0], 128);
    assert_eq!(q_shape[1], 64);
    assert_eq!(k_shape[0], 64);
    assert_eq!(v_shape[0], 64);
    assert_eq!(o_shape[0], 64);
    assert_eq!(o_shape[1], 128);
}

#[test]
fn qk_norm_dims() {
    let attn = tiny_attn();
    let qn = attn.q_norm.weight.dims().unwrap();
    let kn = attn.k_norm.weight.dims().unwrap();
    assert_eq!(qn[0], 32);
    assert_eq!(kn[0], 32);
}

#[test]
fn forward_output_shape() {
    let attn = tiny_attn();

    let x = Tensor::from_slice([0.5f32; 512]).try_reshape([1isize, 8, 64]).unwrap();
    let rope = Tensor::rope_table(10000.0, 8, 32, DType::Float32).unwrap();

    let out = attn.forward(&x, &rope, None).unwrap();
    out.realize().unwrap();
    let s = out.dims().unwrap();
    assert_eq!(s[0], 1);
    assert_eq!(s[1], 8);
    assert_eq!(s[2], 64);

    let v = out.as_vec::<f32>().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn published_weight_shapes() {
    let cfg = qwen3_embedding_0_6b();
    let attn = Qwen3Attention::empty(
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.rms_norm_eps,
        DType::BFloat16,
    );
    let q = attn.q_proj_weight.dims().unwrap();
    let k = attn.k_proj_weight.dims().unwrap();
    let o = attn.o_proj_weight.dims().unwrap();
    // q: (16*128, 1024) = (2048, 1024)
    assert_eq!(q[0], 2048);
    assert_eq!(q[1], 1024);
    // k: (8*128, 1024) = (1024, 1024)
    assert_eq!(k[0], 1024);
    assert_eq!(k[1], 1024);
    // o: (1024, 2048)
    assert_eq!(o[0], 1024);
    assert_eq!(o[1], 2048);
}
