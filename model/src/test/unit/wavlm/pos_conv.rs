use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::state::{HasStateDict, StateDict};
use crate::wavlm::ConvolutionalPositionalEmbedding;

/// `(1, T, 1024) → (1, T, 1024)` after the trim-1 + GELU + add path. We only
/// check shape preservation here; numerical parity is the parity-test's job.
#[test]
fn pos_conv_preserves_shape() {
    let pe = ConvolutionalPositionalEmbedding::empty(1024, 128, 16);
    let t = 799;
    let x = Tensor::zeros(&[1, t, 1024], DType::Float32);
    let y = pe.forward(&x).expect("symbolic forward");
    let shape = y.dims().unwrap();
    assert_eq!(shape, vec![1, t, 1024]);
}

/// Weight-norm reconstruction: build a `(g, v)` pair as a fake state-dict
/// entry, run `load_state_dict`, and confirm the reconstructed `weight`
/// matches `g * v / ||v||_dim01`.
#[test]
#[allow(clippy::needless_range_loop)]
fn pos_conv_weight_norm_reconstruction() {
    let embed_dim = 16;
    let in_per_group = 4; // groups = 4
    let kernel = 8;
    let groups = 4;

    let pe_init = ConvolutionalPositionalEmbedding::empty(embed_dim, kernel, groups);
    // Use the initialized weight as a stand-in `v`; build `g` as all-ones.
    let v = pe_init.weight.clone();
    let _ = v.shape().unwrap(); // ensure shape is usable downstream

    let g = Tensor::ones(&[1, 1, kernel], DType::Float32);
    let bias = pe_init.bias.clone();

    let mut sd = StateDict::new();
    sd.insert("pe.conv.parametrizations.weight.original0".into(), g.clone());
    sd.insert("pe.conv.parametrizations.weight.original1".into(), v.clone());
    sd.insert("pe.conv.bias".into(), bias.clone());

    let mut pe = ConvolutionalPositionalEmbedding::empty(embed_dim, kernel, groups);
    pe.load_state_dict(&sd, "pe").expect("load weight-norm pair");

    // Realize both sides and compare.
    let got = pe.weight.clone();
    got.realize().unwrap();
    let got_vec: Vec<f32> = got.as_vec::<f32>().unwrap();
    let got_shape = got.dims().unwrap();
    assert_eq!(got_shape, vec![embed_dim, in_per_group, kernel]);

    // Compute g * v / ||v||_dim01 manually.
    let v_real = v.clone();
    v_real.realize().unwrap();
    let v_vec: Vec<f32> = v_real.as_vec::<f32>().unwrap();

    // norm[k] = sqrt(sum_{o, ig} v[o, ig, k]^2)
    let mut norm = vec![0f32; kernel];
    for o in 0..embed_dim {
        for ig in 0..in_per_group {
            for k in 0..kernel {
                let idx = (o * in_per_group + ig) * kernel + k;
                norm[k] += v_vec[idx] * v_vec[idx];
            }
        }
    }
    for k in 0..kernel {
        norm[k] = norm[k].sqrt();
    }

    let mut want_vec = vec![0f32; embed_dim * in_per_group * kernel];
    for o in 0..embed_dim {
        for ig in 0..in_per_group {
            for k in 0..kernel {
                let idx = (o * in_per_group + ig) * kernel + k;
                want_vec[idx] = v_vec[idx] / norm[k];
            }
        }
    }

    for (a, b) in got_vec.iter().zip(want_vec.iter()) {
        assert!((a - b).abs() < 1e-5, "weight-norm reconstruction mismatch: {a} vs {b}");
    }
}

/// Loader also accepts a flat `conv.weight` key when the checkpoint was already
/// flattened (e.g. through `nn.utils.remove_weight_norm`).
#[test]
fn pos_conv_flat_weight_load() {
    let pe_init = ConvolutionalPositionalEmbedding::empty(16, 8, 4);
    let w = pe_init.weight.clone();
    let b = pe_init.bias.clone();

    let mut sd = StateDict::new();
    sd.insert("pe.conv.weight".into(), w);
    sd.insert("pe.conv.bias".into(), b);

    let mut pe = ConvolutionalPositionalEmbedding::empty(16, 8, 4);
    pe.load_state_dict(&sd, "pe").expect("flat weight load");
}
