//! Tests: layers.

use std::collections::BTreeSet;

use svod_dtype::DType;
use test_case::test_case;

use crate::Tensor;
use crate::nn::{
    BatchNorm2d, Conv1d, Conv2d, ConvTranspose2d, Embedding, Layer, LayerNorm, Linear, Module, Relu, RmsNorm,
};

/// A reproducible ramp, so a layer and the builder it wraps see identical data.
fn ramp(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin()).collect();
    let dims: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
    Tensor::from_slice(&data).try_reshape(&dims).expect("ramp reshape")
}

fn values(x: &Tensor) -> Vec<f32> {
    x.contiguous().to_vec::<f32>().expect("realize")
}

fn shape(x: &Tensor) -> Vec<usize> {
    x.shape().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
}

fn keys<M: Module>(m: &M, prefix: &str) -> BTreeSet<String> {
    m.state_dict(prefix).into_keys().collect()
}

fn expect<'a>(names: impl IntoIterator<Item = &'a str>) -> BTreeSet<String> {
    names.into_iter().map(str::to_string).collect()
}

// =========================================================================
// forward == the builder call the layer wraps
// =========================================================================

#[test]
fn linear_forward_matches_the_builder() {
    let (x, w, b) = (ramp(&[2, 4]), ramp(&[3, 4]), ramp(&[3]));
    let layer = Linear::new(w.clone(), Some(b.clone()));

    assert_eq!(values(&layer.forward(&x).unwrap()), values(&x.linear().weight(&w).bias(&b).call().unwrap()));
}

#[test]
fn linear_without_bias_omits_it_from_the_call() {
    let (x, w) = (ramp(&[2, 4]), ramp(&[3, 4]));
    let layer = Linear::new(w.clone(), None);

    assert_eq!(values(&layer.forward(&x).unwrap()), values(&x.linear().weight(&w).call().unwrap()));
}

#[test]
fn conv1d_forward_matches_the_builder() {
    let (x, w, b) = (ramp(&[1, 4, 7]), ramp(&[6, 2, 3]), ramp(&[6]));
    let layer =
        Conv1d::new(w.clone(), Some(b.clone())).with_stride(2).with_padding((1, 2)).with_dilation(2).with_groups(2);

    let expected = x.conv1d().weight(&w).bias(&b).stride(2).padding((1, 2)).dilation(2).groups(2).call().unwrap();
    assert_eq!(values(&layer.forward(&x).unwrap()), values(&expected));
}

#[test]
fn conv2d_forward_matches_the_builder() {
    let (x, w, b) = (ramp(&[1, 4, 6, 5]), ramp(&[6, 2, 3, 3]), ramp(&[6]));
    let layer = Conv2d::new(w.clone(), Some(b.clone()))
        .with_stride((2, 1))
        .with_padding(((1, 0), (0, 1)))
        .with_dilation((2, 1))
        .with_groups(2);

    let expected = x
        .conv2d()
        .weight(&w)
        .bias(&b)
        .groups(2)
        .stride(&[2, 1])
        .dilation(&[2, 1])
        .padding(&[(1, 0), (0, 1)])
        .call()
        .unwrap();
    assert_eq!(values(&layer.forward(&x).unwrap()), values(&expected));
}

#[test]
fn conv2d_without_bias_omits_it_from_the_call() {
    let (x, w) = (ramp(&[1, 2, 4, 4]), ramp(&[3, 2, 3, 3]));
    let layer = Conv2d::new(w.clone(), None);

    assert_eq!(values(&layer.forward(&x).unwrap()), values(&x.conv2d().weight(&w).call().unwrap()));
}

#[test]
fn conv_transpose2d_forward_matches_the_builder() {
    let (x, w, b) = (ramp(&[1, 2, 3, 3]), ramp(&[2, 3, 2, 2]), ramp(&[3]));
    let layer = ConvTranspose2d::new(w.clone(), Some(b.clone()))
        .with_stride((2, 2))
        .with_padding(((1, 1), (0, 0)))
        .with_output_padding((1, 0));

    let expected = x
        .conv_transpose2d()
        .weight(&w)
        .bias(&b)
        .groups(1)
        .stride(&[2, 2])
        .dilation(&[1, 1])
        .padding(&[(1, 1), (0, 0)])
        .output_padding(&[1, 0])
        .call()
        .unwrap();
    assert_eq!(values(&layer.forward(&x).unwrap()), values(&expected));
}

#[test]
fn batchnorm2d_forward_matches_the_builder() {
    let x = ramp(&[2, 3, 2, 2]);
    let (w, b, mean) = (ramp(&[3]), ramp(&[3]), ramp(&[3]));
    let var = Tensor::from_slice([0.5f32, 1.5, 2.5]);
    let layer = BatchNorm2d::new(w.clone(), b.clone(), mean.clone(), var.clone(), 1e-3);

    let expected = x.batchnorm().scale(&w).bias(&b).mean(&mean).var(&var).eps(1e-3).call().unwrap();
    assert_eq!(values(&layer.forward(&x).unwrap()), values(&expected));
}

#[test_case(true ; "with bias")]
#[test_case(false ; "without bias")]
fn layernorm_forward_matches_the_builder(bias: bool) {
    let x = ramp(&[2, 3, 4]);
    let (w, b) = (ramp(&[4]), bias.then(|| ramp(&[4])));
    let layer = LayerNorm::new(w.clone(), b.clone(), 1e-4);

    let expected = x.layernorm_with().axis(-1).eps(1e-4).weight(&w).maybe_bias(b.as_ref()).call().unwrap();
    assert_eq!(values(&layer.forward(&x).unwrap()), values(&expected));
}

#[test]
fn layernorm_axis_widens_the_normalized_region() {
    let x = ramp(&[2, 3, 4]);
    let w = ramp(&[3, 4]);
    let layer = LayerNorm::new(w.clone(), None, 1e-4).with_axis(1);

    let expected = x.layernorm_with().axis(1).eps(1e-4).weight(&w).call().unwrap();
    assert_eq!(values(&layer.forward(&x).unwrap()), values(&expected));
}

#[test]
fn rms_norm_forward_matches_the_builder() {
    let (x, w) = (ramp(&[2, 4]), ramp(&[4]));
    let layer = RmsNorm::new(w.clone(), 1e-6);

    assert_eq!(values(&layer.forward(&x).unwrap()), values(&x.rms_norm_with().eps(1e-6).weight(&w).call().unwrap()));
}

#[test]
fn embedding_forward_looks_rows_up() {
    let table = ramp(&[5, 3]);
    let indices = Tensor::from_slice([3i32, 0, 4]);
    let layer = Embedding::new(table.clone());

    assert_eq!(values(&layer.forward(&indices).unwrap()), values(&table.embedding(&indices).unwrap()));
    assert_eq!(shape(&layer.forward(&indices).unwrap()), vec![3, 3]);
}

// =========================================================================
// State-dict keys are PyTorch's
// =========================================================================

#[test]
fn state_dict_keys_match_pytorch_names() {
    let w = ramp(&[2, 2]);
    let v = ramp(&[2]);

    assert_eq!(keys(&Linear::new(w.clone(), Some(v.clone())), ""), expect(["weight", "bias"]));
    assert_eq!(keys(&Linear::new(w.clone(), None), ""), expect(["weight"]));
    assert_eq!(keys(&Conv1d::new(w.clone(), Some(v.clone())), ""), expect(["weight", "bias"]));
    assert_eq!(keys(&Conv2d::new(w.clone(), None), ""), expect(["weight"]));
    assert_eq!(keys(&ConvTranspose2d::new(w.clone(), Some(v.clone())), ""), expect(["weight", "bias"]));
    assert_eq!(
        keys(&BatchNorm2d::new(v.clone(), v.clone(), v.clone(), v.clone(), 1e-5), ""),
        expect(["weight", "bias", "running_mean", "running_var"])
    );
    assert_eq!(keys(&LayerNorm::new(v.clone(), Some(v.clone()), 1e-5), ""), expect(["weight", "bias"]));
    assert_eq!(keys(&LayerNorm::new(v.clone(), None, 1e-5), ""), expect(["weight"]));
    assert_eq!(keys(&RmsNorm::new(v.clone(), 1e-5), ""), expect(["weight"]));
    assert_eq!(keys(&Embedding::new(w), ""), expect(["weight"]));
}

#[test]
fn a_nested_prefix_dots_every_layer_key() {
    let layer = BatchNorm2d::with_dims(3, 1e-5, DType::Float32);
    assert_eq!(
        keys(&layer, "backbone.0.bn"),
        expect([
            "backbone.0.bn.weight",
            "backbone.0.bn.bias",
            "backbone.0.bn.running_mean",
            "backbone.0.bn.running_var",
        ])
    );
}

#[test]
fn a_layer_round_trips_through_its_own_state_dict() {
    let src = Conv2d::new(ramp(&[3, 2, 3, 3]), Some(ramp(&[3])));
    let mut dst = Conv2d::with_dims(2, 3, (3, 3), true, DType::Float32).with_stride((2, 2));
    dst.load_state_dict(&src.state_dict("c"), "c").unwrap();

    assert_eq!(values(&dst.weight), values(&src.weight));
    assert_eq!(values(dst.bias.as_ref().unwrap()), values(src.bias.as_ref().unwrap()));
    // Hyper-parameters are the receiver's; only weights come from the dict.
    assert_eq!(dst.stride, (2, 2));
}

// =========================================================================
// with_dims
// =========================================================================

#[test_case(DType::Float32 ; "f32")]
#[test_case(DType::Float16 ; "f16")]
fn with_dims_shapes_and_dtypes(dtype: DType) {
    let linear = Linear::with_dims(4, 6, true, dtype.clone());
    assert_eq!(shape(&linear.weight), vec![6, 4]);
    assert_eq!(shape(linear.bias.as_ref().unwrap()), vec![6]);
    assert_eq!(linear.weight.dtype(), dtype);
    assert_eq!(linear.bias.as_ref().unwrap().dtype(), dtype);

    let conv1d = Conv1d::with_dims(2, 5, 3, false, dtype.clone());
    assert_eq!(shape(&conv1d.weight), vec![5, 2, 3]);
    assert!(conv1d.bias.is_none());

    let conv2d = Conv2d::with_dims(2, 5, (3, 1), true, dtype.clone());
    assert_eq!(shape(&conv2d.weight), vec![5, 2, 3, 1]);
    assert_eq!(shape(conv2d.bias.as_ref().unwrap()), vec![5]);

    // Transposed convolution keeps PyTorch's `[in, out, kH, kW]` layout.
    let deconv = ConvTranspose2d::with_dims(2, 5, (2, 2), false, dtype.clone());
    assert_eq!(shape(&deconv.weight), vec![2, 5, 2, 2]);

    let embed = Embedding::with_dims(7, 3, dtype.clone());
    assert_eq!(shape(&embed.weight), vec![7, 3]);
    assert_eq!(embed.weight.dtype(), dtype);
}

#[test]
fn with_dims_defaults_to_the_identity_hyper_parameters() {
    let conv = Conv2d::with_dims(2, 3, (3, 3), false, DType::Float32);
    assert_eq!((conv.stride, conv.dilation, conv.groups), ((1, 1), (1, 1), 1));
    assert_eq!(conv.padding, ((0, 0), (0, 0)));

    let deconv = ConvTranspose2d::with_dims(2, 3, (2, 2), false, DType::Float32);
    assert_eq!(deconv.output_padding, (0, 0));
}

#[test]
fn normalization_with_dims_starts_at_the_identity() {
    let ln = LayerNorm::with_dims(3, true, 1e-5, DType::Float32);
    assert_eq!(values(&ln.weight), vec![1.0; 3]);
    assert_eq!(values(ln.bias.as_ref().unwrap()), vec![0.0; 3]);

    let rms = RmsNorm::with_dims(3, 1e-5, DType::Float32);
    assert_eq!(values(&rms.weight), vec![1.0; 3]);

    let bn = BatchNorm2d::with_dims(3, 1e-5, DType::Float32);
    assert_eq!((values(&bn.weight), values(&bn.bias)), (vec![1.0; 3], vec![0.0; 3]));
    assert_eq!((values(&bn.running_mean), values(&bn.running_var)), (vec![0.0; 3], vec![1.0; 3]));

    // An identity batchnorm passes its input through.
    let x = ramp(&[1, 3, 2, 2]);
    let y = bn.forward(&x).unwrap();
    for (got, want) in values(&y).iter().zip(values(&x)) {
        assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
    }
}

#[test]
fn a_kaiming_weight_stays_inside_its_bound() {
    // bound = √(6 / fan_in), fan_in = 4.
    let linear = Linear::with_dims(4, 6, false, DType::Float32);
    let bound = (6.0f32 / 4.0).sqrt();
    assert!(values(&linear.weight).iter().all(|v| v.abs() <= bound), "outside ±{bound}");
}

// =========================================================================
// Layers as trait objects
// =========================================================================

#[test]
fn layers_compose_through_sequential() {
    let x = ramp(&[2, 4]);
    let fc1 = Linear::with_dims(4, 3, true, DType::Float32);
    let norm = LayerNorm::with_dims(3, true, 1e-5, DType::Float32);
    let fc2 = Linear::new(ramp(&[2, 3]), None);

    let got = x.sequential(&[&fc1, &Relu, &norm, &fc2]).unwrap();

    let staged = fc2.forward(&norm.forward(&Relu.forward(&fc1.forward(&x).unwrap()).unwrap()).unwrap()).unwrap();
    assert_eq!(values(&got), values(&staged));
    assert_eq!(shape(&got), vec![2, 2]);
}
