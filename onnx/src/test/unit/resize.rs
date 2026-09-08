//! `Resize` in `nearest` mode: the source index of every output position is a
//! host constant, so the results below must be bit-exact on every device —
//! including those without `double`, where a device-side `f32` transform lands
//! on the wrong side of an exact half-way coordinate.

use crate::test::helpers::*;
use ndarray::Array4;

fn resize(
    x: Tensor,
    scales: Option<&[f32]>,
    sizes: Option<&[i64]>,
    roi: Option<&[f32]>,
    attrs: &[AttributeProto],
) -> Tensor {
    let node = NodeProto { attribute: attrs.to_vec(), ..Default::default() };
    let inputs =
        vec![Some(x), roi.map(Tensor::from_slice), scales.map(Tensor::from_slice), sizes.map(Tensor::from_slice)];
    let outputs = OpRegistry::new().dispatch_multi("Resize", "", &inputs, &node, i64::MAX).unwrap();
    outputs.into_iter().next().unwrap()
}

fn values(t: Tensor, config: &PrepareConfig) -> Vec<f32> {
    let t = t.contiguous();
    t.realize_with(config).unwrap();
    t.as_vec::<f32>().unwrap()
}

fn dims(t: &Tensor) -> Vec<usize> {
    t.dims().unwrap()
}

svod_tensor::codegen_tests! {
    /// 2x2 → 7x8 with `half_pixel`: output row 3 maps to input coordinate
    /// exactly 0.5, so `round_prefer_floor` must yield row 0. In f32 the
    /// transform gives 0.50000006 and picks row 1 instead.
    fn test_resize_nearest_half_pixel_tie(config) {
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap());
        let y = resize(x, None, Some(&[1, 1, 7, 8]), None, &[]);
        assert_eq!(dims(&y), [1, 1, 7, 8]);
        let mut expected: Vec<f32> = Vec::new();
        for row in [1.0f32, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0] {
            expected.extend([row, row, row, row, row + 1.0, row + 1.0, row + 1.0, row + 1.0]);
        }
        assert_eq!(values(y, &config), expected);
    }

    /// `axes` selects and *orders* the resized dims: scales apply to dim 3 then
    /// dim 2. `asymmetric` + `ceil` also exercises the non-half-pixel path.
    fn test_resize_nearest_axes_scales_ceil(config) {
        let x = Tensor::from_ndarray(
            &Array4::from_shape_vec((1, 1, 4, 4), (1..=16).map(|v| v as f32).collect()).unwrap(),
        );
        let attrs = [
            make_attr_string("mode", "nearest"),
            make_attr_string("coordinate_transformation_mode", "asymmetric"),
            make_attr_string("nearest_mode", "ceil"),
            make_attr_ints("axes", &[3, 2]),
        ];
        let y = resize(x, Some(&[0.6, 1.5]), None, None, &attrs);
        assert_eq!(dims(&y), [1, 1, 6, 2]);
        #[rustfmt::skip]
        let expected = vec![
            1.0f32, 3.0,
            5.0, 7.0,
            9.0, 11.0,
            9.0, 11.0,
            13.0, 15.0,
            13.0, 15.0,
        ];
        assert_eq!(values(y, &config), expected);
    }

    /// `tf_crop_and_resize`: positions whose source coordinate leaves the ROI
    /// read `extrapolation_value` rather than the clamped edge pixel.
    fn test_resize_nearest_tf_crop_extrapolation(config) {
        let x = Tensor::from_ndarray(
            &Array4::from_shape_vec((1, 1, 4, 4), (1..=16).map(|v| v as f32).collect()).unwrap(),
        );
        let attrs = [
            make_attr_string("mode", "nearest"),
            make_attr_string("coordinate_transformation_mode", "tf_crop_and_resize"),
            make_attr_float("extrapolation_value", 10.0),
        ];
        let roi = [0.0f32, 0.0, 0.4, 0.6, 1.0, 1.0, 1.2, 1.2];
        let y = resize(x, None, Some(&[1, 1, 3, 3]), Some(&roi), &attrs);
        assert_eq!(dims(&y), [1, 1, 3, 3]);
        #[rustfmt::skip]
        let expected = vec![
            7.0f32, 8.0, 10.0,
            11.0, 12.0, 10.0,
            10.0, 10.0, 10.0,
        ];
        assert_eq!(values(y, &config), expected);
    }

    /// `keep_aspect_ratio_policy` derives one shared scale, so both spatial dims
    /// end up 7 wide even though `sizes` asked for 7x8.
    fn test_resize_nearest_not_larger(config) {
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap());
        let attrs = [
            make_attr_string("keep_aspect_ratio_policy", "not_larger"),
            make_attr_ints("axes", &[2, 3]),
        ];
        let y = resize(x, None, Some(&[7, 8]), None, &attrs);
        assert_eq!(dims(&y), [1, 1, 7, 7]);
        let mut expected: Vec<f32> = Vec::new();
        for row in [1.0f32, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0] {
            expected.extend([row, row, row, row, row + 1.0, row + 1.0, row + 1.0]);
        }
        assert_eq!(values(y, &config), expected);
    }

    /// A scale of exactly 1 is the identity: nothing is gathered, so a
    /// non-spatial dim never has to be concrete.
    fn test_resize_nearest_identity_scales(config) {
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap());
        let y = resize(x, Some(&[1.0, 1.0, 1.0, 1.0]), None, None, &[]);
        assert_eq!(dims(&y), [1, 1, 2, 2]);
        assert_eq!(values(y, &config), vec![1.0f32, 2.0, 3.0, 4.0]);
    }
}
