//! Phase 3 tests: shape_helpers.

use svod_dtype::DType;
use svod_ir::SInt;
use test_case::test_case;

use crate::nn::{CoordinateTransformMode, ResizeMode};
use crate::test::helpers::{RealizeTestExt, assert_close_f32, test_setup};
use crate::{Tensor, Variable};

// =========================================================================
// Helpers
// =========================================================================

/// `[4 | b, rest...]` filled with ones; `b` is a variable bound to 4.
fn input(symbolic: bool, rest: &[usize]) -> Tensor {
    let head = if symbolic { Variable::new("b", 1, 8).bind(4).unwrap().as_sint() } else { SInt::from(4usize) };
    let dims: Vec<SInt> = std::iter::once(head).chain(rest.iter().map(|&d| SInt::from(d))).collect();
    Tensor::full_dynamic(&dims, 1.0f32, DType::Float32).unwrap()
}

/// Extents with `None` marking a symbolic one.
fn dims_opt(t: &Tensor) -> Vec<Option<usize>> {
    t.shape().unwrap().iter().map(|d| d.as_const()).collect()
}

/// Expected leading extent for `input`.
fn head(symbolic: bool) -> Option<usize> {
    (!symbolic).then_some(4)
}

/// `[shape]` filled with its own row-major flat indices.
fn iota(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let dims: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
    Tensor::arange(0, Some(n as i64), None).unwrap().cast(DType::Float32).try_reshape(&dims).unwrap()
}

// =========================================================================
// Symbolic-dim consistency matrix (no codegen)
// =========================================================================

#[test_case(false; "concrete")]
#[test_case(true; "symbolic_batch")]
fn test_chunk_symbolic_passthrough(sym: bool) {
    let x = input(sym, &[6, 5]);
    let parts = x.chunk(2, 1).unwrap();
    assert_eq!(parts.len(), 2);
    for p in &parts {
        assert_eq!(dims_opt(p), vec![head(sym), Some(3), Some(5)]);
    }
}

#[test_case(false; "concrete")]
#[test_case(true; "symbolic_batch")]
fn test_split_symbolic_passthrough(sym: bool) {
    let x = input(sym, &[6, 5]);
    let parts = x.split(&[2, 4], 1).unwrap();
    assert_eq!(dims_opt(&parts[0]), vec![head(sym), Some(2), Some(5)]);
    assert_eq!(dims_opt(&parts[1]), vec![head(sym), Some(4), Some(5)]);
}

#[test_case(false; "concrete")]
#[test_case(true; "symbolic_batch")]
fn test_triangular_symbolic_passthrough(sym: bool) {
    let x = input(sym, &[3, 5]);
    assert_eq!(dims_opt(&x.triu(1).unwrap()), vec![head(sym), Some(3), Some(5)]);
    assert_eq!(dims_opt(&x.tril(-1).unwrap()), vec![head(sym), Some(3), Some(5)]);
}

#[test_case(false; "concrete")]
#[test_case(true; "symbolic_batch")]
fn test_unflatten_symbolic_passthrough(sym: bool) {
    let x = input(sym, &[6, 5]);
    assert_eq!(dims_opt(&x.unflatten(1, &[2, 3]).unwrap()), vec![head(sym), Some(2), Some(3), Some(5)]);
    assert_eq!(dims_opt(&x.unflatten(-1, &[5, 1]).unwrap()), vec![head(sym), Some(6), Some(5), Some(1)]);
}

#[test_case(false; "concrete")]
#[test_case(true; "symbolic_batch")]
fn test_repeat_symbolic_passthrough(sym: bool) {
    let x = input(sym, &[3, 5]);
    let r = x.repeat(&[SInt::from(1usize), SInt::from(2usize), SInt::from(3usize)]).unwrap();
    assert_eq!(dims_opt(&r), vec![head(sym), Some(6), Some(15)]);
}

#[test_case(false; "concrete")]
#[test_case(true; "symbolic_batch")]
fn test_narrow_symbolic_passthrough(sym: bool) {
    let x = input(sym, &[6, 5]);
    assert_eq!(dims_opt(&x.narrow(1, 1usize, 3usize).unwrap()), vec![head(sym), Some(3), Some(5)]);
    assert_eq!(dims_opt(&x.narrow(-1, 2usize, 2usize).unwrap()), vec![head(sym), Some(6), Some(2)]);
}

/// A symbolic `start` narrows to a *constant* extent: the offset becomes a
/// runtime index, not a runtime axis length. Spelling the size as `end - begin`
/// would leave `(t + 1) + t * -1` behind, turning the axis symbolic and
/// stopping the optimizer from upcasting it.
#[test]
fn test_narrow_with_a_symbolic_start_keeps_a_constant_extent() {
    let t = crate::Variable::new("t", 0, 5);
    let x = Tensor::empty(&[6, 5], svod_dtype::DType::Float32);
    let slot = x.narrow(0, t.as_sint(), 1usize).unwrap();
    assert_eq!(dims_opt(&slot), vec![Some(1), Some(5)]);
    assert_eq!(dims_opt(&slot.try_squeeze(Some(0)).unwrap()), vec![Some(5)]);

    // The offset really is the variable, not a folded constant.
    let uses_t = slot
        .uop()
        .toposort()
        .iter()
        .any(|n| matches!(n.op(), svod_ir::Op::Param(p) if p.arg.name.as_deref() == Some("t")));
    assert!(uses_t, "narrow lost the symbolic offset");
}

#[test_case(false; "concrete")]
#[test_case(true; "symbolic_batch")]
fn test_unfold_symbolic_passthrough(sym: bool) {
    let x = input(sym, &[6, 5]);
    assert_eq!(dims_opt(&x.unfold(-1, 3, 1).unwrap()), vec![head(sym), Some(6), Some(3), Some(3)]);
}

#[test_case(false; "concrete")]
#[test_case(true; "symbolic_batch")]
fn test_upsample_symbolic_passthrough(sym: bool) {
    let x = input(sym, &[2, 3, 4]);
    let y = x.upsample(&[2, 2], ResizeMode::Nearest).unwrap();
    assert_eq!(dims_opt(&y), vec![head(sym), Some(2), Some(6), Some(8)]);
}

#[test_case(false; "concrete")]
#[test_case(true; "symbolic_batch")]
fn test_numel_and_numel_sint(sym: bool) {
    let x = input(sym, &[6, 5]);
    assert_eq!(x.numel().ok(), head(sym).map(|b| b * 30));
    assert_eq!(x.numel_sint().unwrap().as_const(), head(sym).map(|b| b * 30));
}

// =========================================================================
// Host references
// =========================================================================

/// Torch `unfold` over a row-major `iota` tensor: every output element is the
/// flat source index it was copied from.
fn unfold_reference(shape: &[usize], dim: usize, size: usize, step: usize) -> (Vec<usize>, Vec<f32>) {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len() - 1).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    let mut out_shape = shape.to_vec();
    out_shape[dim] = (shape[dim] - size) / step + 1;
    out_shape.push(size);

    let total: usize = out_shape.iter().product();
    let mut idx = vec![0usize; out_shape.len()];
    let mut out = Vec::with_capacity(total);
    for _ in 0..total {
        let k = idx[out_shape.len() - 1];
        let src: usize = (0..shape.len()).map(|d| if d == dim { idx[d] * step + k } else { idx[d] } * strides[d]).sum();
        out.push(src as f32);
        for d in (0..idx.len()).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    (out_shape, out)
}

/// The yolo backbone's nearest 2x upsample (gather along H then W).
fn upsample_nearest_2x_gather(x: &Tensor) -> Tensor {
    let b = x.dim(0).unwrap();
    let c = x.dim(1).unwrap();
    let h = x.dim_const(2).unwrap();
    let w = x.dim_const(3).unwrap();

    let h_idx: Vec<i64> = (0..h as i64).flat_map(|v| [v, v]).collect();
    let w_idx: Vec<i64> = (0..w as i64).flat_map(|v| [v, v]).collect();

    let h_index = Tensor::from_slice(&h_idx)
        .try_reshape([SInt::from(1usize), SInt::from(1usize), SInt::from(h * 2), SInt::from(1usize)])
        .unwrap()
        .try_expand([b, c, SInt::from(h * 2), SInt::from(w)])
        .unwrap();
    let x = x.gather(2, &h_index).unwrap();

    let shape = x.shape().unwrap();
    let w_index = Tensor::from_slice(&w_idx)
        .try_reshape([SInt::from(1usize), SInt::from(1usize), SInt::from(1usize), SInt::from(w * 2)])
        .unwrap()
        .try_expand([shape[0].clone(), shape[1].clone(), SInt::from(h * 2), SInt::from(w * 2)])
        .unwrap();
    x.gather(3, &w_index).unwrap()
}

/// Host `triu`/`tril` over a row-major `[rows, cols]` iota tensor.
fn tri_reference(rows: usize, cols: usize, diagonal: isize, upper: bool) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            let (r, c) = ((i / cols) as isize, (i % cols) as isize);
            let keep = if upper { c >= r + diagonal } else { c <= r + diagonal };
            if keep { i as f32 } else { 0.0 }
        })
        .collect()
}

// =========================================================================
// Value tests
// =========================================================================

crate::codegen_tests! {
    #[test_case(&[4, 6, 5], 1, 1, 3; "middle_axis")]
    #[test_case(&[4, 6, 5], -1, 2, 2; "negative_axis")]
    #[test_case(&[4, 6, 5], 0, 0, 4; "whole_axis")]
    #[test_case(&[7], 0, 6, 1; "last_element")]
    fn test_narrow_matches_shrink(config, shape: &[usize], dim: isize, start: usize, len: usize) {
        test_setup();
        let x = iota(shape);
        let ndim = shape.len();
        let d = if dim < 0 { (ndim as isize + dim) as usize } else { dim as usize };
        let ranges: Vec<Option<(isize, isize)>> =
            (0..ndim).map(|i| (i == d).then_some((start as isize, (start + len) as isize))).collect();

        let expect = x.try_shrink(ranges).unwrap();
        let got = x.narrow(dim, start, len).unwrap();
        assert_eq!(got.dims().unwrap(), expect.dims().unwrap());
        assert_eq!(
            got.realize_with_and(&config).as_vec::<f32>().unwrap(),
            expect.realize_with_and(&config).as_vec::<f32>().unwrap()
        );
    }

    fn test_slice_with_open_ends(config) {
        test_setup();
        let x = iota(&[4, 6]);
        let starts = [1i64, 2];
        let full = x.slice_with().starts(&starts).ends(&[4i64, 6]).call().unwrap();
        let expected = full.realize_with_and(&config).as_vec::<f32>().unwrap();

        // Omitted setter, explicit `None`s and the ONNX sentinel all mean "to the end".
        for open in [
            x.slice_with().starts(&starts).call().unwrap(),
            x.slice_with().starts(&starts).ends(&[None, None]).call().unwrap(),
            x.slice_with().starts(&starts).ends(&[i64::MAX, i64::MAX]).call().unwrap(),
        ] {
            assert_eq!(open.dims().unwrap(), full.dims().unwrap());
            assert_eq!(open.realize_with_and(&config).as_vec::<f32>().unwrap(), expected);
        }

        // Mixed: axis 0 bounded, axis 1 open.
        let mixed = x.slice_with().starts(&starts).ends(&[Some(3i64), None]).call().unwrap();
        let mixed_ref = x.slice_with().starts(&starts).ends(&[3i64, 6]).call().unwrap();
        assert_eq!(
            mixed.realize_with_and(&config).as_vec::<f32>().unwrap(),
            mixed_ref.realize_with_and(&config).as_vec::<f32>().unwrap()
        );

        // A negative step with no end walks down to index 0 inclusive.
        let down = x.slice_with().starts(&[3, 5]).steps(&[-1, -1]).call().unwrap();
        let down_ref = x.slice_with().starts(&[3, 5]).ends(&[-5i64, -7]).steps(&[-1, -1]).call().unwrap();
        assert_eq!(down.dims().unwrap(), vec![4, 6]);
        assert_eq!(
            down.realize_with_and(&config).as_vec::<f32>().unwrap(),
            down_ref.realize_with_and(&config).as_vec::<f32>().unwrap()
        );
    }

    #[test_case(&[6], 0, 3, 1; "1d_overlapping")]
    #[test_case(&[7], 0, 3, 2; "1d_strided")]
    #[test_case(&[2, 3, 6], 2, 3, 3; "last_axis_tiled")]
    #[test_case(&[2, 3, 6], -1, 4, 2; "negative_axis")]
    #[test_case(&[2, 5, 3], 1, 2, 1; "middle_axis")]
    #[test_case(&[4, 3], 0, 2, 2; "leading_axis")]
    fn test_unfold_matches_reference(config, shape: &[usize], dim: isize, size: usize, step: usize) {
        test_setup();
        let d = if dim < 0 { (shape.len() as isize + dim) as usize } else { dim as usize };
        let (expected_shape, expected) = unfold_reference(shape, d, size, step);

        let got = iota(shape).unfold(dim, size, step).unwrap();
        assert_eq!(got.dims().unwrap(), expected_shape);
        assert_eq!(got.realize_with_and(&config).as_vec::<f32>().unwrap(), expected);
    }

    fn test_unfold_rejects_bad_params(config) {
        let _ = config;
        let x = iota(&[4]);
        assert!(x.unfold(0, 0, 1).is_err(), "zero window");
        assert!(x.unfold(0, 2, 0).is_err(), "zero step");
        assert!(x.unfold(0, 5, 1).is_err(), "window wider than the axis");
        assert!(x.unfold(1, 2, 1).is_err(), "axis out of range");
    }

    #[test_case(&[1, 2, 3, 4]; "single_batch")]
    #[test_case(&[2, 3, 2, 5]; "multi_batch")]
    fn test_upsample_nearest_matches_gather(config, shape: &[usize]) {
        test_setup();
        let x = iota(shape);
        let expect = upsample_nearest_2x_gather(&x);
        let got = x.upsample(&[2, 2], ResizeMode::Nearest).unwrap();
        assert_eq!(got.dims().unwrap(), expect.dims().unwrap());
        assert_eq!(
            got.realize_with_and(&config).as_vec::<f32>().unwrap(),
            expect.realize_with_and(&config).as_vec::<f32>().unwrap()
        );
    }

    #[test_case(&[1, 1, 3, 4]; "single_channel")]
    #[test_case(&[2, 3, 4, 2]; "multi_channel")]
    fn test_upsample_bilinear_matches_resize(config, shape: &[usize]) {
        test_setup();
        let x = iota(shape);
        // The yolo depth head's `resize_bilinear_2x`.
        let expect = x
            .resize()
            .scales(&[1.0, 1.0, 2.0, 2.0])
            .mode(ResizeMode::Linear)
            .coordinate_transformation_mode(CoordinateTransformMode::AlignCorners)
            .call()
            .unwrap();
        let got = x
            .upsample_with()
            .scale(&[2, 2])
            .mode(ResizeMode::Linear)
            .coordinate_transformation_mode(CoordinateTransformMode::AlignCorners)
            .call()
            .unwrap();
        assert_eq!(got.dims().unwrap(), expect.dims().unwrap());
        assert_close_f32(
            &got.realize_with_and(&config).as_vec::<f32>().unwrap(),
            &expect.realize_with_and(&config).as_vec::<f32>().unwrap(),
            1e-5,
        );
    }

    #[test_case(&[2, 2], None, None; "defaults")]
    #[test_case(&[2, 2], Some(&[1, 1]), None; "stride_override")]
    #[test_case(&[2, 2], None, Some(&[2, 2]); "dilation_override")]
    fn test_pool_defaults(config, kernel: &[usize], stride: Option<&[usize]>, dilation: Option<&[usize]>) {
        test_setup();
        let x = iota(&[1, 1, 6, 6]);
        let ones = vec![1usize; kernel.len()];
        let expect = x.pool(kernel, stride.unwrap_or(kernel), dilation.unwrap_or(&ones)).unwrap();
        let got = x.pool_with().kernel(kernel).maybe_stride(stride).maybe_dilation(dilation).call().unwrap();
        assert_eq!(got.dims().unwrap(), expect.dims().unwrap());
        assert_eq!(
            got.realize_with_and(&config).as_vec::<f32>().unwrap(),
            expect.realize_with_and(&config).as_vec::<f32>().unwrap()
        );
    }

    #[test_case(4, 5, 0; "square_main")]
    #[test_case(4, 5, 2; "above_main")]
    #[test_case(4, 5, -2; "below_main")]
    #[test_case(3, 3, -1; "sub_diagonal")]
    fn test_triangular_isize_diagonal(config, rows: usize, cols: usize, diagonal: isize) {
        test_setup();
        let x = iota(&[rows, cols]);
        let upper = x.triu(diagonal).unwrap();
        let lower = x.tril(diagonal).unwrap();
        assert_eq!(
            upper.realize_with_and(&config).as_vec::<f32>().unwrap(),
            tri_reference(rows, cols, diagonal, true)
        );
        assert_eq!(
            lower.realize_with_and(&config).as_vec::<f32>().unwrap(),
            tri_reference(rows, cols, diagonal, false)
        );
    }
}
