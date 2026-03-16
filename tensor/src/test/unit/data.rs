use ndarray::{Array1, Array3, array};

use crate::Tensor;

// =========================================================================
// Codegen-required tests
// =========================================================================

crate::codegen_tests! {
    fn test_to_vec_computed(config) {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([10.0f32, 20.0, 30.0]);
        let c = (&a + &b).realize_with(&config).unwrap();
        let v = c.to_vec::<f32>().unwrap();
        assert_eq!(v, vec![11.0, 22.0, 33.0]);
    }
}

// === from_ndarray ===

#[test]
fn test_from_ndarray_1d() {
    let arr = array![1.0f32, 2.0, 3.0];
    let t = Tensor::from_ndarray(&arr);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view.shape(), &[3]);
    assert_eq!(view, arr.view().into_dyn());
}

#[test]
fn test_from_ndarray_2d() {
    let arr = array![[1i32, 2], [3, 4]];
    let t = Tensor::from_ndarray(&arr);
    let view = t.array_view::<i32>().unwrap();
    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view, arr.view().into_dyn());
}

#[test]
fn test_from_ndarray_3d() {
    let arr = Array3::from_shape_vec((2, 1, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let t = Tensor::from_ndarray(&arr);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view.shape(), &[2, 1, 3]);
    assert_eq!(view, arr.view().into_dyn());
}

#[test]
fn test_from_ndarray_0d() {
    let arr = ndarray::arr0(42.0f32);
    let t = Tensor::from_ndarray(&arr);
    let v = t.to_vec::<f32>().unwrap();
    assert_eq!(v, vec![42.0]);
}

#[test]
fn test_from_ndarray_empty() {
    let arr = Array1::<f32>::zeros(0);
    let t = Tensor::from_ndarray(&arr);
    let result = t.to_ndarray::<f32>().unwrap();
    assert_eq!(result.shape(), &[0]);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_from_ndarray_view() {
    let arr = array![10.0f32, 20.0, 30.0];
    let view = arr.view();
    let t = Tensor::from_ndarray(&view);
    let tv = t.array_view::<f32>().unwrap();
    assert_eq!(tv, arr.view().into_dyn());
}

#[test]
fn test_from_ndarray_fortran() {
    // Fortran (column-major) — hits slow path, still correct logical order
    let arr = ndarray::Array2::from_shape_vec(ndarray::ShapeBuilder::f((2, 2)), vec![1.0f32, 3.0, 2.0, 4.0]).unwrap();
    let t = Tensor::from_ndarray(&arr);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view, arr.view().into_dyn());
}

// === to_vec ===

#[test]
fn test_to_vec_f32() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let v = t.to_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0]);
}

// === array_view ===

#[test]
fn test_array_view_basic() {
    let arr = array![[1.0f32, 2.0], [3.0, 4.0]];
    let t = Tensor::from_ndarray(&arr);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view, arr.view().into_dyn());
}

#[test]
fn test_array_view_on_from_slice() {
    // from_slice retains buffer — no realize needed
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_array_view_unrealized() {
    let a = Tensor::from_slice([1.0f32, 2.0]);
    let b = Tensor::from_slice([3.0f32, 4.0]);
    let c = &a + &b; // lazy, no buffer
    assert!(c.array_view::<f32>().is_err());
}

#[test]
fn test_array_view_dtype_mismatch() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    assert!(t.array_view::<i32>().is_err());
}

// === array_view_mut ===

#[test]
fn test_array_view_mut_write() {
    let t = Tensor::from_ndarray(&ndarray::Array2::<f32>::zeros((2, 3)));
    t.array_view_mut::<f32>().unwrap()[[1, 2]] = 42.0;
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view[[1, 2]], 42.0);
    assert_eq!(view[[0, 0]], 0.0);
}

#[test]
fn test_array_view_mut_fill() {
    let t = Tensor::from_slice([0.0f32; 4]);
    t.array_view_mut::<f32>().unwrap().fill(7.0);
    assert_eq!(t.array_view::<f32>().unwrap().as_slice().unwrap(), &[7.0; 4]);
}

#[test]
fn test_array_view_mut_unrealized() {
    let a = Tensor::from_slice([1.0f32, 2.0]);
    let c = &a + &a;
    assert!(c.array_view_mut::<f32>().is_err());
}

// === roundtrip ===

#[test]
fn test_roundtrip_ndarray() {
    let original = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let t = Tensor::from_ndarray(&original);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view, original.view().into_dyn());
}

#[test]
fn test_roundtrip_vec() {
    let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let t = Tensor::from_slice(&original);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view.as_slice().unwrap(), &original);
}
