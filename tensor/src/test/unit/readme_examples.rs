use ndarray::array;

use crate::Tensor;

/// Tests that mirror README.md / introduction.md Quick Example exactly.
/// If these tests break, the doc examples are wrong.

#[test]
fn test_readme_quick_example() {
    let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);
    let mut c = &a + &b;
    c.realize().unwrap();
    let view = c.array_view::<f32>().unwrap();
    assert_eq!(view, array![[6.0, 8.0], [10.0, 12.0]].into_dyn());
}

#[test]
fn test_tensor_readme_basic() {
    let a = Tensor::from_ndarray(&array![1.0f32, 2.0, 3.0]);
    let b = Tensor::from_ndarray(&array![4.0f32, 5.0, 6.0]);
    let mut c = &a + &b;
    c.realize().unwrap();
    let view = c.array_view::<f32>().unwrap();
    assert_eq!(view.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
}
