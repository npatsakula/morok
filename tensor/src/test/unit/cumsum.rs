use crate::Tensor;

#[test]
fn test_cumsum_1d() {
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let result = x.cumsum(0).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn test_cumsum_2d_axis0() {
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
    let result = x.cumsum(0).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // Row 0: [1, 2, 3], Row 1: [1+4, 2+5, 3+6] = [5, 7, 9]
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
}

#[test]
fn test_cumsum_2d_axis1() {
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
    let result = x.cumsum(1).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // Row 0: [1, 3, 6], Row 1: [4, 9, 15]
    assert_eq!(vals, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}

#[test]
fn test_cumsum_negative_axis() {
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
    // -1 = last axis = axis 1
    let result = x.cumsum(-1).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}
