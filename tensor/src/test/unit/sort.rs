use crate::*;

fn get_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.uop().shape().unwrap().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
}

#[test]
fn test_sort_1d_ascending() {
    let _ = tracing_subscriber::fmt::try_init();
    let t = Tensor::from_slice([3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
    let (sorted, indices) = t.sort(-1, false).unwrap();
    let sorted = sorted.realize().unwrap();
    let vals = sorted.to_ndarray::<f32>().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]);
    let idx = indices.realize().unwrap().to_ndarray::<i32>().unwrap();
    assert_eq!(idx.as_slice().unwrap(), &[1, 3, 6, 0, 2, 4, 7, 5]);
}

#[test]
fn test_sort_1d_descending() {
    let t = Tensor::from_slice([3.0f32, 1.0, 4.0, 1.0, 5.0]);
    let (sorted, _indices) = t.sort(-1, true).unwrap();
    let sorted = sorted.realize().unwrap();
    let vals = sorted.to_ndarray::<f32>().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[5.0, 4.0, 3.0, 1.0, 1.0]);
}

#[test]
fn test_sort_power_of_2_size() {
    let t = Tensor::from_slice([4.0f32, 2.0, 1.0, 3.0]);
    let (sorted, _) = t.sort(-1, false).unwrap();
    let sorted = sorted.realize().unwrap();
    let vals = sorted.to_ndarray::<f32>().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_sort_single_element() {
    let t = Tensor::from_slice([42.0f32]);
    let (sorted, indices) = t.sort(-1, false).unwrap();
    let sorted = sorted.realize().unwrap();
    let vals = sorted.to_ndarray::<f32>().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[42.0]);
    let idx = indices.realize().unwrap().to_ndarray::<i32>().unwrap();
    assert_eq!(idx.as_slice().unwrap(), &[0]);
}

#[test]
fn test_sort_2d_dim1() {
    let _ = tracing_subscriber::fmt::try_init();
    let t = Tensor::from_slice([3.0f32, 1.0, 2.0, 6.0, 4.0, 5.0]).try_reshape(&[2, 3]).unwrap();
    let (sorted, _) = t.sort(1, false).unwrap();
    let sorted = sorted.realize().unwrap();
    assert_eq!(get_shape(&sorted), vec![2, 3]);
    let vals = sorted.to_ndarray::<f32>().unwrap();
    assert_eq!(vals[[0, 0]], 1.0);
    assert_eq!(vals[[0, 1]], 2.0);
    assert_eq!(vals[[0, 2]], 3.0);
    assert_eq!(vals[[1, 0]], 4.0);
    assert_eq!(vals[[1, 1]], 5.0);
    assert_eq!(vals[[1, 2]], 6.0);
}
