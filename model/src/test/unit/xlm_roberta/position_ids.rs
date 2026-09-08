use crate::xlm_roberta::position_ids_from_input_ids;
use svod_tensor::Tensor;

#[test]
fn position_ids_basic() {
    let ids = Tensor::from_slice([0i64, 100, 200, 300, 2, 1, 1]).try_reshape([1, 7]).unwrap();
    let pos = position_ids_from_input_ids(&ids, 1).unwrap();
    pos.realize().unwrap();
    assert_eq!(pos.as_vec::<i32>().unwrap(), vec![2, 3, 4, 5, 6, 1, 1]);
}

#[test]
fn position_ids_all_real() {
    let ids = Tensor::from_slice([0i64, 100, 200, 2]).try_reshape([1, 4]).unwrap();
    let pos = position_ids_from_input_ids(&ids, 1).unwrap();
    pos.realize().unwrap();
    assert_eq!(pos.as_vec::<i32>().unwrap(), vec![2, 3, 4, 5]);
}

#[test]
fn position_ids_leading_padding() {
    let ids = Tensor::from_slice([1i64, 1, 0, 100, 2]).try_reshape([1, 5]).unwrap();
    let pos = position_ids_from_input_ids(&ids, 1).unwrap();
    pos.realize().unwrap();
    assert_eq!(pos.as_vec::<i32>().unwrap(), vec![1, 1, 2, 3, 4]);
}
