use crate::*;
use ndarray::array;

fn get_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.uop().shape().unwrap().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
}

crate::codegen_tests! {
    fn test_einsum_matrix_multiply(config) {
        let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);
        let result = Tensor::einsum("ij,jk->ik", &[&a, &b]).unwrap();
        let result = result.realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![2, 2]);
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view[[0, 0]], 19.0);
        assert_eq!(view[[0, 1]], 22.0);
        assert_eq!(view[[1, 0]], 43.0);
        assert_eq!(view[[1, 1]], 50.0);
    }

    fn test_einsum_trace(config) {
        let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
        let result = Tensor::einsum("ii->", &[&a]).unwrap();
        let result = result.realize_with(&config).unwrap();
        assert!((result.to_vec::<f32>().unwrap()[0] - 5.0).abs() < 1e-5);
    }

    fn test_einsum_transpose(config) {
        let a = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = Tensor::einsum("ij->ji", &[&a]).unwrap();
        let result = result.realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![3, 2]);
    }

    fn test_einsum_outer_product(config) {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0]);
        let result = Tensor::einsum("i,j->ij", &[&a, &b]).unwrap();
        let result = result.realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![3, 2]);
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view[[0, 0]], 4.0);
        assert_eq!(view[[0, 1]], 5.0);
        assert_eq!(view[[1, 0]], 8.0);
        assert_eq!(view[[1, 1]], 10.0);
        assert_eq!(view[[2, 0]], 12.0);
        assert_eq!(view[[2, 1]], 15.0);
    }

    fn test_einsum_batch_matmul(config) {
        let a = Tensor::from_ndarray(&array![[[1.0f32, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]]);
        let b = Tensor::from_ndarray(&array![[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);
        let result = Tensor::einsum("bij,bjk->bik", &[&a, &b]).unwrap();
        let result = result.realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![2, 2, 2]);
    }

    fn test_einsum_sum_all(config) {
        let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
        let result = Tensor::einsum("ij->", &[&a]).unwrap();
        let result = result.realize_with(&config).unwrap();
        assert!((result.to_vec::<f32>().unwrap()[0] - 10.0).abs() < 1e-5);
    }
}
