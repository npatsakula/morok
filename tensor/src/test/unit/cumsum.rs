use ndarray::array;

use crate::Tensor;

crate::codegen_tests! {
    fn test_cumsum_1d(config) {
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let result = x.cumsum(0).unwrap();
        let result = result.realize_with(&config).unwrap();
        assert_eq!(result.to_vec::<f32>().unwrap(), [1.0, 3.0, 6.0, 10.0]);
    }

    fn test_cumsum_2d_axis0(config) {
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = x.cumsum(0).unwrap();
        let result = result.realize_with(&config).unwrap();
        // Row 0: [1, 2, 3], Row 1: [1+4, 2+5, 3+6] = [5, 7, 9]
        assert_eq!(result.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
    }

    fn test_cumsum_2d_axis1(config) {
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = x.cumsum(1).unwrap();
        let result = result.realize_with(&config).unwrap();
        // Row 0: [1, 3, 6], Row 1: [4, 9, 15]
        assert_eq!(result.to_vec::<f32>().unwrap(), [1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    fn test_cumsum_negative_axis(config) {
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        // -1 = last axis = axis 1
        let result = x.cumsum(-1).unwrap();
        let result = result.realize_with(&config).unwrap();
        assert_eq!(result.to_vec::<f32>().unwrap(), [1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }
}
