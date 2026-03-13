use crate::Tensor;
use ndarray::array;

crate::codegen_tests! {
    // =========================================================================
    // Triu Tests
    // =========================================================================

    fn test_triu_basic(config) {
        // 3x3 matrix -> upper triangular (zero below diagonal)
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let result = x.triu(0).unwrap();
        let result = result.realize_with(&config).unwrap();
        // Expected: [1, 2, 3, 0, 5, 6, 0, 0, 9]
        assert_eq!(result.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }

    fn test_triu_diagonal_positive(config) {
        // k=1: exclude main diagonal
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let result = x.triu(1).unwrap();
        let result = result.realize_with(&config).unwrap();
        // Expected: [0, 2, 3, 0, 0, 6, 0, 0, 0]
        assert_eq!(result.to_vec::<f32>().unwrap(), [0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]);
    }

    fn test_triu_diagonal_negative(config) {
        // k=-1: include one subdiagonal
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let result = x.triu(-1).unwrap();
        let result = result.realize_with(&config).unwrap();
        // Expected: [1, 2, 3, 4, 5, 6, 0, 8, 9]
        assert_eq!(result.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]);
    }

    // =========================================================================
    // Tril Tests
    // =========================================================================

    fn test_tril_basic(config) {
        // 3x3 matrix -> lower triangular (zero above diagonal)
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let result = x.tril(0).unwrap();
        let result = result.realize_with(&config).unwrap();
        // Expected: [1, 0, 0, 4, 5, 0, 7, 8, 9]
        assert_eq!(result.to_vec::<f32>().unwrap(), [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    fn test_tril_diagonal_negative(config) {
        // k=-1: exclude main diagonal
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let result = x.tril(-1).unwrap();
        let result = result.realize_with(&config).unwrap();
        // Expected: [0, 0, 0, 4, 0, 0, 7, 8, 0]
        assert_eq!(result.to_vec::<f32>().unwrap(), [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]);
    }

    fn test_triu_non_square(config) {
        // 2x4 matrix
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let result = x.triu(0).unwrap();
        let result = result.realize_with(&config).unwrap();
        // Expected: [1, 2, 3, 4, 0, 6, 7, 8]
        assert_eq!(result.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0]);
    }
}
