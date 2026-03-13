use morok_dtype::DType;

use crate::Tensor;

crate::codegen_tests! {
    fn test_from_raw_bytes_f32(config) {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = Tensor::from_raw_bytes(&bytes, &[2, 3], DType::Float32).unwrap();
        let shape = t.shape().unwrap();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0].as_const().unwrap(), 2);
        assert_eq!(shape[1].as_const().unwrap(), 3);
        assert_eq!(t.realize_with(&config).unwrap().to_vec::<f32>().unwrap(), [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    fn test_from_raw_bytes_f16(config) {
        let f16_bits: Vec<u16> = vec![0x3C00, 0x4000];
        let bytes: Vec<u8> = f16_bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = Tensor::from_raw_bytes(&bytes, &[2], DType::Float16).unwrap();
        assert_eq!(t.uop().dtype(), DType::Float16);

        let t_f32 = t.cast(DType::Float32).unwrap();
        let vals = t_f32.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-3);
        assert!((vals[1] - 2.0).abs() < 1e-3);
    }

    fn test_eye_square(config) {
        let eye = Tensor::eye(3, 3, DType::Float32).unwrap();
        let shape = eye.shape().unwrap();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0].as_const().unwrap(), 3);
        assert_eq!(shape[1].as_const().unwrap(), 3);
        assert_eq!(
            eye.realize_with(&config).unwrap().to_vec::<f32>().unwrap(),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    fn test_eye_rectangular(config) {
        let eye = Tensor::eye(2, 4, DType::Float32).unwrap();
        let shape = eye.shape().unwrap();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0].as_const().unwrap(), 2);
        assert_eq!(shape[1].as_const().unwrap(), 4);
        assert_eq!(
            eye.realize_with(&config).unwrap().to_vec::<f32>().unwrap(),
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
    }

    fn test_eye_single(config) {
        let eye = Tensor::eye(1, 1, DType::Float32).unwrap().realize_with(&config).unwrap();
        let view = eye.array_view::<f32>().unwrap();
        assert_eq!(view[[0, 0]], 1.0);
    }
}

#[test]
fn test_from_raw_bytes_wrong_length() {
    let bytes = vec![0u8; 10];
    let result = Tensor::from_raw_bytes(&bytes, &[3], DType::Float32);
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("from_raw_bytes"), "Error should mention from_raw_bytes: {err}");
}
