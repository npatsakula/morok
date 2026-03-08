use morok_dtype::DType;

use crate::Tensor;

#[test]
fn test_from_raw_bytes_f32() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = Tensor::from_raw_bytes(&bytes, &[2, 3], DType::Float32).unwrap();
    let arr = t.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[2, 3]);
    assert_eq!(arr.as_slice().unwrap(), &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_from_raw_bytes_f16() {
    // IEEE 754 half-precision: 1.0 = 0x3C00, 2.0 = 0x4000
    let f16_bits: Vec<u16> = vec![0x3C00, 0x4000];
    let bytes: Vec<u8> = f16_bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = Tensor::from_raw_bytes(&bytes, &[2], DType::Float16).unwrap();
    assert_eq!(t.uop().dtype(), DType::Float16);

    // Cast to f32 and verify
    let t_f32 = t.cast(DType::Float32).unwrap();
    let arr = t_f32.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert!((vals[0] - 1.0).abs() < 1e-3);
    assert!((vals[1] - 2.0).abs() < 1e-3);
}

#[test]
fn test_from_raw_bytes_wrong_length() {
    let bytes = vec![0u8; 10]; // 10 bytes != 3 * 4 for f32
    let result = Tensor::from_raw_bytes(&bytes, &[3], DType::Float32);
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("from_raw_bytes"), "Error should mention from_raw_bytes: {err}");
}

#[test]
fn test_eye_square() {
    let eye = Tensor::eye(3, 3, DType::Float32).unwrap();
    let arr = eye.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[3, 3]);
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_eye_rectangular() {
    let eye = Tensor::eye(2, 4, DType::Float32).unwrap();
    let arr = eye.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[2, 4]);
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn test_eye_single() {
    let eye = Tensor::eye(1, 1, DType::Float32).unwrap();
    let arr = eye.to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 1]);
    assert_eq!(arr[[0, 0]], 1.0);
}
