use crate::test::helpers::*;

#[test]
fn test_convert_onnx_dtype_all_supported() {
    use tensor_proto::DataType;

    assert_eq!(convert_onnx_dtype(DataType::Float as i32).unwrap().base(), ScalarDType::Float32);
    assert_eq!(convert_onnx_dtype(DataType::Uint8 as i32).unwrap().base(), ScalarDType::UInt8);
    assert_eq!(convert_onnx_dtype(DataType::Int8 as i32).unwrap().base(), ScalarDType::Int8);
    assert_eq!(convert_onnx_dtype(DataType::Int32 as i32).unwrap().base(), ScalarDType::Int32);
    assert_eq!(convert_onnx_dtype(DataType::Int64 as i32).unwrap().base(), ScalarDType::Int64);
    assert_eq!(convert_onnx_dtype(DataType::Bool as i32).unwrap().base(), ScalarDType::Bool);
    assert_eq!(convert_onnx_dtype(DataType::Double as i32).unwrap().base(), ScalarDType::Float64);
}

#[test]
fn test_convert_onnx_dtype_unsupported() {
    assert!(convert_onnx_dtype(8).is_err()); // String
    assert!(convert_onnx_dtype(999).is_err()); // Unknown
}

#[test]
fn test_extract_tensor_data_raw() {
    let mut tensor = TensorProto::default();
    tensor.raw_data = vec![1, 2, 3, 4];
    let data = extract_tensor_data(&tensor).unwrap();
    assert_eq!(data, vec![1, 2, 3, 4]);
}

#[test]
fn test_attrs_int() {
    use crate::registry::attr::Attrs;

    let mut node = NodeProto::default();
    node.op_type = "Test".to_string();
    node.attribute.push(make_attr_int("axis", 42));

    let mut attrs = Attrs::new(&node);
    assert_eq!(attrs.int("axis", 0), 42);
    assert_eq!(attrs.int("missing", -1), -1);
    attrs.done().unwrap(); // all consumed
}

#[test]
fn test_attrs_ints() {
    use crate::registry::attr::Attrs;

    let mut node = NodeProto::default();
    node.op_type = "Test".to_string();
    node.attribute.push(make_attr_ints("perm", &[1, 2, 0]));

    let mut attrs = Attrs::new(&node);
    assert_eq!(attrs.ints("perm"), vec![1, 2, 0]);
    assert!(attrs.ints("missing").is_empty());
    attrs.done().unwrap();
}

#[test]
fn test_attrs_done_errors_on_unconsumed() {
    use crate::registry::attr::Attrs;

    let mut node = NodeProto::default();
    node.op_type = "Test".to_string();
    node.attribute.push(make_attr_int("axis", 1));
    node.attribute.push(make_attr_float("epsilon", 0.01));

    let mut attrs = Attrs::new(&node);
    let _ = attrs.int("axis", 0); // consume only one
    let err = attrs.done().unwrap_err();
    assert!(err.to_string().contains("epsilon"), "error should mention unconsumed attr: {err}");
}

#[test]
fn test_tensor_from_proto_f32() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensor = make_tensor_proto(raw, vec![2, 3], 1); // FLOAT

    let result = tensor_from_proto(&tensor).unwrap();
    let dims: Vec<usize> = result.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    assert_eq!(dims, vec![2, 3]);
    assert_eq!(result.to_vec::<f32>().unwrap(), [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_create_tensor_int8() {
    let values: Vec<i8> = vec![-128, 0, 127];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = create_tensor_from_raw(&data, &[3], DType::Int8).unwrap();
    assert_eq!(t.to_vec::<i8>().unwrap(), [-128i8, 0, 127]);
}

#[test]
fn test_create_tensor_uint8() {
    let values: Vec<u8> = vec![0, 128, 255];
    let t = create_tensor_from_raw(&values, &[3], DType::UInt8).unwrap();
    assert_eq!(t.to_vec::<u8>().unwrap(), [0u8, 128, 255]);
}

#[test]
fn test_create_tensor_int16() {
    let values: Vec<i16> = vec![-32768, 0, 32767];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = create_tensor_from_raw(&data, &[3], DType::Int16).unwrap();
    assert_eq!(t.to_vec::<i16>().unwrap(), [-32768i16, 0, 32767]);
}

#[test]
fn test_create_tensor_uint16() {
    let values: Vec<u16> = vec![0, 32768, 65535];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = create_tensor_from_raw(&data, &[3], DType::UInt16).unwrap();
    assert_eq!(t.to_vec::<u16>().unwrap(), [0u16, 32768, 65535]);
}

#[test]
fn test_create_tensor_uint32() {
    let values: Vec<u32> = vec![0, 1_000_000, u32::MAX];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = create_tensor_from_raw(&data, &[3], DType::UInt32).unwrap();
    assert_eq!(t.to_vec::<u32>().unwrap(), [0u32, 1_000_000, u32::MAX]);
}

#[test]
fn test_create_tensor_uint64() {
    let values: Vec<u64> = vec![0, 1_000_000_000, u64::MAX];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = create_tensor_from_raw(&data, &[3], DType::UInt64).unwrap();
    assert_eq!(t.to_vec::<u64>().unwrap(), [0u64, 1_000_000_000, u64::MAX]);
}

#[test]
fn test_create_tensor_float16() {
    // IEEE 754 half-precision: 1.0 = 0x3C00, 2.0 = 0x4000, 0.5 = 0x3800
    let f16_bits: Vec<u16> = vec![0x3C00, 0x4000, 0x3800];
    let data: Vec<u8> = f16_bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = create_tensor_from_raw(&data, &[3], DType::Float16).unwrap();
    let t_f32 = t.cast(DType::Float32).unwrap();
    let vals = t_f32.to_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-3);
    assert!((vals[1] - 2.0).abs() < 1e-3);
    assert!((vals[2] - 0.5).abs() < 1e-3);
}

#[test]
fn test_create_tensor_bfloat16() {
    // BFloat16: 1.0 = 0x3F80, 2.0 = 0x4000, 0.5 = 0x3F00
    let bf16_bits: Vec<u16> = vec![0x3F80, 0x4000, 0x3F00];
    let data: Vec<u8> = bf16_bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = create_tensor_from_raw(&data, &[3], DType::BFloat16).unwrap();
    let t_f32 = t.cast(DType::Float32).unwrap();
    let vals = t_f32.to_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-2);
    assert!((vals[1] - 2.0).abs() < 1e-2);
    assert!((vals[2] - 0.5).abs() < 1e-2);
}

#[test]
fn test_extract_float64_from_double_data() {
    let mut tensor = TensorProto::default();
    tensor.data_type = tensor_proto::DataType::Double as i32;
    tensor.dims = vec![2];
    tensor.double_data = vec![1.5, 2.5];

    let data = extract_tensor_data(&tensor).unwrap();
    let values: Vec<f64> = bytemuck::cast_slice(&data).to_vec();
    assert_eq!(values, vec![1.5, 2.5]);
}

#[test]
fn test_extract_uint64_from_uint64_data() {
    let mut tensor = TensorProto::default();
    tensor.data_type = tensor_proto::DataType::Uint64 as i32;
    tensor.dims = vec![2];
    tensor.uint64_data = vec![100, 200];

    let data = extract_tensor_data(&tensor).unwrap();
    let values: Vec<u64> = bytemuck::cast_slice(&data).to_vec();
    assert_eq!(values, vec![100u64, 200]);
}

#[test]
fn test_extract_uint32_from_uint64_data() {
    let mut tensor = TensorProto::default();
    tensor.data_type = tensor_proto::DataType::Uint32 as i32;
    tensor.dims = vec![2];
    tensor.uint64_data = vec![100, 200];

    let data = extract_tensor_data(&tensor).unwrap();
    let values: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    assert_eq!(values, vec![100u32, 200]);
}

#[test]
fn test_external_data_loading() {
    use std::io::Write;

    let dir = std::env::temp_dir().join("morok_test_external_data");
    std::fs::create_dir_all(&dir).unwrap();

    let values: Vec<f32> = vec![1.0, 2.0, 3.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let data_path = dir.join("weights.bin");
    let mut f = std::fs::File::create(&data_path).unwrap();
    f.write_all(&raw).unwrap();

    let mut tensor = TensorProto::default();
    tensor.data_type = tensor_proto::DataType::Float as i32;
    tensor.dims = vec![3];
    tensor.data_location = 1;
    tensor.external_data = vec![
        StringStringEntryProto { key: "location".into(), value: "weights.bin".into() },
        StringStringEntryProto { key: "offset".into(), value: "0".into() },
        StringStringEntryProto { key: "length".into(), value: raw.len().to_string() },
    ];

    let result = tensor_from_proto_ext(&tensor, Some(&dir)).unwrap();
    assert_eq!(result.to_vec::<f32>().unwrap(), [1.0f32, 2.0, 3.0]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_external_data_with_offset() {
    use std::io::Write;

    let dir = std::env::temp_dir().join("morok_test_external_offset");
    std::fs::create_dir_all(&dir).unwrap();

    let padding = vec![0u8; 8];
    let values: Vec<f32> = vec![42.0, 99.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let data_path = dir.join("weights_offset.bin");
    let mut f = std::fs::File::create(&data_path).unwrap();
    f.write_all(&padding).unwrap();
    f.write_all(&raw).unwrap();

    let mut tensor = TensorProto::default();
    tensor.data_type = tensor_proto::DataType::Float as i32;
    tensor.dims = vec![2];
    tensor.data_location = 1;
    tensor.external_data = vec![
        StringStringEntryProto { key: "location".into(), value: "weights_offset.bin".into() },
        StringStringEntryProto { key: "offset".into(), value: "8".into() },
        StringStringEntryProto { key: "length".into(), value: raw.len().to_string() },
    ];

    let result = tensor_from_proto_ext(&tensor, Some(&dir)).unwrap();
    assert_eq!(result.to_vec::<f32>().unwrap(), [42.0f32, 99.0]);

    std::fs::remove_dir_all(&dir).ok();
}
