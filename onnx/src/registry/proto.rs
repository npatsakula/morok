use std::path::Path;

use morok_dtype::{DType, ScalarDType};
use morok_tensor::Tensor;

use crate::error::{Error, Result, UnsupportedDTypeSnafu};
use crate::parser::onnx::{TensorProto, tensor_proto};

/// Convert ONNX DataType to Morok DType.
pub fn convert_onnx_dtype(onnx_dtype: i32) -> Result<DType> {
    use tensor_proto::DataType;

    let scalar = match DataType::try_from(onnx_dtype) {
        Ok(DataType::Float) => ScalarDType::Float32,
        Ok(DataType::Uint8) => ScalarDType::UInt8,
        Ok(DataType::Int8) => ScalarDType::Int8,
        Ok(DataType::Uint16) => ScalarDType::UInt16,
        Ok(DataType::Int16) => ScalarDType::Int16,
        Ok(DataType::Int32) => ScalarDType::Int32,
        Ok(DataType::Int64) => ScalarDType::Int64,
        Ok(DataType::Bool) => ScalarDType::Bool,
        Ok(DataType::Float16) => ScalarDType::Float16,
        Ok(DataType::Double) => ScalarDType::Float64,
        Ok(DataType::Uint32) => ScalarDType::UInt32,
        Ok(DataType::Uint64) => ScalarDType::UInt64,
        Ok(DataType::Bfloat16) => ScalarDType::BFloat16,
        Ok(DataType::Float8e4m3fn) | Ok(DataType::Float8e4m3fnuz) => ScalarDType::FP8E4M3,
        Ok(DataType::Float8e5m2) | Ok(DataType::Float8e5m2fnuz) => ScalarDType::FP8E5M2,
        _ => return UnsupportedDTypeSnafu { dtype: onnx_dtype }.fail(),
    };

    Ok(DType::Scalar(scalar))
}

/// Create a Tensor from ONNX TensorProto (weights/initializers).
pub fn tensor_from_proto(tensor: &TensorProto) -> Result<Tensor> {
    tensor_from_proto_ext(tensor, None)
}

/// Create a Tensor from ONNX TensorProto, with optional model directory for external data.
pub fn tensor_from_proto_ext(tensor: &TensorProto, model_dir: Option<&Path>) -> Result<Tensor> {
    let dtype = convert_onnx_dtype(tensor.data_type)?;
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();

    let raw_data = if tensor.data_location == 1 {
        let dir = model_dir
            .ok_or_else(|| Error::IrConstruction { details: "External data requires model directory path".into() })?;
        load_external_data(tensor, dir)?
    } else {
        extract_tensor_data(tensor)?
    };

    create_tensor_from_raw(&raw_data, &dims, dtype)
}

/// Load external tensor data from file.
///
/// ONNX spec: when `data_location == 1`, `external_data` contains key-value pairs
/// with "location" (file path), "offset", and "length".
pub(crate) fn load_external_data(tensor: &TensorProto, model_dir: &Path) -> Result<Vec<u8>> {
    let mut location = None;
    let mut offset: u64 = 0;
    let mut length: Option<u64> = None;

    for kv in &tensor.external_data {
        match kv.key.as_str() {
            "location" => location = Some(kv.value.clone()),
            "offset" => offset = kv.value.parse().unwrap_or(0),
            "length" => length = kv.value.parse().ok(),
            _ => {}
        }
    }

    let location =
        location.ok_or_else(|| Error::IrConstruction { details: "External data missing 'location' key".into() })?;

    let path = model_dir.join(&location);
    let mut file = std::fs::File::open(&path).map_err(|e| Error::IrConstruction {
        details: format!("External data file not found: {}: {e}", path.display()),
    })?;

    use std::io::{Read, Seek, SeekFrom};
    if offset > 0 {
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| Error::IrConstruction { details: format!("External data seek failed: {e}") })?;
    }

    let data = match length {
        Some(len) => {
            let mut buf = vec![0u8; len as usize];
            file.read_exact(&mut buf)
                .map_err(|e| Error::IrConstruction { details: format!("External data read failed: {e}") })?;
            buf
        }
        None => {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)
                .map_err(|e| Error::IrConstruction { details: format!("External data read failed: {e}") })?;
            buf
        }
    };

    Ok(data)
}

/// Create Tensor from raw bytes, shape, and dtype.
pub(crate) fn create_tensor_from_raw(data: &[u8], dims: &[usize], dtype: DType) -> Result<Tensor> {
    let shape: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
    // Empty data: return empty tensor (avoids bytemuck alignment panic on zero-length slices)
    if data.is_empty() {
        let empty: &[f32] = &[];
        return Tensor::from_slice(empty).try_reshape(&shape).map_err(Error::from);
    }
    macro_rules! typed {
        ($ty:ty) => {{
            let values: Vec<$ty> = bytemuck::cast_slice(data).to_vec();
            Tensor::from_slice(&values).try_reshape(&shape)
        }};
    }
    let tensor = match dtype.base() {
        ScalarDType::Float32 => typed!(f32),
        ScalarDType::Float64 => typed!(f64),
        ScalarDType::Int8 => typed!(i8),
        ScalarDType::Int16 => typed!(i16),
        ScalarDType::Int32 => typed!(i32),
        ScalarDType::Int64 => typed!(i64),
        ScalarDType::UInt8 => typed!(u8),
        ScalarDType::UInt16 => typed!(u16),
        ScalarDType::UInt32 => typed!(u32),
        ScalarDType::UInt64 => typed!(u64),
        ScalarDType::Bool => {
            // ONNX raw_data stores bools as single bytes; int32_data stores as i32.
            let values: Vec<bool> = if data.len() == dims.iter().product::<usize>() {
                data.iter().map(|&v| v != 0).collect()
            } else {
                bytemuck::cast_slice::<_, i32>(data).iter().map(|&v| v != 0).collect()
            };
            Tensor::from_slice(&values).try_reshape(&shape)
        }
        // Float16, BFloat16, FP8: no native Rust type — use raw bytes
        ScalarDType::Float16 | ScalarDType::BFloat16 | ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => {
            Tensor::from_raw_bytes(data, dims, dtype)
        }
        _ => {
            return Err(Error::IrConstruction { details: format!("Unsupported dtype for tensor creation: {dtype:?}") });
        }
    };
    tensor.map_err(Error::from)
}

/// Extract raw data bytes from TensorProto.
pub fn extract_tensor_data(tensor: &TensorProto) -> Result<Vec<u8>> {
    if !tensor.raw_data.is_empty() {
        return Ok(tensor.raw_data.clone());
    }

    let dtype = convert_onnx_dtype(tensor.data_type)?;

    // ONNX proto field mapping (per onnx.proto spec):
    //   float_data (field 4)  → Float32
    //   int32_data (field 5)  → Int32, Int8, UInt8, Int16, UInt16, Bool, Float16, BFloat16, FP8
    //   int64_data (field 7)  → Int64
    //   double_data (field 10) → Float64
    //   uint64_data (field 11) → UInt64, UInt32
    let data = match dtype.base() {
        ScalarDType::Float32 => tensor.float_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::Float64 => tensor.double_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::Int32 => tensor.int32_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::Int64 => tensor.int64_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::UInt32 => tensor.uint64_data.iter().flat_map(|&v| (v as u32).to_le_bytes()).collect(),
        ScalarDType::UInt64 => tensor.uint64_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::UInt8 | ScalarDType::Int8 | ScalarDType::Bool => {
            tensor.int32_data.iter().map(|&v| v as u8).collect()
        }
        ScalarDType::Int16 | ScalarDType::UInt16 | ScalarDType::Float16 | ScalarDType::BFloat16 => {
            tensor.int32_data.iter().flat_map(|&v| (v as u16).to_le_bytes()).collect()
        }
        ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => tensor.int32_data.iter().map(|&v| v as u8).collect(),
        _ => tensor.int32_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
    };

    Ok(data)
}

/// ONNX opset version (domain, version).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpSetId {
    pub domain: String,
    pub version: i64,
}

/// Resolve opset version for a domain from the model's opset imports.
/// Default domain "" is equivalent to "ai.onnx". Returns 1 if not found.
pub fn onnx_opset_version(opsets: &[OpSetId], domain: &str) -> i64 {
    let normalized = if domain == "ai.onnx" { "" } else { domain };
    opsets
        .iter()
        .find(|op| {
            let op_domain = if op.domain == "ai.onnx" { "" } else { op.domain.as_str() };
            op_domain == normalized
        })
        .map(|op| op.version)
        .unwrap_or(1)
}
