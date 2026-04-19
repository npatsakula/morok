use std::collections::HashMap;
use std::path::Path;

use morok_dtype::DType;
use morok_tensor::Tensor;
use snafu::{ResultExt, Snafu};

pub type StateDict = HashMap<String, Tensor>;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("failed to read file: {source}"))]
    Io { source: std::io::Error },
    #[snafu(display("failed to deserialize safetensors"))]
    Safetensors { source: safetensors::SafeTensorError },
    #[snafu(display("unsupported dtype in safetensors: {dtype}"))]
    UnsupportedDtype { dtype: String },
    #[snafu(display("missing key in state dict: {key}"))]
    MissingKey { key: String },
    #[snafu(display("{source}"))]
    Tensor {
        #[snafu(source(from(morok_tensor::error::Error, Box::new)))]
        source: Box<morok_tensor::error::Error>,
    },
}

type Result<T> = std::result::Result<T, Error>;

pub fn load_safetensors(path: &Path) -> Result<StateDict> {
    let data = std::fs::read(path).context(IoSnafu)?;
    let tensors = safetensors::SafeTensors::deserialize(&data).context(SafetensorsSnafu)?;
    let mut sd = StateDict::new();
    for (name, view) in tensors.tensors() {
        let dtype = convert_dtype(view.dtype())?;
        let shape: Vec<usize> = view.shape().to_vec();
        let tensor = Tensor::from_raw_bytes(view.data(), &shape, dtype).context(TensorSnafu)?;
        sd.insert(name.to_string(), tensor);
    }
    Ok(sd)
}

fn convert_dtype(dt: safetensors::Dtype) -> Result<DType> {
    use safetensors::Dtype as ST;
    match dt {
        ST::F32 => Ok(DType::Float32),
        ST::F16 => Ok(DType::Float16),
        ST::BF16 => Ok(DType::BFloat16),
        ST::F64 => Ok(DType::Float64),
        ST::I32 => Ok(DType::Int32),
        ST::I64 => Ok(DType::Int64),
        ST::I16 => Ok(DType::Int16),
        ST::I8 => Ok(DType::Int8),
        ST::U8 => Ok(DType::UInt8),
        ST::BOOL => Ok(DType::Bool),
        other => Err(Error::UnsupportedDtype { dtype: format!("{other:?}") }),
    }
}

pub trait HasStateDict {
    fn state_dict(&self, prefix: &str) -> StateDict;
    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> Result<()>;
}

/// Helper: get a tensor from a state dict by key, returning an error if missing.
pub fn get_tensor(sd: &StateDict, key: &str) -> Result<Tensor> {
    sd.get(key).cloned().ok_or_else(|| Error::MissingKey { key: key.to_string() })
}

/// Helper: format a prefixed key.
pub fn prefixed(prefix: &str, name: &str) -> String {
    if prefix.is_empty() { name.to_string() } else { format!("{prefix}.{name}") }
}
