use std::collections::HashMap;
use std::path::Path;

use snafu::{ResultExt, Snafu};
use svod_dtype::DType;
use svod_tensor::Tensor;

pub type StateDict = HashMap<String, Tensor>;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
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
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
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

/// Load safetensors weights from a HuggingFace checkpoint directory.
///
/// Handles both single-file (`model.safetensors`) and multi-shard
/// (`model-00001-of-0000N.safetensors` + `model.safetensors.index.json`)
/// checkpoints. For single-file, loads `model.safetensors` directly. For
/// multi-shard, parses the index JSON to find all shard files, loads each,
/// and merges into one [`StateDict`].
pub fn load_safetensors_dir(dir: &Path) -> Result<StateDict> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return load_safetensors(&single);
    }

    let index_path = dir.join("model.safetensors.index.json");
    let index_data = std::fs::read_to_string(&index_path).context(IoSnafu)?;
    let index: SafetensorsIndex =
        serde_json::from_str(&index_data).map_err(|e| Error::Io { source: std::io::Error::other(e.to_string()) })?;

    let mut sd = StateDict::new();
    for shard_file in index.unique_shards() {
        let shard_path = dir.join(&shard_file);
        let shard_sd = load_safetensors(&shard_path)?;
        sd.extend(shard_sd);
    }
    Ok(sd)
}

#[derive(serde::Deserialize)]
pub(crate) struct SafetensorsIndex {
    #[serde(rename = "weight_map")]
    weight_map: HashMap<String, String>,
}

impl SafetensorsIndex {
    pub(crate) fn unique_shards(&self) -> Vec<String> {
        let mut shards: Vec<String> = self.weight_map.values().cloned().collect();
        shards.sort();
        shards.dedup();
        shards
    }
}

fn convert_dtype(dt: safetensors::Dtype) -> Result<DType> {
    use safetensors::Dtype as ST;
    match dt {
        ST::F32 => Ok(DType::Float32),
        ST::F16 => Ok(DType::Float16),
        ST::BF16 => Ok(DType::BFloat16),
        ST::F8_E4M3 => Ok(DType::FP8E4M3),
        ST::F8_E5M2 => Ok(DType::FP8E5M2),
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

/// Cast every tensor in a state dict to `dtype`, leaving any tensor that cannot
/// be cast (e.g. an int embedding key) at its original dtype and logging a
/// warning naming the key — tolerant of foreign checkpoints like the per-block
/// loaders, but no longer failing silently (a kept-foreign tensor would
/// otherwise surface as an opaque dtype error far downstream).
pub fn cast_all(sd: &StateDict, dtype: DType) -> StateDict {
    sd.iter()
        .map(|(k, v)| {
            let from = v.uop().dtype();
            let t = if from == dtype {
                v.clone()
            } else {
                match v.cast(dtype.clone()) {
                    Ok(casted) => casted,
                    Err(_) => {
                        tracing::warn!(key = %k, from = ?from, to = ?dtype, "state-dict cast failed; keeping original dtype");
                        v.clone()
                    }
                }
            };
            (k.clone(), t)
        })
        .collect()
}

/// Helper: format a prefixed key.
pub fn prefixed(prefix: &str, name: &str) -> String {
    if prefix.is_empty() { name.to_string() } else { format!("{prefix}.{name}") }
}

/// Insert each named field of `$self` into the state dict under
/// `<prefix>.<field>`. Field idents are used verbatim as keys.
#[macro_export]
macro_rules! state_field {
    ($sd:expr, $prefix:expr, $self:ident, [$($field:ident),+ $(,)?]) => {
        $(
            $sd.insert(
                $crate::state::prefixed($prefix, stringify!($field)),
                $self.$field.clone(),
            );
        )+
    };
}

/// Load each named field of `$self` from the state dict under
/// `<prefix>.<field>`. Mirrors [`state_field!`].
#[macro_export]
macro_rules! load_state_field {
    ($self:ident, $sd:expr, $prefix:expr, [$($field:ident),+ $(,)?]) => {
        $(
            $self.$field = $crate::state::get_tensor(
                $sd,
                &$crate::state::prefixed($prefix, stringify!($field)),
            )?;
        )+
    };
}
