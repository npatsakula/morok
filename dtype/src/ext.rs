use super::*;

pub trait HasDType {
    const DTYPE: DType;
}

macro_rules! impl_dtype_ext {
    ($($ty:ty => $dtype:expr),* $(,)?) => {
        $(impl HasDType for $ty { const DTYPE: DType = $dtype; })*
    };
}

impl_dtype_ext! {
    bool => DType::Bool,
    i8 => DType::Int8, i16 => DType::Int16, i32 => DType::Int32, i64 => DType::Int64,
    u8 => DType::UInt8, u16 => DType::UInt16, u32 => DType::UInt32, u64 => DType::UInt64,
    f32 => DType::Float32, f64 => DType::Float64,
}
