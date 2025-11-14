use crate::*;
use proptest::prelude::*;

#[rustfmt::skip]
impl DType {
    pub fn int_generator() -> impl Strategy<Value = Self> {
        prop_oneof![
            Just(DType::Int8), Just(DType::Int16), Just(DType::Int32), Just(DType::Int64),
            Just(DType::UInt8), Just(DType::UInt16), Just(DType::UInt32), Just(DType::UInt64),
            Just(DType::Index)
        ]
    }

    pub fn float_generator() -> impl Strategy<Value = Self> {
        prop_oneof![
            Just(DType::FP8E4M3), Just(DType::FP8E5M2), Just(DType::Float16),
            Just(DType::BFloat16), Just(DType::Float32), Just(DType::Float64)
        ]
    }

    pub fn scalar_generator() -> impl Strategy<Value = Self> {
        prop_oneof![
            Just(DType::Bool),
            Just(DType::Int8), Just(DType::Int16), Just(DType::Int32), Just(DType::Int64),
            Just(DType::UInt8), Just(DType::UInt16), Just(DType::UInt32), Just(DType::UInt64),
            Just(DType::FP8E4M3), Just(DType::FP8E5M2), Just(DType::Float16),
            Just(DType::BFloat16), Just(DType::Float32), Just(DType::Float64),
            Just(DType::Index)
        ]
    }
}
