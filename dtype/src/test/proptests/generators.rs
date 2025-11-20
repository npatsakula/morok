use crate::*;
use proptest::prelude::*;

pub fn int_dtype() -> impl Strategy<Value = ScalarDType> {
    prop_oneof![
        Just(ScalarDType::Int8),
        Just(ScalarDType::Int16),
        Just(ScalarDType::Int32),
        Just(ScalarDType::Int64),
        Just(ScalarDType::UInt8),
        Just(ScalarDType::UInt16),
        Just(ScalarDType::UInt32),
        Just(ScalarDType::UInt64),
        Just(ScalarDType::Index)
    ]
}

pub fn float_dtype() -> impl Strategy<Value = ScalarDType> {
    prop_oneof![
        Just(ScalarDType::FP8E4M3),
        Just(ScalarDType::FP8E5M2),
        Just(ScalarDType::Float16),
        Just(ScalarDType::BFloat16),
        Just(ScalarDType::Float32),
        Just(ScalarDType::Float64)
    ]
}

pub fn scalar_generator() -> impl Strategy<Value = ScalarDType> {
    prop_oneof![Just(ScalarDType::Bool), int_dtype(), float_dtype()]
}
