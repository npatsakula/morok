use crate::*;
use proptest::prelude::*;

#[rustfmt::skip]
impl DType {
    pub fn int_generator() -> impl Strategy<Value = Self> {
        prop_oneof![
            Just(Self::Int8), Just(Self::Int16), Just(Self::Int32), Just(Self::Int64),
            Just(Self::UInt8), Just(Self::UInt16), Just(Self::UInt32), Just(Self::UInt64)
        ]
    }

    pub fn float_generator() -> impl Strategy<Value = Self> {
        prop_oneof![
            Just(Self::FP8E4M3), Just(Self::FP8E5M2), Just(Self::Float16),
            Just(Self::Float32), Just(Self::Float64)
        ]
    }
}

proptest! {
    #[test]
    fn least_upper_dtype(lhs: DType, rhs: DType) {
        match DType::least_upper_dtype(&[lhs, rhs]) {
            Some(_) => (),
            None => prop_assert!([lhs, rhs].contains(&DType::Float64)),
        }
    }
}
