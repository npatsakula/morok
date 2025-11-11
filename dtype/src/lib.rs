pub mod cast;
pub mod ext;

#[cfg(test)]
pub mod test;

#[derive(Debug, Hash, PartialOrd, Ord)]
#[derive(strum::EnumCount, strum::EnumIter, strum::VariantArray, strum::FromRepr)]
#[derive(enumset::EnumSetType)]
#[cfg_attr(feature = "proptest", derive(proptest_derive::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[enumset(repr = "u16")]
pub enum DType {
    Bool = 0,

    Int8 = 1,
    Int16 = 2,
    Int32 = 3,
    Int64 = 4,

    UInt8 = 5,
    UInt16 = 6,
    UInt32 = 7,
    UInt64 = 8,

    FP8E4M3 = 9,
    FP8E5M2 = 10,
    Float16 = 11,
    BFloat16 = 12,
    Float32 = 13,
    Float64 = 14,

    /// Void type for metadata operations (no data).
    Void = 15,
}

impl DType {
    pub const fn bytes(&self) -> usize {
        match self {
            Self::Bool => 1,
            Self::Int8 => 1,
            Self::Int16 => 2,
            Self::Int32 => 4,
            Self::Int64 => 8,
            Self::UInt8 => 1,
            Self::UInt16 => 2,
            Self::UInt32 => 4,
            Self::UInt64 => 8,
            Self::FP8E4M3 => 1,
            Self::FP8E5M2 => 1,
            Self::Float16 => 2,
            Self::BFloat16 => 2,
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Void => 0,
        }
    }

    pub const fn is_bool(&self) -> bool {
        matches!(self, Self::Bool)
    }

    pub const fn is_signed(&self) -> bool {
        matches!(self, Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64)
    }

    pub const fn is_unsigned(&self) -> bool {
        matches!(self, Self::UInt8 | Self::UInt16 | Self::UInt32 | Self::UInt64)
    }

    pub const fn is_float(&self) -> bool {
        matches!(self, Self::Float16 | Self::Float32 | Self::Float64)
    }

    pub const fn c_style(&self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Int8 => "signed char",
            Self::Int16 => "short",
            Self::Int32 => "int",
            Self::Int64 => "long",
            Self::UInt8 => "unsigned char",
            Self::UInt16 => "unsigned short",
            Self::UInt32 => "unsigned int",
            Self::UInt64 => "unsigned long",
            Self::FP8E4M3 => "float8_e4m3",
            Self::FP8E5M2 => "float8_e5m2",
            Self::Float16 => "half",
            Self::Float32 => "float",
            Self::Float64 => "double",
            Self::BFloat16 => "__bf16",
            Self::Void => "void",
        }
    }
}
