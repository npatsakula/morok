use super::*;
use enumset::EnumSet;

impl DType {
    pub fn can_safe_cast(from: Self, to: Self) -> bool {
        match (from, to) {
            // Same type or from Bool (Bool can cast to anything)
            (f, t) if f == t || f == Self::Bool => true,
            // Unsigned -> Unsigned: only if target is larger
            (f, t) if f.is_unsigned() && t.is_unsigned() => f.bytes() < t.bytes(),
            // Signed -> Signed: only if target is same size or larger
            (f, t) if f.is_signed() && t.is_signed() => f.bytes() <= t.bytes(),
            // Unsigned -> Signed: only if target is strictly larger (to accommodate the extra bit)
            (f, t) if f.is_unsigned() && t.is_signed() => f.bytes() < t.bytes(),
            // Integer -> Float: safe if integer is Int32 or smaller
            (f, t) if !f.is_float() && t.is_float() => f.bytes() <= Self::Int32.bytes(),
            // Float -> Float: only if target is larger
            (f, t) if f.is_float() && t.is_float() => f.bytes() < t.bytes(),
            _ => false,
        }
    }

    const fn promotion_lattice(self) -> &'static [Self] {
        use DType::*;
        match self {
            Bool => &[Int8, UInt8],
            Int8 => &[Int16],
            Int16 => &[Int32],
            Int32 => &[Int64],
            Int64 => &[FP8E4M3, FP8E5M2],
            UInt8 => &[Int16, UInt16],
            UInt16 => &[Int32, UInt32],
            UInt32 => &[Int64, UInt64],
            UInt64 => &[FP8E4M3, FP8E5M2],
            FP8E5M2 => &[Float16, BFloat16],
            FP8E4M3 => &[Float16, BFloat16],
            Float16 => &[Float32],
            BFloat16 => &[Float32],
            Float32 => &[Float64],
            Float64 => &[],
        }
    }

    fn get_recursive_parents(self) -> EnumSet<Self> {
        self.promotion_lattice()
            .iter()
            .fold(EnumSet::only(self), |dtypes, &parent| dtypes.union(parent.get_recursive_parents()))
    }

    pub fn least_upper_dtype(dtypes: &[Self]) -> Option<Self> {
        dtypes
            .iter()
            .map(|dtype| dtype.get_recursive_parents())
            .reduce(|lhs, rhs| lhs.intersection(rhs))
            .and_then(|intersection| intersection.iter().next())
    }
}
