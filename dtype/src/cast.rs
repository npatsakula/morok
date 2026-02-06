use super::*;
use enumset::EnumSet;

impl ScalarDType {
    const fn promotion_lattice(self) -> &'static [Self] {
        use ScalarDType::*;
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
            Float64 | Void | Index => &[],
        }
    }

    fn get_recursive_parents(self) -> EnumSet<Self> {
        self.promotion_lattice()
            .iter()
            .fold(EnumSet::only(self), |dtypes, &parent| dtypes.union(parent.get_recursive_parents()))
    }

    /// Check if casting from `from` to `to` is safe (preserves value).
    pub fn can_safe_cast(self, to: Self) -> bool {
        // Same type (compare discriminants) or from Bool (Bool can cast to anything)
        if self == to || matches!(self, Self::Bool) {
            return true;
        }

        // Index type: can cast from any integer to Index
        if matches!(to, Self::Index) {
            return self.is_int();
        }

        let from_bytes = self.bytes();
        let to_bytes = to.bytes();
        match (self.is_unsigned(), self.is_signed(), self.is_float(), to.is_unsigned(), to.is_signed(), to.is_float()) {
            // Unsigned -> Unsigned: only if target is larger
            (true, _, _, true, _, _) => from_bytes < to_bytes,
            // Signed -> Signed: only if target is same size or larger
            (_, true, _, _, true, _) => from_bytes <= to_bytes,
            // Unsigned -> Signed: only if target is strictly larger
            (true, _, _, _, true, _) => from_bytes < to_bytes,
            // Integer -> Float: safe if integer is Int32 or smaller
            (_, _, false, _, _, true) => from_bytes <= Self::Int32.bytes(),
            // Float -> Float: only if target is larger
            (_, _, true, _, _, true) => from_bytes < to_bytes,
            _ => false,
        }
    }
}

impl DType {
    /// Check if casting from `from` to `to` is safe (preserves value).
    pub fn can_safe_cast(from: Self, to: Self) -> bool {
        // Extract scalars
        let (Some(from_scalar), Some(to_scalar)) = (from.scalar(), to.scalar()) else {
            return false;
        };

        // Check scalar cast is safe
        if !from_scalar.can_safe_cast(to_scalar) {
            return false;
        }

        // Vector counts must match (or broadcast from scalar)
        from.count() == to.count() || from.count() == 1 || to.count() == 1
    }

    /// Find the least upper bound type for a set of dtypes.
    ///
    /// Returns the smallest type that all input types can be safely cast to.
    ///
    /// Type promotion rules:
    /// - Scalar + Scalar → promoted Scalar
    /// - Ptr<T> + Ptr<T> → Ptr<T> (same Ptr types)
    /// - Ptr<T> + Scalar(T) → Scalar(T) (Ptr will be auto-loaded in codegen)
    /// - Ptr<T> + Scalar(U) → promoted Scalar (if T and U are compatible)
    pub fn least_upper_dtype(dtypes: &[Self]) -> Option<Self> {
        if dtypes.is_empty() {
            return None;
        }

        // Check for ImageDType first (they always win in promotion)
        if let Some(img) = dtypes.iter().find(|d| matches!(d, DType::Image { .. })) {
            return Some(img.clone());
        }

        // Check if all types are identical Ptr types
        let first = &dtypes[0];
        if matches!(first, DType::Ptr { .. }) && dtypes.iter().all(|d| d == first) {
            return Some(first.clone());
        }

        // Find common scalar type via promotion lattice intersection
        // Use base() to extract scalar from Ptr types for promotion
        // This allows Ptr<Float32> + Float32 → Float32
        let scalar_result = dtypes
            .iter()
            .map(|d| d.base())
            .map(|s| s.get_recursive_parents())
            .reduce(|lhs, rhs| lhs.intersection(rhs))?
            .iter()
            .min()?; // min by discriminant (= priority: lower = more specific)

        // Return scalar type (Ptr values will be auto-loaded in codegen)
        Some(DType::Scalar(scalar_result))
    }
}
