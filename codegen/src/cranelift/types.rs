//! Cranelift type mappings.

use cranelift_codegen::ir::Type;
use cranelift_codegen::ir::types;
use morok_dtype::{DType, ScalarDType};

/// Convert DType to Cranelift IR type.
pub fn dtype_to_cranelift_type(dtype: &DType) -> Type {
    match dtype {
        DType::Scalar(scalar) => scalar_to_cranelift_type(*scalar),
        DType::Vector { scalar, count } => {
            // Cranelift has limited vector support, fallback to scalar for now
            // TODO: Support Cranelift SIMD types when needed
            let _ = count;
            scalar_to_cranelift_type(*scalar)
        }
        DType::Ptr { .. } => types::I64,   // Pointers are 64-bit
        DType::Image { .. } => types::I64, // Image handles are pointers
    }
}

/// Convert ScalarDType to Cranelift type.
pub fn scalar_to_cranelift_type(scalar: ScalarDType) -> Type {
    match scalar {
        ScalarDType::Bool => types::I8, // Cranelift has no i1, use i8
        ScalarDType::Int8 => types::I8,
        ScalarDType::Int16 => types::I16,
        ScalarDType::Int32 => types::I32,
        ScalarDType::Int64 => types::I64,
        ScalarDType::UInt8 => types::I8, // Unsigned uses same type
        ScalarDType::UInt16 => types::I16,
        ScalarDType::UInt32 => types::I32,
        ScalarDType::UInt64 => types::I64,
        ScalarDType::Float16 => types::F16,
        ScalarDType::BFloat16 => types::F16, // BFloat16 not native, use f16
        ScalarDType::Float32 => types::F32,
        ScalarDType::Float64 => types::F64,
        ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => types::I8, // FP8 stored as i8
        ScalarDType::Void => types::I8,                           // Placeholder for void
        ScalarDType::Index => types::I64,                         // Index type -> i64
    }
}

/// Check if a dtype is a float type.
pub fn is_float(dtype: &DType) -> bool {
    dtype.is_float()
}

/// Check if a dtype is a signed integer type.
pub fn is_signed(dtype: &DType) -> bool {
    dtype.is_signed()
}
