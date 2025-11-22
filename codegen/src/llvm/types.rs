//! LLVM type mappings using inkwell.

use crate::Result;
use inkwell::context::Context;
use inkwell::types::BasicTypeEnum;
use morok_dtype::{DType, ScalarDType};

/// Convert DType to inkwell BasicTypeEnum.
///
/// This function maps morok DTypes to LLVM types via inkwell's type system.
pub fn dtype_to_basic_type<'ctx>(dtype: &DType, context: &'ctx Context) -> BasicTypeEnum<'ctx> {
    match dtype {
        DType::Scalar(scalar) => scalar_to_basic_type(*scalar, context),
        DType::Vector { scalar, count } => vector_to_basic_type(*scalar, *count as u32, context),
        DType::Ptr { .. } => pointer_type(context).into(),
        DType::Image { .. } => {
            // Image types are represented as pointers in LLVM
            pointer_type(context).into()
        }
    }
}

/// Convert ScalarDType to LLVM basic type.
fn scalar_to_basic_type<'ctx>(scalar: ScalarDType, context: &'ctx Context) -> BasicTypeEnum<'ctx> {
    match scalar {
        ScalarDType::Bool => context.bool_type().into(),
        ScalarDType::Int8 => context.i8_type().into(),
        ScalarDType::Int16 => context.i16_type().into(),
        ScalarDType::Int32 => context.i32_type().into(),
        ScalarDType::Int64 => context.i64_type().into(),
        ScalarDType::UInt8 => context.i8_type().into(),
        ScalarDType::UInt16 => context.i16_type().into(),
        ScalarDType::UInt32 => context.i32_type().into(),
        ScalarDType::UInt64 => context.i64_type().into(),
        ScalarDType::Float16 => context.f16_type().into(),
        ScalarDType::BFloat16 => context.f16_type().into(), // Note: LLVM doesn't have native bfloat16, use f16
        ScalarDType::Float32 => context.f32_type().into(),
        ScalarDType::Float64 => context.f64_type().into(),
        ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => context.i8_type().into(), // FP8 stored as i8
        ScalarDType::Void => {
            // Void type shouldn't be used as a basic type
            panic!("Void type cannot be converted to BasicTypeEnum")
        }
        ScalarDType::Index => context.i64_type().into(), // Index type -> i64
    }
}

/// Convert ScalarDType to LLVM vector type.
fn vector_to_basic_type<'ctx>(scalar: ScalarDType, count: u32, context: &'ctx Context) -> BasicTypeEnum<'ctx> {
    let base = scalar_to_basic_type(scalar, context);
    match base {
        BasicTypeEnum::IntType(t) => t.vec_type(count).into(),
        BasicTypeEnum::FloatType(t) => t.vec_type(count).into(),
        _ => base, // Fallback
    }
}

/// Get vector type for a dtype with specified width.
pub fn dtype_to_vector_type<'ctx>(
    dtype: &DType,
    width: u32,
    context: &'ctx Context,
) -> BasicTypeEnum<'ctx> {
    let base_type = dtype_to_basic_type(dtype, context);

    match base_type {
        BasicTypeEnum::IntType(t) => t.vec_type(width).into(),
        BasicTypeEnum::FloatType(t) => t.vec_type(width).into(),
        _ => base_type, // Fallback to base type
    }
}

/// Get pointer type (opaque pointer in LLVM 15+).
pub fn pointer_type<'ctx>(context: &'ctx Context) -> inkwell::types::PointerType<'ctx> {
    // LLVM 21 uses opaque pointers
    context.ptr_type(inkwell::AddressSpace::default())
}

/// Validate that a dtype is supported for code generation.
pub fn validate_dtype(_dtype: &DType) -> Result<()> {
    // All current dtypes are supported
    Ok(())
}
