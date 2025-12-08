//! LLVM type mappings using inkwell.

use inkwell::context::Context;
use inkwell::types::BasicTypeEnum;
use morok_dtype::{DType, ScalarDType};

use crate::llvm::error::{InvalidLlvmTypeSnafu, Result};

/// Convert DType to inkwell BasicTypeEnum.
///
/// This function maps morok DTypes to LLVM types via inkwell's type system.
pub fn dtype_to_basic_type<'ctx>(dtype: &DType, context: &'ctx Context) -> Result<BasicTypeEnum<'ctx>> {
    match dtype {
        DType::Scalar(scalar) => scalar_to_basic_type(*scalar, context),
        DType::Vector { scalar, count } => vector_to_basic_type(*scalar, *count as u32, context),
        DType::Ptr { .. } => Ok(pointer_type(context).into()),
        DType::Image { .. } => {
            // Image types are represented as pointers in LLVM
            Ok(pointer_type(context).into())
        }
    }
}

/// Convert ScalarDType to LLVM basic type.
fn scalar_to_basic_type<'ctx>(scalar: ScalarDType, context: &'ctx Context) -> Result<BasicTypeEnum<'ctx>> {
    match scalar {
        ScalarDType::Bool => Ok(context.bool_type().into()),
        ScalarDType::Int8 => Ok(context.i8_type().into()),
        ScalarDType::Int16 => Ok(context.i16_type().into()),
        ScalarDType::Int32 => Ok(context.i32_type().into()),
        ScalarDType::Int64 => Ok(context.i64_type().into()),
        ScalarDType::UInt8 => Ok(context.i8_type().into()),
        ScalarDType::UInt16 => Ok(context.i16_type().into()),
        ScalarDType::UInt32 => Ok(context.i32_type().into()),
        ScalarDType::UInt64 => Ok(context.i64_type().into()),
        ScalarDType::Float16 => Ok(context.f16_type().into()),
        ScalarDType::BFloat16 => Ok(context.f16_type().into()), // Note: LLVM doesn't have native bfloat16, use f16
        ScalarDType::Float32 => Ok(context.f32_type().into()),
        ScalarDType::Float64 => Ok(context.f64_type().into()),
        ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => Ok(context.i8_type().into()), // FP8 stored as i8
        ScalarDType::Void => InvalidLlvmTypeSnafu { dtype: "Void" }.fail(),
        ScalarDType::Index => Ok(context.i64_type().into()), // Index type -> i64
    }
}

/// Convert ScalarDType to LLVM vector type.
fn vector_to_basic_type<'ctx>(scalar: ScalarDType, count: u32, context: &'ctx Context) -> Result<BasicTypeEnum<'ctx>> {
    let base = scalar_to_basic_type(scalar, context)?;
    Ok(match base {
        BasicTypeEnum::IntType(t) => t.vec_type(count).into(),
        BasicTypeEnum::FloatType(t) => t.vec_type(count).into(),
        _ => base, // Fallback
    })
}

/// Get vector type for a dtype with specified width.
pub fn dtype_to_vector_type<'ctx>(dtype: &DType, width: u32, context: &'ctx Context) -> Result<BasicTypeEnum<'ctx>> {
    let base_type = dtype_to_basic_type(dtype, context)?;

    Ok(match base_type {
        BasicTypeEnum::IntType(t) => t.vec_type(width).into(),
        BasicTypeEnum::FloatType(t) => t.vec_type(width).into(),
        _ => base_type, // Fallback to base type
    })
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
