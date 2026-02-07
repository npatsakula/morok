//! DType → MLIR type conversion and constant/identity helpers.

use melior::ir::r#type::IntegerType;
use melior::ir::{Type, TypeLike};
use melior::{Context, dialect::llvm};
use morok_dtype::{DType, ScalarDType};
use morok_ir::{ConstValue, ReduceOp};

/// Convert a scalar DType to an MLIR type.
pub fn mlir_scalar_type<'c>(ctx: &'c Context, s: ScalarDType) -> Type<'c> {
    match s {
        ScalarDType::Bool => IntegerType::new(ctx, 1).into(),
        ScalarDType::Int8 | ScalarDType::UInt8 | ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => {
            IntegerType::new(ctx, 8).into()
        }
        ScalarDType::Int16 | ScalarDType::UInt16 => IntegerType::new(ctx, 16).into(),
        ScalarDType::Int32 | ScalarDType::UInt32 => IntegerType::new(ctx, 32).into(),
        ScalarDType::Int64 | ScalarDType::UInt64 | ScalarDType::Index => IntegerType::new(ctx, 64).into(),
        ScalarDType::Float16 => Type::float16(ctx),
        ScalarDType::BFloat16 => Type::bfloat16(ctx),
        ScalarDType::Float32 => Type::float32(ctx),
        ScalarDType::Float64 => Type::float64(ctx),
        ScalarDType::Void => llvm::r#type::void(ctx),
    }
}

/// Convert a DType to an MLIR type.
pub fn mlir_type<'c>(ctx: &'c Context, dtype: &DType) -> Type<'c> {
    match dtype {
        DType::Scalar(s) => mlir_scalar_type(ctx, *s),
        DType::Vector { scalar, count } => {
            Type::vector(&[*count as u64], mlir_scalar_type(ctx, *scalar))
        }
        DType::Ptr { .. } => llvm::r#type::pointer(ctx, 0),
        DType::Image { .. } => llvm::r#type::pointer(ctx, 0),
    }
}

/// Get the MLIR pointer type (opaque pointer in address space 0).
pub fn mlir_ptr_type(ctx: &Context) -> Type<'_> {
    llvm::r#type::pointer(ctx, 0)
}

/// Get the bit width of a scalar type (for integer attribute creation).
fn scalar_bits(s: ScalarDType) -> u32 {
    match s {
        ScalarDType::Bool => 1,
        ScalarDType::Int8 | ScalarDType::UInt8 | ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => 8,
        ScalarDType::Int16 | ScalarDType::UInt16 => 16,
        ScalarDType::Int32 | ScalarDType::UInt32 => 32,
        ScalarDType::Int64 | ScalarDType::UInt64 | ScalarDType::Index => 64,
        ScalarDType::Float16 | ScalarDType::BFloat16 => 16,
        ScalarDType::Float32 => 32,
        ScalarDType::Float64 => 64,
        ScalarDType::Void => 0,
    }
}

/// Check if a DType is a float type (scalar or vector of floats).
pub fn is_float(dtype: &DType) -> bool {
    dtype.is_float()
}

/// Check if a DType is signed integer.
pub fn is_signed(dtype: &DType) -> bool {
    dtype.is_signed()
}

/// Get the LLVM intrinsic name suffix for a type (e.g., "f32", "v4f32").
pub fn intrinsic_type_suffix(dtype: &DType) -> String {
    match dtype {
        DType::Vector { scalar, count } => {
            format!("v{}{}", count, scalar_intrinsic_suffix(*scalar))
        }
        DType::Scalar(s) => scalar_intrinsic_suffix(*s).to_string(),
        _ => "f32".to_string(),
    }
}

fn scalar_intrinsic_suffix(s: ScalarDType) -> &'static str {
    match s {
        ScalarDType::Float16 => "f16",
        ScalarDType::BFloat16 => "bf16",
        ScalarDType::Float32 => "f32",
        ScalarDType::Float64 => "f64",
        ScalarDType::Int8 | ScalarDType::UInt8 => "i8",
        ScalarDType::Int16 | ScalarDType::UInt16 => "i16",
        ScalarDType::Int32 | ScalarDType::UInt32 => "i32",
        ScalarDType::Int64 | ScalarDType::UInt64 | ScalarDType::Index => "i64",
        ScalarDType::Bool => "i1",
        _ => "f32",
    }
}

/// Get the number of bits for an MLIR integer attribute given a DType.
pub fn integer_attr_bits(dtype: &DType) -> u32 {
    match dtype {
        DType::Scalar(s) => scalar_bits(*s),
        DType::Vector { scalar, .. } => scalar_bits(*scalar),
        _ => 64,
    }
}

/// Get the identity value for a reduce operation as a ConstValue.
pub fn reduce_identity_value(op: ReduceOp, dtype: &DType) -> ConstValue {
    match op {
        ReduceOp::Add => {
            if dtype.is_float() {
                ConstValue::Float(0.0)
            } else {
                ConstValue::Int(0)
            }
        }
        ReduceOp::Mul => {
            if dtype.is_float() {
                ConstValue::Float(1.0)
            } else {
                ConstValue::Int(1)
            }
        }
        ReduceOp::Max => {
            if dtype.is_float() {
                ConstValue::Float(f64::NEG_INFINITY)
            } else if dtype.is_signed() {
                ConstValue::Int(i64::MIN)
            } else {
                ConstValue::UInt(0)
            }
        }
        ReduceOp::Min => {
            if dtype.is_float() {
                ConstValue::Float(f64::INFINITY)
            } else if dtype.is_signed() {
                ConstValue::Int(i64::MAX)
            } else {
                ConstValue::UInt(u64::MAX)
            }
        }
    }
}
