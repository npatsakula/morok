//! DType → MLIR type conversion and constant/identity helpers.

use melior::ir::Type;
use melior::ir::r#type::IntegerType;
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
        DType::Vector { scalar, count } => Type::vector(&[*count as u64], mlir_scalar_type(ctx, *scalar)),
        DType::Ptr { .. } => llvm::r#type::pointer(ctx, 0),
        DType::Image { .. } => llvm::r#type::pointer(ctx, 0),
    }
}

/// Get the MLIR pointer type (opaque pointer in address space 0).
pub fn mlir_ptr_type(ctx: &Context) -> Type<'_> {
    llvm::r#type::pointer(ctx, 0)
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
