//! Operation-specific code generation using inkwell.

use crate::{Result, UnsupportedOpSnafu};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::{BasicValueEnum, CallSiteValue, FunctionValue};
use inkwell::{FloatPredicate, IntPredicate};
use morok_ir::{BinaryOp, ConstValue, Op, TernaryOp, UnaryOp, UOp};
use snafu::ensure;
use std::rc::Rc;

use super::helpers::ValueMap;
use super::types::dtype_to_basic_type;

/// Generate LLVM IR for a UOp node.
pub fn codegen_uop<'ctx>(
    uop: &Rc<UOp>,
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    values: &mut ValueMap<'ctx>,
) -> Result<Option<BasicValueEnum<'ctx>>> {
    // Check if already generated
    if let Some(value) = values.get(uop.id) {
        return Ok(Some(value));
    }

    let result = match uop.op() {
        Op::Const(val_hash) => Some(codegen_const(val_hash.0, &uop.dtype(), context)?),
        Op::Unary(op, src) => {
            let src_val = codegen_uop(src, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing {
                    what: "source value for unary op".to_string(),
                })?;
            Some(codegen_unary(*op, src_val, &uop.dtype(), context, module, builder)?)
        }
        Op::Binary(op, lhs, rhs) => {
            let lhs_val = codegen_uop(lhs, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing {
                    what: "lhs value for binary op".to_string(),
                })?;
            let rhs_val = codegen_uop(rhs, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing {
                    what: "rhs value for binary op".to_string(),
                })?;
            Some(codegen_binary(*op, lhs_val, rhs_val, &uop.dtype(), context, module, builder)?)
        }
        Op::Ternary(op, a, b, c) => {
            let a_val = codegen_uop(a, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing {
                    what: "first value for ternary op".to_string(),
                })?;
            let b_val = codegen_uop(b, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing {
                    what: "second value for ternary op".to_string(),
                })?;
            let c_val = codegen_uop(c, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing {
                    what: "third value for ternary op".to_string(),
                })?;
            Some(codegen_ternary(*op, a_val, b_val, c_val, &uop.dtype(), context, module, builder)?)
        }
        Op::Cast { src, dtype } => {
            let src_val = codegen_uop(src, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing {
                    what: "source value for cast".to_string(),
                })?;
            Some(codegen_cast(src_val, &src.dtype(), dtype, context, builder)?)
        }
        Op::Load { buffer, index } => {
            let buffer_ptr = codegen_uop(buffer, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing {
                    what: "buffer pointer for load".to_string(),
                })?;
            let index_val = codegen_uop(index, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing {
                    what: "index value for load".to_string(),
                })?;
            Some(codegen_load(buffer_ptr, index_val, &uop.dtype(), context, builder)?)
        }
        Op::Store { buffer, index, value } => {
            let buffer_ptr = codegen_uop(buffer, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing {
                    what: "buffer pointer for store".to_string(),
                })?;
            let index_val = codegen_uop(index, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing {
                    what: "index value for store".to_string(),
                })?;
            let value_val = codegen_uop(value, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing {
                    what: "value for store".to_string(),
                })?;
            codegen_store(buffer_ptr, index_val, value_val, builder)?;
            None // Store doesn't produce a value
        }
        Op::Buffer { .. } => {
            // BUFFER operations should be handled by the renderer and added to ValueMap
            // before codegen starts. If we reach here, it means the buffer wasn't registered.
            return Err(crate::Error::Missing {
                what: format!("Buffer UOp {} should be in ValueMap", uop.id),
            });
        }
        _ => {
            ensure!(
                false,
                UnsupportedOpSnafu {
                    op: format!("{:?}", uop.op())
                }
            );
            unreachable!()
        }
    };

    // Store result for future lookups
    if let Some(val) = result {
        values.insert(uop.id, val);
    }

    Ok(result)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract BasicValueEnum from CallSiteValue.
fn extract_value<'ctx>(call_site: CallSiteValue<'ctx>) -> BasicValueEnum<'ctx> {
    match call_site.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        inkwell::values::ValueKind::Instruction(_) => {
            panic!("Expected basic value from intrinsic call")
        }
    }
}

/// Get LLVM intrinsic function.
fn get_intrinsic<'ctx>(
    name: &str,
    ret_type: BasicValueEnum<'ctx>,
    module: &Module<'ctx>,
) -> Result<FunctionValue<'ctx>> {
    use inkwell::intrinsics::Intrinsic;

    let intrinsic = Intrinsic::find(name).ok_or_else(|| crate::Error::LlvmError {
        reason: format!("Intrinsic {} not found", name),
    })?;

    intrinsic
        .get_declaration(module, &[ret_type.get_type().into()])
        .ok_or_else(|| crate::Error::LlvmError {
            reason: format!("Failed to get declaration for {}", name),
        })
}

/// Call an LLVM intrinsic and extract the result.
fn call_intrinsic<'ctx>(
    intrinsic_name: &str,
    args: &[BasicValueEnum<'ctx>],
    name: &str,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let intrinsic_fn = get_intrinsic(intrinsic_name, args[0], module)?;
    let call_site = builder
        .build_call(intrinsic_fn, &args.iter().map(|v| (*v).into()).collect::<Vec<_>>(), name)
        .map_err(|e| crate::Error::LlvmError {
            reason: format!("build_call {}: {}", intrinsic_name, e),
        })?;
    Ok(extract_value(call_site))
}

/// Get LLVM type suffix for intrinsics (e.g., "f32" for Float32).
fn get_type_suffix(dtype: &morok_dtype::DType) -> Result<String> {
    match dtype.scalar() {
        Some(morok_dtype::ScalarDType::Float16) => Ok("f16".to_string()),
        Some(morok_dtype::ScalarDType::BFloat16) => Ok("f16".to_string()),
        Some(morok_dtype::ScalarDType::Float32) => Ok("f32".to_string()),
        Some(morok_dtype::ScalarDType::Float64) => Ok("f64".to_string()),
        _ => Err(crate::Error::TypeError {
            reason: format!("Type {:?} not supported for intrinsics", dtype),
        }),
    }
}

// ============================================================================
// Codegen Functions
// ============================================================================

/// Generate LLVM constant value.
fn codegen_const<'ctx>(
    val: ConstValue,
    dtype: &morok_dtype::DType,
    context: &'ctx Context,
) -> Result<BasicValueEnum<'ctx>> {
    let llvm_type = dtype_to_basic_type(dtype, context);

    let value = match val {
        ConstValue::Int(i) => llvm_type.into_int_type().const_int(i as u64, true).into(),
        ConstValue::UInt(u) => llvm_type.into_int_type().const_int(u, false).into(),
        ConstValue::Float(f) => llvm_type.into_float_type().const_float(f).into(),
        ConstValue::Bool(b) => context.bool_type().const_int(b as u64, false).into(),
    };

    Ok(value)
}

/// Generate code for unary operations.
fn codegen_unary<'ctx>(
    op: UnaryOp,
    src: BasicValueEnum<'ctx>,
    result_dtype: &morok_dtype::DType,
    _context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let is_float = result_dtype.is_float();

    match op {
        UnaryOp::Neg => {
            if is_float {
                Ok(builder
                    .build_float_neg(src.into_float_value(), "neg")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_float_neg: {}", e),
                    })?
                    .into())
            } else {
                Ok(builder
                    .build_int_neg(src.into_int_value(), "neg")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_neg: {}", e),
                    })?
                    .into())
            }
        }
        UnaryOp::Abs => {
            if is_float {
                let suffix = get_type_suffix(result_dtype)?;
                call_intrinsic(&format!("llvm.fabs.{}", suffix), &[src], "abs", module, builder)
            } else {
                // For integers: x < 0 ? -x : x
                let int_val = src.into_int_value();
                let zero = int_val.get_type().const_zero();
                let is_neg = builder
                    .build_int_compare(IntPredicate::SLT, int_val, zero, "is_neg")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_compare: {}", e),
                    })?;
                let neg_val = builder
                    .build_int_neg(int_val, "neg_val")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_neg: {}", e),
                    })?;
                Ok(builder
                    .build_select(is_neg, neg_val, int_val, "abs")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_select: {}", e),
                    })?)
            }
        }
        UnaryOp::Sqrt => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.sqrt.{}", suffix), &[src], "sqrt", module, builder)
        }
        UnaryOp::Rsqrt => {
            // rsqrt = 1 / sqrt(x)
            let suffix = get_type_suffix(result_dtype)?;
            let sqrt_val = call_intrinsic(
                &format!("llvm.sqrt.{}", suffix),
                &[src],
                "sqrt",
                module,
                builder,
            )?;
            let one = src.into_float_value().get_type().const_float(1.0);
            Ok(builder
                .build_float_div(one, sqrt_val.into_float_value(), "rsqrt")
                .map_err(|e| crate::Error::LlvmError {
                    reason: format!("build_float_div: {}", e),
                })?
                .into())
        }
        UnaryOp::Exp => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.exp.{}", suffix), &[src], "exp", module, builder)
        }
        UnaryOp::Exp2 => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.exp2.{}", suffix), &[src], "exp2", module, builder)
        }
        UnaryOp::Log => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.log.{}", suffix), &[src], "log", module, builder)
        }
        UnaryOp::Log2 => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.log2.{}", suffix), &[src], "log2", module, builder)
        }
        UnaryOp::Sin => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.sin.{}", suffix), &[src], "sin", module, builder)
        }
        UnaryOp::Cos => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.cos.{}", suffix), &[src], "cos", module, builder)
        }
        _ => {
            ensure!(
                false,
                UnsupportedOpSnafu {
                    op: format!("{:?}", op)
                }
            );
            unreachable!()
        }
    }
}

/// Generate code for binary operations.
fn codegen_binary<'ctx>(
    op: BinaryOp,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    result_dtype: &morok_dtype::DType,
    _context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let is_float = result_dtype.is_float();
    let is_signed = result_dtype.is_signed();

    match op {
        BinaryOp::Add => {
            if is_float {
                Ok(builder
                    .build_float_add(lhs.into_float_value(), rhs.into_float_value(), "add")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_float_add: {}", e),
                    })?
                    .into())
            } else {
                Ok(builder
                    .build_int_add(lhs.into_int_value(), rhs.into_int_value(), "add")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_add: {}", e),
                    })?
                    .into())
            }
        }
        BinaryOp::Mul => {
            if is_float {
                Ok(builder
                    .build_float_mul(lhs.into_float_value(), rhs.into_float_value(), "mul")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_float_mul: {}", e),
                    })?
                    .into())
            } else {
                Ok(builder
                    .build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "mul")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_mul: {}", e),
                    })?
                    .into())
            }
        }
        BinaryOp::Sub => {
            if is_float {
                Ok(builder
                    .build_float_sub(lhs.into_float_value(), rhs.into_float_value(), "sub")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_float_sub: {}", e),
                    })?
                    .into())
            } else {
                Ok(builder
                    .build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "sub")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_sub: {}", e),
                    })?
                    .into())
            }
        }
        BinaryOp::Fdiv => Ok(builder
            .build_float_div(lhs.into_float_value(), rhs.into_float_value(), "fdiv")
            .map_err(|e| crate::Error::LlvmError {
                reason: format!("build_float_div: {}", e),
            })?
            .into()),
        BinaryOp::Idiv => {
            if is_signed {
                Ok(builder
                    .build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "idiv")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_signed_div: {}", e),
                    })?
                    .into())
            } else {
                Ok(builder
                    .build_int_unsigned_div(lhs.into_int_value(), rhs.into_int_value(), "idiv")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_unsigned_div: {}", e),
                    })?
                    .into())
            }
        }
        BinaryOp::Mod => {
            if is_float {
                Ok(builder
                    .build_float_rem(lhs.into_float_value(), rhs.into_float_value(), "mod")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_float_rem: {}", e),
                    })?
                    .into())
            } else if is_signed {
                Ok(builder
                    .build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "mod")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_signed_rem: {}", e),
                    })?
                    .into())
            } else {
                Ok(builder
                    .build_int_unsigned_rem(lhs.into_int_value(), rhs.into_int_value(), "mod")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_unsigned_rem: {}", e),
                    })?
                    .into())
            }
        }
        BinaryOp::Max => {
            if is_float {
                let suffix = get_type_suffix(result_dtype)?;
                call_intrinsic(&format!("llvm.maxnum.{}", suffix), &[lhs, rhs], "max", module, builder)
            } else if is_signed {
                call_intrinsic(&format!("llvm.smax.i{}", result_dtype.bytes() * 8), &[lhs, rhs], "max", module, builder)
            } else {
                call_intrinsic(&format!("llvm.umax.i{}", result_dtype.bytes() * 8), &[lhs, rhs], "max", module, builder)
            }
        }
        BinaryOp::Pow => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.pow.{}", suffix), &[lhs, rhs], "pow", module, builder)
        }
        BinaryOp::And => Ok(builder
            .build_and(lhs.into_int_value(), rhs.into_int_value(), "and")
            .map_err(|e| crate::Error::LlvmError {
                reason: format!("build_and: {}", e),
            })?
            .into()),
        BinaryOp::Or => Ok(builder
            .build_or(lhs.into_int_value(), rhs.into_int_value(), "or")
            .map_err(|e| crate::Error::LlvmError {
                reason: format!("build_or: {}", e),
            })?
            .into()),
        BinaryOp::Xor => Ok(builder
            .build_xor(lhs.into_int_value(), rhs.into_int_value(), "xor")
            .map_err(|e| crate::Error::LlvmError {
                reason: format!("build_xor: {}", e),
            })?
            .into()),
        BinaryOp::Shl => Ok(builder
            .build_left_shift(lhs.into_int_value(), rhs.into_int_value(), "shl")
            .map_err(|e| crate::Error::LlvmError {
                reason: format!("build_left_shift: {}", e),
            })?
            .into()),
        BinaryOp::Shr => Ok(builder
            .build_right_shift(lhs.into_int_value(), rhs.into_int_value(), is_signed, "shr")
            .map_err(|e| crate::Error::LlvmError {
                reason: format!("build_right_shift: {}", e),
            })?
            .into()),
        BinaryOp::Lt => {
            if is_float {
                Ok(builder
                    .build_float_compare(FloatPredicate::OLT, lhs.into_float_value(), rhs.into_float_value(), "lt")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_float_compare: {}", e),
                    })?
                    .into())
            } else {
                let pred = if is_signed { IntPredicate::SLT } else { IntPredicate::ULT };
                Ok(builder
                    .build_int_compare(pred, lhs.into_int_value(), rhs.into_int_value(), "lt")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_compare: {}", e),
                    })?
                    .into())
            }
        }
        BinaryOp::Eq => {
            if is_float {
                Ok(builder
                    .build_float_compare(FloatPredicate::OEQ, lhs.into_float_value(), rhs.into_float_value(), "eq")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_float_compare: {}", e),
                    })?
                    .into())
            } else {
                Ok(builder
                    .build_int_compare(IntPredicate::EQ, lhs.into_int_value(), rhs.into_int_value(), "eq")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_compare: {}", e),
                    })?
                    .into())
            }
        }
        _ => {
            ensure!(
                false,
                UnsupportedOpSnafu {
                    op: format!("{:?}", op)
                }
            );
            unreachable!()
        }
    }
}

/// Generate code for ternary operations.
fn codegen_ternary<'ctx>(
    op: TernaryOp,
    a: BasicValueEnum<'ctx>,
    b: BasicValueEnum<'ctx>,
    c: BasicValueEnum<'ctx>,
    result_dtype: &morok_dtype::DType,
    _context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    match op {
        TernaryOp::Where => Ok(builder
            .build_select(a.into_int_value(), b, c, "where")
            .map_err(|e| crate::Error::LlvmError {
                reason: format!("build_select: {}", e),
            })?),
        TernaryOp::MulAcc => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.fma.{}", suffix), &[a, b, c], "fma", module, builder)
        }
    }
}

/// Generate code for cast operations.
fn codegen_cast<'ctx>(
    src: BasicValueEnum<'ctx>,
    src_dtype: &morok_dtype::DType,
    dst_dtype: &morok_dtype::DType,
    context: &'ctx Context,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let dst_type = dtype_to_basic_type(dst_dtype, context);

    let src_is_float = src_dtype.is_float();
    let dst_is_float = dst_dtype.is_float();
    let src_is_signed = src_dtype.is_signed();
    let dst_is_signed = dst_dtype.is_signed();

    match (src_is_float, dst_is_float) {
        (true, true) => {
            let src_val = src.into_float_value();
            let dst_type = dst_type.into_float_type();
            if src_dtype.bytes() > dst_dtype.bytes() {
                Ok(builder
                    .build_float_trunc(src_val, dst_type, "fptrunc")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_float_trunc: {}", e),
                    })?
                    .into())
            } else {
                Ok(builder
                    .build_float_ext(src_val, dst_type, "fpext")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_float_ext: {}", e),
                    })?
                    .into())
            }
        }
        (true, false) => {
            let src_val = src.into_float_value();
            let dst_type = dst_type.into_int_type();
            if dst_is_signed {
                Ok(builder
                    .build_float_to_signed_int(src_val, dst_type, "fptosi")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_float_to_signed_int: {}", e),
                    })?
                    .into())
            } else {
                Ok(builder
                    .build_float_to_unsigned_int(src_val, dst_type, "fptoui")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_float_to_unsigned_int: {}", e),
                    })?
                    .into())
            }
        }
        (false, true) => {
            let src_val = src.into_int_value();
            let dst_type = dst_type.into_float_type();
            if src_is_signed {
                Ok(builder
                    .build_signed_int_to_float(src_val, dst_type, "sitofp")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_signed_int_to_float: {}", e),
                    })?
                    .into())
            } else {
                Ok(builder
                    .build_unsigned_int_to_float(src_val, dst_type, "uitofp")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_unsigned_int_to_float: {}", e),
                    })?
                    .into())
            }
        }
        (false, false) => {
            let src_val = src.into_int_value();
            let dst_type = dst_type.into_int_type();
            if src_dtype.bytes() > dst_dtype.bytes() {
                Ok(builder
                    .build_int_truncate(src_val, dst_type, "trunc")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_truncate: {}", e),
                    })?
                    .into())
            } else if src_is_signed {
                Ok(builder
                    .build_int_s_extend(src_val, dst_type, "sext")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_s_extend: {}", e),
                    })?
                    .into())
            } else {
                Ok(builder
                    .build_int_z_extend(src_val, dst_type, "zext")
                    .map_err(|e| crate::Error::LlvmError {
                        reason: format!("build_int_z_extend: {}", e),
                    })?
                    .into())
            }
        }
    }
}

/// Generate a load instruction.
fn codegen_load<'ctx>(
    buffer_ptr: BasicValueEnum<'ctx>,
    index: BasicValueEnum<'ctx>,
    result_dtype: &morok_dtype::DType,
    context: &'ctx Context,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let ptr_val = buffer_ptr.into_pointer_value();
    let index_val = index.into_int_value();

    // GEP to get the element pointer
    let element_ptr = unsafe {
        builder
            .build_gep(
                dtype_to_basic_type(result_dtype, context),
                ptr_val,
                &[index_val],
                "gep",
            )
            .map_err(|e| crate::Error::LlvmError {
                reason: format!("build_gep: {}", e),
            })?
    };

    // Load from the pointer
    let loaded = builder
        .build_load(dtype_to_basic_type(result_dtype, context), element_ptr, "load")
        .map_err(|e| crate::Error::LlvmError {
            reason: format!("build_load: {}", e),
        })?;

    Ok(loaded)
}

/// Generate a store instruction.
fn codegen_store<'ctx>(
    buffer_ptr: BasicValueEnum<'ctx>,
    index: BasicValueEnum<'ctx>,
    value: BasicValueEnum<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<()> {
    let ptr_val = buffer_ptr.into_pointer_value();
    let index_val = index.into_int_value();

    // GEP to get the element pointer
    let element_ptr = unsafe {
        builder
            .build_gep(
                // Use i8 type for opaque pointer arithmetic
                builder.get_insert_block().unwrap().get_context().i8_type(),
                ptr_val,
                &[index_val],
                "gep",
            )
            .map_err(|e| crate::Error::LlvmError {
                reason: format!("build_gep: {}", e),
            })?
    };

    // Store the value
    builder
        .build_store(element_ptr, value)
        .map_err(|e| crate::Error::LlvmError {
            reason: format!("build_store: {}", e),
        })?;

    Ok(())
}
