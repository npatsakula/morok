//! MLIR operation builders for individual UOp operations.
//!
//! Provides helper functions for building arith + LLVM dialect operations.
//! Control flow (Range/End/If/EndIf) is handled directly in the renderer.

use melior::Context;
use melior::dialect::ods::vector;
use melior::dialect::{arith, arith::CmpfPredicate, arith::CmpiPredicate, ods};
use melior::ir::attribute::{FloatAttribute, IntegerAttribute};
use melior::ir::block::BlockLike;
use melior::ir::operation::OperationBuilder;
use melior::ir::r#type::IntegerType;
use melior::ir::{Block, Location, Type, Value};
use morok_dtype::DType;
use morok_ir::{BinaryOp, ReduceOp, TernaryOp, UnaryOp};

use super::types::mlir_type;

// =============================================================================
// Constants
// =============================================================================

pub fn const_int<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    val: i64,
    int_type: Type<'c>,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let attr = IntegerAttribute::new(int_type, val);
    block.append_operation(arith::constant(ctx, attr.into(), loc)).result(0).unwrap().into()
}

pub fn const_i64<'c>(ctx: &'c Context, block: &Block<'c>, val: i64, loc: Location<'c>) -> Value<'c, 'c> {
    const_int(ctx, block, val, IntegerType::new(ctx, 64).into(), loc)
}

pub fn const_i32<'c>(ctx: &'c Context, block: &Block<'c>, val: i64, loc: Location<'c>) -> Value<'c, 'c> {
    const_int(ctx, block, val, IntegerType::new(ctx, 32).into(), loc)
}

pub fn const_float<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    val: f64,
    float_type: Type<'c>,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let attr = FloatAttribute::new(ctx, float_type, val);
    block.append_operation(arith::constant(ctx, attr.into(), loc)).result(0).unwrap().into()
}

/// Build an MLIR constant from a ConstValue.
///
/// The ConstValue variant must match the target dtype — the IR always produces
/// correctly-typed constants (via `ConstValue::zero`, `reduce_identity`, etc.).
pub fn build_const<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    val: &morok_ir::ConstValue,
    dtype: &DType,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let mlir_ty = mlir_type(ctx, dtype);

    match val {
        morok_ir::ConstValue::Int(i) => {
            debug_assert!(!dtype.is_float(), "Int ConstValue with float dtype {dtype:?}");
            const_int(ctx, block, *i, mlir_ty, loc)
        }
        morok_ir::ConstValue::UInt(u) => {
            debug_assert!(!dtype.is_float(), "UInt ConstValue with float dtype {dtype:?}");
            const_int(ctx, block, *u as i64, mlir_ty, loc)
        }
        morok_ir::ConstValue::Float(f) => {
            debug_assert!(dtype.is_float(), "Float ConstValue with non-float dtype {dtype:?}");
            const_float(ctx, block, *f, mlir_ty, loc)
        }
        morok_ir::ConstValue::Bool(b) => {
            debug_assert!(dtype.is_bool(), "Bool ConstValue with non-bool dtype {dtype:?}");
            const_int(ctx, block, i64::from(*b), IntegerType::new(ctx, 1).into(), loc)
        }
    }
}

/// Build a vector constant.
pub fn build_vconst<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    values: &[morok_ir::ConstValue],
    dtype: &DType,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let vec_type = mlir_type(ctx, dtype);
    let scalar_dtype = dtype.scalar_dtype();

    let scalars: Vec<Value> = values.iter().map(|val| build_const(ctx, block, val, &scalar_dtype, loc)).collect();
    block.append_operation(vector::from_elements(ctx, vec_type, &scalars, loc).into()).result(0).unwrap().into()
}

/// Build a reduce identity constant.
pub fn build_reduce_identity<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    reduce_op: ReduceOp,
    dtype: &DType,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let identity = super::types::reduce_identity_value(reduce_op, dtype);

    if matches!(dtype, DType::Vector { .. }) {
        let scalar_dtype = dtype.scalar_dtype();
        let scalar_val = build_const(ctx, block, &identity, &scalar_dtype, loc);
        let vec_type = mlir_type(ctx, dtype);
        block.append_operation(vector::splat(ctx, vec_type, scalar_val, loc).into()).result(0).unwrap().into()
    } else {
        build_const(ctx, block, &identity, dtype, loc)
    }
}

// =============================================================================
// Binary operations
// =============================================================================

pub fn render_binary<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    op: BinaryOp,
    lhs: Value<'c, 'c>,
    rhs: Value<'c, 'c>,
    dtype: &DType,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let result_type = mlir_type(ctx, dtype);
    let is_f = dtype.is_float();
    let is_s = dtype.is_signed();
    let emit = |op| block.append_operation(op).result(0).unwrap().into();
    let cmp = |fpred, spred, upred| {
        let op = if is_f {
            arith::cmpf(ctx, fpred, lhs, rhs, loc)
        } else if is_s {
            arith::cmpi(ctx, spred, lhs, rhs, loc)
        } else {
            arith::cmpi(ctx, upred, lhs, rhs, loc)
        };
        block.append_operation(op).result(0).unwrap().into()
    };

    match op {
        BinaryOp::Add if is_f => emit(arith::addf(lhs, rhs, loc)),
        BinaryOp::Add => emit(arith::addi(lhs, rhs, loc)),
        BinaryOp::Sub if is_f => emit(arith::subf(lhs, rhs, loc)),
        BinaryOp::Sub => emit(arith::subi(lhs, rhs, loc)),
        BinaryOp::Mul if is_f => emit(arith::mulf(lhs, rhs, loc)),
        BinaryOp::Mul => emit(arith::muli(lhs, rhs, loc)),
        BinaryOp::Fdiv => emit(arith::divf(lhs, rhs, loc)),
        BinaryOp::Idiv if is_s => emit(arith::divsi(lhs, rhs, loc)),
        BinaryOp::Idiv => emit(arith::divui(lhs, rhs, loc)),
        BinaryOp::Mod if is_f => emit(arith::remf(lhs, rhs, loc)),
        BinaryOp::Mod if is_s => emit(arith::remsi(lhs, rhs, loc)),
        BinaryOp::Mod => emit(arith::remui(lhs, rhs, loc)),
        BinaryOp::Max if is_f => emit(ods::arith::maxnumf(ctx, lhs, rhs, loc).into()),
        BinaryOp::Max => {
            let p = if is_s { CmpiPredicate::Sgt } else { CmpiPredicate::Ugt };
            let c = emit(arith::cmpi(ctx, p, lhs, rhs, loc));
            emit(arith::select(c, lhs, rhs, loc))
        }
        BinaryOp::Pow if is_f => emit(ods::math::powf(ctx, lhs, rhs, loc).into()),
        BinaryOp::Pow => {
            let f64_type = Type::float64(ctx);
            let lf = emit(arith::sitofp(lhs, f64_type, loc));
            let rf = emit(arith::sitofp(rhs, f64_type, loc));
            let pf = emit(ods::math::powf(ctx, lf, rf, loc).into());
            emit(arith::fptosi(pf, result_type, loc))
        }
        BinaryOp::Lt => cmp(CmpfPredicate::Olt, CmpiPredicate::Slt, CmpiPredicate::Ult),
        BinaryOp::Le => cmp(CmpfPredicate::Ole, CmpiPredicate::Sle, CmpiPredicate::Ule),
        BinaryOp::Gt => cmp(CmpfPredicate::Ogt, CmpiPredicate::Sgt, CmpiPredicate::Ugt),
        BinaryOp::Ge => cmp(CmpfPredicate::Oge, CmpiPredicate::Sge, CmpiPredicate::Uge),
        BinaryOp::Eq => cmp(CmpfPredicate::Oeq, CmpiPredicate::Eq, CmpiPredicate::Eq),
        BinaryOp::Ne => cmp(CmpfPredicate::Une, CmpiPredicate::Ne, CmpiPredicate::Ne),
        BinaryOp::And => emit(arith::andi(lhs, rhs, loc)),
        BinaryOp::Or => emit(arith::ori(lhs, rhs, loc)),
        BinaryOp::Xor | BinaryOp::Threefry => emit(arith::xori(lhs, rhs, loc)),
        BinaryOp::Shl => emit(arith::shli(lhs, rhs, loc)),
        BinaryOp::Shr if is_s => emit(arith::shrsi(lhs, rhs, loc)),
        BinaryOp::Shr => emit(arith::shrui(lhs, rhs, loc)),
    }
}

// =============================================================================
// Unary operations
// =============================================================================

pub fn render_unary<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    op: UnaryOp,
    src: Value<'c, 'c>,
    dtype: &DType,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let result_type = mlir_type(ctx, dtype);
    let is_f = dtype.is_float();
    let emit = |op| block.append_operation(op).result(0).unwrap().into();

    match op {
        UnaryOp::Neg if is_f => emit(arith::negf(src, loc)),
        UnaryOp::Neg => {
            let zero = const_int(ctx, block, 0, result_type, loc);
            emit(arith::subi(zero, src, loc))
        }
        UnaryOp::Not => {
            let mask = if dtype.is_bool() {
                const_int(ctx, block, 1, IntegerType::new(ctx, 1).into(), loc)
            } else {
                const_int(ctx, block, -1, result_type, loc)
            };
            emit(arith::xori(src, mask, loc))
        }
        UnaryOp::Sqrt => emit(ods::math::sqrt(ctx, src, loc).into()),
        UnaryOp::Exp => emit(ods::math::exp(ctx, src, loc).into()),
        UnaryOp::Exp2 => emit(ods::math::exp_2(ctx, src, loc).into()),
        UnaryOp::Log => emit(ods::math::log(ctx, src, loc).into()),
        UnaryOp::Log2 => emit(ods::math::log_2(ctx, src, loc).into()),
        UnaryOp::Sin => emit(ods::math::sin(ctx, src, loc).into()),
        UnaryOp::Cos => emit(ods::math::cos(ctx, src, loc).into()),
        UnaryOp::Floor => emit(ods::math::floor(ctx, src, loc).into()),
        UnaryOp::Ceil => emit(ods::math::ceil(ctx, src, loc).into()),
        UnaryOp::Trunc => emit(ods::math::trunc(ctx, src, loc).into()),
        UnaryOp::Round => emit(ods::math::round(ctx, src, loc).into()),
        UnaryOp::Abs if is_f => emit(ods::math::absf(ctx, src, loc).into()),
        UnaryOp::Abs => emit(ods::math::absi(ctx, src, loc).into()),
        UnaryOp::Rsqrt => emit(ods::math::rsqrt(ctx, src, loc).into()),
        UnaryOp::Reciprocal => {
            let one = const_float(ctx, block, 1.0, result_type, loc);
            emit(arith::divf(one, src, loc))
        }
        UnaryOp::Tan => emit(ods::math::tan(ctx, src, loc).into()),
        UnaryOp::Sign if is_f => {
            let zero = const_float(ctx, block, 0.0, result_type, loc);
            let gt = emit(arith::cmpf(ctx, CmpfPredicate::Ogt, src, zero, loc));
            let lt = emit(arith::cmpf(ctx, CmpfPredicate::Olt, src, zero, loc));
            let gt_f = emit(arith::uitofp(gt, result_type, loc));
            let lt_f = emit(arith::uitofp(lt, result_type, loc));
            emit(arith::subf(gt_f, lt_f, loc))
        }
        UnaryOp::Sign => {
            let zero = const_int(ctx, block, 0, result_type, loc);
            let (gt_pred, lt_pred) = if dtype.is_signed() {
                (CmpiPredicate::Sgt, CmpiPredicate::Slt)
            } else {
                (CmpiPredicate::Ugt, CmpiPredicate::Ult)
            };
            let gt = emit(arith::cmpi(ctx, gt_pred, src, zero, loc));
            let lt = emit(arith::cmpi(ctx, lt_pred, src, zero, loc));
            let gt_ext = emit(arith::extui(gt, result_type, loc));
            let lt_ext = emit(arith::extui(lt, result_type, loc));
            emit(arith::subi(gt_ext, lt_ext, loc))
        }
        UnaryOp::Erf => emit(ods::math::erf(ctx, src, loc).into()),
        UnaryOp::Square if is_f => emit(arith::mulf(src, src, loc)),
        UnaryOp::Square => emit(arith::muli(src, src, loc)),
    }
}

// =============================================================================
// Cast operations
// =============================================================================

pub fn render_cast<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    src: Value<'c, 'c>,
    from_dtype: &DType,
    to_dtype: &DType,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let to_type = mlir_type(ctx, to_dtype);
    let from_scalar = from_dtype.base();
    let to_scalar = to_dtype.base();

    // Ptr <-> Ptr: no-op, pointers are opaque in LLVM
    if matches!(from_dtype, DType::Ptr { .. }) && matches!(to_dtype, DType::Ptr { .. }) {
        return src;
    }
    // Ptr -> scalar or scalar -> Ptr: invalid IR — ensure_scalar produces Cast(Ptr<T> → T)
    // which should be lowered as a load, not a ptrtoint/inttoptr reinterpretation.
    if matches!(from_dtype, DType::Ptr { .. }) || matches!(to_dtype, DType::Ptr { .. }) {
        unreachable!("Cast between Ptr and scalar types is invalid IR: {from_dtype:?} → {to_dtype:?}");
    }

    // Float -> Float
    if from_scalar.is_float() && to_scalar.is_float() {
        return if to_scalar.bytes() > from_scalar.bytes() {
            block.append_operation(arith::extf(src, to_type, loc)).result(0).unwrap().into()
        } else if to_scalar.bytes() < from_scalar.bytes() {
            block.append_operation(arith::truncf(src, loc)).result(0).unwrap().into()
        } else {
            block.append_operation(arith::bitcast(src, to_type, loc)).result(0).unwrap().into()
        };
    }

    // Int/Bool -> Float
    if !from_scalar.is_float() && to_scalar.is_float() {
        return if from_scalar.is_unsigned() || from_scalar.is_bool() {
            block.append_operation(arith::uitofp(src, to_type, loc)).result(0).unwrap().into()
        } else {
            block.append_operation(arith::sitofp(src, to_type, loc)).result(0).unwrap().into()
        };
    }

    // Float -> Int
    if from_scalar.is_float() && !to_scalar.is_float() {
        return if to_scalar.is_unsigned() {
            block.append_operation(arith::fptoui(src, to_type, loc)).result(0).unwrap().into()
        } else {
            block.append_operation(arith::fptosi(src, to_type, loc)).result(0).unwrap().into()
        };
    }

    // Int -> Int
    let from_bits = if from_scalar.is_bool() { 1 } else { from_scalar.bytes() as u32 * 8 };
    let to_bits = if to_scalar.is_bool() { 1 } else { to_scalar.bytes() as u32 * 8 };

    if from_bits == to_bits {
        block.append_operation(arith::bitcast(src, to_type, loc)).result(0).unwrap().into()
    } else if to_bits < from_bits {
        block.append_operation(arith::trunci(src, to_type, loc)).result(0).unwrap().into()
    } else if from_scalar.is_unsigned() || from_scalar.is_bool() {
        block.append_operation(arith::extui(src, to_type, loc)).result(0).unwrap().into()
    } else {
        block.append_operation(arith::extsi(src, to_type, loc)).result(0).unwrap().into()
    }
}

// =============================================================================
// Vector element insertion/extraction
// =============================================================================

/// Build llvm.insertelement operation manually (not provided by melior dialect helpers).
pub fn insert_element<'c>(
    vec: Value<'c, 'c>,
    elem: Value<'c, 'c>,
    index: Value<'c, 'c>,
    vec_type: Type<'c>,
    loc: Location<'c>,
) -> melior::ir::operation::Operation<'c> {
    OperationBuilder::new("llvm.insertelement", loc)
        .add_operands(&[vec, elem, index])
        .add_results(&[vec_type])
        .build()
        .expect("valid llvm.insertelement")
}

// =============================================================================
// Ternary operations
// =============================================================================

#[allow(clippy::too_many_arguments)]
pub fn render_ternary<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    op: TernaryOp,
    a: Value<'c, 'c>,
    b: Value<'c, 'c>,
    c: Value<'c, 'c>,
    dtype: &DType,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let emit = |op| block.append_operation(op).result(0).unwrap().into();
    match op {
        TernaryOp::Where => emit(arith::select(a, b, c, loc)),
        TernaryOp::MulAcc => {
            if dtype.is_float() {
                emit(ods::math::fma(ctx, a, b, c, loc).into())
            } else {
                let mul = emit(arith::muli(a, b, loc));
                emit(arith::addi(mul, c, loc))
            }
        }
    }
}

// =============================================================================
// Reduce accumulation
// =============================================================================

/// Compute the new accumulator value: `acc_new = reduce_op(acc, src)`. Pure SSA, no memory ops.
pub fn render_reduce_accumulate<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    reduce_op: ReduceOp,
    src: Value<'c, 'c>,
    acc: Value<'c, 'c>,
    dtype: &DType,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let emit = |op| block.append_operation(op).result(0).unwrap().into();
    match reduce_op {
        ReduceOp::Add if dtype.is_float() => emit(arith::addf(acc, src, loc)),
        ReduceOp::Add => emit(arith::addi(acc, src, loc)),
        ReduceOp::Mul if dtype.is_float() => emit(arith::mulf(acc, src, loc)),
        ReduceOp::Mul => emit(arith::muli(acc, src, loc)),
        ReduceOp::Max if dtype.is_float() => emit(ods::arith::maxnumf(ctx, acc, src, loc).into()),
        ReduceOp::Max => {
            let p = if dtype.is_signed() { CmpiPredicate::Sgt } else { CmpiPredicate::Ugt };
            let c = emit(arith::cmpi(ctx, p, acc, src, loc));
            emit(arith::select(c, acc, src, loc))
        }
        ReduceOp::Min if dtype.is_float() => emit(ods::arith::minnumf(ctx, acc, src, loc).into()),
        ReduceOp::Min => {
            let p = if dtype.is_signed() { CmpiPredicate::Slt } else { CmpiPredicate::Ult };
            let c = emit(arith::cmpi(ctx, p, acc, src, loc));
            emit(arith::select(c, acc, src, loc))
        }
    }
}

// =============================================================================
// Vector operations
// =============================================================================

pub fn render_vectorize<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    elements: &[Value<'c, 'c>],
    vec_type: Type<'c>,
    _scalar_type: Type<'c>,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    block.append_operation(vector::from_elements(ctx, vec_type, elements, loc).into()).result(0).unwrap().into()
}

pub fn render_extractelement<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    vec: Value<'c, 'c>,
    index: usize,
    rtype: Type<'c>,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let idx = const_i32(ctx, block, index as i64, loc);
    let op = vector::ExtractElementOperation::builder(ctx, loc).result(rtype).vector(vec).position(idx).build();
    block.append_operation(op.into()).result(0).unwrap().into()
}
