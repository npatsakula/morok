//! MLIR operation builders for individual UOp operations.
//!
//! Provides helper functions for building arith + LLVM dialect operations.
//! Control flow (Range/End/If/EndIf) is handled directly in the renderer.

use melior::dialect::{arith, llvm};
use melior::ir::attribute::{FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute};
use melior::ir::block::BlockLike;
use melior::ir::operation::OperationBuilder;
use melior::ir::r#type::IntegerType;
use melior::ir::{Block, BlockRef, Identifier, Location, Type, Value};
use melior::Context;
use morok_dtype::DType;
use morok_ir::{BinaryOp, ReduceOp, TernaryOp, UnaryOp};

use super::types::{intrinsic_type_suffix, mlir_type};

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
    let attr = FloatAttribute::new(ctx, val, float_type);
    block.append_operation(arith::constant(ctx, attr.into(), loc)).result(0).unwrap().into()
}

/// Build an MLIR constant from a ConstValue.
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
            if dtype.is_float() {
                const_float(ctx, block, *i as f64, mlir_ty, loc)
            } else {
                const_int(ctx, block, *i, mlir_ty, loc)
            }
        }
        morok_ir::ConstValue::UInt(u) => {
            if dtype.is_float() {
                const_float(ctx, block, *u as f64, mlir_ty, loc)
            } else {
                const_int(ctx, block, *u as i64, mlir_ty, loc)
            }
        }
        morok_ir::ConstValue::Float(f) => {
            if dtype.is_float() {
                const_float(ctx, block, *f, mlir_ty, loc)
            } else {
                const_int(ctx, block, *f as i64, mlir_ty, loc)
            }
        }
        morok_ir::ConstValue::Bool(b) => {
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

    let mut current: Value<'c, 'c> = block.append_operation(llvm::undef(vec_type, loc)).result(0).unwrap().into();

    for (i, val) in values.iter().enumerate() {
        let scalar = build_const(ctx, block, val, &scalar_dtype, loc);
        let idx = const_i32(ctx, block, i as i64, loc);
        current = block
            .append_operation(llvm::insert_element(current, scalar, idx, loc))
            .result(0)
            .unwrap()
            .into();
    }
    current
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
        let count = dtype.vcount();

        let mut current: Value = block.append_operation(llvm::undef(vec_type, loc)).result(0).unwrap().into();
        for i in 0..count {
            let idx = const_i32(ctx, block, i as i64, loc);
            current = block
                .append_operation(llvm::insert_element(current, scalar_val, idx, loc))
                .result(0)
                .unwrap()
                .into();
        }
        current
    } else {
        build_const(ctx, block, &identity, dtype, loc)
    }
}

// =============================================================================
// LLVM intrinsic calls
// =============================================================================

pub fn call_intrinsic<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    name: &str,
    args: &[Value<'c, 'c>],
    result_type: Type<'c>,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    block
        .append_operation(llvm::call_intrinsic(
            ctx,
            StringAttribute::new(ctx, name),
            args,
            &[result_type],
            loc,
        ))
        .result(0)
        .unwrap()
        .into()
}

// =============================================================================
// Comparisons
// =============================================================================

fn cmpf_predicate_attr<'c>(ctx: &'c Context, pred: &str) -> melior::ir::Attribute<'c> {
    let pred_val: i64 = match pred {
        "false" => 0,
        "oeq" => 1,
        "ogt" => 2,
        "oge" => 3,
        "olt" => 4,
        "ole" => 5,
        "one" => 6,
        "ord" => 7,
        "ueq" => 8,
        "ugt" => 9,
        "uge" => 10,
        "ult" => 11,
        "ule" => 12,
        "une" => 13,
        "uno" => 14,
        "true" => 15,
        _ => panic!("unknown fcmp predicate: {pred}"),
    };
    IntegerAttribute::new(IntegerType::new(ctx, 64).into(), pred_val).into()
}

fn cmpi_predicate_attr<'c>(ctx: &'c Context, pred: &str) -> melior::ir::Attribute<'c> {
    let pred_val: i64 = match pred {
        "eq" => 0,
        "ne" => 1,
        "slt" => 2,
        "sle" => 3,
        "sgt" => 4,
        "sge" => 5,
        "ult" => 6,
        "ule" => 7,
        "ugt" => 8,
        "uge" => 9,
        _ => panic!("unknown icmp predicate: {pred}"),
    };
    IntegerAttribute::new(IntegerType::new(ctx, 64).into(), pred_val).into()
}

pub fn fcmp<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    pred: &str,
    lhs: Value<'c, 'c>,
    rhs: Value<'c, 'c>,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let i1_type = IntegerType::new(ctx, 1).into();
    block
        .append_operation(
            OperationBuilder::new("arith.cmpf", loc)
                .add_attributes(&[(Identifier::new(ctx, "predicate"), cmpf_predicate_attr(ctx, pred))])
                .add_operands(&[lhs, rhs])
                .add_results(&[i1_type])
                .build()
                .expect("valid arith.cmpf"),
        )
        .result(0)
        .unwrap()
        .into()
}

pub fn icmp<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    pred: &str,
    lhs: Value<'c, 'c>,
    rhs: Value<'c, 'c>,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let i1_type = IntegerType::new(ctx, 1).into();
    block
        .append_operation(
            OperationBuilder::new("arith.cmpi", loc)
                .add_attributes(&[(Identifier::new(ctx, "predicate"), cmpi_predicate_attr(ctx, pred))])
                .add_operands(&[lhs, rhs])
                .add_results(&[i1_type])
                .build()
                .expect("valid arith.cmpi"),
        )
        .result(0)
        .unwrap()
        .into()
}

fn cmp<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    dtype: &DType,
    fpred: &str,
    spred: &str,
    upred: &str,
    lhs: Value<'c, 'c>,
    rhs: Value<'c, 'c>,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    if dtype.is_float() {
        fcmp(ctx, block, fpred, lhs, rhs, loc)
    } else if dtype.is_signed() {
        icmp(ctx, block, spred, lhs, rhs, loc)
    } else {
        icmp(ctx, block, upred, lhs, rhs, loc)
    }
}

// =============================================================================
// Branch helpers
// =============================================================================

/// Build `llvm.br ^dest(%args...)`.
pub fn br<'c>(
    block: &Block<'c>,
    dest: &Block<'c>,
    args: &[Value<'c, 'c>],
    loc: Location<'c>,
) {
    block.append_operation(
        OperationBuilder::new("llvm.br", loc)
            .add_successors(&[dest])
            .add_operands(args)
            .build()
            .expect("valid llvm.br"),
    );
}

/// Build `llvm.cond_br %cond, ^true_dest(%true_args...), ^false_dest(%false_args...)`.
pub fn cond_br<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    condition: Value<'c, 'c>,
    true_dest: &Block<'c>,
    false_dest: &Block<'c>,
    true_args: &[Value<'c, 'c>],
    false_args: &[Value<'c, 'c>],
    loc: Location<'c>,
) {
    use melior::ir::attribute::DenseI32ArrayAttribute;

    let mut operands = vec![condition];
    operands.extend_from_slice(true_args);
    operands.extend_from_slice(false_args);

    block.append_operation(
        OperationBuilder::new("llvm.cond_br", loc)
            .add_operands(&operands)
            .add_successors(&[true_dest, false_dest])
            .add_attributes(&[(
                Identifier::new(ctx, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(ctx, &[1, true_args.len() as i32, false_args.len() as i32]).into(),
            )])
            .build()
            .expect("valid llvm.cond_br"),
    );
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

    match op {
        BinaryOp::Add if is_f => block.append_operation(arith::addf(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Add => block.append_operation(arith::addi(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Sub if is_f => block.append_operation(arith::subf(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Sub => block.append_operation(arith::subi(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Mul if is_f => block.append_operation(arith::mulf(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Mul => block.append_operation(arith::muli(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Fdiv => block.append_operation(arith::divf(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Idiv if is_s => block.append_operation(arith::divsi(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Idiv => block.append_operation(arith::divui(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Mod if is_f => block.append_operation(arith::remf(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Mod if is_s => block.append_operation(arith::remsi(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Mod => block.append_operation(arith::remui(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Max if is_f => {
            let suffix = intrinsic_type_suffix(dtype);
            call_intrinsic(ctx, block, &format!("llvm.maxnum.{suffix}"), &[lhs, rhs], result_type, loc)
        }
        BinaryOp::Max => {
            let p = if is_s { "sgt" } else { "ugt" };
            let c = icmp(ctx, block, p, lhs, rhs, loc);
            block.append_operation(arith::select(c, lhs, rhs, loc)).result(0).unwrap().into()
        }
        BinaryOp::Pow if is_f => {
            let suffix = intrinsic_type_suffix(dtype);
            call_intrinsic(ctx, block, &format!("llvm.pow.{suffix}"), &[lhs, rhs], result_type, loc)
        }
        BinaryOp::Pow => {
            let f64_type = Type::float64(ctx);
            let lf = block.append_operation(arith::sitofp(lhs, f64_type, loc)).result(0).unwrap().into();
            let rf = block.append_operation(arith::sitofp(rhs, f64_type, loc)).result(0).unwrap().into();
            let pf = call_intrinsic(ctx, block, "llvm.pow.f64", &[lf, rf], f64_type, loc);
            block.append_operation(arith::fptosi(pf, result_type, loc)).result(0).unwrap().into()
        }
        BinaryOp::Lt => cmp(ctx, block, dtype, "olt", "slt", "ult", lhs, rhs, loc),
        BinaryOp::Le => cmp(ctx, block, dtype, "ole", "sle", "ule", lhs, rhs, loc),
        BinaryOp::Gt => cmp(ctx, block, dtype, "ogt", "sgt", "ugt", lhs, rhs, loc),
        BinaryOp::Ge => cmp(ctx, block, dtype, "oge", "sge", "uge", lhs, rhs, loc),
        BinaryOp::Eq => cmp(ctx, block, dtype, "oeq", "eq", "eq", lhs, rhs, loc),
        BinaryOp::Ne => cmp(ctx, block, dtype, "une", "ne", "ne", lhs, rhs, loc),
        BinaryOp::And => block.append_operation(arith::andi(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Or => block.append_operation(arith::ori(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Xor | BinaryOp::Threefry => {
            block.append_operation(arith::xori(lhs, rhs, loc)).result(0).unwrap().into()
        }
        BinaryOp::Shl => block.append_operation(arith::shli(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Shr if is_s => block.append_operation(arith::shrsi(lhs, rhs, loc)).result(0).unwrap().into(),
        BinaryOp::Shr => block.append_operation(arith::shrui(lhs, rhs, loc)).result(0).unwrap().into(),
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
    let suffix = intrinsic_type_suffix(dtype);

    match op {
        UnaryOp::Neg if is_f => block.append_operation(arith::negf(src, loc)).result(0).unwrap().into(),
        UnaryOp::Neg => {
            let zero = const_int(ctx, block, 0, result_type, loc);
            block.append_operation(arith::subi(zero, src, loc)).result(0).unwrap().into()
        }
        UnaryOp::Not => {
            let mask = if dtype.is_bool() {
                const_int(ctx, block, 1, IntegerType::new(ctx, 1).into(), loc)
            } else {
                const_int(ctx, block, -1, result_type, loc)
            };
            block.append_operation(arith::xori(src, mask, loc)).result(0).unwrap().into()
        }
        UnaryOp::Sqrt => call_intrinsic(ctx, block, &format!("llvm.sqrt.{suffix}"), &[src], result_type, loc),
        UnaryOp::Exp => call_intrinsic(ctx, block, &format!("llvm.exp.{suffix}"), &[src], result_type, loc),
        UnaryOp::Exp2 => call_intrinsic(ctx, block, &format!("llvm.exp2.{suffix}"), &[src], result_type, loc),
        UnaryOp::Log => call_intrinsic(ctx, block, &format!("llvm.log.{suffix}"), &[src], result_type, loc),
        UnaryOp::Log2 => call_intrinsic(ctx, block, &format!("llvm.log2.{suffix}"), &[src], result_type, loc),
        UnaryOp::Sin => call_intrinsic(ctx, block, &format!("llvm.sin.{suffix}"), &[src], result_type, loc),
        UnaryOp::Cos => call_intrinsic(ctx, block, &format!("llvm.cos.{suffix}"), &[src], result_type, loc),
        UnaryOp::Floor => call_intrinsic(ctx, block, &format!("llvm.floor.{suffix}"), &[src], result_type, loc),
        UnaryOp::Ceil => call_intrinsic(ctx, block, &format!("llvm.ceil.{suffix}"), &[src], result_type, loc),
        UnaryOp::Trunc => call_intrinsic(ctx, block, &format!("llvm.trunc.{suffix}"), &[src], result_type, loc),
        UnaryOp::Round => call_intrinsic(ctx, block, &format!("llvm.round.{suffix}"), &[src], result_type, loc),
        UnaryOp::Abs if is_f => {
            call_intrinsic(ctx, block, &format!("llvm.fabs.{suffix}"), &[src], result_type, loc)
        }
        UnaryOp::Abs => {
            let poison = const_int(ctx, block, 1, IntegerType::new(ctx, 1).into(), loc);
            call_intrinsic(ctx, block, &format!("llvm.abs.{suffix}"), &[src, poison], result_type, loc)
        }
        UnaryOp::Rsqrt => {
            let sqrt_val = call_intrinsic(ctx, block, &format!("llvm.sqrt.{suffix}"), &[src], result_type, loc);
            let one = const_float(ctx, block, 1.0, result_type, loc);
            block.append_operation(arith::divf(one, sqrt_val, loc)).result(0).unwrap().into()
        }
        UnaryOp::Reciprocal => {
            let one = const_float(ctx, block, 1.0, result_type, loc);
            block.append_operation(arith::divf(one, src, loc)).result(0).unwrap().into()
        }
        UnaryOp::Tan => {
            let sin_val = call_intrinsic(ctx, block, &format!("llvm.sin.{suffix}"), &[src], result_type, loc);
            let cos_val = call_intrinsic(ctx, block, &format!("llvm.cos.{suffix}"), &[src], result_type, loc);
            block.append_operation(arith::divf(sin_val, cos_val, loc)).result(0).unwrap().into()
        }
        UnaryOp::Sign if is_f => {
            let zero = const_float(ctx, block, 0.0, result_type, loc);
            let gt = fcmp(ctx, block, "ogt", src, zero, loc);
            let lt = fcmp(ctx, block, "olt", src, zero, loc);
            let gt_f = block.append_operation(arith::uitofp(gt, result_type, loc)).result(0).unwrap().into();
            let lt_f = block.append_operation(arith::uitofp(lt, result_type, loc)).result(0).unwrap().into();
            block.append_operation(arith::subf(gt_f, lt_f, loc)).result(0).unwrap().into()
        }
        UnaryOp::Sign => {
            let zero = const_int(ctx, block, 0, result_type, loc);
            let (gt_pred, lt_pred) = if dtype.is_signed() { ("sgt", "slt") } else { ("ugt", "ult") };
            let gt = icmp(ctx, block, gt_pred, src, zero, loc);
            let lt = icmp(ctx, block, lt_pred, src, zero, loc);
            let gt_ext = block.append_operation(arith::extui(gt, result_type, loc)).result(0).unwrap().into();
            let lt_ext = block.append_operation(arith::extui(lt, result_type, loc)).result(0).unwrap().into();
            block.append_operation(arith::subi(gt_ext, lt_ext, loc)).result(0).unwrap().into()
        }
        UnaryOp::Erf => call_intrinsic(ctx, block, &format!("llvm.erf.{suffix}"), &[src], result_type, loc),
        UnaryOp::Square if is_f => block.append_operation(arith::mulf(src, src, loc)).result(0).unwrap().into(),
        UnaryOp::Square => block.append_operation(arith::muli(src, src, loc)).result(0).unwrap().into(),
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

    // Ptr <-> Ptr
    if matches!(from_dtype, DType::Ptr { .. }) && matches!(to_dtype, DType::Ptr { .. }) {
        return src;
    }
    // Ptr -> Int
    if matches!(from_dtype, DType::Ptr { .. }) {
        return block
            .append_operation(
                OperationBuilder::new("llvm.ptrtoint", loc)
                    .add_operands(&[src])
                    .add_results(&[to_type])
                    .build()
                    .expect("valid ptrtoint"),
            )
            .result(0)
            .unwrap()
            .into();
    }
    // Int -> Ptr
    if matches!(to_dtype, DType::Ptr { .. }) {
        return block
            .append_operation(
                OperationBuilder::new("llvm.inttoptr", loc)
                    .add_operands(&[src])
                    .add_results(&[to_type])
                    .build()
                    .expect("valid inttoptr"),
            )
            .result(0)
            .unwrap()
            .into();
    }

    // Float -> Float
    if from_scalar.is_float() && to_scalar.is_float() {
        return if to_scalar.bytes() > from_scalar.bytes() {
            block.append_operation(arith::extf(src, to_type, loc)).result(0).unwrap().into()
        } else if to_scalar.bytes() < from_scalar.bytes() {
            block.append_operation(arith::truncf(src, to_type, loc)).result(0).unwrap().into()
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
// Ternary operations
// =============================================================================

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
    match op {
        TernaryOp::Where => block.append_operation(arith::select(a, b, c, loc)).result(0).unwrap().into(),
        TernaryOp::MulAcc => {
            if dtype.is_float() {
                let suffix = intrinsic_type_suffix(dtype);
                let result_type = mlir_type(ctx, dtype);
                call_intrinsic(ctx, block, &format!("llvm.fmuladd.{suffix}"), &[a, b, c], result_type, loc)
            } else {
                let mul = block.append_operation(arith::muli(a, b, loc)).result(0).unwrap().into();
                block.append_operation(arith::addi(mul, c, loc)).result(0).unwrap().into()
            }
        }
    }
}

// =============================================================================
// Reduce accumulation
// =============================================================================

/// Build the accumulation step for a reduce operation: load acc, accumulate, store back.
pub fn render_reduce_accumulate<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    reduce_op: ReduceOp,
    src: Value<'c, 'c>,
    acc_ptr: Value<'c, 'c>,
    dtype: &DType,
    loc: Location<'c>,
) {
    let acc_type = mlir_type(ctx, dtype);
    let acc_load: Value = block
        .append_operation(llvm::load(ctx, acc_ptr, acc_type, loc, Default::default()))
        .result(0)
        .unwrap()
        .into();

    let acc_new = match reduce_op {
        ReduceOp::Add if dtype.is_float() => {
            block.append_operation(arith::addf(acc_load, src, loc)).result(0).unwrap().into()
        }
        ReduceOp::Add => block.append_operation(arith::addi(acc_load, src, loc)).result(0).unwrap().into(),
        ReduceOp::Mul if dtype.is_float() => {
            block.append_operation(arith::mulf(acc_load, src, loc)).result(0).unwrap().into()
        }
        ReduceOp::Mul => block.append_operation(arith::muli(acc_load, src, loc)).result(0).unwrap().into(),
        ReduceOp::Max if dtype.is_float() => {
            let suffix = intrinsic_type_suffix(dtype);
            call_intrinsic(ctx, block, &format!("llvm.maxnum.{suffix}"), &[acc_load, src], acc_type, loc)
        }
        ReduceOp::Max => {
            let p = if dtype.is_signed() { "sgt" } else { "ugt" };
            let c = icmp(ctx, block, p, acc_load, src, loc);
            block.append_operation(arith::select(c, acc_load, src, loc)).result(0).unwrap().into()
        }
        ReduceOp::Min if dtype.is_float() => {
            let suffix = intrinsic_type_suffix(dtype);
            call_intrinsic(ctx, block, &format!("llvm.minnum.{suffix}"), &[acc_load, src], acc_type, loc)
        }
        ReduceOp::Min => {
            let p = if dtype.is_signed() { "slt" } else { "ult" };
            let c = icmp(ctx, block, p, acc_load, src, loc);
            block.append_operation(arith::select(c, acc_load, src, loc)).result(0).unwrap().into()
        }
    };

    block.append_operation(llvm::store(ctx, acc_new, acc_ptr, loc, Default::default()));
}

// =============================================================================
// Vector operations
// =============================================================================

pub fn render_vectorize<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    elements: &[Value<'c, 'c>],
    vec_type: Type<'c>,
    scalar_type: Type<'c>,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let mut current: Value = block.append_operation(llvm::undef(vec_type, loc)).result(0).unwrap().into();
    for (i, &elem) in elements.iter().enumerate() {
        let idx = const_i32(ctx, block, i as i64, loc);
        current = block
            .append_operation(llvm::insert_element(current, elem, idx, loc))
            .result(0)
            .unwrap()
            .into();
    }
    current
}

pub fn render_extractelement<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    vec: Value<'c, 'c>,
    index: usize,
    result_type: Type<'c>,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    let idx = const_i32(ctx, block, index as i64, loc);
    block
        .append_operation(llvm::extract_element(vec, idx, result_type, loc))
        .result(0)
        .unwrap()
        .into()
}
