//! Type-aware LLVM builder helpers.
//!
//! These helpers dispatch on float/int/signed types to reduce boilerplate in ops.rs.

use crate::llvm::error::{ArithmeticSnafu, ComparisonSnafu, Result};
use inkwell::builder::Builder;
use inkwell::values::BasicValueEnum;
use inkwell::{FloatPredicate, IntPredicate};
use morok_ir::BinaryOp;
use snafu::ResultExt;

/// Build an addition, dispatching on float/int.
pub fn build_add<'ctx>(
    builder: &Builder<'ctx>,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    is_float: bool,
) -> Result<BasicValueEnum<'ctx>> {
    if is_float {
        Ok(builder
            .build_float_add(lhs.into_float_value(), rhs.into_float_value(), "add")
            .context(ArithmeticSnafu)?
            .into())
    } else {
        Ok(builder.build_int_add(lhs.into_int_value(), rhs.into_int_value(), "add").context(ArithmeticSnafu)?.into())
    }
}

/// Build a subtraction, dispatching on float/int.
pub fn build_sub<'ctx>(
    builder: &Builder<'ctx>,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    is_float: bool,
) -> Result<BasicValueEnum<'ctx>> {
    if is_float {
        Ok(builder
            .build_float_sub(lhs.into_float_value(), rhs.into_float_value(), "sub")
            .context(ArithmeticSnafu)?
            .into())
    } else {
        Ok(builder.build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "sub").context(ArithmeticSnafu)?.into())
    }
}

/// Build a multiplication, dispatching on float/int.
pub fn build_mul<'ctx>(
    builder: &Builder<'ctx>,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    is_float: bool,
) -> Result<BasicValueEnum<'ctx>> {
    if is_float {
        Ok(builder
            .build_float_mul(lhs.into_float_value(), rhs.into_float_value(), "mul")
            .context(ArithmeticSnafu)?
            .into())
    } else {
        Ok(builder.build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "mul").context(ArithmeticSnafu)?.into())
    }
}

/// Build a negation, dispatching on float/int.
pub fn build_neg<'ctx>(
    builder: &Builder<'ctx>,
    src: BasicValueEnum<'ctx>,
    is_float: bool,
) -> Result<BasicValueEnum<'ctx>> {
    if is_float {
        Ok(builder.build_float_neg(src.into_float_value(), "neg").context(ArithmeticSnafu)?.into())
    } else {
        Ok(builder.build_int_neg(src.into_int_value(), "neg").context(ArithmeticSnafu)?.into())
    }
}

/// Build an integer division, dispatching on signed/unsigned.
pub fn build_int_div<'ctx>(
    builder: &Builder<'ctx>,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    is_signed: bool,
) -> Result<BasicValueEnum<'ctx>> {
    if is_signed {
        Ok(builder
            .build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "idiv")
            .context(ArithmeticSnafu)?
            .into())
    } else {
        Ok(builder
            .build_int_unsigned_div(lhs.into_int_value(), rhs.into_int_value(), "idiv")
            .context(ArithmeticSnafu)?
            .into())
    }
}

/// Build a remainder, dispatching on float/int and signed/unsigned.
pub fn build_rem<'ctx>(
    builder: &Builder<'ctx>,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    is_float: bool,
    is_signed: bool,
) -> Result<BasicValueEnum<'ctx>> {
    if is_float {
        Ok(builder
            .build_float_rem(lhs.into_float_value(), rhs.into_float_value(), "mod")
            .context(ArithmeticSnafu)?
            .into())
    } else if is_signed {
        Ok(builder
            .build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "mod")
            .context(ArithmeticSnafu)?
            .into())
    } else {
        Ok(builder
            .build_int_unsigned_rem(lhs.into_int_value(), rhs.into_int_value(), "mod")
            .context(ArithmeticSnafu)?
            .into())
    }
}

/// Get comparison predicates for a binary comparison op.
/// Returns `(float_pred, signed_int_pred, unsigned_int_pred)`.
fn comparison_predicates(op: BinaryOp) -> Option<(FloatPredicate, IntPredicate, IntPredicate)> {
    Some(match op {
        BinaryOp::Lt => (FloatPredicate::OLT, IntPredicate::SLT, IntPredicate::ULT),
        BinaryOp::Le => (FloatPredicate::OLE, IntPredicate::SLE, IntPredicate::ULE),
        BinaryOp::Gt => (FloatPredicate::OGT, IntPredicate::SGT, IntPredicate::UGT),
        BinaryOp::Ge => (FloatPredicate::OGE, IntPredicate::SGE, IntPredicate::UGE),
        BinaryOp::Eq => (FloatPredicate::OEQ, IntPredicate::EQ, IntPredicate::EQ),
        BinaryOp::Ne => (FloatPredicate::ONE, IntPredicate::NE, IntPredicate::NE),
        _ => return None,
    })
}

/// Check if a binary op is a comparison.
pub fn is_comparison(op: BinaryOp) -> bool {
    comparison_predicates(op).is_some()
}

/// Build a comparison operation with type dispatch.
pub fn build_cmp<'ctx>(
    builder: &Builder<'ctx>,
    op: BinaryOp,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    is_float: bool,
    is_signed: bool,
) -> Result<BasicValueEnum<'ctx>> {
    let (float_pred, signed_pred, unsigned_pred) = comparison_predicates(op)
        .ok_or_else(|| crate::llvm::error::Error::InvalidComparisonOp { op: format!("{:?}", op) })?;

    if is_float {
        Ok(builder
            .build_float_compare(float_pred, lhs.into_float_value(), rhs.into_float_value(), "cmp")
            .context(ComparisonSnafu)?
            .into())
    } else {
        let pred = if is_signed { signed_pred } else { unsigned_pred };
        Ok(builder
            .build_int_compare(pred, lhs.into_int_value(), rhs.into_int_value(), "cmp")
            .context(ComparisonSnafu)?
            .into())
    }
}
