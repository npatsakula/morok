//! Type-aware LLVM builder helpers.
//!
//! These helpers dispatch on float/int/signed types to reduce boilerplate in ops.rs.
//! All arithmetic operations support both scalar and vector types.

use crate::llvm::error::{
    ArithmeticSnafu, ComparisonSnafu, Result, VectorInsertSnafu, VectorShuffleSnafu, VectorWidthMismatchSnafu,
};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::values::BasicValueEnum;
use inkwell::{FloatPredicate, IntPredicate};
use morok_ir::BinaryOp;
use snafu::ResultExt;

// ============================================================================
// Binary arithmetic operations (add, sub, mul)
// ============================================================================

/// Generate a binary arithmetic function that dispatches on float/int and scalar/vector.
/// Float operations have fast-math flags applied for FMA fusion and other optimizations.
macro_rules! impl_binary_arith {
    ($fn_name:ident, $float_method:ident, $int_method:ident, $name:literal) => {
        pub fn $fn_name<'ctx>(
            builder: &Builder<'ctx>,
            lhs: BasicValueEnum<'ctx>,
            rhs: BasicValueEnum<'ctx>,
            is_float: bool,
        ) -> Result<BasicValueEnum<'ctx>> {
            if is_float {
                let result: BasicValueEnum<'ctx> = if lhs.is_vector_value() {
                    builder
                        .$float_method(lhs.into_vector_value(), rhs.into_vector_value(), $name)
                        .context(ArithmeticSnafu)?
                        .into()
                } else {
                    builder
                        .$float_method(lhs.into_float_value(), rhs.into_float_value(), $name)
                        .context(ArithmeticSnafu)?
                        .into()
                };
                // Apply fast-math flags for FMA fusion and other float optimizations
                fast_math::apply_fast_math_flags(result);
                Ok(result)
            } else if lhs.is_vector_value() {
                Ok(builder
                    .$int_method(lhs.into_vector_value(), rhs.into_vector_value(), $name)
                    .context(ArithmeticSnafu)?
                    .into())
            } else {
                Ok(builder
                    .$int_method(lhs.into_int_value(), rhs.into_int_value(), $name)
                    .context(ArithmeticSnafu)?
                    .into())
            }
        }
    };
}

impl_binary_arith!(build_add, build_float_add, build_int_add, "add");
impl_binary_arith!(build_sub, build_float_sub, build_int_sub, "sub");
impl_binary_arith!(build_mul, build_float_mul, build_int_mul, "mul");

// ============================================================================
// Unary operations
// ============================================================================

/// Build a negation, dispatching on float/int and scalar/vector.
/// Float operations have fast-math flags applied.
pub fn build_neg<'ctx>(
    builder: &Builder<'ctx>,
    src: BasicValueEnum<'ctx>,
    is_float: bool,
) -> Result<BasicValueEnum<'ctx>> {
    if is_float {
        let result: BasicValueEnum<'ctx> = if src.is_vector_value() {
            builder.build_float_neg(src.into_vector_value(), "neg").context(ArithmeticSnafu)?.into()
        } else {
            builder.build_float_neg(src.into_float_value(), "neg").context(ArithmeticSnafu)?.into()
        };
        fast_math::apply_fast_math_flags(result);
        Ok(result)
    } else if src.is_vector_value() {
        Ok(builder.build_int_neg(src.into_vector_value(), "neg").context(ArithmeticSnafu)?.into())
    } else {
        Ok(builder.build_int_neg(src.into_int_value(), "neg").context(ArithmeticSnafu)?.into())
    }
}

// ============================================================================
// Division and remainder
// ============================================================================

/// Build an integer division, dispatching on signed/unsigned and scalar/vector.
pub fn build_int_div<'ctx>(
    builder: &Builder<'ctx>,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    is_signed: bool,
) -> Result<BasicValueEnum<'ctx>> {
    if is_signed {
        if lhs.is_vector_value() {
            Ok(builder
                .build_int_signed_div(lhs.into_vector_value(), rhs.into_vector_value(), "idiv")
                .context(ArithmeticSnafu)?
                .into())
        } else {
            Ok(builder
                .build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "idiv")
                .context(ArithmeticSnafu)?
                .into())
        }
    } else if lhs.is_vector_value() {
        Ok(builder
            .build_int_unsigned_div(lhs.into_vector_value(), rhs.into_vector_value(), "idiv")
            .context(ArithmeticSnafu)?
            .into())
    } else {
        Ok(builder
            .build_int_unsigned_div(lhs.into_int_value(), rhs.into_int_value(), "idiv")
            .context(ArithmeticSnafu)?
            .into())
    }
}

/// Build a remainder, dispatching on float/int, signed/unsigned, and scalar/vector.
/// Float operations have fast-math flags applied.
pub fn build_rem<'ctx>(
    builder: &Builder<'ctx>,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    is_float: bool,
    is_signed: bool,
) -> Result<BasicValueEnum<'ctx>> {
    if is_float {
        let result: BasicValueEnum<'ctx> = if lhs.is_vector_value() {
            builder
                .build_float_rem(lhs.into_vector_value(), rhs.into_vector_value(), "mod")
                .context(ArithmeticSnafu)?
                .into()
        } else {
            builder
                .build_float_rem(lhs.into_float_value(), rhs.into_float_value(), "mod")
                .context(ArithmeticSnafu)?
                .into()
        };
        fast_math::apply_fast_math_flags(result);
        Ok(result)
    } else if is_signed {
        if lhs.is_vector_value() {
            Ok(builder
                .build_int_signed_rem(lhs.into_vector_value(), rhs.into_vector_value(), "mod")
                .context(ArithmeticSnafu)?
                .into())
        } else {
            Ok(builder
                .build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "mod")
                .context(ArithmeticSnafu)?
                .into())
        }
    } else if lhs.is_vector_value() {
        Ok(builder
            .build_int_unsigned_rem(lhs.into_vector_value(), rhs.into_vector_value(), "mod")
            .context(ArithmeticSnafu)?
            .into())
    } else {
        Ok(builder
            .build_int_unsigned_rem(lhs.into_int_value(), rhs.into_int_value(), "mod")
            .context(ArithmeticSnafu)?
            .into())
    }
}

// ============================================================================
// Comparisons
// ============================================================================

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

/// Build a comparison operation with type dispatch (scalar/vector).
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
        if lhs.is_vector_value() {
            Ok(builder
                .build_float_compare(float_pred, lhs.into_vector_value(), rhs.into_vector_value(), "cmp")
                .context(ComparisonSnafu)?
                .into())
        } else {
            Ok(builder
                .build_float_compare(float_pred, lhs.into_float_value(), rhs.into_float_value(), "cmp")
                .context(ComparisonSnafu)?
                .into())
        }
    } else {
        let pred = if is_signed { signed_pred } else { unsigned_pred };
        if lhs.is_vector_value() {
            Ok(builder
                .build_int_compare(pred, lhs.into_vector_value(), rhs.into_vector_value(), "cmp")
                .context(ComparisonSnafu)?
                .into())
        } else {
            Ok(builder
                .build_int_compare(pred, lhs.into_int_value(), rhs.into_int_value(), "cmp")
                .context(ComparisonSnafu)?
                .into())
        }
    }
}

// ============================================================================
// Vector operations
// ============================================================================

/// Get vector element count from a value (1 for scalars).
pub fn get_vector_count(value: BasicValueEnum<'_>) -> u32 {
    if value.is_vector_value() { value.into_vector_value().get_type().get_size() } else { 1 }
}

/// Broadcast a scalar value to a vector of given count.
///
/// Uses insertelement + shufflevector pattern:
/// 1. Insert scalar into poison vector at index 0
/// 2. Shuffle with zeroinitializer mask to replicate element 0
pub fn broadcast_to_vector<'ctx>(
    builder: &Builder<'ctx>,
    scalar: BasicValueEnum<'ctx>,
    count: u32,
    context: &'ctx Context,
) -> Result<BasicValueEnum<'ctx>> {
    use inkwell::types::VectorType;

    // Get vector type based on scalar type
    let vec_type: VectorType = if scalar.is_float_value() {
        scalar.into_float_value().get_type().vec_type(count)
    } else if scalar.is_int_value() {
        scalar.into_int_value().get_type().vec_type(count)
    } else {
        // Fallback for other types - return as-is
        return Ok(scalar);
    };

    // Insert scalar into poison vector at index 0
    let poison = vec_type.get_poison();
    let zero = context.i32_type().const_zero();
    let single = builder.build_insert_element(poison, scalar, zero, "bcast_insert").context(VectorInsertSnafu)?;

    // Create mask of all zeros to replicate element 0 to all positions
    let i32_type = context.i32_type();
    let mask_values: Vec<_> = (0..count).map(|_| i32_type.const_zero()).collect();
    let mask = VectorType::const_vector(&mask_values);

    // Shuffle to broadcast
    let result = builder.build_shuffle_vector(single, poison, mask, "bcast_shuffle").context(VectorShuffleSnafu)?;

    Ok(result.into())
}

/// Harmonize operands: broadcast scalar to match vector width.
///
/// - If both operands have the same vector width, return them unchanged.
/// - If one is scalar (width 1) and the other is vector, broadcast the scalar.
/// - If both are vectors with different widths, return an error.
pub fn harmonize_operands<'ctx>(
    builder: &Builder<'ctx>,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    context: &'ctx Context,
) -> Result<(BasicValueEnum<'ctx>, BasicValueEnum<'ctx>)> {
    let lhs_count = get_vector_count(lhs);
    let rhs_count = get_vector_count(rhs);

    match (lhs_count, rhs_count) {
        (1, n) if n > 1 => Ok((broadcast_to_vector(builder, lhs, n, context)?, rhs)),
        (n, 1) if n > 1 => Ok((lhs, broadcast_to_vector(builder, rhs, n, context)?)),
        (l, r) if l == r => Ok((lhs, rhs)),
        (l, r) => VectorWidthMismatchSnafu { lhs: l, rhs: r }.fail(),
    }
}

// ============================================================================
// Fast-math flags
// ============================================================================

/// LLVM Fast-Math Flags for floating-point optimizations.
///
/// Following Tinygrad's approach (llvmir.py:69):
/// `flags = " nsz arcp contract afn"`
///
/// Reference: https://llvm.org/docs/LangRef.html#fast-math-flags
pub mod fast_math {
    use inkwell::values::{BasicValue, BasicValueEnum};

    /// NoSignedZeros (1 << 3): Treat -0 and +0 as equivalent.
    pub const FMF_NSZ: u32 = 1 << 3;

    /// AllowReciprocal (1 << 4): Allow optimizations using reciprocal (1/x approximation).
    pub const FMF_ARCP: u32 = 1 << 4;

    /// AllowContract (1 << 5): Allow floating-point contraction (e.g., fusing a*b+c into fma).
    pub const FMF_CONTRACT: u32 = 1 << 5;

    /// ApproxFunc (1 << 6): Allow substitution of approximate calculations for functions.
    pub const FMF_AFN: u32 = 1 << 6;

    /// Default fast-math flags for ML workloads, matching Tinygrad: nsz arcp contract afn.
    pub const FMF_DEFAULT: u32 = FMF_NSZ | FMF_ARCP | FMF_CONTRACT | FMF_AFN;

    /// Apply fast-math flags to a floating-point instruction result.
    ///
    /// This is safe to call on any value - non-FP instructions are silently ignored.
    /// The flags enable LLVM to perform FMA fusion and other float optimizations.
    #[inline]
    pub fn apply_fast_math_flags(value: BasicValueEnum<'_>) {
        if let Some(inst) = value.as_instruction_value() {
            inst.set_fast_math_flags(FMF_DEFAULT);
        }
    }
}
