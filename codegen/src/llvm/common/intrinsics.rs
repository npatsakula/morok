//! Table-driven LLVM intrinsic lookup.
//!
//! Maps operations to LLVM intrinsic names for cleaner codegen.

use morok_ir::prelude::*;

/// Get LLVM intrinsic name for a simple unary float operation.
///
/// Returns `(intrinsic_base, result_name)` for ops that are simple intrinsic calls.
/// Returns `None` for ops that need special handling (Neg, Abs, Rsqrt, Tan, etc.)
pub fn unary_float_intrinsic(op: UnaryOp) -> Option<(&'static str, &'static str)> {
    Some(match op {
        UnaryOp::Sqrt => ("llvm.sqrt", "sqrt"),
        UnaryOp::Exp => ("llvm.exp", "exp"),
        UnaryOp::Exp2 => ("llvm.exp2", "exp2"),
        UnaryOp::Log => ("llvm.log", "log"),
        UnaryOp::Log2 => ("llvm.log2", "log2"),
        UnaryOp::Sin => ("llvm.sin", "sin"),
        UnaryOp::Cos => ("llvm.cos", "cos"),
        UnaryOp::Floor => ("llvm.floor", "floor"),
        UnaryOp::Ceil => ("llvm.ceil", "ceil"),
        UnaryOp::Round => ("llvm.round", "round"),
        _ => return None,
    })
}

/// Get LLVM intrinsic name for max operation.
///
/// Returns the full intrinsic name based on type characteristics.
pub fn max_intrinsic(is_float: bool, is_signed: bool, bits: usize) -> String {
    if is_float {
        format!("llvm.maxnum.f{}", bits)
    } else if is_signed {
        format!("llvm.smax.i{}", bits)
    } else {
        format!("llvm.umax.i{}", bits)
    }
}
