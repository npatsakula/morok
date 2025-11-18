//! Shrinking strategies for UOp graphs.
//!
//! Provides custom shrinking to help proptest find minimal counterexamples
//! when property tests fail.

use std::rc::Rc;

use crate::types::ConstValue;
use crate::{Op, UOp};

/// Shrink a UOp graph to simpler forms.
///
/// Shrinking strategies (in order of application):
/// 1. Replace with a constant (most aggressive)
/// 2. Replace binary ops with one of their operands
/// 3. Replace unary ops with their operand
/// 4. Simplify constants
///
/// Returns an iterator of progressively simpler UOps.
pub fn shrink_uop(uop: &Rc<UOp>) -> Vec<Rc<UOp>> {
    let mut shrunk = Vec::new();

    match uop.op() {
        Op::Binary(_, lhs, rhs) => {
            // Strategy 1: Replace with left operand
            shrunk.push(Rc::clone(lhs));

            // Strategy 2: Replace with right operand
            shrunk.push(Rc::clone(rhs));

            // Strategy 3: Replace with constant 0
            shrunk.push(UOp::const_(uop.dtype().clone(), ConstValue::Int(0)));

            // Strategy 4: Replace with constant 1
            shrunk.push(UOp::const_(uop.dtype().clone(), ConstValue::Int(1)));

            // Strategy 5: Recursively shrink operands (not implemented to keep shrinking simple)
            // This would create new binary ops with shrunk operands, but we keep it simple
            // by only doing simple shrinking strategies above
        }

        Op::Unary(_, src) => {
            // Strategy 1: Replace with source operand
            shrunk.push(Rc::clone(src));

            // Strategy 2: Replace with constant 0
            shrunk.push(UOp::const_(uop.dtype().clone(), ConstValue::Int(0)));

            // Strategy 3: Recursively shrink source (not implemented to keep shrinking simple)
            // This would create new unary ops with shrunk sources
        }

        Op::Const(cv) => {
            // Shrink constants towards zero
            shrunk.extend(shrink_const_value(&cv.0, &uop.dtype()));
        }

        Op::Ternary(_, a, b, c) => {
            // Replace with one of the branches
            shrunk.push(Rc::clone(b)); // true branch
            shrunk.push(Rc::clone(c)); // false branch
            shrunk.push(Rc::clone(a)); // condition (least likely to help)
        }

        _ => {
            // For other ops (DefineVar, Range, etc.), don't shrink
        }
    }

    shrunk
}

/// Shrink a constant value towards zero.
fn shrink_const_value(cv: &ConstValue, dtype: &morok_dtype::DType) -> Vec<Rc<UOp>> {
    let mut shrunk = Vec::new();

    match cv {
        ConstValue::Int(v) if *v != 0 => {
            // Shrink towards zero
            if *v > 0 {
                // Positive: try smaller positive values
                if *v > 1 {
                    shrunk.push(UOp::const_(dtype.clone(), ConstValue::Int(1)));
                }
                if *v > 10 {
                    shrunk.push(UOp::const_(dtype.clone(), ConstValue::Int(v / 2)));
                }
            } else {
                // Negative: try smaller magnitude negative values
                if *v < -1 {
                    shrunk.push(UOp::const_(dtype.clone(), ConstValue::Int(-1)));
                }
                if *v < -10 {
                    shrunk.push(UOp::const_(dtype.clone(), ConstValue::Int(v / 2)));
                }
            }
            // Always try zero
            shrunk.push(UOp::const_(dtype.clone(), ConstValue::Int(0)));
        }

        ConstValue::UInt(v) if *v != 0 => {
            // Shrink towards zero
            if *v > 1 {
                shrunk.push(UOp::const_(dtype.clone(), ConstValue::UInt(1)));
            }
            if *v > 10 {
                shrunk.push(UOp::const_(dtype.clone(), ConstValue::UInt(v / 2)));
            }
            shrunk.push(UOp::const_(dtype.clone(), ConstValue::UInt(0)));
        }

        ConstValue::Float(v) if *v != 0.0 => {
            // Shrink floats towards zero
            if v.is_finite() {
                if v.abs() > 1.0 {
                    shrunk.push(UOp::const_(dtype.clone(), ConstValue::Float(v.signum())));
                }
                if v.abs() > 10.0 {
                    shrunk.push(UOp::const_(dtype.clone(), ConstValue::Float(v / 2.0)));
                }
            }
            shrunk.push(UOp::const_(dtype.clone(), ConstValue::Float(0.0)));
        }

        ConstValue::Bool(true) => {
            shrunk.push(UOp::const_(dtype.clone(), ConstValue::Bool(false)));
        }

        _ => {
            // Already at simplest form
        }
    }

    shrunk
}

/// Get the depth of a UOp graph (maximum path length from root to leaf).
pub fn uop_depth(uop: &Rc<UOp>) -> usize {
    match uop.op() {
        Op::Binary(_, lhs, rhs) => 1 + uop_depth(lhs).max(uop_depth(rhs)),
        Op::Unary(_, src) => 1 + uop_depth(src),
        Op::Ternary(_, a, b, c) => 1 + uop_depth(a).max(uop_depth(b)).max(uop_depth(c)),
        _ => 0, // Leaf node (Const, DefineVar, Range, etc.)
    }
}

/// Count the number of operations in a UOp graph.
pub fn uop_op_count(uop: &Rc<UOp>) -> usize {
    match uop.op() {
        Op::Binary(_, lhs, rhs) => 1 + uop_op_count(lhs) + uop_op_count(rhs),
        Op::Unary(_, src) => 1 + uop_op_count(src),
        Op::Ternary(_, a, b, c) => 1 + uop_op_count(a) + uop_op_count(b) + uop_op_count(c),
        _ => 0, // Leaf nodes don't count as operations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_dtype::DType;

    #[test]
    fn test_shrink_const() {
        let big_const = UOp::const_(DType::Int32, ConstValue::Int(100));
        let shrunk = shrink_uop(&big_const);

        // Should shrink towards zero
        assert!(!shrunk.is_empty());
        assert!(shrunk.iter().any(|u| matches!(u.op(), Op::Const(cv) if cv.0 == ConstValue::Int(0))));
    }

    #[test]
    fn test_uop_depth() {
        let x = UOp::var("x", DType::Int32, 0, 100);
        assert_eq!(uop_depth(&x), 0);

        let x_plus_1 = UOp::new(
            Op::Binary(crate::types::BinaryOp::Add, Rc::clone(&x), UOp::const_(DType::Int32, ConstValue::Int(1))),
            DType::Int32,
        );
        assert_eq!(uop_depth(&x_plus_1), 1);
    }

    #[test]
    fn test_uop_op_count() {
        let x = UOp::var("x", DType::Int32, 0, 100);
        assert_eq!(uop_op_count(&x), 0);

        let x_plus_1 = UOp::new(
            Op::Binary(crate::types::BinaryOp::Add, Rc::clone(&x), UOp::const_(DType::Int32, ConstValue::Int(1))),
            DType::Int32,
        );
        assert_eq!(uop_op_count(&x_plus_1), 1);
    }
}
