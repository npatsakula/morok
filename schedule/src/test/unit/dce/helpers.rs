//! Test helpers for DCE tests.

use morok_ir::types::ConstValue;
use morok_ir::{Op, UOp};
use std::rc::Rc;

use crate::pattern::PatternMatcher;
use crate::symbolic::symbolic_simple;

/// Get the symbolic_simple pattern matcher (reduces duplication).
pub fn get_matcher() -> PatternMatcher {
    symbolic_simple()
}

/// Assert that a UOp transforms to a specific constant value.
///
/// # Panics
/// Panics if the UOp is not a Const or doesn't match the expected value.
pub fn assert_const_value(uop: &Rc<UOp>, expected: ConstValue) {
    match uop.op() {
        Op::Const(cv) => {
            assert_eq!(cv.0, expected, "Expected Const({:?}), got Const({:?})", expected, cv.0);
        }
        other => panic!("Expected Const({:?}), got {:?}", expected, other),
    }
}

/// Assert that an END operation unwraps to its computation (no ranges).
///
/// Returns the computation UOp for further assertions.
///
/// # Panics
/// Panics if the UOp is still an END operation.
pub fn assert_end_unwrapped(uop: &Rc<UOp>) -> Rc<UOp> {
    match uop.op() {
        Op::End { .. } => {
            panic!("Expected END to be unwrapped, but got END operation: {:?}", uop.op())
        }
        _ => uop.clone(),
    }
}

/// Assert that an END operation has a specific number of ranges.
///
/// Returns the computation UOp and ranges for further assertions.
///
/// # Panics
/// Panics if the UOp is not an END operation or range count doesn't match.
pub fn assert_end_range_count(uop: &Rc<UOp>, expected_count: usize) -> (Rc<UOp>, Vec<Rc<UOp>>) {
    match uop.op() {
        Op::End { computation, ranges } => {
            assert_eq!(
                ranges.len(),
                expected_count,
                "Expected END with {} ranges, got {} ranges",
                expected_count,
                ranges.len()
            );
            (Rc::clone(computation), ranges.iter().cloned().collect())
        }
        other => panic!("Expected END operation, got {:?}", other),
    }
}
