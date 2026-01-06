//! Helper functions for pattern matching.
//!
//! These functions are used by the generated pattern matching code to check
//! common conditions like zero/one constants.

use std::sync::Arc;

use crate::ConstValue;

use crate::{Op, UOp};

/// Check if a UOp is a zero constant.
#[inline]
pub fn is_zero(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Const(cv) if cv.0.is_zero())
}

/// Check if a UOp is a one constant.
#[inline]
pub fn is_one(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Const(cv) if cv.0.is_one())
}

/// Check if a UOp is a negative one constant.
#[inline]
pub fn is_neg_one(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Const(cv) if cv.0.is_neg_one())
}

/// Check if a UOp is a non-zero constant.
#[inline]
pub fn is_nonzero(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Const(cv) if !cv.0.is_zero())
}

/// Extract const value if present.
#[inline]
pub fn try_const(uop: &Arc<UOp>) -> Option<&ConstValue> {
    match uop.op() {
        Op::Const(cv) => Some(&cv.0),
        _ => None,
    }
}

/// Check if a UOp matches a constant predicate.
#[inline]
pub fn const_matches<F>(uop: &Arc<UOp>, predicate: F) -> bool
where
    F: FnOnce(&ConstValue) -> bool,
{
    match uop.op() {
        Op::Const(cv) => predicate(&cv.0),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BinaryOp;
    use morok_dtype::DType;

    fn const_int(v: i64) -> Arc<UOp> {
        UOp::const_(DType::Int32, ConstValue::Int(v))
    }

    #[test]
    fn test_is_zero() {
        let zero = const_int(0);
        let one = const_int(1);
        let five = const_int(5);

        assert!(is_zero(&zero));
        assert!(!is_zero(&one));
        assert!(!is_zero(&five));
    }

    #[test]
    fn test_is_one() {
        let zero = const_int(0);
        let one = const_int(1);
        let five = const_int(5);

        assert!(!is_one(&zero));
        assert!(is_one(&one));
        assert!(!is_one(&five));
    }

    #[test]
    fn test_is_neg_one() {
        let zero = const_int(0);
        let neg_one = const_int(-1);
        let one = const_int(1);

        assert!(!is_neg_one(&zero));
        assert!(is_neg_one(&neg_one));
        assert!(!is_neg_one(&one));
    }

    #[test]
    fn test_is_nonzero() {
        let zero = const_int(0);
        let one = const_int(1);
        let neg_five = const_int(-5);

        assert!(!is_nonzero(&zero));
        assert!(is_nonzero(&one));
        assert!(is_nonzero(&neg_five));
    }

    #[test]
    fn test_try_const() {
        let five = const_int(5);
        assert!(try_const(&five).is_some());

        let one = const_int(1);
        let two = const_int(2);
        let add = UOp::new(Op::Binary(BinaryOp::Add, one, two), DType::Int32);
        assert!(try_const(&add).is_none());
    }

    #[test]
    fn test_const_matches() {
        let five = const_int(5);
        let zero = const_int(0);

        assert!(const_matches(&five, |cv| !cv.is_zero()));
        assert!(const_matches(&zero, |cv| cv.is_zero()));
        assert!(!const_matches(&five, |cv| cv.is_zero()));
    }
}
