use super::*;
use crate::types::BinaryOp;
use crate::{ConstValue, Op, UOp};
use morok_dtype::DType;

fn const_int(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(v))
}

fn binary(op: BinaryOp, lhs: Arc<UOp>, rhs: Arc<UOp>) -> Arc<UOp> {
    // Use UOp::new to create binary ops directly for tests
    UOp::new(Op::Binary(op, lhs, rhs), DType::Int32)
}

#[test]
fn test_empty_matcher() {
    let matcher = SimplifiedPatternMatcher::<()>::new();
    assert!(matcher.is_empty());
    assert_eq!(matcher.len(), 0);
}

#[test]
fn test_add_indexed_pattern() {
    let mut matcher = SimplifiedPatternMatcher::<()>::new();

    matcher.add(&[OpKey::Binary(BinaryOp::Add)], |_uop, _ctx| RewriteResult::NoMatch);

    assert_eq!(matcher.len(), 1);
    assert!(!matcher.is_empty());
}

#[test]
fn test_add_wildcard_pattern() {
    let mut matcher = SimplifiedPatternMatcher::<()>::new();

    matcher.add_wildcard(|_uop, _ctx| RewriteResult::NoMatch);

    assert_eq!(matcher.len(), 1);
    assert_eq!(matcher.wildcards.len(), 1);
}

#[test]
fn test_combine_matchers() {
    let mut m1 = SimplifiedPatternMatcher::<()>::new();
    m1.add(&[OpKey::Binary(BinaryOp::Add)], |_, _| RewriteResult::NoMatch);

    let mut m2 = SimplifiedPatternMatcher::<()>::new();
    m2.add(&[OpKey::Binary(BinaryOp::Mul)], |_, _| RewriteResult::NoMatch);

    let combined = m1 + m2;
    assert_eq!(combined.len(), 2);
}

#[test]
fn test_rewrite_basic() {
    let mut matcher = SimplifiedPatternMatcher::<()>::new();

    // Pattern: Add(x, 0) -> x
    matcher.add(&[OpKey::Binary(BinaryOp::Add)], |uop, _ctx| {
        let Op::Binary(BinaryOp::Add, left, right) = uop.op() else {
            return RewriteResult::NoMatch;
        };
        // Check if right is zero
        if let Op::Const(cv) = right.op()
            && cv.0.is_zero()
        {
            return RewriteResult::Rewritten(left.clone());
        }
        // Check if left is zero (commutative)
        if let Op::Const(cv) = left.op()
            && cv.0.is_zero()
        {
            return RewriteResult::Rewritten(right.clone());
        }
        RewriteResult::NoMatch
    });

    // Test: 5 + 0 -> 5
    let five = const_int(5);
    let zero = const_int(0);
    let expr = binary(BinaryOp::Add, five.clone(), zero);

    let result = matcher.rewrite(&expr, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(ref r) if Arc::ptr_eq(r, &five)));

    // Test: 0 + 5 -> 5
    let expr2 = binary(BinaryOp::Add, const_int(0), five.clone());
    let result2 = matcher.rewrite(&expr2, &mut ());
    assert!(matches!(result2, RewriteResult::Rewritten(ref r) if Arc::ptr_eq(r, &five)));

    // Test: 3 + 4 -> NoMatch
    let expr3 = binary(BinaryOp::Add, const_int(3), const_int(4));
    let result3 = matcher.rewrite(&expr3, &mut ());
    assert!(matches!(result3, RewriteResult::NoMatch));
}

#[test]
fn test_wildcard_after_indexed() {
    let mut matcher = SimplifiedPatternMatcher::<()>::new();

    // Indexed pattern that doesn't match
    matcher.add(&[OpKey::Binary(BinaryOp::Add)], |_uop, _ctx| RewriteResult::NoMatch);

    // Wildcard that matches everything
    matcher.add_wildcard(|uop, _ctx| RewriteResult::Rewritten(uop.clone()));

    let expr = binary(BinaryOp::Add, const_int(1), const_int(2));

    // Should fall through to wildcard
    let result = matcher.rewrite(&expr, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
}
