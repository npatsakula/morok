use std::rc::Rc;

use crate::{
    PatternMatcher, UPat,
    pattern::{BindingStore, RewriteResult, VarIntern, matcher::RewriteFn},
};
use morok_dtype::DType;
use morok_ir::{BinaryOp, ConstValue, ConstValueHash, Op, UOp};

fn const_uop(val: i64) -> Rc<UOp> {
    UOp::new(Op::Const(ConstValueHash(ConstValue::Int(val))), DType::Int32)
}

fn binary_uop(op: BinaryOp, a: Rc<UOp>, b: Rc<UOp>) -> Rc<UOp> {
    let dtype = a.dtype().clone();
    UOp::new(Op::Binary(op, a, b), dtype)
}

#[test]
fn test_simple_rewrite() {
    // Pattern: x + 0 -> x
    let mut patterns = vec![];

    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Add], vec![UPat::var("x"), UPat::cvar("zero")]) => |x: &Rc<UOp>, zero: &Rc<UOp>| {
            if let Op::Const(cv) = zero.op()
                && cv.0 == ConstValue::Int(0) {
                    return Some(Rc::clone(x));
                }
            None
        }
    );

    let matcher = PatternMatcher::new(patterns);

    // Test: 5 + 0 should rewrite to 5
    let five = const_uop(5);
    let zero = const_uop(0);
    let add = binary_uop(BinaryOp::Add, five.clone(), zero);

    let result = matcher.rewrite(&add);
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(ref rewritten) = result {
        assert!(Rc::ptr_eq(rewritten, &five));
    }

    // Test: 5 + 3 should not rewrite
    let three = const_uop(3);
    let add2 = binary_uop(BinaryOp::Add, five, three);
    let result2 = matcher.rewrite(&add2);
    assert!(matches!(result2, RewriteResult::NoMatch));
}

#[test]
fn test_multiple_patterns() {
    // Pattern 1: x + 0 -> x
    // Pattern 2: x * 1 -> x
    let mut patterns = vec![];

    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Add], vec![UPat::var("x"), UPat::cvar("c")]) => |x: &Rc<UOp>, c: &Rc<UOp>| {
            if let Op::Const(cv) = c.op()
                && cv.0 == ConstValue::Int(0) {
                    return Some(Rc::clone(x));
                }
            None
        }
    );

    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Mul], vec![UPat::var("x"), UPat::cvar("c")]) => |x: &Rc<UOp>, c: &Rc<UOp>| {
            if let Op::Const(cv) = c.op()
                && cv.0 == ConstValue::Int(1) {
                    return Some(Rc::clone(x));
                }
            None
        }
    );

    let matcher = PatternMatcher::new(patterns);

    let five = const_uop(5);

    // Test: 5 + 0 -> 5
    let add = binary_uop(BinaryOp::Add, five.clone(), const_uop(0));
    let result = matcher.rewrite(&add);
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(ref rewritten) = result {
        assert!(Rc::ptr_eq(rewritten, &five));
    }

    // Test: 5 * 1 -> 5
    let mul = binary_uop(BinaryOp::Mul, five.clone(), const_uop(1));
    let result = matcher.rewrite(&mul);
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(ref rewritten) = result {
        assert!(Rc::ptr_eq(rewritten, &five));
    }
}

#[test]
fn test_no_match() {
    let mut patterns = vec![];

    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Add], vec![UPat::var("x"), UPat::cvar("c")]) => |x: &Rc<UOp>, c: &Rc<UOp>| {
            if let Op::Const(cv) = c.op()
                && cv.0 == ConstValue::Int(0) {
                    return Some(Rc::clone(x));
                }
            None
        }
    );

    let matcher = PatternMatcher::new(patterns);

    // MUL doesn't match ADD pattern
    let mul = binary_uop(BinaryOp::Mul, const_uop(5), const_uop(0));
    let result = matcher.rewrite(&mul);
    assert!(matches!(result, RewriteResult::NoMatch));

    // ADD with non-zero doesn't match the rewrite condition
    let add = binary_uop(BinaryOp::Add, const_uop(5), const_uop(3));
    let result = matcher.rewrite(&add);
    assert!(matches!(result, RewriteResult::NoMatch));
}

#[test]
fn test_wildcard_pattern() {
    // Pattern that matches any UOp and returns a constant
    // Note: Uses manual closure to test low-level pattern matching infrastructure
    let patterns = vec![(
        UPat::var("anything"),
        Box::new(|_bindings: &BindingStore, _intern: &VarIntern| RewriteResult::Rewritten(const_uop(42))) as RewriteFn,
    )];

    let matcher = PatternMatcher::new(patterns);

    // Should match any UOp
    let add = binary_uop(BinaryOp::Add, const_uop(1), const_uop(2));
    let result = matcher.rewrite(&add);
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(ref result_val) = result {
        if let Op::Const(cv) = result_val.op() {
            assert_eq!(cv.0, ConstValue::Int(42));
        } else {
            panic!("Expected Const op");
        }
    }
}

#[test]
fn test_indexed_before_wildcard() {
    // Pattern 1: x + 0 -> x (indexed under Add)
    // Pattern 2: any -> constant 99 (wildcard)
    // Note: Wildcard pattern uses manual closure to test low-level matching infrastructure
    let mut patterns = vec![];

    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Add], vec![UPat::var("x"), UPat::cvar("c")]) => |x: &Rc<UOp>, c: &Rc<UOp>| {
            if let Op::Const(cv) = c.op()
                && cv.0 == ConstValue::Int(0) {
                    return Some(Rc::clone(x));
                }
            None
        }
    );

    patterns.push((
        UPat::var("anything"),
        Box::new(|_bindings: &BindingStore, _intern: &VarIntern| RewriteResult::Rewritten(const_uop(99))) as RewriteFn,
    ));

    let matcher = PatternMatcher::new(patterns);

    let five = const_uop(5);

    // 5 + 0 should match first pattern (indexed) and return 5, not 99
    let add = binary_uop(BinaryOp::Add, five.clone(), const_uop(0));
    let result = matcher.rewrite(&add);
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(ref rewritten) = result {
        assert!(Rc::ptr_eq(rewritten, &five));
    }

    // MUL doesn't match first pattern, so wildcard applies -> 99
    let mul = binary_uop(BinaryOp::Mul, const_uop(5), const_uop(1));
    let result = matcher.rewrite(&mul);
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(ref rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(99));
        } else {
            panic!("Expected Const op");
        }
    }
}
