use crate::pattern::UPat;
use crate::pattern::{PatternMatcher, RewriteResult};
use crate::rangeify::helpers::{get_const_value, is_identity_value};
use morok_dtype::DType;
use morok_ir::{BinaryOp, ConstValue, UOp};
use std::rc::Rc;

#[test]
fn test_pattern_macro_basic() {
    let mut patterns = vec![];

    // Define a simple pattern: x + 0 → x
    pattern!(patterns,
        UPat::var("x") + UPat::cvar("c") => |x, c| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Add, true) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    assert_eq!(patterns.len(), 1);

    // Test that the pattern works
    let matcher = PatternMatcher::new(patterns);

    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let add = five.try_add(&zero).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Rc::ptr_eq(&rewritten, &five));
    }
}

#[test]
fn test_pattern_macro_no_match() {
    let mut patterns = vec![];

    pattern!(patterns,
        UPat::var("x") + UPat::cvar("c") => |x, c| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Add, true) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    let matcher = PatternMatcher::new(patterns);

    // 5 + 3 (not identity) should not match
    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let three = UOp::const_(DType::Int32, ConstValue::Int(3));
    let add = five.try_add(&three).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::NoMatch));
}

#[test]
fn test_pattern_macro_multiple_variables() {
    use crate::rangeify::helpers::is_zero_value;

    let mut patterns = vec![];

    // Pattern: x * 0 → 0 (zero propagation)
    pattern!(patterns,
        UPat::var("x") * UPat::cvar("zero") => |x, zero| {
            let _unused = x;  // Suppress unused variable warning
            let const_val = get_const_value(zero)?;
            if is_zero_value(&const_val, &BinaryOp::Mul) {
                Some(Rc::clone(zero))
            } else {
                None
            }
        }
    );

    let matcher = PatternMatcher::new(patterns);

    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let mul = five.try_mul(&zero).unwrap();

    let result = matcher.rewrite(&mul, &mut ());
    match &result {
        RewriteResult::Rewritten(r) => {
            assert!(Rc::ptr_eq(r, &zero));
        }
        RewriteResult::NoMatch => {
            panic!("Pattern did not match!");
        }
        RewriteResult::Gate(_) => {
            panic!("Unexpected Gate result");
        }
    }
}
