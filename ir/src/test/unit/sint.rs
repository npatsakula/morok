use morok_dtype::DType;

use crate::{ConstValue, SInt, UOp, sint_max, sint_min, sint_prod};

#[test]
fn test_sint_const() {
    let s = SInt::from(42);
    assert!(s.is_const());
    assert!(!s.is_symbolic());
    assert_eq!(s.as_const(), Some(42));
}

#[test]
fn test_sint_symbolic() {
    let uop = UOp::const_(DType::Index, ConstValue::Int(10));
    let s = SInt::from(uop);
    assert!(s.is_const()); // Should simplify to const
    assert_eq!(s.as_const(), Some(10));
}

#[test]
fn test_sint_prod_concrete() {
    let dims = vec![SInt::from(2), SInt::from(3), SInt::from(4)];
    let result = sint_prod(&dims);
    assert_eq!(result.as_const(), Some(24));
}

#[test]
fn test_sint_max_concrete() {
    let vals = vec![SInt::from(10), SInt::from(20), SInt::from(15)];
    let result = sint_max(&vals);
    assert_eq!(result.as_const(), Some(20));
}

#[test]
fn test_sint_min_concrete() {
    let vals = vec![SInt::from(10), SInt::from(20), SInt::from(15)];
    let result = sint_min(&vals);
    assert_eq!(result.as_const(), Some(10));
}

#[test]
fn test_sint_to_uop() {
    let s = SInt::from(42);
    let uop = s.to_uop(DType::Index);
    assert_eq!(uop.dtype(), DType::Index);
}

#[test]
fn test_sint_simplify() {
    let uop = UOp::const_(DType::Index, ConstValue::Int(100));
    let s = SInt::from(uop);
    let simplified = s.simplify();
    assert_eq!(simplified.as_const(), Some(100));
}
