use morok_ir::{BinaryOp, ConstValue, DType, UOp};

use crate::rangeify::helpers::{get_const_value, is_const, is_identity_value, is_zero_value};

#[test]
fn test_is_identity_value() {
    // Add identity
    assert!(is_identity_value(&ConstValue::Int(0), &BinaryOp::Add, false));
    assert!(is_identity_value(&ConstValue::Int(0), &BinaryOp::Add, true));
    assert!(is_identity_value(&ConstValue::Float(0.0), &BinaryOp::Add, false));

    // Mul identity
    assert!(is_identity_value(&ConstValue::Int(1), &BinaryOp::Mul, false));
    assert!(is_identity_value(&ConstValue::Int(1), &BinaryOp::Mul, true));
    assert!(is_identity_value(&ConstValue::Float(1.0), &BinaryOp::Mul, false));

    // Sub only has right identity
    assert!(!is_identity_value(&ConstValue::Int(0), &BinaryOp::Sub, false));
    assert!(is_identity_value(&ConstValue::Int(0), &BinaryOp::Sub, true));

    // Idiv only has right identity
    assert!(!is_identity_value(&ConstValue::Int(1), &BinaryOp::Idiv, false));
    assert!(is_identity_value(&ConstValue::Int(1), &BinaryOp::Idiv, true));

    // Non-identity values
    assert!(!is_identity_value(&ConstValue::Int(2), &BinaryOp::Add, false));
    assert!(!is_identity_value(&ConstValue::Int(0), &BinaryOp::Mul, false));
}

#[test]
fn test_is_zero_value() {
    // Mul zero
    assert!(is_zero_value(&ConstValue::Int(0), &BinaryOp::Mul));
    assert!(is_zero_value(&ConstValue::Float(0.0), &BinaryOp::Mul));

    // And zero
    assert!(is_zero_value(&ConstValue::Int(0), &BinaryOp::And));

    // Non-zero values
    assert!(!is_zero_value(&ConstValue::Int(1), &BinaryOp::Mul));
    assert!(!is_zero_value(&ConstValue::Int(0), &BinaryOp::Add));
}

#[test]
fn test_get_const_value() {
    let c = UOp::const_(DType::Int32, ConstValue::Int(42));
    assert_eq!(get_const_value(&c), Some(ConstValue::Int(42)));

    let x = UOp::define_global(0, DType::Float32);
    assert_eq!(get_const_value(&x), None);
}

#[test]
fn test_is_const() {
    let c = UOp::const_(DType::Int32, ConstValue::Int(42));
    assert!(is_const(&c, &ConstValue::Int(42)));
    assert!(!is_const(&c, &ConstValue::Int(0)));
}
