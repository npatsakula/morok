use super::*;
use morok_dtype::DType;

#[test]
fn test_const_factor_constant() {
    let c = UOp::const_(DType::Int32, ConstValue::Int(6));
    assert_eq!(c.const_factor(), 6);
}

#[test]
fn test_const_factor_multiplication() {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let c = UOp::const_(DType::Int32, ConstValue::Int(6));
    let mul = x.try_mul(&c).unwrap();
    assert_eq!(mul.const_factor(), 6);
}

#[test]
fn test_const_factor_addition() {
    let c1 = UOp::const_(DType::Int32, ConstValue::Int(6));
    let c2 = UOp::const_(DType::Int32, ConstValue::Int(9));
    let add = c1.try_add(&c2).unwrap();
    assert_eq!(add.const_factor(), 3); // GCD(6, 9) = 3
}

#[test]
fn test_divides_constant_exact() {
    let c = UOp::const_(DType::Int32, ConstValue::Int(12));
    let result = c.divides(3);

    assert!(result.is_some());
    if let Some(r) = result {
        if let Op::Const(cv) = r.op() {
            assert_eq!(cv.0, ConstValue::Int(4));
        } else {
            panic!("Expected constant result");
        }
    }
}

#[test]
fn test_divides_constant_not_exact() {
    let c = UOp::const_(DType::Int32, ConstValue::Int(10));
    assert!(c.divides(3).is_none());
}

#[test]
fn test_pop_const_with_constant() {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let c = UOp::const_(DType::Int32, ConstValue::Int(5));
    let add = x.try_add(&c).unwrap();

    let (rest, const_val) = add.pop_const(BinaryOp::Add);

    assert!(Arc::ptr_eq(&rest, &x));
    assert_eq!(const_val, ConstValue::Int(5));
}

#[test]
fn test_pop_const_without_constant() {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let y = UOp::var("y", DType::Int32, 0, 100);
    let add = x.try_add(&y).unwrap();

    let (rest, const_val) = add.pop_const(BinaryOp::Add);

    assert!(Arc::ptr_eq(&rest, &add));
    // No literal const present → identity (Int(0) for ADD on Int32).
    assert_eq!(const_val, ConstValue::Int(0));
}

#[test]
fn test_split_uop_chain() {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let y = UOp::var("y", DType::Int32, 0, 100);
    let z = UOp::var("z", DType::Int32, 0, 100);

    // Build: x + y + z = (x + y) + z
    let xy = x.try_add(&y).unwrap();
    let xyz = xy.try_add(&z).unwrap();

    let terms = xyz.split_uop(BinaryOp::Add);

    assert_eq!(terms.len(), 3);
    assert!(Arc::ptr_eq(&terms[0], &x));
    assert!(Arc::ptr_eq(&terms[1], &y));
    assert!(Arc::ptr_eq(&terms[2], &z));
}

#[test]
fn test_split_uop_single() {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let terms = x.split_uop(BinaryOp::Add);

    assert_eq!(terms.len(), 1);
    assert!(Arc::ptr_eq(&terms[0], &x));
}

#[test]
fn test_gcd() {
    assert_eq!(gcd(12, 8), 4);
    assert_eq!(gcd(17, 19), 1);
    assert_eq!(gcd(100, 50), 50);
    assert_eq!(gcd(-12, 8), 4);
    assert_eq!(gcd(12, -8), 4);
    assert_eq!(gcd(-12, -8), 4);
}

#[test]
fn test_symbolic_gcd_numeric_only() {
    // GCD of 6*x and 4*y → numeric GCD is 2
    let x = UOp::var("x", DType::Index, 0, 10);
    let y = UOp::var("y", DType::Index, 0, 10);
    let six = UOp::const_(DType::Index, ConstValue::Int(6));
    let four = UOp::const_(DType::Index, ConstValue::Int(4));
    let a = x.try_mul(&six).unwrap(); // 6*x
    let b = y.try_mul(&four).unwrap(); // 4*y
    let g = UOp::symbolic_gcd(&[a, b]);
    if let Op::Const(cv) = g.op() {
        assert_eq!(cv.0, ConstValue::Int(2));
    } else {
        panic!("Expected constant GCD, got: {}", g.tree());
    }
}

#[test]
fn test_symbolic_gcd_with_common_factor() {
    // GCD of 6*x and 4*x → 2*x (common symbolic factor x, numeric GCD 2)
    let x = UOp::var("x", DType::Index, 0, 10);
    let six = UOp::const_(DType::Index, ConstValue::Int(6));
    let four = UOp::const_(DType::Index, ConstValue::Int(4));
    let a = x.try_mul(&six).unwrap(); // 6*x (= x*6 internally)
    let b = x.try_mul(&four).unwrap(); // 4*x (= x*4 internally)
    let g = UOp::symbolic_gcd(&[a, b]);
    // Should be 2*x — a MUL node
    assert!(matches!(g.op(), Op::Binary(BinaryOp::Mul, _, _)), "Expected MUL, got: {}", g.tree());
}

#[test]
fn test_const_factor_mul_only_immediate() {
    // (x * 6) * (y * 4) — const_factor should be 1 (no immediate CONST child)
    let x = UOp::var("x", DType::Index, 0, 10);
    let y = UOp::var("y", DType::Index, 0, 10);
    let six = UOp::const_(DType::Index, ConstValue::Int(6));
    let four = UOp::const_(DType::Index, ConstValue::Int(4));
    let a = x.try_mul(&six).unwrap(); // x*6
    let b = y.try_mul(&four).unwrap(); // y*4
    let ab = a.try_mul(&b).unwrap(); // (x*6) * (y*4)
    // Tinygrad: neither immediate child is CONST → returns 1
    assert_eq!(ab.const_factor(), 1);
}

#[test]
fn test_const_factor_vconst() {
    let vc = UOp::vconst(
        vec![ConstValue::Int(6), ConstValue::Int(12), ConstValue::Int(18), ConstValue::Int(24)],
        DType::Int64,
    );
    assert_eq!(vc.const_factor(), 6); // GCD(6, 12, 18, 24) = 6
}

#[test]
fn test_const_factor_vconst_no_common() {
    let vc = UOp::vconst(vec![ConstValue::Int(7), ConstValue::Int(11)], DType::Int64);
    assert_eq!(vc.const_factor(), 1); // GCD(7, 11) = 1
}

#[test]
fn test_divides_vconst() {
    let vc = UOp::vconst(vec![ConstValue::Int(6), ConstValue::Int(12)], DType::Int64);
    let result = vc.divides(3);
    assert!(result.is_some());
    if let Some(r) = result {
        if let Op::VConst { values } = r.op() {
            assert_eq!(values, &[ConstValue::Int(2), ConstValue::Int(4)]);
        } else {
            panic!("Expected VConst result");
        }
    }
}

#[test]
fn test_divides_vconst_not_divisible() {
    let vc = UOp::vconst(
        vec![
            ConstValue::Int(6),
            ConstValue::Int(7), // 7 not divisible by 3
        ],
        DType::Int64,
    );
    assert!(vc.divides(3).is_none());
}

#[test]
fn test_is_increasing_const() {
    let c = UOp::const_(DType::Int32, ConstValue::Int(5));
    assert!(c.is_increasing());

    let neg = UOp::const_(DType::Int32, ConstValue::Int(-5));
    assert!(neg.is_increasing()); // Constants are always "increasing" (irreducible)
}

#[test]
fn test_is_increasing_add() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(5));
    let b = UOp::const_(DType::Int32, ConstValue::Int(3));
    let sum = a.try_add(&b).unwrap();
    assert!(sum.is_increasing());
}

#[test]
fn test_is_increasing_mul_positive_const() {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let two = UOp::const_(DType::Int32, ConstValue::Int(2));
    let scaled = x.try_mul(&two).unwrap();
    assert!(scaled.is_increasing());
}

#[test]
fn test_is_increasing_mul_negative_const() {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let neg = UOp::const_(DType::Int32, ConstValue::Int(-2));
    let scaled = x.try_mul(&neg).unwrap();
    assert!(!scaled.is_increasing()); // Multiplying by negative is not increasing
}

#[test]
fn test_is_increasing_idiv_positive_const() {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let two = UOp::const_(DType::Int32, ConstValue::Int(2));
    let divided = x.idiv(&two);
    assert!(divided.is_increasing());
}

#[test]
fn test_is_increasing_complex() {
    // (x + 5) * 2 should be increasing
    let x = UOp::var("x", DType::Int32, 0, 100);
    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let two = UOp::const_(DType::Int32, ConstValue::Int(2));
    let sum = x.try_add(&five).unwrap();
    let scaled = sum.try_mul(&two).unwrap();
    assert!(scaled.is_increasing());
}
