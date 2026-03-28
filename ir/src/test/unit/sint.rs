use morok_dtype::DType;

use crate::{Op, SInt, UOp, sint_max, sint_min, sint_prod};

#[test]
fn test_sint_const() {
    let s = SInt::from(42);
    assert!(s.is_const());
    assert!(!s.is_symbolic());
    assert_eq!(s.as_const(), Some(42));
}

#[test]
fn test_sint_symbolic() {
    let uop = UOp::index_const(10);
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
    let uop = UOp::index_const(100);
    let s = SInt::from(uop);
    let simplified = s.simplify();
    assert_eq!(simplified.as_const(), Some(100));
}

// =========================================================================
// std::ops arithmetic tests
// =========================================================================

#[test]
fn test_sint_add_concrete() {
    assert_eq!((SInt::from(3) + SInt::from(5)).as_const(), Some(8));
    assert_eq!((SInt::from(10) + 5usize).as_const(), Some(15));
    assert_eq!((5usize + SInt::from(10)).as_const(), Some(15));
}

#[test]
fn test_sint_sub_concrete() {
    assert_eq!((SInt::from(10) - SInt::from(3)).as_const(), Some(7));
    assert_eq!((SInt::from(10) - 3usize).as_const(), Some(7));
}

#[test]
fn test_sint_mul_concrete() {
    assert_eq!((SInt::from(4) * SInt::from(5)).as_const(), Some(20));
    assert_eq!((SInt::from(4) * 5usize).as_const(), Some(20));
    assert_eq!((5usize * SInt::from(4)).as_const(), Some(20));
}

#[test]
fn test_sint_div_concrete() {
    assert_eq!((SInt::from(20) / SInt::from(5)).as_const(), Some(4));
    assert_eq!((SInt::from(20) / 5usize).as_const(), Some(4));
}

#[test]
fn test_sint_ceildiv_concrete() {
    assert_eq!(SInt::from(10).ceildiv(&SInt::from(3)).as_const(), Some(4));
    assert_eq!(SInt::from(9).ceildiv(&SInt::from(3)).as_const(), Some(3));
    assert_eq!(SInt::from(1).ceildiv(&SInt::from(3)).as_const(), Some(1));
}

#[test]
fn test_sint_smax_concrete() {
    assert_eq!(SInt::from(3).smax(&SInt::from(7)).as_const(), Some(7));
    assert_eq!(SInt::from(7).smax(&SInt::from(3)).as_const(), Some(7));
}

#[test]
fn test_sint_smin_concrete() {
    assert_eq!(SInt::from(3).smin(&SInt::from(7)).as_const(), Some(3));
    assert_eq!(SInt::from(7).smin(&SInt::from(3)).as_const(), Some(3));
}

#[test]
fn test_sint_arithmetic_symbolic() {
    let var = UOp::new(Op::DefineVar { name: "N".to_string(), min_val: 1, max_val: 100 }, DType::Index);
    let sym = SInt::from(var);

    // Symbolic + Const → Symbolic
    let result = &sym + 5usize;
    assert!(result.is_symbolic());

    // Const + Symbolic → Symbolic
    let result = 5usize + &sym;
    assert!(result.is_symbolic());

    // Symbolic * Const → Symbolic
    let result = &sym * 3usize;
    assert!(result.is_symbolic());

    // Symbolic ceildiv Const → Symbolic
    let result = sym.ceildiv(&SInt::from(4usize));
    assert!(result.is_symbolic());
}

#[test]
fn test_sint_ref_arithmetic() {
    // Verify ref + ref works (important for pool formulas where values are reused)
    let a = SInt::from(10);
    let b = SInt::from(3);
    assert_eq!((&a + &b).as_const(), Some(13));
    assert_eq!((&a - &b).as_const(), Some(7));
    assert_eq!((&a * &b).as_const(), Some(30));
    assert_eq!((&a / &b).as_const(), Some(3));
}

#[test]
#[should_panic(expected = "arithmetic on SInt::Infer")]
fn test_sint_add_infer_panics() {
    let _ = SInt::Infer + SInt::from(5);
}

#[test]
#[should_panic(expected = "arithmetic on SInt::Infer")]
fn test_sint_mul_infer_panics() {
    let _ = SInt::from(5) * SInt::Infer;
}

#[test]
#[should_panic(expected = "smax on SInt::Infer")]
fn test_sint_smax_infer_panics() {
    let _ = SInt::Infer.smax(&SInt::from(5));
}

#[test]
#[should_panic(expected = "smin on SInt::Infer")]
fn test_sint_smin_infer_panics() {
    let _ = SInt::Infer.smin(&SInt::from(5));
}

#[test]
fn test_sint_display() {
    assert_eq!(format!("{}", SInt::from(42)), "42");
    assert_eq!(format!("{}", SInt::Infer), "-1");
    let var = UOp::new(Op::DefineVar { name: "N".to_string(), min_val: 1, max_val: 100 }, DType::Index);
    assert_eq!(format!("{}", SInt::from(var)), "<symbolic>");
}

#[test]
fn test_sint_pool_formula_concrete() {
    // Verify the pool formula works with concrete values:
    // i=8, k=2, s=2, d=1
    // o = ceildiv(i - d*(k-1), s) = ceildiv(8 - 1*1, 2) = ceildiv(7, 2) = 4
    // f = max(1, ceildiv(o*s - d, i)) = max(1, ceildiv(4*2 - 1, 8)) = max(1, ceildiv(7, 8)) = 1
    let i = SInt::from(8usize);
    let k = 2usize;
    let s = 2usize;
    let d = 1usize;

    let o = (&i - d * (k - 1)).ceildiv(&SInt::from(s));
    assert_eq!(o.as_const(), Some(4));

    let f = SInt::from(1usize).smax(&(&o * s - d).ceildiv(&i));
    assert_eq!(f.as_const(), Some(1));

    // repeat_count = ceildiv(k * (i*f + d), i) = ceildiv(2 * (8*1 + 1), 8) = ceildiv(18, 8) = 3
    let rep = (k * (&i * &f + d)).ceildiv(&i);
    assert_eq!(rep.as_const(), Some(3));
}
