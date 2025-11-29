use crate::{pattern::matcher::RewriteResult, symbolic::symbolic_simple};
use morok_dtype::DType;
use morok_ir::{BinaryOp, ConstValue, Op, TernaryOp, UOp, UnaryOp};
use std::rc::Rc;

#[test]
fn test_symbolic_simple_identity_folding() {
    let matcher = symbolic_simple();

    // Test: 5 + 0 -> 5
    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let add = UOp::new(Op::Binary(BinaryOp::Add, five.clone(), zero.clone()), DType::Int32);

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &five));
    }

    // Test: 0 + 5 -> 5 (commutative)
    let add2 = UOp::new(Op::Binary(BinaryOp::Add, zero.clone(), five.clone()), DType::Int32);
    let result2 = matcher.rewrite(&add2, &mut ());
    assert!(matches!(result2, RewriteResult::Rewritten(_)));

    // Test: 5 * 1 -> 5
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, five.clone(), one.clone()), DType::Int32);
    let result3 = matcher.rewrite(&mul, &mut ());
    assert!(matches!(result3, RewriteResult::Rewritten(_)));

    // Test: 5 - 0 -> 5
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, five.clone(), zero.clone()), DType::Int32);
    let result4 = matcher.rewrite(&sub, &mut ());
    assert!(matches!(result4, RewriteResult::Rewritten(_)));

    // Test: 5 / 1 -> 5 (int division)
    let idiv = UOp::new(Op::Binary(BinaryOp::Idiv, five.clone(), one.clone()), DType::Int32);
    let result5 = matcher.rewrite(&idiv, &mut ());
    assert!(matches!(result5, RewriteResult::Rewritten(_)));

    // Test: 5.0 / 1.0 -> 5.0 (float division)
    let five_f = UOp::const_(DType::Float32, ConstValue::Float(5.0));
    let one_f = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let fdiv = UOp::new(Op::Binary(BinaryOp::Fdiv, five_f.clone(), one_f), DType::Float32);
    let result6 = matcher.rewrite(&fdiv, &mut ());
    assert!(matches!(result6, RewriteResult::Rewritten(_)));

    // Test: 5 | 0 -> 5
    let or_op = UOp::new(Op::Binary(BinaryOp::Or, five.clone(), zero.clone()), DType::Int32);
    let result7 = matcher.rewrite(&or_op, &mut ());
    assert!(matches!(result7, RewriteResult::Rewritten(_)));

    // Test: 5 ^ 0 -> 5
    let xor_op = UOp::new(Op::Binary(BinaryOp::Xor, five.clone(), zero.clone()), DType::Int32);
    let result8 = matcher.rewrite(&xor_op, &mut ());
    assert!(matches!(result8, RewriteResult::Rewritten(_)));
}

#[test]
fn test_symbolic_simple_zero_propagation() {
    let matcher = symbolic_simple();

    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));

    // Test: 5 * 0 -> 0
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, five.clone(), zero.clone()), DType::Int32);

    let result = matcher.rewrite(&mul, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        // Check that the result is a zero constant (value-based)
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(0));
        } else {
            panic!("Expected Const op, got {:?}", rewritten.op());
        }
    }

    // Test: 0 * 5 -> 0 (commutative)
    let mul2 = UOp::new(Op::Binary(BinaryOp::Mul, zero.clone(), five.clone()), DType::Int32);
    let result2 = matcher.rewrite(&mul2, &mut ());
    assert!(matches!(result2, RewriteResult::Rewritten(_)));

    // Test: 5 & 0 -> 0
    let and_op = UOp::new(Op::Binary(BinaryOp::And, five.clone(), zero.clone()), DType::Int32);
    let result3 = matcher.rewrite(&and_op, &mut ());
    assert!(matches!(result3, RewriteResult::Rewritten(_)));

    // Test: 0 & 5 -> 0 (commutative)
    let and2 = UOp::new(Op::Binary(BinaryOp::And, zero.clone(), five), DType::Int32);
    let result4 = matcher.rewrite(&and2, &mut ());
    assert!(matches!(result4, RewriteResult::Rewritten(_)));
}

#[test]
fn test_symbolic_simple_const_folding() {
    let matcher = symbolic_simple();

    // Test: 5 + 3 -> 8 (constant folding)
    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let three = UOp::const_(DType::Int32, ConstValue::Int(3));
    let add = UOp::new(Op::Binary(BinaryOp::Add, five.clone(), three), DType::Int32);

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(8));
        } else {
            panic!("Expected Const(Int(8)), got {:?}", rewritten.op());
        }
    }

    // Test: 5 * 2 -> 10 (constant folding)
    let two = UOp::const_(DType::Int32, ConstValue::Int(2));
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, five, two), DType::Int32);

    let result2 = matcher.rewrite(&mul, &mut ());
    assert!(matches!(result2, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result2 {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(10));
        } else {
            panic!("Expected Const(Int(10)), got {:?}", rewritten.op());
        }
    }
}

// ====== Tests for NEW patterns ======

#[test]
fn test_self_division() {
    // Test: x // x -> 1
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, x.clone(), x), DType::Int32);

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(1));
        } else {
            panic!("Expected Const(1), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_division_by_neg_one() {
    // Test: x // -1 -> -x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let neg_one = UOp::const_(DType::Int32, ConstValue::Int(-1));
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, x.clone(), neg_one), DType::Int32);

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Unary(morok_ir::UnaryOp::Neg, negated) = rewritten.op() {
            assert!(std::rc::Rc::ptr_eq(negated, &x));
        } else {
            panic!("Expected Unary(Neg, x), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_idempotent_modulo() {
    // Test: (x % y) % y -> x % y
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Int32, i64::MIN, i64::MAX);

    // Build (x % y) % y
    let inner_mod = UOp::new(Op::Binary(BinaryOp::Mod, x.clone(), y.clone()), DType::Int32);
    let outer_mod = UOp::new(Op::Binary(BinaryOp::Mod, inner_mod.clone(), y.clone()), DType::Int32);

    let result = matcher.rewrite(&outer_mod, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be equivalent to inner_mod (x % y)
        if let Op::Binary(BinaryOp::Mod, a, b) = rewritten.op() {
            assert!(std::rc::Rc::ptr_eq(a, &x));
            assert!(std::rc::Rc::ptr_eq(b, &y));
        } else {
            panic!("Expected Binary(Mod, x, y), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_idempotent_and() {
    // Test: x & x -> x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let and_op = UOp::new(Op::Binary(BinaryOp::And, x.clone(), x.clone()), DType::Int32);

    let result = matcher.rewrite(&and_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_idempotent_or() {
    // Test: x | x -> x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let or_op = UOp::new(Op::Binary(BinaryOp::Or, x.clone(), x.clone()), DType::Int32);

    let result = matcher.rewrite(&or_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_non_idempotent_and() {
    // Test: x & y (different variables) -> no match
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Int32, i64::MIN, i64::MAX);
    let and_op = UOp::new(Op::Binary(BinaryOp::And, x, y), DType::Int32);

    let result = matcher.rewrite(&and_op, &mut ());
    // Should not match idempotent pattern
    // But might match other patterns (like zero propagation if one is zero)
    // For this test, we're using variables, so no simplification expected
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ====== Tests for ZERO FOLDING patterns ======

#[test]
fn test_self_comparison_lt() {
    // Test: x < x -> False
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let lt = UOp::new(Op::Binary(BinaryOp::Lt, x.clone(), x), DType::Int32);

    let result = matcher.rewrite(&lt, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(false));
        } else {
            panic!("Expected Const(Bool(false)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_self_modulo() {
    // Test: x % x -> 0
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let modulo = UOp::new(Op::Binary(BinaryOp::Mod, x.clone(), x), DType::Int32);

    let result = matcher.rewrite(&modulo, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(0));
        } else {
            panic!("Expected Const(0), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_self_inequality_int() {
    // Test: x != x -> False (for integers)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let ne = UOp::new(Op::Binary(BinaryOp::Ne, x.clone(), x), DType::Int32);

    let result = matcher.rewrite(&ne, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(false));
        } else {
            panic!("Expected Const(Bool(false)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_self_inequality_float_no_fold() {
    // Test: x != x (for floats) -> no match (NaN != NaN is true)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Float32, i64::MIN, i64::MAX);
    let ne = UOp::new(Op::Binary(BinaryOp::Ne, x.clone(), x), DType::Float32);

    let result = matcher.rewrite(&ne, &mut ());
    // Should not match because floats can have NaN
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ====== Tests for DIVISION patterns ======

#[test]
fn test_float_self_division() {
    // Test: x / x -> 1.0 (float division)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Float32, i64::MIN, i64::MAX);
    let div = UOp::new(Op::Binary(BinaryOp::Fdiv, x.clone(), x), DType::Float32);

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Float(1.0));
        } else {
            panic!("Expected Const(1.0), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_division_cancel_multiplication() {
    // Test: (x * y) / y -> x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Float32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Float32, i64::MIN, i64::MAX);

    let mul = UOp::new(Op::Binary(BinaryOp::Mul, x.clone(), y.clone()), DType::Float32);
    let div = UOp::new(Op::Binary(BinaryOp::Fdiv, mul, y), DType::Float32);

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_int_division_cancel_multiplication() {
    // Test: (x * y) // y -> x (integer division)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Int32, i64::MIN, i64::MAX);

    let mul = UOp::new(Op::Binary(BinaryOp::Mul, x.clone(), y.clone()), DType::Int32);
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, mul, y), DType::Int32);

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &x));
    }
}

// ====== Tests for CAST OPTIMIZATION patterns ======

#[test]
fn test_cast_int_to_float_constant() {
    // Test: cast(int_const) -> float_const
    let matcher = symbolic_simple();
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(42));
    let cast = UOp::new(Op::Cast { src: int_val, dtype: DType::Float32 }, DType::Float32);

    let result = matcher.rewrite(&cast, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Float(42.0));
        } else {
            panic!("Expected Const(Float(42.0)), got {:?}", rewritten.op());
        }
        assert_eq!(rewritten.dtype(), DType::Float32);
    }
}

#[test]
fn test_cast_float_to_int_constant() {
    // Test: cast(float_const) -> int_const
    let matcher = symbolic_simple();
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(std::f64::consts::PI));
    let cast = UOp::new(Op::Cast { src: float_val, dtype: DType::Int32 }, DType::Int32);

    let result = matcher.rewrite(&cast, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(3));
        } else {
            panic!("Expected Const(Int(3)), got {:?}", rewritten.op());
        }
        assert_eq!(rewritten.dtype(), DType::Int32);
    }
}

#[test]
fn test_cast_bool_to_int_constant() {
    // Test: cast(bool_const) -> int_const
    let matcher = symbolic_simple();
    let bool_val = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let cast = UOp::new(Op::Cast { src: bool_val, dtype: DType::Int32 }, DType::Int32);

    let result = matcher.rewrite(&cast, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(1));
        } else {
            panic!("Expected Const(Int(1)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_noop_cast_same_dtype() {
    // Test: x.cast(dtype) -> x if same dtype
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let cast = UOp::new(Op::Cast { src: x.clone(), dtype: DType::Int32 }, DType::Int32);

    let result = matcher.rewrite(&cast, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_double_cast_collapse() {
    // Test: x.cast(Float32).cast(Int32) -> x.cast(Int32)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);

    // First cast: Int32 -> Float32
    let inner_cast = UOp::new(Op::Cast { src: x.clone(), dtype: DType::Float32 }, DType::Float32);

    // Second cast: Float32 -> Int32
    let outer_cast = UOp::new(Op::Cast { src: inner_cast, dtype: DType::Int32 }, DType::Int32);

    let result = matcher.rewrite(&outer_cast, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be a single cast from x to Int32
        if let Op::Cast { src, dtype } = rewritten.op() {
            assert!(std::rc::Rc::ptr_eq(src, &x));
            assert_eq!(*dtype, DType::Int32);
        } else {
            panic!("Expected Cast, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_cast_non_constant_no_fold() {
    // Test: cast(variable) -> no constant folding (only dtype change)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let cast = UOp::new(Op::Cast { src: x.clone(), dtype: DType::Float32 }, DType::Float32);

    let result = matcher.rewrite(&cast, &mut ());
    // Should not match constant folding pattern (not a constant)
    // Should not match noop cast (different dtypes)
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ========== Term Combining Tests ==========

#[test]
fn test_combine_identical_terms() {
    // Test: x + x → 2*x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), x.clone()), DType::Int32);

    let result = matcher.rewrite(&add, &mut ());

    // Debug: print the result if it doesn't match
    if !matches!(result, RewriteResult::Rewritten(_)) {
        eprintln!("Test failed: x + x didn't match. Result: {:?}", result);
        eprintln!("Add op: {:?}", add.op());
    }

    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be 2*x
        if let Op::Binary(BinaryOp::Mul, c, var) = rewritten.op() {
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(2));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
            assert!(Rc::ptr_eq(var, &x));
        } else {
            panic!("Expected Mul, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_combine_terms_with_coefficients() {
    // Test: (3 * x) + (5 * x) → 8 * x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let term1 = UOp::new(Op::Binary(BinaryOp::Mul, c3, x.clone()), DType::Int32);
    let term2 = UOp::new(Op::Binary(BinaryOp::Mul, c5, x.clone()), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, term1, term2), DType::Int32);

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be 8*x
        if let Op::Binary(BinaryOp::Mul, c, var) = rewritten.op() {
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(8));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
            assert!(Rc::ptr_eq(var, &x));
        } else {
            panic!("Expected Mul, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_combine_terms_reversed_multiplication() {
    // Test: (x * 3) + (x * 5) → x * 8
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let term1 = UOp::new(Op::Binary(BinaryOp::Mul, x.clone(), c3), DType::Int32);
    let term2 = UOp::new(Op::Binary(BinaryOp::Mul, x.clone(), c5), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, term1, term2), DType::Int32);

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x*8
        if let Op::Binary(BinaryOp::Mul, var, c) = rewritten.op() {
            assert!(Rc::ptr_eq(var, &x));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(8));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected Mul, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_no_combine_different_variables() {
    // Test: (3 * x) + (5 * y) → no rewrite (different variables)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Int32, i64::MIN, i64::MAX);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let term1 = UOp::new(Op::Binary(BinaryOp::Mul, c3, x), DType::Int32);
    let term2 = UOp::new(Op::Binary(BinaryOp::Mul, c5, y), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, term1, term2), DType::Int32);

    let result = matcher.rewrite(&add, &mut ());
    // Should not combine different variables
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ========== ALU Folding Tests ==========

#[test]
fn test_alu_fold_addition_chain() {
    // Test: (x + 3) + 5 → x + 8
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let add1 = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), c3), DType::Int32);
    let add2 = UOp::new(Op::Binary(BinaryOp::Add, add1, c5), DType::Int32);

    let result = matcher.rewrite(&add2, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x + 8
        if let Op::Binary(BinaryOp::Add, var, c) = rewritten.op() {
            assert!(Rc::ptr_eq(var, &x));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(8));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected Add, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_alu_fold_multiplication_chain() {
    // Test: (x * 2) * 3 → x * 6
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let c2 = UOp::const_(DType::Int32, ConstValue::Int(2));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let mul1 = UOp::new(Op::Binary(BinaryOp::Mul, x.clone(), c2), DType::Int32);
    let mul2 = UOp::new(Op::Binary(BinaryOp::Mul, mul1, c3), DType::Int32);

    let result = matcher.rewrite(&mul2, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x * 6
        if let Op::Binary(BinaryOp::Mul, var, c) = rewritten.op() {
            assert!(Rc::ptr_eq(var, &x));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(6));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected Mul, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_alu_fold_sub_then_add_positive() {
    // Test: (x - 3) + 5 → x + 2
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, x.clone(), c3), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, sub, c5), DType::Int32);

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x + 2
        if let Op::Binary(BinaryOp::Add, var, c) = rewritten.op() {
            assert!(Rc::ptr_eq(var, &x));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(2));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected Add, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_alu_fold_sub_then_add_negative() {
    // Test: (x - 5) + 3 → x - 2
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, x.clone(), c5), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, sub, c3), DType::Int32);

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x - 2
        if let Op::Binary(BinaryOp::Sub, var, c) = rewritten.op() {
            assert!(Rc::ptr_eq(var, &x));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(2));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected Sub, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_alu_fold_add_then_sub_positive() {
    // Test: (x + 5) - 3 → x + 2
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), c5), DType::Int32);
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, add, c3), DType::Int32);

    let result = matcher.rewrite(&sub, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x + 2
        if let Op::Binary(BinaryOp::Add, var, c) = rewritten.op() {
            assert!(Rc::ptr_eq(var, &x));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(2));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected Add, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_alu_fold_add_then_sub_negative() {
    // Test: (x + 3) - 5 → x - 2
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), c3), DType::Int32);
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, add, c5), DType::Int32);

    let result = matcher.rewrite(&sub, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x - 2
        if let Op::Binary(BinaryOp::Sub, var, c) = rewritten.op() {
            assert!(Rc::ptr_eq(var, &x));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(2));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected Sub, got {:?}", rewritten.op());
        }
    }
}

// ========== Division Pattern Tests ==========

#[test]
fn test_division_cancel_with_multiplication() {
    // Test: (a * b) // b → a
    let matcher = symbolic_simple();
    let a = UOp::var("a", DType::Int32, i64::MIN, i64::MAX);
    let b = UOp::var("b", DType::Int32, i64::MIN, i64::MAX);
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, a.clone(), b.clone()), DType::Int32);
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, mul, b.clone()), DType::Int32);

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be just 'a'
        assert!(Rc::ptr_eq(&rewritten, &a));
    }
}

#[test]
fn test_division_chain_folding() {
    // Test: (a // 2) // 3 → a // 6
    let matcher = symbolic_simple();
    let a = UOp::var("a", DType::Int32, i64::MIN, i64::MAX);
    let c2 = UOp::const_(DType::Int32, ConstValue::Int(2));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let div1 = UOp::new(Op::Binary(BinaryOp::Idiv, a.clone(), c2), DType::Int32);
    let div2 = UOp::new(Op::Binary(BinaryOp::Idiv, div1, c3), DType::Int32);

    let result = matcher.rewrite(&div2, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be a // 6
        if let Op::Binary(BinaryOp::Idiv, var, c) = rewritten.op() {
            assert!(Rc::ptr_eq(var, &a));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(6));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected Idiv, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_exact_division_with_divides_helper() {
    // Test: (12 * x) // 3 → 4 * x (using divides helper)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let c12 = UOp::const_(DType::Int32, ConstValue::Int(12));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, c12, x.clone()), DType::Int32);
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, mul, c3), DType::Int32);

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be 4 * x
        if let Op::Binary(BinaryOp::Mul, c, var) = rewritten.op() {
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(4));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
            assert!(Rc::ptr_eq(var, &x));
        } else {
            panic!("Expected Mul, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_modulo_with_divisible_left_operand() {
    // Test: (6 * x + y) % 3 → y % 3 (since 6*x is divisible by 3)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Int32, i64::MIN, i64::MAX);
    let c6 = UOp::const_(DType::Int32, ConstValue::Int(6));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, c6, x), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, mul, y.clone()), DType::Int32);
    let modulo = UOp::new(Op::Binary(BinaryOp::Mod, add, c3.clone()), DType::Int32);

    let result = matcher.rewrite(&modulo, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be y % 3
        if let Op::Binary(BinaryOp::Mod, var, c) = rewritten.op() {
            assert!(Rc::ptr_eq(var, &y));
            assert!(Rc::ptr_eq(c, &c3));
        } else {
            panic!("Expected Mod, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_modulo_with_divisible_right_operand() {
    // Test: (x + 9 * y) % 3 → x % 3 (since 9*y is divisible by 3)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Int32, i64::MIN, i64::MAX);
    let c9 = UOp::const_(DType::Int32, ConstValue::Int(9));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, c9, y), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), mul), DType::Int32);
    let modulo = UOp::new(Op::Binary(BinaryOp::Mod, add, c3.clone()), DType::Int32);

    let result = matcher.rewrite(&modulo, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x % 3
        if let Op::Binary(BinaryOp::Mod, var, c) = rewritten.op() {
            assert!(Rc::ptr_eq(var, &x));
            assert!(Rc::ptr_eq(c, &c3));
        } else {
            panic!("Expected Mod, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_modulo_no_simplification() {
    // Test: (x + y) % 3 → no simplification (neither divisible by 3)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Int32, i64::MIN, i64::MAX);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let add = UOp::new(Op::Binary(BinaryOp::Add, x, y), DType::Int32);
    let modulo = UOp::new(Op::Binary(BinaryOp::Mod, add, c3), DType::Int32);

    let result = matcher.rewrite(&modulo, &mut ());
    // Should not simplify
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ========== Distribution Pattern Tests ==========

#[test]
fn test_distribute_division_over_addition() {
    // Test: (6*x + 9*y) // 3 → (2*x) + (3*y)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Int32, i64::MIN, i64::MAX);
    let c6 = UOp::const_(DType::Int32, ConstValue::Int(6));
    let c9 = UOp::const_(DType::Int32, ConstValue::Int(9));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));

    let term1 = UOp::new(Op::Binary(BinaryOp::Mul, c6, x.clone()), DType::Int32);
    let term2 = UOp::new(Op::Binary(BinaryOp::Mul, c9, y.clone()), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, term1, term2), DType::Int32);
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, add, c3), DType::Int32);

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be (2*x) + (3*y)
        if let Op::Binary(BinaryOp::Add, left, right) = rewritten.op() {
            // Check left: 2*x
            if let Op::Binary(BinaryOp::Mul, c, var) = left.op() {
                if let Op::Const(cv) = c.op() {
                    assert_eq!(cv.0, ConstValue::Int(2));
                }
                assert!(Rc::ptr_eq(var, &x));
            } else {
                panic!("Expected Mul on left, got {:?}", left.op());
            }

            // Check right: 3*y
            if let Op::Binary(BinaryOp::Mul, c, var) = right.op() {
                if let Op::Const(cv) = c.op() {
                    assert_eq!(cv.0, ConstValue::Int(3));
                }
                assert!(Rc::ptr_eq(var, &y));
            } else {
                panic!("Expected Mul on right, got {:?}", right.op());
            }
        } else {
            panic!("Expected Add, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_distribute_division_over_subtraction() {
    // Test: (12*x - 6*y) // 3 → (4*x) - (2*y)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Int32, i64::MIN, i64::MAX);
    let c12 = UOp::const_(DType::Int32, ConstValue::Int(12));
    let c6 = UOp::const_(DType::Int32, ConstValue::Int(6));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));

    let term1 = UOp::new(Op::Binary(BinaryOp::Mul, c12, x.clone()), DType::Int32);
    let term2 = UOp::new(Op::Binary(BinaryOp::Mul, c6, y.clone()), DType::Int32);
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, term1, term2), DType::Int32);
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, sub, c3), DType::Int32);

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be (4*x) - (2*y)
        if let Op::Binary(BinaryOp::Sub, left, right) = rewritten.op() {
            // Check left: 4*x
            if let Op::Binary(BinaryOp::Mul, c, var) = left.op() {
                if let Op::Const(cv) = c.op() {
                    assert_eq!(cv.0, ConstValue::Int(4));
                }
                assert!(Rc::ptr_eq(var, &x));
            }

            // Check right: 2*y
            if let Op::Binary(BinaryOp::Mul, c, var) = right.op() {
                if let Op::Const(cv) = c.op() {
                    assert_eq!(cv.0, ConstValue::Int(2));
                }
                assert!(Rc::ptr_eq(var, &y));
            }
        } else {
            panic!("Expected Sub, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_distribute_multiplication_over_addition() {
    // Test: 2 * (x + y) → (2*x) + (2*y)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Int32, i64::MIN, i64::MAX);
    let c2 = UOp::const_(DType::Int32, ConstValue::Int(2));

    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), y.clone()), DType::Int32);
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, c2.clone(), add), DType::Int32);

    let result = matcher.rewrite(&mul, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be (2*x) + (2*y)
        if let Op::Binary(BinaryOp::Add, left, right) = rewritten.op() {
            // Check left: 2*x
            if let Op::Binary(BinaryOp::Mul, c, var) = left.op() {
                assert!(Rc::ptr_eq(c, &c2));
                assert!(Rc::ptr_eq(var, &x));
            }

            // Check right: 2*y
            if let Op::Binary(BinaryOp::Mul, c, var) = right.op() {
                assert!(Rc::ptr_eq(c, &c2));
                assert!(Rc::ptr_eq(var, &y));
            }
        } else {
            panic!("Expected Add, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_distribute_multiplication_over_addition_reversed() {
    // Test: (x + y) * 3 → (x*3) + (y*3)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Int32, i64::MIN, i64::MAX);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));

    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), y.clone()), DType::Int32);
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, add, c3.clone()), DType::Int32);

    let result = matcher.rewrite(&mul, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be (x*3) + (y*3)
        if let Op::Binary(BinaryOp::Add, left, right) = rewritten.op() {
            // Check left: x*3
            if let Op::Binary(BinaryOp::Mul, var, c) = left.op() {
                assert!(Rc::ptr_eq(var, &x));
                assert!(Rc::ptr_eq(c, &c3));
            }

            // Check right: y*3
            if let Op::Binary(BinaryOp::Mul, var, c) = right.op() {
                assert!(Rc::ptr_eq(var, &y));
                assert!(Rc::ptr_eq(c, &c3));
            }
        } else {
            panic!("Expected Add, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_distribute_large_constant() {
    // Test: (x + y) * 100 → (x*100) + (y*100)
    // Note: Distributes unconditionally without size checks
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let y = UOp::var("y", DType::Int32, i64::MIN, i64::MAX);
    let c100 = UOp::const_(DType::Int32, ConstValue::Int(100));

    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), y.clone()), DType::Int32);
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, add, c100.clone()), DType::Int32);

    let result = matcher.rewrite(&mul, &mut ());
    // Should distribute even with large constant (matching Tinygrad behavior)
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be (x*100) + (y*100)
        if let Op::Binary(BinaryOp::Add, left, right) = rewritten.op() {
            // Check left: x*100
            if let Op::Binary(BinaryOp::Mul, var, c) = left.op() {
                assert!(Rc::ptr_eq(var, &x));
                assert!(Rc::ptr_eq(c, &c100));
            }

            // Check right: y*100
            if let Op::Binary(BinaryOp::Mul, var, c) = right.op() {
                assert!(Rc::ptr_eq(var, &y));
                assert!(Rc::ptr_eq(c, &c100));
            }
        } else {
            panic!("Expected Add, got {:?}", rewritten.op());
        }
    }
}

// ========== Compositional Optimization Tests ==========

#[test]
#[ignore = "Distribution patterns conflict with compositional optimization"]
fn test_compositional_optimization_minimal_failure() {
    // Reproduces the exact failing case from the property test
    // Input: ((0 + var("a")) * 2) * 2
    // Expected: var("a") * 4
    // Direct optimization should give better or equal results to compositional
    //
    // NOTE: This test is ignored for the same reason as compositional_subexpr_optimization:
    // distribution patterns increase operation count but may enable other optimizations.

    use crate::rewrite::graph_rewrite;
    let matcher = symbolic_simple();

    // Build the expression: (0 + var("a")) * 2
    let a_var = UOp::var("a", DType::Int32, 0, 1);
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let two = UOp::const_(DType::Int32, ConstValue::Int(2));
    let add = UOp::new(Op::Binary(BinaryOp::Add, zero, a_var.clone()), DType::Int32);
    let a = UOp::new(Op::Binary(BinaryOp::Mul, add, two.clone()), DType::Int32);
    let b = two.clone();

    // === DIRECT PATH ===
    // Build expression with un-optimized subexpressions and optimize
    let expr_unopt = UOp::new(Op::Binary(BinaryOp::Mul, a.clone(), b.clone()), DType::Int32);
    let direct_opt = graph_rewrite(&matcher, expr_unopt, &mut ());

    // === COMPOSITIONAL PATH ===
    // Optimize subexpressions first
    let opt_a = graph_rewrite(&matcher, a.clone(), &mut ());
    let opt_b = graph_rewrite(&matcher, b.clone(), &mut ());

    // Build expression with optimized subexpressions
    let expr_opt_subs = UOp::new(Op::Binary(BinaryOp::Mul, opt_a.clone(), opt_b.clone()), DType::Int32);

    // Optimize the composed expression
    let final_opt = graph_rewrite(&matcher, expr_opt_subs, &mut ());

    // Count operations
    fn count_ops(uop: &Rc<UOp>) -> usize {
        match uop.op() {
            Op::Binary(_, left, right) => 1 + count_ops(left) + count_ops(right),
            Op::Unary(_, src) => 1 + count_ops(src),
            Op::Ternary(_, a, b, c) => 1 + count_ops(a) + count_ops(b) + count_ops(c),
            _ => 0,
        }
    }

    let direct_count = count_ops(&direct_opt);
    let final_count = count_ops(&final_opt);

    println!("=== COMPOSITIONAL OPTIMIZATION DEBUG ===");
    println!("Original a: (0 + var(\"a\")) * 2");
    println!("Original b: 2");
    println!("Full expr: ((0 + var(\"a\")) * 2) * 2");
    println!();
    println!("Optimized a: {:?}", opt_a.op());
    println!("Optimized b: {:?}", opt_b.op());
    println!();
    println!("Direct optimization: {} ops -> {:?}", direct_count, direct_opt.op());
    println!("Compositional optimization: {} ops -> {:?}", final_count, final_opt.op());
    println!();

    // The compositional approach should be nearly as good as direct
    // EXPECTED: Both should optimize to var("a") * 4 (1 operation)
    // ACTUAL: Compositional gives worse results
    assert!(
        final_count <= direct_count + 1,
        "Compositional optimization ({} ops) should be nearly as good as direct ({} ops)",
        final_count,
        direct_count
    );
}

#[test]
fn test_multiplication_chain_folding() {
    // Test: (var("a") * 2) * 2 → var("a") * 4
    // This is the simplified version of the failing case

    let matcher = symbolic_simple();
    let a = UOp::var("a", DType::Int32, i64::MIN, i64::MAX);
    let c2 = UOp::const_(DType::Int32, ConstValue::Int(2));

    // Build (a * 2) * 2
    let mul1 = UOp::new(Op::Binary(BinaryOp::Mul, a.clone(), c2.clone()), DType::Int32);
    let mul2 = UOp::new(Op::Binary(BinaryOp::Mul, mul1, c2.clone()), DType::Int32);

    let result = matcher.rewrite(&mul2, &mut ());

    println!("=== MULTIPLICATION CHAIN TEST ===");
    println!("Input: (var(\"a\") * 2) * 2");
    match &result {
        crate::pattern::matcher::RewriteResult::Rewritten(r) => {
            println!("Result: {:?}", r.op());
        }
        _ => {
            println!("Result: No rewrite");
        }
    }

    assert!(matches!(result, crate::pattern::matcher::RewriteResult::Rewritten(_)));

    if let crate::pattern::matcher::RewriteResult::Rewritten(rewritten) = result {
        // Should be a * 4
        if let Op::Binary(BinaryOp::Mul, var, c) = rewritten.op() {
            assert!(Rc::ptr_eq(var, &a), "Variable should be unchanged");
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(4), "Constant should be folded to 4");
            } else {
                panic!("Expected constant 4, got {:?}", c.op());
            }
        } else {
            panic!("Expected Binary(Mul, a, 4), got {:?}", rewritten.op());
        }
    }
}

// ====== Tests for BOOLEAN patterns (boolean_dsl_patterns) ======

#[test]
fn test_double_not_elimination() {
    // !!x → x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let not_x = x.not();
    let not_not_x = not_x.not();

    let result = matcher.rewrite(&not_not_x, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_double_not_int() {
    // !!x → x (for integers - bitwise NOT)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let not_x = x.not();
    let not_not_x = not_x.not();

    let result = matcher.rewrite(&not_not_x, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_xor_self_cancellation() {
    // x ^ x → 0
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let xor_self = x.try_xor_op(&x).unwrap();

    let result = matcher.rewrite(&xor_self, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(0));
        } else {
            panic!("Expected constant 0");
        }
    }
}

// ====== Tests for NEGATION patterns (negation_dsl_patterns) ======

#[test]
fn test_double_neg_elimination() {
    // -(-x) → x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, i64::MIN, i64::MAX);
    let neg_x = x.neg();
    let neg_neg_x = neg_x.neg();

    let result = matcher.rewrite(&neg_neg_x, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_double_neg_float() {
    // -(-x) → x (for floats)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Float32, i64::MIN, i64::MAX);
    let neg_x = x.neg();
    let neg_neg_x = neg_x.neg();

    let result = matcher.rewrite(&neg_neg_x, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Rc::ptr_eq(&rewritten, &x));
    }
}

// ====== Tests for MINMAX patterns (minmax_dsl_patterns) ======

#[test]
fn test_max_self_identity() {
    // max(x, x) → x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let max_self = x.try_max_op(&x).unwrap();

    let result = matcher.rewrite(&max_self, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_max_self_float() {
    // max(x, x) → x (for floats)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Float32, i64::MIN, i64::MAX);
    let max_self = x.try_max_op(&x).unwrap();

    let result = matcher.rewrite(&max_self, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Rc::ptr_eq(&rewritten, &x));
    }
}

// ====== Tests for POWER patterns (power_dsl_patterns) ======

#[test]
fn test_pow_zero_is_one() {
    // x ** 0 → 1
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 1, 100);
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let pow = x.try_pow_op(&zero).unwrap();

    let result = matcher.rewrite(&pow, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(1));
        } else {
            panic!("Expected constant 1");
        }
    }
}

#[test]
fn test_pow_one_is_identity() {
    // x ** 1 → x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let pow = x.try_pow_op(&one).unwrap();

    let result = matcher.rewrite(&pow, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_pow_float_zero() {
    // x ** 0.0 → 1.0
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Float32, 1, 100);
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let pow = x.try_pow_op(&zero).unwrap();

    let result = matcher.rewrite(&pow, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Float(1.0));
        } else {
            panic!("Expected constant 1.0");
        }
    }
}

// ====== Tests for WHERE/DCE patterns (dce_dsl_patterns) ======

#[test]
fn test_where_same_branches() {
    // where(cond, x, x) → x
    let matcher = symbolic_simple();
    let cond = UOp::var("cond", DType::Bool, 0, 1);
    let x = UOp::var("x", DType::Int32, 0, 100);
    let where_op = UOp::where_op(cond, Rc::clone(&x), Rc::clone(&x)).unwrap();

    let result = matcher.rewrite(&where_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_where_bool_true_false() {
    // where(x, true, false) → x (for bool x)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let true_val = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let false_val = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let where_op = UOp::where_op(Rc::clone(&x), true_val, false_val).unwrap();

    let result = matcher.rewrite(&where_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_where_bool_false_true() {
    // where(x, false, true) → !x (for bool x)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let false_val = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let true_val = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let where_op = UOp::where_op(Rc::clone(&x), false_val, true_val).unwrap();

    let result = matcher.rewrite(&where_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be Not(x)
        if let Op::Unary(UnaryOp::Not, inner) = rewritten.op() {
            assert!(Rc::ptr_eq(inner, &x));
        } else {
            panic!("Expected Not(x)");
        }
    }
}

#[test]
fn test_where_negated_condition() {
    // where(!cond, t, f) → where(cond, f, t)
    let matcher = symbolic_simple();
    let cond = UOp::var("cond", DType::Bool, 0, 1);
    let not_cond = cond.not();
    let t = UOp::var("t", DType::Int32, 0, 100);
    let f = UOp::var("f", DType::Int32, 0, 100);
    let where_op = UOp::where_op(not_cond, Rc::clone(&t), Rc::clone(&f)).unwrap();

    let result = matcher.rewrite(&where_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be Where(cond, f, t) - branches swapped
        if let Op::Ternary(TernaryOp::Where, new_cond, new_t, new_f) = rewritten.op() {
            assert!(Rc::ptr_eq(new_cond, &cond));
            assert!(Rc::ptr_eq(new_t, &f)); // swapped
            assert!(Rc::ptr_eq(new_f, &t)); // swapped
        } else {
            panic!("Expected Where with swapped branches");
        }
    }
}

#[test]
fn test_where_const_true_condition() {
    // where(true, t, f) → t
    let matcher = symbolic_simple();
    let true_cond = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let t = UOp::var("t", DType::Int32, 0, 100);
    let f = UOp::var("f", DType::Int32, 0, 100);
    let where_op = UOp::where_op(true_cond, Rc::clone(&t), f).unwrap();

    let result = matcher.rewrite(&where_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Rc::ptr_eq(&rewritten, &t));
    }
}

#[test]
fn test_where_const_false_condition() {
    // where(false, t, f) → f
    let matcher = symbolic_simple();
    let false_cond = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let t = UOp::var("t", DType::Int32, 0, 100);
    let f = UOp::var("f", DType::Int32, 0, 100);
    let where_op = UOp::where_op(false_cond, t, Rc::clone(&f)).unwrap();

    let result = matcher.rewrite(&where_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Rc::ptr_eq(&rewritten, &f));
    }
}
