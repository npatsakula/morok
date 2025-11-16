use crate::{pattern::matcher::RewriteResult, symbolic::symbolic_simple};
use morok_dtype::DType;
use morok_ir::{BinaryOp, ConstValue, Op, UOp};
use std::rc::Rc;

#[test]
fn test_symbolic_simple_identity_folding() {
    let matcher = symbolic_simple();

    // Test: 5 + 0 -> 5
    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let add = UOp::new(Op::Binary(BinaryOp::Add, five.clone(), zero.clone()), DType::Int32);

    let result = matcher.rewrite(&add);
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &five));
    }

    // Test: 0 + 5 -> 5 (commutative)
    let add2 = UOp::new(Op::Binary(BinaryOp::Add, zero.clone(), five.clone()), DType::Int32);
    let result2 = matcher.rewrite(&add2);
    assert!(matches!(result2, RewriteResult::Rewritten(_)));

    // Test: 5 * 1 -> 5
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, five.clone(), one.clone()), DType::Int32);
    let result3 = matcher.rewrite(&mul);
    assert!(matches!(result3, RewriteResult::Rewritten(_)));

    // Test: 5 - 0 -> 5
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, five.clone(), zero.clone()), DType::Int32);
    let result4 = matcher.rewrite(&sub);
    assert!(matches!(result4, RewriteResult::Rewritten(_)));

    // Test: 5 / 1 -> 5 (int division)
    let idiv = UOp::new(Op::Binary(BinaryOp::Idiv, five.clone(), one.clone()), DType::Int32);
    let result5 = matcher.rewrite(&idiv);
    assert!(matches!(result5, RewriteResult::Rewritten(_)));

    // Test: 5.0 / 1.0 -> 5.0 (float division)
    let five_f = UOp::const_(DType::Float32, ConstValue::Float(5.0));
    let one_f = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let fdiv = UOp::new(Op::Binary(BinaryOp::Fdiv, five_f.clone(), one_f), DType::Float32);
    let result6 = matcher.rewrite(&fdiv);
    assert!(matches!(result6, RewriteResult::Rewritten(_)));

    // Test: 5 | 0 -> 5
    let or_op = UOp::new(Op::Binary(BinaryOp::Or, five.clone(), zero.clone()), DType::Int32);
    let result7 = matcher.rewrite(&or_op);
    assert!(matches!(result7, RewriteResult::Rewritten(_)));

    // Test: 5 ^ 0 -> 5
    let xor_op = UOp::new(Op::Binary(BinaryOp::Xor, five.clone(), zero.clone()), DType::Int32);
    let result8 = matcher.rewrite(&xor_op);
    assert!(matches!(result8, RewriteResult::Rewritten(_)));
}

#[test]
fn test_symbolic_simple_zero_propagation() {
    let matcher = symbolic_simple();

    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));

    // Test: 5 * 0 -> 0
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, five.clone(), zero.clone()), DType::Int32);

    let result = matcher.rewrite(&mul);
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
    let result2 = matcher.rewrite(&mul2);
    assert!(matches!(result2, RewriteResult::Rewritten(_)));

    // Test: 5 & 0 -> 0
    let and_op = UOp::new(Op::Binary(BinaryOp::And, five.clone(), zero.clone()), DType::Int32);
    let result3 = matcher.rewrite(&and_op);
    assert!(matches!(result3, RewriteResult::Rewritten(_)));

    // Test: 0 & 5 -> 0 (commutative)
    let and2 = UOp::new(Op::Binary(BinaryOp::And, zero.clone(), five), DType::Int32);
    let result4 = matcher.rewrite(&and2);
    assert!(matches!(result4, RewriteResult::Rewritten(_)));
}

#[test]
fn test_symbolic_simple_no_match() {
    let matcher = symbolic_simple();

    // Test: 5 + 3 (not identity) -> no match
    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let three = UOp::const_(DType::Int32, ConstValue::Int(3));
    let add = UOp::new(Op::Binary(BinaryOp::Add, five.clone(), three), DType::Int32);

    let result = matcher.rewrite(&add);
    assert!(matches!(result, RewriteResult::NoMatch));

    // Test: 5 * 2 (not identity) -> no match
    let two = UOp::const_(DType::Int32, ConstValue::Int(2));
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, five, two), DType::Int32);

    let result2 = matcher.rewrite(&mul);
    assert!(matches!(result2, RewriteResult::NoMatch));
}

// ====== Tests for NEW patterns ======

#[test]
fn test_self_division() {
    // Test: x // x -> 1
    let matcher = symbolic_simple();
    let x = UOp::define_global(1, DType::Int32);
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, x.clone(), x), DType::Int32);

    let result = matcher.rewrite(&div);
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
    let x = UOp::define_global(1, DType::Int32);
    let neg_one = UOp::const_(DType::Int32, ConstValue::Int(-1));
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, x.clone(), neg_one), DType::Int32);

    let result = matcher.rewrite(&div);
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
    let x = UOp::define_global(1, DType::Int32);
    let y = UOp::define_global(2, DType::Int32);

    // Build (x % y) % y
    let inner_mod = UOp::new(Op::Binary(BinaryOp::Mod, x.clone(), y.clone()), DType::Int32);
    let outer_mod = UOp::new(Op::Binary(BinaryOp::Mod, inner_mod.clone(), y.clone()), DType::Int32);

    let result = matcher.rewrite(&outer_mod);
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
    let x = UOp::define_global(1, DType::Int32);
    let and_op = UOp::new(Op::Binary(BinaryOp::And, x.clone(), x.clone()), DType::Int32);

    let result = matcher.rewrite(&and_op);
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_idempotent_or() {
    // Test: x | x -> x
    let matcher = symbolic_simple();
    let x = UOp::define_global(1, DType::Int32);
    let or_op = UOp::new(Op::Binary(BinaryOp::Or, x.clone(), x.clone()), DType::Int32);

    let result = matcher.rewrite(&or_op);
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_non_idempotent_and() {
    // Test: x & y (different variables) -> no match
    let matcher = symbolic_simple();
    let x = UOp::define_global(1, DType::Int32);
    let y = UOp::define_global(2, DType::Int32);
    let and_op = UOp::new(Op::Binary(BinaryOp::And, x, y), DType::Int32);

    let result = matcher.rewrite(&and_op);
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
    let x = UOp::define_global(1, DType::Int32);
    let lt = UOp::new(Op::Binary(BinaryOp::Lt, x.clone(), x), DType::Int32);

    let result = matcher.rewrite(&lt);
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
    let x = UOp::define_global(1, DType::Int32);
    let modulo = UOp::new(Op::Binary(BinaryOp::Mod, x.clone(), x), DType::Int32);

    let result = matcher.rewrite(&modulo);
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
    let x = UOp::define_global(1, DType::Int32);
    let ne = UOp::new(Op::Binary(BinaryOp::Ne, x.clone(), x), DType::Int32);

    let result = matcher.rewrite(&ne);
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
    let x = UOp::define_global(1, DType::Float32);
    let ne = UOp::new(Op::Binary(BinaryOp::Ne, x.clone(), x), DType::Float32);

    let result = matcher.rewrite(&ne);
    // Should not match because floats can have NaN
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ====== Tests for DIVISION patterns ======

#[test]
fn test_float_self_division() {
    // Test: x / x -> 1.0 (float division)
    let matcher = symbolic_simple();
    let x = UOp::define_global(1, DType::Float32);
    let div = UOp::new(Op::Binary(BinaryOp::Fdiv, x.clone(), x), DType::Float32);

    let result = matcher.rewrite(&div);
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
    let x = UOp::define_global(1, DType::Float32);
    let y = UOp::define_global(2, DType::Float32);

    let mul = UOp::new(Op::Binary(BinaryOp::Mul, x.clone(), y.clone()), DType::Float32);
    let div = UOp::new(Op::Binary(BinaryOp::Fdiv, mul, y), DType::Float32);

    let result = matcher.rewrite(&div);
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_int_division_cancel_multiplication() {
    // Test: (x * y) // y -> x (integer division)
    let matcher = symbolic_simple();
    let x = UOp::define_global(1, DType::Int32);
    let y = UOp::define_global(2, DType::Int32);

    let mul = UOp::new(Op::Binary(BinaryOp::Mul, x.clone(), y.clone()), DType::Int32);
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, mul, y), DType::Int32);

    let result = matcher.rewrite(&div);
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

    let result = matcher.rewrite(&cast);
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
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(3.14));
    let cast = UOp::new(Op::Cast { src: float_val, dtype: DType::Int32 }, DType::Int32);

    let result = matcher.rewrite(&cast);
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

    let result = matcher.rewrite(&cast);
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
    let x = UOp::define_global(1, DType::Int32);
    let cast = UOp::new(Op::Cast { src: x.clone(), dtype: DType::Int32 }, DType::Int32);

    let result = matcher.rewrite(&cast);
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_double_cast_collapse() {
    // Test: x.cast(Float32).cast(Int32) -> x.cast(Int32)
    let matcher = symbolic_simple();
    let x = UOp::define_global(1, DType::Int32);

    // First cast: Int32 -> Float32
    let inner_cast = UOp::new(Op::Cast { src: x.clone(), dtype: DType::Float32 }, DType::Float32);

    // Second cast: Float32 -> Int32
    let outer_cast = UOp::new(Op::Cast { src: inner_cast, dtype: DType::Int32 }, DType::Int32);

    let result = matcher.rewrite(&outer_cast);
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
    let x = UOp::define_global(1, DType::Int32);
    let cast = UOp::new(Op::Cast { src: x.clone(), dtype: DType::Float32 }, DType::Float32);

    let result = matcher.rewrite(&cast);
    // Should not match constant folding pattern (not a constant)
    // Should not match noop cast (different dtypes)
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ========== Term Combining Tests ==========

#[test]
fn test_combine_identical_terms() {
    // Test: x + x → 2*x
    let matcher = symbolic_simple();
    let x = UOp::define_global(1, DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), x.clone()), DType::Int32);

    let result = matcher.rewrite(&add);

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
    let x = UOp::define_global(1, DType::Int32);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let term1 = UOp::new(Op::Binary(BinaryOp::Mul, c3, x.clone()), DType::Int32);
    let term2 = UOp::new(Op::Binary(BinaryOp::Mul, c5, x.clone()), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, term1, term2), DType::Int32);

    let result = matcher.rewrite(&add);
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
    let x = UOp::define_global(1, DType::Int32);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let term1 = UOp::new(Op::Binary(BinaryOp::Mul, x.clone(), c3), DType::Int32);
    let term2 = UOp::new(Op::Binary(BinaryOp::Mul, x.clone(), c5), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, term1, term2), DType::Int32);

    let result = matcher.rewrite(&add);
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
    let x = UOp::define_global(1, DType::Int32);
    let y = UOp::define_global(2, DType::Int32);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let term1 = UOp::new(Op::Binary(BinaryOp::Mul, c3, x), DType::Int32);
    let term2 = UOp::new(Op::Binary(BinaryOp::Mul, c5, y), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, term1, term2), DType::Int32);

    let result = matcher.rewrite(&add);
    // Should not combine different variables
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ========== ALU Folding Tests ==========

#[test]
fn test_alu_fold_addition_chain() {
    // Test: (x + 3) + 5 → x + 8
    let matcher = symbolic_simple();
    let x = UOp::define_global(1, DType::Int32);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let add1 = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), c3), DType::Int32);
    let add2 = UOp::new(Op::Binary(BinaryOp::Add, add1, c5), DType::Int32);

    let result = matcher.rewrite(&add2);
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
    let x = UOp::define_global(1, DType::Int32);
    let c2 = UOp::const_(DType::Int32, ConstValue::Int(2));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let mul1 = UOp::new(Op::Binary(BinaryOp::Mul, x.clone(), c2), DType::Int32);
    let mul2 = UOp::new(Op::Binary(BinaryOp::Mul, mul1, c3), DType::Int32);

    let result = matcher.rewrite(&mul2);
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
    let x = UOp::define_global(1, DType::Int32);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, x.clone(), c3), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, sub, c5), DType::Int32);

    let result = matcher.rewrite(&add);
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
    let x = UOp::define_global(1, DType::Int32);
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, x.clone(), c5), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, sub, c3), DType::Int32);

    let result = matcher.rewrite(&add);
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
    let x = UOp::define_global(1, DType::Int32);
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), c5), DType::Int32);
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, add, c3), DType::Int32);

    let result = matcher.rewrite(&sub);
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
    let x = UOp::define_global(1, DType::Int32);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c5 = UOp::const_(DType::Int32, ConstValue::Int(5));
    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), c3), DType::Int32);
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, add, c5), DType::Int32);

    let result = matcher.rewrite(&sub);
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
    let a = UOp::define_global(1, DType::Int32);
    let b = UOp::define_global(2, DType::Int32);
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, a.clone(), b.clone()), DType::Int32);
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, mul, b.clone()), DType::Int32);

    let result = matcher.rewrite(&div);
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
    let a = UOp::define_global(1, DType::Int32);
    let c2 = UOp::const_(DType::Int32, ConstValue::Int(2));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let div1 = UOp::new(Op::Binary(BinaryOp::Idiv, a.clone(), c2), DType::Int32);
    let div2 = UOp::new(Op::Binary(BinaryOp::Idiv, div1, c3), DType::Int32);

    let result = matcher.rewrite(&div2);
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
    let x = UOp::define_global(1, DType::Int32);
    let c12 = UOp::const_(DType::Int32, ConstValue::Int(12));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, c12, x.clone()), DType::Int32);
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, mul, c3), DType::Int32);

    let result = matcher.rewrite(&div);
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
    let x = UOp::define_global(1, DType::Int32);
    let y = UOp::define_global(2, DType::Int32);
    let c6 = UOp::const_(DType::Int32, ConstValue::Int(6));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, c6, x), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, mul, y.clone()), DType::Int32);
    let modulo = UOp::new(Op::Binary(BinaryOp::Mod, add, c3.clone()), DType::Int32);

    let result = matcher.rewrite(&modulo);
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
    let x = UOp::define_global(1, DType::Int32);
    let y = UOp::define_global(2, DType::Int32);
    let c9 = UOp::const_(DType::Int32, ConstValue::Int(9));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, c9, y), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), mul), DType::Int32);
    let modulo = UOp::new(Op::Binary(BinaryOp::Mod, add, c3.clone()), DType::Int32);

    let result = matcher.rewrite(&modulo);
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
    let x = UOp::define_global(1, DType::Int32);
    let y = UOp::define_global(2, DType::Int32);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let add = UOp::new(Op::Binary(BinaryOp::Add, x, y), DType::Int32);
    let modulo = UOp::new(Op::Binary(BinaryOp::Mod, add, c3), DType::Int32);

    let result = matcher.rewrite(&modulo);
    // Should not simplify
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ========== Distribution Pattern Tests ==========

#[test]
fn test_distribute_division_over_addition() {
    // Test: (6*x + 9*y) // 3 → (2*x) + (3*y)
    let matcher = symbolic_simple();
    let x = UOp::define_global(1, DType::Int32);
    let y = UOp::define_global(2, DType::Int32);
    let c6 = UOp::const_(DType::Int32, ConstValue::Int(6));
    let c9 = UOp::const_(DType::Int32, ConstValue::Int(9));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));

    let term1 = UOp::new(Op::Binary(BinaryOp::Mul, c6, x.clone()), DType::Int32);
    let term2 = UOp::new(Op::Binary(BinaryOp::Mul, c9, y.clone()), DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, term1, term2), DType::Int32);
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, add, c3), DType::Int32);

    let result = matcher.rewrite(&div);
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
    let x = UOp::define_global(1, DType::Int32);
    let y = UOp::define_global(2, DType::Int32);
    let c12 = UOp::const_(DType::Int32, ConstValue::Int(12));
    let c6 = UOp::const_(DType::Int32, ConstValue::Int(6));
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));

    let term1 = UOp::new(Op::Binary(BinaryOp::Mul, c12, x.clone()), DType::Int32);
    let term2 = UOp::new(Op::Binary(BinaryOp::Mul, c6, y.clone()), DType::Int32);
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, term1, term2), DType::Int32);
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, sub, c3), DType::Int32);

    let result = matcher.rewrite(&div);
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
    let x = UOp::define_global(1, DType::Int32);
    let y = UOp::define_global(2, DType::Int32);
    let c2 = UOp::const_(DType::Int32, ConstValue::Int(2));

    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), y.clone()), DType::Int32);
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, c2.clone(), add), DType::Int32);

    let result = matcher.rewrite(&mul);
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
    let x = UOp::define_global(1, DType::Int32);
    let y = UOp::define_global(2, DType::Int32);
    let c3 = UOp::const_(DType::Int32, ConstValue::Int(3));

    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), y.clone()), DType::Int32);
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, add, c3.clone()), DType::Int32);

    let result = matcher.rewrite(&mul);
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
    let x = UOp::define_global(1, DType::Int32);
    let y = UOp::define_global(2, DType::Int32);
    let c100 = UOp::const_(DType::Int32, ConstValue::Int(100));

    let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), y.clone()), DType::Int32);
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, add, c100.clone()), DType::Int32);

    let result = matcher.rewrite(&mul);
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
