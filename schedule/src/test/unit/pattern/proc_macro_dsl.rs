//! Tests for the patterns! proc-macro DSL.

use std::rc::Rc;

use morok_dtype::DType;
use morok_ir::{BinaryOp, ConstValue, Op, UOp};

use crate::pattern::UPat;
use crate::pattern::{BindingStore, BindingStoreExt, VarIntern};
use crate::pattern::matcher::RewriteResult;
use crate::patterns;

/// Helper to create a binary operation UOp
fn binary(op: BinaryOp, lhs: Rc<UOp>, rhs: Rc<UOp>) -> Rc<UOp> {
    let dtype = lhs.dtype();
    UOp::new(Op::Binary(op, lhs, rhs), dtype)
}

#[test]
fn test_simple_add_zero_pattern() {
    let matcher = patterns! {
        Add(x, Const(0)) ~> x
    };

    // Create x + 0
    let x = UOp::const_(DType::Int32, ConstValue::Int(42));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let add = binary(BinaryOp::Add, x.clone(), zero);

    let result = matcher.rewrite(&add);

    match result {
        RewriteResult::Rewritten(rewritten) => {
            assert!(Rc::ptr_eq(&rewritten, &x), "Should rewrite to x");
        }
        _ => panic!("Expected rewrite to succeed"),
    }
}

#[test]
fn test_mul_one_pattern() {
    let matcher = patterns! {
        Mul(x, Const(1)) ~> x
    };

    // Create x * 1
    let x = UOp::const_(DType::Int32, ConstValue::Int(42));
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let mul = binary(BinaryOp::Mul, x.clone(), one);

    let result = matcher.rewrite(&mul);

    match result {
        RewriteResult::Rewritten(rewritten) => {
            assert!(Rc::ptr_eq(&rewritten, &x), "Should rewrite to x");
        }
        _ => panic!("Expected rewrite to succeed"),
    }
}

#[test]
fn test_binding_pattern() {
    let matcher = patterns! {
        Mul(_, zero @ Const(0)) ~> zero
    };

    // Create x * 0
    let x = UOp::const_(DType::Int32, ConstValue::Int(42));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let mul = binary(BinaryOp::Mul, x, zero.clone());

    let result = matcher.rewrite(&mul);

    match result {
        RewriteResult::Rewritten(rewritten) => {
            assert!(Rc::ptr_eq(&rewritten, &zero), "Should rewrite to zero");
        }
        _ => panic!("Expected rewrite to succeed"),
    }
}

#[test]
fn test_multiple_patterns() {
    let matcher = patterns! {
        Add(x, Const(0)) ~> x,
        Mul(x, Const(1)) ~> x,
        Mul(_, zero @ Const(0)) ~> zero,
    };

    // Test x + 0
    let x = UOp::const_(DType::Int32, ConstValue::Int(42));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));

    let add_zero = binary(BinaryOp::Add, x.clone(), zero.clone());
    match matcher.rewrite(&add_zero) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, 0) should rewrite to x"),
    }

    // Test x * 1
    let mul_one = binary(BinaryOp::Mul, x.clone(), one);
    match matcher.rewrite(&mul_one) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Mul(x, 1) should rewrite to x"),
    }

    // Test x * 0
    let mul_zero = binary(BinaryOp::Mul, x, zero.clone());
    match matcher.rewrite(&mul_zero) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &zero)),
        _ => panic!("Mul(x, 0) should rewrite to 0"),
    }
}

#[test]
fn test_no_match() {
    let matcher = patterns! {
        Add(x, Const(0)) ~> x
    };

    // Create x + 1 (should not match)
    let x = UOp::const_(DType::Int32, ConstValue::Int(42));
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let add = binary(BinaryOp::Add, x, one);

    let result = matcher.rewrite(&add);
    assert!(matches!(result, RewriteResult::NoMatch), "x + 1 should not match x + 0 pattern");
}

#[test]
fn test_pattern_matcher_composition() {
    let pm1 = patterns! {
        Add(x, Const(0)) ~> x
    };

    let pm2 = patterns! {
        Mul(x, Const(1)) ~> x
    };

    // Compose pattern matchers
    let combined = pm1 + pm2;

    let x = UOp::const_(DType::Int32, ConstValue::Int(42));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));

    // Test x + 0
    let add_zero = binary(BinaryOp::Add, x.clone(), zero);
    match combined.rewrite(&add_zero) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Combined matcher should handle Add(x, 0)"),
    }

    // Test x * 1
    let mul_one = binary(BinaryOp::Mul, x.clone(), one);
    match combined.rewrite(&mul_one) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Combined matcher should handle Mul(x, 1)"),
    }
}

#[test]
fn test_complex_guard_with_block() {
    // Test that complex guards work - guard checks if const is zero using a block expression
    let matcher = patterns! {
        Add(x, c) if {
            // Complex guard logic: check if c is a zero constant
            match c.op() {
                Op::Const(cv) => matches!(cv.0, ConstValue::Int(0)) || matches!(cv.0, ConstValue::Float(f) if f == 0.0),
                _ => false,
            }
        } ~> x
    };

    // Test x + 0 (int) - should match
    let x = UOp::const_(DType::Int32, ConstValue::Int(42));
    let zero_int = UOp::const_(DType::Int32, ConstValue::Int(0));
    let add_zero_int = binary(BinaryOp::Add, x.clone(), zero_int);

    match matcher.rewrite(&add_zero_int) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, 0) should match with complex guard"),
    }

    // Test x + 0.0 (float) - should match
    let x_f32 = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let zero_float = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let add_zero_float = binary(BinaryOp::Add, x_f32.clone(), zero_float);

    match matcher.rewrite(&add_zero_float) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x_f32)),
        _ => panic!("Add(x, 0.0) should match with complex guard"),
    }

    // Test x + 1 - should NOT match
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let add_one = binary(BinaryOp::Add, x.clone(), one);

    match matcher.rewrite(&add_one) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Add(x, 1) should NOT match zero guard"),
    }
}

#[test]
fn test_guard_with_pointer_equality() {
    // Test guard using Rc::ptr_eq for self-folding patterns like x & x => x
    let matcher = patterns! {
        And(x, y) if Rc::ptr_eq(&x, &y) ~> x
    };

    let a = UOp::const_(DType::Int32, ConstValue::Int(42));

    // Test a & a - should match (same Rc pointer due to hash-consing)
    let and_same = UOp::new(Op::Binary(BinaryOp::And, a.clone(), a.clone()), DType::Int32);

    match matcher.rewrite(&and_same) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &a)),
        _ => panic!("And(x, x) should rewrite to x"),
    }

    // Test a & b - should NOT match (different values = different pointers)
    // Note: Morok uses hash-consing, so Const(42) and Const(42) will share the same Rc
    // We need to use actually different values to test this
    let b = UOp::const_(DType::Int32, ConstValue::Int(99)); // Different value = different Rc
    let and_diff = UOp::new(Op::Binary(BinaryOp::And, a.clone(), b), DType::Int32);

    match matcher.rewrite(&and_diff) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("And(a, b) with different pointers should NOT match"),
    }
}

#[test]
fn test_special_constant_zero() {
    // Test @zero special constant - matches both Int(0) and Float(0.0)
    let matcher = patterns! {
        Add(x, @zero) ~> x
    };

    // Test x + 0 (int)
    let x_int = UOp::const_(DType::Int32, ConstValue::Int(42));
    let zero_int = UOp::const_(DType::Int32, ConstValue::Int(0));
    let add_zero_int = binary(BinaryOp::Add, x_int.clone(), zero_int);

    match matcher.rewrite(&add_zero_int) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x_int)),
        _ => panic!("Add(x, @zero) should match int 0"),
    }

    // Test x + 0.0 (float)
    let x_f32 = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let zero_float = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let add_zero_float = binary(BinaryOp::Add, x_f32.clone(), zero_float);

    match matcher.rewrite(&add_zero_float) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x_f32)),
        _ => panic!("Add(x, @zero) should match float 0.0"),
    }

    // Test x + 1 - should NOT match @zero
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let add_one = binary(BinaryOp::Add, x_int.clone(), one);

    match matcher.rewrite(&add_one) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Add(x, 1) should NOT match @zero"),
    }
}

#[test]
fn test_special_constant_one() {
    // Test @one special constant - matches both Int(1) and Float(1.0)
    let matcher = patterns! {
        Mul(x, @one) ~> x
    };

    // Test x * 1 (int)
    let x_int = UOp::const_(DType::Int32, ConstValue::Int(42));
    let one_int = UOp::const_(DType::Int32, ConstValue::Int(1));
    let mul_one_int = binary(BinaryOp::Mul, x_int.clone(), one_int);

    match matcher.rewrite(&mul_one_int) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x_int)),
        _ => panic!("Mul(x, @one) should match int 1"),
    }

    // Test x * 1.0 (float)
    let x_f32 = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let one_float = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let mul_one_float = binary(BinaryOp::Mul, x_f32.clone(), one_float);

    match matcher.rewrite(&mul_one_float) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x_f32)),
        _ => panic!("Mul(x, @one) should match float 1.0"),
    }

    // Test x * 2 - should NOT match @one
    let two = UOp::const_(DType::Int32, ConstValue::Int(2));
    let mul_two = binary(BinaryOp::Mul, x_int.clone(), two);

    match matcher.rewrite(&mul_two) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Mul(x, 2) should NOT match @one"),
    }
}

#[test]
fn test_special_constant_with_binding() {
    // Test binding with @zero: zero @ @zero
    let matcher = patterns! {
        Mul(_, zero @ @zero) ~> zero
    };

    // Test x * 0 - should return the zero constant
    let x = UOp::const_(DType::Int32, ConstValue::Int(42));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let mul_zero = binary(BinaryOp::Mul, x, zero.clone());

    match matcher.rewrite(&mul_zero) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &zero)),
        _ => panic!("Mul(_, zero @ @zero) should return zero"),
    }

    // Test with float 0.0
    let x_f32 = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let zero_f32 = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let mul_zero_f32 = binary(BinaryOp::Mul, x_f32, zero_f32.clone());

    match matcher.rewrite(&mul_zero_f32) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &zero_f32)),
        _ => panic!("Mul(_, zero @ @zero) should return float zero"),
    }
}

#[test]
fn test_identity_patterns_with_special_constants() {
    // Comprehensive identity pattern test using @zero and @one
    let matcher = patterns! {
        Add(x, @zero) ~> x,
        Add(@zero, x) ~> x,
        Mul(x, @one) ~> x,
        Mul(@one, x) ~> x,
        Mul(_, zero @ @zero) ~> zero,
        Mul(zero @ @zero, _) ~> zero,
    };

    let x = UOp::const_(DType::Int32, ConstValue::Int(42));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));

    // x + 0 => x
    let add_x_zero = binary(BinaryOp::Add, x.clone(), zero.clone());
    match matcher.rewrite(&add_x_zero) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, @zero) failed"),
    }

    // 0 + x => x
    let add_zero_x = binary(BinaryOp::Add, zero.clone(), x.clone());
    match matcher.rewrite(&add_zero_x) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Add(@zero, x) failed"),
    }

    // x * 1 => x
    let mul_x_one = binary(BinaryOp::Mul, x.clone(), one.clone());
    match matcher.rewrite(&mul_x_one) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Mul(x, @one) failed"),
    }

    // 1 * x => x
    let mul_one_x = binary(BinaryOp::Mul, one.clone(), x.clone());
    match matcher.rewrite(&mul_one_x) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Mul(@one, x) failed"),
    }

    // x * 0 => 0
    let mul_x_zero = binary(BinaryOp::Mul, x.clone(), zero.clone());
    match matcher.rewrite(&mul_x_zero) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &zero)),
        _ => panic!("Mul(_, @zero) failed"),
    }

    // 0 * x => 0
    let mul_zero_x = binary(BinaryOp::Mul, zero.clone(), x.clone());
    match matcher.rewrite(&mul_zero_x) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &zero)),
        _ => panic!("Mul(@zero, _) failed"),
    }
}

#[test]
fn test_or_casted_method() {
    // Test the .or_casted() convenience method
    // This should match both constant and CAST(constant)
    use crate::pattern::PatternMatcher;

    // Create pattern: match constant or cast(constant)
    // The "c" name binds to the inner constant in both cases
    let pattern = UPat::cvar("c").or_casted();

    // Build a matcher that returns the bound constant
    let matcher = PatternMatcher::new(vec![(
        pattern,
        Box::new(|bindings: &BindingStore, intern: &VarIntern| {
            if let Some(c) = intern.get_index("c").and_then(|i| bindings.get_by_index(i)) {
                RewriteResult::Rewritten(Rc::clone(c))
            } else {
                RewriteResult::NoMatch
            }
        }),
    )]);

    // Create a constant
    let c = UOp::const_(DType::Int32, ConstValue::Int(42));

    // Test direct constant - should match and return the constant
    match matcher.rewrite(&c) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &c)),
        _ => panic!("Direct constant should match .or_casted()"),
    }

    // Create CAST(constant)
    let cast_c = UOp::cast(c.clone(), DType::Float32);

    // Test cast(constant) - should match and return the inner constant
    match matcher.rewrite(&cast_c) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &c)),
        _ => panic!("Cast(constant) should match .or_casted()"),
    }

    // Create Add - should NOT match
    let add = binary(BinaryOp::Add, c.clone(), c.clone());
    match matcher.rewrite(&add) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Add should NOT match .or_casted()"),
    }
}

#[test]
fn test_or_detach_method() {
    // Test the .or_detach() convenience method
    use crate::pattern::PatternMatcher;

    // Create pattern: match constant or detach(constant)
    // The "c" name binds to the inner constant
    let pattern = UPat::cvar("c").or_detach();

    let matcher = PatternMatcher::new(vec![(
        pattern,
        Box::new(|bindings: &BindingStore, intern: &VarIntern| {
            if let Some(c) = intern.get_index("c").and_then(|i| bindings.get_by_index(i)) {
                RewriteResult::Rewritten(Rc::clone(c))
            } else {
                RewriteResult::NoMatch
            }
        }),
    )]);

    // Create a constant
    let x = UOp::const_(DType::Int32, ConstValue::Int(42));

    // Test direct constant - should match
    match matcher.rewrite(&x) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Direct value should match .or_detach()"),
    }

    // Create DETACH(x)
    let detach_x = UOp::detach(x.clone());

    // Test detach(x) - should also match
    match matcher.rewrite(&detach_x) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Detach(x) should match .or_detach()"),
    }
}

#[test]
fn test_binary_commutative_pattern() {
    // Test the binary_commutative method for matching both orderings
    use crate::pattern::PatternMatcher;

    // Create pattern: match x + 0 OR 0 + x (using permutation)
    let pattern = UPat::binary_commutative(vec![BinaryOp::Add], vec![UPat::var("x"), UPat::zero_const("zero")]);

    let matcher = PatternMatcher::new(vec![(
        pattern,
        Box::new(|bindings: &BindingStore, intern: &VarIntern| {
            if let Some(x) = intern.get_index("x").and_then(|i| bindings.get_by_index(i)) {
                RewriteResult::Rewritten(Rc::clone(x))
            } else {
                RewriteResult::NoMatch
            }
        }),
    )]);

    let x = UOp::const_(DType::Int32, ConstValue::Int(42));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));

    // Test x + 0 - should match
    let add_x_zero = binary(BinaryOp::Add, x.clone(), zero.clone());
    match matcher.rewrite(&add_x_zero) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, 0) should match binary_commutative"),
    }

    // Test 0 + x - should also match!
    let add_zero_x = binary(BinaryOp::Add, zero.clone(), x.clone());
    match matcher.rewrite(&add_zero_x) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Add(0, x) should match binary_commutative"),
    }
}

#[test]
fn test_permute_pattern_with_three_sources() {
    // Test permutation with 3 sources (6 orderings)
    use crate::pattern::upat::{OpFilter, SrcPattern};

    // Manually create a 3-source permutation pattern (ternary-like)
    let pattern = UPat::Match {
        op: Some(vec![OpFilter::Ternary(vec![morok_ir::TernaryOp::Where])]),
        dtype: None,
        src: Some(SrcPattern::Permute(vec![UPat::cvar("a"), UPat::cvar("b"), UPat::cvar("c")])),
        arg: None,
        name: None,
    };

    // Just test that the pattern matches - where(a, b, c) in any order
    let a = UOp::const_(DType::Int32, ConstValue::Int(1));
    let b = UOp::const_(DType::Int32, ConstValue::Int(2));
    let c = UOp::const_(DType::Int32, ConstValue::Int(3));

    // Create where(a, b, c)
    let where_abc = UOp::new(Op::Ternary(morok_ir::TernaryOp::Where, a.clone(), b.clone(), c.clone()), DType::Int32);

    // Match and verify we get bindings
    let results = pattern.match_uop(&where_abc);

    // Should have 6 results (3! permutations), but we only need to check that at least one matches
    // with the correct bindings
    assert!(!results.is_empty(), "Permute pattern should match Where(a, b, c)");

    // Verify at least one result has all three bindings
    let has_all_bindings = results
        .iter()
        .any(|bindings| bindings.contains_key("a") && bindings.contains_key("b") && bindings.contains_key("c"));
    assert!(has_all_bindings, "Should have bindings for a, b, c");
}

#[test]
fn test_struct_field_extraction() {
    // Test struct pattern with field extraction: Cast { src: x, dtype }
    // The dtype field should be extracted and available in the guard/rewrite

    // Create pattern that matches Cast where dtype matches a specific value
    let matcher = patterns! {
        // Match Cast(x) where dtype is Float32, rewrite to x
        Cast { src: x, dtype } if dtype == DType::Float32 ~> x
    };

    // Create an Int32 constant
    let x_int = UOp::const_(DType::Int32, ConstValue::Int(42));

    // Create Cast(x_int) to Float32
    let cast_to_f32 = UOp::cast(x_int.clone(), DType::Float32);

    // This should match (cast to Float32)
    match matcher.rewrite(&cast_to_f32) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x_int)),
        _ => panic!("Cast {{ src: x, dtype }} with dtype == Float32 should match"),
    }

    // Create Cast(x_int) to Int64
    let cast_to_i64 = UOp::cast(x_int.clone(), DType::Int64);

    // This should NOT match (cast to Int64, not Float32)
    match matcher.rewrite(&cast_to_i64) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Cast {{ src: x, dtype }} with dtype == Int64 should NOT match Float32 guard"),
    }
}

#[test]
fn test_struct_field_extraction_permute() {
    // Test struct pattern with field extraction for Permute { src: x, axes }

    let matcher = patterns! {
        // Match Permute where axes has length 2
        Permute { src: x, axes } if axes.len() == 2 ~> x
    };

    // Create a simple tensor
    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Create Permute with 2 axes
    let permute_2 = UOp::new(Op::Permute { src: x.clone(), axes: vec![1, 0] }, DType::Float32);

    // This should match (2 axes)
    match matcher.rewrite(&permute_2) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Permute with 2 axes should match"),
    }

    // Create Permute with 3 axes
    let permute_3 = UOp::new(Op::Permute { src: x.clone(), axes: vec![2, 0, 1] }, DType::Float32);

    // This should NOT match (3 axes, not 2)
    match matcher.rewrite(&permute_3) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Permute with 3 axes should NOT match axes.len() == 2 guard"),
    }
}

// ===== For-Loop Iteration Tests =====

#[test]
fn test_for_loop_unary_expansion() {
    use morok_ir::UnaryOp;

    // Test that for-loop syntax generates patterns for multiple unary ops
    let matcher = patterns! {
        for op in unary [Neg, Sqrt] {
            op(c) ~> {
                // Just return the operand for testing
                Rc::clone(&c)
            }
        }
    };

    // Create Neg(x)
    let x = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let neg_x = UOp::new(Op::Unary(UnaryOp::Neg, x.clone()), DType::Float32);

    match matcher.rewrite(&neg_x) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Neg pattern from for-loop should match"),
    }

    // Create Sqrt(x)
    let sqrt_x = UOp::new(Op::Unary(UnaryOp::Sqrt, x.clone()), DType::Float32);

    match matcher.rewrite(&sqrt_x) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Sqrt pattern from for-loop should match"),
    }

    // Create Exp2(x) - should NOT match (not in the loop)
    let exp2_x = UOp::new(Op::Unary(UnaryOp::Exp2, x.clone()), DType::Float32);

    match matcher.rewrite(&exp2_x) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Exp2 should NOT match (not in for-loop list)"),
    }
}

#[test]
fn test_for_loop_binary_expansion() {
    // Test that for-loop syntax generates patterns for multiple binary ops
    let matcher = patterns! {
        for op in binary [Add, Mul, Sub] {
            op(x, @zero) ~> x
        }
    };

    let x = UOp::const_(DType::Int32, ConstValue::Int(42));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));

    // Test Add(x, 0) => x
    let add_zero = binary(BinaryOp::Add, x.clone(), zero.clone());
    match matcher.rewrite(&add_zero) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, 0) from for-loop should match"),
    }

    // Test Mul(x, 0) => x
    let mul_zero = binary(BinaryOp::Mul, x.clone(), zero.clone());
    match matcher.rewrite(&mul_zero) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Mul(x, 0) from for-loop should match"),
    }

    // Test Sub(x, 0) => x
    let sub_zero = binary(BinaryOp::Sub, x.clone(), zero.clone());
    match matcher.rewrite(&sub_zero) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Sub(x, 0) from for-loop should match"),
    }

    // Test And(x, 0) - should NOT match (not in the loop)
    let and_zero = UOp::new(Op::Binary(BinaryOp::And, x.clone(), zero.clone()), DType::Int32);
    match matcher.rewrite(&and_zero) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("And should NOT match (not in for-loop list)"),
    }
}

#[test]
fn test_for_loop_ternary_expansion() {
    use morok_ir::TernaryOp;

    // Test that for-loop syntax generates patterns for ternary ops
    let matcher = patterns! {
        for op in ternary [Where, MulAcc] {
            op(a, b, c) ~> {
                // For testing, just return the first argument
                Rc::clone(&a)
            }
        }
    };

    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let c = UOp::const_(DType::Float32, ConstValue::Float(3.0));

    // Test Where(a, b, c) => a
    let where_abc = UOp::new(Op::Ternary(TernaryOp::Where, a.clone(), b.clone(), c.clone()), DType::Float32);
    match matcher.rewrite(&where_abc) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &a)),
        _ => panic!("Where pattern from for-loop should match"),
    }

    // Test MulAcc(a, b, c) => a
    let mulacc_abc = UOp::new(Op::Ternary(TernaryOp::MulAcc, a.clone(), b.clone(), c.clone()), DType::Float32);
    match matcher.rewrite(&mulacc_abc) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &a)),
        _ => panic!("MulAcc pattern from for-loop should match"),
    }
}

#[test]
fn test_for_loop_with_op_var_access() {
    use morok_ir::UnaryOp;

    // Test that the operation variable `op` is accessible in the closure
    let matcher = patterns! {
        for op in unary [Neg, Sqrt] {
            op(x) ~> {
                // Use the op variable to verify it's accessible
                // Create a different unary op with the same operand
                match op {
                    UnaryOp::Neg => UOp::new(Op::Unary(UnaryOp::Sqrt, x.clone()), x.dtype()),
                    UnaryOp::Sqrt => UOp::new(Op::Unary(UnaryOp::Neg, x.clone()), x.dtype()),
                    _ => Rc::clone(&x),
                }
            }
        }
    };

    let x = UOp::const_(DType::Float32, ConstValue::Float(42.0));

    // Neg(x) should rewrite to Sqrt(x) (swapped)
    let neg_x = UOp::new(Op::Unary(UnaryOp::Neg, x.clone()), DType::Float32);
    match matcher.rewrite(&neg_x) {
        RewriteResult::Rewritten(r) => {
            assert!(matches!(r.op(), Op::Unary(UnaryOp::Sqrt, _)), "Neg should rewrite to Sqrt");
        }
        _ => panic!("Neg pattern should match"),
    }

    // Sqrt(x) should rewrite to Neg(x) (swapped)
    let sqrt_x = UOp::new(Op::Unary(UnaryOp::Sqrt, x.clone()), DType::Float32);
    match matcher.rewrite(&sqrt_x) {
        RewriteResult::Rewritten(r) => {
            assert!(matches!(r.op(), Op::Unary(UnaryOp::Neg, _)), "Sqrt should rewrite to Neg");
        }
        _ => panic!("Sqrt pattern should match"),
    }
}

#[test]
fn test_for_loop_mixed_with_regular_patterns() {
    use morok_ir::UnaryOp;

    // Test mixing for-loops with regular patterns
    let matcher = patterns! {
        // Regular pattern first
        Add(x, @zero) ~> x,

        // For-loop in the middle
        for op in unary [Neg, Sqrt] {
            op(x) ~> { Rc::clone(&x) }
        },

        // Regular pattern after
        Mul(x, @one) ~> x,
    };

    let x = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Test Add(x, 0) => x
    let add_zero = binary(BinaryOp::Add, x.clone(), zero);
    match matcher.rewrite(&add_zero) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, 0) should match"),
    }

    // Test Neg(x) => x
    let neg_x = UOp::new(Op::Unary(UnaryOp::Neg, x.clone()), DType::Float32);
    match matcher.rewrite(&neg_x) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Neg(x) from for-loop should match"),
    }

    // Test Mul(x, 1) => x
    let mul_one = binary(BinaryOp::Mul, x.clone(), one);
    match matcher.rewrite(&mul_one) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Mul(x, 1) should match"),
    }
}

#[test]
fn test_for_loop_with_guard() {
    use morok_ir::UnaryOp;

    // Test for-loop patterns with guards
    let matcher = patterns! {
        for op in unary [Neg, Sqrt] {
            // Only match if operand is a constant
            op(c) if matches!(c.op(), Op::Const(_)) ~> { Rc::clone(&c) }
        }
    };

    let c = UOp::const_(DType::Float32, ConstValue::Float(42.0));

    // Neg(const) - should match
    let neg_c = UOp::new(Op::Unary(UnaryOp::Neg, c.clone()), DType::Float32);
    match matcher.rewrite(&neg_c) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &c)),
        _ => panic!("Neg(const) should match with guard"),
    }

    // Create a non-constant operand (binary)
    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let y = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let add_xy = binary(BinaryOp::Add, x, y);

    // Neg(add) - should NOT match (operand is not a constant)
    let neg_add = UOp::new(Op::Unary(UnaryOp::Neg, add_xy), DType::Float32);
    match matcher.rewrite(&neg_add) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Neg(non-const) should NOT match with const guard"),
    }
}

#[test]
fn test_for_loop_with_binding() {
    use morok_ir::UnaryOp;

    // Test for-loop patterns with bindings
    let matcher = patterns! {
        for op in unary [Neg, Sqrt] {
            op(inner @ @const) ~> inner
        }
    };

    let c = UOp::const_(DType::Float32, ConstValue::Float(42.0));

    // Neg(const) - should match and return the inner constant
    let neg_c = UOp::new(Op::Unary(UnaryOp::Neg, c.clone()), DType::Float32);
    match matcher.rewrite(&neg_c) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &c)),
        _ => panic!("Neg(inner @ @const) should match and return inner"),
    }

    // Sqrt(const) - should also match
    let sqrt_c = UOp::new(Op::Unary(UnaryOp::Sqrt, c.clone()), DType::Float32);
    match matcher.rewrite(&sqrt_c) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &c)),
        _ => panic!("Sqrt(inner @ @const) should match and return inner"),
    }
}

// ===== ConstWithValue Extraction Tests =====

#[test]
fn test_const_with_value_extraction() {
    // Test automatic ConstValue extraction with c@const(cv)
    let matcher = patterns! {
        // cv is ConstValue, c is &Rc<UOp>
        Add(x, _c@const(cv)) if cv == ConstValue::Int(0) ~> x
    };

    let x = UOp::const_(DType::Int32, ConstValue::Int(42));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));

    // Test x + 0 - should match (cv == 0)
    let add_zero = binary(BinaryOp::Add, x.clone(), zero);
    match matcher.rewrite(&add_zero) {
        RewriteResult::Rewritten(r) => assert!(Rc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, 0) should match with cv == 0"),
    }

    // Test x + 1 - should NOT match (cv != 0)
    let add_one = binary(BinaryOp::Add, x.clone(), one);
    match matcher.rewrite(&add_one) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Add(x, 1) should NOT match cv == 0 guard"),
    }
}

#[test]
fn test_const_with_value_extraction_fallible() {
    // Test ConstValue extraction with fallible pattern
    let matcher = patterns! {
        // Use cv in fallible expression with ?
        Neg(c@const(cv)) => cv.cast(&DType::Float32).map(|casted| UOp::const_(DType::Float32, casted))
    };

    let c = UOp::const_(DType::Int32, ConstValue::Int(42));
    let neg_c = UOp::new(Op::Unary(morok_ir::UnaryOp::Neg, c.clone()), DType::Int32);

    match matcher.rewrite(&neg_c) {
        RewriteResult::Rewritten(r) => {
            // Should create a Float32 constant with casted value
            assert_eq!(r.dtype(), DType::Float32);
        }
        _ => panic!("Neg(c@const(cv)) should match and cast the value"),
    }
}

// ===== Rest Pattern Tests =====

#[test]
fn test_rest_pattern_end() {
    use smallvec::smallvec;

    // Test End(_, ..) matching - verifies the `..` syntax works for variable-arity ops
    let matcher = patterns! {
        // Match any END op and return its computation
        end_op @ End(_, ..) ~> {
            if let Op::End { computation, .. } = end_op.op() {
                Rc::clone(computation)
            } else {
                unreachable!()
            }
        }
    };

    let computation = UOp::const_(DType::Int32, ConstValue::Int(42));
    let range1 = UOp::range(UOp::const_(DType::Index, ConstValue::Int(10)), 0);
    let range2 = UOp::range(UOp::const_(DType::Index, ConstValue::Int(20)), 1);

    // END with 1 range - should match
    let end1 = UOp::end(computation.clone(), smallvec![range1.clone()]);
    match matcher.rewrite(&end1) {
        RewriteResult::Rewritten(r) => {
            assert!(Rc::ptr_eq(&r, &computation), "Should rewrite to computation");
        }
        _ => panic!("End(_, ..) should match END with 1 range"),
    }

    // END with 2 ranges - should also match (that's the point of `..`)
    let end2 = UOp::end(computation.clone(), smallvec![range1.clone(), range2.clone()]);
    match matcher.rewrite(&end2) {
        RewriteResult::Rewritten(r) => {
            assert!(Rc::ptr_eq(&r, &computation), "Should rewrite to computation");
        }
        _ => panic!("End(_, ..) should match END with 2 ranges"),
    }
}

#[test]
fn test_rest_pattern_reduce() {
    use morok_ir::types::ReduceOp;
    use smallvec::smallvec;

    // Test Reduce(_, ..) matching - verifies the `..` syntax works for variable-arity ops
    let matcher = patterns! {
        // Match any REDUCE op and return a constant
        reduce_op @ Reduce(_, ..) ~> UOp::const_(reduce_op.dtype(), ConstValue::Int(99))
    };

    let src = UOp::const_(DType::Int32, ConstValue::Int(42));
    let range1 = UOp::range(UOp::const_(DType::Index, ConstValue::Int(10)), 0);
    let range2 = UOp::range(UOp::const_(DType::Index, ConstValue::Int(20)), 1);

    // REDUCE with 1 range - should match
    let reduce1 = UOp::reduce(src.clone(), smallvec![range1.clone()], ReduceOp::Add);
    match matcher.rewrite(&reduce1) {
        RewriteResult::Rewritten(r) => {
            assert!(matches!(r.op(), Op::Const(_)));
        }
        _ => panic!("Reduce(_, ..) should match REDUCE with 1 range"),
    }

    // REDUCE with 2 ranges - should also match (that's the point of `..`)
    let reduce2 = UOp::reduce(src.clone(), smallvec![range1.clone(), range2.clone()], ReduceOp::Add);
    match matcher.rewrite(&reduce2) {
        RewriteResult::Rewritten(r) => {
            assert!(matches!(r.op(), Op::Const(_)));
        }
        _ => panic!("Reduce(_, ..) should match REDUCE with 2 ranges"),
    }
}

#[test]
fn test_rest_pattern_with_guard() {
    use morok_ir::types::ReduceOp;
    use smallvec::smallvec;

    // Test that guards work correctly with rest patterns
    let matcher = patterns! {
        reduce_op @ Reduce(_, ..) if {
            // Only match REDUCE with Add op
            matches!(reduce_op.op(), Op::Reduce { reduce_op: ReduceOp::Add, .. })
        } ~> UOp::const_(reduce_op.dtype(), ConstValue::Int(0))
    };

    let src = UOp::const_(DType::Int32, ConstValue::Int(42));
    let range = UOp::range(UOp::const_(DType::Index, ConstValue::Int(10)), 0);

    // REDUCE with Add - should match
    let reduce_add = UOp::reduce(src.clone(), smallvec![range.clone()], ReduceOp::Add);
    match matcher.rewrite(&reduce_add) {
        RewriteResult::Rewritten(_) => {}
        _ => panic!("Should match REDUCE Add"),
    }

    // REDUCE with Mul - should NOT match (guard fails)
    let reduce_mul = UOp::reduce(src.clone(), smallvec![range.clone()], ReduceOp::Mul);
    match matcher.rewrite(&reduce_mul) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Should NOT match REDUCE Mul"),
    }
}
