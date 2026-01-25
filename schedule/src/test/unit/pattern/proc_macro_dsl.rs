//! Tests for the patterns! proc-macro DSL.
//!
//! These tests verify that the `patterns!` macro correctly generates
//! `TypedPatternMatcher` instances for pattern-based UOp rewrites.

use std::sync::Arc;

use crate::patterns;
use morok_dtype::DType;
use morok_ir::pattern::RewriteResult;
use morok_ir::{BinaryOp, ConstValue, Op, UOp};

/// Helper to create a binary operation UOp
fn binary(op: BinaryOp, lhs: Arc<UOp>, rhs: Arc<UOp>) -> Arc<UOp> {
    let dtype = lhs.dtype();
    UOp::new(Op::Binary(op, lhs, rhs), dtype)
}

#[test]
fn test_simple_add_zero_pattern() {
    let matcher = patterns! {
        Add(x, Const(0)) ~> x
    };

    // Create x + 0
    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    let add = binary(BinaryOp::Add, x.clone(), zero);

    let result = matcher.rewrite(&add, &mut ());

    match result {
        RewriteResult::Rewritten(rewritten) => {
            assert!(Arc::ptr_eq(&rewritten, &x), "Should rewrite to x");
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
    let x = UOp::native_const(42i32);
    let one = UOp::native_const(1i32);
    let mul = binary(BinaryOp::Mul, x.clone(), one);

    let result = matcher.rewrite(&mul, &mut ());

    match result {
        RewriteResult::Rewritten(rewritten) => {
            assert!(Arc::ptr_eq(&rewritten, &x), "Should rewrite to x");
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
    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    let mul = binary(BinaryOp::Mul, x, zero.clone());

    let result = matcher.rewrite(&mul, &mut ());

    match result {
        RewriteResult::Rewritten(rewritten) => {
            assert!(Arc::ptr_eq(&rewritten, &zero), "Should rewrite to zero");
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
    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    let one = UOp::native_const(1i32);

    let add_zero = binary(BinaryOp::Add, x.clone(), zero.clone());
    match matcher.rewrite(&add_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, 0) should rewrite to x"),
    }

    // Test x * 1
    let mul_one = binary(BinaryOp::Mul, x.clone(), one);
    match matcher.rewrite(&mul_one, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Mul(x, 1) should rewrite to x"),
    }

    // Test x * 0
    let mul_zero = binary(BinaryOp::Mul, x, zero.clone());
    match matcher.rewrite(&mul_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &zero)),
        _ => panic!("Mul(x, 0) should rewrite to 0"),
    }
}

#[test]
fn test_no_match() {
    let matcher = patterns! {
        Add(x, Const(0)) ~> x
    };

    // Create x + 1 (should not match)
    let x = UOp::native_const(42i32);
    let one = UOp::native_const(1i32);
    let add = binary(BinaryOp::Add, x, one);

    let result = matcher.rewrite(&add, &mut ());
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

    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    let one = UOp::native_const(1i32);

    // Test x + 0
    let add_zero = binary(BinaryOp::Add, x.clone(), zero);
    match combined.rewrite(&add_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Combined matcher should handle Add(x, 0)"),
    }

    // Test x * 1
    let mul_one = binary(BinaryOp::Mul, x.clone(), one);
    match combined.rewrite(&mul_one, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
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
    let x = UOp::native_const(42i32);
    let zero_int = UOp::native_const(0i32);
    let add_zero_int = binary(BinaryOp::Add, x.clone(), zero_int);

    match matcher.rewrite(&add_zero_int, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, 0) should match with complex guard"),
    }

    // Test x + 0.0 (float) - should match
    let x_f32 = UOp::native_const(42.0f32);
    let zero_float = UOp::native_const(0.0f32);
    let add_zero_float = binary(BinaryOp::Add, x_f32.clone(), zero_float);

    match matcher.rewrite(&add_zero_float, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x_f32)),
        _ => panic!("Add(x, 0.0) should match with complex guard"),
    }

    // Test x + 1 - should NOT match
    let one = UOp::native_const(1i32);
    let add_one = binary(BinaryOp::Add, x.clone(), one);

    match matcher.rewrite(&add_one, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Add(x, 1) should NOT match zero guard"),
    }
}

#[test]
fn test_guard_with_pointer_equality() {
    // Test guard using Arc::ptr_eq for self-folding patterns like x & x => x
    let matcher = patterns! {
        And(x, y) if Arc::ptr_eq(x, y) ~> x
    };

    let a = UOp::native_const(42i32);

    // Test a & a - should match (same Rc pointer due to hash-consing)
    let and_same = a.try_and_op(&a).unwrap();

    match matcher.rewrite(&and_same, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &a)),
        _ => panic!("And(x, x) should rewrite to x"),
    }

    // Test a & b - should NOT match (different values = different pointers)
    // Note: Morok uses hash-consing, so Const(42) and Const(42) will share the same Rc
    // We need to use actually different values to test this
    let b = UOp::native_const(99i32); // Different value = different Rc
    let and_diff = a.try_and_op(&b).unwrap();

    match matcher.rewrite(&and_diff, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("And(a, b) with different pointers should NOT match"),
    }
}

#[test]
fn test_auto_ptr_eq_duplicate_variable() {
    // Test auto ptr_eq with duplicate variable names: And(x, x) ~> x
    // This should automatically generate Arc::ptr_eq check without explicit guard
    let matcher = patterns! {
        And(x, x) ~> x
    };

    let a = UOp::native_const(42i32);

    // Test a & a - should match (same Rc pointer)
    let and_same = a.try_and_op(&a).unwrap();

    match matcher.rewrite(&and_same, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &a), "And(x, x) should rewrite to x"),
        _ => panic!("And(x, x) should match"),
    }

    // Test a & b - should NOT match (different values = different pointers)
    let b = UOp::native_const(99i32);
    let and_diff = a.try_and_op(&b).unwrap();

    match matcher.rewrite(&and_diff, &mut ()) {
        RewriteResult::NoMatch => {} // Expected - auto ptr_eq check fails
        _ => panic!("And(a, b) with different pointers should NOT match"),
    }
}

#[test]
fn test_auto_ptr_eq_three_args() {
    // Test auto ptr_eq with three duplicate variables: Where(x, x, x) ~> x
    let matcher = patterns! {
        Where(x, x, x) ~> x
    };

    let a = UOp::native_const(42i32);
    let b = UOp::native_const(99i32);

    // Test Where(a, a, a) - should match
    let where_same = UOp::try_where(a.clone(), a.clone(), a.clone()).unwrap();

    match matcher.rewrite(&where_same, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &a), "Where(x, x, x) should rewrite to x"),
        _ => panic!("Where(x, x, x) should match"),
    }

    // Test Where(a, a, b) - should NOT match
    let where_diff = UOp::try_where(a.clone(), a.clone(), b.clone()).unwrap();

    match matcher.rewrite(&where_diff, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Where(a, a, b) should NOT match"),
    }

    // Test Where(a, b, a) - should NOT match (middle differs)
    // This case catches the DuplicateTracker shadowing bug where only first==third is checked
    let where_middle_diff = UOp::try_where(a.clone(), b.clone(), a.clone()).unwrap();

    match matcher.rewrite(&where_middle_diff, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Where(a, b, a) should NOT match Where(x, x, x)"),
    }

    // Test Where(b, a, a) - should NOT match (first differs)
    let where_first_diff = UOp::try_where(b.clone(), a.clone(), a.clone()).unwrap();

    match matcher.rewrite(&where_first_diff, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Where(b, a, a) should NOT match Where(x, x, x)"),
    }
}

#[test]
fn test_special_constant_zero() {
    // Test @zero special constant - matches both Int(0) and Float(0.0)
    let matcher = patterns! {
        Add(x, @zero) ~> x
    };

    // Test x + 0 (int)
    let x_int = UOp::native_const(42i32);
    let zero_int = UOp::native_const(0i32);
    let add_zero_int = binary(BinaryOp::Add, x_int.clone(), zero_int);

    match matcher.rewrite(&add_zero_int, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x_int)),
        _ => panic!("Add(x, @zero) should match int 0"),
    }

    // Test x + 0.0 (float)
    let x_f32 = UOp::native_const(42.0f32);
    let zero_float = UOp::native_const(0.0f32);
    let add_zero_float = binary(BinaryOp::Add, x_f32.clone(), zero_float);

    match matcher.rewrite(&add_zero_float, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x_f32)),
        _ => panic!("Add(x, @zero) should match float 0.0"),
    }

    // Test x + 1 - should NOT match @zero
    let one = UOp::native_const(1i32);
    let add_one = binary(BinaryOp::Add, x_int.clone(), one);

    match matcher.rewrite(&add_one, &mut ()) {
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
    let x_int = UOp::native_const(42i32);
    let one_int = UOp::native_const(1i32);
    let mul_one_int = binary(BinaryOp::Mul, x_int.clone(), one_int);

    match matcher.rewrite(&mul_one_int, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x_int)),
        _ => panic!("Mul(x, @one) should match int 1"),
    }

    // Test x * 1.0 (float)
    let x_f32 = UOp::native_const(42.0f32);
    let one_float = UOp::native_const(1.0f32);
    let mul_one_float = binary(BinaryOp::Mul, x_f32.clone(), one_float);

    match matcher.rewrite(&mul_one_float, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x_f32)),
        _ => panic!("Mul(x, @one) should match float 1.0"),
    }

    // Test x * 2 - should NOT match @one
    let two = UOp::native_const(2i32);
    let mul_two = binary(BinaryOp::Mul, x_int.clone(), two);

    match matcher.rewrite(&mul_two, &mut ()) {
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
    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    let mul_zero = binary(BinaryOp::Mul, x, zero.clone());

    match matcher.rewrite(&mul_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &zero)),
        _ => panic!("Mul(_, zero @ @zero) should return zero"),
    }

    // Test with float 0.0
    let x_f32 = UOp::native_const(42.0f32);
    let zero_f32 = UOp::native_const(0.0f32);
    let mul_zero_f32 = binary(BinaryOp::Mul, x_f32, zero_f32.clone());

    match matcher.rewrite(&mul_zero_f32, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &zero_f32)),
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

    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    let one = UOp::native_const(1i32);

    // x + 0 => x
    let add_x_zero = binary(BinaryOp::Add, x.clone(), zero.clone());
    match matcher.rewrite(&add_x_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, @zero) failed"),
    }

    // 0 + x => x
    let add_zero_x = binary(BinaryOp::Add, zero.clone(), x.clone());
    match matcher.rewrite(&add_zero_x, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Add(@zero, x) failed"),
    }

    // x * 1 => x
    let mul_x_one = binary(BinaryOp::Mul, x.clone(), one.clone());
    match matcher.rewrite(&mul_x_one, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Mul(x, @one) failed"),
    }

    // 1 * x => x
    let mul_one_x = binary(BinaryOp::Mul, one.clone(), x.clone());
    match matcher.rewrite(&mul_one_x, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Mul(@one, x) failed"),
    }

    // x * 0 => 0
    let mul_x_zero = binary(BinaryOp::Mul, x.clone(), zero.clone());
    match matcher.rewrite(&mul_x_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &zero)),
        _ => panic!("Mul(_, @zero) failed"),
    }

    // 0 * x => 0
    let mul_zero_x = binary(BinaryOp::Mul, zero.clone(), x.clone());
    match matcher.rewrite(&mul_zero_x, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &zero)),
        _ => panic!("Mul(@zero, _) failed"),
    }
}

// NOTE: Tests for deprecated UPat API (test_or_casted_method, test_or_detach_method,
// test_binary_commutative_pattern, test_permute_pattern_with_three_sources) were removed
// during migration to TypedPatternMatcher infrastructure.

#[test]
fn test_struct_field_extraction() {
    // Test struct pattern with field extraction: Cast { src: x, dtype }
    // The dtype field should be extracted and available in the guard/rewrite

    // Create pattern that matches Cast where dtype matches a specific value
    let matcher = patterns! {
        // Match Cast(x) where dtype is Float32, rewrite to x
        Cast { src: x, dtype } if *dtype == DType::Float32 ~> x
    };

    // Create an Int32 constant
    let x_int = UOp::native_const(42i32);

    // Create Cast(x_int) to Float32
    let cast_to_f32 = UOp::cast(x_int.clone(), DType::Float32);

    // This should match (cast to Float32)
    match matcher.rewrite(&cast_to_f32, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x_int)),
        _ => panic!("Cast {{ src: x, dtype }} with dtype == Float32 should match"),
    }

    // Create Cast(x_int) to Int64
    let cast_to_i64 = UOp::cast(x_int.clone(), DType::Int64);

    // This should NOT match (cast to Int64, not Float32)
    match matcher.rewrite(&cast_to_i64, &mut ()) {
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
    let x = UOp::native_const(1.0f32);

    // Create Permute with 2 axes
    let permute_2 = UOp::new(Op::Permute { src: x.clone(), axes: vec![1, 0] }, DType::Float32);

    // This should match (2 axes)
    match matcher.rewrite(&permute_2, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Permute with 2 axes should match"),
    }

    // Create Permute with 3 axes
    let permute_3 = UOp::new(Op::Permute { src: x.clone(), axes: vec![2, 0, 1] }, DType::Float32);

    // This should NOT match (3 axes, not 2)
    match matcher.rewrite(&permute_3, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Permute with 3 axes should NOT match axes.len() == 2 guard"),
    }
}

// ===== Nested Struct Pattern Tests =====

#[test]
fn test_nested_struct_pattern() {
    // Test nested struct patterns: Cast { src: Cast { src: x, .. }, dtype }
    // This matches a cast of a cast and extracts the innermost source
    let matcher = patterns! {
        Cast { src: Cast { src: x, .. }, dtype } if *dtype == DType::Float32 ~> x
    };

    // Create an Int32 constant
    let x_int = UOp::native_const(42i32);

    // Create inner cast: Cast(x_int) to Int64
    let inner_cast = UOp::cast(x_int.clone(), DType::Int64);

    // Create outer cast: Cast(inner_cast) to Float32
    let outer_cast = UOp::cast(inner_cast, DType::Float32);

    // This should match (outer cast to Float32)
    match matcher.rewrite(&outer_cast, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &x_int), "Should extract innermost source");
        }
        _ => panic!("Nested Cast pattern should match"),
    }

    // Single cast should NOT match
    let single_cast = UOp::cast(x_int.clone(), DType::Float32);
    match matcher.rewrite(&single_cast, &mut ()) {
        RewriteResult::NoMatch => {} // Expected - not nested
        _ => panic!("Single Cast should NOT match nested pattern"),
    }
}

#[test]
fn test_nested_struct_field_extraction() {
    use morok_ir::types::{AddrSpace, BufferizeOpts};

    // Test nested struct field extraction:
    // Index { buffer: Bufferize { compute, ranges, .. }, indices, gate: None }
    // This should extract `ranges` from the inner Bufferize AND `indices` from the outer Index.
    //
    // Note: We use a simple comparison function for testing since ranges_equal is not available here
    let matcher = patterns! {
        Index { buffer: Bufferize { compute, ranges, .. }, indices, gate: None }
            if ranges.len() == indices.len() ~> compute
    };

    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Global };
    let compute = UOp::native_const(42.0f32);
    let range1 = UOp::range(UOp::index_const(10), 0);
    let range2 = UOp::range(UOp::index_const(20), 1);

    // Create: INDEX(BUFFERIZE(compute, [r1, r2]), [r1, r2])
    let buf = UOp::bufferize(compute.clone(), vec![range1.clone(), range2.clone()], opts);
    let idx = UOp::index().buffer(buf).indices(vec![range1.clone(), range2.clone()]).call().unwrap();

    // Should match and return compute (ranges.len() == indices.len())
    match matcher.rewrite(&idx, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &compute), "Should extract compute from nested pattern");
        }
        _ => panic!("Nested Index(Bufferize) pattern should match with extracted ranges"),
    }
}

#[test]
fn test_nested_struct_field_extraction_mismatch() {
    use morok_ir::types::{AddrSpace, BufferizeOpts};

    // Test that guard fails when ranges.len() != indices.len()
    let matcher = patterns! {
        Index { buffer: Bufferize { compute, ranges, .. }, indices, gate: None }
            if ranges.len() == indices.len() ~> compute
    };

    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Global };
    let compute = UOp::native_const(42.0f32);
    let range1 = UOp::range(UOp::index_const(10), 0);
    let range2 = UOp::range(UOp::index_const(20), 1);

    // Create: INDEX(BUFFERIZE(compute, [r1, r2]), [r1]) - different lengths
    let buf = UOp::bufferize(compute.clone(), vec![range1.clone(), range2], opts);
    let idx = UOp::index().buffer(buf).indices(vec![range1]).call().unwrap();

    // Should NOT match because ranges.len() (2) != indices.len() (1)
    match matcher.rewrite(&idx, &mut ()) {
        RewriteResult::NoMatch => {} // Expected - guard fails
        _ => panic!("Should NOT match when ranges.len() != indices.len()"),
    }
}

// ===== For-Loop Iteration Tests =====

#[test]
fn test_for_loop_unary_expansion() {
    // Test that for-loop syntax generates patterns for multiple unary ops
    #[allow(unused_variables)]
    let matcher = patterns! {
        for op in unary [Neg, Sqrt] {
            op(c) ~> {
                // Just return the operand for testing
                Arc::clone(c)
            }
        }
    };

    // Create Neg(x)
    let x = UOp::native_const(42.0f32);
    let neg_x = x.neg();

    match matcher.rewrite(&neg_x, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Neg pattern from for-loop should match"),
    }

    // Create Sqrt(x)
    let sqrt_x = x.try_sqrt().unwrap();

    match matcher.rewrite(&sqrt_x, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Sqrt pattern from for-loop should match"),
    }

    // Create Exp2(x) - should NOT match (not in the loop)
    let exp2_x = x.try_exp2().unwrap();

    match matcher.rewrite(&exp2_x, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Exp2 should NOT match (not in for-loop list)"),
    }
}

#[test]
fn test_for_loop_binary_expansion() {
    // Test that for-loop syntax generates patterns for multiple binary ops
    #[allow(unused_variables)]
    let matcher = patterns! {
        for op in binary [Add, Mul, Sub] {
            op(x, @zero) ~> x
        }
    };

    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);

    // Test Add(x, 0) => x
    let add_zero = binary(BinaryOp::Add, x.clone(), zero.clone());
    match matcher.rewrite(&add_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, 0) from for-loop should match"),
    }

    // Test Mul(x, 0) => x
    let mul_zero = binary(BinaryOp::Mul, x.clone(), zero.clone());
    match matcher.rewrite(&mul_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Mul(x, 0) from for-loop should match"),
    }

    // Test Sub(x, 0) => x
    let sub_zero = binary(BinaryOp::Sub, x.clone(), zero.clone());
    match matcher.rewrite(&sub_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Sub(x, 0) from for-loop should match"),
    }

    // Test And(x, 0) - should NOT match (not in the loop)
    let and_zero = x.try_and_op(&zero).unwrap();
    match matcher.rewrite(&and_zero, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("And should NOT match (not in for-loop list)"),
    }
}

#[test]
fn test_for_loop_ternary_expansion() {
    // Test that for-loop syntax generates patterns for ternary ops
    #[allow(unused_variables)]
    let matcher = patterns! {
        for op in ternary [Where, MulAcc] {
            op(a, b, c) ~> {
                // For testing, just return the first argument
                Arc::clone(a)
            }
        }
    };

    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let c = UOp::native_const(3.0f32);

    // Test Where(a, b, c) => a
    let where_abc = UOp::try_where(a.clone(), b.clone(), c.clone()).unwrap();
    match matcher.rewrite(&where_abc, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &a)),
        _ => panic!("Where pattern from for-loop should match"),
    }

    // Test MulAcc(a, b, c) => a
    let mulacc_abc = UOp::try_mulacc(a.clone(), b.clone(), c.clone()).unwrap();
    match matcher.rewrite(&mulacc_abc, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &a)),
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
                    UnaryOp::Neg => x.try_sqrt().unwrap(),
                    UnaryOp::Sqrt => x.neg(),
                    _ => Arc::clone(x),
                }
            }
        }
    };

    let x = UOp::native_const(42.0f32);

    // Neg(x) should rewrite to Sqrt(x) (swapped)
    let neg_x = x.neg();
    match matcher.rewrite(&neg_x, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(matches!(r.op(), Op::Unary(UnaryOp::Sqrt, _)), "Neg should rewrite to Sqrt");
        }
        _ => panic!("Neg pattern should match"),
    }

    // Sqrt(x) should rewrite to Neg(x) (swapped)
    let sqrt_x = x.try_sqrt().unwrap();
    match matcher.rewrite(&sqrt_x, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(matches!(r.op(), Op::Unary(UnaryOp::Neg, _)), "Sqrt should rewrite to Neg");
        }
        _ => panic!("Sqrt pattern should match"),
    }
}

#[test]
fn test_for_loop_mixed_with_regular_patterns() {
    // Test mixing for-loops with regular patterns
    #[allow(unused_variables)]
    let matcher = patterns! {
        // Regular pattern first
        Add(x, @zero) ~> x,

        // For-loop in the middle
        for op in unary [Neg, Sqrt] {
            op(x) ~> Arc::clone(x)
        },

        // Regular pattern after
        Mul(x, @one) ~> x,
    };

    let x = UOp::native_const(42.0f32);
    let zero = UOp::native_const(0.0f32);
    let one = UOp::native_const(1.0f32);

    // Test Add(x, 0) => x
    let add_zero = binary(BinaryOp::Add, x.clone(), zero);
    match matcher.rewrite(&add_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, 0) should match"),
    }

    // Test Neg(x) => x
    let neg_x = x.neg();
    match matcher.rewrite(&neg_x, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Neg(x) from for-loop should match"),
    }

    // Test Mul(x, 1) => x
    let mul_one = binary(BinaryOp::Mul, x.clone(), one);
    match matcher.rewrite(&mul_one, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Mul(x, 1) should match"),
    }
}

#[test]
fn test_for_loop_with_guard() {
    // Test for-loop patterns with guards
    #[allow(unused_variables)]
    let matcher = patterns! {
        for op in unary [Neg, Sqrt] {
            // Only match if operand is a constant
            op(c) if matches!(c.op(), Op::Const(_)) ~> Arc::clone(c)
        }
    };

    let c = UOp::native_const(42.0f32);

    // Neg(const) - should match
    let neg_c = c.neg();
    match matcher.rewrite(&neg_c, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &c)),
        _ => panic!("Neg(const) should match with guard"),
    }

    // Create a non-constant operand (binary)
    let x = UOp::native_const(1.0f32);
    let y = UOp::native_const(2.0f32);
    let add_xy = binary(BinaryOp::Add, x, y);

    // Neg(add) - should NOT match (operand is not a constant)
    let neg_add = add_xy.neg();
    match matcher.rewrite(&neg_add, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Neg(non-const) should NOT match with const guard"),
    }
}

#[test]
fn test_for_loop_with_binding() {
    // Test for-loop patterns with bindings
    #[allow(unused_variables)]
    let matcher = patterns! {
        for op in unary [Neg, Sqrt] {
            op(inner @ @const) ~> inner
        }
    };

    let c = UOp::native_const(42.0f32);

    // Neg(const) - should match and return the inner constant
    let neg_c = c.neg();
    match matcher.rewrite(&neg_c, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &c)),
        _ => panic!("Neg(inner @ @const) should match and return inner"),
    }

    // Sqrt(const) - should also match
    let sqrt_c = c.try_sqrt().unwrap();
    match matcher.rewrite(&sqrt_c, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &c)),
        _ => panic!("Sqrt(inner @ @const) should match and return inner"),
    }
}

// ===== ConstWithValue Extraction Tests =====

#[test]
fn test_const_with_value_extraction() {
    // Test automatic ConstValue extraction with c@const(cv)
    let matcher = patterns! {
        // cv is ConstValue, c is &Arc<UOp>
        Add(x, _c@const(cv)) if cv == ConstValue::Int(0) ~> x
    };

    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    let one = UOp::native_const(1i32);

    // Test x + 0 - should match (cv == 0)
    let add_zero = binary(BinaryOp::Add, x.clone(), zero);
    match matcher.rewrite(&add_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, 0) should match with cv == 0"),
    }

    // Test x + 1 - should NOT match (cv != 0)
    let add_one = binary(BinaryOp::Add, x.clone(), one);
    match matcher.rewrite(&add_one, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Add(x, 1) should NOT match cv == 0 guard"),
    }
}

#[test]
fn test_const_with_value_extraction_fallible() {
    // Test ConstValue extraction with fallible pattern
    let matcher = patterns! {
        // Use cv in fallible expression with ?
        Neg(_c@const(cv)) => cv.cast(&DType::Float32).map(|casted| UOp::const_(DType::Float32, casted))
    };

    let c = UOp::native_const(42i32);
    let neg_c = c.neg();

    match matcher.rewrite(&neg_c, &mut ()) {
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
                Arc::clone(computation)
            } else {
                unreachable!()
            }
        }
    };

    let computation = UOp::native_const(42i32);
    let range1 = UOp::range(UOp::index_const(10), 0);
    let range2 = UOp::range(UOp::index_const(20), 1);

    // END with 1 range - should match
    let end1 = UOp::end(computation.clone(), smallvec![range1.clone()]);
    match matcher.rewrite(&end1, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &computation), "Should rewrite to computation");
        }
        _ => panic!("End(_, ..) should match END with 1 range"),
    }

    // END with 2 ranges - should also match (that's the point of `..`)
    let end2 = UOp::end(computation.clone(), smallvec![range1.clone(), range2.clone()]);
    match matcher.rewrite(&end2, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &computation), "Should rewrite to computation");
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

    let src = UOp::native_const(42i32);
    let range1 = UOp::range(UOp::index_const(10), 0);
    let range2 = UOp::range(UOp::index_const(20), 1);

    // REDUCE with 1 range - should match
    let reduce1 = UOp::reduce(src.clone(), smallvec![range1.clone()], ReduceOp::Add);
    match matcher.rewrite(&reduce1, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(matches!(r.op(), Op::Const(_)));
        }
        _ => panic!("Reduce(_, ..) should match REDUCE with 1 range"),
    }

    // REDUCE with 2 ranges - should also match (that's the point of `..`)
    let reduce2 = UOp::reduce(src.clone(), smallvec![range1.clone(), range2.clone()], ReduceOp::Add);
    match matcher.rewrite(&reduce2, &mut ()) {
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

    let src = UOp::native_const(42i32);
    let range = UOp::range(UOp::index_const(10), 0);

    // REDUCE with Add - should match
    let reduce_add = UOp::reduce(src.clone(), smallvec![range.clone()], ReduceOp::Add);
    match matcher.rewrite(&reduce_add, &mut ()) {
        RewriteResult::Rewritten(_) => {}
        _ => panic!("Should match REDUCE Add"),
    }

    // REDUCE with Mul - should NOT match (guard fails)
    let reduce_mul = UOp::reduce(src.clone(), smallvec![range.clone()], ReduceOp::Mul);
    match matcher.rewrite(&reduce_mul, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Should NOT match REDUCE Mul"),
    }
}

// ===== Variable-Arity Prefix Matching Tests =====
// These tests verify that SrcPattern::Tuple uses prefix matching (like Tinygrad's zip() semantics)
// instead of exact-length matching. This allows patterns to match variable-arity ops.

#[test]
fn test_bufferize_variable_ranges() {
    use morok_ir::types::{AddrSpace, BufferizeOpts};

    // Test Bufferize { compute: c, .. } with varying number of ranges
    // This pattern should match Bufferize with 0, 1, 2, or more ranges
    let matcher = patterns! {
        Bufferize { compute: c, .. } if matches!(c.op(), Op::Const(_)) ~> c
    };

    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Global };
    let const_val = UOp::native_const(42.0f32);
    let range1 = UOp::range(UOp::index_const(10), 0);
    let range2 = UOp::range(UOp::index_const(20), 1);

    // Test with 0 ranges
    let buf0 = UOp::bufferize(const_val.clone(), vec![], opts.clone());
    match matcher.rewrite(&buf0, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &const_val), "Should rewrite to const_val with 0 ranges");
        }
        _ => panic!("Bufferize {{ compute: c, .. }} should match with 0 ranges"),
    }

    // Test with 1 range
    let buf1 = UOp::bufferize(const_val.clone(), vec![range1.clone()], opts.clone());
    match matcher.rewrite(&buf1, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &const_val), "Should rewrite to const_val with 1 range");
        }
        _ => panic!("Bufferize {{ compute: c, .. }} should match with 1 range"),
    }

    // Test with 2 ranges
    let buf2 = UOp::bufferize(const_val.clone(), vec![range1, range2], opts);
    match matcher.rewrite(&buf2, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &const_val), "Should rewrite to const_val with 2 ranges");
        }
        _ => panic!("Bufferize {{ compute: c, .. }} should match with 2 ranges"),
    }
}

#[test]
fn test_index_variable_indices() {
    // Test Index { buffer: c, .. } with varying number of indices
    let matcher = patterns! {
        Index { buffer: c, .. } if matches!(c.op(), Op::Const(_)) ~> c
    };

    let const_val = UOp::native_const(42.0f32);
    let idx1 = UOp::index_const(0);
    let idx2 = UOp::index_const(1);

    // Test with 1 index (minimum for Index)
    let index1 = UOp::index().buffer(const_val.clone()).indices(vec![idx1.clone()]).call().unwrap();
    match matcher.rewrite(&index1, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &const_val), "Should rewrite to const_val with 1 index");
        }
        _ => panic!("Index {{ buffer: c, .. }} should match with 1 index"),
    }

    // Test with 2 indices
    let index2 = UOp::index().buffer(const_val.clone()).indices(vec![idx1.clone(), idx2.clone()]).call().unwrap();
    match matcher.rewrite(&index2, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &const_val), "Should rewrite to const_val with 2 indices");
        }
        _ => panic!("Index {{ buffer: c, .. }} should match with 2 indices"),
    }

    // Test with gate (optional field) - use index_gated constructor
    let gate = UOp::const_(DType::Bool, ConstValue::Int(1));
    let index_gated = UOp::index().buffer(const_val.clone()).indices(vec![idx1]).gate(gate).call().unwrap();
    match matcher.rewrite(&index_gated, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &const_val), "Should rewrite to const_val with gate");
        }
        _ => panic!("Index {{ buffer: c, .. }} should match with gate"),
    }
}

#[test]
fn test_index_gate_bare_binding() {
    // Test that bare `gate` field binding extracts Option<Arc<UOp>>
    // This tests if the DSL already supports optional field binding
    let matcher = patterns! {
        Index { buffer: b, indices: _, gate } => {
            // gate should be Option<Arc<UOp>>
            // indices should be SmallVec<[Arc<UOp>; 4]>
            match gate {
                Some(g) => Some(g.clone()),  // Return the gate if present
                None => Some(b.clone()),      // Return the buffer if no gate
            }
        }
    };

    let buffer = UOp::native_const(42.0f32);
    let idx = UOp::index_const(0);
    let gate_val = UOp::const_(DType::Bool, ConstValue::Int(1));

    // Test ungated index
    let ungated = UOp::index().buffer(buffer.clone()).indices(vec![idx.clone()]).call().unwrap();
    match matcher.rewrite(&ungated, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &buffer), "Should return buffer when no gate");
        }
        _ => panic!("Pattern should match ungated Index"),
    }

    // Test gated index
    let gated = UOp::index().buffer(buffer.clone()).indices(vec![idx]).gate(gate_val.clone()).call().unwrap();
    match matcher.rewrite(&gated, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &gate_val), "Should return gate when present");
        }
        _ => panic!("Pattern should match gated Index"),
    }
}

// NOTE: test_prefix_matching_minimum_children was removed - it tested deprecated UPat API

#[test]
fn test_tuple_prefix_semantics_vs_exact() {
    use morok_ir::types::{AddrSpace, BufferizeOpts};

    // Verify that Tuple now uses prefix semantics (matches first N, ignores rest)
    // rather than exact semantics (requires exactly N children)

    // Pattern: Bufferize with compute only (via struct syntax)
    let matcher = patterns! {
        Bufferize { compute: c, .. } ~> c
    };

    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Global };
    let const_val = UOp::native_const(42.0f32);
    let range1 = UOp::range(UOp::index_const(10), 0);
    let range2 = UOp::range(UOp::index_const(20), 1);
    let range3 = UOp::range(UOp::index_const(30), 2);

    // All of these should match because prefix matching ignores extra ranges
    for (n, ranges) in [
        (0, vec![]),
        (1, vec![range1.clone()]),
        (2, vec![range1.clone(), range2.clone()]),
        (3, vec![range1, range2, range3]),
    ] {
        let buf = UOp::bufferize(const_val.clone(), ranges, opts.clone());
        match matcher.rewrite(&buf, &mut ()) {
            RewriteResult::Rewritten(r) => {
                assert!(Arc::ptr_eq(&r, &const_val), "Should rewrite with {} ranges", n);
            }
            _ => panic!("Bufferize {{ compute: c, .. }} should match with {} ranges (prefix semantics)", n),
        }
    }
}

// ===== Alternative Patterns (pipe operator |) Tests =====

#[test]
fn test_alternative_patterns_basic() {
    // Test (Add | Mul) alternative matching
    let matcher = patterns! {
        (Add(x, _y) | Mul(x, _y)) ~> x
    };

    let a = UOp::native_const(5i32);
    let b = UOp::native_const(3i32);

    // Add should match
    let add = binary(BinaryOp::Add, a.clone(), b.clone());
    match matcher.rewrite(&add, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &a), "Add should rewrite to x");
        }
        _ => panic!("Add should match alternative pattern"),
    }

    // Mul should also match
    let mul = binary(BinaryOp::Mul, a.clone(), b.clone());
    match matcher.rewrite(&mul, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &a), "Mul should rewrite to x");
        }
        _ => panic!("Mul should match alternative pattern"),
    }

    // Sub should NOT match
    let sub = binary(BinaryOp::Sub, a.clone(), b.clone());
    match matcher.rewrite(&sub, &mut ()) {
        RewriteResult::NoMatch => {}
        _ => panic!("Sub should NOT match (Add | Mul) pattern"),
    }
}

#[test]
fn test_alternative_patterns_op_shorthand() {
    // Test (Add | Mul)(args) shorthand syntax
    let matcher = patterns! {
        (Add | Mul)(x, @zero) ~> x
    };

    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);

    // Add(x, 0) should match
    let add = binary(BinaryOp::Add, x.clone(), zero.clone());
    match matcher.rewrite(&add, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &x), "Add(x, 0) should rewrite to x");
        }
        _ => panic!("Add(x, 0) should match"),
    }

    // Mul(x, 0) should also match
    let mul = binary(BinaryOp::Mul, x.clone(), zero.clone());
    match matcher.rewrite(&mul, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &x), "Mul(x, 0) should rewrite to x");
        }
        _ => panic!("Mul(x, 0) should match"),
    }
}

#[test]
fn test_alternative_patterns_grouped() {
    // Test simpler alternative: both branches have same structure
    let matcher = patterns! {
        (Add(x, _y) | Mul(x, _y)) ~> x
    };

    let a = UOp::native_const(5i32);
    let b = UOp::native_const(3i32);

    // Add should match
    let add = binary(BinaryOp::Add, a.clone(), b.clone());
    match matcher.rewrite(&add, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &a), "Add(x, y) should rewrite to x");
        }
        _ => panic!("Add should match"),
    }

    // Mul should also match
    let mul = binary(BinaryOp::Mul, a.clone(), b.clone());
    match matcher.rewrite(&mul, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &a), "Mul(x, y) should rewrite to x");
        }
        _ => panic!("Mul should match"),
    }
}

#[test]
fn test_alternative_patterns_with_special_const() {
    // Test alternative with special constants @zero and @one
    let matcher = patterns! {
        (Add(x, @zero) | Add(x, @one)) ~> x
    };

    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    let one = UOp::native_const(1i32);
    let two = UOp::native_const(2i32);

    // Add(x, 0) should match
    let add0 = binary(BinaryOp::Add, x.clone(), zero);
    match matcher.rewrite(&add0, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &x), "Add(x, @zero) should rewrite to x");
        }
        _ => panic!("Add(x, 0) should match @zero"),
    }

    // Add(x, 1) should also match
    let add1 = binary(BinaryOp::Add, x.clone(), one);
    match matcher.rewrite(&add1, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &x), "Add(x, @one) should rewrite to x");
        }
        _ => panic!("Add(x, 1) should match @one"),
    }

    // Add(x, 2) should NOT match
    let add2 = binary(BinaryOp::Add, x.clone(), two);
    match matcher.rewrite(&add2, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Add(x, 2) should NOT match (neither @zero nor @one)"),
    }
}

// NOTE: test_upat_any_direct_api was removed - it tested deprecated UPat API

// ===== Permutation Patterns (bracket syntax) Tests =====

#[test]
fn test_permutation_pattern_basic() {
    // Test Add[x, c] permutation matching (matches both Add(x, c) and Add(c, x))
    let matcher = patterns! {
        Add[x, @const] ~> x
    };

    let x = UOp::native_const(42i32);
    let c = UOp::native_const(5i32);

    // Add(x, c) should match with x bound to first arg
    let add1 = binary(BinaryOp::Add, x.clone(), c.clone());
    match matcher.rewrite(&add1, &mut ()) {
        RewriteResult::Rewritten(_) => {}
        _ => panic!("Add(x, c) should match permutation pattern"),
    }

    // Add(c, x) should also match (permutation tries both orderings)
    let add2 = binary(BinaryOp::Add, c.clone(), x.clone());
    match matcher.rewrite(&add2, &mut ()) {
        RewriteResult::Rewritten(_) => {}
        _ => panic!("Add(c, x) should match permutation pattern"),
    }
}

#[test]
fn test_permutation_pattern_commutative_const_folding() {
    // Simulate commutative constant folding: Add[x, 0] ~> x
    // This should match both Add(x, 0) and Add(0, x)
    let matcher = patterns! {
        Add[x, Const(0)] ~> x
    };

    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);

    // Add(x, 0) should match
    let add1 = binary(BinaryOp::Add, x.clone(), zero.clone());
    match matcher.rewrite(&add1, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &x), "Add(x, 0) should rewrite to x");
        }
        _ => panic!("Add(x, 0) should match"),
    }

    // Add(0, x) should also match and rewrite to x
    let add2 = binary(BinaryOp::Add, zero.clone(), x.clone());
    match matcher.rewrite(&add2, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &x), "Add(0, x) should rewrite to x");
        }
        _ => panic!("Add(0, x) should match"),
    }
}

// ===== Copy Operation Tests =====

#[test]
fn test_copy_struct_pattern() {
    use morok_device::DeviceSpec;

    // Test Copy { src, device } struct pattern matching
    let matcher = patterns! {
        // Match Copy and return the source if it's a constant
        Copy { src: c, .. } if matches!(c.op(), Op::Const(_)) ~> c
    };

    let const_val = UOp::native_const(42.0f32);
    let copy_op = const_val.copy_to_device(DeviceSpec::Cuda { device_id: 0 });

    match matcher.rewrite(&copy_op, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &const_val), "Copy {{ src: c, .. }} should rewrite to c");
        }
        _ => panic!("Copy {{ src: c, .. }} should match when src is constant"),
    }
}

// NOTE: test_copy_f_copy_helper was removed - it tested deprecated UPat API

/// Test context type for `@context` DSL feature.
#[derive(Default)]
struct TestContext {
    counter: u32,
}

impl TestContext {
    fn increment(&mut self) -> u32 {
        self.counter += 1;
        self.counter
    }
}

#[test]
fn test_context_declaration() {
    // Test the @context declaration in patterns! DSL
    // This allows patterns to access a mutable context passed at rewrite time.

    // Create a PatternMatcher<TestContext> using @context
    let matcher = patterns! {
        @context TestContext;

        // Pattern that uses ctx to increment a counter
        x if matches!(x.op(), Op::Const(_)) => {
            let count = ctx.increment();
            if count > 0 {
                Some(Arc::clone(x))
            } else {
                None
            }
        }
    };

    // Create a constant
    let c = UOp::native_const(42i32);

    // Create context
    let mut ctx = TestContext::default();
    assert_eq!(ctx.counter, 0);

    // First rewrite should increment counter
    let result1 = matcher.rewrite(&c, &mut ctx);
    assert!(matches!(result1, RewriteResult::Rewritten(_)));
    assert_eq!(ctx.counter, 1);

    // Second rewrite should increment again
    let result2 = matcher.rewrite(&c, &mut ctx);
    assert!(matches!(result2, RewriteResult::Rewritten(_)));
    assert_eq!(ctx.counter, 2);
}

#[test]
fn test_context_with_graph_rewrite() {
    use crate::rewrite::graph_rewrite_top_down;

    // Test @context with the full graph_rewrite pipeline
    let matcher = patterns! {
        @context TestContext;

        // Replace Add(x, @zero) with x, using context to track rewrites
        Add(x, @zero) => {
            ctx.increment();
            Some(Arc::clone(x))
        }
    };

    let x = UOp::native_const(5i32);
    let zero = UOp::native_const(0i32);
    let add = binary(BinaryOp::Add, x.clone(), zero);

    let mut ctx = TestContext::default();
    let result = graph_rewrite_top_down(&matcher, add, &mut ctx);

    // Should have rewritten Add(5, 0) to 5
    assert!(Arc::ptr_eq(&result, &x));
    // Counter should have been incremented
    assert_eq!(ctx.counter, 1);
}

#[test]
fn test_context_pattern_composition() {
    // Test that PatternMatcher<C> + PatternMatcher<C> works for same context type
    let matcher1 = patterns! {
        @context TestContext;
        Add(x, @zero) => {
            ctx.increment();
            Some(Arc::clone(x))
        }
    };

    let matcher2 = patterns! {
        @context TestContext;
        Mul(x, @one) => {
            ctx.increment();
            ctx.increment(); // Increment twice for mul
            Some(Arc::clone(x))
        }
    };

    // Combine matchers - same context type, so this compiles
    let combined = matcher1 + matcher2;

    let x = UOp::native_const(5i32);
    let zero = UOp::native_const(0i32);
    let one = UOp::native_const(1i32);

    let add_zero = binary(BinaryOp::Add, x.clone(), zero);
    let mul_one = binary(BinaryOp::Mul, x.clone(), one);

    let mut ctx = TestContext::default();

    // Test add pattern
    let result1 = combined.rewrite(&add_zero, &mut ctx);
    assert!(matches!(result1, RewriteResult::Rewritten(_)));
    assert_eq!(ctx.counter, 1); // Add pattern increments once

    // Test mul pattern
    let result2 = combined.rewrite(&mul_one, &mut ctx);
    assert!(matches!(result2, RewriteResult::Rewritten(_)));
    assert_eq!(ctx.counter, 3); // Mul pattern increments twice (1 + 2 = 3)
}

#[test]
fn test_commutative_pattern_with_special_zero() {
    // Test Add[x, @zero] commutative pattern - should match both orderings
    let matcher = patterns! {
        Add[x, @zero] ~> x
    };

    let x = UOp::var("a", morok_dtype::DType::Int32, 0, i64::MAX);
    let zero = UOp::native_const(0i32);

    // Add(x, 0) should match
    let add_x_zero = binary(BinaryOp::Add, x.clone(), zero.clone());
    match matcher.rewrite(&add_x_zero, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &x), "Add(x, 0) should rewrite to x");
        }
        _ => panic!("Add[x, @zero] should match Add(x, 0)"),
    }

    // Add(0, x) should also match (commutative)
    let add_zero_x = binary(BinaryOp::Add, zero.clone(), x.clone());
    match matcher.rewrite(&add_zero_x, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &x), "Add(0, x) should rewrite to x");
        }
        _ => panic!("Add[x, @zero] should match Add(0, x) via commutativity"),
    }
}

#[test]
fn test_commutative_pattern_with_graph_rewrite() {
    use crate::rewrite::graph_rewrite_top_down;

    // Test Add[x, @zero] with graph_rewrite - like the failing property test
    let matcher = patterns! {
        Add[x, @zero] ~> x
    };

    let x = UOp::var("a", morok_dtype::DType::Int32, 0, i64::MAX);
    let zero = UOp::native_const(0i32);

    // Add(0, x) via graph_rewrite
    let add_zero_x = binary(BinaryOp::Add, zero.clone(), x.clone());
    let result = graph_rewrite_top_down(&matcher, add_zero_x, &mut ());

    assert!(Arc::ptr_eq(&result, &x), "graph_rewrite(Add(0, x)) should simplify to x");
}

#[test]
fn test_symbolic_simple_add_zero() {
    use crate::rewrite::graph_rewrite_top_down;
    use crate::symbolic::patterns::{constant_folding_dsl_patterns, identity_and_zero_patterns};

    // Test combining two matchers
    let matcher = constant_folding_dsl_patterns() + identity_and_zero_patterns();

    let x = UOp::var("a", morok_dtype::DType::Int32, 0, i64::MAX);
    let zero = UOp::native_const(0i32);

    // Add(0, x) via graph_rewrite with combined patterns
    let add_zero_x = binary(BinaryOp::Add, zero.clone(), x.clone());
    let result = graph_rewrite_top_down(&matcher, add_zero_x, &mut ());

    assert!(Arc::ptr_eq(&result, &x), "combined patterns + graph_rewrite(Add(0, x)) should simplify to x");
}

// ===== Option Pattern Tests (gate: None, gate: Some(g)) =====

#[test]
fn test_option_none_pattern() {
    // Test gate: None pattern matching
    let matcher = patterns! {
        Index { buffer: b, indices: _, gate: None } ~> b
    };

    let buffer = UOp::native_const(42.0f32);
    let idx = UOp::index_const(0);

    // Ungated index should match
    let ungated = UOp::index().buffer(buffer.clone()).indices(vec![idx.clone()]).call().unwrap();
    match matcher.rewrite(&ungated, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &buffer), "Should extract buffer from ungated Index");
        }
        _ => panic!("Index with gate: None should match"),
    }

    // Gated index should NOT match
    let gate = UOp::const_(DType::Bool, ConstValue::Int(1));
    let gated = UOp::index().buffer(buffer.clone()).indices(vec![idx]).gate(gate).call().unwrap();
    match matcher.rewrite(&gated, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Index with gate: Some(_) should NOT match gate: None pattern"),
    }
}

#[test]
fn test_option_some_pattern() {
    // Test gate: Some(g) pattern matching
    let matcher = patterns! {
        Index { buffer: _, indices: _, gate: Some(g) } ~> g
    };

    let buffer = UOp::native_const(42.0f32);
    let idx = UOp::index_const(0);
    let gate = UOp::const_(DType::Bool, ConstValue::Int(1));

    // Gated index should match and extract the gate
    let gated = UOp::index().buffer(buffer.clone()).indices(vec![idx.clone()]).gate(gate.clone()).call().unwrap();
    match matcher.rewrite(&gated, &mut ()) {
        RewriteResult::Rewritten(r) => {
            assert!(Arc::ptr_eq(&r, &gate), "Should extract gate from gated Index");
        }
        _ => panic!("Index with gate: Some(g) should match"),
    }

    // Ungated index should NOT match
    let ungated = UOp::index().buffer(buffer.clone()).indices(vec![idx]).call().unwrap();
    match matcher.rewrite(&ungated, &mut ()) {
        RewriteResult::NoMatch => {} // Expected
        _ => panic!("Index with gate: None should NOT match gate: Some(g) pattern"),
    }
}

#[test]
fn test_nested_index_with_gate_none() {
    // Test the exact pattern from flatten_cascaded_index in DSL form
    let matcher = patterns! {
        Index {
            buffer: Index { buffer: real_buffer, indices: inner_indices, gate: None },
            indices: outer_indices,
            gate: None
        } if outer_indices.len() == 1 && inner_indices.len() == 1 => |real_buffer, inner_indices| {
            UOp::index().buffer(real_buffer.clone()).indices(vec![inner_indices[0].clone()]).call().ok()
        }
    };

    let real_buffer = UOp::native_const(42.0f32);
    let idx1 = UOp::index_const(5);
    let idx2 = UOp::index_const(10);

    // Create nested Index: INDEX(INDEX(real_buffer, [idx1]), [idx2])
    let inner_idx = UOp::index().buffer(real_buffer.clone()).indices(vec![idx1.clone()]).call().unwrap();
    let outer_idx = UOp::index().buffer(inner_idx.clone()).indices(vec![idx2.clone()]).call().unwrap();

    // Should match and return INDEX(real_buffer, [idx1])
    match matcher.rewrite(&outer_idx, &mut ()) {
        RewriteResult::Rewritten(r) => {
            // Result should be INDEX(real_buffer, [idx1])
            if let Op::Index { buffer, indices, gate } = r.op() {
                assert!(Arc::ptr_eq(buffer, &real_buffer), "Buffer should be real_buffer");
                assert_eq!(indices.len(), 1, "Should have 1 index");
                assert!(Arc::ptr_eq(&indices[0], &idx1), "Index should be idx1 from inner");
                assert!(gate.is_none(), "Gate should be None");
            } else {
                panic!("Result should be Index op");
            }
        }
        _ => panic!("Nested Index pattern should match"),
    }

    // With a gate on outer, should NOT match
    let gate = UOp::const_(DType::Bool, ConstValue::Int(1));
    let gated_outer = UOp::index().buffer(inner_idx.clone()).indices(vec![idx2.clone()]).gate(gate).call().unwrap();
    match matcher.rewrite(&gated_outer, &mut ()) {
        RewriteResult::NoMatch => {} // Expected - outer gate is Some
        _ => panic!("Should NOT match when outer gate is Some"),
    }
}

// ===== Wildcard For-Loop Tests ([*] syntax) =====

#[test]
fn test_for_loop_binary_wildcard() {
    // Test that `binary [*]` expands to ALL binary ops
    #[allow(unused_variables)]
    let matcher = patterns! {
        for op in binary [*] {
            op(x, @zero) ~> x
        }
    };

    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);

    // Test Add(x, 0) => x
    let add_zero = binary(BinaryOp::Add, x.clone(), zero.clone());
    match matcher.rewrite(&add_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Add(x, 0) from binary [*] should match"),
    }

    // Test Mul(x, 0) => x
    let mul_zero = binary(BinaryOp::Mul, x.clone(), zero.clone());
    match matcher.rewrite(&mul_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Mul(x, 0) from binary [*] should match"),
    }

    // Test Xor(x, 0) => x (less common op, but should match with [*])
    let xor_zero = x.try_xor_op(&zero).unwrap();
    match matcher.rewrite(&xor_zero, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &x)),
        _ => panic!("Xor(x, 0) from binary [*] should match"),
    }
}

#[test]
fn test_for_loop_unary_wildcard() {
    // Test that `unary [*]` expands to ALL unary ops
    #[allow(unused_variables)]
    let matcher = patterns! {
        for op in unary [*] {
            op(c) if matches!(c.op(), Op::Const(_)) ~> c
        }
    };

    let c = UOp::native_const(42.0f32);

    // Test Neg(const)
    let neg_c = c.neg();
    match matcher.rewrite(&neg_c, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &c)),
        _ => panic!("Neg(const) from unary [*] should match"),
    }

    // Test Sqrt(const)
    let sqrt_c = c.try_sqrt().unwrap();
    match matcher.rewrite(&sqrt_c, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &c)),
        _ => panic!("Sqrt(const) from unary [*] should match"),
    }

    // Test Exp2(const) - verifies [*] includes all ops
    let exp2_c = c.try_exp2().unwrap();
    match matcher.rewrite(&exp2_c, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &c)),
        _ => panic!("Exp2(const) from unary [*] should match"),
    }
}

#[test]
fn test_for_loop_ternary_wildcard() {
    // Test that `ternary [*]` expands to ALL ternary ops
    #[allow(unused_variables)]
    let matcher = patterns! {
        for op in ternary [*] {
            op(a, b, c) ~> {
                // For testing, just return the first argument
                Arc::clone(a)
            }
        }
    };

    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let c = UOp::native_const(3.0f32);

    // Test Where(a, b, c) => a
    let where_abc = UOp::try_where(a.clone(), b.clone(), c.clone()).unwrap();
    match matcher.rewrite(&where_abc, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &a)),
        _ => panic!("Where from ternary [*] should match"),
    }

    // Test MulAcc(a, b, c) => a
    let mulacc_abc = UOp::try_mulacc(a.clone(), b.clone(), c.clone()).unwrap();
    match matcher.rewrite(&mulacc_abc, &mut ()) {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, &a)),
        _ => panic!("MulAcc from ternary [*] should match"),
    }
}
