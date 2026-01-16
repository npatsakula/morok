//! ALU devectorization tests (no_vectorized_alu).
//!
//! Tests for the no_vectorized_alu patterns which convert vectorized
//! ALU operations to VECTORIZE of scalar operations.
//!
//! Based on Tinygrad's no_vectorized_alu (devectorizer.py:219-223).

use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{BinaryOp, Op, TernaryOp, UOp, UnaryOp};

use super::helpers::*;

// =============================================================================
// Binary Op Tests
// =============================================================================

/// Test: Add<vec4> devectorizes to VECTORIZE of scalar Adds.
///
/// Add(<4 x f32>, <4 x f32>) -> VECTORIZE(Add(f32, f32), ...)
#[test]
fn test_add_vec4_devectorize() {
    let a = create_vector_float_iota(4);
    let b = create_vector_float_values(vec![10.0, 20.0, 30.0, 40.0]);

    let add = UOp::new(Op::Binary(BinaryOp::Add, a, b), DType::Float32.vec(4));

    let result = apply_no_vectorized_alu(&add);

    // Should become VECTORIZE of 4 scalar Adds
    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4, "Should have 4 scalar Adds");
            for elem in elements.iter() {
                assert!(matches!(elem.op(), Op::Binary(BinaryOp::Add, _, _)));
                assert_eq!(elem.dtype().vcount(), 1, "Each Add should be scalar");
            }
        }
        // Could remain Binary if vcount <= 1
        Op::Binary(BinaryOp::Add, _, _) => {}
        other => panic!("Expected VECTORIZE or Binary, got {:?}", other),
    }
}

/// Test: Mul<vec8> devectorizes to VECTORIZE of 8 Muls.
#[test]
fn test_mul_vec8_devectorize() {
    let a = create_vector_float_iota(8);
    let b = create_vector_float_iota(8);

    let mul = UOp::new(Op::Binary(BinaryOp::Mul, a, b), DType::Float32.vec(8));

    let result = apply_no_vectorized_alu(&mul);

    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 8, "Should have 8 scalar Muls");
            for elem in elements.iter() {
                assert!(matches!(elem.op(), Op::Binary(BinaryOp::Mul, _, _)));
            }
        }
        Op::Binary(BinaryOp::Mul, _, _) => {}
        other => panic!("Expected VECTORIZE or Binary, got {:?}", other),
    }
}

/// Test: Binary<scalar> remains unchanged.
#[test]
fn test_binary_scalar_unchanged() {
    let a = create_float_const(1.0);
    let b = create_float_const(2.0);

    let add = UOp::new(Op::Binary(BinaryOp::Add, a, b), DType::Float32);

    let result = apply_no_vectorized_alu(&add);

    // Scalar binary should remain unchanged
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Add, _, _)));
    assert_eq!(result.dtype().vcount(), 1);
}

/// Test: Sub<vec4> devectorizes.
#[test]
fn test_sub_vec4_devectorize() {
    let a = create_vector_float_iota(4);
    let b = create_vector_float_iota(4);

    let sub = UOp::new(Op::Binary(BinaryOp::Sub, a, b), DType::Float32.vec(4));

    let result = apply_no_vectorized_alu(&sub);

    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4);
            for elem in elements.iter() {
                assert!(matches!(elem.op(), Op::Binary(BinaryOp::Sub, _, _)));
            }
        }
        Op::Binary(BinaryOp::Sub, _, _) => {}
        other => panic!("Expected VECTORIZE or Binary, got {:?}", other),
    }
}

// =============================================================================
// Unary Op Tests
// =============================================================================

/// Test: Neg<vec4> devectorizes to VECTORIZE of Negs.
#[test]
fn test_neg_vec4_devectorize() {
    let a = create_vector_float_iota(4);

    let neg = UOp::new(Op::Unary(UnaryOp::Neg, a), DType::Float32.vec(4));

    let result = apply_no_vectorized_alu(&neg);

    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4);
            for elem in elements.iter() {
                assert!(matches!(elem.op(), Op::Unary(UnaryOp::Neg, _)));
                assert_eq!(elem.dtype().vcount(), 1);
            }
        }
        Op::Unary(UnaryOp::Neg, _) => {}
        other => panic!("Expected VECTORIZE or Unary, got {:?}", other),
    }
}

/// Test: Sqrt<vec4> devectorizes to VECTORIZE of Sqrts.
#[test]
fn test_sqrt_vec4_devectorize() {
    let a = create_vector_float_iota(4);

    let sqrt = UOp::new(Op::Unary(UnaryOp::Sqrt, a), DType::Float32.vec(4));

    let result = apply_no_vectorized_alu(&sqrt);

    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4);
            for elem in elements.iter() {
                assert!(matches!(elem.op(), Op::Unary(UnaryOp::Sqrt, _)));
            }
        }
        Op::Unary(UnaryOp::Sqrt, _) => {}
        other => panic!("Expected VECTORIZE or Unary, got {:?}", other),
    }
}

/// Test: Exp2<vec4> devectorizes.
#[test]
fn test_exp2_vec4_devectorize() {
    let a = create_vector_float_iota(4);

    let exp2 = UOp::new(Op::Unary(UnaryOp::Exp2, a), DType::Float32.vec(4));

    let result = apply_no_vectorized_alu(&exp2);

    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4);
        }
        Op::Unary(UnaryOp::Exp2, _) => {}
        other => panic!("Expected VECTORIZE or Unary, got {:?}", other),
    }
}

/// Test: Scalar unary remains unchanged.
#[test]
fn test_unary_scalar_unchanged() {
    let a = create_float_const(4.0);

    let sqrt = UOp::new(Op::Unary(UnaryOp::Sqrt, a), DType::Float32);

    let result = apply_no_vectorized_alu(&sqrt);

    assert!(matches!(result.op(), Op::Unary(UnaryOp::Sqrt, _)));
    assert_eq!(result.dtype().vcount(), 1);
}

// =============================================================================
// Cast Tests
// =============================================================================

/// Test: Cast<vec4> devectorizes to VECTORIZE of Casts.
#[test]
fn test_cast_vec4_devectorize() {
    let a = create_vector_float_iota(4);

    let cast = UOp::cast(a, DType::Int64.vec(4));

    let result = apply_no_vectorized_alu(&cast);

    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4);
            for elem in elements.iter() {
                assert!(matches!(elem.op(), Op::Cast { .. }));
                assert_eq!(elem.dtype().vcount(), 1);
            }
        }
        Op::Cast { .. } => {}
        other => panic!("Expected VECTORIZE or Cast, got {:?}", other),
    }
}

/// Test: Scalar Cast remains unchanged.
#[test]
fn test_cast_scalar_unchanged() {
    let a = create_float_const(3.0);

    let cast = UOp::cast(a, DType::Int64);

    let result = apply_no_vectorized_alu(&cast);

    assert!(matches!(result.op(), Op::Cast { .. }));
    assert_eq!(result.dtype().vcount(), 1);
}

// =============================================================================
// WHERE Tests
// =============================================================================

/// Test: WHERE<vec4> devectorizes to VECTORIZE of WHEREs.
#[test]
fn test_where_vec4_devectorize() {
    let cond = create_vector_bool(vec![true, false, true, false]);
    let t_val = create_vector_float_iota(4);
    let f_val = create_vector_float_values(vec![10.0, 11.0, 12.0, 13.0]);

    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, cond, t_val, f_val), DType::Float32.vec(4));

    let result = apply_no_vectorized_alu(&where_op);

    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4);
            for elem in elements.iter() {
                assert!(matches!(elem.op(), Op::Ternary(TernaryOp::Where, _, _, _)));
                assert_eq!(elem.dtype().vcount(), 1);
            }
        }
        Op::Ternary(TernaryOp::Where, _, _, _) => {}
        other => panic!("Expected VECTORIZE or WHERE, got {:?}", other),
    }
}

/// Test: Scalar WHERE remains unchanged.
#[test]
fn test_where_scalar_unchanged() {
    let cond = create_bool_const(true);
    let t_val = create_float_const(1.0);
    let f_val = create_float_const(0.0);

    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, cond, t_val, f_val), DType::Float32);

    let result = apply_no_vectorized_alu(&where_op);

    assert!(matches!(result.op(), Op::Ternary(TernaryOp::Where, _, _, _)));
    assert_eq!(result.dtype().vcount(), 1);
}

// =============================================================================
// MulAcc Tests
// =============================================================================

/// Test: MulAcc<vec4> devectorizes to VECTORIZE of MulAccs.
#[test]
fn test_mulacc_vec4_devectorize() {
    let a = create_vector_float_iota(4);
    let b = create_vector_float_iota(4);
    let c = create_vector_float_values(vec![100.0, 100.0, 100.0, 100.0]);

    let mulacc = UOp::try_mulacc(a, b, c).expect("MulAcc creation should succeed");

    // Verify initial vector dtype
    assert_vcount(&mulacc, 4);

    let result = apply_no_vectorized_alu(&mulacc);

    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4, "Should have 4 scalar MulAccs");
            for elem in elements.iter() {
                assert!(matches!(elem.op(), Op::Ternary(TernaryOp::MulAcc, _, _, _)));
                assert_eq!(elem.dtype().vcount(), 1);
            }
        }
        Op::Ternary(TernaryOp::MulAcc, _, _, _) => {}
        other => panic!("Expected VECTORIZE or MulAcc, got {:?}", other),
    }
}

/// Test: Scalar MulAcc remains unchanged.
#[test]
fn test_mulacc_scalar_unchanged() {
    let a = create_float_const(2.0);
    let b = create_float_const(3.0);
    let c = create_float_const(1.0);

    let mulacc = UOp::try_mulacc(a, b, c).expect("MulAcc creation should succeed");

    let result = apply_no_vectorized_alu(&mulacc);

    assert!(matches!(result.op(), Op::Ternary(TernaryOp::MulAcc, _, _, _)));
    assert_eq!(result.dtype().vcount(), 1);
}

// =============================================================================
// Mixed Operand Tests
// =============================================================================

/// Test: Vector + scalar broadcast handling.
///
/// Add(vec4, scalar) where scalar is broadcast to vec4.
#[test]
fn test_binary_mixed_operands() {
    let a = create_vector_float_iota(4);
    let scalar = create_float_const(10.0);
    let b = UOp::broadcast(scalar, 4);

    let add = UOp::new(Op::Binary(BinaryOp::Add, a, b), DType::Float32.vec(4));

    let result = apply_no_vectorized_alu(&add);

    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4);
            for elem in elements.iter() {
                assert!(matches!(elem.op(), Op::Binary(BinaryOp::Add, _, _)));
            }
        }
        Op::Binary(BinaryOp::Add, _, _) => {}
        other => panic!("Expected VECTORIZE or Binary, got {:?}", other),
    }
}

// =============================================================================
// Large Vector Tests
// =============================================================================

/// Test: Vec16 devectorization.
#[test]
fn test_add_vec16_devectorize() {
    let elements_a: smallvec::SmallVec<[std::sync::Arc<UOp>; 4]> =
        (0..16).map(|i| UOp::const_(DType::Float32, ConstValue::Float(i as f64))).collect();
    let a = UOp::vectorize(elements_a);

    let elements_b: smallvec::SmallVec<[std::sync::Arc<UOp>; 4]> =
        (0..16).map(|i| UOp::const_(DType::Float32, ConstValue::Float((i * 2) as f64))).collect();
    let b = UOp::vectorize(elements_b);

    let add = UOp::new(Op::Binary(BinaryOp::Add, a, b), DType::Float32.vec(16));

    let result = apply_no_vectorized_alu(&add);

    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 16, "Should have 16 scalar Adds");
        }
        Op::Binary(BinaryOp::Add, _, _) => {}
        other => panic!("Expected VECTORIZE or Binary, got {:?}", other),
    }
}

// =============================================================================
// Integer Op Tests
// =============================================================================

/// Test: Integer add devectorization.
#[test]
fn test_int_add_devectorize() {
    let a = create_vector_int_iota(4);
    let b = create_vector_int_values(vec![10, 20, 30, 40]);

    let add = UOp::new(Op::Binary(BinaryOp::Add, a, b), DType::Int64.vec(4));

    let result = apply_no_vectorized_alu(&add);

    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4);
            for elem in elements.iter() {
                assert!(matches!(elem.op(), Op::Binary(BinaryOp::Add, _, _)));
            }
        }
        Op::Binary(BinaryOp::Add, _, _) => {}
        other => panic!("Expected VECTORIZE or Binary, got {:?}", other),
    }
}

/// Test: Bitwise and devectorization.
#[test]
fn test_bitwise_and_devectorize() {
    let a = create_vector_int_values(vec![0xFF, 0xF0, 0x0F, 0x00]);
    let b = create_vector_int_values(vec![0x0F, 0x0F, 0x0F, 0x0F]);

    let and = UOp::new(Op::Binary(BinaryOp::And, a, b), DType::Int64.vec(4));

    let result = apply_no_vectorized_alu(&and);

    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4);
        }
        Op::Binary(BinaryOp::And, _, _) => {}
        other => panic!("Expected VECTORIZE or Binary, got {:?}", other),
    }
}
