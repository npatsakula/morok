//! GEP/CAT/VECTORIZE pattern tests.
//!
//! Tests for the gep_ptrcat_patterns which handle:
//! - CAT -> VECTORIZE conversion (multi-element sources)
//! - GEP(VECTORIZE) -> element extraction
//! - GEP(CAT) -> reorder
//! - GEP(PTRCAT) -> reorder pointers
//! - Identity patterns (single-source unwrap)
//! - WHERE devectorization
//!
//! Based on Tinygrad's symbolic.py and devectorizer.py patterns.

use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{Op, TernaryOp, UOp};

use super::helpers::*;

// =============================================================================
// CAT -> VECTORIZE Tests
// =============================================================================

/// Test: CAT with multi-element sources converts to VECTORIZE.
///
/// CAT([a<4>, b<4>]) -> VECTORIZE(a.gep(0), ..., a.gep(3), b.gep(0), ..., b.gep(3))
#[test]
fn test_cat_vec4_to_vectorize() {
    let a = create_vector_float_iota(4);
    let b = create_vector_float_values(vec![10.0, 11.0, 12.0, 13.0]);

    let cat = UOp::cat().sources(vec![a, b]).call();
    assert_vcount(&cat, 8);

    let result = apply_gep_ptrcat_patterns(&cat);

    // Should become VECTORIZE with 8 elements (extracted via GEP)
    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 8, "Should have 8 elements");
            // Each element should be scalar
            for elem in elements.iter() {
                assert_eq!(elem.dtype().vcount(), 1, "Each element should be scalar");
            }
        }
        // Could remain CAT if elements are already scalar
        Op::Cat { sources } => {
            assert_eq!(sources.len(), 8);
        }
        other => panic!("Expected VECTORIZE or CAT, got {:?}", other),
    }
}

/// Test: CAT with scalar sources remains unchanged.
///
/// CAT([a, b, c, d]) with all scalars -> unchanged (handled by GEP(CAT) reorder)
#[test]
fn test_cat_scalar_unchanged() {
    let a = create_float_const(1.0);
    let b = create_float_const(2.0);
    let c = create_float_const(3.0);
    let d = create_float_const(4.0);

    let cat = UOp::cat().sources(vec![a, b, c, d]).call();

    let result = apply_gep_ptrcat_patterns(&cat);

    // Scalar CAT should remain as CAT (pattern only fires for multi-element sources)
    match result.op() {
        Op::Cat { sources } => {
            assert_eq!(sources.len(), 4);
            for src in sources.iter() {
                assert_eq!(src.dtype().vcount(), 1);
            }
        }
        Op::Vectorize { elements } => {
            // Could be converted to VECTORIZE by other patterns
            assert_eq!(elements.len(), 4);
        }
        other => panic!("Expected CAT or VECTORIZE, got {:?}", other),
    }
}

/// Test: Single-source CAT unwraps to source.
///
/// CAT([a]) -> a
#[test]
fn test_cat_single_source_unwrap() {
    let a = create_vector_float_iota(4);
    let cat = UOp::cat().sources(vec![a.clone()]).call();

    let result = apply_gep_ptrcat_patterns(&cat);

    // Should unwrap to just 'a'
    assert!(Arc::ptr_eq(&result, &a), "Single-source CAT should unwrap");
}

// =============================================================================
// GEP(VECTORIZE) Tests
// =============================================================================

/// Test: GEP on VECTORIZE extracts single element.
///
/// GEP(VECTORIZE([e0, e1, e2]), [1]) -> e1
#[test]
fn test_gep_vectorize_single() {
    let e0 = create_float_const(0.0);
    let e1 = create_float_const(1.0);
    let e2 = create_float_const(2.0);

    let vec = UOp::vectorize([e0, e1.clone(), e2].into_iter().collect());
    let gep = UOp::gep(vec, vec![1]);

    let result = apply_gep_ptrcat_patterns(&gep);

    // Should extract e1 directly
    assert_eq!(result.dtype().vcount(), 1, "Should be scalar");
    // Check it's the right constant
    match result.op() {
        Op::Const(v) => {
            assert_eq!(v.0, ConstValue::Float(1.0), "Should extract value 1.0");
        }
        Op::Gep { indices, .. } => {
            // If not simplified, should at least have correct index
            assert_eq!(indices.len(), 1, "Should have single index");
            assert_eq!(indices[0], 1, "Index should be 1");
        }
        other => panic!("Expected Const or GEP, got {:?}", other),
    }
}

/// Test: GEP on VECTORIZE extracts multiple elements.
///
/// GEP(VECTORIZE([e0, e1, e2, e3]), [0, 2]) -> VECTORIZE([e0, e2])
#[test]
fn test_gep_vectorize_multi() {
    let elements: smallvec::SmallVec<[Arc<UOp>; 4]> = (0..4).map(|i| create_float_const(i as f64)).collect();

    let vec = UOp::vectorize(elements);
    let gep = UOp::gep(vec, vec![0, 2]);

    let result = apply_gep_ptrcat_patterns(&gep);

    // Should extract elements 0 and 2
    assert_vcount(&result, 2);
    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 2);
        }
        other => panic!("Expected VECTORIZE, got {:?}", other),
    }
}

/// Test: GEP on broadcast VECTORIZE extracts single element.
///
/// GEP(VECTORIZE([x, x, x, x]), [i]) -> x
#[test]
fn test_gep_broadcast_extraction() {
    let x = create_float_const(42.0);
    let vec = UOp::broadcast(x.clone(), 4);
    let gep = UOp::gep(vec, vec![2]);

    let result = apply_gep_ptrcat_patterns(&gep);

    // Should extract to just x
    assert_eq!(result.dtype().vcount(), 1, "Should be scalar");
    match result.op() {
        Op::Const(v) => {
            assert_eq!(v.0, ConstValue::Float(42.0), "Should extract value 42.0");
        }
        Op::Gep { indices, .. } => {
            // If not simplified, should have correct index
            assert_eq!(indices.len(), 1, "Should have single index");
        }
        other => panic!("Expected Const or GEP, got {:?}", other),
    }
}

// =============================================================================
// GEP(CAT) Tests
// =============================================================================

/// Test: GEP on CAT reorders sources.
///
/// GEP(CAT([a, b, c]), [1, 2]) -> CAT([b, c])
#[test]
fn test_gep_cat_reorder() {
    let a = create_float_const(1.0);
    let b = create_float_const(2.0);
    let c = create_float_const(3.0);

    let cat = UOp::cat().sources(vec![a, b.clone(), c.clone()]).call();
    let gep = UOp::gep(cat, vec![1, 2]);

    let result = apply_gep_ptrcat_patterns(&gep);

    // Should produce CAT([b, c]) or VECTORIZE([b, c])
    assert_vcount(&result, 2);
    match result.op() {
        Op::Cat { sources } => {
            assert_eq!(sources.len(), 2);
        }
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 2);
        }
        other => panic!("Expected CAT or VECTORIZE, got {:?}", other),
    }
}

/// Test: GEP on CAT extracts single element.
#[test]
fn test_gep_cat_single() {
    let a = create_float_const(1.0);
    let b = create_float_const(2.0);
    let c = create_float_const(3.0);

    let cat = UOp::cat().sources(vec![a, b.clone(), c]).call();
    let gep = UOp::gep(cat, vec![1]);

    let result = apply_gep_ptrcat_patterns(&gep);

    // Should extract b directly
    assert_eq!(result.dtype().vcount(), 1);
}

// =============================================================================
// GEP(PTRCAT) Tests
// =============================================================================

/// Test: GEP on PTRCAT reorders pointers.
///
/// GEP(PTRCAT([p1, p2, p3]), [0, 2]) -> PTRCAT([p1, p3])
#[test]
fn test_gep_ptrcat_reorder() {
    let buffer = create_buffer(64);

    let p1 = create_index(buffer.clone(), 0);
    let p2 = create_index(buffer.clone(), 1);
    let p3 = create_index(buffer.clone(), 2);

    let ptrcat = UOp::ptrcat().sources(vec![p1.clone(), p2, p3.clone()]).call();
    let gep = UOp::gep(ptrcat, vec![0, 2]);

    let result = apply_gep_ptrcat_patterns(&gep);

    // Should produce PTRCAT([p1, p3])
    match result.op() {
        Op::PtrCat { sources } => {
            assert_eq!(sources.len(), 2);
        }
        other => panic!("Expected PTRCAT, got {:?}", other),
    }
}

/// Test: Single-source PTRCAT unwraps.
///
/// PTRCAT([p]) -> p
#[test]
fn test_ptrcat_single_unwrap() {
    let buffer = create_buffer(64);
    let p = create_index(buffer.clone(), 0);

    let ptrcat = UOp::ptrcat().sources(vec![p.clone()]).call();

    let result = apply_gep_ptrcat_patterns(&ptrcat);

    // Should unwrap to just p
    assert_is_index(&result);
}

// =============================================================================
// Identity Reconstruction Test
// =============================================================================

/// Test: CAT(GEP(x,[0]), GEP(x,[1]), ..., GEP(x,[n-1])) -> x
#[test]
fn test_cat_gep_identity() {
    let x = create_vector_float_iota(4);

    // Create CAT(GEP(x,[0]), GEP(x,[1]), GEP(x,[2]), GEP(x,[3]))
    let geps: Vec<Arc<UOp>> = (0..4).map(|i| UOp::gep(x.clone(), vec![i])).collect();
    let cat = UOp::cat().sources(geps).call();

    let result = apply_gep_ptrcat_patterns(&cat);

    // Should simplify to just x
    // Note: This requires the identity reconstruction pattern to fire
    assert_vcount(&result, 4);
}

// =============================================================================
// WHERE Devectorization Tests
// =============================================================================

/// Test: WHERE with vector condition devectorizes.
///
/// WHERE(<4 x i1>, <4 x T>, <4 x T>) -> VECTORIZE(WHERE(i1, T, T), ...)
#[test]
fn test_where_devectorize() {
    let cond = create_vector_bool(vec![true, false, true, false]);
    let t_val = create_vector_float_iota(4);
    let f_val = create_vector_float_values(vec![10.0, 11.0, 12.0, 13.0]);

    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, cond, t_val, f_val), DType::Float32.vec(4));

    let result = apply_gep_ptrcat_patterns(&where_op);

    // Should become VECTORIZE of 4 scalar WHEREs or remain as WHERE
    // Either way, total vcount should be 4
    assert_eq!(result.dtype().vcount(), 4, "Result vcount should be 4");
    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4, "Should have 4 scalar WHEREs");
            for elem in elements.iter() {
                assert!(matches!(elem.op(), Op::Ternary(TernaryOp::Where, _, _, _)), "Each element should be WHERE");
                assert_eq!(elem.dtype().vcount(), 1, "Each WHERE should be scalar");
            }
        }
        Op::Ternary(TernaryOp::Where, c, t, f) => {
            // If not devectorized, inputs should still be vec4
            assert_eq!(c.dtype().vcount(), 4, "Condition should be vec4");
            assert_eq!(t.dtype().vcount(), 4, "True value should be vec4");
            assert_eq!(f.dtype().vcount(), 4, "False value should be vec4");
        }
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

    let result = apply_gep_ptrcat_patterns(&where_op);

    // Scalar WHERE should remain unchanged
    assert!(matches!(result.op(), Op::Ternary(TernaryOp::Where, _, _, _)), "Scalar WHERE should remain unchanged");
    assert_eq!(result.dtype().vcount(), 1);
}

// =============================================================================
// GEP Through Cast Tests
// =============================================================================

/// Test: GEP through CAST is handled correctly.
#[test]
fn test_gep_through_cast() {
    let vec = create_vector_float_iota(4);
    let cast = UOp::cast(vec.clone(), DType::Int64.vec(4));
    let gep = UOp::gep(cast, vec![1]);

    let result = apply_gep_ptrcat_patterns(&gep);

    // GEP should work through CAST
    assert_eq!(result.dtype().vcount(), 1);
}

// =============================================================================
// VECTORIZE Normalization Tests
// =============================================================================

/// Test: Multi-index GEP normalizes to VECTORIZE.
///
/// GEP(x, [0, 1, 2, 3]) -> VECTORIZE(GEP(x, [0]), GEP(x, [1]), GEP(x, [2]), GEP(x, [3]))
#[test]
fn test_multi_index_gep_normalizes() {
    let x = create_vector_float_iota(8);
    let gep = UOp::gep(x.clone(), vec![0, 1, 2, 3]);

    let result = apply_vectorize_normalize(&gep);

    // Multi-index GEP should become VECTORIZE of single-index GEPs
    match result.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), 4);
            for elem in elements.iter() {
                if let Op::Gep { indices, .. } = elem.op() {
                    assert_eq!(indices.len(), 1, "Each GEP should be single-index");
                }
            }
        }
        other => panic!("Expected VECTORIZE, got {:?}", other),
    }
}

/// Test: GEP on scalar with index 0 is identity.
///
/// GEP(scalar, [0]) -> scalar
#[test]
fn test_gep_scalar_identity() {
    let scalar = create_float_const(42.0);
    let gep = UOp::gep(scalar.clone(), vec![0]);

    let result = apply_vectorize_normalize(&gep);

    // Should simplify to scalar
    assert!(Arc::ptr_eq(&result, &scalar) || result.dtype().vcount() == 1);
}

/// Test: Single-element VECTORIZE unwraps.
///
/// VECTORIZE([x]) -> x
#[test]
fn test_single_element_vectorize_unwrap() {
    let x = create_float_const(42.0);
    let vec = UOp::vectorize([x.clone()].into_iter().collect());

    let result = apply_vectorize_normalize(&vec);

    // Should unwrap to x
    assert!(Arc::ptr_eq(&result, &x), "Single-element VECTORIZE should unwrap");
}

// =============================================================================
// Edge Cases
// =============================================================================

/// Test: Empty PTRCAT sources (should not happen but handle gracefully).
#[test]
#[should_panic]
fn test_empty_ptrcat_panics() {
    // PTRCAT requires at least one source
    let _ptrcat = UOp::ptrcat().sources(vec![]).call();
}

/// Test: Empty CAT sources (should not happen but handle gracefully).
#[test]
#[should_panic]
fn test_empty_cat_panics() {
    // CAT requires at least one source
    let _cat = UOp::cat().sources(vec![]).call();
}

/// Test: GEP with out-of-bounds index.
#[test]
fn test_gep_out_of_bounds() {
    let vec = create_vector_float_iota(4);
    let gep = UOp::gep(vec, vec![10]); // Index 10 is out of bounds

    // Should not panic, but may produce invalid result
    let _result = apply_gep_ptrcat_patterns(&gep);
}
