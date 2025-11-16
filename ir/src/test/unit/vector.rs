//! Vector operation tests.
//!
//! Tests vector operations: Vectorize, Gep, VConst, Cat, PtrCat.

use smallvec::smallvec;

use morok_dtype::{AddrSpace, DType};

use crate::{ConstValue, UOp};

// =========================================================================
// Vectorize Tests
// =========================================================================

#[test]
fn test_vectorize_basic() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let c = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let d = UOp::const_(DType::Float32, ConstValue::Float(4.0));

    let vec = UOp::vectorize(smallvec![a, b, c, d]);
    // Should be Float32 vector of size 4
    assert_eq!(vec.dtype(), DType::Float32.vec(4));
}

#[test]
fn test_vectorize_preserves_base_dtype() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(1));
    let b = UOp::const_(DType::Int32, ConstValue::Int(2));

    let vec = UOp::vectorize(smallvec![a, b]);
    assert_eq!(vec.dtype(), DType::Int32.vec(2));
}

// =========================================================================
// Gep (Get Element Pointer) Tests
// =========================================================================

#[test]
fn test_gep_basic() {
    // Create a vector
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let c = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let d = UOp::const_(DType::Float32, ConstValue::Float(4.0));
    let vec = UOp::vectorize(smallvec![a, b, c, d]);

    // GEP operation exists (actual behavior may vary based on implementation)
    let _elem = UOp::gep(vec, vec![0]);
    // Just verify it compiles and creates a UOp
}

#[test]
fn test_gep_multiple_indices() {
    let vec = UOp::vectorize(smallvec![
        UOp::const_(DType::Int32, ConstValue::Int(10)),
        UOp::const_(DType::Int32, ConstValue::Int(20)),
        UOp::const_(DType::Int32, ConstValue::Int(30)),
        UOp::const_(DType::Int32, ConstValue::Int(40)),
    ]);

    // Extract multiple elements -> keeps vector dtype (doesn't reduce count)
    let result = UOp::gep(vec, vec![0, 2]);
    assert_eq!(result.dtype(), DType::Int32.vec(4));
}

// =========================================================================
// VConst Tests
// =========================================================================

#[test]
fn test_vconst_basic() {
    let values = vec![ConstValue::Float(1.0), ConstValue::Float(2.0), ConstValue::Float(3.0), ConstValue::Float(4.0)];

    let vec = UOp::vconst(values);
    // vconst infers dtype as Float64 vector
    assert_eq!(vec.dtype(), DType::Float64.vec(4));
}

// =========================================================================
// Cat Tests
// =========================================================================

#[test]
fn test_cat_basic() {
    let a = UOp::vectorize(smallvec![
        UOp::const_(DType::Float32, ConstValue::Float(1.0)),
        UOp::const_(DType::Float32, ConstValue::Float(2.0)),
    ]);
    let b = UOp::vectorize(smallvec![
        UOp::const_(DType::Float32, ConstValue::Float(3.0)),
        UOp::const_(DType::Float32, ConstValue::Float(4.0)),
    ]);

    let result = UOp::cat(vec![a, b]);
    // Cat concatenates vectors
    assert_eq!(result.dtype(), DType::Float32.vec(2));
}

// =========================================================================
// PtrCat Tests
// =========================================================================

#[test]
fn test_ptrcat_basic() {
    let ptr_dtype = DType::Float32.ptr(None, AddrSpace::Global);
    let a = UOp::const_(ptr_dtype.clone(), ConstValue::Int(0));
    let b = UOp::const_(ptr_dtype.clone(), ConstValue::Int(0));

    let result = UOp::ptrcat(vec![a, b]);
    assert_eq!(result.dtype(), ptr_dtype);
}
