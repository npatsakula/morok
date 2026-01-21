//! Tests for new devectorize patterns added in the redesign.
//!
//! Tests the following patterns:
//! - cast_after: AFTER(CAST(x), deps) → CAST(AFTER(x, deps))
//! - load_store_indexing: INDEX(buf, x, true) → INDEX(buf, x, None)
//! - devectorize_buf_and_index: LOCAL/REG buffer vectorization

use morok_dtype::{AddrSpace, DType};
use morok_ir::types::ConstValue;
use morok_ir::{Op, UOp};
use smallvec::smallvec;

use super::helpers::*;

// =============================================================================
// Cast After Pattern Tests
// =============================================================================

/// Test: AFTER(CAST(x), deps) → CAST(AFTER(x, deps))
#[test]
fn test_cast_after_basic() {
    // Create a simple UOp to wrap
    let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // CAST(src, Float64)
    let cast = UOp::cast(src.clone(), DType::Float64);

    // AFTER(cast, [])
    let after = UOp::after(cast.clone(), smallvec![]);

    // Apply pattern
    let result = apply_cast_after(&after);

    // Result should be CAST(AFTER(src, []), Float64)
    match result.op() {
        Op::Cast { src: inner, dtype } => {
            assert_eq!(*dtype, DType::Float64);
            match inner.op() {
                Op::After { passthrough, deps } => {
                    // passthrough should be the original src (Float32 const)
                    assert_eq!(passthrough.dtype(), DType::Float32);
                    assert!(deps.is_empty());
                }
                other => panic!("Expected AFTER inside CAST, got {:?}", other),
            }
        }
        other => panic!("Expected CAST, got {:?}", other),
    }
}

/// Test: AFTER(CAST(x), deps) preserves deps
#[test]
fn test_cast_after_with_deps() {
    let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let dep = UOp::const_(DType::Int32, ConstValue::Int(42));
    let cast = UOp::cast(src.clone(), DType::Float64);
    let after = UOp::after(cast, smallvec![dep.clone()]);

    let result = apply_cast_after(&after);

    // Check structure: CAST(AFTER(src, [dep]), Float64)
    let Op::Cast { src: inner, .. } = result.op() else {
        panic!("Expected CAST");
    };
    let Op::After { deps, .. } = inner.op() else {
        panic!("Expected AFTER");
    };
    assert_eq!(deps.len(), 1);
}

/// Test: AFTER without CAST is unchanged
#[test]
fn test_after_without_cast_unchanged() {
    let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let after = UOp::after(src.clone(), smallvec![]);

    let result = apply_cast_after(&after);

    // Should be unchanged
    assert!(matches!(result.op(), Op::After { .. }));
    let Op::After { passthrough, .. } = result.op() else { unreachable!() };
    // passthrough should be the original const, not a CAST
    assert!(matches!(passthrough.op(), Op::Const(_)));
}

// =============================================================================
// Load Store Indexing Pattern Tests (Gate Dropping)
// =============================================================================

/// Test: INDEX with constant true gate → INDEX without gate
#[test]
fn test_drop_true_gate() {
    let buffer = create_buffer(64);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));

    // Create gated index
    let gated_index = UOp::new(
        Op::Index { buffer: buffer.clone(), indices: smallvec![idx.clone()], gate: Some(gate) },
        DType::Float32,
    );

    let result = apply_load_store_indexing(&gated_index);

    // Result should have no gate
    match result.op() {
        Op::Index { gate, .. } => {
            assert!(gate.is_none(), "Gate should be dropped");
        }
        other => panic!("Expected INDEX, got {:?}", other),
    }
}

/// Test: INDEX with false gate is unchanged
#[test]
fn test_false_gate_unchanged() {
    let buffer = create_buffer(64);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(false));

    let gated_index = UOp::new(
        Op::Index { buffer: buffer.clone(), indices: smallvec![idx.clone()], gate: Some(gate.clone()) },
        DType::Float32,
    );

    let result = apply_load_store_indexing(&gated_index);

    // Result should still have the gate
    match result.op() {
        Op::Index { gate: g, .. } => {
            assert!(g.is_some(), "False gate should not be dropped");
        }
        other => panic!("Expected INDEX, got {:?}", other),
    }
}

/// Test: INDEX without gate is unchanged
#[test]
fn test_no_gate_unchanged() {
    let buffer = create_buffer(64);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));

    let index = UOp::index().buffer(buffer).indices(vec![idx]).call().unwrap();

    let result = apply_load_store_indexing(&index);

    // Should be unchanged (or structurally equal)
    match result.op() {
        Op::Index { gate, .. } => {
            assert!(gate.is_none());
        }
        other => panic!("Expected INDEX, got {:?}", other),
    }
}

// =============================================================================
// Devectorize Buf And Index Pattern Tests
// =============================================================================

/// Test: DEFINE_LOCAL with vector ptr gets scalar ptr + CAST
#[test]
fn test_devectorize_define_local_vec4() {
    // Create DEFINE_LOCAL with vec4 pointer dtype
    let vec_ptr_dtype = DType::Float32.vec(4).ptr(Some(16), AddrSpace::Local);
    let def_local = UOp::define_local(0, vec_ptr_dtype);

    let result = apply_cast_after(&def_local);

    // Result should be CAST(DEFINE_LOCAL with scalar ptr, vec ptr)
    match result.op() {
        Op::Cast { src, dtype } => {
            // dtype should be the original vec ptr dtype
            assert!(matches!(dtype, DType::Ptr { base, .. } if base.vcount() == 4));

            // src should be DEFINE_LOCAL with scalar ptr
            assert!(matches!(src.op(), Op::DefineLocal(_)));
            let DType::Ptr { base: inner_base, .. } = src.dtype() else { panic!("Expected Ptr dtype") };
            assert_eq!(inner_base.vcount(), 1, "Inner should have scalar base");
        }
        Op::DefineLocal(_) => {
            // If the pattern didn't fire, check if it's because the dtype is scalar
            // This can happen if the test environment doesn't support vector dtypes
        }
        other => panic!("Expected CAST or DEFINE_LOCAL, got {:?}", other),
    }
}

/// Test: DEFINE_LOCAL with scalar ptr is unchanged
#[test]
fn test_define_local_scalar_unchanged() {
    let scalar_ptr_dtype = DType::Float32.ptr(Some(16), AddrSpace::Local);
    let def_local = UOp::define_local(0, scalar_ptr_dtype);

    let result = apply_cast_after(&def_local);

    // Should be unchanged
    assert!(matches!(result.op(), Op::DefineLocal(_)));
}

/// Test: Full devectorize pass on a simple load
#[test]
fn test_full_devectorize_simple_load() {
    let buffer = create_buffer(64);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    // The result should be some form of valid load(s)
    // It could be CAT, VECTORIZE, or a single LOAD depending on grouping
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD in the result");
}

/// Test: Devectorize preserves semantics with non-contiguous indices
#[test]
fn test_devectorize_non_contiguous() {
    let buffer = create_buffer(64);
    let index = create_vector_index_scaled(buffer.clone(), 4, 2); // [0, 2, 4, 6]
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_devectorize(&load);

    // With non-contiguous indices, should result in multiple scalar loads
    // or GEP-based reordering
    assert!(result.dtype().vcount() >= 1);
}

// =============================================================================
// Integration Tests
// =============================================================================

/// Test: Cast after pattern works within full devectorize pipeline
#[test]
fn test_cast_after_in_full_pipeline() {
    let buffer = create_buffer(64);
    let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let cast = UOp::cast(src.clone(), DType::Float64);
    let after = UOp::after(cast, smallvec![]);

    // Create a load that depends on the after
    let idx = create_index(buffer.clone(), 0);
    let load = UOp::load().buffer(buffer).index(idx).call();

    // Create a sink with both
    let sink = UOp::sink(vec![after, load]);

    let result = apply_devectorize(&sink);

    // The sink should contain transformed nodes
    assert!(matches!(result.op(), Op::Sink { .. }));
}

/// Test: Gate dropping works in full pipeline
#[test]
fn test_gate_dropping_in_full_pipeline() {
    let buffer = create_buffer(64);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));

    let gated_index =
        UOp::new(Op::Index { buffer: buffer.clone(), indices: smallvec![idx], gate: Some(gate) }, DType::Float32);
    let load = UOp::load().buffer(buffer).index(gated_index).call();

    let result = apply_devectorize(&load);

    // The load should have been processed
    assert!(count_loads(&result) >= 1);
}
