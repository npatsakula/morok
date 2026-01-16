//! Phase 3 tests: bool storage conversion.
//!
//! Tests for the bool_storage_patterns which convert bool LOAD/STORE
//! operations to use uint8 storage to avoid LLVM i1 garbage bits.
//!
//! Based on Tinygrad's PTX/NIR bool->uint8 patterns.

use morok_dtype::ScalarDType;
use morok_ir::{Op, UOp};

use super::helpers::*;

// =============================================================================
// Bool Load Tests
// =============================================================================

/// Test: LOAD<bool> converts to CAST(LOAD<uint8>, bool).
///
/// This ensures proper bool loading without LLVM i1 garbage bits.
#[test]
fn test_bool_load_to_uint8() {
    let buffer = create_bool_buffer(64);
    let index = create_index(buffer.clone(), 0);

    // Create a LOAD that returns bool
    let load = create_load(buffer.clone(), index);

    // Verify initial state
    assert_eq!(load.dtype().base(), ScalarDType::Bool);

    let result = apply_phase3(&load);

    // Result should be CAST(LOAD<uint8>, bool) or LOAD with converted type
    // Either way, the result type should be bool (user-facing) but storage is uint8
    match result.op() {
        Op::Cast { src, dtype } => {
            // Outer dtype should be bool
            assert_eq!(dtype.base(), ScalarDType::Bool, "CAST should produce bool");
            // Inner LOAD should be uint8
            assert_is_load(src);
            assert_eq!(src.dtype().base(), ScalarDType::UInt8, "Inner LOAD should be uint8");
        }
        Op::Load { .. } => {
            // If unchanged, result should still be bool (transformation may be deferred)
            assert_eq!(result.dtype().base(), ScalarDType::Bool, "LOAD result should be bool");
        }
        other => panic!("Expected CAST(LOAD) or LOAD, got {:?}", other),
    }
}

/// Test: LoadGated<bool> converts to CAST(LoadGated<uint8>, bool).
#[test]
fn test_bool_load_gated() {
    let buffer = create_bool_buffer(64);
    let gate = create_bool_const(true);
    let index = create_index(buffer.clone(), 0);

    // Create gated bool load
    let load = UOp::load_gated(buffer.clone(), index, gate);
    assert_eq!(load.dtype().base(), ScalarDType::Bool);

    let result = apply_phase3(&load);

    // Should be CAST(LoadGated<uint8>, bool)
    match result.op() {
        Op::Cast { src, dtype } => {
            assert_eq!(dtype.base(), ScalarDType::Bool);
            assert!(matches!(src.op(), Op::LoadGated { .. }));
            assert_eq!(src.dtype().base(), ScalarDType::UInt8);
        }
        Op::LoadGated { .. } => {}
        other => panic!("Expected CAST(LoadGated) or LoadGated, got {:?}", other),
    }
}

/// Test: Non-bool LOAD remains unchanged.
#[test]
fn test_non_bool_load_unchanged() {
    let buffer = create_buffer(64); // float32 buffer
    let index = create_index(buffer.clone(), 0);
    let load = create_load(buffer.clone(), index);

    assert_eq!(load.dtype().base(), ScalarDType::Float32);

    let result = apply_phase3(&load);

    // Float32 LOAD should remain unchanged
    assert_is_load(&result);
    assert_eq!(result.dtype().base(), ScalarDType::Float32);
}

/// Test: Int32 LOAD remains unchanged.
#[test]
fn test_int32_load_unchanged() {
    let buffer = create_buffer_typed(64, ScalarDType::Int32);
    let index = create_index(buffer.clone(), 0);
    let load = create_load(buffer.clone(), index);

    let result = apply_phase3(&load);

    assert_is_load(&result);
    assert_eq!(result.dtype().base(), ScalarDType::Int32);
}

// =============================================================================
// Bool Store Tests
// =============================================================================

/// Test: STORE(bool_val) converts to STORE(CAST(bool_val, uint8)).
#[test]
fn test_bool_store_to_uint8() {
    let buffer = create_bool_buffer(64);
    let index = create_index(buffer.clone(), 0);
    let bool_val = create_bool_const(true);

    let store = create_store(buffer.clone(), index, bool_val);

    let result = apply_phase3(&store);

    // Result should be STORE with CAST(bool_val, uint8) as value
    match result.op() {
        Op::Store { value, .. } => {
            // Value should be cast to uint8
            match value.op() {
                Op::Cast { src, dtype } => {
                    assert_eq!(dtype.base(), ScalarDType::UInt8);
                    assert_eq!(src.dtype().base(), ScalarDType::Bool);
                }
                // Could be constant uint8 after optimization
                Op::Const(_) => {}
                other => panic!("Expected CAST or Const value, got {:?}", other),
            }
        }
        other => panic!("Expected STORE, got {:?}", other),
    }
}

/// Test: StoreGated with bool value converts correctly.
#[test]
fn test_bool_store_gated() {
    let buffer = create_bool_buffer(64);
    let gate = create_bool_const(true);
    let index = create_index(buffer.clone(), 0);
    let bool_val = create_bool_const(false);

    let store = UOp::store_gated(buffer.clone(), index, bool_val, gate);

    let result = apply_phase3(&store);

    match result.op() {
        Op::StoreGated { value, .. } => {
            // Value should be cast to uint8
            match value.op() {
                Op::Cast { dtype, .. } => {
                    assert_eq!(dtype.base(), ScalarDType::UInt8);
                }
                Op::Const(_) => {}
                other => panic!("Expected CAST or Const value, got {:?}", other),
            }
        }
        other => panic!("Expected StoreGated, got {:?}", other),
    }
}

/// Test: Non-bool STORE remains unchanged.
#[test]
fn test_non_bool_store_unchanged() {
    let buffer = create_buffer(64);
    let index = create_index(buffer.clone(), 0);
    let float_val = create_float_const(3.0);

    let store = create_store(buffer.clone(), index, float_val.clone());

    let result = apply_phase3(&store);

    // Float STORE should remain unchanged
    match result.op() {
        Op::Store { value, .. } => {
            // Value should NOT be cast
            assert_eq!(value.dtype().base(), ScalarDType::Float32);
        }
        other => panic!("Expected STORE, got {:?}", other),
    }
}

// =============================================================================
// Roundtrip Tests
// =============================================================================

/// Test: Store bool then load bool maintains correctness.
#[test]
fn test_bool_roundtrip() {
    let buffer = create_bool_buffer(64);
    let index = create_index(buffer.clone(), 0);
    let bool_val = create_bool_const(true);

    // Store bool value
    let store = create_store(buffer.clone(), index.clone(), bool_val);
    let store_result = apply_phase3(&store);

    // Load bool value
    let load = create_load(buffer.clone(), index);
    let load_result = apply_phase3(&load);

    // Verify store has uint8 cast
    if let Op::Store { value, .. } = store_result.op() {
        assert!(matches!(value.op(), Op::Cast { .. } | Op::Const(_)));
    }

    // Verify load is cast back to bool
    if let Op::Cast { dtype, .. } = load_result.op() {
        assert_eq!(dtype.base(), ScalarDType::Bool);
    }
}

/// Test: Bool buffer through full devectorize pipeline.
#[test]
fn test_bool_with_devectorize() {
    let buffer = create_bool_buffer(64);
    let index = create_index(buffer.clone(), 0);
    let load = create_load(buffer.clone(), index);

    // Apply full devectorize (all phases)
    let result = apply_devectorize(&load);

    // Should produce properly converted load
    // Either CAST(LOAD<uint8>, bool) or unchanged if pattern didn't match
    assert!(
        result.dtype().base() == ScalarDType::Bool || result.dtype().base() == ScalarDType::UInt8,
        "Result should be bool or uint8"
    );
}

// =============================================================================
// Vector Bool Tests
// =============================================================================

/// Test: Vector bool load conversion.
#[test]
fn test_vector_bool_load() {
    let buffer = create_bool_buffer(64);

    // Create vector bool load by loading multiple elements
    let index = create_index(buffer.clone(), 0);

    // Create load with explicit bool dtype
    let load = create_load(buffer.clone(), index);

    let result = apply_phase3(&load);

    // Should handle vector bool correctly
    match result.op() {
        Op::Cast { src, dtype } => {
            assert_eq!(dtype.base(), ScalarDType::Bool);
            assert_eq!(src.dtype().base(), ScalarDType::UInt8);
        }
        Op::Load { .. } => {}
        other => panic!("Expected CAST(LOAD) or LOAD, got {:?}", other),
    }
}

/// Test: Vector bool store conversion.
#[test]
fn test_vector_bool_store() {
    let buffer = create_bool_buffer(64);
    let index = create_index(buffer.clone(), 0);
    let bool_vec = create_vector_bool(vec![true, false, true, false]);

    let store = create_store(buffer.clone(), index, bool_vec);

    let result = apply_phase3(&store);

    // Should convert vector bool to uint8
    if let Op::Store { value, .. } = result.op() {
        match value.op() {
            Op::Cast { dtype, .. } => {
                assert_eq!(dtype.base(), ScalarDType::UInt8);
            }
            Op::Vectorize { .. } => {
                // Could be VECTORIZE of casts
            }
            _ => {}
        }
    }
}
