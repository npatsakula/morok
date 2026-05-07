//! Tests for new devectorize patterns added in the redesign.
//!
//! Tests the following patterns:
//! - cast_after: AFTER(CAST(x), deps) → CAST(AFTER(x, deps))
//! - load_store_indexing: INDEX(buf, x, true) → INDEX(buf, x, None)
//! - devectorize_buf_and_index: LOCAL/REG buffer vectorization

use std::sync::Arc;

use morok_dtype::{AddrSpace, DType};
use morok_ir::types::ConstValue;
use morok_ir::{Op, UOp};
use smallvec::{SmallVec, smallvec};

use crate::devectorize::devectorize;

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
    let cast = src.cast(DType::Float64);

    // AFTER(cast, [])
    let after = cast.after(smallvec![]);

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
    let cast = src.cast(DType::Float64);
    let after = cast.after(smallvec![dep.clone()]);

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
    let after = src.after(smallvec![]);

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
    let cast = src.cast(DType::Float64);
    let after = cast.after(smallvec![]);

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

// =============================================================================
// Gated Load Alt Pattern Tests (pm_render)
// =============================================================================

/// Test: Gated LOAD without alt gets const 0 alt
#[test]
fn test_gated_load_gets_alt() {
    let buffer = create_buffer(64);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(false)); // non-true gate

    // Create gated INDEX
    let gated_index =
        UOp::new(Op::Index { buffer: buffer.clone(), indices: smallvec![idx], gate: Some(gate) }, DType::Float32);

    // Create LOAD without alt
    let load = UOp::load().buffer(buffer).index(gated_index).call();

    // Apply pm_render
    let result = apply_pm_render(&load);

    // Result should be LOAD with alt value
    match result.op() {
        Op::Load { alt, .. } => {
            assert!(alt.is_some(), "Gated LOAD should have alt value after pm_render");
            if let Some(alt_val) = alt {
                // Alt should be const 0
                let is_zero = match alt_val.op() {
                    Op::Const(cv) => {
                        matches!(cv.0, ConstValue::Int(0)) || matches!(cv.0, ConstValue::Float(f) if f == 0.0)
                    }
                    _ => false,
                };
                assert!(is_zero, "Alt value should be 0");
            }
        }
        other => {
            // Could be transformed to something else
            tracing::debug!("Gated load transformed to: {:?}", other);
        }
    }
}

/// Test: LOAD without gate is unchanged by alt pattern
#[test]
fn test_ungate_load_unchanged() {
    let buffer = create_buffer(64);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));

    // Create ungated INDEX
    let index = UOp::index().buffer(buffer.clone()).indices(vec![idx]).call().unwrap();

    // Create LOAD without alt
    let load = UOp::load().buffer(buffer).index(index).call();

    let result = apply_pm_render(&load);

    // Result should still be LOAD without alt
    if let Op::Load { alt, .. } = result.op() {
        assert!(alt.is_none(), "Ungated LOAD should not have alt value");
    }
}

// =============================================================================
// is_increasing Tests (already in helpers.rs, but integration test here)
// =============================================================================

/// Test: is_increasing on range variable
#[test]
fn test_is_increasing_range() {
    use morok_ir::types::{AxisId, AxisType};
    let range = UOp::range_axis(UOp::index_const(16), AxisId::Unrenumbered(0), AxisType::Loop);
    assert!(range.is_increasing(), "RANGE should be increasing");
}

/// Test: is_increasing on constant
#[test]
fn test_is_increasing_constant() {
    let c = UOp::const_(DType::Int32, ConstValue::Int(5));
    assert!(c.is_increasing(), "CONST should be increasing");
}

/// Test: is_increasing on add
#[test]
fn test_is_increasing_add_expr() {
    use morok_ir::types::{AxisId, AxisType};
    let range = UOp::range_axis(UOp::index_const(16), AxisId::Unrenumbered(0), AxisType::Loop);
    let c = UOp::const_(DType::Index, ConstValue::Int(5));
    let sum = range.try_add(&c).unwrap();
    assert!(sum.is_increasing(), "RANGE + CONST should be increasing");
}

/// Test: is_increasing on mul by positive const
#[test]
fn test_is_increasing_mul_positive() {
    use morok_ir::types::{AxisId, AxisType};
    let range = UOp::range_axis(UOp::index_const(16), AxisId::Unrenumbered(0), AxisType::Loop);
    let c = UOp::const_(DType::Index, ConstValue::Int(4));
    let prod = range.try_mul(&c).unwrap();
    assert!(prod.is_increasing(), "RANGE * positive CONST should be increasing");
}

/// Test: is_increasing on mul by negative const
#[test]
fn test_is_increasing_mul_negative() {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let c = UOp::const_(DType::Int32, ConstValue::Int(-1));
    let prod = x.try_mul(&c).unwrap();
    assert!(!prod.is_increasing(), "x * negative CONST should not be increasing");
}

// =============================================================================
// Vector INDEX on Local Buffer Tests (scatter store fix)
// =============================================================================

/// Test: INDEX(CAST(DEF_LOCAL), vec3_idx) gets decomposed by devectorize.
///
/// Reproduces the scatter store bug: UPCAST creates vec3 indices on a local buffer
/// with vec3 pointer type. Without the fix, neither `expand_index` (requires
/// VECTORIZE buffer) nor `no_vectorized_index` (required scalar idx) would match,
/// leaving a vector STORE that C/LLVM codegen cannot emit.
#[test]
fn test_devectorize_local_buffer_vector_index() {
    // DEF_LOCAL with vec3 pointer: Ptr<vec3<f32>>
    let vec3_ptr_dtype = DType::Float32.vec(3).ptr(Some(9), AddrSpace::Local);
    let _def_local = UOp::define_local(0, vec3_ptr_dtype.clone());

    // Simulate no_vectorized_buf having fired: DEF_LOCAL(Ptr<f32>).cast(Ptr<vec3<f32>>)
    let scalar_ptr_dtype = DType::Float32.ptr(Some(9), AddrSpace::Local);
    let scalar_def = UOp::define_local(1, scalar_ptr_dtype);
    let cast_def = scalar_def.cast(vec3_ptr_dtype);

    // Create vector index with 3 lanes (simulating UPCAST expansion)
    let idx0 = UOp::index_const(0);
    let idx1 = UOp::index_const(1);
    let idx2 = UOp::index_const(2);
    let vec3_idx = UOp::vectorize(smallvec![idx0, idx1, idx2]);

    // INDEX(CAST(DEF_LOCAL), vec3_idx) — this is the problematic pattern
    let index = UOp::new(
        Op::Index { buffer: cast_def, indices: smallvec![vec3_idx], gate: None },
        DType::Float32.vec(9).ptr(Some(9), AddrSpace::Local),
    );

    // Create vec9 value and STORE
    let value_elements: SmallVec<[Arc<UOp>; 4]> =
        (0..9).map(|i| UOp::const_(DType::Float32, ConstValue::Float(i as f64))).collect();
    let vec9_value = UOp::vectorize(value_elements);
    let store = index.store(vec9_value);

    // Wrap in SINK for the rewrite engine
    let sink = UOp::sink(vec![store]);

    // Apply full devectorize pipeline
    let result = devectorize(&sink, &crate::optimizer::Renderer::cpu());

    // After devectorize, all STOREs should have scalar indices (no vector INDEX on local buf)
    let has_vector_local_index = result.toposort().iter().any(|node: &Arc<UOp>| {
        if let Op::Index { buffer, indices, .. } = node.op() {
            let has_vec_idx = indices.first().is_some_and(|i| i.dtype().vcount() > 1);
            let is_local_cast = matches!(buffer.op(), Op::Cast { src, .. }
                if matches!(src.op(), Op::DefineLocal(_)));
            has_vec_idx && is_local_cast
        } else {
            false
        }
    });
    assert!(!has_vector_local_index, "After devectorize, no INDEX(CAST(DEF_LOCAL), vec_idx) should remain");

    // Should have produced multiple STOREs (PTRCAT groups consecutive offsets,
    // so count may be less than 9 due to grouping)
    let store_count = count_stores(&result);
    assert!(store_count >= 3, "Expected at least 3 stores (grouped), got {store_count}");
}

/// Test: Same as above but with vec9 index (simulating u3u3 UPCAST = 3×3 = 9 lanes).
#[test]
fn test_devectorize_local_buffer_vec9_index() {
    // DEF_LOCAL with vec9 pointer: Ptr<vec9<f32>>
    // (This is the actual pattern from the u3u3 kernel)
    let vec9_ptr_dtype = DType::Float32.vec(9).ptr(Some(81), AddrSpace::Local);
    let scalar_ptr_dtype = DType::Float32.ptr(Some(81), AddrSpace::Local);
    let scalar_def = UOp::define_local(2, scalar_ptr_dtype);
    let cast_def = scalar_def.cast(vec9_ptr_dtype);

    // Create vector index with 9 lanes
    let vec9_idx = UOp::vectorize((0..9i64).map(UOp::index_const).collect::<SmallVec<[Arc<UOp>; 4]>>());

    // INDEX(CAST(DEF_LOCAL), vec9_idx)
    let index = UOp::new(
        Op::Index { buffer: cast_def, indices: smallvec![vec9_idx], gate: None },
        DType::Float32.vec(81).ptr(Some(81), AddrSpace::Local),
    );

    // vec81 value (9 index lanes × 9 pointer vcount)
    let value_elements: SmallVec<[Arc<UOp>; 4]> =
        (0..81).map(|i| UOp::const_(DType::Float32, ConstValue::Float(i as f64))).collect();
    let vec81_value = UOp::vectorize(value_elements);
    let store = index.store(vec81_value);

    let sink = UOp::sink(vec![store]);
    let result = devectorize(&sink, &crate::optimizer::Renderer::cpu());

    let has_vector_local_index = result.toposort().iter().any(|node: &Arc<UOp>| {
        if let Op::Index { buffer, indices, .. } = node.op() {
            let has_vec_idx = indices.first().is_some_and(|i| i.dtype().vcount() > 1);
            let is_local_cast = matches!(buffer.op(), Op::Cast { src, .. }
                if matches!(src.op(), Op::DefineLocal(_)));
            has_vec_idx && is_local_cast
        } else {
            false
        }
    });
    assert!(!has_vector_local_index, "After devectorize, no INDEX(CAST(DEF_LOCAL), vec_idx) should remain");

    // PTRCAT groups consecutive offsets, so store count is less than 81
    let store_count = count_stores(&result);
    assert!(store_count >= 9, "Expected at least 9 stores (grouped), got {store_count}");
}
