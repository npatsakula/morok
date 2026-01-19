//! Test helpers for devectorize.rs tests.
//!
//! Provides builders for creating test UOps and assertion helpers.
//! Mirrors Tinygrad's test patterns for memory access operations.

use std::sync::Arc;

use morok_dtype::{AddrSpace, DType, ScalarDType};
use morok_ir::types::ConstValue;
use morok_ir::{Op, UOp};
use smallvec::SmallVec;

use crate::devectorize::{
    bool_storage_patterns, correct_load_store_patterns, devectorize, load_store_folding_patterns,
    load_store_indexing_patterns, no_vectorized_alu, pm_render,
};
use crate::rewrite::graph_rewrite_bottom_up;

// =============================================================================
// Phase Application Helpers
// =============================================================================

/// Apply full devectorize pass to a UOp.
///
/// Now uses single-pass rewriting (aligned with Tinygrad), followed by pm_render
/// to convert CAT to VECTORIZE for rendering.
pub fn apply_devectorize(uop: &Arc<UOp>) -> Arc<UOp> {
    let devectorized = devectorize(uop);
    // Also run pm_render to convert CAT to VECTORIZE (required for codegen)
    graph_rewrite_bottom_up(&pm_render(), devectorized, &mut ())
}

/// Apply load_store_folding patterns only.
///
/// Includes: expand_index, GEP movement, PTRCAT distribution.
pub fn apply_load_store_folding(uop: &Arc<UOp>) -> Arc<UOp> {
    let patterns = load_store_folding_patterns();
    graph_rewrite_bottom_up(&patterns, uop.clone(), &mut ())
}

/// Apply correct_load_store patterns only.
///
/// Includes: split_load, split_store (CAST(INDEX) patterns).
pub fn apply_correct_load_store(uop: &Arc<UOp>) -> Arc<UOp> {
    let patterns = correct_load_store_patterns();
    graph_rewrite_bottom_up(&patterns, uop.clone(), &mut ())
}

/// Apply bool storage patterns only.
///
/// Converts bool LOAD/STORE to uint8.
pub fn apply_bool_storage(uop: &Arc<UOp>) -> Arc<UOp> {
    let phase3 = bool_storage_patterns();
    graph_rewrite_bottom_up(&phase3, uop.clone(), &mut ())
}

/// Apply pm_render patterns (post-devectorize rendering).
///
/// Includes: CAT→VECTORIZE, multi-index GEP→VECTORIZE, unwrap single-element.
pub fn apply_pm_render(uop: &Arc<UOp>) -> Arc<UOp> {
    let patterns = pm_render();
    graph_rewrite_bottom_up(&patterns, uop.clone(), &mut ())
}

/// Apply ALU devectorization patterns.
pub fn apply_no_vectorized_alu(uop: &Arc<UOp>) -> Arc<UOp> {
    let patterns = no_vectorized_alu();
    graph_rewrite_bottom_up(&patterns, uop.clone(), &mut ())
}

/// Apply pm_render patterns for VECTORIZE normalization.
///
/// (Legacy name for compatibility - now uses pm_render)
pub fn apply_vectorize_normalize(uop: &Arc<UOp>) -> Arc<UOp> {
    apply_pm_render(uop)
}

/// Apply load_store_indexing patterns (gate dropping).
pub fn apply_load_store_indexing(uop: &Arc<UOp>) -> Arc<UOp> {
    let patterns = load_store_indexing_patterns();
    graph_rewrite_bottom_up(&patterns, uop.clone(), &mut ())
}

/// Apply cast_after pattern.
pub fn apply_cast_after(uop: &Arc<UOp>) -> Arc<UOp> {
    use crate::devectorize::devectorize_patterns;
    let patterns = devectorize_patterns();
    graph_rewrite_bottom_up(&patterns, uop.clone(), &mut ())
}

// =============================================================================
// Legacy Phase Names (Backward Compatibility)
// =============================================================================

/// Legacy name for load_store_folding patterns (Phase 1).
///
/// Includes expand_index patterns.
pub fn apply_phase1(uop: &Arc<UOp>) -> Arc<UOp> {
    apply_load_store_folding(uop)
}

/// Legacy name for load_store_folding + correct_load_store + pm_render patterns (Phase 2).
///
/// Includes: GEP movement, PTRCAT distribution, split patterns, and GEP/CAT normalization.
pub fn apply_phase2(uop: &Arc<UOp>) -> Arc<UOp> {
    let patterns = load_store_folding_patterns() + correct_load_store_patterns() + pm_render();
    graph_rewrite_bottom_up(&patterns, uop.clone(), &mut ())
}

/// Legacy name for bool_storage patterns (Phase 3).
pub fn apply_phase3(uop: &Arc<UOp>) -> Arc<UOp> {
    apply_bool_storage(uop)
}

/// Legacy name for pm_render patterns.
///
/// (Was gep_ptrcat_patterns)
pub fn apply_gep_ptrcat_patterns(uop: &Arc<UOp>) -> Arc<UOp> {
    apply_pm_render(uop)
}

// =============================================================================
// Buffer Builders
// =============================================================================

/// Create a global buffer with float32 element type.
///
/// Returns a BUFFER UOp with Ptr dtype pointing to float32 data.
pub fn create_buffer(size: usize) -> Arc<UOp> {
    create_buffer_typed(size, ScalarDType::Float32)
}

/// Create a global buffer with specified element type.
pub fn create_buffer_typed(size: usize, scalar: ScalarDType) -> Arc<UOp> {
    let dtype = DType::Scalar(scalar).ptr(Some(size), AddrSpace::Global);
    UOp::new_buffer(morok_dtype::DeviceSpec::Cpu, size, dtype)
}

/// Create a local (shared) memory buffer.
pub fn create_buffer_local(size: usize, scalar: ScalarDType) -> Arc<UOp> {
    let dtype = DType::Scalar(scalar).ptr(Some(size), AddrSpace::Local);
    UOp::new_buffer(morok_dtype::DeviceSpec::Cpu, size, dtype)
}

/// Create a bool buffer.
pub fn create_bool_buffer(size: usize) -> Arc<UOp> {
    create_buffer_typed(size, ScalarDType::Bool)
}

// =============================================================================
// Index Builders
// =============================================================================

/// Create a scalar INDEX operation.
///
/// INDEX(buffer, [idx]) with scalar index.
pub fn create_index(buffer: Arc<UOp>, idx: i64) -> Arc<UOp> {
    let idx_uop = UOp::const_(DType::Index, ConstValue::Int(idx));
    UOp::index(buffer, vec![idx_uop]).expect("index creation should succeed")
}

/// Create a vector INDEX with iota pattern: [0, 1, 2, ..., count-1].
///
/// INDEX(buffer, VECTORIZE([0, 1, 2, ..., count-1]))
pub fn create_vector_index_iota(buffer: Arc<UOp>, count: usize) -> Arc<UOp> {
    let indices: SmallVec<[Arc<UOp>; 4]> =
        (0..count).map(|i| UOp::const_(DType::Index, ConstValue::Int(i as i64))).collect();
    let vec_idx = UOp::vectorize(indices);
    let idx_dtype = buffer.dtype().base();
    UOp::new(Op::Index { buffer, indices: smallvec::smallvec![vec_idx], gate: None }, DType::Scalar(idx_dtype))
}

/// Create a vector INDEX with offset: [offset, offset+1, offset+2, ..., offset+count-1].
pub fn create_vector_index_offset(buffer: Arc<UOp>, count: usize, offset: i64) -> Arc<UOp> {
    let indices: SmallVec<[Arc<UOp>; 4]> =
        (0..count).map(|i| UOp::const_(DType::Index, ConstValue::Int(offset + i as i64))).collect();
    let vec_idx = UOp::vectorize(indices);
    let idx_dtype = buffer.dtype().base();
    UOp::new(Op::Index { buffer, indices: smallvec::smallvec![vec_idx], gate: None }, DType::Scalar(idx_dtype))
}

/// Create a vector INDEX with scaled pattern: [0*scale, 1*scale, 2*scale, ..., (count-1)*scale].
///
/// This creates strided access patterns.
pub fn create_vector_index_scaled(buffer: Arc<UOp>, count: usize, scale: i64) -> Arc<UOp> {
    let indices: SmallVec<[Arc<UOp>; 4]> =
        (0..count).map(|i| UOp::const_(DType::Index, ConstValue::Int(i as i64 * scale))).collect();
    let vec_idx = UOp::vectorize(indices);
    let idx_dtype = buffer.dtype().base();
    UOp::new(Op::Index { buffer, indices: smallvec::smallvec![vec_idx], gate: None }, DType::Scalar(idx_dtype))
}

/// Create a vector INDEX with explicit values.
pub fn create_vector_index_values(buffer: Arc<UOp>, values: Vec<i64>) -> Arc<UOp> {
    let indices: SmallVec<[Arc<UOp>; 4]> =
        values.iter().map(|&v| UOp::const_(DType::Index, ConstValue::Int(v))).collect();
    let vec_idx = UOp::vectorize(indices);
    let idx_dtype = buffer.dtype().base();
    UOp::new(Op::Index { buffer, indices: smallvec::smallvec![vec_idx], gate: None }, DType::Scalar(idx_dtype))
}

/// Create a gated vector INDEX.
pub fn create_vector_index_gated(buffer: Arc<UOp>, count: usize, gate: Arc<UOp>) -> Arc<UOp> {
    let indices: SmallVec<[Arc<UOp>; 4]> =
        (0..count).map(|i| UOp::const_(DType::Index, ConstValue::Int(i as i64))).collect();
    let vec_idx = UOp::vectorize(indices);
    let idx_dtype = buffer.dtype().base();
    UOp::new(Op::Index { buffer, indices: smallvec::smallvec![vec_idx], gate: Some(gate) }, DType::Scalar(idx_dtype))
}

/// Create an INDEX with symbolic root + offset pattern.
///
/// INDEX(buffer, [range_var * scale + offset])
/// Used for testing root extraction and grouping.
pub fn create_index_with_range(buffer: Arc<UOp>, axis_id: usize, bound: i64, scale: i64, offset: i64) -> Arc<UOp> {
    use morok_ir::{AxisId, AxisType, BinaryOp};

    let range = UOp::new(
        Op::Range {
            end: UOp::const_(DType::Index, ConstValue::Int(bound)),
            axis_id: AxisId::Renumbered(axis_id),
            axis_type: AxisType::Loop,
        },
        DType::Index,
    );

    // range * scale + offset
    let scaled = if scale == 1 {
        range
    } else {
        UOp::new(Op::Binary(BinaryOp::Mul, range, UOp::const_(DType::Index, ConstValue::Int(scale))), DType::Index)
    };

    let idx = if offset == 0 {
        scaled
    } else {
        UOp::new(Op::Binary(BinaryOp::Add, scaled, UOp::const_(DType::Index, ConstValue::Int(offset))), DType::Index)
    };

    UOp::index(buffer, vec![idx]).expect("index creation should succeed")
}

// =============================================================================
// Load/Store Builders
// =============================================================================

/// Create a LOAD operation.
pub fn create_load(buffer: Arc<UOp>, index: Arc<UOp>) -> Arc<UOp> {
    UOp::load().buffer(buffer).index(index).call()
}

/// Create a STORE operation.
pub fn create_store(buffer: Arc<UOp>, index: Arc<UOp>, value: Arc<UOp>) -> Arc<UOp> {
    UOp::store(buffer, index, value)
}

/// Create a vector LOAD with iota index.
pub fn create_vector_load_iota(buffer: Arc<UOp>, count: usize) -> Arc<UOp> {
    let index = create_vector_index_iota(buffer.clone(), count);
    UOp::load().buffer(buffer).index(index).call()
}

/// Create a vector STORE with iota index.
pub fn create_vector_store_iota(buffer: Arc<UOp>, count: usize, value: Arc<UOp>) -> Arc<UOp> {
    let index = create_vector_index_iota(buffer.clone(), count);
    UOp::store(buffer, index, value)
}

// =============================================================================
// Value Builders
// =============================================================================

/// Create a scalar float constant.
pub fn create_float_const(value: f64) -> Arc<UOp> {
    UOp::const_(DType::Float32, ConstValue::Float(value))
}

/// Create a scalar int constant.
pub fn create_int_const(value: i64) -> Arc<UOp> {
    UOp::const_(DType::Int64, ConstValue::Int(value))
}

/// Create a scalar bool constant.
pub fn create_bool_const(value: bool) -> Arc<UOp> {
    UOp::const_(DType::Bool, ConstValue::Bool(value))
}

/// Create a vector float constant with iota pattern.
pub fn create_vector_float_iota(count: usize) -> Arc<UOp> {
    let elements: SmallVec<[Arc<UOp>; 4]> =
        (0..count).map(|i| UOp::const_(DType::Float32, ConstValue::Float(i as f64))).collect();
    UOp::vectorize(elements)
}

/// Create a vector int constant with iota pattern.
pub fn create_vector_int_iota(count: usize) -> Arc<UOp> {
    let elements: SmallVec<[Arc<UOp>; 4]> =
        (0..count).map(|i| UOp::const_(DType::Int64, ConstValue::Int(i as i64))).collect();
    UOp::vectorize(elements)
}

/// Create a vector constant from explicit float values.
pub fn create_vector_float_values(values: Vec<f64>) -> Arc<UOp> {
    let elements: SmallVec<[Arc<UOp>; 4]> =
        values.into_iter().map(|v| UOp::const_(DType::Float32, ConstValue::Float(v))).collect();
    UOp::vectorize(elements)
}

/// Create a vector constant from explicit int values.
pub fn create_vector_int_values(values: Vec<i64>) -> Arc<UOp> {
    let elements: SmallVec<[Arc<UOp>; 4]> =
        values.into_iter().map(|v| UOp::const_(DType::Int64, ConstValue::Int(v))).collect();
    UOp::vectorize(elements)
}

/// Create a vector bool constant.
pub fn create_vector_bool(values: Vec<bool>) -> Arc<UOp> {
    let elements: SmallVec<[Arc<UOp>; 4]> =
        values.into_iter().map(|v| UOp::const_(DType::Bool, ConstValue::Bool(v))).collect();
    UOp::vectorize(elements)
}

// =============================================================================
// Assertion Helpers
// =============================================================================

/// Assert that a UOp is a PTRCAT with expected source count.
pub fn assert_is_ptrcat(uop: &Arc<UOp>, expected_count: usize) {
    match uop.op() {
        Op::PtrCat { sources } => {
            assert_eq!(
                sources.len(),
                expected_count,
                "PTRCAT source count mismatch: expected {}, got {}",
                expected_count,
                sources.len()
            );
        }
        other => panic!("Expected PTRCAT, got {:?}", other),
    }
}

/// Assert that a UOp is a CAT with expected source count.
pub fn assert_is_cat(uop: &Arc<UOp>, expected_count: usize) {
    match uop.op() {
        Op::Cat { sources } => {
            assert_eq!(
                sources.len(),
                expected_count,
                "CAT source count mismatch: expected {}, got {}",
                expected_count,
                sources.len()
            );
        }
        other => panic!("Expected CAT, got {:?}", other),
    }
}

/// Assert that a UOp is a VECTORIZE with expected element count.
pub fn assert_is_vectorize(uop: &Arc<UOp>, expected_count: usize) {
    match uop.op() {
        Op::Vectorize { elements } => {
            assert_eq!(
                elements.len(),
                expected_count,
                "VECTORIZE element count mismatch: expected {}, got {}",
                expected_count,
                elements.len()
            );
        }
        other => panic!("Expected VECTORIZE, got {:?}", other),
    }
}

/// Assert that a UOp has expected vcount (vector width).
pub fn assert_vcount(uop: &Arc<UOp>, expected: usize) {
    assert_eq!(uop.dtype().vcount(), expected, "vcount mismatch: expected {}, got {}", expected, uop.dtype().vcount());
}

/// Assert dtype matches expected.
pub fn assert_dtype(uop: &Arc<UOp>, expected: DType) {
    assert_eq!(uop.dtype(), expected, "dtype mismatch");
}

/// Assert base scalar dtype matches expected.
pub fn assert_base_dtype(uop: &Arc<UOp>, expected: ScalarDType) {
    assert_eq!(uop.dtype().base(), expected, "base dtype mismatch");
}

/// Assert that a UOp is a LOAD.
pub fn assert_is_load(uop: &Arc<UOp>) {
    assert!(matches!(uop.op(), Op::Load { .. }), "Expected LOAD, got {:?}", uop.op());
}

/// Assert that a UOp is a STORE.
pub fn assert_is_store(uop: &Arc<UOp>) {
    assert!(matches!(uop.op(), Op::Store { .. }), "Expected STORE, got {:?}", uop.op());
}

/// Assert that a UOp is a GEP with expected indices.
pub fn assert_is_gep(uop: &Arc<UOp>, expected_indices: &[usize]) {
    match uop.op() {
        Op::Gep { indices, .. } => {
            assert_eq!(
                indices, expected_indices,
                "GEP indices mismatch: expected {:?}, got {:?}",
                expected_indices, indices
            );
        }
        other => panic!("Expected GEP, got {:?}", other),
    }
}

/// Assert that a UOp is a CAST.
pub fn assert_is_cast(uop: &Arc<UOp>) {
    assert!(matches!(uop.op(), Op::Cast { .. }), "Expected CAST, got {:?}", uop.op());
}

/// Assert that a UOp is a GROUP with expected source count.
pub fn assert_is_group(uop: &Arc<UOp>, expected_count: usize) {
    match uop.op() {
        Op::Group { sources } => {
            assert_eq!(
                sources.len(),
                expected_count,
                "GROUP source count mismatch: expected {}, got {}",
                expected_count,
                sources.len()
            );
        }
        other => panic!("Expected GROUP, got {:?}", other),
    }
}

/// Assert that a UOp is an INDEX.
pub fn assert_is_index(uop: &Arc<UOp>) {
    assert!(matches!(uop.op(), Op::Index { .. }), "Expected INDEX, got {:?}", uop.op());
}

// =============================================================================
// Op Counting Helpers
// =============================================================================

/// Count operations matching a predicate in the UOp tree.
pub fn count_ops<F>(uop: &Arc<UOp>, predicate: F) -> usize
where
    F: Fn(&Arc<UOp>) -> bool,
{
    let mut count = 0;
    count_ops_recursive(uop, &predicate, &mut count);
    count
}

fn count_ops_recursive<F>(uop: &Arc<UOp>, predicate: &F, count: &mut usize)
where
    F: Fn(&Arc<UOp>) -> bool,
{
    if predicate(uop) {
        *count += 1;
    }
    for child in uop.op().children() {
        count_ops_recursive(child, predicate, count);
    }
}

/// Count LOAD operations in the tree.
pub fn count_loads(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Load { .. }))
}

/// Count STORE operations in the tree.
pub fn count_stores(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Store { .. }))
}

/// Count INDEX operations in the tree.
pub fn count_indices(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Index { .. }))
}

/// Count PTRCAT operations in the tree.
pub fn count_ptrcats(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::PtrCat { .. }))
}

/// Count CAT operations in the tree.
pub fn count_cats(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Cat { .. }))
}

/// Count VECTORIZE operations in the tree.
pub fn count_vectorizes(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Vectorize { .. }))
}

/// Count GEP operations in the tree.
pub fn count_geps(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Gep { .. }))
}

/// Count CAST operations in the tree.
pub fn count_casts(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Cast { .. }))
}

// =============================================================================
// Unwrap Helpers
// =============================================================================

/// Unwrap PTRCAT and return sources.
pub fn unwrap_ptrcat(uop: &Arc<UOp>) -> SmallVec<[Arc<UOp>; 4]> {
    match uop.op() {
        Op::PtrCat { sources } => sources.clone(),
        other => panic!("Expected PTRCAT, got {:?}", other),
    }
}

/// Unwrap CAT and return sources.
pub fn unwrap_cat(uop: &Arc<UOp>) -> SmallVec<[Arc<UOp>; 4]> {
    match uop.op() {
        Op::Cat { sources } => sources.clone(),
        other => panic!("Expected CAT, got {:?}", other),
    }
}

/// Unwrap VECTORIZE and return elements.
pub fn unwrap_vectorize(uop: &Arc<UOp>) -> SmallVec<[Arc<UOp>; 4]> {
    match uop.op() {
        Op::Vectorize { elements } => elements.clone(),
        other => panic!("Expected VECTORIZE, got {:?}", other),
    }
}

/// Unwrap LOAD and return (buffer, index).
pub fn unwrap_load(uop: &Arc<UOp>) -> (Arc<UOp>, Arc<UOp>) {
    match uop.op() {
        Op::Load { buffer, index } => (buffer.clone(), index.clone()),
        other => panic!("Expected LOAD, got {:?}", other),
    }
}

/// Unwrap STORE and return (buffer, index, value).
pub fn unwrap_store(uop: &Arc<UOp>) -> (Arc<UOp>, Arc<UOp>, Arc<UOp>) {
    match uop.op() {
        Op::Store { buffer, index, value, .. } => (buffer.clone(), index.clone(), value.clone()),
        other => panic!("Expected STORE, got {:?}", other),
    }
}

/// Unwrap GEP and return (vector, indices).
pub fn unwrap_gep(uop: &Arc<UOp>) -> (Arc<UOp>, Vec<usize>) {
    match uop.op() {
        Op::Gep { vector, indices } => (vector.clone(), indices.clone()),
        other => panic!("Expected GEP, got {:?}", other),
    }
}

/// Unwrap CAST and return (src, dtype).
pub fn unwrap_cast(uop: &Arc<UOp>) -> (Arc<UOp>, DType) {
    match uop.op() {
        Op::Cast { src, dtype } => (src.clone(), dtype.clone()),
        other => panic!("Expected CAST, got {:?}", other),
    }
}

/// Unwrap INDEX and return (buffer, indices, gate).
pub fn unwrap_index(uop: &Arc<UOp>) -> (Arc<UOp>, SmallVec<[Arc<UOp>; 4]>, Option<Arc<UOp>>) {
    match uop.op() {
        Op::Index { buffer, indices, gate } => (buffer.clone(), indices.clone(), gate.clone()),
        other => panic!("Expected INDEX, got {:?}", other),
    }
}

/// Unwrap GROUP and return sources.
pub fn unwrap_group(uop: &Arc<UOp>) -> Vec<Arc<UOp>> {
    match uop.op() {
        Op::Group { sources } => sources.to_vec(),
        other => panic!("Expected GROUP, got {:?}", other),
    }
}
