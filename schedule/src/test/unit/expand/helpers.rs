//! Test helpers for expand.rs tests.
//!
//! Provides builders for creating test UOps and assertion helpers.
//! Mirrors Tinygrad's test patterns from TestExpander.

use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{Op, UOp};

use crate::expand::pre_expand;
use crate::rewrite::graph_rewrite_bottom_up;

/// Apply expander rewrite to a UOp (main entry point for tests).
///
/// Uses pre_expand which runs both phases:
/// 1. Range(Unroll/Upcast) → UNROLL conversion
/// 2. Fix REDUCE/STORE + expand ops with UNROLL + CONTRACT
pub fn expander_rewrite(uop: &Arc<UOp>) -> Arc<UOp> {
    pre_expand(uop)
}

/// Apply phase2 expander only (skip Range→UNROLL conversion).
///
/// Useful for testing do_expand, do_contract directly when starting
/// from manually constructed UNROLL/CONTRACT operations.
pub fn phase2_only(uop: &Arc<UOp>) -> Arc<UOp> {
    let phase2 = crate::expand::phase2_expand();
    graph_rewrite_bottom_up(&phase2, uop.clone(), &mut ())
}

// =============================================================================
// UOp Builders (mirror Tinygrad patterns)
// =============================================================================

/// Create UNROLL with iota pattern: UNROLL(VCONST([0,1,...,N-1]), [(axis_id, N)]).
///
/// This is the standard pattern from Tinygrad's Range(Unroll) conversion.
/// IMPORTANT: UNROLL dtype is SCALAR (matching Tinygrad), not the vec dtype from VCONST.
pub fn create_unroll_iota(axis_id: usize, count: usize) -> Arc<UOp> {
    let values: Vec<ConstValue> = (0..count as i64).map(ConstValue::Int).collect();
    let vconst = UOp::vconst(values);
    // Use scalar dtype for UNROLL wrapper (Tinygrad: dtypes.int, not dtypes.int.vec(N))
    vconst.unroll_with_dtype(vec![(axis_id, count)], DType::Int64)
}

/// Create UNROLL with scaled iota: UNROLL(VCONST([0*scale,1*scale,...]), [(axis_id, N)]).
///
/// Used to test expansion with different value patterns.
/// IMPORTANT: UNROLL dtype is SCALAR (matching Tinygrad).
pub fn create_unroll_scaled(axis_id: usize, count: usize, scale: i64) -> Arc<UOp> {
    let values: Vec<ConstValue> = (0..count as i64).map(|i| ConstValue::Int(i * scale)).collect();
    let vconst = UOp::vconst(values);
    vconst.unroll_with_dtype(vec![(axis_id, count)], DType::Int64)
}

/// Create UNROLL with explicit values.
/// IMPORTANT: UNROLL dtype is SCALAR (matching Tinygrad).
pub fn create_unroll_values(axis_id: usize, values: Vec<i64>) -> Arc<UOp> {
    let const_values: Vec<ConstValue> = values.into_iter().map(ConstValue::Int).collect();
    let count = const_values.len();
    let vconst = UOp::vconst(const_values);
    vconst.unroll_with_dtype(vec![(axis_id, count)], DType::Int64)
}

/// Create UNROLL with multiple axes.
///
/// Creates VCONST with product(axis_sizes) elements, numbered 0..N.
/// Matches Tinygrad's multi-axis UNROLL pattern.
/// IMPORTANT: UNROLL dtype is SCALAR (matching Tinygrad).
pub fn create_unroll_multi_axis(axes: Vec<(usize, usize)>) -> Arc<UOp> {
    let total_count: usize = axes.iter().map(|(_, sz)| sz).product();
    let values: Vec<ConstValue> = (0..total_count as i64).map(ConstValue::Int).collect();
    let vconst = UOp::vconst(values);
    vconst.unroll_with_dtype(axes, DType::Int64)
}

/// Create UNROLL with multiple axes and custom dtype.
pub fn create_unroll_multi_axis_with_dtype(axes: Vec<(usize, usize)>, dtype: DType) -> Arc<UOp> {
    let total_count: usize = axes.iter().map(|(_, sz)| sz).product();
    let values: Vec<ConstValue> = (0..total_count as i64).map(ConstValue::Int).collect();
    let vconst = UOp::vconst(values);
    vconst.unroll_with_dtype(axes, dtype)
}

/// Create a simple integer VCONST.
pub fn create_vconst_int(values: Vec<i64>) -> Arc<UOp> {
    let const_values: Vec<ConstValue> = values.into_iter().map(ConstValue::Int).collect();
    UOp::vconst(const_values)
}

/// Create a CONTRACT operation.
pub fn create_contract(src: Arc<UOp>, axes: Vec<(usize, usize)>) -> Arc<UOp> {
    src.contract(axes)
}

/// Create a CONTRACT operation with explicit void dtype (for STORE-like operations).
pub fn create_contract_void(src: Arc<UOp>, axes: Vec<(usize, usize)>) -> Arc<UOp> {
    UOp::new(morok_ir::Op::Contract { src: src.clone(), upcast_ranges: axes }, DType::Void)
}

// =============================================================================
// Assertion Helpers
// =============================================================================

/// Assert that a UOp is an UNROLL with the expected axes.
///
/// # Panics
/// Panics if the UOp is not an UNROLL or axes don't match.
pub fn assert_is_unroll(uop: &Arc<UOp>, expected_axes: &[(usize, usize)]) {
    match uop.op() {
        Op::Unroll { unroll_axes, .. } => {
            assert_eq!(
                unroll_axes.as_slice(),
                expected_axes,
                "Expected UNROLL axes {:?}, got {:?}",
                expected_axes,
                unroll_axes
            );
        }
        other => panic!("Expected UNROLL, got {:?}", other),
    }
}

/// Assert that a UOp is an UNROLL and return its source and axes.
///
/// # Panics
/// Panics if the UOp is not an UNROLL.
pub fn unwrap_unroll(uop: &Arc<UOp>) -> (Arc<UOp>, Vec<(usize, usize)>) {
    match uop.op() {
        Op::Unroll { src, unroll_axes } => (src.clone(), unroll_axes.clone()),
        other => panic!("Expected UNROLL, got {:?}", other),
    }
}

/// Assert that a UOp is a VCONST with the expected integer values.
///
/// # Panics
/// Panics if the UOp is not a VCONST or values don't match.
pub fn assert_is_vconst(uop: &Arc<UOp>, expected_values: &[i64]) {
    match uop.op() {
        Op::VConst { values } => {
            let actual: Vec<i64> = values
                .iter()
                .map(|cv| match cv {
                    ConstValue::Int(i) => *i,
                    other => panic!("Expected Int, got {:?}", other),
                })
                .collect();
            assert_eq!(actual, expected_values, "VCONST values mismatch");
        }
        other => panic!("Expected VCONST, got {:?}", other),
    }
}

/// Extract integer values from a VCONST UOp.
///
/// # Panics
/// Panics if the UOp is not a VCONST or contains non-integer values.
pub fn unwrap_vconst(uop: &Arc<UOp>) -> Vec<i64> {
    match uop.op() {
        Op::VConst { values } => values
            .iter()
            .map(|cv| match cv {
                ConstValue::Int(i) => *i,
                ConstValue::UInt(u) => *u as i64,
                other => panic!("Expected Int, got {:?}", other),
            })
            .collect(),
        other => panic!("Expected VCONST, got {:?}", other),
    }
}

/// Unwrap UNROLL and extract VCONST values from its source.
///
/// This is the main pattern for checking expanded results.
pub fn unwrap_unroll_vconst(uop: &Arc<UOp>) -> Vec<i64> {
    let (src, _) = unwrap_unroll(uop);
    unwrap_vconst(&src)
}

/// Assert that a UOp is a GEP with the expected indices.
///
/// # Panics
/// Panics if the UOp is not a GEP or indices don't match.
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

/// Unwrap GEP and return (source, indices).
///
/// # Panics
/// Panics if the UOp is not a GEP.
pub fn unwrap_gep(uop: &Arc<UOp>) -> (Arc<UOp>, Vec<usize>) {
    match uop.op() {
        Op::Gep { vector, indices } => (vector.clone(), indices.clone()),
        other => panic!("Expected GEP, got {:?}", other),
    }
}

/// Assert that a UOp is a VECTORIZE with the expected element count.
///
/// # Panics
/// Panics if the UOp is not a VECTORIZE or element count doesn't match.
pub fn assert_is_vectorize(uop: &Arc<UOp>, expected_count: usize) {
    match uop.op() {
        Op::Vectorize { elements } => {
            assert_eq!(elements.len(), expected_count, "VECTORIZE element count mismatch");
        }
        other => panic!("Expected VECTORIZE, got {:?}", other),
    }
}

/// Assert that a UOp is a CONTRACT with the expected axes.
///
/// # Panics
/// Panics if the UOp is not a CONTRACT or axes don't match.
pub fn assert_is_contract(uop: &Arc<UOp>, expected_axes: &[(usize, usize)]) {
    match uop.op() {
        Op::Contract { upcast_ranges, .. } => {
            assert_eq!(upcast_ranges.as_slice(), expected_axes, "CONTRACT axes mismatch");
        }
        other => panic!("Expected CONTRACT, got {:?}", other),
    }
}

/// Unwrap CONTRACT and return (source, axes).
///
/// # Panics
/// Panics if the UOp is not a CONTRACT.
pub fn unwrap_contract(uop: &Arc<UOp>) -> (Arc<UOp>, Vec<(usize, usize)>) {
    match uop.op() {
        Op::Contract { src, upcast_ranges } => (src.clone(), upcast_ranges.clone()),
        other => panic!("Expected CONTRACT, got {:?}", other),
    }
}

/// Assert the dtype matches expected.
pub fn assert_dtype(uop: &Arc<UOp>, expected: DType) {
    assert_eq!(uop.dtype(), expected, "dtype mismatch");
}

/// Assert the vcount (vector width) of a dtype.
pub fn assert_vcount(uop: &Arc<UOp>, expected: usize) {
    assert_eq!(uop.dtype().vcount(), expected, "vcount mismatch: expected {}, got {}", expected, uop.dtype().vcount());
}

// =============================================================================
// BUFFERIZE Builders
// =============================================================================

use morok_ir::BufferizeOpts;

/// Create a BUFFERIZE operation.
pub fn create_bufferize(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>, opts: BufferizeOpts) -> Arc<UOp> {
    UOp::new(Op::Bufferize { compute, ranges: ranges.into_iter().collect(), opts }, DType::Void)
}

/// Create a BUFFERIZE with global memory target.
pub fn create_bufferize_global(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    use morok_dtype::DeviceSpec;
    create_bufferize(compute, ranges, BufferizeOpts::new(DeviceSpec::Cpu))
}

// =============================================================================
// BUFFERIZE Assertion Helpers
// =============================================================================

/// Assert that a UOp is a BUFFERIZE.
pub fn assert_is_bufferize(uop: &Arc<UOp>) {
    assert!(matches!(uop.op(), Op::Bufferize { .. }), "Expected BUFFERIZE, got {:?}", uop.op());
}

/// Assert that a BUFFERIZE has CONTRACT as its compute source.
pub fn assert_bufferize_compute_is_contract(uop: &Arc<UOp>) {
    match uop.op() {
        Op::Bufferize { compute, .. } => {
            assert!(
                matches!(compute.op(), Op::Contract { .. }),
                "Expected BUFFERIZE.compute to be CONTRACT, got {:?}",
                compute.op()
            );
        }
        other => panic!("Expected BUFFERIZE, got {:?}", other),
    }
}

/// Unwrap BUFFERIZE and return (compute, ranges, opts).
pub fn unwrap_bufferize(uop: &Arc<UOp>) -> (Arc<UOp>, smallvec::SmallVec<[Arc<UOp>; 4]>, BufferizeOpts) {
    match uop.op() {
        Op::Bufferize { compute, ranges, opts } => (compute.clone(), ranges.clone(), opts.clone()),
        other => panic!("Expected BUFFERIZE, got {:?}", other),
    }
}

// =============================================================================
// END Builders
// =============================================================================

/// Create an END operation.
pub fn create_end(computation: Arc<UOp>, ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    computation.end(ranges.into_iter().collect())
}

// =============================================================================
// END Assertion Helpers
// =============================================================================

/// Assert that a UOp is an END.
pub fn assert_is_end(uop: &Arc<UOp>) {
    assert!(matches!(uop.op(), Op::End { .. }), "Expected END, got {:?}", uop.op());
}

/// Assert that an END has CONTRACT as its computation.
pub fn assert_is_end_with_contract(uop: &Arc<UOp>) {
    match uop.op() {
        Op::End { computation, .. } => {
            assert!(
                matches!(computation.op(), Op::Contract { .. }),
                "Expected END.computation to be CONTRACT, got {:?}",
                computation.op()
            );
        }
        other => panic!("Expected END, got {:?}", other),
    }
}

/// Assert that an END has the expected number of ranges.
pub fn assert_end_ranges_count(uop: &Arc<UOp>, expected: usize) {
    match uop.op() {
        Op::End { ranges, .. } => {
            assert_eq!(
                ranges.len(),
                expected,
                "END ranges count mismatch: expected {}, got {}",
                expected,
                ranges.len()
            );
        }
        other => panic!("Expected END, got {:?}", other),
    }
}

/// Unwrap END and return (computation, ranges).
pub fn unwrap_end(uop: &Arc<UOp>) -> (Arc<UOp>, smallvec::SmallVec<[Arc<UOp>; 4]>) {
    match uop.op() {
        Op::End { computation, ranges } => (computation.clone(), ranges.clone()),
        other => panic!("Expected END, got {:?}", other),
    }
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

/// Count BUFFERIZE operations in the tree.
pub fn count_bufferizes(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Bufferize { .. }))
}

/// Count END operations in the tree.
pub fn count_ends(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::End { .. }))
}

/// Count CONTRACT operations in the tree.
pub fn count_contracts(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Contract { .. }))
}

/// Count UNROLL operations in the tree.
pub fn count_unrolls(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Unroll { .. }))
}
