//! Tests for gpudims store masking with ended ranges.
//!
//! These tests validate the fix for Stage 12 bug where `toposort().filter(Range)`
//! was returning ALL ranges in the graph instead of only active (in-scope) ranges.
//! The fix uses `in_scope_ranges()` which correctly excludes ended ranges.

use std::collections::HashSet;
use std::sync::Arc;

use morok_dtype::{AddrSpace, DType};
use morok_ir::types::{AxisId, AxisType};
use morok_ir::{Op, UOp};

/// Helper: Create a LOCAL RANGE with the given end value.
fn create_local_range(end_value: i64, axis_id: usize) -> Arc<UOp> {
    let end = UOp::const_(DType::Index, morok_ir::types::ConstValue::Int(end_value));
    UOp::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Local)
}

/// Helper: Create a GLOBAL buffer (Ptr to Global memory).
fn create_global_buffer(buf_id: usize) -> Arc<UOp> {
    UOp::define_global(buf_id, DType::Float32.ptr(Some(1024), AddrSpace::Global))
}

/// Helper: Create an INDEX into a buffer with given indices.
fn create_index(buffer: Arc<UOp>, indices: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(indices).call().expect("index should succeed")
}

// =============================================================================
// in_scope_ranges Tests (Underlying Mechanism)
// =============================================================================

/// Test: Range is in scope when actively used.
#[test]
fn test_in_scope_ranges_basic() {
    let range = create_local_range(16, 0);
    let value = range.add(&UOp::index_const(1)); // Use range in computation

    #[allow(clippy::mutable_key_type)]
    let in_scope = value.in_scope_ranges();

    // Range should be in scope
    assert!(in_scope.iter().any(|key| key.0.id == range.id), "Range should be in scope: found {:?}", in_scope);
}

/// Test: Range is NOT in scope after END.
///
/// This is the key behavior that the bug fix relies on.
#[test]
fn test_in_scope_ranges_after_end() {
    let range = create_local_range(16, 0);
    let computation = range.add(&UOp::index_const(1));

    // End the range
    let ended = computation.end(smallvec::smallvec![range.clone()]);

    #[allow(clippy::mutable_key_type)]
    let in_scope = ended.in_scope_ranges();

    // Range should NOT be in scope after END
    assert!(
        !in_scope.iter().any(|key| key.0.id == range.id),
        "Range should NOT be in scope after END: found {:?}",
        in_scope
    );
}

/// Test: Multiple ranges, some ended.
///
/// Validates that only non-ended ranges are in scope.
#[test]
fn test_in_scope_ranges_partial_end() {
    let range1 = create_local_range(16, 0);
    let range2 = create_local_range(32, 1);

    // Computation uses both ranges
    let computation = range1.add(&range2);

    // End only range1
    let after_end1 = computation.end(smallvec::smallvec![range1.clone()]);

    // Create another computation that uses range2, with dependency on ended computation
    // Use AFTER to express this sequencing (after_end1 is Void, so we can't use add)
    let another_computation = range2.add(&UOp::index_const(5));
    let final_comp = another_computation.after(smallvec::smallvec![after_end1]);

    #[allow(clippy::mutable_key_type)]
    let in_scope = final_comp.in_scope_ranges();

    // range1 should NOT be in scope (it was ended), range2 should be
    assert!(!in_scope.iter().any(|key| key.0.id == range1.id), "range1 should NOT be in scope");
    assert!(in_scope.iter().any(|key| key.0.id == range2.id), "range2 should be in scope");
}

// =============================================================================
// Store Mask Behavior Tests (Integration)
// =============================================================================

/// Test: Verify toposort vs in_scope_ranges difference for ended ranges.
///
/// This test demonstrates the bug that was fixed:
/// - toposort() returns ALL ranges in the graph (including ended ones)
/// - in_scope_ranges() returns only active ranges
#[test]
fn test_toposort_vs_in_scope_difference() {
    let range = create_local_range(16, 0);
    let computation = range.add(&UOp::index_const(42));

    // End the range
    let ended = computation.end(smallvec::smallvec![range.clone()]);

    // Create an index that doesn't use the range directly
    let buffer = create_global_buffer(0);
    let idx = UOp::index_const(0);
    let index = create_index(buffer, vec![idx]);

    // Create final graph: index depends on ended via AFTER (since ended is Void)
    let final_graph = index.after(smallvec::smallvec![ended]);

    // toposort() would include the range (it's in the graph history)
    let topo_ranges: HashSet<u64> =
        final_graph.toposort().iter().filter(|u| matches!(u.op(), Op::Range { .. })).map(|u| u.id).collect();

    // in_scope_ranges() at final_graph should NOT include the ended range
    // because the range is not "in scope" - it was ended before this point
    let final_in_scope: HashSet<u64> = final_graph.in_scope_ranges().iter().map(|key| key.0.id).collect();

    // The range appears in toposort (it's in the graph)...
    assert!(topo_ranges.contains(&range.id), "Range should be in toposort of final_graph");
    // ...but NOT in in_scope_ranges (it was ended before final_graph)
    assert!(!final_in_scope.contains(&range.id), "Range should NOT be in final_graph's in_scope_ranges");
}

/// Test: INDEX in_scope_ranges when range is still active.
///
/// Validates that an INDEX correctly includes active ranges.
#[test]
fn test_index_in_scope_with_active_range() {
    let range = create_local_range(16, 0);
    let buffer = create_global_buffer(0);

    // INDEX that uses the range
    let index = create_index(buffer, vec![range.clone()]);

    let in_scope: HashSet<u64> = index.in_scope_ranges().iter().map(|key| key.0.id).collect();

    // Range should be in scope
    assert!(in_scope.contains(&range.id), "Active range should be in INDEX's scope");
}

/// Test: INDEX in_scope_ranges when range is NOT used (but exists in graph).
///
/// This validates the scenario where a LOCAL range exists but is not used
/// in the index computation - the range should still be in scope unless ended.
#[test]
fn test_index_scope_with_unused_but_active_range() {
    let range = create_local_range(16, 0);
    let buffer = create_global_buffer(0);

    // INDEX that does NOT use the range (uses constant instead)
    let constant_idx = UOp::index_const(0);

    // Build a graph where range is visible but not used by the index
    // range → computation → {index, computation}
    let _computation = range.add(&UOp::index_const(1));
    let index = create_index(buffer, vec![constant_idx]);

    // The index doesn't use range, so range won't be in its in_scope_ranges
    let in_scope: HashSet<u64> = index.in_scope_ranges().iter().map(|key| key.0.id).collect();

    // Range is NOT in the index's scope because it's not a source
    // This is the correct behavior - the store mask logic looks for
    // LOCAL ranges that are NOT in the index computation
    assert!(!in_scope.contains(&range.id), "Unused range should NOT be in INDEX's scope");
}
