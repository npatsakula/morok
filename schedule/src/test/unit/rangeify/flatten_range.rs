//! Tests for range flattening and canonicalization.
//!
//! Validates that flatten_range correctly unnests and canonicalizes RANGE operations
//! for kernel deduplication.

use std::sync::Arc;

use morok_ir::UOp;

use crate::rangeify::transforms::{flatten_range_impl, flatten_ranges};

#[test]
fn test_flatten_range_impl_non_supported_op() {
    // Operations that don't support flattening should return None
    let const_op = UOp::native_const(1.0f32);

    let result = flatten_range_impl(&const_op);
    assert!(result.is_none());
}

#[test]
fn test_flatten_range_impl_no_ranges() {
    // STORE operation with no ranges should return None
    let index = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store = index.store(value);

    let result = flatten_range_impl(&store);
    assert!(result.is_none());
}

#[test]
fn test_flatten_ranges_identity() {
    // Graph with no nested ranges should return unchanged
    let computation = UOp::native_const(1.0f32);
    let flattened = flatten_ranges(&computation);

    // Should return identical graph (same pointer)
    assert!(Arc::ptr_eq(&flattened, &computation));
}

// ===== Nesting Tests =====

#[test]
fn test_flatten_range_nested_end() {
    // END(END(x, [r1]), [r2]) → END(x, [r1, r2])
    use morok_ir::Op;
    use smallvec::smallvec;

    let computation = UOp::native_const(1.0f32);
    let r1 = UOp::range(UOp::index_const(10), 0);
    let r2 = UOp::range(UOp::index_const(20), 1);

    // Create nested END: END(END(computation, [r1]), [r2])
    let inner_end = computation.clone().end(smallvec![r1.clone()]);
    let outer_end = inner_end.end(smallvec![r2.clone()]);

    // Flatten
    let flattened = flatten_range_impl(&outer_end);

    // Should have flattened
    assert!(flattened.is_some(), "Nested END should be flattened");

    let flattened = flattened.unwrap();
    if let Op::End { ranges, .. } = flattened.op() {
        // Should contain both ranges (order may vary due to toposort)
        assert_eq!(ranges.len(), 2, "Should have 2 ranges after flattening");
    } else {
        panic!("Expected END operation");
    }
}

#[test]
fn test_flatten_range_deeply_nested() {
    // END(END(END(x, [r1]), [r2]), [r3]) → END(x, [r1, r2, r3])
    use morok_ir::Op;
    use smallvec::smallvec;

    let computation = UOp::native_const(1.0f32);
    let r1 = UOp::range(UOp::index_const(10), 0);
    let r2 = UOp::range(UOp::index_const(20), 1);
    let r3 = UOp::range(UOp::index_const(30), 2);

    // Create 3-level nesting
    let end1 = computation.clone().end(smallvec![r1.clone()]);
    let end2 = end1.end(smallvec![r2.clone()]);
    let end3 = end2.end(smallvec![r3.clone()]);

    // Flatten
    let flattened = flatten_range_impl(&end3);

    assert!(flattened.is_some(), "Deeply nested END should be flattened");

    let flattened = flattened.unwrap();
    if let Op::End { ranges, .. } = flattened.op() {
        assert_eq!(ranges.len(), 3, "Should have 3 ranges after deep flattening");
    } else {
        panic!("Expected END operation");
    }
}

#[test]
fn test_flatten_range_preserves_computation() {
    // Flattening should preserve the inner computation
    use morok_ir::Op;
    use smallvec::smallvec;

    // Create a binary computation: 1.0 + 2.0
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let add = a.try_add(&b).unwrap();

    let r1 = UOp::range(UOp::index_const(10), 0);
    let r2 = UOp::range(UOp::index_const(20), 1);

    // END(END(add, [r1]), [r2])
    let inner_end = add.clone().end(smallvec![r1.clone()]);
    let outer_end = inner_end.end(smallvec![r2.clone()]);

    let flattened = flatten_range_impl(&outer_end);

    assert!(flattened.is_some());
    let flattened = flattened.unwrap();

    if let Op::End { computation, ranges } = flattened.op() {
        // Computation should be preserved (the inner END, which contains the add)
        // Note: flatten_range_impl only flattens one level at a time
        assert_eq!(ranges.len(), 2);

        // The computation is now the inner END (with its single range)
        // To fully flatten, we'd need to call flatten_ranges which uses toposort
        if let Op::End { computation: inner_comp, .. } = computation.op() {
            // Inner computation should be the ADD
            assert!(matches!(inner_comp.op(), Op::Binary(..)));
        }
    } else {
        panic!("Expected END operation");
    }
}

#[test]
fn test_flatten_ranges_full_graph() {
    // Test full graph flattening via flatten_ranges
    use morok_ir::Op;
    use smallvec::smallvec;

    let computation = UOp::native_const(1.0f32);
    let r1 = UOp::range(UOp::index_const(10), 0);
    let r2 = UOp::range(UOp::index_const(20), 1);

    // Create nested structure
    let inner_end = computation.clone().end(smallvec![r1.clone()]);
    let outer_end = inner_end.end(smallvec![r2.clone()]);

    // Full graph flattening
    let flattened = flatten_ranges(&outer_end);

    // Should have applied transformation (different pointer)
    assert!(!Arc::ptr_eq(&flattened, &outer_end), "Graph should be transformed");

    // Result should still be an END
    assert!(matches!(flattened.op(), Op::End { .. }));
}

#[test]
fn test_flatten_range_single_range() {
    // END with single range should still work
    use morok_ir::Op;
    use smallvec::smallvec;

    let computation = UOp::native_const(1.0f32);
    let r1 = UOp::range(UOp::index_const(10), 0);

    let end = computation.clone().end(smallvec![r1.clone()]);

    let flattened = flatten_range_impl(&end);

    // Should succeed (single range is still valid)
    assert!(flattened.is_some());

    if let Op::End { ranges, .. } = flattened.unwrap().op() {
        assert_eq!(ranges.len(), 1);
    }
}
