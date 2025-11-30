//! Comprehensive tests for RangeifyContext state tracking.
//!
//! Tests verify that RangeifyContext correctly manages transformation state:
//! - Range ID generation and uniqueness
//! - Transform recording and retrieval
//! - Edge cases and large-scale scenarios
//! - Thread-safety considerations

use std::rc::Rc;

use morok_dtype::DType;
use morok_ir::{BufferizeOpts, ConstValue, Op, UOp};

use crate::rangeify::RangeifyContext;

// ===== Basic Functionality Tests =====

#[test]
fn test_new_context_is_empty() {
    let ctx = RangeifyContext::new();
    assert_eq!(ctx.range_counter, 0, "New context should have counter at 0");
    assert_eq!(ctx.range_map.len(), 0, "New context should have empty range_map");
}

#[test]
fn test_default_context_is_empty() {
    let ctx = RangeifyContext::default();
    assert_eq!(ctx.range_counter, 0, "Default context should have counter at 0");
    assert_eq!(ctx.range_map.len(), 0, "Default context should have empty range_map");
}

#[test]
fn test_next_range_id_increments() {
    let mut ctx = RangeifyContext::new();

    assert_eq!(ctx.next_range_id(), 0, "First ID should be 0");
    assert_eq!(ctx.next_range_id(), 1, "Second ID should be 1");
    assert_eq!(ctx.next_range_id(), 2, "Third ID should be 2");
    assert_eq!(ctx.range_counter, 3, "Counter should be at 3 after 3 allocations");
}

#[test]
fn test_record_and_retrieve_transform() {
    let mut ctx = RangeifyContext::new();

    let original = UOp::native_const(1.0f32);
    let rangeified = UOp::native_const(2.0f32);

    ctx.record_transform(original.clone(), rangeified.clone());

    let retrieved = ctx.get_rangeified(&original);
    assert!(retrieved.is_some(), "Should find recorded transformation");
    assert!(Rc::ptr_eq(retrieved.unwrap(), &rangeified), "Retrieved value should be the same Rc as recorded");
}

#[test]
fn test_get_missing_returns_none() {
    let ctx = RangeifyContext::new();

    let uop = UOp::native_const(1.0f32);
    assert!(ctx.get_rangeified(&uop).is_none(), "Should return None for missing transformation");
}

// ===== Multiple Transform Tests =====

#[test]
fn test_multiple_transforms() {
    let mut ctx = RangeifyContext::new();

    let original1 = UOp::native_const(1.0f32);
    let rangeified1 = UOp::native_const(2.0f32);

    let original2 = UOp::native_const(10i32);
    let rangeified2 = UOp::native_const(20i32);

    let original3 = UOp::native_const(true);
    let rangeified3 = UOp::native_const(false);

    ctx.record_transform(original1.clone(), rangeified1.clone());
    ctx.record_transform(original2.clone(), rangeified2.clone());
    ctx.record_transform(original3.clone(), rangeified3.clone());

    assert_eq!(ctx.range_map.len(), 3, "Should have 3 recorded transforms");

    // Verify each transform is independently retrievable
    assert!(Rc::ptr_eq(ctx.get_rangeified(&original1).unwrap(), &rangeified1));
    assert!(Rc::ptr_eq(ctx.get_rangeified(&original2).unwrap(), &rangeified2));
    assert!(Rc::ptr_eq(ctx.get_rangeified(&original3).unwrap(), &rangeified3));
}

#[test]
fn test_overwrite_transform() {
    let mut ctx = RangeifyContext::new();

    let original = UOp::native_const(1.0f32);
    let rangeified1 = UOp::native_const(2.0f32);
    let rangeified2 = UOp::native_const(3.0f32);

    ctx.record_transform(original.clone(), rangeified1.clone());
    ctx.record_transform(original.clone(), rangeified2.clone());

    // Should have the second value
    let retrieved = ctx.get_rangeified(&original).unwrap();
    assert!(Rc::ptr_eq(retrieved, &rangeified2), "Should retrieve the most recently recorded transform");
    assert_eq!(ctx.range_map.len(), 1, "Should still have only 1 entry (overwritten)");
}

// ===== Complex Operation Tests =====

#[test]
fn test_transform_with_binary_ops() {
    let mut ctx = RangeifyContext::new();

    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let original = a.try_add(&b).unwrap();

    let c = UOp::native_const(3.0f32);
    let d = UOp::native_const(4.0f32);
    let rangeified = c.try_add(&d).unwrap();

    ctx.record_transform(original.clone(), rangeified.clone());

    let retrieved = ctx.get_rangeified(&original).unwrap();
    assert!(Rc::ptr_eq(retrieved, &rangeified));
}

#[test]
fn test_transform_with_nested_structure() {
    let mut ctx = RangeifyContext::new();

    // Create nested computation: (a + b) * c
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let sum = a.try_add(&b).unwrap();
    let c = UOp::native_const(3.0f32);
    let original = sum.try_mul(&c).unwrap();

    // Create rangeified version
    let d = UOp::native_const(4.0f32);
    let rangeified = d;

    ctx.record_transform(original.clone(), rangeified.clone());

    let retrieved = ctx.get_rangeified(&original).unwrap();
    assert!(Rc::ptr_eq(retrieved, &rangeified));
}

// ===== Edge Case Tests =====

#[test]
fn test_transform_same_value() {
    let mut ctx = RangeifyContext::new();

    let value = UOp::native_const(1.0f32);

    // Record a transform where original == rangeified
    ctx.record_transform(value.clone(), value.clone());

    let retrieved = ctx.get_rangeified(&value).unwrap();
    assert!(Rc::ptr_eq(retrieved, &value), "Should correctly handle self-transform");
}

#[test]
fn test_range_id_large_count() {
    let mut ctx = RangeifyContext::new();

    // Allocate many IDs to test large counter values
    for i in 0..1000 {
        assert_eq!(ctx.next_range_id(), i, "ID should match iteration count");
    }

    assert_eq!(ctx.range_counter, 1000, "Counter should reach 1000");
}

#[test]
fn test_many_transforms() {
    let mut ctx = RangeifyContext::new();

    // Record many transforms
    let count = 100;
    let mut originals = Vec::new();
    let mut rangeifieds = Vec::new();

    for i in 0..count {
        let original = UOp::native_const(i);
        let rangeified = UOp::native_const(i * 2);

        originals.push(original.clone());
        rangeifieds.push(rangeified.clone());

        ctx.record_transform(original, rangeified);
    }

    assert_eq!(ctx.range_map.len(), count as usize, "Should have all transforms recorded");

    // Verify all transforms are retrievable
    for (i, (original, rangeified)) in originals.iter().zip(rangeifieds.iter()).enumerate() {
        let retrieved = ctx.get_rangeified(original);
        assert!(retrieved.is_some(), "Transform {} should be retrievable", i);
        assert!(Rc::ptr_eq(retrieved.unwrap(), rangeified), "Transform {} should match", i);
    }
}

// ===== UOpKey Equality Tests =====

#[test]
fn test_transform_with_equivalent_uops() {
    let mut ctx = RangeifyContext::new();

    // Create two separate UOps with same value
    let original1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let original2 = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    let rangeified = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    ctx.record_transform(original1.clone(), rangeified.clone());

    // Since original1 and original2 are structurally equal and use hash consing,
    // they should be the same Rc
    assert!(Rc::ptr_eq(&original1, &original2), "Hash consing should make equivalent UOps share the same Rc");

    // So we should be able to retrieve using original2
    let retrieved = ctx.get_rangeified(&original2);
    assert!(retrieved.is_some(), "Should find transform using equivalent UOp");
    assert!(Rc::ptr_eq(retrieved.unwrap(), &rangeified));
}

// ===== Integration with Real Operations =====

#[test]
fn test_transform_with_reshape() {
    let mut ctx = RangeifyContext::new();

    let tensor = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let new_shape = UOp::index_const(10);
    let reshape = UOp::new(Op::Reshape { src: tensor.clone(), new_shape }, tensor.dtype());

    let rangeified = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    ctx.record_transform(reshape.clone(), rangeified.clone());

    let retrieved = ctx.get_rangeified(&reshape).unwrap();
    assert!(Rc::ptr_eq(retrieved, &rangeified));
}

#[test]
fn test_transform_with_bufferize() {
    let mut ctx = RangeifyContext::new();

    let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let range = UOp::index_const(10);
    let bufferize = UOp::bufferize(compute, vec![range], BufferizeOpts::local());

    let rangeified = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    ctx.record_transform(bufferize.clone(), rangeified.clone());

    let retrieved = ctx.get_rangeified(&bufferize).unwrap();
    assert!(Rc::ptr_eq(retrieved, &rangeified));
}

// ===== Counter Independence Tests =====

#[test]
fn test_range_counter_independent_of_transforms() {
    let mut ctx = RangeifyContext::new();

    // Allocate some IDs
    ctx.next_range_id();
    ctx.next_range_id();

    // Record some transforms
    let original = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let rangeified = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    ctx.record_transform(original, rangeified);

    // Allocate more IDs
    assert_eq!(ctx.next_range_id(), 2, "Counter should be independent of transform recording");
    assert_eq!(ctx.next_range_id(), 3);
}

#[test]
fn test_transforms_independent_of_counter() {
    let mut ctx = RangeifyContext::new();

    // Record some transforms
    for i in 0..10 {
        let original = UOp::const_(DType::Int32, ConstValue::Int(i as i64));
        let rangeified = UOp::const_(DType::Int32, ConstValue::Int((i * 2) as i64));
        ctx.record_transform(original, rangeified);
    }

    // Counter should still be at 0 since we didn't call next_range_id()
    assert_eq!(ctx.range_counter, 0, "Transform recording shouldn't affect counter");
    assert_eq!(ctx.range_map.len(), 10, "Should have all transforms");

    // Now allocate IDs
    assert_eq!(ctx.next_range_id(), 0);
    assert_eq!(ctx.next_range_id(), 1);
}

// ===== Stress Tests =====

#[test]
fn test_interleaved_operations() {
    let mut ctx = RangeifyContext::new();

    for i in 0..50 {
        // Allocate an ID
        let id = ctx.next_range_id();
        assert_eq!(id, i);

        // Record a transform
        let original = UOp::const_(DType::Int32, ConstValue::Int(i as i64));
        let rangeified = UOp::const_(DType::Int32, ConstValue::Int((i * 3) as i64));
        ctx.record_transform(original.clone(), rangeified.clone());

        // Verify immediate retrieval
        let retrieved = ctx.get_rangeified(&original).unwrap();
        assert!(Rc::ptr_eq(retrieved, &rangeified));
    }

    assert_eq!(ctx.range_counter, 50);
    assert_eq!(ctx.range_map.len(), 50);
}
