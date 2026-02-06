use std::{f32::consts::PI, sync::Arc};

use crate::rangeify::patterns::buffer_folding;
use crate::rewrite::graph_rewrite;
use morok_dtype::DType;
use morok_ir::{ConstValue, UOp};

// Helper functions for creating test UOps
fn create_const(val: i64) -> Arc<UOp> {
    UOp::native_const(val as i32)
}

fn create_range(end: i64, axis_id: usize) -> Arc<UOp> {
    UOp::range_const(end, axis_id)
}

fn create_bufferize(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::bufferize_global(compute, ranges)
}

// Pattern 1: Noop Buffer Removal Tests

#[test]
fn test_noop_bufferize_same_ranges() {
    // INDEX(BUFFERIZE(x, R), R) → x
    let x = UOp::define_global(1, DType::Float32);
    let range = create_range(10, 0);
    let ranges = vec![range.clone()];

    let bufferized = create_bufferize(x.clone(), ranges.clone());
    let indexed = UOp::index().buffer(bufferized).indices(ranges).call().unwrap();

    let matcher = buffer_folding();
    let result = graph_rewrite(&matcher, indexed, &mut ());

    // Should fold to just x
    assert!(Arc::ptr_eq(&result, &x), "Noop BUFFERIZE should be removed");
}

#[test]
fn test_noop_bufferize_different_ranges() {
    // INDEX(BUFFERIZE(x, R1), R2) where R1 != R2 should NOT fold
    let x = UOp::define_global(1, DType::Float32);
    let range1 = create_range(10, 0);
    let range2 = create_range(10, 1); // Different axis_id

    let bufferized = create_bufferize(x.clone(), vec![range1]);
    let indexed = UOp::index().buffer(bufferized.clone()).indices(vec![range2]).call().unwrap();

    let matcher = buffer_folding();
    let result = graph_rewrite(&matcher, indexed.clone(), &mut ());

    // Should NOT fold - ranges are different
    assert!(Arc::ptr_eq(&result, &indexed), "Should not fold with different ranges");
}

#[test]
fn test_noop_bufferize_multiple_ranges() {
    // INDEX(BUFFERIZE(x, [R1, R2]), [R1, R2]) → x
    let x = UOp::define_global(1, DType::Float32);
    let range1 = create_range(10, 0);
    let range2 = create_range(20, 1);
    let ranges = vec![range1.clone(), range2.clone()];

    let bufferized = create_bufferize(x.clone(), ranges.clone());
    let indexed = UOp::index().buffer(bufferized).indices(ranges).call().unwrap();

    let matcher = buffer_folding();
    let result = graph_rewrite(&matcher, indexed, &mut ());

    // Should fold to just x
    assert!(Arc::ptr_eq(&result, &x), "Noop BUFFERIZE with multiple ranges should be removed");
}

// Pattern 2: BUFFERIZE(CONST) → CONST Tests

#[test]
fn test_bufferize_const_folding() {
    // BUFFERIZE(CONST, ranges) → CONST
    let const_val = create_const(42);
    let range = create_range(10, 0);

    let bufferized = create_bufferize(const_val.clone(), vec![range]);

    let matcher = buffer_folding();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should fold to just the constant
    assert!(Arc::ptr_eq(&result, &const_val), "BUFFERIZE(CONST) should fold to CONST");
}

#[test]
fn test_bufferize_different_const_types() {
    // Test with different constant types
    let test_cases = vec![
        (DType::Int32, ConstValue::Int(100)),
        (DType::Float32, ConstValue::Float(PI.into())),
        (DType::Bool, ConstValue::Bool(true)),
    ];

    for (dtype, val) in test_cases {
        let const_val = UOp::const_(dtype.clone(), val);
        let range = create_range(5, 0);

        let bufferized = create_bufferize(const_val.clone(), vec![range]);

        let matcher = buffer_folding();
        let result = graph_rewrite(&matcher, bufferized, &mut ());

        assert!(Arc::ptr_eq(&result, &const_val), "BUFFERIZE(CONST) should fold for {:?}", dtype);
    }
}

// Pattern 3: INDEX(CONST) → CONST Tests

#[test]
fn test_index_const_folding() {
    // INDEX(CONST, ranges) → CONST
    let const_val = create_const(7);
    let range = create_range(10, 0);

    let indexed = UOp::index().buffer(const_val.clone()).indices(vec![range]).call().unwrap();

    let matcher = buffer_folding();
    let result = graph_rewrite(&matcher, indexed, &mut ());

    // Should fold to just the constant
    assert!(Arc::ptr_eq(&result, &const_val), "INDEX(CONST) should fold to CONST");
}

#[test]
fn test_index_const_multiple_indices() {
    // INDEX(CONST, [R1, R2, R3]) → CONST
    let const_val = UOp::native_const(2.5f32);
    let ranges = vec![create_range(10, 0), create_range(20, 1), create_range(30, 2)];

    let indexed = UOp::index().buffer(const_val.clone()).indices(ranges).call().unwrap();

    let matcher = buffer_folding();
    let result = graph_rewrite(&matcher, indexed, &mut ());

    assert!(Arc::ptr_eq(&result, &const_val), "INDEX(CONST) with multiple indices should fold");
}

// Pattern 4: COPY(CONST) → CONST Tests

#[test]
fn test_copy_const_folding() {
    // COPY(CONST, device) → CONST
    let const_val = create_const(99);
    let device = UOp::device(morok_device::DeviceSpec::Cpu);

    let copy = const_val.copy(device);

    let matcher = buffer_folding();
    let result = graph_rewrite(&matcher, copy, &mut ());

    // Should fold to just the constant (device doesn't matter for constants)
    assert!(Arc::ptr_eq(&result, &const_val), "COPY(CONST) should fold to CONST");
}

#[test]
fn test_copy_const_different_devices() {
    // Test copying constants to different devices - all should fold
    let const_val = UOp::native_const(1.5f32);

    let devices = vec![morok_device::DeviceSpec::Cpu, morok_device::DeviceSpec::Cuda { device_id: 0 }];

    for device_spec in devices {
        let device = UOp::device(device_spec);
        let copy = const_val.copy(device);

        let matcher = buffer_folding();
        let result = graph_rewrite(&matcher, copy, &mut ());

        assert!(Arc::ptr_eq(&result, &const_val), "COPY(CONST) should fold regardless of device");
    }
}

// Combined/Integration Tests

#[test]
fn test_nested_constant_folding() {
    // INDEX(BUFFERIZE(CONST, R1), R1) should fold through both patterns
    let const_val = create_const(123);
    let range = create_range(15, 0);

    let bufferized = create_bufferize(const_val.clone(), vec![range.clone()]);
    let indexed = UOp::index().buffer(bufferized).indices(vec![range]).call().unwrap();

    let matcher = buffer_folding();
    let result = graph_rewrite(&matcher, indexed, &mut ());

    // Should fold all the way to the constant
    assert!(Arc::ptr_eq(&result, &const_val), "Nested constant operations should fold completely");
}

#[test]
fn test_noop_fold_non_const_operations() {
    // INDEX(BUFFERIZE(x, R), R) should fold to x even for non-constant operations
    // Use symbolic variables instead of define_global (which requires pointer dtype)
    let x = UOp::var("x", DType::Float32, 0, 100);
    let y = UOp::var("y", DType::Float32, 0, 100);

    let add = x.try_add(&y).unwrap();

    let range = create_range(10, 0);
    let bufferized = create_bufferize(add.clone(), vec![range.clone()]);
    let indexed = UOp::index().buffer(bufferized.clone()).indices(vec![range]).call().unwrap();

    let matcher = buffer_folding();
    let result = graph_rewrite(&matcher, indexed.clone(), &mut ());

    // Should fold - noop buffer removal works for all operations
    assert!(Arc::ptr_eq(&result, &add), "Noop BUFFERIZE+INDEX should fold regardless of operation type");
}
