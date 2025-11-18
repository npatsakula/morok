use std::collections::HashMap;
use std::rc::Rc;

use morok_device::DeviceSpec;
use morok_dtype::DType;

use crate::{ConstValue, Op, UOp};

#[test]
fn test_const_creation() {
    let c1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    assert_eq!(c1.dtype(), DType::Float32);
    assert!(matches!(c1.op(), Op::Const(_)));
}

#[test]
fn test_hash_consing() {
    // Create two identical constants
    let c1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let c2 = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // They should be the same object
    assert!(Rc::ptr_eq(&c1, &c2), "Hash consing should return same Rc for identical UOps");
}

#[test]
fn test_hash_consing_with_src() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    // Create a + b twice
    let add1 = UOp::try_add_op(a.clone(), b.clone()).unwrap();
    let add2 = UOp::try_add_op(a.clone(), b.clone()).unwrap();

    // Should be the same object
    assert!(Rc::ptr_eq(&add1, &add2), "Hash consing should work with src nodes");
}

#[test]
fn test_binary_operations() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    let add = UOp::try_add_op(a.clone(), b.clone()).unwrap();
    assert_eq!(add.dtype(), DType::Float32);
    assert_eq!(add.op().children().len(), 2);

    let mul = UOp::try_mul_op(a.clone(), b.clone()).unwrap();
    assert_eq!(mul.dtype(), DType::Float32);
}

#[test]
fn test_unary_operations() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(4.0));

    let sqrt = UOp::sqrt(&a).unwrap();
    assert_eq!(sqrt.dtype(), DType::Float32);
    assert_eq!(sqrt.op().children().len(), 1);
}

#[test]
fn test_cast() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.5));
    let cast = UOp::cast(a.clone(), DType::Int32);

    assert_eq!(cast.dtype(), DType::Int32);
}

#[test]
fn test_comparison() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    let cmp = UOp::cmplt(&a, &b).unwrap();
    assert_eq!(cmp.dtype(), DType::Bool);
}

#[test]
fn test_toposort() {
    // Build graph: (a + b) * c
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let c = UOp::const_(DType::Float32, ConstValue::Float(3.0));

    let add = UOp::try_add_op(a.clone(), b.clone()).unwrap();
    let mul = UOp::try_mul_op(add.clone(), c.clone()).unwrap();

    let sorted = mul.toposort();

    // All nodes should be present
    assert!(sorted.len() >= 5); // a, b, c, add, mul

    // Check that dependencies come before dependents
    let positions: HashMap<_, _> = sorted.iter().enumerate().map(|(i, node)| (Rc::as_ptr(node), i)).collect();

    for node in &sorted {
        let node_pos = positions[&Rc::as_ptr(node)];
        for child in node.op().children() {
            let child_pos = positions[&Rc::as_ptr(child)];
            assert!(child_pos < node_pos, "Dependencies must come before dependents");
        }
    }
}

#[test]
fn test_toposort_shared_node() {
    // Build graph: x = a + b; y = a + c; z = x * y
    // Node 'a' is shared between x and y
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let c = UOp::const_(DType::Float32, ConstValue::Float(3.0));

    let x = UOp::try_add_op(a.clone(), b.clone()).unwrap();
    let y = UOp::try_add_op(a.clone(), c.clone()).unwrap();
    let z = UOp::try_mul_op(x.clone(), y.clone()).unwrap();

    let sorted = z.toposort();

    // Node 'a' should appear only once
    let a_ptr = Rc::as_ptr(&a);
    let a_count = sorted.iter().filter(|node| Rc::as_ptr(node) == a_ptr).count();
    assert_eq!(a_count, 1, "Shared node 'a' should appear exactly once");
}

#[test]
fn test_buffer_creation() {
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    assert!(matches!(buf.op(), Op::Buffer { .. }));
    assert_eq!(buf.dtype(), DType::Float32);

    if let Op::Buffer { size, .. } = buf.op() {
        assert_eq!(*size, 100);
    } else {
        panic!("Expected Buffer op");
    }
}

#[test]
fn test_buffer_hash_consing() {
    // Two buffers with same device and size should NOT be the same
    // (due to different UNIQUE identifiers)
    let buf1 = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let buf2 = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    assert!(!Rc::ptr_eq(&buf1, &buf2), "Different buffers should have different UNIQUE ids");
}

#[test]
fn test_buffer_view() {
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 1000, DType::Float32);
    let view = UOp::buffer_view(buf, 100, 50);

    assert!(matches!(view.op(), Op::BufferView { .. }));
    assert_eq!(view.dtype(), DType::Float32);

    if let Op::BufferView { size, offset, .. } = view.op() {
        assert_eq!(*size, 100);
        assert_eq!(*offset, 50);
    } else {
        panic!("Expected BufferView op");
    }
}

#[test]
fn test_index_operation() {
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let idx = UOp::const_(DType::Index, ConstValue::UInt(10));

    let indexed = UOp::index(buf, vec![idx]).expect("index should succeed");
    assert!(matches!(indexed.op(), Op::Index { .. }));
    assert_eq!(indexed.op().children().len(), 2); // buffer + 1 index
}

#[test]
fn test_device_and_unique() {
    let dev = UOp::device(DeviceSpec::Cpu);
    assert!(matches!(dev.op(), Op::Device(_)));
    if let Op::Device(spec) = dev.op() {
        assert_eq!(*spec, DeviceSpec::Cpu);
    }

    let uniq = UOp::unique(Some(42));
    assert!(matches!(uniq.op(), Op::Unique(42)));

    let uniq_auto = UOp::unique(None);
    assert!(matches!(uniq_auto.op(), Op::Unique(_)));
}

#[test]
fn test_children_method() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let add = UOp::try_add_op(a.clone(), b.clone()).unwrap();

    let children = add.op().children();
    assert_eq!(children.len(), 2);
    assert!(Rc::ptr_eq(children[0], &a));
    assert!(Rc::ptr_eq(children[1], &b));
}

#[test]
fn test_for_each_child() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let add = UOp::try_add_op(a.clone(), b.clone()).unwrap();

    let mut children = Vec::new();
    add.op().map_child(|child| children.push(child.clone()));

    assert_eq!(children.len(), 2);
    assert!(Rc::ptr_eq(&children[0], &a));
    assert!(Rc::ptr_eq(&children[1], &b));
}

// ============================================================================
// Cached Property Tests
// ============================================================================

#[test]
fn test_shape_property_scalar() {
    // Scalar constant should have empty shape
    let scalar = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let shape = scalar.shape();

    assert!(shape.is_some(), "Scalar should have shape");
    assert_eq!(shape.unwrap().len(), 0, "Scalar should have empty shape");
}

#[test]
fn test_shape_property_lazy_evaluation() {
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::ShapeProperty;

    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let add = UOp::try_add_op(a.clone(), b.clone()).unwrap();

    // VERIFY: Cache is empty before first access (lazy evaluation)
    assert!(ShapeProperty::cache(&add).get().is_none(), "Cache should be empty before first access");

    // First access triggers computation
    let shape1 = ShapeProperty::get(&add);
    assert!(shape1.is_some());

    // VERIFY: Cache is now populated
    assert!(ShapeProperty::cache(&add).get().is_some(), "Cache should be populated after first access");

    // Second access retrieves from cache (same pointer)
    let shape2 = ShapeProperty::get(&add);

    // VERIFY: Both accesses return the same cached reference
    assert!(std::ptr::eq(shape1, shape2), "Second access should return same cached reference");
}

#[test]
fn test_ranges_property_no_ranges() {
    // Simple arithmetic with no RANGE ops
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let add = UOp::try_add_op(a, b).unwrap();

    let ranges = add.ranges();
    assert_eq!(ranges.len(), 0, "No RANGE ops in simple arithmetic");
}

#[test]
fn test_ranges_property_with_range() {
    use crate::AxisType;

    // Create a RANGE op
    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range(end, 0, AxisType::Loop);

    // Create some computation that uses the range
    let idx = UOp::cast(range.clone(), DType::Float32);

    let ranges = idx.ranges();
    assert_eq!(ranges.len(), 1, "Should find one RANGE op");
    assert!(Rc::ptr_eq(&ranges[0], &range));
}

#[test]
fn test_ranges_property_lazy_evaluation() {
    use crate::AxisType;
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::RangesProperty;

    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range(end, 0, AxisType::Loop);
    let idx = UOp::cast(range.clone(), DType::Float32);

    // VERIFY: Cache is empty before first access (lazy evaluation)
    assert!(RangesProperty::cache(&idx).get().is_none(), "Cache should be empty before first access");

    // First access triggers computation
    let ranges1 = RangesProperty::get(&idx);
    assert_eq!(ranges1.len(), 1);

    // VERIFY: Cache is now populated
    assert!(RangesProperty::cache(&idx).get().is_some(), "Cache should be populated after first access");

    // Second access retrieves from cache (same pointer)
    let ranges2 = RangesProperty::get(&idx);

    // VERIFY: Both accesses return the same cached reference
    assert!(std::ptr::eq(ranges1, ranges2), "Second access should return same cached reference");
    assert!(Rc::ptr_eq(&ranges1[0], &ranges2[0]));
}

#[test]
fn test_in_scope_ranges_simple() {
    use crate::AxisType;

    // Create a RANGE op
    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range(end, 0, AxisType::Loop);

    // RANGE itself should have itself in scope
    let in_scope = range.in_scope_ranges();
    assert_eq!(in_scope.len(), 1, "RANGE should have itself in scope");

    // Create computation that uses the range
    let idx = UOp::cast(range.clone(), DType::Float32);
    let in_scope_idx = idx.in_scope_ranges();
    assert_eq!(in_scope_idx.len(), 1, "Computation should inherit RANGE scope");
}

#[test]
fn test_in_scope_ranges_lazy_evaluation() {
    use crate::AxisType;
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::InScopeRangesProperty;

    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range(end, 0, AxisType::Loop);
    let idx = UOp::cast(range.clone(), DType::Float32);

    // VERIFY: Cache is empty before first access (lazy evaluation)
    assert!(InScopeRangesProperty::cache(&idx).get().is_none(), "Cache should be empty before first access");

    // First access triggers computation
    let in_scope1 = InScopeRangesProperty::get(&idx);
    assert_eq!(in_scope1.len(), 1);

    // VERIFY: Cache is now populated
    assert!(InScopeRangesProperty::cache(&idx).get().is_some(), "Cache should be populated after first access");

    // Second access retrieves from cache (same pointer)
    let in_scope2 = InScopeRangesProperty::get(&idx);

    // VERIFY: Both accesses return the same cached reference
    assert!(std::ptr::eq(in_scope1, in_scope2), "Second access should return same cached reference");
}

#[test]
fn test_in_scope_ranges_after_end() {
    use crate::AxisType;
    use smallvec::smallvec;

    // Create a RANGE and computation
    let end_val = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range(end_val, 0, AxisType::Loop);
    let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Create END operation
    let end_op = UOp::end(compute.clone(), smallvec![range.clone()]);

    // After END, the range should no longer be in scope
    let in_scope = end_op.in_scope_ranges();
    assert_eq!(in_scope.len(), 0, "After END, range should not be in scope");
}

#[test]
fn test_in_scope_ranges_nested() {
    use crate::AxisType;
    use smallvec::smallvec;

    // Create two nested RANGEs
    let end1 = UOp::const_(DType::Index, ConstValue::Int(10));
    let _range1 = UOp::range(end1, 0, AxisType::Loop);

    let end2 = UOp::const_(DType::Index, ConstValue::Int(20));
    let range2 = UOp::range(end2, 1, AxisType::Loop);

    // Computation that uses both ranges
    let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Both ranges should be in scope
    let in_scope = compute.in_scope_ranges();
    assert_eq!(in_scope.len(), 0, "Const has no ranges in scope initially");

    // After ending range2, only range1 should be in scope
    let after_end2 = UOp::end(compute.clone(), smallvec![range2.clone()]);
    let in_scope_after = after_end2.in_scope_ranges();
    assert_eq!(in_scope_after.len(), 0, "After END, ranges are not propagated to parent");
}

#[test]
fn test_toposort_filtered_basic() {
    // Build graph: a -> b -> c
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::try_add_op(a.clone(), UOp::const_(DType::Float32, ConstValue::Float(2.0))).unwrap();
    let c = UOp::try_mul_op(b.clone(), UOp::const_(DType::Float32, ConstValue::Float(3.0))).unwrap();

    // Filter to only include 'c'
    let filtered = c.toposort_filtered(|node| Rc::ptr_eq(node, &c));

    // Should only contain 'c' since gate blocks traversal of children
    assert_eq!(filtered.len(), 1, "Filtered toposort should only include nodes passing gate");
    assert!(Rc::ptr_eq(&filtered[0], &c));
}

#[test]
fn test_toposort_filtered_all() {
    // Build graph: a + b
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let add = UOp::try_add_op(a.clone(), b.clone()).unwrap();

    // Filter that accepts all nodes
    let filtered = add.toposort_filtered(|_| true);

    // Should be same as regular toposort
    let regular = add.toposort();
    assert_eq!(filtered.len(), regular.len());
}

#[test]
fn test_toposort_filtered_none() {
    // Build graph
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Filter that rejects all nodes
    let filtered = a.toposort_filtered(|_| false);

    // Should be empty (gate blocks traversal)
    assert_eq!(filtered.len(), 0, "Gate blocking all nodes should return empty");
}

#[test]
fn test_multiple_properties_coexist() {
    // Create a constant (has shape)
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    // Create an addition operation
    let add = UOp::try_add_op(a, b).unwrap();

    // Access shape property (const operations have shape)
    let shape = add.shape();
    assert!(shape.is_some());
    assert_eq!(shape.unwrap().len(), 0); // Scalar

    // Access ranges property (no ranges in this graph)
    let ranges = add.ranges();
    assert_eq!(ranges.len(), 0);

    // Access in_scope_ranges property
    let in_scope = add.in_scope_ranges();
    assert_eq!(in_scope.len(), 0);

    // All should be cached independently
    let shape2 = add.shape();
    let ranges2 = add.ranges();
    let in_scope2 = add.in_scope_ranges();

    assert_eq!(shape, shape2);
    assert_eq!(ranges.len(), ranges2.len());
    assert_eq!(in_scope.len(), in_scope2.len());
}
