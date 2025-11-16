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
