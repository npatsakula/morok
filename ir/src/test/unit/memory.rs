//! Tests for memory and buffer operations constructors.

use morok_dtype::DType;
use morok_dtype::DeviceSpec;

use crate::types::{AddrSpace, AxisId, AxisType, BufferizeOpts};
use crate::{Op, UOp};

#[test]
fn test_bufferize() {
    let compute = UOp::native_const(1.0f32);
    let r1 = UOp::range_axis(UOp::native_const(10i32), AxisId::Renumbered(0), AxisType::Loop);
    let r2 = UOp::range_axis(UOp::native_const(20i32), AxisId::Renumbered(1), AxisType::Loop);

    let opts = BufferizeOpts::new(DeviceSpec::Cpu);
    let bufferize = UOp::bufferize(compute.clone(), vec![r1, r2], opts);

    // Should have same dtype as compute
    assert_eq!(bufferize.dtype(), DType::Float32);

    // Should be Bufferize op
    if let Op::Bufferize { compute: c, ranges, opts: o } = bufferize.op() {
        assert!(std::sync::Arc::ptr_eq(c, &compute));
        assert_eq!(ranges.len(), 2);
        assert_eq!(o.device, Some(DeviceSpec::Cpu));
        assert_eq!(o.addrspace, AddrSpace::Global);
    } else {
        panic!("Expected Bufferize op");
    }
}

#[test]
fn test_bufferize_local() {
    let compute = UOp::native_const(1.0f32);
    let r = UOp::range_axis(UOp::native_const(10i32), AxisId::Renumbered(0), AxisType::Loop);

    let opts = BufferizeOpts::local();
    let bufferize = UOp::bufferize(compute, vec![r], opts);

    if let Op::Bufferize { opts: o, .. } = bufferize.op() {
        assert_eq!(o.addrspace, AddrSpace::Local);
    } else {
        panic!("Expected Bufferize op");
    }
}

#[test]
fn test_load() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index_const(0);

    let load = UOp::load(buffer.clone(), index.clone());

    // Should have same dtype as buffer
    assert_eq!(load.dtype(), DType::Float32);

    // Should be Load op
    if let Op::Load { buffer: b, index: i } = load.op() {
        assert!(std::sync::Arc::ptr_eq(b, &buffer));
        assert!(std::sync::Arc::ptr_eq(i, &index));
    } else {
        panic!("Expected Load op");
    }
}

#[test]
fn test_load_gated() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index_const(0);
    let gate = UOp::native_const(true);

    let load = UOp::load_gated(buffer.clone(), index.clone(), gate.clone());

    // Should have same dtype as buffer
    assert_eq!(load.dtype(), DType::Float32);

    // Should be LoadGated op
    if let Op::LoadGated { buffer: b, index: i, gate: g } = load.op() {
        assert!(std::sync::Arc::ptr_eq(b, &buffer));
        assert!(std::sync::Arc::ptr_eq(i, &index));
        assert!(std::sync::Arc::ptr_eq(g, &gate));
    } else {
        panic!("Expected LoadGated op");
    }
}

#[test]
fn test_store() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index_const(0);
    let value = UOp::native_const(42.0f32);

    let store = UOp::store(buffer.clone(), index.clone(), value.clone());

    // Store should have Void dtype
    assert_eq!(store.dtype(), DType::Void);

    // Should be Store op
    if let Op::Store { buffer: b, index: i, value: v, .. } = store.op() {
        assert!(std::sync::Arc::ptr_eq(b, &buffer));
        assert!(std::sync::Arc::ptr_eq(i, &index));
        assert!(std::sync::Arc::ptr_eq(v, &value));
    } else {
        panic!("Expected Store op");
    }
}

#[test]
fn test_store_gated() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index_const(0);
    let value = UOp::native_const(42.0f32);
    let gate = UOp::native_const(true);

    let store = UOp::store_gated(buffer.clone(), index.clone(), value.clone(), gate.clone());

    // Store should have Void dtype
    assert_eq!(store.dtype(), DType::Void);

    // Should be StoreGated op
    if let Op::StoreGated { buffer: b, index: i, value: v, gate: g, .. } = store.op() {
        assert!(std::sync::Arc::ptr_eq(b, &buffer));
        assert!(std::sync::Arc::ptr_eq(i, &index));
        assert!(std::sync::Arc::ptr_eq(v, &value));
        assert!(std::sync::Arc::ptr_eq(g, &gate));
    } else {
        panic!("Expected StoreGated op");
    }
}

#[test]
fn test_define_global() {
    let dg = UOp::define_global(0, DType::Float32);

    assert_eq!(dg.dtype(), DType::Float32);

    if let Op::DefineGlobal(id) = dg.op() {
        assert_eq!(*id, 0);
    } else {
        panic!("Expected DefineGlobal op");
    }
}

#[test]
fn test_define_local() {
    let dl = UOp::define_local(1, DType::Int32);

    assert_eq!(dl.dtype(), DType::Int32);

    if let Op::DefineLocal(id) = dl.op() {
        assert_eq!(*id, 1);
    } else {
        panic!("Expected DefineLocal op");
    }
}
