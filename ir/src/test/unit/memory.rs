//! Tests for memory and buffer operations constructors.

use svod_dtype::DType;
use svod_dtype::DeviceSpec;

use crate::ops;
use crate::types::{AddrSpace, AxisId, AxisType, BufferizeOpts};
use crate::{Op, UOp};

/// GETADDR reads a buffer's device address: a plain scalar u64 with no address space of
/// its own, carrying its device as typed metadata rather than as a source.
#[test]
fn getaddr_is_a_scalar_device_address() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let address = buffer.getaddr(None);

    assert_eq!(address.dtype(), DType::UInt64);
    assert_eq!(address.shape().unwrap().unwrap().as_slice(), &[]);
    assert_eq!(address.addrspace(), None);
    assert!(address.tree().contains("GETADDR(CPU)"));
    match address.op() {
        Op::GetAddr(ops::GetAddr { src, device }) => {
            assert!(std::sync::Arc::ptr_eq(src, &buffer));
            assert_eq!(device, &DeviceSpec::Cpu);
            assert_eq!(src.addrspace(), Some(AddrSpace::Global));
        }
        op => panic!("expected GETADDR, got {op:?}"),
    }

    let graph = crate::CanonicalGraph::from_root("hcq", &address).unwrap();
    let node = graph.nodes.last().unwrap();
    assert_eq!(node.op, "GETADDR");
    assert_eq!(node.src.len(), 1);
    assert_eq!(node.arg, crate::CanonicalArg::Device { name: "CPU".to_string() });

    assert_ne!(address.id, buffer.getaddr(Some(DeviceSpec::Cuda { device_id: 0 })).id, "device is part of identity");
    assert!(std::sync::Arc::ptr_eq(&address.with_sources(vec![buffer]), &address));

    // Already-scalar sources have no address to take.
    let scalar = UOp::native_const(1u64);
    assert!(std::sync::Arc::ptr_eq(&scalar.getaddr(Some(DeviceSpec::Cpu)), &scalar));
}

#[test]
fn test_stage_records_ranges_and_address_space() {
    let compute = UOp::native_const(1.0f32);
    let r1 = UOp::range_axis(UOp::native_const(10i32), AxisId::Renumbered(0), AxisType::Loop);
    let r2 = UOp::range_axis(UOp::native_const(20i32), AxisId::Renumbered(1), AxisType::Loop);

    let stage = UOp::stage(compute.clone(), vec![r1, r2], BufferizeOpts::new(DeviceSpec::Cpu));
    assert_eq!(stage.dtype(), DType::Float32, "STAGE keeps the computed dtype");
    let Op::Stage(ops::Stage { compute: staged, ranges, opts }) = stage.op() else {
        panic!("expected STAGE, got {:?}", stage.op())
    };
    assert!(std::sync::Arc::ptr_eq(staged, &compute));
    assert_eq!(ranges.len(), 2);
    assert_eq!(opts.device, Some(DeviceSpec::Cpu));
    assert_eq!(opts.addrspace, AddrSpace::Global);

    let local = UOp::stage(compute, vec![], BufferizeOpts::local());
    let Op::Stage(ops::Stage { opts, .. }) = local.op() else { panic!("expected STAGE, got {:?}", local.op()) };
    assert_eq!(opts.addrspace, AddrSpace::Local);
}

#[test]
fn test_load() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let offset = UOp::index_const(0);
    let index = UOp::index().buffer(buffer.clone()).indices(vec![offset]).call().unwrap();

    let load = UOp::load().index(index.clone()).call();
    let gated = UOp::load().index(index.clone()).alt(UOp::native_const(0.0f32)).gate(UOp::native_const(true)).call();

    assert_eq!(load.dtype(), DType::Float32, "LOAD has the buffer's dtype");
    match load.op() {
        Op::Load(ops::Load { index: i, alt: None, gate: None }) => assert!(std::sync::Arc::ptr_eq(i, &index)),
        op => panic!("expected an ungated Load, got {op:?}"),
    }
    assert!(matches!(gated.op(), Op::Load(ops::Load { alt: Some(_), gate: Some(_), .. })));
}

#[test]
fn test_store() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index_offset = UOp::index_const(0);
    let value = UOp::native_const(42.0f32);

    // Create INDEX op first (STORE's index field is an INDEX op)
    let index = UOp::index().buffer(buffer.clone()).indices(vec![index_offset]).call().unwrap();

    // Use store_value() on INDEX (preferred API)
    let store = index.store_value(value.clone());

    // Store should have Void dtype
    assert_eq!(store.dtype(), DType::Void);

    // Should be Store op with index pointing to buffer
    if let Op::Store(ops::Store { index: i, value: v, .. }) = store.op() {
        assert!(std::sync::Arc::ptr_eq(i, &index));
        assert!(std::sync::Arc::ptr_eq(v, &value));
        // Verify buffer can be accessed via store_buffer()
        assert!(std::sync::Arc::ptr_eq(store.store_buffer().unwrap(), &buffer));
    } else {
        panic!("Expected Store op");
    }
}

#[test]
fn test_codegen_param() {
    // Per-kernel codegen PARAM: scalar storage dtype and global address metadata.
    let p = UOp::param(0, 1024, DType::Float32, None);

    assert_eq!(p.dtype(), DType::Float32);
    assert_eq!(p.shape().unwrap().unwrap()[0].as_const(), Some(1024));

    if let Op::Param(ops::Param { arg, .. }) = p.op() {
        assert_eq!(arg.slot, 0);
        assert_eq!(arg.dtype, DType::Float32);
        assert_eq!(arg.addrspace, Some(svod_dtype::AddrSpace::Global));
        assert!(arg.device.is_none());
    } else {
        panic!("Expected Param op");
    }
}

/// INDEX and LOAD take the buffer's scalar element dtype; a shaped index contributes a
/// shape, never lanes in the dtype.
#[test]
fn test_index_and_load_keep_scalar_element_dtype() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 16, DType::Float32);
    let scalar = UOp::index().buffer(buffer.clone()).indices(vec![UOp::index_const(0)]).call().unwrap();
    assert_eq!(scalar.dtype(), DType::Float32);

    let offsets = UOp::stack(smallvec::smallvec![UOp::index_const(0), UOp::index_const(1)]);
    let index = UOp::index().buffer(buffer).indices(vec![offsets]).call().unwrap();
    let load = UOp::load().index(index.clone()).call();

    assert_eq!(index.dtype(), DType::Float32);
    assert_eq!(load.dtype(), DType::Float32);
    assert_eq!(index.shape().unwrap().unwrap().as_slice(), &[crate::SInt::Const(2)]);
    assert_eq!(load.shape().unwrap().unwrap().as_slice(), &[crate::SInt::Const(2)]);
}

#[test]
fn test_local_buffer() {
    let dl = UOp::buffer(1, 4, DType::Int32, AddrSpace::Local, None);

    assert_eq!(dl.dtype(), DType::Int32);

    if let Op::Buffer(ops::Buffer { arg, .. }) = dl.op() {
        assert_eq!(arg.slot, 1);
        assert_eq!(arg.addrspace, Some(AddrSpace::Local));
        assert_eq!(dl.shape().unwrap().unwrap().as_slice(), &[crate::SInt::Const(4)]);
    } else {
        panic!("Expected local Buffer op");
    }
}
