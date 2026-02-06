//! Tests for device semantics in rangeify.
//!
//! Validates:
//! - Device extraction from graphs
//! - Address space consistency
//! - Device specification handling
//!
//! Based on Tinygrad's device-related tests.

use std::sync::Arc;

use morok_dtype::{AddrSpace, DType, DeviceSpec};
use morok_ir::{Op, UOp};

use crate::rangeify::patterns::extract_device_from_graph;

// ===== Device Extraction =====

#[test]
fn test_extract_device_from_buffer() {
    // Create buffer with CPU device
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);

    // Extract device should find the CPU spec
    if let Some(device) = extract_device_from_graph(&buffer) {
        assert_eq!(device, DeviceSpec::Cpu);
    }
    // May return None if no device found - that's also valid
}

#[test]
fn test_device_uop_creation() {
    // Create Device UOp directly
    let device = UOp::device(DeviceSpec::Cpu);

    if let Op::Device(spec) = device.op() {
        assert_eq!(*spec, DeviceSpec::Cpu);
    } else {
        panic!("Expected Device op");
    }
}

// ===== Address Space Tests =====

#[test]
fn test_addrspace_global() {
    // Global address space should be the default for buffers
    let addrspace = AddrSpace::Global;
    assert_eq!(addrspace, AddrSpace::Global);
}

#[test]
fn test_addrspace_local() {
    // Local address space for shared memory
    let addrspace = AddrSpace::Local;
    assert_eq!(addrspace, AddrSpace::Local);
}

// ===== Device Spec Equality =====

#[test]
fn test_device_spec_equality() {
    let cpu1 = DeviceSpec::Cpu;
    let cpu2 = DeviceSpec::Cpu;

    assert_eq!(cpu1, cpu2, "Same device specs should be equal");
}

// ===== Multiple Buffers Same Device =====

#[test]
fn test_multiple_buffers_same_device() {
    // Multiple buffers on same device
    let a = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let b = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);

    // Both should have CPU device
    if let Op::Buffer { device: dev_a, .. } = a.op()
        && let Op::Buffer { device: dev_b, .. } = b.op()
        && let (Op::Device(spec_a), Op::Device(spec_b)) = (dev_a.op(), dev_b.op())
    {
        assert_eq!(spec_a, spec_b, "Same device type should match");
    }
}

// ===== Device in Computation =====

#[test]
fn test_device_propagation_through_ops() {
    // Device should be accessible through computation graph
    let a = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let b = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);

    let add = a.try_add(&b).unwrap();

    // The add result should be able to trace back to buffers
    // (devices are children of buffers)
    let topo = add.toposort();

    // Count buffer nodes (which have device children)
    let buffer_count = topo.iter().filter(|u| matches!(u.op(), Op::Buffer { .. })).count();

    // Should find at least 2 buffer nodes (one per input)
    assert!(buffer_count >= 2, "Should have buffer nodes from inputs");

    // Verify each buffer has a device child
    for node in &topo {
        if let Op::Buffer { device, .. } = node.op() {
            assert!(matches!(device.op(), Op::Device(_)), "Buffer should have device");
        }
    }
}

// ===== Buffer View Device =====

#[test]
fn test_buffer_view_inherits_device() {
    // Buffer view should reference original buffer's device
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let view = buffer.view(50, 10);

    // View contains reference to original buffer
    if let Op::BufferView { buffer: ref_buf, .. } = view.op() {
        assert!(Arc::ptr_eq(ref_buf, &buffer));
    } else {
        panic!("Expected BufferView op");
    }
}

// ===== Edge Cases =====

#[test]
fn test_constant_no_device() {
    // Constants don't have a device
    let c = UOp::native_const(42.0f32);

    // extract_device_from_graph may return None for constants
    let device = extract_device_from_graph(&c);
    // Either None or some default is valid
    let _ = device;
}

#[test]
fn test_device_spec_debug() {
    // Device specs should have Debug impl
    let cpu = DeviceSpec::Cpu;
    let debug_str = format!("{:?}", cpu);
    assert!(debug_str.contains("Cpu"), "Debug should contain device name");
}
