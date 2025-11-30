//! Tests for cycle detection in kernel splitting.
//!
//! Validates that find_bufs correctly detects buffer access conflicts and that
//! as_buf properly extracts buffers from wrapper operations.

use std::rc::Rc;

use morok_device::DeviceSpec;
use morok_dtype::DType;
use morok_ir::{UOp, UOpKey};

use crate::rangeify::cycle_detection::{OpAccessType, as_buf, find_bufs};

#[test]
fn test_find_bufs_store_only() {
    // Create a kernel that only STOREs to a buffer
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store = UOp::store(buffer.clone(), index, value);

    // Should succeed - only STORE access
    #[allow(clippy::mutable_key_type)]
    let buf_accesses = find_bufs(&store);

    // Verify we found one buffer with STORE access
    assert_eq!(buf_accesses.len(), 1);
    let buf_key = UOpKey(buffer.clone());
    assert_eq!(buf_accesses.get(&buf_key), Some(&OpAccessType::Store));
}

#[test]
fn test_find_bufs_load_only() {
    // Create a computation that only LOADs from a buffer
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index_const(0);
    let loaded = UOp::load(buffer.clone(), index.clone());

    // Wrap in a STORE to a different buffer (kernel output)
    let out_buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let store = UOp::store(out_buffer.clone(), index, loaded);

    // Should succeed - input buffer only LOADed, output buffer only STOREd
    #[allow(clippy::mutable_key_type)]
    let buf_accesses = find_bufs(&store);

    // Verify we found two buffers with correct access types
    assert_eq!(buf_accesses.len(), 2);
    let in_buf_key = UOpKey(buffer.clone());
    let out_buf_key = UOpKey(out_buffer.clone());
    assert_eq!(buf_accesses.get(&in_buf_key), Some(&OpAccessType::Load));
    assert_eq!(buf_accesses.get(&out_buf_key), Some(&OpAccessType::Store));
}

#[test]
#[should_panic(expected = "buffer accessed with conflicting ops")]
fn test_find_bufs_conflicting_access() {
    // Create a kernel that both LOADs and STOREs to the same buffer
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index_const(0);

    // First LOAD from the buffer
    let loaded = UOp::load(buffer.clone(), index.clone());

    // Then STORE back to the same buffer (conflict!)
    let store = UOp::store(buffer.clone(), index, loaded);

    // Should panic with "buffer accessed with conflicting ops"
    find_bufs(&store);
}

#[test]
fn test_find_bufs_multiple_buffers() {
    // Create a kernel with multiple input buffers and one output
    let buf1 = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let buf2 = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let out_buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index_const(0);

    // LOAD from both input buffers
    let load1 = UOp::load(buf1.clone(), index.clone());
    let load2 = UOp::load(buf2.clone(), index.clone());

    // Add them together
    let sum = load1.try_add_op(&load2).unwrap();

    // STORE to output buffer
    let store = UOp::store(out_buf.clone(), index, sum);

    // Should succeed
    #[allow(clippy::mutable_key_type)]
    let buf_accesses = find_bufs(&store);

    // Verify all three buffers tracked correctly
    assert_eq!(buf_accesses.len(), 3);
    assert_eq!(buf_accesses.get(&UOpKey(buf1.clone())), Some(&OpAccessType::Load));
    assert_eq!(buf_accesses.get(&UOpKey(buf2.clone())), Some(&OpAccessType::Load));
    assert_eq!(buf_accesses.get(&UOpKey(out_buf.clone())), Some(&OpAccessType::Store));
}

#[test]
fn test_find_bufs_gated_operations() {
    // Test with LoadGated and StoreGated
    let in_buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let out_buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index_const(0);
    let gate = UOp::native_const(true);

    // LoadGated from input
    let loaded = UOp::load_gated(in_buf.clone(), index.clone(), gate.clone());

    // StoreGated to output
    let store = UOp::store_gated(out_buf.clone(), index, loaded, gate);

    // Should succeed
    #[allow(clippy::mutable_key_type)]
    let buf_accesses = find_bufs(&store);

    // Verify both buffers tracked
    assert_eq!(buf_accesses.len(), 2);
    assert_eq!(buf_accesses.get(&UOpKey(in_buf.clone())), Some(&OpAccessType::Load));
    assert_eq!(buf_accesses.get(&UOpKey(out_buf.clone())), Some(&OpAccessType::Store));
}

#[test]
fn test_as_buf_mselect() {
    // Test as_buf extracts buffer from MSelect
    let buffer = UOp::unique(Some(0));
    let mselect = UOp::mselect(buffer.clone(), 0);

    let extracted = as_buf(&mselect);
    assert!(Rc::ptr_eq(&extracted, &buffer));
}

#[test]
fn test_as_buf_mstack() {
    // Test as_buf extracts first buffer from MStack
    let buf1 = UOp::unique(Some(1));
    let buf2 = UOp::unique(Some(2));
    let mstack = UOp::mstack(vec![buf1.clone(), buf2].into());

    let extracted = as_buf(&mstack);
    assert!(Rc::ptr_eq(&extracted, &buf1));
}

#[test]
fn test_as_buf_after() {
    // Test as_buf extracts passthrough from After
    let buffer = UOp::unique(Some(0));
    let computation = UOp::noop();
    let after = UOp::after(buffer.clone(), smallvec::smallvec![computation]);

    let extracted = as_buf(&after);
    assert!(Rc::ptr_eq(&extracted, &buffer));
}
