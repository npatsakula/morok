use crate::{Buffer, BufferSpec, CpuAllocator};
use std::sync::Arc;
use svod_dtype::DType;

#[cfg(feature = "cuda")]
use crate::CudaAllocator;

#[test]
fn test_lazy_allocation() {
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::new(allocator, DType::Float32, vec![10], BufferSpec::default());

    assert!(!buffer.is_allocated());
    buffer.ensure_allocated().unwrap();
    assert!(buffer.is_allocated());
}

#[test]
fn test_buffer_alias() {
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![10], BufferSpec::default()).unwrap();

    let view = buffer.view(4, 16).unwrap();
    assert_eq!(view.offset(), 4);
    assert_eq!(view.size(), 16);
}

#[test]
fn test_view_has_distinct_handle_id() {
    // Each Buffer value (including views) carries its own handle id.
    // Disjoint views of one allocation must compare as different handles
    // so the parallel-hazard model can treat them as independent.
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![16], BufferSpec::default()).unwrap();
    let view_a = buffer.view(0, 16).unwrap();
    let view_b = buffer.view(16, 16).unwrap();

    assert_ne!(buffer.id(), view_a.id(), "view must have a fresh handle id distinct from its base");
    assert_ne!(view_a.id(), view_b.id(), "two distinct views must have distinct handle ids");
}

#[test]
fn test_view_shares_storage_id() {
    // Storage identity must be shared between a base and its views; this is
    // what alias detection in the memory planner relies on.
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![16], BufferSpec::default()).unwrap();
    let view = buffer.view(8, 16).unwrap();

    assert_eq!(buffer.storage_id(), view.storage_id(), "view must share its base's storage id");
}

#[test]
fn test_independent_buffers_have_distinct_storage_ids() {
    let allocator = Arc::new(CpuAllocator);
    let a = Buffer::allocate(allocator.clone(), DType::Float32, vec![8], BufferSpec::default()).unwrap();
    let b = Buffer::allocate(allocator, DType::Float32, vec![8], BufferSpec::default()).unwrap();

    assert_ne!(a.storage_id(), b.storage_id(), "independent allocations must have distinct storage ids");
    assert_ne!(a.id(), b.id());
}

#[test]
fn test_invalid_view() {
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![10], BufferSpec::default()).unwrap();

    // Try to create a view that exceeds buffer size
    let result = buffer.view(36, 16);
    assert!(result.is_err());
}

fn byte_buffer(allocator: Arc<CpuAllocator>, len: usize) -> Buffer {
    Buffer::allocate(allocator, DType::UInt8, vec![len], BufferSpec::default()).unwrap()
}

#[test]
fn test_copyin_at_writes_region_and_checks_bounds() {
    let mut buffer = byte_buffer(Arc::new(CpuAllocator), 8);
    buffer.copyin(&[0; 8]).unwrap();

    buffer.copyin_at(3, &[1, 2, 3]).unwrap();
    let mut actual = [0; 8];
    buffer.copyout(&mut actual).unwrap();
    assert_eq!(actual, [0, 0, 0, 1, 2, 3, 0, 0]);

    assert!(buffer.copyin_at(7, &[1, 2]).is_err());
    assert!(buffer.copyin_at(usize::MAX, &[1]).is_err());
}

#[test]
fn test_copy_region_from_copies_partial_regions_and_checks_both_bounds() {
    let allocator = Arc::new(CpuAllocator);
    let mut src = byte_buffer(allocator.clone(), 8);
    let mut dst = byte_buffer(allocator, 8);
    src.copyin(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
    dst.copyin(&[9; 8]).unwrap();

    dst.copy_region_from(2, &src, 3, 3).unwrap();
    let mut actual = [0; 8];
    dst.copyout(&mut actual).unwrap();
    assert_eq!(actual, [9, 9, 3, 4, 5, 9, 9, 9]);

    assert!(dst.copy_region_from(6, &src, 0, 3).is_err());
    assert!(dst.copy_region_from(0, &src, 7, 2).is_err());
    assert!(dst.copy_region_from(usize::MAX, &src, 0, 1).is_err());
}

#[test]
fn test_copy_within_allows_non_overlapping_regions_and_rejects_overlap() {
    let mut buffer = byte_buffer(Arc::new(CpuAllocator), 8);
    buffer.copyin(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();

    buffer.copy_within(4, 0, 4).unwrap();
    let mut actual = [0; 8];
    buffer.copyout(&mut actual).unwrap();
    assert_eq!(actual, [0, 1, 2, 3, 0, 1, 2, 3]);

    assert!(buffer.copy_within(2, 0, 4).is_err());
    assert!(buffer.copy_within(7, 0, 2).is_err());
    assert!(buffer.copy_within(0, usize::MAX, 1).is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn test_unified_memory_allocation() {
    let allocator = match CudaAllocator::new(0) {
        Ok(alloc) => Arc::new(alloc),
        Err(_) => {
            eprintln!("CUDA not available, skipping test");
            return;
        }
    };

    let options = BufferSpec { cpu_access: true, ..Default::default() };
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![10], options).unwrap();

    assert!(buffer.is_allocated());
    assert!(buffer.allocator().name() == "CUDA");
}

#[cfg(feature = "cuda")]
#[test]
fn test_unified_memory_cpu_access() {
    let allocator = match CudaAllocator::new(0) {
        Ok(alloc) => Arc::new(alloc),
        Err(_) => {
            eprintln!("CUDA not available, skipping test");
            return;
        }
    };

    let options = BufferSpec { cpu_access: true, ..Default::default() };
    let mut buffer = Buffer::allocate(allocator, DType::Float32, vec![10], options).unwrap();

    // Write data from CPU
    let input_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let input_bytes: &[u8] = unsafe { std::slice::from_raw_parts(input_data.as_ptr() as *const u8, 40) };
    buffer.copyin(input_bytes).unwrap();

    // Read data back to CPU
    let mut output_data = vec![0f32; 10];
    let output_bytes: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(output_data.as_mut_ptr() as *mut u8, 40) };
    buffer.copyout(output_bytes).unwrap();

    // Verify data
    assert_eq!(input_data, output_data);
}

#[cfg(feature = "cuda")]
#[test]
fn test_unified_memory_view() {
    let allocator = match CudaAllocator::new(0) {
        Ok(alloc) => Arc::new(alloc),
        Err(_) => {
            eprintln!("CUDA not available, skipping test");
            return;
        }
    };

    let options = BufferSpec { cpu_access: true, ..Default::default() };
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![10], options).unwrap();

    // Create view into unified buffer
    let view = buffer.view(8, 16).unwrap();
    assert_eq!(view.offset(), 8);
    assert_eq!(view.size(), 16);
}

#[cfg(feature = "cuda")]
#[test]
fn test_copy_device_to_unified() {
    let allocator = match CudaAllocator::new(0) {
        Ok(alloc) => Arc::new(alloc),
        Err(_) => {
            eprintln!("CUDA not available, skipping test");
            return;
        }
    };

    // Create device-only buffer
    let device_opts = BufferSpec { cpu_access: false, ..Default::default() };
    let mut device_buf = Buffer::allocate(allocator.clone(), DType::Float32, vec![10], device_opts).unwrap();

    // Create unified buffer
    let unified_opts = BufferSpec { cpu_access: true, ..Default::default() };
    let mut unified_buf = Buffer::allocate(allocator, DType::Float32, vec![10], unified_opts).unwrap();

    // Write test data to device buffer
    let input_data: Vec<f32> = (0..10).map(|i| i as f32 * 2.0).collect();
    let input_bytes: &[u8] = unsafe { std::slice::from_raw_parts(input_data.as_ptr() as *const u8, 40) };
    device_buf.copyin(input_bytes).unwrap();

    // Copy from device to unified
    unified_buf.copy_from(&device_buf).unwrap();

    // Read from unified buffer via CPU
    let mut output_data = vec![0f32; 10];
    let output_bytes: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(output_data.as_mut_ptr() as *mut u8, 40) };
    unified_buf.copyout(output_bytes).unwrap();

    // Verify data
    assert_eq!(input_data, output_data);
}

#[cfg(feature = "cuda")]
#[test]
fn test_copy_unified_to_device() {
    let allocator = match CudaAllocator::new(0) {
        Ok(alloc) => Arc::new(alloc),
        Err(_) => {
            eprintln!("CUDA not available, skipping test");
            return;
        }
    };

    // Create unified buffer
    let unified_opts = BufferSpec { cpu_access: true, ..Default::default() };
    let mut unified_buf = Buffer::allocate(allocator.clone(), DType::Float32, vec![10], unified_opts).unwrap();

    // Create device-only buffer
    let device_opts = BufferSpec { cpu_access: false, ..Default::default() };
    let mut device_buf = Buffer::allocate(allocator, DType::Float32, vec![10], device_opts).unwrap();

    // Write test data to unified buffer
    let input_data: Vec<f32> = (0..10).map(|i| i as f32 * 3.0).collect();
    let input_bytes: &[u8] = unsafe { std::slice::from_raw_parts(input_data.as_ptr() as *const u8, 40) };
    unified_buf.copyin(input_bytes).unwrap();

    // Copy from unified to device
    device_buf.copy_from(&unified_buf).unwrap();

    // Read from device buffer
    let mut output_data = vec![0f32; 10];
    let output_bytes: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(output_data.as_mut_ptr() as *mut u8, 40) };
    device_buf.copyout(output_bytes).unwrap();

    // Verify data
    assert_eq!(input_data, output_data);
}

#[cfg(feature = "cuda")]
#[test]
fn test_copy_unified_to_unified() {
    let allocator = match CudaAllocator::new(0) {
        Ok(alloc) => Arc::new(alloc),
        Err(_) => {
            eprintln!("CUDA not available, skipping test");
            return;
        }
    };

    let options = BufferSpec { cpu_access: true, ..Default::default() };
    let mut src_buf = Buffer::allocate(allocator.clone(), DType::Float32, vec![10], options).unwrap();
    let mut dst_buf = Buffer::allocate(allocator, DType::Float32, vec![10], options).unwrap();

    // Write test data to source
    let input_data: Vec<f32> = (0..10).map(|i| i as f32 + 5.0).collect();
    let input_bytes: &[u8] = unsafe { std::slice::from_raw_parts(input_data.as_ptr() as *const u8, 40) };
    src_buf.copyin(input_bytes).unwrap();

    // Copy unified to unified (uses direct CPU access)
    dst_buf.copy_from(&src_buf).unwrap();

    // Read from destination
    let mut output_data = vec![0f32; 10];
    let output_bytes: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(output_data.as_mut_ptr() as *mut u8, 40) };
    dst_buf.copyout(output_bytes).unwrap();

    // Verify data
    assert_eq!(input_data, output_data);
}

#[cfg(feature = "cuda")]
#[test]
fn test_unified_memory_zero_init() {
    let allocator = match CudaAllocator::new(0) {
        Ok(alloc) => Arc::new(alloc),
        Err(_) => {
            eprintln!("CUDA not available, skipping test");
            return;
        }
    };

    let options = BufferSpec { cpu_access: true, ..Default::default() };
    let buffer =
        Buffer::allocate_with_zero_init(allocator, DType::Float32, vec![10], options, /*zero_init=*/ true).unwrap();

    // Read data and verify it's zeroed
    let mut output_data = vec![1f32; 10]; // Initialize with non-zero
    let output_bytes: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(output_data.as_mut_ptr() as *mut u8, 40) };
    buffer.copyout(output_bytes).unwrap();

    // All values should be zero
    assert_eq!(output_data, vec![0f32; 10]);
}

#[test]
fn test_fork_views_preserves_geometry_and_contents() {
    let allocator = Arc::new(CpuAllocator);
    let base = Buffer::allocate(allocator, DType::Float32, vec![8], BufferSpec::default()).unwrap();
    let mut head = base.view(0, 16).unwrap();
    let mut tail = base.view(16, 16).unwrap();
    head.copyin(&[1u8; 16]).unwrap();
    tail.copyin(&[2u8; 16]).unwrap();

    // Snapshot fork: ONE fresh storage, every view re-minted at its offset
    // with the original bytes.
    let forked = Buffer::fork_views(&[&head, &tail], true).unwrap();
    assert_eq!(forked.len(), 2);
    assert_eq!(forked[0].storage_id(), forked[1].storage_id(), "views must land on one storage");
    assert_ne!(forked[0].storage_id(), base.storage_id());
    assert_eq!((forked[0].offset(), forked[1].offset()), (0, 16));
    let mut bytes = [0u8; 16];
    forked[1].copyout(&mut bytes).unwrap();
    assert_eq!(bytes, [2u8; 16]);

    // A bare fork shares nothing: writes to it never reach the original.
    let mut bare = Buffer::fork_views(&[&head], false).unwrap();
    bare[0].copyin(&[7u8; 16]).unwrap();
    head.copyout(&mut bytes).unwrap();
    assert_eq!(bytes, [1u8; 16]);
}

#[test]
fn test_fork_views_rejects_mixed_storages() {
    let allocator = Arc::new(CpuAllocator);
    let left = Buffer::allocate(allocator.clone(), DType::Float32, vec![4], BufferSpec::default()).unwrap();
    let right = Buffer::allocate(allocator, DType::Float32, vec![4], BufferSpec::default()).unwrap();
    assert!(Buffer::fork_views(&[&left, &right], false).is_err());
}
