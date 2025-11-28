use crate::{Buffer, BufferOptions, CpuAllocator};
use morok_dtype::DType;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::CudaAllocator;

#[test]
fn test_lazy_allocation() {
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::new(allocator, DType::Float32, vec![10], BufferOptions::default());

    assert!(!buffer.is_allocated());
    buffer.ensure_allocated().unwrap();
    assert!(buffer.is_allocated());
}

#[test]
fn test_buffer_view() {
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![10], BufferOptions::default()).unwrap();

    let view = buffer.view(4, 16).unwrap();
    assert_eq!(view.offset(), 4);
    assert_eq!(view.size(), 16);
}

#[test]
fn test_invalid_view() {
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![10], BufferOptions::default()).unwrap();

    // Try to create a view that exceeds buffer size
    let result = buffer.view(36, 16);
    assert!(result.is_err());
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

    let options = BufferOptions { cpu_accessible: true, zero_init: false };
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

    let options = BufferOptions { cpu_accessible: true, zero_init: false };
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

    let options = BufferOptions { cpu_accessible: true, zero_init: false };
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
    let device_opts = BufferOptions { cpu_accessible: false, zero_init: false };
    let mut device_buf = Buffer::allocate(allocator.clone(), DType::Float32, vec![10], device_opts).unwrap();

    // Create unified buffer
    let unified_opts = BufferOptions { cpu_accessible: true, zero_init: false };
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
    let unified_opts = BufferOptions { cpu_accessible: true, zero_init: false };
    let mut unified_buf = Buffer::allocate(allocator.clone(), DType::Float32, vec![10], unified_opts).unwrap();

    // Create device-only buffer
    let device_opts = BufferOptions { cpu_accessible: false, zero_init: false };
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

    let options = BufferOptions { cpu_accessible: true, zero_init: false };
    let mut src_buf = Buffer::allocate(allocator.clone(), DType::Float32, vec![10], options.clone()).unwrap();
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

    let options = BufferOptions { cpu_accessible: true, zero_init: true };
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![10], options).unwrap();

    // Read data and verify it's zeroed
    let mut output_data = vec![1f32; 10]; // Initialize with non-zero
    let output_bytes: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(output_data.as_mut_ptr() as *mut u8, 40) };
    buffer.copyout(output_bytes).unwrap();

    // All values should be zero
    assert_eq!(output_data, vec![0f32; 10]);
}
