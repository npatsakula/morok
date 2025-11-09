use crate::{Buffer, BufferOptions, CpuAllocator};
use morok_dtype::DType;
use std::sync::Arc;

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
