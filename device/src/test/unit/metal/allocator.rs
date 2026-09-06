use std::sync::Arc;

use svod_dtype::DType;

use super::metal_alloc_or_skip;
use crate::Buffer;
use crate::allocator::{Allocator, BufferSpec, RawBuffer};

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 7 % 251) as u8).collect()
}

#[test]
fn alloc_copyin_copyout_roundtrip() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let buffer = alloc._alloc(4096, &BufferSpec::default(), false).expect("alloc");
    assert!(matches!(buffer, RawBuffer::Metal { size: 4096, .. }));
    assert!(buffer.cpu_accessible());
    let data = pattern(4096);
    alloc._copyin(&buffer, 0, &data).expect("copyin");
    let mut back = vec![0u8; 4096];
    alloc._copyout(&mut back, &buffer, 0).expect("copyout");
    assert_eq!(back, data);
    // Offset copies address a sub-range.
    let mut tail = vec![0u8; 16];
    alloc._copyout(&mut tail, &buffer, 4080).expect("copyout tail");
    assert_eq!(tail, &data[4080..]);
    alloc._free(buffer, &BufferSpec::default());
}

#[test]
fn zero_initializes_fresh_and_recycled_allocations() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let alloc: Arc<dyn Allocator> = Arc::new(crate::allocator::LruAllocator::new(Box::new(alloc)));
    let spec = BufferSpec::default();
    let mut first = Buffer::new_with_zero_init(alloc.clone(), DType::UInt8, vec![256], spec, true);
    first.ensure_allocated().unwrap();
    assert!(first.as_slice::<u8>().unwrap().iter().all(|byte| *byte == 0));
    first.copyin(&pattern(256)).unwrap();
    let recycled_base = first.raw_data_ptr();
    drop(first);
    // The LRU hands the same allocation back; it must be re-zeroed.
    let second = Buffer::new_with_zero_init(alloc, DType::UInt8, vec![256], spec, true);
    second.ensure_allocated().unwrap();
    assert_eq!(second.raw_data_ptr(), recycled_base, "expected LRU reuse");
    assert!(second.as_slice::<u8>().unwrap().iter().all(|byte| *byte == 0));
}

#[test]
fn transfer_copies_between_and_within_buffers() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let spec = BufferSpec::default();
    let src = alloc._alloc(64, &spec, false).unwrap();
    let dst = alloc._alloc(64, &spec, true).unwrap();
    let data = pattern(64);
    alloc._copyin(&src, 0, &data).unwrap();
    alloc._transfer(&dst, 8, &src, 0, 32).unwrap();
    let mut back = vec![0u8; 64];
    alloc._copyout(&mut back, &dst, 0).unwrap();
    assert_eq!(&back[8..40], &data[..32]);
    assert!(back[..8].iter().chain(&back[40..]).all(|byte| *byte == 0));
    // Overlapping views (memory planning) must behave like memmove.
    alloc._transfer(&src, 4, &src, 0, 32).unwrap();
    alloc._copyout(&mut back, &src, 0).unwrap();
    assert_eq!(&back[4..36], &data[..32]);
    alloc._free(src, &spec);
    alloc._free(dst, &spec);
}

#[test]
fn free_unregisters_the_host_pointer() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let buffer = alloc._alloc(128, &BufferSpec::default(), false).unwrap();
    let RawBuffer::Metal { contents, .. } = &buffer else { unreachable!() };
    let pointer = contents.as_ptr();
    assert!(alloc.dev.resolve(pointer).is_ok());
    assert_eq!(alloc.dev.resolve(unsafe { pointer.add(127) }).unwrap().1, 127);
    alloc._free(buffer, &BufferSpec::default());
    assert!(alloc.dev.resolve(pointer).is_err());
}

#[test]
fn buffer_views_expose_typed_host_slices() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let alloc: Arc<dyn Allocator> = Arc::new(alloc);
    let mut buffer = Buffer::new(alloc, DType::Float32, vec![8], BufferSpec::default());
    let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
    buffer.copyin(bytemuck_bytes(&values)).unwrap();
    assert_eq!(buffer.as_slice::<f32>().unwrap(), &values[..]);
    let view = buffer.view(8, 8).unwrap();
    assert_eq!(view.as_slice::<f32>().unwrap(), &values[2..4]);
    assert_eq!(unsafe { view.as_raw_ptr() }, unsafe { buffer.as_raw_ptr().add(8) });
}

fn bytemuck_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; the slice is re-viewed byte-wise.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}
