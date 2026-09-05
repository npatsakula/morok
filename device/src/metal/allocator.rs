//! `MTLResourceStorageModeShared` allocations: one unified-memory buffer per
//! allocation, host-visible through `contents`.

use std::ptr::NonNull;
use std::sync::Arc;

use svod_dtype::DeviceSpec;

use super::device::MetalDevice;
use super::objc::{AutoreleasePool, Id, MTL_RESOURCE_STORAGE_MODE_SHARED, NSUInteger, ObjcId};
use crate::allocator::{Allocator, BufferSpec, RawBuffer};
use crate::error::UnsupportedSnafu;
use crate::{Error, Result};

#[derive(Clone)]
pub struct MetalAllocator {
    pub dev: Arc<MetalDevice>,
    pub device_id: usize,
}

impl MetalAllocator {
    pub fn new(device_id: usize) -> Result<Self> {
        Ok(Self { dev: MetalDevice::open(device_id)?, device_id })
    }

    /// Host pointer at `offset` into a Metal buffer, after draining the device:
    /// shared-storage host access is not ordered against committed work.
    fn host_ptr(&self, buffer: &RawBuffer, offset: usize) -> Result<*mut u8> {
        match buffer {
            RawBuffer::Metal { contents, .. } => {
                self.dev.synchronize()?;
                // SAFETY: the caller bounds `offset` by the allocation.
                Ok(unsafe { contents.as_ptr().add(offset) })
            }
            other => unreachable!("MetalAllocator used with a non-Metal buffer: {other:?}"),
        }
    }
}

impl std::fmt::Debug for MetalAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalAllocator").field("device_id", &self.device_id).field("device", &self.dev).finish()
    }
}

impl Allocator for MetalAllocator {
    /// `BufferSpec` flags are ignored: on Apple silicon every buffer is
    /// unified memory, so shared storage is both device-local and host-visible.
    fn _alloc(&self, size: usize, _options: &BufferSpec, zero: bool) -> Result<RawBuffer> {
        let objc = self.dev.objc();
        let _pool = AutoreleasePool::push(objc);
        let alloc_len = size.max(1);
        // SAFETY: `newBufferWithLength:options:` takes two NSUIntegers; +1 or nil.
        let buffer = unsafe {
            ObjcId::adopt(objc.send2::<NSUInteger, NSUInteger, Id>(
                self.dev.mtl(),
                objc.sels.new_buffer_with_length_options,
                alloc_len as NSUInteger,
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            ))
        }
        .ok_or_else(|| Error::Runtime { message: format!("Metal OOM allocating {size} bytes") })?;
        // SAFETY: `contents` returns the host mapping of a shared buffer.
        let contents = NonNull::new(unsafe { objc.send0::<*mut u8>(buffer.as_raw(), objc.sels.contents) })
            .ok_or_else(|| Error::Runtime { message: "MTLBuffer has no host mapping".into() })?;
        if zero {
            // SAFETY: fresh allocation of `alloc_len` bytes, no GPU work references it yet.
            unsafe { std::ptr::write_bytes(contents.as_ptr(), 0, alloc_len) };
        }
        self.dev.register_buffer(contents.as_ptr() as usize, alloc_len, buffer.clone());
        Ok(RawBuffer::Metal { buffer, contents, size, device: self.dev.clone() })
    }

    fn _free(&self, buffer: RawBuffer, _options: &BufferSpec) {
        let RawBuffer::Metal { contents, size, device, .. } = &buffer else {
            tracing::debug!(?buffer, "MetalAllocator::free called with non-Metal buffer; dropping");
            return;
        };
        // Committed kernels may still read this buffer; a failed drain cannot
        // propagate from a free, so quarantine the allocation instead.
        if let Err(error) = device.synchronize() {
            tracing::warn!(?error, size, "MetalAllocator::free: drain failed; allocation quarantined");
            std::mem::forget(buffer);
            return;
        }
        device.unregister_buffer(contents.as_ptr() as usize);
        drop(buffer);
    }

    fn _copyin(&self, dest: &RawBuffer, dest_off: usize, src: &[u8]) -> Result<()> {
        let dst = self.host_ptr(dest, dest_off)?;
        // SAFETY: the caller bounds `dest_off + src.len()`; host access is
        // exclusive after the drain.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        Ok(())
    }

    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> Result<()> {
        let source = self.host_ptr(src, src_off)?;
        // SAFETY: as `_copyin`.
        unsafe { std::ptr::copy_nonoverlapping(source, dest.as_mut_ptr(), dest.len()) };
        Ok(())
    }

    fn _transfer(&self, dest: &RawBuffer, dest_off: usize, src: &RawBuffer, src_off: usize, sz: usize) -> Result<()> {
        if !matches!((dest, src), (RawBuffer::Metal { .. }, RawBuffer::Metal { .. })) {
            return UnsupportedSnafu { op: "transfer" }.fail();
        }
        let dst = self.host_ptr(dest, dest_off)?;
        let source = self.host_ptr(src, src_off)?;
        // SAFETY: bounded by the caller; `copy` tolerates the overlapping
        // views memory planning can produce.
        unsafe { std::ptr::copy(source, dst, sz) };
        Ok(())
    }

    /// Without this override `Buffer::synchronize()` would be a silent no-op.
    fn synchronize(&self) -> Result<()> {
        self.dev.synchronize()
    }

    fn name(&self) -> &str {
        "METAL"
    }

    fn device_spec(&self) -> DeviceSpec {
        DeviceSpec::Metal { device_id: self.device_id }
    }
}
