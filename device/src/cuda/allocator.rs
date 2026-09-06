//! Device-local allocations by default; host-visible buffers are managed
//! memory (device-resident, migrated on host touch), `host` buffers are
//! pinned host memory mapped into the device. Host ↔ device copies are one
//! synchronous `cuMemcpy` up to the bounce size and stage through the
//! device's pinned bounce buffer on the copy stream above it.

use std::ptr::NonNull;
use std::sync::Arc;

use svod_dtype::DeviceSpec;

use super::device::{CudaDevice, STAGING_BYTES};
use super::sys::{CU_MEM_ATTACH_GLOBAL, CU_MEMHOSTALLOC_DEVICEMAP, CU_MEMHOSTALLOC_PORTABLE, CUdeviceptr};
use crate::allocator::{Allocator, BufferSpec, RawBuffer};
use crate::error::UnsupportedSnafu;
use crate::{Error, Result};

/// What backs a [`RawBuffer::Cuda`] allocation, which decides its free call
/// and its host access path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaMemory {
    /// `cuMemAlloc`: device memory, no host mapping.
    Device,
    /// `cuMemAllocManaged`: one address valid on both sides, resident where
    /// last touched.
    Managed,
    /// `cuMemHostAlloc(DEVICEMAP)`: pinned host memory read by kernels over
    /// the bus.
    Pinned,
}

#[derive(Clone)]
pub struct CudaAllocator {
    pub dev: Arc<CudaDevice>,
    pub device_id: usize,
}

impl CudaAllocator {
    pub fn new(device_id: usize) -> Result<Self> {
        Ok(Self { dev: CudaDevice::open(device_id)?, device_id })
    }

    fn cuda_buffer(&self, buffer: &RawBuffer) -> (CUdeviceptr, Option<NonNull<u8>>, usize, CudaMemory) {
        match buffer {
            RawBuffer::Cuda { device_ptr, host_ptr, size, memory, .. } => (*device_ptr, *host_ptr, *size, *memory),
            other => unreachable!("CudaAllocator used with a non-CUDA buffer: {other:?}"),
        }
    }

    fn alloc_failed(&self, size: usize, error: Error) -> Error {
        let usage =
            self.dev.memory_info().map(|(free, total)| format!(" (free {free} / total {total})")).unwrap_or_default();
        Error::CudaAllocFailed { size, reason: format!("{error}{usage}") }
    }

    /// Copy `len` bytes in bounce-buffer-sized chunks through the device's
    /// pinned staging memory; `step(staging, done, chunk)` moves one chunk and
    /// must leave the staging memory reusable (stream synchronized).
    fn staged(&self, len: usize, mut step: impl FnMut(NonNull<u8>, usize, usize) -> Result<()>) -> Result<()> {
        self.dev.with_staging(|staging, capacity| {
            let mut done = 0;
            while done < len {
                let chunk = capacity.min(len - done);
                step(staging, done, chunk)?;
                done += chunk;
            }
            Ok(())
        })
    }
}

impl std::fmt::Debug for CudaAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaAllocator").field("device_id", &self.device_id).field("device", &self.dev).finish()
    }
}

impl Allocator for CudaAllocator {
    /// `host` → pinned; `cpu_access` → managed (when the device supports
    /// coherent managed access, else plain device memory); otherwise device.
    fn _alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer> {
        let api = self.dev.enter()?;
        let alloc_len = size.max(1);
        let memory = if options.host {
            CudaMemory::Pinned
        } else if options.cpu_access && self.dev.limits().managed_memory {
            CudaMemory::Managed
        } else {
            CudaMemory::Device
        };
        let mut device_ptr: CUdeviceptr = 0;
        let mut host_ptr = None;
        // SAFETY: out-pointers to live slots; flags per `cuda.h`.
        let result = unsafe {
            match memory {
                CudaMemory::Device => (api.mem_alloc)(&mut device_ptr, alloc_len).check("cuMemAlloc"),
                CudaMemory::Managed => {
                    (api.mem_alloc_managed)(&mut device_ptr, alloc_len, CU_MEM_ATTACH_GLOBAL).check("cuMemAllocManaged")
                }
                CudaMemory::Pinned => {
                    let mut raw = std::ptr::null_mut();
                    (api.mem_host_alloc)(&mut raw, alloc_len, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP)
                        .check("cuMemHostAlloc")
                        .and_then(|()| {
                            (api.mem_host_get_device_pointer)(&mut device_ptr, raw, 0)
                                .check("cuMemHostGetDevicePointer")
                                .inspect_err(|_| {
                                    (api.mem_free_host)(raw);
                                })
                        })
                        .map(|()| host_ptr = NonNull::new(raw.cast::<u8>()))
                }
            }
        };
        result.map_err(|error| self.alloc_failed(size, error))?;
        if memory == CudaMemory::Managed {
            host_ptr = NonNull::new(device_ptr as usize as *mut u8);
        }
        let buffer = RawBuffer::Cuda { device_ptr, host_ptr, size, memory, device: Arc::clone(&self.dev) };
        if zero {
            self.dev.zero(device_ptr, alloc_len)?;
        }
        Ok(buffer)
    }

    fn _free(&self, buffer: RawBuffer, _options: &BufferSpec) {
        let RawBuffer::Cuda { device_ptr, host_ptr, size, memory, device } = &buffer else {
            tracing::debug!(?buffer, "CudaAllocator::free called with non-CUDA buffer; dropping");
            return;
        };
        // In-flight kernels may still reference the allocation; a failed
        // drain (or a poisoned context) cannot propagate from a free, so the
        // allocation is quarantined instead.
        if let Err(error) = device.synchronize() {
            tracing::warn!(?error, size, "CudaAllocator::free: drain failed; allocation quarantined");
            std::mem::forget(buffer);
            return;
        }
        let api = device.api();
        // SAFETY: the allocation this buffer owns, freed by the call that made it.
        let result = unsafe {
            match memory {
                CudaMemory::Device | CudaMemory::Managed => (api.mem_free)(*device_ptr),
                CudaMemory::Pinned => (api.mem_free_host)(host_ptr.map_or(std::ptr::null_mut(), |p| p.as_ptr().cast())),
            }
        };
        if let Err(error) = result.check("cuMemFree") {
            tracing::warn!(?error, size, "CudaAllocator::free failed");
        }
    }

    /// Host access is not ordered against the plan streams, so every path
    /// drains the context first. Pinned memory is then a plain `memcpy`;
    /// device and managed memory take one synchronous `cuMemcpyHtoD` up to
    /// the bounce size and chunked pinned staging above it.
    fn _copyin(&self, dest: &RawBuffer, dest_off: usize, src: &[u8]) -> Result<()> {
        let (device_ptr, host_ptr, _, memory) = self.cuda_buffer(dest);
        self.dev.synchronize()?;
        if src.is_empty() {
            return Ok(());
        }
        if let (CudaMemory::Pinned, Some(host)) = (memory, host_ptr) {
            // SAFETY: the caller bounds `dest_off + src.len()` by the allocation.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), host.as_ptr().add(dest_off), src.len()) };
            return Ok(());
        }
        let api = self.dev.api();
        let dst = device_ptr + dest_off as u64;
        if src.len() <= STAGING_BYTES {
            // SAFETY: the destination range is bounded by the caller; the copy
            // retires before the call returns, so `src` is free afterwards.
            return self.dev.check(unsafe { (api.memcpy_htod)(dst, src.as_ptr().cast(), src.len()) }, "cuMemcpyHtoD");
        }
        let stream = self.dev.copy_stream();
        self.staged(src.len(), |staging, done, chunk| {
            // SAFETY: `chunk` fits the bounce buffer; the destination range is
            // bounded by the caller; the copy is waited for before reuse.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr().add(done), staging.as_ptr(), chunk);
                self.dev.check(
                    (api.memcpy_htod_async)(dst + done as u64, staging.as_ptr().cast(), chunk, stream),
                    "cuMemcpyHtoDAsync",
                )?;
            }
            self.dev.stream_synchronize(stream)
        })
    }

    /// Mirror of [`Self::_copyin`].
    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> Result<()> {
        let (device_ptr, host_ptr, _, memory) = self.cuda_buffer(src);
        self.dev.synchronize()?;
        if dest.is_empty() {
            return Ok(());
        }
        if let (CudaMemory::Pinned, Some(host)) = (memory, host_ptr) {
            // SAFETY: as `_copyin`.
            unsafe { std::ptr::copy_nonoverlapping(host.as_ptr().add(src_off), dest.as_mut_ptr(), dest.len()) };
            return Ok(());
        }
        let api = self.dev.api();
        let source = device_ptr + src_off as u64;
        let len = dest.len();
        let out = dest.as_mut_ptr();
        if len <= STAGING_BYTES {
            // SAFETY: the source range is bounded by the caller; `dest` is
            // written before the call returns.
            return self.dev.check(unsafe { (api.memcpy_dtoh)(out.cast(), source, len) }, "cuMemcpyDtoH");
        }
        let stream = self.dev.copy_stream();
        self.staged(len, |staging, done, chunk| {
            // SAFETY: as `_copyin`; the host read happens after the copy retired.
            unsafe {
                self.dev.check(
                    (api.memcpy_dtoh_async)(staging.as_ptr().cast(), source + done as u64, chunk, stream),
                    "cuMemcpyDtoHAsync",
                )?;
                self.dev.stream_synchronize(stream)?;
                std::ptr::copy_nonoverlapping(staging.as_ptr(), out.add(done), chunk);
            }
            Ok(())
        })
    }

    /// Device-to-device: one async copy on the copy stream and one stream
    /// wait. The context is drained first because a plan stream may still be
    /// writing `src` or reading `dest` (no per-storage producer tracking yet).
    /// An overlapping range within one allocation (memory planning) bounces
    /// through a temporary so it keeps memmove semantics.
    fn _transfer(&self, dest: &RawBuffer, dest_off: usize, src: &RawBuffer, src_off: usize, sz: usize) -> Result<()> {
        if !matches!((dest, src), (RawBuffer::Cuda { .. }, RawBuffer::Cuda { .. })) {
            return UnsupportedSnafu { op: "transfer" }.fail();
        }
        let (dst_base, ..) = self.cuda_buffer(dest);
        let (src_base, ..) = self.cuda_buffer(src);
        let dst = dst_base + dest_off as u64;
        let source = src_base + src_off as u64;
        self.dev.synchronize()?;
        if sz == 0 || dst == source {
            return Ok(());
        }
        let api = self.dev.api();
        let stream = self.dev.copy_stream();
        let overlaps = dst < source + sz as u64 && source < dst + sz as u64;
        if overlaps {
            let bounce = self._alloc(sz, &BufferSpec { cpu_access: false, ..BufferSpec::default() }, false)?;
            let (tmp, ..) = self.cuda_buffer(&bounce);
            // SAFETY: three device ranges of `sz` bytes each, bounded by the caller.
            let result = unsafe {
                self.dev
                    .check((api.memcpy_dtod_async)(tmp, source, sz, stream), "cuMemcpyDtoDAsync")
                    .and_then(|()| self.dev.check((api.memcpy_dtod_async)(dst, tmp, sz, stream), "cuMemcpyDtoDAsync"))
                    .and_then(|()| self.dev.stream_synchronize(stream))
            };
            self._free(bounce, &BufferSpec::default());
            return result;
        }
        // SAFETY: two device ranges bounded by the caller.
        self.dev.check(unsafe { (api.memcpy_dtod_async)(dst, source, sz, stream) }, "cuMemcpyDtoDAsync")?;
        self.dev.stream_synchronize(stream)
    }

    fn synchronize(&self) -> Result<()> {
        self.dev.synchronize()
    }

    fn name(&self) -> &str {
        "CUDA"
    }

    fn device_spec(&self) -> DeviceSpec {
        DeviceSpec::Cuda { device_id: self.device_id }
    }

    fn supports_device_local(&self) -> bool {
        true
    }
}
