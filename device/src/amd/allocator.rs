//! `AmdAllocator`: KFD-direct VRAM/GTT allocator.
//!
//! Each `alloc` reserves a host VA
//! via PROT_NONE mmap, hands the VA to `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`,
//! optionally maps it host-visible via `mmap(drm_fd, ...)`, and binds it
//! into the GPU page table with `AMDKFD_IOC_MAP_MEMORY_TO_GPU`.
//!
//! Always-compiled-on-Linux: construction returns `Err(NoAmdGpu)` cleanly on
//! hosts without `/dev/kfd`, so the existence of `AmdAllocator` doesn't gate
//! anything; the runtime AMD path simply isn't reachable.

#![cfg(unix)]

use std::sync::Arc;

use svod_dtype::DeviceSpec;
use tracing::debug;

use crate::allocator::{Allocator, BufferSpec, RawBuffer};
use crate::amd::device::AmdDevice;
use crate::amd::iface::AllocKind;
use crate::amd::va_registry::AllocTag;
use crate::error::{Result, UnsupportedSnafu};

/// VRAM-/GTT-backed buffer allocator routed through KFD ioctls.
pub struct AmdAllocator {
    pub dev: Arc<AmdDevice>,
    pub device_id: usize,
}

impl AmdAllocator {
    /// Open the `device_id`-th KFD GPU node and bind a VM.
    ///
    /// Returns `Err(NoAmdGpu)` cleanly when the host has no AMD GPU, no
    /// `/dev/kfd`, or the index is out of range. Never panics.
    pub fn new(device_id: usize) -> Result<Self> {
        let dev = AmdDevice::open(device_id)?;
        Ok(Self { dev, device_id })
    }

    /// The SDMA copy queue, required by the device-only copy arms. Present
    /// whenever a device-local buffer exists (`_alloc` only drops `cpu_access`
    /// once `has_sdma_queue` is true, which the factory sets after installing
    /// the queue), so this `Err` is effectively unreachable in practice.
    fn copy_queue(&self) -> Result<&Arc<crate::amd::queue::AmdCopyQueue>> {
        self.dev
            .core()
            .copy_queue()
            .ok_or(crate::error::Error::Unsupported { op: "device-only VRAM copy requires the SDMA copy queue" })
    }

    /// Allocate GTT-pinned system memory with `COHERENT | UNCACHED | PUBLIC`
    /// flags — host-visible, uncached, suitable for queue rings, GART pages,
    /// and signal slots. The `uncached` branch sets **GTT**, not VRAM
    /// (uncached and VRAM are mutually exclusive in the flag composition).
    pub fn alloc_uncached(&self, size: usize) -> Result<RawBuffer> {
        do_alloc(&self.dev, size, AllocKind::UncachedGtt, /*cpu_accessible=*/ true, /*zero_init=*/ true)
    }
}

impl std::fmt::Debug for AmdAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdAllocator")
            .field("device_id", &self.device_id)
            .field("arch", &self.dev.arch.mcpu())
            .field("gpu_id", &self.dev.node.gpu_id)
            .finish()
    }
}

impl Allocator for AmdAllocator {
    fn _alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer> {
        // Force cpu_access when there is no SDMA copy queue: without SDMA the
        // only way to move data is the host `memmove` path, which needs a host
        // mapping: `cpu_access = options.cpu_access || !has_sdma_queue`.
        let cpu_access = options.cpu_access || !self.dev.has_sdma_queue();
        // The seam only zero-fills host-mapped buffers and now *rejects*
        // `zero && !cpu_access`. So ask it to zero only when host-mapped; for a
        // device-only buffer pass `zero=false` and SDMA-zero it ourselves below,
        // keeping `zero=true` honored regardless of host visibility.
        let seam_zero = zero && cpu_access;
        let buf = do_alloc(&self.dev, size, AllocKind::DeviceVram, cpu_access, seam_zero)?;
        if zero && let RawBuffer::AmdDevice { host_ptr: None, gpu_addr, size: bsize, .. } = &buf {
            self.copy_queue()?.device_zero(*gpu_addr, *bsize)?;
        }
        Ok(buf)
    }

    fn _copyin(&self, dest: &RawBuffer, dest_off: usize, src: &[u8]) -> Result<()> {
        match dest {
            RawBuffer::AmdDevice { host_ptr: Some(ptr), .. } => {
                // Drain first: a recycled VA may still be referenced by an
                // in-flight kernel (dispatch is async + the LRU recycles
                // without syncing). Direct host writes aren't ordered on the
                // GPU timeline, so synchronize. Matches the no-copy-queue
                // `_copyin` contract.
                self.dev.synchronize()?;
                // SAFETY: BAR-backed VRAM mapping valid for the buffer's lifetime;
                // scheduler exclusivity. `dest_off + src.len()` is bounded by the caller.
                let dst = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().add(dest_off), src.len()) };
                dst.copy_from_slice(src);
                // Coherence: the next compute dispatch reading this VA does a full
                // `acquire_mem` (L2 invalidate) in its prologue, so the host write
                // is visible without extra bookkeeping here.
                Ok(())
            }
            RawBuffer::AmdDevice { host_ptr: None, gpu_addr, .. } => {
                // Device-local VRAM: drain in-flight kernels on the recycled VA,
                // then stage host→device through the SDMA copy queue.
                // PERF: `synchronize()` drains every connector timeline, not just
                // this buffer's producer (RawBuffer has no last-writer timeline).
                // Fine today — final outputs stay host-visible, so the hot readback
                // avoids this arm. Scope to the producing timeline if device-only
                // intermediate transfers ever become hot (see review finding #10).
                self.dev.synchronize()?;
                self.copy_queue()?.host_to_device(gpu_addr + dest_off as u64, src)
            }
            other => unreachable!("AmdAllocator::_copyin on non-AMD buffer: {other:?}"),
        }
    }

    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> Result<()> {
        match src {
            RawBuffer::AmdDevice { host_ptr: Some(ptr), .. } => {
                // Dispatch is async (`AmdProgram::execute` does not block), so
                // drain the device timeline before reading GPU-written results.
                self.dev.synchronize()?;
                let src_slice = unsafe { std::slice::from_raw_parts(ptr.as_ptr().add(src_off), dest.len()) };
                dest.copy_from_slice(src_slice);
                Ok(())
            }
            RawBuffer::AmdDevice { host_ptr: None, gpu_addr, .. } => {
                // Drain the producing kernels, then stage device→host through
                // the SDMA copy queue (which host-waits each chunk before the
                // memcpy out).
                self.dev.synchronize()?;
                self.copy_queue()?.device_to_host(dest, gpu_addr + src_off as u64)
            }
            other => unreachable!("AmdAllocator::_copyout on non-AMD buffer: {other:?}"),
        }
    }

    fn _transfer(&self, dest: &RawBuffer, dest_off: usize, src: &RawBuffer, src_off: usize, sz: usize) -> Result<()> {
        match (dest, src) {
            (
                RawBuffer::AmdDevice { host_ptr: Some(dst_ptr), .. },
                RawBuffer::AmdDevice { host_ptr: Some(src_ptr), .. },
            ) => {
                // Drain before the host memmove: `src` may still be written by
                // an in-flight kernel and `dst` may be an in-flight target.
                // Host pointer access isn't ordered on any GPU timeline, so we
                // fence the whole device first — same contract as `_copyin`/
                // `_copyout`.
                self.dev.synchronize()?;
                let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr.as_ptr().add(dest_off), sz) };
                let src_slice = unsafe { std::slice::from_raw_parts(src_ptr.as_ptr().add(src_off), sz) };
                dst.copy_from_slice(src_slice);
                // Coherence handled by the consumer dispatch's full L2 acquire
                // prologue (cf. `_copyin`).
                Ok(())
            }
            (RawBuffer::AmdDevice { gpu_addr: dst_gpu, .. }, RawBuffer::AmdDevice { gpu_addr: src_gpu, .. }) => {
                // At least one side is device-only VRAM (no host mapping):
                // direct device→device SDMA copy (both VAs always exist).
                self.dev.synchronize()?;
                self.copy_queue()?.device_to_device(dst_gpu + dest_off as u64, src_gpu + src_off as u64, sz)
            }
            _ => UnsupportedSnafu { op: "transfer" }.fail(),
        }
    }

    fn _free(&self, buffer: RawBuffer, _options: &BufferSpec) {
        let (gpu_addr, host_ptr, size, handle, device) = match buffer {
            RawBuffer::AmdDevice { gpu_addr, host_ptr, size, handle, device } => {
                (gpu_addr, host_ptr, size, handle, device)
            }
            // Wrong-allocator-for-buffer-type would be a programming bug.
            // Falling through means we leak the buffer; CPU/CUDA arms would
            // just drop their backing storage.
            other => {
                debug!(?other, "AmdAllocator::free called with non-AMD buffer; dropping");
                return;
            }
        };
        // 0. Drain the device's submitted work before tearing down the
        //    mapping. `device.synchronize()` → `core.synchronize_all()` drains
        //    EVERY connector registered on this core — this is the per-VM
        //    fence: all queues share one page table, so unmapping `gpu_addr`
        //    while ANY queue's CP still references it faults the whole VM.
        //    Draining all timelines guarantees every in-flight reader (on any
        //    connector) has retired before the unmap. Logged-and-ignore
        //    on failure: free is called from `Drop`, so we can't propagate.
        if let Err(e) = device.synchronize() {
            tracing::warn!(?e, gpu_addr, "AmdAllocator::free: device synchronize failed; freeing anyway");
        }
        // The unmap + munmap + free (host or PROT_NONE reservation share the
        // same VA region) is the backend's job.
        let _ = host_ptr;
        device.core().iface().free_raw(gpu_addr, size, handle);
    }

    /// Drain all in-flight GPU work on this device. Without this override the
    /// trait default is a no-op, so `Buffer::synchronize()` (which delegates
    /// to `allocator.synchronize()`) would silently NOT fence the AMD timeline
    /// — e.g. `Buffer::copy_from`'s cross-device staging read relies on it.
    fn synchronize(&self) -> Result<()> {
        self.dev.synchronize()
    }

    fn name(&self) -> &str {
        "AMD"
    }

    fn device_spec(&self) -> DeviceSpec {
        DeviceSpec::Amd { device_id: self.device_id }
    }

    /// AMD keeps intermediates device-local: the SDMA copy queue provides the
    /// host↔device and device→device copies the scheduler needs for buffers
    /// allocated with `cpu_access: false`. Only meaningful once an SDMA queue is
    /// installed (otherwise `_alloc` forces `cpu_access`), but the placement
    /// decision is safe regardless — a host-mapped fallback still works.
    fn supports_device_local(&self) -> bool {
        true
    }
}

/// Shared body for VRAM and GTT allocations. The KFD ioctls (VA reservation,
/// alloc, host mmap, map_memory_to_gpu) live behind the backend seam; this just
/// attaches the owning `AmdDevice` to the result so `RawBuffer::AmdDevice` can
/// keep the KFD/DRM fds alive for the buffer's lifetime.
fn do_alloc(
    dev: &Arc<AmdDevice>,
    size: usize,
    kind: AllocKind,
    cpu_accessible: bool,
    zero_init: bool,
) -> Result<RawBuffer> {
    // Diagnostic tag for the VA registry: scratch is tagged at its own call
    // site (`device::alloc_scratch`); everything routed through here is either a
    // VRAM data/code/kernarg buffer or GTT control memory.
    let tag = match kind {
        AllocKind::DeviceVram => AllocTag::Vram,
        AllocKind::UncachedGtt => AllocTag::Gtt,
    };
    let r = dev.core().iface().alloc_raw(size, kind, tag, cpu_accessible, zero_init)?;
    Ok(RawBuffer::AmdDevice {
        gpu_addr: r.gpu_va,
        host_ptr: r.host_ptr,
        size: r.size,
        handle: r.handle,
        device: Arc::clone(dev),
    })
}
