//! AMD KFD-direct command queues.
//!
//! - [`AmdComputeQueue`]: 16 MiB AQL ring, doorbell-driven kernel dispatch.
//! - [`AmdCopyQueue`]: SDMA queue for device↔device / device↔host copies.
//!
//! Both share the same KFD `AMDKFD_IOC_CREATE_QUEUE` mechanism but use
//! different `queue_type` codes. AQL packets are 64 bytes (`HsaKernelDispatchPacket`
//! + `HsaBarrierAndPacket`); SDMA submissions are raw dword sequences.
//!
//! Dispatch goes through `dispatch_pm4` (single-XCC PM4 ring, fenced on the
//! `PoolQueue`'s monotonic counter) or the native AQL path (multi-XCC CDNA,
//! completion via per-op signals). `AmdCopyQueue::copy_fenced` stages
//! host↔device / device↔device copies via SDMA, fenced on its own timeline.

#![cfg(unix)]

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use parking_lot::Mutex;
use tracing::debug;

use crate::allocator::{Allocator, BufferSpec};

use crate::amd::AmdAllocator;
use crate::amd::connector::PoolQueue;
use crate::amd::device::AmdDeviceCore;
use crate::amd::signal::Timeline;
use crate::amd::sys::hsa::{
    hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM, hsa_kernel_dispatch_packet_t,
    hsa_packet_header_t_HSA_PACKET_HEADER_BARRIER, hsa_packet_header_t_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
    hsa_packet_header_t_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE, hsa_packet_header_t_HSA_PACKET_HEADER_TYPE,
    hsa_packet_type_t_HSA_PACKET_TYPE_BARRIER_AND, hsa_packet_type_t_HSA_PACKET_TYPE_INVALID,
    hsa_packet_type_t_HSA_PACKET_TYPE_VENDOR_SPECIFIC, hsa_signal_t, kernel_dispatch_header,
};
use crate::amd::sys::kfd;
use crate::amd::sys::pm4;
use crate::amd::sys::sdma;
use crate::error::{Error, Result};

/// AQL packets are exactly 64 bytes.
pub const AQL_PACKET_BYTES: usize = 64;
/// Packet-header dword (dword 0) with type = INVALID. Written into a ring slot
/// before the body, and used to pre-fill the ring at creation, so the AQL packet
/// processor treats an unpublished/half-written slot as "not yet produced"
/// rather than latching a torn packet (ROCr fills the ring with this).
const INVALID_AQL_HEADER: u32 = hsa_packet_type_t_HSA_PACKET_TYPE_INVALID << hsa_packet_header_t_HSA_PACKET_HEADER_TYPE;
/// 16 MiB ring — the compute-ring default size.
pub const COMPUTE_RING_BYTES: usize = 16 * 1024 * 1024;
/// SDMA ring is smaller; 1 MiB is plenty for short copy bursts.
pub const COPY_RING_BYTES: usize = 1024 * 1024;

/// Conservative upper bound on the dwords a single PM4 dispatch writes to the
/// ring (wait, HDP flush, acquire_mem, the SET_SH_REG stream, DISPATCH_DIRECT,
/// RELEASE_MEM — a typical dispatch is ~150). Bounds in-flight dispatches so
/// the host can never lap the ring.
const MAX_DISPATCH_DWORDS: usize = 1024;
/// Max un-retired dispatches allowed before back-pressure blocks the host.
/// Chosen so the combined ring footprint stays at half the ring even in the
/// worst case (`* MAX_DISPATCH_DWORDS`), leaving generous margin while still
/// letting the host run thousands of dispatches ahead of the GPU.
const RING_MAX_INFLIGHT: u64 = (COMPUTE_RING_BYTES / 4 / MAX_DISPATCH_DWORDS / 2) as u64;

/// Build an AQL `hsa_barrier_and_packet_t` (64 bytes) with no dependencies and
/// the given `completion_signal` handle. Used as a graph batch's terminator:
/// because each kernel-dispatch packet sets the header BARRIER bit (the chain is
/// serialised), a trailing barrier_and completes — and fires `completion_signal`
/// (countdown decrement) — only once the last kernel in the batch retires.
///
/// Layout: `header`@0, `dep_signal[5]`@8..48 (all 0 = no deps), `completion_signal`@56.
pub fn build_barrier_and(completion_signal: u64) -> [u32; 16] {
    // Compose the 16-bit header in u32 (generated enum consts are `c_uint`):
    // BARRIER_AND type + barrier bit + system-scope acquire/release fences.
    let header: u32 = hsa_packet_type_t_HSA_PACKET_TYPE_BARRIER_AND
        | (1 << hsa_packet_header_t_HSA_PACKET_HEADER_BARRIER)
        | (hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM << hsa_packet_header_t_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
        | (hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM << hsa_packet_header_t_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    let mut p = [0u32; 16];
    p[0] = header; // dw0: header (low 16) + reserved (high 16 = 0)
    p[14] = completion_signal as u32; // completion_signal.handle @ byte 56
    p[15] = (completion_signal >> 32) as u32;
    p
}

/// Build an AQL `hsa_barrier_and_packet_t` (64 bytes) that gates a following
/// dispatch on up to five producer completion signals — WITHOUT the header
/// BARRIER bit. Because the BARRIER bit is clear, this packet does NOT wait for
/// prior packets in the ring to complete: it only blocks until each (non-zero)
/// `deps[i]` signal value reaches 0, so earlier *independent* kernels keep
/// running. The system-scope acquire/release fences make the producers' writes
/// visible to the consumer. Used by DAG-driven graph dispatch to enforce true
/// data dependencies while leaving the per-dispatch BARRIER bit stripped.
///
/// Up to the first five `deps` are written into `dep_signal[0..5]` (dwords
/// 2/3, 4/5, 6/7, 8/9, 10/11); callers chain multiple packets for >5 deps.
/// `completion` (0 = none) at dwords 14/15.
///
/// Layout: `header`@0, `dep_signal[5]`@8..48, `completion_signal`@56.
pub fn build_barrier_and_deps(deps: &[u64], completion: u64) -> [u32; 16] {
    // BARRIER_AND type + system-scope acquire/release fences, but NO barrier bit.
    let header: u32 = hsa_packet_type_t_HSA_PACKET_TYPE_BARRIER_AND
        | (hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM << hsa_packet_header_t_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
        | (hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM << hsa_packet_header_t_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    let mut p = [0u32; 16];
    p[0] = header; // dw0: header (low 16) + reserved (high 16 = 0)
    // dep_signal[i] handle at dwords 2+2*i / 3+2*i (byte 8 + 8*i). Up to 5.
    for (i, &dep) in deps.iter().take(5).enumerate() {
        p[2 + 2 * i] = dep as u32;
        p[3 + 2 * i] = (dep >> 32) as u32;
    }
    p[14] = completion as u32; // completion_signal.handle @ byte 56
    p[15] = (completion >> 32) as u32;
    p
}

/// Build an AQL vendor-specific packet (64 bytes) pointing the AQL packet
/// processor at a PM4 indirect buffer (`pm4_addr`, `pm4_count` dwords). Used to
/// run a PM4 `ACQUIRE_MEM` (full instruction-/scalar-cache + L2 invalidate) on
/// the AQL queue: the AQL header acquire fence covers DATA caches only — NOT the
/// instruction cache — so a code object placed on a recycled VA must be preceded
/// by this explicit invalidate, mirroring ROCr `GpuAgent::InvalidateCodeCaches`
/// at code-object load. The BARRIER bit + system-scope fences serialise it ahead
/// of the following dispatch on every XCC. Unlike the old vendor-IB path, this
/// carries no `RELEASE_MEM`/signal write, so the multi-XCC timeline-slot race
/// that motivated deleting vendor IBs does not apply (broadcast cache-invalidate
/// is idempotent across XCCs).
pub fn build_aql_vendor_ib_packet(pm4_addr: u64, pm4_count: u32) -> [u32; 16] {
    let header: u32 = hsa_packet_type_t_HSA_PACKET_TYPE_VENDOR_SPECIFIC
        | (1 << hsa_packet_header_t_HSA_PACKET_HEADER_BARRIER)
        | (hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM << hsa_packet_header_t_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
        | (hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM << hsa_packet_header_t_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    let mut p = [0u32; 16];
    // dw0: AQL header (low 16) | AMD vendor-IB format count = 1 (high 16).
    p[0] = header | (1 << 16);
    p[1] = pm4::packet3(pm4::PACKET3_INDIRECT_BUFFER, 2);
    p[2] = pm4_addr as u32;
    p[3] = (pm4_addr >> 32) as u32;
    p[4] = pm4_count | pm4::INDIRECT_BUFFER_VALID;
    p[5] = 10; // poll interval
    p
}

/// Pack a kernel-dispatch packet describing a single launch.
///
/// `kernel_object` = GPU VA of the kernel descriptor (from the loaded code
/// object, via `AmdProgram`).
#[allow(clippy::too_many_arguments)]
pub fn build_dispatch_packet(
    workgroup_size: [u16; 3],
    grid_size: [u32; 3],
    private_segment_size: u32,
    group_segment_size: u32,
    kernel_object: u64,
    kernarg_address: u64,
    completion_signal: u64,
) -> hsa_kernel_dispatch_packet_t {
    build_dispatch_packet_barrier(
        workgroup_size,
        grid_size,
        private_segment_size,
        group_segment_size,
        kernel_object,
        kernarg_address,
        completion_signal,
        /*barrier=*/ true,
    )
}

/// Like [`build_dispatch_packet`] but with explicit control over the header
/// BARRIER bit. When `barrier` is `false` the bit is cleared, so the AQL packet
/// processor does NOT wait for all prior packets to COMPLETE before launching
/// this kernel — letting independent kernels overlap. True data dependencies
/// are then carried by preceding `barrier_and` packets (see
/// [`build_barrier_and_deps`]). All other fields are identical to
/// [`build_dispatch_packet`].
#[allow(clippy::too_many_arguments)]
pub fn build_dispatch_packet_barrier(
    workgroup_size: [u16; 3],
    grid_size: [u32; 3],
    private_segment_size: u32,
    group_segment_size: u32,
    kernel_object: u64,
    kernarg_address: u64,
    completion_signal: u64,
    barrier: bool,
) -> hsa_kernel_dispatch_packet_t {
    let dims: u16 = if grid_size[2] > 1 {
        3
    } else if grid_size[1] > 1 {
        2
    } else {
        1
    };
    let mut p = hsa_kernel_dispatch_packet_t::default();
    // `header` (u16) and `setup` (dims, in bits 0-1) share a union with the
    // `full_header` u32; setting the latter writes both in one little-endian
    // store (header = low half, setup = high half). Clear ONLY the BARRIER bit
    // in the low 16 when `barrier == false`; the `dims << 16` high half and the
    // fence-scope bits are preserved.
    let mut full_header = u32::from(kernel_dispatch_header()) | (u32::from(dims) << 16);
    if !barrier {
        full_header &= !(1u32 << hsa_packet_header_t_HSA_PACKET_HEADER_BARRIER);
    }
    p.__bindgen_anon_1.full_header = full_header;
    p.workgroup_size_x = workgroup_size[0];
    p.workgroup_size_y = workgroup_size[1];
    p.workgroup_size_z = workgroup_size[2];
    p.grid_size_x = grid_size[0];
    p.grid_size_y = grid_size[1];
    p.grid_size_z = grid_size[2];
    p.private_segment_size = private_segment_size;
    p.group_segment_size = group_segment_size;
    p.kernel_object = kernel_object;
    p.kernarg_address = kernarg_address as *mut std::os::raw::c_void;
    p.completion_signal = hsa_signal_t { handle: completion_signal };
    p
}

/// Compute queue. Wraps either a `KFD_IOC_QUEUE_TYPE_COMPUTE` (PM4) ring on
/// single-XCC GPUs (gfx11/12 default) or a `KFD_IOC_QUEUE_TYPE_COMPUTE_AQL`
/// ring on multi-XCC CDNA. The two paths share the same KFD setup, doorbell
/// mapping, and submit primitive — the only differences are the packet
/// format we write into the ring and whether the GART contains an
/// `amd_queue_t` AQL descriptor.
///
/// `inner` is guarded by a per-queue `Mutex`: the brief critical section is the
/// packet write + doorbell ring (and the rare scratch-descriptor patch), so a
/// shared `Arc<AmdComputeQueue>` is safe when more owners than queues co-tenant
/// one ring. Cross-queue parallelism comes from the pool holding many queues,
/// each with its own lock — the MES interleaves them on the CP pipes.
pub struct AmdComputeQueue {
    inner: Mutex<QueueInner>,
    /// Immutable device identity (kfd_fd, drm_fd, node, arch, poison latch).
    core: Arc<AmdDeviceCore>,
    /// `true` when this queue submits raw PM4 dwords directly; `false` when
    /// it submits AQL packets (with PM4 wrapped in AQL vendor IB packets).
    /// Decided at queue creation from `num_xcc`, fixed for the queue's lifetime.
    is_pm4: bool,
}

/// Copy queue (SDMA). Stages host↔device and device↔device copies for
/// device-local VRAM buffers. Each [`copy_fenced`](AmdCopyQueue::copy_fenced)
/// is serialised under `inner` and fenced on its own [`Timeline`]: the SDMA
/// `fence` packet writes the timeline value into the GTT signal slot the host
/// busy-polls, so completion needs no interrupt/TRAP.
pub struct AmdCopyQueue {
    inner: Mutex<QueueInner>,
    core: Arc<AmdDeviceCore>,
    timeline: Arc<Timeline>,
    /// Host-visible GTT bounce buffer for host↔device staging. A device-local
    /// VRAM buffer has no host mapping, so `_copyin`/`_copyout` memcpy through
    /// this and DMA the other leg. Locked for the whole chunked transfer so
    /// concurrent copies don't clobber it.
    staging: Mutex<StagingBuf>,
}

struct StagingBuf {
    _buf: crate::allocator::RawBuffer,
    host: NonNull<u8>,
    gpu: u64,
    size: usize,
}

// SAFETY: `host`/`gpu` address a stable GTT mapping owned by `_buf`; all access
// is serialised under `AmdCopyQueue::staging`'s Mutex.
unsafe impl Send for StagingBuf {}

struct QueueInner {
    /// 16 MiB ring buffer; host-visible so we can write packets directly.
    ring_host: NonNull<u8>,
    ring_size: usize,
    /// Per-queue doorbell (`*mut u64` MMIO).
    doorbell: NonNull<u64>,
    /// mmap base of the doorbell page, kept so the queue can `munmap` it on
    /// teardown (each queue maps its own page).
    doorbell_base: NonNull<u8>,
    /// Host pointer to the GART-resident `write_dispatch_id` slot — KFD
    /// reads this in addition to the doorbell. It must be updated before
    /// every doorbell ring. Skipping it makes the GPU's
    /// command processor see the doorbell change but stall on a stale wptr.
    write_ptr_host: NonNull<u64>,
    /// Host base of the GART page (the `AmdQueueT` descriptor). On AQL queues
    /// the scratch fields at fixed offsets are patched here when the
    /// connector's scratch buffer is (re)allocated; see `set_aql_scratch`.
    gart_host: NonNull<u8>,
    /// Host base of the `queue_inactive_signal` amd_signal_t (AQL only). The CP
    /// trap handler writes its exception code into `value` (`+8`) and halts the
    /// queue; reading it tells us WHY a wedge happened (e.g. `0x401`).
    qinactive_host: Option<NonNull<u8>>,
    /// Index of the next packet (in AQL_PACKET_BYTES-sized slots). For SDMA
    /// queues this is the next byte offset; type checks ensure callers don't
    /// confuse them.
    write_idx: u64,
    /// Owned KFD queue id (held for the future destroy ioctl; reading it
    /// inside the queue isn't useful since the ioctl takes it directly).
    #[allow(dead_code)]
    queue_id: u32,
    /// Owned bookkeeping buffers we need to keep alive. The EOP and ctx-save
    /// buffers stay alive for the lifetime of the queue — KFD reads them
    /// asynchronously as part of the compute dispatch hardware state.
    _ring_buf: crate::allocator::RawBuffer,
    _gart_buf: crate::allocator::RawBuffer,
    _eop_buf: Option<crate::allocator::RawBuffer>,
    _ctx_buf: Option<crate::allocator::RawBuffer>,
    _qinactive_buf: Option<crate::allocator::RawBuffer>,
}

// SAFETY: ring/doorbell access goes through Mutex; underlying buffers are
// allocator-owned and stable.
unsafe impl Send for QueueInner {}
unsafe impl Sync for QueueInner {}

impl Drop for QueueInner {
    /// Free the queue's KFD-allocated VRAM/GTT backings. `RawBuffer` itself
    /// has no `Drop` (the existing `AmdAllocator::_free` consumes RawBuffer
    /// by destructure), so a queue dropped directly — as happens for
    /// pool queues — would otherwise leak ~50 MiB of ring + GART +
    /// EOP + ctx-save every time. We call the in-place free path
    /// (`RawBuffer::free_amd_device_in_place`) for each. `AmdComputeQueue::
    /// Drop` has already invoked `kfd_destroy_queue` AND `PoolQueue::Drop`
    /// has drained the queue, so the GPU is idle on these buffers.
    ///
    /// Skipped during panic unwind: `PoolQueue::Drop` and
    /// `AmdComputeQueue::Drop` both skip their drain/destroy on panic, so
    /// the GPU's CP may still be reading the ring/GART. Unmapping them here
    /// would fault the VM mid-unwind and could crash before the panic's
    /// diagnostics flush. Accept the buffer leak — the process is unwinding and
    /// the OS reclaims at exit.
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        self._ring_buf.free_amd_device_in_place();
        self._gart_buf.free_amd_device_in_place();
        if let Some(eop) = self._eop_buf.as_ref() {
            eop.free_amd_device_in_place();
        }
        if let Some(ctx) = self._ctx_buf.as_ref() {
            ctx.free_amd_device_in_place();
        }
    }
}

impl QueueInner {
    /// Append raw PM4 dwords to the ring, wrapping at dword granularity.
    /// `write_idx` is counted in dwords for PM4 queues.
    /// Caller holds the queue lock — this is part of one atomic dispatch.
    ///
    /// Ring overflow is prevented up-stream by `wait_dispatch_headroom` (which
    /// bounds in-flight dispatches via the timeline signal), so a single push
    /// must never exceed the per-dispatch budget.
    fn push_pm4(&mut self, dwords: &[u32]) {
        let ring_dwords = self.ring_size / 4;
        debug_assert!(
            dwords.len() <= MAX_DISPATCH_DWORDS,
            "single dispatch ({} dwords) exceeds MAX_DISPATCH_DWORDS ({MAX_DISPATCH_DWORDS}); \
             raise the bound or lower RING_MAX_INFLIGHT",
            dwords.len(),
        );
        let mut idx = (self.write_idx as usize) % ring_dwords;
        for &dw in dwords {
            // SAFETY: ring_host points to ring_size bytes; idx < ring_dwords.
            unsafe { std::ptr::write_volatile((self.ring_host.as_ptr() as *mut u32).add(idx), dw) };
            idx = (idx + 1) % ring_dwords;
        }
        self.write_idx += dwords.len() as u64;
    }

    /// Write one 64-byte AQL packet at the current slot. `write_idx` counts
    /// 64-byte slots for AQL queues.
    fn push_aql(&mut self, bytes: &[u8]) {
        debug_assert_eq!(bytes.len(), AQL_PACKET_BYTES);
        let off = (self.write_idx as usize * AQL_PACKET_BYTES) % self.ring_size;
        // SAFETY: ring_host is mmapped + size-validated; off bounded by ring_size.
        let dst = unsafe { self.ring_host.as_ptr().add(off) };
        let real_header = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        // ROCr `BlitKernel::PopulateQueue` publish protocol: write the packet body
        // with an INVALID-type header FIRST, fence, then store the real header
        // LAST. On a host-visible device/large-BAR ring the packet processor can
        // otherwise latch a valid-typed packet over a torn body (stale grid_size /
        // kernarg_address / completion_signal) and stall the CP with no fault —
        // rptr frozen on the packet, queue_inactive=0 (the exact wedge we see).
        unsafe {
            // dwords 1..16 (body) then dword 0 = INVALID header.
            std::ptr::copy_nonoverlapping(bytes.as_ptr().add(4), dst.add(4), AQL_PACKET_BYTES - 4);
            std::ptr::write_volatile(dst as *mut u32, INVALID_AQL_HEADER);
        }
        // Body must be globally visible before the real header is published.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        // SAFETY: same slot; single 32-bit store flips the type to valid last.
        unsafe { std::ptr::write_volatile(dst as *mut u32, real_header) };
        self.write_idx += 1;
    }

    /// Publish the current `write_idx` to GART + ring the doorbell. AQL uses
    /// the **last completed** slot (`write_idx - 1`); PM4 uses the **next**
    /// dword (`write_idx`).
    fn ring_doorbell(&self, is_pm4: bool) {
        // GART wptr first: without it KFD sees the doorbell
        // change but reads a stale wptr.
        unsafe { std::ptr::write_volatile(self.write_ptr_host.as_ptr(), self.write_idx) };
        // SeqCst (not Release): on x86 a Release fence is a no-op, so it does NOT
        // drain the CPU write-combining buffer or order the wptr/packet stores
        // ahead of the doorbell MMIO write. SeqCst lowers to an `mfence`, which
        // does. Without it the CP can observe the doorbell while the wptr /
        // ring packet / kernarg stores are still buffered (stale read → wedge).
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let doorbell_value = if is_pm4 { self.write_idx } else { self.write_idx - 1 };
        // SAFETY: doorbell is mmapped MMIO; aligned 64-bit store.
        unsafe { std::ptr::write_volatile(self.doorbell.as_ptr(), doorbell_value) };
    }
}

impl AmdComputeQueue {
    /// Create a compute queue. The queue kind is selected by `is_aql =
    /// xccs > 1`. Single-XCC GPUs (the gfx11/12 default) use the
    /// PM4 path (`KFD_IOC_QUEUE_TYPE_COMPUTE`), submitting raw PM4 dwords
    /// directly into the ring. Multi-XCC CDNA falls back to AQL, where each
    /// dispatch is a 64-byte AQL packet and PM4 helpers are wrapped via
    /// the vendor IB packet.
    /// Predict whether `create` would build a PM4 queue for this device,
    /// WITHOUT allocating anything. Used by `AmdGraph::capture` to skip the
    /// (multi-MiB) per-graph connector build on AQL hardware where the graph
    /// path is unsupported anyway. Same logic as `create`'s `is_pm4` decision.
    pub fn will_use_pm4(core: &AmdDeviceCore) -> bool {
        let force_aql = std::env::var("SVOD_AMD_AQL").ok().map(|s| s != "0").unwrap_or(false);
        !force_aql && core.node.num_xcc.max(1) == 1
    }

    pub fn create(allocator: &AmdAllocator) -> Result<Box<Self>> {
        let core = allocator.dev.core();
        // `SVOD_AMD_AQL=1` forces AQL even on single-XCC, useful for
        // bisecting PM4 vs AQL bring-up issues.
        let is_pm4 = Self::will_use_pm4(core);
        let queue_type = if is_pm4 { kfd::KFD_IOC_QUEUE_TYPE_COMPUTE } else { kfd::KFD_IOC_QUEUE_TYPE_COMPUTE_AQL };
        let inner = create_queue(allocator, queue_type, COMPUTE_RING_BYTES, !is_pm4, /*needs_cwsr=*/ true)?;
        debug!(gpu_id = core.node.gpu_id, num_xcc = core.node.num_xcc, is_pm4 = is_pm4, "AmdComputeQueue created");
        Ok(Box::new(Self { inner: Mutex::new(inner), core: Arc::clone(core), is_pm4 }))
    }

    /// `true` when this queue submits raw PM4 dwords (single-XCC); `false`
    /// for the AQL path. Read by callers in `program.rs` to pick the right
    /// dispatch builder.
    pub fn is_pm4(&self) -> bool {
        self.is_pm4
    }

    /// Block until at most `RING_MAX_INFLIGHT` dispatches are un-retired, so a
    /// host running `wait=false` faster than the GPU can't lap the ring and
    /// overwrite unconsumed packets. Bounds the combined ring footprint to
    /// `RING_MAX_INFLIGHT * MAX_DISPATCH_DWORDS` (half the ring).
    ///
    /// Gates on the queue's PM4 counter SIGNAL — the proven completion
    /// primitive `drain_all` already uses — not the PM4 read pointer (whose
    /// COMPUTE-queue semantics are unreliable, which would deadlock a spin).
    /// The dispatches we wait on were submitted (doorbell rung) in prior calls,
    /// so the GPU will signal them; the wait always makes progress.
    fn wait_dispatch_headroom(&self, pool: &PoolQueue) -> Result<()> {
        let last_reserved = pool.pm4_value().saturating_sub(1);
        if last_reserved > RING_MAX_INFLIGHT {
            let target = last_reserved - RING_MAX_INFLIGHT;
            pool.pm4_signal().wait_signal_value(target, 30_000).inspect_err(|e| self.core.poison(&e.to_string()))?;
        }
        Ok(())
    }

    /// Atomically build + submit one PM4 (single-XCC) kernel dispatch.
    ///
    /// The queue's `inner` lock serializes packet assembly + ring blit +
    /// doorbell; the caller additionally holds `pool.dispatch_lock` across the
    /// whole op so the PM4 counter reservation and ring submission stay ordered
    /// against co-tenant owners on this shared queue.
    ///
    /// Sequence:
    /// `wait(counter, prev) → memory_barrier → exec → signal(counter, next)`.
    /// Returns the counter value this dispatch signals.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_pm4(
        &self,
        pool: &PoolQueue,
        rsrc1: u32,
        rsrc2: u32,
        rsrc3: u32,
        prog_addr: u64,
        enable_private_segment_sgpr: bool,
        user_data: &[u32],
        local: [u32; 3],
        grid: [u32; 3],
        wave32: bool,
        target_major: u32,
    ) -> Result<u64> {
        debug_assert!(self.is_pm4, "dispatch_pm4 called on AQL queue");
        debug_assert!(
            Arc::ptr_eq(&self.core, pool.core()),
            "dispatch_pm4: pool core ≠ queue core (queue gpu_id={}, pool gpu_id={}); \
             cross-device dispatch silently corrupts scratch/counter VAs",
            self.core.node.gpu_id,
            pool.core().node.gpu_id,
        );
        // The caller (`execute_on`) holds `pool.dispatch_lock` across the whole
        // op (kernarg bump + write + this dispatch), so the PM4 counter state
        // and ring stay consistent across the back-pressure and wrap waits
        // against co-tenant owners on this shared queue.
        // Keep the PM4 counter < 2^32 (drain+reset at the watermark) before
        // reserving this dispatch's value.
        pool.ensure_pm4_headroom()?;
        // Ring back-pressure: block if too many dispatches are in flight, so an
        // async (`wait=false`) burst can't lap the ring. Outside the inner lock.
        self.wait_dispatch_headroom(pool)?;
        let counter_addr = pool.pm4_signal().value_addr();
        let scratch_addr = pool.scratch_gpu_va();
        let tmpring_size = pool.tmpring_size();
        // Assemble the full USER_DATA prefix here, under the lock, so the scratch
        // SGPR descriptor (words 0-3) is derived from the SAME `scratch_addr` as
        // the `COMPUTE_DISPATCH_SCRATCH_BASE` register below. Building it in
        // `AmdProgram::execute` (outside the lock) let a concurrent scratch
        // realloc slip in between the two reads, so the descriptor and the
        // register could point at different buffers. The scratch base address
        // is read exactly once and reused for the descriptor and the register.
        let mut full_user_data: Vec<u32> = Vec::with_capacity(user_data.len() + 4);
        if enable_private_segment_sgpr {
            full_user_data.push(scratch_addr as u32);
            full_user_data.push((scratch_addr >> 32) as u32 | (1u32 << 31));
            full_user_data.push(0xFFFF_FFFF);
            full_user_data.push(0x20c1_4000);
        }
        full_user_data.extend_from_slice(user_data);
        let mut g = self.inner.lock();
        let prev = pool.pm4_value().saturating_sub(1);
        let next = pool.next_pm4();

        let mut q: Vec<u32> = Vec::with_capacity(96);
        // wait(counter, prev): no-op on the first dispatch (prev == 0).
        q.extend_from_slice(&pm4::wait_reg_mem(counter_addr, prev as u32, 0xFFFF_FFFF));
        // memory_barrier: HDP flush handshake + a FULL acquire (L2 invalidate),
        // emitted unconditionally per dispatch. It makes host-/SDMA-written inputs
        // visible and invalidates stale L2 regardless of producer. `build_exec_pm4`
        // below then does the narrow per-exec acquire; the full→narrow pair is the
        // standard submit-barrier-then-exec acquire sequence.
        q.extend_from_slice(&pm4::hdp_flush());
        if target_major == 9 {
            q.extend_from_slice(&pm4::acquire_mem_gfx9());
        } else {
            q.extend_from_slice(&pm4::acquire_mem());
        }
        // exec: SET_SH_REG stream + DISPATCH_DIRECT.
        build_exec_pm4(
            &mut q,
            rsrc1,
            rsrc2,
            rsrc3,
            prog_addr,
            &full_user_data,
            scratch_addr,
            tmpring_size,
            local,
            grid,
            wave32,
            target_major,
        );
        // signal(counter, next): RELEASE_MEM after a system-scope cache flush.
        q.extend_from_slice(&pm4::release_mem(
            counter_addr,
            next as u32,
            /*cache_flush=*/ true,
            target_major == 9,
        ));

        g.push_pm4(&q);
        g.ring_doorbell(/*is_pm4=*/ true);
        Ok(next)
    }

    /// Push a pre-built PM4 dword stream into the ring with ONE doorbell — the
    /// primitive behind `AmdHwQueue::submit` (the graph's atomic submit).
    /// Blits `cmds` into the ring, advances the write index, rings the doorbell.
    ///
    /// `dwords` is normally the 4-dword `PACKET3_INDIRECT_BUFFER` reference to
    /// the graph's bound `hw_page`; the CP then runs the whole captured chain
    /// inline.
    ///
    /// With per-connector queues, each graph owns its connector and submits
    /// through ITS OWN ring. The graph's own `comp_queue` `Mutex<AmdHwQueue>`
    /// serialises capture vs replay within one graph; this primitive only
    /// takes the queue's inner lock.
    pub fn submit_dwords(&self, dwords: &[u32]) -> Result<()> {
        debug_assert!(self.is_pm4, "submit_dwords on AQL queue");
        if let Some(err) = self.core.poison_error() {
            return Err(err);
        }
        // Ring contiguity is serialized by the inner `Mutex` (one writer
        // produces a contiguous run + doorbell). No `Release` fence here —
        // `ring_doorbell` already issues its own publication barrier.
        let mut g = self.inner.lock();
        g.push_pm4(dwords);
        g.ring_doorbell(/*is_pm4=*/ true);
        Ok(())
    }

    /// Re-blit a captured sequence of 64-byte AQL packets into the ring with ONE
    /// doorbell — the AQL (multi-XCC) analogue of [`submit_dwords`], used by the
    /// graph replay. Each packet is either a vendor IB (pointing at a captured
    /// PM4 run) or a native kernel-dispatch packet; the AQL packet processor runs
    /// them in order. Back-pressure is the graph's per-replay timeline wait (one
    /// replay in flight), so no `wait_dispatch_headroom` is needed here.
    pub fn submit_aql(&self, packets: &[[u32; 16]]) -> Result<()> {
        debug_assert!(!self.is_pm4, "submit_aql on PM4 queue");
        if let Some(err) = self.core.poison_error() {
            return Err(err);
        }
        // Ring contiguity is serialized by the inner `Mutex` (one writer
        // produces a contiguous run of packets + one doorbell).
        let mut g = self.inner.lock();
        for p in packets {
            g.push_aql(dwords_as_bytes(p));
        }
        g.ring_doorbell(/*is_pm4=*/ false);
        Ok(())
    }

    /// Submit ONE native AQL kernel-dispatch packet: push it + ring the
    /// doorbell, nothing else. No PM4 prologue, no `RELEASE_MEM` epilogue, no
    /// `PRED_EXEC` gating, no timeline reservation — completion is whatever the
    /// packet's own `completion_signal` field points at (the hardware
    /// countdown). The packet's system-scope acquire/release fences handle
    /// coherence and FIFO queue order handles sequencing, so the old vendor-IB
    /// prologue + XCC0-gated RELEASE_MEM are gone.
    pub fn dispatch_aql_native(&self, packet: &hsa_kernel_dispatch_packet_t) -> Result<()> {
        debug_assert!(!self.is_pm4, "dispatch_aql_native on PM4 queue");
        debug_assert_eq!(size_of::<hsa_kernel_dispatch_packet_t>(), AQL_PACKET_BYTES);
        if let Some(err) = self.core.poison_error() {
            return Err(err);
        }
        // Ring write + doorbell are serialized by the inner `Mutex`; the caller
        // (`execute_on`) additionally holds `pool.dispatch_lock` so the packet's
        // kernarg slot is the one just bumped.
        // SAFETY: `hsa_kernel_dispatch_packet_t` is `#[repr(C)]` and exactly
        // `AQL_PACKET_BYTES` (debug-asserted above).
        let bytes = unsafe { std::slice::from_raw_parts(packet as *const _ as *const u8, AQL_PACKET_BYTES) };
        let mut g = self.inner.lock();
        g.push_aql(bytes);
        g.ring_doorbell(/*is_pm4=*/ false);
        Ok(())
    }

    /// Dispatch a kernel whose completion is detected via a TRAILING
    /// `barrier_and` packet (which carries `completion`), NOT the kernel packet's
    /// own `completion_signal` field. On multi-XCC the kernel packet's native
    /// per-call completion_signal intermittently strands (the queue goes idle and
    /// the signal never fires — see memory `amd-multi-xcc-aql-hang`); a
    /// `barrier_and` gates on the kernel's retirement across all XCCs and fires
    /// its signal reliably (the same gated-signal mechanism the graph batch
    /// terminator uses, HW-confirmed to fire once on 8 XCCs). The kernel packet
    /// passed here MUST have `completion_signal = 0`.
    ///
    /// When `ib_gpu != 0`, a one-shot I-cache-invalidate vendor-IB is prepended
    /// (first dispatch of a program; recycled-VA stale-I-cache guard). Everything
    /// goes into ONE atomic ring write + doorbell, in order:
    /// `[icache ACQUIRE_MEM]? -> kernel dispatch -> barrier_and(completion)`.
    /// AQL queues only.
    pub fn dispatch_aql_with_barrier_signal(
        &self,
        ib_gpu: u64,
        ib_dwords: u32,
        packet: &hsa_kernel_dispatch_packet_t,
        completion: u64,
    ) -> Result<()> {
        debug_assert!(!self.is_pm4, "dispatch_aql_with_barrier_signal on PM4 queue");
        // SAFETY: `hsa_kernel_dispatch_packet_t` is `#[repr(C)]`, exactly 64 bytes
        // (= 16 dwords); reinterpret as the dword array `submit_aql` takes.
        let mut kd = [0u32; 16];
        unsafe { std::ptr::copy_nonoverlapping(packet as *const _ as *const u32, kd.as_mut_ptr(), 16) };
        let bar = build_barrier_and(completion);
        if ib_gpu != 0 {
            self.submit_aql(&[build_aql_vendor_ib_packet(ib_gpu, ib_dwords), kd, bar])
        } else {
            self.submit_aql(&[kd, bar])
        }
    }

    /// Patch the AQL `amd_queue_t` scratch descriptor in the GART page. The AQL
    /// packet processor reads private-segment (scratch) config from here, so it
    /// must be refreshed whenever the connector's scratch buffer is allocated or
    /// grown. No-op on PM4 queues, where scratch goes through registers per
    /// dispatch. The caller holds the queue idle (the connector drains its
    /// timeline before a scratch realloc), so no in-flight dispatch can observe
    /// a half-written descriptor.
    pub(crate) fn set_aql_scratch(&self, desc: &crate::amd::device::AqlScratchDesc) {
        if self.is_pm4 {
            return;
        }
        use crate::amd::sys::hsa;
        // Called only while the queue is drained (the caller holds it idle), so
        // the descriptor write can't race a dispatch; locking briefly to read the
        // stable GART base pointer.
        let base = self.inner.lock().gart_host.as_ptr();
        // SAFETY: `base` is the GART page we mmapped; every offset lands inside
        // the 256-byte AmdQueueT descriptor that occupies the page.
        unsafe {
            std::ptr::write_volatile(base.add(hsa::OFFSET_COMPUTE_TMPRING_SIZE) as *mut u32, desc.tmpring_size);
            let rd = base.add(hsa::OFFSET_SCRATCH_RESOURCE_DESCRIPTOR) as *mut u32;
            for (i, w) in desc.resource_descriptor.iter().enumerate() {
                std::ptr::write_volatile(rd.add(i), *w);
            }
            std::ptr::write_volatile(
                base.add(hsa::OFFSET_SCRATCH_BACKING_MEMORY_LOCATION) as *mut u64,
                desc.backing_va,
            );
            std::ptr::write_volatile(
                base.add(hsa::OFFSET_SCRATCH_WAVE64_LANE_BYTE_SIZE) as *mut u32,
                desc.wave64_lane_byte_size,
            );
            // Force the host writes through to the GART page before returning:
            // the page is mapped write-combining, so a CPU `Release` fence alone
            // does not guarantee the command processor sees the new descriptor on
            // the next dispatch. A read-back drains the WC buffers (and serialises
            // with the fence below). Without this, a mid-stream scratch grow
            // intermittently leaves the CP reading the stale descriptor — which,
            // once the old backing is freed, points at unmapped VRAM and wedges
            // the queue with no fault event.
            std::ptr::read_volatile(base.add(hsa::OFFSET_COMPUTE_TMPRING_SIZE) as *const u32);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    }

    /// Read the AQL scratch descriptor back out of the GART page — the bytes the
    /// firmware actually sees. Test-only: validates that `set_aql_scratch` wrote
    /// the right values at the right offsets on real hardware.
    #[cfg(test)]
    pub(crate) fn read_aql_scratch(&self) -> crate::amd::device::AqlScratchDesc {
        use crate::amd::sys::hsa;
        let base = self.inner.lock().gart_host.as_ptr();
        unsafe {
            let rd = base.add(hsa::OFFSET_SCRATCH_RESOURCE_DESCRIPTOR) as *const u32;
            crate::amd::device::AqlScratchDesc {
                tmpring_size: std::ptr::read_volatile(base.add(hsa::OFFSET_COMPUTE_TMPRING_SIZE) as *const u32),
                resource_descriptor: [
                    std::ptr::read_volatile(rd),
                    std::ptr::read_volatile(rd.add(1)),
                    std::ptr::read_volatile(rd.add(2)),
                    std::ptr::read_volatile(rd.add(3)),
                ],
                backing_va: std::ptr::read_volatile(base.add(hsa::OFFSET_SCRATCH_BACKING_MEMORY_LOCATION) as *const u64),
                wave64_lane_byte_size: std::ptr::read_volatile(
                    base.add(hsa::OFFSET_SCRATCH_WAVE64_LANE_BYTE_SIZE) as *const u32
                ),
            }
        }
    }

    /// The CP exception code the trap handler wrote into `queue_inactive_signal`
    /// when it halted the queue (e.g. `0x401` insufficient-scratch), or `None`
    /// if no exception is recorded (value `0`) / on PM4 queues. Used to turn a
    /// blind dispatch timeout into a diagnosable error.
    pub(crate) fn inactive_exception(&self) -> Option<i64> {
        let h = self.inner.lock().qinactive_host?;
        // SAFETY: host-visible amd_signal_t; `value` is the i64 at +8.
        let code = unsafe { std::ptr::read_volatile(h.as_ptr().add(8) as *const i64) };
        (code != 0).then_some(code)
    }
}

impl Drop for AmdComputeQueue {
    /// Destroy the in-kernel KFD compute queue object. Without this, every
    /// `kfd_create_queue` ioctl leaves a queue id permanently registered with
    /// the kernel until process exit — and the per-process compute-queue
    /// limit (typically 32) is over LIFETIME creations, not concurrent ones,
    /// so a long-running process that creates+drops plans (BEAM-style) would
    /// eventually hit the cap with zero live connectors. The userspace ring
    /// / GART / EOP / ctx-save buffers free via the underlying
    /// `RawBuffer::Drop` chain when `self.inner` drops next.
    ///
    /// `PoolQueue::Drop` has already drained the queue before
    /// reaching this point on the happy path. During panic unwind the
    /// pool queue skips its drain to keep teardown bounded — destroying
    /// the KFD queue with in-flight CP work risks a kernel-side fault that
    /// crashes the process before useful diagnostics flush, so we also
    /// skip and accept the queue-id leak (process exit reclaims it).
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        // `&mut self` → exclusive; `get_mut` needs no unsafe.
        let inner = self.inner.get_mut();
        let (queue_id, doorbell_base) = (inner.queue_id, inner.doorbell_base);
        self.core.iface().teardown_ring(queue_id, doorbell_base);
    }
}

impl Drop for AmdCopyQueue {
    fn drop(&mut self) {
        let (queue_id, doorbell_base) = {
            let g = self.inner.lock();
            (g.queue_id, g.doorbell_base)
        };
        self.core.iface().teardown_ring(queue_id, doorbell_base);
    }
}

/// View a `[u32; 16]` AQL packet as its 64 little-endian bytes.
fn dwords_as_bytes(p: &[u32; 16]) -> &[u8] {
    // SAFETY: 16 u32 == 64 bytes, contiguous, any bit pattern valid.
    unsafe { std::slice::from_raw_parts(p.as_ptr() as *const u8, AQL_PACKET_BYTES) }
}

/// Build the PM4 SET_SH_REG + DISPATCH_DIRECT stream for a single-XCC dispatch,
/// appending into `q` (minus SQTT/PMC/dispatch_ptr).
/// The shader entry point is pre-shifted right by 8 (COMPUTE_PGM_LO/HI hold the
/// upper bits of a 256-byte-aligned address). `wave32` comes from
/// `kd.kernel_code_properties & 0x400`; `cs_w32_en` applies to gfx10+ (every
/// non-CDNA arch — RDNA2/3/4), gfx9 (CDNA, wave64) ignores it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_exec_pm4(
    q: &mut Vec<u32>,
    rsrc1: u32,
    rsrc2: u32,
    rsrc3: u32,
    prog_addr: u64,
    user_data: &[u32],
    scratch_addr: u64,
    tmpring_size: u32,
    local: [u32; 3],
    grid: [u32; 3],
    wave32: bool,
    target_major: u32,
) {
    // 1. Pre-dispatch cache-invalidate, narrowed to skip L2 on both arches: the
    //    prologue + the prior dispatch's EOP release already handle L2, so this
    //    only needs the per-CU caches (gfx10+ GCR NO_GLI_GL2; gfx9 CP_COHER
    //    minus the TC/L2 actions).
    if target_major == 9 {
        q.extend_from_slice(&pm4::acquire_mem_gfx9_narrow());
    } else {
        q.extend_from_slice(&pm4::acquire_mem_with(pm4::GCR_FLAGS_NO_GLI_GL2));
    }

    // 2. Shader address: COMPUTE_PGM_LO/HI hold (prog_addr >> 8).
    let prog_shr = prog_addr >> 8;
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_PGM_LO, &[prog_shr as u32, (prog_shr >> 32) as u32]));

    // 3. RSRC1/2 together; RSRC3 separately (gfx9 uses a different SH offset).
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_PGM_RSRC1, &[rsrc1, rsrc2]));
    let rsrc3_reg = if target_major == 9 { pm4::COMPUTE_PGM_RSRC3_GFX9 } else { pm4::COMPUTE_PGM_RSRC3 };
    q.extend(pm4::set_sh_reg(rsrc3_reg, &[rsrc3]));

    // 4. Scratch / tmpring (valid base required for wave init on RDNA3+ even
    //    when SCRATCH_EN=0). COMPUTE_DISPATCH_SCRATCH_BASE is written on ALL
    //    arches incl. gfx10.3 (RDNA2): RDNA2 uses "architected flat scratch" —
    //    LLVM emits scratch accesses relative to FLAT_SCRATCH, which HW
    //    initialises from this register, so it MUST be set. tinygrad ops_amd
    //    writes it unconditionally; the old `has_scratch_base_registers` gate
    //    (gfx11+/CDNA only) was wrong for gfx10.3 and left FLAT_SCRATCH unset.
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_TMPRING_SIZE, &[tmpring_size]));
    let scratch_shr = scratch_addr >> 8;
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_DISPATCH_SCRATCH_BASE_LO, &[scratch_shr as u32, (scratch_shr >> 32) as u32]));

    // 5. Restart points always zero (no preempt-resume).
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_RESTART_X, &[0, 0, 0]));

    // 6. COMPUTE_USER_DATA_0..N — user SGPR pre-load (scratch desc + kernarg ptr),
    //    assembled by the caller.
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_USER_DATA_0, user_data));

    // 7. RESOURCE_LIMITS: 0 = no per-SH wave caps.
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_RESOURCE_LIMITS, &[0]));

    // 8. START_X..NUM_THREAD_Z + 2 reserved (local size in NUM_THREAD_*).
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_START_X, &[0, 0, 0, local[0], local[1], local[2], 0, 0]));

    // 9. Launch.
    let mut di = pm4::DISPATCH_INITIATOR_FORCE_START_AT_000 | pm4::DISPATCH_INITIATOR_COMPUTE_SHADER_EN;
    if target_major != 9 && wave32 {
        di |= pm4::DISPATCH_INITIATOR_CS_W32_EN;
    }
    q.extend_from_slice(&pm4::dispatch_direct(grid, di));

    // 10. CS_PARTIAL_FLUSH so the next dispatch sees clean state.
    q.extend_from_slice(&pm4::event_write(pm4::CS_PARTIAL_FLUSH, pm4::EVENT_INDEX_PARTIAL_FLUSH));
}

/// Per-copy completion-wait timeout. A staging copy that never signals means
/// a wedged SDMA engine — fail loud rather than spin forever.
const COPY_TIMEOUT_MS: u64 = 30_000;

/// Host-visible bounce-buffer size. One SDMA packet covers `SDMA_MAX_COPY_BYTES`
/// (4 MiB), so a staging buffer that size makes each chunk a single copy.
const STAGING_BYTES: usize = sdma::SDMA_MAX_COPY_BYTES;

impl AmdCopyQueue {
    pub fn create(allocator: &AmdAllocator) -> Result<Arc<Self>> {
        let inner = create_queue(
            allocator,
            kfd::KFD_IOC_QUEUE_TYPE_SDMA,
            COPY_RING_BYTES,
            /*aql=*/ false,
            /*needs_cwsr=*/ false,
        )?;
        let core = Arc::clone(allocator.dev.core());
        let signal = core
            .signal_pool()
            .ok_or_else(|| Error::AmdAllocFailed { reason: "copy queue needs the signal pool installed first".into() })?
            .acquire()?;
        let timeline = Timeline::new(Arc::new(signal));
        let staging_buf = allocator.alloc_uncached(STAGING_BYTES)?;
        let (gpu, host) = match &staging_buf {
            crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => return Err(Error::AmdAllocFailed { reason: "staging buffer requires host-visible GTT".into() }),
        };
        let staging = Mutex::new(StagingBuf { _buf: staging_buf, host, gpu, size: STAGING_BYTES });
        Ok(Arc::new(Self { inner: Mutex::new(inner), core, timeline, staging }))
    }

    /// Stage host bytes into device-local VRAM at `dst_gpu`, chunked through the
    /// bounce buffer. Each chunk: memcpy host→staging, DMA staging→device.
    pub fn host_to_device(&self, dst_gpu: u64, src: &[u8]) -> Result<()> {
        let st = self.staging.lock();
        let mut off = 0usize;
        while off < src.len() {
            let n = (src.len() - off).min(st.size);
            // SAFETY: staging host mapping spans `st.size`; `n <= st.size`.
            unsafe { std::ptr::copy_nonoverlapping(src[off..].as_ptr(), st.host.as_ptr(), n) };
            self.copy_fenced(st.gpu, dst_gpu + off as u64, n)?;
            off += n;
        }
        Ok(())
    }

    /// Read device-local VRAM at `src_gpu` into host bytes via the bounce
    /// buffer. Each chunk: DMA device→staging, memcpy staging→host.
    pub fn device_to_host(&self, dst: &mut [u8], src_gpu: u64) -> Result<()> {
        let st = self.staging.lock();
        let mut off = 0usize;
        while off < dst.len() {
            let n = (dst.len() - off).min(st.size);
            self.copy_fenced(src_gpu + off as u64, st.gpu, n)?;
            // SAFETY: staging host mapping spans `st.size`; `n <= st.size`.
            unsafe { std::ptr::copy_nonoverlapping(st.host.as_ptr(), dst[off..].as_mut_ptr(), n) };
            off += n;
        }
        Ok(())
    }

    /// Direct device→device VRAM copy (no staging needed).
    pub fn device_to_device(&self, dst_gpu: u64, src_gpu: u64, size: usize) -> Result<()> {
        self.copy_fenced(src_gpu, dst_gpu, size)
    }

    /// Zero `size` bytes of device-local VRAM at `dst_gpu`. The hardware
    /// zero-init path only covers host-mapped buffers (`iface` skips device-only
    /// VRAM), so a device-local `zero=true` allocation is filled here: zero the
    /// staging buffer once, then DMA it over the destination in chunks.
    pub fn device_zero(&self, dst_gpu: u64, size: usize) -> Result<()> {
        let st = self.staging.lock();
        // Each chunk DMAs at most `min(remaining, st.size)` bytes from the front
        // of staging, so only that many need to be zero — no point memsetting the
        // whole 4 MiB buffer when zeroing a small allocation.
        // SAFETY: staging host mapping spans `st.size`; `size.min(st.size) ≤ st.size`.
        unsafe { std::ptr::write_bytes(st.host.as_ptr(), 0, size.min(st.size)) };
        let mut off = 0usize;
        while off < size {
            let n = (size - off).min(st.size);
            self.copy_fenced(st.gpu, dst_gpu + off as u64, n)?;
            off += n;
        }
        Ok(())
    }

    /// Copy `size` bytes `src` → `dst` (both GPU VAs) and block until the SDMA
    /// engine has finished. Chunks at `SDMA_MAX_COPY_BYTES`, fences the batch on
    /// the queue's timeline, rings the doorbell, then busy-polls the signal slot
    /// the fence writes. The reserve→push→doorbell sequence is serialised under
    /// `inner`; the busy-poll wait runs after the lock is released (so the lock
    /// is never held across a multi-second GPU wait).
    pub fn copy_fenced(&self, src: u64, dst: u64, size: usize) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        {
            let mut g = self.inner.lock();
            let mut off = 0usize;
            while off < size {
                let n = (size - off).min(sdma::SDMA_MAX_COPY_BYTES);
                push_sdma(&mut g, &sdma::copy_linear(src + off as u64, dst + off as u64, n));
                off += n;
            }
            // Reserve + fence the timeline value the host waits on.
            let target = self.timeline.next();
            push_sdma(&mut g, &sdma::fence(self.timeline.value_addr(), target as u32));
            // GART wptr first, then doorbell — same ordering as the compute
            // queue. SDMA doorbell + wptr are byte counters (= write_idx).
            unsafe { std::ptr::write_volatile(g.write_ptr_host.as_ptr(), g.write_idx) };
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            unsafe { std::ptr::write_volatile(g.doorbell.as_ptr(), g.write_idx) };
        }
        // Block on the fence value (== `current() - 1`) outside the lock; `drain`
        // also wrap-resets the 32-bit timeline once past the watermark, with the
        // engine idle. Coherence for the copied data is handled by the consuming
        // compute dispatch's full `acquire_mem` prologue (it invalidates L2), so
        // no extra cache bookkeeping is needed here.
        self.timeline.drain(COPY_TIMEOUT_MS)
    }
}

/// Append SDMA dwords into the byte-indexed copy ring, padding with NOPs
/// (op 0 = zero) to the ring end when a packet would straddle the wrap point —
/// the SDMA engine misparses a torn packet. `write_idx` is a monotonic byte
/// counter; the doorbell publishes it verbatim.
fn push_sdma(g: &mut QueueInner, dwords: &[u32]) {
    let bytes = std::mem::size_of_val(dwords);
    let pos = (g.write_idx as usize) % g.ring_size;
    if pos + bytes > g.ring_size {
        let pad = g.ring_size - pos;
        // SAFETY: ring_host spans ring_size bytes; pos + pad == ring_size.
        unsafe { std::ptr::write_bytes(g.ring_host.as_ptr().add(pos), 0, pad) };
        g.write_idx += pad as u64;
    }
    let pos = (g.write_idx as usize) % g.ring_size;
    // SAFETY: pos + bytes ≤ ring_size after the wrap guard above.
    unsafe { std::ptr::copy_nonoverlapping(dwords.as_ptr() as *const u8, g.ring_host.as_ptr().add(pos), bytes) };
    g.write_idx += bytes as u64;
}

impl std::fmt::Debug for AmdComputeQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdComputeQueue").field("gpu_id", &self.core.node.gpu_id).finish_non_exhaustive()
    }
}

impl std::fmt::Debug for AmdCopyQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdCopyQueue").field("gpu_id", &self.core.node.gpu_id).finish_non_exhaustive()
    }
}

fn create_queue(
    allocator: &AmdAllocator,
    queue_type: u32,
    ring_size: usize,
    aql: bool,
    needs_cwsr: bool,
) -> Result<QueueInner> {
    let dev = allocator.dev.core();
    // Ring + GART are both VRAM with COHERENT | UNCACHED | PUBLIC flags
    // (uncached + cpu-accessible). Using plain VRAM (no UNCACHED) makes
    // KFD reject the create_queue ioctl with EINVAL.
    let ring_buf = allocator.alloc_uncached(ring_size)?;
    let (ring_gpu, ring_host) = match &ring_buf {
        crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => return Err(Error::AmdAllocFailed { reason: "queue ring requires host-visible buffer".into() }),
    };
    // Pre-fill every AQL ring slot's header with INVALID (ROCr does this) so the
    // packet processor never treats an unpublished slot (zeroed = VENDOR_SPECIFIC)
    // as a real packet if it reads ahead. PM4 rings don't use AQL headers.
    if aql {
        for slot in 0..(ring_size / AQL_PACKET_BYTES) {
            // SAFETY: slot*64 < ring_size; dword 0 is the header.
            unsafe {
                std::ptr::write_volatile(
                    ring_host.as_ptr().add(slot * AQL_PACKET_BYTES) as *mut u32,
                    INVALID_AQL_HEADER,
                )
            };
        }
    }
    // GART page holds the AQL queue descriptor (`amd_queue_t`, 256 bytes).
    // rptr/wptr live at fixed offsets inside it; KFD reads the descriptor
    // when wiring up the queue. The GART page is a 0x100-byte uncached,
    // cpu-accessible allocation.
    let gart_buf = allocator.alloc_uncached(0x100)?;
    let (gart_gpu, gart_host) = match &gart_buf {
        crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => return Err(Error::AmdAllocFailed { reason: "GART page requires host-visible buffer".into() }),
    };

    let mut qinactive_buf: Option<crate::allocator::RawBuffer> = None;
    let mut qinactive_host: Option<NonNull<u8>> = None;
    if aql {
        // A host-visible amd_signal_t the CP trap handler writes its exception
        // code into (e.g. 0x401 insufficient-scratch) when it halts the queue.
        // Without a real handle the CP can't report WHY it halted (silent wedge).
        let qi_buf = allocator.alloc_uncached(64)?;
        let (qi_gpu, qi_host) = match &qi_buf {
            crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => {
                return Err(Error::AmdAllocFailed {
                    reason: "queue_inactive_signal requires host-visible buffer".into(),
                });
            }
        };
        // SAFETY: fresh 64-byte buffer. amd_signal_t: kind=USER@0, value=0@8.
        unsafe {
            std::ptr::write_bytes(qi_host.as_ptr(), 0, 64);
            std::ptr::write_volatile(
                qi_host.as_ptr() as *mut i64,
                crate::amd::sys::hsa::amd_signal_kind_t_AMD_SIGNAL_KIND_USER as i64,
            );
        }
        // Initialize the GART descriptor.
        // max_cu_id is total CUs across all XCCs - 1 (cu_cnt*xccs-1).
        let cu_cnt = dev.node.simd_count.max(1) / dev.node.simd_per_cu.max(1);
        let waves_per_cu = dev.node.max_waves_per_simd * dev.node.simd_per_cu;
        // `queue_properties` is the u32 storage field; the generated property
        // constants are `amd_queue_properties_t` (c_int), narrowed here.
        let desc = crate::amd::sys::hsa::amd_queue_t {
            queue_properties: (crate::amd::sys::hsa::amd_queue_properties_t_AMD_QUEUE_PROPERTIES_IS_PTR64
                | crate::amd::sys::hsa::amd_queue_properties_t_AMD_QUEUE_PROPERTIES_ENABLE_PROFILING)
                as u32,
            read_dispatch_id_field_base_byte_offset: crate::amd::sys::hsa::OFFSET_READ_DISPATCH_ID as u32,
            max_cu_id: cu_cnt.saturating_sub(1),
            max_wave_id: waves_per_cu.saturating_sub(1),
            queue_inactive_signal: crate::amd::sys::hsa::hsa_signal_t { handle: qi_gpu },
            ..Default::default()
        };
        // SAFETY: gart_host points to a 4 KiB region we just allocated.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &desc as *const _ as *const u8,
                gart_host.as_ptr(),
                std::mem::size_of::<crate::amd::sys::hsa::amd_queue_t>(),
            );
        }
        qinactive_buf = Some(qi_buf);
        qinactive_host = Some(qi_host);
    }

    // Both AQL and plain COMPUTE queues use the same rptr/wptr offsets — the
    // `amd_queue_t::{read,write}_dispatch_id` byte offsets are passed
    // unconditionally. KFD validates these
    // against the queue type's expected layout; using (0, 8) for plain
    // COMPUTE produces EINVAL.
    let wptr_offset: u64 = crate::amd::sys::hsa::OFFSET_WRITE_DISPATCH_ID as u64;
    let rptr_offset: u64 = crate::amd::sys::hsa::OFFSET_READ_DISPATCH_ID as u64;
    // Compute queues need EOP + ctx-save buffers. Sizing:
    //   ctx_save_restore_size (ioctl arg) = wg_data_size + ctl_stack_size
    //   cwsr_buffer_size (alloc size)     = round_up((ctx_save_restore_size
    //                                          + debug_memory_size) * xccs,
    //                                          PAGESIZE)
    // The buffer is larger than what we
    // tell KFD by `debug_memory_size * xccs` — that overflow region is where
    // KFD writes debug-trap state. Undersizing causes corruption when CWSR
    // fires; oversizing is harmless.
    //
    // EOP and ctx-save are *plain VRAM* (no PUBLIC/COHERENT/UNCACHED flags):
    // they're written by the GPU during preemption and never read from the
    // CPU, so the default allocation flags suffice.
    // SDMA queues take no EOP/ctx-save buffers (CWSR is a compute-shader
    // preemption mechanism) — both are zero for SDMA. Compute queues
    // size them per the CWSR contract below.
    let (eop_buf, ctx_buf, eop_gpu, ctx_gpu, eop_size, ctx_save_restore_size, ctl_stack_size) = if needs_cwsr {
        let (wg_data_size, ctl_stack_size, debug_memory_size) = compute_ctx_sizes(dev);
        let xccs = dev.node.num_xcc.max(1) as usize;
        let ctx_save_restore_size = wg_data_size + ctl_stack_size;
        let cwsr_buffer_size = ((ctx_save_restore_size + debug_memory_size) * xccs).next_multiple_of(0x1000);
        let plain = BufferSpec { cpu_access: false, nolru: true, ..Default::default() };
        let eop_buf = allocator.alloc(0x1000, &plain, /*zero=*/ false)?;
        // ctx-save MUST be host-visible and zeroed: we write the per-XCC CWSR
        // header (`HsaUserContextSaveAreaHeader`) the CP reads on every context
        // save/restore (MES preempts a busy queue as routine runlist scheduling).
        // Without the header, a restore reads garbage `DebugOffset`/`DebugSize`
        // and the queue silently strands (rptr frozen, no fault) — the exact
        // multi-XCC wedge. Mirrors libhsakmt `fill_cwsr_header`.
        let ctx_spec = BufferSpec { cpu_access: true, nolru: true, ..Default::default() };
        let ctx_buf = allocator.alloc(cwsr_buffer_size, &ctx_spec, /*zero=*/ true)?;
        let eop_gpu = match &eop_buf {
            crate::allocator::RawBuffer::AmdDevice { gpu_addr, .. } => *gpu_addr,
            _ => 0,
        };
        let (ctx_gpu, ctx_host) = match &ctx_buf {
            crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => return Err(Error::AmdAllocFailed { reason: "ctx-save buffer requires host-visible buffer".into() }),
        };
        // Per-XCC `HsaUserContextSaveAreaHeader` (40 bytes): DebugOffset@16,
        // DebugSize@20 (ErrorReason@24 / ErrorEventId@32 stay 0 — no event).
        // SAFETY: ctx_host is the zeroed cwsr_buffer_size region; each header sits
        // at `i * ctx_save_restore_size`, well within the buffer.
        unsafe {
            for i in 0..xccs {
                let hdr = ctx_host.as_ptr().add(i * ctx_save_restore_size);
                std::ptr::write_volatile(hdr.add(16) as *mut u32, ((xccs - i) * ctx_save_restore_size) as u32);
                std::ptr::write_volatile(hdr.add(20) as *mut u32, (debug_memory_size * xccs) as u32);
            }
        }
        (Some(eop_buf), Some(ctx_buf), eop_gpu, ctx_gpu, 0x1000u64, ctx_save_restore_size as u32, ctl_stack_size as u32)
    } else {
        (None, None, 0u64, 0u64, 0u64, 0u32, 0u32)
    };
    let _ = aql; // queue_type already encodes AQL vs plain COMPUTE

    // CREATE_QUEUE + doorbell mmap through the backend seam. The ring/GART/EOP/
    // ctx buffers above are allocated by us (above the seam); the iface only
    // activates the HQD (register the queue + map its doorbell).
    let desc = crate::amd::iface::RingDesc {
        ring_gpu,
        gart_gpu,
        wptr_offset,
        rptr_offset,
        eop_gpu,
        eop_size,
        ctx_gpu,
        ctx_save_restore_size,
        ctl_stack_size,
        ring_size,
        gpu_id: dev.node.gpu_id,
        queue_type,
    };
    let qh = dev.iface().setup_ring(&desc)?;
    let queue_id = qh.queue_id;
    let doorbell = qh.doorbell;
    let doorbell_base = qh.doorbell_base;

    // SAFETY: gart_host points to the GART page we just mmapped; the
    // write_dispatch_id field lives at a fixed offset inside the AmdQueueT
    // descriptor we wrote into the page.
    let write_ptr_host = unsafe { NonNull::new_unchecked(gart_host.as_ptr().add(wptr_offset as usize) as *mut u64) };

    Ok(QueueInner {
        ring_host,
        ring_size,
        doorbell,
        doorbell_base,
        write_ptr_host,
        gart_host,
        write_idx: 0,
        queue_id,
        qinactive_host,
        _ring_buf: ring_buf,
        _gart_buf: gart_buf,
        _eop_buf: eop_buf,
        _ctx_buf: ctx_buf,
        _qinactive_buf: qinactive_buf,
    })
}

/// Compute (wg_data_size, ctl_stack_size, debug_memory_size) for the ctx-save /
/// restore region.
fn compute_ctx_sizes(dev: &AmdDeviceCore) -> (usize, usize, usize) {
    const PAGE: usize = 0x1000;
    let sgrp_per_cu: usize = 0x4000;
    let hwreg_per_cu: usize = 0x1000;
    let is_cdna4 = dev.arch == svod_dtype::AmdArch::Gfx950;
    let lds_per_cu: usize = if is_cdna4 { (dev.node.lds_size_in_kb as usize) << 10 } else { 0x10000 };

    // VGPR-per-CU, mirroring ROCr's `hsakmt_get_vgpr_size_per_cu` (libhsakmt
    // queues.c): CDNA (gfx9.x) uses 0x80000, the listed RDNA3/RDNA4 tuples use
    // 0x60000, and everything below gfx1100 — all RDNA2 (gfx10.3) — plus Gfx1102
    // use the 0x40000 default. The kernel rejects CREATE_QUEUE with EINVAL if the
    // CWSR buffer derived from this is short, so it must match the runtime.
    let vgpr_per_cu: usize = match dev.arch {
        svod_dtype::AmdArch::Gfx942 | svod_dtype::AmdArch::Gfx950 => 0x80000,
        svod_dtype::AmdArch::Gfx1100
        | svod_dtype::AmdArch::Gfx1101
        | svod_dtype::AmdArch::Gfx1151
        | svod_dtype::AmdArch::Gfx1200
        | svod_dtype::AmdArch::Gfx1201 => 0x60000,
        svod_dtype::AmdArch::Gfx1030
        | svod_dtype::AmdArch::Gfx1031
        | svod_dtype::AmdArch::Gfx1032
        | svod_dtype::AmdArch::Gfx1033
        | svod_dtype::AmdArch::Gfx1034
        | svod_dtype::AmdArch::Gfx1035
        | svod_dtype::AmdArch::Gfx1036
        | svod_dtype::AmdArch::Gfx1102 => 0x40000,
    };

    let xccs = dev.node.num_xcc.max(1) as usize;
    let cu_cnt = ((dev.node.simd_count.max(1) / dev.node.simd_per_cu.max(1)) as usize / xccs).max(1);
    let waves_per_cu = (dev.node.max_waves_per_simd as usize) * (dev.node.simd_per_cu as usize);
    let wave_cnt = if dev.arch.is_cdna() {
        // gfx9 caps waves at min(cu_cnt*40, se_cnt*xccs*512). se_cnt is the
        // shader-engine count per XCC (array_count / simd_arrays_per_engine /
        // xccs). KFD >= 6.11 sizes the CWSR debug-memory area from this exact
        // formula and rejects CREATE_QUEUE with EINVAL if our buffer is short,
        // so it must match the kernel's `kfd_queue_acquire_buffers` value.
        let se_cnt = (dev.node.array_count as usize / (dev.node.simd_arrays_per_engine.max(1) as usize) / xccs).max(1);
        (cu_cnt * 40).min(se_cnt * xccs * 512)
    } else {
        cu_cnt * waves_per_cu
    };

    let wg_data_size = (vgpr_per_cu + sgrp_per_cu + lds_per_cu + hwreg_per_cu) * cu_cnt;
    let wg_data_size = wg_data_size.next_multiple_of(PAGE);

    let waves_factor = if dev.arch.is_cdna() { 8 } else { 12 };
    let mut ctl_stack_size = (waves_factor * wave_cnt + 8 + 40).next_multiple_of(PAGE);
    // gfx10 (RDNA2) HW design caps the control stack at 0x7000 (sufficient for
    // AQL, limited by SPI events). ROCr clamps it for `(gfxv & 0x3f0000) ==
    // 0xA0000` (queues.c), tinygrad for `target[0] == 10`; an unclamped larger
    // value risks CREATE_QUEUE rejection on gfx10.
    if dev.arch.is_rdna2() {
        ctl_stack_size = ctl_stack_size.min(0x7000);
    }
    // `debug_memory_size = round_up(wave_cnt * 32, 64)`.
    let debug_memory_size = (wave_cnt * 32).next_multiple_of(64);

    (wg_data_size, ctl_stack_size, debug_memory_size)
}
