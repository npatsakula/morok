//! `AmdIface`: the device-backend seam.
//!
//! `AmdDeviceCore` no longer issues KFD ioctls directly — it routes every
//! memory alloc/free, queue ring setup/teardown, and event wait through an
//! `Arc<dyn AmdIface>`. The only implementor today is [`KfdIface`], a verbatim
//! relocation of the KFD-direct path that previously lived inline across
//! `allocator.rs`, `device.rs`, and `queue.rs`. KFD-specific state (the kfd /
//! drm fds, KFD ABI version, event ids, and the per-process event page) lives
//! on `KfdIface`, off the core. A future userspace AM driver becomes a second
//! `AmdIface` implementor without touching any call site above the seam.
//!
//! This is a pure refactor: flag composition, ioctl argument layout, error
//! mapping, and the fault-message strings are reproduced bit-for-bit.

#![cfg(unix)]

use std::os::fd::{AsRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use libc::{
    MAP_ANONYMOUS, MAP_FIXED, MAP_NORESERVE, MAP_PRIVATE, MAP_SHARED, PROT_NONE, PROT_READ, PROT_WRITE, mmap, munmap,
};
use tracing::debug;

use crate::amd::sys::{ioctl, kfd};
use crate::amd::topology::AmdNode;
use crate::amd::va_registry::{AllocTag, VaRegistry};
use crate::error::{Error, Result};

/// Size of the doorbell MMIO window mapped per queue (two 4 KiB pages).
const DOORBELL_PAGE_BYTES: usize = 0x2000;

/// Backend seam for the AMD device. Every KFD ioctl that `AmdDeviceCore` (and
/// its allocator / queue / connector helpers) needs is funnelled through one
/// of these five methods, so a second backend (the userspace AM driver) can be
/// dropped in without editing any call site.
pub trait AmdIface: Send + Sync + std::fmt::Debug {
    /// Reserve a host VA, KFD-allocate `size` bytes per `kind`, optionally map
    /// host-visible, and bind it into the GPU page table. `zero` zero-fills the
    /// allocation, but this seam can only honor it for host-mapped buffers —
    /// `zero=true` with `cpu_access=false` is rejected with an error (the caller
    /// owns device-zeroing via the SDMA copy queue). `tag` records the
    /// allocation's purpose in the VA registry so a later fault can be resolved
    /// back to it.
    fn alloc_raw(
        &self,
        size: usize,
        kind: AllocKind,
        tag: AllocTag,
        cpu_access: bool,
        zero: bool,
    ) -> Result<AllocResult>;
    /// Unmap from the GPU, drop the host mapping, and free the KFD allocation.
    /// Best-effort (called from `Drop`); ioctl errors are swallowed.
    fn free_raw(&self, gpu_va: u64, size: usize, handle: u64);
    /// Create a KFD compute/SDMA queue over a pre-allocated ring + GART and
    /// return its queue id + mmapped doorbell.
    fn setup_ring(&self, desc: &RingDesc) -> Result<QueueHandle>;
    /// Destroy the in-kernel queue object and `munmap` the queue's doorbell
    /// page (`doorbell_base` is the mmap base from [`QueueHandle`]). Best-effort.
    fn teardown_ring(&self, queue_id: u32, doorbell_base: NonNull<u8>);
    /// Block up to `timeout_ms` on the device's completion + fault events.
    /// `Ok(Some(Error::Runtime{..}))` on a fault, `Ok(None)` on a normal
    /// wake-up/timeout, `Err` if the WAIT_EVENTS ioctl itself failed.
    fn wait_events(&self, timeout_ms: u32) -> Result<Option<Error>>;
}

/// Allocation flavor — selects the KFD flag set built in [`AmdIface::alloc_raw`].
#[derive(Clone, Copy, Debug)]
pub enum AllocKind {
    /// Device VRAM (GTT on APUs). WRITABLE | EXECUTABLE | NO_SUBSTITUTE, plus
    /// PUBLIC when `cpu_access` — the tinygrad/ROCr base flag set for every
    /// data/code/scratch buffer.
    DeviceVram,
    /// GTT-pinned, host-visible, uncached system memory (rings, signal slots,
    /// the event page). Carries fine-grained `COHERENT`, which the completion
    /// signal handshake relies on.
    UncachedGtt,
    /// GTT-pinned, host-visible, *cached* coherent memory — ROCr's flag set for
    /// the queue rptr/wptr control page and CWSR ctx-save (`queues.c`
    /// `allocate_exec_aligned_memory_gpu`, Uncached=0). Same as `UncachedGtt`
    /// minus the UNCACHED bit.
    CoherentGtt,
}

/// Result of [`AmdIface::alloc_raw`]: everything `RawBuffer::AmdDevice` needs
/// except the owning device handle (the allocator adds that at the call site).
pub struct AllocResult {
    pub gpu_va: u64,
    pub host_ptr: Option<NonNull<u8>>,
    pub handle: u64,
    pub size: usize,
}

// SAFETY: `host_ptr` is a raw mapping pointer with no thread affinity; access
// is synchronized by the scheduler / single-owner invariant exactly as
// `RawBuffer` asserts. Needed only so `Result<AllocResult>` crosses the trait.
unsafe impl Send for AllocResult {}

/// Inputs for [`AmdIface::setup_ring`]: the caller pre-allocates the ring /
/// GART / EOP / ctx-save backings (above the seam) and hands their GPU VAs +
/// sizes plus the descriptor offsets here.
pub struct RingDesc {
    pub ring_gpu: u64,
    pub gart_gpu: u64,
    pub wptr_offset: u64,
    pub rptr_offset: u64,
    pub eop_gpu: u64,
    pub eop_size: u64,
    pub ctx_gpu: u64,
    pub ctx_save_restore_size: u32,
    pub ctl_stack_size: u32,
    pub ring_size: usize,
    pub gpu_id: u32,
    pub queue_type: u32,
}

/// Result of [`AmdIface::setup_ring`]: the kernel-assigned queue id and the
/// per-queue doorbell pointer.
pub struct QueueHandle {
    pub queue_id: u32,
    /// mmap base of the doorbell page, kept so the page can be `munmap`'d on
    /// teardown (each queue maps its own page).
    pub doorbell_base: NonNull<u8>,
    pub doorbell: NonNull<u64>,
}

// SAFETY: `doorbell` is an MMIO mapping pointer; the owning `QueueInner`
// already asserts `Send`/`Sync` over it. Needed so `Result<QueueHandle>`
// crosses the trait.
unsafe impl Send for QueueHandle {}

/// KFD-direct backend. Owns the per-process/per-device KFD state that used to
/// live on `AmdDeviceCore`.
#[derive(Debug)]
pub struct KfdIface {
    /// Shared `/dev/kfd` handle (one per process).
    kfd_fd: Arc<OwnedFd>,
    drm_fd: OwnedFd,
    /// Its own clone of the topology node (gpu_id, drm minor, etc.).
    node: AmdNode,
    // The following are captured at bring-up and retained for parity with the
    // pre-refactor `AmdDeviceCore` (KFD ABI gating + event-page bookkeeping).
    // They are read only during `open`/diagnostics today, so allow the
    // not-currently-reread fields rather than churn the layout.
    #[allow(dead_code)]
    kfd_version: (u32, u32),
    /// VA + size of the GTT-pinned event page (held to keep it mapped).
    #[allow(dead_code)]
    event_page_va: u64,
    #[allow(dead_code)]
    event_page_size: usize,
    /// KFD ids for the SIGNAL events used by queue completion / fault paths.
    queue_event_id: u32,
    #[allow(dead_code)]
    queue_event_slot_index: u32,
    #[allow(dead_code)]
    queue_event_mailbox_ptr: u64,
    mem_fault_event_id: u32,
    hw_fault_event_id: u32,
    /// VA → allocation registry: every `alloc_raw` range, plus a bounded ring
    /// of recently-freed ranges, so a fault VA can be resolved to its owning
    /// (or stale, or nearest) allocation. See [`crate::amd::va_registry`].
    va: VaRegistry,
    /// One-shot latch so the terminal fault is logged at `error` exactly once,
    /// even though `wait_events(0)` may re-observe the (non-auto-reset) fault
    /// event on subsequent poll-fault calls.
    fault_logged: AtomicBool,
}

impl KfdIface {
    /// Open `/dev/kfd` + the node's DRM render fd, acquire the VM, enable the
    /// runtime, bind the event page, and create the three completion / fault
    /// events. Verbatim move of the bring-up block from `AmdDevice::open_uncached`.
    pub fn open(node: &AmdNode) -> Result<Self> {
        let kfd_fd = crate::amd::device::ensure_global_kfd()?;
        let drm_path = format!("/dev/dri/renderD{}", node.drm_render_minor);
        let drm_fd = crate::amd::device::open_owned(drm_path.as_str())?;

        // GET_VERSION captures the KFD ABI version so we can gate RUNTIME_ENABLE
        // (which only exists on kfd >= 1.14).
        let mut ver_args = kfd::kfd_ioctl_get_version_args { major_version: 0, minor_version: 0 };
        if let Err(e) = unsafe { ioctl::kfd_get_version(kfd_fd.as_raw_fd(), &mut ver_args as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_GET_VERSION", errno: e as i32 });
        }
        let kfd_version = (ver_args.major_version, ver_args.minor_version);

        // ACQUIRE_VM tells KFD to register this DRM fd as the process's VM
        // for this GPU. Required before any alloc/map ioctls.
        let mut args = kfd::kfd_ioctl_acquire_vm_args { drm_fd: drm_fd.as_raw_fd() as u32, gpu_id: node.gpu_id };
        // SAFETY: kfd_fd is a valid open fd; `args` is a well-typed ioctl
        // argument matching the AMDKFD_IOC_ACQUIRE_VM signature.
        let rc = unsafe { ioctl::kfd_acquire_vm(kfd_fd.as_raw_fd(), &mut args as *mut _) };
        if let Err(e) = rc {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_ACQUIRE_VM", errno: e as i32 });
        }

        // SET_MEMORY_POLICY — ROCr issues this once per GPU right after
        // ACQUIRE_VM (fmm_init_process_apertures): default NONCOHERENT cache
        // policy, COHERENT alternate. We carve no alternate aperture (per-alloc
        // COHERENT flags select fine-grain instead), so base/size stay 0.
        let mut policy = kfd::kfd_ioctl_set_memory_policy_args {
            gpu_id: node.gpu_id,
            default_policy: kfd::KFD_IOC_CACHE_POLICY_NONCOHERENT,
            alternate_policy: kfd::KFD_IOC_CACHE_POLICY_COHERENT,
            ..Default::default()
        };
        if let Err(e) = unsafe { ioctl::kfd_set_memory_policy(kfd_fd.as_raw_fd(), &mut policy as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_SET_MEMORY_POLICY", errno: e as i32 });
        }

        // SCRATCH_BACKING_VA — ROCr programs the per-VMID scratch aperture base
        // (SH_HIDDEN_PRIVATE_BASE) before any queue exists; KFD reports the
        // process scratch aperture in GET_PROCESS_APERTURES. Without it, gfx10
        // dispatches that spill resolve scratch against an unprogrammed base.
        let mut apertures = kfd::kfd_ioctl_get_process_apertures_args::default();
        if let Err(e) = unsafe { ioctl::kfd_get_process_apertures(kfd_fd.as_raw_fd(), &mut apertures as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_GET_PROCESS_APERTURES", errno: e as i32 });
        }
        let scratch_base = apertures.process_apertures[..apertures.num_of_nodes.min(7) as usize]
            .iter()
            .find(|a| a.gpu_id == node.gpu_id)
            .map(|a| a.scratch_base)
            .unwrap_or(0);
        if scratch_base != 0 {
            let mut sb =
                kfd::kfd_ioctl_set_scratch_backing_va_args { va_addr: scratch_base, gpu_id: node.gpu_id, pad: 0 };
            if let Err(e) = unsafe { ioctl::kfd_set_scratch_backing_va(kfd_fd.as_raw_fd(), &mut sb as *mut _) } {
                return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_SET_SCRATCH_BACKING_VA", errno: e as i32 });
            }
        }

        // XNACK off — explicit, matching ROCr's init on non-XNACK parts. Old
        // kernels reject the ioctl (ENOTTY); best-effort.
        let mut xnack = kfd::kfd_ioctl_set_xnack_mode_args { xnack_enabled: 0 };
        if let Err(e) = unsafe { ioctl::kfd_set_xnack_mode(kfd_fd.as_raw_fd(), &mut xnack as *mut _) } {
            tracing::warn!(?e, "SET_XNACK_MODE(0) rejected; continuing with kernel default");
        }

        // RUNTIME_ENABLE — only on KFD >= 1.14; older kernels reject the
        // ioctl with ENOTTY. KFD selects enable-vs-disable by
        // `mode_mask & KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK` (bit 0), so a zero
        // mask invokes runtime *disable* — the opposite of intent. ROCr always
        // enables at init (libhsakmt `hsaKmtRuntimeEnable`, debug.c). The enabled
        // runtime state is what registers the process for KFD eviction→restore +
        // queue-resume, which a memory-pressured APU (iGPU driving display +
        // compute) depends on; without it, an evicted-but-live BO is never
        // restored and the CP faults NotPresent on it. No TTMP (no CWSR
        // trap-debug); r_debug = 0 (no debugger attached).
        // Bisect gate: KFD couples runtime-enable to per-process debug state;
        // we pass r_debug=0 (ROCr passes a real pointer). Skip the ioctl to
        // isolate whether the enabled-debug-runtime state perturbs queue/BO
        // restore on gfx10.3 iGPUs.
        if std::env::var_os("SVOD_AMD_NO_RUNTIME_ENABLE").is_some() {
            tracing::warn!("SVOD_AMD_NO_RUNTIME_ENABLE set; skipping RUNTIME_ENABLE");
        } else if kfd_version >= (1, 14) {
            const KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK: u32 = 1;
            let mut rt = kfd::kfd_ioctl_runtime_enable_args {
                mode_mask: KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK,
                ..Default::default()
            };
            if let Err(e) = unsafe { ioctl::kfd_runtime_enable(kfd_fd.as_raw_fd(), &mut rt as *mut _) } {
                return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_RUNTIME_ENABLE", errno: e as i32 });
            }
        }

        // Event-page setup: allocated and bound
        // exactly once per process; subsequent devices reuse it by calling
        // `MAP_MEMORY_TO_GPU` for their `gpu_id`. Without the bound event page,
        // AMDKFD_IOC_CREATE_QUEUE returns EINVAL.
        let (event_page_va, event_page_size) = {
            let ep = crate::amd::device::ensure_event_page(&kfd_fd, &drm_fd, node)?;
            (ep.va, ep.size)
        };
        let mut qe = kfd::kfd_ioctl_create_event_args {
            event_type: kfd::KFD_IOC_EVENT_SIGNAL,
            auto_reset: 1,
            ..Default::default()
        };
        if let Err(e) = unsafe { ioctl::kfd_create_event(kfd_fd.as_raw_fd(), &mut qe as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_EVENT(queue signal)", errno: e as i32 });
        }
        let mut mem_event =
            kfd::kfd_ioctl_create_event_args { event_type: kfd::KFD_IOC_EVENT_MEMORY, ..Default::default() };
        if let Err(e) = unsafe { ioctl::kfd_create_event(kfd_fd.as_raw_fd(), &mut mem_event as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_EVENT(mem fault)", errno: e as i32 });
        }
        let mut hw_event =
            kfd::kfd_ioctl_create_event_args { event_type: kfd::KFD_IOC_EVENT_HW_EXCEPTION, ..Default::default() };
        if let Err(e) = unsafe { ioctl::kfd_create_event(kfd_fd.as_raw_fd(), &mut hw_event as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_EVENT(hw fault)", errno: e as i32 });
        }

        // The mailbox sits at event_page + slot_index * 8. SDMA fence packets
        // write the queue event_id here to wake up `WAIT_EVENTS` from `sleep()`.
        let queue_event_mailbox_ptr = event_page_va + (qe.event_slot_index as u64) * 8;

        debug!(
            node = node.node_id,
            gpu_id = node.gpu_id,
            kfd_version = ?kfd_version,
            queue_event_id = qe.event_id,
            mem_fault_event_id = mem_event.event_id,
            hw_fault_event_id = hw_event.event_id,
            "KfdIface opened"
        );

        Ok(Self {
            kfd_fd,
            drm_fd,
            node: node.clone(),
            kfd_version,
            event_page_va,
            event_page_size,
            queue_event_id: qe.event_id,
            queue_event_slot_index: qe.event_slot_index,
            queue_event_mailbox_ptr,
            mem_fault_event_id: mem_event.event_id,
            hw_fault_event_id: hw_event.event_id,
            va: VaRegistry::default(),
            fault_logged: AtomicBool::new(false),
        })
    }
}

impl AmdIface for KfdIface {
    fn alloc_raw(
        &self,
        size: usize,
        kind: AllocKind,
        tag: AllocTag,
        cpu_accessible: bool,
        zero_init: bool,
    ) -> Result<AllocResult> {
        // KFD VA reservation + map are page-granular; a 0-byte mmap is EINVAL.
        let size = size.max(1).next_multiple_of(0x1000);
        let flags = compose_flags(kind, cpu_accessible, self.node.is_apu());
        let va = reserve_va(size)?;
        let mut args = kfd::kfd_ioctl_alloc_memory_of_gpu_args {
            va_addr: va as u64,
            size: size as u64,
            gpu_id: self.node.gpu_id,
            flags,
            ..Default::default()
        };
        if let Err(e) = unsafe { ioctl::kfd_alloc_memory_of_gpu(self.kfd_fd.as_raw_fd(), &mut args as *mut _) } {
            unsafe { munmap(va as *mut _, size) };
            return Err(map_alloc_err(e, cpu_accessible));
        }
        let mem_handle = args.handle;
        let mmap_offset = args.mmap_offset;

        let host_ptr = if cpu_accessible {
            let p = unsafe {
                mmap(
                    va as *mut _,
                    size,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED,
                    self.drm_fd.as_raw_fd(),
                    mmap_offset as i64,
                )
            };
            if p == libc::MAP_FAILED || !std::ptr::eq(p, va) {
                self.free_kfd(mem_handle);
                unsafe { munmap(va as *mut _, size) };
                return Err(Error::AmdAllocFailed {
                    reason: "host-visible mmap failed (enable resizable BAR for VRAM, or check GTT availability)"
                        .into(),
                });
            }
            Some(unsafe { NonNull::new_unchecked(p as *mut u8) })
        } else {
            None
        };

        let mut gpu_id = self.node.gpu_id;
        let mut map_args = kfd::kfd_ioctl_map_memory_to_gpu_args {
            handle: mem_handle,
            device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
            n_devices: 1,
            n_success: 0,
        };
        if let Err(e) = unsafe { ioctl::kfd_map_memory_to_gpu(self.kfd_fd.as_raw_fd(), &mut map_args as *mut _) } {
            self.free_kfd(mem_handle);
            unsafe { munmap(va as *mut _, size) };
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_MAP_MEMORY_TO_GPU", errno: e as i32 });
        }
        if map_args.n_success != 1 {
            self.free_kfd(mem_handle);
            unsafe { munmap(va as *mut _, size) };
            return Err(Error::AmdAllocFailed {
                reason: format!("KFD map_memory_to_gpu reported {} success(es)", map_args.n_success),
            });
        }

        if zero_init {
            match host_ptr {
                Some(p) => unsafe { std::ptr::write_bytes(p.as_ptr(), 0, size) },
                // This seam has no copy queue, so it cannot zero a device-only
                // (non-host-mapped) buffer. Fail loud rather than silently return
                // garbage: `AmdAllocator::_alloc` passes `zero=false` here for
                // device-only buffers and zeroes them via SDMA itself, so a
                // `zero_init && !cpu_access` request reaching this point is a
                // caller bug (e.g. a future path that bypasses `_alloc`).
                None => {
                    self.free_kfd(mem_handle);
                    unsafe { munmap(va as *mut _, size) };
                    return Err(Error::AmdAllocFailed {
                        reason: "zero_init requested for a device-only (non-host-mapped) allocation; \
                                 the caller must device-zero (e.g. via the SDMA copy queue)"
                            .into(),
                    });
                }
            }
        }

        self.va.insert(va as u64, size, mem_handle, tag);
        debug!(size, gpu_addr = va as u64, handle = mem_handle, tag = ?tag, cpu_accessible, "AmdAllocator alloc done");

        Ok(AllocResult { gpu_va: va as u64, host_ptr, handle: mem_handle, size })
    }

    fn free_raw(&self, gpu_va: u64, size: usize, handle: u64) {
        // Drop from the live registry into the freed-history ring *before* the
        // unmap, so a fault VA that lands here is classified as RECENTLY-FREED
        // (the use-after-free signal) rather than as a live allocation.
        self.va.remove(gpu_va);
        debug!(gpu_va, size, handle, "AmdIface free_raw");
        // 1. Unmap from GPU.
        let mut gpu_id = self.node.gpu_id;
        let mut unmap_args = kfd::kfd_ioctl_unmap_memory_from_gpu_args {
            handle,
            device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
            n_devices: 1,
            n_success: 0,
        };
        // SAFETY: fd is alive; handle is from a successful alloc.
        let unmap_rc = unsafe { ioctl::kfd_unmap_memory_from_gpu(self.kfd_fd.as_raw_fd(), &mut unmap_args as *mut _) };
        // A failed/partial unmap leaves the GPU PTEs for `gpu_va` intact while we
        // go on to free the handle and (below) release the VA. If the VA is then
        // recycled by a later alloc, those stale PTEs surface as a live-but-
        // `NotPresent` fault. KFD's unmap is synchronous (PTE clear + TLB flush
        // in-kernel), so a non-error return means the mapping is gone — surface
        // any failure loudly instead of silently recycling a half-mapped VA.
        match unmap_rc {
            Err(e) => {
                tracing::warn!(?e, gpu_va, handle, "free_raw: UNMAP_MEMORY_FROM_GPU failed — VA recycle may fault")
            }
            Ok(_) if unmap_args.n_success != 1 => {
                tracing::warn!(n_success = unmap_args.n_success, gpu_va, handle, "free_raw: UNMAP partial")
            }
            Ok(_) => {}
        }
        // 2. Free the KFD allocation BEFORE releasing the VA (ROCr `__fmm_release`
        //    order, fmm.c: free the BO — which finalizes PTE teardown — then let
        //    the VA become reusable). Releasing the VA first would let a
        //    concurrent alloc map a fresh BO over a VA whose old BO isn't freed.
        let mut free_args = kfd::kfd_ioctl_free_memory_of_gpu_args { handle };
        // SAFETY: fd alive; handle from a successful alloc.
        let _ = unsafe { ioctl::kfd_free_memory_of_gpu(self.kfd_fd.as_raw_fd(), &mut free_args as *mut _) };
        // 3. Quarantine the VA instead of releasing it (ROCr `reserved_aperture_release`,
        //    fmm.c): re-mmap PROT_NONE drops the host pages but keeps the range
        //    reserved, so neither the OS nor a later alloc can recycle a VA the GPU
        //    might still reference through a stale PTE or in-flight packet. The
        //    cost is bounded by total bytes ever freed in 47-bit address space;
        //    only on ENOMEM fall back to a real munmap.
        // SAFETY: gpu_va is the VA returned by our own mmap.
        let p = unsafe {
            mmap(gpu_va as *mut _, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0)
        };
        if p == libc::MAP_FAILED {
            unsafe { munmap(gpu_va as *mut _, size) };
        }
    }

    fn setup_ring(&self, desc: &RingDesc) -> Result<QueueHandle> {
        let mut args = kfd::kfd_ioctl_create_queue_args {
            ring_base_address: desc.ring_gpu,
            write_pointer_address: desc.gart_gpu + desc.wptr_offset,
            read_pointer_address: desc.gart_gpu + desc.rptr_offset,
            doorbell_offset: 0,
            ring_size: desc.ring_size as u32,
            gpu_id: desc.gpu_id,
            queue_type: desc.queue_type,
            queue_percentage: kfd::KFD_MAX_QUEUE_PERCENTAGE,
            queue_priority: 7,
            ..Default::default()
        };
        if desc.eop_gpu != 0 {
            args.eop_buffer_address = desc.eop_gpu;
            args.eop_buffer_size = desc.eop_size;
        }
        if desc.ctx_gpu != 0 {
            args.ctx_save_restore_address = desc.ctx_gpu;
            args.ctx_save_restore_size = desc.ctx_save_restore_size;
            args.ctl_stack_size = desc.ctl_stack_size;
        }

        // SAFETY: kfd_fd is alive; args is well-typed.
        if let Err(e) = unsafe { ioctl::kfd_create_queue(self.kfd_fd.as_raw_fd(), &mut args as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_QUEUE", errno: e as i32 });
        }

        let (doorbell_base, doorbell) = self.doorbell_mmap(args.doorbell_offset)?;
        debug!(queue_id = args.queue_id, doorbell_offset = args.doorbell_offset, "AMD queue created");
        Ok(QueueHandle { queue_id: args.queue_id, doorbell_base, doorbell })
    }

    fn teardown_ring(&self, queue_id: u32, doorbell_base: NonNull<u8>) {
        let mut args = kfd::kfd_ioctl_destroy_queue_args { queue_id, ..Default::default() };
        // SAFETY: `kfd_fd` is alive (held via Arc<OwnedFd>); the queue_id was
        // returned by KFD on the matching create_queue call.
        let rc = unsafe { ioctl::kfd_destroy_queue(self.kfd_fd.as_raw_fd(), &mut args as *mut _) };
        if let Err(e) = rc {
            tracing::warn!(?e, queue_id, "teardown_ring: kfd_destroy_queue failed");
        }
        // Release the per-queue doorbell MMIO page mapped in `doorbell_mmap`.
        // SAFETY: `doorbell_base` is the mmap base returned for this queue and
        // is no longer referenced once the queue is destroyed.
        if unsafe { munmap(doorbell_base.as_ptr().cast(), DOORBELL_PAGE_BYTES) } != 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            tracing::warn!(queue_id, errno, "teardown_ring: doorbell munmap failed");
        }
    }

    fn wait_events(&self, timeout_ms: u32) -> Result<Option<Error>> {
        let mut events: [kfd::kfd_event_data; 3] = [Default::default(); 3];
        events[0].event_id = self.queue_event_id;
        events[1].event_id = self.mem_fault_event_id;
        events[2].event_id = self.hw_fault_event_id;
        let mut args = kfd::kfd_ioctl_wait_events_args {
            events_ptr: events.as_mut_ptr() as u64,
            num_events: events.len() as u32,
            wait_for_all: 0,
            timeout: timeout_ms,
            wait_result: 0,
        };
        // SAFETY: kfd_fd is alive; args + events live for the duration of the call.
        let rc = unsafe { ioctl::kfd_wait_events(self.kfd_fd.as_raw_fd(), &mut args as *mut _) };
        if let Err(e) = rc {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_WAIT_EVENTS", errno: e as i32 });
        }
        // Inspect each event's union payload. `gpu_id != 0` signals the fault
        // was actually written by KFD (the union is zero-initialized when
        // nothing fired). Slot [1] = memory fault, slot [2] = hw exception.
        // SAFETY: bindgen union access — we read whichever payload type
        // matches the event we registered.
        let mem = unsafe { events[1].__bindgen_anon_1.memory_exception_data };
        if mem.gpu_id != 0 {
            // Copy each field into a local before the format/tracing calls
            // below. The bindgen union payload is `#[repr(C)]` (naturally
            // aligned), so the by-value copies are for clarity, not to avoid an
            // unaligned reference.
            let gpu_id = mem.gpu_id;
            let va = { mem.va };
            let not_present = { mem.failure.NotPresent };
            let read_only = { mem.failure.ReadOnly };
            let no_execute = { mem.failure.NoExecute };
            let imprecise = { mem.failure.imprecise };
            let error_type = { mem.ErrorType };
            // Resolve the raw VA to its owning / stale / nearest allocation —
            // turns "fault at 0x7f…" into "fault +0x40 into a RECENTLY-FREED
            // scratch region", which is what actually localizes the bug.
            let class = self.va.classify(va);
            let va_hex = format!("{va:#x}");
            let message = format!(
                "AMD GPU memory fault on gpu_id={gpu_id} va={va_hex} \
                 (NotPresent={not_present} ReadOnly={read_only} NoExecute={no_execute} \
                 Imprecise={imprecise} ErrorType={error_type}) — {class}",
            );
            // Log at the fault site: the panic that eventually surfaces this is
            // a delayed re-throw at the next `synchronize()`, far from here. Once
            // only — the memory-fault event is not auto-reset, so subsequent
            // `wait_events(0)` poll-fault calls re-observe the same fault.
            if !self.fault_logged.swap(true, Ordering::Relaxed) {
                tracing::error!(
                    gpu_id,
                    va = va_hex.as_str(),
                    not_present,
                    read_only,
                    no_execute,
                    imprecise,
                    error_type,
                    classification = %class,
                    "AMD GPU memory fault"
                );
            }
            return Ok(Some(Error::Runtime { message }));
        }
        let hw = unsafe { events[2].__bindgen_anon_1.hw_exception_data };
        if hw.gpu_id != 0 {
            let message = format!(
                "AMD GPU hardware exception on gpu_id={} reset_type={} reset_cause={} memory_lost={}",
                hw.gpu_id, hw.reset_type, hw.reset_cause, hw.memory_lost,
            );
            return Ok(Some(Error::Runtime { message }));
        }
        Ok(None)
    }
}

impl KfdIface {
    fn free_kfd(&self, handle: u64) {
        let mut args = kfd::kfd_ioctl_free_memory_of_gpu_args { handle };
        // SAFETY: self.kfd_fd is alive; handle is from a successful alloc.
        let _ = unsafe { ioctl::kfd_free_memory_of_gpu(self.kfd_fd.as_raw_fd(), &mut args as *mut _) };
    }

    /// Map a single doorbell window from `/dev/kfd` and return the mmap base
    /// (kept for a later `munmap`) plus the per-queue doorbell pointer. KFD
    /// doorbells are page-aligned regions of MMIO addresses; the per-queue
    /// doorbell address = page_base + (doorbell_offset & 0x1fff). The base is
    /// returned separately because `mmap` only guarantees 4 KiB alignment, so
    /// the page base is not recoverable from the offset pointer alone.
    fn doorbell_mmap(&self, doorbell_offset: u64) -> Result<(NonNull<u8>, NonNull<u64>)> {
        let page_base = doorbell_offset & !0x1fff;
        // SAFETY: standard libc::mmap; protections set for read+write MMIO.
        let p = unsafe {
            mmap(
                std::ptr::null_mut(),
                DOORBELL_PAGE_BYTES,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                self.kfd_fd.as_raw_fd(),
                page_base as i64,
            )
        };
        if p == libc::MAP_FAILED {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(Error::AmdAllocFailed { reason: format!("doorbell mmap failed (errno {errno})") });
        }
        let base = NonNull::new(p as *mut u8).expect("doorbell page non-null");
        let offset_in_page = (doorbell_offset & 0x1fff) as usize;
        // SAFETY: offset_in_page < DOORBELL_PAGE_BYTES; alignment to u64 holds.
        let ptr = unsafe { base.as_ptr().add(offset_in_page) as *mut u64 };
        Ok((base, NonNull::new(ptr).expect("doorbell page non-null")))
    }
}

/// Compose the KFD `KFD_IOC_ALLOC_MEM_FLAGS_*` set for an allocation. This is
/// the ONLY place the flags are built; it reproduces the four pre-refactor flag
/// sets bit-for-bit (see the module doc / call sites).
///
/// `is_apu` selects the device-memory backing: discrete GPUs use dedicated VRAM,
/// but APUs (integrated, unified memory) have none, so `DeviceVram` is allocated
/// from GTT (system memory the GPU reaches via the GART). The modifier bits are
/// otherwise identical, so the coherence contract the copy paths rely on (the
/// GPU's L2-acquire dispatch prologue) is unchanged — GTT just replaces VRAM as
/// the heap. The control-structure (`UncachedGtt`) set is already GTT and
/// arch-independent.
pub(crate) fn compose_flags(kind: AllocKind, cpu_access: bool, is_apu: bool) -> u32 {
    match kind {
        AllocKind::DeviceVram => {
            let heap = if is_apu { kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT } else { kfd::KFD_IOC_ALLOC_MEM_FLAGS_VRAM };
            let mut flags = heap
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
            if cpu_access {
                flags |= kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC;
            }
            flags
        }
        AllocKind::UncachedGtt => {
            kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED
        }
        AllocKind::CoherentGtt => {
            kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT
        }
    }
}

/// Reserve `size` bytes of host VA so KFD can bind VRAM into it.
fn reserve_va(size: usize) -> Result<*mut libc::c_void> {
    // SAFETY: standard libc::mmap signature; no aliasing concerns at this point.
    let p = unsafe { mmap(std::ptr::null_mut(), size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0) };
    if p == libc::MAP_FAILED {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        return Err(Error::AmdAllocFailed { reason: format!("VA reservation mmap failed (errno {errno})") });
    }
    Ok(p)
}

fn map_alloc_err(e: nix::errno::Errno, cpu_accessible: bool) -> Error {
    match e {
        nix::errno::Errno::ENOMEM => Error::AmdAllocFailed { reason: "ENOMEM (VRAM exhausted)".into() },
        nix::errno::Errno::EINVAL if cpu_accessible => {
            Error::AmdAllocFailed { reason: "EINVAL on host-visible VRAM alloc — enable resizable BAR".into() }
        }
        other => Error::AmdIoctl { ioctl: "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU", errno: other as i32 },
    }
}
