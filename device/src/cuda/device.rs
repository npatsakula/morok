//! The opened CUDA device: primary context, attribute limits, the copy and
//! dispatch streams, the base event that zeroes the GPU-clock timeline, and
//! the poison latch.

use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int};
use std::ptr::{NonNull, null_mut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use svod_dtype::CudaArch;

use super::sys::{
    Api, CU_EVENT_DEFAULT, CU_EVENT_DISABLE_TIMING, CU_MEMHOSTALLOC_PORTABLE, CU_STREAM_NON_BLOCKING, CUcontext,
    CUdevice, CUdeviceptr, CUevent, CUresult, CUstream, api, attribute,
};
use crate::error::{Error, Result, TimelineTimeoutSnafu};

static DEVICE_CACHE: LazyLock<Mutex<HashMap<usize, Arc<CudaDevice>>>> = LazyLock::new(Default::default);
static HAS_DEVICES: OnceLock<bool> = OnceLock::new();

/// Bounce buffer for large pageable host ↔ device copies (`cuMemcpy*Async`
/// needs pinned host memory to be asynchronous); transfers up to this size
/// go through one synchronous `cuMemcpy` instead.
pub(crate) const STAGING_BYTES: usize = 4 << 20;
/// Poll cadence of timed event waits; the driver offers no timed wait.
const EVENT_POLL: Duration = Duration::from_micros(200);

/// Whether the driver loads, initializes, and reports at least one device.
/// Memoized; never panics; `false` on any failure.
pub fn has_devices() -> bool {
    *HAS_DEVICES.get_or_init(|| api().and_then(|api| api.init().and_then(|()| api.device_count())).is_ok_and(|n| n > 0))
}

/// Static device limits (`cuDeviceGetAttribute`): the launch-bound check,
/// the optimizer profile's shared-memory budget and
/// [`crate::KernelResources`] occupancy.
#[derive(Debug, Clone, Copy)]
pub struct CudaLimits {
    pub sm_count: u32,
    pub max_threads_per_block: u32,
    pub max_threads_per_sm: u32,
    pub shared_per_block: u32,
    pub warp_size: u32,
    /// `cuMemAllocManaged` is usable and host access is coherent with running
    /// kernels: the backing of host-visible buffers.
    pub managed_memory: bool,
}

pub struct CudaDevice {
    api: &'static Api,
    device_id: usize,
    handle: CUdevice,
    context: CUcontext,
    name: String,
    arch: CudaArch,
    limits: CudaLimits,
    /// Host ↔ device copies and memsets (`CudaAllocator`).
    copy_stream: CUstream,
    /// Per-call `Program::execute` dispatches.
    dispatch_stream: CUstream,
    /// Recorded at open: the zero of every GPU-clock timestamp.
    base_event: CUevent,
    staging: Mutex<Option<NonNull<u8>>>,
    poisoned: AtomicBool,
    poison_message: OnceLock<String>,
}

// SAFETY: every field is either immutable after open or guarded; the staging
// pointer is pinned host memory used only under its mutex.
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDevice")
            .field("device_id", &self.device_id)
            .field("name", &self.name)
            .field("arch", &self.arch)
            .field("poisoned", &self.is_poisoned())
            .finish_non_exhaustive()
    }
}

impl CudaDevice {
    /// Process-global cached open of CUDA device `device_id`. Never panics:
    /// `DeviceUnavailable` without a driver, `NoCudaGpu` without a device,
    /// `CudaDriver` for driver failures.
    pub fn open(device_id: usize) -> Result<Arc<Self>> {
        let mut cache = DEVICE_CACHE.lock();
        if let Some(device) = cache.get(&device_id) {
            return Ok(Arc::clone(device));
        }
        let device = Arc::new(Self::open_uncached(device_id)?);
        cache.insert(device_id, Arc::clone(&device));
        Ok(device)
    }

    fn open_uncached(device_id: usize) -> Result<Self> {
        let api = api()?;
        api.init().map_err(|error| Error::NoCudaGpu { reason: error.to_string() })?;
        let count = api.device_count()?;
        if device_id >= count {
            return Err(Error::NoCudaGpu {
                reason: format!("device_id {device_id} out of range; {count} device(s) present"),
            });
        }
        let mut handle: CUdevice = 0;
        // SAFETY: out-pointer to a live integer; the ordinal was range-checked.
        unsafe { (api.device_get)(&mut handle, device_id as c_int) }.check("cuDeviceGet")?;
        let mut context = CUcontext::NULL;
        // SAFETY: out-pointer to a live handle slot.
        unsafe { (api.device_primary_ctx_retain)(&mut context, handle) }.check("cuDevicePrimaryCtxRetain")?;
        // Everything below runs in the retained context; release it on failure.
        let release = ContextGuard { api, handle };
        // SAFETY: a context this process retains.
        unsafe { (api.ctx_set_current)(context) }.check("cuCtxSetCurrent")?;

        let attribute = |id: i32| -> Result<u32> {
            let mut value: c_int = 0;
            // SAFETY: out-pointer to a live integer.
            unsafe { (api.device_get_attribute)(&mut value, id, handle) }.check("cuDeviceGetAttribute")?;
            Ok(u32::try_from(value).unwrap_or(0))
        };
        let mut name = [0 as c_char; 256];
        // SAFETY: the driver writes at most `len` bytes including the NUL.
        unsafe { (api.device_get_name)(name.as_mut_ptr(), name.len() as c_int, handle) }.check("cuDeviceGetName")?;
        // SAFETY: NUL-terminated by the driver (the buffer is zeroed anyway).
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
        let major = attribute(attribute::COMPUTE_CAPABILITY_MAJOR)?;
        let minor = attribute(attribute::COMPUTE_CAPABILITY_MINOR)?;
        let arch = CudaArch::from_compute_capability(
            u8::try_from(major).unwrap_or(u8::MAX),
            u8::try_from(minor).unwrap_or(u8::MAX),
        );
        let limits = CudaLimits {
            sm_count: attribute(attribute::MULTIPROCESSOR_COUNT)?,
            max_threads_per_block: attribute(attribute::MAX_THREADS_PER_BLOCK)?,
            max_threads_per_sm: attribute(attribute::MAX_THREADS_PER_MULTIPROCESSOR)?,
            shared_per_block: attribute(attribute::MAX_SHARED_MEMORY_PER_BLOCK)?,
            warp_size: attribute(attribute::WARP_SIZE)?,
            managed_memory: attribute(attribute::MANAGED_MEMORY)? == 1
                && attribute(attribute::CONCURRENT_MANAGED_ACCESS)? == 1,
        };

        let stream = || -> Result<CUstream> {
            let mut stream = CUstream::NULL;
            // SAFETY: out-pointer to a live handle slot.
            unsafe { (api.stream_create)(&mut stream, CU_STREAM_NON_BLOCKING) }.check("cuStreamCreate")?;
            Ok(stream)
        };
        let copy_stream = stream()?;
        let dispatch_stream = stream()?;
        let mut base_event = CUevent::NULL;
        // SAFETY: out-pointer to a live handle slot; the event is then
        // recorded on the legacy default stream and waited for, so it is
        // complete (and timestamped) before the device is handed out.
        unsafe {
            (api.event_create)(&mut base_event, CU_EVENT_DEFAULT).check("cuEventCreate")?;
            (api.event_record)(base_event, CUstream::NULL).check("cuEventRecord")?;
            (api.event_synchronize)(base_event).check("cuEventSynchronize")?;
        }
        let (driver_major, driver_minor) = api.driver_version()?;
        tracing::info!(
            device_id,
            name,
            %arch,
            sms = limits.sm_count,
            managed = limits.managed_memory,
            driver = format!("{driver_major}.{driver_minor}"),
            "opened CUDA device"
        );
        std::mem::forget(release);
        Ok(Self {
            api,
            device_id,
            handle,
            context,
            name,
            arch,
            limits,
            copy_stream,
            dispatch_stream,
            base_event,
            staging: Mutex::new(None),
            poisoned: AtomicBool::new(false),
            poison_message: OnceLock::new(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn arch(&self) -> CudaArch {
        self.arch
    }

    pub fn limits(&self) -> &CudaLimits {
        &self.limits
    }

    pub(crate) fn api(&self) -> &'static Api {
        self.api
    }

    pub(crate) fn copy_stream(&self) -> CUstream {
        self.copy_stream
    }

    pub(crate) fn dispatch_stream(&self) -> CUstream {
        self.dispatch_stream
    }

    pub(crate) fn base_event(&self) -> CUevent {
        self.base_event
    }

    /// Make this device's context current on the calling thread (the driver
    /// keeps it per thread) and refuse to proceed on a poisoned device. Every
    /// entry point of the backend starts here.
    pub(crate) fn enter(&self) -> Result<&'static Api> {
        if let Some(error) = self.poison_error() {
            return Err(error);
        }
        // SAFETY: a context this device retains for its whole lifetime.
        unsafe { (self.api.ctx_set_current)(self.context) }.check("cuCtxSetCurrent")?;
        Ok(self.api)
    }

    /// [`CUresult::check`] that also latches sticky (context-killing) errors.
    pub(crate) fn check(&self, result: CUresult, call: &'static str) -> Result<()> {
        let outcome = result.check(call);
        if let Err(error) = &outcome
            && result.is_sticky()
        {
            self.poison(&error.to_string());
        }
        outcome
    }

    /// Wait for every stream of this context.
    pub fn synchronize(&self) -> Result<()> {
        let api = self.enter()?;
        // SAFETY: plain call in the current context.
        self.check(unsafe { (api.ctx_synchronize)() }, "cuCtxSynchronize")
    }

    pub fn stream_synchronize(&self, stream: CUstream) -> Result<()> {
        let api = self.enter()?;
        // SAFETY: a live stream of this context.
        self.check(unsafe { (api.stream_synchronize)(stream) }, "cuStreamSynchronize")
    }

    /// `(free, total)` bytes of device memory.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        let api = self.enter()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: out-pointers to live size_t values.
        unsafe { (api.mem_get_info)(&mut free, &mut total) }.check("cuMemGetInfo")?;
        Ok((free, total))
    }

    /// Zero `size` bytes at `device_ptr` after draining in-flight work.
    pub(crate) fn zero(&self, device_ptr: CUdeviceptr, size: usize) -> Result<()> {
        self.synchronize()?;
        let api = self.api;
        // SAFETY: the caller owns `size` bytes at `device_ptr`.
        self.check(unsafe { (api.memset_d8_async)(device_ptr, 0, size, self.copy_stream) }, "cuMemsetD8Async")?;
        self.stream_synchronize(self.copy_stream)
    }

    /// Run `f` with the pinned bounce buffer (allocated on first use). The
    /// lock serializes staged copies, which share the copy stream anyway.
    pub(crate) fn with_staging<T>(&self, f: impl FnOnce(NonNull<u8>, usize) -> Result<T>) -> Result<T> {
        let mut staging = self.staging.lock();
        let pointer = match *staging {
            Some(pointer) => pointer,
            None => {
                let api = self.enter()?;
                let mut raw = null_mut();
                // SAFETY: out-pointer to a live pointer slot.
                unsafe { (api.mem_host_alloc)(&mut raw, STAGING_BYTES, CU_MEMHOSTALLOC_PORTABLE) }
                    .check("cuMemHostAlloc")?;
                let pointer = NonNull::new(raw.cast::<u8>())
                    .ok_or_else(|| Error::Runtime { message: "cuMemHostAlloc returned null".into() })?;
                *staging = Some(pointer);
                pointer
            }
        };
        f(pointer, STAGING_BYTES)
    }

    /// `true` once a sticky driver error has poisoned the context.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Latch a fatal error: the context is unusable, the message is kept.
    pub fn poison(&self, message: &str) {
        let _ = self.poison_message.set(message.to_string());
        self.poisoned.store(true, Ordering::Release);
    }

    pub fn poison_error(&self) -> Option<Error> {
        self.is_poisoned().then(|| Error::Runtime {
            message: self.poison_message.get().cloned().unwrap_or_else(|| "CUDA device poisoned".into()),
        })
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        let api = self.api;
        // SAFETY: handles this device created; the context is released last.
        unsafe {
            if (api.ctx_set_current)(self.context) != CUresult::SUCCESS {
                return;
            }
            if let Some(staging) = self.staging.get_mut().take() {
                (api.mem_free_host)(staging.as_ptr().cast());
            }
            (api.event_destroy)(self.base_event);
            (api.stream_destroy)(self.copy_stream);
            (api.stream_destroy)(self.dispatch_stream);
            (api.device_primary_ctx_release)(self.handle);
        }
    }
}

/// Releases the primary context if `open_uncached` fails midway.
struct ContextGuard {
    api: &'static Api,
    handle: CUdevice,
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        // SAFETY: balances the retain that created this guard.
        unsafe { (self.api.device_primary_ctx_release)(self.handle) };
    }
}

/// An owned stream of a device.
pub struct CudaStream {
    dev: Arc<CudaDevice>,
    raw: CUstream,
}

impl CudaStream {
    /// A non-blocking stream (not ordered against the legacy default stream).
    pub fn new(dev: Arc<CudaDevice>) -> Result<Self> {
        let api = dev.enter()?;
        let mut raw = CUstream::NULL;
        // SAFETY: out-pointer to a live handle slot.
        unsafe { (api.stream_create)(&mut raw, CU_STREAM_NON_BLOCKING) }.check("cuStreamCreate")?;
        Ok(Self { dev, raw })
    }

    pub fn raw(&self) -> CUstream {
        self.raw
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.dev
    }

    pub fn synchronize(&self) -> Result<()> {
        self.dev.stream_synchronize(self.raw)
    }

    /// Record a fresh event at the current tail of this stream.
    pub fn record(&self, timing: bool) -> Result<Arc<CudaEvent>> {
        let event = CudaEvent::new(Arc::clone(&self.dev), timing)?;
        event.record(self.raw)?;
        Ok(Arc::new(event))
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let api = self.dev.api();
        // SAFETY: a stream this value created; destruction is deferred by the
        // driver until its work retires.
        unsafe { (api.stream_destroy)(self.raw) };
    }
}

/// An owned event of a device.
pub struct CudaEvent {
    dev: Arc<CudaDevice>,
    raw: CUevent,
}

impl CudaEvent {
    /// `timing` events carry GPU timestamps (`cuEventElapsedTime`);
    /// completion-only events skip them and are cheaper to record.
    pub fn new(dev: Arc<CudaDevice>, timing: bool) -> Result<Self> {
        let api = dev.enter()?;
        let mut raw = CUevent::NULL;
        let flags = if timing { CU_EVENT_DEFAULT } else { CU_EVENT_DISABLE_TIMING };
        // SAFETY: out-pointer to a live handle slot.
        unsafe { (api.event_create)(&mut raw, flags) }.check("cuEventCreate")?;
        Ok(Self { dev, raw })
    }

    pub fn raw(&self) -> CUevent {
        self.raw
    }

    pub fn record(&self, stream: CUstream) -> Result<()> {
        let api = self.dev.enter()?;
        // SAFETY: live event and stream of this context.
        self.dev.check(unsafe { (api.event_record)(self.raw, stream) }, "cuEventRecord")
    }

    /// Whether the recorded work has completed (`cuEventQuery`). An event
    /// never recorded counts as completed, as the driver defines it.
    pub fn completed(&self) -> Result<bool> {
        let api = self.dev.enter()?;
        // SAFETY: a live event.
        match unsafe { (api.event_query)(self.raw) } {
            CUresult::SUCCESS => Ok(true),
            CUresult::NOT_READY => Ok(false),
            other => self.dev.check(other, "cuEventQuery").map(|()| true),
        }
    }

    /// Block until completion; `timeout_ms == 0` waits forever.
    pub fn wait(&self, timeout_ms: u64) -> Result<()> {
        if timeout_ms == 0 {
            let api = self.dev.enter()?;
            // SAFETY: a live event.
            return self.dev.check(unsafe { (api.event_synchronize)(self.raw) }, "cuEventSynchronize");
        }
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        while !self.completed()? {
            if Instant::now() >= deadline {
                return TimelineTimeoutSnafu { what: "CUDA event", target: 1u64, current: 0u64, waited_ms: timeout_ms }
                    .fail();
            }
            std::thread::sleep(EVENT_POLL);
        }
        Ok(())
    }

    /// Milliseconds on the GPU clock from `start` to `self`; both must be
    /// completed timing events.
    pub fn elapsed_ms_since(&self, start: CUevent) -> Result<f32> {
        let api = self.dev.enter()?;
        let mut ms = 0f32;
        // SAFETY: out-pointer to a live float; both events are live.
        self.dev.check(unsafe { (api.event_elapsed_time)(&mut ms, start, self.raw) }, "cuEventElapsedTime")?;
        Ok(ms)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let api = self.dev.api();
        // SAFETY: an event this value created; the driver defers destruction
        // until it completes.
        unsafe { (api.event_destroy)(self.raw) };
    }
}

impl std::fmt::Debug for CudaEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaEvent({:p})", self.raw.0)
    }
}
