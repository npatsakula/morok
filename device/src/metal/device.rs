//! The opened Metal device: `MTLDevice` + command queue, the in-flight
//! command-buffer list, and the host-pointer → `MTLBuffer` registry.

use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;
use std::sync::{Arc, LazyLock, OnceLock};

use parking_lot::Mutex;

use svod_dtype::MetalFamily;

use super::objc::{
    AutoreleasePool, Id, MTL_COMMAND_BUFFER_STATUS_COMPLETED, MTL_GPU_FAMILY_APPLE_BASE, MTL_GPU_FAMILY_MAC2,
    NSInteger, NSUInteger, Objc, ObjcBool, ObjcId, ns_error_message, ns_string_to_string, objc,
};
use crate::{Error, Result};

const COMMAND_QUEUE_DEPTH: NSUInteger = 1024;
/// Newest Apple GPU generation probed with `supportsFamily:`; unknown values
/// simply answer NO, so this only needs to stay ahead of shipping hardware.
const NEWEST_APPLE_GENERATION: u8 = 12;

static DEVICE_CACHE: LazyLock<Mutex<HashMap<usize, Arc<MetalDevice>>>> = LazyLock::new(Default::default);
static HAS_DEVICES: OnceLock<bool> = OnceLock::new();

/// Whether the Apple frameworks load and a default GPU exists. Memoized; the
/// probe device is released immediately and no queue is created.
pub fn has_devices() -> bool {
    *HAS_DEVICES.get_or_init(|| {
        let Ok(objc) = objc() else { return false };
        let _pool = AutoreleasePool::push(objc);
        // SAFETY: returns a +1 device or nil.
        unsafe { ObjcId::adopt((objc.create_system_default_device)()) }.is_some()
    })
}

/// Live host-address ranges → the record registered with them. The greatest
/// base `<= address` is the only range that can contain it.
#[derive(Debug)]
pub(crate) struct PointerRegistry<T> {
    live: BTreeMap<usize, (T, usize)>,
}

impl<T> Default for PointerRegistry<T> {
    fn default() -> Self {
        Self { live: BTreeMap::new() }
    }
}

impl<T> PointerRegistry<T> {
    pub(crate) fn insert(&mut self, base: usize, len: usize, record: T) {
        self.live.insert(base, (record, len));
    }

    pub(crate) fn remove(&mut self, base: usize) -> Option<T> {
        self.live.remove(&base).map(|(record, _)| record)
    }

    /// The record containing `address` and the offset into it.
    pub(crate) fn resolve(&self, address: usize) -> Result<(&T, usize)> {
        if let Some((&base, (record, len))) = self.live.range(..=address).next_back()
            && address < base.saturating_add(*len)
        {
            return Ok((record, address - base));
        }
        let describe = |entry: Option<(&usize, &(T, usize))>| match entry {
            Some((base, (_, len))) => format!("[{base:#x}, {:#x})", base.saturating_add(*len)),
            None => "none".to_string(),
        };
        Err(Error::Runtime {
            message: format!(
                "Metal buffer pointer {address:#x} is in no registered MTLBuffer (nearest below {}, nearest above {})",
                describe(self.live.range(..=address).next_back()),
                describe(self.live.range((Bound::Excluded(address), Bound::Unbounded)).next()),
            ),
        })
    }
}

pub struct MetalDevice {
    objc: &'static Objc,
    device_id: usize,
    mtl: ObjcId,
    queue: ObjcId,
    name: String,
    family: MetalFamily,
    /// Committed, possibly unfinished command buffers; drained by [`Self::synchronize`].
    in_flight: Mutex<Vec<ObjcId>>,
    /// `contents` base → (MTLBuffer, allocated length).
    registry: Mutex<PointerRegistry<ObjcId>>,
    /// Metal 4 timestamp queue, opened on first profiled replay; `None` below macOS 26.
    mtl4: OnceLock<Option<super::mtl4::Mtl4Profiler>>,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDevice")
            .field("device_id", &self.device_id)
            .field("name", &self.name)
            .field("family", &self.family)
            .finish_non_exhaustive()
    }
}

impl MetalDevice {
    /// Process-global cached open. Only the system default device (`id 0`) is
    /// supported; enumerating `MTLCopyAllDevices` is a follow-up.
    pub fn open(device_id: usize) -> Result<Arc<Self>> {
        let mut cache = DEVICE_CACHE.lock();
        if let Some(device) = cache.get(&device_id) {
            return Ok(device.clone());
        }
        if device_id != 0 {
            return Err(Error::DeviceUnavailable { reason: format!("Metal device {device_id} does not exist") });
        }
        let device = Arc::new(Self::open_default(device_id)?);
        cache.insert(device_id, device.clone());
        Ok(device)
    }

    fn open_default(device_id: usize) -> Result<Self> {
        let objc = objc()?;
        let _pool = AutoreleasePool::push(objc);
        let sels = &objc.sels;
        // SAFETY: +1 device or nil.
        let mtl = unsafe { ObjcId::adopt((objc.create_system_default_device)()) }
            .ok_or_else(|| Error::DeviceUnavailable { reason: "MTLCreateSystemDefaultDevice returned nil".into() })?;
        // SAFETY: `newCommandQueueWithMaxCommandBufferCount:` takes NSUInteger, returns +1 or nil.
        let queue = unsafe {
            ObjcId::adopt(objc.send1::<NSUInteger, Id>(
                mtl.as_raw(),
                sels.new_command_queue_with_max,
                COMMAND_QUEUE_DEPTH,
            ))
        }
        .ok_or_else(|| Error::DeviceUnavailable { reason: "Metal command queue creation failed".into() })?;
        // SAFETY: `name` returns an autoreleased NSString.
        let name = unsafe { ns_string_to_string(objc, objc.send0::<Id>(mtl.as_raw(), sels.name)) };
        let supports = |family: NSInteger| -> bool {
            // SAFETY: `supportsFamily:` takes NSInteger and returns BOOL.
            unsafe { objc.send1::<NSInteger, ObjcBool>(mtl.as_raw(), sels.supports_family, family) != 0 }
        };
        let family = (1..=NEWEST_APPLE_GENERATION)
            .rev()
            .find(|generation| supports(MTL_GPU_FAMILY_APPLE_BASE + NSInteger::from(*generation)))
            .map(MetalFamily::Apple)
            .unwrap_or(if supports(MTL_GPU_FAMILY_MAC2) { MetalFamily::Mac2 } else { MetalFamily::Unknown });
        tracing::info!(device_id, name, %family, "opened Metal device");
        Ok(Self {
            objc,
            device_id,
            mtl,
            queue,
            name,
            family,
            in_flight: Mutex::new(Vec::new()),
            registry: Mutex::new(PointerRegistry::default()),
            mtl4: OnceLock::new(),
        })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// The highest `MTLGPUFamily` the GPU supports.
    pub fn family(&self) -> MetalFamily {
        self.family
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Paravirtualized GPUs (CI VMs) break indirect command buffers.
    pub fn supports_graph(&self) -> bool {
        !self.name.to_lowercase().contains("virtual")
    }

    /// Per-dispatch GPU timestamps, when the OS offers Metal 4.
    pub fn mtl4(&self) -> Option<&super::mtl4::Mtl4Profiler> {
        self.mtl4.get_or_init(|| super::mtl4::Mtl4Profiler::open(self)).as_ref()
    }

    pub(crate) fn objc(&self) -> &'static Objc {
        self.objc
    }

    pub(crate) fn mtl(&self) -> Id {
        self.mtl.as_raw()
    }

    pub(crate) fn queue(&self) -> Id {
        self.queue.as_raw()
    }

    pub(crate) fn register_buffer(&self, base: usize, len: usize, buffer: ObjcId) {
        self.registry.lock().insert(base, len, buffer);
    }

    pub(crate) fn unregister_buffer(&self, base: usize) -> Option<ObjcId> {
        self.registry.lock().remove(base)
    }

    /// The `MTLBuffer` and byte offset a host pointer refers to. The returned
    /// object is kept alive by the `RawBuffer` that owns the pointer.
    pub(crate) fn resolve(&self, pointer: *mut u8) -> Result<(Id, NSUInteger)> {
        let registry = self.registry.lock();
        let (buffer, offset) = registry.resolve(pointer as usize)?;
        Ok((buffer.as_raw(), offset as NSUInteger))
    }

    /// [`Self::resolve`] plus an owning reference, for holders that outlive the
    /// `RawBuffer` (an indirect command buffer keeps its bindings across frees).
    pub(crate) fn resolve_retained(&self, pointer: *mut u8) -> Result<(ObjcId, NSUInteger)> {
        let registry = self.registry.lock();
        let (buffer, offset) = registry.resolve(pointer as usize)?;
        Ok((buffer.clone(), offset as NSUInteger))
    }

    /// A fresh command buffer on the queue and a compute encoder on it, both
    /// retained. The caller must `endEncoding` before releasing the encoder.
    pub(crate) fn begin_compute(&self) -> Result<(ObjcId, ObjcId)> {
        let sels = &self.objc.sels;
        // SAFETY: `commandBuffer` / `computeCommandEncoder` return autoreleased objects.
        let command_buffer =
            unsafe { ObjcId::retain(self.objc, self.objc.send0::<Id>(self.queue(), sels.command_buffer)) }
                .ok_or_else(|| Error::Runtime { message: "Metal command queue returned no command buffer".into() })?;
        let encoder = unsafe {
            ObjcId::retain(self.objc, self.objc.send0::<Id>(command_buffer.as_raw(), sels.compute_command_encoder))
        }
        .ok_or_else(|| Error::Runtime { message: "Metal command buffer returned no compute encoder".into() })?;
        Ok((command_buffer, encoder))
    }

    /// [`Self::resolve_retained`] for a flattened list of host addresses.
    pub(crate) fn resolve_all(&self, addresses: &[u64]) -> Result<Vec<(ObjcId, NSUInteger)>> {
        let registry = self.registry.lock();
        addresses
            .iter()
            .map(|address| {
                let (buffer, offset) = registry.resolve(*address as usize)?;
                Ok((buffer.clone(), offset as NSUInteger))
            })
            .collect()
    }

    /// Block until a committed command buffer retires; its `NSError`, if any,
    /// becomes `Error::Runtime` prefixed with `what`.
    pub(crate) fn wait_command_buffer(&self, command_buffer: Id, what: &str) -> Result<()> {
        // SAFETY: blocking wait, then an autoreleased NSError (or nil).
        let message = unsafe {
            self.objc.send0::<()>(command_buffer, self.objc.sels.wait_until_completed);
            ns_error_message(self.objc, self.objc.send0::<Id>(command_buffer, self.objc.sels.error))
        };
        message.map_or(Ok(()), |message| Err(Error::Runtime { message: format!("{what} failed: {message}") }))
    }

    /// `(GPUStartTime, GPUEndTime)` in seconds of a completed command buffer.
    pub(crate) fn gpu_times(&self, command_buffer: Id) -> (f64, f64) {
        // SAFETY: two `CFTimeInterval` (double) accessors.
        unsafe {
            (
                self.objc.send0::<f64>(command_buffer, self.objc.sels.gpu_start_time),
                self.objc.send0::<f64>(command_buffer, self.objc.sels.gpu_end_time),
            )
        }
    }

    /// Track a committed command buffer, dropping the ones already completed.
    pub(crate) fn push_in_flight(&self, command_buffer: ObjcId) {
        let mut in_flight = self.in_flight.lock();
        in_flight.retain(|cb| {
            // SAFETY: `status` returns an NSUInteger.
            let status = unsafe { self.objc.send0::<NSUInteger>(cb.as_raw(), self.objc.sels.status) };
            status != MTL_COMMAND_BUFFER_STATUS_COMPLETED
        });
        in_flight.push(command_buffer);
    }

    /// Wait for every committed command buffer and surface the first failure.
    ///
    /// The list lock is held across the wait: a thread that arrives while
    /// another is draining must not see an empty list and return early while
    /// its own kernel is still being waited on by the other thread.
    pub fn synchronize(&self) -> Result<()> {
        let mut in_flight = self.in_flight.lock();
        if in_flight.is_empty() {
            return Ok(());
        }
        let _pool = AutoreleasePool::push(self.objc);
        let mut first_error = None;
        for command_buffer in in_flight.drain(..) {
            // SAFETY: `label` returns an autoreleased NSString or nil.
            let label = unsafe {
                ns_string_to_string(self.objc, self.objc.send0::<Id>(command_buffer.as_raw(), self.objc.sels.label))
            };
            let result = self.wait_command_buffer(command_buffer.as_raw(), &format!("Metal command buffer '{label}'"));
            if first_error.is_none() {
                first_error = result.err();
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}
