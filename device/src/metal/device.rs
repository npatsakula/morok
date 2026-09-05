//! The opened Metal device: `MTLDevice` + command queue, the in-flight
//! command-buffer list, and the host-pointer → `MTLBuffer` registry.

use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;
use std::sync::{Arc, LazyLock, OnceLock};

use parking_lot::Mutex;

use super::objc::{
    AutoreleasePool, Id, MTL_COMMAND_BUFFER_STATUS_COMPLETED, MTL_GPU_FAMILY_APPLE, MTL_GPU_FAMILY_MAC2, NSInteger,
    NSUInteger, Objc, ObjcBool, ObjcId, ns_error_message, ns_string_to_string, objc,
};
use crate::{Error, Result};

const COMMAND_QUEUE_DEPTH: NSUInteger = 1024;

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
    family: String,
    /// Committed, possibly unfinished command buffers; drained by [`Self::synchronize`].
    in_flight: Mutex<Vec<ObjcId>>,
    /// `contents` base → (MTLBuffer, allocated length).
    registry: Mutex<PointerRegistry<ObjcId>>,
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
        let family = MTL_GPU_FAMILY_APPLE
            .iter()
            .chain(std::iter::once(&MTL_GPU_FAMILY_MAC2))
            .find(|(family, _)| supports(*family))
            .map_or("Unknown", |(_, label)| label)
            .to_string();
        tracing::info!(device_id, name, family, "opened Metal device");
        Ok(Self {
            objc,
            device_id,
            mtl,
            queue,
            name,
            family,
            in_flight: Mutex::new(Vec::new()),
            registry: Mutex::new(PointerRegistry::default()),
        })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// `MTLGPUFamily` label such as `Apple9` or `Mac2`.
    pub fn family(&self) -> &str {
        &self.family
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Paravirtualized GPUs (CI VMs) break indirect command buffers.
    pub fn supports_graph(&self) -> bool {
        !self.name.to_lowercase().contains("virtual")
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
            // SAFETY: plain void selector, then an autoreleased NSError (or nil).
            let message = unsafe {
                self.objc.send0::<()>(command_buffer.as_raw(), self.objc.sels.wait_until_completed);
                ns_error_message(self.objc, self.objc.send0::<Id>(command_buffer.as_raw(), self.objc.sels.error))
            };
            if let Some(message) = message
                && first_error.is_none()
            {
                // SAFETY: `label` returns an autoreleased NSString or nil.
                let label = unsafe {
                    ns_string_to_string(self.objc, self.objc.send0::<Id>(command_buffer.as_raw(), self.objc.sels.label))
                };
                first_error =
                    Some(Error::Runtime { message: format!("Metal command buffer '{label}' failed: {message}") });
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}
