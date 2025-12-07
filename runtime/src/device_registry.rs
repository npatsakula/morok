//! Device factory registry for runtime device creation and caching.
//!
//! This module provides a registry for full Device objects (renderer + compiler + runtime + allocator).
//! It's separate from `morok_device::registry::DeviceRegistry` (which only manages allocators)
//! to avoid circular dependencies between `device` and `runtime` crates.

use std::collections::HashMap;
use std::sync::Arc;

use morok_device::Result as DeviceResult;
use morok_device::device::Device;
use morok_device::registry::DeviceRegistry;
use morok_dtype::DeviceSpec;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use snafu::ResultExt;

use crate::error::{DeviceSnafu, Result, UnsupportedDeviceSnafu};

/// Factory function that creates a Device for a given DeviceSpec.
///
/// The factory receives both the device specification and the allocator registry,
/// allowing it to obtain the correct allocator for the device.
///
/// Returns `DeviceResult<Device>` (from morok_device) since device creation
/// errors come from the device crate.
pub type DeviceFactory = Arc<dyn Fn(&DeviceSpec, &DeviceRegistry) -> DeviceResult<Device> + Send + Sync>;

/// Registry for full Device objects with caching and factory registration.
///
/// # Thread Safety
///
/// This registry uses `parking_lot::RwLock` for efficient concurrent access:
/// - Multiple readers can access cached devices simultaneously
/// - Writers acquire exclusive lock only when creating new devices
/// - Double-checked locking pattern prevents redundant device creation
///
/// # Example
///
/// ```ignore
/// // Get a device (creates if not cached)
/// let alloc_registry = morok_device::registry::registry();
/// let device = DEVICE_FACTORIES.device(&DeviceSpec::Cpu, alloc_registry)?;
///
/// // Register a custom factory
/// DEVICE_FACTORIES.register_factory("CUSTOM", Arc::new(|spec, reg| {
///     // Create custom device...
/// }));
/// ```
pub struct DeviceFactoryRegistry {
    /// Cached device instances (DeviceSpec -> Device)
    devices: RwLock<HashMap<DeviceSpec, Arc<Device>>>,
    /// Registered factories (device type string -> factory function)
    factories: RwLock<HashMap<String, DeviceFactory>>,
}

impl DeviceFactoryRegistry {
    /// Create a new registry with built-in device factories registered.
    pub fn new() -> Self {
        let registry = Self { devices: RwLock::new(HashMap::new()), factories: RwLock::new(HashMap::new()) };

        // Register built-in CPU factory
        registry
            .register_factory("CPU", Arc::new(|_spec, alloc_reg| crate::devices::cpu::create_cpu_device(alloc_reg)));

        // Future: Register CUDA, Metal, WebGPU factories when implemented
        // registry.register_factory("CUDA", Arc::new(|spec, reg| create_cuda_device(spec, reg)));

        registry
    }

    /// Register a device factory for a device type.
    ///
    /// The device type string is case-insensitive (converted to uppercase).
    /// This allows plugins or extensions to register new device types at runtime.
    ///
    /// # Arguments
    ///
    /// * `device_type` - Device type identifier (e.g., "CPU", "CUDA", "METAL")
    /// * `factory` - Factory function that creates Device instances
    pub fn register_factory(&self, device_type: &str, factory: DeviceFactory) {
        self.factories.write().insert(device_type.to_uppercase(), factory);
    }

    /// Get or create a Device for the given specification.
    ///
    /// This method uses double-checked locking for efficiency:
    /// 1. Fast path: Read lock to check cache
    /// 2. Slow path: Write lock to create and cache new device
    ///
    /// # Arguments
    ///
    /// * `spec` - Device specification (e.g., `DeviceSpec::Cpu`)
    /// * `alloc_registry` - Allocator registry for obtaining device allocators
    ///
    /// # Returns
    ///
    /// Arc-wrapped Device for the specification, or error if device type unsupported.
    pub fn device(&self, spec: &DeviceSpec, alloc_registry: &DeviceRegistry) -> Result<Arc<Device>> {
        // Fast path: read lock to check cache
        if let Some(dev) = self.devices.read().get(spec) {
            return Ok(Arc::clone(dev));
        }

        // Slow path: write lock to create
        let mut devices = self.devices.write();

        // Double-check after acquiring write lock (another thread may have created it)
        if let Some(dev) = devices.get(spec) {
            return Ok(Arc::clone(dev));
        }

        // Look up factory for this device type
        let device_type = spec.base_type();
        let factory = self
            .factories
            .read()
            .get(device_type)
            .cloned()
            .ok_or_else(|| UnsupportedDeviceSnafu { device: device_type.to_string() }.build())?;

        // Create device via factory
        let device = factory(spec, alloc_registry).context(DeviceSnafu)?;
        let arc = Arc::new(device);
        devices.insert(spec.clone(), Arc::clone(&arc));
        Ok(arc)
    }
}

impl Default for DeviceFactoryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global device factory registry.
///
/// This static instance is lazily initialized on first access,
/// with built-in device factories automatically registered.
///
/// # Example
///
/// ```ignore
/// let device = morok_runtime::DEVICE_FACTORIES
///     .device(&DeviceSpec::Cpu, morok_device::registry::registry())?;
/// ```
pub static DEVICE_FACTORIES: Lazy<DeviceFactoryRegistry> = Lazy::new(DeviceFactoryRegistry::new);
