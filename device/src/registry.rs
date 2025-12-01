use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::RwLock;

pub use morok_dtype::DeviceSpec;

use crate::allocator::{Allocator, CpuAllocator, LruAllocator};
use crate::error::{InvalidDeviceSnafu, Result};

/// Extension trait for DeviceSpec to add parsing functionality.
///
/// This is in the device crate because parsing depends on feature flags
/// and error types that are device-specific.
pub trait DeviceSpecExt {
    /// Parse a device string into a DeviceSpec.
    ///
    /// Examples:
    /// - "CPU" -> DeviceSpec::Cpu
    /// - "CUDA:0" -> DeviceSpec::Cuda { device_id: 0 }
    /// - "cuda" -> DeviceSpec::Cuda { device_id: 0 } (default to device 0)
    fn parse(s: &str) -> Result<DeviceSpec>;
}

impl DeviceSpecExt for DeviceSpec {
    fn parse(s: &str) -> Result<Self> {
        let s = s.to_uppercase();
        let parts: Vec<&str> = s.split(':').collect();

        match parts[0] {
            "CPU" => Ok(DeviceSpec::Cpu),
            #[cfg(feature = "cuda")]
            "CUDA" | "GPU" => {
                let device_id = if parts.len() > 1 {
                    parts[1].parse().map_err(|_| crate::error::Error::InvalidDevice { device: s.to_string() })?
                } else {
                    0
                };
                Ok(DeviceSpec::Cuda { device_id })
            }
            #[cfg(not(feature = "cuda"))]
            "CUDA" | "GPU" => {
                let device_id = if parts.len() > 1 {
                    parts[1].parse().map_err(|_| crate::error::Error::InvalidDevice { device: s.to_string() })?
                } else {
                    0
                };
                Ok(DeviceSpec::Cuda { device_id })
            }
            #[cfg(feature = "metal")]
            "METAL" => {
                let device_id = if parts.len() > 1 {
                    parts[1].parse().map_err(|_| crate::error::Error::InvalidDevice { device: s.to_string() })?
                } else {
                    0
                };
                Ok(DeviceSpec::Metal { device_id })
            }
            #[cfg(not(feature = "metal"))]
            "METAL" => {
                let device_id = if parts.len() > 1 {
                    parts[1].parse().map_err(|_| crate::error::Error::InvalidDevice { device: s.to_string() })?
                } else {
                    0
                };
                Ok(DeviceSpec::Metal { device_id })
            }
            #[cfg(feature = "webgpu")]
            "WEBGPU" => Ok(DeviceSpec::WebGpu),
            #[cfg(not(feature = "webgpu"))]
            "WEBGPU" => Ok(DeviceSpec::WebGpu),
            _ => InvalidDeviceSnafu { device: s }.fail(),
        }
    }
}

pub struct DeviceRegistry {
    devices: RwLock<HashMap<DeviceSpec, Arc<dyn Allocator>>>,
}

impl DeviceRegistry {
    fn new() -> Self {
        Self { devices: RwLock::new(HashMap::new()) }
    }

    /// Get or create a device allocator.
    pub fn get(&self, spec: &DeviceSpec) -> Result<Arc<dyn Allocator>> {
        // Fast path: read lock
        {
            let devices = self.devices.read();
            if let Some(allocator) = devices.get(spec) {
                return Ok(Arc::clone(allocator));
            }
        }

        // Slow path: write lock to create
        let mut devices = self.devices.write();

        // Double-check after acquiring write lock
        if let Some(allocator) = devices.get(spec) {
            return Ok(Arc::clone(allocator));
        }

        // Create new allocator
        let allocator = self.create_allocator(spec)?;
        devices.insert(spec.clone(), Arc::clone(&allocator));
        Ok(allocator)
    }

    /// Get a device by parsing a device string.
    pub fn get_device(&self, device: &str) -> Result<Arc<dyn Allocator>> {
        let spec = <DeviceSpec as DeviceSpecExt>::parse(device)?;
        self.get(&spec)
    }

    fn create_allocator(&self, spec: &DeviceSpec) -> Result<Arc<dyn Allocator>> {
        let base: Box<dyn Allocator> = match spec {
            DeviceSpec::Cpu => Box::new(CpuAllocator),
            #[cfg(feature = "cuda")]
            DeviceSpec::Cuda { device_id } => Box::new(crate::allocator::CudaAllocator::new(*device_id)?),
            #[cfg(not(feature = "cuda"))]
            DeviceSpec::Cuda { .. } => unimplemented!("Cuda allocator - to be implemented"),
            DeviceSpec::Metal { .. } => unimplemented!("Metal allocator - to be implemented"),
            DeviceSpec::WebGpu => unimplemented!("WebGPU allocator - to be implemented"),
        };

        // Wrap with LRU cache (already thread-safe via Mutex)
        let lru = LruAllocator::new(base);

        Ok(Arc::new(lru))
    }
}

/// Global device registry instance.
static REGISTRY: Lazy<DeviceRegistry> = Lazy::new(DeviceRegistry::new);

/// Get the global device registry.
pub fn registry() -> &'static DeviceRegistry {
    &REGISTRY
}

/// Convenience function to get a device allocator by string.
pub fn get_device(device: &str) -> Result<Arc<dyn Allocator>> {
    registry().get_device(device)
}

/// Convenience function to get CPU allocator.
pub fn cpu() -> Result<Arc<dyn Allocator>> {
    registry().get(&DeviceSpec::Cpu)
}

/// Convenience function to get CUDA allocator.
#[cfg(feature = "cuda")]
pub fn cuda(device_id: usize) -> Result<Arc<dyn Allocator>> {
    registry().get(&DeviceSpec::Cuda { device_id })
}
