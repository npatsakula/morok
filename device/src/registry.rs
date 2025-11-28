use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::RwLock;

use crate::allocator::{Allocator, CpuAllocator, LruAllocator};
use crate::error::{InvalidDeviceSnafu, Result};

/// Device specification parsed from a device string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceSpec {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda {
        device_id: usize,
    },
    #[cfg(feature = "metal")]
    Metal {
        device_id: usize,
    },
    #[cfg(feature = "webgpu")]
    WebGpu,
}

impl DeviceSpec {
    /// Get maximum buffer count for this device.
    ///
    /// Returns None if the device has no buffer limit (effectively unlimited).
    ///
    /// Known limits:
    /// - Metal: 31 buffers (Apple Silicon hardware limit)
    /// - WebGPU: 8 buffers (WebGPU specification limit)
    /// - CPU/CUDA: None (no practical limit)
    pub fn max_buffers(&self) -> Option<usize> {
        match self {
            DeviceSpec::Cpu => None,
            #[cfg(feature = "cuda")]
            DeviceSpec::Cuda { .. } => None, // 128+ buffers, effectively unlimited
            #[cfg(feature = "metal")]
            DeviceSpec::Metal { .. } => Some(31),
            #[cfg(feature = "webgpu")]
            DeviceSpec::WebGpu => Some(8),
        }
    }

    /// Parse a device string into a DeviceSpec.
    ///
    /// Examples:
    /// - "CPU" -> DeviceSpec::Cpu
    /// - "CUDA:0" -> DeviceSpec::Cuda { device_id: 0 }
    /// - "cuda" -> DeviceSpec::Cuda { device_id: 0 } (default to device 0)
    pub fn parse(s: &str) -> Result<Self> {
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
            #[cfg(feature = "metal")]
            "METAL" => {
                let _device_id = if parts.len() > 1 {
                    parts[1].parse().map_err(|_| crate::error::Error::InvalidDevice { device: s.to_string() })?
                } else {
                    0
                };
                unimplemented!("Metal device support - to be implemented")
            }
            #[cfg(feature = "webgpu")]
            "WEBGPU" => {
                unimplemented!("WebGPU device support - to be implemented")
            }
            _ => InvalidDeviceSnafu { device: s }.fail(),
        }
    }

    /// Canonicalize the device spec to a standard string representation.
    pub fn canonicalize(&self) -> String {
        match self {
            DeviceSpec::Cpu => "CPU".to_string(),
            #[cfg(feature = "cuda")]
            DeviceSpec::Cuda { device_id } => format!("CUDA:{}", device_id),
            #[cfg(feature = "metal")]
            DeviceSpec::Metal { .. } => unimplemented!("Metal device support - to be implemented"),
            #[cfg(feature = "webgpu")]
            DeviceSpec::WebGpu => unimplemented!("WebGPU device support - to be implemented"),
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
        let spec = DeviceSpec::parse(device)?;
        self.get(&spec)
    }

    fn create_allocator(&self, spec: &DeviceSpec) -> Result<Arc<dyn Allocator>> {
        let base: Box<dyn Allocator> = match spec {
            DeviceSpec::Cpu => Box::new(CpuAllocator),
            #[cfg(feature = "cuda")]
            DeviceSpec::Cuda { device_id } => Box::new(crate::allocator::CudaAllocator::new(*device_id)?),
            #[cfg(feature = "metal")]
            DeviceSpec::Metal { .. } => unimplemented!("Metal allocator - to be implemented"),
            #[cfg(feature = "webgpu")]
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
