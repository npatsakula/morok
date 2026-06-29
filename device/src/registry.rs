use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::RwLock;

pub use svod_dtype::DeviceSpec;

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
        // DISK: preserve path case (don't uppercase)
        if s.len() >= 5 && s[..5].eq_ignore_ascii_case("DISK:") {
            return Ok(DeviceSpec::Disk { path: std::path::PathBuf::from(&s[5..]) });
        }

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
            "AMD" | "HIP" => {
                // Format: AMD | AMD:N. The arch lives on the opened
                // `AmdDevice` (resolved from KFD topology); it's intentionally
                // not part of the DeviceSpec to avoid identity ambiguity
                // (two specs for one physical device).
                let device_id = if parts.len() > 1 {
                    parts[1].parse().map_err(|_| crate::error::Error::InvalidDevice { device: s.to_string() })?
                } else {
                    0
                };
                Ok(DeviceSpec::Amd { device_id })
            }
            _ => InvalidDeviceSnafu { device: s }.fail(),
        }
    }
}

#[derive(Default)]
pub struct DeviceRegistry {
    devices: RwLock<HashMap<DeviceSpec, Arc<dyn Allocator>>>,
}

impl DeviceRegistry {
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
        // DISK: no LRU caching — DiskAllocator is used directly, not wrapped in LruAllocator.
        if let DeviceSpec::Disk { path } = spec {
            return Ok(Arc::new(crate::allocator::DiskAllocator::new(path.clone())));
        }

        let base: Box<dyn Allocator> = match spec {
            DeviceSpec::Cpu => Box::new(CpuAllocator),
            #[cfg(feature = "cuda")]
            DeviceSpec::Cuda { device_id } => Box::new(crate::allocator::CudaAllocator::new(*device_id)?),
            #[cfg(not(feature = "cuda"))]
            DeviceSpec::Cuda { .. } => {
                return Err(crate::error::Error::DeviceUnavailable {
                    reason: "CUDA device requested but the `cuda` feature is not enabled".into(),
                });
            }
            DeviceSpec::Amd { device_id, .. } => Box::new(crate::amd::AmdAllocator::new(*device_id)?),
            DeviceSpec::Metal { .. } => {
                return Err(crate::error::Error::DeviceUnavailable {
                    reason: "Metal allocator is not yet implemented".into(),
                });
            }
            DeviceSpec::WebGpu => {
                return Err(crate::error::Error::DeviceUnavailable {
                    reason: "WebGPU allocator is not yet implemented".into(),
                });
            }
            DeviceSpec::Disk { .. } => unreachable!(),
        };

        // Wrap with LRU cache (already thread-safe via Mutex)
        let lru = LruAllocator::new(base);

        Ok(Arc::new(lru))
    }
}

/// Global device registry instance.
static REGISTRY: Lazy<DeviceRegistry> = Lazy::new(DeviceRegistry::default);

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

/// Read the gfx arch **and CU count** of AMD device `device_id` from KFD topology
/// in a single probe. The CU count is `simd_count / simd_per_cu` (the whole-device
/// total, which varies by SKU and virtualized partition) — the real number kernels
/// size their grid-saturation heuristics against instead of a hard-coded constant.
pub fn resolve_amd_arch_and_cu(device_id: usize) -> Result<(svod_dtype::AmdArch, usize)> {
    let nodes = crate::amd::topology::enumerate();
    let node = nodes.get(device_id).ok_or_else(|| crate::error::Error::NoAmdGpu {
        reason: format!("device_id {device_id} out of range; {} GPU node(s) present", nodes.len()),
    })?;
    let arch = svod_dtype::AmdArch::from_gfx_target_version(node.gfx_target_version).ok_or_else(|| {
        crate::error::Error::DeviceUnavailable {
            reason: format!("unsupported gfx_target_version {} on AMD device {device_id}", node.gfx_target_version),
        }
    })?;
    let cu_count = (node.simd_count / node.simd_per_cu.max(1)).max(1) as usize;
    Ok((arch, cu_count))
}

/// Read the gfx arch of AMD device `device_id` from KFD topology. Used by
/// `DeviceSpec::parse("AMD:N")` so the resulting spec encodes the real arch
/// (not a hard-coded default that would break the kernel cache).
pub fn resolve_amd_arch_from_topology(device_id: usize) -> Result<svod_dtype::AmdArch> {
    resolve_amd_arch_and_cu(device_id).map(|(arch, _)| arch)
}

/// Convenience function to get CUDA allocator.
#[cfg(feature = "cuda")]
pub fn cuda(device_id: usize) -> Result<Arc<dyn Allocator>> {
    registry().get(&DeviceSpec::Cuda { device_id })
}
