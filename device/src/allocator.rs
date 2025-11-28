use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice, UnifiedSlice};
use snafu::ResultExt;

use crate::error::{CudaSnafu, Result};

/// Opaque handle to device memory.
///
/// Uses `RefCell` for interior mutability with runtime borrow checking.
/// Safe for single-threaded use (Buffer is !Send + !Sync).
#[derive(Debug)]
pub enum RawBuffer {
    Cpu {
        data: RefCell<Box<[u8]>>,
        cpu_accessible: bool,
    },
    #[cfg(feature = "cuda")]
    CudaDevice {
        data: RefCell<CudaSlice<u8>>,
        device: Arc<CudaContext>,
    },
    #[cfg(feature = "cuda")]
    CudaUnified {
        data: RefCell<UnifiedSlice<u8>>,
        device: Arc<CudaContext>,
    },
}

impl RawBuffer {
    /// Get the size of the buffer in bytes.
    pub fn size(&self) -> usize {
        match self {
            RawBuffer::Cpu { data, .. } => data.borrow().len(),
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { data, .. } => data.borrow().len(),
            #[cfg(feature = "cuda")]
            RawBuffer::CudaUnified { data, .. } => data.borrow().len(),
        }
    }

    /// Get whether this buffer is CPU-accessible.
    pub fn cpu_accessible(&self) -> bool {
        match self {
            RawBuffer::Cpu { cpu_accessible, .. } => *cpu_accessible,
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { .. } => false,
            #[cfg(feature = "cuda")]
            RawBuffer::CudaUnified { .. } => true,
        }
    }
}

/// Options for buffer allocation.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "proptest", derive(proptest_derive::Arbitrary))]
pub struct BufferOptions {
    /// Whether to zero-initialize the buffer.
    pub zero_init: bool,
    /// Hint that this buffer will be accessed from CPU (for unified memory).
    ///
    /// NOTE: CUDA unified memory is not yet implemented. Setting this to `true`
    /// with CudaAllocator will panic in debug builds.
    pub cpu_accessible: bool,
}

pub trait Allocator: Send + Sync + std::fmt::Debug {
    fn alloc(&self, size: usize, options: &BufferOptions) -> Result<RawBuffer>;
    fn free(&self, _buffer: RawBuffer, _options: &BufferOptions) {}
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
    fn name(&self) -> &str;
}

/// CPU allocator using system memory.
#[derive(Debug, Clone)]
pub struct CpuAllocator;

impl Allocator for CpuAllocator {
    fn alloc(&self, size: usize, options: &BufferOptions) -> Result<RawBuffer> {
        let data = vec![0u8; size].into_boxed_slice();
        Ok(RawBuffer::Cpu { data: RefCell::new(data), cpu_accessible: options.cpu_accessible })
    }

    fn name(&self) -> &str {
        "CPU"
    }
}

/// CUDA allocator using GPU memory.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaAllocator {
    device: Arc<CudaContext>,
    device_id: usize,
}

#[cfg(feature = "cuda")]
impl CudaAllocator {
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaContext::new(device_id).context(CudaSnafu)?;
        Ok(Self { device, device_id })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

#[cfg(feature = "cuda")]
impl Allocator for CudaAllocator {
    fn alloc(&self, size: usize, options: &BufferOptions) -> Result<RawBuffer> {
        if options.cpu_accessible {
            // Allocate unified memory (CPU-accessible)
            let mut data = unsafe { self.device.alloc_unified::<u8>(size, true) }.context(CudaSnafu)?;

            if options.zero_init {
                self.device.default_stream().memset_zeros(&mut data).context(CudaSnafu)?;
            }

            Ok(RawBuffer::CudaUnified { data: RefCell::new(data), device: Arc::clone(&self.device) })
        } else {
            // Allocate device-only memory (faster GPU access)
            let stream = self.device.default_stream();
            let data =
                if options.zero_init { stream.alloc_zeros::<u8>(size) } else { unsafe { stream.alloc::<u8>(size) } }
                    .context(CudaSnafu)?;

            Ok(RawBuffer::CudaDevice { data: RefCell::new(data), device: Arc::clone(&self.device) })
        }
    }

    fn synchronize(&self) -> Result<()> {
        self.device.default_stream().synchronize().context(CudaSnafu)
    }

    fn name(&self) -> &str {
        "CUDA"
    }
}

/// Cache key for buffer reuse in LRU allocator.
///
/// Includes size and cpu_accessible (hardware property that affects allocation).
/// zero_init is NOT included - it's a software operation handled after cache retrieval.
///
/// Design rationale (following Tinygrad):
/// - cpu_accessible is included because it represents different memory types:
///   - false: Device-only memory (cuMemAlloc) - faster GPU access
///   - true: Unified memory (cuMemAllocManaged) - CPU-accessible, not yet implemented
/// - These are immutable hardware properties that cannot be changed post-allocation
/// - Buffers allocated with different cpu_accessible values cannot be safely reused
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct CacheKey {
    size: usize,
    cpu_accessible: bool,
}

/// LRU allocator that caches freed buffers for reuse.
#[derive(Debug)]
pub(crate) struct LruAllocator {
    inner: Box<dyn Allocator>,
    cache: Mutex<HashMap<CacheKey, Vec<RawBuffer>>>,
    max_buffers_per_size: usize,
    name: String,
}

impl LruAllocator {
    pub fn new(inner: Box<dyn Allocator>) -> Self {
        Self::with_capacity(inner, 32)
    }

    pub fn with_capacity(inner: Box<dyn Allocator>, max_buffers_per_size: usize) -> Self {
        let name = inner.name().to_string();
        Self { inner, cache: Mutex::new(HashMap::new()), max_buffers_per_size, name }
    }

    /// Get the number of cached buffers for a specific size and cpu_accessible flag.
    /// Only available in tests for cache introspection.
    #[cfg(test)]
    pub(crate) fn cache_count(&self, size: usize, cpu_accessible: bool) -> usize {
        let key = CacheKey { size, cpu_accessible };
        let cache = self.cache.lock().unwrap();
        cache.get(&key).map(|v| v.len()).unwrap_or(0)
    }

    /// Get the total number of cached buffers across all keys.
    /// Only available in tests for cache introspection.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn total_cached(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.values().map(|v| v.len()).sum()
    }
}

impl Allocator for LruAllocator {
    fn alloc(&self, size: usize, options: &BufferOptions) -> Result<RawBuffer> {
        let key = CacheKey { size, cpu_accessible: options.cpu_accessible };

        // Try cache first
        let buffer = {
            let mut cache = self.cache.lock().unwrap();
            if let Some(buffers) = cache.get_mut(&key)
                && let Some(buffer) = buffers.pop()
            {
                if buffers.is_empty() {
                    cache.remove(&key);
                }
                Some(buffer)
            } else {
                None
            }
        }; // Drop lock before expensive allocation

        // If found in cache, optionally zero and return
        if let Some(buffer) = buffer {
            if options.zero_init {
                // Zero the cached buffer if requested
                match &buffer {
                    RawBuffer::Cpu { data, .. } => {
                        data.borrow_mut().fill(0);
                    }
                    #[cfg(feature = "cuda")]
                    RawBuffer::CudaDevice { data, device } => {
                        let mut cuda_data = data.borrow_mut();
                        device.default_stream().memset_zeros(&mut *cuda_data).context(CudaSnafu)?;
                    }
                    #[cfg(feature = "cuda")]
                    RawBuffer::CudaUnified { data, device } => {
                        let mut unified_data = data.borrow_mut();
                        device.default_stream().memset_zeros(&mut *unified_data).context(CudaSnafu)?;
                    }
                }
            }
            return Ok(buffer);
        }

        // Cache miss - allocate from inner
        match self.inner.alloc(size, options) {
            Ok(buffer) => Ok(buffer),
            Err(e) => {
                // On allocation failure, clear cache and retry
                self.cache.lock().unwrap().clear();
                self.inner.alloc(size, options).map_err(|_| e)
            }
        }
    }

    fn free(&self, buffer: RawBuffer, options: &BufferOptions) {
        let key = CacheKey { size: buffer.size(), cpu_accessible: options.cpu_accessible };

        let mut cache = self.cache.lock().unwrap();
        let buffers = cache.entry(key).or_default();
        if buffers.len() < self.max_buffers_per_size {
            buffers.push(buffer);
        }
    }

    fn synchronize(&self) -> Result<()> {
        self.inner.synchronize()
    }

    fn name(&self) -> &str {
        &self.name
    }
}
