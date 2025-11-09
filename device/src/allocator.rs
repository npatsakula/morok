use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice};
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
    },
    #[cfg(feature = "cuda")]
    Cuda {
        data: RefCell<CudaSlice<u8>>,
        device: Arc<CudaContext>,
    },
}

impl RawBuffer {
    /// Get the size of the buffer in bytes.
    pub fn size(&self) -> usize {
        match self {
            RawBuffer::Cpu { data } => data.borrow().len(),
            #[cfg(feature = "cuda")]
            RawBuffer::Cuda { data, .. } => data.borrow().len(),
        }
    }
}

/// Options for buffer allocation.
#[derive(Debug, Clone, Default)]
pub struct BufferOptions {
    /// Whether to zero-initialize the buffer.
    pub zero_init: bool,
    /// Hint that this buffer will be accessed from CPU (for unified memory).
    pub cpu_accessible: bool,
}

pub trait Allocator: Send + Sync + std::fmt::Debug {
    fn alloc(&self, size: usize, options: &BufferOptions) -> Result<RawBuffer>;
    fn free(&self, _buffer: RawBuffer) {}
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
    fn name(&self) -> &str;
}

/// CPU allocator using system memory.
#[derive(Debug, Clone)]
pub struct CpuAllocator;

impl Allocator for CpuAllocator {
    fn alloc(&self, size: usize, _options: &BufferOptions) -> Result<RawBuffer> {
        let data = vec![0u8; size].into_boxed_slice();
        Ok(RawBuffer::Cpu { data: RefCell::new(data) })
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
        let stream = self.device.default_stream();
        let data = if options.zero_init { stream.alloc_zeros::<u8>(size) } else { unsafe { stream.alloc::<u8>(size) } }
            .context(CudaSnafu)?;

        Ok(RawBuffer::Cuda { data: RefCell::new(data), device: Arc::clone(&self.device) })
    }

    fn synchronize(&self) -> Result<()> {
        self.device.default_stream().synchronize().context(CudaSnafu)
    }

    fn name(&self) -> &str {
        "CUDA"
    }
}

/// Cache key for buffer reuse in LRU allocator.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct CacheKey {
    size: usize,
    zero_init: bool,
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
}

impl Allocator for LruAllocator {
    fn alloc(&self, size: usize, options: &BufferOptions) -> Result<RawBuffer> {
        let key = CacheKey { size, zero_init: options.zero_init };

        // Try cache first
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(buffers) = cache.get_mut(&key)
                && let Some(buffer) = buffers.pop()
            {
                if buffers.is_empty() {
                    cache.remove(&key);
                }
                return Ok(buffer);
            }
        } // Drop lock before expensive allocation

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

    fn free(&self, buffer: RawBuffer) {
        let key = CacheKey { size: buffer.size(), zero_init: false };

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
