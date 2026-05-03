use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use morok_dtype::DType;
use smallvec::{SmallVec, smallvec};

use morok_dtype::ext::HasDType;
use snafu::ResultExt;

use crate::allocator::{Allocator, BufferOptions, RawBuffer};
use crate::error::{
    InvalidViewSnafu, NdarrayShapeSnafu, NotCpuAccessibleSnafu, Result, SizeMismatchSnafu, TypeMismatchSnafu,
};

/// Global counter for unique buffer IDs.
///
/// Uses `AtomicU64` to generate unique IDs across threads.
/// IDs are monotonically increasing and never reused.
static BUFFER_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_buffer_id() -> u64 {
    BUFFER_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Unique identifier for a buffer handle.
///
/// Mirrors tinygrad's distinct-identity-per-`BUFFER_VIEW` semantics: each
/// `Buffer` value carries its own `BufferId`, including views — so two
/// disjoint slices of a shared arena have different ids and the parallel
/// hazard model can treat them as independent. Use [`Buffer::storage_id`]
/// when storage-identity (rather than handle-identity) matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u64);

#[cfg(feature = "cuda")]
use crate::error::CudaSnafu;
#[cfg(feature = "cuda")]
use snafu::ResultExt;

/// Shared buffer data that can be referenced by multiple views.
#[derive(Debug)]
struct BufferData {
    /// Stable per-storage identifier minted when the underlying allocation
    /// is created. Distinct from the per-handle [`Buffer::id`]: every
    /// `Buffer` value (including views) gets a fresh handle id, but every
    /// view of one allocation shares the same `storage_id`. Used by code
    /// that needs storage identity (e.g. alias detection in the memory
    /// planner) without falling into the `Arc::as_ptr` aliasing trap.
    storage_id: BufferId,
    /// Lazily-initialized raw buffer (lock-free after first allocation).
    raw: OnceLock<RawBuffer>,
    allocator: Arc<dyn Allocator>,
    /// Total size of the underlying allocation in bytes.
    total_size: usize,
    /// Allocation options.
    options: BufferOptions,
}

impl BufferData {
    fn new(allocator: Arc<dyn Allocator>, size: usize, options: BufferOptions) -> Self {
        Self { storage_id: BufferId(next_buffer_id()), raw: OnceLock::new(), allocator, total_size: size, options }
    }

    /// Ensure the buffer is allocated, allocating if necessary.
    /// Uses lock-free OnceLock for efficient repeated checks.
    fn ensure_allocated(&self) -> Result<()> {
        if self.raw.get().is_some() {
            return Ok(());
        }

        // Allocate - if another thread beat us, that's fine
        let raw = self.allocator.alloc(self.total_size, &self.options)?;

        // Try to set - if another thread beat us, free this allocation
        if let Err(raw) = self.raw.set(raw) {
            // Another thread won the race - free our allocation
            self.allocator.free(raw, &self.options);
        }

        Ok(())
    }

    /// Check if the buffer is currently allocated.
    fn is_allocated(&self) -> bool {
        self.raw.get().is_some()
    }

    /// Get raw buffer reference (buffer must be allocated).
    fn raw(&self) -> &RawBuffer {
        self.raw.get().expect("buffer not allocated")
    }
}

impl Drop for BufferData {
    fn drop(&mut self) {
        // Free the buffer if it was allocated
        if let Some(raw) = self.raw.take() {
            self.allocator.free(raw, &self.options);
        }
    }
}

/// A device buffer that may be a view into another buffer.
///
/// Handle-identity (`id`) is per-`Buffer` value, including views — mirroring
/// tinygrad, where each `BUFFER_VIEW` produces a distinct UOp identity.
/// Storage-identity (the underlying `Arc<BufferData>`) is shared between a
/// buffer and its views; use [`Buffer::storage_id`] to compare it.
#[derive(Debug, Clone)]
pub struct Buffer {
    /// Per-handle unique identifier. Views get fresh ids; storage is shared
    /// via `data` independently.
    id: BufferId,
    /// Shared data for the base allocation.
    data: Arc<BufferData>,
    /// Offset into the base buffer (in bytes).
    offset: usize,
    /// Size of this view (in bytes).
    size: usize,
    /// Data type of the buffer elements.
    dtype: DType,
    /// Shape of the tensor (stack-allocated for 0-4D tensors).
    shape: SmallVec<[usize; 4]>,
}

impl Buffer {
    /// Create a new buffer with lazy allocation.
    pub fn new(allocator: Arc<dyn Allocator>, dtype: DType, shape: Vec<usize>, options: BufferOptions) -> Self {
        let size = dtype.bytes() * shape.iter().product::<usize>();
        Self {
            id: BufferId(next_buffer_id()),
            data: Arc::new(BufferData::new(allocator, size, options)),
            offset: 0,
            size,
            dtype,
            shape: SmallVec::from_vec(shape),
        }
    }

    /// Create a new buffer with immediate allocation.
    pub fn allocate(
        allocator: Arc<dyn Allocator>,
        dtype: DType,
        shape: Vec<usize>,
        options: BufferOptions,
    ) -> Result<Self> {
        let buffer = Self::new(allocator, dtype, shape, options);
        buffer.ensure_allocated()?;
        Ok(buffer)
    }

    /// Create a view into this buffer.
    ///
    /// The view shares storage with `self` (same `Arc<BufferData>`) but gets
    /// a **fresh `BufferId`** so the runtime parallel-hazard model treats
    /// disjoint views of one arena as independent — mirroring tinygrad's
    /// `BUFFER_VIEW`-as-distinct-identity semantics. Use
    /// [`Buffer::storage_id`] to compare storage identity instead.
    pub fn view(&self, offset: usize, size: usize) -> Result<Self> {
        // Validate view parameters
        if offset + size > self.size {
            return InvalidViewSnafu { offset, size, buffer_size: self.size }.fail();
        }

        Ok(Self {
            id: BufferId(next_buffer_id()),
            data: Arc::clone(&self.data),
            offset: self.offset + offset,
            size,
            dtype: self.dtype.clone(),
            // For views, shape is not well-defined without reshaping logic
            shape: smallvec![size / self.dtype.bytes()],
        })
    }

    /// Ensure the underlying buffer is allocated.
    pub fn ensure_allocated(&self) -> Result<()> {
        self.data.ensure_allocated()
    }

    /// Check if the buffer is allocated.
    pub fn is_allocated(&self) -> bool {
        self.data.is_allocated()
    }

    /// Get the size of this buffer view in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the offset of this view in bytes.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// Get the shape of this buffer.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get a byte slice of the buffer data (CPU-accessible buffers only).
    ///
    /// Zero-copy. For realized tensors after `realize()`, this is safe because
    /// the scheduler guarantees no concurrent kernel writes.
    ///
    /// # Errors
    /// - `NotAllocated` if buffer hasn't been allocated
    /// - `NotCpuAccessible` for CUDA device buffers (use `copyout` instead)
    pub fn as_host_bytes(&self) -> Result<&[u8]> {
        self.ensure_allocated()?;
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: After realize(), no kernels are executing.
                // The scheduler guarantees exclusive access during kernel execution;
                // user code only accesses buffers between kernel runs.
                let bytes = unsafe { &(&(*data.get()))[self.offset..self.offset + self.size] };
                Ok(bytes)
            }
            RawBuffer::Mmap { data, .. } => Ok(&data[self.offset..self.offset + self.size]),
            #[cfg(feature = "cuda")]
            _ => NotCpuAccessibleSnafu.fail(),
        }
    }

    /// Get a mutable byte slice of the buffer data (CPU-accessible buffers only).
    ///
    /// # Safety contract (same as `as_host_bytes`)
    /// Caller must ensure no kernels are executing concurrently.
    ///
    /// # Errors
    /// - `NotAllocated` if buffer hasn't been allocated
    /// - `NotCpuAccessible` for CUDA device buffers
    #[allow(clippy::mut_from_ref)] // interior mutability via UnsafeCell
    pub fn as_host_bytes_mut(&self) -> Result<&mut [u8]> {
        self.ensure_allocated()?;
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: Same invariant as as_host_bytes — user code only
                // accesses buffers between kernel runs. UnsafeCell provides
                // interior mutability through the shared Arc<BufferData>.
                let bytes = unsafe { &mut (&mut *data.get())[self.offset..self.offset + self.size] };
                Ok(bytes)
            }
            // Mmap is read-only — no mutable access
            RawBuffer::Mmap { .. } => NotCpuAccessibleSnafu.fail(),
            #[cfg(feature = "cuda")]
            _ => NotCpuAccessibleSnafu.fail(),
        }
    }

    /// Typed immutable view over CPU-accessible buffer memory.
    ///
    /// Returns an ndarray view shaped according to the buffer's concrete dimensions.
    /// Only works for CPU-accessible buffers — fails for device-only CUDA memory.
    ///
    /// # Errors
    /// - `TypeMismatch` if `T::DTYPE` doesn't match buffer dtype
    /// - `NotCpuAccessible` for non-CPU-accessible buffers
    /// - `NotAllocated` if buffer hasn't been allocated
    pub fn as_array<T: HasDType>(&self) -> Result<ndarray::ArrayViewD<'_, T>> {
        self.ensure_allocated()?;
        if self.dtype != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: self.dtype.clone() }.fail();
        }
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => {
                let bytes = unsafe { &(&(*data.get()))[self.offset..self.offset + self.size] };
                let count = bytes.len() / T::DTYPE.bytes();
                let typed = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) };
                ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&self.shape), typed).context(NdarrayShapeSnafu)
            }
            RawBuffer::Mmap { data, .. } => {
                let bytes = &data[self.offset..self.offset + self.size];
                let count = bytes.len() / T::DTYPE.bytes();
                let typed = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) };
                ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&self.shape), typed).context(NdarrayShapeSnafu)
            }
            #[cfg(feature = "cuda")]
            _ => NotCpuAccessibleSnafu.fail(),
        }
    }

    /// Typed mutable view over CPU-accessible buffer memory.
    ///
    /// Same as [`Self::as_array`] but allows writes. Caller must ensure no
    /// kernels are executing concurrently (safety is the caller's
    /// responsibility).
    ///
    /// # Errors
    /// Same as [`Self::as_array`].
    #[allow(clippy::mut_from_ref)]
    pub fn as_array_mut<T: HasDType>(&self) -> Result<ndarray::ArrayViewMutD<'_, T>> {
        self.ensure_allocated()?;
        if self.dtype != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: self.dtype.clone() }.fail();
        }
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, cpu_accessible } if *cpu_accessible => {
                let bytes = unsafe { &mut (&mut *data.get())[self.offset..self.offset + self.size] };
                let count = bytes.len() / T::DTYPE.bytes();
                let typed = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, count) };
                ndarray::ArrayViewMutD::from_shape(ndarray::IxDyn(&self.shape), typed).context(NdarrayShapeSnafu)
            }
            _ => NotCpuAccessibleSnafu.fail(),
        }
    }

    /// Zero-copy typed slice view (CPU-accessible only).
    pub fn as_slice<T: HasDType>(&self) -> Result<&[T]> {
        self.ensure_allocated()?;
        if self.dtype != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: self.dtype.clone() }.fail();
        }
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, cpu_accessible } if *cpu_accessible => {
                let bytes = unsafe { &(&(*data.get()))[self.offset..self.offset + self.size] };
                let count = bytes.len() / T::DTYPE.bytes();
                Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) })
            }
            _ => NotCpuAccessibleSnafu.fail(),
        }
    }

    /// Read a single scalar value from the buffer (CPU-accessible only).
    ///
    /// Panics if the buffer contains more than one element.
    pub fn item<T: HasDType + Copy>(&self) -> Result<T> {
        let slice = self.as_slice::<T>()?;
        assert_eq!(slice.len(), 1, "item() requires exactly 1 element, got {}", slice.len());
        Ok(slice[0])
    }

    /// Get the allocator used by this buffer.
    pub fn allocator(&self) -> &dyn Allocator {
        &*self.data.allocator
    }

    /// Get an `Arc`-cloned handle to the allocator, suitable for constructing
    /// new buffers on the same device (used by the arena memory planner to
    /// allocate per-lane arenas matching prototype buffers' device).
    pub fn allocator_arc(&self) -> Arc<dyn Allocator> {
        Arc::clone(&self.data.allocator)
    }

    /// Get the unique identifier for this buffer **handle**.
    ///
    /// Each `Buffer` value (including each view) carries its own `BufferId`;
    /// disjoint views of one arena therefore have different ids. Used by the
    /// runtime parallel-hazard model. To compare storage identity (i.e. "do
    /// these two buffers share the same underlying allocation"), use
    /// [`Buffer::storage_id`] instead.
    pub fn id(&self) -> BufferId {
        self.id
    }

    /// Size of the underlying allocation in bytes (shared by every view of
    /// this buffer's storage). Distinct from [`Buffer::size`], which returns
    /// the view's size — for a non-view buffer the two are equal; for a view
    /// into an arena, `total_size` reports the arena's allocation size while
    /// `size` reports just the view's window.
    pub fn total_size(&self) -> usize {
        self.data.total_size
    }

    /// Stable identifier for this buffer's underlying allocation.
    ///
    /// Equal across a base buffer and all of its views, distinct between
    /// independent allocations. Unlike a heap-pointer probe, this id is
    /// minted once at allocation time and never reused — safe to use as a
    /// hash key or alias-detection key without worrying about
    /// allocator-reuse aliasing.
    pub fn storage_id(&self) -> BufferId {
        self.data.storage_id
    }

    /// Copy data from host memory into this buffer.
    pub fn copyin(&mut self, src: &[u8]) -> Result<()> {
        self.ensure_allocated()?;

        let expected = self.size;
        let actual = src.len();
        snafu::ensure!(expected == actual, SizeMismatchSnafu { expected, actual });

        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: Scheduler guarantees exclusive access during buffer operations
                let slice = unsafe {
                    let data_mut = &mut *data.get();
                    &mut data_mut[self.offset..self.offset + self.size]
                };
                slice.copy_from_slice(src);
                Ok(())
            }
            RawBuffer::Mmap { .. } => panic!("DISK device is read-only: copyin not supported"),
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { data, device } => {
                // SAFETY: Scheduler guarantees exclusive access
                let cuda_data = unsafe { &mut *data.get() };
                let mut view = cuda_data.slice_mut(self.offset..self.offset + self.size);
                device.default_stream().memcpy_htod(src, &mut view).context(CudaSnafu)
            }
            #[cfg(feature = "cuda")]
            RawBuffer::CudaUnified { data, .. } => {
                // SAFETY: Scheduler guarantees exclusive access
                let unified_data = unsafe { &mut *data.get() };
                let slice = unified_data.as_mut_slice().context(CudaSnafu)?;
                let target = &mut slice[self.offset..self.offset + self.size];
                target.copy_from_slice(src);
                Ok(())
            }
        }
    }

    /// Copy data from this buffer to host memory.
    pub fn copyout(&self, dst: &mut [u8]) -> Result<()> {
        self.ensure_allocated()?;

        let expected = self.size;
        let actual = dst.len();
        snafu::ensure!(expected == actual, SizeMismatchSnafu { expected, actual });

        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: Scheduler guarantees no concurrent writes during buffer operations
                let data_ref = unsafe { &*data.get() };
                dst.copy_from_slice(&data_ref[self.offset..self.offset + self.size]);
                Ok(())
            }
            RawBuffer::Mmap { data, .. } => {
                dst.copy_from_slice(&data[self.offset..self.offset + self.size]);
                Ok(())
            }
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { data, device } => {
                device.synchronize().context(CudaSnafu)?;
                // SAFETY: Scheduler guarantees no concurrent writes
                let cuda_data = unsafe { &*data.get() };
                let view = cuda_data.slice(self.offset..self.offset + self.size);
                device.default_stream().memcpy_dtoh(&view, dst).context(CudaSnafu)
            }
            #[cfg(feature = "cuda")]
            RawBuffer::CudaUnified { data, .. } => {
                // SAFETY: Scheduler guarantees no concurrent writes
                let unified_data = unsafe { &*data.get() };
                let slice = unified_data.as_slice().context(CudaSnafu)?;
                let source = &slice[self.offset..self.offset + self.size];
                dst.copy_from_slice(source);
                Ok(())
            }
        }
    }

    /// Copy data from another buffer to this buffer.
    pub fn copy_from(&mut self, src: &Buffer) -> Result<()> {
        self.ensure_allocated()?;
        src.ensure_allocated()?;

        let expected = self.size;
        let actual = src.size;
        snafu::ensure!(expected == actual, SizeMismatchSnafu { expected, actual });

        let dst_raw = self.data.raw();
        let src_raw = src.data.raw();

        // SAFETY: Scheduler guarantees exclusive access to dst and read access to src.
        // src and dst are different buffers (enforced by borrow checker at call site).
        match (dst_raw, src_raw) {
            // CPU -> CPU
            (RawBuffer::Cpu { data: dst_data, .. }, RawBuffer::Cpu { data: src_data, .. }) => {
                let dst_mut = unsafe { &mut *dst_data.get() };
                let src_ref = unsafe { &*src_data.get() };
                let dst_slice = &mut dst_mut[self.offset..self.offset + self.size];
                let src_slice = &src_ref[src.offset..src.offset + src.size];
                dst_slice.copy_from_slice(src_slice);
                Ok(())
            }
            // Mmap -> CPU
            (RawBuffer::Cpu { data: dst_data, .. }, RawBuffer::Mmap { data: src_data, .. }) => {
                let dst_mut = unsafe { &mut *dst_data.get() };
                let dst_slice = &mut dst_mut[self.offset..self.offset + self.size];
                let src_slice = &src_data[src.offset..src.offset + src.size];
                dst_slice.copy_from_slice(src_slice);
                Ok(())
            }
            // Mmap as destination is not supported (read-only)
            (RawBuffer::Mmap { .. }, _) => panic!("DISK device is read-only: copy_from not supported"),
            // CudaDevice -> CudaDevice
            #[cfg(feature = "cuda")]
            (
                RawBuffer::CudaDevice { data: dst_data, device: dst_device },
                RawBuffer::CudaDevice { data: src_data, .. },
            ) => {
                let dst_cuda = unsafe { &mut *dst_data.get() };
                let src_cuda = unsafe { &*src_data.get() };
                let mut dst_view = dst_cuda.slice_mut(self.offset..self.offset + self.size);
                let src_view = src_cuda.slice(src.offset..src.offset + src.size);
                dst_device.default_stream().memcpy_dtod(&src_view, &mut dst_view).context(CudaSnafu)
            }
            // CPU -> CudaDevice
            #[cfg(feature = "cuda")]
            (RawBuffer::CudaDevice { data: dst_data, device }, RawBuffer::Cpu { data: src_data, .. }) => {
                let dst_cuda = unsafe { &mut *dst_data.get() };
                let src_ref = unsafe { &*src_data.get() };
                let mut dst_view = dst_cuda.slice_mut(self.offset..self.offset + self.size);
                let src_slice = &src_ref[src.offset..src.offset + src.size];
                device.default_stream().memcpy_htod(src_slice, &mut dst_view).context(CudaSnafu)
            }
            // CudaDevice -> CPU
            #[cfg(feature = "cuda")]
            (RawBuffer::Cpu { data: dst_data, .. }, RawBuffer::CudaDevice { data: src_data, device }) => {
                let dst_mut = unsafe { &mut *dst_data.get() };
                let src_cuda = unsafe { &*src_data.get() };
                let dst_slice = &mut dst_mut[self.offset..self.offset + self.size];
                let src_view = src_cuda.slice(src.offset..src.offset + src.size);
                device.default_stream().memcpy_dtoh(&src_view, dst_slice).context(CudaSnafu)
            }
            // CudaUnified -> CudaUnified (direct CPU access)
            #[cfg(feature = "cuda")]
            (RawBuffer::CudaUnified { data: dst_data, .. }, RawBuffer::CudaUnified { data: src_data, .. }) => {
                let dst_unified = unsafe { &mut *dst_data.get() };
                let src_unified = unsafe { &*src_data.get() };
                let dst_slice = dst_unified.as_mut_slice().context(CudaSnafu)?;
                let src_slice = src_unified.as_slice().context(CudaSnafu)?;
                let dst_target = &mut dst_slice[self.offset..self.offset + self.size];
                let src_source = &src_slice[src.offset..src.offset + src.size];
                dst_target.copy_from_slice(src_source);
                Ok(())
            }
            // CPU -> CudaUnified (direct CPU access)
            #[cfg(feature = "cuda")]
            (RawBuffer::CudaUnified { data: dst_data, .. }, RawBuffer::Cpu { data: src_data, .. }) => {
                let dst_unified = unsafe { &mut *dst_data.get() };
                let src_ref = unsafe { &*src_data.get() };
                let dst_slice = dst_unified.as_mut_slice().context(CudaSnafu)?;
                let dst_target = &mut dst_slice[self.offset..self.offset + self.size];
                let src_source = &src_ref[src.offset..src.offset + src.size];
                dst_target.copy_from_slice(src_source);
                Ok(())
            }
            // CudaUnified -> CPU (direct CPU access)
            #[cfg(feature = "cuda")]
            (RawBuffer::Cpu { data: dst_data, .. }, RawBuffer::CudaUnified { data: src_data, .. }) => {
                let dst_mut = unsafe { &mut *dst_data.get() };
                let src_unified = unsafe { &*src_data.get() };
                let src_slice = src_unified.as_slice().context(CudaSnafu)?;
                let dst_target = &mut dst_mut[self.offset..self.offset + self.size];
                let src_source = &src_slice[src.offset..src.offset + src.size];
                dst_target.copy_from_slice(src_source);
                Ok(())
            }
            // CudaDevice -> CudaUnified (device-to-host memcpy)
            #[cfg(feature = "cuda")]
            (
                RawBuffer::CudaUnified { data: dst_data, device: dst_device },
                RawBuffer::CudaDevice { data: src_data, .. },
            ) => {
                let src_cuda = unsafe { &*src_data.get() };
                let src_view = src_cuda.slice(src.offset..src.offset + src.size);
                // Get CPU-accessible slice from unified memory
                let dst_unified = unsafe { &mut *dst_data.get() };
                let mut dst_target = dst_unified.slice_mut(self.offset..self.offset + self.size);
                // Copy directly from device to unified memory (via host access)
                dst_device.default_stream().memcpy_dtod(&src_view, &mut dst_target).context(CudaSnafu)
            }
            // CudaUnified -> CudaDevice (host-to-device memcpy)
            #[cfg(feature = "cuda")]
            (RawBuffer::CudaDevice { data: dst_data, device }, RawBuffer::CudaUnified { data: src_data, .. }) => {
                let dst_cuda = unsafe { &mut *dst_data.get() };
                let mut dst_view = dst_cuda.slice_mut(self.offset..self.offset + self.size);
                // Get CPU-accessible slice from unified memory
                let src_unified = unsafe { &*src_data.get() };
                let src_source = src_unified.slice(src.offset..src.offset + src.size);
                // Copy directly from unified memory to device (via host access)
                device.default_stream().memcpy_htod(&src_source, &mut dst_view).context(CudaSnafu)
            }
        }
    }

    /// Synchronize the device (wait for all operations to complete).
    pub fn synchronize(&self) -> Result<()> {
        self.data.allocator.synchronize()
    }

    /// Get a raw pointer to the buffer data for kernel execution.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid while the buffer is allocated.
    /// The caller must ensure:
    /// - Buffer remains allocated during pointer lifetime
    /// - No conflicting accesses occur during kernel execution
    /// - Pointer is not used after buffer is freed
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not yet allocated.
    pub unsafe fn as_raw_ptr(&self) -> *mut u8 {
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: Caller is responsible for ensuring no conflicting access.
                // This is already an unsafe function - caller guarantees exclusive access.
                unsafe { (&mut *data.get()).as_mut_ptr().add(self.offset) }
            }
            RawBuffer::Mmap { data, .. } => {
                // Read-only mmap: writing through this pointer is UB.
                unsafe { data.as_ptr().add(self.offset) as *mut u8 }
            }
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { .. } | RawBuffer::CudaUnified { .. } => {
                // TODO: CUDA device memory support for kernels
                // This requires proper stream management and synchronization
                // For MVP, we only support CPU execution
                unimplemented!("CUDA buffer raw pointers not yet supported for kernel execution")
            }
        }
    }

    /// Get the raw data pointer for testing buffer identity.
    ///
    /// This is used in tests to verify cache reuse by comparing pointer addresses.
    /// Returns the pointer to the underlying buffer data.
    #[cfg(test)]
    pub(crate) fn raw_data_ptr(&self) -> usize {
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: Only reading the pointer address for test comparison
                unsafe { (*data.get()).as_ptr() as usize }
            }
            RawBuffer::Mmap { data, .. } => data.as_ptr() as usize,
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { data, .. } => {
                // For CUDA device memory, we use the CudaSlice's internal pointer
                // SAFETY: Only reading pointer address for test comparison
                unsafe { &*data.get() as *const _ as usize }
            }
            #[cfg(feature = "cuda")]
            RawBuffer::CudaUnified { data, .. } => {
                // For CUDA unified memory, we use the UnifiedSlice's internal pointer
                // SAFETY: Only reading pointer address for test comparison
                unsafe { &*data.get() as *const _ as usize }
            }
        }
    }
}
