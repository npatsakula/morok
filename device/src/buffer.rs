use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

use morok_dtype::DType;
use smallvec::{SmallVec, smallvec};
use snafu::ResultExt;

use crate::allocator::{Allocator, BufferOptions, RawBuffer};
use crate::error::{InvalidViewSnafu, Result, SizeMismatchSnafu};

#[cfg(feature = "cuda")]
use crate::error::CudaSnafu;

/// Shared buffer data that can be referenced by multiple views.
#[derive(Debug)]
struct BufferData {
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
        Self { raw: OnceLock::new(), allocator, total_size: size, options }
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
/// This type is `!Send + !Sync` to prevent accidental sharing across threads,
/// matching Tinygrad's single-threaded execution model.
#[derive(Debug, Clone)]
pub struct Buffer {
    /// Shared data for the base allocation.
    data: Rc<BufferData>,
    /// Offset into the base buffer (in bytes).
    offset: usize,
    /// Size of this view (in bytes).
    size: usize,
    /// Data type of the buffer elements.
    dtype: DType,
    /// Shape of the tensor (stack-allocated for 0-4D tensors).
    #[allow(dead_code)]
    shape: SmallVec<[usize; 4]>,
    /// Marker to make Buffer `!Send + !Sync` (single-threaded only).
    _not_send_sync: PhantomData<Rc<()>>,
}

impl Buffer {
    /// Create a new buffer with lazy allocation.
    pub fn new(allocator: Arc<dyn Allocator>, dtype: DType, shape: Vec<usize>, options: BufferOptions) -> Self {
        let size = dtype.bytes() * shape.iter().product::<usize>();
        Self {
            data: Rc::new(BufferData::new(allocator, size, options)),
            offset: 0,
            size,
            dtype,
            shape: SmallVec::from_vec(shape),
            _not_send_sync: PhantomData,
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
    pub fn view(&self, offset: usize, size: usize) -> Result<Self> {
        // Validate view parameters
        if offset + size > self.size {
            return InvalidViewSnafu { offset, size, buffer_size: self.size }.fail();
        }

        Ok(Self {
            data: Rc::clone(&self.data),
            offset: self.offset + offset,
            size,
            dtype: self.dtype.clone(),
            // For views, shape is not well-defined without reshaping logic
            shape: smallvec![size / self.dtype.bytes()],
            _not_send_sync: PhantomData,
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

    /// Get the allocator used by this buffer.
    pub fn allocator(&self) -> &dyn Allocator {
        &*self.data.allocator
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
                let mut data_mut = data.borrow_mut();
                let slice = &mut data_mut[self.offset..self.offset + self.size];
                slice.copy_from_slice(src);
                Ok(())
            }
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { data, device } => {
                let mut cuda_data = data.borrow_mut();
                let mut view = cuda_data.slice_mut(self.offset..self.offset + self.size);
                device.default_stream().memcpy_htod(src, &mut view).context(CudaSnafu)
            }
            #[cfg(feature = "cuda")]
            RawBuffer::CudaUnified { data, .. } => {
                let mut unified_data = data.borrow_mut();
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
                let data_ref = data.borrow();
                dst.copy_from_slice(&data_ref[self.offset..self.offset + self.size]);
                Ok(())
            }
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { data, device } => {
                device.synchronize().context(CudaSnafu)?;
                let cuda_data = data.borrow();
                let view = cuda_data.slice(self.offset..self.offset + self.size);
                device.default_stream().memcpy_dtoh(&view, dst).context(CudaSnafu)
            }
            #[cfg(feature = "cuda")]
            RawBuffer::CudaUnified { data, .. } => {
                let unified_data = data.borrow();
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

        match (dst_raw, src_raw) {
            // CPU -> CPU
            (RawBuffer::Cpu { data: dst_data, .. }, RawBuffer::Cpu { data: src_data, .. }) => {
                let mut dst_mut = dst_data.borrow_mut();
                let src_ref = src_data.borrow();
                let dst_slice = &mut dst_mut[self.offset..self.offset + self.size];
                let src_slice = &src_ref[src.offset..src.offset + src.size];
                dst_slice.copy_from_slice(src_slice);
                Ok(())
            }
            // CudaDevice -> CudaDevice
            #[cfg(feature = "cuda")]
            (
                RawBuffer::CudaDevice { data: dst_data, device: dst_device },
                RawBuffer::CudaDevice { data: src_data, .. },
            ) => {
                let mut dst_cuda = dst_data.borrow_mut();
                let src_cuda = src_data.borrow();
                let mut dst_view = dst_cuda.slice_mut(self.offset..self.offset + self.size);
                let src_view = src_cuda.slice(src.offset..src.offset + src.size);
                dst_device.default_stream().memcpy_dtod(&src_view, &mut dst_view).context(CudaSnafu)
            }
            // CPU -> CudaDevice
            #[cfg(feature = "cuda")]
            (RawBuffer::CudaDevice { data: dst_data, device }, RawBuffer::Cpu { data: src_data, .. }) => {
                let mut dst_cuda = dst_data.borrow_mut();
                let src_ref = src_data.borrow();
                let mut dst_view = dst_cuda.slice_mut(self.offset..self.offset + self.size);
                let src_slice = &src_ref[src.offset..src.offset + src.size];
                device.default_stream().memcpy_htod(src_slice, &mut dst_view).context(CudaSnafu)
            }
            // CudaDevice -> CPU
            #[cfg(feature = "cuda")]
            (RawBuffer::Cpu { data: dst_data, .. }, RawBuffer::CudaDevice { data: src_data, device }) => {
                let mut dst_mut = dst_data.borrow_mut();
                let src_cuda = src_data.borrow();
                let dst_slice = &mut dst_mut[self.offset..self.offset + self.size];
                let src_view = src_cuda.slice(src.offset..src.offset + src.size);
                device.default_stream().memcpy_dtoh(&src_view, dst_slice).context(CudaSnafu)
            }
            // CudaUnified -> CudaUnified (direct CPU access)
            #[cfg(feature = "cuda")]
            (RawBuffer::CudaUnified { data: dst_data, .. }, RawBuffer::CudaUnified { data: src_data, .. }) => {
                let mut dst_unified = dst_data.borrow_mut();
                let src_unified = src_data.borrow();
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
                let mut dst_unified = dst_data.borrow_mut();
                let src_ref = src_data.borrow();
                let dst_slice = dst_unified.as_mut_slice().context(CudaSnafu)?;
                let dst_target = &mut dst_slice[self.offset..self.offset + self.size];
                let src_source = &src_ref[src.offset..src.offset + src.size];
                dst_target.copy_from_slice(src_source);
                Ok(())
            }
            // CudaUnified -> CPU (direct CPU access)
            #[cfg(feature = "cuda")]
            (RawBuffer::Cpu { data: dst_data, .. }, RawBuffer::CudaUnified { data: src_data, .. }) => {
                let mut dst_mut = dst_data.borrow_mut();
                let src_unified = src_data.borrow();
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
                // let mut dst_unified = dst_data.borrow_mut();
                let src_cuda = src_data.borrow();
                let src_view = src_cuda.slice(src.offset..src.offset + src.size);
                // Get CPU-accessible slice from unified memory
                let mut dst_unified = dst_data.borrow_mut();
                let mut dst_target = dst_unified.slice_mut(self.offset..self.offset + self.size);
                // Copy directly from device to unified memory (via host access)
                dst_device.default_stream().memcpy_dtod(&src_view, &mut dst_target).context(CudaSnafu)
            }
            // CudaUnified -> CudaDevice (host-to-device memcpy)
            #[cfg(feature = "cuda")]
            (RawBuffer::CudaDevice { data: dst_data, device }, RawBuffer::CudaUnified { data: src_data, .. }) => {
                let mut dst_cuda = dst_data.borrow_mut();
                let mut dst_view = dst_cuda.slice_mut(self.offset..self.offset + self.size);
                // Get CPU-accessible slice from unified memory
                let src_unified = src_data.borrow();
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

    /// Get the raw data pointer for testing buffer identity.
    ///
    /// This is used in tests to verify cache reuse by comparing pointer addresses.
    /// Returns the pointer to the underlying buffer data.
    #[cfg(test)]
    pub(crate) fn raw_data_ptr(&self) -> usize {
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => data.borrow().as_ptr() as usize,
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { data, .. } => {
                // For CUDA device memory, we use the CudaSlice's internal pointer
                // CudaSlice wraps a device pointer, we cast the RefCell ref to get a stable address
                &*data.borrow() as *const _ as usize
            }
            #[cfg(feature = "cuda")]
            RawBuffer::CudaUnified { data, .. } => {
                // For CUDA unified memory, we use the UnifiedSlice's internal pointer
                // UnifiedSlice wraps a managed pointer, we cast the RefCell ref to get a stable address
                &*data.borrow() as *const _ as usize
            }
        }
    }
}
