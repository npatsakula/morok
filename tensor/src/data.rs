use bon::bon;
use snafu::ResultExt;
use std::sync::Arc;

use svod_device::{Buffer, registry};
use svod_dtype::DType;
use svod_dtype::ext::HasDType;
use svod_ir::{DeviceSpec, SInt, UOp, shape::Shape};

use crate::Tensor;
use crate::error::*;
use crate::tensor_registry;
use svod_dtype::default_device::default_device;

#[bon]
impl Tensor {
    /// Create tensor from slice on the active default device (CPU unless
    /// overridden via `set_default_device` or the `SVOD_DEVICE` env var).
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// ```
    pub fn from_slice<T: HasDType, C: AsRef<[T]>>(source: C) -> Self {
        let source = source.as_ref();
        Self::from_bytes_shaped(
            unsafe { std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * T::DTYPE.bytes()) },
            &[source.len()],
            T::DTYPE,
            default_device(),
        )
    }

    /// Create tensor from slice with explicit device specification using builder pattern.
    #[builder]
    pub fn from_slice_with<T: HasDType, C: AsRef<[T]>>(
        source: C,
        #[builder(default = default_device())] device: DeviceSpec,
    ) -> Self {
        let source = source.as_ref();
        Self::from_bytes_shaped(
            unsafe { std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * T::DTYPE.bytes()) },
            &[source.len()],
            T::DTYPE,
            device,
        )
    }
}

impl Tensor {
    /// Core: create a tensor from raw bytes with a known shape.
    ///
    /// Builds the buffer UOp with the target shape directly — no reshape,
    /// so the returned tensor retains its buffer for zero-copy `array_view`.
    /// Routes to whichever allocator the registry returns for `device`; for
    /// AMD this means data is mmapped through the host-visible VRAM aperture
    /// and the GPU sees the buffer directly.
    fn from_bytes_shaped(bytes: &[u8], shape: &[usize], dtype: DType, device: DeviceSpec) -> Self {
        Self::from_bytes_shaped_spec(bytes, shape, dtype, device, Default::default())
    }

    /// [`from_bytes_shaped`](Self::from_bytes_shaped) with an explicit
    /// [`svod_device::BufferSpec`]. `cpu_access: false` keeps the buffer
    /// device-local (no host mapping): the init bytes and any later host
    /// access stage through the backend's copy engine (`copyin`/`copyout`),
    /// and device→device `copy_from` stays on-device. For state buffers the
    /// host shouldn't observe.
    pub fn from_bytes_shaped_spec(
        bytes: &[u8],
        shape: &[usize],
        dtype: DType,
        device: DeviceSpec,
        spec: svod_device::BufferSpec,
    ) -> Self {
        let numel: usize = shape.iter().product();
        let ir_shape = Shape::from_iter(shape.iter().map(|&d| SInt::Const(d)));

        let buffer_uop = UOp::new_buffer(device.clone(), numel, dtype.clone());
        let buffer_uop_id = buffer_uop.id;

        let allocator = registry::registry().get(&device).unwrap_or_else(|e| {
            panic!(
                "Failed to get allocator for {device:?}: {e}\n\
                 Hint: set SVOD_DEVICE=CPU (or unset it) to fall back to the CPU backend."
            )
        });

        let mut buffer = Buffer::new(allocator, dtype.clone(), shape.to_vec(), spec);
        buffer.copyin(bytes).expect("Buffer write always successful");

        let buffer_arc = Arc::new(buffer);
        let uop = buffer_uop.try_reshape(&ir_shape).expect("shape matches element count");

        let entry = tensor_registry::register_tensor_with_buffer(uop, buffer_arc.clone(), buffer_uop_id);
        Self::with_buffer(entry, buffer_arc)
    }

    /// Weight-loading constructor: weights with the same checkpoint
    /// provenance share ONE immutable device storage across model instances
    /// (see [`crate::weight_cache`]). The tensor still gets its own BUFFER
    /// UOp identity; sharing is at the storage level, so the planner and
    /// `replicate` treat it like any other pre-allocated read-only input.
    pub fn from_shared_weight(key: crate::weight_cache::WeightKey, bytes: &[u8]) -> Result<Self> {
        let numel: usize = key.shape.iter().product();
        let expected = numel * key.dtype.bytes();
        if bytes.len() != expected {
            return Err(Error::IrConstruction {
                details: format!("from_shared_weight: data length {} != expected {expected}", bytes.len()),
            });
        }
        let ir_shape = Shape::from_iter(key.shape.iter().map(|&d| SInt::Const(d)));
        let (device, dtype) = (key.device.clone(), key.dtype.clone());
        let buffer_arc = crate::weight_cache::shared_weight_buffer(key, bytes);
        let buffer_uop = UOp::new_buffer(device, numel, dtype);
        let buffer_uop_id = buffer_uop.id;
        let uop = buffer_uop.try_reshape(&ir_shape).expect("shape matches element count");
        let entry = tensor_registry::register_tensor_with_buffer(uop, buffer_arc.clone(), buffer_uop_id);
        Ok(Self::with_buffer(entry, buffer_arc))
    }

    /// Create tensor from raw bytes with explicit dtype and shape.
    ///
    /// The bytes are interpreted as little-endian values of the given dtype.
    /// Length must equal `product(shape) * dtype.bytes()`.
    /// Used for types without a native Rust representation (Float16, BFloat16, FP8).
    pub fn from_raw_bytes(data: &[u8], shape: &[usize], dtype: DType) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let expected_bytes = numel * dtype.bytes();
        if data.len() != expected_bytes {
            return Err(Error::IrConstruction {
                details: format!(
                    "from_raw_bytes: data length {} != expected {} ({} elements * {} bytes)",
                    data.len(),
                    expected_bytes,
                    numel,
                    dtype.bytes()
                ),
            });
        }
        Ok(Self::from_bytes_shaped(data, shape, dtype, default_device()))
    }

    /// Create tensor from an ndarray (owned `Array` or `ArrayView`).
    ///
    /// When the array is already C-contiguous, uses the backing slice directly
    /// (no intermediate allocation). Falls back to `.iter().cloned().collect()`
    /// for Fortran-order or non-contiguous layouts.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let view = t.array_view::<f32>().unwrap();
    /// assert_eq!(view[[1, 2]], 6.0);
    /// ```
    pub fn from_ndarray<T, S, D>(array: &ndarray::ArrayBase<S, D>) -> Self
    where
        T: HasDType + Clone,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        let shape: Vec<usize> = array.shape().to_vec();
        if array.is_empty() {
            let t = Self::empty_zero(T::DTYPE);
            if shape.len() <= 1 {
                return t;
            }
            let isize_shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
            return t.try_reshape(&isize_shape).expect("empty reshape matches");
        }
        // Fast path: C-contiguous — use backing slice directly, no intermediate Vec
        if let Some(slice) = array.as_slice() {
            let bytes =
                unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * T::DTYPE.bytes()) };
            Self::from_bytes_shaped(bytes, &shape, T::DTYPE, default_device())
        } else {
            // Slow path: Fortran-order or non-contiguous — collect in logical order
            let data: Vec<T> = array.iter().cloned().collect();
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * T::DTYPE.bytes()) };
            Self::from_bytes_shaped(bytes, &shape, T::DTYPE, default_device())
        }
    }

    /// Get a reference to the underlying buffer.
    ///
    /// Returns `None` for lazy tensors that haven't been realized yet.
    /// Returns `Some(buffer)` for input tensors and realized tensors.
    pub fn buffer(&self) -> Option<Buffer> {
        // Check local field first, then entry, then global registry by base UOp ID.
        if let Some(buf) = self.buffer.as_ref().or_else(|| self.entry.buffer()) {
            return Some((**buf).clone());
        }
        crate::tensor_registry::get_buffer_arc(self.uop().base().id).map(|arc| (*arc).clone())
    }

    /// Read realized tensor data as an ndarray.
    ///
    /// The tensor must have a buffer (from `from_slice`, `realize()`, etc.).
    /// Returns error if the tensor has not been realized.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let result = t.as_ndarray::<f32>().unwrap();
    /// assert_eq!(result.shape(), &[3]);
    /// ```
    pub fn as_ndarray<T: HasDType + Default + Clone>(&self) -> Result<ndarray::ArrayD<T>> {
        use ndarray::{ArrayD, IxDyn};

        let uop = self.uop();
        let shape = uop.shape().context(UOpSnafu)?.ok_or(Error::NoShape)?;

        // Refuse symbolic shapes — matches Tinygrad: assert all_int(self.shape)
        if shape.iter().any(|dim| dim.as_const().is_none()) {
            return SymbolicShapeSnafu.fail();
        }

        let dims: Vec<usize> = shape.iter().map(|dim| dim.as_const().unwrap()).collect();

        if dims.contains(&0) {
            let arr = ArrayD::from_shape_vec(IxDyn(&dims), vec![]).context(NdarrayShapeSnafu)?;
            return Ok(arr);
        }

        let buffer = self.buffer().ok_or(Error::NoBuffer)?;

        if buffer.dtype() != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: buffer.dtype() }.fail();
        }

        let count = buffer.size() / T::DTYPE.bytes();
        let mut data = vec![T::default(); count];
        buffer
            .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, count * T::DTYPE.bytes()) })
            .context(DeviceSnafu)?;

        let arr = ArrayD::from_shape_vec(IxDyn(&dims), data).context(NdarrayShapeSnafu)?;
        Ok(arr)
    }

    /// Read realized tensor data as a flat `Vec<T>`.
    ///
    /// The tensor must have a buffer (from `from_slice`, `realize()`, etc.).
    /// Returns error if the tensor has not been realized.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let v = t.as_vec::<f32>().unwrap();
    /// assert_eq!(v, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn as_vec<T: HasDType + Default + Clone>(&self) -> Result<Vec<T>> {
        let uop = self.uop();
        if let Ok(Some(shape)) = uop.shape() {
            // Refuse symbolic shapes — matches Tinygrad: assert all_int(self.shape)
            if shape.iter().any(|dim| dim.as_const().is_none()) {
                return SymbolicShapeSnafu.fail();
            }
            if shape.iter().any(|dim| dim.as_const() == Some(0)) {
                return Ok(vec![]);
            }
        }

        let buffer = self.buffer().ok_or(Error::NoBuffer)?;

        if buffer.dtype() != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: buffer.dtype() }.fail();
        }

        let count = buffer.size() / T::DTYPE.bytes();
        let mut data = vec![T::default(); count];
        buffer
            .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, count * T::DTYPE.bytes()) })
            .context(DeviceSnafu)?;

        Ok(data)
    }

    /// Typed immutable view into the buffer, shaped by the tensor's logical shape.
    ///
    /// Uses the tensor's concrete shape for multidimensional indexing.
    /// Falls back to the buffer's flat shape for symbolic tensors.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let t = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    /// let view = t.array_view::<f32>().unwrap();
    /// assert_eq!(view[[0, 1]], 2.0);
    /// ```
    pub fn array_view<T: HasDType>(&self) -> Result<ndarray::ArrayViewD<'_, T>> {
        let buffer_arc = self.buffer.as_ref().or_else(|| self.entry.buffer()).ok_or(Error::NoBuffer)?;
        let flat = buffer_arc.as_array::<T>().context(DeviceSnafu)?;
        // Reshape to tensor's logical shape if concrete
        if let Ok(shape) = self.shape() {
            let dims: Vec<usize> = shape.iter().filter_map(|d| d.as_const()).collect();
            if dims.len() == shape.len() {
                return flat.into_shape_with_order(ndarray::IxDyn(&dims)).context(NdarrayShapeSnafu);
            }
        }
        Ok(flat)
    }

    /// Typed mutable view into the buffer, shaped by the tensor's logical shape.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let t = Tensor::from_ndarray(&array![[0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    /// t.array_view_mut::<f32>().unwrap()[[1, 2]] = 42.0;
    /// assert_eq!(t.array_view::<f32>().unwrap()[[1, 2]], 42.0);
    /// ```
    pub fn array_view_mut<T: HasDType>(&self) -> Result<ndarray::ArrayViewMutD<'_, T>> {
        let buffer_arc = self.buffer.as_ref().or_else(|| self.entry.buffer()).ok_or(Error::NoBuffer)?;
        let flat = buffer_arc.as_array_mut::<T>().context(DeviceSnafu)?;
        if let Ok(shape) = self.shape() {
            let dims: Vec<usize> = shape.iter().filter_map(|d| d.as_const()).collect();
            if dims.len() == shape.len() {
                return flat.into_shape_with_order(ndarray::IxDyn(&dims)).context(NdarrayShapeSnafu);
            }
        }
        Ok(flat)
    }
}
