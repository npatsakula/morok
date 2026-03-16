use bon::bon;
use snafu::ResultExt;
use std::sync::Arc;

use morok_device::{Buffer, registry};
use morok_dtype::DType;
use morok_dtype::ext::HasDType;
use morok_ir::{DeviceSpec, SInt, UOp, shape::Shape};

use crate::Tensor;
use crate::error::*;
use crate::tensor_registry;

#[bon]
impl Tensor {
    /// Create tensor from slice on CPU (default device).
    ///
    /// # Examples
    /// ```
    /// # use morok_tensor::Tensor;
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// ```
    pub fn from_slice<T: HasDType, C: AsRef<[T]>>(source: C) -> Self {
        let source = source.as_ref();
        Self::from_bytes_shaped(
            unsafe { std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * T::DTYPE.bytes()) },
            &[source.len()],
            T::DTYPE,
            DeviceSpec::Cpu,
        )
    }

    /// Create tensor from slice with explicit device specification using builder pattern.
    #[builder]
    pub fn from_slice_with<T: HasDType, C: AsRef<[T]>>(
        source: C,
        #[builder(default = DeviceSpec::Cpu)] device: DeviceSpec,
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
    fn from_bytes_shaped(bytes: &[u8], shape: &[usize], dtype: DType, device: DeviceSpec) -> Self {
        let numel: usize = shape.iter().product();
        let ir_shape = Shape::from_iter(shape.iter().map(|&d| SInt::Const(d)));

        let buffer_uop = UOp::new_buffer(device.clone(), numel, dtype.clone());
        let buffer_uop_id = buffer_uop.id;

        let allocator = match &device {
            DeviceSpec::Cpu => registry::cpu().expect("CPU always should be accessible"),
            _ => registry::cpu().expect("CPU fallback for unsupported device"),
        };

        let mut buffer = Buffer::new(allocator, dtype.clone(), vec![numel], Default::default());
        buffer.copyin(bytes).expect("Buffer write always successful");

        let buffer_arc = Arc::new(buffer);
        let uop = buffer_uop.try_reshape(&ir_shape).expect("shape matches element count");

        let entry = tensor_registry::register_tensor_with_buffer(uop, buffer_arc.clone(), buffer_uop_id);
        Self::with_buffer(entry, buffer_arc)
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
        Ok(Self::from_bytes_shaped(data, shape, dtype, DeviceSpec::Cpu))
    }

    /// Create tensor from an ndarray (owned `Array` or `ArrayView`).
    ///
    /// When the array is already C-contiguous, uses the backing slice directly
    /// (no intermediate allocation). Falls back to `.iter().cloned().collect()`
    /// for Fortran-order or non-contiguous layouts.
    ///
    /// # Examples
    /// ```
    /// # use morok_tensor::Tensor;
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
            let t = Self::empty(T::DTYPE);
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
            Self::from_bytes_shaped(bytes, &shape, T::DTYPE, DeviceSpec::Cpu)
        } else {
            // Slow path: Fortran-order or non-contiguous — collect in logical order
            let data: Vec<T> = array.iter().cloned().collect();
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * T::DTYPE.bytes()) };
            Self::from_bytes_shaped(bytes, &shape, T::DTYPE, DeviceSpec::Cpu)
        }
    }

    /// Get a reference to the underlying buffer.
    ///
    /// Returns `None` for lazy tensors that haven't been realized yet.
    /// Returns `Some(buffer)` for input tensors and realized tensors.
    pub fn buffer(&self) -> Option<Buffer> {
        self.buffer.as_ref().map(|arc_buf| (**arc_buf).clone())
    }

    /// Extract data as `ndarray::ArrayD<T>`.
    ///
    /// Calls `.contiguous().realize()` internally to ensure correct layout,
    /// then copies data out of the buffer.
    ///
    /// # Examples
    /// ```
    /// # use morok_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let result = t.to_ndarray::<f32>().unwrap();
    /// assert_eq!(result.shape(), &[3]);
    /// ```
    pub fn to_ndarray<T: HasDType + Default + Clone>(&self) -> Result<ndarray::ArrayD<T>> {
        use ndarray::{ArrayD, IxDyn};

        let uop = self.uop();
        let shape = uop.shape().context(UOpSnafu)?.ok_or(Error::NoShape)?;
        let dims: Vec<usize> = shape.iter().map(|dim| dim.as_const().unwrap_or(1)).collect();

        // Zero-size tensor: return empty ndarray without realization (matches Tinygrad)
        if dims.contains(&0) {
            let arr = ArrayD::from_shape_vec(IxDyn(&dims), vec![]).context(NdarrayShapeSnafu)?;
            return Ok(arr);
        }

        // Following Tinygrad's approach: `.numpy()` always calls `.contiguous().realize()`.
        // Never use a cached buffer directly — movement ops (permute, shrink, pad, etc.)
        // change the logical-to-physical mapping, so the underlying buffer may not match
        // the tensor's logical shape.
        let realized = self.clone().contiguous().realize()?;
        let buffer = realized.buffer().ok_or(Error::NoBuffer)?;

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

    /// Extract data as a flat `Vec<T>`.
    ///
    /// Calls `.contiguous().realize()` internally to ensure correct layout,
    /// then copies data out of the buffer.
    ///
    /// # Examples
    /// ```
    /// # use morok_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let v = t.to_vec::<f32>().unwrap();
    /// assert_eq!(v, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn to_vec<T: HasDType + Default + Clone>(&self) -> Result<Vec<T>> {
        // Zero-size tensor: return empty vec without realization (matches to_ndarray)
        let uop = self.uop();
        if let Ok(Some(shape)) = uop.shape()
            && shape.iter().any(|dim| dim.as_const() == Some(0)) {
                return Ok(vec![]);
            }

        let realized = self.clone().contiguous().realize()?;
        let buffer = realized.buffer().ok_or(Error::NoBuffer)?;

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

    /// Get a zero-copy typed view into the realized buffer.
    ///
    /// CPU-only. The tensor must have a buffer (from `from_slice`, `from_ndarray`,
    /// or `realize()`). For tensors with movement ops (permute, etc.),
    /// call `.contiguous().realize()` first.
    ///
    /// # Examples
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::array;
    /// let t = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    /// let view = t.array_view::<f32>().unwrap();
    /// assert_eq!(view[[0, 1]], 2.0);
    /// ```
    pub fn array_view<T: HasDType>(&self) -> Result<ndarray::ArrayViewD<'_, T>> {
        let buffer_arc = self.buffer.as_ref().ok_or(Error::NoBuffer)?;

        if buffer_arc.dtype() != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: buffer_arc.dtype() }.fail();
        }

        let bytes = buffer_arc.as_host_bytes().context(DeviceSnafu)?;
        let count = bytes.len() / T::DTYPE.bytes();
        let data = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) };
        let shape = self.shape()?;
        let dims: Vec<usize> = shape.iter().map(|d| d.as_const().unwrap_or(1)).collect();
        ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&dims), data).context(NdarrayShapeSnafu)
    }

    /// Get a mutable zero-copy typed view into the buffer.
    ///
    /// Allows direct writes into the tensor's backing memory.
    /// Same requirements as `array_view` — CPU-only, buffer must exist.
    ///
    /// # Examples
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::array;
    /// let t = Tensor::from_ndarray(&array![[0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    /// t.array_view_mut::<f32>().unwrap()[[1, 2]] = 42.0;
    /// assert_eq!(t.array_view::<f32>().unwrap()[[1, 2]], 42.0);
    /// ```
    pub fn array_view_mut<T: HasDType>(&self) -> Result<ndarray::ArrayViewMutD<'_, T>> {
        let buffer_arc = self.buffer.as_ref().ok_or(Error::NoBuffer)?;

        if buffer_arc.dtype() != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: buffer_arc.dtype() }.fail();
        }

        let bytes = buffer_arc.as_host_bytes_mut().context(DeviceSnafu)?;
        let count = bytes.len() / T::DTYPE.bytes();
        let data = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, count) };
        let shape = self.shape()?;
        let dims: Vec<usize> = shape.iter().map(|d| d.as_const().unwrap_or(1)).collect();
        ndarray::ArrayViewMutD::from_shape(ndarray::IxDyn(&dims), data).context(NdarrayShapeSnafu)
    }
}
