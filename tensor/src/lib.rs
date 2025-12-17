use bon::bon;
use snafu::ResultExt;
use std::sync::Arc;
use tracing::trace;

use morok_device::{Buffer, registry};
use morok_dtype::ext::HasDType;
use morok_ir::{DeviceSpec, SInt, UOp, shape::Shape};

pub mod error;
use error::*;

pub mod activation;
pub mod arithmetic;
pub mod bitwise;
pub mod broadcast;
pub mod buffer_registry;
pub mod conditional;
pub mod math;
pub mod matmul;
pub mod memory_planner;
pub mod realize;
pub mod reduce;
pub mod schedule;
pub mod shape_ops;
pub mod tensor_registry;
pub mod traits;

// Re-export for public API
pub use tensor_registry::apply_map_to_tensors;

/// Information about a rendered kernel.
///
/// This is the public API returned by `tensor.kernels()`.
#[derive(Clone, Debug)]
pub struct KernelInfo {
    /// Kernel name (e.g., "kernel")
    pub name: String,
    /// Generated code (LLVM IR, CUDA PTX, etc.)
    pub code: String,
    /// Entry point function name
    pub entry_point: String,
    /// Backend that generated this kernel
    pub backend: String,
}

/// Tensor represents a multi-dimensional array with lazy evaluation.
///
/// Operations like addition and multiplication build a computation graph
/// without allocating buffers. Buffers are only allocated when:
/// - Creating input tensors via `from_slice()`
/// - Evaluating the computation graph via `realize()`
///
/// # Global Graph Substitution
///
/// Tensors are registered in a global registry to support atomic graph substitution.
/// When rangeify transforms a UOp (e.g., NEG → BUFFERIZE(NEG)), all tensors
/// referencing it are updated atomically via `apply_map_to_tensors()`.
///
/// This is critical for diamond patterns (like argmin's NEG feeding both MAX and EQ)
/// where different consumers must see the same transformed version.
///
/// # Buffer Ownership (RAII)
///
/// Tensors own their buffers via `Arc<Buffer>`. When all Tensor clones referencing
/// a buffer are dropped, the buffer is automatically freed. This provides RAII
/// cleanup without manual buffer management.
///
/// # Examples
///
/// ```
/// # use morok_tensor::Tensor;
/// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
/// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
/// let c = &a + &b;  // Lazy - only builds UOp graph
/// let realized = c.realize().unwrap();  // Executes the computation
/// ```
pub struct Tensor {
    /// Registry entry holding the computation graph (supports global substitution)
    entry: Arc<tensor_registry::TensorEntry>,
    /// Owned buffer for RAII cleanup. None for lazy tensors.
    buffer: Option<Arc<Buffer>>,
}

// Manual Clone impl to share Arc<Buffer> across clones
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self { entry: Arc::clone(&self.entry), buffer: self.buffer.clone() }
    }
}

#[bon]
impl Tensor {
    /// Create tensor without buffer (for lazy computation graphs).
    fn new(uop: Arc<UOp>) -> Self {
        let entry = tensor_registry::register_tensor(uop);
        Self { entry, buffer: None }
    }

    /// Create tensor with existing buffer (for input tensors and realize results).
    pub(crate) fn with_buffer(entry: Arc<tensor_registry::TensorEntry>, buffer: Arc<Buffer>) -> Self {
        Self { entry, buffer: Some(buffer) }
    }

    /// Get the current UOp for this tensor.
    ///
    /// This reads from the registry, so it reflects any global substitutions.
    pub fn uop(&self) -> Arc<UOp> {
        self.entry.uop.read().clone()
    }

    /// Get kernels for THIS tensor.
    ///
    /// Note: Kernel tracking is not yet implemented with the new registry.
    /// This returns an empty list for now.
    pub fn kernels(&self) -> Vec<KernelInfo> {
        // TODO: Implement kernel tracking with the new registry
        Vec::new()
    }

    /// Create 1D tensor with evenly spaced values.
    ///
    /// Generates values in the range `[start, stop)` with given step size.
    /// If `stop` is None, treats `start` as stop and starts from 0.
    ///
    /// # Arguments
    /// * `start` - Starting value (or stop value if `stop` is None)
    /// * `stop` - Ending value (exclusive), defaults to None
    /// * `step` - Step size, defaults to 1
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::arange(5, None, None)?;         // [0, 1, 2, 3, 4]
    /// let t = Tensor::arange(2, Some(10), Some(2))?;  // [2, 4, 6, 8]
    /// let t = Tensor::arange(10, Some(0), Some(-2))?; // [10, 8, 6, 4, 2]
    /// ```
    pub fn arange(start: i64, stop: Option<i64>, step: Option<i64>) -> Result<Self> {
        // Handle start/stop convention: arange(5) = arange(0, 5, 1)
        let (actual_start, actual_stop) = match stop {
            Some(s) => (start, s),
            None => (0, start),
        };

        let actual_step = step.unwrap_or(1);

        // Calculate number of elements
        if actual_step == 0 {
            return Err(Error::SymbolicShapeUnsupported { operation: "arange with step=0".to_string() });
        }

        let count = if actual_step > 0 {
            if actual_stop <= actual_start {
                0
            } else {
                ((actual_stop - actual_start + actual_step - 1) / actual_step) as usize
            }
        } else if actual_stop >= actual_start {
            0
        } else {
            ((actual_start - actual_stop - actual_step - 1) / (-actual_step)) as usize
        };

        // Generate values
        let values: Vec<i32> = (0..count).map(|i| (actual_start + i as i64 * actual_step) as i32).collect();

        // Create tensor from values
        Ok(Self::from_slice(&values))
    }

    /// Create tensor from slice on CPU (default device).
    ///
    /// For explicit device specification, use `from_slice_with`.
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// assert_eq!(a.device(), DeviceSpec::Cpu);
    /// ```
    pub fn from_slice<T: HasDType, C: AsRef<[T]>>(source: C) -> Self {
        Self::from_slice_on(source, DeviceSpec::Cpu)
    }

    /// Create tensor from slice with explicit device specification using builder pattern.
    ///
    /// # Examples
    /// ```ignore
    /// // CPU tensor with builder
    /// let a = Tensor::from_slice_with(&[1.0f32, 2.0, 3.0]).call();
    ///
    /// // Explicit device
    /// let b = Tensor::from_slice_with(&[1.0f32, 2.0, 3.0])
    ///     .device(DeviceSpec::Cuda { device_id: 0 })
    ///     .call();
    /// ```
    #[builder]
    pub fn from_slice_with<T: HasDType, C: AsRef<[T]>>(
        source: C,
        #[builder(default = DeviceSpec::Cpu)] device: DeviceSpec,
    ) -> Self {
        Self::from_slice_on(source, device)
    }

    /// Internal: Create tensor from slice on specified device.
    fn from_slice_on<T: HasDType, C: AsRef<[T]>>(source: C, device: DeviceSpec) -> Self {
        let source = source.as_ref();
        let shape = Shape::from_iter([SInt::Const(source.len())]);
        let dtype = T::DTYPE;

        let buffer_uop = UOp::new_buffer(device.clone(), source.len(), dtype.clone());

        // Get allocator for specified device
        let allocator = match &device {
            DeviceSpec::Cpu => registry::cpu().expect("CPU always should be accessible"),
            // For non-CPU devices, try to get from registry or fall back to CPU for now
            _ => registry::cpu().expect("CPU fallback for unsupported device"),
        };

        let mut buffer = Buffer::new(allocator, dtype.clone(), vec![source.len()], Default::default());
        let bytes = unsafe { std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * dtype.bytes()) };
        buffer.copyin(bytes).expect("Buffer write always successful");

        // RAII: Wrap buffer in Arc for ownership
        let buffer_arc = Arc::new(buffer);

        // SECONDARY: Register for schedule lookup (backwards compat during migration)
        buffer_registry::get_or_create_buffer(buffer_uop.id, || Ok((*buffer_arc).clone()))
            .expect("Buffer registration failed");

        let uop = buffer_uop.try_reshape(&shape).expect("this reshape is always successful");

        // PRIMARY: Create tensor with buffer (RAII ownership)
        let entry = tensor_registry::register_tensor(uop);
        Self::with_buffer(entry, buffer_arc)
    }

    /// Get a reference to the underlying buffer.
    ///
    /// First checks tensor-owned buffer (RAII), then falls back to registry lookup
    /// for backwards compatibility during migration.
    ///
    /// Uses `.base()` to walk through movement operations to find the actual buffer.
    pub fn buffer(&self) -> Option<Buffer> {
        // First: check tensor-owned buffer (no lock!)
        if let Some(arc_buf) = &self.buffer {
            return Some((**arc_buf).clone());
        }
        // Fallback: registry lookup (backwards compat during migration)
        let uop = self.uop();
        let base_id = uop.base().id;
        trace!(uop.id = uop.id, base.id = base_id, "buffer lookup (registry fallback)");
        buffer_registry::get_buffer(base_id)
    }

    /// Get device specification from underlying UOp graph.
    ///
    /// Returns the device where this tensor's data resides.
    /// For lazy tensors (not yet realized), returns the target device.
    /// Defaults to CPU if no device is found in the graph.
    ///
    /// # Examples
    /// ```ignore
    /// let cpu_tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// assert_eq!(cpu_tensor.device(), DeviceSpec::Cpu);
    /// ```
    pub fn device(&self) -> DeviceSpec {
        self.uop().device_spec().unwrap_or(DeviceSpec::Cpu)
    }

    /// Move tensor to a different device.
    ///
    /// Creates a lazy COPY operation. Data is not transferred until `realize()`.
    /// If already on target device, returns a clone (no-op).
    ///
    /// # Examples
    /// ```ignore
    /// let cpu_tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let gpu_tensor = cpu_tensor.to(DeviceSpec::Cuda { device_id: 0 });
    /// let realized = gpu_tensor.realize()?;  // Actually transfers data
    /// ```
    pub fn to(&self, device: DeviceSpec) -> Self {
        if self.device() == device {
            return self.clone();
        }

        let copy_uop = self.uop().copy_to_device(device);
        Self::new(copy_uop)
    }

    /// Cast tensor to a different dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let t_int = t.cast(DType::Int32)?;
    /// ```
    pub fn cast(&self, dtype: morok_dtype::DType) -> Result<Self> {
        let casted = UOp::cast(self.uop(), dtype);
        Ok(Self::new(casted))
    }

    /// Extract data as ndarray::ArrayD<T> (for testing).
    ///
    /// This method is primarily intended for testing and validation.
    /// It extracts the computed tensor data into an ndarray with the proper shape.
    ///
    /// # Type Parameters
    /// * `T` - The output type, must implement HasDType and match the tensor's dtype
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let result = t.realize()?.to_ndarray::<f32>()?;
    /// assert_eq!(result.shape(), &[3]);
    /// ```
    ///
    /// # Errors
    /// Returns error if:
    /// - Tensor has no buffer (unrealized)
    /// - Type T doesn't match tensor's dtype
    /// - Shape cannot be extracted
    /// - Buffer read fails
    pub fn to_ndarray<T: HasDType + Default + Clone>(&self) -> Result<ndarray::ArrayD<T>> {
        use ndarray::{ArrayD, IxDyn};

        // Get buffer
        let buffer = self.buffer().ok_or(Error::NoBuffer)?;

        // Validate dtype matches
        if buffer.dtype() != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: buffer.dtype() }.fail();
        }

        // Get shape
        let uop = self.uop();
        let shape = uop.shape().context(UOpSnafu)?.ok_or(Error::NoShape)?;

        // Convert shape to usize dimensions
        let dims: Vec<usize> = shape.iter().map(|dim| dim.as_const().unwrap_or(1)).collect();

        // Extract data
        let count = buffer.size() / T::DTYPE.bytes();
        let mut data = vec![T::default(); count];
        buffer
            .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, count * T::DTYPE.bytes()) })
            .context(DeviceSnafu)?;

        // Create ndarray with proper shape
        let arr = ArrayD::from_shape_vec(IxDyn(&dims), data).context(NdarrayShapeSnafu)?;

        Ok(arr)
    }

    /// Update the UOp for this tensor directly.
    ///
    /// This is used internally after realization to update the tensor's UOp
    /// to point to the materialized buffer.
    pub(crate) fn set_uop(&self, uop: Arc<UOp>) {
        *self.entry.uop.write() = uop;
    }
}

#[cfg(test)]
mod test;
