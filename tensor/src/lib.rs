use bon::bon;
use snafu::ResultExt;
use std::{cell::RefCell, rc::Rc, sync::Arc};

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
pub mod realize;
pub mod reduce;
pub mod schedule;
pub mod shape_ops;
pub mod traits;

/// Reference to a kernel used by a tensor.
///
/// Each tensor tracks which kernels were compiled during its realization.
/// This allows `tensor.kernels()` to return only the kernels for that specific tensor,
/// not a global list contaminated by other tensors.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct KernelRef {
    /// AST UOp ID (for global dedup cache lookup)
    ast_id: u64,
    /// Device it was compiled for
    device: String,
    /// Rendered code (for debugging)
    code: String,
    /// Entry point name
    entry_point: String,
}

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
/// # Examples
///
/// ```
/// # use morok_tensor::Tensor;
/// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
/// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
/// let c = &a + &b;  // Lazy - only builds UOp graph
/// let realized = c.realize().unwrap();  // Executes the computation
/// println!("Kernels used: {:?}", realized.kernels());
/// ```
#[derive(Clone)]
pub struct Tensor {
    /// The computation graph (lazy)
    uop: Arc<UOp>,
    /// Kernels used by this tensor (populated by realize())
    /// Stored in Rc<RefCell> to allow mutation during realize()
    kernels: Rc<RefCell<Vec<KernelRef>>>,
}

#[bon]
impl Tensor {
    fn new(uop: Arc<UOp>) -> Self {
        Self { uop, kernels: Rc::new(RefCell::new(Vec::new())) }
    }

    /// Get kernels for THIS tensor only (not global pollution).
    ///
    /// Returns information about each kernel that was compiled and executed
    /// during this tensor's realization. This list is specific to this tensor
    /// and won't include kernels from other tensors.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    /// let c = (&a + &b).realize()?;
    ///
    /// // Only shows the kernel for this specific addition
    /// for kernel in c.kernels() {
    ///     println!("Kernel: {}", kernel.name);
    ///     println!("Code:\n{}", kernel.code);
    /// }
    /// ```
    pub fn kernels(&self) -> Vec<KernelInfo> {
        self.kernels
            .borrow()
            .iter()
            .map(|k| KernelInfo {
                name: k.entry_point.clone(),
                code: k.code.clone(),
                entry_point: k.entry_point.clone(),
                backend: "llvm".to_string(),
            })
            .collect()
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
        buffer_registry::get_or_create_buffer(buffer_uop.id, || Ok(buffer)).expect("Buffer registration failed");

        let uop = buffer_uop.try_reshape(&shape).expect("this reshape is always successful");
        Self::new(uop)
    }

    /// Get a reference to the underlying buffer from the registry.
    /// This allows multi-buffer operations to access buffers from multiple tensors.
    /// Uses `.base()` to walk through movement operations to find the actual buffer.
    pub fn buffer(&self) -> Option<Buffer> {
        buffer_registry::get_buffer(self.uop.base().id)
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
        self.uop.device_spec().unwrap_or(DeviceSpec::Cpu)
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

        let copy_uop = self.uop.copy_to_device(device);
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
        let casted = UOp::cast(self.uop.clone(), dtype);
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
        let shape = self.uop.shape().context(UOpSnafu)?.ok_or(Error::NoShape)?;

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
}

#[cfg(test)]
mod test;
