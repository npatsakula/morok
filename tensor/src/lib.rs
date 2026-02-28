use bon::bon;
use snafu::ResultExt;
use std::sync::Arc;

use morok_device::{Buffer, registry};
use morok_dtype::DType;
use morok_dtype::ext::HasDType;
use morok_ir::{ConstValue, DeviceSpec, SInt, UOp, shape::Shape};

pub mod error;
use error::*;

pub mod activation;
pub mod arithmetic;
pub mod bitwise;
pub mod broadcast;
pub mod conditional;
pub mod indexing;
pub mod math;
pub mod matmul;
pub mod memory_planner;
pub mod nn;
pub mod realize;
pub mod reduce;
pub mod schedule;
pub mod shape_ops;
pub mod tensor_registry;
pub mod traits;

// Re-export for public API
pub use tensor_registry::apply_map_to_tensors;

/// Reduction operations supported by cumulative reduce (`_cumalu`).
#[derive(Debug, Clone, Copy)]
enum CumReduceOp {
    Add,
    #[allow(dead_code)]
    Max,
}

impl CumReduceOp {
    /// Identity element for this operation as f64, used as pad fill value.
    fn identity_value(&self, dtype: DType) -> f64 {
        match self {
            CumReduceOp::Add => 0.0,
            CumReduceOp::Max => {
                if dtype.is_int() {
                    i64::MIN as f64
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }
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

    /// Create a new tensor from a UOp, preserving buffer from self.
    ///
    /// Used by movement operations (reshape, permute, etc.) that create
    /// new view UOps but share the underlying buffer.
    fn with_same_buffer(&self, uop: Arc<UOp>) -> Self {
        let entry = tensor_registry::register_tensor(uop);
        Self { entry, buffer: self.buffer.clone() }
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

    /// Create an empty (0-element) tensor with the given dtype and shape `[0]`.
    ///
    /// Matches Tinygrad's `Tensor([], dtype=dtype)`. No buffer is allocated.
    pub fn empty(dtype: DType) -> Self {
        let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 0, dtype);
        let shape = Shape::from_iter([SInt::Const(0)]);
        let uop = buffer_uop.try_reshape(&shape).expect("empty reshape to [0]");
        Self::new(uop)
    }

    /// Create a tensor filled with a constant value, broadcast to the given shape.
    pub fn full(shape: &[usize], value: impl Into<ConstValue>, dtype: DType) -> Result<Self> {
        let scalar = Self::const_(value, dtype);
        if shape.is_empty() {
            return Ok(scalar);
        }
        let expand_shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
        scalar.try_reshape(&vec![1; shape.len()])?.try_expand(&expand_shape)
    }

    /// Cumulative reduce along an axis using a sliding-window approach.
    ///
    /// Decomposes prefix-sum/prefix-max/prefix-prod into existing ops:
    /// pad → pool (sliding windows) → reduce. Fully lazy, O(1) graph nodes.
    fn _cumalu(&self, axis: isize, reduce: CumReduceOp) -> Result<Self> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let axis_idx = Self::normalize_axis(axis, ndim)?;
        let n = shape[axis_idx]
            .as_const()
            .ok_or_else(|| Error::SymbolicShapeUnsupported { operation: "_cumalu".to_string() })?;

        if n <= 1 {
            return Ok(self.clone());
        }

        // 1. Transpose target axis to last
        let x = if axis_idx != ndim - 1 { self.try_transpose(axis_idx as isize, -1)? } else { self.clone() };

        // 2. Pad left with (n-1) identity elements
        let identity = reduce.identity_value(self.uop().dtype());
        let mut padding = vec![(0isize, 0isize); ndim];
        padding[ndim - 1] = ((n - 1) as isize, 0);
        let x = x.try_pad_value(&padding, identity)?;

        // 3. Pool with kernel=n, stride=1
        let x = x.pool(&[n], &[1], &[1])?;

        // 4. Reduce last dim
        let x = match reduce {
            CumReduceOp::Add => x.sum(-1isize)?,
            CumReduceOp::Max => x.max(-1isize)?,
        };

        // 5. Transpose back
        if axis_idx != ndim - 1 { x.try_transpose(axis_idx as isize, -1) } else { Ok(x) }
    }

    /// Create 1D tensor with evenly spaced values and explicit dtype.
    ///
    /// Generates values in the range `[start, stop)` with given step size.
    /// If `stop` is None, treats `start` as stop and starts from 0.
    ///
    /// Uses lazy `full(step)._cumalu(0, Add) + (start - step)` which
    /// `reduce_collapse` simplifies into `RANGE * step + offset`.
    /// Create 1D tensor with evenly spaced values (integer parameters).
    ///
    /// Matches Tinygrad's `Tensor.arange()`: `full(step) → cumsum → + (start - step)`.
    pub fn arange_with_dtype(start: i64, stop: Option<i64>, step: Option<i64>, dtype: DType) -> Result<Self> {
        let (start, stop) = match stop {
            Some(s) => (start, s),
            None => (0, start),
        };
        let step = step.unwrap_or(1);
        if step == 0 {
            return Err(Error::SymbolicShapeUnsupported { operation: "arange with step=0".to_string() });
        }
        Self::arange_inner(start as f64, stop as f64, step as f64, dtype, false)
    }

    /// Create 1D tensor with evenly spaced Int32 values.
    pub fn arange(start: i64, stop: Option<i64>, step: Option<i64>) -> Result<Self> {
        Self::arange_with_dtype(start, stop, step, DType::Int32)
    }

    /// Create 1D tensor with evenly spaced values (float parameters).
    ///
    /// Handles float start/stop/step natively, matching Tinygrad's `Tensor.arange()`.
    pub fn arange_f64(start: f64, stop: f64, step: f64, dtype: DType) -> Result<Self> {
        if step == 0.0 {
            return Err(Error::SymbolicShapeUnsupported { operation: "arange with step=0".to_string() });
        }
        Self::arange_inner(start, stop, step, dtype, true)
    }

    /// Shared implementation: `full(count, step) → cumsum → + (start - step)`.
    fn arange_inner(start: f64, stop: f64, step: f64, dtype: DType, is_float: bool) -> Result<Self> {
        let count = ((stop - start) / step).ceil() as i64;
        if count <= 0 {
            return Ok(Self::empty(dtype));
        }
        let count = count as usize;
        let val = |v: f64| if is_float { ConstValue::Float(v) } else { ConstValue::Int(v as i64) };
        let step_tensor = Self::full(&[count], val(step), dtype.clone())?;
        let cumsum = step_tensor._cumalu(0, CumReduceOp::Add)?;
        let offset = Self::const_(val(start - step), dtype.clone());
        cumsum.try_add(&offset)?.cast(dtype)
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
        let buffer_uop_id = buffer_uop.id;

        // Get allocator for specified device
        let allocator = match &device {
            DeviceSpec::Cpu => registry::cpu().expect("CPU always should be accessible"),
            // For non-CPU devices, try to get from registry or fall back to CPU for now
            _ => registry::cpu().expect("CPU fallback for unsupported device"),
        };

        let mut buffer = Buffer::new(allocator, dtype.clone(), vec![source.len()], Default::default());
        let bytes = unsafe { std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * dtype.bytes()) };
        buffer.copyin(bytes).expect("Buffer write always successful");

        // Wrap buffer in Arc for RAII ownership
        let buffer_arc = Arc::new(buffer);

        let uop = buffer_uop.try_reshape(&shape).expect("this reshape is always successful");

        // Register tensor with buffer (also adds to buffer index for schedule lookups)
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
            return Err(error::Error::IrConstruction {
                details: format!(
                    "from_raw_bytes: data length {} != expected {} ({} elements * {} bytes)",
                    data.len(),
                    expected_bytes,
                    numel,
                    dtype.bytes()
                ),
            });
        }

        let flat_shape = Shape::from_iter([SInt::Const(numel)]);
        let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, numel, dtype.clone());
        let buffer_uop_id = buffer_uop.id;

        let allocator = registry::cpu().expect("CPU always accessible");
        let mut buffer = Buffer::new(allocator, dtype.clone(), vec![numel], Default::default());
        buffer.copyin(data).expect("Buffer write always successful");

        let buffer_arc = Arc::new(buffer);
        let uop = buffer_uop.try_reshape(&flat_shape).expect("flat reshape always succeeds");

        let entry = tensor_registry::register_tensor_with_buffer(uop, buffer_arc.clone(), buffer_uop_id);
        let tensor = Self::with_buffer(entry, buffer_arc);
        let isize_shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
        tensor.try_reshape(&isize_shape)
    }

    // === Constant Constructors ===

    /// Create a scalar constant tensor.
    ///
    /// Creates a 0-dimensional tensor containing a single constant value.
    /// The constant is embedded directly in the IR and does not allocate
    /// a buffer until realized (if needed).
    ///
    /// # Arguments
    /// * `value` - The constant value (will be converted to ConstValue)
    /// * `dtype` - The data type for the tensor
    ///
    /// # Examples
    /// ```ignore
    /// // Float constant
    /// let pi = Tensor::const_(3.14159, DType::Float32);
    ///
    /// // Integer constant
    /// let forty_two = Tensor::const_(42i64, DType::Int64);
    /// ```
    pub fn const_<T: Into<ConstValue>>(value: T, dtype: DType) -> Self {
        let const_val = value.into();
        let uop = UOp::const_(dtype, const_val);
        Self::new(uop)
    }

    /// Create a scalar constant tensor with dtype auto-inferred from value.
    ///
    /// Convenience method that infers dtype from the Rust type.
    ///
    /// # Examples
    /// ```ignore
    /// let f = Tensor::from_const(3.14f32);  // DType::Float32
    /// let i = Tensor::from_const(42i32);    // DType::Int32
    /// let b = Tensor::from_const(true);     // DType::Bool
    /// ```
    pub fn from_const<T: Into<ConstValue> + HasDType>(value: T) -> Self {
        let dtype = T::DTYPE;
        Self::const_(value, dtype)
    }

    /// Get a reference to the underlying buffer.
    ///
    /// Tensors own their buffers via RAII. Input tensors get their buffer
    /// from `from_slice()`, realized tensors get theirs from `realize()`.
    ///
    /// Returns `None` for lazy tensors that haven't been realized yet.
    /// Returns `Some(buffer)` for input tensors and realized tensors.
    pub fn buffer(&self) -> Option<Buffer> {
        // Tensor-owned buffer via RAII (no locks, no registry lookup)
        self.buffer.as_ref().map(|arc_buf| (**arc_buf).clone())
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
        let casted = self.uop().cast(dtype);
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

        // If no buffer, materialize the tensor.
        // Get shape first to check for zero-size tensors
        let uop = self.uop();
        let shape = uop.shape().context(UOpSnafu)?.ok_or(Error::NoShape)?;
        let dims: Vec<usize> = shape.iter().map(|dim| dim.as_const().unwrap_or(1)).collect();

        // Zero-size tensor: return empty ndarray without realization (matches Tinygrad)
        if dims.contains(&0) {
            let arr = ArrayD::from_shape_vec(IxDyn(&dims), vec![]).context(NdarrayShapeSnafu)?;
            return Ok(arr);
        }

        // Following Tinygrad's approach: `.numpy()` calls `.contiguous().realize()` first.
        let buffer = match self.buffer() {
            Some(buf) => buf,
            None => {
                let realized = self.clone().contiguous().realize()?;
                realized.buffer().ok_or(Error::NoBuffer)?
            }
        };

        // Validate dtype matches
        if buffer.dtype() != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: buffer.dtype() }.fail();
        }

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

    /// Ensure this tensor has contiguous memory layout.
    ///
    /// Creates a CONTIGUOUS UOp that forces materialization when realized.
    /// Following Tinygrad's approach, calling `.contiguous().realize()` on
    /// a pure constant tensor will create an actual buffer.
    ///
    /// # Examples
    /// ```ignore
    /// // Force a constant to be materialized
    /// let c = Tensor::const_(5.0f32, DType::Float32);
    /// let realized = c.contiguous().realize()?;
    /// assert!(realized.buffer().is_some());
    /// ```
    pub fn contiguous(&self) -> Self {
        let uop = self.uop();
        if matches!(uop.op(), morok_ir::Op::Contiguous { .. }) {
            return self.clone();
        }
        let contiguous_uop = uop.contiguous();
        Self::new(contiguous_uop)
    }
}

#[cfg(test)]
mod test;
