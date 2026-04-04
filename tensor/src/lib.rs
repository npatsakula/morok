use bon::bon;
use std::sync::Arc;

use morok_device::Buffer;
use morok_dtype::DType;
use morok_dtype::ext::HasDType;
use morok_ir::{ConstValue, DeviceSpec, Op, SInt, UOp, shape::Shape};
use snafu::ResultExt;

/// Extract max value from an SInt for buffer allocation.
///
/// Concrete dims return their value. Symbolic dims (DefineVar, Bind)
/// return `max_val` from the underlying Variable, enabling rebinding
/// without reallocation. Matches Tinygrad's `x.vmax`.
fn sint_vmax(s: &SInt) -> usize {
    match s {
        SInt::Const(v) => *v,
        SInt::Symbolic(uop) => match uop.op() {
            Op::DefineVar { max_val, .. } => *max_val as usize,
            Op::Bind { var, .. } => match var.op() {
                Op::DefineVar { max_val, .. } => *max_val as usize,
                _ => 1,
            },
            _ => 1,
        },
        SInt::Infer => panic!("cannot compute vmax of SInt::Infer"),
    }
}

pub mod error;
use error::*;

pub mod activation;
pub mod arithmetic;
pub mod bitwise;
pub mod broadcast;
pub mod conditional;
pub mod config;
pub mod data;
pub mod einsum;
pub mod indexing;
pub mod math;
pub mod matmul;
pub mod memory_planner;
pub mod nn;
pub mod realize;
pub mod reduce;
pub mod schedule;
pub(crate) mod schedule_cache;
pub mod shape_ops;
pub mod tensor_registry;
pub mod traits;
pub mod transformer;
pub mod variable;

// Re-export for public API
pub use config::PrepareConfig;
pub use morok_runtime::CpuBackend;
pub use tensor_registry::apply_map_to_tensors;
pub use variable::{BoundVariable, Variable};

/// Reduction operations supported by cumulative reduce (`_cumalu`).
#[derive(Debug, Clone, Copy)]
enum CumReduceOp {
    Add,
    Mul,
    #[allow(dead_code)]
    Max,
}

impl CumReduceOp {
    /// Identity element for this operation as f64, used as pad fill value.
    fn identity_value(&self, dtype: DType) -> f64 {
        match self {
            CumReduceOp::Add => 0.0,
            CumReduceOp::Mul => 1.0,
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
/// let mut c = &a + &b;  // Lazy - only builds UOp graph
/// c.realize().unwrap();  // Executes the computation
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

    /// Check if this tensor has zero total elements (any shape dimension is 0).
    fn has_zero_elements(&self) -> bool {
        match self.uop().shape() {
            Ok(Some(shape)) => shape.iter().any(|dim| dim.as_const() == Some(0)),
            _ => false,
        }
    }

    /// Ensure buffer is attached if the UOp has buffer identity.
    ///
    /// When `apply_map_to_tensors` substitutes a tensor's UOp with a realized
    /// BUFFER+RESHAPE, the Tensor struct's `buffer` field isn't updated.
    /// This method looks up the buffer from the registry and attaches it.
    pub(crate) fn ensure_buffer(&self) {
        let buffer_id = self.uop().base().id;
        if let Some(buf_arc) = tensor_registry::get_buffer_arc(buffer_id) {
            self.entry.set_buffer(buf_arc);
        }
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

    /// Create an uninitialized buffer-backed tensor with the given shape and dtype.
    ///
    /// No device memory is allocated — only the BUFFER UOp is created.
    /// Use `assign()` to bind real data before `realize()`.
    /// Matches Tinygrad's `Tensor.empty(*shape)`.
    pub fn empty(shape: &[usize], dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, numel, dtype);
        let ir_shape = Shape::from_iter(shape.iter().map(|&d| SInt::Const(d)));
        let uop = buffer_uop.try_reshape(&ir_shape).expect("shape matches element count");
        Self::new(uop)
    }

    /// Create an uninitialized buffer-backed tensor with symbolic (dynamic) dimensions.
    ///
    /// Buffer is sized to `prod(vmax)` — each symbolic dim uses its Variable's
    /// max_val for allocation. This enables rebinding to any value in [min, max]
    /// without reallocation. Matches Tinygrad's
    /// `prod([x.vmax if isinstance(x, UOp) else x for x in shape])`.
    pub fn empty_dynamic(shape: &[SInt], dtype: DType) -> Self {
        let numel: usize = shape.iter().map(sint_vmax).product();
        let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, numel, dtype);
        let ir_shape = Shape::from_iter(shape.iter().cloned());
        let uop = buffer_uop.try_reshape(&ir_shape).expect("shape valid for reshape");
        Self::new(uop)
    }

    /// Create an empty 0-element tensor with the given dtype and shape `[0]`.
    pub fn empty_zero(dtype: DType) -> Self {
        Self::empty(&[0], dtype)
    }

    /// Create a tensor filled with a constant value, broadcast to the given shape.
    pub fn full(shape: &[usize], value: impl Into<ConstValue>, dtype: DType) -> Result<Self> {
        let scalar = Self::const_(value, dtype);
        if shape.is_empty() {
            return Ok(scalar);
        }
        let expand_shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
        scalar.try_reshape(vec![1; shape.len()])?.try_expand(&expand_shape)
    }

    /// Create a tensor filled with a constant value, using symbolic (dynamic) dimensions.
    ///
    /// Dimensions can be concrete (`SInt::Const`) or symbolic (`SInt::Symbolic`
    /// from [`Variable::bind()`](crate::Variable::bind)).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use morok_tensor::{Tensor, Variable};
    /// use morok_dtype::DType;
    ///
    /// let batch = Variable::new("batch", 1, 32);
    /// let x = Tensor::full_dynamic(&[batch.bind(16)?.into(), 784.into()], 0.0, DType::Float32)?;
    /// ```
    pub fn full_dynamic(shape: &[SInt], value: impl Into<ConstValue>, dtype: DType) -> Result<Self> {
        let const_uop = UOp::const_(dtype.clone(), value.into());
        if shape.is_empty() {
            return Ok(Self::new(const_uop));
        }
        // Reshape scalar to [1, 1, ...] then expand to target shape.
        // Expand handles both concrete and symbolic (SInt::Symbolic) dims.
        let ones: Shape = vec![SInt::Const(1); shape.len()].into();
        let target: Shape = shape.to_vec().into();
        let reshaped = const_uop.try_reshape(&ones).context(error::UOpSnafu)?;
        let expanded = reshaped.try_expand(&target).context(error::UOpSnafu)?;
        Ok(Self::new(expanded))
    }

    /// Create a zero-filled tensor with symbolic (dynamic) dimensions.
    pub fn zeros_dynamic(shape: &[SInt], dtype: DType) -> Result<Self> {
        Self::full_dynamic(shape, ConstValue::zero(dtype.base()), dtype)
    }

    /// Create a one-filled tensor with symbolic (dynamic) dimensions.
    pub fn ones_dynamic(shape: &[SInt], dtype: DType) -> Result<Self> {
        Self::full_dynamic(shape, ConstValue::one(dtype.base()), dtype)
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
            CumReduceOp::Mul => x.prod(-1isize)?,
            CumReduceOp::Max => x.max(-1isize)?,
        };

        // 5. Transpose back
        if axis_idx != ndim - 1 { x.try_transpose(axis_idx as isize, -1) } else { Ok(x) }
    }

    /// Cumulative sum along an axis.
    pub fn cumsum(&self, axis: isize) -> Result<Self> {
        self._cumalu(axis, CumReduceOp::Add)
    }

    /// Cumulative product along an axis.
    pub fn cumprod(&self, axis: isize) -> Result<Self> {
        self._cumalu(axis, CumReduceOp::Mul)
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

    /// Create 1D tensor with `steps` evenly spaced values from `start` to `end` (inclusive).
    pub fn linspace(start: f64, end: f64, steps: usize, dtype: DType) -> Result<Self> {
        if steps == 0 {
            return Ok(Self::empty_zero(dtype));
        }
        if steps == 1 {
            return Self::full(&[1], start, dtype);
        }
        let t = Self::arange(steps as i64, None, None)?;
        let scale = Self::const_((end - start) / (steps as f64 - 1.0), DType::Float64);
        let offset = Self::const_(start, DType::Float64);
        t.cast(DType::Float64)?.try_mul(&scale)?.try_add(&offset)?.cast(dtype)
    }

    /// Shared implementation: `full(count, step) → cumsum → + (start - step)`.
    fn arange_inner(start: f64, stop: f64, step: f64, dtype: DType, is_float: bool) -> Result<Self> {
        let count = ((stop - start) / step).ceil() as i64;
        if count <= 0 {
            return Ok(Self::empty_zero(dtype));
        }
        let count = count as usize;
        let val = |v: f64| if is_float { ConstValue::Float(v) } else { ConstValue::Int(v as i64) };
        let step_tensor = Self::full(&[count], val(step), dtype.clone())?;
        let cumsum = step_tensor._cumalu(0, CumReduceOp::Add)?;
        let offset = Self::const_(val(start - step), dtype.clone());
        cumsum.try_add(&offset)?.cast(dtype)
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
    /// let mut gpu_tensor = cpu_tensor.to(DeviceSpec::Cuda { device_id: 0 });
    /// gpu_tensor.realize()?;  // Actually transfers data
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

    /// Bitcast tensor to a different dtype (reinterpret bits, same byte size required).
    pub fn bitcast(&self, dtype: morok_dtype::DType) -> Result<Self> {
        Ok(Self::new(self.uop().bitcast(dtype)))
    }

    /// Assign a value tensor to this tensor in-place.
    ///
    /// Creates an ASSIGN UOp linking this tensor's BUFFER to the value.
    /// The assignment is resolved lazily during `realize()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let placeholder = Tensor::empty(&[2, 3], DType::Float32);
    /// let real_data = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .try_reshape(&[2, 3]).unwrap();
    /// placeholder.assign(&real_data);
    /// ```
    pub fn assign(&self, value: &Tensor) {
        let target_uop = self.uop();
        let assign_uop = UOp::assign(target_uop.clone(), value.uop());
        // Track for side-realization during realize() (Tinygrad: _pending_assigns).
        let buffer_id = target_uop.base().id;
        tensor_registry::add_pending_assign(buffer_id, assign_uop.clone());
        self.set_uop(assign_uop);
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
    /// let mut c = Tensor::const_(5.0f32, DType::Float32).contiguous();
    /// c.realize()?;
    /// assert!(c.buffer().is_some());
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

impl Tensor {
    /// Helper to broadcast a scalar constant to match this tensor's shape.
    pub(crate) fn broadcast_scalar(&self, value: ConstValue) -> Result<Self> {
        let shape = self.shape()?;
        let scalar = Self::new(UOp::const_(self.uop().dtype(), value));
        scalar.broadcast_to(&shape)
    }

    /// Broadcast a dtype-aware zero to match this tensor's shape.
    pub fn zero(&self) -> Result<Self> {
        let sdtype = self.uop().dtype().scalar().expect("scalar dtype");
        self.broadcast_scalar(ConstValue::zero(sdtype))
    }

    /// Broadcast a dtype-aware one to match this tensor's shape.
    pub fn one(&self) -> Result<Self> {
        let sdtype = self.uop().dtype().scalar().expect("scalar dtype");
        self.broadcast_scalar(ConstValue::one(sdtype))
    }

    /// Identity matrix of shape `[n, m]` with the given dtype.
    pub fn eye(n: usize, m: usize, dtype: DType) -> Result<Self> {
        let rows = Self::arange(n as i64, None, None)?.try_unsqueeze(-1)?;
        let cols = Self::arange(m as i64, None, None)?;
        rows.try_eq(&cols)?.cast(dtype)
    }
}

#[bon]
impl Tensor {
    /// Cumulative sum with exclusive and reverse options.
    #[builder]
    pub fn cumsum_with(
        &self,
        axis: isize,
        #[builder(default = false)] exclusive: bool,
        #[builder(default = false)] reverse: bool,
    ) -> Result<Self> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let axis_idx = Self::normalize_axis(axis, ndim)?;
        let mut result = self.clone();
        if reverse {
            result = result.flip(&[axis_idx as isize])?;
        }
        if exclusive {
            let dim_size = shape[axis_idx].as_const().unwrap() as isize;
            let mut pad_spec: Vec<(isize, isize)> = vec![(0, 0); ndim];
            pad_spec[axis_idx] = (1, 0);
            result = result.try_pad(&pad_spec)?;
            let mut shrink_spec: Vec<(isize, isize)> =
                result.shape()?.iter().map(|s| (0, s.as_const().unwrap() as isize)).collect();
            shrink_spec[axis_idx] = (0, dim_size);
            result = result.try_shrink(&shrink_spec)?;
        }
        result = result.cumsum(axis_idx as isize)?;
        if reverse {
            result = result.flip(&[axis_idx as isize])?;
        }
        Ok(result)
    }

    /// Cumulative product with exclusive and reverse options.
    #[builder]
    pub fn cumprod_with(
        &self,
        axis: isize,
        #[builder(default = false)] exclusive: bool,
        #[builder(default = false)] reverse: bool,
    ) -> Result<Self> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let axis_idx = Self::normalize_axis(axis, ndim)?;
        let mut result = self.clone();
        if reverse {
            result = result.flip(&[axis_idx as isize])?;
        }
        if exclusive {
            let dim_size = shape[axis_idx].as_const().unwrap() as isize;
            let mut pad_spec: Vec<(isize, isize)> = vec![(0, 0); ndim];
            pad_spec[axis_idx] = (1, 0);
            result = result.try_pad_value(&pad_spec, 1.0)?;
            let mut shrink_spec: Vec<(isize, isize)> =
                result.shape()?.iter().map(|s| (0, s.as_const().unwrap() as isize)).collect();
            shrink_spec[axis_idx] = (0, dim_size);
            result = result.try_shrink(&shrink_spec)?;
        }
        result = result.cumprod(axis_idx as isize)?;
        if reverse {
            result = result.flip(&[axis_idx as isize])?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod test;
