use snafu::ResultExt;
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use morok_device::{Buffer, registry};
use morok_dtype::ext::HasDType;
use morok_ir::{DeviceSpec, SInt, UOp, shape::Shape};

pub mod error;
use error::*;

pub mod activation;
pub mod arithmetic;
pub mod bitwise;
pub mod broadcast;
pub mod conditional;
pub mod math;
pub mod matmul;
pub mod reduce;
pub mod shape_ops;
pub mod traits;

// Thread-local buffer registry (matches Tinygrad's single-threaded model)
// Maps UOp id -> actual device Buffer
thread_local! {
    static BUFFERS: RefCell<HashMap<u64, Buffer>> = RefCell::new(HashMap::new());
}

/// Tensor represents a multi-dimensional array with lazy evaluation.
///
/// Operations like addition and multiplication build a computation graph
/// without allocating buffers. Buffers are only allocated when:
/// - Creating input tensors via `from_slice()`
/// - Evaluating the computation graph (future feature)
///
/// # Examples
///
/// ```
/// # use morok_tensor::Tensor;
/// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
/// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
/// let c = &a + &b;  // Lazy - only builds UOp graph
/// let d = &c * &a;  // Chain operations
/// ```
#[derive(Clone)]
pub struct Tensor {
    uop: Rc<UOp>,
}

impl Tensor {
    fn new(uop: Rc<UOp>) -> Self {
        Self { uop }
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

    pub fn from_slice<T: HasDType, C: AsRef<[T]>>(source: C) -> Self {
        let source = source.as_ref();
        let shape = Shape::from_iter([SInt::Const(source.len())]);
        let dtype = T::DTYPE;

        let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, source.len(), dtype.clone());

        let device = registry::cpu().expect("CPU always should be accessible");
        let mut buffer = Buffer::new(device, dtype.clone(), vec![source.len()], Default::default());
        let bytes = unsafe { std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * dtype.bytes()) };
        buffer.copyin(bytes).expect("CPU buffer write always successful");
        BUFFERS.with(|buffers| buffers.borrow_mut().insert(buffer_uop.id, buffer));

        let uop = buffer_uop.try_reshape(&shape).expect("this reshape is always successful");
        Self { uop }
    }

    /// Get a reference to the underlying buffer from the registry.
    /// This allows multi-buffer operations to access buffers from multiple tensors.
    /// Uses `.base()` to walk through movement operations to find the actual buffer.
    pub fn buffer(&self) -> Option<Buffer> {
        BUFFERS.with(|buffers| buffers.borrow().get(&self.uop.base().id).cloned())
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
}

#[cfg(test)]
mod test;
