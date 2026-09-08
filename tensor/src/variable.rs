//! Symbolic variables for dynamic tensor dimensions.
//!
//! Variables allow tensor shapes to contain symbolic dimensions (e.g., batch size,
//! sequence length) that are resolved to concrete values at execution time.
//!
//! # Tinygrad Alignment
//!
//! Matches Tinygrad's `Variable = UOp` where `Variable("i", 1, 10)` creates
//! an ALU `PARAM` UOp and `.bind(val)` produces a `BIND(PARAM, CONST)` UOp.
//!
//! # Example
//!
//! ```ignore
//! use svod_tensor::{Variable, Tensor};
//! use svod_dtype::DType;
//!
//! let batch = Variable::new("batch", 1, 32);
//! let bound = batch.bind(16)?;
//!
//! let x = Tensor::full_dynamic(&[bound.as_sint(), 784.into()], 0.0, DType::Float32)?;
//! x.realize()?;
//! ```

use std::sync::Arc;

use svod_ir::{ConstValue, Op, SInt, UOp};

use crate::error::{Result, VariableOutOfRangeSnafu};
use snafu::ensure;
use svod_ir::ops;

/// A symbolic variable for dynamic tensor dimensions.
///
/// Wraps an ALU `PARAM` UOp with known bounds `[min_val, max_val]`.
/// Variables are created unbound, then bound to concrete values via [`bind()`](Self::bind).
///
/// The same `Variable` can be bound to different values for different executions,
/// enabling dynamic batch sizes, sequence lengths, etc.
#[derive(Clone)]
pub struct Variable {
    uop: Arc<UOp>,
}

impl Variable {
    /// Create a new symbolic variable with inclusive bounds.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique variable name (used as kernel parameter name)
    /// * `min_val` - Minimum allowed value (inclusive)
    /// * `max_val` - Maximum allowed value (inclusive)
    ///
    /// # Panics
    ///
    /// Panics if `min_val > max_val`.
    pub fn new(name: &str, min_val: i64, max_val: i64) -> Self {
        assert!(min_val <= max_val, "Variable '{name}': min_val ({min_val}) > max_val ({max_val})");
        Self { uop: UOp::define_var(name.to_string(), min_val, max_val) }
    }

    /// Bind this variable to a concrete value.
    ///
    /// Returns a [`BoundVariable`] whose [`as_sint()`](BoundVariable::as_sint) can be
    /// used as a tensor dimension.
    ///
    /// # Errors
    ///
    /// Returns [`VariableOutOfRange`](crate::error::ErrorKind::VariableOutOfRange) if
    /// `val` is outside `[min_val, max_val]`.
    pub fn bind(&self, val: i64) -> Result<BoundVariable> {
        let (min, max) = self.bounds();
        ensure!(val >= min && val <= max, VariableOutOfRangeSnafu { name: self.name().to_string(), val, min, max });
        let val_uop = UOp::const_(self.uop.dtype(), ConstValue::Int(val));
        let bind_uop = self.uop.bind(val_uop);
        Ok(BoundVariable { var: self.clone(), value: val, uop: bind_uop })
    }

    /// Variable name.
    pub fn name(&self) -> &str {
        match self.uop.op() {
            Op::Param(ops::Param { arg, .. }) => arg.name.as_deref().expect("variable PARAM has a name"),
            _ => unreachable!("Variable always wraps Param"),
        }
    }

    /// Inclusive bounds `(min_val, max_val)`.
    pub fn bounds(&self) -> (i64, i64) {
        match self.uop.op() {
            Op::Param(ops::Param { arg, .. }) => {
                let (min, max) = arg.vmin_vmax.as_ref().expect("variable PARAM has bounds");
                let ConstValue::Int(min) = min.0 else { unreachable!("integer variable minimum") };
                let ConstValue::Int(max) = max.0 else { unreachable!("integer variable maximum") };
                (min, max)
            }
            _ => unreachable!("Variable always wraps Param"),
        }
    }

    /// Get the underlying variable `PARAM` UOp as an `SInt`.
    ///
    /// This is useful for constructing shapes that use the variable's max value
    /// for buffer allocation (unbound variable → allocate to max).
    pub fn as_sint(&self) -> SInt {
        SInt::Symbolic(self.uop.clone())
    }

    /// Get the underlying variable `PARAM` UOp.
    pub fn uop(&self) -> &Arc<UOp> {
        &self.uop
    }
}

impl std::fmt::Debug for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (min, max) = self.bounds();
        write!(f, "Variable({:?}, {}, {})", self.name(), min, max)
    }
}

/// A variable bound to a concrete value.
///
/// Created by [`Variable::bind()`]. Use [`as_sint()`](Self::as_sint) to get an `SInt`
/// suitable as a tensor dimension.
#[derive(Clone)]
pub struct BoundVariable {
    var: Variable,
    value: i64,
    uop: Arc<UOp>,
}

impl BoundVariable {
    /// Get an `SInt` representing this bound variable.
    ///
    /// The returned `SInt::Symbolic` contains `BIND(DEFINE_VAR, CONST(value))`,
    /// which flows through the existing shape infrastructure (reshape, permute,
    /// expand, binary ops all handle `SInt::Symbolic`).
    pub fn as_sint(&self) -> SInt {
        SInt::Symbolic(self.uop.clone())
    }

    /// The bound concrete value.
    pub fn value(&self) -> i64 {
        self.value
    }

    /// The underlying variable definition.
    pub fn variable(&self) -> &Variable {
        &self.var
    }

    /// Decompose into the variable and its bound value.
    pub fn unbind(self) -> (Variable, i64) {
        (self.var, self.value)
    }

    /// Get `(name, value)` pair for use with `ExecutionPlan::execute_with_vars`.
    pub fn as_var_val(&self) -> (&str, i64) {
        (self.variable().name(), self.value())
    }

    /// Get the underlying `BIND` UOp.
    pub fn uop(&self) -> &Arc<UOp> {
        &self.uop
    }
}

impl std::fmt::Debug for BoundVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BoundVariable({:?} = {})", self.var.name(), self.value)
    }
}

// Allow using BoundVariable directly as SInt in shape expressions
impl From<BoundVariable> for SInt {
    fn from(bv: BoundVariable) -> SInt {
        bv.as_sint()
    }
}

impl From<&BoundVariable> for SInt {
    fn from(bv: &BoundVariable) -> SInt {
        bv.as_sint()
    }
}
