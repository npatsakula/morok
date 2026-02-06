//! Type definitions for the kernel optimization layer.
//!
//! This module defines the core types used in kernel optimization:
//! - `OptOps`: Optimization operations (UPCAST, LOCAL, UNROLL, etc.)
//! - `Opt`: An optimization descriptor combining operation and parameters
//!
//! Note: `AxisType` is re-exported from `morok_ir` where it's defined.
use std::fmt;

// Re-export AxisType from IR (where it belongs)
pub use morok_ir::AxisType;

use super::error::*;

/// Optimization operations for kernel transformation.
///
/// Each operation represents a specific kernel optimization strategy:
/// - Parallelization: `LOCAL`, `THREAD`, `GLOBAL`
/// - Vectorization: `UPCAST`
/// - Loop optimization: `UNROLL`, `GROUP`, `GROUPTOP`
/// - Layout: `SWAP`, `PADTO`
/// - Hardware acceleration: `TC` (Tensor Cores)
/// - Configuration: `NOLOCALS`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OptOps {
    /// Apply tensor core optimization (hardware matrix multiplication).
    TC,
    /// Vectorize by combining multiple iterations (SIMD).
    UPCAST,
    /// Unroll reduction loops for better instruction-level parallelism.
    UNROLL,
    /// Move axis to local/shared memory (GPU workgroup dimension).
    LOCAL,
    /// Add CPU threading for parallelism.
    THREAD,
    /// Split reduction with shared memory synchronization (inner split).
    GROUP,
    /// Split reduction with shared memory synchronization (outer split).
    GROUPTOP,
    /// Disable local memory usage.
    NOLOCALS,
    /// Pad dimension to make it divisible by required amounts.
    PADTO,
    /// Swap two global axes.
    SWAP,
}

impl fmt::Display for OptOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TC => write!(f, "TC"),
            Self::UPCAST => write!(f, "UPCAST"),
            Self::UNROLL => write!(f, "UNROLL"),
            Self::LOCAL => write!(f, "LOCAL"),
            Self::THREAD => write!(f, "THREAD"),
            Self::GROUP => write!(f, "GROUP"),
            Self::GROUPTOP => write!(f, "GROUPTOP"),
            Self::NOLOCALS => write!(f, "NOLOCALS"),
            Self::PADTO => write!(f, "PADTO"),
            Self::SWAP => write!(f, "SWAP"),
        }
    }
}

/// Argument types for optimization operations.
///
/// Some operations take a single integer argument (amount, axis, size),
/// while TC takes a tuple of configuration parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OptArg {
    /// Single integer argument (amount, axis index, size).
    Int(usize),
    /// Tuple for tensor core configuration: (tc_select, tc_opt_level, use_tensor_cores).
    TensorCore { tc_select: i32, opt_level: usize, use_tc: usize },
    /// Tuple for SWAP operation: (other_axis).
    Swap { other_axis: usize },
}

impl OptArg {
    /// Get the type name of this OptArg variant.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Int(_) => "Int",
            Self::TensorCore { .. } => "TensorCore",
            Self::Swap { .. } => "Swap",
        }
    }

    /// Extract integer value, returning error if not an Int variant.
    pub fn int(&self) -> Result<usize, OptError> {
        match self {
            Self::Int(v) => Ok(*v),
            _ => InvalidArgTypeSnafu { expected: "Int", found: self.type_name() }.fail(),
        }
    }

    /// Extract tensor core configuration, returning error if not a TensorCore variant.
    pub fn tc(&self) -> Result<(i32, usize, usize), OptError> {
        match self {
            Self::TensorCore { tc_select, opt_level, use_tc } => Ok((*tc_select, *opt_level, *use_tc)),
            _ => InvalidArgTypeSnafu { expected: "TensorCore", found: self.type_name() }.fail(),
        }
    }

    /// Extract swap configuration, returning error if not a Swap variant.
    pub fn swap(&self) -> Result<usize, OptError> {
        match self {
            Self::Swap { other_axis } => Ok(*other_axis),
            _ => InvalidArgTypeSnafu { expected: "Swap", found: self.type_name() }.fail(),
        }
    }
}

impl From<usize> for OptArg {
    fn from(v: usize) -> Self {
        Self::Int(v)
    }
}

/// An optimization descriptor combining operation type and parameters.
///
/// # Examples
///
/// ```ignore
/// // Upcast axis 2 by amount 4
/// let opt = Opt::new(OptOps::UPCAST, Some(2), OptArg::Int(4));
///
/// // Apply tensor cores
/// let opt = Opt::new(OptOps::TC, None, OptArg::TensorCore {
///     tc_select: -1,
///     opt_level: 2,
///     use_tc: 1,
/// });
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Opt {
    /// The optimization operation to perform.
    pub op: OptOps,
    /// Optional axis index (logical, not physical).
    ///
    /// Some operations (like NOLOCALS, TC) don't operate on a specific axis.
    /// For operations like UNROLL and GROUP, this is an index into a filtered
    /// list (unrollable_dims or reduce axes), not an absolute axis index.
    pub axis: Option<usize>,
    /// Operation-specific argument (amount, size, or configuration).
    pub arg: OptArg,
}

impl Opt {
    /// Create a new optimization descriptor.
    pub fn new(op: OptOps, axis: Option<usize>, arg: OptArg) -> Self {
        Self { op, axis, arg }
    }

    /// Create an UPCAST optimization.
    pub fn upcast(axis: usize, amount: usize) -> Self {
        Self::new(OptOps::UPCAST, Some(axis), OptArg::Int(amount))
    }

    /// Create a LOCAL optimization.
    pub fn local(axis: usize, amount: usize) -> Self {
        Self::new(OptOps::LOCAL, Some(axis), OptArg::Int(amount))
    }

    /// Create an UNROLL optimization.
    pub fn unroll(axis: usize, amount: usize) -> Self {
        Self::new(OptOps::UNROLL, Some(axis), OptArg::Int(amount))
    }

    /// Create a GROUP optimization.
    pub fn group(axis: usize, amount: usize) -> Self {
        Self::new(OptOps::GROUP, Some(axis), OptArg::Int(amount))
    }

    /// Create a GROUPTOP optimization.
    pub fn grouptop(axis: usize, amount: usize) -> Self {
        Self::new(OptOps::GROUPTOP, Some(axis), OptArg::Int(amount))
    }

    /// Create a THREAD optimization.
    pub fn thread(axis: usize, amount: usize) -> Self {
        Self::new(OptOps::THREAD, Some(axis), OptArg::Int(amount))
    }

    /// Create a PADTO optimization.
    pub fn padto(axis: usize, size: usize) -> Self {
        Self::new(OptOps::PADTO, Some(axis), OptArg::Int(size))
    }

    /// Create a SWAP optimization.
    pub fn swap(axis: usize, other_axis: usize) -> Self {
        Self::new(OptOps::SWAP, Some(axis), OptArg::Swap { other_axis })
    }

    /// Create a TC (tensor core) optimization.
    pub fn tc(axis: Option<usize>, tc_select: i32, opt_level: usize, use_tc: usize) -> Self {
        Self::new(OptOps::TC, axis, OptArg::TensorCore { tc_select, opt_level, use_tc })
    }

    /// Create a NOLOCALS optimization.
    pub fn nolocals() -> Self {
        Self::new(OptOps::NOLOCALS, None, OptArg::Int(0))
    }
}

impl fmt::Display for Opt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.op)?;
        if let Some(axis) = self.axis {
            write!(f, "{}, ", axis)?;
        }
        match &self.arg {
            OptArg::Int(v) => write!(f, "{}", v),
            OptArg::TensorCore { tc_select, opt_level, use_tc } => {
                write!(f, "tc_sel={}, opt={}, use={}", tc_select, opt_level, use_tc)
            }
            OptArg::Swap { other_axis } => write!(f, "swap={}", other_axis),
        }?;
        write!(f, ")")
    }
}
