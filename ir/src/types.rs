//! Type definitions for IR operations.
//!
//! This module contains all the fundamental type enums and structs used throughout
//! the IR, including operation types, constant values, and metadata structures.

use std::hash::{Hash, Hasher};
use std::mem::discriminant;

use morok_device::DeviceSpec;
use morok_dtype::{DType, ScalarDType};

/// Constant value that can be stored in a UOp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstValue {
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
}

/// Helper macro to cast to target width and back to storage type (for proper truncation/extension).
macro_rules! cast_via {
    ($v:expr, $target:ty, $storage:ty) => {
        ($v as $target) as $storage
    };
}

/// Macro to generate casting logic by delegating to helper functions.
macro_rules! impl_cast {
    ($self:expr, $to:expr) => {
        match ($self, $to) {
            (ConstValue::Bool(v), dt) => cast_bool(v, dt)?,
            (ConstValue::Int(v), dt) => cast_int(v, dt)?,
            (ConstValue::UInt(v), dt) => cast_uint(v, dt)?,
            (ConstValue::Float(v), dt) => cast_float(v, dt)?,
        }
    };
}

#[inline]
fn cast_bool(v: bool, to: ScalarDType) -> Option<ConstValue> {
    use ScalarDType::*;
    Some(match to {
        Bool => ConstValue::Bool(v),
        Int8 | Int16 | Int32 | Int64 => ConstValue::Int(v as i64),
        UInt8 | UInt16 | UInt32 | UInt64 => ConstValue::UInt(v as u64),
        Float16 | BFloat16 | Float32 | Float64 => ConstValue::Float(v as u8 as f64),
        _ => return None,
    })
}

#[inline]
fn cast_int(v: i64, to: ScalarDType) -> Option<ConstValue> {
    use ScalarDType::*;
    Some(match to {
        Bool => ConstValue::Bool(v != 0),
        Int8 => ConstValue::Int(cast_via!(v, i8, i64)),
        Int16 => ConstValue::Int(cast_via!(v, i16, i64)),
        Int32 => ConstValue::Int(cast_via!(v, i32, i64)),
        Int64 => ConstValue::Int(v),
        UInt8 => ConstValue::UInt(cast_via!(v, u8, u64)),
        UInt16 => ConstValue::UInt(cast_via!(v, u16, u64)),
        UInt32 => ConstValue::UInt(cast_via!(v, u32, u64)),
        UInt64 => ConstValue::UInt(v as u64),
        Float16 | BFloat16 | Float32 | Float64 => ConstValue::Float(v as f64),
        _ => return None,
    })
}

#[inline]
fn cast_uint(v: u64, to: ScalarDType) -> Option<ConstValue> {
    use ScalarDType::*;
    Some(match to {
        Bool => ConstValue::Bool(v != 0),
        Int8 => ConstValue::Int(cast_via!(v, i8, i64)),
        Int16 => ConstValue::Int(cast_via!(v, i16, i64)),
        Int32 => ConstValue::Int(cast_via!(v, i32, i64)),
        Int64 => ConstValue::Int(v as i64),
        UInt8 => ConstValue::UInt(cast_via!(v, u8, u64)),
        UInt16 => ConstValue::UInt(cast_via!(v, u16, u64)),
        UInt32 => ConstValue::UInt(cast_via!(v, u32, u64)),
        UInt64 => ConstValue::UInt(v),
        Float16 | BFloat16 | Float32 | Float64 => ConstValue::Float(v as f64),
        _ => return None,
    })
}

#[inline]
fn cast_float(v: f64, to: ScalarDType) -> Option<ConstValue> {
    use ScalarDType::*;
    Some(match to {
        Bool => ConstValue::Bool(v != 0.0),
        Int8 => ConstValue::Int(cast_via!(v, i8, i64)),
        Int16 => ConstValue::Int(cast_via!(v, i16, i64)),
        Int32 => ConstValue::Int(cast_via!(v, i32, i64)),
        Int64 => ConstValue::Int(v as i64),
        // Float-to-unsigned: route through i64 first (matches Tinygrad behavior)
        UInt8 => ConstValue::UInt(cast_via!(v as i64, u8, u64)),
        UInt16 => ConstValue::UInt(cast_via!(v as i64, u16, u64)),
        UInt32 => ConstValue::UInt(cast_via!(v as i64, u32, u64)),
        UInt64 => ConstValue::UInt((v as i64) as u64),
        Float16 | BFloat16 | Float32 | Float64 => ConstValue::Float(v),
        _ => return None,
    })
}

impl ConstValue {
    pub const fn dtype(&self) -> DType {
        match self {
            ConstValue::Int(_) => DType::Int64,
            ConstValue::UInt(_) => DType::UInt64,
            ConstValue::Float(_) => DType::Float64,
            ConstValue::Bool(_) => DType::Bool,
        }
    }

    pub const fn zero(dtype: ScalarDType) -> Self {
        use ScalarDType::*;
        match dtype {
            Bool => Self::Bool(false),
            Int8 | Int16 | Int32 | Int64 => Self::Int(0),
            UInt8 | UInt16 | UInt32 | UInt64 => Self::UInt(0),
            FP8E4M3 | FP8E5M2 | Float16 | BFloat16 | Float32 | Float64 => Self::Float(0.0),
            Void | Index => Self::Int(0), // TODO: remove this types from scalars
        }
    }

    pub const fn one(dtype: ScalarDType) -> Self {
        use ScalarDType::*;
        match dtype {
            Bool => Self::Bool(true),
            Int8 | Int16 | Int32 | Int64 => Self::Int(1),
            UInt8 | UInt16 | UInt32 | UInt64 => Self::UInt(1),
            FP8E4M3 | FP8E5M2 | Float16 | BFloat16 | Float32 | Float64 => Self::Float(1.0),
            Void | Index => Self::Int(1), // TODO: remove this types from scalars
        }
    }

    /// Cast this constant value to the target dtype.
    ///
    /// Returns `None` if:
    /// - The target dtype is not a scalar type
    /// - The target dtype is not representable as a ConstValue (e.g., Void, Index, special float formats)
    ///
    /// # Safety and Semantics
    ///
    /// This method performs constant folding for cast operations and allows ALL casts
    /// (including lossy ones like float->int) since the user explicitly wrote the cast operation.
    ///
    /// Uses Rust's `as` operator for conversions, which follows C semantics:
    /// - Truncation for narrowing conversions (e.g., i64 -> i32)
    /// - Wrap-around for unsigned overflow
    /// - Truncation toward zero for float-to-int conversions
    ///
    /// For multi-stage conversions (e.g., casting through intermediate types),
    /// the value is cast to the target width and then extended back to the storage type.
    /// Example: i64 -> i8 -> i64 ensures proper sign extension.
    pub fn cast(&self, dtype: &DType) -> Option<Self> {
        let scalar_dtype = dtype.scalar()?;

        Some(impl_cast!(*self, scalar_dtype))
    }
}

/// Memory address space for buffer allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddrSpace {
    /// Global/device memory.
    Global,
    /// Local/shared memory.
    Local,
    /// Register memory.
    Reg,
}

/// Options for BUFFERIZE operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferizeOpts {
    /// Device specification or None for local buffers.
    pub device: Option<DeviceSpec>,
    /// Address space (GLOBAL or LOCAL).
    pub addrspace: AddrSpace,
}

impl BufferizeOpts {
    pub fn new(device: DeviceSpec) -> Self {
        Self { device: Some(device), addrspace: AddrSpace::Global }
    }

    pub fn local() -> Self {
        Self { device: None, addrspace: AddrSpace::Local }
    }
}

/// Axis type for loop ranges and reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisType {
    /// Outer kernel-level scheduling dimension (doesn't go inside kernels).
    ///
    /// Used to mark ranges that exist at the scheduling/orchestration level
    /// but don't become part of kernel execution. These ranges are used during
    /// kernel splitting to identify boundaries.
    Outer,
    /// GPU grid dimension.
    Global,
    /// Warp/wavefront dimension.
    Warp,
    /// GPU block/workgroup dimension (local memory scope).
    Local,
    /// Regular loop.
    Loop,
    /// Grouped reduction.
    GroupReduce,
    /// Reduction axis.
    Reduce,
    /// Vectorization axis (upcast).
    Upcast,
    /// Unrolled loop.
    Unroll,
    /// Thread dimension.
    Thread,
}

impl AxisType {
    /// Returns true if this axis type represents a kernel boundary.
    ///
    /// Kernel boundary ranges (Outer) exist at the scheduling level and
    /// don't go inside individual kernels. During kernel splitting, operations
    /// with outer ranges are skipped from being packaged into KERNEL ops.
    pub const fn is_kernel_boundary(&self) -> bool {
        matches!(self, Self::Outer)
    }
}

/// Reduction operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Sum reduction (a + b).
    Add,
    /// Product reduction (a * b).
    Mul,
    /// Maximum reduction (max(a, b)).
    Max,
}

/// Unary operation types.
///
/// All unary operations preserve the input dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Negation: -x
    Neg,
    /// Square root: √x
    Sqrt,
    /// Base-2 exponential: 2^x
    Exp2,
    /// Base-2 logarithm: log₂(x)
    Log2,
    /// Sine: sin(x) (float only)
    Sin,
    /// Reciprocal: 1/x
    Reciprocal,
    /// Truncate towards zero (remove fractional part)
    Trunc,
}

/// Binary operation types.
///
/// Arithmetic operations (Add, Mul, Sub, Mod, Max, Pow, Idiv, Fdiv) preserve the LHS dtype.
/// Comparison operations (Lt, Eq, Ne) always return DType::Bool.
/// Bitwise operations (And, Or, Xor, Shl, Shr) preserve dtype and require int/bool types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    // Arithmetic operations
    /// Addition: a + b
    Add,
    /// Multiplication: a * b
    Mul,
    /// Subtraction: a - b
    Sub,
    /// Modulo: a % b (C-style remainder)
    ///
    /// Uses C/Rust semantics where result has the sign of the dividend (first operand).
    /// This matches Tinygrad's MOD and C's % operator.
    ///
    /// **NOT** Python's modulo operator (which has sign of the divisor).
    ///
    /// Examples: -9 % 5 = -4 (Python gives 1), 9 % -5 = 4 (Python gives -1)
    Mod,
    /// Maximum: max(a, b)
    Max,
    /// Power: a^b
    Pow,
    /// Integer division: a / b (truncated toward zero)
    ///
    /// Uses C-style truncation, NOT floor division.
    /// This matches Tinygrad's IDIV and C's / operator for integers.
    ///
    /// **NOT** Python's // floor division (which rounds toward -∞).
    ///
    /// Examples: -9 / 5 = -1 (Python's // gives -2), 9 / -5 = -1 (Python's // gives -2)
    Idiv,
    /// Float division: a / b (exact IEEE 754 division)
    ///
    /// Only used for float dtypes. Performs exact floating-point division.
    /// Matches Tinygrad's FDIV.
    Fdiv,

    // Comparison operations
    /// Less than: a < b
    Lt,
    /// Equality: a == b
    Eq,
    /// Inequality: a != b
    Ne,

    // Bitwise operations (int/bool only)
    /// Bitwise AND: a & b
    And,
    /// Bitwise OR: a | b
    Or,
    /// Bitwise XOR: a ^ b
    Xor,
    /// Left shift: a << b
    Shl,
    /// Right shift: a >> b
    Shr,

    // Special operations
    /// Threefry PRNG: threefry(x, key) -> uint64
    Threefry,
}

impl BinaryOp {
    /// Returns true if this is a comparison operation.
    pub fn is_comparison(self) -> bool {
        matches!(self, Self::Lt | Self::Eq | Self::Ne)
    }

    /// Returns true if this is an arithmetic operation.
    pub fn is_arithmetic(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Sub | Self::Mod | Self::Max | Self::Pow | Self::Idiv | Self::Fdiv)
    }

    /// Returns true if this is a bitwise operation.
    pub fn is_bitwise(self) -> bool {
        matches!(self, Self::And | Self::Or | Self::Xor | Self::Shl | Self::Shr)
    }

    /// Returns true if this operation is associative.
    pub fn is_associative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::And | Self::Or | Self::Max)
    }

    /// Returns true if this operation is commutative.
    pub fn is_commutative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Eq | Self::Ne | Self::And | Self::Or | Self::Xor | Self::Max)
    }

    /// Returns true if this operation is idempotent (f(x, x) = x).
    pub fn is_idempotent(self) -> bool {
        matches!(self, Self::Or | Self::And | Self::Max)
    }
}

/// Ternary operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TernaryOp {
    /// Conditional selection: condition ? true_val : false_val
    Where,
    /// Multiply-accumulate: a * b + c (fused operation)
    MulAcc,
}

/// Metadata for WMMA (Warp Matrix Multiply-Accumulate) operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WmmaMetadata {
    /// Operation name (e.g., "WMMA_INSTRUCTION").
    pub name: String,
    /// Matrix dimensions (N, M, K).
    pub dims: (usize, usize, usize),
    /// Input matrix dtype.
    pub dtype_in: DType,
    /// Output/accumulator dtype.
    pub dtype_out: DType,
    /// Target device string.
    pub device: String,
    /// Thread count.
    pub threads: usize,
    /// Upcast axes for vectorization.
    pub upcast_axes: Vec<(usize, usize)>,
    /// Reduction axes.
    pub reduce_axes: Vec<(usize, usize)>,
}

/// Wrapper for ConstValue that implements Eq and Hash.
///
/// Floats don't implement Eq/Hash due to IEEE 754 NaN semantics (NaN != NaN).
/// This wrapper uses bitwise comparison: two floats are equal if their bit patterns match.
/// This means:
/// - NaN values with identical bit patterns are considered equal
/// - Different NaN representations are not equal
/// - This is consistent with hash consing requirements
#[derive(Debug, Clone, Copy)]
pub struct ConstValueHash(pub ConstValue);

impl PartialEq for ConstValueHash {
    fn eq(&self, other: &Self) -> bool {
        match (self.0, other.0) {
            (ConstValue::Int(a), ConstValue::Int(b)) => a == b,
            (ConstValue::UInt(a), ConstValue::UInt(b)) => a == b,
            (ConstValue::Float(a), ConstValue::Float(b)) => a.to_bits() == b.to_bits(),
            (ConstValue::Bool(a), ConstValue::Bool(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for ConstValueHash {}

impl Hash for ConstValueHash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (discriminant(&self.0)).hash(state);
        match self.0 {
            ConstValue::Int(v) => v.hash(state),
            ConstValue::UInt(v) => v.hash(state),
            ConstValue::Float(v) => v.to_bits().hash(state),
            ConstValue::Bool(v) => v.hash(state),
        }
    }
}
