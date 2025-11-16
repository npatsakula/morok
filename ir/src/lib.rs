use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::mem::discriminant;
use std::rc::{Rc, Weak};

use smallvec::SmallVec;
use snafu::ensure;

#[cfg(test)]
pub mod test;

use morok_device::DeviceSpec;
use morok_dtype::DType;

pub mod error;
pub mod ops;
pub mod shape;
pub mod sint;

pub use error::{Error, IndexTypeMismatchSnafu, Result};
pub use sint::{SInt, sint_max, sint_min, sint_prod};

// Thread-local counter for unique identifiers.
//
// Design choice: Uses Cell<usize> for single-threaded efficiency.
// - Cell is !Send + !Sync, preventing accidental multi-threading
// - Zero overhead compared to atomics (no memory barriers)
// - Matches Tinygrad's single-threaded execution model
thread_local! {
    static UNIQUE_COUNTER: Cell<usize> = const { Cell::new(0) };
}

fn next_unique_id() -> usize {
    UNIQUE_COUNTER.with(|counter| {
        let id = counter.get();
        counter.set(id + 1);
        id
    })
}

// Thread-local counter for UOp stable IDs.
//
// Provides monotonic IDs that never repeat, eliminating ABA problem.
// Uses u64 to provide 2^64 unique IDs (effectively unlimited).
thread_local! {
    static UOP_ID_COUNTER: Cell<u64> = const { Cell::new(0) };
}

fn next_uop_id() -> u64 {
    UOP_ID_COUNTER.with(|counter| {
        let id = counter.get();
        counter.set(id + 1);
        id
    })
}

/// Constant value that can be stored in a UOp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstValue {
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
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

/// Operation type with typed operands.
///
/// Each operation encodes its operand structure directly in the enum variant.
/// This provides compile-time verification of operand count and types.
///
/// Design choices:
/// - Fixed-arity ops grouped by arity: Unary, Binary, Ternary
/// - Special ops with extra data remain separate: Cast (dtype), MSelect (device_index)
/// - Variable-arity ops use SmallVec: Index { indices: SmallVec<[Rc<UOp>; 4]> }
/// - SmallVec avoids heap allocation for common cases (≤4 children)
/// - Gated operations use separate variants (LoadGated vs Load) for type safety
///
/// Note: PartialEq, Eq, and Hash are NOT derived because Op contains Rc<UOp>.
/// Hash consing uses UOpKey which compares by pointer equality instead.
#[derive(Debug, Clone)]
pub enum Op {
    // Nullary operations (6 variants)
    Const(ConstValueHash),
    Unique(usize),
    Device(DeviceSpec),
    Noop,
    DefineGlobal(usize),
    DefineLocal(usize),

    // Graph organization operations (2 variants)
    Sink {
        sources: SmallVec<[Rc<UOp>; 4]>,
    },
    Group {
        sources: SmallVec<[Rc<UOp>; 4]>,
    },

    // Grouped operations (3 variants)
    Unary(UnaryOp, Rc<UOp>),
    Binary(BinaryOp, Rc<UOp>, Rc<UOp>),
    Ternary(TernaryOp, Rc<UOp>, Rc<UOp>, Rc<UOp>),

    // Type operations (2 variants)
    Cast {
        src: Rc<UOp>,
        dtype: DType,
    },
    BitCast {
        src: Rc<UOp>,
        dtype: DType,
    },

    // Special operations (2 variants)
    MSelect {
        buffer: Rc<UOp>,
        device_index: usize,
    },
    Special {
        end: Rc<UOp>,
        name: String,
    },

    // Buffer operations (high-level, 7 variants)
    Buffer {
        unique: Rc<UOp>,
        device: Rc<UOp>,
        size: usize,
    },
    BufferView {
        buffer: Rc<UOp>,
        size: usize,
        offset: usize,
    },
    Bufferize {
        compute: Rc<UOp>,
        ranges: SmallVec<[Rc<UOp>; 4]>,
        opts: BufferizeOpts,
    },
    Index {
        buffer: Rc<UOp>,
        indices: SmallVec<[Rc<UOp>; 4]>,
        gate: Option<Rc<UOp>>,
    },
    PointerIndex {
        ptr: Rc<UOp>,
        offset: Rc<UOp>,
    },
    Copy {
        src: Rc<UOp>,
        device: Rc<UOp>,
    },
    MStack {
        buffers: SmallVec<[Rc<UOp>; 4]>,
    },

    // Movement/Reshape operations (7 variants)
    Reshape {
        src: Rc<UOp>,
        new_shape: Rc<UOp>,
    },
    Permute {
        src: Rc<UOp>,
        axes: Vec<usize>,
    },
    Expand {
        src: Rc<UOp>,
        new_shape: Rc<UOp>,
    },
    Pad {
        src: Rc<UOp>,
        begin_pads: Rc<UOp>,
        end_pads: Rc<UOp>,
    },
    Shrink {
        src: Rc<UOp>,
        begins: Rc<UOp>,
        ends: Rc<UOp>,
    },
    Flip {
        src: Rc<UOp>,
        axes: Vec<bool>,
    },
    Multi {
        src: Rc<UOp>,
        axis: usize,
    },

    // Reduction operations (3 variants)
    ReduceAxis {
        src: Rc<UOp>,
        reduce_op: ReduceOp,
        axes: Vec<usize>,
    },
    Reduce {
        src: Rc<UOp>,
        ranges: SmallVec<[Rc<UOp>; 4]>,
        reduce_op: ReduceOp,
    },
    AllReduce {
        src: Rc<UOp>,
        device: Rc<UOp>,
        reduce_op: ReduceOp,
    },

    // Control flow operations (5 variants)
    If {
        condition: Rc<UOp>,
        body: SmallVec<[Rc<UOp>; 4]>,
    },
    EndIf {
        if_op: Rc<UOp>,
    },
    Range {
        end: Rc<UOp>,
        axis_id: usize,
        axis_type: AxisType,
    },
    End {
        range_or_reduce: Rc<UOp>,
    },
    Barrier {
        src: Rc<UOp>,
        deps: SmallVec<[Rc<UOp>; 4]>,
    },

    // Vector operations (5 variants)
    Vectorize {
        elements: SmallVec<[Rc<UOp>; 4]>,
    },
    Gep {
        vector: Rc<UOp>,
        indices: Vec<usize>,
    },
    VConst {
        values: Vec<ConstValue>,
    },
    /// Concatenate vectors into larger vector (expander op).
    /// Like VECTORIZE but sources can be vectors themselves.
    /// Output vcount = sum of all input vcounts.
    Cat {
        sources: SmallVec<[Rc<UOp>; 4]>,
    },
    /// Concatenate pointers into vectorized pointer (expander op).
    /// Used for grouping memory accesses in devectorizer.
    PtrCat {
        sources: SmallVec<[Rc<UOp>; 4]>,
    },

    // Symbolic/Define operations (3 variants)
    DefineVar {
        name: String,
        min_val: i64,
        max_val: i64,
    },
    Bind {
        var: Rc<UOp>,
        value: Rc<UOp>,
    },
    DefineReg {
        size: usize,
    },

    // Advanced operations (12 variants)
    Wmma {
        a: Rc<UOp>,
        b: Rc<UOp>,
        c: Rc<UOp>,
        metadata: WmmaMetadata,
    },
    Contract {
        src: Rc<UOp>,
        upcast_ranges: Vec<(usize, usize)>,
    },
    Unroll {
        src: Rc<UOp>,
        unroll_axes: Vec<(usize, usize)>,
    },
    Kernel {
        ast: Option<Rc<UOp>>,
    },
    Assign {
        target: Rc<UOp>,
        value: Rc<UOp>,
    },
    Detach {
        src: Rc<UOp>,
    },
    Contiguous {
        src: Rc<UOp>,
    },
    ContiguousBackward {
        src: Rc<UOp>,
    },
    After {
        passthrough: Rc<UOp>,
        deps: SmallVec<[Rc<UOp>; 4]>,
    },
    Precast {
        src: Rc<UOp>,
    },
    Custom {
        deps: SmallVec<[Rc<UOp>; 4]>,
        code: String,
    },
    CustomI {
        deps: SmallVec<[Rc<UOp>; 4]>,
        code: String,
    },

    // Memory operations (low-level, after kernel splitting, 4 variants)
    Load {
        buffer: Rc<UOp>,
        index: Rc<UOp>,
    },
    LoadGated {
        buffer: Rc<UOp>,
        index: Rc<UOp>,
        gate: Rc<UOp>,
    },
    Store {
        buffer: Rc<UOp>,
        index: Rc<UOp>,
        value: Rc<UOp>,
    },
    StoreGated {
        buffer: Rc<UOp>,
        index: Rc<UOp>,
        value: Rc<UOp>,
        gate: Rc<UOp>,
    },
}

impl Op {
    /// Get all child UOps as a Vec of references.
    ///
    /// This is the convenient API for traversing the graph.
    /// Allocates a Vec but is simple to use.
    pub fn children(&self) -> SmallVec<[&Rc<UOp>; 4]> {
        match self {
            // Nullary operations
            Self::Const(_)
            | Self::Unique(_)
            | Self::Device(_)
            | Self::Noop
            | Self::DefineGlobal(_)
            | Self::DefineLocal(_)
            | Self::VConst { .. }
            | Self::DefineVar { .. }
            | Self::DefineReg { .. } => SmallVec::new(),

            // Graph organization operations
            Self::Sink { sources } | Self::Group { sources } => sources.iter().collect(),

            // Grouped operations
            Self::Unary(_, x) => SmallVec::from_slice(&[x]),
            Self::Binary(_, a, b) => SmallVec::from_slice(&[a, b]),
            Self::Ternary(_, a, b, c) => SmallVec::from_slice(&[a, b, c]),

            // Type operations
            Self::Cast { src, .. } | Self::BitCast { src, .. } => SmallVec::from_slice(&[src]),

            // Special operations
            Self::MSelect { buffer, .. } => SmallVec::from_slice(&[buffer]),
            Self::Special { end, .. } => SmallVec::from_slice(&[end]),

            // Buffer operations
            Self::Buffer { unique, device, .. } => SmallVec::from_slice(&[unique, device]),
            Self::BufferView { buffer, .. } => SmallVec::from_slice(&[buffer]),
            Self::Bufferize { compute, ranges, .. } => {
                let mut children = SmallVec::from_slice(&[compute]);
                children.extend(ranges.iter());
                children
            }
            Self::Index { buffer, indices, gate } => {
                let mut children = SmallVec::from_slice(&[buffer]);
                children.extend(indices.iter());
                children.extend(gate);
                children
            }
            Self::PointerIndex { ptr, offset } => SmallVec::from_slice(&[ptr, offset]),
            Self::Copy { src, device } => SmallVec::from_slice(&[src, device]),
            Self::MStack { buffers } => buffers.iter().collect(),

            // Movement operations
            Self::Reshape { src, new_shape } => SmallVec::from_slice(&[src, new_shape]),
            Self::Permute { src, .. } | Self::Flip { src, .. } | Self::Multi { src, .. } => {
                SmallVec::from_slice(&[src])
            }
            Self::Expand { src, new_shape } => SmallVec::from_slice(&[src, new_shape]),
            Self::Pad { src, begin_pads, end_pads } => SmallVec::from_slice(&[src, begin_pads, end_pads]),
            Self::Shrink { src, begins, ends } => SmallVec::from_slice(&[src, begins, ends]),

            // Reduction operations
            Self::ReduceAxis { src, .. } => SmallVec::from_slice(&[src]),
            Self::Reduce { src, ranges, .. } => {
                let mut children = SmallVec::from_slice(&[src]);
                children.extend(ranges.iter());
                children
            }
            Self::AllReduce { src, device, .. } => SmallVec::from_slice(&[src, device]),

            // Control flow operations
            Self::If { condition, body } => {
                let mut children = SmallVec::from_slice(&[condition]);
                children.extend(body.iter());
                children
            }
            Self::EndIf { if_op } => SmallVec::from_slice(&[if_op]),
            Self::Range { end, .. } => SmallVec::from_slice(&[end]),
            Self::End { range_or_reduce } => SmallVec::from_slice(&[range_or_reduce]),
            Self::Barrier { src, deps } => {
                let mut children = SmallVec::from_slice(&[src]);
                children.extend(deps.iter());
                children
            }

            // Vector operations
            Self::Vectorize { elements } => elements.iter().collect(),
            Self::Gep { vector, .. } => SmallVec::from_slice(&[vector]),
            Self::Cat { sources } | Self::PtrCat { sources } => sources.iter().collect(),

            // Symbolic/Define operations
            Self::Bind { var, value } => SmallVec::from_slice(&[var, value]),

            // Advanced operations
            Self::Wmma { a, b, c, .. } => SmallVec::from_slice(&[a, b, c]),
            Self::Contract { src, .. }
            | Self::Unroll { src, .. }
            | Self::Detach { src }
            | Self::Contiguous { src }
            | Self::ContiguousBackward { src }
            | Self::Precast { src } => SmallVec::from_slice(&[src]),
            Self::Kernel { ast } => {
                let mut children = SmallVec::new();
                children.extend(ast);
                children
            }
            Self::Assign { target, value } => SmallVec::from_slice(&[target, value]),
            Self::After { passthrough, deps } => {
                let mut children = SmallVec::from_slice(&[passthrough]);
                children.extend(deps.iter());
                children
            }
            Self::Custom { deps, .. } | Self::CustomI { deps, .. } => deps.iter().collect(),

            // Memory operations
            Self::Load { buffer, index } => SmallVec::from_slice(&[buffer, index]),
            Self::LoadGated { buffer, index, gate } => SmallVec::from_slice(&[buffer, index, gate]),
            Self::Store { buffer, index, value } => SmallVec::from_slice(&[buffer, index, value]),
            Self::StoreGated { buffer, index, value, gate } => SmallVec::from_slice(&[buffer, index, value, gate]),
        }
    }

    /// Get child UOp stable IDs for hash consing.
    ///
    /// Returns SmallVec of IDs, optimized for common case of ≤4 children (inline storage).
    fn src_ids(&self) -> SmallVec<[u64; 4]> {
        self.children().into_iter().map(|child| child.id).collect()
    }

    /// Apply a function to each child UOp.
    pub fn map_child<F>(&self, mut f: F)
    where
        F: FnMut(&Rc<UOp>),
    {
        for child in self.children() {
            f(child);
        }
    }
}

/// Micro-operation node in the computation graph.
///
/// UOps form a DAG where operations reference their inputs through the Op enum.
/// Hash consing ensures that structurally identical UOps share the same allocation.
///
/// Shape inference is lazy and cached - computed on first access via `shape()` method.
#[derive(Debug)]
pub struct UOp {
    /// Unique stable ID for this UOp instance.
    /// Used for identity-based caching instead of fragile raw pointers.
    id: u64,
    op: Op,
    dtype: DType,
    /// Cached shape - computed lazily on first access.
    /// OnceCell provides thread-safe lazy initialization.
    shape_cache: std::cell::OnceCell<Option<shape::Shape>>,
}

/// Cache key for hash consing.
///
/// Uses stable UOp IDs for child UOps to avoid infinite recursion during hashing.
/// IDs are monotonic and never reused, eliminating ABA problem from pointer-based approach.
#[derive(Eq, PartialEq, Hash)]
struct UOpKey {
    op_discriminant: std::mem::Discriminant<Op>,
    dtype: DType,
    src_ids: SmallVec<[u64; 4]>,
    // Store additional data that's not in src_ids
    op_data: OpData,
}

/// Non-recursive data from Op variants for hashing.
///
/// Captures operation-specific data that std::mem::discriminant doesn't include.
/// This is critical for hash consing correctness - without this, operations like
/// Add and Mul would be treated as identical since they have the same discriminant.
#[derive(Eq, PartialEq, Hash, Clone)]
enum OpData {
    // Nullary operations
    Const(ConstValueHash),
    Unique(usize),
    Device(DeviceSpec),
    DefineGlobal(usize),
    DefineLocal(usize),

    // Grouped operations
    Unary(UnaryOp),
    Binary(BinaryOp),
    Ternary(TernaryOp),

    // Type operations
    CastDType(DType),
    BitCastDType(DType),

    // Special operations
    MSelectIdx(usize),
    SpecialName(String),

    // Buffer operations
    BufferSize(usize),
    BufferView(usize, usize),
    Bufferize(BufferizeOpts),

    // Movement/Reshape operations
    PermuteAxes(Vec<usize>),
    FlipAxes(Vec<bool>),
    MultiAxis(usize),

    // Reduction operations
    ReduceAxisData(ReduceOp, Vec<usize>),
    ReduceOp(ReduceOp),
    AllReduceOp(ReduceOp),

    // Control flow operations
    RangeData(usize, AxisType),

    // Vector operations
    GepIndices(Vec<usize>),
    VConstValues(Vec<ConstValueHash>),

    // Symbolic/Define operations
    DefineVarData(String, i64, i64),
    DefineRegSize(usize),

    // Advanced operations
    WmmaData(WmmaMetadata),
    ContractRanges(Vec<(usize, usize)>),
    UnrollAxes(Vec<(usize, usize)>),
    CustomCode(String),

    // Operations with only children (no extra semantic data)
    None,
}

impl UOpKey {
    fn new(op: &Op, dtype: DType) -> Self {
        let op_discriminant = discriminant(op);
        let src_ids = op.src_ids();

        let op_data = match op {
            // Nullary operations
            Op::Const(c) => OpData::Const(*c),
            Op::Unique(id) => OpData::Unique(*id),
            Op::Device(d) => OpData::Device(d.clone()),
            Op::DefineGlobal(slot) => OpData::DefineGlobal(*slot),
            Op::DefineLocal(slot) => OpData::DefineLocal(*slot),

            // Grouped operations
            Op::Unary(unary_op, _) => OpData::Unary(*unary_op),
            Op::Binary(binary_op, _, _) => OpData::Binary(*binary_op),
            Op::Ternary(ternary_op, _, _, _) => OpData::Ternary(*ternary_op),

            // Type operations
            Op::Cast { dtype, .. } => OpData::CastDType(dtype.clone()),
            Op::BitCast { dtype, .. } => OpData::BitCastDType(dtype.clone()),

            // Special operations
            Op::MSelect { device_index, .. } => OpData::MSelectIdx(*device_index),
            Op::Special { name, .. } => OpData::SpecialName(name.clone()),

            // Buffer operations
            Op::Buffer { size, .. } => OpData::BufferSize(*size),
            Op::BufferView { size, offset, .. } => OpData::BufferView(*size, *offset),
            Op::Bufferize { opts, .. } => OpData::Bufferize(opts.clone()),

            // Movement/Reshape operations
            Op::Permute { axes, .. } => OpData::PermuteAxes(axes.clone()),
            Op::Flip { axes, .. } => OpData::FlipAxes(axes.clone()),
            Op::Multi { axis, .. } => OpData::MultiAxis(*axis),

            // Reduction operations
            Op::ReduceAxis { reduce_op, axes, .. } => OpData::ReduceAxisData(*reduce_op, axes.clone()),
            Op::Reduce { reduce_op, .. } => OpData::ReduceOp(*reduce_op),
            Op::AllReduce { reduce_op, .. } => OpData::AllReduceOp(*reduce_op),

            // Control flow operations
            Op::Range { axis_id, axis_type, .. } => OpData::RangeData(*axis_id, *axis_type),

            // Vector operations
            Op::Gep { indices, .. } => OpData::GepIndices(indices.clone()),
            Op::VConst { values } => OpData::VConstValues(values.iter().map(|v| ConstValueHash(*v)).collect()),

            // Symbolic/Define operations
            Op::DefineVar { name, min_val, max_val } => OpData::DefineVarData(name.clone(), *min_val, *max_val),
            Op::DefineReg { size } => OpData::DefineRegSize(*size),

            // Advanced operations
            Op::Wmma { metadata, .. } => OpData::WmmaData(metadata.clone()),
            Op::Contract { upcast_ranges, .. } => OpData::ContractRanges(upcast_ranges.clone()),
            Op::Unroll { unroll_axes, .. } => OpData::UnrollAxes(unroll_axes.clone()),
            Op::Custom { code, .. } | Op::CustomI { code, .. } => OpData::CustomCode(code.clone()),

            // All other operations have no semantic data beyond children and discriminant
            _ => OpData::None,
        };

        Self { op_discriminant, dtype, src_ids, op_data }
    }
}

// Thread-local hash consing cache.
//
// Maps UOpKey to weak references. When all strong references are dropped,
// the entry is automatically cleaned up.
//
// Design choice: Uses Rc + thread_local instead of Arc + DashMap.
// - Rc is !Send + !Sync, preventing accidental threading bugs
// - No atomic refcounting overhead (faster cloning)
// - Thread-local cache avoids locking entirely
// - Matches Tinygrad's single-threaded model
// - Can switch to Arc + concurrent hashmap later if needed for parallelism
thread_local! {
    static CACHE: RefCell<HashMap<UOpKey, Weak<UOp>>> = RefCell::new(HashMap::new());
}

impl UOp {
    /// Create a new UOp with hash consing.
    ///
    /// If an identical UOp already exists, returns a reference to it.
    /// Otherwise, creates a new UOp and caches it.
    pub fn new(op: Op, dtype: DType) -> Rc<Self> {
        let key = UOpKey::new(&op, dtype.clone());

        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();

            // Check if we already have this UOp
            if let Some(weak) = cache.get(&key)
                && let Some(rc) = weak.upgrade()
            {
                return rc;
            }

            // Create new UOp with unique stable ID
            let uop = Rc::new(Self { id: next_uop_id(), op, dtype, shape_cache: std::cell::OnceCell::new() });

            // Cache it
            cache.insert(key, Rc::downgrade(&uop));

            uop
        })
    }

    /// Get the operation.
    pub fn op(&self) -> &Op {
        &self.op
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// Get the shape of this UOp.
    ///
    /// Shape is computed lazily on first access and cached.
    /// Returns None if shape cannot be determined (e.g., for control flow ops).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use morok_ir::{UOp, ConstValue};
    /// # use morok_dtype::DType;
    /// let scalar = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    /// assert_eq!(scalar.shape().map(|s| s.len()), Some(0)); // Scalar has empty shape
    /// ```
    pub fn shape(&self) -> Option<&shape::Shape> {
        self.shape_cache.get_or_init(|| shape::infer_shape_from_op(self)).as_ref()
    }

    // Construction helpers

    /// Create a constant UOp.
    pub fn const_(dtype: DType, value: ConstValue) -> Rc<Self> {
        Self::new(Op::Const(ConstValueHash(value)), dtype)
    }

    /// Create a unique identifier.
    pub fn unique(num: Option<usize>) -> Rc<Self> {
        let id = num.unwrap_or_else(next_unique_id);
        Self::new(Op::Unique(id), DType::Void)
    }

    /// Create a device specification.
    pub fn device(device: DeviceSpec) -> Rc<Self> {
        Self::new(Op::Device(device), DType::Void)
    }

    /// Create a no-op.
    pub fn noop() -> Rc<Self> {
        Self::new(Op::Noop, DType::Void)
    }

    /// Create a cast operation.
    pub fn cast(src: Rc<Self>, dtype: DType) -> Rc<Self> {
        Self::new(Op::Cast { src, dtype: dtype.clone() }, dtype)
    }

    // Macro-generated helpers for repetitive operations

    /// Create a new buffer.
    ///
    /// Equivalent to: `UOp(Ops.BUFFER, dtype, (unique(), device(device_spec)), size)`
    pub fn new_buffer(device: DeviceSpec, size: usize, dtype: DType) -> Rc<Self> {
        let unique = Self::unique(None);
        let dev = Self::device(device);
        Self::new(Op::Buffer { unique, device: dev, size }, dtype)
    }

    /// Create a buffer view.
    pub fn buffer_view(buffer: Rc<Self>, size: usize, offset: usize) -> Rc<Self> {
        let dtype = buffer.dtype.clone();
        Self::new(Op::BufferView { buffer, size, offset }, dtype)
    }

    /// Create an index operation.
    pub fn index(buffer: Rc<Self>, indices: Vec<Rc<Self>>) -> Result<Rc<Self>> {
        // Validate that all indices have Index dtype
        for idx in &indices {
            let idx_dtype = idx.dtype();
            ensure!(idx_dtype == DType::Index, IndexTypeMismatchSnafu { actual: idx_dtype });
        }

        let dtype = buffer.dtype.clone();
        let indices = SmallVec::from_vec(indices);
        Ok(Self::new(Op::Index { buffer, indices, gate: None }, dtype))
    }

    /// Create a gated index operation.
    pub fn index_gated(buffer: Rc<Self>, indices: Vec<Rc<Self>>, gate: Rc<Self>) -> Result<Rc<Self>> {
        // Validate that all indices have Index dtype
        for idx in &indices {
            let idx_dtype = idx.dtype();
            ensure!(idx_dtype == DType::Index, IndexTypeMismatchSnafu { actual: idx_dtype });
        }

        let dtype = buffer.dtype.clone();
        let indices = SmallVec::from_vec(indices);
        Ok(Self::new(Op::Index { buffer, indices, gate: Some(gate) }, dtype))
    }

    /// Copy to a different device.
    pub fn copy_to_device(self: &Rc<Self>, device: DeviceSpec) -> Rc<Self> {
        let dev = Self::device(device);
        Self::new(Op::Copy { src: self.clone(), device: dev }, self.dtype.clone())
    }

    /// Create a sink operation (graph termination).
    ///
    /// Sink marks outputs that must be evaluated. All sources are dependencies.
    pub fn sink(sources: Vec<Rc<Self>>) -> Rc<Self> {
        Self::new(Op::Sink { sources: SmallVec::from_vec(sources) }, DType::Void)
    }

    /// Create a group operation (merging/organizing related ops).
    ///
    /// Group is a NOOP that helps organize related operations together.
    /// It passes through the first source while ensuring all sources are dependencies.
    pub fn group(sources: Vec<Rc<Self>>) -> Rc<Self> {
        let dtype = if sources.is_empty() { DType::Void } else { sources[0].dtype.clone() };
        Self::new(Op::Group { sources: SmallVec::from_vec(sources) }, dtype)
    }

    /// Create a pointer index operation (pointer arithmetic).
    ///
    /// Performs pointer + offset arithmetic for address calculation in kernels.
    /// Both ptr and offset should have Index dtype.
    pub fn pointer_index(ptr: Rc<Self>, offset: Rc<Self>) -> Result<Rc<Self>> {
        let ptr_dtype = ptr.dtype();
        let offset_dtype = offset.dtype();
        ensure!(ptr_dtype == DType::Index, IndexTypeMismatchSnafu { actual: ptr_dtype });
        ensure!(offset_dtype == DType::Index, IndexTypeMismatchSnafu { actual: offset_dtype });
        Ok(Self::new(Op::PointerIndex { ptr, offset }, DType::Index))
    }

    /// Create a CAT operation (concatenate vectors).
    ///
    /// Combines multiple scalar or vector values into a single larger vector.
    /// This is an expander-level operation used during kernel optimization.
    ///
    /// Like VECTORIZE but sources can be vectors themselves.
    /// Output dtype vcount = sum of all input vcounts.
    ///
    /// Example: CAT(vec4, vec2) → vec6
    ///
    /// Note: This operation should be lowered before final codegen.
    pub fn cat(sources: Vec<Rc<Self>>) -> Rc<Self> {
        assert!(!sources.is_empty(), "CAT requires at least one source");
        let dtype = sources[0].dtype.clone();
        Self::new(Op::Cat { sources: SmallVec::from_vec(sources) }, dtype)
    }

    /// Create a PTRCAT operation (concatenate pointers).
    ///
    /// Combines multiple pointer indices into a vectorized pointer.
    /// This is an expander-level operation used in devectorizer for grouping memory accesses.
    ///
    /// Output dtype vcount = sum of all source base counts.
    ///
    /// Note: This operation should be lowered to multiple LOAD/STORE before final codegen.
    pub fn ptrcat(sources: Vec<Rc<Self>>) -> Rc<Self> {
        assert!(!sources.is_empty(), "PTRCAT requires at least one source");
        let dtype = sources[0].dtype.clone();
        Self::new(Op::PtrCat { sources: SmallVec::from_vec(sources) }, dtype)
    }

    /// Topological sort of the computation graph.
    ///
    /// Returns nodes in an order where all dependencies come before their dependents.
    pub fn toposort(self: &Rc<Self>) -> Vec<Rc<Self>> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut stack = vec![(self.clone(), false)];

        while let Some((node, processed)) = stack.pop() {
            let ptr = Rc::as_ptr(&node);

            if visited.contains(&ptr) {
                continue;
            }

            if processed {
                visited.insert(ptr);
                result.push(node);
            } else {
                stack.push((node.clone(), true));

                // Use for_each_child for zero-allocation traversal
                let mut children = Vec::new();
                node.op.map_child(|child| {
                    if !visited.contains(&Rc::as_ptr(child)) {
                        children.push(child.clone());
                    }
                });

                // Push in reverse order for proper traversal
                for child in children.into_iter().rev() {
                    stack.push((child, false));
                }
            }
        }

        result
    }
}

// Macro-generated helper methods for arithmetic operations
macro_rules! unary_ops {
    ($($name:ident => $op:ident),* $(,)?) => {
        impl UOp {
            $(
                pub fn $name(arg: &Rc<Self>) -> Rc<Self> {
                    let dtype = arg.dtype.clone();
                    Self::new(Op::Unary(UnaryOp::$op, arg.clone()), dtype)
                }
            )*
        }
    }
}

macro_rules! cmp_ops {
    ($($name:ident => $op:ident),* $(,)?) => {
        impl UOp {
            $(
                pub fn $name(lhs: &Rc<Self>, rhs: &Rc<Self>) -> Result<Rc<Self>> {
                    // Use type promotion to validate types and find common type
                    let (lhs, rhs, _) = Self::promote_and_cast(lhs.clone(), rhs.clone())?;
                    Ok(Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), DType::Bool))
                }
            )*
        }
    }
}

// Macro for transcendental functions that require float dtype
macro_rules! transcendental_ops {
    ($($name:ident => $op:ident),* $(,)?) => {
        impl UOp {
            $(
                pub fn $name(arg: &Rc<Self>) -> Result<Rc<Self>> {
                    let dtype = arg.dtype();
                    if !dtype.is_float() {
                        return Err(Error::InvalidDTypeForOp {
                            operation: stringify!($name),
                            dtype: dtype.clone()
                        });
                    }
                    Ok(Self::new(Op::Unary(UnaryOp::$op, arg.clone()), dtype))
                }
            )*
        }
    }
}

// Negation works on any numeric type
unary_ops! {
    neg => Neg,
}

// Transcendental functions require float types
transcendental_ops! {
    sqrt => Sqrt,
    exp2 => Exp2,
    log2 => Log2,
}

cmp_ops! {
    cmplt => Lt,
    cmpeq => Eq,
    cmpne => Ne,
}

impl Clone for UOp {
    fn clone(&self) -> Self {
        Self { id: self.id, op: self.op.clone(), dtype: self.dtype.clone(), shape_cache: std::cell::OnceCell::new() }
    }
}

/// Trait for converting scalar values into UOps.
///
/// This allows operator overloading to work with mixed scalar/UOp operands.
/// For example: `uop + 5.0` or `5.0 + uop`.
pub trait IntoUOp {
    fn into_uop(self, dtype: DType) -> Rc<UOp>;
}

impl IntoUOp for f32 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Float(self as f64))
    }
}

impl IntoUOp for f64 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Float(self))
    }
}

impl IntoUOp for i32 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Int(self as i64))
    }
}

impl IntoUOp for i64 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Int(self))
    }
}

impl IntoUOp for u32 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::UInt(self as u64))
    }
}

impl IntoUOp for u64 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::UInt(self))
    }
}

impl IntoUOp for bool {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Bool(self))
    }
}

/// Index specification for multi-dimensional slicing.
///
/// Similar to NumPy/ndarray indexing:
/// - `Single(idx)`: Select single element (like `arr[5]`)
/// - `Range{start, end, step}`: Slice range (like `arr[0:10:2]`)
/// - `Full`: Select all elements (like `arr[:]`)
/// - `NewAxis`: Add new dimension (like `arr[np.newaxis]`)
///
/// # Example
/// ```ignore
/// use morok_ir::{s, IndexSpec, UOp};
///
/// // Using macro syntax
/// let specs = vec![
///     s![idx],              // Single index
///     s![..],               // Full slice
///     s![start, end],       // Range
///     s![start, end, step], // Range with step
///     s![NewAxis],          // New axis
/// ];
/// ```
#[derive(Debug, Clone)]
pub enum IndexSpec {
    /// Single integer index - selects one element and removes dimension.
    Single(Rc<UOp>),

    /// Range with optional step - selects multiple elements.
    Range { start: Rc<UOp>, end: Rc<UOp>, step: Option<Rc<UOp>> },

    /// Full slice - selects all elements along this dimension.
    Full,

    /// New axis - adds a dimension of size 1.
    NewAxis,
}

/// Slice macro for creating IndexSpec instances.
///
/// Similar to ndarray's `s![]` macro, provides syntactic sugar for slicing.
///
/// # Syntax
/// - `s![idx]` → `IndexSpec::Single(idx)`
/// - `s![..]` → `IndexSpec::Full`
/// - `s![start, end]` → `IndexSpec::Range{start, end, step: None}`
/// - `s![start, end, step]` → `IndexSpec::Range{start, end, step: Some(step)}`
/// - `s![NewAxis]` → `IndexSpec::NewAxis`
///
/// # Example
/// ```ignore
/// let buf = UOp::new_buffer(DeviceSpec::Cpu, 1000, DType::Float32);
/// let idx = UOp::const_(DType::Int32, ConstValue::Int(5));
/// let start = UOp::const_(DType::Int32, ConstValue::Int(0));
/// let end = UOp::const_(DType::Int32, ConstValue::Int(10));
///
/// let slice = UOp::slice(buf, vec![
///     s![start, end],  // Range 0..10
///     s![idx],         // Single index at 5
///     s![..],          // Full slice
/// ]);
/// ```
#[macro_export]
macro_rules! s {
    // Full slice: s![..]
    (..) => {
        $crate::IndexSpec::Full
    };

    // Single index: s![idx]
    ($idx:expr) => {
        $crate::IndexSpec::Single($idx)
    };

    // Range without step: s![start, end]
    ($start:expr, $end:expr) => {
        $crate::IndexSpec::Range { start: $start, end: $end, step: None }
    };

    // Range with step: s![start, end, step]
    ($start:expr, $end:expr, $step:expr) => {
        $crate::IndexSpec::Range { start: $start, end: $end, step: Some($step) }
    };

    // NewAxis: s![NewAxis]
    (NewAxis) => {
        $crate::IndexSpec::NewAxis
    };
}

impl UOp {
    /// Multi-dimensional slicing with IndexSpec.
    ///
    /// # Example
    /// ```ignore
    /// let buf = UOp::new_buffer(DeviceSpec::Cpu, 1000, DType::Float32);
    /// let start = UOp::const_(DType::Int32, ConstValue::Int(0));
    /// let end = UOp::const_(DType::Int32, ConstValue::Int(10));
    ///
    /// // Slice first 10 elements
    /// let slice = UOp::slice(buf, vec![
    ///     IndexSpec::Range { start, end, step: None }
    /// ]);
    /// ```
    pub fn slice(buffer: Rc<Self>, specs: Vec<IndexSpec>) -> Result<Rc<Self>> {
        let mut indices = Vec::new();

        for spec in specs {
            match spec {
                IndexSpec::Single(idx) => {
                    // Single index - just use it directly
                    indices.push(idx);
                }
                IndexSpec::Range { start, end: _, step: _ } => {
                    // Range indexing - for now, just use start as a simple index
                    // TODO: Proper range expansion requires loop IR and range operations
                    indices.push(start);
                }
                IndexSpec::Full => {
                    // Full slice - skip (means "all elements")
                    // TODO: Proper handling requires understanding dimension size
                }
                IndexSpec::NewAxis => {
                    // NewAxis - adds dimension
                    // TODO: Requires reshape operation
                }
            }
        }

        if indices.is_empty() {
            // No actual indexing, just return buffer
            Ok(buffer)
        } else {
            Self::index(buffer, indices)
        }
    }

    /// Gated slicing - conditional access with gate.
    ///
    /// Similar to `slice` but with a boolean gate for conditional indexing.
    pub fn slice_gated(buffer: Rc<Self>, specs: Vec<IndexSpec>, gate: Rc<Self>) -> Result<Rc<Self>> {
        let mut indices = Vec::new();

        for spec in specs {
            match spec {
                IndexSpec::Single(idx) => indices.push(idx),
                IndexSpec::Range { start, .. } => indices.push(start),
                IndexSpec::Full | IndexSpec::NewAxis => {}
            }
        }

        if indices.is_empty() { Ok(buffer) } else { Self::index_gated(buffer, indices, gate) }
    }
}
