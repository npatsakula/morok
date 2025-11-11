use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::mem::discriminant;
use std::rc::{Rc, Weak};

use smallvec::SmallVec;

use morok_device::DeviceSpec;
use morok_dtype::DType;

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

/// Unary operation types.
///
/// All unary operations preserve the input dtype.
/// Not requires int/bool dtype.
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
    /// Bitwise NOT: ~x (int/bool only)
    Not,
}

/// Binary operation types.
///
/// Arithmetic operations (Add, Mul, Sub, Div, Rem) preserve the LHS dtype.
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
    /// Division: a / b
    Div,
    /// Remainder/Modulo: a % b
    Rem,

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
}

impl BinaryOp {
    /// Returns true if this is a comparison operation.
    pub fn is_comparison(self) -> bool {
        matches!(self, Self::Lt | Self::Eq | Self::Ne)
    }

    /// Returns true if this is an arithmetic operation.
    pub fn is_arithmetic(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Sub | Self::Div | Self::Rem)
    }

    /// Returns true if this is a bitwise operation.
    pub fn is_bitwise(self) -> bool {
        matches!(self, Self::And | Self::Or | Self::Xor | Self::Shl | Self::Shr)
    }

    /// Returns true if this operation is associative.
    pub fn is_associative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::And | Self::Or | Self::Xor)
    }

    /// Returns true if this operation is commutative.
    pub fn is_commutative(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::Eq | Self::Ne | Self::And | Self::Or | Self::Xor
        )
    }
}

/// Ternary operation types.
///
/// Currently unused, but reserved for future WHERE operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TernaryOp {
    /// Conditional selection: condition ? true_val : false_val
    Where,
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

    // Grouped operations (2 variants, down from 11)
    Unary(UnaryOp, Rc<UOp>),
    Binary(BinaryOp, Rc<UOp>, Rc<UOp>),

    // Special unary operations with extra data (2 variants)
    Cast { src: Rc<UOp>, dtype: DType },
    MSelect { buffer: Rc<UOp>, device_index: usize },

    // Buffer operations (high-level, 6 variants)
    Buffer { unique: Rc<UOp>, device: Rc<UOp>, size: usize },
    BufferView { buffer: Rc<UOp>, size: usize, offset: usize },
    Bufferize { compute: Rc<UOp>, ranges: SmallVec<[Rc<UOp>; 4]>, opts: BufferizeOpts },
    Index { buffer: Rc<UOp>, indices: SmallVec<[Rc<UOp>; 4]>, gate: Option<Rc<UOp>> },
    Copy { src: Rc<UOp>, device: Rc<UOp> },
    MStack { buffers: SmallVec<[Rc<UOp>; 4]> },

    // Memory operations (low-level, after kernel splitting, 4 variants)
    Load { buffer: Rc<UOp>, index: Rc<UOp> },
    LoadGated { buffer: Rc<UOp>, index: Rc<UOp>, gate: Rc<UOp> },
    Store { buffer: Rc<UOp>, index: Rc<UOp>, value: Rc<UOp> },
    StoreGated { buffer: Rc<UOp>, index: Rc<UOp>, value: Rc<UOp>, gate: Rc<UOp> },
}

impl Op {
    /// Get all child UOps as a Vec of references.
    ///
    /// This is the convenient API for traversing the graph.
    /// Allocates a Vec but is simple to use.
    pub fn children(&self) -> Vec<&Rc<UOp>> {
        match self {
            // Nullary operations
            Self::Const(_)
            | Self::Unique(_)
            | Self::Device(_)
            | Self::Noop
            | Self::DefineGlobal(_)
            | Self::DefineLocal(_) => vec![],

            // Grouped operations
            Self::Unary(_, x) => vec![x],
            Self::Binary(_, a, b) => vec![a, b],

            // Special unary operations
            Self::Cast { src, .. } => vec![src],
            Self::MSelect { buffer, .. } => vec![buffer],

            // Buffer operations
            Self::Buffer { unique, device, .. } => vec![unique, device],
            Self::BufferView { buffer, .. } => vec![buffer],
            Self::Bufferize { compute, ranges, .. } => {
                let mut children = vec![compute];
                children.extend(ranges.iter());
                children
            }
            Self::Index { buffer, indices, gate } => {
                let mut children = vec![buffer];
                children.extend(indices.iter());
                if let Some(g) = gate {
                    children.push(g);
                }
                children
            }
            Self::Copy { src, device } => vec![src, device],
            Self::MStack { buffers } => buffers.iter().collect(),

            // Memory operations
            Self::Load { buffer, index } => vec![buffer, index],
            Self::LoadGated { buffer, index, gate } => vec![buffer, index, gate],
            Self::Store { buffer, index, value } => vec![buffer, index, value],
            Self::StoreGated { buffer, index, value, gate } => vec![buffer, index, value, gate],
        }
    }

    /// Visit all child UOps with a visitor function.
    ///
    /// This is the zero-allocation API for performance-critical code.
    /// More efficient than children() but slightly less convenient.
    pub fn for_each_child<F>(&self, mut f: F)
    where
        F: FnMut(&Rc<UOp>),
    {
        match self {
            // Nullary operations
            Self::Const(_)
            | Self::Unique(_)
            | Self::Device(_)
            | Self::Noop
            | Self::DefineGlobal(_)
            | Self::DefineLocal(_) => {}

            // Grouped operations
            Self::Unary(_, x) => f(x),
            Self::Binary(_, a, b) => {
                f(a);
                f(b);
            }

            // Special unary operations
            Self::Cast { src, .. } => f(src),
            Self::MSelect { buffer, .. } => f(buffer),

            // Buffer operations
            Self::Buffer { unique, device, .. } => {
                f(unique);
                f(device);
            }
            Self::BufferView { buffer, .. } => f(buffer),
            Self::Bufferize { compute, ranges, .. } => {
                f(compute);
                ranges.iter().for_each(&mut f);
            }
            Self::Index { buffer, indices, gate } => {
                f(buffer);
                indices.iter().for_each(&mut f);
                if let Some(g) = gate {
                    f(g);
                }
            }
            Self::Copy { src, device } => {
                f(src);
                f(device);
            }
            Self::MStack { buffers } => buffers.iter().for_each(f),

            // Memory operations
            Self::Load { buffer, index } => {
                f(buffer);
                f(index);
            }
            Self::LoadGated { buffer, index, gate } => {
                f(buffer);
                f(index);
                f(gate);
            }
            Self::Store { buffer, index, value } => {
                f(buffer);
                f(index);
                f(value);
            }
            Self::StoreGated { buffer, index, value, gate } => {
                f(buffer);
                f(index);
                f(value);
                f(gate);
            }
        }
    }

    /// Get child UOp pointers for hash consing.
    ///
    /// Returns raw pointers to avoid reference counting overhead during hashing.
    fn src_ptrs(&self) -> Vec<*const UOp> {
        let mut ptrs = Vec::new();
        self.for_each_child(|child| ptrs.push(Rc::as_ptr(child)));
        ptrs
    }
}

/// Micro-operation node in the computation graph.
///
/// UOps form a DAG where operations reference their inputs through the Op enum.
/// Hash consing ensures that structurally identical UOps share the same allocation.
#[derive(Debug)]
pub struct UOp {
    op: Op,
    dtype: DType,
}

/// Cache key for hash consing.
///
/// Uses pointer addresses for child UOps to avoid infinite recursion during hashing.
#[derive(Eq, PartialEq, Hash)]
struct UOpKey {
    op_discriminant: std::mem::Discriminant<Op>,
    dtype: DType,
    src_ptrs: Vec<*const UOp>,
    // Store additional data that's not in src_ptrs
    op_data: OpData,
}

/// Non-recursive data from Op variants for hashing.
#[derive(Eq, PartialEq, Hash)]
enum OpData {
    Const(ConstValueHash),
    Unique(usize),
    Device(DeviceSpec),
    DefineGlobal(usize),
    DefineLocal(usize),
    Unary(UnaryOp),
    Binary(BinaryOp),
    CastDType(DType),
    MSelectIdx(usize),
    BufferSize(usize),
    BufferView(usize, usize),
    Bufferize(BufferizeOpts),
    None,
}

impl UOpKey {
    fn new(op: &Op, dtype: DType) -> Self {
        let op_discriminant = discriminant(op);
        let src_ptrs = op.src_ptrs();

        let op_data = match op {
            Op::Const(c) => OpData::Const(*c),
            Op::Unique(id) => OpData::Unique(*id),
            Op::Device(d) => OpData::Device(d.clone()),
            Op::DefineGlobal(slot) => OpData::DefineGlobal(*slot),
            Op::DefineLocal(slot) => OpData::DefineLocal(*slot),
            Op::Unary(unary_op, _) => OpData::Unary(*unary_op),
            Op::Binary(binary_op, _, _) => OpData::Binary(*binary_op),
            Op::Cast { dtype, .. } => OpData::CastDType(*dtype),
            Op::MSelect { device_index, .. } => OpData::MSelectIdx(*device_index),
            Op::Buffer { size, .. } => OpData::BufferSize(*size),
            Op::BufferView { size, offset, .. } => OpData::BufferView(*size, *offset),
            Op::Bufferize { opts, .. } => OpData::Bufferize(opts.clone()),
            _ => OpData::None,
        };

        Self { op_discriminant, dtype, src_ptrs, op_data }
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
        let key = UOpKey::new(&op, dtype);

        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();

            // Check if we already have this UOp
            if let Some(weak) = cache.get(&key)
                && let Some(rc) = weak.upgrade()
            {
                return rc;
            }

            // Create new UOp
            let uop = Rc::new(Self { op, dtype });

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
        self.dtype
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
        Self::new(Op::Cast { src, dtype }, dtype)
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
        let dtype = buffer.dtype;
        Self::new(Op::BufferView { buffer, size, offset }, dtype)
    }

    /// Create an index operation.
    pub fn index(buffer: Rc<Self>, indices: Vec<Rc<Self>>) -> Rc<Self> {
        let dtype = buffer.dtype;
        let indices = SmallVec::from_vec(indices);
        Self::new(Op::Index { buffer, indices, gate: None }, dtype)
    }

    /// Create a gated index operation.
    pub fn index_gated(buffer: Rc<Self>, indices: Vec<Rc<Self>>, gate: Rc<Self>) -> Rc<Self> {
        let dtype = buffer.dtype;
        let indices = SmallVec::from_vec(indices);
        Self::new(Op::Index { buffer, indices, gate: Some(gate) }, dtype)
    }

    /// Copy to a different device.
    pub fn copy_to_device(self: &Rc<Self>, device: DeviceSpec) -> Rc<Self> {
        let dev = Self::device(device);
        Self::new(Op::Copy { src: self.clone(), device: dev }, self.dtype)
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
                node.op.for_each_child(|child| {
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
                pub fn $name(arg: Rc<Self>) -> Rc<Self> {
                    let dtype = arg.dtype;
                    Self::new(Op::Unary(UnaryOp::$op, arg), dtype)
                }
            )*
        }
    }
}

macro_rules! binary_ops {
    ($($name:ident => $op:ident),* $(,)?) => {
        impl UOp {
            $(
                pub fn $name(lhs: Rc<Self>, rhs: Rc<Self>) -> Rc<Self> {
                    let dtype = lhs.dtype; // TODO: proper type promotion
                    Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), dtype)
                }
            )*
        }
    }
}

macro_rules! cmp_ops {
    ($($name:ident => $op:ident),* $(,)?) => {
        impl UOp {
            $(
                pub fn $name(lhs: Rc<Self>, rhs: Rc<Self>) -> Rc<Self> {
                    Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), DType::Bool)
                }
            )*
        }
    }
}

unary_ops! {
    neg => Neg,
    sqrt => Sqrt,
    exp2 => Exp2,
    log2 => Log2,
}

binary_ops! {
    add => Add,
    mul => Mul,
    sub => Sub,
    div => Div,
}

cmp_ops! {
    cmplt => Lt,
    cmpeq => Eq,
    cmpne => Ne,
}

impl Clone for UOp {
    fn clone(&self) -> Self {
        Self { op: self.op.clone(), dtype: self.dtype }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_const_creation() {
        let c1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        assert_eq!(c1.dtype(), DType::Float32);
        assert!(matches!(c1.op(), Op::Const(_)));
    }

    #[test]
    fn test_hash_consing() {
        // Create two identical constants
        let c1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let c2 = UOp::const_(DType::Float32, ConstValue::Float(1.0));

        // They should be the same object
        assert!(Rc::ptr_eq(&c1, &c2), "Hash consing should return same Rc for identical UOps");
    }

    #[test]
    fn test_hash_consing_with_src() {
        let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));

        // Create a + b twice
        let add1 = UOp::add(a.clone(), b.clone());
        let add2 = UOp::add(a.clone(), b.clone());

        // Should be the same object
        assert!(Rc::ptr_eq(&add1, &add2), "Hash consing should work with src nodes");
    }

    #[test]
    fn test_binary_operations() {
        let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));

        let add = UOp::add(a.clone(), b.clone());
        assert_eq!(add.dtype(), DType::Float32);
        assert_eq!(add.op().children().len(), 2);

        let mul = UOp::mul(a.clone(), b.clone());
        assert_eq!(mul.dtype(), DType::Float32);
    }

    #[test]
    fn test_unary_operations() {
        let a = UOp::const_(DType::Float32, ConstValue::Float(4.0));

        let sqrt = UOp::sqrt(a.clone());
        assert_eq!(sqrt.dtype(), DType::Float32);
        assert_eq!(sqrt.op().children().len(), 1);
    }

    #[test]
    fn test_cast() {
        let a = UOp::const_(DType::Float32, ConstValue::Float(1.5));
        let cast = UOp::cast(a.clone(), DType::Int32);

        assert_eq!(cast.dtype(), DType::Int32);
    }

    #[test]
    fn test_comparison() {
        let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));

        let cmp = UOp::cmplt(a.clone(), b.clone());
        assert_eq!(cmp.dtype(), DType::Bool);
    }

    #[test]
    fn test_toposort() {
        // Build graph: (a + b) * c
        let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
        let c = UOp::const_(DType::Float32, ConstValue::Float(3.0));

        let add = UOp::add(a.clone(), b.clone());
        let mul = UOp::mul(add.clone(), c.clone());

        let sorted = mul.toposort();

        // All nodes should be present
        assert!(sorted.len() >= 5); // a, b, c, add, mul

        // Check that dependencies come before dependents
        let positions: HashMap<_, _> = sorted.iter().enumerate().map(|(i, node)| (Rc::as_ptr(node), i)).collect();

        for node in &sorted {
            let node_pos = positions[&Rc::as_ptr(node)];
            for child in node.op().children() {
                let child_pos = positions[&Rc::as_ptr(child)];
                assert!(child_pos < node_pos, "Dependencies must come before dependents");
            }
        }
    }

    #[test]
    fn test_toposort_shared_node() {
        // Build graph: x = a + b; y = a + c; z = x * y
        // Node 'a' is shared between x and y
        let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
        let c = UOp::const_(DType::Float32, ConstValue::Float(3.0));

        let x = UOp::add(a.clone(), b.clone());
        let y = UOp::add(a.clone(), c.clone());
        let z = UOp::mul(x.clone(), y.clone());

        let sorted = z.toposort();

        // Node 'a' should appear only once
        let a_ptr = Rc::as_ptr(&a);
        let a_count = sorted.iter().filter(|node| Rc::as_ptr(node) == a_ptr).count();
        assert_eq!(a_count, 1, "Shared node 'a' should appear exactly once");
    }

    #[test]
    fn test_buffer_creation() {
        let buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
        assert!(matches!(buf.op(), Op::Buffer { .. }));
        assert_eq!(buf.dtype(), DType::Float32);

        if let Op::Buffer { size, .. } = buf.op() {
            assert_eq!(*size, 100);
        } else {
            panic!("Expected Buffer op");
        }
    }

    #[test]
    fn test_buffer_hash_consing() {
        // Two buffers with same device and size should NOT be the same
        // (due to different UNIQUE identifiers)
        let buf1 = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
        let buf2 = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
        assert!(!Rc::ptr_eq(&buf1, &buf2), "Different buffers should have different UNIQUE ids");
    }

    #[test]
    fn test_buffer_view() {
        let buf = UOp::new_buffer(DeviceSpec::Cpu, 1000, DType::Float32);
        let view = UOp::buffer_view(buf, 100, 50);

        assert!(matches!(view.op(), Op::BufferView { .. }));
        assert_eq!(view.dtype(), DType::Float32);

        if let Op::BufferView { size, offset, .. } = view.op() {
            assert_eq!(*size, 100);
            assert_eq!(*offset, 50);
        } else {
            panic!("Expected BufferView op");
        }
    }

    #[test]
    fn test_index_operation() {
        let buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
        let idx = UOp::const_(DType::Int32, ConstValue::Int(10));

        let indexed = UOp::index(buf, vec![idx]);
        assert!(matches!(indexed.op(), Op::Index { .. }));
        assert_eq!(indexed.op().children().len(), 2); // buffer + 1 index
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_copy_to_device() {
        let buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
        let gpu_copy = buf.copy_to_device(DeviceSpec::Cuda { device_id: 0 });

        assert!(matches!(gpu_copy.op(), Op::Copy { .. }));
        assert_eq!(gpu_copy.dtype(), DType::Float32);
        assert_eq!(gpu_copy.op().children().len(), 2); // source buffer + target device
    }

    #[test]
    fn test_device_and_unique() {
        let dev = UOp::device(DeviceSpec::Cpu);
        assert!(matches!(dev.op(), Op::Device(_)));
        if let Op::Device(spec) = dev.op() {
            assert_eq!(*spec, DeviceSpec::Cpu);
        }

        let uniq = UOp::unique(Some(42));
        assert!(matches!(uniq.op(), Op::Unique(42)));

        let uniq_auto = UOp::unique(None);
        assert!(matches!(uniq_auto.op(), Op::Unique(_)));
    }

    #[test]
    fn test_children_method() {
        let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
        let add = UOp::add(a.clone(), b.clone());

        let children = add.op().children();
        assert_eq!(children.len(), 2);
        assert!(Rc::ptr_eq(children[0], &a));
        assert!(Rc::ptr_eq(children[1], &b));
    }

    #[test]
    fn test_for_each_child() {
        let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
        let add = UOp::add(a.clone(), b.clone());

        let mut children = Vec::new();
        add.op().for_each_child(|child| children.push(child.clone()));

        assert_eq!(children.len(), 2);
        assert!(Rc::ptr_eq(&children[0], &a));
        assert!(Rc::ptr_eq(&children[1], &b));
    }
}
