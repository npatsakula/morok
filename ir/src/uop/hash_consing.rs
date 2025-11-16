//! Hash consing infrastructure for UOp deduplication.
//!
//! This module implements the caching system that ensures structurally identical
//! UOps share the same memory allocation (hash consing).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::Hash;
use std::mem::discriminant;
use std::rc::{Rc, Weak};

use smallvec::SmallVec;

use crate::op::Op;
use crate::types::*;
use crate::uop::core::UOp;
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

pub(crate) fn next_unique_id() -> usize {
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

pub(crate) fn next_uop_id() -> u64 {
    UOP_ID_COUNTER.with(|counter| {
        let id = counter.get();
        counter.set(id + 1);
        id
    })
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

/// Get child UOp stable IDs for hash consing.
///
/// Returns SmallVec of IDs, optimized for common case of ≤4 children (inline storage).
fn src_ids(op: &Op) -> SmallVec<[u64; 4]> {
    op.children().into_iter().map(|child| child.id).collect()
}

impl UOpKey {
    fn new(op: &Op, dtype: DType) -> Self {
        let op_discriminant = discriminant(op);
        let src_ids = src_ids(op);

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
            let uop = Rc::new(Self {
                id: next_uop_id(),
                op,
                dtype,
                shape_cache: std::cell::OnceCell::new(),
                ranges_cache: std::cell::OnceCell::new(),
            });

            // Cache it
            cache.insert(key, Rc::downgrade(&uop));

            uop
        })
    }
}
