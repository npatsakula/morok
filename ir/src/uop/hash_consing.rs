//! Hash consing infrastructure for UOp deduplication.
//!
//! This module implements the caching system that ensures structurally identical
//! UOps share the same memory allocation (hash consing).
//!
//! # Thread Safety
//!
//! Uses a global lock-free concurrent HashMap (papaya) for cross-thread deduplication.
//! Creating the same UOp in different threads returns the same `Arc<UOp>`, so
//! `Arc::ptr_eq` works correctly across thread boundaries.
//!
//! # Memory Management
//!
//! UOps are stored as `Arc<UOp>` (not `Weak<UOp>`) for simpler atomic operations.
//! This means UOps are kept alive by the cache until explicitly cleaned up via
//! `gc_unused_uops()`. Call this after tensor operations complete to free memory.

use std::hash::Hash;
use std::mem::discriminant;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use papaya::HashMap;
use smallvec::SmallVec;

use crate::op::Op;
use crate::types::*;
use crate::uop::core::UOp;
use morok_dtype::DType;
use morok_dtype::DeviceSpec;

// Global atomic counter for unique identifiers.
//
// Uses AtomicUsize for thread-safe ID generation across all threads.
// Ordering::Relaxed is sufficient since we only need uniqueness, not synchronization.
static UNIQUE_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn next_unique_id() -> usize {
    UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// Global atomic counter for UOp stable IDs.
//
// Provides monotonic IDs that never repeat, eliminating ABA problem.
// Uses u64 to provide 2^64 unique IDs (effectively unlimited).
static UOP_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn next_uop_id() -> u64 {
    UOP_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Cache key for hash consing.
///
/// Uses stable UOp IDs for child UOps to avoid infinite recursion during hashing.
/// IDs are monotonic and never reused, eliminating ABA problem from pointer-based approach.
#[derive(Eq, PartialEq, Hash, Clone)]
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
    RangeData(AxisId, AxisType),

    // Vector operations
    GepIndices(Vec<usize>),
    VConstValues(Vec<ConstValueHash>),

    // Symbolic/Define operations
    DefineVarData(String, i64),
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
            Op::DefineVar { name, max_val } => OpData::DefineVarData(name.clone(), *max_val),
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

// Global hash consing cache using lock-free concurrent HashMap.
//
// Design: Stores Arc<UOp> (not Weak) for simpler atomic get-or-insert operations.
// - Cross-thread deduplication: same UOpKey → same Arc<UOp> across all threads
// - Lock-free reads and writes via papaya's epoch-based reclamation
// - Requires explicit GC via gc_unused_uops() after tensor operations
//
// Memory lifecycle:
// 1. UOps created via UOp::new() are cached indefinitely
// 2. After tensor.realize(), call gc_unused_uops() to clean up
// 3. UOps with strong_count == 1 (only cache holds them) are removed
static UOPS: OnceLock<HashMap<UOpKey, Arc<UOp>>> = OnceLock::new();

fn uops() -> &'static HashMap<UOpKey, Arc<UOp>> {
    UOPS.get_or_init(HashMap::new)
}

/// Remove UOps that are only referenced by the cache (strong_count == 1).
///
/// Call this after tensor operations complete to free memory.
/// UOps still referenced elsewhere will be kept.
///
/// # Example
///
/// ```ignore
/// let result = tensor.realize()?;
/// gc_unused_uops();  // Clean up intermediate UOps
/// ```
pub fn gc_unused_uops() {
    let map = uops();
    let guard = map.guard();

    // Collect keys to remove (can't mutate while iterating)
    let to_remove: Vec<UOpKey> =
        map.iter(&guard).filter(|(_, arc)| Arc::strong_count(arc) == 1).map(|(k, _)| k.clone()).collect();

    // Remove dead entries
    for key in to_remove {
        map.remove(&key, &guard);
    }
}

impl UOp {
    /// Create a new UOp with hash consing.
    ///
    /// If an identical UOp already exists (in any thread), returns a reference to it.
    /// Otherwise, creates a new UOp and caches it globally.
    ///
    /// # Thread Safety
    ///
    /// This function is thread-safe. Creating the same UOp from different threads
    /// will return the same `Arc<UOp>`, so `Arc::ptr_eq` works across threads.
    #[track_caller]
    pub fn new(op: Op, dtype: DType) -> Arc<Self> {
        // Capture caller location BEFORE entering any closures
        // This is critical for #[track_caller] to work correctly
        let caller_location = std::panic::Location::caller();
        let key = UOpKey::new(&op, dtype.clone());

        let guard = uops().guard();

        // Atomic get-or-insert: if another thread races us, we both get the same Arc
        let uop = uops().get_or_insert_with(
            key,
            || {
                Arc::new(Self {
                    id: next_uop_id(),
                    op,
                    dtype,
                    shape_cache: std::sync::OnceLock::new(),
                    ranges_cache: std::sync::OnceLock::new(),
                    in_scope_ranges_cache: std::sync::OnceLock::new(),
                    vmin_vmax_cache: std::sync::OnceLock::new(),
                    metadata: None,
                })
            },
            &guard,
        );

        // Record provenance in the creating thread (thread-local, debug only)
        // Note: Only the thread that actually created the UOp should record provenance,
        // but get_or_insert_with doesn't tell us if we inserted. For simplicity,
        // we record on every call - duplicate entries are harmless for debugging.
        use crate::provenance::PROVENANCE_TRACKER;
        PROVENANCE_TRACKER.with(|tracker| {
            tracker.borrow_mut().capture(uop.id, caller_location);
        });

        Arc::clone(uop)
    }

    /// Attach metadata to this UOp, creating a new instance.
    ///
    /// Metadata is NOT part of hash consing - this method creates a new UOp
    /// with a different ID but the same operation structure. This allows
    /// attaching metadata (like kernel info) after optimization.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let ast = /* ... optimized AST ... */;
    /// let with_info = ast.with_metadata(KernelInfo::new("r_g16l16", vec![], false));
    /// ```
    pub fn with_metadata<T: std::any::Any + Send + Sync + 'static>(self: &Arc<Self>, metadata: T) -> Arc<Self> {
        Arc::new(Self {
            id: next_uop_id(),
            op: self.op.clone(),
            dtype: self.dtype.clone(),
            shape_cache: std::sync::OnceLock::new(),
            ranges_cache: std::sync::OnceLock::new(),
            in_scope_ranges_cache: std::sync::OnceLock::new(),
            vmin_vmax_cache: std::sync::OnceLock::new(),
            metadata: Some(Arc::new(metadata)),
        })
    }

    /// Get metadata of a specific type if it exists.
    ///
    /// Returns `None` if no metadata is attached or if the metadata is of a different type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(info) = ast.metadata::<KernelInfo>() {
    ///     println!("Kernel name: {}", info.name);
    /// }
    /// ```
    pub fn metadata<T: std::any::Any + Send + Sync>(&self) -> Option<std::sync::Arc<T>> {
        self.metadata.as_ref()?.clone().downcast::<T>().ok()
    }
}
