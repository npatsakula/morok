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
//! # Memory Management (Tinygrad-aligned)
//!
//! UOps are stored as `Weak<UOp>` references in the cache. When no strong references
//! remain (outside the cache), the UOp is automatically eligible for cleanup.
//! Dead weak references are cleaned up lazily on next access or via `gc_dead_refs()`.
//!
//! This matches Tinygrad's approach using `weakref.WeakKeyDictionary` - no manual
//! cleanup calls required in user code.

use std::hash::Hash;
use std::mem::discriminant;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};

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
    // DefineGlobal and DefineLocal include unique IDs to prevent hash consing
    // across different kernels/realizations. Each kernel's DEFINE_GLOBAL(0)
    // must be a distinct UOp, even though they have the same slot number.
    DefineGlobal(usize, usize), // (slot, unique_id)
    DefineLocal(usize, usize),  // (slot, unique_id)

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
    BufferData(usize, usize), // (unique_id, size) - each buffer is unique
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
    DefineVarData(String, i64, i64), // (name, min_val, max_val)
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
            // DEFINE_GLOBAL/LOCAL need unique IDs to prevent hash consing across kernels
            Op::DefineGlobal(slot) => OpData::DefineGlobal(*slot, next_unique_id()),
            Op::DefineLocal(slot) => OpData::DefineLocal(*slot, next_unique_id()),

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

            // Buffer operations - include unique ID to prevent collision
            Op::Buffer { unique, size, .. } => {
                if let Op::Unique(id) = unique.op() {
                    OpData::BufferData(*id, *size)
                } else {
                    // Fallback: use UOp's stable id
                    OpData::BufferData(unique.id as usize, *size)
                }
            }
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

// Global hash consing cache using lock-free concurrent HashMap.
//
// Design: Stores Weak<UOp> for automatic memory management (Tinygrad-aligned).
// - Cross-thread deduplication: same UOpKey → same Arc<UOp> across all threads
// - Lock-free reads and writes via papaya's epoch-based reclamation
// - Automatic cleanup: when no strong refs remain, weak ref becomes dead
// - Dead refs cleaned lazily on next access or via gc_dead_refs()
//
// Memory lifecycle (matches Tinygrad's weakref.WeakKeyDictionary):
// 1. UOps created via UOp::new() store Weak refs in cache
// 2. Strong refs held by Tensor, Scheduler, etc. keep UOps alive
// 3. When all strong refs dropped, UOp deallocated, weak ref becomes dead
// 4. Dead weak refs cleaned up lazily or via gc_dead_refs()
static UOPS: OnceLock<HashMap<UOpKey, Weak<UOp>>> = OnceLock::new();

fn uops() -> &'static HashMap<UOpKey, Weak<UOp>> {
    UOPS.get_or_init(HashMap::new)
}

/// Remove dead weak references from the cache.
///
/// This is optional - dead refs are also cleaned lazily on next access.
/// Call this if you want to proactively free cache memory.
///
/// # Example
///
/// ```ignore
/// // After dropping many tensors, optionally clean up cache
/// gc_dead_refs();
/// ```
pub fn gc_dead_refs() {
    let map = uops();
    let guard = map.guard();

    // Collect keys with dead weak refs
    let to_remove: Vec<UOpKey> =
        map.iter(&guard).filter(|(_, weak)| weak.upgrade().is_none()).map(|(k, _)| k.clone()).collect();

    // Remove dead entries
    for key in to_remove {
        map.remove(&key, &guard);
    }
}

/// Legacy alias for gc_dead_refs (for compatibility).
///
/// With weak references, UOps are automatically cleaned up when no longer
/// referenced. This function now just cleans up dead weak refs in the cache.
#[deprecated(note = "UOp cache now uses weak refs - cleanup is automatic. Use gc_dead_refs() to clean cache.")]
pub fn gc_unused_uops() {
    gc_dead_refs();
}

/// Get the set of IDs for UOps currently alive in the cache.
///
/// This is used by kernel cache GC to determine which compiled kernels
/// can be safely removed (those whose AST IDs are no longer live).
///
/// # Returns
///
/// A HashSet containing the IDs of all currently cached UOps (only live ones).
pub fn live_uop_ids() -> std::collections::HashSet<u64> {
    let map = uops();
    let guard = map.guard();
    map.iter(&guard).filter_map(|(_, weak)| weak.upgrade().map(|arc| arc.id)).collect()
}

impl UOp {
    /// Create a new UOp with hash consing.
    ///
    /// If an identical UOp already exists (in any thread) and is still alive,
    /// returns a reference to it. Otherwise, creates a new UOp and caches it.
    ///
    /// # Thread Safety
    ///
    /// This function is thread-safe. Creating the same UOp from different threads
    /// will return the same `Arc<UOp>`, so `Arc::ptr_eq` works across threads.
    ///
    /// # Memory Management
    ///
    /// The cache stores weak references. UOps are automatically cleaned up when
    /// no strong references remain (Tinygrad-aligned behavior).
    #[track_caller]
    pub fn new(op: Op, dtype: DType) -> Arc<Self> {
        use papaya::{Compute, Operation};

        // Capture caller location BEFORE entering any closures
        let caller_location = std::panic::Location::caller();
        let key = UOpKey::new(&op, dtype.clone());
        let guard = uops().guard();

        // Fast path: check if valid entry exists
        if let Some(weak) = uops().get(&key, &guard)
            && let Some(arc) = weak.upgrade() {
                // Valid entry found - record provenance and return
                use crate::provenance::PROVENANCE_TRACKER;
                PROVENANCE_TRACKER.with(|tracker| {
                    tracker.borrow_mut().capture(arc.id, caller_location);
                });
                return arc;
            }
            // Dead weak ref - will be replaced below

        // Create new UOp (will be used if we win the race)
        let new_arc = Arc::new(Self {
            id: next_uop_id(),
            op,
            dtype,
            shape_cache: std::sync::OnceLock::new(),
            ranges_cache: std::sync::OnceLock::new(),
            in_scope_ranges_cache: std::sync::OnceLock::new(),
            vmin_vmax_cache: std::sync::OnceLock::new(),
            metadata: None,
        });
        let new_weak = Arc::downgrade(&new_arc);

        // Atomic insert: insert our weak ref, but if someone else has a valid one, use theirs
        // Note: papaya's Insert replaces existing entries, which handles dead weak refs
        let result = uops().compute(
            key,
            |entry| match entry {
                Some((_, existing_weak)) => {
                    if let Some(existing_arc) = existing_weak.upgrade() {
                        // Valid entry exists - abort with it (reuse existing)
                        Operation::Abort(existing_arc)
                    } else {
                        // Dead entry - replace with ours
                        Operation::Insert(new_weak.clone())
                    }
                }
                None => {
                    // No entry - insert ours
                    Operation::Insert(new_weak.clone())
                }
            },
            &guard,
        );

        // Determine which Arc to return based on compute result
        let final_arc = match result {
            Compute::Inserted(_, _) | Compute::Updated { .. } => new_arc,
            Compute::Aborted(existing_arc) => existing_arc,
            _ => new_arc, // Fallback for Unchanged/Removed (shouldn't happen)
        };

        // Record provenance
        use crate::provenance::PROVENANCE_TRACKER;
        PROVENANCE_TRACKER.with(|tracker| {
            tracker.borrow_mut().capture(final_arc.id, caller_location);
        });

        final_arc
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
