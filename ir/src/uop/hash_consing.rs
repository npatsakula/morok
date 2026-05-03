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

use std::hash::{Hash, Hasher};
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
///
/// Performance: hash is pre-computed during construction and cached in `cached_hash`.
/// This avoids re-hashing on every HashMap lookup (the previous bottleneck: 57% of CPU
/// in xxhash). Follows Tinygrad's approach where UOp hash is `id()`-based (~nanoseconds).
#[derive(Clone)]
struct UOpKey {
    op_discriminant: std::mem::Discriminant<Op>,
    dtype: DType,
    src_hashes: SmallVec<[u64; 4]>,
    op_data: OpData,
    tag: Option<SmallVec<[usize; 2]>>,
    /// Pre-computed hash — avoids re-hashing on every HashMap operation.
    cached_hash: u64,
}

impl Hash for UOpKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Use pre-computed hash directly — O(1) regardless of OpData complexity
        state.write_u64(self.cached_hash);
    }
}

impl PartialEq for UOpKey {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: different hashes → definitely not equal
        self.cached_hash == other.cached_hash
            && self.op_discriminant == other.op_discriminant
            && self.dtype == other.dtype
            && self.src_hashes == other.src_hashes
            && self.op_data == other.op_data
            && self.tag == other.tag
    }
}

impl Eq for UOpKey {}

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
    LUnique(usize),
    Device(DeviceSpec),
    // DefineLocal includes unique ID to prevent hash consing across kernels.
    DefineLocal(usize, usize), // (slot, unique_id)

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
    //
    // `local` distinguishes buffers tagged by `Op::LUnique` (per-kernel local
    // counter starting at 0 — see `schedule/src/rangeify/kernel.rs`'s
    // `next_lunique`) from buffers tagged by `Op::Unique` (the global atomic
    // `next_unique_id`). Without the discriminator, `BufferData(0, size)`
    // could collide between an LUnique slot 0 and a Unique with global id 0.
    BufferData { local: bool, id: usize, size: usize },
    ParamData(usize, usize), // (slot, size) — dedup by structure, matching Tinygrad's UOp cache
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
    DefineRegData(usize, usize),     // (size, id)

    // Advanced operations
    WmmaData(Box<WmmaMetadata>),
    ContractRanges(Vec<(usize, usize)>),
    UnrollAxes(Vec<(usize, usize)>),
    CustomCode(String),
    CustomFunctionKind(CustomFunctionKind),
    CallInfoData(CallInfo),
    SourceCode(String),
    ProgramBinaryBytes(Vec<u8>),
    SinkInfo(Option<crate::types::KernelInfo>),

    // Movement operations with extra data
    ContiguousOpts(Vec<crate::types::ContiguousHint>),

    // Tuple operations
    GetTupleIndex(usize),

    // Operations with only children (no extra semantic data)
    None,
}

/// Get child UOp structural hashes for hash consing.
///
/// Uses `content_hash` (structural) instead of `id` (identity) so that
/// structurally identical children produce the same key — even if they're
/// different `Arc` pointers. This makes hash consing truly structural,
/// matching Tinygrad's behavior where `id()` works because hash consing
/// guarantees same structure = same object.
///
/// Returns SmallVec of hashes, optimized for common case of ≤4 children (inline storage).
fn src_hashes(op: &Op) -> SmallVec<[u64; 4]> {
    op.children().into_iter().map(|child| child.content_hash).collect()
}

impl UOpKey {
    fn new(op: &Op, dtype: DType, tag: &Option<SmallVec<[usize; 2]>>) -> Self {
        let op_discriminant = discriminant(op);
        let src_hashes = src_hashes(op);

        let op_data = match op {
            Op::Const(c) => OpData::Const(*c),
            Op::Unique(id) => OpData::Unique(*id),
            Op::LUnique(id) => OpData::LUnique(*id),
            Op::Device(d) => OpData::Device(d.clone()),
            Op::DefineLocal(slot) => OpData::DefineLocal(*slot, next_unique_id()),
            Op::Unary(unary_op, _) => OpData::Unary(*unary_op),
            Op::Binary(binary_op, _, _) => OpData::Binary(*binary_op),
            Op::Ternary(ternary_op, _, _, _) => OpData::Ternary(*ternary_op),
            Op::Cast { dtype, .. } => OpData::CastDType(dtype.clone()),
            Op::BitCast { dtype, .. } => OpData::BitCastDType(dtype.clone()),
            Op::MSelect { device_index, .. } => OpData::MSelectIdx(*device_index),
            Op::Special { name, .. } => OpData::SpecialName(name.clone()),
            Op::Buffer { unique, size, .. } => match unique.op() {
                Op::Unique(id) => OpData::BufferData { local: false, id: *id, size: *size },
                Op::LUnique(id) => OpData::BufferData { local: true, id: *id, size: *size },
                // Fallback: use UOp's stable id (already globally unique).
                _ => OpData::BufferData { local: false, id: unique.id as usize, size: *size },
            },
            Op::BufferView { size, offset, .. } => OpData::BufferView(*size, *offset),
            Op::Bufferize { opts, .. } => OpData::Bufferize(opts.clone()),
            Op::Permute { axes, .. } => OpData::PermuteAxes(axes.clone()),
            Op::Flip { axes, .. } => OpData::FlipAxes(axes.clone()),
            Op::Multi { axis, .. } => OpData::MultiAxis(*axis),
            Op::ReduceAxis { reduce_op, axes, .. } => OpData::ReduceAxisData(*reduce_op, axes.clone()),
            Op::Reduce { reduce_op, .. } => OpData::ReduceOp(*reduce_op),
            Op::AllReduce { reduce_op, .. } => OpData::AllReduceOp(*reduce_op),
            Op::Range { axis_id, axis_type, .. } => OpData::RangeData(*axis_id, *axis_type),
            Op::Gep { indices, .. } => OpData::GepIndices(indices.clone()),
            Op::VConst { values } => OpData::VConstValues(values.iter().map(|v| ConstValueHash(*v)).collect()),
            Op::DefineVar { name, min_val, max_val } => OpData::DefineVarData(name.clone(), *min_val, *max_val),
            Op::DefineReg { size, id } => OpData::DefineRegData(*size, *id),
            Op::Wmma { metadata, .. } => OpData::WmmaData(metadata.clone().into()),
            Op::Contract { upcast_ranges, .. } => OpData::ContractRanges(upcast_ranges.clone()),
            Op::Unroll { unroll_axes, .. } => OpData::UnrollAxes(unroll_axes.clone()),
            Op::Custom { code, .. } | Op::CustomI { code, .. } => OpData::CustomCode(code.clone()),
            Op::CustomFunction { kind, .. } => OpData::CustomFunctionKind(kind.clone()),
            Op::Call { info, .. } | Op::Function { info, .. } => OpData::CallInfoData(info.clone()),
            Op::Sink { info, .. } => OpData::SinkInfo(info.clone()),
            Op::Source { code } => OpData::SourceCode(code.clone()),
            Op::ProgramBinary { bytes } => OpData::ProgramBinaryBytes(bytes.clone()),
            Op::Contiguous { opts, .. } => OpData::ContiguousOpts(opts.to_vec()),
            Op::Param { slot, size, .. } => OpData::ParamData(*slot, *size),
            // All remaining ops encode semantic data entirely through children
            // (captured by src_hashes) — no extra OpData needed.
            Op::Noop | Op::Invalid => OpData::None,
            // Multi-child ops: children ARE the data
            Op::Group { .. }
            | Op::Vectorize { .. }
            | Op::Cat { .. }
            | Op::PtrCat { .. }
            | Op::MStack { .. }
            | Op::Barrier { .. }
            | Op::Linear { .. }
            | Op::Program { .. }
            | Op::Tuple { .. } => OpData::None,
            Op::GetTuple { index, .. } => OpData::GetTupleIndex(*index),
            // Movement ops: shape/bounds are Arc<UOp> children
            Op::Reshape { .. } | Op::Expand { .. } | Op::Pad { .. } | Op::Shrink { .. } => OpData::None,
            // Memory/control: all fields are Arc<UOp> children
            Op::Index { .. } | Op::PointerIndex { .. } | Op::Copy { .. } | Op::Load { .. } | Op::Store { .. } => {
                OpData::None
            }
            Op::If { .. } | Op::EndIf { .. } | Op::End { .. } | Op::After { .. } => OpData::None,
            // Single-source ops with no extra data
            Op::Detach { .. } | Op::ContiguousBackward { .. } | Op::Precast { .. } => OpData::None,
            // Binding: children encode all semantics
            Op::Bind { .. } => OpData::None,
        };

        // Pre-compute hash using xxhash (fast, non-cryptographic).
        // Cached to avoid re-hashing on every HashMap lookup — the previous
        // bottleneck was 57% of CPU time spent in xxhash due to repeated hashing.
        let cached_hash = {
            use xxhash_rust::xxh64::Xxh64;
            let mut h = Xxh64::new(0);
            op_discriminant.hash(&mut h);
            dtype.hash(&mut h);
            for id in &src_hashes {
                h.write_u64(*id);
            }
            op_data.hash(&mut h);
            tag.hash(&mut h);
            h.finish()
        };

        Self { op_discriminant, dtype, src_hashes, op_data, tag: tag.clone(), cached_hash }
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
    #[inline]
    #[track_caller]
    pub fn new(op: Op, dtype: DType) -> Arc<Self> {
        Self::new_tagged(op, dtype, None)
    }

    /// Create a UOp with an explicit tag (Tinygrad: `UOp(op, dtype, src, arg, tag)`).
    /// Tag participates in hash consing — same structure + different tag = different UOp.
    #[track_caller]
    pub fn new_tagged(op: Op, dtype: DType, tag: Option<SmallVec<[usize; 2]>>) -> Arc<Self> {
        use papaya::{Compute, Operation};

        let caller_location = std::panic::Location::caller();
        let key = UOpKey::new(&op, dtype.clone(), &tag);
        let guard = uops().guard();

        // Fast path: check if valid entry exists
        if let Some(weak) = uops().get(&key, &guard)
            && let Some(arc) = weak.upgrade()
        {
            use crate::provenance::PROVENANCE_TRACKER;
            PROVENANCE_TRACKER.with(|tracker| {
                tracker.borrow_mut().capture(arc.id, caller_location);
            });
            return arc;
        }

        let content_hash = {
            use xxhash_rust::xxh64::Xxh64;
            let mut h = Xxh64::new(0);
            std::mem::discriminant(&op).hash(&mut h);
            dtype.hash(&mut h);
            for child in op.children() {
                h.write_u64(child.content_hash);
            }
            key.op_data.hash(&mut h);
            h.finish()
        };

        let new_arc = Arc::new(Self {
            id: next_uop_id(),
            op,
            dtype,
            content_hash,
            tag,
            shape_cache: std::sync::OnceLock::new(),
            ranges_cache: std::sync::OnceLock::new(),
            in_scope_ranges_cache: std::sync::OnceLock::new(),
            vmin_vmax_cache: std::sync::OnceLock::new(),
            sound_vmin_vmax_cache: std::sync::OnceLock::new(),
            has_index_in_sources_cache: std::sync::OnceLock::new(),
            backward_slice_cache: std::sync::OnceLock::new(),
            metadata: None,
        });
        let new_weak = Arc::downgrade(&new_arc);

        let result = uops().compute(
            key,
            |entry| match entry {
                Some((_, existing_weak)) => {
                    if let Some(existing_arc) = existing_weak.upgrade() {
                        Operation::Abort(existing_arc)
                    } else {
                        Operation::Insert(new_weak.clone())
                    }
                }
                None => Operation::Insert(new_weak.clone()),
            },
            &guard,
        );

        let final_arc = match result {
            Compute::Inserted(_, _) | Compute::Updated { .. } => new_arc,
            Compute::Aborted(existing_arc) => existing_arc,
            _ => new_arc,
        };

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
        self.with_metadata_raw(Arc::new(metadata))
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

    /// Get raw metadata (type-erased).
    ///
    /// Used to preserve metadata across graph rewrites that create new root nodes.
    pub fn metadata_raw(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        self.metadata.clone()
    }

    /// Attach raw metadata (type-erased), creating a new instance.
    ///
    /// Used to re-attach metadata that was saved before graph rewrites.
    pub fn with_metadata_raw(self: &Arc<Self>, metadata: Arc<dyn std::any::Any + Send + Sync>) -> Arc<Self> {
        Arc::new(Self {
            id: next_uop_id(),
            op: self.op.clone(),
            dtype: self.dtype.clone(),
            content_hash: self.content_hash, // same structure, same content hash
            tag: self.tag.clone(),
            shape_cache: std::sync::OnceLock::new(),
            ranges_cache: std::sync::OnceLock::new(),
            in_scope_ranges_cache: std::sync::OnceLock::new(),
            vmin_vmax_cache: std::sync::OnceLock::new(),
            sound_vmin_vmax_cache: std::sync::OnceLock::new(),
            has_index_in_sources_cache: std::sync::OnceLock::new(),
            backward_slice_cache: std::sync::OnceLock::new(),
            metadata: Some(metadata),
        })
    }
}
