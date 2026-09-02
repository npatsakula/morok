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
use svod_dtype::DType;
use svod_dtype::DeviceSpec;

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
    src_ids: SmallVec<[u64; 4]>,
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

/// Forwards the single pre-computed xxh64 value `UOpKey::hash` writes.
///
/// The table's `BuildHasher` was `RandomState`, so every probe ran SipHash over
/// an 8-byte buffer holding a digest we had already computed. Tinygrad's `ucache`
/// has the same property for free: its key is a tuple of five pointers hashed by
/// CPython's identity hash.
#[derive(Default)]
struct PrecomputedHasher(u64);

impl Hasher for PrecomputedHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("UOpKey::hash must write exactly one pre-computed u64");
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

type PrecomputedHash = std::hash::BuildHasherDefault<PrecomputedHasher>;

impl PartialEq for UOpKey {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: different hashes → definitely not equal
        self.cached_hash == other.cached_hash
            && self.op_discriminant == other.op_discriminant
            && self.dtype == other.dtype
            && self.src_ids == other.src_ids
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
    CopyDevice(DeviceSpec),

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
    ParamData(Box<ParamArg>),
    SliceSize(usize),
    Stage(BufferizeOpts),

    // Movement/Reshape operations
    PermuteAxes(Vec<usize>),
    FlipAxes(Vec<bool>),
    MultiAxis(usize),

    // Reduction operations
    ReduceAxisData(ReduceOp, Vec<usize>),
    ReduceData(ReduceOp, usize),
    AllReduceData(ReduceOp, DeviceSpec),

    // Control flow operations
    RangeData(AxisId, AxisType),

    // Vector operations
    VConstValues(Vec<ConstValueHash>),

    // Symbolic/Define operations
    DefineVarData(String, i64, i64), // (name, min_val, max_val)

    // Advanced operations
    WmmaData(Box<WmmaMetadata>),
    CustomCode(String),
    CustomFunctionKind(CustomFunctionKind),
    CallInfoData(CallInfo),
    SourceData(Box<(String, Option<SourceStageIdentity>)>),
    ProgramBinaryData(Box<(Vec<u8>, Option<BinaryStageIdentity>)>),
    ProgramData(Box<(ProgramInfo, Option<SourceStageIdentity>, Option<BinaryStageIdentity>)>),
    SinkInfo(Option<crate::types::KernelInfo>),

    // Movement operations with extra data
    ContiguousOpts(Vec<crate::types::ContiguousHint>),

    // Tuple operations
    GetTupleIndex(usize),

    // Operations with only children (no extra semantic data)
    None,

    GetAddrDevice(DeviceSpec),
    // Tail variant preserves all pre-existing OpData hash discriminants.
    InsArg(InsArg),
}

// The hash-cons table stores one `UOpKey` per live UOp, and every `UOp::new`
// probes it with a freshly built key, so `OpData`'s footprint is paid on the
// hottest path in the compiler. Keep the rare, fat payloads behind a `Box`.
const _: () = assert!(size_of::<OpData>() <= 128, "OpData grew: box the new payload");

/// Child identities for in-process hash consing. Children are already
/// hash-consed, while IDs distinguish equal-content nodes with different tags
/// and cannot alias on a content-hash collision.
fn src_ids(op: &Op) -> SmallVec<[u64; 4]> {
    op.children().into_iter().map(|child| child.id).collect()
}

impl UOpKey {
    fn new(op: &Op, dtype: DType, tag: &Option<SmallVec<[usize; 2]>>) -> Self {
        let op_discriminant = discriminant(op);
        let src_ids = src_ids(op);

        let op_data = match op {
            Op::Const(c) => OpData::Const(*c),
            Op::Unique(id) => OpData::Unique(*id),
            Op::LUnique(id) => OpData::LUnique(*id),
            Op::Unary(unary_op, _) => OpData::Unary(*unary_op),
            Op::Binary(binary_op, _, _) => OpData::Binary(*binary_op),
            Op::Ternary(ternary_op, _, _, _) => OpData::Ternary(*ternary_op),
            Op::Cast { dtype, .. } => OpData::CastDType(dtype.clone()),
            Op::BitCast { dtype, .. } => OpData::BitCastDType(dtype.clone()),
            Op::MSelect { device_index, .. } => OpData::MSelectIdx(*device_index),
            Op::Special { name, .. } => OpData::SpecialName(name.clone()),
            Op::GetAddr { device, .. } => OpData::GetAddrDevice(device.clone()),
            Op::Copy { device, .. } => OpData::CopyDevice(device.clone()),
            Op::Buffer { arg, .. } | Op::Param { arg, .. } => OpData::ParamData(arg.clone().into()),
            Op::Slice { size, .. } => OpData::SliceSize(*size),
            Op::Stage { opts, .. } => OpData::Stage(opts.clone()),
            Op::Permute { axes, .. } => OpData::PermuteAxes(axes.clone()),
            Op::Flip { axes, .. } => OpData::FlipAxes(axes.clone()),
            Op::Multi { axis, .. } => OpData::MultiAxis(*axis),
            Op::ReduceAxis { reduce_op, axes, .. } => OpData::ReduceAxisData(*reduce_op, axes.clone()),
            Op::Reduce { reduce_op, num_axes, .. } => OpData::ReduceData(*reduce_op, *num_axes),
            Op::AllReduce { reduce_op, device, .. } => OpData::AllReduceData(*reduce_op, device.clone()),
            Op::Range { axis_id, axis_type, .. } => OpData::RangeData(axis_id.clone(), *axis_type),
            Op::VConst { values } => OpData::VConstValues(values.iter().map(|v| ConstValueHash(*v)).collect()),
            Op::DefineVar { name, min_val, max_val } => OpData::DefineVarData(name.clone(), *min_val, *max_val),
            Op::Wmma { metadata, .. } => OpData::WmmaData(metadata.clone().into()),
            Op::Custom { code, .. } | Op::CustomI { code, .. } => OpData::CustomCode(code.clone()),
            Op::CustomFunction { kind, .. } => OpData::CustomFunctionKind(kind.clone()),
            Op::Call { info, .. } | Op::Function { info, .. } => OpData::CallInfoData(info.clone()),
            Op::Sink { info, .. } => OpData::SinkInfo(info.clone()),
            Op::Source { code, identity } => OpData::SourceData((code.clone(), identity.clone()).into()),
            Op::ProgramBinary { bytes, identity } => {
                OpData::ProgramBinaryData((bytes.clone(), identity.clone()).into())
            }
            Op::Program { info, source, binary, .. } => OpData::ProgramData(
                (
                    info.clone(),
                    source.as_ref().and_then(|stage| match stage.op() {
                        Op::Source { identity, .. } => identity.clone(),
                        _ => None,
                    }),
                    binary.as_ref().and_then(|stage| match stage.op() {
                        Op::ProgramBinary { identity, .. } => identity.clone(),
                        _ => None,
                    }),
                )
                    .into(),
            ),
            Op::Ins { arg, .. } => OpData::InsArg(arg.clone()),
            Op::Contiguous { opts, .. } => OpData::ContiguousOpts(opts.to_vec()),
            // All remaining ops encode semantic data entirely through children
            // (captured by src_ids) — no extra OpData needed.
            Op::Noop => OpData::None,
            // Multi-child ops: children ARE the data
            Op::Group { .. }
            | Op::Stack { .. }
            | Op::MStack { .. }
            | Op::Barrier { .. }
            | Op::Linear { .. }
            | Op::Tuple { .. } => OpData::None,
            Op::GetTuple { index, .. } => OpData::GetTupleIndex(*index),
            // Movement ops: shape/bounds are Arc<UOp> children
            Op::Reshape { .. } | Op::Expand { .. } | Op::Pad { .. } | Op::Shrink { .. } => OpData::None,
            // Memory/control: all fields are Arc<UOp> children
            Op::Index { .. } | Op::Load { .. } | Op::Store { .. } => OpData::None,
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
            for id in &src_ids {
                h.write_u64(*id);
            }
            op_data.hash(&mut h);
            tag.hash(&mut h);
            h.finish()
        };

        Self { op_discriminant, dtype, src_ids, op_data, tag: tag.clone(), cached_hash }
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
static UOPS: OnceLock<HashMap<UOpKey, Weak<UOp>, PrecomputedHash>> = OnceLock::new();

fn uops() -> &'static HashMap<UOpKey, Weak<UOp>, PrecomputedHash> {
    UOPS.get_or_init(HashMap::default)
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

        if let Op::Load { index, alt, gate } = &op {
            assert_eq!(dtype, index.dtype(), "LOAD dtype must match its address dtype");
            assert_eq!(alt.is_some(), gate.is_some(), "LOAD requires either index only or index, alt, and gate");
            if let (Some(alt), Some(gate)) = (alt, gate) {
                assert_eq!(gate.dtype(), DType::Bool, "LOAD gate must have bool dtype");
                assert!(Self::is_invalid_marker(alt) || alt.dtype() == dtype, "LOAD alt dtype must match LOAD dtype");
            }
        }

        let caller_location = std::panic::Location::caller();
        let key = UOpKey::new(&op, dtype.clone(), &tag);
        let guard = uops().guard();

        // Fast path: check if valid entry exists
        // No provenance capture here: an interning hit returns a node that already
        // has its `Created` event, and this branch is the majority of the ~1M
        // `UOp::new` calls in one resnet50 schedule.
        if let Some(weak) = uops().get(&key, &guard)
            && let Some(arc) = weak.upgrade()
        {
            return arc;
        }

        // One walk feeds both the structural hash and the early-reject mask of child op kinds.
        let (content_hash, src_ops) = {
            use xxhash_rust::xxh64::Xxh64;
            let mut h = Xxh64::new(0);
            let mut src_ops = crate::op::OpMask::EMPTY;
            std::mem::discriminant(&op).hash(&mut h);
            dtype.hash(&mut h);
            for child in op.children() {
                h.write_u64(child.content_hash);
                src_ops = src_ops.union(crate::op::OpMask::of_op(child.op()));
            }
            key.op_data.hash(&mut h);
            (h.finish(), src_ops)
        };

        let new_arc = Arc::new(Self {
            id: next_uop_id(),
            op,
            dtype,
            content_hash,
            src_ops,
            tag,
            shape_cache: std::sync::OnceLock::new(),
            ranges_cache: std::sync::OnceLock::new(),
            in_scope_ranges_cache: std::sync::OnceLock::new(),
            vmin_vmax_cache: std::sync::OnceLock::new(),
            sound_vmin_vmax_cache: std::sync::OnceLock::new(),
            has_index_in_sources_cache: std::sync::OnceLock::new(),
            backward_slice_cache: std::sync::OnceLock::new(),
            has_weak_float_cache: std::sync::OnceLock::new(),
            device_spec_cache: std::sync::OnceLock::new(),
            addrspace_cache: std::sync::OnceLock::new(),
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

        crate::provenance::record_created(final_arc.id, caller_location);

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
            src_ops: self.src_ops,
            tag: self.tag.clone(),
            shape_cache: std::sync::OnceLock::new(),
            ranges_cache: std::sync::OnceLock::new(),
            in_scope_ranges_cache: std::sync::OnceLock::new(),
            vmin_vmax_cache: std::sync::OnceLock::new(),
            sound_vmin_vmax_cache: std::sync::OnceLock::new(),
            has_index_in_sources_cache: std::sync::OnceLock::new(),
            backward_slice_cache: std::sync::OnceLock::new(),
            has_weak_float_cache: std::sync::OnceLock::new(),
            device_spec_cache: std::sync::OnceLock::new(),
            addrspace_cache: std::sync::OnceLock::new(),
            metadata: Some(metadata),
        })
    }
}

// ============================================================================
// Drop hook (buffer lifetime tracking)
// ============================================================================

/// Hook fired with the node's id when an `Op::Buffer` UOp is dropped.
///
/// Installed once by the buffer-owning layer (`svod-tensor`'s registry) to
/// expire its uop-id-keyed entries automatically: ids are per-allocation and
/// never reused, so once the node is gone no live graph can look the id up
/// again. The hook runs inside `Drop`, potentially deep in a graph teardown —
/// it must not construct UOps and must not block (a lock-free map removal is
/// the intended shape).
static UOP_DROP_HOOK: std::sync::OnceLock<fn(u64)> = std::sync::OnceLock::new();

/// Install the buffer-uop drop hook. First caller wins; later installs are
/// ignored (install-once by design — the owning layer is a singleton).
pub fn set_uop_drop_hook(hook: fn(u64)) {
    let _ = UOP_DROP_HOOK.set(hook);
}

thread_local! {
    /// Deferred-teardown queue: `.0` is true while a drain is active on this
    /// thread; `.1` holds `Op` payloads taken from dying nodes encountered
    /// during that drain (their child `Arc`s keep the subtree alive until the
    /// drain reaches them).
    static TEARDOWN: std::cell::RefCell<(bool, Vec<Op>)> = const { std::cell::RefCell::new((false, Vec::new())) };
}

impl Drop for UOp {
    fn drop(&mut self) {
        // Buffer nodes only: the hook exists for buffer-lifetime tracking, and
        // graph rewriting churns millions of transient non-buffer nodes per
        // prepare — their drop must stay allocation-free and branch-cheap.
        // The intern table is deliberately NOT touched here: dead `Weak`
        // tombstones are overwritten lazily by `new_tagged`, and rebuilding a
        // `UOpKey` in every drop frame would allocate.
        if matches!(self.op, Op::Buffer { .. })
            && let Some(hook) = UOP_DROP_HOOK.get()
        {
            hook(self.id);
        }

        // Flatten the recursive teardown: a long dependency chain would
        // otherwise recurse once per node in the drop glue and overflow the
        // stack. When some child dies with this node, take ownership of the
        // whole `Op` payload and route it through a thread-local queue: the
        // outermost dying node drains the queue iteratively, and every nested
        // `UOp::drop` re-entered from a drained `Op` merely enqueues its own
        // payload and returns — so the glue recursion stays constant-depth
        // and the total work stays linear in the number of dying nodes.
        let mut has_dying = false;
        self.op.map_child(|child| has_dying |= Arc::strong_count(child) == 1);
        if !has_dying {
            // Every child is shared: the glue only decrements refcounts.
            return;
        }

        let op = std::mem::replace(&mut self.op, Op::Unique(usize::MAX));
        let root_op = TEARDOWN.with(|state| {
            let mut state = state.borrow_mut();
            if state.0 {
                state.1.push(op);
                None
            } else {
                state.0 = true;
                Some(op)
            }
        });
        // Nested drop inside an active drain: the drainer owns `op` now.
        let Some(root_op) = root_op else { return };
        // Drop payloads OUTSIDE the RefCell borrow: each may re-enter
        // `UOp::drop` for children, which locks TEARDOWN again to enqueue.
        drop(root_op);
        while let Some(op) = TEARDOWN.with(|state| state.borrow_mut().1.pop()) {
            drop(op);
        }
        TEARDOWN.with(|state| state.borrow_mut().0 = false);
    }
}
