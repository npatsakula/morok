//! Hash consing infrastructure for UOp deduplication.
//!
//! Structurally identical UOps share one allocation, so `Arc::ptr_eq` (and the
//! id-based `PartialEq`) is exact structural equality.
//!
//! # Table layout
//!
//! The global lock-free table (papaya) maps each live node's structural hash to a
//! `Weak` handle on the node itself; there is no second copy of the node's
//! identity. Lookups probe with the candidate `(op, dtype, tag)` and compare
//! against the live node behind each colliding entry. Equality is the derived
//! `Op` equality: children compare by interned id, payloads by value.
//!
//! # Memory lifecycle (Tinygrad `ucache` + `__del__`)
//!
//! A node's entry lives exactly as long as the node: `UOp::drop` removes it by
//! allocation identity, so the table never accumulates dead entries.

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use papaya::{Equivalent, HashMap};
use smallvec::SmallVec;

use crate::op::Op;
use crate::ops;
use crate::origin::{self, OriginId};
use crate::uop::core::UOp;
use svod_dtype::DType;

type Tag = Option<SmallVec<[usize; 2]>>;

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

/// Structural hash of `(dtype, op, origin)`: children contribute their own content
/// hash, so the value is deterministic across runs and independent of interning
/// order. The origin is mixed in only when present, so an origin-free graph hashes
/// exactly as it did before origins existed.
fn content_hash(op: &Op, dtype: &DType, origin: Option<OriginId>) -> u64 {
    let mut h = xxhash_rust::xxh64::Xxh64::new(0);
    dtype.hash(&mut h);
    op.hash(&mut h);
    if let Some(origin) = origin {
        h.write_u32(origin.get());
    }
    h.finish()
}

/// Ops an origin must never split.
///
/// BUFFER/PARAM/UNIQUE/LUNIQUE are identities rather than structures: buffer ids key
/// the realize/tensor tables and PARAM positions key kernel dedup.
///
/// CONST is the one *value* here. A literal is the only node two scopes build
/// independently yet identically — every other node is distinguished by its
/// operands, which already carry the scope that produced them. Splitting a literal
/// buys no attribution (a constant costs nothing to execute) and defeats every
/// structural-equality rewrite, `WHERE(_, t, t) -> t` among them. That matters
/// beyond code quality: the kernel cut re-merges the split literals with
/// `without_origins`, so a rewrite the pre-cut passes could not see fires after the
/// CALL ABI is fixed and drops a PARAM the CALL still binds.
///
/// Structural nodes join them: shape stacks (a child of every BUFFER and PARAM,
/// so a variable built in two scopes would otherwise become two variables),
/// variable bindings, vector constants, and index-typed arithmetic, which is
/// shape algebra rather than work a kernel performs.
fn origin_opaque(op: &Op, dtype: &DType) -> bool {
    matches!(
        op,
        Op::Buffer(..)
            | Op::Param(..)
            | Op::Unique(..)
            | Op::LUnique(..)
            | Op::Const(..)
            | Op::VConst(..)
            | Op::Stack(..)
            | Op::Bind(..)
            | Op::DefineVar(..)
            | Op::Noop
    ) || *dtype == DType::Index
        || *dtype == DType::WeakInt
}

/// Table hash: the content hash mixed with the tag, which participates in
/// interning but not in the cross-run content hash.
fn intern_hash(content_hash: u64, tag: &Tag) -> u64 {
    match tag {
        None => content_hash,
        Some(tag) => {
            let mut h = xxhash_rust::xxh64::Xxh64::new(content_hash);
            tag.hash(&mut h);
            h.finish()
        }
    }
}

/// Forwards the single pre-computed hash every key and probe writes.
///
/// Tinygrad's `ucache` has the same property for free: its key is a tuple of
/// pointers hashed by CPython's identity hash.
#[derive(Default)]
struct PrecomputedHasher(u64);

impl Hasher for PrecomputedHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("intern keys must write exactly one pre-computed u64");
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

type PrecomputedHash = std::hash::BuildHasherDefault<PrecomputedHasher>;

/// Table entry: the node's intern hash plus a weak handle on the node.
struct InternKey {
    hash: u64,
    node: Weak<UOp>,
}

impl Hash for InternKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

/// Two entries are equal when they are the same allocation or both nodes are
/// alive and structurally equal. A dead entry equals nothing but itself, so a
/// node created while its predecessor is mid-teardown never aliases it.
impl PartialEq for InternKey {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.node, &other.node)
            || match (self.node.upgrade(), other.node.upgrade()) {
                (Some(a), Some(b)) => same_structure(&a, &b),
                _ => false,
            }
    }
}

impl Eq for InternKey {}

fn same_structure(a: &UOp, b: &UOp) -> bool {
    a.dtype == b.dtype && a.tag == b.tag && a.origin() == b.origin() && a.op == b.op
}

/// Lookup probe for a candidate that has not been allocated yet.
struct Probe<'a> {
    hash: u64,
    op: &'a Op,
    dtype: &'a DType,
    tag: &'a Tag,
    origin: Option<OriginId>,
}

impl Hash for Probe<'_> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl Equivalent<InternKey> for Probe<'_> {
    fn equivalent(&self, key: &InternKey) -> bool {
        key.node.upgrade().is_some_and(|node| {
            node.dtype == *self.dtype && node.tag == *self.tag && node.origin() == self.origin && node.op == *self.op
        })
    }
}

/// Removal probe: matches an entry by allocation identity, live or dead.
struct ByPtr {
    hash: u64,
    ptr: *const UOp,
}

impl Hash for ByPtr {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl Equivalent<InternKey> for ByPtr {
    fn equivalent(&self, key: &InternKey) -> bool {
        std::ptr::eq(Weak::as_ptr(&key.node), self.ptr)
    }
}

static UOPS: OnceLock<HashMap<InternKey, (), PrecomputedHash>> = OnceLock::new();

fn uops() -> &'static HashMap<InternKey, (), PrecomputedHash> {
    UOPS.get_or_init(HashMap::default)
}

/// Get the set of IDs for UOps currently alive in the cache.
///
/// This is used by kernel cache GC to determine which compiled kernels
/// can be safely removed (those whose AST IDs are no longer live).
pub fn live_uop_ids() -> std::collections::HashSet<u64> {
    let map = uops();
    let guard = map.guard();
    map.keys(&guard).filter_map(|key| key.node.upgrade().map(|arc| arc.id)).collect()
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
    #[inline]
    pub fn new(op: Op, dtype: DType) -> Arc<Self> {
        Self::new_tagged(op, dtype, None)
    }

    /// Create a UOp with an explicit tag (Tinygrad: `UOp(op, dtype, src, arg, tag)`).
    /// Tag participates in hash consing — same structure + different tag = different UOp.
    #[inline]
    pub fn new_tagged(op: Op, dtype: DType, tag: Tag) -> Arc<Self> {
        Self::new_with_origin(op, dtype, tag, origin::current())
    }

    /// Create a UOp under an explicit origin instead of the ambient scope. Rewrites
    /// use this to carry a node's origin across a rebuild.
    pub fn new_with_origin(op: Op, dtype: DType, tag: Tag, origin: Option<OriginId>) -> Arc<Self> {
        use papaya::{Compute, Operation};

        if let Op::Load(ops::Load { index, alt, gate }) = &op {
            assert_eq!(dtype, index.dtype(), "LOAD dtype must match its address dtype");
            assert_eq!(alt.is_some(), gate.is_some(), "LOAD requires either index only or index, alt, and gate");
            if let (Some(alt), Some(gate)) = (alt, gate) {
                assert_eq!(gate.dtype(), DType::Bool, "LOAD gate must have bool dtype");
                assert!(Self::is_invalid_marker(alt) || alt.dtype() == dtype, "LOAD alt dtype must match LOAD dtype");
            }
        }

        let origin = origin.filter(|_| !origin_opaque(&op, &dtype));
        let content_hash = content_hash(&op, &dtype, origin);
        let hash = intern_hash(content_hash, &tag);
        let guard = uops().guard();

        // Fast path: a live structurally equal node already exists. This branch is
        // the majority of the ~1M `UOp::new` calls in one resnet50 schedule.
        if let Some((key, _)) = uops().get_key_value(&Probe { hash, op: &op, dtype: &dtype, tag: &tag, origin }, &guard)
            && let Some(arc) = key.node.upgrade()
        {
            return arc;
        }

        let mut src_ops = crate::op::OpMask::EMPTY;
        op.map_child(|child| src_ops = src_ops.union(crate::op::OpMask::of_op(child.op())));

        let new_arc = Arc::new(Self::fresh(op, dtype, tag, origin, content_hash, src_ops, None));
        let result = uops().compute(
            InternKey { hash, node: Arc::downgrade(&new_arc) },
            |entry| match entry.and_then(|(existing, _)| existing.node.upgrade()) {
                Some(existing) => Operation::Abort(existing),
                None => Operation::Insert(()),
            },
            &guard,
        );

        match result {
            Compute::Aborted(existing) => existing,
            _ => new_arc,
        }
    }

    /// Attach metadata to this UOp, creating a new instance.
    ///
    /// Metadata is NOT part of hash consing - this method creates a new UOp
    /// with a different ID but the same operation structure. This allows
    /// attaching metadata (like kernel info) after optimization.
    pub fn with_metadata<T: std::any::Any + Send + Sync + 'static>(self: &Arc<Self>, metadata: T) -> Arc<Self> {
        self.with_metadata_raw(Arc::new(metadata))
    }

    /// Get metadata of a specific type if it exists.
    ///
    /// Returns `None` if no metadata is attached or if the metadata is of a different type.
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
    /// The result is deliberately not interned: it shares structure and content
    /// hash with `self` but carries its own identity.
    pub fn with_metadata_raw(self: &Arc<Self>, metadata: Arc<dyn std::any::Any + Send + Sync>) -> Arc<Self> {
        Arc::new(Self::fresh(
            self.op.clone(),
            self.dtype.clone(),
            self.tag.clone(),
            self.origin,
            self.content_hash,
            self.src_ops,
            Some(metadata),
        ))
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
        // Retire the intern entry by allocation identity. Nodes that were never
        // interned (metadata copies, losers of an insertion race) simply miss.
        uops().pin().remove(&ByPtr { hash: intern_hash(self.content_hash, &self.tag), ptr: self });

        // Buffer nodes only: the hook exists for buffer-lifetime tracking, and
        // graph rewriting churns millions of transient non-buffer nodes per
        // prepare — their drop must stay allocation-free and branch-cheap.
        if matches!(self.op, Op::Buffer(..))
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
