//! Global tensor registry for atomic graph substitution.
//!
//! This module implements Tinygrad's `all_tensors` pattern using papaya's lock-free HashMap.
//! When rangeify transforms a UOp (e.g., NEG → BUFFERIZE(NEG)), the `becomes_map` must be
//! applied to ALL tensors that reference it - not just the one being realized.
//!
//! Without this, diamond patterns (like argmin's NEG feeding both MAX and EQ) fail because
//! different consumers see different versions of the same producer.
//!
//! # Thread Safety
//!
//! All operations are lock-free and thread-safe. Uses papaya's epoch-based reclamation
//! for concurrent access and parking_lot::RwLock for interior UOp mutation.
//!
//! # Memory Management (Tinygrad-aligned)
//!
//! Tensors are stored as `Weak<TensorEntry>` in the registry. When all strong references
//! (held by `Tensor` structs) are dropped, the entry becomes eligible for cleanup.
//! Dead weak refs are cleaned lazily on access or via `gc_dead_refs()`.
//!
//! This matches Tinygrad's `weakref.WeakKeyDictionary` pattern - no manual cleanup required.
//!
//! # Buffer Storage (Tinygrad-aligned)
//!
//! Buffers are stored in a separate map (`BUFFERS`) indexed by UOp ID.
//! This matches Tinygrad's `buffers: weakref.WeakKeyDictionary[UOp, Buffer]` pattern:
//! - Key is UOp ID (analogous to weak UOp reference)
//! - Value is Arc<Buffer> (strong reference keeps buffer alive)
//! - When UOp is dropped, buffer entry cleaned up via `gc_dead_refs()`
//!
//! TensorEntry also caches the buffer for direct access via tensor.buffer().

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use morok_device::Buffer;
use morok_ir::{Op, UOp, UOpKey};
use papaya::HashMap as PapayaMap;
use parking_lot::RwLock;

/// Atomic counter for unique tensor IDs.
static TENSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_tensor_id() -> u64 {
    TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Entry in the global tensor registry.
///
/// Uses RwLock for interior mutability of the UOp during global substitution.
/// The RwLock allows concurrent reads (typical tensor operations) with exclusive
/// writes only during `apply_map_to_tensors`.
///
/// Buffer can be set at construction (input tensors) or later (realized tensors).
/// Uses OnceLock for thread-safe one-time initialization.
pub struct TensorEntry {
    /// Unique tensor ID (stable across UOp updates).
    pub id: u64,
    /// The computation graph (mutable for global substitution).
    pub uop: RwLock<Arc<UOp>>,
    /// The materialized buffer (can be set once via OnceLock).
    buffer: OnceLock<Arc<Buffer>>,
}

impl TensorEntry {
    /// Get the buffer if materialized.
    pub fn buffer(&self) -> Option<&Arc<Buffer>> {
        self.buffer.get()
    }

    /// Set the buffer (can only be called once, subsequent calls are no-ops).
    /// Returns true if buffer was set, false if already set.
    pub fn set_buffer(&self, buffer: Arc<Buffer>) -> bool {
        self.buffer.set(buffer).is_ok()
    }
}

// Global tensor registry using lock-free concurrent HashMap.
//
// Design: Stores Weak<TensorEntry> for automatic memory management (Tinygrad-aligned).
// - Tensor structs hold Arc<TensorEntry> (strong refs)
// - Registry holds Weak<TensorEntry> (weak refs)
// - When Tensor is dropped, TensorEntry can be cleaned up
// - Dead weak refs cleaned lazily on access or via gc_dead_refs()
static TENSORS: OnceLock<PapayaMap<u64, Weak<TensorEntry>>> = OnceLock::new();

// Direct buffer storage: UOp ID → Arc<Buffer> (Tinygrad-aligned).
//
// This is Morok's equivalent of Tinygrad's `buffers: weakref.WeakKeyDictionary[UOp, Buffer]`.
// - Buffers stored independently of TensorEntry
// - Stays alive as long as mapping exists
// - Cleaned up when corresponding UOp is dropped (via gc_dead_refs)
static BUFFERS: OnceLock<PapayaMap<u64, Arc<Buffer>>> = OnceLock::new();

// Secondary index: UOp ID → Tensor ID.
// Used for tensor lookups during schedule creation.
// Entries become stale when corresponding tensor is dropped - cleaned lazily.
static UOP_TO_TENSOR: OnceLock<PapayaMap<u64, u64>> = OnceLock::new();

fn tensors() -> &'static PapayaMap<u64, Weak<TensorEntry>> {
    TENSORS.get_or_init(PapayaMap::new)
}

fn buffers() -> &'static PapayaMap<u64, Arc<Buffer>> {
    BUFFERS.get_or_init(PapayaMap::new)
}

fn uop_to_tensor() -> &'static PapayaMap<u64, u64> {
    UOP_TO_TENSOR.get_or_init(PapayaMap::new)
}

/// Register a new tensor without buffer (for lazy computation graphs).
///
/// Thread-safe: each call creates a unique tensor ID.
/// The registry stores a weak reference; the caller holds the strong reference.
///
/// # Arguments
///
/// * `uop` - The tensor's computation graph
///
/// # Returns
///
/// Arc to the registered TensorEntry (caller owns the strong reference)
pub fn register_tensor(uop: Arc<UOp>) -> Arc<TensorEntry> {
    let id = next_tensor_id();
    let entry = Arc::new(TensorEntry { id, uop: RwLock::new(uop), buffer: OnceLock::new() });

    // Store weak ref in registry - entry stays alive as long as caller holds Arc
    let guard = tensors().guard();
    tensors().insert(id, Arc::downgrade(&entry), &guard);

    entry
}

/// Register a new tensor with buffer (for input tensors and realized tensors).
///
/// Stores buffer in both:
/// 1. BUFFERS map (indexed by UOp ID) - for schedule buffer lookups
/// 2. TensorEntry.buffer - for direct tensor access
///
/// The registry stores a weak reference; the caller holds the strong reference.
///
/// # Arguments
///
/// * `uop` - The tensor's computation graph
/// * `buffer` - The materialized buffer
/// * `buffer_uop_id` - The UOp ID to index under (for lookups)
///
/// # Returns
///
/// Arc to the registered TensorEntry (caller owns the strong reference)
pub fn register_tensor_with_buffer(uop: Arc<UOp>, buffer: Arc<Buffer>, buffer_uop_id: u64) -> Arc<TensorEntry> {
    let id = next_tensor_id();
    let entry = Arc::new(TensorEntry { id, uop: RwLock::new(uop), buffer: OnceLock::from(buffer.clone()) });

    // Store weak ref in tensor registry
    let guard = tensors().guard();
    tensors().insert(id, Arc::downgrade(&entry), &guard);

    // Store buffer directly indexed by UOp ID (Tinygrad-aligned)
    let buf_guard = buffers().guard();
    buffers().insert(buffer_uop_id, buffer, &buf_guard);

    // Also map UOp ID → Tensor ID for tensor lookups
    let uop_guard = uop_to_tensor().guard();
    uop_to_tensor().insert(buffer_uop_id, id, &uop_guard);

    entry
}

/// Get buffer by UOp ID.
///
/// Direct lookup from BUFFERS map (Tinygrad-aligned).
/// Used by collect_input_buffers() during schedule creation.
/// Returns None if no buffer registered for this UOp ID.
pub fn get_buffer(uop_id: u64) -> Option<Buffer> {
    let guard = buffers().guard();
    buffers().get(&uop_id, &guard).map(|arc_buf| (**arc_buf).clone())
}

/// Remove buffer and UOp ID → Tensor ID mapping.
///
/// Called during cleanup to free buffer and index entries.
pub fn remove_buffer(uop_id: u64) {
    // Remove from BUFFERS map
    let buf_guard = buffers().guard();
    buffers().remove(&uop_id, &buf_guard);

    // Remove from UOp → Tensor index
    let guard = uop_to_tensor().guard();
    uop_to_tensor().remove(&uop_id, &guard);
}

/// Get count of buffers in the registry (for testing).
pub fn buffer_count() -> usize {
    buffers().len()
}

/// Clear all buffers and UOp → Tensor mappings from the index (for testing).
pub fn clear_buffer_index() {
    // Clear buffers
    let buf_guard = buffers().guard();
    buffers().clear(&buf_guard);

    // Clear UOp → Tensor index
    let guard = uop_to_tensor().guard();
    uop_to_tensor().clear(&guard);
}

/// Register a buffer for an existing tensor.
///
/// Used by realize() to associate output buffers with tensors for schedule lookups.
/// Stores buffer in both BUFFERS map and TensorEntry.
///
/// # Arguments
///
/// * `uop_id` - The UOp ID to index under (for lookups)
/// * `tensor_id` - The tensor ID that owns this buffer
/// * `buffer` - The materialized buffer
pub fn register_buffer(uop_id: u64, tensor_id: u64, buffer: Arc<Buffer>) {
    // Store buffer directly indexed by UOp ID (Tinygrad-aligned)
    let buf_guard = buffers().guard();
    buffers().insert(uop_id, buffer.clone(), &buf_guard);

    // Also set buffer on the TensorEntry for direct tensor access
    if let Some(entry) = get_tensor(tensor_id) {
        entry.set_buffer(buffer);
    }

    // Map UOp ID → Tensor ID for tensor lookups
    let guard = uop_to_tensor().guard();
    uop_to_tensor().insert(uop_id, tensor_id, &guard);
}

/// Get a tensor entry by ID.
///
/// Thread-safe read operation. Returns None if tensor was dropped.
pub fn get_tensor(id: u64) -> Option<Arc<TensorEntry>> {
    let guard = tensors().guard();
    tensors().get(&id, &guard)?.upgrade()
}

/// Remove a tensor from the registry.
///
/// Thread-safe removal operation.
pub fn remove_tensor(id: u64) {
    let guard = tensors().guard();
    tensors().remove(&id, &guard);
}

/// Remove dead weak references and stale buffers from the registry.
///
/// This is optional - dead refs are cleaned lazily on access.
/// Call this to proactively free registry memory.
///
/// Buffers are cleaned based on UOp liveness (matching Tinygrad's WeakKeyDictionary
/// pattern where buffer entries are removed when the UOp key is garbage collected).
pub fn gc_dead_refs() {
    // Get live UOp IDs from the UOp cache
    let live_uop_ids = morok_ir::uop::live_uop_ids();

    // Clean dead tensor weak refs
    let map = tensors();
    let guard = map.guard();
    let to_remove: Vec<u64> =
        map.iter(&guard).filter(|(_, weak)| weak.upgrade().is_none()).map(|(k, _)| *k).collect();
    for id in to_remove {
        map.remove(&id, &guard);
    }

    // Clean stale buffers (UOp no longer alive)
    let buf_map = buffers();
    let buf_guard = buf_map.guard();
    let stale_bufs: Vec<u64> =
        buf_map.iter(&buf_guard).filter(|(uop_id, _)| !live_uop_ids.contains(uop_id)).map(|(id, _)| *id).collect();
    for uop_id in stale_bufs {
        buf_map.remove(&uop_id, &buf_guard);
    }

    // Clean stale UOp→Tensor mappings
    let uop_map = uop_to_tensor();
    let uop_guard = uop_map.guard();
    let stale_uop_ids: Vec<u64> = uop_map
        .iter(&uop_guard)
        .filter(|(uop_id, tensor_id)| {
            // Remove if UOp is dead OR tensor is dead
            !live_uop_ids.contains(uop_id)
                || tensors().get(*tensor_id, &guard).is_none_or(|w| w.upgrade().is_none())
        })
        .map(|(uop_id, _)| *uop_id)
        .collect();
    for uop_id in stale_uop_ids {
        uop_map.remove(&uop_id, &uop_guard);
    }
}

/// Legacy alias for gc_dead_refs (for compatibility).
///
/// With weak references, tensors are automatically cleaned up when no longer
/// referenced. This function now just cleans up dead weak refs in the registry.
#[deprecated(note = "Tensor registry now uses weak refs - cleanup is automatic. Use gc_dead_refs() to clean registry.")]
pub fn gc_unused_tensors() {
    gc_dead_refs();
}

/// Clear all tensors, buffers, and mappings from the registry.
///
/// Primarily useful for testing to ensure test isolation.
pub fn clear_all() {
    // Clear tensors
    let guard = tensors().guard();
    tensors().clear(&guard);

    // Clear buffers
    let buf_guard = buffers().guard();
    buffers().clear(&buf_guard);

    // Clear UOp → Tensor index
    let uop_guard = uop_to_tensor().guard();
    uop_to_tensor().clear(&uop_guard);
}

/// Apply a transformation map to ALL live tensors globally.
///
/// This is Morok's equivalent of Tinygrad's `_apply_map_to_tensors`.
/// When rangeify creates a becomes_map (old UOp → new UOp), this function
/// ensures ALL tensors see the same transformed versions.
///
/// # Arguments
///
/// * `becomes_map` - Mapping from original UOps to their transformed versions
///
/// # Thread Safety
///
/// This function acquires write locks on affected tensors during the update phase.
/// Other tensors can still be read/written concurrently.
#[allow(clippy::mutable_key_type)]
pub fn apply_map_to_tensors(becomes_map: &HashMap<UOpKey, Arc<UOp>>) {
    if becomes_map.is_empty() {
        return;
    }

    let map = tensors();
    let guard = map.guard();

    // Phase 1: Find affected tensors (read-only scan, skip dead weak refs)
    let affected: Vec<Arc<TensorEntry>> = map
        .iter(&guard)
        .filter_map(|(_, weak)| {
            let entry = weak.upgrade()?; // Skip dead entries
            let is_affected = {
                let uop = entry.uop.read();
                // Check if tensor's root UOp is in map
                if becomes_map.contains_key(&UOpKey(uop.clone())) {
                    true
                } else {
                    // Check if any node in the graph is in map
                    uop.toposort().iter().any(|n| becomes_map.contains_key(&UOpKey(n.clone())))
                }
            }; // uop lock dropped here
            if is_affected { Some(entry) } else { None }
        })
        .collect();

    if affected.is_empty() {
        return;
    }

    // Phase 2: Create SINK of affected tensor UOps
    let sources: Vec<Arc<UOp>> = affected.iter().map(|e| e.uop.read().clone()).collect();
    let sink = UOp::sink(sources.clone());

    // Phase 3: Atomic substitution across all affected UOps
    let new_sink = sink.substitute(becomes_map);

    // Phase 4: Update each tensor's UOp (acquires write locks)
    if let Op::Sink { sources: new_sources } = new_sink.op() {
        for (entry, (old, new)) in affected.iter().zip(sources.iter().zip(new_sources.iter())) {
            if !Arc::ptr_eq(old, new) {
                *entry.uop.write() = new.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_dtype::DType;
    use morok_ir::ConstValue;

    #[test]
    fn test_register_and_get() {
        let _guard = crate::test::helpers::test_setup();

        let uop = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let entry = register_tensor(uop.clone());

        let retrieved = get_tensor(entry.id).expect("Should find tensor");
        assert_eq!(retrieved.id, entry.id);
        assert!(Arc::ptr_eq(&*retrieved.uop.read(), &uop));
    }

    #[test]
    fn test_apply_map_updates_tensors() {
        let _guard = crate::test::helpers::test_setup();

        // Create two tensors sharing a common UOp
        let shared = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let t1_uop = shared.neg();
        let t2_uop = shared.neg(); // Same as t1_uop due to hash consing

        let t1 = register_tensor(t1_uop.clone());
        let t2 = register_tensor(t2_uop.clone());

        // Create a replacement for the shared const
        let replacement = UOp::const_(DType::Float32, ConstValue::Float(2.0));

        #[allow(clippy::mutable_key_type)]
        let mut becomes_map = HashMap::new();
        becomes_map.insert(UOpKey(shared.clone()), replacement.clone());

        // Apply the map
        apply_map_to_tensors(&becomes_map);

        // Both tensors should now reference the replacement
        let t1_new = t1.uop.read();
        let t2_new = t2.uop.read();

        // The root NEG should now have the replacement as its source
        assert!(!Arc::ptr_eq(&*t1_new, &t1_uop), "t1 should be updated");
        assert!(!Arc::ptr_eq(&*t2_new, &t2_uop), "t2 should be updated");
    }
}
