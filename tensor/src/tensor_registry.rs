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
//! # Memory Management
//!
//! Tensors are stored as `Arc<TensorEntry>`. Call `gc_unused_tensors()` after operations
//! complete to clean up tensors only referenced by the registry.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

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
pub struct TensorEntry {
    /// Unique tensor ID (stable across UOp updates).
    pub id: u64,
    /// The computation graph (mutable for global substitution).
    pub uop: RwLock<Arc<UOp>>,
}

// Global tensor registry using lock-free concurrent HashMap.
//
// Maps tensor_id → Arc<TensorEntry>.
// Tensors live until explicitly removed via gc_unused_tensors().
static TENSORS: OnceLock<PapayaMap<u64, Arc<TensorEntry>>> = OnceLock::new();

fn tensors() -> &'static PapayaMap<u64, Arc<TensorEntry>> {
    TENSORS.get_or_init(PapayaMap::new)
}

/// Register a new tensor and return its entry.
///
/// Thread-safe: each call creates a unique tensor ID.
///
/// # Arguments
///
/// * `uop` - The tensor's computation graph
///
/// # Returns
///
/// Arc to the registered TensorEntry
pub fn register_tensor(uop: Arc<UOp>) -> Arc<TensorEntry> {
    let id = next_tensor_id();
    let entry = Arc::new(TensorEntry { id, uop: RwLock::new(uop) });

    let guard = tensors().guard();
    tensors().insert(id, entry.clone(), &guard);

    entry
}

/// Get a tensor entry by ID.
///
/// Thread-safe read operation.
pub fn get_tensor(id: u64) -> Option<Arc<TensorEntry>> {
    let guard = tensors().guard();
    tensors().get(&id, &guard).cloned()
}

/// Remove a tensor from the registry.
///
/// Thread-safe removal operation.
pub fn remove_tensor(id: u64) {
    let guard = tensors().guard();
    tensors().remove(&id, &guard);
}

/// Remove tensors that are only referenced by the registry (strong_count == 1).
///
/// Call this after tensor operations complete to free memory.
/// Tensors still referenced elsewhere will be kept.
pub fn gc_unused_tensors() {
    let map = tensors();
    let guard = map.guard();

    // Collect IDs to remove (can't mutate while iterating)
    let to_remove: Vec<u64> =
        map.iter(&guard).filter(|(_, arc)| Arc::strong_count(arc) == 1).map(|(k, _)| *k).collect();

    // Remove dead entries
    for id in to_remove {
        map.remove(&id, &guard);
    }
}

/// Clear all tensors from the registry.
///
/// Primarily useful for testing to ensure test isolation.
pub fn clear_all() {
    let guard = tensors().guard();
    tensors().clear(&guard);
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

    // Phase 1: Find affected tensors (read-only scan)
    let affected: Vec<Arc<TensorEntry>> = map
        .iter(&guard)
        .filter_map(|(_, entry)| {
            let uop = entry.uop.read();
            // Check if tensor's root UOp is in map
            if becomes_map.contains_key(&UOpKey(uop.clone())) {
                return Some(entry.clone());
            }
            // Check if any node in the graph is in map
            if uop.toposort().iter().any(|n| becomes_map.contains_key(&UOpKey(n.clone()))) {
                return Some(entry.clone());
            }
            None
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
        clear_all();

        let uop = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let entry = register_tensor(uop.clone());

        let retrieved = get_tensor(entry.id).expect("Should find tensor");
        assert_eq!(retrieved.id, entry.id);
        assert!(Arc::ptr_eq(&*retrieved.uop.read(), &uop));
    }

    #[test]
    fn test_apply_map_updates_tensors() {
        clear_all();

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
