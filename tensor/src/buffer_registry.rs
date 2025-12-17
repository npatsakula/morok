//! Buffer registry for UOp ID → Buffer mapping.
//!
//! This module provides a global concurrent registry that maps UOp IDs to allocated buffers.
//! Uses papaya's lock-free HashMap for thread-safe access across parallel tensor operations.
//!
//! # Thread Safety
//!
//! All operations are thread-safe. Multiple threads can allocate and access buffers
//! concurrently without explicit synchronization.
//!
//! # Memory Management
//!
//! Buffers live as long as they are in the registry. Call `remove_buffer()` or
//! `clear_all()` to free them explicitly.
//!
//! For automatic cleanup, use [`BufferScope`] which removes non-persistent buffers
//! when dropped.

use std::collections::HashSet;
use std::sync::OnceLock;

use morok_device::Buffer;
use papaya::HashMap;
use tracing::trace;

use crate::Result;

// Global buffer registry using lock-free concurrent HashMap.
//
// Maps UOp ID -> Buffer for materialized tensors.
// Buffers live until explicitly removed via remove_buffer() or clear_all().
static BUFFERS: OnceLock<HashMap<u64, Buffer>> = OnceLock::new();

fn buffers() -> &'static HashMap<u64, Buffer> {
    BUFFERS.get_or_init(HashMap::new)
}

/// Get or create buffer for a UOp.
///
/// Thread-safe: if multiple threads call this with the same `uop_id` concurrently,
/// exactly one will create the buffer (the one whose create_fn succeeds first),
/// and all others will receive a clone of that buffer.
///
/// # Arguments
///
/// * `uop_id` - The UOp ID to look up
/// * `create_fn` - Function to create the buffer if not found
///
/// # Returns
///
/// The buffer (either from cache or freshly created)
///
/// # Errors
///
/// Returns error if buffer creation fails
pub fn get_or_create_buffer<F>(uop_id: u64, create_fn: F) -> Result<Buffer>
where
    F: FnOnce() -> Result<Buffer>,
{
    let map = buffers();
    let guard = map.guard();

    // Fast path: buffer already exists
    if let Some(buf) = map.get(&uop_id, &guard) {
        trace!(uop_id, buffer.id = buf.id().0, "get_or_create_buffer: existing");
        return Ok(buf.clone());
    }

    // Slow path: create buffer (expensive operation)
    let buffer = create_fn()?;

    trace!(uop_id, buffer.id = buffer.id().0, "get_or_create_buffer: creating");

    // Atomic insert - if another thread beat us, use their buffer
    // papaya's get_or_insert_with would be cleaner but create_fn is fallible
    use papaya::{Compute, Operation};
    match map.compute(
        uop_id,
        |entry| match entry {
            Some((_, existing)) => Operation::Abort(existing.clone()),
            None => Operation::Insert(buffer.clone()),
        },
        &guard,
    ) {
        Compute::Inserted(_, buf) => Ok(buf.clone()),
        Compute::Aborted(buf) => Ok(buf),
        _ => Ok(buffer),
    }
}

/// Get existing buffer (returns None if not found).
///
/// Thread-safe read operation.
///
/// # Arguments
///
/// * `uop_id` - The UOp ID to look up
///
/// # Returns
///
/// The buffer if found, None otherwise
pub fn get_buffer(uop_id: u64) -> Option<Buffer> {
    let guard = buffers().guard();
    let result = buffers().get(&uop_id, &guard).cloned();
    if let Some(ref buf) = result {
        trace!(uop_id, buffer.id = buf.id().0, buffer.size = buf.size(), "get_buffer: found");
    } else {
        trace!(uop_id, "get_buffer: not found");
    }
    result
}

/// Remove buffer (for explicit cleanup).
///
/// Thread-safe removal operation.
///
/// # Arguments
///
/// * `uop_id` - The UOp ID to remove
pub fn remove_buffer(uop_id: u64) {
    let guard = buffers().guard();
    buffers().remove(&uop_id, &guard);
}

/// Clear all buffers from the registry.
///
/// This is primarily useful for testing to ensure test isolation.
/// Thread-safe.
pub fn clear_all() {
    let guard = buffers().guard();
    buffers().clear(&guard);
}

/// Get the count of buffers in the registry.
///
/// Useful for debugging and testing memory management.
pub fn buffer_count() -> usize {
    buffers().len()
}

// ============================================================================
// BUFFER SCOPE (RAII CLEANUP)
// ============================================================================

/// RAII scope for automatic buffer cleanup.
///
/// When a `BufferScope` is dropped, it removes all registered buffers from the
/// global registry except those marked as persistent. This enables automatic
/// cleanup of intermediate buffers after execution.
///
/// # Example
///
/// ```ignore
/// let mut scope = BufferScope::new();
///
/// // Register intermediate buffers
/// scope.register(intermediate_id);
///
/// // Mark output buffer as persistent
/// scope.mark_persistent(output_id);
///
/// // When scope is dropped, only intermediate_id is removed
/// ```
#[derive(Debug, Default)]
pub struct BufferScope {
    /// IDs of buffers registered in this scope.
    registered: Vec<u64>,
    /// IDs of buffers that should persist (not removed on drop).
    persistent: HashSet<u64>,
}

impl BufferScope {
    /// Create a new empty buffer scope.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a buffer ID in this scope.
    ///
    /// When the scope is dropped, this buffer will be removed from the registry
    /// unless it has been marked as persistent.
    pub fn register(&mut self, uop_id: u64) {
        self.registered.push(uop_id);
    }

    /// Mark a buffer as persistent (won't be cleaned up when scope is dropped).
    ///
    /// Use this for output buffers that need to survive after execution.
    pub fn mark_persistent(&mut self, uop_id: u64) {
        self.persistent.insert(uop_id);
    }

    /// Get the number of registered buffers.
    pub fn registered_count(&self) -> usize {
        self.registered.len()
    }

    /// Get the number of persistent buffers.
    pub fn persistent_count(&self) -> usize {
        self.persistent.len()
    }

    /// Manually release non-persistent buffers without dropping the scope.
    ///
    /// After calling this, the scope is empty and can be reused.
    pub fn release(&mut self) {
        for id in self.registered.drain(..) {
            if !self.persistent.contains(&id) {
                trace!(uop_id = id, "BufferScope: releasing buffer");
                remove_buffer(id);
            }
        }
        self.persistent.clear();
    }
}

impl Drop for BufferScope {
    fn drop(&mut self) {
        for &id in &self.registered {
            if !self.persistent.contains(&id) {
                trace!(uop_id = id, "BufferScope: releasing buffer on drop");
                remove_buffer(id);
            }
        }
    }
}
