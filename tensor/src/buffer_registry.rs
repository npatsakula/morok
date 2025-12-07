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

use std::sync::OnceLock;

use morok_device::Buffer;
use papaya::HashMap;

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
        return Ok(buf.clone());
    }

    // Slow path: create buffer (expensive operation)
    let buffer = create_fn()?;

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
    buffers().get(&uop_id, &guard).cloned()
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
