//! Buffer registry for UOp ID → Buffer mapping.
//!
//! This module provides a simple thread-local registry that maps UOp IDs to allocated buffers.
//! This replaces the global `BUFFERS` thread-local, providing better encapsulation.

use std::cell::RefCell;
use std::collections::HashMap;

use morok_device::Buffer;

use crate::Result;

thread_local! {
    /// Maps UOp ID -> Buffer for materialized tensors.
    ///
    /// Buffers are NOT weak refs - they live as long as referenced by schedule items
    /// or until explicitly removed.
    static BUFFER_MAP: RefCell<HashMap<u64, Buffer>> = RefCell::new(HashMap::new());
}

/// Get or create buffer for a UOp.
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
    BUFFER_MAP.with(|map| {
        if let Some(buf) = map.borrow().get(&uop_id) {
            return Ok(buf.clone());
        }

        let buffer = create_fn()?;
        map.borrow_mut().insert(uop_id, buffer.clone());
        Ok(buffer)
    })
}

/// Get existing buffer (returns None if not found).
///
/// # Arguments
///
/// * `uop_id` - The UOp ID to look up
///
/// # Returns
///
/// The buffer if found, None otherwise
pub fn get_buffer(uop_id: u64) -> Option<Buffer> {
    BUFFER_MAP.with(|map| map.borrow().get(&uop_id).cloned())
}

/// Remove buffer (for explicit cleanup if needed).
///
/// # Arguments
///
/// * `uop_id` - The UOp ID to remove
pub fn remove_buffer(uop_id: u64) {
    BUFFER_MAP.with(|map| map.borrow_mut().remove(&uop_id));
}

/// Clear all buffers from the registry.
///
/// This is primarily useful for testing to ensure test isolation.
pub fn clear_all() {
    BUFFER_MAP.with(|map| map.borrow_mut().clear());
}
