//! Global kernel deduplication cache.
//!
//! This module provides a simple thread-local cache that maps UOp IDs to compiled kernels.
//! Thanks to hash consing in `ir/src/uop/hash_consing.rs`, identical ASTs automatically
//! have identical IDs, making deduplication trivial.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::LlvmKernel;

/// Cached kernel that can be reused across tensors.
#[derive(Clone)]
pub struct CachedKernel {
    pub kernel: Arc<LlvmKernel>,
    pub device: String,
    pub code: String,
    pub entry_point: String,
}

thread_local! {
    /// Global kernel dedup cache: UOp ID -> cached kernel.
    ///
    /// Thanks to hash consing, identical ASTs have identical IDs!
    /// This allows trivial kernel reuse across different tensors.
    static KERNEL_DEDUP: RefCell<HashMap<u64, Arc<CachedKernel>>> = RefCell::new(HashMap::new());
}

/// Get or compile a kernel by UOp ID.
///
/// # Arguments
///
/// * `ast_id` - The UOp ID of the kernel AST (from hash consing)
/// * `device` - Device string (e.g., "CPU", "CUDA:0")
/// * `compile_fn` - Function to compile the kernel if not cached
///
/// # Returns
///
/// Arc to the cached kernel (either from cache or freshly compiled)
///
/// # Errors
///
/// Returns error if compilation fails
pub fn get_or_compile_kernel<F, E>(
    ast_id: u64,
    device: &str,
    compile_fn: F,
) -> Result<Arc<CachedKernel>, E>
where
    F: FnOnce() -> Result<CachedKernel, E>,
{
    KERNEL_DEDUP.with(|cache| {
        let map = cache.borrow();

        // Check cache
        if let Some(cached) = map.get(&ast_id) {
            // Verify device matches
            if cached.device == device {
                return Ok(cached.clone());
            }
        }

        // Compile fresh
        drop(map);
        let compiled = compile_fn()?;
        let cached = Arc::new(compiled);

        // Store in cache
        cache.borrow_mut().insert(ast_id, cached.clone());
        Ok(cached)
    })
}

/// Clear all cached kernels.
///
/// This is primarily useful for testing to ensure test isolation.
pub fn clear_all() {
    KERNEL_DEDUP.with(|cache| cache.borrow_mut().clear());
}
