//! Global kernel deduplication cache.
//!
//! This module provides a global concurrent cache that maps (UOp ID, device) pairs to compiled kernels.
//! Uses papaya's lock-free HashMap for thread-safe access across parallel tensor operations.
//!
//! # Thread Safety
//!
//! All operations are thread-safe. Multiple threads can look up and compile kernels
//! concurrently without explicit synchronization.
//!
//! # Deduplication
//!
//! Thanks to hash consing in `ir/src/uop/hash_consing.rs`, identical ASTs automatically
//! have identical IDs, making kernel deduplication trivial. The key includes both the
//! AST ID and the device string to support multi-GPU systems where the same kernel
//! might be compiled differently for different devices.

use std::sync::{Arc, OnceLock};

use morok_device::device::Program;
use papaya::HashMap;

/// Cached kernel that can be reused across tensors.
///
/// Note: This struct does not implement Clone because `Box<dyn Program>` is not Clone.
/// Use `Arc<CachedKernel>` for sharing.
pub struct CachedKernel {
    /// The compiled, executable program.
    pub program: Box<dyn Program>,
    /// Device string (e.g., "CPU", "CUDA:0").
    pub device: String,
    /// Generated source code (for debugging/profiling).
    pub code: String,
    /// Entry point name.
    pub entry_point: String,
    /// Global work size for dispatch (GPU backends, CPU threading).
    /// For CPU threading: [thread_count, 1, 1]
    pub global_size: Option<[usize; 3]>,
    /// Local work size for dispatch (GPU backends).
    pub local_size: Option<[usize; 3]>,
}

// SAFETY: CachedKernel is Send + Sync because:
// - Box<dyn Program> is Send + Sync (Program trait requires Send + Sync)
// - String is Send + Sync
unsafe impl Send for CachedKernel {}
unsafe impl Sync for CachedKernel {}

/// Cache key: (AST ID, device string).
///
/// Using both AST ID and device allows the same logical kernel to be compiled
/// differently for different devices (e.g., CPU vs CUDA, or CUDA:0 vs CUDA:1).
type KernelKey = (u64, String);

// Global kernel dedup cache using lock-free concurrent HashMap.
//
// Maps (UOp ID, device) -> Arc<CachedKernel>.
// Kernels live until explicitly cleared via clear_all().
static KERNELS: OnceLock<HashMap<KernelKey, Arc<CachedKernel>>> = OnceLock::new();

fn kernels() -> &'static HashMap<KernelKey, Arc<CachedKernel>> {
    KERNELS.get_or_init(HashMap::new)
}

/// Get or compile a kernel by UOp ID and device.
///
/// Thread-safe: if multiple threads call this with the same key concurrently,
/// exactly one will compile the kernel, and all others will receive a clone
/// of the Arc to that kernel.
///
/// # Arguments
///
/// * `ast_id` - The UOp ID of the kernel AST (from hash consing)
/// * `device` - Device string (e.g., "CPU", "CUDA:0")
/// * `compile_fn` - Function to compile the kernel if not cached
///
/// # Returns
///
/// Arc to the cached kernel (either from cache or freshly compiled).
///
/// # Errors
///
/// Returns error if compilation fails
pub fn get_or_compile_kernel<F, E>(ast_id: u64, device: &str, compile_fn: F) -> Result<Arc<CachedKernel>, E>
where
    F: FnOnce() -> Result<CachedKernel, E>,
{
    let key = (ast_id, device.to_string());
    let map = kernels();
    let guard = map.guard();

    // Fast path: kernel already cached
    if let Some(cached) = map.get(&key, &guard) {
        return Ok(Arc::clone(cached));
    }

    // Slow path: compile kernel (expensive)
    let compiled = compile_fn()?;
    let cached = Arc::new(compiled);

    // Atomic insert - if another thread beat us, use their kernel
    use papaya::{Compute, Operation};
    match map.compute(
        key,
        |entry| match entry {
            Some((_, existing)) => Operation::Abort(Arc::clone(existing)),
            None => Operation::Insert(Arc::clone(&cached)),
        },
        &guard,
    ) {
        Compute::Inserted(_, kernel) => Ok(Arc::clone(kernel)),
        Compute::Aborted(kernel) => Ok(kernel),
        _ => Ok(cached),
    }
}

/// Clear all cached kernels.
///
/// This is primarily useful for testing to ensure test isolation.
/// Thread-safe.
pub fn clear_all() {
    let guard = kernels().guard();
    kernels().clear(&guard);
}

/// Remove kernels whose AST IDs are no longer in the live UOp set.
///
/// Call this after `gc_unused_uops()` to clean up compiled kernels for
/// discarded UOps. This prevents kernel cache memory accumulation during
/// beam search and other optimization passes.
///
/// # Arguments
///
/// * `live_ids` - Set of UOp IDs that are still alive in the UOp cache
///
/// # Example
///
/// ```ignore
/// morok_ir::uop::gc_unused_uops();
/// let live_ids = morok_ir::uop::live_uop_ids();
/// morok_runtime::kernel_cache::gc_unused_kernels(&live_ids);
/// ```
pub fn gc_unused_kernels(live_ids: &std::collections::HashSet<u64>) {
    let map = kernels();
    let guard = map.guard();

    // Collect keys to remove (can't mutate while iterating)
    let to_remove: Vec<KernelKey> =
        map.iter(&guard).filter(|((ast_id, _), _)| !live_ids.contains(ast_id)).map(|(k, _)| k.clone()).collect();

    // Remove dead entries
    for key in to_remove {
        map.remove(&key, &guard);
    }
}
