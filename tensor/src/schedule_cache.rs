//! Schedule-level cache for the tensor realize pipeline.
//!
//! Caches the expensive rangeify + try_get_kernel_graph pipeline output so that
//! structurally identical computations (same shape, different data) skip
//! those passes and go straight to buffer allocation + codegen.
//!
//! The per-kernel OPT_CACHE in `prepare_execution_plan` already caches
//! optimization + compilation. This cache sits one level above, deduplicating
//! the schedule creation step itself.

use std::sync::{Arc, OnceLock};

use papaya::HashMap;

use morok_ir::UOp;

/// Compute a deterministic content hash for a UOp tree.
///
/// Unlike UOp IDs (which depend on allocation order and weak-ref lifetimes),
/// this hash is purely structural — identical trees always produce the same hash.
/// Matches the role of Tinygrad's `UOp.key` (SHA-256 of tree structure).
pub(crate) fn content_hash(uop: &UOp) -> u64 {
    uop.content_hash
}

/// Cache key: (content_hash of normalized sink, codegen backend string).
type ScheduleCacheKey = (u64, &'static str);

/// Cached output of rangeify + try_get_kernel_graph pipeline.
///
/// Contains the kernel graph, but NOT buffer allocations.
/// Buffer allocation happens fresh each time in `create_schedule`, because
/// buffers are tensor-specific (different input data).
pub(crate) struct CachedSchedule {
    /// Pre-schedule artifact after rangeify + callable extraction.
    ///
    /// Stores callable AST/dependency structure (without concrete buffers),
    /// mirroring Tinygrad's cached `pre_schedule` stage.
    pub pre_schedule: Arc<crate::schedule::PreSchedule>,
}

/// Global schedule-level cache.
///
/// Keyed by (content_hash, codegen_backend) so identical computations
/// on the same backend share the cached pipeline output.
static SCHEDULE_CACHE: OnceLock<HashMap<ScheduleCacheKey, Arc<CachedSchedule>>> = OnceLock::new();

pub(crate) fn schedule_cache() -> &'static HashMap<ScheduleCacheKey, Arc<CachedSchedule>> {
    SCHEDULE_CACHE.get_or_init(HashMap::new)
}

/// Compute the cache key for a tensor + config.
///
/// Uses pre-schedule cache normalization (BUFFER->PARAM + strip BIND
/// runtime values), then content-hashes the result.
#[cfg(test)]
pub(crate) fn cache_key_for(tensor: &crate::Tensor, config: &crate::PrepareConfig) -> Option<(u64, &'static str)> {
    let sink = UOp::sink(vec![tensor.uop().contiguous()]);
    let normalized = crate::realize::normalize_for_schedule_cache(&sink).ok()?;
    let param_buffers = normalized.param_buffers;
    let codegen = crate::realize::resolve_codegen(&param_buffers, config).ok()?;
    Some((content_hash(&normalized.normalized), codegen))
}
