//! Beam search auto-tuning for kernel optimization.
//!
//! Implements a beam search algorithm that explores the optimization space
//! to find high-performance kernel configurations. This is slower than
//! heuristic-based optimization but can achieve ML-quality performance.
//!
//! # Algorithm
//!
//! 1. Start with base scheduler
//! 2. Generate all valid actions (OptOps applications)
//! 3. Compile and time each candidate
//! 4. Keep top K (beam width) by timing
//! 5. Repeat until no improvement or timeout
//!
//! # Caching
//!
//! Results are cached to disk using sled. The cache key is a hash of
//! (ast_hash, beam_width, device_name). Caching can be disabled via
//! the IGNORE_BEAM_CACHE environment variable.

use std::sync::Arc;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;

use morok_ir::{AxisType, ConstValue, Op, UOp};

use super::Scheduler;
use super::config::BeamConfig;
use super::error::*;
use super::opts::apply_opt;
use super::types::Opt;

/// Minimum measurable improvement before BEAM stops iterating.
///
/// Default 10 ns — below typical measurement noise. Override via
/// `MOROK_BEAM_MIN_PROGRESS` (nanoseconds; set to `0` to disable).
fn beam_min_progress() -> Duration {
    static CACHED: Lazy<Duration> = Lazy::new(|| {
        let nanos: u64 = std::env::var("MOROK_BEAM_MIN_PROGRESS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
        Duration::from_nanos(nanos)
    });
    *CACHED
}

// ============================================================================
// ACTION SPACE
// ============================================================================

/// Thread-count amounts considered by beam search.
///
/// Static set `[2,3,4,5,8,12,16,24,32,64]` filtered by `max_threads`. We don't
/// pre-filter by divisor patterns — `apply_thread` enforces divisibility against
/// the chosen axis at apply time, and the true divisibility depends on
/// post-action shape.
fn thread_action_amounts(max_threads: usize) -> Vec<usize> {
    const AMOUNTS: [usize; 10] = [2, 3, 4, 5, 8, 12, 16, 24, 32, 64];
    AMOUNTS.iter().copied().filter(|&t| t <= max_threads).collect()
}

/// Pre-computed action space for beam search (~500 actions).
pub static BEAM_ACTIONS: Lazy<Vec<Opt>> = Lazy::new(|| {
    let mut actions = Vec::with_capacity(600);

    // UPCAST: axes 0-7, amounts [0, 2, 3, 4, 5, 7]
    // amount=0 means "full size" - handled specially in apply
    for axis in 0..8 {
        for &amt in &[0, 2, 3, 4, 5, 7] {
            actions.push(Opt::upcast(axis, amt));
        }
    }

    // UNROLL: axes 0-4, amounts [0, 4, 7]
    for axis in 0..5 {
        for &amt in &[0, 4, 7] {
            actions.push(Opt::unroll(axis, amt));
        }
    }

    // LOCAL: axes 0-5, amounts [2, 3, 4, 8, 13, 16, 29]
    for axis in 0..6 {
        for &amt in &[2, 3, 4, 8, 13, 16, 29] {
            actions.push(Opt::local(axis, amt));
        }
    }
    // Hand-tuned LOCAL extras outside the grid.
    actions.push(Opt::local(0, 32));
    actions.push(Opt::local(6, 2));

    // GROUPTOP: axes 0-2, amounts [13, 16, 28, 29, 32, 49, 64, 256]
    for axis in 0..3 {
        for &amt in &[13, 16, 28, 29, 32, 49, 64, 256] {
            actions.push(Opt::grouptop(axis, amt));
        }
    }

    // GROUP: axes 0-2, amounts [0, 4, 8, 16]
    for axis in 0..3 {
        for &amt in &[0, 4, 8, 16] {
            actions.push(Opt::group(axis, amt));
        }
    }

    // TC: tensor cores. 1 default-axis action + 9 axis variants = 10 actions.
    // Survivors after post-compile dedup are unchanged compared to a wider
    // brute-force enumeration because `seen_libs` collapses duplicate kernels.
    const TC_AXIS_CHOICES: usize = 9;
    const TC_OPT_DEFAULT: usize = 0;
    const TC_OPT_AXIS: usize = 2;
    actions.push(Opt::tc(Some(0), -1, TC_OPT_DEFAULT, 1));
    for axis_choice in 0..TC_AXIS_CHOICES {
        actions.push(Opt::tc(Some(axis_choice), -1, TC_OPT_AXIS, 1));
    }

    // SWAP: axis pairs
    for a0 in 0..5 {
        for a1 in (a0 + 1)..5 {
            actions.push(Opt::swap(a0, a1));
        }
    }

    // THREAD: CPU parallelization with smart divisor selection
    // Include thread counts that divide common tensor sizes (64, 128, 256, 512, 1024)
    let max_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let thread_amounts = thread_action_amounts(max_threads);
    for axis in 0..3 {
        for &amt in &thread_amounts {
            actions.push(Opt::thread(axis, amt));
        }
    }

    // NOLOCALS — only when explicitly enabled via `MOROK_NOLOCALS`.
    if std::env::var("MOROK_NOLOCALS").is_ok() {
        actions.push(Opt::nolocals());
    }

    actions
});

// ============================================================================
// ACTION GENERATION & FILTERING
// ============================================================================

/// Generate all valid next-states from the current scheduler.
///
/// Applies each action from `BEAM_ACTIONS` and filters to those that:
/// 1. Apply successfully (divisibility, bounds, etc.)
/// 2. Pass limit checks (upcast size, local size, UOp count)
fn generate_actions(scheduler: &Scheduler, config: &BeamConfig) -> Vec<Scheduler> {
    BEAM_ACTIONS
        .iter()
        .filter_map(|action| {
            // Clone scheduler and try to apply action
            let mut candidate = scheduler.clone();
            match apply_opt(&mut candidate, action, true) {
                Ok(()) if validate_limits(&candidate, config) => Some(candidate),
                _ => None,
            }
        })
        .collect()
}

/// Validate that a scheduler state is within configured limits.
fn validate_limits(scheduler: &Scheduler, config: &BeamConfig) -> bool {
    // Calculate upcast size (product of UPCAST/UNROLL dimensions)
    let upcast_sz = product_of_axes(scheduler, &[AxisType::Upcast, AxisType::Unroll]);

    // Calculate local size (product of LOCAL/WARP/GROUP_REDUCE dimensions)
    let local_sz = product_of_axes(scheduler, &[AxisType::Local, AxisType::Warp, AxisType::GroupReduce]);

    // Check UOp count
    let uop_count = scheduler.ast().toposort().len();

    upcast_sz <= config.max_upcast && local_sz <= config.max_local && uop_count <= config.max_uops
}

/// Calculate product of dimension sizes for given axis types.
fn product_of_axes(scheduler: &Scheduler, types: &[AxisType]) -> usize {
    scheduler
        .rngs()
        .iter()
        .filter_map(|rng| {
            if let Op::Range { axis_type, end, .. } = rng.op()
                && types.contains(axis_type)
                && let Op::Const(cv) = end.op()
                && let ConstValue::Int(sz) = cv.0
            {
                Some(sz as usize)
            } else {
                None
            }
        })
        .product::<usize>()
        .max(1)
}

// ============================================================================
// BEAM SEARCH ALGORITHM
// ============================================================================

/// Beam search result containing optimized scheduler and timing.
pub struct BeamResult {
    /// Optimized scheduler state.
    pub scheduler: Scheduler,
    /// Best timing achieved.
    pub timing: Duration,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Total candidates evaluated.
    pub candidates_evaluated: usize,
}

/// Metrics returned by the `compile_and_time` closure for each candidate.
///
/// Timing drives ranking; the IR hash drives `seen_libs` dedup; the compute-op
/// count drives the `least_compute_ops*1000` filter.
#[derive(Debug, Clone, Copy)]
pub struct CandidateMetrics {
    /// Best execution timing across the run loop (`min(tms)`).
    pub timing: Duration,
    /// Hash of the post-codegen IR — kernels that lower to the same IR are
    /// guaranteed to compile to the same object, so we skip duplicates.
    pub ir_hash: u64,
    /// Cheap upper bound on the kernel's compute work; used by the
    /// `least_compute_ops*1000` filter to discard degenerate candidates.
    pub compute_ops: u64,
}

/// Hash a UOp tree to a `u64` for `seen_libs` dedup.
///
/// Uses the pre-computed `content_hash` field on `UOp` (see
/// `ir/src/uop/hash_consing.rs`), which is the same structural hash the
/// hash-consing cache and `schedule_cache` rely on. O(1) — read the cached
/// field instead of re-walking the graph.
pub fn hash_post_codegen_ir(uop: &Arc<UOp>) -> u64 {
    uop.content_hash
}

/// Symbolic estimate of compute ops in a kernel.
///
/// Each ALU/Ternary/Reduce/WMMA node contributes `prod(enclosing-RANGE sizes)`
/// flops. Symbolic RANGE ends resolve to the midpoint of their `vmin`/`vmax`
/// bounds (matching the `(vmax+vmin)/2` choice in the BEAM timing path), so
/// dynamic-shape kernels participate in the `least_compute_ops*1000` bloat
/// filter.
pub fn compute_ops_estimate(uop: &Arc<UOp>) -> u64 {
    let topo = uop.toposort();

    // Pre-compute the size contribution of every loop-bound node — RANGE for
    // ordinary loops, SPECIAL for hardware-provided indices.
    let mut range_size: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    for node in &topo {
        let end = match node.op() {
            Op::Range { end, .. } => Some(end),
            Op::Special { end, .. } => Some(end),
            _ => None,
        };
        if let Some(end) = end {
            range_size.insert(node.id, range_size_estimate(end));
        }
    }

    // Each ALU/Reduce/WMMA accumulates `prod(in-scope range sizes)`. Backward
    // slice membership tells us which RANGEs the node sits inside, mirroring
    // tinygrad's `mult_stack` discipline structurally.
    let mut flops: u64 = 0;
    for node in &topo {
        let is_alu =
            matches!(node.op(), Op::Binary(..) | Op::Unary(..) | Op::Ternary(..) | Op::Reduce { .. } | Op::Wmma { .. });
        if !is_alu {
            continue;
        }
        let bws = node.backward_slice_ids();
        let mut weight: u64 = 1;
        for (rid, sz) in &range_size {
            if bws.contains(rid) {
                weight = weight.saturating_mul(*sz);
            }
        }
        flops = flops.saturating_add(weight);
    }
    flops
}

/// Estimate a RANGE end's iteration count.
///
/// Concrete `Const(Int)` ends use the value directly; everything else falls
/// back to the midpoint of the `end` UOp's symbolic `vmin`/`vmax` bounds, so
/// dynamic-shape ranges still contribute a representative number of flops.
fn range_size_estimate(end: &Arc<UOp>) -> u64 {
    if let Op::Const(cv) = end.op()
        && let Some(v) = cv.0.try_int()
    {
        return (v.max(1)) as u64;
    }
    let vmin = end.vmin().try_int().unwrap_or(1);
    let vmax = end.vmax().try_int().unwrap_or(vmin);
    (((vmin + vmax) / 2).max(1)) as u64
}

/// Run beam search optimization.
///
/// # Arguments
///
/// * `scheduler` - Initial scheduler state
/// * `config` - Beam search configuration
/// * `compile_and_time` - Function to compile and time a scheduler state
///
/// # Returns
///
/// `BeamResult` containing the best scheduler found and performance metrics.
///
/// # Example
///
/// ```ignore
/// let config = BeamConfig::default();
/// let compile_and_time = |s: &Scheduler, early_stop: Option<Duration>| {
///     let ast = s.get_optimized_ast(None);
///     let kernel = compile_kernel(&ast)?;
///     let bench = benchmark_kernel(&kernel, ..., early_stop)?;
///     Some(CandidateMetrics { timing: bench.min, ir_hash: ..., compute_ops: ... })
/// };
///
/// let result = beam_search(scheduler, &config, compile_and_time)?;
/// println!("Best time: {:?}", result.timing);
/// ```
pub fn beam_search<F>(scheduler: Scheduler, config: &BeamConfig, compile_and_time: F) -> Result<BeamResult, OptError>
where
    F: Fn(&Scheduler, Option<Duration>) -> Option<CandidateMetrics> + Sync,
{
    let start = Instant::now();
    let mut iterations = 0;
    let mut candidates_evaluated = 0;

    // Initialize beam with `Duration::MAX` so the first iteration has no
    // incumbent to beat. Avoids one wasted compile+time per `beam_search`
    // invocation (also charged on cache replay through `OPT_CACHE`).
    let mut beam: Vec<(Scheduler, Duration)> = vec![(scheduler.clone(), Duration::MAX)];

    while start.elapsed() < config.timeout {
        iterations += 1;

        // 1. EXPAND: Generate all valid next states from current beam (sequential)
        // Note: Scheduler is not Sync due to OnceCell caches, so expansion is sequential
        let candidates: Vec<Scheduler> = beam.iter().flat_map(|(s, _)| generate_actions(s, config)).collect();

        if candidates.is_empty() {
            break;
        }

        // Per-iteration state — both reset at the top of every iteration.
        // `seen_libs` dedups kernels that lower to the same post-codegen IR;
        // `least_compute_ops` anchors the 1000× compute-bloat filter.
        let mut seen_libs: std::collections::HashSet<u64> = std::collections::HashSet::with_capacity(candidates.len());
        let mut least_compute_ops: u64 = u64::MAX;

        // Reject any candidate whose first run already exceeds 3× the current beam best.
        let beam_best = beam.first().map(|(_, t)| *t);
        let early_stop = beam_best.and_then(|t| t.checked_mul(3));

        // 2. COMPILE & TIME: Evaluate performance
        let mut timed: Vec<(Scheduler, Duration)> = Vec::new();
        for s in candidates {
            let Some(metrics) = compile_and_time(&s, early_stop) else { continue };

            if !seen_libs.insert(metrics.ir_hash) {
                continue;
            }
            least_compute_ops = least_compute_ops.min(metrics.compute_ops);
            if least_compute_ops.saturating_mul(1000) < metrics.compute_ops {
                continue;
            }

            timed.push((s, metrics.timing));
        }

        candidates_evaluated += timed.len();

        if timed.is_empty() {
            break;
        }

        // 3. SORT: Sort by timing (best first)
        let mut sorted = timed;
        sorted.sort_by_key(|(_, t)| *t);

        // 4. CHECK TERMINATION — exit when the new best is already below
        //    the progress floor (fast-enough kernel) OR when the gain over
        //    the incumbent is sub-noise. Sub-noise gains don't justify a
        //    next compile round.
        let best_new = sorted[0].1;
        let best_old = beam.first().map(|(_, t)| *t).unwrap_or(Duration::MAX);
        let min_progress = beam_min_progress();
        let absolute_floor = best_new < min_progress;
        let no_real_gain = best_old.saturating_sub(best_new) < min_progress;

        if absolute_floor || no_real_gain {
            // When exiting AND we did improve, pin the beam to the single
            // new winner so callers see it.
            if best_new < best_old {
                beam = sorted.into_iter().take(1).collect();
            }
            break;
        }

        // 5. PRUNE: Keep top K by timing
        beam = sorted.into_iter().take(config.beam_width).collect();
    }

    let (best_scheduler, best_timing) = beam.into_iter().next().unwrap_or((scheduler, Duration::MAX));

    Ok(BeamResult { scheduler: best_scheduler, timing: best_timing, iterations, candidates_evaluated })
}

// ============================================================================
// REPLAY
// ============================================================================

/// Replay a sequence of optimizations on a scheduler.
///
/// Used to restore cached beam search results.
pub fn replay_opts(mut scheduler: Scheduler, opts: &[Opt]) -> Result<Scheduler, OptError> {
    for opt in opts {
        apply_opt(&mut scheduler, opt, true)?;
    }
    Ok(scheduler)
}

/// Get the applied optimizations from a scheduler.
pub fn get_applied_opts(scheduler: &Scheduler) -> &[Opt] {
    &scheduler.applied_opts
}

// ============================================================================
// CACHING
// ============================================================================

/// Global sled database for beam search cache.
///
/// Lazy-initialized on first access. Returns None if cache directory
/// cannot be created or database cannot be opened.
static CACHE_DB: Lazy<Option<sled::Db>> = Lazy::new(|| {
    let cache_dir = dirs::cache_dir()?.join("morok");
    std::fs::create_dir_all(&cache_dir).ok()?;
    sled::open(cache_dir.join("beam_cache")).ok()
});

/// Cache key for beam search results.
///
/// Includes the limit configuration (max_upcast, max_local, max_uops) so that
/// changing caps invalidates cached entries: replaying opts produced under a
/// looser cap could reintroduce a kernel that no longer satisfies the new cap.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    /// Hash of the AST structure.
    ast_hash: u64,
    /// Beam width used for search.
    beam_width: usize,
    /// Renderer/TC backend.
    device: morok_ir::RendererDevice,
    /// Upcast/unroll product cap at search time.
    max_upcast: usize,
    /// Local/warp/group_reduce product cap at search time.
    max_local: usize,
    /// UOp count cap at search time.
    max_uops: usize,
}

impl CacheKey {
    /// Create a cache key from a scheduler and config.
    fn from_scheduler(scheduler: &Scheduler, config: &BeamConfig) -> Self {
        // Use structural hash for cross-run stability. The recursive Hash for UOp
        // traverses (dtype, op) of the entire DAG — same AST structure produces
        // the same hash regardless of process-local ids.
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        scheduler.ast().hash(&mut hasher);
        let ast_hash = hasher.finish();

        Self {
            ast_hash,
            beam_width: config.beam_width,
            device: scheduler.ren.device,
            max_upcast: config.max_upcast,
            max_local: config.max_local,
            max_uops: config.max_uops,
        }
    }

    /// Convert to bytes for database key.
    fn to_bytes(&self) -> Vec<u8> {
        let device_str = self.device.canonical();
        let mut bytes = Vec::with_capacity(48 + device_str.len());
        bytes.extend_from_slice(&self.ast_hash.to_le_bytes());
        bytes.extend_from_slice(&self.beam_width.to_le_bytes());
        bytes.extend_from_slice(&self.max_upcast.to_le_bytes());
        bytes.extend_from_slice(&self.max_local.to_le_bytes());
        bytes.extend_from_slice(&self.max_uops.to_le_bytes());
        bytes.extend_from_slice(device_str.as_bytes());
        bytes
    }
}

/// Serialize applied opts to bytes for caching using bincode.
fn serialize_opts(opts: &[Opt]) -> Vec<u8> {
    bincode::serialize(opts).expect("Opt serialization should not fail")
}

/// Deserialize opts from cached bytes using bincode.
fn deserialize_opts(bytes: &[u8]) -> Option<Vec<Opt>> {
    bincode::deserialize(bytes).ok()
}

/// Get cached beam search result.
fn cache_get(key: &CacheKey) -> Option<Vec<Opt>> {
    let db = CACHE_DB.as_ref()?;
    let bytes = db.get(key.to_bytes()).ok()??;
    deserialize_opts(&bytes)
}

/// Store beam search result in cache.
fn cache_put(key: &CacheKey, opts: &[Opt]) {
    if let Some(db) = CACHE_DB.as_ref()
        && db.insert(key.to_bytes(), serialize_opts(opts)).is_ok()
    {
        // Flush to disk to ensure persistence across runs
        let _ = db.flush();
    }
}

/// Remove a stale cache entry.
fn cache_invalidate(key: &CacheKey) {
    if let Some(db) = CACHE_DB.as_ref() {
        let _ = db.remove(key.to_bytes());
        let _ = db.flush();
    }
}

/// Run beam search with disk caching.
///
/// Checks the cache before running beam search. If a cached result exists,
/// replays the optimizations instead of searching. Results are cached after
/// successful search.
///
/// # Arguments
///
/// * `scheduler` - Initial scheduler state
/// * `config` - Beam search configuration (includes disable_cache flag)
/// * `compile_and_time` - Function to compile and time a scheduler state
///
/// # Returns
///
/// `BeamResult` containing the best scheduler found.
pub fn beam_search_cached<F>(
    scheduler: Scheduler,
    config: &BeamConfig,
    compile_and_time: F,
) -> Result<BeamResult, OptError>
where
    F: Fn(&Scheduler, Option<Duration>) -> Option<CandidateMetrics> + Sync,
{
    let key = CacheKey::from_scheduler(&scheduler, config);

    // Check cache (unless disabled)
    if !config.disable_cache
        && let Some(cached_opts) = cache_get(&key)
    {
        // Replay cached optimizations. If replay fails (stale entry from code changes),
        // or the replayed scheduler exceeds the current limits (looser cap at search
        // time, tighter cap now), invalidate and fall through to fresh search.
        tracing::info!(opts_count = cached_opts.len(), "Beam cache HIT - replaying opts");
        match replay_opts(scheduler.clone(), &cached_opts) {
            Ok(replayed) if validate_limits(&replayed, config) => {
                let timing = compile_and_time(&replayed, None).map(|m| m.timing).unwrap_or(Duration::MAX);
                return Ok(BeamResult { scheduler: replayed, timing, iterations: 0, candidates_evaluated: 0 });
            }
            Ok(_) => {
                tracing::warn!("Beam cache replayed scheduler violates limits - invalidating");
                cache_invalidate(&key);
            }
            Err(e) => {
                tracing::warn!(?e, "Beam cache replay failed (stale entry?) - invalidating");
                cache_invalidate(&key);
            }
        }
    }

    tracing::info!("Beam cache MISS - running search");
    // Run beam search
    let result = beam_search(scheduler, config, compile_and_time)?;

    // Cache result (unless disabled)
    if !config.disable_cache {
        cache_put(&key, &result.scheduler.applied_opts);
    }

    Ok(result)
}

/// Clear the beam search cache.
///
/// Useful for testing or when invalidating cached results.
pub fn clear_cache() {
    if let Some(db) = CACHE_DB.as_ref() {
        let _ = db.clear();
    }
}

#[cfg(test)]
#[path = "../test/unit/optimizer/beam_internal.rs"]
mod tests;
