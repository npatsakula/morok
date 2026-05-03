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

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;

use morok_ir::{AxisType, ConstValue, Op};

use super::Scheduler;
use super::config::BeamConfig;
use super::error::*;
use super::opts::apply_opt;
use super::types::Opt;

/// Minimum improvement (over the current beam best) required to keep iterating.
///
/// Mirrors tinygrad `BEAM_MIN_PROGRESS` (search.py:138) — default 10 ns. When the
/// best candidate of a round improves on the beam by less than this delta, we
/// stop instead of churning over noise. Read once via `OnceLock` to avoid
/// per-iteration env var lookups.
fn beam_min_progress() -> Duration {
    static CACHE: OnceLock<Duration> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let micros = std::env::var("MOROK_BEAM_MIN_PROGRESS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.01);
        Duration::from_nanos((micros * 1_000.0).round().max(0.0) as u64)
    })
}

// ============================================================================
// ACTION SPACE
// ============================================================================

/// Generate thread counts that are likely to divide common tensor sizes.
///
/// Instead of fixed power-of-2, includes all values up to max_threads that
/// divide common sizes (64, 128, 256, 512, 1024). This ensures beam search
/// can find optimal thread counts for various tensor dimensions.
fn thread_action_amounts(max_threads: usize) -> Vec<usize> {
    const COMMON_SIZES: [usize; 5] = [64, 128, 256, 512, 1024];

    let mut amounts: Vec<usize> = (2..=max_threads).filter(|&t| COMMON_SIZES.iter().any(|&sz| sz % t == 0)).collect();
    amounts.sort_unstable();
    amounts.dedup();
    amounts
}

/// Pre-computed action space for beam search (~500 actions).
///
/// Based on tinygrad's beam search action generation.
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

    // TC: tensor cores
    const TC_AXIS_CHOICES: usize = 9;

    // Auto-select without fixed axis choice.
    actions.push(Opt::tc(None, -1, 0, 1));
    actions.push(Opt::tc(None, -1, 2, 1));

    // Auto-select with explicit axis choices.
    for axis_choice in 0..TC_AXIS_CHOICES {
        actions.push(Opt::tc(Some(axis_choice), -1, 0, 1));
        actions.push(Opt::tc(Some(axis_choice), -1, 2, 1));
    }

    // Specific tensor cores, with and without explicit axis choices.
    for tc_select in 0..9 {
        actions.push(Opt::tc(None, tc_select, 2, 1));
        for axis_choice in 0..TC_AXIS_CHOICES {
            actions.push(Opt::tc(Some(axis_choice), tc_select, 2, 1));
        }
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

    // NOLOCALS
    actions.push(Opt::nolocals());

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
/// let compile_and_time = |s: &Scheduler| {
///     let ast = s.get_optimized_ast(None);
///     let kernel = compile_kernel(&ast)?;
///     let timing = benchmark_kernel(&kernel)?;
///     Some(timing)
/// };
///
/// let result = beam_search(scheduler, &config, compile_and_time)?;
/// println!("Best time: {:?}", result.timing);
/// ```
pub fn beam_search<F>(scheduler: Scheduler, config: &BeamConfig, compile_and_time: F) -> Result<BeamResult, OptError>
where
    F: Fn(&Scheduler) -> Option<Duration> + Sync,
{
    let start = Instant::now();
    let mut iterations = 0;
    let mut candidates_evaluated = 0;

    // Initialize beam with starting state
    let initial_timing = compile_and_time(&scheduler).unwrap_or(Duration::MAX);
    let mut beam: Vec<(Scheduler, Duration)> = vec![(scheduler.clone(), initial_timing)];

    while start.elapsed() < config.timeout {
        iterations += 1;

        // 1. EXPAND: Generate all valid next states from current beam (sequential)
        // Note: Scheduler is not Sync due to OnceCell caches, so expansion is sequential
        let candidates: Vec<Scheduler> = beam.iter().flat_map(|(s, _)| generate_actions(s, config)).collect();

        if candidates.is_empty() {
            break;
        }

        // 2. COMPILE & TIME: Evaluate performance
        // The compile_and_time function should handle parallelism internally if needed
        let timed: Vec<(Scheduler, Duration)> = candidates
            .into_iter()
            .filter_map(|s| {
                let timing = compile_and_time(&s)?;
                Some((s, timing))
            })
            .collect();

        candidates_evaluated += timed.len();

        if timed.is_empty() {
            break;
        }

        // 3. SORT: Sort by timing (best first)
        let mut sorted = timed;
        sorted.sort_by_key(|(_, t)| *t);

        // 4. CHECK TERMINATION: improvement below min_progress threshold, or
        //    suspiciously-zero candidate timing (mirrors tinygrad search.py:182).
        let best_new = sorted[0].1;
        let best_old = beam.first().map(|(_, t)| *t).unwrap_or(Duration::MAX);
        let min_progress = beam_min_progress();

        if best_new < min_progress || best_old.saturating_sub(best_new) < min_progress {
            // Adopt best candidate if it still improves on the beam, then stop.
            if best_new < best_old {
                beam = sorted.into_iter().take(1).collect();
            }
            break;
        }

        // 5. PRUNE: Keep top K by timing
        beam = sorted.into_iter().take(config.beam_width).collect();

        // Memory management: With weak references in the UOp cache (Tinygrad-aligned),
        // discarded candidates are automatically cleaned up when their Arcs are dropped.
        // No manual GC calls needed.
    }

    // Return best result
    let (best_scheduler, best_timing) = beam.into_iter().next().unwrap_or((scheduler, Duration::MAX));

    Ok(BeamResult { scheduler: best_scheduler, timing: best_timing, iterations, candidates_evaluated })
}

/// Run beam search with timeout check per iteration.
///
/// Similar to `beam_search` but includes additional timeout checks
/// to avoid long-running searches and early cutoff for slow candidates.
pub fn beam_search_with_timeout<F>(
    scheduler: Scheduler,
    config: &BeamConfig,
    compile_and_time: F,
) -> Result<BeamResult, OptError>
where
    F: Fn(&Scheduler) -> Option<Duration> + Sync,
{
    let start = Instant::now();
    let mut iterations = 0;
    let mut candidates_evaluated = 0;

    let initial_timing = compile_and_time(&scheduler).unwrap_or(Duration::MAX);
    let mut beam: Vec<(Scheduler, Duration)> = vec![(scheduler.clone(), initial_timing)];

    // Early termination threshold (3x the best time so far)
    let mut cutoff = initial_timing.saturating_mul(3);

    while start.elapsed() < config.timeout {
        iterations += 1;

        // Check timeout before expansion
        if start.elapsed() > config.timeout {
            break;
        }

        let candidates: Vec<Scheduler> = beam.iter().flat_map(|(s, _)| generate_actions(s, config)).collect();

        if candidates.is_empty() {
            break;
        }

        // Time with cutoff for early termination
        let timed: Vec<(Scheduler, Duration)> = candidates
            .into_iter()
            .filter_map(|s| {
                let timing = compile_and_time(&s)?;
                // Skip if clearly worse than cutoff
                if timing > cutoff {
                    return None;
                }
                Some((s, timing))
            })
            .collect();

        candidates_evaluated += timed.len();

        if timed.is_empty() {
            break;
        }

        let mut sorted = timed;
        sorted.sort_by_key(|(_, t)| *t);

        let best_new = sorted[0].1;
        let best_old = beam.first().map(|(_, t)| *t).unwrap_or(Duration::MAX);
        let min_progress = beam_min_progress();

        if best_new < min_progress || best_old.saturating_sub(best_new) < min_progress {
            if best_new < best_old {
                beam = sorted.into_iter().take(1).collect();
            }
            break;
        }

        // Update cutoff based on new best
        cutoff = best_new.saturating_mul(3);

        beam = sorted.into_iter().take(config.beam_width).collect();

        // Memory management: With weak refs, discarded candidates are auto-cleaned.
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
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    /// Hash of the AST structure.
    ast_hash: u64,
    /// Beam width used for search.
    beam_width: usize,
    /// Device name (e.g., "cpu", "cuda").
    device: String,
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

        Self { ast_hash, beam_width: config.beam_width, device: scheduler.ren.device.clone() }
    }

    /// Convert to bytes for database key.
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(24);
        bytes.extend_from_slice(&self.ast_hash.to_le_bytes());
        bytes.extend_from_slice(&self.beam_width.to_le_bytes());
        bytes.extend_from_slice(self.device.as_bytes());
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
    F: Fn(&Scheduler) -> Option<Duration> + Sync,
{
    let key = CacheKey::from_scheduler(&scheduler, config);

    // Check cache (unless disabled)
    if !config.disable_cache
        && let Some(cached_opts) = cache_get(&key)
    {
        // Replay cached optimizations. If replay fails (stale entry from code changes),
        // invalidate and fall through to fresh search.
        tracing::info!(opts_count = cached_opts.len(), "Beam cache HIT - replaying opts");
        match replay_opts(scheduler.clone(), &cached_opts) {
            Ok(replayed) => {
                let timing = compile_and_time(&replayed).unwrap_or(Duration::MAX);
                return Ok(BeamResult { scheduler: replayed, timing, iterations: 0, candidates_evaluated: 0 });
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
