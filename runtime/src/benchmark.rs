//! Kernel benchmarking infrastructure for auto-tuning.
//!
//! Provides timing utilities for measuring kernel execution performance,
//! used by beam search optimization to compare candidate kernels.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use morok_device::device::Program;

use crate::Result;

/// Configuration for kernel benchmarking.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup runs (not timed).
    pub warmup_runs: usize,
    /// Number of timing runs.
    pub timing_runs: usize,
    /// Whether to return minimum time (true) or mean (false).
    pub take_minimum: bool,
    /// If set, abort the timing loop the moment any single run exceeds this
    /// threshold. Used by beam search to skip candidates clearly slower than
    /// the current best (typically `early_stop = beam[0].timing * 3`).
    pub early_stop: Option<Duration>,
    /// Invalidate L2 between runs by streaming through a scratch buffer.
    /// Stabilises rankings — without this, second/third runs hit hot caches
    /// and bias beam toward smaller-tile candidates.
    pub clear_l2: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        // 3 timing runs, take the minimum — variance from rayon dispatch
        // and OS scheduling is much larger than per-run overhead, so the
        // min of 3 is a tighter estimate of the kernel's true cost than
        // any longer-running statistic.
        Self { warmup_runs: 0, timing_runs: 3, take_minimum: true, early_stop: None, clear_l2: false }
    }
}

/// Result of kernel benchmarking.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Minimum execution time.
    pub min: Duration,
    /// Mean execution time.
    pub mean: Duration,
    /// All timing measurements.
    pub runs: Vec<Duration>,
}

impl BenchmarkResult {
    /// Get the timing value based on config preference.
    pub fn timing(&self, take_minimum: bool) -> Duration {
        if take_minimum { self.min } else { self.mean }
    }
}

/// Benchmark a compiled kernel's execution time.
///
/// Runs warmup iterations (discarded), then timing iterations.
/// Returns min/mean/all timings.
///
/// # Safety
///
/// All buffer pointers must be valid for the duration of benchmarking.
/// The kernel will be executed multiple times.
///
/// # Example
///
/// ```ignore
/// let config = BenchmarkConfig::default();
/// let result = unsafe { benchmark_kernel(&kernel, &buffers, &vals, None, None, &config)? };
/// println!("Min time: {:?}", result.min);
/// ```
pub unsafe fn benchmark_kernel(
    kernel: &dyn Program,
    buffers: &[*mut u8],
    vals: &[i64],
    global_size: Option<[usize; 3]>,
    local_size: Option<[usize; 3]>,
    config: &BenchmarkConfig,
) -> Result<BenchmarkResult> {
    // Warmup runs (discard timing)
    for _ in 0..config.warmup_runs {
        unsafe { kernel.execute(buffers, vals, global_size, local_size)? };
    }

    // Timing runs
    let mut runs = Vec::with_capacity(config.timing_runs);
    for i in 0..config.timing_runs {
        if config.clear_l2 && i > 0 {
            invalidate_l2();
        }
        let start = Instant::now();
        unsafe { kernel.execute(buffers, vals, global_size, local_size)? };
        runs.push(start.elapsed());

        // Min-of-runs early stop: abort only when the best run so far still
        // exceeds the threshold. A single jitter outlier in an otherwise
        // competitive candidate must not disqualify it — `take_minimum=true`
        // already discards tail noise from the final result.
        if let Some(threshold) = config.early_stop
            && runs.iter().copied().min().expect("runs non-empty after push") > threshold
        {
            break;
        }
    }

    // Calculate statistics
    let min = runs.iter().copied().min().unwrap_or(Duration::ZERO);
    let total: Duration = runs.iter().sum();
    let mean = total / runs.len().max(1) as u32;

    Ok(BenchmarkResult { min, mean, runs })
}

/// Force rayon's global thread pool to materialise.
///
/// Subsequent rayon calls dispatch in O(1), but the lazy initialisation can
/// dominate the first 1-2 measurements at the small kernel sizes BEAM-time
/// uses. Call this once before a benchmark loop to remove that bias.
pub fn warmup_thread_pool() {
    rayon::join(|| (), || ());
}

/// Stream through a 16 MiB scratch buffer to evict L2 between timing runs.
///
/// Apple M1 P-core L2 is 12 MiB, A14/M2 L2 caches are similar; 16 MiB is
/// large enough to fully evict L2 on common Apple Silicon and x86 desktop
/// CPUs. The scratch buffer is allocated once (per process) via `OnceLock`
/// and reused across calls. `black_box` prevents the compiler from eliding
/// the read.
fn invalidate_l2() {
    const SCRATCH_BYTES: usize = 16 * 1024 * 1024;
    static SCRATCH: OnceLock<Vec<u8>> = OnceLock::new();
    let scratch = SCRATCH.get_or_init(|| vec![0u8; SCRATCH_BYTES]);

    let mut acc: u8 = 0;
    let stride = 64; // touch one byte per cache line
    let mut i = 0;
    while i < scratch.len() {
        acc = acc.wrapping_add(scratch[i]);
        i += stride;
    }
    std::hint::black_box(acc);
}

#[cfg(test)]
#[path = "test/unit/benchmark.rs"]
mod tests;
