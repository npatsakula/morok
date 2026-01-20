//! Kernel benchmarking infrastructure for auto-tuning.
//!
//! Provides timing utilities for measuring kernel execution performance,
//! used by beam search optimization to compare candidate kernels.

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
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self { warmup_runs: 1, timing_runs: 3, take_minimum: true }
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
    for _ in 0..config.timing_runs {
        let start = Instant::now();
        unsafe { kernel.execute(buffers, vals, global_size, local_size)? };
        runs.push(start.elapsed());
    }

    // Calculate statistics
    let min = runs.iter().copied().min().unwrap_or(Duration::ZERO);
    let total: Duration = runs.iter().sum();
    let mean = total / runs.len().max(1) as u32;

    Ok(BenchmarkResult { min, mean, runs })
}

/// Benchmark with early stopping if clearly slower than threshold.
///
/// Useful for beam search to skip obviously slow candidates.
///
/// # Safety
///
/// Same safety requirements as `benchmark_kernel`.
pub unsafe fn benchmark_kernel_with_cutoff(
    kernel: &dyn Program,
    buffers: &[*mut u8],
    vals: &[i64],
    global_size: Option<[usize; 3]>,
    local_size: Option<[usize; 3]>,
    config: &BenchmarkConfig,
    cutoff: Duration,
) -> Result<Option<BenchmarkResult>> {
    // Warmup runs
    for _ in 0..config.warmup_runs {
        unsafe { kernel.execute(buffers, vals, global_size, local_size)? };
    }

    // First timing run - check against cutoff
    let start = Instant::now();
    unsafe { kernel.execute(buffers, vals, global_size, local_size)? };
    let first = start.elapsed();

    // Early exit if clearly slower (3x cutoff)
    if first > cutoff * 3 {
        return Ok(None);
    }

    // Remaining timing runs
    let mut runs = vec![first];
    for _ in 1..config.timing_runs {
        let start = Instant::now();
        unsafe { kernel.execute(buffers, vals, global_size, local_size)? };
        runs.push(start.elapsed());
    }

    let min = runs.iter().copied().min().unwrap_or(Duration::ZERO);
    let total: Duration = runs.iter().sum();
    let mean = total / runs.len().max(1) as u32;

    Ok(Some(BenchmarkResult { min, mean, runs }))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockKernel {
        name: String,
        sleep_micros: u64,
    }

    impl Program for MockKernel {
        unsafe fn execute(
            &self,
            _buffers: &[*mut u8],
            _vals: &[i64],
            _global_size: Option<[usize; 3]>,
            _local_size: Option<[usize; 3]>,
        ) -> morok_device::Result<()> {
            std::thread::sleep(Duration::from_micros(self.sleep_micros));
            Ok(())
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_benchmark_basic() {
        let kernel = MockKernel { name: "test".into(), sleep_micros: 100 };
        let config = BenchmarkConfig { warmup_runs: 1, timing_runs: 3, take_minimum: true };

        let result = unsafe { benchmark_kernel(&kernel, &[], &[], None, None, &config) }.unwrap();

        assert_eq!(result.runs.len(), 3);
        assert!(result.min >= Duration::from_micros(100));
        assert!(result.min <= result.mean);
    }

    #[test]
    fn test_benchmark_with_cutoff_passes() {
        let kernel = MockKernel { name: "fast".into(), sleep_micros: 50 };
        let config = BenchmarkConfig::default();
        let cutoff = Duration::from_millis(1);

        let result = unsafe { benchmark_kernel_with_cutoff(&kernel, &[], &[], None, None, &config, cutoff) }.unwrap();

        assert!(result.is_some());
    }

    #[test]
    fn test_benchmark_with_cutoff_fails() {
        let kernel = MockKernel { name: "slow".into(), sleep_micros: 10000 };
        let config = BenchmarkConfig::default();
        let cutoff = Duration::from_micros(100); // Very tight cutoff

        let result = unsafe { benchmark_kernel_with_cutoff(&kernel, &[], &[], None, None, &config, cutoff) }.unwrap();

        assert!(result.is_none());
    }
}
