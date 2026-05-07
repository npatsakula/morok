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
    let config =
        BenchmarkConfig { warmup_runs: 1, timing_runs: 3, take_minimum: true, early_stop: None, clear_l2: false };

    let result = unsafe { benchmark_kernel(&kernel, &[], &[], None, None, &config) }.unwrap();

    assert_eq!(result.runs.len(), 3);
    assert!(result.min >= Duration::from_micros(100));
    assert!(result.min <= result.mean);
}

#[test]
fn test_benchmark_early_stop() {
    let kernel = MockKernel { name: "slow".into(), sleep_micros: 10000 };
    let config = BenchmarkConfig {
        warmup_runs: 0,
        timing_runs: 5,
        take_minimum: true,
        early_stop: Some(Duration::from_micros(100)),
        clear_l2: false,
    };

    let result = unsafe { benchmark_kernel(&kernel, &[], &[], None, None, &config) }.unwrap();

    // Each run is ~10ms; the very first exceeds the 100µs threshold and
    // the loop bails out, so we record exactly one run instead of five.
    assert_eq!(result.runs.len(), 1);
}

#[test]
fn test_benchmark_early_stop_passes_under_cutoff() {
    let kernel = MockKernel { name: "fast".into(), sleep_micros: 50 };
    let cutoff = Duration::from_millis(1);
    let config = BenchmarkConfig { early_stop: Some(cutoff * 3), ..BenchmarkConfig::default() };

    let result = unsafe { benchmark_kernel(&kernel, &[], &[], None, None, &config) }.unwrap();

    assert_eq!(result.runs.len(), config.timing_runs);
    assert!(result.min < cutoff);
}

#[test]
fn test_benchmark_early_stop_aborts_over_cutoff() {
    let kernel = MockKernel { name: "slow".into(), sleep_micros: 10000 };
    let cutoff = Duration::from_micros(100);
    let config = BenchmarkConfig { early_stop: Some(cutoff * 3), ..BenchmarkConfig::default() };

    let result = unsafe { benchmark_kernel(&kernel, &[], &[], None, None, &config) }.unwrap();

    assert_eq!(result.runs.len(), 1);
    assert!(result.min > cutoff * 3);
}
