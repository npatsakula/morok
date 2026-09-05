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
        _wait: bool,
    ) -> svod_device::Result<()> {
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
    // Wide margin: a 50us sleep can take >1ms wall on a loaded CI runner, and
    // this test only claims that early-stop does NOT trigger under the cutoff.
    let cutoff = Duration::from_millis(100);
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

/// A backend with GPU stamps reports the device time, not the (longer) wall
/// time around the synchronous dispatch.
struct StampedKernel;

impl Program for StampedKernel {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        std::thread::sleep(Duration::from_millis(5));
        Ok(())
    }

    unsafe fn execute_timed(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> svod_device::Result<Option<Duration>> {
        unsafe { self.execute(buffers, vals, global_size, local_size, true)? };
        Ok(Some(Duration::from_micros(7)))
    }

    fn name(&self) -> &str {
        "stamped"
    }
}

#[test]
fn benchmark_prefers_gpu_stamped_durations() {
    let result =
        unsafe { benchmark_kernel(&StampedKernel, &[], &[], None, None, &BenchmarkConfig::default()) }.unwrap();
    assert!(result.runs.iter().all(|run| *run == Duration::from_micros(7)), "{:?}", result.runs);
}
