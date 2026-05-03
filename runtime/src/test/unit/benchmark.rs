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
