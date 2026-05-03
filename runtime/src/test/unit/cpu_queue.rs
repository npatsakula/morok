use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct QueueProgram {
    calls: Arc<AtomicUsize>,
}

impl morok_device::Program for QueueProgram {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
    ) -> morok_device::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn name(&self) -> &str {
        "queue_program"
    }
}

#[test]
fn test_cpu_queue_creation() {
    let queue = CpuQueue::new();
    assert_eq!(queue.device(), &DeviceSpec::Cpu);
}

#[test]
fn test_cpu_queue_submit_empty() {
    let mut queue = CpuQueue::new();
    queue.submit().unwrap();
}

#[test]
fn test_cpu_queue_memory_barrier() {
    let mut queue = CpuQueue::new();
    queue.memory_barrier();
    queue.submit().unwrap();
}

#[test]
fn test_cpu_queue_signal_is_not_dropped() {
    let mut queue = CpuQueue::new();
    let signal = CpuTimelineSignal::new();

    queue.signal(&signal, 7);
    queue.submit().unwrap();

    assert_eq!(signal.value(), 7);
}

#[test]
fn test_dyn_queue_wait_signal_are_forwarded() {
    let signal = CpuTimelineSignal::new();
    let mut queue = morok_device::DynQueue::new(CpuQueue::new());

    queue.signal(&signal, 11).wait(&signal, 11).submit().unwrap();

    assert_eq!(signal.value(), 11);
}

#[test]
fn test_cpu_queue_exec_runs_on_submit() {
    let mut queue = CpuQueue::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let program = Arc::new(QueueProgram { calls: calls.clone() });

    queue.exec(program, &[], &ExecParams::default());
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    queue.submit().unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
