use super::*;
use std::sync::Arc;
use std::thread;

#[test]
fn test_cpu_signal_basic() {
    let signal = CpuTimelineSignal::new();
    assert_eq!(signal.value(), 0);

    signal.set(5);
    assert_eq!(signal.value(), 5);

    assert!(signal.is_reached(5));
    assert!(signal.is_reached(3));
    assert!(!signal.is_reached(10));
}

#[test]
fn test_cpu_signal_set_is_monotonic() {
    let signal = CpuTimelineSignal::new();
    signal.set(5);
    signal.set(3);

    assert_eq!(signal.value(), 5);
    signal.wait(5, 100).unwrap();
}

#[test]
fn test_cpu_signal_wait_already_reached() {
    let signal = CpuTimelineSignal::new();
    signal.set(10);

    // Should return immediately
    signal.wait(5, 100).unwrap();
    signal.wait(10, 100).unwrap();
}

#[test]
fn test_cpu_signal_wait_concurrent() {
    let signal = Arc::new(CpuTimelineSignal::new());
    let signal_clone = Arc::clone(&signal);

    let waiter = thread::spawn(move || {
        signal_clone.wait(5, 5000).unwrap();
        signal_clone.value()
    });

    // Give waiter time to block
    thread::sleep(std::time::Duration::from_millis(10));

    // Set the signal
    signal.set(5);

    let result = waiter.join().unwrap();
    assert!(result >= 5);
}

#[test]
fn test_cpu_signal_timeout() {
    let signal = CpuTimelineSignal::new();

    // Should timeout waiting for value 10
    let result = signal.wait(10, 50);
    assert!(result.is_err());
}
