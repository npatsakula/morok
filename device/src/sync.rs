//! Timeline synchronization primitives for parallel execution.
//!
//! This module provides device-agnostic synchronization using timeline signals,
//! which are monotonically increasing counters that enable ordering of operations
//! across devices.
//!
//! # Design
//!
//! Timeline signals abstract over:
//! - CPU: `AtomicU64` with parking_lot condvar for waiting
//! - CUDA: Event pools keyed by timeline value
//! - Metal: `MTLSharedEvent` (future)
//! - HIP: Similar to CUDA (future)
//!
//! # Example
//!
//! ```ignore
//! let signal = CpuTimelineSignal::new();
//!
//! // Producer thread
//! signal.set(1);  // Signal completion of operation 1
//!
//! // Consumer thread
//! signal.wait(1, 1000)?;  // Wait for operation 1 to complete
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::error::{Result, RuntimeSnafu};
use snafu::ensure;

/// Monotonic timeline signal for synchronization.
///
/// Timeline signals provide a way to order operations across different execution
/// contexts (threads, devices, queues). The signal value only increases, and
/// waiters block until the signal reaches or exceeds the target value.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` for cross-thread use.
pub trait TimelineSignal: Send + Sync + std::fmt::Debug {
    /// Get the current signal value.
    fn value(&self) -> u64;

    /// Set the signal to a new value.
    ///
    /// # Panics
    ///
    /// May panic if `value` is less than the current value (implementation-defined).
    fn set(&self, value: u64);

    /// Wait for the signal to reach or exceed `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - The target value to wait for
    /// * `timeout_ms` - Maximum time to wait in milliseconds (0 = infinite)
    ///
    /// # Returns
    ///
    /// `Ok(())` if the signal reached the target value, or `Err` on timeout.
    fn wait(&self, value: u64, timeout_ms: u64) -> Result<()>;

    /// Check if the signal has reached `value` without blocking.
    fn is_reached(&self, value: u64) -> bool {
        self.value() >= value
    }
}

/// CPU-based timeline signal using atomics and condvar.
///
/// Efficient for CPU-only workloads. Uses `AtomicU64` for the counter and
/// `parking_lot::Condvar` for efficient waiting.
#[derive(Debug)]
pub struct CpuTimelineSignal {
    /// Current timeline value (monotonically increasing).
    value: AtomicU64,
    /// Mutex for condvar waiting (protects nothing, just for condvar).
    mutex: Mutex<()>,
    /// Condvar for waiting threads.
    condvar: Condvar,
}

impl Default for CpuTimelineSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuTimelineSignal {
    /// Create a new CPU timeline signal starting at 0.
    pub fn new() -> Self {
        Self { value: AtomicU64::new(0), mutex: Mutex::new(()), condvar: Condvar::new() }
    }

    /// Create a new CPU timeline signal with an initial value.
    pub fn with_initial(initial: u64) -> Self {
        Self { value: AtomicU64::new(initial), mutex: Mutex::new(()), condvar: Condvar::new() }
    }
}

impl TimelineSignal for CpuTimelineSignal {
    fn value(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    fn set(&self, value: u64) {
        // Store the new value
        self.value.store(value, Ordering::Release);

        // Wake all waiters to check the new value
        self.condvar.notify_all();
    }

    fn wait(&self, target: u64, timeout_ms: u64) -> Result<()> {
        // Fast path: already reached
        if self.value.load(Ordering::Acquire) >= target {
            return Ok(());
        }

        let mut guard = self.mutex.lock();

        if timeout_ms == 0 {
            // Infinite wait
            while self.value.load(Ordering::Acquire) < target {
                self.condvar.wait(&mut guard);
            }
            Ok(())
        } else {
            // Timed wait
            let deadline = Instant::now() + Duration::from_millis(timeout_ms);

            while self.value.load(Ordering::Acquire) < target {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    ensure!(
                        self.value.load(Ordering::Acquire) >= target,
                        RuntimeSnafu {
                            message: format!(
                                "timeline signal timeout: waited {}ms for value {}, current {}",
                                timeout_ms,
                                target,
                                self.value.load(Ordering::Acquire)
                            )
                        }
                    );
                    return Ok(());
                }

                let result = self.condvar.wait_for(&mut guard, remaining);
                if result.timed_out() && self.value.load(Ordering::Acquire) < target {
                    return RuntimeSnafu {
                        message: format!(
                            "timeline signal timeout: waited {}ms for value {}, current {}",
                            timeout_ms,
                            target,
                            self.value.load(Ordering::Acquire)
                        ),
                    }
                    .fail();
                }
            }
            Ok(())
        }
    }
}

#[cfg(feature = "cuda")]
pub mod cuda {
    //! CUDA-specific timeline signal using event pools.

    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
    use parking_lot::Mutex;

    use super::TimelineSignal;
    use crate::error::{CudaSnafu, Result};
    use snafu::ResultExt;

    /// CUDA timeline signal using event pools.
    ///
    /// Each timeline value maps to a CUDA event. When `set(n)` is called,
    /// we record an event at value `n` on the stream. When `wait(n)` is called,
    /// we synchronize on that event.
    #[derive(Debug)]
    pub struct CudaTimelineSignal {
        /// Current timeline value.
        value: AtomicU64,
        /// Event pool: timeline value → recorded event.
        events: Mutex<HashMap<u64, Arc<CudaEvent>>>,
        /// CUDA context for creating events.
        context: Arc<CudaContext>,
        /// Stream for recording events.
        stream: Arc<CudaStream>,
    }

    impl CudaTimelineSignal {
        /// Create a new CUDA timeline signal.
        pub fn new(context: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self {
            Self { value: AtomicU64::new(0), events: Mutex::new(HashMap::new()), context, stream }
        }

        /// Record an event at the given timeline value.
        ///
        /// This should be called after submitting work to the stream.
        pub fn record(&self, value: u64) -> Result<()> {
            let event = self.context.create_event(None).context(CudaSnafu)?;
            self.stream.record(&event).context(CudaSnafu)?;

            let mut events = self.events.lock();
            events.insert(value, Arc::new(event));

            // Update the timeline value
            self.value.fetch_max(value, Ordering::Release);

            // Clean up old events (keep last 16)
            if events.len() > 32 {
                let current = self.value.load(Ordering::Acquire);
                events.retain(|&v, _| v > current.saturating_sub(16));
            }

            Ok(())
        }

        /// Get the stream for this signal.
        pub fn stream(&self) -> &Arc<CudaStream> {
            &self.stream
        }
    }

    impl TimelineSignal for CudaTimelineSignal {
        fn value(&self) -> u64 {
            self.value.load(Ordering::Acquire)
        }

        fn set(&self, value: u64) {
            // For CUDA, set() should be called via record() after stream work.
            // This is a fallback that just updates the counter.
            self.value.fetch_max(value, Ordering::Release);
        }

        fn wait(&self, target: u64, timeout_ms: u64) -> Result<()> {
            // Fast path: already reached
            if self.value.load(Ordering::Acquire) >= target {
                return Ok(());
            }

            // Find the event for this timeline value
            let event = {
                let events = self.events.lock();
                // Find the smallest event >= target
                events.iter().filter(|(&v, _)| v >= target).min_by_key(|(&v, _)| v).map(|(_, e)| Arc::clone(e))
            };

            if let Some(event) = event {
                if timeout_ms == 0 {
                    // Synchronous wait
                    event.synchronize().context(CudaSnafu)?;
                } else {
                    // Polling wait with timeout
                    let start = std::time::Instant::now();
                    let timeout = std::time::Duration::from_millis(timeout_ms);

                    while !event.is_ready() {
                        if start.elapsed() > timeout {
                            return crate::error::RuntimeSnafu {
                                message: format!(
                                    "CUDA timeline signal timeout: waited {}ms for value {}",
                                    timeout_ms, target
                                ),
                            }
                            .fail();
                        }
                        std::thread::sleep(std::time::Duration::from_micros(100));
                    }
                }
            } else {
                // No event recorded yet - spin wait
                let start = std::time::Instant::now();
                let timeout = if timeout_ms == 0 {
                    std::time::Duration::MAX
                } else {
                    std::time::Duration::from_millis(timeout_ms)
                };

                while self.value.load(Ordering::Acquire) < target {
                    if start.elapsed() > timeout {
                        return crate::error::RuntimeSnafu {
                            message: format!(
                                "CUDA timeline signal timeout: waited {}ms for value {}, current {}",
                                timeout_ms,
                                target,
                                self.value.load(Ordering::Acquire)
                            ),
                        }
                        .fail();
                    }
                    std::thread::yield_now();
                }
            }

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
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
}
