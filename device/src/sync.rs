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

use std::any::Any;
use std::sync::Arc;
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
pub trait TimelineSignal: Send + Sync + std::fmt::Debug + Any {
    /// Return this signal as `Any` for checked type-erased queue dispatch.
    fn as_any(&self) -> &dyn Any;

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
#[derive(Debug, Clone)]
pub struct CpuTimelineSignal {
    inner: Arc<CpuTimelineSignalInner>,
}

#[derive(Debug)]
struct CpuTimelineSignalInner {
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
        Self {
            inner: Arc::new(CpuTimelineSignalInner {
                value: AtomicU64::new(0),
                mutex: Mutex::new(()),
                condvar: Condvar::new(),
            }),
        }
    }

    /// Create a new CPU timeline signal with an initial value.
    pub fn with_initial(initial: u64) -> Self {
        Self {
            inner: Arc::new(CpuTimelineSignalInner {
                value: AtomicU64::new(initial),
                mutex: Mutex::new(()),
                condvar: Condvar::new(),
            }),
        }
    }
}

impl TimelineSignal for CpuTimelineSignal {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn value(&self) -> u64 {
        self.inner.value.load(Ordering::Acquire)
    }

    fn set(&self, value: u64) {
        let previous = self.inner.value.fetch_max(value, Ordering::AcqRel);
        if value > previous {
            self.inner.condvar.notify_all();
        }
    }

    fn wait(&self, target: u64, timeout_ms: u64) -> Result<()> {
        // Fast path: already reached
        if self.inner.value.load(Ordering::Acquire) >= target {
            return Ok(());
        }

        let mut guard = self.inner.mutex.lock();

        if timeout_ms == 0 {
            // Infinite wait
            while self.inner.value.load(Ordering::Acquire) < target {
                self.inner.condvar.wait(&mut guard);
            }
            Ok(())
        } else {
            // Timed wait
            let deadline = Instant::now() + Duration::from_millis(timeout_ms);

            while self.inner.value.load(Ordering::Acquire) < target {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    ensure!(
                        self.inner.value.load(Ordering::Acquire) >= target,
                        RuntimeSnafu {
                            message: format!(
                                "timeline signal timeout: waited {}ms for value {}, current {}",
                                timeout_ms,
                                target,
                                self.inner.value.load(Ordering::Acquire)
                            )
                        }
                    );
                    return Ok(());
                }

                let result = self.inner.condvar.wait_for(&mut guard, remaining);
                if result.timed_out() && self.inner.value.load(Ordering::Acquire) < target {
                    return RuntimeSnafu {
                        message: format!(
                            "timeline signal timeout: waited {}ms for value {}, current {}",
                            timeout_ms,
                            target,
                            self.inner.value.load(Ordering::Acquire)
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

    use std::any::Any;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
    use parking_lot::Mutex;

    use super::TimelineSignal;
    use crate::error::{CudaSnafu, Result};
    use snafu::ResultExt;

    /// Number of event slots in the ring. 64 is well above typical in-flight
    /// depth and bounds the worst-case event memory.
    const EVENT_RING_SIZE: usize = 64;

    /// One occupied slot in the event ring.
    #[derive(Debug)]
    struct EventSlot {
        timeline_value: u64,
        event: Arc<CudaEvent>,
    }

    /// CUDA timeline signal using a fixed-size ring of event slots.
    ///
    /// Each `record(n)` lands in the next slot of the ring. Waiters look up
    /// the smallest slot whose `timeline_value >= target` and synchronise on
    /// its event. When the ring wraps and a slot is overwritten, the previous
    /// `Arc<CudaEvent>` is released only after every waiter that fetched it
    /// drops their clone (Arc lifetime semantics) — slots are never torn out
    /// from under outstanding waiters.
    #[derive(Debug)]
    pub struct CudaTimelineSignal {
        /// Current timeline value.
        value: AtomicU64,
        /// Ring of recorded (timeline_value, event) slots and a next-write cursor.
        ring: Mutex<EventRing>,
        /// CUDA context for creating events.
        context: Arc<CudaContext>,
        /// Stream for recording events.
        stream: Arc<CudaStream>,
    }

    #[derive(Debug)]
    struct EventRing {
        slots: [Option<EventSlot>; EVENT_RING_SIZE],
        next: usize,
    }

    impl EventRing {
        fn new() -> Self {
            Self { slots: std::array::from_fn(|_| None), next: 0 }
        }
    }

    impl CudaTimelineSignal {
        /// Create a new CUDA timeline signal.
        pub fn new(context: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self {
            Self { value: AtomicU64::new(0), ring: Mutex::new(EventRing::new()), context, stream }
        }

        /// Record an event at the given timeline value.
        ///
        /// Called after submitting work to the stream. The new (value, event)
        /// pair occupies the next ring slot, overwriting whatever was there.
        /// Overwriting is safe: any waiter that looked up that slot already
        /// holds an `Arc<CudaEvent>` clone, which keeps the event alive past
        /// the slot's overwrite — i.e. no event is dropped while a waiter
        /// still references it.
        pub fn record(&self, value: u64) -> Result<()> {
            let event = self.context.create_event(None).context(CudaSnafu)?;
            self.stream.record(&event).context(CudaSnafu)?;

            let mut ring = self.ring.lock();
            let slot_idx = ring.next;
            ring.slots[slot_idx] = Some(EventSlot { timeline_value: value, event: Arc::new(event) });
            ring.next = (slot_idx + 1) % EVENT_RING_SIZE;
            drop(ring);

            // Update the timeline value. AcqRel keeps load-half non-Relaxed so concurrent
            // record/set observe each other's monotonic updates.
            self.value.fetch_max(value, Ordering::AcqRel);

            Ok(())
        }

        /// Get the stream for this signal.
        pub fn stream(&self) -> &Arc<CudaStream> {
            &self.stream
        }
    }

    impl TimelineSignal for CudaTimelineSignal {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn value(&self) -> u64 {
            self.value.load(Ordering::Acquire)
        }

        fn set(&self, value: u64) {
            // For CUDA, set() should be called via record() after stream work.
            // This is a fallback that just updates the counter.
            self.value.fetch_max(value, Ordering::AcqRel);
        }

        fn wait(&self, target: u64, timeout_ms: u64) -> Result<()> {
            // Fast path: already reached
            if self.value.load(Ordering::Acquire) >= target {
                return Ok(());
            }

            // Find the smallest event in the ring with timeline_value >= target.
            // Cloning the Arc keeps the event alive even if the slot is later
            // overwritten by a recycling record() — no torn waiters.
            let event = {
                let ring = self.ring.lock();
                ring.slots
                    .iter()
                    .filter_map(|slot| slot.as_ref().filter(|s| s.timeline_value >= target))
                    .min_by_key(|s| s.timeline_value)
                    .map(|s| Arc::clone(&s.event))
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
                // No event matched. This can happen when (a) no record() has
                // landed for `target` yet, or (b) we lost a race against
                // sufficient record()s to wrap the ring and overwrite the
                // slot satisfying `target` between the fast-path load above
                // and this lookup. Re-check the timeline counter before
                // entering the spin so a race-loser exits immediately rather
                // than busy-yielding for an already-completed target.
                if self.value.load(Ordering::Acquire) >= target {
                    return Ok(());
                }

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
#[path = "test/unit/sync.rs"]
mod tests;
