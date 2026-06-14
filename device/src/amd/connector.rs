//! `PoolQueue`: a SHARED dispatch queue + lightweight per-owner contexts.
//!
//! A `PoolQueue` bundles the dispatch state that backs a single KFD compute
//! queue: the queue itself (ring + doorbell), a kernarg bump arena, the scratch
//! backing, the PM4 monotonic completion counter, and the pool-level in-flight
//! AQL signal list. It is a SHARED resource — multiple owners hold an
//! `Arc<PoolQueue>` and co-tenant the same ring. Dispatch atomicity comes from
//! `dispatch_lock`: an owner holds it
//! across a whole op (kernarg bump + write + ring submission) so kernarg order
//! ≡ ring order on a shared queue. Cross-queue parallelism comes from the pool
//! holding several `PoolQueue`s — the GPU's MES interleaves their independent
//! rings on the CP pipes.
//!
//! [`OwnerCtx`] is the lightweight per-owner handle. It holds an
//! `Arc<PoolQueue>` plus the owner's own completion bookkeeping (its last AQL
//! signal / its last PM4 counter value), so `OwnerCtx::synchronize` can drain
//! only this owner's work — the owner-local fast path. The whole-pool drain
//! (`PoolQueue::drain_all`, reached via `AmdDeviceCore::synchronize_all`) is the
//! host-visibility/free fence used by `AmdAllocator::_copyin`/`_copyout`/`_free`.

#![cfg(unix)]

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::amd::AmdAllocator;
use crate::amd::device::{AmdDeviceCore, ScratchState, alloc_scratch};
use crate::amd::kernarg::KernargArena;
use crate::amd::queue::AmdComputeQueue;
use crate::amd::signal::{AmdSignal, TIMELINE_WRAP_WATERMARK, Timeline};
use crate::error::{Error, Result};

/// A SHARED dispatch queue. Multiple owners hold `Arc<PoolQueue>` and co-tenant
/// the same KFD compute queue; the pool holds a bounded number of these so
/// distinct owners can run on distinct queues (cross-queue parallelism) without
/// exhausting KFD queues.
///
/// Owns: KFD compute queue, kernarg arena, scratch backing, PM4 completion
/// counter, pool-level in-flight AQL signals. `dispatch_lock` serializes a
/// whole op (bump + write + submit) so a shared ring's kernarg order matches
/// its ring order.
#[derive(Debug)]
pub struct PoolQueue {
    /// Shared immutable identity. Cloned across all queues backed by the same
    /// physical AMD:N (and across `AmdDevice` for back-compat).
    core: Arc<AmdDeviceCore>,
    /// The KFD compute queue — own ring + doorbell + GART. Its `inner` is a
    /// `parking_lot::Mutex<QueueInner>`, so a shared `Arc<PoolQueue>` can be
    /// dispatched by several co-tenant owners safely; the brief critical
    /// section is the packet write + doorbell. Distinct `PoolQueue`s' queues
    /// are interleaved by the GPU's MES, not a CPU lock.
    queue: Box<AmdComputeQueue>,
    /// Kernel-argument bump arena (16 MiB GTT). One per `PoolQueue`. The
    /// `dispatch_lock` held across bump + write + dispatch makes the bump cursor
    /// order match the ring submission order, so a wrapped slot is provably free
    /// once the whole pool drains. Freed on the queue's drop via
    /// `Drop for KernargArena`, after `Drop for PoolQueue` has drained.
    arena: Box<KernargArena>,
    /// Scratch backing. Grown on demand by [`ensure_has_local_memory`](Self::ensure_has_local_memory)
    /// under the `dispatch_lock` (park-and-grow on a live multi-XCC queue).
    scratch_state: Mutex<ScratchState>,
    /// PM4 monotonic completion counter + its signal. Used by the PM4 single-XCC
    /// dispatch path and the SDMA-style monotonic drain. (The SDMA copy queue
    /// has its own separate `Timeline`; this one is untouched by SDMA.) The AQL
    /// `execute_on` path uses `inflight` instead.
    pm4_counter: Arc<Timeline>,
    /// Serializes a whole op (kernarg bump + write + ring submission) on this
    /// shared queue, and the scratch park-and-grow. Lock order is always
    /// `dispatch_lock` → queue inner `Mutex`, never the reverse.
    dispatch_lock: Mutex<()>,
    /// Pool-level in-flight native AQL completion signals (FIFO) — ALL owners'
    /// in-flight work on this queue. Each AQL dispatch arms a pool signal to 1;
    /// the packet processor decrements it to 0 on completion. Retired front
    /// slots are reclaimed on the next [`acquire_signal`](Self::acquire_signal)
    /// or drained by [`drain_all`](Self::drain_all). `synchronize_all` drains it
    /// via the core's `Weak<PoolQueue>` registry without touching the queue.
    inflight: Mutex<VecDeque<Arc<AmdSignal>>>,
}

impl PoolQueue {
    /// Build a pool queue with its own KFD compute queue + kernarg arena +
    /// pre-reserved scratch. The PM4 counter signal is acquired from the core's
    /// shared `SignalPool` (which the factory must have installed first —
    /// `AmdDeviceCore::install_signal_pool`).
    ///
    /// Registers `Weak::self` in the core's queue list so device-wide
    /// `synchronize_all` (called by `AmdAllocator::_copyin`/`_copyout`/`_free`)
    /// drains every queue before any host-visible buffer free.
    pub fn new_with_resources(core: Arc<AmdDeviceCore>, allocator: &AmdAllocator) -> Result<Arc<Self>> {
        // Order matters: every step that allocates must come BEFORE
        // `alloc_scratch`. Earlier-built resources (`AmdComputeQueue`,
        // `KernargArena`, signal slot, counter Arc) all have RAII cleanup, so a
        // failure before the scratch alloc unwinds via `?`. The scratch backing
        // is the lone raw KFD allocation — keeping it last means a failure
        // before then leaks nothing.
        let queue = AmdComputeQueue::create(allocator)?;
        let arena = KernargArena::new(allocator, &core)?;
        let pool = core.signal_pool().cloned().ok_or_else(|| Error::Runtime {
            message: "PoolQueue::new_with_resources: signal pool not installed on core — \
                      install via AmdDeviceCore::install_signal_pool before building any queue"
                .into(),
        })?;
        let pm4_counter = Timeline::new(Arc::new(pool.acquire()?));
        // Pre-reserve a generous scratch buffer once (ROCr-style stable pool): the
        // mid-stream `set_aql_scratch` REALLOC (new VA + KFD-unmap of the old) on a
        // live multi-XCC queue silently wedges the CP. Pre-sizing here so no typical
        // kernel triggers a grow.
        let (scratch_va, scratch_size, tmpring_size, size_per_thread, scratch_handle, aql_desc) =
            alloc_scratch(core.iface(), &core.node, &core.arch, 2048)?;
        let q = Arc::new(Self {
            core,
            queue,
            arena,
            scratch_state: Mutex::new(ScratchState {
                gpu_va: scratch_va,
                size_per_thread,
                tmpring_size,
                handle: scratch_handle,
                size: scratch_size,
                aql_desc,
            }),
            pm4_counter,
            dispatch_lock: Mutex::new(()),
            inflight: Mutex::new(VecDeque::new()),
        });
        // Register in the core so `synchronize_all` can drain this queue's
        // in-flight work via `Weak<PoolQueue>`, reading only signal slots and
        // never touching the queue. Opportunistic GC of dropped entries.
        {
            let mut list = q.core.connectors.lock();
            list.retain(|w| w.strong_count() > 0);
            list.push(Arc::downgrade(&q));
        }
        // Publish the initial scratch descriptor into the AQL queue's GART page
        // (no-op on PM4 queues). Must happen before the first dispatch.
        q.queue().set_aql_scratch(&aql_desc);
        Ok(q)
    }

    /// Borrow this queue's KFD compute queue.
    #[inline]
    pub fn queue(&self) -> &AmdComputeQueue {
        &self.queue
    }

    /// Borrow this queue's kernarg arena.
    #[inline]
    pub fn arena(&self) -> &KernargArena {
        &self.arena
    }

    /// The immutable core this queue dispatches against.
    #[inline]
    pub fn core(&self) -> &Arc<AmdDeviceCore> {
        &self.core
    }

    /// Acquire the dispatch lock. An owner holds this across a whole op (kernarg
    /// bump + write + ring submission) so kernarg order ≡ ring order on the
    /// shared queue, and the scratch park-and-grow holds it to fence co-tenant
    /// dispatches during the swap. Lock order: `dispatch_lock` → queue inner.
    #[inline]
    pub fn dispatch_guard(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.dispatch_lock.lock()
    }

    /// PM4 completion-counter signal (forwards to the shared `Timeline`).
    pub fn pm4_signal(&self) -> &Arc<AmdSignal> {
        self.pm4_counter.signal()
    }

    /// Reserve the next PM4 counter value (`fetch_add(1)`). The caller emits a
    /// `RELEASE_MEM` packet that writes this value to the counter's signal slot.
    pub fn next_pm4(&self) -> u64 {
        self.pm4_counter.next()
    }

    /// Highest submitted PM4 counter value (the value the next `signal` packet
    /// would write). A drain waits until the GPU has written `value - 1`.
    pub fn pm4_value(&self) -> u64 {
        self.pm4_counter.current()
    }

    /// Current scratch buffer GPU VA. Read under the scratch mutex; tiny lock
    /// window on the dispatch hot path.
    pub fn scratch_gpu_va(&self) -> u64 {
        self.scratch_state.lock().gpu_va
    }

    /// Packed `COMPUTE_TMPRING_SIZE` for the current scratch buffer.
    pub fn tmpring_size(&self) -> u32 {
        self.scratch_state.lock().tmpring_size
    }

    /// Atomic snapshot of the scratch state — one lock window, so the PM4
    /// dispatch's VA, tmpring, and SRD words all describe the same buffer
    /// even when a concurrent grow swaps it.
    pub(crate) fn scratch_snapshot(&self) -> crate::amd::device::ScratchState {
        *self.scratch_state.lock()
    }

    /// Drain ALL submitted GPU work on this queue (every owner's). Blocks until
    /// the PM4 counter observes `pm4_value() - 1`, then waits every in-flight
    /// AQL signal. Reads only signal slots — never takes `dispatch_lock` — so
    /// holding `dispatch_lock` across a caller that also calls `drain_all`
    /// (e.g. `ensure_has_local_memory`) is deadlock-free.
    pub fn drain_all(&self) -> Result<()> {
        if let Some(err) = self.core.poison_error() {
            return Err(err);
        }
        // Drain the monotonic PM4 counter (PM4 / SDMA-style work)...
        self.pm4_counter.drain(30_000).inspect_err(|e| self.core.poison(&e.to_string()))?;
        // ...then every in-flight native completion signal (AQL per-op work).
        // Snapshot under the lock, wait outside it, then drop the retired ones —
        // `retain` keeps any signal a concurrent dispatch armed after the
        // snapshot, so we never lose track of still-pending work.
        let snapshot: Vec<Arc<AmdSignal>> = self.inflight.lock().iter().cloned().collect();
        for sig in &snapshot {
            sig.wait_done(30_000).inspect_err(|e| self.core.poison(&e.to_string()))?;
        }
        self.inflight.lock().retain(|s| !s.is_done());
        Ok(())
    }

    /// Acquire a fresh native completion signal for one AQL dispatch. First
    /// reclaims retired in-flight signals (FIFO — they return their pool slots
    /// on drop), then takes a new slot. If the pool is momentarily exhausted,
    /// blocks on the OLDEST in-flight dispatch (the queue head — pool-level, so
    /// this is whichever owner's dispatch is oldest): the back-pressure that
    /// bounds how far the host can run ahead of the GPU. The returned signal is
    /// armed to 1; the caller places [`signal_handle`](AmdSignal::signal_handle)
    /// in the dispatch packet and registers it via [`register_inflight`](Self::register_inflight).
    pub fn acquire_signal(&self) -> Result<Arc<AmdSignal>> {
        let pool = self
            .core
            .signal_pool()
            .cloned()
            .ok_or_else(|| Error::Runtime { message: "acquire_signal: signal pool not installed".into() })?;
        loop {
            // Reclaim any retired signals from the front (FIFO completion order).
            {
                let mut inflight = self.inflight.lock();
                while inflight.front().is_some_and(|s| s.is_done()) {
                    inflight.pop_front();
                }
            }
            match pool.acquire() {
                Ok(sig) => {
                    sig.arm(1);
                    return Ok(Arc::new(sig));
                }
                // Pool exhausted: block on the oldest in-flight dispatch (queue
                // head), drop it, and retry. If nothing is in flight there is no
                // slot to wait for, so surface the exhaustion error.
                Err(e) => {
                    let oldest = self.inflight.lock().front().cloned();
                    match oldest {
                        Some(sig) => {
                            sig.wait_done(30_000).inspect_err(|e| self.core.poison(&e.to_string()))?;
                            self.inflight.lock().pop_front();
                        }
                        None => return Err(e),
                    }
                }
            }
        }
    }

    /// Reserve a raw signal slot held for a graph's lifetime — armed to 1 but
    /// NOT registered in `inflight`. Unlike [`acquire_signal`](Self::acquire_signal),
    /// these are not FIFO in-flight completion signals: a DAG graph holds one per
    /// kernel for its whole lifetime and re-arms them each replay, so they must
    /// not be reclaimed by the in-flight FIFO. Returns `Err` when the pool is
    /// exhausted (the caller falls back to blanket-BARRIER capture).
    pub fn reserve_signal(&self) -> Result<Arc<AmdSignal>> {
        let pool = self
            .core
            .signal_pool()
            .cloned()
            .ok_or_else(|| Error::Runtime { message: "reserve_signal: signal pool not installed".into() })?;
        let sig = pool.acquire()?;
        sig.arm(1);
        Ok(Arc::new(sig))
    }

    /// Register a dispatched AQL completion signal as in-flight (FIFO,
    /// pool-level). Kept alive until it retires and a later acquire/drain
    /// reclaims it.
    pub fn register_inflight(&self, sig: Arc<AmdSignal>) {
        self.inflight.lock().push_back(sig);
    }

    /// Free slots in the shared signal pool (0 if not installed). Graph capture
    /// checks this before reserving one slot per kernel so a large graph can't
    /// drain the pool below the headroom per-op dispatch + PM4 counters need.
    pub fn signal_free(&self) -> usize {
        self.core.signal_pool().map(|p| p.free()).unwrap_or(0)
    }

    /// Keep the PM4 counter below 2^32 on the dispatch hot path.
    ///
    /// A drain resets the counter on wraparound, but a queue dispatched in a
    /// long `wait=false` PM4 loop never hits one, so the full-u64 counter would
    /// climb past 2^32 while the GPU's `RELEASE_MEM` writes only the low 32
    /// bits. A later drain would then wait for a full-u64 `target` the slot can
    /// never reach → false 30 s timeout. Calling this before reserving each
    /// counter value forces the drain+reset at the 2^31 watermark, so the
    /// reserved value stays `< 2^32` and the `as u32` truncations stay lossless.
    /// The dispatch lock held across the op makes the check + drain sequential.
    pub fn ensure_pm4_headroom(&self) -> Result<()> {
        if self.pm4_counter.current() > TIMELINE_WRAP_WATERMARK {
            self.drain_all()?;
        }
        Ok(())
    }

    /// Ensure the queue's scratch backing has at least `private_segment_size`
    /// bytes per thread, growing it on demand. The old scratch buffer is freed
    /// (drain → unmap → munmap → free).
    pub fn ensure_has_local_memory(&self, private_segment_size: u32) -> Result<()> {
        let current = self.scratch_state.lock().size_per_thread;
        if private_segment_size <= current {
            return Ok(());
        }
        // Serialize the realloc (alloc new → swap → drain → free old) against
        // concurrent co-tenant dispatchers on this shared queue. Holding the
        // dispatch lock blocks every co-tenant from enqueuing the stale scratch
        // VA during the swap. `drain_all` waits only on signals
        // (never takes `dispatch_lock`), so this is deadlock-free.
        let _g = self.dispatch_lock.lock();
        // Re-check under the guard: another dispatcher may have grown scratch
        // while we waited for the lock.
        if private_segment_size <= self.scratch_state.lock().size_per_thread {
            return Ok(());
        }
        let (va, size, tmpring, rounded, handle, aql_desc) =
            alloc_scratch(self.core.iface(), &self.core.node, &self.core.arch, private_segment_size)?;
        let swapped = {
            let mut state = self.scratch_state.lock();
            if rounded > state.size_per_thread {
                let old = (state.gpu_va, state.size, state.handle);
                *state = ScratchState {
                    gpu_va: va,
                    size_per_thread: rounded,
                    tmpring_size: tmpring,
                    handle,
                    size,
                    aql_desc,
                };
                Some(old)
            } else {
                None
            }
        };
        match swapped {
            // Park-and-grow — the only safe ordering on a live multi-XCC queue.
            // Drain to idle first so no dispatch is mid-flight or still reading
            // the old scratch; publish the NEW descriptor (self-flushed to GART)
            // so the live `amd_queue_t` never points at soon-to-be-unmapped
            // VRAM; only THEN free the old backing. Skipping the drain (or
            // freeing before republish) silently wedges the CP. No-op republish
            // on PM4 queues.
            Some((old_va, old_size, old_handle)) => {
                if let Err(e) = self.drain_all() {
                    tracing::warn!(?e, "scratch grow: drain failed; proceeding with republish");
                }
                self.queue.set_aql_scratch(&aql_desc);
                self.core.iface().free_raw(old_va, old_size, old_handle);
            }
            // Lost the race — another dispatcher already grew scratch past our
            // target. Free the buffer we redundantly allocated.
            None => self.free_scratch(va, size, handle),
        }
        Ok(())
    }

    /// Drain → unmap → munmap → free a scratch backing buffer. Old scratch is no
    /// longer referenced once the queue drains.
    fn free_scratch(&self, va: u64, size: usize, handle: u64) {
        if let Err(e) = self.drain_all() {
            tracing::warn!(?e, va, "scratch realloc: drain failed; freeing anyway");
        }
        self.core.iface().free_raw(va, size, handle);
    }
}

impl Drop for PoolQueue {
    /// Drain in-flight GPU work before the queue dies (device close — the LAST
    /// `Arc<PoolQueue>` dropping), so a downstream host read of a buffer this
    /// queue wrote doesn't race a still-running kernel, and free the scratch
    /// backing (`ScratchState` is `Copy` with no `Drop`).
    ///
    /// Skipped during panic unwind: `drain_all` can block up to ~30 s per queue
    /// and an unwinding test would pay N × 30 s before teardown. The in-flight
    /// work is then abandoned — the caller saw a panic anyway.
    fn drop(&mut self) {
        if std::thread::panicking() {
            tracing::warn!(
                "PoolQueue drop during panic unwind: skipping drain; \
                 in-flight GPU work + scratch backing abandoned"
            );
            return;
        }
        if let Err(e) = self.drain_all() {
            tracing::warn!(?e, "PoolQueue drop: drain failed (in-flight work lost)");
        }
        let state = *self.scratch_state.lock();
        self.free_scratch(state.gpu_va, state.size, state.handle);
    }
}

/// Lightweight per-owner dispatch context. Holds an `Arc<PoolQueue>` (the shared
/// queue this owner dispatches on) plus the owner's own completion bookkeeping,
/// so [`synchronize`](Self::synchronize) can wait on ONLY this owner's work (the
/// owner-local fast path). The whole-pool drain
/// (`AmdDeviceCore::synchronize_all` → `PoolQueue::drain_all`) is the
/// host-visibility/free fence.
pub struct OwnerCtx {
    pool: Arc<PoolQueue>,
    /// AQL: this owner's last in-flight completion signal.
    my_newest: Mutex<Option<Arc<AmdSignal>>>,
    /// PM4: this owner's last reserved counter value (0 = none yet).
    pm4_high: AtomicU64,
}

impl OwnerCtx {
    pub fn new(pool: Arc<PoolQueue>) -> Self {
        Self { pool, my_newest: Mutex::new(None), pm4_high: AtomicU64::new(0) }
    }

    /// Access the shared queue (queue / arena / scratch / acquire_signal /
    /// dispatch_guard).
    #[inline]
    pub fn pool(&self) -> &Arc<PoolQueue> {
        &self.pool
    }

    /// Record this owner's most recent in-flight AQL signal (the one to wait on
    /// in the owner-local `synchronize`).
    pub fn set_newest(&self, sig: Arc<AmdSignal>) {
        *self.my_newest.lock() = Some(sig);
    }

    /// Record this owner's highest reserved PM4 counter value.
    pub fn set_pm4_high(&self, v: u64) {
        self.pm4_high.store(v, Ordering::Release);
    }

    /// Owner-local drain: wait on ONLY this owner's last submitted work. AQL:
    /// wait its newest signal. PM4: wait the shared counter to reach this
    /// owner's high value. Polls the device poison latch and bails on fault.
    pub fn synchronize(&self) -> Result<()> {
        if let Some(err) = self.pool.core.poison_error() {
            return Err(err);
        }
        let newest = self.my_newest.lock().clone();
        if let Some(sig) = newest {
            sig.wait_done(30_000).inspect_err(|e| self.pool.core.poison(&e.to_string()))?;
            return Ok(());
        }
        let high = self.pm4_high.load(Ordering::Acquire);
        if high > 0 {
            self.pool
                .pm4_signal()
                .wait_signal_value(high, 30_000)
                .inspect_err(|e| self.pool.core.poison(&e.to_string()))?;
        }
        Ok(())
    }

    /// Identity of the shared queue this owner dispatches on — used by the
    /// concurrency test to assert distinct owners landed on distinct queues.
    #[cfg(test)]
    pub fn queue_ptr(&self) -> *const PoolQueue {
        Arc::as_ptr(&self.pool)
    }
}

impl std::fmt::Debug for OwnerCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnerCtx").finish_non_exhaustive()
    }
}

/// An `OwnerCtx` IS the per-plan execution context: it holds the leased queue
/// for the plan's lifetime, so every kernel dispatches onto the same ring.
impl crate::device::PlanContext for OwnerCtx {
    unsafe fn dispatch(
        &self,
        program: &dyn crate::device::Program,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> Result<Option<Arc<dyn crate::DispatchTimestamps>>> {
        // A plan is single-backend, and this context was minted by one of its
        // AmdPrograms, so every kernel it dispatches is an AmdProgram. The
        // downcast recovers our own concrete type — a construction invariant,
        // not a runtime check.
        let amd = program
            .as_any()
            .downcast_ref::<crate::amd::AmdProgram>()
            .expect("AMD PlanContext dispatched a non-AMD program");
        self.pool().ensure_has_local_memory(amd.private_segment_size())?;
        let sig = unsafe {
            amd.execute_on(self, buffers, vals, global_size, local_size, /*wait=*/ false)?
        };
        Ok(sig.map(|s| s as Arc<dyn crate::DispatchTimestamps>))
    }

    fn synchronize(&self) -> Result<()> {
        OwnerCtx::synchronize(self)
    }
}
