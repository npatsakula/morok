//! In-flight computation dedup ("singleflight") for the global prepare-time
//! caches. N threads missing the same key run the expensive computation once:
//! the winner computes while losers park on a condvar, then re-check the
//! cache. A failed (or panicked) winner wakes the losers with nothing
//! inserted, so the next loser retries and becomes the winner — errors
//! propagate to their caller without being cached.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

type Slot = Arc<(Mutex<bool>, Condvar)>;

pub(crate) struct Singleflight<K> {
    inflight: Mutex<HashMap<K, Slot>>,
}

impl<K: Hash + Eq + Clone> Singleflight<K> {
    pub(crate) fn new() -> Self {
        Self { inflight: Mutex::new(HashMap::new()) }
    }

    /// Run `compute` for `key` unless another thread already is; in that case
    /// wait for it and re-`lookup`. `compute` is responsible for inserting
    /// its result into the underlying cache before returning, or the losers
    /// will recompute.
    pub(crate) fn run<V, E>(
        &self,
        key: K,
        lookup: impl Fn() -> Option<V>,
        compute: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E> {
        loop {
            if let Some(hit) = lookup() {
                return Ok(hit);
            }
            match self.claim(key.clone()) {
                Ok(_ticket) => {
                    return match lookup() {
                        // Raced an earlier winner between our first lookup and
                        // slot registration — take the published result.
                        Some(hit) => Ok(hit),
                        None => compute(),
                    };
                }
                Err(slot) => {
                    let mut done = slot.0.lock();
                    while !*done {
                        slot.1.wait(&mut done);
                    }
                    // winner finished (or failed) — re-check the cache
                }
            }
        }
    }

    /// Become the computing thread for `key` without waiting: `Some` when no
    /// other thread holds it. Batch callers claim several keys up front and
    /// must never block on a foreign key while holding tickets, or two
    /// batches claiming in different orders would deadlock; `run` on the
    /// foreign keys after the batch is the safe way to wait.
    pub(crate) fn try_claim(&self, key: K) -> Option<Ticket<'_, K>> {
        self.claim(key).ok()
    }

    /// [`try_claim`](Self::try_claim) for a cache miss: `lookup` runs again
    /// under the ticket, so a winner that published between the caller's
    /// lookup and the claim is not recomputed (under a second kernel name).
    pub(crate) fn try_claim_miss<V>(&self, key: K, lookup: impl Fn() -> Option<V>) -> Option<Ticket<'_, K>> {
        if lookup().is_some() {
            return None;
        }
        let ticket = self.try_claim(key)?;
        lookup().is_none().then_some(ticket)
    }

    fn claim(&self, key: K) -> Result<Ticket<'_, K>, Slot> {
        let mut inflight = self.inflight.lock();
        match inflight.entry(key.clone()) {
            std::collections::hash_map::Entry::Occupied(entry) => Err(entry.get().clone()),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let slot = entry.insert(Arc::new((Mutex::new(false), Condvar::new()))).clone();
                Ok(Ticket { flight: self, key, slot })
            }
        }
    }
}

/// The winner's hold on a key. Dropping it — including on unwind, so a
/// panicking computation cannot strand waiters — wakes every loser, which
/// then re-checks the cache.
pub(crate) struct Ticket<'a, K: Hash + Eq + Clone> {
    flight: &'a Singleflight<K>,
    key: K,
    slot: Slot,
}

impl<K: Hash + Eq + Clone> Drop for Ticket<'_, K> {
    fn drop(&mut self) {
        self.flight.inflight.lock().remove(&self.key);
        *self.slot.0.lock() = true;
        self.slot.1.notify_all();
    }
}

#[cfg(test)]
#[path = "test/unit/singleflight.rs"]
mod tests;
