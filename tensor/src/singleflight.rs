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
        let mut compute = Some(compute);
        loop {
            if let Some(hit) = lookup() {
                return Ok(hit);
            }
            let (slot, winner) = {
                let mut inflight = self.inflight.lock();
                match inflight.entry(key.clone()) {
                    std::collections::hash_map::Entry::Occupied(entry) => (entry.get().clone(), false),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        (entry.insert(Arc::new((Mutex::new(false), Condvar::new()))).clone(), true)
                    }
                }
            };
            if !winner {
                let mut done = slot.0.lock();
                while !*done {
                    slot.1.wait(&mut done);
                }
                continue; // winner finished (or failed) — re-check the cache
            }
            // Winner. The guard wakes waiters even on unwind, so a panicking
            // computation cannot strand them.
            let _wake = WakeOnDrop { flight: self, key: &key, slot: &slot };
            return match lookup() {
                // Raced an earlier winner between our first lookup and slot
                // registration — take the published result.
                Some(hit) => Ok(hit),
                None => compute.take().expect("winner runs once")(),
            };
        }
    }
}

struct WakeOnDrop<'a, K: Hash + Eq + Clone> {
    flight: &'a Singleflight<K>,
    key: &'a K,
    slot: &'a Slot,
}

impl<K: Hash + Eq + Clone> Drop for WakeOnDrop<'_, K> {
    fn drop(&mut self) {
        self.flight.inflight.lock().remove(self.key);
        *self.slot.0.lock() = true;
        self.slot.1.notify_all();
    }
}
