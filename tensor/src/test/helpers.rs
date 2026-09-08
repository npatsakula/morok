//! Test helpers for realize() validation.

use crate::{PrepareConfig, Tensor};

/// Convenience extension for tests: realize in-place and return `&Self` for chaining reads.
pub trait RealizeTestExt {
    fn realize_with_and(&self, config: &PrepareConfig) -> &Self;
}

impl RealizeTestExt for Tensor {
    /// Realize with explicit config, panic on error, return `&self` for reading.
    fn realize_with_and(&self, config: &PrepareConfig) -> &Self {
        self.realize_with(config).unwrap();
        self
    }
}

/// Setup function to call at the start of each test.
///
/// No-op today: the kernel cache is intentionally process-static and
/// deduped by `(ast_id, device)`, so cross-test entries don't interfere —
/// identical ASTs hand back the same `Arc<CachedKernel>`, distinct ASTs
/// occupy distinct slots, and the cache is never torn down mid-process.
pub fn test_setup() {}

/// Compare float slices with tolerance.
#[track_caller]
pub fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "Length mismatch: {} != {}", actual.len(), expected.len());

    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < tol, "Mismatch at index {}: {} != {} (diff: {})", i, a, e, (a - e).abs());
    }
}
