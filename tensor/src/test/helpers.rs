//! Test helpers for realize() validation.

use crate::Tensor;
use ndarray::ArrayD;

// Mutex to serialize tests that use global caches.
// With papaya-based global caches, tests running in parallel can interfere.
// Tests that call test_setup() acquire this mutex to prevent races.
//
// Using parking_lot::Mutex to avoid mutex poisoning when tests panic.
pub static CACHE_TEST_MUTEX: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Setup function to call at the start of each test to ensure isolation.
///
/// This acquires a mutex and cleans up global caches to prevent cross-test contamination.
///
/// Note: Both UOp cache and tensor registry use weak references (Tinygrad-aligned),
/// so entries are automatically cleaned up when no longer referenced. We call
/// gc_dead_refs() to proactively clean up dead weak refs in both caches.
///
/// # Returns
///
/// A MutexGuard that must be held for the duration of the test.
/// The guard will be dropped automatically at the end of the test function.
pub fn test_setup() -> parking_lot::MutexGuard<'static, ()> {
    let guard = CACHE_TEST_MUTEX.lock();
    morok_ir::uop::gc_dead_refs(); // Clean up dead UOp weak refs
    crate::tensor_registry::gc_dead_refs(); // Clean up dead tensor weak refs
    crate::tensor_registry::clear_buffer_index();
    morok_runtime::kernel_cache::clear_all();
    guard
}

/// Compare float arrays with tolerance.
///
/// # Arguments
/// * `actual` - The actual computed values
/// * `expected` - The expected reference values (flat array)
/// * `tol` - Absolute tolerance for floating point comparison
///
/// # Panics
/// Panics if:
/// - Array lengths don't match
/// - Any value differs by more than the tolerance
#[track_caller]
pub fn assert_close_f32(actual: &ArrayD<f32>, expected: &[f32], tol: f32) {
    let actual_flat: Vec<f32> = actual.iter().copied().collect();
    assert_eq!(actual_flat.len(), expected.len(), "Length mismatch: {} != {}", actual_flat.len(), expected.len());

    for (i, (a, e)) in actual_flat.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < tol, "Mismatch at index {}: {} != {} (diff: {})", i, a, e, (a - e).abs());
    }
}

/// Realize tensor and extract f32 ndarray.
///
/// This is a convenience wrapper for `.realize()?.to_ndarray::<f32>()?`.
///
/// # Panics
/// Panics if realize or to_ndarray fails.
pub fn realize_f32(t: Tensor) -> ArrayD<f32> {
    t.realize().expect("realize failed").to_ndarray::<f32>().expect("to_ndarray failed")
}

/// Realize tensor and extract i32 ndarray.
///
/// This is a convenience wrapper for `.realize()?.to_ndarray::<i32>()?`.
///
/// # Panics
/// Panics if realize or to_ndarray fails.
pub fn realize_i32(t: Tensor) -> ArrayD<i32> {
    t.realize().expect("realize failed").to_ndarray::<i32>().expect("to_ndarray failed")
}

/// Compare i32 arrays for exact equality.
///
/// # Arguments
/// * `actual` - The actual computed values
/// * `expected` - The expected reference values (flat array)
///
/// # Panics
/// Panics if:
/// - Array lengths don't match
/// - Any value differs
#[track_caller]
pub fn assert_eq_i32(actual: &ArrayD<i32>, expected: &[i32]) {
    let actual_flat: Vec<i32> = actual.iter().copied().collect();
    assert_eq!(actual_flat.len(), expected.len(), "Length mismatch: {} != {}", actual_flat.len(), expected.len());

    for (i, (a, e)) in actual_flat.iter().zip(expected).enumerate() {
        assert_eq!(*a, *e, "Mismatch at index {}: {} != {}", i, a, e);
    }
}
