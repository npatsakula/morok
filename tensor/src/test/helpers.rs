//! Test helpers for realize() validation.

use crate::Tensor;
use ndarray::ArrayD;

/// Setup function to call at the start of each test to ensure isolation.
///
/// This clears all thread-local caches (buffer registry, kernel cache)
/// to prevent cross-test contamination.
pub fn test_setup() {
    crate::buffer_registry::clear_all();
    morok_runtime::kernel_cache::clear_all();
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
    assert_eq!(
        actual_flat.len(),
        expected.len(),
        "Length mismatch: {} != {}",
        actual_flat.len(),
        expected.len()
    );

    for (i, (a, e)) in actual_flat.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "Mismatch at index {}: {} != {} (diff: {})",
            i,
            a,
            e,
            (a - e).abs()
        );
    }
}

/// Realize tensor and extract f32 ndarray.
///
/// This is a convenience wrapper for `.realize()?.to_ndarray::<f32>()?`.
///
/// # Panics
/// Panics if realize or to_ndarray fails.
pub fn realize_f32(t: Tensor) -> ArrayD<f32> {
    t.realize()
        .expect("realize failed")
        .to_ndarray::<f32>()
        .expect("to_ndarray failed")
}
