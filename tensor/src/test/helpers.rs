//! Test helpers for realize() validation.

use crate::Tensor;
use ndarray::ArrayD;

/// Setup function to call at the start of each test.
///
/// Buffer UOp IDs are globally unique (monotonic counter via `Op::Unique`),
/// so buffer entries never collide across parallel tests — no registry
/// clearing or mutex serialization needed.
///
/// The kernel name dedup counter is the only non-RAII global state that
/// needs reset between tests.
pub fn test_setup() {
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
