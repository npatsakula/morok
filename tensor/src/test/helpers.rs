//! Test helpers for realize() validation.

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

/// Compare float slices with tolerance.
#[track_caller]
pub fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "Length mismatch: {} != {}", actual.len(), expected.len());

    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < tol, "Mismatch at index {}: {} != {} (diff: {})", i, a, e, (a - e).abs());
    }
}
