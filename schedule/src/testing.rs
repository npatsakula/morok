//! Shared test tracing infrastructure.
//!
//! Provides a unified JSON tracing setup for all crates. Requires the `testing` feature.
//!
//! ```ignore
//! morok_schedule::testing::setup_test_tracing();
//! ```
//!
//! Controlled by `RUST_LOG` (e.g., `RUST_LOG=morok_onnx::importer=trace`).
//! Compatible with `scripts/extract-ir.sh` (JSON structured output).

/// Initialize a JSON tracing subscriber for tests.
///
/// Safe to call multiple times — only the first call initializes.
/// Outputs JSON lines to the test writer (captured by `cargo test`,
/// visible with `--nocapture`). Uses `RUST_LOG` for filtering.
pub fn setup_test_tracing() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .json()
            .with_current_span(false)
            .with_span_list(false)
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_test_writer()
            .init();
    });
}
