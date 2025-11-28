//! Configuration constants for hand-coded heuristics.
//!
//! These constants control the behavior of automatic kernel optimization.
//! Based on Tinygrad's configuration system, adapted for Rust.

/// Tensor core usage level.
///
/// - 0: Disabled
/// - 1: Enabled for suitable patterns
/// - 2: Force shape-only matching (more aggressive)
pub const USE_TC: usize = 1;

/// Tensor core optimization level.
///
/// - 0: Strict matching (exact dimensions)
/// - 1: Relaxed matching (allows some flexibility)
/// - 2: Padded matching (adds padding if needed)
pub const TC_OPT: usize = 2;

/// Tensor core selection.
///
/// - -1: Auto-select best tensor core configuration
/// - 0-N: Use specific tensor core configuration by index
pub const TC_SELECT: i32 = -1;

/// Enable matrix-vector multiplication optimization.
pub const MV_ENABLED: bool = true;

/// Matrix-vector block size (rows per workgroup).
pub const MV_BLOCKSIZE: usize = 4;

/// Matrix-vector threads per row (for reduction).
pub const MV_THREADS_PER_ROW: usize = 8;

/// Matrix-vector rows per thread (for output).
pub const MV_ROWS_PER_THREAD: usize = 4;

/// Disable local memory usage globally.
///
/// Useful for debugging or when local memory causes issues.
pub const NOLOCALS: bool = false;

/// Debug verbosity level.
///
/// - 0: No debug output
/// - 1-3: Increasing verbosity
/// - 4+: Very detailed debug output
pub const DEBUG: usize = 0;
