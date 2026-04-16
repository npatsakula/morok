//! Per-kernel execution profiling.
//!
//! Provides structured timing data for kernel execution via
//! [`ExecutionPlan::execute_profiled()`](crate::ExecutionPlan::execute_profiled).

use std::sync::Arc;
use std::time::Duration;

use morok_dtype::DeviceSpec;

use crate::kernel_cache::CachedKernel;

/// Per-kernel timing from a profiled execution.
///
/// Holds an `Arc<CachedKernel>` for zero-copy access to kernel metadata
/// (entry point, generated code, global/local size, variable names).
///
/// # Example
///
/// ```ignore
/// let plan = tensor.prepare()?;
/// let profiles = plan.execute_profiled()?;
///
/// for (i, p) in profiles.iter().enumerate() {
///     println!("{:4} {:>8.3}ms  {}  ({} bufs, {:?})",
///         i, p.elapsed.as_secs_f64() * 1000.0,
///         p.kernel.entry_point, p.num_buffers, p.device);
/// }
/// ```
pub struct KernelProfile {
    /// Compiled kernel (entry_point, code, global_size, local_size, var_names).
    pub kernel: Arc<CachedKernel>,
    /// Device this kernel executed on.
    pub device: DeviceSpec,
    /// Number of buffer arguments.
    pub num_buffers: usize,
    /// Wall-clock execution time.
    pub elapsed: Duration,
}
