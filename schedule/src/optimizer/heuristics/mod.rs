//! Hand-coded heuristics for automatic kernel optimization.
//!
//! This module implements Tinygrad-style optimization heuristics that provide
//! reasonable performance without expensive auto-tuning. The heuristics analyze
//! kernel patterns and apply optimization primitives (UPCAST, LOCAL, UNROLL, etc.)
//! in a systematic order.
//!
//! # Optimization Strategy
//!
//! The heuristics are applied in this order:
//! 1. **Tensor Cores**: Hardware-accelerated matmul (if pattern matches)
//! 2. **Image Float4**: GPU image vectorization (if applicable)
//! 3. **Matrix-Vector**: Specialized matvec optimization
//! 4. **Grouped Reduction**: Two-stage reductions with synchronization
//! 5. **Masked Upcasts**: Upcast small masked dimensions (≤7)
//! 6. **Heuristic Upcasts**: Stride-based ranking of upcast candidates
//! 7. **Unroll**: Loop unrolling for reduction axes
//! 8. **Default Upcast**: Fallback vectorization if nothing else worked
//! 9. **Local Dims**: GPU workgroup/shared memory configuration
//! 10. **Threading**: CPU multi-threading configuration
//!
//! # Usage
//!
//! ```ignore
//! use morok_schedule::optimizer::Scheduler;
//! use morok_schedule::optimizer::heuristics::hand_coded_optimizations;
//!
//! let mut scheduler = Scheduler::new(ast, renderer);
//! hand_coded_optimizations(&mut scheduler);
//! let optimized_ast = scheduler.get_optimized_ast(None);
//! ```

pub mod complex;
pub mod config;
pub mod helpers;
pub mod intermediate;
pub mod simple;

// Re-exports
pub use complex::*;
pub use config::*;
pub use helpers::*;
pub use intermediate::*;
pub use simple::*;

use crate::optimizer::Scheduler;

/// Apply hand-coded optimization heuristics to a kernel.
///
/// This is the main entry point for automatic optimization. It applies
/// a series of heuristics in a specific order to optimize the kernel
/// for the target hardware.
///
/// The heuristics are applied in this order:
/// 1. **Tensor Cores**: Hardware-accelerated matmul (if pattern matches)
/// 2. **Image Float4**: GPU image vectorization (if applicable)
/// 3. **Grouped Reduction**: Two-stage reductions (if reduction is large)
/// 4. **Masked Upcasts**: Upcast small masked dimensions
/// 5. **Heuristic Upcasts**: Stride-based upcast selection
/// 6. **Unroll**: Loop unrolling for small reductions
/// 7. **Default Upcast**: Fallback vectorization
/// 8. **Local Dims**: GPU workgroup configuration
/// 9. **Threading**: CPU parallelization
///
/// The function mutates the scheduler by applying various OptOps.
pub fn hand_coded_optimizations(scheduler: &mut Scheduler) {
    // 1. Try tensor cores first (highest performance for matmul)
    if try_tensor_cores(scheduler) {
        // If tensor cores applied, skip most other optimizations
        // as TC handles vectorization and tiling internally
        apply_local_dims(scheduler);
        return;
    }

    // 2. Try image-specific optimizations
    if apply_image_upcasts(scheduler) {
        // Image upcasts may be sufficient, continue with other opts
    }

    // 3. Try grouped reduction for large reductions
    try_grouped_reduction(scheduler);

    // 4. Apply masked upcasts (small masked dimensions)
    apply_masked_upcasts(scheduler);

    // 5. Apply sophisticated upcast heuristics
    apply_heuristic_upcasts(scheduler);

    // 6. Unroll small reduction loops
    apply_unroll(scheduler);

    // 7. Fallback upcast if nothing else applied
    if !scheduler.upcasted() {
        apply_default_upcast(scheduler);
    }

    // 8. Configure local dimensions for GPU
    apply_local_dims(scheduler);

    // 9. Apply threading for CPU
    apply_threading(scheduler);
}
