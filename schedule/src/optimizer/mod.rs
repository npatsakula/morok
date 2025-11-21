//! Kernel optimization layer for morok-schedule.
//!
//! This module implements hardware-aware kernel optimization based on Tinygrad's approach.
//! It provides a `Scheduler` that applies optimization primitives (OptOps) to transform
//! kernel execution for better performance on specific backends.
//!
//! # Architecture
//!
//! The optimization process follows this flow:
//!
//! 1. **Initialization**: Create `Scheduler` from UOp AST + `Renderer` (backend capabilities)
//! 2. **Initial Transform**: Convert eligible LOOP axes to GLOBAL (parallelization)
//! 3. **Optimization**: Apply `Opt` operations via `apply_opt()`
//!    - UPCAST: Vectorization (SIMD)
//!    - LOCAL: GPU workgroup dimensions (shared memory)
//!    - UNROLL: Loop unrolling for reductions
//!    - GROUP: Two-stage reductions with synchronization
//!    - TC: Tensor core acceleration
//!    - PADTO, SWAP, THREAD, NOLOCALS: Layout and configuration
//! 4. **Finalization**: Extract optimized AST with `get_optimized_ast()`
//!
//! # Optimization Strategies
//!
//! - **Hand-coded heuristics** (`heuristics` module): Fast, reasonable performance
//! - **Beam search** (`beam` module, optional): Slow, ML-quality performance
//!
//! # Example
//!
//! ```ignore
//! use morok_schedule::optimizer::{Scheduler, Renderer, Opt, OptOps};
//!
//! // Create scheduler with CUDA backend
//! let renderer = Renderer::cuda();
//! let mut scheduler = Scheduler::new(kernel_ast, renderer);
//!
//! // Apply optimizations
//! scheduler.convert_loop_to_global();
//! scheduler.apply_opt(Opt::upcast(0, 4), true)?; // Vectorize axis 0 by 4
//! scheduler.apply_opt(Opt::local(1, 16), true)?; // Local memory for axis 1
//!
//! // Get optimized kernel
//! let optimized_ast = scheduler.get_optimized_ast(None);
//! ```

pub mod error;
pub mod kernel_info;
pub mod opts;
pub mod renderer;
pub mod scheduler;
pub mod types;

// Re-exports
pub use kernel_info::KernelInfo;
pub use opts::apply_opt;
pub use renderer::{Renderer, TcOpt, TensorCore};
pub use scheduler::Scheduler;
pub use types::{AxisType, Opt, OptArg, OptOps};
