//! NVIDIA GPU LLVM IR text generation (NVPTX).
//!
//! Composed against [`cpu::render_uop`] as the base, exactly like `amd/`:
//! NVPTX-specific ops (`Special`, `Barrier`, LOCAL buffers, `Log2`, `Wmma`)
//! are intercepted here, everything else (ALU, INDEX, LOAD, STORE, CAST,
//! RANGE) falls through to the CPU emitter unchanged. clang lowers the module
//! to PTX text (`--target=nvptx64-nvidia-cuda -march=sm_XY`), which the CUDA
//! driver JITs at load.
//!
//! [`cpu::render_uop`]: crate::llvm::cpu::render_uop

pub mod ops;
pub mod wmma;

pub use ops::{globaltimer, render_uop, shfl_bfly};
