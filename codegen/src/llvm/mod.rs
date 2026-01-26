//! LLVM IR code generation.
//!
//! This module generates LLVM IR code from optimized UOp graphs.
//!
//! # Module Structure
//!
//! - `common/`: Shared utilities (types, ctx) for CPU and GPU
//! - `cpu/`: CPU-specific rendering
//! - `gpu/`: Future GPU-specific rendering (HIP, CUDA, Metal)
//! - `text/`: Main entry point that orchestrates rendering

pub mod common;
pub mod cpu;
pub mod gpu;
pub mod text;

pub use cpu::render_uop as cpu_render_uop;
pub use text::LlvmTextRenderer;
