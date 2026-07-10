//! **`hk`** — a name-faithful, LLVM-IR-verified 1:1 port of HipKittens' `micro_tk` BF16→FP32 GEMM
//! leaf helpers into tk2's tile-IR DSL (gfx942 / MI300X / CDNA3).
//!
//! Source kernel: `submodules/HipKittens/kernels/gemm/bf16fp32/cdna3/8192_256_256_64_16/256_256_64_16.cpp`.
//! Ground-truth IR: `hk-micro_tk.ll` (kernel `@_Z8micro_tk13micro_globals`, ROCm 7.2.4). Every helper
//! here keeps HipKittens' **exact identifiers** (`mma_ABt`, `st_bf`, `load`, `make_srsrc`, …) so a
//! reader can point at the Rust name and the `.cpp` name and see the same thing — a deliberate,
//! justified style exception (`#[allow(non_snake_case)]` / `#[allow(non_camel_case_types)]`) for the
//! 1:1 port. Each helper is a THIN wrapper over an existing tk2 builder / movement primitive, and each
//! is verified against the oracle IR by a headless unit test in `src/test/hk_ir.rs`.
//!
//! The leaf helpers (`memory`/`mma`/`sync`/`swizzle`/`types`) compose into [`gemm::micro_tk`] — the
//! assembled kernel whose rolled K-loop matches HK's oracle CFG (verified headlessly + on gfx942).
//!
//! ## Tiling constants (from the source, hard-coded for the 8192³ reference port)
//! `BLOCK_SIZE = 256`, `K_STEP = 64`, `REG_BLOCK = BLOCK_SIZE/4 = 64`, `DOT_SLICE = 16`,
//! `NUM_WARPS = 8`, `NUM_THREADS = 512`, `BUFFER_SIZE = (256·64)/512 = 32` bf16/thread.

pub mod gemm;
pub mod memory;
pub mod mma;
pub mod swizzle;
pub mod sync;
pub mod types;

pub use gemm::{micro_globals, micro_tk};

// ── HK's kernel-local tiling constants (`256_256_64_16.cpp`) ─────────────────
/// Output tile edge (M and N per block).
pub const BLOCK_SIZE: usize = 256;
/// K chunk staged in LDS per mainloop iteration.
pub const K_STEP: usize = 64;
/// `BLOCK_SIZE / 4` — the per-warp accumulator edge.
pub const REG_BLOCK: usize = BLOCK_SIZE / 4;
/// K-slice per MFMA (one 16×16×16 MFMA K).
pub const DOT_SLICE: usize = 16;
/// `2 (warp_row) × 4 (warp_col)`.
pub const NUM_WARPS: usize = 8;
/// `WARP_THREADS(64) · NUM_WARPS`.
pub const NUM_THREADS: usize = 512;
/// `(BLOCK_SIZE · K_STEP) / NUM_THREADS` — bf16 elements per thread per staged tile (= 4 × float4).
pub const BUFFER_SIZE: usize = (BLOCK_SIZE * K_STEP) / NUM_THREADS;
