//! `svod-tk2` — a from-scratch Rust tile-IR DSL for AMD (gfx942) matmul kernels: a typed builder
//! emits an interned, hash-consed tile-IR carrying both the algorithm and the schedule as data;
//! composable combinators/passes rewrite it; a verified lowering emits device-UOp → linearizer →
//! codegen → LLVM. The target is HipKittens' compiler-visible matmul perf. Design: `tk2/DESIGN.md`.
//!
//! The modules:
//! - [`ir`] — the interned, hash-consed tile-IR ADT (§1-§2): ONE DAG carrying algorithm + schedule
//!   as data, with ordering as first-class edges. [`build`] is the typed builder over it.
//! - [`movement`] — the `LdsView`/`LdsStage`/`SharedTile` handles carrying LDS addressing as data.
//! - [`schedule`] — the typestate cluster-pipeline DSL (`MemScope`/`ComputeScope` + the 8-wave
//!   ping-pong [`schedule::pipeline`]); [`pipeline`] is the driver form backing the asm clustered kernel.
//! - [`pass`]/[`passes`] — the strategy-combinator runner + the `.apply`-able [`SwizzlePass`]/
//!   [`VectorizePass`] refinements.
//! - [`lower`] — the verified lowering to a device-UOp SINK → svod's `do_linearize → type_verify →
//!   render` path (§D); [`launch`] dispatches on device.
//!
//! [`kernels`] keeps two matmul kernels: the compiler-visible `pipe2` and the asm `clustered` HK copies.

pub mod build;
pub mod error;
pub mod graph;
pub mod ir;
pub mod kernels;
pub mod launch;
pub mod lower;
pub(crate) mod movement;
pub mod pass;
pub mod passes;
pub(crate) mod pipeline;
pub mod schedule;

pub use build::{Builder, Elem, F32};
pub use error::{Error, Result};
pub use graph::graph_kernel;
pub use ir::{Node, TileId, TileIr};
pub use kernels::{Program, matmul_lds_kblock_mw_clustered, matmul_lds_kblock_mw_pipe2};
pub use pass::{Band, Fold, Pass, Pipeline, Strategy};
pub use passes::{SwizzlePass, VectorizePass};
pub use schedule::{
    Carry, Committed, ComputeScope, Gathered, InFlight, MemScope, PipelineCx, SteadyOut, compute_cluster, mem_cluster,
    pipeline,
};

#[cfg(test)]
mod test;
