//! `svod-tk2` — a from-scratch Rust tile-IR DSL (the replacement for the svod-tk
//! layer). **Step 1: the minimal end-to-end skeleton** — the architecture spine
//! "naive tile-IR → verified lowering → correct (slow) device-UOp → runs on
//! device", with NO optimizations yet. The settled design lives in `tk2/DESIGN.md`.
//!
//! The four pieces (each a module):
//! - [`ir`] — the interned, hash-consed tile-IR ADT (§1-§2): ONE DAG carrying
//!   algorithm + (eventually) schedule as data, with ordering as first-class edges.
//! - [`build`] — the typed builder emitting `Copy` [`ir::TileId`] handles, gently
//!   typed (sealed dtype trait) per §OPEN-2.
//! - [`lower`] — the verified lowering to a device-UOp SINK, then through svod's
//!   existing `program_from_sink → do_linearize → type_verify → render` path (§D).
//! - [`pass`] — the strategy-combinator pass runner with bands + contracts + a
//!   nanopass identity-default folder (§2.6). No real passes yet — scaffolding.
//!
//! [`kernels`] authors the two proof kernels; [`launch`] dispatches on device.

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

pub use build::{Builder, Elem, F32};
pub use error::{Error, Result};
pub use graph::graph_kernel;
pub use ir::{Node, TileId, TileIr};
pub use kernels::{
    Program, elementwise_add, lds_carry_loop, lds_roundtrip, matmul, matmul_lds, matmul_lds_kblock,
    matmul_lds_kblock_ks, matmul_lds_kblock_mw, matmul_lds_kblock_mw_clustered, matmul_lds_kblock_mw_pipe,
    matmul_lds_kblock_mw_resident, matmul_lds_kblock_sw, matmul_lds_kblock_vec, matmul_lds_tiled, sum_reduce,
};
pub use pass::{Band, Fold, Pass, Pipeline, Strategy};
pub use passes::{ConstFoldPass, SwizzlePass, UnrollPass, VectorizePass, optimize_addressing};

#[cfg(test)]
mod test;
