//! Graph rewrite engine with fixed-point iteration.
//!
//! This module implements the core graph rewriting algorithm that applies
//! pattern-based transformations to UOp graphs.

pub mod engine;

pub use engine::{
    GraphRewriteOutput, NoMatcher, graph_rewrite, graph_rewrite_bottom_up, graph_rewrite_bottom_up_with_map,
    graph_rewrite_with_bpm, graph_rewrite_with_map,
};

// Backward compatibility - deprecated
#[allow(deprecated)]
pub use engine::{graph_rewrite_top_down, graph_rewrite_top_down_with_map};
