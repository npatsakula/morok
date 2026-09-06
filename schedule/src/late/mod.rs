//! Late code generation rewrites.

pub mod coalesce;
pub mod dtype;
pub mod gater;

pub use coalesce::{
    AddImageContext, indexing_simplify, memory_coalescing, pm_lower_grouped_shrink, pm_simplify_add_image,
};
pub use dtype::{DemoteFloat, demote_unsupported_floats};
pub use gater::pm_move_gates_from_index;
