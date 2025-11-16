//! Kernel splitting pipeline orchestration.
//!
//! This module implements the run_kernel_split_pipeline function that orchestrates
//! the complete transformation from high-level BUFFERIZE operations to executable
//! KERNEL operations.
//!
//! # Pipeline Overview
//!
//! The full Tinygrad pipeline has 13 stages (see research notes), but this
//! implementation focuses on the core kernel splitting functionality:
//!
//! 1. **BUFFERIZE → STORE**: Convert memory allocation ops to explicit stores
//! 2. **STORE → KERNEL**: Split computation graph at store boundaries
//!
//! Additional stages (symbolic simplification, cost-based optimization, etc.)
//! will be added as pattern matchers are implemented.

use std::rc::Rc;

use morok_ir::UOp;

use super::{bufferize_to_store::bufferize_to_store, kernel_context::KernelContext, split_kernel::split_store};

/// Run the kernel splitting pipeline.
///
/// This function orchestrates the transformation from high-level operations
/// (BUFFERIZE, etc.) to low-level KERNEL operations ready for code generation.
///
/// # Current Implementation
///
/// The pipeline currently implements 2 core stages:
///
/// 1. **Stage 1: BUFFERIZE → STORE conversion**
///    - Walks the graph bottom-up
///    - Converts each BUFFERIZE to STORE + DEFINE_GLOBAL/DEFINE_LOCAL
///    - Tracks buffer allocations in KernelContext
///
/// 2. **Stage 2: STORE → KERNEL splitting**
///    - Walks the graph bottom-up
///    - Splits computation at STORE/END boundaries
///    - Creates KERNEL operations with proper sources
///
/// # Future Stages
///
/// To match Tinygrad's full pipeline, we'll need to add:
/// - Symbolic simplification (constant folding, algebraic rewrites)
/// - Cost-based bufferize removal
/// - Range optimization and flattening
/// - Device-specific buffer limits
///
/// These will be added via pattern matchers integrated with the graph_rewrite engine.
///
/// # Arguments
///
/// * `root` - The root UOp of the computation graph
///
/// # Returns
///
/// The transformed graph with KERNEL operations
///
/// # Example
///
/// ```ignore
/// // Input graph with BUFFERIZE ops:
/// // BUFFERIZE(compute, ranges, opts)
///
/// let kernels = run_kernel_split_pipeline(graph);
///
/// // Output graph with KERNEL ops:
/// // KERNEL([buffers, vars], SINK(STORE(...)))
/// ```
pub fn run_kernel_split_pipeline(root: Rc<UOp>) -> Rc<UOp> {
    let mut ctx = KernelContext::new();

    // **STAGE 1: BUFFERIZE → STORE Conversion**
    //
    // Walk the graph bottom-up and convert all BUFFERIZE operations to
    // STORE operations with explicit buffer allocation.
    //
    // This populates ctx.buffer_map with BUFFERIZE → DEFINE_GLOBAL/DEFINE_LOCAL mappings.
    let after_bufferize = transform_bottom_up(&root, &mut ctx, bufferize_to_store);

    // **STAGE 2: STORE → KERNEL Splitting**
    //
    // Walk the graph bottom-up and split at STORE/END boundaries to create
    // KERNEL operations.
    //
    // Uses buffer_map from Stage 1 to populate KERNEL sources.
    let after_split = transform_bottom_up(&after_bufferize, &mut ctx, split_store);

    after_split
}

/// Apply a transformation function bottom-up on a graph.
///
/// This helper walks the computation graph in bottom-up order (children before parents)
/// and applies the given transformation function to each node.
///
/// # Algorithm
///
/// 1. Recursively transform all sources (children) first
/// 2. Reconstruct the current node with transformed sources
/// 3. Apply the transformation function to the reconstructed node
/// 4. Return the final result (or original if no transformation)
///
/// # Arguments
///
/// * `uop` - The UOp to transform
/// * `ctx` - Mutable kernel context for tracking state
/// * `transform_fn` - Function to apply to each node
///
/// # Returns
///
/// The transformed UOp
fn transform_bottom_up<F>(uop: &Rc<UOp>, ctx: &mut KernelContext, transform_fn: F) -> Rc<UOp>
where
    F: Fn(&Rc<UOp>, &mut KernelContext) -> Option<Rc<UOp>> + Copy,
{
    // **Step 1: Recursively transform all sources**
    let sources = uop.op().sources();

    if sources.is_empty() {
        // Leaf node - try to transform directly
        return transform_fn(uop, ctx).unwrap_or_else(|| uop.clone());
    }

    // Transform all sources
    let mut transformed_sources = Vec::with_capacity(sources.len());
    let mut any_changed = false;

    for src in sources {
        let transformed = transform_bottom_up(&src, ctx, transform_fn);
        if !Rc::ptr_eq(&transformed, &src) {
            any_changed = true;
        }
        transformed_sources.push(transformed);
    }

    // **Step 2: Reconstruct with transformed sources**
    let reconstructed = if any_changed { uop.with_sources(transformed_sources) } else { uop.clone() };

    // **Step 3: Try to transform the reconstructed node**
    transform_fn(&reconstructed, ctx).unwrap_or(reconstructed)
}
