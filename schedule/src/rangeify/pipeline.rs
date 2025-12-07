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
//! 3. **Dependency Resolution**: Extract inter-kernel dependencies from AFTER nodes
//!
//! Additional stages (symbolic simplification, cost-based optimization, etc.)
//! will be added as pattern matchers are implemented.

use std::collections::HashMap;
use std::sync::Arc;

use morok_ir::{Op, UOp};

use super::{bufferize_to_store::bufferize_to_store, kernel_context::KernelContext, split_kernel::split_store};

/// Run the kernel splitting pipeline.
///
/// This function orchestrates the transformation from high-level operations
/// (BUFFERIZE, etc.) to low-level KERNEL operations ready for code generation.
///
/// # Current Implementation
///
/// The pipeline currently implements 3 core stages:
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
/// 3. **Stage 3: Dependency resolution**
///    - Extracts inter-kernel dependencies from AFTER nodes
///    - Populates `ctx.kernel_deps` for scheduling
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
/// The transformed graph with KERNEL operations and populated KernelContext
///
/// # Example
///
/// ```ignore
/// // Input graph with BUFFERIZE ops:
/// // BUFFERIZE(compute, ranges, opts)
///
/// let (kernels, ctx) = run_kernel_split_pipeline(graph);
/// // ctx.kernel_deps contains inter-kernel dependencies
///
/// // Output graph with KERNEL ops:
/// // KERNEL([buffers, vars], SINK(STORE(...)))
/// ```
pub fn run_kernel_split_pipeline(root: Arc<UOp>) -> (Arc<UOp>, KernelContext) {
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

    // **STAGE 3: Dependency Resolution**
    //
    // Extract inter-kernel dependencies from AFTER nodes.
    // This populates ctx.kernel_deps for use by the scheduler.
    resolve_kernel_dependencies(&after_split, &mut ctx);

    // Return both the transformed graph and the context
    (after_split, ctx)
}

/// Resolve dependencies between kernels by analyzing AFTER nodes.
///
/// This function walks the graph and identifies producer→consumer relationships
/// between kernels via shared buffers. When kernel A writes to buffer B
/// (wrapped in an AFTER node), and kernel C reads from buffer B,
/// kernel C depends on kernel A.
///
/// # How it works
///
/// 1. Build a map: buffer_id → (AFTER node, producing KERNEL)
/// 2. For each KERNEL, find buffers it reads (from its sources)
/// 3. If a read buffer has a producer, record the dependency in ctx.kernel_deps
///
/// # Arguments
///
/// * `root` - The root UOp after kernel splitting
/// * `ctx` - KernelContext to populate with dependencies
fn resolve_kernel_dependencies(root: &Arc<UOp>, ctx: &mut KernelContext) {
    // Build map: buffer_id → producing KERNEL
    // An AFTER node wraps a buffer and depends on the kernel that produces it
    let mut buffer_producers: HashMap<u64, Arc<UOp>> = HashMap::new();

    for node in root.toposort() {
        if let Op::After { passthrough, deps } = node.op() {
            // The passthrough is the buffer (DEFINE_GLOBAL)
            if let Op::DefineGlobal(_) = passthrough.op() {
                // Find the KERNEL in deps
                for dep in deps {
                    if let Op::Kernel { .. } = dep.op() {
                        buffer_producers.insert(passthrough.id, dep.clone());
                        break;
                    }
                }
            }
        }
    }

    // For each KERNEL, find what buffers it reads and record dependencies
    for node in root.toposort() {
        if let Op::Kernel { sources, ast } = node.op() {
            // Find buffers this kernel reads from (via its sources)
            for src in sources {
                // If this source is an AFTER, extract the underlying buffer
                let buffer_id = match src.op() {
                    Op::After { passthrough, .. } => {
                        if let Op::DefineGlobal(_) = passthrough.op() {
                            Some(passthrough.id)
                        } else {
                            None
                        }
                    }
                    Op::DefineGlobal(_) => Some(src.id),
                    _ => None,
                };

                // If this buffer has a producer kernel, record the dependency
                if let Some(buf_id) = buffer_id
                    && let Some(producer) = buffer_producers.get(&buf_id)
                    && !Arc::ptr_eq(producer, &node)
                {
                    ctx.add_dependency(buf_id, producer.clone(), node.clone());
                }
            }

            // Also check the AST for LOAD operations that read from buffers
            for ast_node in ast.toposort() {
                if let Op::Load { buffer, .. } = ast_node.op() {
                    let buffer_id = match buffer.op() {
                        Op::DefineGlobal(_) => Some(buffer.id),
                        _ => None,
                    };

                    if let Some(buf_id) = buffer_id
                        && let Some(producer) = buffer_producers.get(&buf_id)
                        && !Arc::ptr_eq(producer, &node)
                    {
                        // Avoid duplicate dependencies
                        let already_tracked = ctx.kernel_deps.iter().any(|d| {
                            d.buffer_id == buf_id
                                && Arc::ptr_eq(&d.producer, producer)
                                && Arc::ptr_eq(&d.consumer, &node)
                        });
                        if !already_tracked {
                            ctx.add_dependency(buf_id, producer.clone(), node.clone());
                        }
                    }
                }
            }
        }
    }
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
fn transform_bottom_up<F>(uop: &Arc<UOp>, ctx: &mut KernelContext, transform_fn: F) -> Arc<UOp>
where
    F: Fn(&Arc<UOp>, &mut KernelContext) -> Option<Arc<UOp>> + Copy,
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
        if !Arc::ptr_eq(&transformed, &src) {
            any_changed = true;
        }
        transformed_sources.push(transformed);
    }

    // **Step 2: Reconstruct with transformed sources**
    let reconstructed = if any_changed { uop.with_sources(transformed_sources) } else { uop.clone() };

    // **Step 3: Try to transform the reconstructed node**
    transform_fn(&reconstructed, ctx).unwrap_or(reconstructed)
}
