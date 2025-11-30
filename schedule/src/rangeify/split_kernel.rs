//! Kernel splitting at STORE boundaries.
//!
//! This module implements the split_store function that splits the computation graph
//! into individual kernels at STORE operation boundaries. It follows Tinygrad's
//! kernel splitting algorithm (schedule/rangeify.py:471-497).
//!
//! The algorithm:
//! 1. Filters operations to only split at kernel boundaries (OUTER ranges only)
//! 2. Applies transformation pipeline to convert high-level ops to kernel IR
//! 3. Creates SINK operation wrapping the computation
//! 4. Creates KERNEL operation with proper buffer and variable arguments

use std::rc::Rc;

use morok_ir::{AxisType, Op, UOp};
use smallvec::SmallVec;

use super::codegen_patterns::rangeify_codegen_patterns;
use super::cycle_detection::find_bufs;
use super::kernel_context::KernelContext;
use super::split_patterns::to_define_global_patterns;
use crate::rewrite::graph_rewrite_bottom_up;

/// Find first COPY or BUFFER_VIEW operation in the computation graph.
///
/// When a kernel's computation contains a COPY or BUFFER_VIEW operation,
/// the kernel's AST should be that operation directly rather than wrapped
/// in a SINK. This allows:
/// - **COPY:** Scheduler to detect cross-device transfers and assign elevated priority
/// - **BUFFER_VIEW:** Runtime to extract view parameters (size, offset) for sub-buffer creation
///
/// If multiple COPY/BUFFER_VIEW operations exist, returns the first one found
/// during topological traversal (typically the one deepest in the graph).
///
/// # Arguments
///
/// * `uop` - The computation graph to search
///
/// # Returns
///
/// * `Some(op)` - The first COPY or BUFFER_VIEW operation found
/// * `None` - If no such operation exists in the graph
///
/// # Example
///
/// ```ignore
/// let copy = UOp::new(Op::Copy { src: buffer, device }, DType::Float32);
/// let store = UOp::store(output, vec![idx], copy);
///
/// // Returns Some(copy), not None
/// assert!(find_copy_or_buffer_view(&store).is_some());
/// ```
///
/// Based on Tinygrad's "hack for COPY" (rangeify.py:494-499).
fn find_copy_or_buffer_view(uop: &Rc<UOp>) -> Option<Rc<UOp>> {
    // Traverse the computation graph in topological order
    for node in uop.toposort() {
        match node.op() {
            Op::Copy { .. } | Op::BufferView { .. } => {
                return Some(node);
            }
            _ => continue,
        }
    }
    None
}

/// Split STORE and END operations into individual kernels.
///
/// This function determines whether a STORE or END operation should be split into
/// a separate kernel. The decision is based on analyzing the operation's RANGE
/// dependencies - kernels are only created when all ranges are OUTER (meaning we're
/// at the outermost scheduling level).
///
/// # Algorithm
///
/// 1. **Filtering:**
///    - Only split when all RANGEs are AxisType::Outer
///    - Skip END operations that close OUTER ranges (they're control flow markers)
///
/// 2. **Transformation Pipeline:**
///    - Apply to_define_global patterns (BUFFER → DEFINE_GLOBAL, etc.)
///    - Apply rangeify_codegen patterns (remove_noop, get_contiguous, fix_after_broadcast)
///    - The graph rewriting engine propagates substitutions automatically
///
/// 3. **Validation:**
///    - Check for buffer access cycles (LOAD vs STORE conflicts)
///
/// 4. **Kernel AST Creation:**
///    - Wrap computation in SINK operation (for normal kernels)
///    - Use COPY/BUFFER_VIEW directly as AST (for special kernels)
///
/// 5. **KERNEL Creation:**
///    - Create KERNEL with sources from KernelContext
///    - Sources = all accessed buffers + all BIND variables
///
/// # Arguments
///
/// * `x` - The STORE or END operation to potentially split
/// * `ctx` - Mutable kernel context for tracking transformations
///
/// # Returns
///
/// * `Some(kernel)` - A KERNEL operation if split was performed
/// * `None` - If the operation is not ready to split
///
/// # Example
///
/// ```ignore
/// // Input: STORE with all OUTER ranges
/// let store = /* ... */;
/// let mut ctx = KernelContext::new();
///
/// if let Some(kernel) = split_store(&store, &mut ctx) {
///     // kernel contains the split computation
/// }
/// ```
///
/// Based on Tinygrad's split_store (schedule/rangeify.py:471-497).
pub fn split_store(uop: &Rc<UOp>, ctx: &mut KernelContext) -> Option<Rc<UOp>> {
    // **FILTERING CRITERION 0: Skip if has non-OUTER ranges in scope**
    // Matches Tinygrad line 472:
    // if len([r for r in x.ranges if r.arg[-1] != AxisType.OUTER]): return None
    //
    // Only split at operations where ALL in-scope ranges are OUTER.
    // This ensures we only create kernels at proper kernel boundaries.
    if uop.has_non_outer_ranges() {
        return None;
    }

    // **FILTERING CRITERION 1: Verify operation type**
    //
    // We support two cases:
    // 1. END wrapping STORE (normal pipeline output from bufferize_to_store)
    //    Structure: END(computation=STORE(...), ranges=[RANGE(...)])
    //    We preserve the entire END structure (don't extract STORE!)
    //
    // 2. Bare STORE (for unit tests or edge cases)
    //    Process the STORE directly.
    //
    // IMPORTANT: Unlike our previous implementation, we DO NOT extract STORE from END.
    // Tinygrad preserves END structure inside the KERNEL's AST.
    // Final structure: KERNEL(..., ast=SINK(END(STORE, RANGE)))
    //
    // Based on Tinygrad rangeify.py:480-507
    let computation = match uop.op() {
        Op::End { computation, ranges } => {
            // Verify the END wraps a STORE operation
            match computation.op() {
                Op::Store { .. } | Op::StoreGated { .. } => {
                    // Skip END operations that close OUTER ranges (control flow markers)
                    // Tinygrad line 485: if x.op is Ops.END and x.src[1].op is Ops.RANGE and x.src[1].arg[-1] == AxisType.OUTER
                    for r in ranges.iter() {
                        if let Op::Range { axis_type, .. } = r.op()
                            && *axis_type == AxisType::Outer
                        {
                            return None;
                        }
                    }
                    // Keep the whole END, don't extract STORE
                    uop.clone()
                }
                // END wrapping non-STORE - skip (control flow marker)
                _ => return None,
            }
        }
        Op::Store { .. } | Op::StoreGated { .. } => {
            // Bare STORE - process directly
            uop.clone()
        }
        _ => return None,
    };

    // **STEP 1: Apply transformation pipeline**
    // Tinygrad line 489: ret = graph_rewrite(x, to_define_global+..., bottom_up=True)
    //
    // The graph_rewrite will:
    // - Transform BUFFER → DEFINE_GLOBAL in STORE children
    // - Preserve END structure (no pattern removes END)
    // - Result: END(STORE(DEFINE_GLOBAL, ...), RANGE(...))
    //
    // Apply to_define_global patterns via graph_rewrite (bottom-up)
    // Bottom-up traversal ensures deep nodes (BUFFER inside INDEX inside ADD) are processed.
    let transformed = {
        let matcher = to_define_global_patterns();
        graph_rewrite_bottom_up(&matcher, computation, ctx)
    };

    // **STEP 1.5a: Apply movement op patterns**
    // This pushes movement ops through INDEX operations:
    // INDEX(RESHAPE(buffer), ranges) → INDEX(buffer, transformed_ranges)
    //
    // This step is critical for input buffers with movement ops (RESHAPE(BUFFER), etc.)
    // After this step, movement ops are eliminated and indices are properly transformed.
    //
    // Based on Tinygrad's pm_mops (schedule/rangeify.py:18-25).
    // Bottom-up ensures movement ops deep in the graph are processed.
    let transformed = {
        let movement_matcher = super::movement_patterns::movement_op_patterns();
        graph_rewrite_bottom_up(&movement_matcher, transformed, &mut ())
    };

    // **STEP 1.5b: Apply codegen preparation patterns**
    // Apply rangeify_codegen patterns (remove_noop, get_contiguous, fix_after_broadcast)
    // These prepare the IR for final code generation:
    // - NOOP → zero constant
    // - CONTIGUOUS markers removed
    // - AFTER wrapping EXPAND fixed
    // - INDEX on DEFINE_GLOBAL gets LOAD wrapper
    // Bottom-up ensures INDEX operations are processed after their children.
    let transformed = {
        let codegen_matcher = rangeify_codegen_patterns();
        graph_rewrite_bottom_up(&codegen_matcher, transformed, &mut ())
    };

    // **STEP 2: Validate no buffer access cycles**
    // Check that no buffer is accessed with conflicting operations (LOAD vs STORE).
    // This prevents creating invalid kernels. Panics if cycles detected.
    // Based on Tinygrad's find_bufs (schedule/rangeify.py:413-417)
    #[allow(clippy::mutable_key_type)]
    let _buf_accesses = find_bufs(&transformed);

    // **STEP 3: Create kernel AST (SINK or special operations)**
    //
    // Normal case: Wrap computation in SINK
    // Special cases: Use COPY or BUFFER_VIEW directly as kernel AST
    //
    // Rationale:
    // - COPY kernels need direct AST for scheduler priority assignment
    // - BUFFER_VIEW kernels need direct AST for view parameter extraction
    //
    // If both COPY and BUFFER_VIEW exist, the first one found (deepest in toposort)
    // is used as the AST. This is deterministic and matches Tinygrad's behavior.
    //
    // Based on Tinygrad's "hack for COPY" (rangeify.py:494-499).
    let ast = if let Some(special_op) = find_copy_or_buffer_view(&transformed) {
        // Found COPY or BUFFER_VIEW - use it directly as kernel AST
        special_op
    } else {
        // Normal computational kernel - wrap in SINK
        UOp::sink(vec![transformed])
    };

    // **STEP 4: Build kernel sources from transformed AST**
    // Sources = all DEFINE_GLOBAL/DEFINE_LOCAL nodes + all BIND variables
    //
    // Walk the transformed AST to collect all buffer definitions.
    // This is necessary because input buffers (LOADs) also become DEFINE_GLOBAL
    // via the debuf pattern, and we need to include them in kernel sources.
    //
    // Based on Tinygrad's split_store (schedule/rangeify.py:510-512).
    let mut sources: SmallVec<[Rc<UOp>; 4]> = SmallVec::new();

    // Collect all DEFINE_GLOBAL and DEFINE_LOCAL from the transformed AST
    // These are the kernel's buffer arguments (both inputs and outputs)
    for node in ast.toposort() {
        match node.op() {
            Op::DefineGlobal(_) | Op::DefineLocal(_) => {
                sources.push(node);
            }
            _ => {}
        }
    }

    // Add all variables from context (BIND tracking)
    for var_key in &ctx.vars {
        sources.push(var_key.0.clone());
    }

    // **STEP 5: Create KERNEL operation**
    // Use the AST we created in STEP 3 (either SINK or special operation)
    let kernel = UOp::kernel(sources, ast);

    Some(kernel)
}
