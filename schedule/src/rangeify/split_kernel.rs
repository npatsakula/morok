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

use std::cell::RefCell;
use std::rc::Rc;

use morok_ir::{AxisType, Op, UOp};
use smallvec::SmallVec;

use super::codegen_patterns::rangeify_codegen_patterns;
use super::cycle_detection::find_bufs;
use super::kernel_context::KernelContext;
use super::split_patterns::to_define_global_patterns;
use crate::rewrite::graph_rewrite;

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
/// 4. **SINK Creation:**
///    - Wrap the computation in a SINK operation
///    - TODO: Add support for COPY/BUFFER_VIEW special cases
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
    // Apply to_define_global patterns via graph_rewrite
    let transformed = {
        // Wrap context in RefCell for pattern access
        let ctx_ref = Rc::new(RefCell::new(ctx.clone()));

        // Build pattern matcher with context
        let matcher = to_define_global_patterns(ctx_ref.clone());

        // Apply patterns to computation graph
        let result = graph_rewrite(&matcher, computation);

        // Extract updated context (patterns may have modified it)
        *ctx = ctx_ref.borrow().clone();

        result
    };

    // **STEP 1.5: Apply codegen preparation patterns**
    // Apply rangeify_codegen patterns (remove_noop, get_contiguous, fix_after_broadcast)
    // These prepare the IR for final code generation:
    // - NOOP → zero constant
    // - CONTIGUOUS markers removed
    // - AFTER wrapping EXPAND fixed
    let transformed = {
        let ctx_ref = Rc::new(RefCell::new(ctx.clone()));
        let codegen_matcher = rangeify_codegen_patterns(ctx_ref.clone());
        let result = graph_rewrite(&codegen_matcher, transformed);
        *ctx = ctx_ref.borrow().clone();
        result
    };

    // **STEP 2: Validate no buffer access cycles**
    // Check that no buffer is accessed with conflicting operations (LOAD vs STORE).
    // This prevents creating invalid kernels. Panics if cycles detected.
    // Based on Tinygrad's find_bufs (schedule/rangeify.py:413-417)
    let _buf_accesses = find_bufs(&transformed);

    // **STEP 3: Create SINK operation**
    // Wrap the transformed computation (either STORE or END(STORE, RANGE)) in a SINK
    // Tinygrad line 501: ret = ret.sink(...)
    let sink = UOp::sink(vec![transformed]);

    // **STEP 4: Build kernel sources from context**
    // Sources = all accessed buffers + all BIND variables
    //
    // Note: buffer_map is populated by bufferize_to_store() when BUFFERIZE ops
    // are converted to STORE ops. Each BUFFERIZE gets a DEFINE_GLOBAL/DEFINE_LOCAL
    // tracked in the context.
    //
    // vars will be populated when we implement BIND handling patterns.
    let mut sources: SmallVec<[Rc<UOp>; 4]> = SmallVec::new();

    // Add all buffers from context
    sources.extend(ctx.buffer_map.values().cloned());

    // Add all variables from context
    for var_key in &ctx.vars {
        sources.push(var_key.0.clone());
    }

    // **STEP 5: Create KERNEL operation**
    let kernel = UOp::kernel(sources, sink);

    Some(kernel)
}
