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

use morok_dtype::DType;
use morok_ir::{AxisType, Op, UOp};
use smallvec::SmallVec;

use super::kernel_context::KernelContext;
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
/// 2. **Transformation:**
///    - Apply to_define_global patterns (BUFFER → DEFINE_GLOBAL, etc.)
///    - The graph rewriting engine propagates substitutions automatically
///
/// 3. **SINK Creation:**
///    - Wrap the computation in a SINK operation
///    - TODO: Add support for COPY/BUFFER_VIEW special cases
///
/// 4. **KERNEL Creation:**
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
pub fn split_store(x: &Rc<UOp>, ctx: &mut KernelContext) -> Option<Rc<UOp>> {
    // TODO: Implement range filtering
    // For now, we'll implement a simplified version that doesn't check ranges

    // **FILTERING CRITERION 1: Only process STORE and END operations**
    match x.op() {
        Op::Store { .. } | Op::StoreGated { .. } => {
            // Process STORE operations
        }
        Op::End { .. } => {
            // Process END operations
            // TODO: Skip END operations for OUTER ranges
        }
        _ => return None,
    }

    // **STEP 1: Apply transformation pipeline**
    // This is where we would apply to_define_global patterns via graph_rewrite
    // For now, we'll create a simple kernel structure

    // TODO: Implement graph_rewrite integration with to_define_global patterns
    // let transformed = graph_rewrite(x.clone(), to_define_global, ctx);

    // **STEP 2: Create SINK operation**
    // Wrap the computation in a SINK
    let sink = UOp::new(Op::Sink { sources: smallvec::smallvec![x.clone()] }, DType::Void);

    // **STEP 3: Build kernel sources from context**
    // Sources = all accessed buffers + all BIND variables
    let mut sources: SmallVec<[Rc<UOp>; 4]> = SmallVec::new();

    // Add buffers from context (these were tracked during transformation)
    // TODO: ctx.map.values() when we integrate with graph_rewrite

    // Add variables from context
    // TODO: ctx.vars.keys() when we integrate with graph_rewrite

    // **STEP 4: Create KERNEL operation**
    let kernel = UOp::kernel(sources, sink);

    Some(kernel)
}

/// Helper function to check if an operation has only OUTER ranges.
///
/// This determines whether an operation is ready to be split into a kernel.
/// An operation is ready when all its RANGE dependencies are AxisType::Outer,
/// meaning we're at the outermost scheduling level.
///
/// # Arguments
///
/// * `x` - The operation to check
///
/// # Returns
///
/// * `true` - All ranges are OUTER (ready to split)
/// * `false` - Some ranges are not OUTER (not ready yet)
fn all_ranges_outer(x: &Rc<UOp>) -> bool {
    // TODO: Implement range checking
    // This requires tracking which RANGEs each operation depends on
    // For now, we'll return true to allow splitting
    true
}

/// Helper function to check if an END operation closes an OUTER range.
///
/// END operations that close OUTER ranges are control flow markers and should
/// not be converted into kernels themselves.
///
/// # Arguments
///
/// * `x` - The END operation to check
///
/// # Returns
///
/// * `true` - This END closes an OUTER range (skip it)
/// * `false` - This END doesn't close an OUTER range (process it)
fn is_outer_end(x: &Rc<UOp>) -> bool {
    match x.op() {
        Op::End { ranges, .. } => {
            // Check if any of the ranges being closed are OUTER
            ranges.iter().any(|r| match r.op() {
                Op::Range { axis_type, .. } => *axis_type == AxisType::Outer,
                _ => false,
            })
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_dtype::DType;
    use morok_ir::ConstValue;

    #[test]
    fn test_split_store_basic() {
        let mut ctx = KernelContext::new();

        // Create a simple STORE operation
        let buffer = UOp::unique(Some(0));
        let index = UOp::const_(DType::Index, ConstValue::Int(0));
        let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let store = UOp::new(Op::Store { buffer: buffer.clone(), index, value }, DType::Void);

        // Try to split
        let result = split_store(&store, &mut ctx);

        // Should return a KERNEL
        assert!(result.is_some());
        let kernel = result.unwrap();
        assert!(matches!(kernel.op(), Op::Kernel { .. }));
    }

    #[test]
    fn test_split_store_non_store_returns_none() {
        let mut ctx = KernelContext::new();

        // Create a non-STORE operation
        let const_op = UOp::const_(DType::Float32, ConstValue::Float(1.0));

        // Try to split
        let result = split_store(&const_op, &mut ctx);

        // Should return None
        assert!(result.is_none());
    }

    #[test]
    fn test_is_outer_end() {
        // Create an END with an OUTER range
        let range_outer = UOp::range(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Outer);
        let store = UOp::noop();
        let end = UOp::end(store, smallvec::smallvec![range_outer]);

        // Should be detected as OUTER end
        assert!(is_outer_end(&end));

        // Create an END with a Loop range
        let range_loop = UOp::range(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Loop);
        let end_loop = UOp::end(UOp::noop(), smallvec::smallvec![range_loop]);

        // Should NOT be detected as OUTER end
        assert!(!is_outer_end(&end_loop));
    }
}
