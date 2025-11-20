//! BUFFERIZE to STORE conversion.
//!
//! This module implements the conversion of high-level BUFFERIZE operations
//! into low-level STORE operations with explicit buffer allocation.
//!
//! The transformation converts:
//! - BUFFERIZE(compute, ranges, opts) with GLOBAL addrspace
//!   → DEFINE_GLOBAL + STORE wrapped in END operations
//!
//! - BUFFERIZE(compute, ranges, opts) with LOCAL addrspace
//!   → DEFINE_LOCAL + STORE wrapped in END operations + BARRIER

use std::rc::Rc;

use morok_ir::{AddrSpace, Op, UOp};
use smallvec::SmallVec;

use super::kernel_context::KernelContext;

/// Convert BUFFERIZE operation to STORE with buffer allocation and END wrapping.
///
/// This function performs the core transformation from high-level BUFFERIZE to
/// low-level memory operations. It:
///
/// 1. Creates appropriate buffer allocation (DEFINE_GLOBAL or DEFINE_LOCAL)
/// 2. Creates STORE operation to write computed value
/// 3. Wraps STORE in END operations for each range (innermost to outermost)
/// 4. Adds BARRIER for local buffers (synchronization)
///
/// # Arguments
///
/// * `bufferize_op` - The BUFFERIZE operation to convert
/// * `ctx` - Mutable kernel context for tracking buffer allocations
///
/// # Returns
///
/// * `Some(uop)` - The converted operation (STORE wrapped in ENDs and optionally BARRIER)
/// * `None` - If the input is not a BUFFERIZE operation
///
/// # Example
///
/// ```ignore
/// // Input: BUFFERIZE(compute, [range1, range2], {addrspace: GLOBAL})
/// // Output: END(range2, END(range1, STORE(DEFINE_GLOBAL(0), index, compute)))
/// ```
pub fn bufferize_to_store(bufferize_op: &Rc<UOp>, ctx: &mut KernelContext) -> Option<Rc<UOp>> {
    // Extract BUFFERIZE components
    let (compute, ranges, opts) = match bufferize_op.op() {
        Op::Bufferize { compute, ranges, opts } => (compute, ranges, opts),
        _ => return None,
    };

    // Check if we've already allocated a buffer for this BUFFERIZE
    // If so, reuse it to avoid creating duplicate buffers for the same operation
    let buffer = if let Some(existing_buffer) = ctx.get_buffer(bufferize_op) {
        // Reuse existing buffer
        existing_buffer.clone()
    } else {
        // Create new buffer allocation based on address space
        let buffer = if opts.addrspace == AddrSpace::Global {
            // Global memory: DEFINE_GLOBAL
            let global_id = ctx.next_global();
            UOp::define_global(global_id, compute.dtype())
        } else {
            // Local/shared memory: DEFINE_LOCAL
            let local_id = ctx.next_local();
            UOp::define_local(local_id, compute.dtype())
        };

        // Track the buffer in context for later reference
        ctx.map_buffer(bufferize_op.clone(), buffer.clone());

        buffer
    };

    // Create index for the STORE
    // If we have ranges, create an INDEX operation
    // Otherwise, store directly to the buffer
    let store_target = if !ranges.is_empty() {
        // Create INDEX with the buffer and all ranges
        UOp::index(buffer.clone(), ranges.to_vec()).expect("Failed to create INDEX for BUFFERIZE-to-STORE conversion")
    } else {
        // No ranges - store directly to buffer
        buffer.clone()
    };

    // Create STORE operation
    let store = UOp::store(buffer.clone(), store_target, compute.clone());

    // Wrap STORE in END operation with all ranges
    // END references the computation (STORE) and closes all the ranges
    let mut result = if !ranges.is_empty() {
        UOp::end(store.clone(), ranges.clone())
    } else {
        store
    };

    // For local buffers, add BARRIER for synchronization
    // This ensures all threads in a workgroup have completed their stores
    // before any thread proceeds to read from the local buffer
    if opts.addrspace == AddrSpace::Local {
        result = UOp::barrier(result, SmallVec::new());
    }

    Some(result)
}
