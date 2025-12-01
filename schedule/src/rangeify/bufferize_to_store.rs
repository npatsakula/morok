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

use morok_ir::{AddrSpace, ConstValue, Op, UOp};
use smallvec::SmallVec;

use super::kernel_context::KernelContext;

/// Calculate buffer size from RANGE operations.
///
/// Size = product of all range bounds.
/// For ranges with constant bounds, returns the product.
///
/// # Panics
///
/// Panics if any range has a symbolic (non-constant) bound.
/// This matches Tinygrad's behavior: `assert isinstance(size, int), "no symbolic sized buffers"`
fn calculate_size_from_ranges(ranges: &SmallVec<[Rc<UOp>; 4]>) -> usize {
    if ranges.is_empty() {
        return 1;
    }

    ranges
        .iter()
        .map(|r| {
            if let Op::Range { end, .. } = r.op() {
                // Extract constant bound from end (via vmax for symbolic simplification)
                match end.vmax() {
                    ConstValue::Int(v) if *v > 0 => *v as usize,
                    ConstValue::UInt(v) if *v > 0 => *v as usize,
                    other => panic!(
                        "Cannot allocate buffer with symbolic size: range bound resolved to {:?}. \
                         Buffers require concrete sizes (Tinygrad: 'no symbolic sized buffers')",
                        other
                    ),
                }
            } else {
                // Non-RANGE operations contribute size 1 (scalar)
                1
            }
        })
        .product()
}

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
        // Calculate buffer size from ranges (like Tinygrad's prod(shape))
        let size = calculate_size_from_ranges(ranges);

        // Create Ptr dtype with embedded size (like Tinygrad's dtype.ptr(size=...))
        let base_dtype = compute.dtype();
        let ptr_dtype = base_dtype.ptr(Some(size), opts.addrspace);

        // Create new buffer allocation based on address space
        let buffer = if opts.addrspace == AddrSpace::Global {
            // Global memory: DEFINE_GLOBAL
            let global_id = ctx.next_global();
            UOp::define_global(global_id, ptr_dtype)
        } else {
            // Local/shared memory: DEFINE_LOCAL
            let local_id = ctx.next_local();
            UOp::define_local(local_id, ptr_dtype)
        };

        // DON'T track buffer here - handle_after will do it later
        // Following Tinygrad's architecture: ctx.map is populated by handle_after pattern

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
    let mut do_store = if !ranges.is_empty() { UOp::end(store.clone(), ranges.clone()) } else { store };

    // For local buffers, add BARRIER for synchronization
    // This ensures all threads in a workgroup have completed their stores
    // before any thread proceeds to read from the local buffer
    if opts.addrspace == AddrSpace::Local {
        do_store = UOp::barrier(do_store, SmallVec::new());
    }

    // Following Tinygrad: return buffer.after(do_store)
    // This creates the AFTER operation for dependency tracking
    let result = UOp::after(buffer.clone(), SmallVec::from_elem(do_store, 1));

    // Track the mapping: BUFFERIZE → AFTER
    // This is critical for output buffers to be included in kernel sources.
    // For input buffers, debuf pattern maps BUFFER → BUFFER (identity).
    // For output buffers (BUFFERIZE), we map BUFFERIZE → AFTER here.
    //
    // Based on Tinygrad: ctx.map[buffer] = buffer.after(do_store)
    // where buffer is the DEFINE_GLOBAL created for the BUFFERIZE output.
    ctx.map_buffer(bufferize_op.clone(), result.clone());

    Some(result)
}
