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

use morok_dtype::DType;
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
            UOp::new(Op::DefineGlobal(global_id), compute.dtype())
        } else {
            // Local/shared memory: DEFINE_LOCAL
            let local_id = ctx.next_local();
            UOp::new(Op::DefineLocal(local_id), compute.dtype())
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
    let store =
        UOp::new(Op::Store { buffer: buffer.clone(), index: store_target, value: compute.clone() }, DType::Void);

    // Wrap STORE in END operation with all ranges
    // END references the computation (STORE) and closes all the ranges
    let mut result = if !ranges.is_empty() {
        UOp::new(Op::End { computation: store.clone(), ranges: ranges.clone() }, DType::Void)
    } else {
        store
    };

    // For local buffers, add BARRIER for synchronization
    // This ensures all threads in a workgroup have completed their stores
    // before any thread proceeds to read from the local buffer
    if opts.addrspace == AddrSpace::Local {
        result = UOp::new(Op::Barrier { src: result, deps: SmallVec::new() }, DType::Void);
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_dtype::DType;
    use morok_ir::{AxisType, BufferizeOpts, ConstValue};

    #[test]
    fn test_bufferize_to_store_global() {
        let mut ctx = KernelContext::new();

        // Create a simple BUFFERIZE with one range
        let compute = UOp::const_(DType::Float32, ConstValue::Float(42.0));
        let range = UOp::new(
            Op::Range { end: UOp::const_(DType::Index, ConstValue::Int(10)), axis_id: 0, axis_type: AxisType::Loop },
            DType::Index,
        );

        let bufferize = UOp::new(
            Op::Bufferize {
                compute: compute.clone(),
                ranges: smallvec::smallvec![range.clone()],
                opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
            },
            DType::Float32,
        );

        // Convert to STORE
        let result = bufferize_to_store(&bufferize, &mut ctx);

        assert!(result.is_some());
        let result = result.unwrap();

        // Should be an END operation
        assert!(matches!(result.op(), Op::End { .. }));

        // Should have created a DEFINE_GLOBAL
        assert_eq!(ctx.global_counter, 1);
        assert_eq!(ctx.local_counter, 0);

        // Should NOT have a BARRIER (global buffer)
        assert!(!matches!(result.op(), Op::Barrier { .. }));
    }

    #[test]
    fn test_bufferize_to_store_local_with_barrier() {
        let mut ctx = KernelContext::new();

        // Create BUFFERIZE with LOCAL addrspace
        let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let range = UOp::new(
            Op::Range { end: UOp::const_(DType::Index, ConstValue::Int(5)), axis_id: 0, axis_type: AxisType::Loop },
            DType::Index,
        );

        let bufferize = UOp::new(
            Op::Bufferize {
                compute: compute.clone(),
                ranges: smallvec::smallvec![range],
                opts: BufferizeOpts { device: None, addrspace: AddrSpace::Local },
            },
            DType::Float32,
        );

        // Convert to STORE
        let result = bufferize_to_store(&bufferize, &mut ctx);

        assert!(result.is_some());
        let result = result.unwrap();

        // Should be wrapped in BARRIER
        assert!(matches!(result.op(), Op::Barrier { .. }));

        // Should have created a DEFINE_LOCAL
        assert_eq!(ctx.global_counter, 0);
        assert_eq!(ctx.local_counter, 1);
    }

    #[test]
    fn test_bufferize_to_store_no_ranges() {
        let mut ctx = KernelContext::new();

        // Create BUFFERIZE with no ranges (scalar store)
        let compute = UOp::const_(DType::Float32, ConstValue::Float(3.14));

        let bufferize = UOp::new(
            Op::Bufferize {
                compute: compute.clone(),
                ranges: SmallVec::new(),
                opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
            },
            DType::Float32,
        );

        // Convert to STORE
        let result = bufferize_to_store(&bufferize, &mut ctx);

        assert!(result.is_some());
        let result = result.unwrap();

        // With no ranges, should be a STORE directly (no END wrapping)
        assert!(matches!(result.op(), Op::Store { .. }));
    }

    #[test]
    fn test_bufferize_to_store_multiple_ranges() {
        let mut ctx = KernelContext::new();

        // Create BUFFERIZE with multiple ranges
        let compute = UOp::const_(DType::Int32, ConstValue::Int(100));
        let range1 = UOp::new(
            Op::Range { end: UOp::const_(DType::Index, ConstValue::Int(4)), axis_id: 0, axis_type: AxisType::Loop },
            DType::Index,
        );
        let range2 = UOp::new(
            Op::Range { end: UOp::const_(DType::Index, ConstValue::Int(8)), axis_id: 1, axis_type: AxisType::Loop },
            DType::Index,
        );

        let bufferize = UOp::new(
            Op::Bufferize {
                compute: compute.clone(),
                ranges: smallvec::smallvec![range1, range2],
                opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
            },
            DType::Int32,
        );

        // Convert to STORE
        let result = bufferize_to_store(&bufferize, &mut ctx);

        assert!(result.is_some());
        let result = result.unwrap();

        // Should be wrapped in END
        assert!(matches!(result.op(), Op::End { .. }));

        // Should have STORE as computation and 2 ranges
        if let Op::End { computation, ranges } = result.op() {
            // Computation should be STORE
            assert!(matches!(computation.op(), Op::Store { .. }));
            // Should have 2 ranges
            assert_eq!(ranges.len(), 2);
        } else {
            panic!("Expected END operation");
        }
    }

    #[test]
    fn test_non_bufferize_returns_none() {
        let mut ctx = KernelContext::new();

        // Create a non-BUFFERIZE operation
        let const_op = UOp::const_(DType::Float32, ConstValue::Float(1.0));

        // Should return None
        let result = bufferize_to_store(&const_op, &mut ctx);
        assert!(result.is_none());
    }

    #[test]
    fn test_buffer_tracked_in_context() {
        let mut ctx = KernelContext::new();

        let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let bufferize = UOp::new(
            Op::Bufferize {
                compute,
                ranges: SmallVec::new(),
                opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
            },
            DType::Float32,
        );

        // Before conversion, buffer should not be tracked
        assert!(!ctx.has_buffer(&bufferize));

        // Convert to STORE
        bufferize_to_store(&bufferize, &mut ctx);

        // After conversion, buffer should be tracked
        assert!(ctx.has_buffer(&bufferize));

        // Should be able to get the DEFINE_GLOBAL
        let replacement = ctx.get_buffer(&bufferize).unwrap();
        assert!(matches!(replacement.op(), Op::DefineGlobal(_)));
    }
}
