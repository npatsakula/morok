//! Buffer access cycle detection for kernel splitting validation.
//!
//! This module implements buffer access validation to prevent creating invalid
//! kernels where a buffer is accessed with conflicting operation types (LOAD vs STORE).
//!
//! Based on Tinygrad's find_bufs (schedule/rangeify.py:413-417).

use std::collections::HashMap;
use std::rc::Rc;

use morok_ir::{Op, UOp, UOpKey};

/// Buffer access types for cycle detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpAccessType {
    /// Buffer is being read from (LOAD operation)
    Load,
    /// Buffer is being written to (STORE operation)
    Store,
}

/// Extract the underlying buffer from operations like MSelect, MStack, After.
///
/// This helper unwraps buffer-like operations to get the actual buffer being accessed.
/// Used to normalize buffer references for conflict detection.
///
/// # Arguments
///
/// * `uop` - The operation that may wrap a buffer
///
/// # Returns
///
/// The underlying buffer, or the original UOp if it's already a buffer
///
/// # Example
///
/// ```ignore
/// let mselect = UOp::new(Op::MSelect { buffer, selector, device_index }, dtype);
/// let buf = as_buf(&mselect);
/// assert!(Rc::ptr_eq(&buf, &buffer));
/// ```
pub fn as_buf(uop: &Rc<UOp>) -> Rc<UOp> {
    match uop.op() {
        Op::MSelect { buffer, .. } => buffer.clone(),
        Op::MStack { buffers } if !buffers.is_empty() => buffers[0].clone(),
        Op::After { passthrough, .. } => passthrough.clone(),
        _ => uop.clone(),
    }
}

/// Detect conflicting buffer accesses in a kernel.
///
/// This function validates that no buffer is accessed with both LOAD and STORE
/// operations within the same kernel, which would create an access conflict.
///
/// # Algorithm
///
/// 1. Perform topological sort excluding AFTER operations (they're dependency markers)
/// 2. For each LOAD/STORE operation found:
///    - Extract the buffer being accessed
///    - Check if we've seen this buffer before
///    - If yes, verify the access type matches (both LOAD or both STORE)
///    - If conflict detected, panic with error message
///
/// # Arguments
///
/// * `store` - The STORE operation representing the kernel's computation
///
/// # Returns
///
/// A mapping of buffers to their access types. Panics if conflicts are detected.
///
/// # Panics
///
/// Panics if a buffer is accessed with conflicting operation types (e.g., both
/// LOAD and STORE in the same kernel).
///
/// # Example
///
/// ```ignore
/// use morok_ir::{UOp, Op, DType, ConstValue};
/// use morok_schedule::rangeify::cycle_detection::find_bufs;
///
/// let buffer = UOp::unique(Some(0));
/// let index = UOp::const_(DType::Index, ConstValue::Int(0));
/// let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
/// let store = UOp::new(Op::Store { buffer, index, value }, DType::Void);
///
/// // This should succeed - only STORE access
/// let buf_accesses = find_bufs(&store);
/// ```
///
/// Based on Tinygrad's find_bufs (schedule/rangeify.py:413-417):
/// ```python
/// def find_bufs(store:UOp) -> dict[UOp, OpAccessType]:
///   ret:dict[UOp, OpAccessType] = {}
///   for idx in store.toposort((UOp.AFTER,)):
///     if idx.op is Ops.INDEX:
///       if idx.src[0] in ret and ret[idx.src[0]] != idx.arg:
///         raise RuntimeError(f"buffer accessed with conflicting ops: {idx.src[0]}")
///       ret[idx.src[0]] = idx.arg
///   return ret
/// ```
pub fn find_bufs(store: &Rc<UOp>) -> HashMap<UOpKey, OpAccessType> {
    let mut ret: HashMap<UOpKey, OpAccessType> = HashMap::new();

    // Toposort with gate: exclude AFTER operations from traversal
    // AFTER represents "buffer after computation" and is a dependency marker,
    // not an actual buffer access operation
    let nodes = store.toposort_filtered(|uop| !matches!(uop.op(), Op::After { .. }));

    for node in nodes {
        // Check for LOAD operations (reading from buffer)
        if let Op::Load { buffer, .. } | Op::LoadGated { buffer, .. } = node.op() {
            let buf = as_buf(buffer);
            let buf_key = UOpKey(buf.clone());

            // Check for conflicting access
            if let Some(&existing_access) = ret.get(&buf_key) {
                if existing_access != OpAccessType::Load {
                    panic!(
                        "buffer accessed with conflicting ops: {:?} (existing: {:?}, new: {:?})",
                        buf,
                        existing_access,
                        OpAccessType::Load
                    );
                }
            }

            ret.insert(buf_key, OpAccessType::Load);
        }

        // Check for STORE operations (writing to buffer)
        if let Op::Store { buffer, .. } | Op::StoreGated { buffer, .. } = node.op() {
            let buf = as_buf(buffer);
            let buf_key = UOpKey(buf.clone());

            // Check for conflicting access
            if let Some(&existing_access) = ret.get(&buf_key) {
                if existing_access != OpAccessType::Store {
                    panic!(
                        "buffer accessed with conflicting ops: {:?} (existing: {:?}, new: {:?})",
                        buf,
                        existing_access,
                        OpAccessType::Store
                    );
                }
            }

            ret.insert(buf_key, OpAccessType::Store);
        }
    }

    ret
}
