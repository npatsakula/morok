//! Buffer access cycle detection: validate no LOAD/STORE conflicts in kernels.

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

/// Unwrap buffer-like ops (MSelect, MStack, After) to get the underlying buffer.
pub fn as_buf(uop: &Rc<UOp>) -> Rc<UOp> {
    match uop.op() {
        Op::MSelect { buffer, .. } => buffer.clone(),
        Op::MStack { buffers } if !buffers.is_empty() => buffers[0].clone(),
        Op::After { passthrough, .. } => passthrough.clone(),
        _ => uop.clone(),
    }
}

/// Detect conflicting buffer accesses. Panics if same buffer has both LOAD and STORE.
#[allow(clippy::mutable_key_type)]
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
            if let Some(&existing_access) = ret.get(&buf_key)
                && existing_access != OpAccessType::Load
            {
                panic!(
                    "buffer accessed with conflicting ops: {:?} (existing: {:?}, new: {:?})",
                    buf,
                    existing_access,
                    OpAccessType::Load
                );
            }

            ret.insert(buf_key, OpAccessType::Load);
        }

        // Check for STORE operations (writing to buffer)
        if let Op::Store { buffer, .. } | Op::StoreGated { buffer, .. } = node.op() {
            let buf = as_buf(buffer);
            let buf_key = UOpKey(buf.clone());

            // Check for conflicting access
            if let Some(&existing_access) = ret.get(&buf_key)
                && existing_access != OpAccessType::Store
            {
                panic!(
                    "buffer accessed with conflicting ops: {:?} (existing: {:?}, new: {:?})",
                    buf,
                    existing_access,
                    OpAccessType::Store
                );
            }

            ret.insert(buf_key, OpAccessType::Store);
        }
    }

    ret
}
