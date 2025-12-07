//! Simple debug printing for UOp ASTs.

use crate::{Op, UOp};
use std::sync::Arc;

/// Print a toposorted list of UOps
pub fn print_ast(root: &Arc<UOp>, label: &str, _max_depth: usize) {
    eprintln!("=== {} ===", label);
    for (i, node) in root.toposort().iter().enumerate() {
        // Print sources IDs
        let sources: Vec<u64> = node.op().sources().iter().map(|s| s.id).collect();
        let op_name = match node.op() {
            Op::DefineGlobal(id) => format!("DEFINE_GLOBAL({})", id),
            Op::DefineLocal(id) => format!("DEFINE_LOCAL({})", id),
            Op::DefineVar { name, max_val } => {
                format!("DEFINE_VAR('{}', max={})", name, max_val)
            }
            Op::Buffer { .. } => "BUFFER".to_string(),
            Op::Bufferize { .. } => "BUFFERIZE".to_string(),
            Op::Load { .. } => "LOAD".to_string(),
            Op::Store { .. } => "STORE".to_string(),
            Op::Index { .. } => "INDEX".to_string(),
            Op::Binary(bop, ..) => format!("{:?}", bop),
            Op::Reduce { reduce_op, ranges, .. } => {
                let range_ids: Vec<u64> = ranges.iter().map(|r| r.id).collect();
                format!("REDUCE(op={:?}, ranges={:?})", reduce_op, range_ids)
            }
            Op::Bind { .. } => "BIND".to_string(),
            Op::Range { axis_id, axis_type, .. } => format!("RANGE(axis={:?}, type={:?})", axis_id, axis_type),
            Op::Const(val) => format!("CONST({:?})", val),
            Op::Sink { .. } => "SINK".to_string(),
            Op::Kernel { .. } => "KERNEL".to_string(),
            Op::Vectorize { elements } => format!("VECTORIZE(len={})", elements.len()),
            _ => format!("{:?}", node.op()),
        };
        eprintln!("  [{}] id={}: {} srcs={:?}", i, node.id, op_name, sources);
    }
    eprintln!("=== End {} ===\n", label);
}
