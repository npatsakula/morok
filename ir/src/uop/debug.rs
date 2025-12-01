//! Simple debug printing for UOp ASTs.

use std::rc::Rc;
use crate::{UOp, Op};

/// Print a toposorted list of UOps
pub fn print_ast(root: &Rc<UOp>, label: &str, _max_depth: usize) {
    eprintln!("=== {} ===", label);
    for (i, node) in root.toposort().iter().enumerate() {
        let op_name = match node.op() {
            Op::DefineGlobal(id) => format!("DEFINE_GLOBAL({})", id),
            Op::DefineLocal(id) => format!("DEFINE_LOCAL({})", id),
            Op::DefineVar { name, min_val, max_val } => format!("DEFINE_VAR('{}', min={}, max={})", name, min_val, max_val),
            Op::Buffer { .. } => "BUFFER".to_string(),
            Op::Bufferize { .. } => "BUFFERIZE".to_string(),
            Op::Load { .. } => "LOAD".to_string(),
            Op::Store { .. } => "STORE".to_string(),
            Op::Index { .. } => "INDEX".to_string(),
            Op::Binary(bop, ..) => format!("{:?}", bop),
            Op::Reduce { .. } => "REDUCE".to_string(),
            Op::Bind { .. } => "BIND".to_string(),
            Op::Range { axis_id, axis_type, .. } => format!("RANGE(axis={:?}, type={:?})", axis_id, axis_type),
            Op::Const(val) => format!("CONST({:?})", val),
            Op::Sink { .. } => "SINK".to_string(),
            Op::Kernel { .. } => "KERNEL".to_string(),
            _ => format!("{:?}", node.op()),
        };
        eprintln!("  [{}] id={}: {}", i, node.id, op_name);
    }
    eprintln!("=== End {} ===\n", label);
}
