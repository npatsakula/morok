//! Helper structures for Cranelift code generation.

use std::sync::Arc;

use cranelift_codegen::ir::Block;
use cranelift_frontend::Variable;

use morok_ir::{Op, UOp};

/// Loop context for tracking loop structure during codegen.
pub(crate) struct LoopContext {
    pub header_block: Block,
    pub body_block: Block,
    pub exit_block: Block,
    pub loop_var: Variable,
}

/// Collect all buffer and variable parameters from a UOp graph.
///
/// Returns (buffers, variables) in a consistent order:
/// - Buffers: DEFINE_GLOBAL sorted by ID, DEFINE_LOCAL sorted by ID, BUFFER sorted by UOp ID
/// - Variables: DEFINE_VAR sorted by name
pub(crate) fn collect_buffers_and_vars(root: &Arc<UOp>) -> (Vec<Arc<UOp>>, Vec<Arc<UOp>>) {
    let nodes = root.toposort();

    // Collect buffers
    let mut buffers = Vec::new();
    for node in &nodes {
        match node.op() {
            Op::Buffer { .. } | Op::DefineGlobal(_) | Op::DefineLocal(_) => {
                buffers.push(node.clone());
            }
            _ => {}
        }
    }

    // Sort buffers by internal ID
    buffers.sort_by_key(|b| match b.op() {
        Op::DefineGlobal(id) => *id as u64,
        Op::DefineLocal(id) => (*id as u64) + (1u64 << 32),
        Op::Buffer { .. } => b.id + (1u64 << 48),
        _ => b.id,
    });

    // Collect variables
    let mut variables = Vec::new();
    for node in &nodes {
        if let Op::DefineVar { .. } = node.op() {
            variables.push(node.clone());
        }
    }

    // Sort variables by name
    variables.sort_by(|a, b| {
        let name_a = match a.op() {
            Op::DefineVar { name, .. } => name,
            _ => unreachable!("filtered to only DefineVar above"),
        };
        let name_b = match b.op() {
            Op::DefineVar { name, .. } => name,
            _ => unreachable!("filtered to only DefineVar above"),
        };
        name_a.cmp(name_b)
    });

    (buffers, variables)
}
