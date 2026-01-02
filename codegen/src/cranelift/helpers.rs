//! Helper structures for Cranelift code generation.

use cranelift_codegen::ir::Block;
use cranelift_frontend::Variable;

/// Loop context for tracking loop structure during codegen.
pub(crate) struct LoopContext {
    pub header_block: Block,
    pub body_block: Block,
    pub exit_block: Block,
    pub loop_var: Variable,
}
