//! Helper functions for LLVM code generation using inkwell.

use inkwell::basic_block::BasicBlock;
use inkwell::values::{BasicValueEnum, IntValue, PhiValue};
use std::collections::HashMap;

/// Loop context for tracking RANGE-generated basic blocks.
///
/// When codegen encounters a RANGE operation, it creates this structure
/// to track all the basic blocks and values needed to complete the loop
/// when END is encountered.
///
/// Loop structure (latch-test pattern matching Tinygrad):
/// ```text
/// loop_entry_N:    → unconditional branch to latch
/// loop_latch_N:    → PHI node, increment, condition check, conditional branch
/// loop_body_N:     → loop body code (positioned here after RANGE)
/// loop_footer_N:   → branch back to latch (END branches here)
/// loop_exit_N:     → continuation after loop (positioned here after END)
/// ```
#[derive(Clone)]
pub struct LoopContext<'ctx> {
    /// Loop latch block - contains PHI, increment, and condition check
    pub latch_block: BasicBlock<'ctx>,
    /// Loop footer block - END branches here, then jumps to latch
    pub footer_block: BasicBlock<'ctx>,
    /// Loop exit block - continuation after loop completes
    pub exit_block: BasicBlock<'ctx>,
    /// PHI node for loop counter (incoming edges from entry and footer)
    pub phi: PhiValue<'ctx>,
    /// Incremented counter value (for PHI incoming edge from footer)
    pub incremented: IntValue<'ctx>,
}

/// Value tracker for UOp to LLVM value mapping.
///
/// Maps UOp IDs to their corresponding LLVM values (inkwell's BasicValueEnum).
/// This allows us to look up previously generated values when processing the graph.
/// Also tracks loop contexts for RANGE/END pairs, and "processed but no value" UOps.
pub struct ValueMap<'ctx> {
    uop_to_value: HashMap<u64, BasicValueEnum<'ctx>>,
    loop_contexts: HashMap<u64, LoopContext<'ctx>>,
    /// UOps that were processed but don't produce a value (like END, SINK).
    /// This prevents re-processing when they're encountered again.
    processed_no_value: std::collections::HashSet<u64>,
}

impl<'ctx> ValueMap<'ctx> {
    pub fn new() -> Self {
        Self {
            uop_to_value: HashMap::new(),
            loop_contexts: HashMap::new(),
            processed_no_value: std::collections::HashSet::new(),
        }
    }

    /// Store a value for a UOp ID.
    pub fn insert(&mut self, uop_id: u64, value: BasicValueEnum<'ctx>) {
        self.uop_to_value.insert(uop_id, value);
    }

    /// Get a value for a UOp ID.
    pub fn get(&self, uop_id: u64) -> Option<BasicValueEnum<'ctx>> {
        self.uop_to_value.get(&uop_id).copied()
    }

    /// Check if a UOp has been compiled (either has value or was processed with no value).
    pub fn contains(&self, uop_id: u64) -> bool {
        self.uop_to_value.contains_key(&uop_id) || self.processed_no_value.contains(&uop_id)
    }

    /// Mark a UOp as processed even though it doesn't produce a value.
    pub fn mark_processed(&mut self, uop_id: u64) {
        self.processed_no_value.insert(uop_id);
    }

    /// Store loop context for a RANGE operation.
    pub fn insert_loop(&mut self, range_id: u64, ctx: LoopContext<'ctx>) {
        self.loop_contexts.insert(range_id, ctx);
    }

    /// Get loop context for a RANGE operation.
    pub fn get_loop(&self, range_id: u64) -> Option<&LoopContext<'ctx>> {
        self.loop_contexts.get(&range_id)
    }
}

impl<'ctx> Default for ValueMap<'ctx> {
    fn default() -> Self {
        Self::new()
    }
}
