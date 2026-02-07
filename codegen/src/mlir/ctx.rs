//! Render context for MLIR code generation.
//!
//! Maps UOp IDs to MLIR Values and manages loop/reduce/if block tracking.
//! Uses raw `MlirBlock` handles for block storage to avoid Melior lifetime issues.

use std::collections::HashMap;

use melior::ir::{BlockRef, Value};
use mlir_sys::MlirBlock;

/// Loop block handles for Range/End control flow.
pub struct LoopBlocks {
    pub latch: MlirBlock,
    pub body: MlirBlock,
    pub footer: MlirBlock,
    pub exit: MlirBlock,
}

impl LoopBlocks {
    /// Get a safe BlockRef for the latch block.
    ///
    /// # Safety
    /// The underlying block must still be alive (owned by the function region).
    pub unsafe fn latch_ref<'c, 'a>(&self) -> BlockRef<'c, 'a> {
        unsafe { BlockRef::from_raw(self.latch) }
    }

    pub unsafe fn body_ref<'c, 'a>(&self) -> BlockRef<'c, 'a> {
        unsafe { BlockRef::from_raw(self.body) }
    }

    pub unsafe fn footer_ref<'c, 'a>(&self) -> BlockRef<'c, 'a> {
        unsafe { BlockRef::from_raw(self.footer) }
    }

    pub unsafe fn exit_ref<'c, 'a>(&self) -> BlockRef<'c, 'a> {
        unsafe { BlockRef::from_raw(self.exit) }
    }
}

/// Pending reduce accumulator info.
pub struct PendingReduce<'c> {
    pub acc_ptr: Value<'c, 'c>,
    pub result_type: melior::ir::Type<'c>,
}

/// Maps UOp ID → MLIR Value and manages control flow blocks.
pub struct RenderContext<'c> {
    values: HashMap<u64, Value<'c, 'c>>,
    loop_blocks: HashMap<usize, LoopBlocks>,
    if_end_blocks: HashMap<u64, MlirBlock>,
    pending_reduces: HashMap<u64, PendingReduce<'c>>,
    /// The current insertion block (raw handle).
    current_block: MlirBlock,
}

impl<'c> RenderContext<'c> {
    /// Create a new context with the given entry block as current.
    pub fn new(entry_block: MlirBlock) -> Self {
        Self {
            values: HashMap::new(),
            loop_blocks: HashMap::new(),
            if_end_blocks: HashMap::new(),
            pending_reduces: HashMap::new(),
            current_block: entry_block,
        }
    }

    /// Get a safe BlockRef for the current insertion block.
    ///
    /// # Safety
    /// The underlying block must still be alive (owned by the function region).
    pub unsafe fn current_block_ref<'a>(&self) -> BlockRef<'c, 'a> {
        unsafe { BlockRef::from_raw(self.current_block) }
    }

    /// Set the current insertion block.
    pub fn set_current_block(&mut self, block: MlirBlock) {
        self.current_block = block;
    }

    /// Register a value for a UOp ID.
    pub fn register(&mut self, id: u64, value: Value<'c, 'c>) {
        self.values.insert(id, value);
    }

    /// Get the value for a UOp ID.
    pub fn get(&self, id: u64) -> Value<'c, 'c> {
        self.values
            .get(&id)
            .copied()
            .unwrap_or_else(|| panic!("UOp {} not in MLIR context", id))
    }

    /// Try to get a value.
    pub fn try_get(&self, id: u64) -> Option<Value<'c, 'c>> {
        self.values.get(&id).copied()
    }

    /// Check if a UOp is registered.
    pub fn contains(&self, id: u64) -> bool {
        self.values.contains_key(&id)
    }

    /// Register loop blocks for an axis ID.
    pub fn register_loop(&mut self, axis_id: usize, blocks: LoopBlocks) {
        self.loop_blocks.insert(axis_id, blocks);
    }

    /// Get loop blocks for an axis ID.
    pub fn get_loop(&self, axis_id: usize) -> Option<&LoopBlocks> {
        self.loop_blocks.get(&axis_id)
    }

    /// Register an end block for an if statement.
    pub fn register_if_end(&mut self, if_id: u64, end_block: MlirBlock) {
        self.if_end_blocks.insert(if_id, end_block);
    }

    /// Take the end block for an if statement.
    pub fn take_if_end(&mut self, if_id: u64) -> Option<MlirBlock> {
        self.if_end_blocks.remove(&if_id)
    }

    /// Register a pending reduce accumulator.
    pub fn register_reduce_pending(
        &mut self,
        reduce_id: u64,
        acc_ptr: Value<'c, 'c>,
        result_type: melior::ir::Type<'c>,
    ) {
        self.pending_reduces
            .insert(reduce_id, PendingReduce { acc_ptr, result_type });
    }

    /// Peek at a pending reduce without removing it.
    pub fn peek_pending_reduce(&self, reduce_id: u64) -> Option<&PendingReduce<'c>> {
        self.pending_reduces.get(&reduce_id)
    }

    /// Take all pending reduces.
    pub fn take_pending_reduces(&mut self) -> HashMap<u64, PendingReduce<'c>> {
        std::mem::take(&mut self.pending_reduces)
    }
}
