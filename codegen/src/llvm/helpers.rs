//! Helper functions for LLVM code generation using inkwell.

use inkwell::values::BasicValueEnum;
use std::collections::HashMap;

/// Value tracker for UOp to LLVM value mapping.
///
/// Maps UOp IDs to their corresponding LLVM values (inkwell's BasicValueEnum).
/// This allows us to look up previously generated values when processing the graph.
pub struct ValueMap<'ctx> {
    uop_to_value: HashMap<u64, BasicValueEnum<'ctx>>,
}

impl<'ctx> ValueMap<'ctx> {
    pub fn new() -> Self {
        Self { uop_to_value: HashMap::new() }
    }

    /// Store a value for a UOp ID.
    pub fn insert(&mut self, uop_id: u64, value: BasicValueEnum<'ctx>) {
        self.uop_to_value.insert(uop_id, value);
    }

    /// Get a value for a UOp ID.
    pub fn get(&self, uop_id: u64) -> Option<BasicValueEnum<'ctx>> {
        self.uop_to_value.get(&uop_id).copied()
    }

    /// Check if a UOp has been compiled.
    pub fn contains(&self, uop_id: u64) -> bool {
        self.uop_to_value.contains_key(&uop_id)
    }
}

impl<'ctx> Default for ValueMap<'ctx> {
    fn default() -> Self {
        Self::new()
    }
}
