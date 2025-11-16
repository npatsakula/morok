//! Kernel context for tracking state during kernel splitting.
//!
//! This module provides the KernelContext struct which tracks buffer allocations,
//! variable bindings, and range numbering during the kernel splitting phase.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use morok_ir::{UOp, UOpKey};

/// Context for tracking state during kernel splitting.
///
/// KernelContext maintains the state needed to convert a computation graph
/// into individual executable kernels. It tracks:
/// - Global buffer allocations (DEFINE_GLOBAL numbering)
/// - Local buffer allocations (DEFINE_LOCAL numbering)
/// - Buffer replacement mappings (BUFFER → DEFINE_GLOBAL/AFTER)
/// - Kernel-local variable bindings
/// - Range renumbering for deduplication
#[derive(Clone)]
pub struct KernelContext {
    /// Counter for DEFINE_GLOBAL numbering.
    ///
    /// Each BUFFER operation gets converted to a DEFINE_GLOBAL with a unique ID.
    /// This counter ensures IDs are unique across all kernels.
    pub global_counter: usize,

    /// Counter for DEFINE_LOCAL numbering.
    ///
    /// Local buffers (shared memory) get DEFINE_LOCAL operations with unique IDs.
    pub local_counter: usize,

    /// Map from original buffers to their DEFINE_GLOBAL/AFTER replacements.
    ///
    /// During kernel splitting, BUFFER operations are replaced with either:
    /// - DEFINE_GLOBAL: For global memory allocations
    /// - AFTER: For dependency tracking between kernels
    ///
    /// This map tracks these replacements so later operations can reference
    /// the correct buffer.
    pub buffer_map: HashMap<UOpKey, Rc<UOp>>,

    /// Variable bindings for kernel-local vars.
    ///
    /// Tracks which variables are bound within this kernel scope.
    /// Used to unbind variables when converting BIND operations.
    pub vars: HashSet<UOpKey>,

    /// Range renumbering counter (for deduplication).
    ///
    /// RANGE operations are renumbered starting from 0 within each kernel.
    /// This enables deduplication of identical kernels that differ only in
    /// their range IDs.
    pub range_counter: usize,
}

impl KernelContext {
    /// Create a new empty kernel context.
    pub fn new() -> Self {
        Self { global_counter: 0, local_counter: 0, buffer_map: HashMap::new(), vars: HashSet::new(), range_counter: 0 }
    }

    /// Allocate and return the next DEFINE_GLOBAL ID.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut ctx = KernelContext::new();
    /// let id1 = ctx.next_global();  // 0
    /// let id2 = ctx.next_global();  // 1
    /// ```
    pub fn next_global(&mut self) -> usize {
        let id = self.global_counter;
        self.global_counter += 1;
        id
    }

    /// Allocate and return the next DEFINE_LOCAL ID.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut ctx = KernelContext::new();
    /// let id1 = ctx.next_local();  // 0
    /// let id2 = ctx.next_local();  // 1
    /// ```
    pub fn next_local(&mut self) -> usize {
        let id = self.local_counter;
        self.local_counter += 1;
        id
    }

    /// Allocate and return the next RANGE ID.
    ///
    /// Used for renumbering ranges within a kernel for deduplication.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut ctx = KernelContext::new();
    /// let id1 = ctx.next_range();  // 0
    /// let id2 = ctx.next_range();  // 1
    /// ```
    pub fn next_range(&mut self) -> usize {
        let id = self.range_counter;
        self.range_counter += 1;
        id
    }

    /// Check if a buffer is already mapped.
    pub fn has_buffer(&self, buf: &Rc<UOp>) -> bool {
        self.buffer_map.contains_key(&UOpKey(buf.clone()))
    }

    /// Get the replacement for a buffer, if it exists.
    pub fn get_buffer(&self, buf: &Rc<UOp>) -> Option<&Rc<UOp>> {
        self.buffer_map.get(&UOpKey(buf.clone()))
    }

    /// Map a buffer to its replacement.
    pub fn map_buffer(&mut self, original: Rc<UOp>, replacement: Rc<UOp>) {
        self.buffer_map.insert(UOpKey(original), replacement);
    }

    /// Add a variable to the kernel-local variable set.
    pub fn add_var(&mut self, var: Rc<UOp>) {
        self.vars.insert(UOpKey(var));
    }

    /// Check if a variable is kernel-local.
    pub fn has_var(&self, var: &Rc<UOp>) -> bool {
        self.vars.contains(&UOpKey(var.clone()))
    }
}

impl Default for KernelContext {
    fn default() -> Self {
        Self::new()
    }
}
