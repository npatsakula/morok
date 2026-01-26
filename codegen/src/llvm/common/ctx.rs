//! Render context for LLVM IR text generation.
//!
//! Maps UOp IDs to LLVM variable names and manages naming.
//! Shared between CPU and GPU backends.

use std::collections::HashMap;
use std::sync::Arc;

use morok_ir::{ConstValue, Op, prelude::*};

use super::types::{lconst, ldt};

/// Pending reduce load info.
pub struct PendingReduce {
    pub acc_ptr: String,
    pub dtype: String,
}

/// Maps UOp ID → LLVM variable name.
pub struct RenderContext {
    names: HashMap<u64, String>,
    range_values: HashMap<usize, String>,
    counter: usize,
    /// Pending reduce final loads: reduce_id -> (acc_ptr, dtype)
    pending_reduces: HashMap<u64, PendingReduce>,
}

impl RenderContext {
    pub fn new() -> Self {
        Self { names: HashMap::new(), range_values: HashMap::new(), counter: 0, pending_reduces: HashMap::new() }
    }

    /// Get or create variable name for UOp.
    ///
    /// For constants, returns literal value.
    /// For definitions, returns argument name.
    /// For other ops, returns a generated variable name.
    pub fn name(&mut self, uop: &Arc<UOp>) -> String {
        if let Some(name) = self.names.get(&uop.id) {
            return name.clone();
        }

        let name = match uop.op() {
            Op::Const(cv) => lconst(&cv.0, &uop.dtype()),
            Op::VConst { values } => self.render_vconst(values, uop),
            Op::DefineGlobal(id) => format!("%data{id}"),
            Op::DefineLocal(id) => format!("%local{id}"),
            Op::DefineVar { name, .. } => format!("%{name}"),
            Op::DefineReg { .. } => {
                let n = format!("%reg{}", self.counter);
                self.counter += 1;
                n
            }
            Op::Range { axis_id, .. } => {
                // Range variables are named by axis_id
                format!("%r{}", axis_id.value())
            }
            _ => {
                let n = format!("%v{}", self.counter);
                self.counter += 1;
                n
            }
        };

        self.names.insert(uop.id, name.clone());
        name
    }

    /// Render a vector constant.
    fn render_vconst(&self, values: &[ConstValue], uop: &Arc<UOp>) -> String {
        let scalar_type = ldt(&morok_dtype::DType::Scalar(uop.dtype().base()));

        // Format as LLVM vector constant: <type val, type val, ...>
        let elements: Vec<String> = values
            .iter()
            .map(|v| {
                let val = lconst(v, &uop.dtype());
                format!("{scalar_type} {val}")
            })
            .collect();

        format!("<{}>", elements.join(", "))
    }

    /// Get existing name (panics if not found).
    pub fn get(&self, uop: &Arc<UOp>) -> &str {
        self.names
            .get(&uop.id)
            .map(|s| s.as_str())
            .unwrap_or_else(|| panic!("UOp {} ({:?}) not in context", uop.id, uop.op()))
    }

    /// Try to get existing name.
    pub fn try_get(&self, uop: &Arc<UOp>) -> Option<&str> {
        self.names.get(&uop.id).map(|s| s.as_str())
    }

    /// Check if a UOp is already registered.
    pub fn contains(&self, id: u64) -> bool {
        self.names.contains_key(&id)
    }

    /// Alias one ID to another's name.
    pub fn alias(&mut self, id: u64, name: String) {
        self.names.insert(id, name);
    }

    /// Pre-register a name for a UOp ID.
    pub fn register(&mut self, id: u64, name: String) {
        self.names.insert(id, name);
    }

    /// Get current variable counter.
    pub fn counter(&self) -> usize {
        self.counter
    }

    /// Register a range value by axis_id.
    pub fn register_range(&mut self, axis_id: usize, name: String) {
        self.range_values.insert(axis_id, name);
    }

    /// Get a range value by axis_id.
    pub fn get_range(&self, axis_id: usize) -> Option<&str> {
        self.range_values.get(&axis_id).map(|s| s.as_str())
    }

    /// Register a pending reduce final load.
    pub fn register_reduce_pending(&mut self, reduce_id: u64, acc_ptr: String, dtype: String) {
        self.pending_reduces.insert(reduce_id, PendingReduce { acc_ptr, dtype });
    }

    /// Take all pending reduces (empties map).
    pub fn take_pending_reduces(&mut self) -> HashMap<u64, PendingReduce> {
        std::mem::take(&mut self.pending_reduces)
    }

    /// Check if there are pending reduces.
    pub fn has_pending_reduces(&self) -> bool {
        !self.pending_reduces.is_empty()
    }
}

impl Default for RenderContext {
    fn default() -> Self {
        Self::new()
    }
}
