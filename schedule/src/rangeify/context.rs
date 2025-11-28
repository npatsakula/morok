//! Rangeify context for tracking state during transformation.

use morok_ir::{UOp, UOpKey};
use std::collections::HashMap;
use std::rc::Rc;

/// Context for rangeify transformation.
///
/// Tracks state during the rangeify transformation, including:
/// - Mapping from original UOps to their rangeified versions
/// - Counter for generating unique range IDs
#[derive(Default)]
pub struct RangeifyContext {
    /// Maps old UOps to their rangeified versions.
    ///
    /// This allows us to track how each node in the original graph
    /// has been transformed during the rangeify process.
    ///
    /// Uses UOpKey for HashMap keys since Rc<UOp> doesn't implement Hash/Eq.
    pub range_map: HashMap<UOpKey, Rc<UOp>>,

    /// Counter for generating unique range IDs.
    ///
    /// Each RANGE operation needs a unique axis_id to distinguish
    /// different loop dimensions. This counter ensures we never
    /// reuse IDs within a single transformation.
    pub range_counter: usize,
}

impl RangeifyContext {
    /// Create a new empty rangeify context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the next available range ID.
    ///
    /// Increments the internal counter and returns the previous value.
    /// This ensures each range gets a unique ID.
    pub fn next_range_id(&mut self) -> usize {
        let id = self.range_counter;
        self.range_counter += 1;
        id
    }

    /// Record that a UOp has been transformed.
    ///
    /// Maps the original UOp to its rangeified version so we can
    /// track the transformation.
    pub fn record_transform(&mut self, original: Rc<UOp>, rangeified: Rc<UOp>) {
        self.range_map.insert(UOpKey(original), rangeified);
    }

    /// Get the rangeified version of a UOp, if it exists.
    pub fn get_rangeified(&self, original: &Rc<UOp>) -> Option<&Rc<UOp>> {
        self.range_map.get(&UOpKey(original.clone()))
    }
}
