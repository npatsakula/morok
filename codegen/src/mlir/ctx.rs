//! Render context for MLIR code generation.
//!
//! Maps UOp IDs to MLIR Values and manages scf loop/if region tracking.

use std::collections::HashMap;

use melior::ir::{BlockRef, Region, Type, Value};

use super::amx::AmxLoopState;

/// Info for an in-progress `scf.for` loop being built incrementally.
pub struct ScfLoopInfo<'c, 'a> {
    /// Block where the `scf.for` op will be appended once the loop closes.
    pub parent_block: BlockRef<'c, 'a>,
    /// The owned Region (body) — transferred to `scf.for` at End time.
    pub region: Region<'c>,
    /// UOp ID of the Range node (used to register scf.for results).
    pub range_id: u64,
    /// Axis ID from the Range op.
    pub axis_id: usize,
    /// Original MLIR type of the loop IV (e.g. i32).
    pub range_type: Type<'c>,
    /// Lower bound in `index` type.
    pub lb: Value<'c, 'a>,
    /// Upper bound in `index` type.
    pub ub: Value<'c, 'a>,
    /// Step value in `index` type.
    pub step: Value<'c, 'a>,
    /// Initial values for iter_args (reduce identity constants).
    pub init_values: Vec<Value<'c, 'a>>,
    /// Result types of the `scf.for` (one per reduce).
    pub result_types: Vec<Type<'c>>,
    /// UOp IDs of the reduces associated with this loop, in order.
    pub reduce_ids: Vec<u64>,
    /// Current accumulator values to yield at `scf.yield` time.
    /// Initialized from body block arguments; updated by Reduce ops.
    pub yield_values: Vec<Value<'c, 'a>>,
}

/// Info for an in-progress `scf.if` being built incrementally.
pub struct ScfIfInfo<'c, 'a> {
    /// Block where the `scf.if` op will be appended.
    pub parent_block: BlockRef<'c, 'a>,
    /// The condition value for the `scf.if`.
    pub condition: Value<'c, 'a>,
    /// The owned then-Region — transferred to `scf.if` at EndIf time.
    pub then_region: Region<'c>,
}

/// Maps UOp ID -> MLIR Value and manages structured control flow regions.
pub struct RenderContext<'c, 'a> {
    values: HashMap<u64, Value<'c, 'a>>,
    /// Stack of in-progress `scf.for` loops (innermost on top).
    scf_loop_stack: Vec<ScfLoopInfo<'c, 'a>>,
    /// Stack of in-progress `scf.if` regions, keyed by If node ID.
    scf_if_stack: Vec<(u64, ScfIfInfo<'c, 'a>)>,
    /// The current insertion block reference.
    current_block: BlockRef<'c, 'a>,
    /// The function entry block (for hoisting allocas out of loops).
    entry_block: BlockRef<'c, 'a>,
    /// AMX loop state when inside a hoisted WMMA reduce loop.
    amx_loop_state: Option<AmxLoopState<'c>>,
    /// Whether AMX SET has been emitted for this kernel.
    amx_set_emitted: bool,
}

impl<'c, 'a> RenderContext<'c, 'a> {
    pub fn new(entry_block: BlockRef<'c, 'a>) -> Self {
        Self {
            values: HashMap::new(),
            scf_loop_stack: Vec::new(),
            scf_if_stack: Vec::new(),
            current_block: entry_block,
            entry_block,
            amx_loop_state: None,
            amx_set_emitted: false,
        }
    }

    pub fn entry_block(&self) -> BlockRef<'c, 'a> {
        self.entry_block
    }

    pub fn current_block(&self) -> BlockRef<'c, 'a> {
        self.current_block
    }

    pub fn set_current_block(&mut self, block: BlockRef<'c, 'a>) {
        self.current_block = block;
    }

    pub fn register(&mut self, id: u64, value: Value<'c, 'a>) {
        self.values.insert(id, value);
    }

    pub fn get(&self, id: u64) -> Value<'c, 'a> {
        self.values.get(&id).copied().unwrap_or_else(|| panic!("UOp {} not in MLIR context", id))
    }

    pub fn try_get(&self, id: u64) -> Option<Value<'c, 'a>> {
        self.values.get(&id).copied()
    }

    pub fn contains(&self, id: u64) -> bool {
        self.values.contains_key(&id)
    }

    // -- scf.for loop management --

    pub fn push_scf_loop(&mut self, info: ScfLoopInfo<'c, 'a>) {
        self.scf_loop_stack.push(info);
    }

    pub fn pop_scf_loop(&mut self) -> ScfLoopInfo<'c, 'a> {
        self.scf_loop_stack.pop().expect("scf loop stack underflow")
    }

    /// Find the loop on the stack that owns a given reduce, and update its yield value.
    /// Returns the index of the reduce within that loop's iter_args.
    pub fn update_reduce_yield(&mut self, reduce_id: u64, new_value: Value<'c, 'a>) {
        for loop_info in self.scf_loop_stack.iter_mut().rev() {
            if let Some(idx) = loop_info.reduce_ids.iter().position(|&id| id == reduce_id) {
                loop_info.yield_values[idx] = new_value;
                return;
            }
        }
        panic!("reduce {} not found in any scf loop on stack", reduce_id);
    }

    // -- scf.if management --

    pub fn push_scf_if(&mut self, if_id: u64, info: ScfIfInfo<'c, 'a>) {
        self.scf_if_stack.push((if_id, info));
    }

    pub fn pop_scf_if(&mut self, if_id: u64) -> ScfIfInfo<'c, 'a> {
        let idx = self
            .scf_if_stack
            .iter()
            .rposition(|(id, _)| *id == if_id)
            .unwrap_or_else(|| panic!("scf.if {} not found on stack", if_id));
        self.scf_if_stack.remove(idx).1
    }

    // -- AMX loop state management --

    pub fn set_amx_loop_state(&mut self, state: AmxLoopState<'c>) {
        self.amx_loop_state = Some(state);
    }

    pub fn amx_loop_state(&self) -> Option<&AmxLoopState<'c>> {
        self.amx_loop_state.as_ref()
    }

    pub fn take_amx_loop_state(&mut self) -> Option<AmxLoopState<'c>> {
        self.amx_loop_state.take()
    }

    // -- AMX kernel-wide state management --

    pub fn amx_set_emitted(&self) -> bool {
        self.amx_set_emitted
    }

    pub fn mark_amx_set_emitted(&mut self) {
        self.amx_set_emitted = true;
    }
}
