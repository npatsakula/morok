//! Target-neutral instruction-selection interfaces.

use std::collections::HashMap;
use std::sync::Arc;

use svod_ir::ops;
use svod_ir::{Op, UOp, UOpKey};

/// Context used by Tinygrad's pre-isel pass: fresh temporaries count downward
/// from -1 so they cannot collide with normal non-negative identifiers.
#[derive(Debug, Clone)]
pub struct PreIselContext {
    next: i64,
}

impl Default for PreIselContext {
    fn default() -> Self {
        Self { next: -1 }
    }
}

impl PreIselContext {
    pub fn next_temp(&mut self) -> i64 {
        let value = self.next;
        self.next -= 1;
        value
    }
}

/// Instruction-selection graph facts, corresponding to Tinygrad's
/// `IselContext`: consumers, deterministic function arguments, and vreg IDs.
#[derive(Debug)]
pub struct IselContext {
    uses: HashMap<UOpKey, Vec<Arc<UOp>>>,
    pub func_args: Vec<Arc<UOp>>,
    next_vreg: usize,
}

impl IselContext {
    pub fn new(sink: &Arc<UOp>) -> Self {
        let topo = sink.toposort();
        let mut uses: HashMap<UOpKey, Vec<Arc<UOp>>> = topo.iter().cloned().map(|u| (UOpKey(u), Vec::new())).collect();
        for user in &topo {
            for source in user.op().sources() {
                if let Some(consumers) = uses.get_mut(&UOpKey(source)) {
                    consumers.push(user.clone());
                }
            }
        }

        let mut func_args: Vec<_> =
            topo.into_iter().filter(|u| matches!(u.op(), Op::Param(..) | Op::Special(..))).collect();
        func_args.sort_by_key(|u| match u.op() {
            Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_some() => (0u8, arg.slot as u64, String::new()),
            Op::Param(..) => (1, u.content_hash, String::new()),
            Op::Special(ops::Special { name, .. }) => (2, 0, name.clone()),
            _ => unreachable!(),
        });

        Self { uses, func_args, next_vreg: 0 }
    }

    pub fn uses(&self, uop: &Arc<UOp>) -> &[Arc<UOp>] {
        self.uses.get(&UOpKey(uop.clone())).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn next_vreg(&mut self) -> usize {
        let value = self.next_vreg;
        self.next_vreg += 1;
        value
    }
}
