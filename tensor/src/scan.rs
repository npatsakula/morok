//! Schedule-level scan loops: one compiled step kernel, launched `T` times.
//!
//! A [`ScanVar`] is an ordinary symbolic [`Variable`](crate::Variable) that a
//! builder uses *only* as an additive index offset (`gx.narrow(0, t, 1)`), so
//! every axis extent it touches stays a constant and the optimizer keeps
//! upcasting and vectorizing the step. After rangeify, [`wrap_scan_loops`]
//! turns the kernels that read the variable into the body of the schedule-level
//! loop `RANGE → CALL … CALL → END(CALL, [RANGE])`, which `create_pre_schedule`
//! replays once per slot with the counter bound into the kernel's arguments.
//!
//! The rewrite is deliberately post-rangeify: rangeify is index-functional and
//! would never synthesize a recurrence, but it happily indexes a symbolic
//! offset, so the step is compiled exactly once and re-launched.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use smallvec::SmallVec;
use svod_ir::{AxisId, AxisType, Op, SInt, UOp, UOpKey, ops};

use crate::Variable;

/// Name prefix that marks a variable as a loop counter rather than a runtime
/// input. Everything the rewrite needs is in the variable itself, so there is
/// no side table to keep in sync.
const SCAN_PREFIX: &str = "__scan";

static SCAN_SEQ: AtomicUsize = AtomicUsize::new(0);

/// A loop counter for a schedule-level scan.
///
/// The name is unique per instance: two scans sharing one name would be merged
/// into a single loop, which is only correct when their bodies are mutually
/// independent.
#[derive(Clone, Debug)]
pub(crate) struct ScanVar {
    var: Variable,
}

impl ScanVar {
    pub(crate) fn new(trips: usize) -> Self {
        assert!(trips > 0, "scan trip count must be positive");
        let name = format!("{SCAN_PREFIX}{}", SCAN_SEQ.fetch_add(1, Ordering::Relaxed));
        Self { var: Variable::new(&name, 0, trips as i64 - 1) }
    }

    /// The counter as an index offset.
    pub(crate) fn index(&self) -> SInt {
        self.var.as_sint()
    }
}

/// The `(name, node, trip count)` of the scan variable a kernel body reads.
///
/// The trip count is the variable's own upper bound — a counter over `[0, T)`
/// is declared as `vmin = 0, vmax = T - 1`.
fn body_scan_var(body: &Arc<UOp>) -> Option<(String, Arc<UOp>, i64)> {
    body.toposort_call_aware(false).into_iter().find_map(|node| {
        let Op::Param(ops::Param { arg, .. }) = node.op() else { return None };
        let name = arg.name.as_deref()?;
        if arg.addrspace.is_some() || !name.starts_with(SCAN_PREFIX) {
            return None;
        }
        let (_, vmax) = arg.vmin_vmax.as_ref()?;
        Some((name.to_string(), node.clone(), vmax.0.try_int()? + 1))
    })
}

/// Wrap every kernel that reads a scan variable in a schedule-level loop.
///
/// Returns the graph unchanged when no kernel reads one, which is every graph
/// that has no recurrence in it.
pub(crate) fn wrap_scan_loops(root: Arc<UOp>) -> Arc<UOp> {
    // `toposort_call_aware` is topological, so the last member of a group is
    // the one every other member precedes — the only sound place for the END.
    let mut groups: Vec<(Arc<UOp>, i64, Vec<Arc<UOp>>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for node in root.toposort_call_aware(false) {
        let Op::Call(ops::Call { body, .. }) = node.op() else { continue };
        let Some((name, var, trips)) = body_scan_var(body) else { continue };
        match index.get(&name) {
            Some(&i) => groups[i].2.push(node.clone()),
            None => {
                index.insert(name, groups.len());
                groups.push((var, trips, vec![node.clone()]));
            }
        }
    }
    if groups.is_empty() {
        return root;
    }

    let mut substitutions: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    for (axis, (var, trips, calls)) in groups.iter().enumerate() {
        let range = UOp::range_axis(UOp::index_const(*trips), AxisId::Renumbered(axis), AxisType::Loop);
        let bind = var.bind(range.clone());

        let last = calls.len() - 1;
        for (i, call) in calls.iter().enumerate() {
            let Op::Call(ops::Call { body, args, info }) = call.op() else { unreachable!("filtered above") };
            let mut args: SmallVec<[Arc<UOp>; 4]> = args.clone();
            args.push(bind.clone());
            let looped = body.call(args, (**info).clone());
            // The END closes the loop after the body's last kernel, and takes
            // the original CALL's place in the graph so it stays reachable
            // without becoming a SINK output.
            let replacement = if i == last { looped.end(SmallVec::from_vec(vec![range.clone()])) } else { looped };
            substitutions.insert(UOpKey(call.clone()), replacement);
        }
    }
    root.substitute(&substitutions)
}
