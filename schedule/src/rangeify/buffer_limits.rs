//! Buffer limit enforcement for device-specific constraints.
//!
//! Forces bufferization when buffer count exceeds device limits:
//! - Metal: 31 buffers, WebGPU: 8 buffers, CPU/CUDA: no limit

use std::collections::HashSet;
use std::rc::Rc;

use morok_device::DeviceSpec;
use morok_ir::{AddrSpace, BufferizeOpts, Op, UOp, UOpKey};

use super::buffer_cost::collect_accessed_buffers;
use crate::pattern::UPat;
use crate::pattern::matcher::{PatternMatcher, RewriteResult};

/// Extract device specification from a UOp graph (first device found).
#[allow(clippy::mutable_key_type)]
pub fn extract_device_from_graph(root: &Rc<UOp>) -> Option<DeviceSpec> {
    let mut visited = HashSet::new();

    fn visit(uop: &Rc<UOp>, visited: &mut HashSet<UOpKey>) -> Option<DeviceSpec> {
        let key = UOpKey(Rc::clone(uop));
        if !visited.insert(key) {
            return None; // Already visited
        }

        match uop.op() {
            Op::Device(spec) => return Some(spec.clone()),
            Op::Buffer { device, .. } => {
                // Device is a UOp that should be Op::Device
                if let Op::Device(spec) = device.op() {
                    return Some(spec.clone());
                }
            }
            Op::Bufferize { opts, .. } => {
                // BufferizeOpts may contain optional device
                if let Some(device_spec) = &opts.device {
                    return Some(device_spec.clone());
                }
            }
            _ => {}
        }

        // Recursively visit children
        for child in uop.op().sources() {
            if let Some(device) = visit(&child, visited) {
                return Some(device);
            }
        }

        None
    }

    visit(root, &mut visited)
}

/// Check if operation is elementwise (Binary or Ternary).
pub fn is_elementwise(uop: &Rc<UOp>) -> bool {
    matches!(uop.op(), Op::Binary(..) | Op::Ternary(..))
}

/// Create pattern matcher for buffer limit enforcement.
/// Forces bufferization of elementwise sources when buffer count > max_buffers - 1.
pub fn buffer_limit_patterns(max_buffers: usize) -> PatternMatcher {
    use crate::pattern::matcher::RewriteFn;
    use crate::pattern::{BindingStore, BindingStoreExt, VarIntern};

    let mut patterns = vec![];

    // Pattern: Binary/Ternary operations with too many buffers → Force bufferize elementwise sources
    let limit = max_buffers; // Copy for closure capture
    patterns.push((
        UPat::var("op"),
        Box::new(move |bindings: &BindingStore, intern: &VarIntern, _ctx: &mut ()| {
            let Some(op) = intern.get_index("op").and_then(|i| bindings.get_by_index(i)) else {
                return RewriteResult::NoMatch;
            };

            // Only process Binary and Ternary operations (elementwise)
            let sources = match op.op() {
                Op::Binary(_, left, right) => vec![left.clone(), right.clone()],
                Op::Ternary(_, cond, true_val, false_val) => {
                    vec![cond.clone(), true_val.clone(), false_val.clone()]
                }
                _ => return RewriteResult::NoMatch,
            };

            // Count buffers accessed by all sources
            let mut all_buffers = Vec::new();
            for src in &sources {
                all_buffers.extend(collect_accessed_buffers(src));
            }

            // Deduplicate
            #[allow(clippy::mutable_key_type)]
            let mut seen = HashSet::new();
            all_buffers.retain(|b| seen.insert(UOpKey(Rc::clone(b))));

            // Check if exceeds limit (-1 for output buffer)
            if all_buffers.len() > limit.saturating_sub(1) {
                // Force bufferize elementwise sources
                let mut new_sources = Vec::new();
                let mut any_changed = false;

                for src in &sources {
                    let new_src = if is_elementwise(src) { force_bufferize(src) } else { Rc::clone(src) };

                    if !Rc::ptr_eq(&new_src, src) {
                        any_changed = true;
                    }
                    new_sources.push(new_src);
                }

                // If any source changed, reconstruct the operation
                if any_changed {
                    let dtype = op.dtype();
                    let rewritten = match op.op() {
                        Op::Binary(bin_op, _, _) => {
                            UOp::new(Op::Binary(*bin_op, new_sources[0].clone(), new_sources[1].clone()), dtype)
                        }
                        Op::Ternary(tern_op, _, _, _) => UOp::new(
                            Op::Ternary(
                                *tern_op,
                                new_sources[0].clone(),
                                new_sources[1].clone(),
                                new_sources[2].clone(),
                            ),
                            dtype,
                        ),
                        _ => unreachable!(),
                    };
                    return RewriteResult::Rewritten(rewritten);
                }
            }

            RewriteResult::NoMatch
        }) as RewriteFn<()>,
    ));

    PatternMatcher::new(patterns)
}

/// Force bufferization of a computation to GLOBAL memory.
fn force_bufferize(src: &Rc<UOp>) -> Rc<UOp> {
    // Collect all ranges from the source computation
    let ranges = collect_ranges(src);

    if ranges.is_empty() {
        // No ranges to bufferize, return original
        return Rc::clone(src);
    }

    // Create BUFFERIZE with GLOBAL address space
    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Global };
    let bufferized = UOp::bufferize(Rc::clone(src), ranges.clone(), opts);

    // Wrap in INDEX to make it usable
    UOp::index(bufferized, ranges).unwrap_or_else(|_| Rc::clone(src))
}

/// Collect all RANGE operations from a computation tree.
#[allow(clippy::mutable_key_type)]
fn collect_ranges(src: &Rc<UOp>) -> Vec<Rc<UOp>> {
    let mut ranges = Vec::new();
    let mut visited = HashSet::new();

    fn visit(uop: &Rc<UOp>, ranges: &mut Vec<Rc<UOp>>, visited: &mut HashSet<UOpKey>) {
        let key = UOpKey(Rc::clone(uop));
        if !visited.insert(key) {
            return; // Already visited
        }

        if matches!(uop.op(), Op::Range { .. }) {
            ranges.push(Rc::clone(uop));
        }

        for child in uop.op().sources() {
            visit(&child, ranges, visited);
        }
    }

    visit(src, &mut ranges, &mut visited);

    // Deduplicate while preserving order
    let mut seen = HashSet::new();
    ranges.retain(|r| seen.insert(UOpKey(Rc::clone(r))));

    ranges
}
