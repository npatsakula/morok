//! Cheap, symbolic cost estimates over a kernel AST.
//!
//! These walk the UOp graph (no execution) to approximate a kernel's compute
//! work. Used by the BEAM `least_compute_ops` bloat filter and by the runtime
//! profiler's roofline (GFLOP/s) column.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::op::Op;
use crate::ops;
use crate::uop::UOp;

/// Symbolic estimate of compute ops in a kernel.
///
/// Each ALU/Ternary/Reduce/WMMA node contributes `prod(enclosing-RANGE sizes)`
/// flops. Symbolic RANGE ends resolve to the midpoint of their `vmin`/`vmax`
/// bounds (matching the `(vmax+vmin)/2` choice in the BEAM timing path), so
/// dynamic-shape kernels participate in the `least_compute_ops*1000` bloat
/// filter.
pub fn compute_ops_estimate(uop: &Arc<UOp>) -> u64 {
    let topo = uop.toposort();
    let pos: FxHashMap<u64, usize> = topo.iter().enumerate().map(|(i, node)| (node.id, i)).collect();

    // One bit per loop-bound node — RANGE for ordinary loops, SPECIAL for
    // hardware-provided indices — plus its iteration count.
    let mut range_sizes: Vec<u64> = Vec::new();
    let mut range_bit: FxHashMap<u64, usize> = FxHashMap::default();
    for node in &topo {
        if let Op::Range(ops::Range { end, .. }) | Op::Special(ops::Special { end, .. }) = node.op() {
            range_bit.insert(node.id, range_sizes.len());
            range_sizes.push(range_size_estimate(end));
        }
    }

    // Bottom-up fold over the toposort: `enclosing[node]` is the union of its
    // children's enclosing ranges plus itself. Structurally the same
    // information tinygrad tracks with its `mult_stack` discipline.
    let words = range_sizes.len().div_ceil(64);
    let mut enclosing = vec![0u64; topo.len() * words];
    let mut flops: u64 = 0;
    for (i, node) in topo.iter().enumerate() {
        let (done, rest) = enclosing.split_at_mut(i * words);
        let mask = &mut rest[..words];
        node.op().map_child(|child| {
            let j = pos[&child.id]; // children precede parents in a toposort
            for (w, cw) in mask.iter_mut().zip(&done[j * words..(j + 1) * words]) {
                *w |= cw;
            }
        });
        if let Some(&bit) = range_bit.get(&node.id) {
            mask[bit / 64] |= 1 << (bit % 64);
        }

        // Each ALU/Reduce/WMMA accumulates `prod(enclosing range sizes)`.
        if matches!(node.op(), Op::Binary(..) | Op::Unary(..) | Op::Ternary(..) | Op::Reduce(..) | Op::Wmma(..)) {
            let mut weight: u64 = 1;
            for (w, word) in mask.iter().enumerate() {
                let mut bits = *word;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    weight = weight.saturating_mul(range_sizes[w * 64 + bit]);
                    bits &= bits - 1;
                }
            }
            flops = flops.saturating_add(weight);
        }
    }
    flops
}

/// Estimate a RANGE end's iteration count.
///
/// Concrete `Const(Int)` ends use the value directly; everything else falls
/// back to the midpoint of the `end` UOp's symbolic `vmin`/`vmax` bounds, so
/// dynamic-shape ranges still contribute a representative number of flops.
fn range_size_estimate(end: &Arc<UOp>) -> u64 {
    if let Op::Const(cv) = end.op()
        && let Some(v) = cv.0.try_int()
    {
        return (v.max(1)) as u64;
    }
    let vmin = end.vmin().try_int().unwrap_or(1);
    let vmax = end.vmax().try_int().unwrap_or(vmin);
    (((vmin + vmax) / 2).max(1)) as u64
}

#[cfg(test)]
#[path = "../test/unit/uop/cost.rs"]
mod tests;
