//! Pattern matching for tensor core opportunities.
//!
//! Detects matrix multiplication patterns suitable for tensor core acceleration.

use crate::optimizer::{Scheduler, error::*};
use morok_ir::{Op, ReduceOp, UOp};
use std::rc::Rc;

/// Information about a detected matmul pattern.
#[derive(Debug, Clone)]
pub struct MatmulPattern {
    /// The REDUCE operation being optimized.
    pub reduce_op: Rc<UOp>,
    /// First input operand (M dimension).
    pub in0: Rc<UOp>,
    /// Second input operand (N dimension).
    pub in1: Rc<UOp>,
    /// Ranges unique to in0 (M dimensions).
    pub in0_ranges: Vec<Rc<UOp>>,
    /// Ranges unique to in1 (N dimensions).
    pub in1_ranges: Vec<Rc<UOp>>,
    /// Reduction ranges (K dimensions).
    pub red_ranges: Vec<Rc<UOp>>,
    /// All possible axis choices (N, M, K) combinations.
    pub axis_choices: Vec<(Rc<UOp>, Rc<UOp>, Rc<UOp>)>,
}

/// Detect matmul pattern in the scheduler's AST.
///
/// Looks for: REDUCE(ADD, MUL(in0, in1), ...reduce_ranges)
///
/// # Returns
///
/// - `Ok(Some(pattern))` if matmul pattern found
/// - `Ok(None)` if no pattern (not an error, just not applicable)
/// - `Err(_)` on validation errors
pub fn detect_matmul(scheduler: &Scheduler) -> Result<Option<MatmulPattern>, OptError> {
    // 1. Find first REDUCE operation
    let reduce_op = match scheduler.reduceop() {
        Some(op) => op,
        None => return Ok(None), // No reduce, not a matmul
    };

    // 2. Verify it's a sum reduction
    if let Op::Reduce { reduce_op: reduce_type, ranges: _, src } = reduce_op.op() {
        if *reduce_type != ReduceOp::Add {
            return Ok(None); // Only ADD reductions can be matmuls
        }

        // 3. Extract MUL operation (possibly under CAST)
        let mul = if let Op::Cast { src: cast_src, .. } = src.op() { cast_src.clone() } else { src.clone() };

        // 4. Verify it's a multiplication
        if let Op::Binary(morok_ir::BinaryOp::Mul, a, b) = mul.op() {
            let in0 = a.clone();
            let in1 = b.clone();

            // 5. Extract ranges for each input
            let in0_all_ranges = get_ranges(&in0);
            let in1_all_ranges = get_ranges(&in1);

            // 6. Get reduction ranges
            let red_ranges = if let Op::Reduce { ranges, .. } = reduce_op.op() {
                ranges.iter().cloned().collect::<Vec<_>>()
            } else {
                vec![]
            };

            // 7. Find unique ranges (M and N dimensions)
            let mut in0_ranges = Vec::with_capacity(in0_all_ranges.len());
            for r in &in0_all_ranges {
                if !in1_all_ranges.iter().any(|r2| Rc::ptr_eq(r, r2)) {
                    in0_ranges.push(r.clone());
                }
            }

            let mut in1_ranges = Vec::with_capacity(in1_all_ranges.len());
            for r in &in1_all_ranges {
                if !in0_all_ranges.iter().any(|r2| Rc::ptr_eq(r, r2)) {
                    in1_ranges.push(r.clone());
                }
            }

            // 8. Sort by axis_id descending (canonical ordering)
            in0_ranges.sort_by_key(|r| std::cmp::Reverse(get_axis_id(r)));
            in1_ranges.sort_by_key(|r| std::cmp::Reverse(get_axis_id(r)));
            let mut red_ranges_sorted = red_ranges.clone();
            red_ranges_sorted.sort_by_key(|r| std::cmp::Reverse(get_axis_id(r)));

            // 9. Generate all axis choices (Cartesian product)
            // Note: Tinygrad swaps in1 and in0 in the product
            let capacity = in1_ranges.len() * in0_ranges.len() * red_ranges_sorted.len();
            let mut axis_choices = Vec::with_capacity(capacity);
            for n_range in &in1_ranges {
                for m_range in &in0_ranges {
                    for k_range in &red_ranges_sorted {
                        axis_choices.push((n_range.clone(), m_range.clone(), k_range.clone()));
                    }
                }
            }

            if axis_choices.is_empty() {
                return Ok(None); // No valid axis combinations
            }

            Ok(Some(MatmulPattern {
                reduce_op,
                in0,
                in1,
                in0_ranges,
                in1_ranges,
                red_ranges: red_ranges_sorted,
                axis_choices,
            }))
        } else {
            Ok(None) // Not a MUL operation
        }
    } else {
        Ok(None) // Not a REDUCE operation
    }
}

/// Get all RANGE UOps used by the given UOp (via backward slice).
fn get_ranges(uop: &Rc<UOp>) -> Vec<Rc<UOp>> {
    uop.backward_slice().into_iter().filter(|node| matches!(node.op(), Op::Range { .. })).collect()
}

/// Extract axis_id from a RANGE UOp.
fn get_axis_id(range: &Rc<UOp>) -> usize {
    if let Op::Range { axis_id, .. } = range.op() {
        axis_id.value()
    } else {
        0 // Fallback (shouldn't happen)
    }
}
