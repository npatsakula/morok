//! Pre-expander pass for UNROLL/UPCAST range handling.
//!
//! Transforms the kernel AST before codegen to properly handle UNROLL optimization.
//!
//! # Problem
//!
//! When `shift_to` applies UNROLL optimization, it substitutes Range ops
//! with arithmetic expressions like `replaced_rng * amount + new_rng`.
//! This creates two issues:
//! 1. REDUCE.ranges contains arithmetic expressions instead of Range ops
//! 2. Range(Unroll) ops in expressions try to create loops without ENDs
//!
//! # Solution (Tinygrad-aligned)
//!
//! Two-phase expansion:
//! 1. **Pre-expansion**: Convert Range(Unroll/Upcast) → UNROLL op with constant vector
//! 2. **Main expansion**: Propagate vectorization through operations using UNROLL
//!
//! The key insight from Tinygrad's expander.py:
//! - UNROLL(src=VCONST([0,1,...,N-1]), axes=[(axis_id, N)]) holds all iterations as a vector
//! - Operations using UNROLL get replicated/vectorized via do_expand
//! - CONTRACT collapses vectorized results back to scalar for REDUCE/STORE
//!
//! # Implementation
//!
//! - `convert_range_to_unroll`: Range(Unroll) → UNROLL(VCONST([0..N]))
//! - `fix_reduce_unroll`: Extract ranges from arithmetic expressions in REDUCE
//! - `do_expand`: Replicate operations that use UNROLL inputs

use std::sync::Arc;

use morok_ir::{AxisType, BinaryOp, ConstValue, Op, PatternMatcher, UOp};
use smallvec::SmallVec;

/// Run pre-expansion pass on kernel AST.
///
/// Call this AFTER optimization but BEFORE codegen.
/// Two phases:
/// 1. Convert Range(Unroll/Upcast) → UNROLL ops with constant vectors
/// 2. Fix REDUCE operations with arithmetic expressions in ranges
/// 3. Expand operations that use UNROLL inputs
///
/// Uses bottom-up traversal to ensure all nodes are visited, including
/// REDUCE nodes nested inside KERNEL/STORE structures.
pub fn pre_expand(ast: &Arc<UOp>) -> Arc<UOp> {
    use crate::rewrite::graph_rewrite_bottom_up;

    // Phase 1: Convert Range(Unroll/Upcast) to UNROLL ops
    let phase1 = phase1_range_to_unroll();
    let ast = graph_rewrite_bottom_up(&phase1, ast.clone(), &mut ());

    // Phase 2: Fix REDUCE with non-Range entries and expand operations
    let phase2 = phase2_expand();
    graph_rewrite_bottom_up(&phase2, ast, &mut ())
}

/// Phase 1: Convert Range(Unroll/Upcast) → UNROLL ops with constant vectors.
///
/// Tinygrad pattern (expander.py:143-147):
/// ```python
/// (UPat(Ops.RANGE, name="r"),
///  lambda r: UOp(Ops.UNROLL, r.dtype, (UOp.const(r.dtype.vec(s), tuple(range(s))),), ((r.arg[0],s),))
///  if r.arg[1] in {AxisType.UNROLL, AxisType.UPCAST} else None)
/// ```
fn phase1_range_to_unroll() -> PatternMatcher {
    crate::patterns! {
        // Convert Range(Unroll/Upcast) to UNROLL op with constant vector
        range if matches!(range.op(), Op::Range { axis_type: AxisType::Unroll | AxisType::Upcast, .. }) => {
            convert_range_to_unroll(range)
        },
    }
}

/// Phase 2: Fix REDUCE and expand operations using UNROLL.
fn phase2_expand() -> PatternMatcher {
    crate::patterns! {
        // Fix REDUCE with non-Range entries in ranges
        reduce if matches!(reduce.op(), Op::Reduce { .. }) => fix_reduce_unroll(reduce),

        // Expand binary operations that have UNROLL inputs
        // Skip if expression contains Range(Reduce) - those must stay scalar for loop counter
        binary if matches!(binary.op(), Op::Binary(..))
            && has_unroll_input(binary)
            && !contains_reduce_range(binary) => do_expand(binary),

        // Expand unary operations that have UNROLL inputs
        unary if matches!(unary.op(), Op::Unary(..))
            && has_unroll_input(unary)
            && !contains_reduce_range(unary) => do_expand(unary),

        // Expand cast operations that have UNROLL inputs
        cast if matches!(cast.op(), Op::Cast { .. })
            && has_unroll_input(cast)
            && !contains_reduce_range(cast) => do_expand(cast),
    }
}

/// Convert Range(Unroll/Upcast) to UNROLL op with constant vector [0, 1, ..., N-1].
fn convert_range_to_unroll(range: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Range { end, axis_id, axis_type } = range.op() else {
        return None;
    };

    // Only convert Unroll/Upcast axis types
    if !matches!(axis_type, AxisType::Unroll | AxisType::Upcast) {
        return None;
    }

    // Extract constant size
    let size = extract_const_size(end)?;
    if size == 0 {
        return None;
    }

    // Create constant vector [0, 1, 2, ..., size-1]
    let values: Vec<ConstValue> = (0..size as i64).map(ConstValue::Int).collect();
    let vconst = UOp::vconst(values);

    // Wrap in UNROLL op with axis metadata
    Some(UOp::unroll(vconst, vec![(axis_id.value(), size)]))
}

/// Check if any input to this operation is an UNROLL op.
fn has_unroll_input(uop: &Arc<UOp>) -> bool {
    uop.op().sources().iter().any(|src| matches!(src.op(), Op::Unroll { .. }))
}

/// Check if a UOp or any of its dependencies contains a Range(Reduce) op.
/// We must NOT expand operations that depend on reduce loop counters.
fn contains_reduce_range(uop: &Arc<UOp>) -> bool {
    if matches!(uop.op(), Op::Range { axis_type: AxisType::Reduce, .. }) {
        return true;
    }
    uop.op().sources().iter().any(|src| contains_reduce_range(src))
}

/// Expand an operation that has UNROLL inputs.
///
/// Tinygrad's do_expand (expander.py:22-65):
/// 1. Collect all UNROLL inputs and their axes
/// 2. For each source:
///    - If UNROLL with matching axes: unwrap to get inner value
///    - If UNROLL with different axes: use GEP to remap
///    - If not UNROLL: broadcast/replicate to match expansion size
/// 3. Create expanded operation with vectorized dtype
/// 4. Wrap result in UNROLL
fn do_expand(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let sources = uop.op().sources();

    // Collect UNROLL sources
    let unroll_sources: Vec<_> = sources
        .iter()
        .filter(|s| matches!(s.op(), Op::Unroll { .. }))
        .collect();

    if unroll_sources.is_empty() {
        return None;
    }

    // Get expansion info from first UNROLL (simplified: assume all have same axes)
    let first_unroll = unroll_sources[0];
    let Op::Unroll { src: inner_src, unroll_axes } = first_unroll.op() else {
        return None;
    };

    let expand_size: usize = unroll_axes.iter().map(|(_, sz)| sz).product();
    if expand_size == 0 {
        return None;
    }

    // Process each source
    let mut new_sources: SmallVec<[Arc<UOp>; 4]> = SmallVec::new();

    for src in sources.iter() {
        if let Op::Unroll { src: inner, unroll_axes: src_axes } = src.op() {
            if src_axes == unroll_axes {
                // Same expansion: unwrap
                new_sources.push(inner.clone());
            } else {
                // Different expansion: for now, just unwrap (simplified)
                // TODO: implement GEP swizzling for mismatched axes
                new_sources.push(inner.clone());
            }
        } else {
            // Non-UNROLL: broadcast by replicating
            // For scalar constants, create a vector of repeated values
            if let Op::Const(cv) = src.op() {
                let values: Vec<ConstValue> = (0..expand_size).map(|_| cv.0.clone()).collect();
                new_sources.push(UOp::vconst(values));
            } else {
                // For other ops, use Vectorize to replicate
                let elements: SmallVec<[Arc<UOp>; 4]> =
                    (0..expand_size).map(|_| src.clone()).collect();
                new_sources.push(UOp::vectorize(elements));
            }
        }
    }

    // Determine output dtype from the inner source (already vectorized)
    let output_dtype = inner_src.dtype();

    // Create the expanded operation with vectorized output
    let new_op = match uop.op() {
        Op::Binary(op, _, _) => {
            if new_sources.len() >= 2 {
                Some(UOp::new(
                    Op::Binary(*op, new_sources[0].clone(), new_sources[1].clone()),
                    output_dtype.clone(),
                ))
            } else {
                None
            }
        }
        Op::Unary(op, _) => {
            if !new_sources.is_empty() {
                Some(UOp::new(Op::Unary(*op, new_sources[0].clone()), output_dtype.clone()))
            } else {
                None
            }
        }
        Op::Cast { dtype, .. } => {
            if !new_sources.is_empty() {
                Some(UOp::cast(new_sources[0].clone(), dtype.clone()))
            } else {
                None
            }
        }
        _ => None,
    };

    // Wrap result in UNROLL to maintain expansion metadata
    new_op.map(|op| UOp::unroll(op, unroll_axes.clone()))
}

/// Fix a REDUCE operation that has arithmetic expressions in its ranges.
///
/// When `shift_to` applies UNROLL, it substitutes Range ops with expressions
/// like `replaced_rng * amount + new_rng`. This function extracts the embedded
/// Range ops and properly restructures the REDUCE.
fn fix_reduce_unroll(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce { src, reduce_op, ranges } = reduce.op() else {
        return None;
    };

    // Check if any range is not a Range op (i.e., is an arithmetic expression)
    let has_non_range = ranges.iter().any(|r| !matches!(r.op(), Op::Range { .. }));
    if !has_non_range {
        return None; // Nothing to fix
    }

    let mut fixed_ranges: SmallVec<[Arc<UOp>; 4]> = SmallVec::new();
    let mut unroll_axes: Vec<(usize, usize)> = Vec::new();

    for range in ranges.iter() {
        match range.op() {
            // Already a Range op - check axis type
            Op::Range { axis_type, axis_id, end } => {
                match axis_type {
                    AxisType::Reduce | AxisType::Loop => {
                        // Keep in REDUCE.ranges
                        fixed_ranges.push(range.clone());
                    }
                    AxisType::Unroll | AxisType::Upcast => {
                        // Move to CONTRACT wrapper
                        if let Some(size) = extract_const_size(end) {
                            unroll_axes.push((axis_id.value(), size));
                        } else {
                            // Can't extract size, keep as is
                            fixed_ranges.push(range.clone());
                        }
                    }
                    _ => {
                        // Other axis types (Global, Local, etc.) - keep as is
                        fixed_ranges.push(range.clone());
                    }
                }
            }

            // Arithmetic expression from shift_to substitution
            // Pattern: ADD(MUL(replaced_rng, Const), new_rng) or
            //          ADD(MUL(new_rng, Const), replaced_rng)
            Op::Binary(BinaryOp::Add, left, right) => {
                if let Some((reduce_range, unroll_info)) = extract_ranges_from_expr(left, right) {
                    // Add reduce range to fixed_ranges
                    fixed_ranges.push(reduce_range);

                    // Add unroll axis info if extracted
                    if let Some((axis_id, size)) = unroll_info {
                        unroll_axes.push((axis_id, size));
                    }
                } else {
                    // Can't extract, keep as is (will likely fail in codegen)
                    fixed_ranges.push(range.clone());
                }
            }

            // Unknown pattern - keep as is
            _ => {
                fixed_ranges.push(range.clone());
            }
        }
    }

    // Build the fixed REDUCE, preserving the original dtype
    let original_dtype = reduce.dtype();
    let fixed_src = if unroll_axes.is_empty() {
        src.clone()
    } else {
        // Wrap source in CONTRACT to document the unrolled axes
        UOp::contract(src.clone(), unroll_axes)
    };

    // Create new REDUCE with fixed ranges but preserve original dtype
    Some(UOp::new(
        Op::Reduce { src: fixed_src, ranges: fixed_ranges, reduce_op: *reduce_op },
        original_dtype.clone(),
    ))
}

/// Extract Reduce and Unroll ranges from a shift_to arithmetic expression.
///
/// shift_to creates expressions like:
/// - top=false: `replaced_rng * amount + new_rng`
/// - top=true:  `new_rng * old_sz + replaced_rng`
///
/// Returns (reduce_range, Option<(axis_id, size)>) for the unroll axis.
fn extract_ranges_from_expr(
    left: &Arc<UOp>,
    right: &Arc<UOp>,
) -> Option<(Arc<UOp>, Option<(usize, usize)>)> {
    // Pattern 1 (top=false): ADD(MUL(replaced_rng, Const), new_rng)
    if let Op::Binary(BinaryOp::Mul, mul_left, _mul_right) = left.op() {
        if let Op::Range { axis_type: AxisType::Reduce, .. } = mul_left.op() {
            // mul_left is the Reduce range
            let unroll_info = extract_unroll_info(right);
            return Some((mul_left.clone(), unroll_info));
        }
    }

    // Pattern 2 (top=true): ADD(MUL(new_rng, Const), replaced_rng)
    if let Op::Range { axis_type: AxisType::Reduce, .. } = right.op() {
        if let Op::Binary(BinaryOp::Mul, mul_left, _) = left.op() {
            let unroll_info = extract_unroll_info(mul_left);
            return Some((right.clone(), unroll_info));
        }
    }

    None
}

/// Extract unroll axis info (axis_id, size) from a Range op or UNROLL op.
///
/// After Phase 1, Range(Unroll) has been converted to UNROLL, so we need
/// to handle both cases.
fn extract_unroll_info(uop: &Arc<UOp>) -> Option<(usize, usize)> {
    // Handle Range(Unroll) case (before Phase 1 conversion)
    if let Op::Range { axis_type: AxisType::Unroll, axis_id, end } = uop.op() {
        if let Some(size) = extract_const_size(end) {
            return Some((axis_id.value(), size));
        }
    }
    // Handle UNROLL op case (after Phase 1 conversion)
    if let Op::Unroll { unroll_axes, .. } = uop.op() {
        if let Some(&(axis_id, size)) = unroll_axes.first() {
            return Some((axis_id, size));
        }
    }
    None
}

/// Extract constant size from a Range's end value.
fn extract_const_size(end: &Arc<UOp>) -> Option<usize> {
    if let Op::Const(cv) = end.op() {
        match cv.0 {
            ConstValue::Int(i) if i > 0 => Some(i as usize),
            ConstValue::UInt(u) => Some(u as usize),
            _ => None,
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_dtype::DType;
    use morok_ir::ReduceOp;

    #[test]
    fn test_pre_expand_passthrough() {
        // A simple REDUCE with proper Range ops should pass through unchanged
        let end = UOp::const_(DType::Index, ConstValue::Int(32));
        let range = UOp::range_axis(end, morok_ir::AxisId::Renumbered(0), AxisType::Reduce);
        let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
        let reduce = UOp::reduce(src, smallvec::smallvec![range.clone()], ReduceOp::Add);

        let result = pre_expand(&reduce);

        // Should be unchanged (though may be a new node due to graph_rewrite)
        if let Op::Reduce { ranges, .. } = result.op() {
            assert_eq!(ranges.len(), 1);
            assert!(matches!(ranges[0].op(), Op::Range { axis_type: AxisType::Reduce, .. }));
        } else {
            panic!("Expected REDUCE op");
        }
    }

    #[test]
    fn test_extract_const_size() {
        let end = UOp::const_(DType::Index, ConstValue::Int(64));
        assert_eq!(extract_const_size(&end), Some(64));
    }
}
