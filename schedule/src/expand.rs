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

use std::collections::HashMap;
use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::{AxisType, BinaryOp, ConstValue, Op, PatternMatcher, UOp};
use smallvec::SmallVec;

// ============================================================================
// Swizzle Helpers (ported from Tinygrad's expander.py:8-20)
// ============================================================================

/// Compute linear index from axis positions (row-major, reverse iteration).
///
/// Based on Tinygrad's `_expand_arg_to_idx` (expander.py:8-13).
fn expand_arg_to_idx(args: &[(usize, usize)], rpk: &HashMap<usize, usize>) -> usize {
    let mut idx = 0;
    let mut mul = 1;
    for &(axis, m) in args.iter().rev() {
        idx += rpk.get(&axis).unwrap_or(&0) * mul;
        mul *= m;
    }
    idx
}

/// Generate all combinations of axis positions.
///
/// Based on Tinygrad's `_choices_from_args` (expander.py:15-16).
/// For args = [(0, 2), (1, 3)], generates:
/// [{0: 0, 1: 0}, {0: 0, 1: 1}, {0: 0, 1: 2}, {0: 1, 1: 0}, ...]
fn choices_from_args(args: &[(usize, usize)]) -> Vec<HashMap<usize, usize>> {
    let mut result = vec![HashMap::new()];
    for &(axis, m) in args {
        result = result
            .into_iter()
            .flat_map(|map| {
                (0..m).map(move |v| {
                    let mut new_map = map.clone();
                    new_map.insert(axis, v);
                    new_map
                })
            })
            .collect();
    }
    result
}

/// Compute swizzle indices for GEP when UNROLL axes don't match.
///
/// Based on Tinygrad's `_swizzle_args` (expander.py:18-20).
/// Maps indices from expansion layout (cargs) to source layout (eargs),
/// zeroing out any axes in exclude_args.
fn swizzle_args(cargs: &[(usize, usize)], eargs: &[(usize, usize)], exclude_args: &[usize]) -> Vec<usize> {
    choices_from_args(cargs)
        .into_iter()
        .map(|rpk| {
            let mut rpk_with_zeros = rpk.clone();
            for &ax in exclude_args {
                rpk_with_zeros.insert(ax, 0);
            }
            expand_arg_to_idx(eargs, &rpk_with_zeros)
        })
        .collect()
}

// ============================================================================
// Range Detection Helper
// ============================================================================

/// Check if a UOp contains any runtime scalar ops (Range or DefineVar).
///
/// Expressions containing Range or DefineVar ops must not be broadcast/vectorized,
/// as they represent runtime scalar values (loop counters, kernel parameters).
fn contains_runtime_scalar(uop: &Arc<UOp>) -> bool {
    match uop.op() {
        Op::Range { .. } | Op::DefineVar { .. } => true,
        op => op.sources().iter().any(contains_runtime_scalar),
    }
}

// ============================================================================
// Main Expansion Pass
// ============================================================================

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
///
fn phase1_range_to_unroll() -> PatternMatcher {
    crate::patterns! {
        // Convert Range(Unroll/Upcast) to UNROLL op with constant vector
        range if matches!(range.op(), Op::Range { axis_type: AxisType::Unroll | AxisType::Upcast, .. }) => {
            convert_range_to_unroll(range)
        },
    }
}

/// Phase 2: Fix REDUCE/STORE and expand all operations using UNROLL.
///
/// Based on Tinygrad's expander PatternMatcher (expander.py:84-108).
fn phase2_expand() -> PatternMatcher {
    crate::patterns! {
        // Fix REDUCE with non-Range entries in ranges (use .. to match any ranges)
        reduce @ Reduce(_, ..) => fix_reduce_unroll(reduce),

        // Fix STORE with UNROLL in range args
        store @ Store(_, _, _) => fix_store_unroll(store),
        store @ StoreGated(_, _, _, _) => fix_store_unroll(store),

        // Handle END with UNROLL ranges
        end @ End(_, ..) => end_unrolls(end),

        // Collapse nested UNROLL: UNROLL(UNROLL(x, inner), outer) → UNROLL(x, inner+outer)
        outer @ Unroll(Unroll(_, ..), ..) => collapse_double_unroll(outer),

        // Remove empty UNROLL: UNROLL(x, ()) → x
        unroll @ Unroll(_, ..) => unwrap_empty_unroll(unroll),

        // Main expansion: ALL expandable ops with UNROLL inputs
        // Uses is_expandable() check and range_ending_src_index() for proper range handling
        op if op.op().is_expandable() && has_unroll_input(op) => do_expand(op),
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

/// Expand an operation that has UNROLL inputs.
///
/// Based on Tinygrad's do_expand (expander.py:22-65):
/// 1. Only expand if operation is expandable (ALU, LOAD, STORE, etc.)
/// 2. Collect all UNROLL inputs and unify their axes
/// 3. For each source:
///    - UNROLL with same axes: unwrap
///    - UNROLL with different axes: GEP swizzle to remap
///    - Range-position sources (per range_ending_src_index): pass through
///    - INDEX non-pointer (i >= 1): pass through
///    - Already vectorized (dtype.vcount > 1): CAT to replicate
///    - Scalar: broadcast via VECTORIZE
/// 4. Create expanded operation with vectorized dtype
/// 5. Wrap result in UNROLL
fn do_expand(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let op = uop.op();

    // Only expand operations that are marked expandable
    if !op.is_expandable() {
        return None;
    }

    // Don't expand INDEX operations that return pointers (Tinygrad expander.py:84-86)
    if matches!(op, Op::Index { .. }) && matches!(uop.dtype(), DType::Ptr { .. }) {
        return None;
    }

    let sources = op.sources();

    // Collect all UNROLL sources with their indices
    let unroll_sources: Vec<(usize, &Arc<UOp>)> =
        sources.iter().enumerate().filter(|(_, s)| matches!(s.op(), Op::Unroll { .. })).collect();

    if unroll_sources.is_empty() {
        return None;
    }

    // Collect exclude_args for WMMA (zero out reduce axes)
    let exclude_args: Vec<usize> = if let Op::Wmma { metadata, .. } = op {
        metadata.reduce_axes.iter().map(|(ax, _)| *ax).collect()
    } else {
        vec![]
    };

    // Collect all expansion args from UNROLL sources
    let all_expand_args: Vec<Vec<(usize, usize)>> = unroll_sources
        .iter()
        .filter_map(|(_, s)| if let Op::Unroll { unroll_axes, .. } = s.op() { Some(unroll_axes.clone()) } else { None })
        .collect();

    // Determine final expansion axes
    let expand_args: Vec<(usize, usize)> =
        if all_expand_args.iter().all(|a| a == &all_expand_args[0]) && exclude_args.is_empty() {
            // All same and no exclusions: use as-is (optimization)
            all_expand_args[0].clone()
        } else {
            // Otherwise, sort and dedupe, excluding exclude_args
            let mut combined: Vec<(usize, usize)> = all_expand_args.into_iter().flatten().collect();
            combined.sort_by_key(|(ax, _)| *ax);
            combined.dedup();
            combined.into_iter().filter(|(ax, _)| !exclude_args.contains(ax)).collect()
        };

    let expand_sz: usize = expand_args.iter().map(|(_, sz)| sz).product();
    if expand_sz == 0 {
        return None;
    }

    // Get range_start index for this operation (sources at/after are range args)
    let range_start_idx = op.range_ending_src_index();

    // Process each source
    let mut new_sources: SmallVec<[Arc<UOp>; 4]> = SmallVec::new();

    for (i, src) in sources.iter().enumerate() {
        if let Op::Unroll { src: inner, unroll_axes: src_axes } = src.op() {
            if *src_axes == expand_args {
                // Same expansion: unwrap
                new_sources.push(inner.clone());
            } else {
                // Different expansion: GEP swizzle
                let swizzle_indices = swizzle_args(&expand_args, src_axes, &exclude_args);

                // If inner dtype has count > 1, adjust indices for interleaving
                let inner_count = inner.dtype().vcount();
                let final_indices: Vec<usize> = if inner_count > 1 {
                    swizzle_indices
                        .iter()
                        .flat_map(|&idx| (0..inner_count).map(move |j| idx * inner_count + j))
                        .collect()
                } else {
                    swizzle_indices
                };
                new_sources.push(UOp::gep(inner.clone(), final_indices));
            }
        } else {
            // Non-UNROLL source: check special cases

            // Case 1: Range-position sources pass through unchanged
            if let Some(range_idx) = range_start_idx
                && i >= range_idx
            {
                new_sources.push(src.clone());
                continue;
            }

            // Case 2: Buffer (source 0) for memory ops passes through unchanged
            // Don't broadcast pointers - INDEX, LOAD, STORE all have buffer as source 0
            if i == 0
                && matches!(
                    op,
                    Op::Index { .. }
                        | Op::Load { .. }
                        | Op::LoadGated { .. }
                        | Op::Store { .. }
                        | Op::StoreGated { .. }
                )
            {
                new_sources.push(src.clone());
                continue;
            }

            // Case 3: INDEX indices (i >= 1) pass through when returning Ptr
            // Don't vectorize indices for pointer computation (used by STORE for GEP)
            if matches!(op, Op::Index { .. }) && i >= 1 && matches!(uop.dtype(), DType::Ptr { .. }) {
                new_sources.push(src.clone());
                continue;
            }

            // Case 3b: LOAD/STORE/LoadGated/StoreGated index (source 1) passes through
            // Don't broadcast the index expression - it needs to remain as-is for linearization
            if i == 1 && matches!(op, Op::Load { .. } | Op::LoadGated { .. } | Op::Store { .. } | Op::StoreGated { .. })
            {
                new_sources.push(src.clone());
                continue;
            }

            // Case 3c: Expressions containing runtime scalars (Range, DefineVar) pass through
            // These represent runtime values (loop counters, kernel params) - can't be vectorized.
            // This handles shift_to expressions like `Binary(Range[Reduce] * 4)`
            if contains_runtime_scalar(src) {
                new_sources.push(src.clone());
                continue;
            }

            // Case 4: Already vectorized (dtype.count > 1) -> CAT to replicate
            let src_count = src.dtype().vcount();
            if src_count > 1 {
                let cat_sources: Vec<Arc<UOp>> = (0..expand_sz).map(|_| src.clone()).collect();
                new_sources.push(UOp::cat(cat_sources));
            } else {
                // Case 4: Scalar -> broadcast
                new_sources.push(UOp::broadcast(src.clone(), expand_sz));
            }
        }
    }

    // Compute output dtype: vectorize by expand_sz
    let base_dtype = uop.dtype();
    let base_count = base_dtype.vcount();
    let new_dtype = if let Some(scalar) = base_dtype.scalar() {
        DType::Scalar(scalar).vec(base_count * expand_sz)
    } else {
        base_dtype.clone()
    };

    // Create the expanded operation
    let new_op = reconstruct_op_with_new_sources(op, &new_sources, &new_dtype)?;

    // Wrap result in UNROLL
    Some(UOp::unroll(new_op, expand_args))
}

/// Reconstruct an operation with new sources and dtype.
///
/// This handles all expandable operation types, creating a new UOp
/// with the given sources and dtype.
fn reconstruct_op_with_new_sources(op: &Op, sources: &SmallVec<[Arc<UOp>; 4]>, dtype: &DType) -> Option<Arc<UOp>> {
    match op {
        // ALU operations
        Op::Unary(unary_op, _) => sources.first().map(|s| UOp::new(Op::Unary(*unary_op, s.clone()), dtype.clone())),

        Op::Binary(binary_op, _, _) => {
            if sources.len() >= 2 {
                Some(UOp::new(Op::Binary(*binary_op, sources[0].clone(), sources[1].clone()), dtype.clone()))
            } else {
                None
            }
        }

        Op::Ternary(ternary_op, _, _, _) => {
            if sources.len() >= 3 {
                Some(UOp::new(
                    Op::Ternary(*ternary_op, sources[0].clone(), sources[1].clone(), sources[2].clone()),
                    dtype.clone(),
                ))
            } else {
                None
            }
        }

        // Type operations
        Op::Cast { dtype: cast_dtype, .. } => sources.first().map(|s| UOp::cast(s.clone(), cast_dtype.clone())),

        Op::BitCast { dtype: bitcast_dtype, .. } => sources
            .first()
            .map(|s| UOp::new(Op::BitCast { src: s.clone(), dtype: bitcast_dtype.clone() }, dtype.clone())),

        // Vector operations
        Op::Gep { indices, .. } => {
            // For GEP, recalculate indices for expanded vector
            sources.first().map(|s| {
                let src_count = s.dtype().vcount();
                let expand_sz = dtype.vcount() / src_count.max(1);
                let new_indices: Vec<usize> = indices
                    .iter()
                    .flat_map(|&idx| (0..expand_sz).map(move |e| idx + e * (src_count / expand_sz.max(1))))
                    .collect();
                UOp::gep(s.clone(), new_indices)
            })
        }

        Op::Vectorize { .. } => Some(UOp::vectorize(sources.clone())),

        // Memory operations - preserve structure
        Op::Load { .. } => {
            if sources.len() >= 2 {
                Some(UOp::new(Op::Load { buffer: sources[0].clone(), index: sources[1].clone() }, dtype.clone()))
            } else {
                None
            }
        }

        Op::LoadGated { .. } => {
            if sources.len() >= 3 {
                Some(UOp::new(
                    Op::LoadGated { buffer: sources[0].clone(), index: sources[1].clone(), gate: sources[2].clone() },
                    dtype.clone(),
                ))
            } else {
                None
            }
        }

        Op::Store { .. } => {
            if sources.len() >= 3 {
                Some(UOp::new(
                    Op::Store { buffer: sources[0].clone(), index: sources[1].clone(), value: sources[2].clone() },
                    dtype.clone(),
                ))
            } else {
                None
            }
        }

        Op::StoreGated { .. } => {
            if sources.len() >= 4 {
                Some(UOp::new(
                    Op::StoreGated {
                        buffer: sources[0].clone(),
                        index: sources[1].clone(),
                        value: sources[2].clone(),
                        gate: sources[3].clone(),
                    },
                    dtype.clone(),
                ))
            } else {
                None
            }
        }

        Op::Index { gate, .. } => {
            // First source is buffer, rest are indices (gate handled separately)
            if !sources.is_empty() {
                let buffer = sources[0].clone();
                let indices: SmallVec<[Arc<UOp>; 4]> = if gate.is_some() && sources.len() > 1 {
                    sources[1..sources.len() - 1].iter().cloned().collect()
                } else {
                    sources[1..].iter().cloned().collect()
                };
                let new_gate = if gate.is_some() { sources.last().cloned() } else { None };
                Some(UOp::new(Op::Index { buffer, indices, gate: new_gate }, dtype.clone()))
            } else {
                None
            }
        }

        // Buffer operations
        Op::Bufferize { opts, .. } => {
            if !sources.is_empty() {
                let compute = sources[0].clone();
                let ranges: SmallVec<[Arc<UOp>; 4]> = sources[1..].iter().cloned().collect();
                Some(UOp::new(Op::Bufferize { compute, ranges, opts: opts.clone() }, dtype.clone()))
            } else {
                None
            }
        }

        // Control flow
        Op::Reduce { reduce_op, .. } => {
            if !sources.is_empty() {
                let src = sources[0].clone();
                let ranges: SmallVec<[Arc<UOp>; 4]> = sources[1..].iter().cloned().collect();
                Some(UOp::new(Op::Reduce { src, ranges, reduce_op: *reduce_op }, dtype.clone()))
            } else {
                None
            }
        }

        Op::End { .. } => {
            if !sources.is_empty() {
                let computation = sources[0].clone();
                let ranges: SmallVec<[Arc<UOp>; 4]> = sources[1..].iter().cloned().collect();
                Some(UOp::new(Op::End { computation, ranges }, dtype.clone()))
            } else {
                None
            }
        }

        Op::After { .. } => {
            if !sources.is_empty() {
                let passthrough = sources[0].clone();
                let deps: SmallVec<[Arc<UOp>; 4]> = sources[1..].iter().cloned().collect();
                Some(UOp::new(Op::After { passthrough, deps }, dtype.clone()))
            } else {
                None
            }
        }

        // Tensor core - preserve metadata
        Op::Wmma { metadata, .. } => {
            if sources.len() >= 3 {
                Some(UOp::wmma(sources[0].clone(), sources[1].clone(), sources[2].clone(), metadata.clone()))
            } else {
                None
            }
        }

        // Other operations: not expandable or handled elsewhere
        _ => None,
    }
}

/// Fix REDUCE operations that have non-Range entries in their ranges.
///
/// This handles two scenarios:
/// - Range ops with Unroll/Upcast axis_type → move to CONTRACT wrapper
/// - Arithmetic expressions from shift_to substitution → extract the Reduce range
///
/// NOTE: This is more complex than Tinygrad's fix_reduce_unroll because we're NOT
/// running Phase 1 (Range→UNROLL conversion), so we handle Range(Unroll) directly.
fn fix_reduce_unroll(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce { src, reduce_op, ranges } = reduce.op() else {
        return None;
    };

    // Check if any range is not a simple Range(Reduce/Loop) op
    let has_non_reduce_range = ranges.iter().any(|r| {
        match r.op() {
            Op::Range { axis_type, .. } => !matches!(axis_type, AxisType::Reduce | AxisType::Loop),
            _ => true, // Arithmetic expressions or other ops need fixing
        }
    });

    if !has_non_reduce_range {
        return None; // Nothing to fix
    }

    let mut fixed_ranges: SmallVec<[Arc<UOp>; 4]> = SmallVec::new();
    let mut unroll_axes: Vec<(usize, usize)> = Vec::new();

    for range in ranges.iter() {
        match range.op() {
            // Range op - check axis type
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
            // Pattern: ADD(MUL(replaced_rng, Const), new_rng)
            Op::Binary(BinaryOp::Add, left, right) => {
                if let Some((reduce_range, unroll_info)) = extract_ranges_from_expr(left, right) {
                    fixed_ranges.push(reduce_range);
                    if let Some((axis_id, size)) = unroll_info {
                        unroll_axes.push((axis_id, size));
                    }
                } else {
                    // Can't extract, keep as is
                    fixed_ranges.push(range.clone());
                }
            }

            // UNROLL op (from Phase 1 if it were enabled)
            Op::Unroll { unroll_axes: axes, .. } => {
                unroll_axes.extend(axes.iter().cloned());
            }

            // Unknown pattern - keep as is
            _ => {
                fixed_ranges.push(range.clone());
            }
        }
    }

    // Build the fixed REDUCE
    let fixed_src = if unroll_axes.is_empty() {
        src.clone()
    } else {
        // Wrap source in CONTRACT to document the unrolled axes
        UOp::contract(src.clone(), unroll_axes)
    };

    // Use source dtype for REDUCE result if source is vectorized.
    // This handles UPCAST on parallel axes - the REDUCE should produce vector output
    // with element-wise accumulation, not scalar output with horizontal reduction.
    let result_dtype = if fixed_src.dtype().vcount() > 1 {
        fixed_src.dtype()
    } else {
        reduce.dtype()
    };

    Some(UOp::new(Op::Reduce { src: fixed_src, ranges: fixed_ranges, reduce_op: *reduce_op }, result_dtype))
}

/// Extract Reduce and Unroll ranges from a shift_to arithmetic expression.
///
/// shift_to creates expressions like:
/// - top=false: `replaced_rng * amount + new_rng`
/// - top=true:  `new_rng * old_sz + replaced_rng`
///
/// Returns (reduce_range, Option<(axis_id, size)>) for the unroll axis.
#[allow(clippy::type_complexity)]
fn extract_ranges_from_expr(left: &Arc<UOp>, right: &Arc<UOp>) -> Option<(Arc<UOp>, Option<(usize, usize)>)> {
    // Pattern 1 (top=false): ADD(MUL(replaced_rng, Const), new_rng)
    if let Op::Binary(BinaryOp::Mul, mul_left, _mul_right) = left.op()
        && let Op::Range { axis_type: AxisType::Reduce, .. } = mul_left.op()
    {
        let unroll_info = extract_unroll_info(right);
        return Some((mul_left.clone(), unroll_info));
    }

    // Pattern 2 (top=true): ADD(MUL(new_rng, Const), replaced_rng)
    if let Op::Range { axis_type: AxisType::Reduce, .. } = right.op()
        && let Op::Binary(BinaryOp::Mul, mul_left, _) = left.op()
    {
        let unroll_info = extract_unroll_info(mul_left);
        return Some((right.clone(), unroll_info));
    }

    None
}

/// Extract unroll axis info (axis_id, size) from a Range op or UNROLL op.
///
/// After Phase 1, Range(Unroll) has been converted to UNROLL, so we need
/// to handle both cases.
fn extract_unroll_info(uop: &Arc<UOp>) -> Option<(usize, usize)> {
    // Handle Range(Unroll) case (before Phase 1 conversion)
    if let Op::Range { axis_type: AxisType::Unroll, axis_id, end } = uop.op()
        && let Some(size) = extract_const_size(end)
    {
        return Some((axis_id.value(), size));
    }
    // Handle UNROLL op case (after Phase 1 conversion)
    if let Op::Unroll { unroll_axes, .. } = uop.op()
        && let Some(&(axis_id, size)) = unroll_axes.first()
    {
        return Some((axis_id, size));
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

// ============================================================================
// Additional Expansion Patterns (from Tinygrad expander.py)
// ============================================================================

/// Fix a STORE operation that has UNROLL in its range arguments.
///
/// Based on Tinygrad's fix_store_unroll (expander.py:123-126).
/// Wraps the STORE in CONTRACT with the UNROLL axes.
fn fix_store_unroll(store: &Arc<UOp>) -> Option<Arc<UOp>> {
    let ranges = match store.op() {
        Op::Store { .. } => {
            // Store has no explicit ranges in our model
            // But it might have UNROLL sources
            return None;
        }
        Op::StoreGated { gate, .. } => {
            // Check if gate is UNROLL
            if matches!(gate.op(), Op::Unroll { .. }) {
                vec![gate.clone()]
            } else {
                return None;
            }
        }
        _ => return None,
    };

    // Collect UNROLL axes
    let store_expand: Vec<_> = ranges.iter().filter(|r| matches!(r.op(), Op::Unroll { .. })).collect();

    if store_expand.is_empty() {
        return None;
    }

    // Collect all axes from UNROLL sources
    let contract_axes: Vec<(usize, usize)> = store_expand
        .iter()
        .filter_map(|u| if let Op::Unroll { unroll_axes, .. } = u.op() { Some(unroll_axes.clone()) } else { None })
        .flatten()
        .collect();

    if contract_axes.is_empty() {
        return None;
    }

    // Wrap STORE in CONTRACT
    Some(UOp::contract(store.clone(), contract_axes))
}

/// Handle END operations that have UNROLL in their ranges.
///
/// Based on Tinygrad's end_unrolls (expander.py:78-82).
/// Converts UNROLL ranges to CONTRACT wrapper.
fn end_unrolls(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::End { computation, ranges } = uop.op() else {
        return None;
    };

    // Partition ranges into UNROLL vs non-UNROLL
    let (unrolls, non_unrolls): (Vec<_>, Vec<_>) = ranges.iter().partition(|r| matches!(r.op(), Op::Unroll { .. }));

    if unrolls.is_empty() {
        return None;
    }

    // Collect all axes from UNROLLs
    let all_axes: Vec<(usize, usize)> = unrolls
        .iter()
        .filter_map(|u| if let Op::Unroll { unroll_axes, .. } = u.op() { Some(unroll_axes.clone()) } else { None })
        .flatten()
        .collect();

    // Wrap computation in CONTRACT
    let contracted = UOp::contract(computation.clone(), all_axes);

    // Create new END with contracted computation and non-UNROLL ranges
    let new_ranges: SmallVec<[Arc<UOp>; 4]> = non_unrolls.into_iter().cloned().collect();

    Some(UOp::new(Op::End { computation: contracted, ranges: new_ranges }, uop.dtype()))
}

/// Collapse nested UNROLL operations.
///
/// Based on Tinygrad's pattern (expander.py:94-95):
/// UNROLL(UNROLL(x, inner), outer) → UNROLL(x, inner + outer)
fn collapse_double_unroll(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Unroll { src, unroll_axes: outer_axes } = uop.op() else {
        return None;
    };

    let Op::Unroll { src: inner_src, unroll_axes: inner_axes } = src.op() else {
        return None;
    };

    // Combine axes: inner + outer
    let combined: Vec<(usize, usize)> = inner_axes.iter().chain(outer_axes.iter()).cloned().collect();

    Some(UOp::unroll(inner_src.clone(), combined))
}

/// Remove empty UNROLL operations.
///
/// Based on Tinygrad's pattern (expander.py:104):
/// UNROLL(x, ()) → x
fn unwrap_empty_unroll(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Unroll { src, unroll_axes } = uop.op() else {
        return None;
    };

    if unroll_axes.is_empty() { Some(src.clone()) } else { None }
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
