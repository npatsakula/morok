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
use morok_ir::AxisType;
use morok_ir::prelude::*;

use crate::TypedPatternMatcher;
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
/// Tinygrad pattern (expander.py:143-147, Python syntax):
/// ```text
/// (UPat(Ops.RANGE, name="r"),
///  lambda r: UOp(Ops.UNROLL, r.dtype, (UOp.const(r.dtype.vec(s), tuple(range(s))),), ((r.arg[0],s),))
///  if r.arg[1] in {AxisType.UNROLL, AxisType.UPCAST} else None)
/// ```
///
fn phase1_range_to_unroll() -> TypedPatternMatcher {
    crate::patterns! {
        // Convert Range(Unroll) to UNROLL op with constant vector
        // NOTE: Range(Upcast) is NOT converted here - it's preserved for fix_reduce_unroll
        // to detect and set Vector dtype for K-vectorization. It gets converted in Phase 2.
        range if matches!(range.op(), Op::Range { axis_type: AxisType::Unroll, .. }) => |range| {
            convert_range_to_unroll(range)
        },
    }
}

/// Phase 2: Fix REDUCE/STORE and expand all operations using UNROLL.
///
/// Based on Tinygrad's expander TypedPatternMatcher (expander.py:84-108).
fn phase2_expand() -> TypedPatternMatcher {
    // Pattern order MUST match Tinygrad's pm_pre_expander + expander order:
    // 1. convert_range_to_unroll (Range → UNROLL)
    // 2. fix_reduce_unroll (REDUCE with UNROLL → CONTRACT(REDUCE))
    // 3. fix_store_unroll (STORE with UNROLL → CONTRACT(STORE))  <- BEFORE do_expand!
    // 4. do_expand (expand ops with UNROLL sources)
    // 5. do_contract (CONTRACT(UNROLL) → GEP)
    //
    // Critical: fix_store_unroll MUST run BEFORE do_expand!
    // Otherwise do_expand processes STORE first, changing the tree structure.
    crate::patterns! {
        // =====================================================================
        // Phase 2a: Range conversion (pm_pre_expander pattern 1)
        // =====================================================================

        // Convert Range(Upcast) or Range(Unroll) to UNROLL op
        // This runs FIRST so that UNROLL is available for subsequent patterns
        range if matches!(range.op(), Op::Range { axis_type: AxisType::Upcast | AxisType::Unroll, .. }) => |range| {
            convert_range_to_unroll(range)
        },

        // =====================================================================
        // Phase 2b: Pre-expansion REDUCE/STORE fixes (pm_pre_expander patterns 2-3)
        // =====================================================================

        // Fix REDUCE with non-Range entries in ranges
        // This detects Upcast axes and sets Vector dtype for K-vectorization
        reduce @ Reduce(_, ..) => |reduce| fix_reduce_unroll(reduce),

        // Fix STORE with UNROLL in ranges/index - wrap in CONTRACT
        // MUST run BEFORE do_expand! Tinygrad's fix_store_unroll is in pm_pre_expander.
        store if matches!(store.op(), Op::Store { .. }) => |store| fix_store_unroll(store),
        store if matches!(store.op(), Op::StoreGated { .. }) => |store| fix_store_unroll(store),

        // Handle END with UNROLL ranges
        end @ End(_, ..) => |end| end_unrolls(end),

        // =====================================================================
        // Phase 2c: Lift UNROLL out of Binary for proper propagation
        // =====================================================================
        // Must run BEFORE do_expand so parent ops see UNROLL as direct source.
        // Converts Binary(op, X, UNROLL) → UNROLL(Binary(op, X, unwrap))
        binary if is_binary_with_single_unroll(binary) => |binary| lift_unroll_from_binary(binary),

        // =====================================================================
        // Phase 2d: Core expansion (expander patterns)
        // =====================================================================

        // Main expansion: ALL expandable ops with UNROLL inputs
        // Uses is_expandable() check and range_ending_src_index() for proper range handling.
        op if op.op().is_expandable() && has_unroll_input(op) => |op| do_expand(op),

        // Contract UNROLL via GEP extraction
        contract @ Contract(_, ..) => |contract| do_contract(contract),

        // =====================================================================
        // Phase 2e: Cleanup
        // =====================================================================

        // Collapse nested UNROLL: UNROLL(UNROLL(x, inner), outer) → UNROLL(x, inner+outer)
        outer @ Unroll(Unroll(_, ..), ..) => |outer| collapse_double_unroll(outer),

        // Remove empty UNROLL: UNROLL(x, ()) → x
        unroll @ Unroll(_, ..) => |unroll| unwrap_empty_unroll(unroll),
    }
}

/// Convert Range(Unroll/Upcast) to UNROLL op with constant vector [0, 1, ..., N-1].
fn convert_range_to_unroll(range: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Range { end, axis_id, axis_type } = range.op() else {
        return None;
    };

    tracing::debug!(
        axis_type = ?axis_type,
        axis_id = ?axis_id,
        "convert_range_to_unroll: checking Range"
    );

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
    let unroll = UOp::unroll(vconst, vec![(axis_id.value(), size)]);

    tracing::debug!(
        axis_type = ?axis_type,
        axis_id = ?axis_id,
        size = size,
        unroll_id = unroll.id,
        "convert_range_to_unroll: CONVERTED Range to UNROLL"
    );

    Some(unroll)
}

/// Check if any input to this operation is an UNROLL op.
fn has_unroll_input(uop: &Arc<UOp>) -> bool {
    let sources = uop.op().sources();
    let has_unroll = sources.iter().any(|src| matches!(src.op(), Op::Unroll { .. }));

    // Debug: trace PointerIndex and Store ops
    if matches!(uop.op(), Op::PointerIndex { .. } | Op::Store { .. } | Op::Index { .. }) {
        tracing::debug!(
            op = ?std::mem::discriminant(uop.op()),
            has_unroll = has_unroll,
            source_ops = ?sources.iter().map(|s| std::mem::discriminant(s.op())).collect::<Vec<_>>(),
            "has_unroll_input: checking memory op"
        );
    }

    has_unroll
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

    // NOTE: We previously skipped INDEX with Ptr dtype here, but that was incorrect.
    // Tinygrad's do_expand (expander.py:97-98) DOES expand INDEX operations.
    // The special handling at lines 50-51 only affects non-UNROLL sources for non-Ptr INDEX.

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
    tracing::debug!(expand_args = ?expand_args, expand_sz = expand_sz, "do_expand: computed expansion parameters");
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

    // Store/StoreGated don't produce values, so don't wrap them in UNROLL
    // (UNROLL implies a value that can be used by parent operations)
    // Instead, just return the expanded Store directly with vectorized sources.
    if matches!(op, Op::Store { .. } | Op::StoreGated { .. }) {
        tracing::debug!(op_type = ?std::mem::discriminant(op), "do_expand: returning Store without UNROLL wrapper");
        return Some(new_op);
    }

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
                // Comparison operations always produce Bool dtype, vectorized if needed.
                // This matches Tinygrad's behavior (uop/ops.py:412):
                //   if op in {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}:
                //       out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
                let result_dtype =
                    if binary_op.is_comparison() { DType::Bool.vec(dtype.vcount()) } else { dtype.clone() };
                Some(UOp::new(Op::Binary(*binary_op, sources[0].clone(), sources[1].clone()), result_dtype))
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
        // NOTE: Use the vectorized `dtype` (new_dtype), not the original `cast_dtype`.
        // When expanding with expand_sz=N, the output dtype becomes vec<N x T> not scalar T.
        Op::Cast { .. } => sources.first().map(|s| UOp::cast(s.clone(), dtype.clone())),

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

        Op::Vectorize { .. } => {
            // After expansion, sources may be vectors (from broadcast/CAT).
            // VECTORIZE requires: dtype.vcount() == sources.len()
            // But if sources are vectors, we need CAT to flatten them.
            //
            // Example with expand_sz=4:
            //   Original: VECTORIZE([a, b]) -> vec(2)
            //   - a becomes broadcast(a, 4) = VECTORIZE -> vec(4)
            //   - b becomes broadcast(b, 4) = VECTORIZE -> vec(4)
            //   - Result: CAT -> vec(8), not VECTORIZE(vec(4), vec(4))

            let all_scalar = sources.iter().all(|s| s.dtype().vcount() == 1);

            if all_scalar {
                // All sources are still scalars - use normal VECTORIZE
                Some(UOp::vectorize(sources.clone()))
            } else {
                // Some sources are vectors - flatten into CAT
                // CAT correctly computes dtype as scalar.vec(sum_of_vcounts)
                Some(UOp::cat(sources.iter().cloned().collect()))
            }
        }

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
                    Op::Store {
                        buffer: sources[0].clone(),
                        index: sources[1].clone(),
                        value: sources[2].clone(),
                        ranges: sources[3..].iter().cloned().collect(),
                    },
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
                        ranges: sources[4..].iter().cloned().collect(),
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

        Op::PointerIndex { .. } => {
            // PointerIndex has (ptr, offset)
            if sources.len() >= 2 {
                Some(UOp::new(Op::PointerIndex { ptr: sources[0].clone(), offset: sources[1].clone() }, dtype.clone()))
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

// ============================================================================
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

    tracing::debug!(
        src_op = ?std::mem::discriminant(src.op()),
        ranges_len = ranges.len(),
        "fix_reduce_unroll: checking REDUCE"
    );

    // Skip if src is already CONTRACT-wrapped (already processed)
    // This prevents infinite rewrite loops when pattern matching
    if matches!(src.op(), Op::Contract { .. }) {
        return None;
    }

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
    // Separate Unroll (loop unrolling) from Upcast (vectorization)
    // Only Upcast axes create vectorized accumulators
    let mut unroll_axes: Vec<(usize, usize)> = Vec::new();
    let mut upcast_axes: Vec<(usize, usize)> = Vec::new();

    for range in ranges.iter() {
        match range.op() {
            // Range op - check axis type
            Op::Range { axis_type, axis_id, end } => {
                match axis_type {
                    AxisType::Reduce | AxisType::Loop => {
                        // Keep in REDUCE.ranges
                        fixed_ranges.push(range.clone());
                    }
                    AxisType::Unroll => {
                        // Loop unrolling - move to CONTRACT wrapper
                        if let Some(size) = extract_const_size(end) {
                            unroll_axes.push((axis_id.value(), size));
                        } else {
                            fixed_ranges.push(range.clone());
                        }
                    }
                    AxisType::Upcast => {
                        // SIMD vectorization - creates vectorized accumulator
                        if let Some(size) = extract_const_size(end) {
                            upcast_axes.push((axis_id.value(), size));
                        } else {
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
                        // Check if the new range is Upcast or Unroll
                        if is_upcast_range(right) || is_upcast_range(left) {
                            upcast_axes.push((axis_id, size));
                        } else {
                            unroll_axes.push((axis_id, size));
                        }
                    }
                } else {
                    // Can't extract ranges from this Binary expression.
                    // The no-op check at the end will prevent infinite loops.
                    tracing::debug!(
                        "fix_reduce_unroll: unhandled Binary in ranges: left={:?}, right={:?}",
                        left.op(),
                        right.op()
                    );
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

    // Combine all axes for CONTRACT wrapper
    let contract_axes: Vec<(usize, usize)> = unroll_axes.iter().chain(upcast_axes.iter()).copied().collect();
    let has_contract = !contract_axes.is_empty();

    // Safety check: Only return Some() if we actually changed something.
    // This prevents infinite rewrite loops when extraction fails - if ranges
    // contain unhandled Binary expressions, we'd create a near-identical REDUCE
    // that triggers the pattern again indefinitely.
    let ranges_unchanged =
        fixed_ranges.len() == ranges.len() && fixed_ranges.iter().zip(ranges.iter()).all(|(a, b)| Arc::ptr_eq(a, b));

    if ranges_unchanged && !has_contract {
        return None;
    }

    // Build the fixed REDUCE
    let fixed_src = if has_contract {
        // Wrap source in CONTRACT to document the unrolled/upcasted axes
        UOp::contract(src.clone(), contract_axes)
    } else {
        src.clone()
    };

    // Determine output dtype:
    // - If Upcast axes exist (from UPCAST on reduce axis): output VECTOR dtype
    //   This creates vectorized accumulators; horizontal reduction happens after END
    // - Otherwise: output scalar (pm_horizontal_reduce handles vectorized sources)
    let result_dtype = if !upcast_axes.is_empty() {
        // Vectorized accumulator pattern: UPCAST was applied to reduce axis
        // Output vector dtype so pm_horizontal_reduce skips this REDUCE
        // Codegen will perform horizontal reduction after the reduce loop
        let total_upcast: usize = upcast_axes.iter().map(|(_, sz)| sz).product();
        DType::Vector { scalar: reduce.dtype().base(), count: total_upcast }
    } else {
        // Standard pattern: scalar output, horizontal reduce before loop if needed
        reduce.dtype()
    };

    Some(UOp::new(Op::Reduce { src: fixed_src, ranges: fixed_ranges, reduce_op: *reduce_op }, result_dtype))
}

/// Check if a UOp is a Range(Upcast) op.
fn is_upcast_range(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Range { axis_type: AxisType::Upcast, .. })
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

    // Pattern 3: Nested shift_to - ADD(inner_add, Range(Upcast/Unroll))
    // Handles expressions from multiple shift_to applications:
    //   (replaced_rng * a + range1) + range2
    // Recursively extract reduce_range from inner Add, get unroll_info from outer range.
    if let Op::Binary(BinaryOp::Add, inner_left, inner_right) = left.op()
        && let Op::Range { axis_type, .. } = right.op()
        && matches!(axis_type, AxisType::Upcast | AxisType::Unroll)
        && let Some((reduce_range, _)) = extract_ranges_from_expr(inner_left, inner_right)
    {
        let unroll_info = extract_unroll_info(right);
        return Some((reduce_range, unroll_info));
    }

    // Pattern 4: Nested shift_to reversed - ADD(Range(Upcast/Unroll), inner_add)
    if let Op::Binary(BinaryOp::Add, inner_left, inner_right) = right.op()
        && let Op::Range { axis_type, .. } = left.op()
        && matches!(axis_type, AxisType::Upcast | AxisType::Unroll)
        && let Some((reduce_range, _)) = extract_ranges_from_expr(inner_left, inner_right)
    {
        let unroll_info = extract_unroll_info(left);
        return Some((reduce_range, unroll_info));
    }

    None
}

/// Extract unroll/upcast axis info (axis_id, size) from a Range op or UNROLL op.
///
/// After Phase 1, Range(Unroll) has been converted to UNROLL, so we need
/// to handle both cases. Also handles Range(Upcast) for vectorized accumulators.
fn extract_unroll_info(uop: &Arc<UOp>) -> Option<(usize, usize)> {
    // Handle Range(Unroll) or Range(Upcast) case (before Phase 1 conversion)
    if let Op::Range { axis_type, axis_id, end } = uop.op()
        && matches!(axis_type, AxisType::Unroll | AxisType::Upcast)
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

/// Based on Tinygrad's fix_store_unroll (expander.py:123-126).
///
/// Tinygrad's implementation:
/// ```python
/// def fix_store_unroll(x:UOp):
///   store_expand, store_range = partition(x.src[2:], lambda y: y.op is Ops.UNROLL)
///   if len(store_expand) == 0: return None
///   return UOp(Ops.CONTRACT, dtypes.void, (x.replace(src=x.src[:2]+tuple(store_range)),),
///              tuple(flatten(x.arg for x in store_expand)), tag=1)
/// ```
///
/// This ONLY partitions DIRECT children in STORE.ranges (src[2:]).
/// It does NOT search nested expressions or the index subtree.
/// UNROLLs in the index/value are handled by do_expand, not here.
fn fix_store_unroll(store: &Arc<UOp>) -> Option<Arc<UOp>> {
    match store.op() {
        Op::Store { buffer, index, value, ranges } => {
            // Partition ranges into direct UNROLL ops vs non-UNROLL
            // Matching Tinygrad: partition(x.src[2:], lambda y: y.op is Ops.UNROLL)
            let (store_expand, store_range): (Vec<_>, Vec<_>) =
                ranges.iter().partition(|r| matches!(r.op(), Op::Unroll { .. }));

            if store_expand.is_empty() {
                return None;
            }

            // Collect axes from UNROLL ops
            let contract_axes: Vec<(usize, usize)> = store_expand
                .iter()
                .filter_map(|u| match u.op() {
                    Op::Unroll { unroll_axes, .. } => Some(unroll_axes.clone()),
                    _ => None,
                })
                .flatten()
                .collect();

            tracing::debug!(
                contract_axes = ?contract_axes,
                num_store_range = store_range.len(),
                num_store_expand = store_expand.len(),
                "fix_store_unroll: partitioned STORE.ranges"
            );

            // Create new STORE with only non-UNROLL ranges
            let new_store = UOp::store_with_ranges(
                buffer.clone(),
                index.clone(),
                value.clone(),
                store_range.into_iter().cloned().collect(),
            );

            // Wrap in CONTRACT with void dtype (matching Tinygrad)
            Some(UOp::contract(new_store, contract_axes))
        }
        Op::StoreGated { buffer, index, value, gate, ranges } => {
            // Partition ranges into direct UNROLL ops vs non-UNROLL
            let (store_expand, store_range): (Vec<_>, Vec<_>) =
                ranges.iter().partition(|r| matches!(r.op(), Op::Unroll { .. }));

            if store_expand.is_empty() {
                return None;
            }

            // Collect axes from UNROLL ops
            let contract_axes: Vec<(usize, usize)> = store_expand
                .iter()
                .filter_map(|u| match u.op() {
                    Op::Unroll { unroll_axes, .. } => Some(unroll_axes.clone()),
                    _ => None,
                })
                .flatten()
                .collect();

            // Create new STORE with only non-UNROLL ranges
            let new_store = UOp::store_gated_with_ranges(
                buffer.clone(),
                index.clone(),
                value.clone(),
                gate.clone(),
                store_range.into_iter().cloned().collect(),
            );

            // Wrap in CONTRACT
            Some(UOp::contract(new_store, contract_axes))
        }
        _ => None,
    }
}

/// Handle END operations that have UNROLL in their ranges.
///
/// Based on Tinygrad's end_unrolls (expander.py:78-82).
/// Converts UNROLL ranges to CONTRACT wrapper.
fn end_unrolls(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::End { computation, ranges } = uop.op() else {
        return None;
    };

    tracing::debug!(
        ranges_len = ranges.len(),
        has_unroll = ranges.iter().any(|r| matches!(r.op(), Op::Unroll { .. })),
        "end_unrolls: checking END"
    );

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

/// Contract UNROLL to extract elements via GEP.
///
/// Based on Tinygrad's do_contract (expander.py:67-76).
///
/// CONTRACT(UNROLL(src, unroll_axes), contract_axes) transforms to:
/// - If contract_axes covers all unroll_axes: GEP(src, indices)
/// - If partially covers: UNROLL(GEP(src, indices), remaining_axes)
///
/// This prevents vector width multiplication by reorganizing the expanded data.
fn do_contract(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Contract { src: contract_src, upcast_ranges: contract_axes } = uop.op() else {
        return None;
    };

    tracing::debug!(
        contract_axes = ?contract_axes,
        contract_src_op = contract_src.op().as_ref(),
        "do_contract: matched CONTRACT op"
    );

    // Handle CONTRACT(UNROLL(...)) pattern
    if let Op::Unroll { src: unroll_inner, unroll_axes } = contract_src.op() {
        tracing::debug!(
            unroll_axes = ?unroll_axes,
            contract_axes = ?contract_axes,
            "do_contract: CONTRACT(UNROLL(...)) pattern matched"
        );

        // Compute remaining axes (unroll_axes not in contract_axes)
        let remaining_axes: Vec<(usize, usize)> =
            unroll_axes.iter().filter(|(ax, _)| !contract_axes.iter().any(|(cax, _)| cax == ax)).cloned().collect();

        tracing::debug!(
            remaining_axes = ?remaining_axes,
            "do_contract: computed remaining axes"
        );

        // Compute GEP indices using swizzle pattern
        let gep_indices = if remaining_axes.is_empty() {
            // Full contraction: sequential indices for contract_axes
            swizzle_args(contract_axes, unroll_axes, &[])
        } else {
            // Partial contraction: indices accounting for remaining axes
            let exclude: Vec<usize> = remaining_axes.iter().map(|(ax, _)| *ax).collect();
            swizzle_args(contract_axes, unroll_axes, &exclude)
        };

        tracing::debug!(gep_indices_len = gep_indices.len(), "do_contract: computed GEP indices");

        // Create GEP to extract elements
        let gep_result = UOp::gep(unroll_inner.clone(), gep_indices);

        // Return based on remaining axes
        return if remaining_axes.is_empty() {
            tracing::debug!("do_contract: full contraction -> GEP");
            // Fully contracted: return GEP result
            Some(gep_result)
        } else {
            tracing::debug!(remaining_axes = ?remaining_axes, "do_contract: partial contraction -> UNROLL(GEP)");
            // Partially contracted: wrap in UNROLL with remaining axes
            // Use CONTRACT's dtype for the per-iteration element type
            Some(UOp::unroll_with_dtype(gep_result, remaining_axes, uop.dtype()))
        };
    }

    // CONTRACT without UNROLL: convert to VECTORIZE (repeat elements)
    // Based on Tinygrad's: if ex.op is not Ops.UNROLL: return UOp(Ops.VECTORIZE, ...)
    if !matches!(contract_src.op(), Op::Unroll { .. }) {
        // For void types (STORE), Tinygrad returns VECTORIZE(void, (src,) * count)
        // For void with count=1, this is essentially identity. We unwrap to let
        // the graph continue processing (STORE's sources may still need expansion).
        if uop.dtype() == DType::Void {
            tracing::debug!("do_contract: unwrapping CONTRACT for void type (STORE)");
            return Some(contract_src.clone());
        }

        let count: usize = contract_axes.iter().map(|(_, sz)| sz).product();
        if count > 1 {
            let sources: SmallVec<[Arc<UOp>; 4]> = (0..count).map(|_| contract_src.clone()).collect();
            return Some(UOp::vectorize(sources));
        }
    }

    None
}

/// Check if Binary has exactly one UNROLL source (for pattern guard).
fn is_binary_with_single_unroll(uop: &Arc<UOp>) -> bool {
    if let Op::Binary(op, left, right) = uop.op() {
        let left_is_unroll = matches!(left.op(), Op::Unroll { .. });
        let right_is_unroll = matches!(right.op(), Op::Unroll { .. });
        let result = left_is_unroll != right_is_unroll; // XOR: exactly one is UNROLL
        if result {
            tracing::debug!(
                op = ?op,
                left_is_unroll = left_is_unroll,
                right_is_unroll = right_is_unroll,
                "is_binary_with_single_unroll: MATCHED"
            );
        }
        result
    } else {
        false
    }
}

/// Lift UNROLL out of Binary expressions for proper expansion propagation.
///
/// Pattern: Binary(op, non_unroll, UNROLL(src, axes)) → UNROLL(Binary(op, non_unroll, src), axes)
///
/// This ensures UNROLL is at the outer level so has_unroll_input() can see it
/// and do_expand() can propagate vectorization to parent operations.
fn lift_unroll_from_binary(binary: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Binary(op, left, right) = binary.op() else { return None };

    // Skip comparisons - they need both operands expanded, which do_expand handles.
    // If we lift UNROLL here, the non-UNROLL operand (e.g., input load) won't get expanded.
    if op.is_comparison() {
        return None;
    }

    let left_is_unroll = matches!(left.op(), Op::Unroll { .. });
    let right_is_unroll = matches!(right.op(), Op::Unroll { .. });

    // Only handle single UNROLL case (both UNROLL is handled by do_expand)
    if left_is_unroll == right_is_unroll {
        return None;
    }

    let (unroll_axes, unroll_inner, non_unroll, unroll_on_left) = if left_is_unroll {
        let Op::Unroll { src, unroll_axes } = left.op() else { return None };
        (unroll_axes.clone(), src.clone(), right.clone(), true)
    } else {
        let Op::Unroll { src, unroll_axes } = right.op() else { return None };
        (unroll_axes.clone(), src.clone(), left.clone(), false)
    };

    // Compute expansion size from UNROLL inner dtype
    let expand_sz = unroll_inner.dtype().vcount();

    tracing::debug!(
        op = ?op,
        unroll_axes = ?unroll_axes,
        unroll_on_left = unroll_on_left,
        expand_sz = expand_sz,
        binary_dtype = ?binary.dtype(),
        unroll_inner_dtype = ?unroll_inner.dtype(),
        non_unroll_dtype = ?non_unroll.dtype(),
        "lift_unroll_from_binary: LIFTING"
    );

    // Create new Binary with unwrapped UNROLL source (preserve operand order)
    let (new_left, new_right) = if unroll_on_left {
        (unroll_inner.clone(), non_unroll.clone())
    } else {
        (non_unroll.clone(), unroll_inner.clone())
    };

    // Use unroll_inner's dtype - for non-comparison ops, the result dtype matches operand dtype
    let new_binary = UOp::new(Op::Binary(*op, new_left, new_right), unroll_inner.dtype());

    Some(UOp::unroll(new_binary, unroll_axes.to_vec()))
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

    #[test]
    fn test_vectorize_expansion_with_mixed_sources() {
        // Test that VECTORIZE with mixed scalar/vector sources after expansion
        // produces CAT instead of invalid VECTORIZE.
        //
        // This tests the fix for: "Invalid VECTORIZE operand count: 2, expected 4"
        // which occurred when beam search used width >= 3.

        // Create an UNROLL operation (simulates expanded loop)
        let values = UOp::vconst(vec![ConstValue::Int(0), ConstValue::Int(1), ConstValue::Int(2)]);
        let unroll = UOp::unroll(values, vec![(0, 3)]);

        // Create a scalar constant
        let scalar = UOp::const_(DType::Float32, ConstValue::Float(1.0));

        // Create a Binary op with UNROLL - this will trigger expansion
        // The scalar source will be broadcast, creating a vector
        let binary = UOp::new(Op::Binary(morok_ir::BinaryOp::Add, scalar.clone(), unroll.clone()), DType::Float32);

        // Run pre_expand - should not panic
        let result = pre_expand(&binary);

        // Result should be wrapped in UNROLL with expanded inner op
        assert!(
            matches!(result.op(), Op::Unroll { .. } | Op::Binary(..)),
            "Expected UNROLL or Binary, got {:?}",
            result.op()
        );
    }

    #[test]
    fn test_vectorize_all_scalar_sources() {
        // When all sources are scalar after expansion, VECTORIZE should be used
        let scalar_a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let scalar_b = UOp::const_(DType::Float32, ConstValue::Float(2.0));

        // Create VECTORIZE with scalars only (no UNROLL)
        let vectorize = UOp::vectorize(smallvec::smallvec![scalar_a, scalar_b]);

        // No expansion needed - should pass through unchanged
        let result = pre_expand(&vectorize);

        // Should still be VECTORIZE (or equivalent)
        assert_eq!(result.dtype().vcount(), 2);
    }

    #[test]
    fn test_fix_reduce_unroll_returns_none_on_unextractable_binary() {
        // Test that fix_reduce_unroll returns None when ranges contain
        // unextractable Binary expressions. This prevents infinite rewrite loops.
        //
        // This tests the fix for: "Rewrite iteration limit (1000) exceeded"
        // which occurred when beam search used width >= 3.

        // Create a Binary expression that doesn't match any extraction pattern:
        // ADD(MUL(Const, Const), Const) - no Range ops at all!
        // This simulates a malformed expression that might result from some edge case.
        let unextractable = UOp::new(
            Op::Binary(
                morok_ir::BinaryOp::Add,
                UOp::new(
                    Op::Binary(
                        morok_ir::BinaryOp::Mul,
                        UOp::const_(DType::Index, ConstValue::Int(2)),
                        UOp::const_(DType::Index, ConstValue::Int(3)),
                    ),
                    DType::Index,
                ),
                UOp::const_(DType::Index, ConstValue::Int(1)),
            ),
            DType::Index,
        );

        let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
        let reduce = UOp::reduce(src, smallvec::smallvec![unextractable], ReduceOp::Add);

        // fix_reduce_unroll should return None because:
        // 1. The Binary expression can't be extracted (no Range ops)
        // 2. The range is kept as-is (no change)
        // 3. No CONTRACT is added
        // -> ranges_unchanged && !has_contract -> return None
        let result = fix_reduce_unroll(&reduce);
        assert!(result.is_none(), "Expected None when extraction fails and nothing changes");
    }

    #[test]
    fn test_fix_reduce_unroll_handles_nested_binary() {
        // Test that nested Binary expressions (from multiple shift_to) are handled.
        // Pattern: ADD(ADD(MUL(reduce_range, 4), range_upcast1), range_upcast2)

        let end = UOp::const_(DType::Index, ConstValue::Int(64));
        let reduce_range = UOp::range_axis(end.clone(), morok_ir::AxisId::Renumbered(0), AxisType::Reduce);

        // Inner shift_to result: MUL(reduce_range, 4) + Range(Upcast)
        let upcast_end = UOp::const_(DType::Index, ConstValue::Int(4));
        let upcast_range = UOp::range_axis(upcast_end.clone(), morok_ir::AxisId::Renumbered(1), AxisType::Upcast);

        let mul = UOp::new(
            Op::Binary(morok_ir::BinaryOp::Mul, reduce_range.clone(), UOp::const_(DType::Index, ConstValue::Int(4))),
            DType::Index,
        );
        let inner_add = UOp::new(Op::Binary(morok_ir::BinaryOp::Add, mul, upcast_range.clone()), DType::Index);

        // Outer shift_to: inner_add + Range(Upcast)
        let upcast_range2 = UOp::range_axis(upcast_end, morok_ir::AxisId::Renumbered(2), AxisType::Upcast);
        let nested_binary = UOp::new(Op::Binary(morok_ir::BinaryOp::Add, inner_add, upcast_range2), DType::Index);

        let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
        let reduce = UOp::reduce(src, smallvec::smallvec![nested_binary], ReduceOp::Add);

        // fix_reduce_unroll should extract the reduce_range and handle the upcast axes
        let result = fix_reduce_unroll(&reduce);
        assert!(result.is_some(), "Expected Some when nested Binary can be extracted");

        // Check the result has CONTRACT wrapper (for upcast axes)
        if let Some(fixed) = result
            && let Op::Reduce { src: fixed_src, .. } = fixed.op()
        {
            assert!(matches!(fixed_src.op(), Op::Contract { .. }), "Expected CONTRACT wrapper for upcast axes");
        }
    }
}
// TEMP DEBUG: Add detailed tracing to do_expand
