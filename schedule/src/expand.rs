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
// Helper Functions
// ============================================================================

/// Extract broadcast info: (source, count) if VECTORIZE has all identical elements.
///
/// A broadcast VECTORIZE has the form: VECTORIZE([x, x, x, ...]) where all elements
/// point to the same UOp. Returns None if not a broadcast pattern.
fn broadcast_info(uop: &Arc<UOp>) -> Option<(Arc<UOp>, usize)> {
    let Op::Vectorize { elements } = uop.op() else { return None };
    let first = elements.first()?;
    elements.iter().skip(1).all(|e| Arc::ptr_eq(e, first)).then(|| (first.clone(), elements.len()))
}

/// Fix BUFFERIZE with UNROLL sources by wrapping them in CONTRACT.
///
/// Based on Tinygrad expander.py:91-92:
///   (UPat(Ops.BUFFERIZE, src=(UPat(Ops.UNROLL), UPat(Ops.UNROLL)), name="x"),
///    lambda x: x.replace(src=tuple(UOp(Ops.CONTRACT, dtype=s.dtype.vec(x.src[1].src[0].dtype.count),
///                                      src=(s,), arg=x.src[1].arg) for s in x.src)))
///
/// When BUFFERIZE has two UNROLL sources, wrap each in CONTRACT using the second UNROLL's axes.
fn fix_bufferize_unroll(bufferize: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Bufferize { compute, ranges, opts } = bufferize.op() else {
        return None;
    };

    // Check if compute is UNROLL
    let Op::Unroll { src: _, unroll_axes: _ } = compute.op() else {
        return None;
    };

    // Check if we have at least one range that is UNROLL
    let unroll_range = ranges.iter().find(|r| matches!(r.op(), Op::Unroll { .. }))?;
    let Op::Unroll { src: _, unroll_axes: range_axes } = unroll_range.op() else {
        return None;
    };

    // Get the contract axes from the range UNROLL
    let contract_axes = range_axes.clone();

    // Wrap compute in CONTRACT
    let contracted_compute = compute.contract(contract_axes.clone());

    // Wrap each UNROLL range in CONTRACT, pass through non-UNROLL ranges
    let contracted_ranges: SmallVec<[Arc<UOp>; 4]> = ranges
        .iter()
        .map(|r| if matches!(r.op(), Op::Unroll { .. }) { r.contract(contract_axes.clone()) } else { r.clone() })
        .collect();

    Some(UOp::new(
        Op::Bufferize { compute: contracted_compute, ranges: contracted_ranges, opts: opts.clone() },
        bufferize.dtype(),
    ))
}

// ============================================================================
// Swizzle Helpers (ported from Tinygrad's expander.py:8-20)
// ============================================================================

/// Compute linear index from axis positions (row-major, reverse iteration).
///
/// Based on Tinygrad's `_expand_arg_to_idx` (expander.py:8-13).
pub fn expand_arg_to_idx(args: &[(usize, usize)], rpk: &HashMap<usize, usize>) -> usize {
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
pub fn choices_from_args(args: &[(usize, usize)]) -> Vec<HashMap<usize, usize>> {
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
pub fn swizzle_args(cargs: &[(usize, usize)], eargs: &[(usize, usize)], exclude_args: &[usize]) -> Vec<usize> {
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
pub fn phase2_expand() -> TypedPatternMatcher {
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
        // Phase 2a: Push broadcast through AFTER/END (Tinygrad expander.py:84-85)
        // =====================================================================
        // These patterns push AFTER and END inside broadcast (VECTORIZE with all same elements).
        // This is necessary for WMMA and complex kernel generation.

        // Push AFTER inside broadcast: AFTER(VECTORIZE([x;n]), deps) → VECTORIZE([AFTER(x, deps);n])
        After { passthrough, deps, .. } if broadcast_info(passthrough).is_some() => |after| {
            let (src, count) = broadcast_info(passthrough)?;
            let elements: SmallVec<[Arc<UOp>; 4]> = std::iter::repeat_n(src.after(deps.clone()), count).collect();
            Some(UOp::vectorize(elements))
        },

        // Push END inside broadcast: END(VECTORIZE([x;n]), ranges) → VECTORIZE([END(x, ranges);n])
        End { computation, ranges, .. } if broadcast_info(computation).is_some() => |end| {
            let (src, count) = broadcast_info(computation)?;
            let elements: SmallVec<[Arc<UOp>; 4]> = std::iter::repeat_n(src.end(ranges.clone()), count).collect();
            Some(UOp::vectorize(elements))
        },

        // =====================================================================
        // Phase 2b: Range conversion (pm_pre_expander pattern 1)
        // =====================================================================

        // Convert Range(Upcast) or Range(Unroll) to UNROLL op
        // This runs FIRST so that UNROLL is available for subsequent patterns
        // NOTE: REDUCE ranges are NOT converted here - they're handled by pm_reduce
        // which transforms REDUCE ops to accumulator patterns with END(ranges)
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

        // Handle END with UNROLL ranges
        end @ End(_, ..) => |end| end_unrolls(end),

        // BUFFERIZE with two UNROLL sources: wrap both in CONTRACT
        // (Tinygrad expander.py:91-92)
        bufferize if matches!(bufferize.op(), Op::Bufferize { .. }) => |bufferize| {
            fix_bufferize_unroll(bufferize)
        },

        // =====================================================================
        // Phase 2c: Core expansion (expander patterns)
        // =====================================================================

        // Collapse nested UNROLL BEFORE do_expand (Tinygrad: expander.py:94-95)
        // CRITICAL: Must run before do_expand to prevent exponential dtype growth.
        // If do_expand sees UNROLL(UNROLL(x)), it processes incorrectly.
        // UNROLL(UNROLL(x, inner), outer) → UNROLL(x, inner + outer)
        outer @ Unroll { src: Unroll { src: inner_src, unroll_axes: inner_axes, .. }, unroll_axes: outer_axes, .. } => |outer| {
            let combined: Vec<(usize, usize)> = inner_axes.iter().chain(outer_axes.iter()).cloned().collect();
            Some(inner_src.unroll_with_dtype(combined, outer.dtype().scalar_dtype()))
        },

        // Main expansion: ALL expandable ops with UNROLL inputs
        // Uses is_expandable() check and range_ending_src_index() for proper range handling.
        op if op.op().is_expandable() && has_unroll_input(op) => |op| do_expand(op),

        // Contract UNROLL via GEP extraction
        contract @ Contract(_, ..) => |contract| do_contract(contract),

        // BARRIER with UNROLL source: push BARRIER inside UNROLL
        // (Tinygrad expander.py:101-102 - "BARRIERs aren't actually expanded")
        Barrier { src: Unroll { src: inner, unroll_axes, .. }, deps, .. } => |barrier| {
            let inner_barrier = UOp::new(Op::Barrier { src: inner.clone(), deps: deps.clone() }, inner.dtype());
            Some(inner_barrier.unroll(unroll_axes.clone()))
        },

        // =====================================================================
        // Phase 2e: Cleanup
        // =====================================================================

        // Remove empty UNROLL: UNROLL(x, ()) → x
        Unroll { src, unroll_axes, .. } if unroll_axes.is_empty() ~> src,
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

    // Convert Unroll, Upcast, and Reduce axis types to UNROLL op
    // Tinygrad applies OptOps.UNROLL to REDUCE axes (changing type to UNROLL),
    // then expander eliminates the Range. Morok handles this directly by
    // converting REDUCE ranges here, achieving the same result.
    if !matches!(axis_type, AxisType::Unroll | AxisType::Upcast | AxisType::Reduce) {
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
    // Use the Range's dtype for the wrapper (matching Tinygrad: r.dtype).
    // This preserves type information through the expansion phase.
    let unroll = vconst.unroll_with_dtype(vec![(axis_id.value(), size)], range.dtype());

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
            // Check if this is a float UNROLL at Load/Store index position
            // Load/Store index at position 1 should be integer. If UNROLL wraps float data,
            // GEP swizzle would preserve float dtype, causing codegen panic.
            // Only skip GEP when inner is float - integer UNROLL needs normal expansion.
            let is_float_at_load_store_index =
                i == 1 && matches!(op, Op::Load { .. } | Op::Store { .. }) && !inner.dtype().base().is_int();

            if is_float_at_load_store_index {
                // Float data at index position - this is an upstream bug
                // Unwrap without GEP swizzle to avoid making things worse
                // Codegen's defensive check will catch this as an error
                new_sources.push(inner.clone());
                continue;
            }

            if *src_axes == expand_args {
                // Same expansion: unwrap
                new_sources.push(inner.clone());
            } else {
                // Different expansion: GEP swizzle
                let swizzle_indices = swizzle_args(&expand_args, src_axes, &exclude_args);

                // If UNROLL wrapper dtype has count > 1, adjust indices for interleaving.
                // Tinygrad (expander.py:43): if src.dtype.count > 1 (src is UNROLL wrapper)
                // This handles cases where the ORIGINAL operation had vector dtype.
                // Use UNROLL wrapper's dtype (src.dtype()), NOT inner's vectorized dtype!
                let wrapper_count = src.dtype().vcount();
                let final_indices: Vec<usize> = if wrapper_count > 1 {
                    swizzle_indices
                        .iter()
                        .flat_map(|&idx| (0..wrapper_count).map(move |j| idx * wrapper_count + j))
                        .collect()
                } else {
                    swizzle_indices
                };
                new_sources.push(inner.gep(final_indices));
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

            // Case 2: Buffer (source 0) for LOAD passes through unchanged
            // INDEX buffer is broadcast to enable devectorize expand_index pattern matching
            // (Tinygrad expander.py:56-58 broadcasts scalar sources including buffer)
            //
            // NOTE: This does NOT apply to STORE! For STORE, sources are [index, value, ranges...]
            // and the index (source 0) needs normal expansion. Only LOAD has buffer as source 0.
            if i == 0 && matches!(op, Op::Load { .. }) {
                new_sources.push(src.clone());
                continue;
            }

            // Case 3b: INDEX indices (sources at i >= 1) pass through when INDEX has element dtype
            // Tinygrad expander.py:50-51: INDEX indices are not vectorized when dtype is not PtrDType
            // This preserves the memory access pattern - each expanded INDEX uses the same indices
            if i >= 1 && matches!(op, Op::Index { .. }) && !matches!(uop.dtype(), DType::Ptr { .. }) {
                new_sources.push(src.clone());
                continue;
            }

            // Case 3d: REDUCE source for full reductions - pass through without CAT
            // A full reduction has no output-dimension ranges (Loop, Global, Outer).
            if i == 0 && matches!(op, Op::Reduce { .. }) {
                let range_start = op.range_ending_src_index().unwrap_or(1);
                fn contains_output_range(uop: &Arc<UOp>) -> bool {
                    match uop.op() {
                        Op::Range { axis_type, .. } => {
                            matches!(axis_type, AxisType::Loop | AxisType::Global | AxisType::Outer)
                        }
                        _ => uop.op().sources().iter().any(contains_output_range),
                    }
                }
                let is_full_reduction = !sources[range_start..].iter().any(contains_output_range);
                if is_full_reduction {
                    new_sources.push(src.clone());
                    continue;
                }
            }

            // Case 4: Already vectorized (dtype.count > 1) -> CAT to replicate
            let src_count = src.dtype().vcount();
            if src_count > 1 {
                let cat_sources: Vec<Arc<UOp>> = (0..expand_sz).map(|_| src.clone()).collect();
                new_sources.push(UOp::cat().sources(cat_sources).call());
            } else {
                // Case 4: Scalar -> broadcast
                new_sources.push(src.broadcast(expand_sz));
            }
        }
    }

    // Compute output dtype: vectorize by expand_sz
    // ALL ops (including REDUCE) get vectorized dtype when expanded.
    // This matches Tinygrad's do_expand (expander.py:64):
    //   nsrc = UOp(root.op, root.dtype.scalar().vec(root.dtype.count*expand_sz), ...)
    //
    // For REDUCE: Creates N independent accumulators (register blocking).
    // Each lane accumulates independently, producing N output values.
    //
    // NOTE: With K-vectorization disabled, this is the only vectorization path.
    // fix_reduce_unroll may still set Vector dtype for K-axis UPCAST if enabled,
    // but that's now opt-in via MOROK_K_VECTORIZE.
    let base_dtype = uop.dtype();
    let base_count = base_dtype.vcount();
    let new_dtype = if let Some(scalar) = base_dtype.scalar() {
        DType::Scalar(scalar).vec(base_count * expand_sz)
    } else {
        base_dtype.clone()
    };

    // =========================================================================
    // Special Case 1: GEP needs index recalculation
    // =========================================================================
    // For GEP, recalculate indices for expanded vector.
    // The source has been expanded by CAT'ing `expand_sz` copies of the original vector.
    // We need to extract the corresponding elements from each copy.
    if let Op::Gep { indices, .. } = op {
        let src = new_sources.first()?;
        let src_count = src.dtype().vcount();
        let output_count = new_dtype.vcount();
        // Tinygrad alignment: GEP expansion expects scalar output dtype or multiple indices
        debug_assert!(
            new_dtype.vcount() == 1 || indices.len() > 1,
            "GEP expansion expects scalar output or multi-index, got vcount={} with {} indices",
            new_dtype.vcount(),
            indices.len()
        );
        // expand_sz = how many copies of the original output we need
        let gep_expand_sz = output_count / indices.len().max(1);
        // stride = original source count = src_count / expand_sz
        let stride = src_count / gep_expand_sz.max(1);
        let new_indices: Vec<usize> =
            indices.iter().flat_map(|&idx| (0..gep_expand_sz).map(move |e| idx + e * stride)).collect();
        let gep_result = src.gep(new_indices);
        return Some(gep_result.unroll_with_dtype(expand_args, base_dtype.scalar_dtype()));
    }

    // =========================================================================
    // Special Case 2: END needs range filtering
    // =========================================================================
    // Filter to only include actual RANGE ops in the ranges field.
    // During expansion, non-RANGE ops (CONST, Add, etc.) may be present
    // from broadcast/CAT transformations. END.ranges must only contain RANGEs.
    if matches!(op, Op::End { .. }) {
        let computation = new_sources.first()?.clone();
        let ranges: SmallVec<[Arc<UOp>; 4]> =
            new_sources[1..].iter().filter(|s| matches!(s.op(), Op::Range { .. })).cloned().collect();
        // If no ranges remain, just return the computation wrapped in UNROLL
        if ranges.is_empty() {
            return Some(computation.unroll_with_dtype(expand_args, base_dtype.scalar_dtype()));
        }
        let end_op = UOp::new(Op::End { computation, ranges }, new_dtype.clone());
        return Some(end_op.unroll_with_dtype(expand_args, base_dtype.scalar_dtype()));
    }

    // =========================================================================
    // Special Case 3: AFTER validation
    // =========================================================================
    // Don't create AFTER with Range/End passthrough (violates Tinygrad semantics)
    if matches!(op, Op::After { .. }) {
        let passthrough = new_sources.first()?;
        if matches!(passthrough.op(), Op::Range { .. } | Op::End { .. }) {
            return None;
        }
    }

    // =========================================================================
    // Special Case 4: Comparison dtype override
    // =========================================================================
    // Comparison operations always produce Bool dtype, vectorized if needed.
    // This matches Tinygrad's behavior (uop/ops.py:412).
    let final_dtype = match op {
        Op::Binary(kind, _, _) if kind.is_comparison() => DType::Bool.vec(new_dtype.vcount()),
        _ => new_dtype.clone(),
    };

    // Create the expanded operation using replace() infrastructure
    let new_op = uop.replace().dtype(final_dtype).src(new_sources.to_vec()).call();

    // Store doesn't produce a value, so don't wrap it in UNROLL
    // (UNROLL implies a value that can be used by parent operations)
    if matches!(op, Op::Store { .. }) {
        return Some(new_op);
    }

    // Wrap result in UNROLL with SCALAR dtype (not vectorized inner dtype!)
    // Tinygrad (expander.py:65): UOp(Ops.UNROLL, root.dtype, (nsrc,), expand_args)
    // The UNROLL wrapper dtype is the ELEMENT dtype (original scalar dtype).
    // This is critical for swizzle adjustment in parent do_expand calls.
    Some(new_op.unroll_with_dtype(expand_args, base_dtype.scalar_dtype()))
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
pub(crate) fn fix_reduce_unroll(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce { src, reduce_op, ranges } = reduce.op() else {
        return None;
    };

    tracing::debug!(
        src_op = src.op().as_ref(),
        ranges_len = ranges.len(),
        ranges_types = ?ranges.iter().map(|r| match r.op() {
            Op::Range { axis_type, .. } => format!("Range({:?})", axis_type),
            _ => r.op().as_ref().to_string(),
        }).collect::<Vec<_>>(),
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

    // Assertion from Tinygrad expander.py:116 - verify axes are valid
    // (All axis sizes must be > 0 and the product must be reasonable)
    debug_assert!(
        contract_axes.iter().all(|(_, sz)| *sz > 0),
        "CONTRACT axes must have positive sizes: {:?}",
        contract_axes
    );

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
        src.contract(contract_axes)
    } else {
        src.clone()
    };

    // Check if output-dimension ranges remain (Loop/Global/Outer)
    let has_output_ranges = fixed_ranges
        .iter()
        .any(|r| matches!(r.op(), Op::Range { axis_type: AxisType::Loop | AxisType::Global | AxisType::Outer, .. }));

    // Vector dtype only if: upcast axes exist AND output dimensions remain
    // Full reductions (no output dims) collapse to scalar via horizontal reduce
    let result_dtype = if !upcast_axes.is_empty() && has_output_ranges {
        let total_upcast: usize = upcast_axes.iter().map(|(_, sz)| sz).product();
        reduce.dtype().scalar_dtype().vec(total_upcast)
    } else {
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
        Op::Store { index, value, ranges } => {
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
            let new_store = index.store_with_ranges(value.clone(), store_range.into_iter().cloned().collect());

            // Wrap in CONTRACT with void dtype (matching Tinygrad)
            Some(new_store.contract(contract_axes))
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
    let contracted = computation.contract(all_axes);

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

    // Assertion from Tinygrad expander.py:72
    // Contract dtype count should match axis product (unless void for STORE)
    let axis_product: usize = contract_axes.iter().map(|(_, sz)| sz).product();
    debug_assert!(
        uop.dtype() == DType::Void || uop.dtype().vcount() == axis_product,
        "Contract dtype count {} != axis product {}",
        uop.dtype().vcount(),
        axis_product
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
        let gep_result = unroll_inner.gep(gep_indices);

        // Return based on remaining axes
        return if remaining_axes.is_empty() {
            tracing::debug!("do_contract: full contraction -> GEP");
            // Fully contracted: return GEP result
            Some(gep_result)
        } else {
            tracing::debug!(remaining_axes = ?remaining_axes, "do_contract: partial contraction -> UNROLL(GEP)");
            // Partially contracted: wrap in UNROLL with remaining axes
            // Use scalar dtype for UNROLL wrapper to avoid wrapper_count inflation
            Some(gep_result.unroll_with_dtype(remaining_axes, uop.dtype().scalar_dtype()))
        };
    }

    // CONTRACT without UNROLL: convert to VECTORIZE (repeat elements)
    // Based on Tinygrad's do_contract (expander.py:70):
    //   if ex.op is not Ops.UNROLL: return UOp(Ops.VECTORIZE, con.dtype, con.src*con.dtype.count)
    //
    // IMPORTANT: Tinygrad uses dtype.count (vector count), NOT axis product!
    // For non-void, dtype.count == axis_product (enforced by assertion above).
    //
    // Special case for void: STORE and other void ops are side-effects, not values.
    // VECTORIZE of void doesn't make sense semantically. Just unwrap the CONTRACT.
    // This matches Tinygrad where void.count=1 creates single-element VECTORIZE
    // which is immediately simplified to the source.
    if !matches!(contract_src.op(), Op::Unroll { .. }) {
        let count = uop.dtype().vcount();
        if count == 1 {
            // Void or scalar: just unwrap CONTRACT (equivalent to identity VECTORIZE)
            return Some(contract_src.clone());
        }
        let sources: SmallVec<[Arc<UOp>; 4]> = (0..count).map(|_| contract_src.clone()).collect();
        return Some(UOp::vectorize(sources));
    }

    None
}


