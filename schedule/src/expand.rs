//! Pre-expander pass for UNROLL/CONTRACT expansion.
//!
//! Transforms kernel AST before codegen to handle UNROLL optimization.
//!
//! # Pipeline (Tinygrad-aligned)
//!
//! Two-phase expansion:
//! 1. **Range conversion**: Range(Unroll/Upcast) → UNROLL(VCONST([0..N]))
//! 2. **Expansion**:
//!    - fix_reduce_unroll: Partition REDUCE.ranges, move UNROLL to CONTRACT
//!    - fix_store_unroll: Partition STORE.ranges, wrap in CONTRACT
//!    - do_expand: Propagate vectorization through ops with UNROLL inputs
//!    - do_contract: CONTRACT(UNROLL) → GEP extraction
//!
//! # Key Insight
//!
//! UNROLL holds all loop iterations as a vector. Operations using UNROLL
//! get replicated/vectorized. CONTRACT collapses back to scalar for output.

use std::collections::HashMap;
use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::prelude::*;
use morok_ir::{AxisId, AxisType};

use crate::TypedPatternMatcher;
use smallvec::{SmallVec, smallvec};

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert ConstValue to usize (for Range end values).
fn const_to_usize(cv: &ConstValue) -> Option<usize> {
    match cv {
        ConstValue::Int(i) if *i > 0 => Some(*i as usize),
        ConstValue::UInt(u) => Some(*u as usize),
        _ => None,
    }
}

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
/// Matches BUFFERIZE with compute=UNROLL and exactly one range that is UNROLL.
/// Wraps BOTH in CONTRACT using the range UNROLL's axes.
fn fix_bufferize_unroll(bufferize: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Bufferize { compute, ranges, opts } = bufferize.op() else {
        return None;
    };

    // Check if compute is UNROLL
    let Op::Unroll { .. } = compute.op() else {
        return None;
    };

    // Must have exactly one range that is UNROLL (Tinygrad: src=(UNROLL, UNROLL))
    if ranges.len() != 1 {
        return None;
    }
    let Op::Unroll { src: range_inner, unroll_axes: contract_axes } = ranges[0].op() else {
        return None;
    };

    // Contract dtype: s.dtype.vec(x.src[1].src[0].dtype.count)
    // = source dtype vectorized by second UNROLL's inner count
    let inner_count = range_inner.dtype().vcount();

    // Wrap compute in CONTRACT
    let contracted_compute = UOp::new(
        Op::Contract { src: compute.clone(), upcast_ranges: contract_axes.clone() },
        compute.dtype().vec(inner_count),
    );

    // Wrap range in CONTRACT
    let contracted_range = UOp::new(
        Op::Contract { src: ranges[0].clone(), upcast_ranges: contract_axes.clone() },
        ranges[0].dtype().vec(inner_count),
    );

    Some(UOp::new(
        Op::Bufferize { compute: contracted_compute, ranges: smallvec![contracted_range], opts: opts.clone() },
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
///
/// # Tinygrad Pipeline Alignment (Stage 9)
///
/// Tinygrad: `sym + pm_pre_expander + pm_group_for_reduce + expander`
///
/// Our phases:
/// 1. Convert Range(Unroll/Upcast) → UNROLL ops with constant vectors
/// 2. Apply expansion patterns with symbolic simplification
///
/// # Traversal Direction
///
/// Uses bottom-up traversal. Note that Tinygrad's `bottom_up=False` is actually
/// a hybrid that processes children first, then applies patterns - it's NOT
/// purely top-down. Morok's bottom-up matches this behavior better because:
/// - Range(Upcast) → UNROLL conversion must complete before fix_reduce_unroll
/// - Pattern dependencies require children to be transformed first
pub fn pre_expand(ast: &Arc<UOp>) -> Arc<UOp> {
    use crate::rewrite::graph_rewrite_bottom_up;
    use crate::symbolic::symbolic_simple;

    // Phase 1: Convert Range(Unroll/Upcast) to UNROLL ops
    let phase1 = phase1_range_to_unroll();
    let ast = graph_rewrite_bottom_up(&phase1, ast.clone(), &mut ());

    // Phase 2: Expander + symbolic (Tinygrad: sym + pm_pre_expander + expander)
    // Combines symbolic simplification with expansion for single-pass efficiency.
    // Uses bottom-up: children transformed before parents, matching Tinygrad's
    // actual behavior (despite their "bottom_up=False" naming).
    let phase2 = symbolic_simple() + phase2_expand();
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
        range @ Range { end: _end @const(cv), axis_id, axis_type }
            if matches!(axis_type, AxisType::Unroll) => |range| {
            let size = const_to_usize(&cv)?;
            let values: Vec<ConstValue> = (0..size as i64).map(ConstValue::Int).collect();
            let vconst = UOp::vconst(values);
            Some(vconst.unroll_with_dtype(vec![(axis_id.value(), size)], range.dtype()))
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

        // Convert Range(Upcast/Unroll) to UNROLL op
        // This runs FIRST so that UNROLL is available for subsequent patterns
        // NOTE: Reduce ranges are NOT converted here - they remain as ranges for REDUCE.ranges
        range @ Range { end: _end @const(cv), axis_id, axis_type }
            if matches!(axis_type, AxisType::Upcast | AxisType::Unroll) => |range| {
            let size = const_to_usize(&cv)?;
            let values: Vec<ConstValue> = (0..size as i64).map(ConstValue::Int).collect();
            let vconst = UOp::vconst(values);
            Some(vconst.unroll_with_dtype(vec![(axis_id.value(), size)], range.dtype()))
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

        // Collapse nested UNROLL BEFORE do_expand (Tinygrad expander.py:94-95)
        // UNROLL(UNROLL(x, inner), outer) → UNROLL(x, inner + outer)
        outer @ Unroll { src: Unroll { src: inner_src, unroll_axes: inner_axes, .. }, unroll_axes: outer_axes, .. } => |outer| {
            let combined: Vec<(usize, usize)> = inner_axes.iter().chain(outer_axes.iter()).cloned().collect();
            Some(inner_src.unroll_with_dtype(combined, outer.dtype()))
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
            // Tinygrad expander.py:47-49
            if let Some(range_idx) = range_start_idx
                && i >= range_idx
            {
                new_sources.push(src.clone());
                continue;
            }

            // Case 2: INDEX indices (sources at i >= 1) pass through when INDEX has element dtype
            // Tinygrad expander.py:50-51: INDEX indices are not vectorized when dtype is not PtrDType
            if i >= 1 && matches!(op, Op::Index { .. }) && !matches!(uop.dtype(), DType::Ptr { .. }) {
                new_sources.push(src.clone());
                continue;
            }

            // Case 3: Already vectorized (dtype.count > 1) -> CAT to replicate
            // Tinygrad expander.py:52-54
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

    // GEP: recalculate indices for expanded vector
    // Tinygrad expander.py:60-63
    if let Op::Gep { indices, .. } = op {
        debug_assert_eq!(base_dtype.vcount(), 1, "GEP expansion expects scalar output dtype");
        let src = new_sources.first()?;
        let src_count = src.dtype().vcount();
        // Tinygrad: tuple(range(arg[0], src_count, src_count // expand_sz))
        let stride = src_count / expand_sz;
        let new_indices: Vec<usize> =
            indices.iter().flat_map(|&idx| (0..expand_sz).map(move |e| idx + e * stride)).collect();
        // Fall through to normal path with modified arg
        let gep_result = src.gep(new_indices);
        return Some(gep_result.unroll(expand_args));
    }

    // Create the expanded operation using replace() infrastructure
    // Tinygrad expander.py:64
    let new_op = uop.replace().dtype(new_dtype.clone()).src(new_sources.to_vec()).call();

    // Wrap result in UNROLL with original dtype
    // Tinygrad expander.py:65: UOp(Ops.UNROLL, root.dtype, (nsrc,), expand_args)
    Some(new_op.unroll_with_dtype(expand_args, base_dtype))
}

// ============================================================================
/// Fix REDUCE operations with UNROLL ranges.
///
/// Matches Tinygrad expander.py:112-121 exactly:
/// ```python
/// def fix_reduce_unroll(x:UOp):
///   reduce_range, reduce_expand = partition(x.src[1:], lambda y: y.op is Ops.RANGE)
///   if len(reduce_expand) == 0: return None
///   reduce_expand = [x for x in reduce_expand if x.op is not Ops.CONST]
///   assert all(x.op is Ops.UNROLL for x in reduce_expand), f"not all UNROLLS in {reduce_expand}"
///   ret = x.src[0]
///   if len(contract_axis:=flatten(x.arg for x in reduce_expand)):
///     ret = UOp(Ops.CONTRACT, x.dtype.vec(prod(x[1] for x in contract_axis)), (ret,), tuple(contract_axis), tag=1)
///   return x.replace(src=(ret,)+tuple(reduce_range))
/// ```
pub(crate) fn fix_reduce_unroll(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce { src, reduce_op, ranges } = reduce.op() else {
        return None;
    };

    // Partition: RANGE ops vs everything else
    let (reduce_range, reduce_expand): (Vec<_>, Vec<_>) =
        ranges.iter().partition(|r| matches!(r.op(), Op::Range { .. }));

    if reduce_expand.is_empty() {
        return None;
    }

    // Filter out CONST
    let reduce_expand: Vec<_> = reduce_expand.into_iter().filter(|r| !matches!(r.op(), Op::Const(_))).collect();

    if reduce_expand.is_empty() {
        return None;
    }

    // All non-Range, non-Const must be UNROLL (Tinygrad assertion)
    debug_assert!(
        reduce_expand.iter().all(|r| matches!(r.op(), Op::Unroll { .. })),
        "not all UNROLLS in {:?}",
        reduce_expand.iter().map(|r| r.op().as_ref()).collect::<Vec<_>>()
    );

    // Collect contract axes from UNROLL ops
    let contract_axes: Vec<(usize, usize)> = reduce_expand
        .iter()
        .filter_map(|u| match u.op() {
            Op::Unroll { unroll_axes, .. } => Some(unroll_axes.clone()),
            _ => None,
        })
        .flatten()
        .collect();

    // Wrap source in CONTRACT if axes exist
    let contracted_src = if !contract_axes.is_empty() {
        let total: usize = contract_axes.iter().map(|(_, sz)| sz).product();
        UOp::new(Op::Contract { src: src.clone(), upcast_ranges: contract_axes }, reduce.dtype().vec(total))
    } else {
        src.clone()
    };

    // Return REDUCE with only Range ops (Tinygrad: x.replace(src=(ret,)+tuple(reduce_range)))
    Some(UOp::new(
        Op::Reduce { src: contracted_src, ranges: reduce_range.into_iter().cloned().collect(), reduce_op: *reduce_op },
        reduce.dtype(),
    ))
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
fn end_unrolls(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::End { computation, ranges } = uop.op() else { return None };

    let (unrolls, non_unrolls): (Vec<_>, Vec<_>) = ranges.iter().partition(|r| matches!(r.op(), Op::Unroll { .. }));
    if unrolls.is_empty() {
        return None;
    }

    let all_axes: Vec<(usize, usize)> = unrolls
        .iter()
        .filter_map(|u| match u.op() {
            Op::Unroll { unroll_axes, .. } => Some(unroll_axes.clone()),
            _ => None,
        })
        .flatten()
        .collect();

    let contracted = computation.contract(all_axes);
    Some(UOp::new(Op::End { computation: contracted, ranges: non_unrolls.into_iter().cloned().collect() }, uop.dtype()))
}

/// Contract UNROLL to extract elements via GEP.
///
/// Based on Tinygrad's do_contract (expander.py:67-76).
fn do_contract(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Contract { src: contract_src, upcast_ranges: contract_axes } = uop.op() else {
        return None;
    };

    // CONTRACT without UNROLL → VECTORIZE
    let Op::Unroll { src: unroll_inner, unroll_axes } = contract_src.op() else {
        let count = uop.dtype().vcount();
        if count == 1 {
            return Some(contract_src.clone());
        }
        let sources: SmallVec<[Arc<UOp>; 4]> = (0..count).map(|_| contract_src.clone()).collect();
        return Some(UOp::vectorize(sources));
    };

    // Assertion from Tinygrad expander.py:72
    debug_assert!(
        uop.dtype() == DType::Void || uop.dtype().vcount() == contract_axes.iter().map(|(_, sz)| sz).product::<usize>(),
        "Contract dtype count mismatch"
    );

    // Compute remaining axes and GEP indices
    let remaining_axes: Vec<_> =
        unroll_axes.iter().filter(|(ax, _)| !contract_axes.iter().any(|(cax, _)| cax == ax)).cloned().collect();

    let exclude: Vec<usize> = remaining_axes.iter().map(|(ax, _)| *ax).collect();
    let gep_indices = swizzle_args(contract_axes, unroll_axes, &exclude);
    let gep_result = unroll_inner.gep(gep_indices);

    // Tinygrad expander.py:76: UOp(Ops.UNROLL, con.dtype, (ex.src[0].gep(...),), new_ex_args)
    Some(gep_result.unroll_with_dtype(remaining_axes, uop.dtype()))
}

// ============================================================================
// GROUP_REDUCE Pattern (pm_group_for_reduce)
// ============================================================================

/// Transform REDUCE with GROUP_REDUCE ranges to shared memory pattern.
///
/// Based on Tinygrad's fix_group_for_reduce (expander.py:128-141):
/// ```python
/// def fix_group_for_reduce(x:UOp):
///   reduce_gfr, reduce_r = partition(x.src[1:], lambda u: u.op is Ops.RANGE and u.arg[1] == AxisType.GROUP_REDUCE)
///   if len(reduce_gfr) == 0: return None
///
///   upstream_locals = [u for u in x.toposort() if u.op is Ops.RANGE and u.arg[1] == AxisType.LOCAL]
///
///   ret = x.replace(src=(x.src[0],)+tuple(reduce_r))
///   reduce_loop = [x.replace(arg=(x.arg[0]+100, AxisType.REDUCE)) for x in reduce_gfr]
///   buf = ret.bufferize(*upstream_locals, *reduce_gfr, arg=BufferizeOpts(reduce_gfr[0].arg[0], AddrSpace.LOCAL))
///            .index(*upstream_locals, *reduce_loop)
///
///   return buf.reduce(*reduce_loop, arg=x.arg)
/// ```
///
/// # Transformation
///
/// GROUP_REDUCE enables two-stage reduction:
/// 1. First reduce within each group (normal REDUCE with non-GROUP_REDUCE ranges)
/// 2. Store to shared memory indexed by LOCAL + GROUP_REDUCE ranges
/// 3. Load from shared memory indexed by LOCAL + new REDUCE ranges
/// 4. Final reduce across the new REDUCE ranges
///
/// # Why Shared Memory?
///
/// Tensor core operations produce partial results that need cross-warp reduction.
/// GROUP_REDUCE stages this through shared memory for efficient communication.
fn fix_group_for_reduce(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce { src, reduce_op, ranges } = reduce.op() else {
        return None;
    };

    // Partition ranges into GROUP_REDUCE vs other
    let (reduce_gfr, reduce_r): (Vec<_>, Vec<_>) =
        ranges.iter().partition(|r| matches!(r.op(), Op::Range { axis_type: AxisType::GroupReduce, .. }));

    if reduce_gfr.is_empty() {
        return None;
    }

    // Find upstream LOCAL ranges in the computation graph
    let upstream_locals: Vec<Arc<UOp>> = reduce
        .toposort()
        .into_iter()
        .filter(|u| matches!(u.op(), Op::Range { axis_type: AxisType::Local, .. }))
        .collect();

    // Step 1: Create partial reduce with non-GROUP_REDUCE ranges
    // Tinygrad: ret = x.replace(src=(x.src[0],)+tuple(reduce_r))
    let partial_reduce = if reduce_r.is_empty() {
        src.clone()
    } else {
        UOp::new(
            Op::Reduce { src: src.clone(), ranges: reduce_r.into_iter().cloned().collect(), reduce_op: *reduce_op },
            reduce.dtype(),
        )
    };

    // Step 2: Create renumbered REDUCE ranges (axis_id + 100)
    // Tinygrad: reduce_loop = [x.replace(arg=(x.arg[0]+100, AxisType.REDUCE)) for x in reduce_gfr]
    let reduce_loops: Vec<Arc<UOp>> = reduce_gfr
        .iter()
        .filter_map(|r| {
            let Op::Range { end, axis_id, .. } = r.op() else { return None };
            Some(UOp::range_axis(end.clone(), AxisId::Renumbered(axis_id.value() + 100), AxisType::Reduce))
        })
        .collect();

    // Step 3: Create BUFFERIZE → INDEX → REDUCE
    // Buffer ranges: [LOCAL...] + [GROUP_REDUCE...]
    let buf_ranges: Vec<Arc<UOp>> =
        upstream_locals.iter().cloned().chain(reduce_gfr.iter().map(|r| (*r).clone())).collect();

    let buf = UOp::bufferize_local(partial_reduce, buf_ranges);

    // Index ranges: [LOCAL...] + [new REDUCE...]
    let idx_ranges: Vec<Arc<UOp>> = upstream_locals.iter().cloned().chain(reduce_loops.iter().cloned()).collect();

    let indexed = UOp::index().buffer(buf).indices(idx_ranges).call().ok()?;

    // Step 4: Final reduction
    Some(indexed.reduce(reduce_loops.into_iter().collect(), *reduce_op))
}

/// Pattern matcher for GROUP_REDUCE transformation.
///
/// Based on Tinygrad's pm_group_for_reduce (expander.py:153-156).
///
/// This should be applied AFTER pm_pre_expander but BEFORE the main expander.
pub fn pm_group_for_reduce() -> TypedPatternMatcher {
    crate::patterns! {
        // Match REDUCE ops and transform GROUP_REDUCE ranges
        reduce @ Reduce(_, ..) => |reduce| fix_group_for_reduce(reduce),
    }
}
