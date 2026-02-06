//! GPU dimension injection for kernel execution.
//!
//! This module implements `pm_add_gpudims`, which transforms RANGE operations
//! with GLOBAL/LOCAL axis types into SPECIAL UOps representing GPU thread indices.
//!
//! Based on Tinygrad's `gpudims.py`.
//!
//! # Pipeline Position
//!
//! Runs between `pm_reduce` (Stage 11) and `pm_add_loads` (Stage 13):
//! - After reduction is lowered to accumulator patterns
//! - Before loads are explicitly extracted from INDEX ops
//!
//! # Transformation
//!
//! ```text
//! RANGE(end, axis_id, GLOBAL) → gidxN (SPECIAL with global thread index)
//! RANGE(end, axis_id, LOCAL)  → lidxN (SPECIAL with local thread index)
//! ```
//!
//! Dimension limiting is applied to fit within hardware constraints:
//! - Grouping: Merge adjacent dimensions that fit within limits
//! - Splitting: Factor dimensions that exceed limits

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::types::{AxisType, ConstValue};
use morok_ir::{Op, UOp};
use smallvec::SmallVec;

use crate::optimizer::Renderer;
use crate::pattern::TypedPatternMatcher;

/// Pattern matcher for GPU dimension injection.
///
/// Matches SINK operations and transforms GLOBAL/LOCAL ranges to SPECIAL ops.
/// Must run after pm_reduce and before pm_add_loads.
///
/// # Context
///
/// Requires `&Renderer` context to access device limits (global_max, local_max).
pub fn pm_add_gpudims() -> TypedPatternMatcher<Renderer> {
    crate::patterns! {
        @context Renderer;
        // Match SINK with at least one source
        sink @ Sink { sources: _sources } => |sink| add_gpudims(ctx, sink),
    }
}

/// Main transformation: inject GPU dimensions into SINK.
///
/// Follows Tinygrad's `add_gpudims` function (gpudims.py:59-103):
/// 1. Collect all RANGE operations from topology
/// 2. Check for existing SPECIAL ops (skip if found)
/// 3. Categorize ranges by axis type (GLOBAL/THREAD vs LOCAL/WARP/GROUP_REDUCE)
/// 4. Create SPECIAL indices with dimension limiting
/// 5. Substitute RANGE ops with computed indices
fn add_gpudims(ctx: &Renderer, sink: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Sink { sources } = sink.op() else {
        return None;
    };

    // Collect topology (all UOps reachable from sink)
    let topo = sink.toposort();

    // Check for existing SPECIAL ops - if found, gpudims already applied
    if topo.iter().any(|u| matches!(u.op(), Op::Special { .. })) {
        return None;
    }

    // Collect all RANGE operations, keyed by (axis_id, axis_type)
    // We exclude axis_type from the key matching for categorization, but track it
    let mut all_ranges: HashMap<(usize, AxisType), Arc<UOp>> = HashMap::new();
    for u in &topo {
        if let Op::Range { axis_id, axis_type, .. } = u.op() {
            all_ranges.insert((axis_id.value(), *axis_type), u.clone());
        }
    }

    if all_ranges.is_empty() {
        return None;
    }

    // Categorize ranges by axis type
    // Global dims: GLOBAL, THREAD
    // Local dims: LOCAL, WARP, GROUP_REDUCE
    let mut global_dims: Vec<(usize, AxisType)> = Vec::new();
    let mut local_dims: Vec<(usize, AxisType)> = Vec::new();

    for (axis_id, axis_type) in all_ranges.keys() {
        match axis_type {
            AxisType::Global | AxisType::Thread => {
                if !global_dims.iter().any(|(id, _)| *id == *axis_id) {
                    global_dims.push((*axis_id, *axis_type));
                }
            }
            AxisType::Local | AxisType::Warp | AxisType::GroupReduce => {
                if !local_dims.iter().any(|(id, _)| *id == *axis_id) {
                    local_dims.push((*axis_id, *axis_type));
                }
            }
            _ => {}
        }
    }

    // Sort by axis_id for consistent ordering
    global_dims.sort_by_key(|(id, _)| *id);
    local_dims.sort_by_key(|(id, _)| *id);

    // No GPU dimensions to inject
    if global_dims.is_empty() && local_dims.is_empty() {
        return None;
    }

    // Extract shapes from RANGE operations (the end values)
    let get_ranges_for_dims = |dims: &[(usize, AxisType)]| -> Vec<Arc<UOp>> {
        dims.iter().filter_map(|(axis_id, axis_type)| all_ranges.get(&(*axis_id, *axis_type))).cloned().collect()
    };

    let global_ranges = get_ranges_for_dims(&global_dims);
    let local_ranges = get_ranges_for_dims(&local_dims);

    // Extract dimension sizes from ranges
    let extract_shape = |ranges: &[Arc<UOp>]| -> Vec<Arc<UOp>> {
        ranges
            .iter()
            .filter_map(|r| match r.op() {
                Op::Range { end, .. } => Some(end.clone()),
                _ => None,
            })
            .collect()
    };

    let global_shape = extract_shape(&global_ranges);
    let local_shape = extract_shape(&local_ranges);

    // Generate GPU indices
    let global_max = ctx.global_max.as_deref();
    let local_max_product = ctx.local_max;

    // For locals, we use product limit rather than per-dimension
    // Convert to per-dimension limits if needed
    let local_max: Option<Vec<usize>> = local_max_product.map(|max| {
        // Simple heuristic: distribute limit evenly if multiple dimensions
        let n = local_shape.len().max(1);
        let per_dim = (max as f64).powf(1.0 / n as f64).floor() as usize;
        vec![per_dim.max(1); n]
    });
    let local_max_slice = local_max.as_deref();

    // Create global indices (gidx0, gidx1, ...)
    let global_idxs = get_grouped_dims("gidx", &global_shape, global_max, true);
    // Create local indices (lidx0, lidx1, ...)
    let local_idxs = get_grouped_dims("lidx", &local_shape, local_max_slice, false);

    // Clone local_idxs for later use in store masking
    let local_idxs_for_masks = local_idxs.clone();

    // Combine indices in order: global, then local
    let all_idxs: Vec<Arc<UOp>> = global_idxs.into_iter().chain(local_idxs).collect();

    // Build substitution map: RANGE -> corresponding index
    let mut subs: HashMap<u64, Arc<UOp>> = HashMap::new();
    let all_dims: Vec<(usize, AxisType)> = global_dims.iter().chain(local_dims.iter()).cloned().collect();

    for (i, (axis_id, axis_type)) in all_dims.iter().enumerate() {
        if *axis_type == AxisType::Reduce {
            // Don't replace reduce axes (they stay as loops)
            continue;
        }
        if let Some(range_uop) = all_ranges.get(&(*axis_id, *axis_type))
            && i < all_idxs.len()
        {
            subs.insert(range_uop.id, all_idxs[i].clone());
        }
    }

    // Handle STORE masking for global stores with missing local indices
    // When a STORE to global memory doesn't use all local indices,
    // we need to mask the store to only execute when unused local indices are 0
    let store_subs = compute_store_masks(&topo, &all_ranges, &local_dims, &local_idxs_for_masks);
    for (id, masked_idx) in store_subs {
        subs.insert(id, masked_idx);
    }

    // Apply substitutions to rebuild the sink
    if subs.is_empty() {
        return None;
    }

    let new_sources: SmallVec<[Arc<UOp>; 4]> = sources.iter().map(|s| substitute(s, &subs)).collect();

    Some(UOp::new(Op::Sink { sources: new_sources }, sink.dtype().clone()))
}

/// Compute store masks for global stores with missing local indices.
///
/// Based on Tinygrad's gpudims.py:86-96.
/// When a STORE to global memory doesn't use all local indices,
/// we add a mask so the store only executes when missing locals are 0.
fn compute_store_masks(
    topo: &[Arc<UOp>],
    all_ranges: &HashMap<(usize, AxisType), Arc<UOp>>,
    local_dims: &[(usize, AxisType)],
    local_idxs: &[Arc<UOp>],
) -> HashMap<u64, Arc<UOp>> {
    let mut masks: HashMap<u64, Arc<UOp>> = HashMap::new();

    for uop in topo {
        let Op::Store { index, .. } = uop.op() else {
            continue;
        };

        // Check if store targets global memory
        // In Morok, we check if the INDEX's buffer has Global addrspace
        let is_global_store = match index.op() {
            Op::Index { buffer, .. } => match buffer.dtype() {
                DType::Ptr { addrspace, .. } => addrspace == morok_dtype::AddrSpace::Global,
                _ => true, // Assume global if not a pointer type
            },
            _ => continue,
        };

        if !is_global_store {
            continue;
        }

        // Find local ranges NOT used in the index computation.
        // Use in_scope_ranges() to get only active (not ended) ranges,
        // rather than toposort().filter(Range) which returns ALL ranges in the graph.
        let index_ranges: HashSet<u64> = index.in_scope_ranges().iter().map(|key| key.0.id).collect();

        let mut missing_locals: Vec<Arc<UOp>> = Vec::new();
        for (i, (axis_id, axis_type)) in local_dims.iter().enumerate() {
            if let Some(range_uop) = all_ranges.get(&(*axis_id, *axis_type))
                && !index_ranges.contains(&range_uop.id)
                && i < local_idxs.len()
            {
                missing_locals.push(local_idxs[i].clone());
            }
        }

        if missing_locals.is_empty() {
            continue;
        }

        // Create mask: (missing_local_1 == 0) & (missing_local_2 == 0) & ...
        // Using eq() and and_() panicking wrappers for cleaner code
        let zero = UOp::index_const(0);
        let mut mask: Option<Arc<UOp>> = None;
        for local_idx in missing_locals {
            let eq_zero = local_idx.eq(&zero);
            mask = Some(match mask {
                None => eq_zero,
                Some(m) => m.and_(&eq_zero),
            });
        }

        // Add gate to INDEX if mask exists
        if let (Some(mask), Op::Index { buffer, indices, gate }) = (mask, index.op()) {
            let new_gate = match gate {
                Some(existing) => existing.and_(&mask),
                None => mask,
            };
            // Use INDEX builder pattern
            let new_index = UOp::index()
                .buffer(buffer.clone())
                .indices(indices.clone())
                .gate(new_gate)
                .call()
                .expect("gpudims: INDEX gate construction failed");
            masks.insert(index.id, new_index);
        }
    }

    masks
}

/// Substitute UOps according to the substitution map.
fn substitute(uop: &Arc<UOp>, subs: &HashMap<u64, Arc<UOp>>) -> Arc<UOp> {
    // Check if this exact UOp should be substituted
    if let Some(replacement) = subs.get(&uop.id) {
        return replacement.clone();
    }

    // Recursively substitute children
    let children = uop.op().children();
    if children.is_empty() {
        return uop.clone();
    }

    let new_children: Vec<Arc<UOp>> = children.iter().map(|c| substitute(c, subs)).collect();

    // Check if any children changed
    let changed = children.iter().zip(&new_children).any(|(old, new)| old.id != new.id);

    if !changed {
        return uop.clone();
    }

    // Rebuild with new children
    uop.replace().src(new_children).call()
}

/// Extract i64 value from ConstValue.
fn const_to_i64(cv: &ConstValue) -> Option<i64> {
    match cv {
        ConstValue::Int(v) => Some(*v),
        ConstValue::UInt(v) => Some(*v as i64),
        ConstValue::Bool(v) => Some(*v as i64),
        ConstValue::Float(v) => Some(*v as i64),
    }
}

/// Create GPU thread indices with dimension limiting.
///
/// Based on Tinygrad's `get_grouped_dims` (gpudims.py:28-57).
///
/// # Arguments
///
/// * `prefix` - Index name prefix ("gidx" or "lidx")
/// * `dims` - Dimension sizes as UOps
/// * `max_sizes` - Hardware limits per dimension (None = unlimited)
/// * `reverse` - Reverse dimension ordering (true for global indices)
///
/// # Returns
///
/// Vector of SPECIAL UOps representing thread indices.
fn get_grouped_dims(prefix: &str, dims: &[Arc<UOp>], max_sizes: Option<&[usize]>, reverse: bool) -> Vec<Arc<UOp>> {
    if dims.is_empty() {
        return vec![];
    }

    // Try to get concrete dimension values for grouping/splitting
    let concrete_dims: Option<Vec<usize>> = dims
        .iter()
        .map(|d| match d.op() {
            Op::Const(c) => const_to_i64(&c.0).map(|v| v as usize),
            _ => None,
        })
        .collect();

    // Apply dimension limiting if we have concrete values and max_sizes
    let limited_dims: Vec<usize> = match (&concrete_dims, max_sizes) {
        (Some(dims_vec), Some(max)) => limit_dims(dims_vec, max),
        (Some(dims_vec), None) => dims_vec.clone(),
        (None, _) => {
            // Symbolic dimensions: can't limit, just create SPECIAL for each
            return dims.iter().enumerate().map(|(i, d)| UOp::special(d.clone(), format!("{}{}", prefix, i))).collect();
        }
    };

    // Create raw indices as SPECIAL UOps
    let raw_idxs: Vec<Arc<UOp>> = limited_dims
        .iter()
        .enumerate()
        .map(|(i, &size)| UOp::special(UOp::index_const(size as i64), format!("{}{}", prefix, i)))
        .collect();

    // Handle dimension count mismatch
    let original_len = dims.len();
    let limited_len = limited_dims.len();

    let result = if limited_len < original_len {
        // Contraction: more original dims than limited dims
        // Need to decompose indices back to original dimension count
        decompose_contracted_dims(&raw_idxs, &limited_dims, concrete_dims.as_ref().unwrap())
    } else if limited_len > original_len {
        // Expansion: fewer original dims than limited dims
        // Need to combine indices to match original dimension count
        combine_expanded_dims(&raw_idxs, &limited_dims, concrete_dims.as_ref().unwrap())
    } else if limited_dims != *concrete_dims.as_ref().unwrap() {
        // Same count but different values: flatten and unflatten
        flatten_unflatten_dims(&raw_idxs, &limited_dims, concrete_dims.as_ref().unwrap())
    } else {
        raw_idxs
    };

    if reverse { result.into_iter().rev().collect() } else { result }
}

/// Limit dimensions to fit within hardware constraints.
///
/// Tries grouping first, then splitting if needed.
fn limit_dims(dims: &[usize], max_sizes: &[usize]) -> Vec<usize> {
    // First try grouping
    if let Some(grouped) = group_dims(dims, max_sizes) {
        return grouped;
    }

    // If grouping fails, try splitting
    split_dims(dims, max_sizes)
}

/// Group adjacent dimensions to fit within hardware limits.
///
/// Based on Tinygrad's `_group_dims` (gpudims.py:7-16).
fn group_dims(dims: &[usize], max_sizes: &[usize]) -> Option<Vec<usize>> {
    let mut result = dims.to_vec();

    // Keep trying to group until we fit or can't group anymore
    while result.len() > max_sizes.len() || result.iter().zip(max_sizes).any(|(d, m)| *d > *m) {
        let mut grouped = false;
        for i in 0..max_sizes.len().min(result.len().saturating_sub(1)) {
            if i + 1 < result.len() {
                let product = result[i].saturating_mul(result[i + 1]);
                if product <= max_sizes[i] {
                    // Merge dims[i] and dims[i+1]
                    result = result[..i]
                        .iter()
                        .chain(std::iter::once(&product))
                        .chain(result[i + 2..].iter())
                        .cloned()
                        .collect();
                    grouped = true;
                    break;
                }
            }
        }
        if !grouped {
            return None;
        }
    }

    Some(result)
}

/// Split dimensions that exceed hardware limits.
///
/// Based on Tinygrad's `_split_dims` (gpudims.py:18-26).
fn split_dims(dims: &[usize], max_sizes: &[usize]) -> Vec<usize> {
    // Pad to 3 dimensions (typical GPU max)
    let mut result: Vec<usize> = dims.to_vec();
    while result.len() < 3 {
        result.push(1);
    }

    // Split dimensions that exceed limits
    for i in 0..result.len() {
        let max = if i < max_sizes.len() { max_sizes[i] } else { usize::MAX };
        while result[i] > max {
            // Find smallest divisor
            let div = find_smallest_divisor(result[i]);
            if div == 1 {
                // Prime number that can't be split - give up
                break;
            }
            // Split: move factor to next dimension
            let next = (i + 1) % result.len();
            result[i] /= div;
            result[next] *= div;
        }
    }

    // Trim trailing 1s
    while result.len() > 1 && result.last() == Some(&1) {
        result.pop();
    }

    result
}

/// Find the smallest divisor of n (excluding 1).
fn find_smallest_divisor(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let sqrt_n = (n as f64).sqrt().ceil() as usize;
    for d in 2..=sqrt_n {
        if n.is_multiple_of(d) {
            return d;
        }
    }
    1 // n is prime
}

/// Decompose contracted dimensions back to original count.
///
/// When we grouped dimensions (limited_len < original_len), we need
/// to decompose the indices using divmod.
fn decompose_contracted_dims(raw_idxs: &[Arc<UOp>], limited_dims: &[usize], original_dims: &[usize]) -> Vec<Arc<UOp>> {
    // Get contraction mapping
    let contraction = get_contraction(original_dims, limited_dims);
    let Some(contraction) = contraction else {
        // Fallback: return raw indices
        return raw_idxs.to_vec();
    };

    let mut result: Vec<Arc<UOp>> = Vec::new();

    for (idx, group) in raw_idxs.iter().zip(&contraction) {
        let mut current = idx.clone();
        for &dim_idx in group.iter().rev().skip(1).collect::<Vec<_>>().into_iter().rev() {
            let dim_size = original_dims[dim_idx];
            let dim_uop = UOp::index_const(dim_size as i64);
            // Extract: result[dim_idx] = current % dim_size
            result.push(current.mod_(&dim_uop));
            // Shift: current = current / dim_size (integer division)
            current = current.idiv(&dim_uop);
        }
        result.push(current);
    }

    result
}

/// Get contraction mapping: which original dims map to each limited dim.
///
/// Uses accumulated product matching instead of greedy grouping.
/// This handles non-consecutive dimension groups like [2,5,2] → [10,2].
///
/// # Algorithm
///
/// 1. Compute accumulated products for both original and limited dims
/// 2. Find split points where accumulated products match
/// 3. Build index ranges from split points
///
/// # Example
///
/// ```text
/// original_dims = [2, 5, 2], limited_dims = [10, 2]
/// acc_old = [2, 10, 20]
/// acc_new = [10, 20]
/// split = [2, 3]  (indices where acc_old matches acc_new)
/// result = [[0, 1], [2]]  (dims 0,1 → limited 0; dim 2 → limited 1)
/// ```
fn get_contraction(original_dims: &[usize], limited_dims: &[usize]) -> Option<Vec<Vec<usize>>> {
    if original_dims.is_empty() && limited_dims.is_empty() {
        return Some(vec![]);
    }
    if limited_dims.is_empty() {
        return None;
    }

    // Accumulated products for original dims
    let acc_old: Vec<usize> = original_dims
        .iter()
        .scan(1usize, |s, &x| {
            *s = s.saturating_mul(x);
            Some(*s)
        })
        .collect();

    // Accumulated products for limited dims
    let acc_new: Vec<usize> = limited_dims
        .iter()
        .scan(1usize, |s, &x| {
            *s = s.saturating_mul(x);
            Some(*s)
        })
        .collect();

    // Find split points: for each accumulated product in acc_new,
    // find the index in acc_old that matches
    let mut split = Vec::with_capacity(acc_new.len());
    for &acc in &acc_new {
        if acc == 1 {
            // Special case: leading 1s don't consume any original dims
            split.push(0);
        } else {
            match acc_old.iter().position(|&x| x == acc) {
                Some(idx) => split.push(idx + 1), // +1 because we want the index AFTER the match
                None => return None,              // No valid contraction
            }
        }
    }

    // Build index ranges from split points
    let mut result = Vec::with_capacity(split.len());
    let mut prev = 0;
    for (i, &s) in split.iter().enumerate() {
        if i == split.len() - 1 {
            // Last group: take remaining indices
            result.push((prev..original_dims.len()).collect());
        } else {
            result.push((prev..s).collect());
            prev = s;
        }
    }

    Some(result)
}

/// Combine expanded dimensions to match original count.
fn combine_expanded_dims(raw_idxs: &[Arc<UOp>], limited_dims: &[usize], original_dims: &[usize]) -> Vec<Arc<UOp>> {
    let a = limited_dims.len();
    let b = original_dims.len();

    match (a, b) {
        (2, 1) => {
            // idx = raw_idxs[0] * limited_dims[1] + raw_idxs[1]
            let mul = raw_idxs[0].mul(&UOp::index_const(limited_dims[1] as i64));
            vec![mul.add(&raw_idxs[1])]
        }
        (3, 1) => {
            // idx = (raw_idxs[0] * limited_dims[1] + raw_idxs[1]) * limited_dims[2] + raw_idxs[2]
            let mul1 = raw_idxs[0].mul(&UOp::index_const(limited_dims[1] as i64));
            let add1 = mul1.add(&raw_idxs[1]);
            let mul2 = add1.mul(&UOp::index_const(limited_dims[2] as i64));
            vec![mul2.add(&raw_idxs[2])]
        }
        (3, 2) => {
            // idx0 = raw_idxs[0] * limited_dims[1] + raw_idxs[1]
            // idx1 = raw_idxs[2]
            let mul = raw_idxs[0].mul(&UOp::index_const(limited_dims[1] as i64));
            vec![mul.add(&raw_idxs[1]), raw_idxs[2].clone()]
        }
        _ => raw_idxs.to_vec(),
    }
}

/// Flatten and unflatten when dimensions are same count but different values.
fn flatten_unflatten_dims(raw_idxs: &[Arc<UOp>], limited_dims: &[usize], original_dims: &[usize]) -> Vec<Arc<UOp>> {
    let n = limited_dims.len();
    if n == 2 {
        // flat = raw_idxs[0] * limited_dims[1] + raw_idxs[1]
        let mul = raw_idxs[0].mul(&UOp::index_const(limited_dims[1] as i64));
        let flat = mul.add(&raw_idxs[1]);
        // unflatten
        let dim1_uop = UOp::index_const(original_dims[1] as i64);
        vec![flat.idiv(&dim1_uop), flat.mod_(&dim1_uop)]
    } else if n == 3 {
        // flat = raw_idxs[0] * (limited_dims[1] * limited_dims[2]) + raw_idxs[1] * limited_dims[2] + raw_idxs[2]
        let l12 = UOp::index_const((limited_dims[1] * limited_dims[2]) as i64);
        let l2 = UOp::index_const(limited_dims[2] as i64);
        let mul0 = raw_idxs[0].mul(&l12);
        let mul1 = raw_idxs[1].mul(&l2);
        let add0 = mul0.add(&mul1);
        let flat = add0.add(&raw_idxs[2]);
        // unflatten
        let d1 = original_dims[1];
        let d2 = original_dims[2];
        let d12 = UOp::index_const((d1 * d2) as i64);
        let d1_uop = UOp::index_const(d1 as i64);
        let d2_uop = UOp::index_const(d2 as i64);
        vec![flat.idiv(&d12), flat.idiv(&d2_uop).mod_(&d1_uop), flat.mod_(&d2_uop)]
    } else {
        raw_idxs.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_dims_already_fits() {
        // Dims already fit, no grouping needed
        let result = group_dims(&[4, 4], &[16, 16, 16]);
        assert_eq!(result, Some(vec![4, 4]));
    }

    #[test]
    fn test_group_dims_needs_grouping() {
        // 4 dims need to be grouped to fit into 3 max_sizes
        // [4, 4, 4, 4] can't fit directly into [256, 256, 256] (4 dims > 3 max_sizes)
        // Should group: [4*4, 4, 4] = [16, 4, 4]
        let result = group_dims(&[4, 4, 4, 4], &[256, 256, 256]);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_group_dims_no_change() {
        // Dims already fit
        let result = group_dims(&[8, 8, 8], &[256, 256, 256]);
        assert_eq!(result, Some(vec![8, 8, 8]));
    }

    #[test]
    fn test_group_dims_impossible() {
        // Can't fit 1000 into max 10
        let result = group_dims(&[1000], &[10]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_split_dims_simple() {
        // 100 exceeds 64, should split
        let result = split_dims(&[100], &[64, 64, 64]);
        // 100 / 2 = 50, then 50 / 2 = 25 fits
        assert!(result.iter().all(|&d| d <= 64));
    }

    #[test]
    fn test_find_smallest_divisor() {
        assert_eq!(find_smallest_divisor(1), 1);
        assert_eq!(find_smallest_divisor(2), 2); // 2 is the smallest divisor of 2 (excluding 1)
        assert_eq!(find_smallest_divisor(3), 1); // 3 is prime
        assert_eq!(find_smallest_divisor(4), 2);
        assert_eq!(find_smallest_divisor(9), 3);
        assert_eq!(find_smallest_divisor(100), 2);
    }

    #[test]
    fn test_get_contraction_non_consecutive() {
        // [2, 5, 2] → [10, 2]: dims 0,1 fuse to 10; dim 2 stays as 2
        // acc_old = [2, 10, 20], acc_new = [10, 20]
        let result = get_contraction(&[2, 5, 2], &[10, 2]);
        assert_eq!(result, Some(vec![vec![0, 1], vec![2]]));
    }

    #[test]
    fn test_get_contraction_identity() {
        // [4, 4, 4] → [4, 4, 4]: no grouping
        let result = get_contraction(&[4, 4, 4], &[4, 4, 4]);
        assert_eq!(result, Some(vec![vec![0], vec![1], vec![2]]));
    }

    #[test]
    fn test_get_contraction_all_fused() {
        // [2, 3, 4] → [24]: all dims fuse to one
        let result = get_contraction(&[2, 3, 4], &[24]);
        assert_eq!(result, Some(vec![vec![0, 1, 2]]));
    }

    #[test]
    fn test_get_contraction_empty() {
        let result = get_contraction(&[], &[]);
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn test_get_contraction_invalid() {
        // [2, 3, 4] → [5, 4]: 2*3 = 6 != 5
        let result = get_contraction(&[2, 3, 4], &[5, 4]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_get_contraction_partial() {
        // [2, 4, 3] → [8, 3]: dims 0,1 fuse to 8; dim 2 stays as 3
        let result = get_contraction(&[2, 4, 3], &[8, 3]);
        assert_eq!(result, Some(vec![vec![0, 1], vec![2]]));
    }
}
