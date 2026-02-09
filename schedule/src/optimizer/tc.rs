//! Tensor Core (TC) optimization - Hardware-accelerated matrix multiplication.
//!
//! Implements pattern matching, selection, swizzle, and application for tensor core ops.
//! Supports NVIDIA (WMMA), AMD (Matrix Cores), Intel, and Apple (AMX) hardware.

use std::collections::HashMap;
use std::sync::Arc;

use morok_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, ReduceOp, UOp, UOpKey, WmmaMetadata, WmmaUpcastAxes};
use smallvec::{SmallVec, smallvec};

use crate::optimizer::{
    Renderer, Scheduler,
    error::*,
    renderer::{SwizzleAxis, TcOpt, TensorCore},
};

// ============================================================================
// PATTERN MATCHING
// ============================================================================

/// Information about a detected matmul pattern.
#[derive(Debug, Clone)]
pub struct MatmulPattern {
    pub reduce_op: Arc<UOp>,
    pub in0: Arc<UOp>,
    pub in1: Arc<UOp>,
    pub in0_ranges: Vec<Arc<UOp>>,
    pub in1_ranges: Vec<Arc<UOp>>,
    pub red_ranges: Vec<Arc<UOp>>,
    pub axis_choices: Vec<(Arc<UOp>, Arc<UOp>, Arc<UOp>)>,
}

/// Detect matmul pattern: REDUCE(ADD, MUL(in0, in1), ...reduce_ranges)
pub fn detect_matmul(scheduler: &Scheduler) -> Result<Option<MatmulPattern>, OptError> {
    let reduce_op = match scheduler.reduceop() {
        Some(op) => op,
        None => return Ok(None),
    };

    let Op::Reduce { reduce_op: reduce_type, ranges: _, src } = reduce_op.op() else {
        return Ok(None);
    };

    if *reduce_type != ReduceOp::Add {
        return Ok(None);
    }

    // Extract MUL operation (possibly under CAST)
    let mul = if let Op::Cast { src: cast_src, .. } = src.op() { cast_src.clone() } else { src.clone() };

    let Op::Binary(BinaryOp::Mul, a, b) = mul.op() else {
        return Ok(None);
    };

    let (in0, in1) = (a.clone(), b.clone());
    let in0_all_ranges = get_ranges(&in0);
    let in1_all_ranges = get_ranges(&in1);

    let red_ranges: Vec<_> =
        if let Op::Reduce { ranges, .. } = reduce_op.op() { ranges.iter().cloned().collect() } else { vec![] };

    // Find unique ranges (M and N dimensions)
    let in0_ranges: Vec<_> =
        in0_all_ranges.iter().filter(|r| !in1_all_ranges.iter().any(|r2| Arc::ptr_eq(r, r2))).cloned().collect();

    let in1_ranges: Vec<_> =
        in1_all_ranges.iter().filter(|r| !in0_all_ranges.iter().any(|r2| Arc::ptr_eq(r, r2))).cloned().collect();

    // Sort by axis_id descending
    let mut in0_ranges = in0_ranges;
    let mut in1_ranges = in1_ranges;
    let mut red_ranges = red_ranges;
    in0_ranges.sort_by_key(|r| std::cmp::Reverse(get_axis_id(r)));
    in1_ranges.sort_by_key(|r| std::cmp::Reverse(get_axis_id(r)));
    red_ranges.sort_by_key(|r| std::cmp::Reverse(get_axis_id(r)));

    // Generate all axis choices (N, M, K) using explicit loops to avoid closure ownership issues
    let mut axis_choices = Vec::with_capacity(in1_ranges.len() * in0_ranges.len() * red_ranges.len());
    for n in &in1_ranges {
        for m in &in0_ranges {
            for k in &red_ranges {
                axis_choices.push((n.clone(), m.clone(), k.clone()));
            }
        }
    }

    if axis_choices.is_empty() {
        return Ok(None);
    }

    Ok(Some(MatmulPattern { reduce_op, in0, in1, in0_ranges, in1_ranges, red_ranges, axis_choices }))
}

fn get_ranges(uop: &Arc<UOp>) -> Vec<Arc<UOp>> {
    uop.backward_slice().into_iter().filter(|node| matches!(node.op(), Op::Range { .. })).collect()
}

fn get_axis_id(range: &Arc<UOp>) -> usize {
    if let Op::Range { axis_id, .. } = range.op() { axis_id.value() } else { 0 }
}

fn get_range_size(range: &Arc<UOp>) -> Option<i64> {
    if let Op::Range { end, .. } = range.op()
        && let Op::Const(cv) = end.op()
        && let ConstValue::Int(size) = cv.0
    {
        return Some(size);
    }
    None
}

// ============================================================================
// SELECTION
// ============================================================================

/// Result of tensor core selection.
#[derive(Debug, Clone)]
pub struct TcSelection {
    pub tc_index: usize,
    pub axes: (Arc<UOp>, Arc<UOp>, Arc<UOp>),
}

/// Select appropriate tensor core for the given matmul pattern.
pub fn select_tensor_core(
    pattern: &MatmulPattern,
    renderer: &Renderer,
    tc_select: i32,
    axis_choice: usize,
) -> Result<Option<TcSelection>, OptError> {
    let tensor_cores = if tc_select == -1 {
        &renderer.tensor_cores[..]
    } else {
        let idx = tc_select as usize;
        if idx >= renderer.tensor_cores.len() {
            return ValidationFailedSnafu { op: "TC", reason: "tc_select index out of bounds" }.fail();
        }
        &renderer.tensor_cores[idx..idx + 1]
    };

    let (in0_scalar, in1_scalar, out_scalar) =
        (pattern.in0.dtype().scalar(), pattern.in1.dtype().scalar(), pattern.reduce_op.dtype().scalar());

    for (tc_idx, tc) in tensor_cores.iter().enumerate() {
        let (tc_in_scalar, tc_out_scalar) = (tc.dtype_in.scalar(), tc.dtype_out.scalar());

        if in0_scalar != tc_in_scalar || in1_scalar != tc_in_scalar || out_scalar != tc_out_scalar {
            continue;
        }

        if axis_choice >= pattern.axis_choices.len() {
            continue;
        }

        let axes = pattern.axis_choices[axis_choice].clone();

        let actual_tc_idx = if tc_select == -1 {
            renderer.tensor_cores.iter().position(|t| std::ptr::eq(t, tc)).unwrap_or(tc_idx)
        } else {
            tc_select as usize
        };

        return Ok(Some(TcSelection { tc_index: actual_tc_idx, axes }));
    }

    Ok(None)
}

// ============================================================================
// SWIZZLE
// ============================================================================

/// Generate the base shape from tensor core opts.
pub fn base_shape(tc: &TensorCore) -> Vec<SwizzleAxis> {
    let reduce_count = (tc.dims.2 as f64).log2().floor() as usize;
    let mut ret = Vec::with_capacity(tc.opts.len() + reduce_count);
    let (mut u_cnt, mut l_cnt) = (0, 0);

    for opt in &tc.opts {
        match opt {
            TcOpt::Upcast(_) => {
                ret.push(SwizzleAxis::Upcast(u_cnt));
                u_cnt += 1;
            }
            TcOpt::Local(_) => {
                ret.push(SwizzleAxis::Local(l_cnt));
                l_cnt += 1;
            }
        }
    }
    for i in 0..reduce_count {
        ret.push(SwizzleAxis::Reduce(i));
    }
    ret
}

fn generate_remaps(tc: &TensorCore) -> Vec<HashMap<SwizzleAxis, SwizzleAxis>> {
    let local_count = tc.opts.iter().filter(|opt| opt.is_local()).count();
    let upcast_count = tc.opts.iter().filter(|opt| opt.is_upcast()).count();
    let reduce_count = (tc.dims.2 as f64).log2().floor() as usize;

    let mut fwd_shape = Vec::with_capacity(local_count + upcast_count + reduce_count);
    (0..local_count).for_each(|i| fwd_shape.push(SwizzleAxis::Local(i)));
    (0..upcast_count).for_each(|i| fwd_shape.push(SwizzleAxis::Upcast(i)));
    (0..reduce_count).for_each(|i| fwd_shape.push(SwizzleAxis::Reduce(i)));

    [&tc.swizzle.0, &tc.swizzle.1]
        .iter()
        .map(|part| {
            let mut flattened = Vec::new();
            flattened.extend_from_slice(&part.0);
            flattened.extend_from_slice(&part.1);
            flattened.extend_from_slice(&part.2);

            fwd_shape.iter().enumerate().filter_map(|(i, &key)| flattened.get(i).map(|&v| (key, v))).collect()
        })
        .collect()
}

/// Compute permutation indices for the given shape.
pub fn permutes_for_shape(tc: &TensorCore, shape: &[SwizzleAxis]) -> (Vec<usize>, Vec<usize>) {
    let remaps = generate_remaps(tc);
    let perms: Vec<Vec<usize>> = remaps
        .iter()
        .map(|remap| {
            shape
                .iter()
                .enumerate()
                .map(|(i, &axis)| remap.get(&axis).and_then(|&r| shape.iter().position(|&s| s == r)).unwrap_or(i))
                .collect()
        })
        .collect();

    (perms[0].clone(), perms[1].clone())
}

/// Get the number of reduce axes for the tensor core (log2 of K dimension).
pub fn get_reduce_axes_count(tc: &TensorCore) -> usize {
    (tc.dims.2 as f64).log2().floor() as usize
}

/// Compute inverse permutation.
fn argsort(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

// ============================================================================
// A-TILE PACKING
// ============================================================================

/// Pre-pack a TC operand into a contiguous scratch buffer.
///
/// When operand A has strided memory access (e.g., row-major A in AMX matmul),
/// each K iteration requires `tile_size` separate cache line accesses. This function
/// creates a copy loop that packs the tile into contiguous memory, so the reduction
/// loop reads one cache line per K iteration instead of `tile_size`.
///
/// The copy loop uses fresh RANGE nodes (distinct axis_ids) so it becomes a separate
/// loop from the downstream reduction. An AFTER dependency ensures correct ordering.
fn pack_tc_operand(
    src: &Arc<UOp>,
    reduce_range: &Arc<UOp>,
    contract_ranges: &[&Arc<UOp>],
    next_axis_id: &mut usize,
) -> Result<Arc<UOp>, OptError> {
    // 1. Compute buffer dimensions
    let k_size = get_range_size(reduce_range).expect("ICE: reduce range must have const size") as usize;
    let contract_sizes: Vec<usize> = contract_ranges
        .iter()
        .map(|r| get_range_size(r).expect("ICE: contract range must have const size") as usize)
        .collect();
    let tile_size: usize = contract_sizes.iter().product();
    let buf_total = k_size * tile_size;
    let element_dtype = src.dtype().scalar_dtype();

    // 2. Create scratch buffer (register-allocated)
    let buf = UOp::define_reg_typed(buf_total, element_dtype);

    // 3. Create fresh RANGE nodes for the copy loop (2 loops: K × tile_size)
    let k_end = match reduce_range.op() {
        Op::Range { end, .. } => end.clone(),
        _ => unreachable!(),
    };
    let k_clone = UOp::range_axis(k_end, AxisId::Renumbered(*next_axis_id), AxisType::Loop);
    *next_axis_id += 1;

    // Single flat range for the entire tile (replaces N nested binary ranges)
    let m_flat = UOp::range_axis(UOp::index_const(tile_size as i64), AxisId::Renumbered(*next_axis_id), AxisType::Loop);
    *next_axis_id += 1;

    // 4. Substitute original ranges → decomposed sub-indices of m_flat in src expression
    //
    // The contract_ranges are N binary (size-2) Upcast ranges from shift_to splits.
    // The src expression references them individually. We decompose m_flat back into
    // sub-indices: sub_idx[i] = (m_flat / contract_strides[i]) % contract_sizes[i]
    let contract_dims: Vec<i64> = contract_sizes.iter().map(|&s| s as i64).collect();
    let contract_strides = crate::passes::linearize_index::compute_row_major_strides(&contract_dims);

    #[allow(clippy::mutable_key_type)]
    let subst: HashMap<UOpKey, Arc<UOp>> = {
        let mut map = HashMap::with_capacity(1 + contract_ranges.len());
        map.insert(UOpKey(reduce_range.clone()), k_clone.clone());
        for (i, orig) in contract_ranges.iter().enumerate() {
            let sub_idx = if contract_strides[i] == 1 {
                m_flat
                    .try_mod(&UOp::index_const(contract_sizes[i] as i64))
                    .map_err(|_| ValidationFailedSnafu { op: "TC pack", reason: "sub-index mod failed" }.build())?
            } else {
                let divided = m_flat
                    .try_div(&UOp::index_const(contract_strides[i]))
                    .map_err(|_| ValidationFailedSnafu { op: "TC pack", reason: "sub-index div failed" }.build())?;
                divided
                    .try_mod(&UOp::index_const(contract_sizes[i] as i64))
                    .map_err(|_| ValidationFailedSnafu { op: "TC pack", reason: "sub-index mod failed" }.build())?
            };
            map.insert(UOpKey((*orig).clone()), sub_idx);
        }
        map
    };
    let src_cloned = src.substitute(&subst);

    // 5. Store: buf[k_clone * tile_size + m_flat] = src_cloned
    let tile_size_const = UOp::index_const(tile_size as i64);
    let store_idx = k_clone
        .try_mul(&tile_size_const)
        .and_then(|k_offset| k_offset.try_add(&m_flat))
        .map_err(|_| ValidationFailedSnafu { op: "TC pack", reason: "store index creation failed" }.build())?;

    let store_ptr = UOp::index()
        .buffer(buf.clone())
        .indices(vec![store_idx])
        .ptr(true)
        .call()
        .map_err(|_| ValidationFailedSnafu { op: "TC pack", reason: "store index creation failed" }.build())?;
    let store = store_ptr.store(src_cloned);

    let end = store.end(smallvec![k_clone, m_flat]);
    let buf_ready = buf.after(smallvec![end]);

    // 6. Read: LOAD(INDEX(buf_ready, [k * tile_size + m_linear])) using ORIGINAL ranges
    let read_dims: Vec<i64> = std::iter::once(k_size as i64).chain(contract_sizes.iter().map(|&s| s as i64)).collect();
    let read_strides = crate::passes::linearize_index::compute_row_major_strides(&read_dims);
    let read_indices: Vec<Arc<UOp>> =
        std::iter::once(reduce_range.clone()).chain(contract_ranges.iter().map(|r| (*r).clone())).collect();
    let read_idx = crate::passes::linearize_index::build_linear_index(&read_indices, &read_strides);

    let read_ptr = UOp::index()
        .buffer(buf_ready.clone())
        .indices(vec![read_idx])
        .ptr(true)
        .call()
        .map_err(|_| ValidationFailedSnafu { op: "TC pack", reason: "read index creation failed" }.build())?;

    Ok(UOp::load().buffer(buf_ready).index(read_ptr).call())
}

// ============================================================================
// APPLICATION
// ============================================================================

/// Apply tensor core optimization to the scheduler.
pub fn apply(
    scheduler: &mut Scheduler,
    tc_select: i32,
    tc_opt: usize,
    use_tensor_cores: usize,
) -> Result<[Arc<UOp>; 3], OptError> {
    // Validate
    if !scheduler.applied_opts.is_empty() {
        return ValidationFailedSnafu { op: "TC", reason: "tensor core opts must be first" }.fail();
    }
    if use_tensor_cores == 0 || use_tensor_cores > 2 {
        return ValidationFailedSnafu { op: "TC", reason: "use_tensor_cores must be 1 or 2" }.fail();
    }
    if tc_opt > 2 {
        return ValidationFailedSnafu { op: "TC", reason: "tc_opt must be 0, 1, or 2" }.fail();
    }
    if tc_select < -1 {
        return ValidationFailedSnafu { op: "TC", reason: "tc_select must be >= -1" }.fail();
    }

    // Detect pattern
    let pattern = detect_matmul(scheduler)?
        .ok_or_else(|| ValidationFailedSnafu { op: "TC", reason: "no matmul pattern detected" }.build())?;

    // Select tensor core
    let tc_selection = (0..pattern.axis_choices.len())
        .find_map(|axis_choice| select_tensor_core(&pattern, &scheduler.ren, tc_select, axis_choice).ok().flatten())
        .ok_or_else(|| ValidationFailedSnafu { op: "TC", reason: "no compatible tensor core found" }.build())?;

    // Clone the TensorCore to avoid borrow conflicts when applying PADTO
    let tc = scheduler.ren.tensor_cores[tc_selection.tc_index].clone();
    let (n_range, m_range, k_range) = &tc_selection.axes;
    // Mutable axes array - may be updated after PADTO
    let mut axes = [n_range.clone(), m_range.clone(), k_range.clone()];

    // Padding check and application (tc_opt >= 2)
    // When tc_opt >= 2, we use PADTO to align non-divisible dimensions
    // instead of rejecting them outright.
    // Track whether any axis was padded — if so, B operand needs packing
    // because PADTO gates break devectorization of the B source expression.
    let mut padded = false;

    if tc_opt >= 2 {
        // Collect padding operations needed (can't mutate axes while iterating)
        let tc_dims = [tc.dims.0, tc.dims.1, tc.dims.2];
        let mut padding_ops: Vec<(usize, usize, usize)> = Vec::new(); // (axes_idx, scheduler_idx, tc_dim)

        for (i, (axis, &tc_dim)) in axes.iter().zip(&tc_dims).enumerate() {
            let dim_size = get_range_size(axis);
            if let Some(size) = dim_size
                && !(size as usize).is_multiple_of(tc_dim)
            {
                let axis_idx = scheduler.rngs().iter().position(|r| Arc::ptr_eq(r, axis)).ok_or_else(|| {
                    ValidationFailedSnafu { op: "TC", reason: "axis not found in scheduler ranges" }.build()
                })?;
                padding_ops.push((i, axis_idx, tc_dim));
            }
        }

        padded = !padding_ops.is_empty();

        // Apply padding operations sequentially
        for (axes_idx, scheduler_idx, tc_dim) in padding_ops {
            crate::optimizer::opts::apply_opt(scheduler, &crate::optimizer::Opt::padto(scheduler_idx, tc_dim), false)
                .map_err(|_| {
                ValidationFailedSnafu {
                    op: "TC",
                    reason: "padding failed (may exceed 4x work limit or have unsafe ops)",
                }
                .build()
            })?;

            // Update axes to the new padded range (PADTO substitutes the old range)
            axes[axes_idx] = scheduler.rngs()[scheduler_idx].clone();
        }
    } else {
        // Without tc_opt >= 2, reject non-divisible dimensions
        for (i, axis) in axes.iter().enumerate() {
            let dim_size = get_range_size(axis);
            let tc_dim = match i {
                0 => tc.dims.0,
                1 => tc.dims.1,
                _ => tc.dims.2,
            };
            if let Some(size) = dim_size
                && !(size as usize).is_multiple_of(tc_dim)
            {
                return ValidationFailedSnafu { op: "TC", reason: "dimension not divisible by tensor core size" }
                    .fail();
            }
        }
    }

    // Create WARP dimension
    let mut warp = UOp::range_axis(
        UOp::const_(morok_dtype::DType::Index, ConstValue::Int(tc.threads as i64)),
        AxisId::Renumbered(scheduler.maxarg() + 1),
        AxisType::Warp,
    );

    // Step 1: Apply TC opts via shift_to — splits each axis into (reduced, new_rng)
    let two = UOp::const_(morok_dtype::DType::Index, ConstValue::Int(2));
    let mut ne: Vec<Arc<UOp>> = Vec::with_capacity(tc.opts.len());

    for opt in &tc.opts {
        match opt {
            TcOpt::Upcast(dim) => {
                let (replaced, new_rng) = scheduler.shift_to(axes[*dim].clone(), 2, AxisType::Upcast, false, None)?;
                axes[*dim] = replaced;
                ne.push(new_rng);
            }
            TcOpt::Local(dim) => {
                let warp_mod = warp
                    .try_mod(&two)
                    .map_err(|_| ValidationFailedSnafu { op: "TC", reason: "warp mod failed" }.build())?;
                let (replaced, new_rng) =
                    scheduler.shift_to(axes[*dim].clone(), 2, AxisType::Local, false, Some(warp_mod))?;
                axes[*dim] = replaced;
                warp = warp
                    .try_div(&two)
                    .map_err(|_| ValidationFailedSnafu { op: "TC", reason: "warp div failed" }.build())?;
                ne.push(new_rng);
            }
        }
    }

    // K-dimension UNROLL splits
    for (_idx, amt) in tc.get_reduce_axes() {
        let (replaced, new_rng) = scheduler.shift_to(axes[2].clone(), amt, AxisType::Unroll, false, None)?;
        axes[2] = replaced;
        ne.push(new_rng);
    }

    // Build WMMA UOp (if use_tensor_cores == 1)
    if use_tensor_cores == 1 {
        // Step 2: Re-extract sources from updated AST
        let updated_reduce = scheduler
            .reduceop()
            .ok_or_else(|| ValidationFailedSnafu { op: "TC", reason: "REDUCE missing after shift_to" }.build())?;

        // Validate that the REDUCE still contains MUL pattern after shift_to
        let reduce_src = match updated_reduce.op() {
            Op::Reduce { src, .. } => src.clone(),
            _ => unreachable!(),
        };
        let mul = match reduce_src.op() {
            Op::Cast { src, .. } => src.clone(),
            _ => reduce_src.clone(),
        };
        if !matches!(mul.op(), Op::Binary(BinaryOp::Mul, ..)) {
            return ValidationFailedSnafu { op: "TC", reason: "expected MUL inside REDUCE" }.fail();
        }

        // Step 3: Apply swizzle permutation via placeholders
        let bshape = base_shape(&tc);
        let (perm_a, perm_b) = permutes_for_shape(&tc, &bshape);
        let inv_a = argsort(&perm_a);
        let inv_b = argsort(&perm_b);

        // Create placeholder UOps with unique axis_ids
        let ph_base = scheduler.maxarg() + 100;
        let placeholders: Vec<Arc<UOp>> = (0..ne.len())
            .map(|i| {
                UOp::range_axis(
                    UOp::const_(morok_dtype::DType::Index, ConstValue::Int(2)),
                    AxisId::Renumbered(ph_base + i),
                    AxisType::Upcast,
                )
            })
            .collect();

        // Substitute ne → placeholders in REDUCE subtree
        #[allow(clippy::mutable_key_type)]
        let subst_to_ph: HashMap<UOpKey, Arc<UOp>> =
            ne.iter().zip(&placeholders).map(|(n, ph)| (UOpKey(n.clone()), ph.clone())).collect();
        let ret = updated_reduce.substitute(&subst_to_ph);

        // Re-extract sources from substituted REDUCE
        let ret_src = match ret.op() {
            Op::Reduce { src, .. } => src.clone(),
            _ => unreachable!(),
        };
        let ret_mul = match ret_src.op() {
            Op::Cast { src, .. } => src.clone(),
            _ => ret_src.clone(),
        };
        let (ret_a, ret_b) = match ret_mul.op() {
            Op::Binary(BinaryOp::Mul, a, b) => (a.clone(), b.clone()),
            _ => unreachable!(),
        };

        // Substitute placeholders → permuted ne for each source
        #[allow(clippy::mutable_key_type)]
        let subst_a: HashMap<UOpKey, Arc<UOp>> =
            placeholders.iter().enumerate().map(|(i, ph)| (UOpKey(ph.clone()), ne[inv_a[i]].clone())).collect();
        #[allow(clippy::mutable_key_type)]
        let subst_b: HashMap<UOpKey, Arc<UOp>> =
            placeholders.iter().enumerate().map(|(i, ph)| (UOpKey(ph.clone()), ne[inv_b[i]].clone())).collect();

        let src_a = ret_a.substitute(&subst_a);
        let src_b = ret_b.substitute(&subst_b);

        // Step 4: Build tc_upcast_axes from ne ranges
        //
        // `ne` mirrors `tc.opts` order (upcast and local interleaved), with reduce
        // entries appended after `ne[tc.opts.len()..]`. We must filter by opt type
        // to extract only upcast entries, not assume positional layout.
        let upcast_ne: Vec<&Arc<UOp>> =
            tc.opts.iter().zip(ne.iter()).filter(|(opt, _)| opt.is_upcast()).map(|(_, rng)| rng).collect();
        let reduce_ne: Vec<&Arc<UOp>> = ne[tc.opts.len()..].iter().collect();

        // base_upcast_ne: reversed([reduce, upcast]) = [upcast_reversed, reduce_reversed]
        let mut base_upcast_ne: Vec<&Arc<UOp>> = Vec::new();
        base_upcast_ne.extend(&reduce_ne);
        base_upcast_ne.extend(&upcast_ne);
        base_upcast_ne.reverse();

        let base_upcast_axes: Vec<(usize, usize)> = base_upcast_ne
            .iter()
            .map(|rng| match rng.op() {
                Op::Range { axis_id, .. } => (axis_id.value(), 2),
                _ => unreachable!(),
            })
            .collect();

        // Slice by log2(elements_per_thread)
        let n_a = (tc.elements_per_thread.0 as f64).log2() as usize;
        let n_b = (tc.elements_per_thread.1 as f64).log2() as usize;
        let n_c = (tc.elements_per_thread.2 as f64).log2() as usize;
        let a_axes = base_upcast_axes[..n_a].to_vec();
        let b_axes = base_upcast_axes[..n_b].to_vec();
        let c_axes = base_upcast_axes[..n_c].to_vec();

        // Pack operand A if configured (AMX: contiguous scratch buffer for strided access)
        let mut next_axis_id = scheduler.maxarg() + 200;
        let src_a = if tc.pack_a {
            let contract_range_refs: Vec<&Arc<UOp>> = base_upcast_ne[..n_a].to_vec();
            pack_tc_operand(&src_a, &axes[2], &contract_range_refs, &mut next_axis_id)?
        } else {
            src_a
        };

        // Pack operand B when PADTO was applied — PADTO gates break devectorization
        // by creating per-element validity masks that prevent merging into contiguous
        // vector loads. Packing B into a scratch buffer resolves this: the copy loop
        // handles gated reads at the scalar level, and WMMA reads from contiguous memory.
        let src_b = if padded {
            let contract_range_refs: Vec<&Arc<UOp>> = base_upcast_ne[..n_b].to_vec();
            pack_tc_operand(&src_b, &axes[2], &contract_range_refs, &mut next_axis_id)?
        } else {
            src_b
        };

        // Step 5: Construct WMMA
        // Compute TC reduce axis IDs early (needed for metadata)
        let tc_reduce_aids: Vec<usize> = ne[tc.opts.len()..]
            .iter()
            .filter_map(|r| match r.op() {
                Op::Range { axis_id, .. } => Some(axis_id.value()),
                _ => None,
            })
            .collect();

        let metadata = WmmaMetadata {
            name: format!(
                "WMMA_{}_{}_{}_{}_{}",
                tc.dims.0,
                tc.dims.1,
                tc.dims.2,
                wmma_dtype_name(&tc.dtype_in),
                wmma_dtype_name(&tc.dtype_out),
            ),
            dims: tc.dims,
            dtype_in: tc.dtype_in.clone(),
            dtype_out: tc.dtype_out.clone(),
            device: scheduler.ren.device.clone(),
            threads: tc.threads,
            upcast_axes: WmmaUpcastAxes { a: a_axes.clone(), b: b_axes.clone(), c: c_axes.clone() },
            reduce_axes: tc_reduce_aids.clone(),
            tile_grid: tc.tile_grid,
        };

        let a_contract = src_a.contract(a_axes);
        let b_contract = src_b.contract(b_axes);
        let zero_acc = if tc.dtype_out.is_float() {
            UOp::const_(tc.dtype_out.clone(), ConstValue::Float(0.0))
        } else {
            UOp::const_(tc.dtype_out.clone(), ConstValue::Int(0))
        };
        let wmma = UOp::wmma(a_contract, b_contract, zero_acc, metadata);
        let mut tc_uop = wmma.unroll_with_dtype(c_axes, tc.dtype_out.clone());

        // Preserve extra reduce ranges (exclude TC reduce axis_ids)
        if let Op::Reduce { ranges, .. } = updated_reduce.op() {
            let extra: SmallVec<[Arc<UOp>; 4]> = ranges
                .iter()
                .filter(|r| match r.op() {
                    Op::Range { axis_id, .. } => !tc_reduce_aids.contains(&axis_id.value()),
                    _ => false,
                })
                .cloned()
                .collect();
            if !extra.is_empty() {
                tc_uop = tc_uop.reduce(extra, ReduceOp::Add);
            }
        }

        // Substitute REDUCE → WMMA chain in the AST
        #[allow(clippy::mutable_key_type)]
        let mut subst_map = HashMap::new();
        subst_map.insert(UOpKey(updated_reduce), tc_uop);
        let new_ast = scheduler.ast().substitute(&subst_map);
        scheduler.set_ast(new_ast);
    }

    Ok(axes)
}

/// Short dtype name for WMMA function identifiers (matches Tinygrad convention).
fn wmma_dtype_name(dtype: &morok_ir::prelude::DType) -> &'static str {
    use morok_dtype::ScalarDType;
    match dtype.base() {
        ScalarDType::Float32 => "float",
        ScalarDType::Float16 => "half",
        ScalarDType::BFloat16 => "bfloat",
        ScalarDType::Float64 => "double",
        ScalarDType::Int32 => "int",
        ScalarDType::Int8 => "int8",
        _ => "unknown",
    }
}

// ============================================================================
// MODULE SHIMS (backwards compatibility for tests)
// ============================================================================

/// Pattern matching functions (was opts::tc::matching).
pub mod matching {
    pub use super::{MatmulPattern, detect_matmul};
}

/// Selection functions (was opts::tc::selection).
pub mod selection {
    pub use super::{TcSelection, select_tensor_core};
}

/// Swizzle functions (was opts::tc::swizzle).
pub mod swizzle {
    pub use super::{base_shape, get_reduce_axes_count, permutes_for_shape};
}
