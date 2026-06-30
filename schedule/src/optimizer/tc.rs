//! Tensor Core (TC) optimization - Hardware-accelerated matrix multiplication.
//!
//! Implements pattern matching, selection, swizzle, and application for tensor core ops.
//! Supports NVIDIA (WMMA), AMD (Matrix Cores), Intel, and Apple (AMX) hardware.

use std::collections::HashMap;
use std::sync::Arc;

use smallvec::SmallVec;
use svod_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, ReduceOp, UOp, UOpKey, WmmaMetadata, WmmaUpcastAxes};

use crate::argsort;
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

    // Use `.scalar()` (returns Option) instead of `.base()` so Image dtypes
    // don't silently masquerade as Float32 (`base()` maps `Image` → Float32).
    // Reject Image dtypes outright — TCs operate on plain Scalar/Vector dtypes.
    let in0_dt = &pattern.in0.dtype();
    let in1_dt = &pattern.in1.dtype();
    let out_dt = &pattern.reduce_op.dtype();
    if in0_dt.is_image() || in1_dt.is_image() || out_dt.is_image() {
        return Ok(None);
    }
    let Some(in0_scalar) = in0_dt.scalar() else { return Ok(None) };
    let Some(in1_scalar) = in1_dt.scalar() else { return Ok(None) };
    let Some(out_scalar) = out_dt.scalar() else { return Ok(None) };

    for (tc_idx, tc) in tensor_cores.iter().enumerate() {
        if tc.dtype_in.is_image() || tc.dtype_out.is_image() {
            continue;
        }
        let (Some(tc_in_scalar), Some(tc_out_scalar)) = (tc.dtype_in.scalar(), tc.dtype_out.scalar()) else {
            continue;
        };

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

// ============================================================================
// APPLICATION
// ============================================================================

fn apply_axis_choice_impl(
    scheduler: &mut Scheduler,
    pattern: &MatmulPattern,
    tc_select: i32,
    tc_opt: usize,
    use_tensor_cores: usize,
    axis_choice: usize,
) -> Result<[Arc<UOp>; 3], OptError> {
    let tc_selection = select_tensor_core(pattern, &scheduler.ren, tc_select, axis_choice)?
        .ok_or_else(|| ValidationFailedSnafu { op: "TC", reason: "no compatible tensor core found" }.build())?;

    // Record which TC was actually picked; beam's `validate_limits` reads
    // this to compute the correct `tc_up` divisor when the renderer offers
    // multiple TC variants.
    scheduler.selected_tc_index = Some(tc_selection.tc_index);

    // Clone the TensorCore to avoid borrow conflicts when applying PADTO
    let tc = scheduler.ren.tensor_cores[tc_selection.tc_index].clone();
    let (n_range, m_range, k_range) = &tc_selection.axes;
    // Mutable axes array - may be updated after PADTO
    let mut axes = [n_range.clone(), m_range.clone(), k_range.clone()];

    // Padding check and application (tc_opt >= 2)
    // When tc_opt >= 2, we use PADTO to align non-divisible dimensions
    // instead of rejecting them outright.
    if tc_opt >= 2 {
        // Collect padding operations needed (can't mutate axes while iterating)
        let tc_dims = [tc.dims.0, tc.dims.1, tc.dims.2];
        let mut padding_ops: Vec<(usize, usize, usize)> = Vec::new(); // (axes_idx, scheduler_idx, tc_dim)

        for (i, (axis, &tc_dim)) in axes.iter().zip(&tc_dims).enumerate() {
            match get_range_size(axis) {
                Some(size) => {
                    if !(size as usize).is_multiple_of(tc_dim) {
                        let axis_idx = scheduler.rngs().iter().position(|r| Arc::ptr_eq(r, axis)).ok_or_else(|| {
                            ValidationFailedSnafu { op: "TC", reason: "axis not found in scheduler ranges" }.build()
                        })?;
                        padding_ops.push((i, axis_idx, tc_dim));
                    }
                }
                // PADTO can't pad an unknown extent, and even a provably
                // divisible symbolic axis fires pathological tilings in the
                // divisibility-keyed heuristics — symbolic axes never TC.
                None => {
                    return ValidationFailedSnafu { op: "TC", reason: "symbolic dimension cannot use tensor cores" }
                        .fail();
                }
            }
        }

        // Apply padding operations sequentially. Propagate the inner OptError
        // directly rather than collapsing it — the PADTO failure detail (4x work
        // limit, unsafe ops, symbolic extent) is more actionable than a generic
        // "padding failed" message.
        for (axes_idx, scheduler_idx, tc_dim) in padding_ops {
            crate::optimizer::opts::apply_opt(scheduler, &crate::optimizer::Opt::padto(scheduler_idx, tc_dim), false)?;

            // Update axes to the new padded range (PADTO substitutes the old range)
            axes[axes_idx] = scheduler.rngs()[scheduler_idx].clone();
        }
    } else {
        // Without tc_opt >= 2, reject non-divisible dimensions. Symbolic dims
        // never TC: silently skipping the check lowers a tile loop over an
        // extent the tile may not cover.
        for (i, axis) in axes.iter().enumerate() {
            let tc_dim = match i {
                0 => tc.dims.0,
                1 => tc.dims.1,
                _ => tc.dims.2,
            };
            match get_range_size(axis) {
                Some(size) if (size as usize).is_multiple_of(tc_dim) => {}
                _ => {
                    return ValidationFailedSnafu { op: "TC", reason: "dimension not divisible by tensor core size" }
                        .fail();
                }
            }
        }
    }

    // Create WARP dimension
    let mut warp = UOp::range_axis(
        UOp::const_(svod_dtype::DType::Index, ConstValue::Int(tc.threads as i64)),
        AxisId::Renumbered(scheduler.maxarg() + 1),
        AxisType::Warp,
    );

    // Step 1: Apply TC opts via shift_to — splits each axis into (reduced, new_rng)
    let two = UOp::const_(svod_dtype::DType::Index, ConstValue::Int(2));
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
                    UOp::const_(svod_dtype::DType::Index, ConstValue::Int(2)),
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
            device: scheduler.ren.device,
            threads: tc.threads,
            upcast_axes: WmmaUpcastAxes { a: a_axes.clone(), b: b_axes.clone(), c: c_axes.clone() },
            reduce_axes: tc_reduce_aids.clone(),
            tile_grid: tc.tile_grid,
            asm: false,
        };

        // Tag the WMMA structure finalized (see `TAG_TC_FINAL`) so the expander
        // keeps the operand CONTRACTs / WMMA / output UNROLL distinct from the
        // raw operand subtrees and expands the WMMA per output tile.
        let tc_tag = smallvec::smallvec![crate::devectorize::TAG_TC_FINAL];
        let a_contract = src_a.contract(a_axes).with_tag(tc_tag.clone());
        let b_contract = src_b.contract(b_axes).with_tag(tc_tag.clone());
        // The WMMA C/accumulator operand carries the full per-thread D-register
        // width (`elements_per_thread.2`, == prod(c_axes)), NOT a scalar — see
        // tinygrad postrange.py:300-303 which builds the zero accumulator as
        // `dtype_out.vec(elements_per_thread[2])`. A scalar-0 here desyncs the
        // C operand from A/B/D when the expander replicates the WMMA over the
        // M/N output tiles: do_expand broadcasts a scalar C by `expand_sz`
        // (e.g. 16) giving a count-16 operand, while A/B/D become count-64,
        // and `devectorize_wmma` then can't group C into per-tile slices.
        let c_count = tc.elements_per_thread.2;
        let zero_scalar = if tc.dtype_out.is_float() {
            UOp::const_(tc.dtype_out.clone(), ConstValue::Float(0.0))
        } else {
            UOp::const_(tc.dtype_out.clone(), ConstValue::Int(0))
        };
        let zero_acc = zero_scalar.broadcast(c_count);
        let wmma = UOp::wmma(a_contract, b_contract, zero_acc, metadata).with_tag(tc_tag.clone());
        let mut tc_uop = wmma.unroll_with_dtype(c_axes, tc.dtype_out.clone()).with_tag(tc_tag.clone());

        // Re-wrap the WMMA in a REDUCE over the residual reduction ranges — the
        // K-tile loop left once the matrix core folds the contraction axes
        // (`tc_reduce_aids`). `shift_to` splits K and substitutes the composite
        // index back into the operand expressions, so the residual range no
        // longer lives in `updated_reduce.ranges` (which collapses to empty) but
        // in the WMMA's backward slice. Collect it from the slice, keeping only
        // `Reduce`-typed ranges the core did not consume — the slice also carries
        // Global/Warp/Upcast ranges, which must NOT be wrapped. Without this
        // REDUCE, pm_reduce
        // never builds the carried accumulator + loop-close `End`, so codegen
        // emits a bare WMMA with a const-0 C operand and an unterminated loop.
        let mut extra: SmallVec<[Arc<UOp>; 4]> = tc_uop
            .backward_slice()
            .into_iter()
            .filter(|r| matches!(r.op(), Op::Range { axis_id, axis_type: AxisType::Reduce, .. } if !tc_reduce_aids.contains(&axis_id.value())))
            .collect();
        // Deterministic nesting (outer = lowest axis_id); slice may list a range once.
        extra.sort_by_key(get_axis_id);
        extra.dedup_by_key(|r| get_axis_id(r));
        if !extra.is_empty() {
            tc_uop = tc_uop.reduce(extra, ReduceOp::Add);
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

fn tc_reject_reason(err: &OptError) -> &'static str {
    match err {
        OptError::ValidationFailed { reason, .. } => reason,
        OptError::InvalidArgType { .. } => "invalid argument type",
        OptError::AxisOutOfBounds { .. } => "axis out of bounds",
        OptError::DivisionError { .. } => "division constraint violated",
        OptError::SymbolicDivisionError { .. } => "symbolic divisibility constraint",
        OptError::ExpectedRangeOperation => "expected range operation",
        OptError::MissingAxisParameter => "missing axis parameter",
        OptError::UnsupportedFeature { .. } => "unsupported backend feature",
        OptError::DeviceLimitExceeded { .. } => "device limit exceeded",
    }
}

/// Apply tensor core optimization to the scheduler.
///
/// If `axis_choice` is provided, only that axis candidate is attempted.
/// Otherwise all axis candidates are tried in order until one succeeds.
pub fn apply_with_axis_choice(
    scheduler: &mut Scheduler,
    tc_select: i32,
    tc_opt: usize,
    use_tensor_cores: usize,
    axis_choice: Option<usize>,
) -> Result<[Arc<UOp>; 3], OptError> {
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

    let pattern = detect_matmul(scheduler)?
        .ok_or_else(|| ValidationFailedSnafu { op: "TC", reason: "no matmul pattern detected" }.build())?;

    let choices: Vec<usize> = if let Some(choice) = axis_choice {
        if choice >= pattern.axis_choices.len() {
            return ValidationFailedSnafu { op: "TC", reason: "axis choice out of bounds" }.fail();
        }
        vec![choice]
    } else {
        (0..pattern.axis_choices.len()).collect()
    };

    let mut failures: Vec<(usize, &'static str)> = Vec::new();
    let tc_choices: Vec<i32> = if tc_select == -1 {
        (0..scheduler.ren.tensor_cores.len()).map(|idx| idx as i32).collect()
    } else {
        vec![tc_select]
    };
    let mut last_err: Option<OptError> = None;

    // Cap total trials to bound compile time when both axis_choices and
    // tensor_cores are large. 64 covers realistic combinations (>16 axes ×
    // >4 TC variants is exceedingly rare) without aborting useful searches.
    const TC_RETRY_BUDGET: usize = 64;
    let mut trials = 0usize;

    'outer: for choice in choices {
        for &tc_choice in &tc_choices {
            if trials >= TC_RETRY_BUDGET {
                tracing::debug!(
                    trials,
                    budget = TC_RETRY_BUDGET,
                    "tensor core retry budget exhausted; aborting search"
                );
                break 'outer;
            }
            trials += 1;

            let mut trial = scheduler.clone();
            match apply_axis_choice_impl(&mut trial, &pattern, tc_choice, tc_opt, use_tensor_cores, choice) {
                Ok(axes) => {
                    *scheduler = trial;
                    return Ok(axes);
                }
                Err(err) => {
                    let reason = tc_reject_reason(&err);
                    tracing::debug!(
                        axis_choice = choice,
                        tc_select = tc_choice,
                        reason,
                        error = %err,
                        "tensor core axis choice rejected"
                    );
                    failures.push((choice, reason));
                    last_err = Some(err);
                }
            }
        }
    }

    tracing::debug!(requested_axis_choice = ?axis_choice, failures = ?failures, "tensor core optimization rejected");

    if let Some(err) = last_err {
        Err(err)
    } else {
        ValidationFailedSnafu { op: "TC", reason: "no compatible tensor core found" }.fail()
    }
}

/// Apply tensor core optimization, auto-trying axis choices.
pub fn apply(
    scheduler: &mut Scheduler,
    tc_select: i32,
    tc_opt: usize,
    use_tensor_cores: usize,
) -> Result<[Arc<UOp>; 3], OptError> {
    apply_with_axis_choice(scheduler, tc_select, tc_opt, use_tensor_cores, None)
}

/// Short dtype name for WMMA function identifiers.
fn wmma_dtype_name(dtype: &svod_ir::prelude::DType) -> &'static str {
    use svod_dtype::ScalarDType;
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
