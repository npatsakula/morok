//! Tensor Core (TC) optimization - Hardware-accelerated matrix multiplication.
//!
//! Implements pattern matching, selection, swizzle, and application for tensor core ops.
//! Supports NVIDIA (WMMA), AMD (Matrix Cores), Intel, and Apple (AMX) hardware.

use std::collections::HashMap;
use std::sync::Arc;

use morok_ir::{AxisType, BinaryOp, ConstValue, Op, ReduceOp, UOp, WmmaMetadata};

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

type UpcastAxes = (Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<(usize, usize)>);

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

/// Build upcast axes configuration for WMMA construction.
pub fn build_upcast_axes(tc: &TensorCore, _new_ranges: &[usize]) -> UpcastAxes {
    (vec![(0, tc.elements_per_thread.0)], vec![(1, tc.elements_per_thread.1)], vec![(2, tc.elements_per_thread.2)])
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
) -> Result<(), OptError> {
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

    let tc = &scheduler.ren.tensor_cores[tc_selection.tc_index];
    let (n_range, m_range, k_range) = &tc_selection.axes;
    let axes = [n_range.clone(), m_range.clone(), k_range.clone()];

    // Padding check (tc_opt >= 2)
    if tc_opt >= 2 {
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
                return ValidationFailedSnafu { op: "TC", reason: "dimension not divisible (padding not implemented)" }
                    .fail();
            }
        }
    }

    // Create WARP dimension
    let warp = UOp::range_axis(
        UOp::const_(morok_dtype::DType::Index, ConstValue::Int(tc.threads as i64)),
        morok_ir::AxisId::Renumbered(scheduler.maxarg() + 1),
        AxisType::Warp,
    );

    // Apply TC opts
    let mut new_ranges = Vec::with_capacity(tc.opts.len());
    let _warp = warp;

    for opt in &tc.opts {
        let dim = match opt {
            TcOpt::Local(d) | TcOpt::Upcast(d) => *d,
        };
        let axis_idx = scheduler.rngs().iter().position(|r| Arc::ptr_eq(r, &axes[dim]));
        if let Some(idx) = axis_idx {
            new_ranges.push(idx);
        } else {
            return ValidationFailedSnafu { op: "TC", reason: "axis not found in scheduler ranges" }.fail();
        }
    }

    // Unroll reduction
    for (_idx, amt) in tc.get_reduce_axes() {
        new_ranges.push(amt);
    }

    // Build WMMA UOp (if use_tensor_cores == 1)
    if use_tensor_cores == 1 {
        let (a_axes, b_axes, c_axes) = build_upcast_axes(tc, &new_ranges);
        let metadata = WmmaMetadata {
            name: format!("WMMA_{}x{}x{}", tc.dims.0, tc.dims.1, tc.dims.2),
            dims: tc.dims,
            dtype_in: tc.dtype_in.clone(),
            dtype_out: tc.dtype_out.clone(),
            device: scheduler.ren.device.clone(),
            threads: tc.threads,
            upcast_axes: c_axes.clone(),
            reduce_axes: vec![],
        };

        let a_contract = UOp::contract(pattern.in0.clone(), a_axes);
        let b_contract = UOp::contract(pattern.in1.clone(), b_axes);
        let zero_acc = UOp::const_(tc.dtype_out.clone(), ConstValue::Float(0.0));
        let wmma = UOp::wmma(a_contract, b_contract, zero_acc, metadata);
        let tc_uop = UOp::unroll(wmma, c_axes);

        #[allow(clippy::mutable_key_type)]
        let mut subst_map = HashMap::new();
        subst_map.insert(morok_ir::UOpKey(pattern.reduce_op.clone()), tc_uop);
        let new_ast = scheduler.ast().substitute(&subst_map);
        scheduler.set_ast(new_ast);
    }

    Ok(())
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
    pub use super::{base_shape, build_upcast_axes, get_reduce_axes_count, permutes_for_shape};
}
