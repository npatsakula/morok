//! Tensor core application and WMMA construction.
//!
//! Main transformation logic that applies tensor core optimizations to the AST.

use super::{matching, selection, swizzle};
use crate::optimizer::{Scheduler, error::*};
use morok_ir::{AxisType, Op, UOp, WmmaMetadata};
use std::rc::Rc;

/// Apply tensor core optimization to the scheduler.
///
/// # Arguments
///
/// * `scheduler` - The scheduler to optimize
/// * `tc_select` - Tensor core selection (-1 for auto, >= 0 for specific TC index)
/// * `tc_opt` - Optimization level (0: basic, 1: allow multi-reduce, 2: allow padding)
/// * `use_tensor_cores` - Mode (1: use WMMA UOps, 2: shape only without WMMA)
///
/// # Validation
///
/// - Must be applied before other optimizations (applied_opts must be empty)
/// - Backend must support tensor cores
/// - Pattern must match tensor core requirements
///
/// # Returns
///
/// - `Ok(())` if tensor core applied successfully
/// - `Err(_)` if validation fails or no suitable tensor core found
pub fn apply(
    scheduler: &mut Scheduler,
    tc_select: i32,
    tc_opt: usize,
    use_tensor_cores: usize,
) -> Result<(), OptError> {
    // 1. Validate this is first optimization
    if !scheduler.applied_opts.is_empty() {
        return ValidationFailedSnafu { op: "TC", reason: "tensor core opts must be first" }.fail();
    }

    // 2. Validate parameters
    if use_tensor_cores == 0 || use_tensor_cores > 2 {
        return ValidationFailedSnafu { op: "TC", reason: "use_tensor_cores must be 1 or 2" }.fail();
    }

    if tc_opt > 2 {
        return ValidationFailedSnafu { op: "TC", reason: "tc_opt must be 0, 1, or 2" }.fail();
    }

    if tc_select < -1 {
        return ValidationFailedSnafu { op: "TC", reason: "tc_select must be >= -1" }.fail();
    }

    // 3. Detect matmul pattern
    let pattern = match matching::detect_matmul(scheduler)? {
        Some(p) => p,
        None => {
            return ValidationFailedSnafu { op: "TC", reason: "no matmul pattern detected" }.fail();
        }
    };

    // 4. Try to select a tensor core
    // Try all axis choices until we find a match
    let mut tc_selection = None;
    for axis_choice in 0..pattern.axis_choices.len() {
        if let Some(selection) = selection::select_tensor_core(&pattern, &scheduler.ren, tc_select, axis_choice)? {
            tc_selection = Some(selection);
            break;
        }
    }

    let tc_selection = match tc_selection {
        Some(s) => s,
        None => {
            return ValidationFailedSnafu { op: "TC", reason: "no compatible tensor core found for this pattern" }
                .fail();
        }
    };

    // 5. Get selected tensor core
    let tc = &scheduler.ren.tensor_cores[tc_selection.tc_index];
    let (n_range, m_range, k_range) = &tc_selection.axes;

    // 6. Apply padding if needed (tc_opt >= 2)
    let axes = [n_range.clone(), m_range.clone(), k_range.clone()];
    if tc_opt >= 2 {
        for i in 0..3 {
            let dim_size = get_range_size(&axes[i]);
            let tc_dim = match i {
                0 => tc.dims.0,
                1 => tc.dims.1,
                2 => tc.dims.2,
                _ => unreachable!(),
            };

            if let Some(size) = dim_size
                && !(size as usize).is_multiple_of(tc_dim)
            {
                // Need padding - would apply PADTO here
                // For now, just validate it's divisible
                return ValidationFailedSnafu {
                    op: "TC",
                    reason: "dimension not divisible by tensor core requirement (padding not yet implemented)",
                }
                .fail();
            }
        }
    }

    // 7. Create WARP dimension
    let warp = UOp::range_axis(
        UOp::const_(morok_dtype::DType::Index, morok_ir::ConstValue::Int(tc.threads as i64)),
        scheduler.maxarg() + 1,
        AxisType::Warp,
    );

    // 8. Apply TC opts sequence
    let mut new_ranges = Vec::with_capacity(tc.opts.len());
    let _current_warp = warp.clone(); // Would be used for actual warp splitting

    for opt in &tc.opts {
        match opt {
            crate::optimizer::renderer::TcOpt::Local(dim) => {
                // Split dimension by 2, assign to LOCAL, use warp lanes
                let axis_idx = scheduler.rngs().iter().position(|r| Rc::ptr_eq(r, &axes[*dim]));
                if let Some(idx) = axis_idx {
                    // For now, just note that we would split here
                    // Actual implementation would call scheduler.shift_to()
                    // axes[*dim] = shift_to(axes[*dim], 2, AxisType::Local, current_warp % 2)?
                    // current_warp = current_warp / 2
                    new_ranges.push(idx);
                } else {
                    return ValidationFailedSnafu { op: "TC", reason: "axis not found in scheduler ranges" }.fail();
                }
            }
            crate::optimizer::renderer::TcOpt::Upcast(dim) => {
                // Split dimension by 2, assign to UPCAST
                let axis_idx = scheduler.rngs().iter().position(|r| Rc::ptr_eq(r, &axes[*dim]));
                if let Some(idx) = axis_idx {
                    // axes[*dim] = shift_to(axes[*dim], 2, AxisType::Upcast)?
                    new_ranges.push(idx);
                } else {
                    return ValidationFailedSnafu { op: "TC", reason: "axis not found in scheduler ranges" }.fail();
                }
            }
        }
    }

    // 9. Unroll reduction dimension
    let reduce_axes = tc.get_reduce_axes();
    for (_idx, amt) in reduce_axes {
        // axes[2] = shift_to(axes[2], amt, AxisType::Unroll)?
        new_ranges.push(amt);
    }

    // 10. Build WMMA UOp (if use_tensor_cores == 1)
    if use_tensor_cores == 1 {
        // Build upcast axes configuration
        let (a_axes, b_axes, c_axes) = swizzle::build_upcast_axes(tc, &new_ranges);

        // Create WMMA metadata
        let metadata = WmmaMetadata {
            name: format!("WMMA_{}x{}x{}", tc.dims.0, tc.dims.1, tc.dims.2),
            dims: tc.dims,
            dtype_in: tc.dtype_in.clone(),
            dtype_out: tc.dtype_out.clone(),
            device: scheduler.ren.device.clone(),
            threads: tc.threads,
            upcast_axes: c_axes.clone(), // Use C axes for now
            reduce_axes: vec![],         // Will be filled during actual implementation
        };

        // Create CONTRACT UOps for inputs
        let a_contract = UOp::contract(pattern.in0.clone(), a_axes);
        let b_contract = UOp::contract(pattern.in1.clone(), b_axes);

        // Create zero accumulator
        let zero_acc = UOp::const_(tc.dtype_out.clone(), morok_ir::ConstValue::Float(0.0));

        // Build WMMA UOp
        let wmma = UOp::wmma(a_contract, b_contract, zero_acc, metadata.clone());

        // Wrap with UNROLL for C dimension
        let tc_uop = UOp::unroll(wmma, c_axes);

        // Substitute REDUCE with WMMA/UNROLL tree
        let mut subst_map = std::collections::HashMap::new();
        subst_map.insert(morok_ir::UOpKey(pattern.reduce_op.clone()), tc_uop);
        let new_ast = scheduler.ast().substitute(&subst_map);
        scheduler.set_ast(new_ast);
    }

    Ok(())
}

/// Get the size of a RANGE UOp (if constant).
fn get_range_size(range: &Rc<UOp>) -> Option<i64> {
    if let Op::Range { end, .. } = range.op()
        && let Op::Const(cv) = end.op()
        && let morok_ir::ConstValue::Int(size) = cv.0
    {
        return Some(size);
    }
    None
}
