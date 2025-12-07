//! Patterns for kernel splitting: BUFFER→DEFINE_GLOBAL, AFTER handling, range renumbering.

use std::sync::Arc;

use morok_dtype::AddrSpace;
use morok_ir::{AxisId, AxisType, ConstValue, DType, Op, UOp};

use super::kernel_context::KernelContext;
use crate::pattern::matcher::PatternMatcher;

/// Replace BUFFER with DEFINE_GLOBAL or DEFINE_LOCAL.
///
/// **Critical:** Embeds buffer size in the Ptr dtype following Tinygrad's pattern.
/// This ensures `compute_buffer_size()` can extract size from the DefineGlobal dtype
/// instead of searching through the AST.
pub fn debuf(buf: &Arc<UOp>, ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    // Extract buffer information including size
    let (ptr_dtype, addrspace) = match buf.op() {
        Op::Buffer { size, .. } => {
            // Get base dtype from the buffer
            let base_dtype = buf.dtype();
            // Embed size in Ptr dtype (like Tinygrad's dtype.ptr(size=...))
            let ptr_dtype = base_dtype.ptr(Some(*size), AddrSpace::Global);
            (ptr_dtype, AddrSpace::Global)
        }
        _ => return None,
    };

    // Create DEFINE_GLOBAL or DEFINE_LOCAL based on address space
    let replacement = if addrspace == AddrSpace::Global {
        let global_id = ctx.next_global();
        UOp::define_global(global_id, ptr_dtype)
    } else {
        let local_id = ctx.next_local();
        UOp::define_local(local_id, ptr_dtype)
    };

    // Track the buffer mapping: BUFFER → DEFINE_GLOBAL
    // This is needed so kernel sources include the DEFINE_GLOBAL for each input buffer.
    // Output buffers are tracked via bufferize_to_store as BUFFERIZE → AFTER(DEFINE_GLOBAL).
    ctx.map_buffer(buf.clone(), replacement.clone());

    Some(replacement)
}

/// Handle AFTER: extract buffer and track dependency.
pub fn handle_after(after: &Arc<UOp>, ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    // Extract the passthrough (first source)
    let passthrough = match after.op() {
        Op::After { passthrough, .. } => passthrough,
        _ => return None,
    };

    // Get the underlying buffer
    // If the passthrough is MSTACK or MSELECT, unwrap to the first source
    let buf = match passthrough.op() {
        Op::MStack { buffers } if !buffers.is_empty() => buffers[0].clone(),
        Op::MSelect { buffer, .. } => buffer.clone(),
        _ => passthrough.clone(),
    };

    // Skip AFTER for local buffers (they don't need cross-kernel dependency tracking)
    // Local buffers are kernel-scoped and synchronized via BARRIER operations
    if matches!(buf.dtype(), DType::Ptr { addrspace: AddrSpace::Local, .. }) {
        return Some(buf);
    }

    // Track the mapping: buffer -> AFTER operation for global buffers
    // This will be used when building kernel arguments
    ctx.map_buffer(buf.clone(), after.clone());

    // Return the buffer itself (strip AFTER but remember it)
    Some(buf)
}

/// Remove BIND: extract var and track it.
pub fn unbind_kernel(bind: &Arc<UOp>, ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    // Verify this is a BIND operation
    let (var, _value) = match bind.op() {
        Op::Bind { var, value } => (var, value),
        _ => return None,
    };

    // Track the variable in context
    ctx.add_var(var.clone());

    // Return just the variable (unbind it)
    Some(var.clone())
}

/// Renumber RANGE axis_id starting from 0 for kernel deduplication.
///
/// Uses enum-based guard that is naturally idempotent:
/// - Only matches ranges with `AxisId::Unrenumbered`
/// - Produces ranges with `AxisId::Renumbered`, so won't match again
pub fn renumber_range(range: &Arc<UOp>, ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    // Verify this is a RANGE operation
    let (end, old_axis_id, axis_type) = match range.op() {
        Op::Range { end, axis_id, axis_type } => (end, *axis_id, *axis_type),
        _ => return None,
    };

    // Guard: only renumber unrenumbered ranges (type-safe idempotence)
    match old_axis_id {
        AxisId::Unrenumbered(_) => {}
        AxisId::Renumbered(_) => return None,
    }

    // OUTER ranges: convert to BIND(DEFINE_VAR, RANGE) pattern
    // This follows Tinygrad's approach where OUTER ranges become variables
    // that are passed as kernel parameters at runtime.
    if axis_type == AxisType::Outer {
        // Create variable name based on axis_id
        let var_name = format!("range_{}", old_axis_id.value());

        // Extract max value from range end (vmax = end - 1)
        let vmax = match end.vmax() {
            morok_ir::ConstValue::Int(v) => *v - 1,
            morok_ir::ConstValue::UInt(v) => (*v - 1) as i64,
            _ => {
                // If we can't extract a concrete vmax, fall back to simple renumbering
                // This handles symbolic/dynamic ranges
                let new_axis_id = AxisId::Renumbered(ctx.next_range());
                let new_range = UOp::range_axis(end.clone(), new_axis_id, axis_type);
                return Some(new_range);
            }
        };

        // Create DEFINE_VAR with max=vmax (min is always 0)
        let var = UOp::define_var(var_name, vmax);

        // Create renumbered RANGE
        let new_axis_id = AxisId::Renumbered(ctx.next_range());
        let renumbered_range = UOp::range_axis(end.clone(), new_axis_id, axis_type);

        // Wrap in BIND(var, range)
        return Some(UOp::bind(var, renumbered_range));
    }

    // Non-OUTER ranges: just renumber axis_id
    let new_axis_id = AxisId::Renumbered(ctx.next_range());
    let new_range = UOp::range_axis(end.clone(), new_axis_id, axis_type);

    Some(new_range)
}

/// Remove spurious sources from CONST and DEFINE_VAR.
pub fn cleanup_const(op: &Arc<UOp>, _ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    // Check if this is CONST or DEFINE_VAR with sources
    let should_clean = matches!(op.op(), Op::Const(_) | Op::DefineVar { .. });

    if !should_clean {
        return None;
    }

    // Get the current sources
    let sources = op.op().sources();
    if sources.is_empty() {
        return None;
    }

    // Create new operation with no sources
    let cleaned = match op.op() {
        Op::Const(val) => UOp::const_(op.dtype(), val.0),
        Op::DefineVar { name, max_val } => UOp::var(name.clone(), op.dtype(), *max_val),
        _ => unreachable!(),
    };

    Some(cleaned)
}

/// Replace RANGE(end=0) with CONST(0).
pub fn remove_zero_range(range: &Arc<UOp>, _ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    // Verify this is a RANGE operation
    let end = match range.op() {
        Op::Range { end, .. } => end,
        _ => return None,
    };

    // Check if end is a constant 0
    let is_zero = match end.op() {
        Op::Const(val) => match val.0 {
            ConstValue::Int(i) => i == 0,
            ConstValue::UInt(u) => u == 0,
            _ => false,
        },
        _ => false,
    };

    if !is_zero {
        return None;
    }

    // Replace with constant 0
    let zero = UOp::index_const(0);

    Some(zero)
}

/// Create patterns for to_define_global transformation.
pub fn to_define_global_patterns() -> PatternMatcher<KernelContext> {
    crate::patterns! {
        @context KernelContext;

        buf if matches!(buf.op(), Op::Buffer { .. }) => debuf(buf, ctx),
        b if matches!(b.op(), Op::Bind { .. }) => unbind_kernel(b, ctx),
        after if matches!(after.op(), Op::After { .. }) => handle_after(after, ctx),
        c if matches!(c.op(), Op::Const(_) | Op::DefineVar { .. }) => cleanup_const(c, ctx),
        r if matches!(r.op(), Op::Range { .. }) => remove_zero_range(r, ctx),
        r if matches!(r.op(), Op::Range { .. }) => renumber_range(r, ctx),
    }
}
