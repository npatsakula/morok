//! Patterns for kernel splitting transformation.
//!
//! This module implements the to_define_global patterns used during kernel splitting.
//! These patterns replace BUFFER operations with DEFINE_GLOBAL/DEFINE_LOCAL, handle
//! AFTER dependencies, and prepare the graph for kernel extraction.
//!
//! Based on Tinygrad's to_define_global PatternMatcher (schedule/rangeify.py:410-425).

use std::cell::RefCell;
use std::rc::Rc;

use morok_dtype::AddrSpace;
use morok_ir::{ConstValue, DType, Op, UOp};

use super::kernel_context::KernelContext;
use crate::pattern::UPat;
use crate::pattern::matcher::{PatternMatcher, RewriteFn, RewriteResult};

/// Replace BUFFER with DEFINE_GLOBAL or DEFINE_LOCAL.
///
/// This is the core pattern for converting high-level BUFFER operations into
/// low-level buffer allocations. Each BUFFER is replaced with either:
/// - DEFINE_GLOBAL(id) for global memory
/// - DEFINE_LOCAL(id) for local/shared memory
///
/// The buffer is also tracked in the context for building the kernel's argument list.
///
/// # Arguments
///
/// * `bindings` - Variable bindings from pattern match (must contain "buf")
/// * `ctx` - Kernel context for tracking buffer allocations
///
/// # Returns
///
/// * `Rewritten(define_op)` - The DEFINE_GLOBAL or DEFINE_LOCAL replacement
/// * `NoMatch` - If the binding doesn't contain a buffer
///
/// # Example
///
/// ```ignore
/// // Before: BUFFER(unique, device, size)
/// // After:  DEFINE_GLOBAL(0) or DEFINE_LOCAL(0)
/// ```
pub fn debuf(buf: &Rc<UOp>, ctx: &mut KernelContext) -> RewriteResult {
    // Extract buffer information
    let (dtype, addrspace) = match buf.op() {
        Op::Buffer { .. } => {
            // Get dtype from the buffer
            let dtype = buf.dtype();
            // For now, assume global unless we add addrspace tracking
            (dtype, AddrSpace::Global)
        }
        _ => return RewriteResult::NoMatch,
    };

    // Create DEFINE_GLOBAL or DEFINE_LOCAL based on address space
    let replacement = if addrspace == AddrSpace::Global {
        let global_id = ctx.next_global();
        UOp::define_global(global_id, dtype)
    } else {
        let local_id = ctx.next_local();
        UOp::define_local(local_id, dtype)
    };

    // Track the buffer in context (maps original buffer to itself for later reference)
    // Note: In Tinygrad this is ctx.map[buf] = buf, meaning the original buffer
    // becomes part of the kernel's argument list.
    ctx.map_buffer(buf.clone(), buf.clone());

    RewriteResult::Rewritten(replacement)
}

/// Handle AFTER operations for dependency tracking.
///
/// AFTER represents "buffer after computation completes". This pattern extracts
/// the underlying buffer and tracks the dependency relationship in the context.
///
/// For output buffers created by a kernel, the context maps the buffer to the
/// AFTER operation, which will be used when building kernel dependencies.
///
/// # Arguments
///
/// * `bindings` - Variable bindings from pattern match (must contain "after")
/// * `ctx` - Kernel context for tracking dependencies
///
/// # Returns
///
/// * `Rewritten(buffer)` - The extracted buffer (AFTER is removed but tracked)
/// * `NoMatch` - If not an AFTER operation
///
/// # Example
///
/// ```ignore
/// // Before: AFTER(BUFFER, store_computation)
/// // After:  BUFFER (but ctx.map[BUFFER] = AFTER for dependencies)
/// ```
pub fn handle_after(after: &Rc<UOp>, ctx: &mut KernelContext) -> RewriteResult {
    // Verify this is an AFTER operation
    if !matches!(after.op(), Op::After { .. }) {
        return RewriteResult::NoMatch;
    }

    // Extract the passthrough (first source)
    let passthrough = match after.op() {
        Op::After { passthrough, .. } => passthrough,
        _ => unreachable!(),
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
        return RewriteResult::Rewritten(buf);
    }

    // Track the mapping: buffer -> AFTER operation for global buffers
    // This will be used when building kernel arguments
    ctx.map_buffer(buf.clone(), after.clone());

    // Return the buffer itself (strip AFTER but remember it)
    RewriteResult::Rewritten(buf)
}

/// Remove BIND operations for kernel-local variables.
///
/// BIND operations tie variables to specific values within a kernel scope.
/// When splitting kernels, we need to unbind these variables so they can
/// be passed as kernel arguments instead.
///
/// # Arguments
///
/// * `bindings` - Variable bindings from pattern match (must contain "b")
/// * `ctx` - Kernel context for tracking variables
///
/// # Returns
///
/// * `Rewritten(var)` - Just the variable (BIND removed, but var tracked)
/// * `NoMatch` - If not a BIND operation
///
/// # Example
///
/// ```ignore
/// // Before: BIND(var, value)
/// // After:  var (but ctx.vars tracks it)
/// ```
pub fn unbind_kernel(bind: &Rc<UOp>, ctx: &mut KernelContext) -> RewriteResult {
    // Verify this is a BIND operation
    let (var, _value) = match bind.op() {
        Op::Bind { var, value } => (var, value),
        _ => return RewriteResult::NoMatch,
    };

    // Track the variable in context
    ctx.add_var(var.clone());

    // Return just the variable (unbind it)
    RewriteResult::Rewritten(var.clone())
}

/// Renumber RANGE operations within a kernel.
///
/// To enable kernel deduplication, RANGE operations are renumbered starting
/// from 0 within each kernel. This ensures that two functionally identical
/// kernels have the same RANGE numbering.
///
/// # Arguments
///
/// * `bindings` - Variable bindings from pattern match (must contain "r")
/// * `ctx` - Kernel context for tracking range numbering
///
/// # Returns
///
/// * `Rewritten(new_range)` - RANGE with renumbered axis_id
/// * `NoMatch` - If not a RANGE or already numbered correctly
///
/// # Example
///
/// ```ignore
/// // Before: RANGE(end, axis_id=5, type=Loop)
/// // After:  RANGE(end, axis_id=0, type=Loop) (first range in kernel)
/// ```
pub fn renumber_range(range: &Rc<UOp>, ctx: &mut KernelContext) -> RewriteResult {
    // Verify this is a RANGE operation
    let (end, old_axis_id, axis_type) = match range.op() {
        Op::Range { end, axis_id, axis_type } => (end, *axis_id, *axis_type),
        _ => return RewriteResult::NoMatch,
    };

    // Get the new axis ID (automatically increments)
    let new_axis_id = ctx.next_range();

    // Only rewrite if the ID changed
    if new_axis_id == old_axis_id {
        return RewriteResult::NoMatch;
    }

    // Create new RANGE with renumbered axis_id
    let new_range = UOp::range_axis(end.clone(), new_axis_id, axis_type);

    RewriteResult::Rewritten(new_range)
}

/// Clean up spurious sources from CONST and DEFINE_VAR operations.
///
/// During graph transformations, CONST and DEFINE_VAR operations may
/// accidentally acquire sources. This pattern removes them.
///
/// # Arguments
///
/// * `bindings` - Variable bindings from pattern match (must contain "c")
///
/// # Returns
///
/// * `Rewritten(cleaned_op)` - Operation with sources removed
/// * `NoMatch` - If operation has no sources
///
/// # Example
///
/// ```ignore
/// // Before: CONST(42, dtype=Int32) with spurious sources
/// // After:  CONST(42, dtype=Int32) with no sources
/// ```
pub fn cleanup_const(op: &Rc<UOp>, _ctx: &mut KernelContext) -> RewriteResult {
    // Check if this is CONST or DEFINE_VAR with sources
    let should_clean = matches!(op.op(), Op::Const(_) | Op::DefineVar { .. });

    if !should_clean {
        return RewriteResult::NoMatch;
    }

    // Get the current sources
    let sources = op.op().sources();
    if sources.is_empty() {
        return RewriteResult::NoMatch;
    }

    // Create new operation with no sources
    let cleaned = match op.op() {
        Op::Const(val) => UOp::const_(op.dtype(), val.0),
        Op::DefineVar { name, min_val, max_val } => UOp::var(name.clone(), op.dtype(), *min_val, *max_val),
        _ => unreachable!(),
    };

    RewriteResult::Rewritten(cleaned)
}

/// Remove RANGE operations with zero size.
///
/// RANGE operations with end=0 represent empty loops and should be
/// replaced with a constant 0.
///
/// # Arguments
///
/// * `bindings` - Variable bindings from pattern match (must contain "r")
///
/// # Returns
///
/// * `Rewritten(const_0)` - Constant 0 if range has zero size
/// * `NoMatch` - If range is non-empty
///
/// # Example
///
/// ```ignore
/// // Before: RANGE(end=CONST(0), axis_id=0, type=Loop)
/// // After:  CONST(0, dtype=Index)
/// ```
pub fn remove_zero_range(range: &Rc<UOp>, _ctx: &mut KernelContext) -> RewriteResult {
    // Verify this is a RANGE operation
    let end = match range.op() {
        Op::Range { end, .. } => end,
        _ => return RewriteResult::NoMatch,
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
        return RewriteResult::NoMatch;
    }

    // Replace with constant 0
    use morok_dtype::DType;
    let zero = UOp::const_(DType::Index, ConstValue::Int(0));

    RewriteResult::Rewritten(zero)
}

/// Create patterns for `to_define_global` transformation with KernelContext.
///
/// This function creates a PatternMatcher whose patterns have access to KernelContext
/// via closure capture. The patterns implement Tinygrad's to_define_global transformation:
///
/// 1. **debuf**: BUFFER → DEFINE_GLOBAL/DEFINE_LOCAL
/// 2. **unbind_kernel**: BIND(var, value) → var (track variable)
/// 3. **handle_after**: AFTER/MSTACK/MSELECT → underlying buffer (track dependencies)
/// 4. **cleanup_const**: Remove spurious sources from CONST/DEFINE_VAR
/// 5. **remove_zero_range**: RANGE(end=0) → CONST(0)
/// 6. **renumber_range**: Renumber RANGEs for deduplication
///
/// # Arguments
///
/// * `ctx` - Shared reference to KernelContext for tracking buffers, variables, and ranges
///
/// # Returns
///
/// A PatternMatcher with all to_define_global patterns
///
/// # Example
///
/// ```ignore
/// use std::cell::RefCell;
/// use std::rc::Rc;
///
/// let ctx = Rc::new(RefCell::new(KernelContext::new()));
/// let matcher = to_define_global_patterns(Rc::clone(&ctx));
///
/// // Apply patterns via graph_rewrite
/// let result = graph_rewrite(&matcher, computation);
/// ```
///
/// Based on Tinygrad's to_define_global (schedule/rangeify.py:419-434).
pub fn to_define_global_patterns(ctx: Rc<RefCell<KernelContext>>) -> PatternMatcher {
    let mut patterns: Vec<(UPat, RewriteFn)> = vec![];

    // Pattern 1: debuf - BUFFER → DEFINE_GLOBAL/DEFINE_LOCAL
    pattern_ctx_mut!(patterns, Rc::clone(&ctx),
        UPat::var("buf") => |buf, ctx| {
            // Only match BUFFER operations
            if !matches!(buf.op(), Op::Buffer { .. }) {
                return RewriteResult::NoMatch;
            }
            debuf(buf, ctx)
        }
    );

    // Pattern 2: unbind_kernel - BIND → var
    pattern_ctx_mut!(patterns, Rc::clone(&ctx),
        UPat::var("b") => |b, ctx| {
            // Only match BIND operations
            if !matches!(b.op(), Op::Bind { .. }) {
                return RewriteResult::NoMatch;
            }
            unbind_kernel(b, ctx)
        }
    );

    // Pattern 3: handle_after - AFTER/MSTACK/MSELECT → buffer
    pattern_ctx_mut!(patterns, Rc::clone(&ctx),
        UPat::var("after") => |after, ctx| {
            // Only match AFTER, MSTACK, or MSELECT operations
            if !matches!(after.op(), Op::After { .. } | Op::MStack { .. } | Op::MSelect { .. }) {
                return RewriteResult::NoMatch;
            }
            handle_after(after, ctx)
        }
    );

    // Pattern 4: cleanup_const - Remove spurious sources from CONST/DEFINE_VAR
    pattern_ctx_mut!(patterns, Rc::clone(&ctx),
        UPat::var("c") => |c, ctx| {
            // Only match CONST or DEFINE_VAR with sources
            if !matches!(c.op(), Op::Const(_) | Op::DefineVar { .. }) {
                return RewriteResult::NoMatch;
            }
            cleanup_const(c, ctx)
        }
    );

    // Pattern 5: remove_zero_range - RANGE(end=0) → CONST(0)
    pattern_ctx_mut!(patterns, Rc::clone(&ctx),
        UPat::var("r") => |r, ctx| {
            // Only match RANGE operations
            if !matches!(r.op(), Op::Range { .. }) {
                return RewriteResult::NoMatch;
            }
            remove_zero_range(r, ctx)
        }
    );

    // Pattern 6: renumber_range - Renumber RANGEs for deduplication
    pattern_ctx_mut!(patterns, Rc::clone(&ctx),
        UPat::var("r") => |r, ctx| {
            // Only match RANGE operations
            if !matches!(r.op(), Op::Range { .. }) {
                return RewriteResult::NoMatch;
            }
            renumber_range(r, ctx)
        }
    );

    PatternMatcher::new(patterns)
}
