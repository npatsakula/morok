//! Codegen preparation patterns (rangeify_codegen).
//!
//! This module implements patterns that prepare kernel IR for code generation by:
//! - Removing NOOP placeholders with actual zero values
//! - Handling broadcast operations
//! - Generating LOAD operations for memory accesses
//! - Fixing AFTER operations
//!
//! Based on Tinygrad's rangeify_codegen PatternMatcher (schedule/rangeify.py:440-465).

use std::rc::Rc;

use morok_dtype::DType;
use morok_ir::{ConstValue, Op, UOp, UOpKey};
use smallvec::SmallVec;

use crate::pattern::matcher::PatternMatcher;

/// Replace NOOP operations with actual zero values.
///
/// NOOP is a placeholder operation that represents an uninitialized value.
/// Before code generation, we must replace it with an actual zero value:
/// - Scalar types → single zero constant
/// - Vector types → vector of zeros
///
/// # Arguments
///
/// * `noop` - The NOOP operation to replace
///
/// # Returns
///
/// * `Some(zero)` - A zero constant matching the NOOP's dtype
/// * `None` - If the operation is not a NOOP
///
/// # Example
///
/// ```ignore
/// // Scalar NOOP
/// let noop = UOp::noop(); // dtype = Float32
/// let zero = remove_noop(&noop); // CONST(0.0, Float32)
///
/// // Vector NOOP
/// let noop_vec = UOp::noop(); // dtype = Vec<Float32, 4>
/// let zero_vec = remove_noop(&noop_vec); // VEC([0.0, 0.0, 0.0, 0.0])
/// ```
///
/// Based on Tinygrad's remove_noop (schedule/rangeify.py:449-450):
/// ```python
/// def remove_noop(x:UOp) -> UOp|None:
///   if x.op is Ops.NOOP: return x.dtype.base.vec(x.dtype.count) if x.dtype.count > 1 else x.dtype.base.as_const(0)
/// ```
pub fn remove_noop(noop: &Rc<UOp>) -> Option<Rc<UOp>> {
    // Only match NOOP operations
    if !matches!(noop.op(), Op::Noop) {
        return None;
    }

    let dtype = noop.dtype();

    // Get the base scalar type (works for both scalars and vectors)
    let base = dtype.base();

    // Don't create zero for Void type
    if base == morok_dtype::ScalarDType::Void {
        return None;
    }

    // Check if this is a vector type
    if dtype.is_vector() {
        let count = dtype.count();
        let zero_value = ConstValue::zero(base);

        // Create a vector of zeros using vectorize
        let zeros: SmallVec<[Rc<UOp>; 4]> = (0..count).map(|_| UOp::const_(DType::Scalar(base), zero_value)).collect();

        return Some(UOp::vectorize(zeros));
    }

    // Scalar case: create zero constant matching dtype
    let zero_value = ConstValue::zero(base);
    Some(UOp::const_(dtype, zero_value))
}

/// Remove CONTIGUOUS markers.
///
/// CONTIGUOUS is a marker operation that indicates memory layout has been verified.
/// After memory layout analysis, these markers can be safely removed before codegen.
///
/// # Arguments
///
/// * `contiguous` - The CONTIGUOUS operation to remove
///
/// # Returns
///
/// * `Some(source)` - The source of the CONTIGUOUS operation
/// * `None` - If the operation is not CONTIGUOUS
///
/// # Example
///
/// ```ignore
/// // Before: CONTIGUOUS(tensor)
/// // After:  tensor
/// ```
///
/// Based on Tinygrad's get_contiguous (schedule/rangeify.py:447):
/// ```python
/// def get_contiguous(x:UOp) -> UOp|None:
///   if x.op is Ops.CONTIGUOUS: return x.src[0]
/// ```
pub fn get_contiguous(contiguous: &Rc<UOp>) -> Option<Rc<UOp>> {
    // Only match CONTIGUOUS operations
    if !matches!(contiguous.op(), Op::Contiguous { .. }) {
        return None;
    }

    // Return the source (remove the CONTIGUOUS wrapper)
    Some(contiguous.op().sources()[0].clone())
}

/// Fix AFTER operations wrapping EXPAND (broadcast).
///
/// When an AFTER operation wraps an EXPAND (broadcast), we need to unwrap it by
/// replacing the AFTER's passthrough with the EXPAND's source. This prevents issues
/// with local AFTER operations that shouldn't exist.
///
/// # Arguments
///
/// * `after` - The AFTER operation to fix
///
/// # Returns
///
/// * `Some(fixed)` - AFTER with EXPAND unwrapped from passthrough
/// * `None` - If not an AFTER wrapping EXPAND, or if it's a local AFTER (panic)
///
/// # Panics
///
/// Panics if the EXPAND source has RANGE parents, indicating a local AFTER which
/// is not allowed.
///
/// # Example
///
/// ```ignore
/// // Before: AFTER(passthrough=EXPAND(source), deps=[...])
/// // After:  AFTER(passthrough=source, deps=[...])
/// ```
///
/// Based on Tinygrad's fix_after_broadcast (schedule/rangeify.py:453-456):
/// ```python
/// def fix_after_broadcast(x:UOp) -> UOp|None:
///   if x.op is Ops.AFTER and (b:=x.src[0]).op is Ops.BROADCAST:
///     if any(u.op is Ops.RANGE for u in b.src[0].sparents): raise RuntimeError("can't have a local AFTER")
///     return x.replace(src=(b.src[0],)+x.src[1:])
/// ```
///
/// Note: We use EXPAND instead of BROADCAST (they're equivalent in our IR).
pub fn fix_after_broadcast(after: &Rc<UOp>) -> Option<Rc<UOp>> {
    // Only match AFTER operations
    let (passthrough, deps) = match after.op() {
        Op::After { passthrough, deps } => (passthrough, deps),
        _ => return None,
    };

    // Check if passthrough is an EXPAND (broadcast) operation
    let expand_src = match passthrough.op() {
        Op::Expand { src, .. } => src,
        _ => return None,
    };

    // Check if expand source has RANGE parents (indicates local buffer)
    // This requires building a consumer map to check parent relationships
    #[allow(clippy::mutable_key_type)]
    let consumer_map = expand_src.get_consumer_map();
    let has_range_parents = consumer_map
        .get(&UOpKey(expand_src.clone()))
        .map(|consumers| consumers.iter().any(|c| matches!(c.op(), Op::Range { .. })))
        .unwrap_or(false);

    if has_range_parents {
        panic!("can't have a local AFTER");
    }

    // Replace AFTER's passthrough with expand source
    // New structure: AFTER(passthrough=expand_src, deps=deps)
    let new_after = UOp::after(expand_src.clone(), deps.clone());

    Some(new_after)
}

/// Create patterns for codegen preparation.
///
/// This function builds a PatternMatcher with patterns that prepare kernel IR
/// for code generation. Currently implements:
///
/// 1. **remove_noop**: NOOP → zero constant
/// 2. **get_contiguous**: Remove CONTIGUOUS markers
/// 3. **fix_after_broadcast**: Fix AFTER wrapping EXPAND (broadcast)
///
/// Future patterns to add (from MISSING_RANGEIFY_PATTERNS.md):
/// 4. **add_load_to_index (broadcast)**: Generate LOAD for broadcast INDEX (investigation needed)
/// 5. **add_load_to_index (GEP)**: Generate LOAD for GEP INDEX (investigation needed)
///
/// # Returns
///
/// A PatternMatcher<()> with all codegen preparation patterns (no context needed)
///
/// # Example
///
/// ```ignore
/// let matcher = rangeify_codegen_patterns();
///
/// // Apply patterns via graph_rewrite
/// let result = graph_rewrite(&matcher, computation, &mut ());
/// ```
///
/// Based on Tinygrad's rangeify_codegen (schedule/rangeify.py:440-465).
pub fn rangeify_codegen_patterns() -> PatternMatcher<()> {
    // NOTE: add_load_to_index patterns from Tinygrad are NOT needed in Morok.
    //
    // Architectural difference:
    // - Tinygrad: INDEX operation serves dual purpose (pointer-returning vs value-returning)
    //   - Patterns convert value-returning INDEX → pointer INDEX + LOAD
    // - Morok: INDEX and LOAD are separate operations from the start
    //   - INDEX: High-level buffer indexing (used in bufferize_to_store)
    //   - LOAD/STORE: Low-level memory operations (created during kernel splitting)
    //
    // Our bufferize_to_store.rs already creates STORE operations that take INDEX
    // as an argument, so no conversion pattern is needed.

    // NOTE: flatten_range patterns are NOT integrated here.
    // flatten_range_impl() in flatten_range.rs requires a consumer map and substitution pass.
    // It's currently implemented as a direct transformation function, not a pattern.
    // Future work: Add substitution pass infrastructure to support flatten_range integration.

    crate::patterns! {
        noop if matches!(noop.op(), Op::Noop) => remove_noop(noop),
        cont if matches!(cont.op(), Op::Contiguous { .. }) => get_contiguous(cont),
        after if matches!(after.op(), Op::After { .. }) => fix_after_broadcast(after),
    }
}
