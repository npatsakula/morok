//! Standard cached properties for UOps.
//!
//! This module defines the standard set of cached properties using the
//! `cached_property!` macro infrastructure.
//!
//! # Available Properties
//!
//! - [`ShapeProperty`] - Shape inference for tensor operations
//! - [`RangesProperty`] - All RANGE operations in the graph
//! - [`InScopeRangesProperty`] - RANGE operations currently in scope
//! - [`VminVmaxProperty`] - Range analysis (min/max values) for operations

use crate::cached_property;
use crate::types::ConstValue;
use crate::{Op, UOpKey};
use std::collections::HashSet;
use std::sync::Arc;

// ============================================================================
// Shape Property
// ============================================================================

cached_property! {
    /// Cached shape property.
    ///
    /// Computes the shape of a UOp via shape inference rules.
    /// Returns `Ok(None)` for control flow operations (SINK, END, CALL wrappers, etc.),
    /// `Ok(Some(shape))` for tensor operations, and `Err` for shape mismatches.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use morok_ir::uop::properties::ShapeProperty;
    /// use morok_ir::uop::cached_property::CachedProperty;
    ///
    /// let shape_result = ShapeProperty::get(&my_uop);
    /// ```
    ShapeProperty: crate::Result<Option<crate::shape::Shape>> {
        cache_field: shape_cache,
        compute: |uop| crate::shape::infer_shape_from_op(uop)
    }
}

// ============================================================================
// Ranges Property
// ============================================================================

cached_property! {
    /// Cached ranges property (recursive, like Tinygrad's approach).
    ///
    /// Returns all RANGE operations in the computation graph, computed from
    /// children's cached values. O(N) total on first access, O(1) after.
    ///
    /// This is different from `in_scope_ranges` which only returns ranges
    /// that are currently "active" (not yet ended).
    RangesProperty: Vec<Arc<crate::UOp>> {
        cache_field: ranges_cache,
        compute: |uop| {
            let mut seen = std::collections::HashSet::new();
            let mut result = Vec::new();
            // Self first if RANGE (matches Tinygrad: {self:None} | self._ranges)
            if matches!(uop.op, Op::Range { .. }) {
                seen.insert(uop.id);
                result.push(uop.clone());
            }
            uop.op.map_child(|src| {
                for r in RangesProperty::get(src) {
                    if seen.insert(r.id) {
                        result.push(r.clone());
                    }
                }
            });
            result
        }
    }
}

// ============================================================================
// In-Scope Ranges Property
// ============================================================================

cached_property! {
    /// Cached in-scope ranges property (recursive, like Tinygrad's `@recursive_property`).
    ///
    /// Returns only the RANGE operations that are "in scope" at this UOp,
    /// meaning they are currently active (not yet ended).
    ///
    /// Computed from children's cached values (guaranteed available by
    /// `CachedProperty::get()`'s filtered toposort + bottom-up processing):
    /// 1. Merge in-scope ranges from all source operations
    /// 2. Remove ranges ended by this operation (`op.ended_ranges()`)
    /// 3. Add self if this is a RANGE operation
    ///
    /// This is O(N) total for the first access on a graph, then O(1) for
    /// subsequent accesses on overlapping subgraphs (cached per-node).
    InScopeRangesProperty: HashSet<UOpKey> {
        cache_field: in_scope_ranges_cache,
        compute: |uop| {
            #[allow(clippy::mutable_key_type)]
            let mut result: HashSet<UOpKey> = HashSet::new();

            // Step 1: Merge from all sources' cached in_scope_ranges
            uop.op.map_child(|src| {
                for r in InScopeRangesProperty::get(src).iter() {
                    result.insert(r.clone());
                }
            });

            // Step 2: Remove ended ranges (using existing op.ended_ranges())
            for ended in uop.op.ended_ranges() {
                match ended.op() {
                    Op::Range { .. } => {
                        result.remove(&UOpKey(ended.clone()));
                    }
                    _ => {
                        // Non-RANGE ended (like AFTER) — remove all its in-scope ranges
                        for r in InScopeRangesProperty::get(ended).iter() {
                            result.remove(r);
                        }
                    }
                }
            }

            // Step 3: Add self if RANGE
            if matches!(uop.op, Op::Range { .. }) {
                result.insert(UOpKey(uop.clone()));
            }

            result
        }
    }
}

// ============================================================================
// Backward Slice Property
// ============================================================================

cached_property! {
    /// Cached backward slice: all node IDs reachable from this UOp (including self).
    ///
    /// Tinygrad equivalent: `@functools.cached_property backward_slice` (ops.py:155).
    /// O(N) total on first access, O(1) membership test via `HashSet::contains`.
    ///
    /// This replaces the uncached `backward_slice()` DFS for membership tests.
    /// The old `backward_slice()` returning `Vec<Arc<UOp>>` is kept for callers
    /// that need iteration with full UOp access.
    BackwardSliceProperty: std::collections::HashSet<u64> {
        cache_field: backward_slice_cache,
        compute: |uop| {
            let mut result = std::collections::HashSet::new();
            result.insert(uop.id);
            uop.op.map_child(|src| {
                for &id in BackwardSliceProperty::get(src).iter() {
                    result.insert(id);
                }
            });
            result
        }
    }
}

// ============================================================================
// VminVmax Property
// ============================================================================

cached_property! {
    /// Cached vmin/vmax range analysis property.
    ///
    /// Computes the minimum and maximum possible values for a UOp based on
    /// operation semantics and input ranges. Returns a tuple of (vmin, vmax)
    /// where both values are ConstValue types.
    ///
    /// The analysis is conservative - when in doubt, it returns the full dtype
    /// bounds to avoid incorrect optimizations.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use morok_ir::uop::properties::VminVmaxProperty;
    /// use morok_ir::uop::cached_property::CachedProperty;
    ///
    /// let (vmin, vmax) = VminVmaxProperty::get(&my_uop);
    /// println!("Value range: [{:?}, {:?}]", vmin, vmax);
    /// ```
    VminVmaxProperty: (ConstValue, ConstValue) {
        cache_field: vmin_vmax_cache,
        compute: |uop| {
            crate::uop::range_eval::compute_sound_vmin_vmax(uop)
                .unwrap_or_else(|| crate::uop::range_eval::dtype_bounds(&uop.dtype))
        }
    }
}

// Sound vmin/vmax: returns None for ops without provably correct bounds.
// Use this for optimizations that collapse expressions to constants.
cached_property! {
    SoundVminVmaxProperty: Option<(ConstValue, ConstValue)> {
        cache_field: sound_vmin_vmax_cache,
        compute: |uop| crate::uop::range_eval::compute_sound_vmin_vmax(uop)
    }
}
