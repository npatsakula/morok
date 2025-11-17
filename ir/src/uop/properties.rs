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

use crate::cached_property;
use crate::{Op, UOpKey};
use std::collections::HashSet;
use std::rc::Rc;

// ============================================================================
// Shape Property
// ============================================================================

cached_property! {
    /// Cached shape property.
    ///
    /// Computes the shape of a UOp via shape inference rules.
    /// Shape is `None` for control flow operations (SINK, END, KERNEL, etc.)
    /// and `Some(shape)` for tensor operations.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use morok_ir::uop::properties::ShapeProperty;
    /// use morok_ir::uop::cached_property::CachedProperty;
    ///
    /// let shape = ShapeProperty::get(&my_uop);
    /// ```
    ShapeProperty: Option<crate::shape::Shape> {
        cache_field: shape_cache,
        compute: |uop| crate::shape::infer_shape_from_op(uop)
    }
}

// ============================================================================
// Ranges Property
// ============================================================================

cached_property! {
    /// Cached ranges property.
    ///
    /// Returns all RANGE operations in the computation graph, collected via
    /// toposort and filtering.
    ///
    /// This is different from `in_scope_ranges` which only returns ranges
    /// that are currently "active" (not yet ended).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use morok_ir::uop::properties::RangesProperty;
    /// use morok_ir::uop::cached_property::CachedProperty;
    ///
    /// let all_ranges = RangesProperty::get(&my_uop);
    /// println!("Found {} ranges in graph", all_ranges.len());
    /// ```
    RangesProperty: Vec<Rc<crate::UOp>> {
        cache_field: ranges_cache,
        compute: |uop| {
            uop.toposort()
                .into_iter()
                .filter(|node| matches!(node.op, Op::Range { .. }))
                .collect()
        }
    }
}

// ============================================================================
// In-Scope Ranges Property
// ============================================================================

cached_property! {
    /// Cached in-scope ranges property.
    ///
    /// Returns only the RANGE operations that are "in scope" at this UOp,
    /// meaning they are currently active (not yet ended).
    ///
    /// This is computed bottom-up via toposort:
    /// 1. Merge ranges from all source operations
    /// 2. Remove ranges that are ended by this operation
    /// 3. Add self if this is a RANGE operation
    ///
    /// # Example
    ///
    /// ```ignore
    /// use morok_ir::uop::properties::InScopeRangesProperty;
    /// use morok_ir::uop::cached_property::CachedProperty;
    ///
    /// let in_scope = InScopeRangesProperty::get(&my_uop);
    /// for range in in_scope {
    ///     println!("Range {} is in scope", range.0.id);
    /// }
    /// ```
    InScopeRangesProperty: HashSet<UOpKey> {
        cache_field: in_scope_ranges_cache,
        compute: |uop| uop.compute_in_scope_ranges()
    }
}
