//! Cached property infrastructure for UOps.
//!
//! This module provides a reusable pattern for graph properties that need to be:
//! - Computed lazily (on first access)
//! - Cached permanently (in OnceCell)
//! - Computed bottom-up via toposort
//! - Optimized with filtered toposort (skip cached nodes)
//!
//! # Architecture
//!
//! The pattern consists of three components:
//!
//! 1. **OnceCell cache fields** in the UOp struct (explicit, visible)
//! 2. **CachedProperty trait** providing the computation logic
//! 3. **cached_property! macro** reducing boilerplate
//!
//! # Performance
//!
//! The key optimization is **filtered toposort**. Instead of traversing the entire
//! graph on every cache miss, we only traverse nodes that don't have the property
//! cached yet.
//!
//! For a graph with 10,000 nodes where 9,900 are already cached:
//! - **Without filtering**: 10,000 nodes visited
//! - **With filtering**: 100 nodes visited
//! - **Speedup**: 100x
//!
//! # Example
//!
//! ```ignore
//! use morok_ir::cached_property;
//!
//! // Define a new cached property
//! cached_property! {
//!     MyProperty: MyType {
//!         cache_field: my_property_cache,
//!         compute: |uop| {
//!             // Computation logic here
//!             // Can call MyProperty::get(child) on children
//!             // (they're guaranteed to be computed already)
//!         }
//!     }
//! }
//!
//! // Use it in UOp's public API
//! impl UOp {
//!     pub fn my_property(&self: &Arc<Self>) -> &MyType {
//!         MyProperty::get(self)
//!     }
//! }
//! ```

use std::sync::Arc;
use std::sync::OnceLock;

use crate::UOp;

/// Trait for computed properties that can be cached on UOps.
///
/// Properties are computed bottom-up via filtered toposort, ensuring:
/// 1. Dependencies are computed before dependents (toposort order)
/// 2. Already-cached nodes are skipped (filtered toposort)
/// 3. Each node's property is computed exactly once (OnceLock)
///
/// # Implementation Pattern
///
/// The trait provides a default `get()` implementation that:
/// 1. Returns cached value if available (fast path)
/// 2. Otherwise, performs filtered toposort to find uncached nodes
/// 3. Computes properties bottom-up, caching each result
/// 4. Returns the final cached value
///
/// ```ignore
/// impl CachedProperty for MyProperty {
///     fn get(uop: &Arc<UOp>) -> &Self::Value {
///         // Fast path: already cached
///         if let Some(val) = Self::cache(uop).get() {
///             return val;
///         }
///
///         // Filtered toposort: only uncached nodes
///         let uncached = uop.toposort_filtered(|n| Self::cache(n).get().is_none());
///
///         // Compute bottom-up
///         for node in uncached {
///             Self::cache(&node).get_or_init(|| Self::compute(&node));
///         }
///
///         Self::cache(uop).get().unwrap()
///     }
/// }
/// ```
pub trait CachedProperty: Sized + 'static {
    /// The type of the cached value.
    type Value: Clone;

    /// Compute this property for a single node.
    ///
    /// When this is called, all children are guaranteed to have this
    /// property already computed (via toposort order), so you can safely
    /// call `Self::get(child)` on any child node.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn compute(uop: &Arc<UOp>) -> Self::Value {
    ///     match &uop.op {
    ///         Op::Unary(_, src) => {
    ///             // Safe: src is a child, so it's already computed
    ///             let src_val = Self::get(src);
    ///             // ... use src_val
    ///         }
    ///         // ...
    ///     }
    /// }
    /// ```
    fn compute(uop: &Arc<UOp>) -> Self::Value;

    /// Get the cache cell for this property.
    ///
    /// The cache field must be a `OnceLock<Self::Value>` field in the UOp struct.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn cache(uop: &Arc<UOp>) -> &OnceLock<Self::Value> {
    ///     &uop.my_property_cache
    /// }
    /// ```
    fn cache(uop: &Arc<UOp>) -> &OnceLock<Self::Value>;

    /// Get cached value, computing if needed via filtered toposort.
    ///
    /// This is the main entry point - **use this instead of `compute()`**.
    ///
    /// # Performance
    ///
    /// - **First access**: O(N) where N = number of uncached nodes in graph
    /// - **Subsequent access**: O(1) (cached)
    ///
    /// Key optimization: Uses filtered toposort to skip already-cached nodes,
    /// making incremental updates very fast (10-100x speedup for large graphs).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let shape = ShapeProperty::get(&uop);  // Computes if needed
    /// let shape_again = ShapeProperty::get(&uop);  // Instant (cached)
    /// ```
    fn get(uop: &Arc<UOp>) -> &Self::Value {
        // Fast path: already cached
        if let Some(val) = Self::cache(uop).get() {
            return val;
        }

        // Slow path: filtered toposort + bottom-up computation
        // Only traverse nodes that don't have this property cached yet
        let uncached = uop.toposort_filtered(|node| Self::cache(node).get().is_none());

        // Compute bottom-up (dependencies before dependents)
        for node in uncached {
            Self::cache(&node).get_or_init(|| Self::compute(&node));
        }

        // Must be cached now
        Self::cache(uop).get().expect("property must be cached after toposort")
    }
}

/// Define a cached property on UOp.
///
/// This macro generates a marker struct and implements the `CachedProperty` trait,
/// reducing boilerplate from ~50 lines to ~10 lines per property.
///
/// # Syntax
///
/// ```ignore
/// cached_property! {
///     PropertyName: ReturnType {
///         cache_field: cache_field_name,
///         compute: |uop| { /* computation */ }
///     }
/// }
/// ```
///
/// # Requirements
///
/// 1. The `cache_field` must exist in the UOp struct as `OnceLock<ReturnType>`
/// 2. The `compute` closure must have signature `Fn(&Arc<UOp>) -> ReturnType`
/// 3. `ReturnType` must implement `Clone`
///
/// # Example
///
/// ```ignore
/// use morok_ir::cached_property;
/// use morok_ir::shape::Shape;
///
/// cached_property! {
///     ShapeProperty: Option<Shape> {
///         cache_field: shape_cache,
///         compute: |uop| crate::shape::infer_shape_from_op(uop)
///     }
/// }
///
/// // Now you can use:
/// let shape = ShapeProperty::get(&my_uop);
/// ```
///
/// # Generated Code
///
/// The macro expands to:
///
/// ```ignore
/// pub struct ShapeProperty;
///
/// impl CachedProperty for ShapeProperty {
///     type Value = Option<Shape>;
///
///     fn compute(uop: &Arc<UOp>) -> Self::Value {
///         (|uop| crate::shape::infer_shape_from_op(uop))(uop)
///     }
///
///     fn cache(uop: &Arc<UOp>) -> &OnceCell<Self::Value> {
///         &uop.shape_cache
///     }
/// }
/// ```
#[macro_export]
macro_rules! cached_property {
    (
        $(#[$meta:meta])*
        $name:ident: $value_type:ty {
            cache_field: $cache_field:ident,
            compute: $compute:expr
        }
    ) => {
        $(#[$meta])*
        pub struct $name;

        impl $crate::uop::cached_property::CachedProperty for $name {
            type Value = $value_type;

            fn compute(uop: &std::sync::Arc<$crate::UOp>) -> Self::Value {
                let compute_fn: fn(&std::sync::Arc<$crate::UOp>) -> Self::Value = $compute;
                compute_fn(uop)
            }

            fn cache(uop: &std::sync::Arc<$crate::UOp>) -> &std::sync::OnceLock<Self::Value> {
                &uop.$cache_field
            }
        }
    };
}
