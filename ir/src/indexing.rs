//! Multi-dimensional indexing and slicing support.
//!
//! This module provides NumPy-style indexing through [`IndexSpec`] and the [`s!`](crate::s) macro.
//!
//! For UOp slice methods, see [`crate::uop::constructors::memory`].

use std::sync::Arc;

use crate::uop::UOp;

// Note: Rc and UOp are used by IndexSpec::Single and IndexSpec::Range

/// Index specification for multi-dimensional slicing.
///
/// Similar to NumPy/ndarray indexing:
/// - `Single(idx)`: Select single element (like `arr[5]`)
/// - `Range{start, end, step}`: Slice range (like `arr[0:10:2]`)
/// - `Full`: Select all elements (like `arr[:]`)
/// - `NewAxis`: Add new dimension (like `arr[np.newaxis]`)
///
/// # Example
/// ```ignore
/// use morok_ir::{s, IndexSpec, UOp};
///
/// // Using macro syntax
/// let specs = vec![
///     s![idx],              // Single index
///     s![..],               // Full slice
///     s![start, end],       // Range
///     s![start, end, step], // Range with step
///     s![NewAxis],          // New axis
/// ];
/// ```
#[derive(Debug, Clone)]
pub enum IndexSpec {
    /// Single integer index - selects one element and removes dimension.
    Single(Arc<UOp>),

    /// Range with optional step - selects multiple elements.
    Range { start: Arc<UOp>, end: Arc<UOp>, step: Option<Arc<UOp>> },

    /// Full slice - selects all elements along this dimension.
    Full,

    /// New axis - adds a dimension of size 1.
    NewAxis,
}

/// Slice macro for creating IndexSpec instances.
///
/// Similar to ndarray's `s![]` macro, provides syntactic sugar for slicing.
///
/// # Syntax
/// - `s![idx]` → `IndexSpec::Single(idx)`
/// - `s![..]` → `IndexSpec::Full`
/// - `s![start, end]` → `IndexSpec::Range{start, end, step: None}`
/// - `s![start, end, step]` → `IndexSpec::Range{start, end, step: Some(step)}`
/// - `s![NewAxis]` → `IndexSpec::NewAxis`
///
/// # Example
/// ```ignore
/// let buf = UOp::new_buffer(DeviceSpec::Cpu, 1000, DType::Float32);
/// let idx = UOp::const_(DType::Int32, ConstValue::Int(5));
/// let start = UOp::const_(DType::Int32, ConstValue::Int(0));
/// let end = UOp::const_(DType::Int32, ConstValue::Int(10));
///
/// let slice = UOp::slice(buf, vec![
///     s![start, end],  // Range 0..10
///     s![idx],         // Single index at 5
///     s![..],          // Full slice
/// ]);
/// ```
#[macro_export]
macro_rules! s {
    // Full slice: s![..]
    (..) => {
        $crate::IndexSpec::Full
    };

    // Single index: s![idx]
    ($idx:expr) => {
        $crate::IndexSpec::Single($idx)
    };

    // Range without step: s![start, end]
    ($start:expr, $end:expr) => {
        $crate::IndexSpec::Range { start: $start, end: $end, step: None }
    };

    // Range with step: s![start, end, step]
    ($start:expr, $end:expr, $step:expr) => {
        $crate::IndexSpec::Range { start: $start, end: $end, step: Some($step) }
    };

    // NewAxis: s![NewAxis]
    (NewAxis) => {
        $crate::IndexSpec::NewAxis
    };
}

// UOp::slice and UOp::slice_gated methods have been moved to uop/constructors/memory.rs
