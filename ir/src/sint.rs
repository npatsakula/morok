//! Symbolic Integer (SInt) - dimensions that can be either concrete or symbolic.
//!
//! This module provides the SInt type, which represents a dimension that can be:
//! - A concrete compile-time constant (`usize`)
//! - A symbolic runtime expression (`Arc<UOp>`)
//!
//! This follows Tinygrad's approach where shapes can contain mixed concrete/symbolic
//! dimensions, enabling dynamic shapes (variable batch sizes, dynamic sequences, etc.).

use std::sync::Arc;

use crate::UOp;

/// Symbolic Integer - either a concrete value or a symbolic UOp expression.
///
/// # Examples
///
/// ```rust
/// # use morok_ir::{SInt, UOp, ConstValue, Op};
/// # use morok_dtype::DType;
/// // Concrete dimension
/// let static_dim = SInt::from(32);
/// assert!(static_dim.is_const());
/// assert_eq!(static_dim.as_const(), Some(32));
///
/// // Symbolic dimension - use DefineVar for truly dynamic dimensions
/// let batch_size = UOp::new(
///     Op::DefineVar { name: "batch".to_string(), min_val: 1, max_val: 1024 },
///     DType::Index,
/// );
/// let dynamic_dim = SInt::from(batch_size);
/// assert!(!dynamic_dim.is_const());
/// ```
#[derive(Debug, Clone)]
pub enum SInt {
    /// Concrete compile-time constant dimension.
    Const(usize),

    /// Symbolic runtime expression (must have dtype Index or Int).
    Symbolic(Arc<UOp>),
}

// Manual implementations using stable ID equality for Symbolic (consistent with hash consing)
impl PartialEq for SInt {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SInt::Const(a), SInt::Const(b)) => a == b,
            (SInt::Symbolic(a), SInt::Symbolic(b)) => a.id == b.id,
            (SInt::Const(_), SInt::Symbolic(_)) | (SInt::Symbolic(_), SInt::Const(_)) => false,
        }
    }
}

impl Eq for SInt {}

impl std::hash::Hash for SInt {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            SInt::Const(v) => v.hash(state),
            SInt::Symbolic(uop) => uop.id.hash(state),
        }
    }
}

impl SInt {
    /// Check if this is a concrete constant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use morok_ir::SInt;
    /// let s = SInt::from(42);
    /// assert!(s.is_const());
    /// ```
    pub fn is_const(&self) -> bool {
        matches!(self, SInt::Const(_))
    }

    /// Check if this is a symbolic expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use morok_ir::{SInt, UOp, ConstValue, Op};
    /// # use morok_dtype::DType;
    /// let uop =
    ///     UOp::new(Op::DefineVar { name: "n".to_string(), min_val: 1, max_val: 100 }, DType::Index);
    /// let s = SInt::from(uop);
    /// assert!(s.is_symbolic());
    /// ```
    pub fn is_symbolic(&self) -> bool {
        matches!(self, SInt::Symbolic(_))
    }

    /// Get concrete value if this is a constant, None otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use morok_ir::SInt;
    /// let s = SInt::from(42);
    /// assert_eq!(s.as_const(), Some(42));
    ///
    /// # use morok_ir::{UOp, ConstValue, Op};
    /// # use morok_dtype::DType;
    /// let uop =
    ///     UOp::new(Op::DefineVar { name: "n".to_string(), min_val: 1, max_val: 100 }, DType::Index);
    /// let s = SInt::from(uop);
    /// assert_eq!(s.as_const(), None);
    /// ```
    pub fn as_const(&self) -> Option<usize> {
        match self {
            SInt::Const(v) => Some(*v),
            SInt::Symbolic(_) => None,
        }
    }

    /// Get symbolic UOp if this is symbolic, None otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use morok_ir::{SInt, UOp, ConstValue, Op};
    /// # use morok_dtype::DType;
    /// let uop =
    ///     UOp::new(Op::DefineVar { name: "n".to_string(), min_val: 1, max_val: 100 }, DType::Index);
    /// let s = SInt::from(uop.clone());
    /// assert!(s.as_symbolic().is_some());
    /// ```
    pub fn as_symbolic(&self) -> Option<&Arc<UOp>> {
        match self {
            SInt::Const(_) => None,
            SInt::Symbolic(uop) => Some(uop),
        }
    }

    /// Convert to UOp (const or passthrough).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use morok_ir::{SInt, ConstValue};
    /// # use morok_dtype::DType;
    /// let s = SInt::from(42);
    /// let uop = s.to_uop(DType::Index);
    /// assert_eq!(uop.dtype(), DType::Index);
    /// ```
    pub fn to_uop(&self, dtype: morok_dtype::DType) -> Arc<UOp> {
        match self {
            SInt::Const(v) => UOp::const_(dtype, crate::ConstValue::Int(*v as i64)),
            SInt::Symbolic(uop) => {
                if uop.dtype() != dtype {
                    UOp::cast(uop.clone(), dtype)
                } else {
                    uop.clone()
                }
            }
        }
    }

    /// Try to simplify symbolic expression to concrete value if possible.
    ///
    /// For const values, returns self. For symbolic, attempts to evaluate
    /// if it's a constant expression.
    pub fn simplify(&self) -> Self {
        match self {
            SInt::Const(_) => self.clone(),
            SInt::Symbolic(uop) => {
                // Try to extract constant value from UOp if it's a Const op
                if let crate::Op::Const(const_hash) = uop.op() {
                    match const_hash.0 {
                        crate::ConstValue::Int(v) if v >= 0 => SInt::Const(v as usize),
                        crate::ConstValue::UInt(v) => SInt::Const(v as usize),
                        _ => self.clone(),
                    }
                } else {
                    // Could implement symbolic simplification here (const folding, etc.)
                    self.clone()
                }
            }
        }
    }
}

// =========================================================================
// Conversions
// =========================================================================

impl From<usize> for SInt {
    fn from(value: usize) -> Self {
        SInt::Const(value)
    }
}

impl From<Arc<UOp>> for SInt {
    fn from(value: Arc<UOp>) -> Self {
        // Try to extract constant value if possible
        let sint = SInt::Symbolic(value);
        sint.simplify()
    }
}

impl From<&Arc<UOp>> for SInt {
    fn from(value: &Arc<UOp>) -> Self {
        SInt::from(value.clone())
    }
}

// =========================================================================
// Utilities
// =========================================================================

/// Compute product of SInt values.
///
/// If all values are concrete, returns Const(product).
/// Otherwise, constructs symbolic multiplication UOp.
///
/// # Examples
///
/// ```rust
/// # use morok_ir::{SInt, sint_prod};
/// let dims = vec![SInt::from(2), SInt::from(3), SInt::from(4)];
/// let result = sint_prod(&dims);
/// assert_eq!(result.as_const(), Some(24));
/// ```
pub fn sint_prod(values: &[SInt]) -> SInt {
    sint_aggregate(values, 1, |vals| vals.into_iter().product(), |a, b| a.try_mul(&b), "multiplication")
}

/// Helper for aggregating SInt values with both concrete and symbolic paths.
fn sint_aggregate<F, G>(
    values: &[SInt],
    empty_val: usize,
    concrete_op: F,
    symbolic_op: G,
    op_name: &'static str,
) -> SInt
where
    F: FnOnce(Vec<usize>) -> usize,
    G: Fn(Arc<UOp>, Arc<UOp>) -> crate::Result<Arc<UOp>>,
{
    use morok_dtype::DType;

    if values.is_empty() {
        return SInt::Const(empty_val);
    }

    // If all concrete, compute directly
    if values.iter().all(|s| s.is_const()) {
        let concrete_vals: Vec<usize> = values.iter().map(|s| s.as_const().unwrap()).collect();
        return SInt::Const(concrete_op(concrete_vals));
    }

    // Build symbolic operation tree
    let uops: Vec<_> = values.iter().map(|s| s.to_uop(DType::Index)).collect();
    let result = uops
        .into_iter()
        .reduce(|acc, uop| symbolic_op(acc, uop).unwrap_or_else(|_| panic!("Index {} should not fail", op_name)));

    SInt::from(result.expect("Aggregation of non-empty vec should succeed"))
}

/// Compute maximum of SInt values (symbolic-aware).
///
/// # Examples
///
/// ```rust
/// # use morok_ir::{SInt, sint_max};
/// let a = SInt::from(10);
/// let b = SInt::from(20);
/// let result = sint_max(&[a, b]);
/// assert_eq!(result.as_const(), Some(20));
/// ```
pub fn sint_max(values: &[SInt]) -> SInt {
    sint_aggregate(values, 0, |vals| vals.into_iter().max().unwrap(), |a, b| a.try_max(&b), "max")
}

/// Compute minimum of SInt values (symbolic-aware).
///
/// Note: Currently falls back to first element for symbolic cases since
/// UOp::try_min_op is not yet implemented.
///
/// # Examples
///
/// ```rust
/// # use morok_ir::{SInt, sint_min};
/// let a = SInt::from(10);
/// let b = SInt::from(20);
/// let result = sint_min(&[a, b]);
/// assert_eq!(result.as_const(), Some(10));
/// ```
pub fn sint_min(values: &[SInt]) -> SInt {
    if values.is_empty() {
        return SInt::Const(usize::MAX);
    }

    // If all concrete, compute min directly
    if values.iter().all(|s| s.is_const()) {
        let minimum = values.iter().map(|s| s.as_const().unwrap()).min().unwrap();
        return SInt::Const(minimum);
    }

    // TODO: Implement symbolic min when UOp::try_min_op is available
    // For now, fall back to first element for symbolic cases
    values[0].clone()
}
