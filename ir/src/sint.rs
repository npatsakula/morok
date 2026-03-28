//! Symbolic Integer (SInt) - dimensions that can be either concrete or symbolic.
//!
//! This module provides the SInt type, which represents a dimension that can be:
//! - A concrete compile-time constant (`usize`)
//! - A symbolic runtime expression (`Arc<UOp>`)
//! - An inference placeholder (`-1` in reshape)
//!
//! Arithmetic via `std::ops` traits matches Tinygrad's approach: concrete values fold
//! inline via native arithmetic, symbolic values produce UOp graph nodes. No
//! simplification at construction time — the pipeline handles it.
//!
//! `SInt::Infer` panics on any arithmetic — it must be resolved at the API boundary
//! (e.g. `try_reshape`) before computing.

use std::fmt;
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

    /// Infer this dimension from the total element count (reshape -1 placeholder).
    Infer,
}

// Manual implementations using stable ID equality for Symbolic (consistent with hash consing)
impl PartialEq for SInt {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SInt::Const(a), SInt::Const(b)) => a == b,
            (SInt::Symbolic(a), SInt::Symbolic(b)) => a.id == b.id,
            (SInt::Infer, SInt::Infer) => true,
            _ => false,
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
            SInt::Infer => {}
        }
    }
}

impl fmt::Display for SInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SInt::Const(v) => write!(f, "{v}"),
            SInt::Symbolic(_) => write!(f, "<symbolic>"),
            SInt::Infer => write!(f, "-1"),
        }
    }
}

impl SInt {
    /// Check if this is a concrete constant.
    pub fn is_const(&self) -> bool {
        matches!(self, SInt::Const(_))
    }

    /// Check if this is an infer placeholder (-1).
    pub fn is_infer(&self) -> bool {
        matches!(self, SInt::Infer)
    }

    /// Check if this is a symbolic expression.
    pub fn is_symbolic(&self) -> bool {
        matches!(self, SInt::Symbolic(_))
    }

    /// Check if this dimension is concrete (not symbolic and not infer).
    pub fn is_concrete(&self) -> bool {
        matches!(self, SInt::Const(_))
    }

    /// Get concrete value if this is a constant, None otherwise.
    pub fn as_const(&self) -> Option<usize> {
        match self {
            SInt::Const(v) => Some(*v),
            SInt::Symbolic(_) | SInt::Infer => None,
        }
    }

    /// Get symbolic UOp if this is symbolic, None otherwise.
    pub fn as_symbolic(&self) -> Option<&Arc<UOp>> {
        match self {
            SInt::Symbolic(uop) => Some(uop),
            SInt::Const(_) | SInt::Infer => None,
        }
    }

    /// Convert to UOp (const or passthrough). Panics on Infer.
    pub fn to_uop(&self, dtype: morok_dtype::DType) -> Arc<UOp> {
        match self {
            SInt::Const(v) => UOp::const_(dtype, crate::ConstValue::Int(*v as i64)),
            SInt::Symbolic(uop) => {
                if uop.dtype() != dtype {
                    uop.cast(dtype)
                } else {
                    uop.clone()
                }
            }
            SInt::Infer => panic!("cannot convert SInt::Infer to UOp — resolve -1 first"),
        }
    }

    /// Try to simplify symbolic expression to concrete value if possible.
    ///
    /// For const values, returns self. For symbolic, attempts to evaluate
    /// if it's a constant expression. No full symbolic simplification —
    /// that's deferred to the pipeline (matching Tinygrad).
    pub fn simplify(&self) -> Self {
        match self {
            SInt::Const(_) | SInt::Infer => self.clone(),
            SInt::Symbolic(uop) => {
                if let crate::Op::Const(const_hash) = uop.op() {
                    match const_hash.0 {
                        crate::ConstValue::Int(v) if v >= 0 => SInt::Const(v as usize),
                        crate::ConstValue::UInt(v) => SInt::Const(v as usize),
                        _ => self.clone(),
                    }
                } else {
                    self.clone()
                }
            }
        }
    }

    /// Ceiling division: ceildiv(a, b) = (a + b - 1) / b.
    /// Both operands must be positive. Panics on Infer.
    pub fn ceildiv(&self, rhs: &SInt) -> SInt {
        (self + rhs - 1usize) / rhs
    }

    /// Maximum of two SInt values. Panics on Infer.
    pub fn smax(&self, rhs: &SInt) -> SInt {
        match (self, rhs) {
            (SInt::Infer, _) | (_, SInt::Infer) => {
                panic!("smax on SInt::Infer — resolve -1 before computing")
            }
            (SInt::Const(a), SInt::Const(b)) => SInt::Const(*a.max(b)),
            _ => {
                let a = self.to_uop(morok_dtype::DType::Index);
                let b = rhs.to_uop(morok_dtype::DType::Index);
                SInt::Symbolic(a.try_max(&b).unwrap())
            }
        }
    }

    /// Minimum of two SInt values. Panics on Infer.
    /// Follows Tinygrad: `min(a, b) = -max(-a, -b)`.
    pub fn smin(&self, rhs: &SInt) -> SInt {
        match (self, rhs) {
            (SInt::Infer, _) | (_, SInt::Infer) => {
                panic!("smin on SInt::Infer — resolve -1 before computing")
            }
            (SInt::Const(a), SInt::Const(b)) => SInt::Const(*a.min(b)),
            _ => {
                let a = self.to_uop(morok_dtype::DType::Index);
                let b = rhs.to_uop(morok_dtype::DType::Index);
                // min(a, b) = -max(-a, -b)
                let neg_max = a.neg().try_max(&b.neg()).unwrap();
                SInt::Symbolic(neg_max.neg())
            }
        }
    }
}

// =========================================================================
// std::ops arithmetic
// =========================================================================

macro_rules! impl_sint_binop {
    ($trait:ident, $method:ident, $concrete_op:tt, $uop_method:ident) => {
        // Primary: &SInt op &SInt
        impl std::ops::$trait for &SInt {
            type Output = SInt;
            fn $method(self, rhs: &SInt) -> SInt {
                match (self, rhs) {
                    (SInt::Infer, _) | (_, SInt::Infer) => {
                        panic!("arithmetic on SInt::Infer — resolve -1 before computing")
                    }
                    (SInt::Const(a), SInt::Const(b)) => SInt::Const(a $concrete_op b),
                    _ => {
                        let a = self.to_uop(morok_dtype::DType::Index);
                        let b = rhs.to_uop(morok_dtype::DType::Index);
                        SInt::Symbolic(a.$uop_method(&b).unwrap())
                    }
                }
            }
        }
        // Forward: owned variants
        impl std::ops::$trait for SInt {
            type Output = SInt;
            fn $method(self, rhs: SInt) -> SInt { (&self).$method(&rhs) }
        }
        impl std::ops::$trait<&SInt> for SInt {
            type Output = SInt;
            fn $method(self, rhs: &SInt) -> SInt { (&self).$method(rhs) }
        }
        impl std::ops::$trait<SInt> for &SInt {
            type Output = SInt;
            fn $method(self, rhs: SInt) -> SInt { self.$method(&rhs) }
        }
        // Convenience: SInt op usize
        impl std::ops::$trait<usize> for &SInt {
            type Output = SInt;
            fn $method(self, rhs: usize) -> SInt { self.$method(&SInt::Const(rhs)) }
        }
        impl std::ops::$trait<usize> for SInt {
            type Output = SInt;
            fn $method(self, rhs: usize) -> SInt { (&self).$method(&SInt::Const(rhs)) }
        }
        // Convenience: usize op SInt
        impl std::ops::$trait<SInt> for usize {
            type Output = SInt;
            fn $method(self, rhs: SInt) -> SInt { (&SInt::Const(self)).$method(&rhs) }
        }
        impl std::ops::$trait<&SInt> for usize {
            type Output = SInt;
            fn $method(self, rhs: &SInt) -> SInt { (&SInt::Const(self)).$method(rhs) }
        }
    };
}

impl_sint_binop!(Add, add, +, try_add);
impl_sint_binop!(Mul, mul, *, try_mul);
impl_sint_binop!(Div, div, /, try_div);

// Sub is implemented separately to guard against usize underflow.
// SInt represents non-negative dimension sizes; concrete a - b must have a >= b.
impl std::ops::Sub for &SInt {
    type Output = SInt;
    fn sub(self, rhs: &SInt) -> SInt {
        match (self, rhs) {
            (SInt::Infer, _) | (_, SInt::Infer) => {
                panic!("arithmetic on SInt::Infer — resolve -1 before computing")
            }
            (SInt::Const(a), SInt::Const(b)) => {
                assert!(a >= b, "SInt subtraction underflow: {a} - {b} would be negative");
                SInt::Const(a - b)
            }
            _ => {
                let a = self.to_uop(morok_dtype::DType::Index);
                let b = rhs.to_uop(morok_dtype::DType::Index);
                SInt::Symbolic(a.try_sub(&b).unwrap())
            }
        }
    }
}
impl std::ops::Sub for SInt {
    type Output = SInt;
    fn sub(self, rhs: SInt) -> SInt {
        (&self).sub(&rhs)
    }
}
impl std::ops::Sub<&SInt> for SInt {
    type Output = SInt;
    fn sub(self, rhs: &SInt) -> SInt {
        (&self).sub(rhs)
    }
}
impl std::ops::Sub<SInt> for &SInt {
    type Output = SInt;
    fn sub(self, rhs: SInt) -> SInt {
        self.sub(&rhs)
    }
}
impl std::ops::Sub<usize> for &SInt {
    type Output = SInt;
    fn sub(self, rhs: usize) -> SInt {
        self.sub(&SInt::Const(rhs))
    }
}
impl std::ops::Sub<usize> for SInt {
    type Output = SInt;
    fn sub(self, rhs: usize) -> SInt {
        (&self).sub(&SInt::Const(rhs))
    }
}
impl std::ops::Sub<SInt> for usize {
    type Output = SInt;
    fn sub(self, rhs: SInt) -> SInt {
        (&SInt::Const(self)).sub(&rhs)
    }
}
impl std::ops::Sub<&SInt> for usize {
    type Output = SInt;
    fn sub(self, rhs: &SInt) -> SInt {
        (&SInt::Const(self)).sub(rhs)
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

impl From<isize> for SInt {
    /// Converts isize to SInt. `-1` becomes `SInt::Infer` (reshape inference placeholder).
    /// Panics on other negative values.
    fn from(value: isize) -> Self {
        if value == -1 {
            SInt::Infer
        } else {
            assert!(value >= 0, "negative dimension {value} is invalid (only -1 for inference is allowed)");
            SInt::Const(value as usize)
        }
    }
}

impl From<&isize> for SInt {
    fn from(value: &isize) -> Self {
        SInt::from(*value)
    }
}

impl From<i32> for SInt {
    fn from(value: i32) -> Self {
        if value == -1 {
            SInt::Infer
        } else {
            assert!(value >= 0, "negative dimension {value} is invalid (only -1 for inference is allowed)");
            SInt::Const(value as usize)
        }
    }
}

impl From<&i32> for SInt {
    fn from(value: &i32) -> Self {
        SInt::from(*value)
    }
}

impl From<i64> for SInt {
    fn from(value: i64) -> Self {
        if value == -1 {
            SInt::Infer
        } else {
            assert!(value >= 0, "negative dimension {value} is invalid (only -1 for inference is allowed)");
            SInt::Const(value as usize)
        }
    }
}

impl From<&i64> for SInt {
    fn from(value: &i64) -> Self {
        SInt::from(*value)
    }
}

impl From<&usize> for SInt {
    fn from(value: &usize) -> Self {
        SInt::Const(*value)
    }
}

impl From<&SInt> for SInt {
    fn from(value: &SInt) -> Self {
        value.clone()
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
// Shrink range conversion
// =========================================================================

/// A shrink range that may be `None` (keep dim), concrete, or symbolic.
///
/// Intermediate representation used by [`IntoShrinkRange`]. isize values
/// are preserved for negative-index resolution at the tensor layer.
#[derive(Debug, Clone)]
pub enum ShrinkRange {
    /// Keep entire dimension (identity).
    None,
    /// Concrete range with possibly-negative indices (resolved by tensor layer).
    Isize(isize, isize),
    /// SInt range (concrete or symbolic, always non-negative).
    Sint(SInt, SInt),
}

/// Trait for types convertible to a shrink range specification.
pub trait IntoShrinkRange {
    fn into_shrink_range(self) -> ShrinkRange;
}

impl IntoShrinkRange for Option<(isize, isize)> {
    fn into_shrink_range(self) -> ShrinkRange {
        match self {
            Some((b, e)) => ShrinkRange::Isize(b, e),
            ::core::option::Option::None => ShrinkRange::None,
        }
    }
}

impl IntoShrinkRange for &Option<(isize, isize)> {
    fn into_shrink_range(self) -> ShrinkRange {
        (*self).into_shrink_range()
    }
}

impl IntoShrinkRange for (isize, isize) {
    fn into_shrink_range(self) -> ShrinkRange {
        ShrinkRange::Isize(self.0, self.1)
    }
}

impl IntoShrinkRange for &(isize, isize) {
    fn into_shrink_range(self) -> ShrinkRange {
        ShrinkRange::Isize(self.0, self.1)
    }
}

// (i32, i32) → concrete range (ONNX often uses i32)
impl IntoShrinkRange for (i32, i32) {
    fn into_shrink_range(self) -> ShrinkRange {
        ShrinkRange::Isize(self.0 as isize, self.1 as isize)
    }
}

impl IntoShrinkRange for &(i32, i32) {
    fn into_shrink_range(self) -> ShrinkRange {
        ShrinkRange::Isize(self.0 as isize, self.1 as isize)
    }
}

// (usize, usize) → concrete range
impl IntoShrinkRange for (usize, usize) {
    fn into_shrink_range(self) -> ShrinkRange {
        ShrinkRange::Sint(SInt::Const(self.0), SInt::Const(self.1))
    }
}

impl IntoShrinkRange for &(usize, usize) {
    fn into_shrink_range(self) -> ShrinkRange {
        ShrinkRange::Sint(SInt::Const(self.0), SInt::Const(self.1))
    }
}

impl IntoShrinkRange for Option<(SInt, SInt)> {
    fn into_shrink_range(self) -> ShrinkRange {
        match self {
            Some((b, e)) => ShrinkRange::Sint(b, e),
            ::core::option::Option::None => ShrinkRange::None,
        }
    }
}

impl IntoShrinkRange for &Option<(SInt, SInt)> {
    fn into_shrink_range(self) -> ShrinkRange {
        self.clone().into_shrink_range()
    }
}

impl IntoShrinkRange for (SInt, SInt) {
    fn into_shrink_range(self) -> ShrinkRange {
        ShrinkRange::Sint(self.0, self.1)
    }
}

impl IntoShrinkRange for &(SInt, SInt) {
    fn into_shrink_range(self) -> ShrinkRange {
        ShrinkRange::Sint(self.0.clone(), self.1.clone())
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
    values.iter().fold(SInt::Const(1), |acc, v| &acc * v)
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
    assert!(!values.is_empty(), "sint_max requires at least one value");
    values.iter().skip(1).fold(values[0].clone(), |acc, v| acc.smax(v))
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
    assert!(!values.is_empty(), "sint_min requires at least one value");
    values.iter().skip(1).fold(values[0].clone(), |acc, v| acc.smin(v))
}
