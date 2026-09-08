//! Scalar-or-tensor operand adapter for the elementwise APIs.
//!
//! Every binary op ([`Tensor::try_add`](crate::Tensor::try_add) and friends,
//! [`maximum`](crate::Tensor::maximum), [`where_`](crate::Tensor::where_),
//! [`masked_fill`](crate::Tensor::masked_fill), the `clamp` bounds and the
//! `std::ops` operators) takes `impl Into<Operand<'_>>`, so `&t`, `t` and a bare
//! `2.0` are all valid right-hand sides. A scalar is materialized as a constant
//! in the *other* operand's dtype.

use svod_dtype::{DType, ScalarDType};
use svod_ir::ConstValue;

use crate::Tensor;

/// Right-hand side of an elementwise op.
pub enum Operand<'a> {
    /// Borrowed tensor — used as is.
    Tensor(&'a Tensor),
    /// Owned tensor, so `a + b` by value keeps working.
    Owned(Tensor),
    /// Constant scalar, materialized in the other operand's dtype.
    Scalar(ConstValue),
}

impl Operand<'_> {
    /// Dtype of the tensor this operand carries, if any.
    pub(crate) fn dtype(&self) -> Option<DType> {
        match self {
            Self::Tensor(t) => Some(t.dtype()),
            Self::Owned(t) => Some(t.dtype()),
            Self::Scalar(_) => None,
        }
    }

    /// Resolve to a tensor, materializing a scalar with `dtype`.
    #[track_caller]
    pub(crate) fn materialize(self, dtype: DType) -> Tensor {
        match self {
            Self::Tensor(t) => t.clone(),
            Self::Owned(t) => t,
            Self::Scalar(value) => Tensor::const_(value, dtype),
        }
    }

    /// Dtype a bare scalar defaults to when no tensor operand fixes one.
    pub(crate) fn scalar_dtype(&self) -> DType {
        match self {
            Self::Scalar(ConstValue::Float(_)) => DType::Float32,
            Self::Scalar(ConstValue::UInt(_)) => DType::UInt32,
            Self::Scalar(ConstValue::Bool(_)) => DType::Bool,
            Self::Scalar(_) => DType::Int32,
            Self::Tensor(t) => t.dtype(),
            Self::Owned(t) => t.dtype(),
        }
    }
}

impl Tensor {
    /// Resolve an operand against this tensor's dtype.
    #[track_caller]
    pub(crate) fn operand<'a>(&self, value: impl Into<Operand<'a>>) -> Tensor {
        value.into().materialize(self.dtype())
    }
}

impl<'a> From<&'a Tensor> for Operand<'a> {
    fn from(t: &'a Tensor) -> Self {
        Self::Tensor(t)
    }
}

/// So `&x` where `x: &Tensor` is accepted alongside a plain `x`.
impl<'a> From<&&'a Tensor> for Operand<'a> {
    fn from(t: &&'a Tensor) -> Self {
        Self::Tensor(t)
    }
}

impl From<Tensor> for Operand<'_> {
    fn from(t: Tensor) -> Self {
        Self::Owned(t)
    }
}

impl From<ConstValue> for Operand<'_> {
    fn from(v: ConstValue) -> Self {
        Self::Scalar(v)
    }
}

macro_rules! impl_scalar_operand {
    ($($ty:ty),+ $(,)?) => { $(
        impl From<$ty> for Operand<'_> {
            fn from(v: $ty) -> Self {
                Self::Scalar(ConstValue::from(v))
            }
        }
    )+ };
}

impl_scalar_operand!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);

impl From<usize> for Operand<'_> {
    fn from(v: usize) -> Self {
        Self::Scalar(ConstValue::UInt(v as u64))
    }
}

impl From<isize> for Operand<'_> {
    fn from(v: isize) -> Self {
        Self::Scalar(ConstValue::Int(v as i64))
    }
}

/// Dtype the two branches of [`Tensor::select`] agree on: the first tensor
/// operand's, or the scalar's natural dtype when both are scalars.
pub(crate) fn common_dtype(a: &Operand<'_>, b: &Operand<'_>) -> DType {
    a.dtype().or_else(|| b.dtype()).unwrap_or_else(|| match (a.scalar_dtype(), b.scalar_dtype()) {
        // A float branch beside an int branch widens to the float.
        (DType::Scalar(ScalarDType::Float32), _) | (_, DType::Scalar(ScalarDType::Float32)) => DType::Float32,
        (lhs, _) => lhs,
    })
}
