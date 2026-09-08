use super::*;

/// Operator sugar for the fallible binary ops.
///
/// Every operator returns [`Result<Tensor>`], so shape or dtype mismatches stay
/// recoverable: `(&a + &b)?`. The right-hand side is any [`Operand`], so a
/// tensor (owned or borrowed) and a bare scalar both work; the left-hand side
/// may be an owned or borrowed [`Tensor`], or a scalar via [`impl_scalar_lhs`].
macro_rules! impl_binary_op {
    ($($trait:ident :: $method:ident => $try_method:ident),* $(,)?) => { $(
        impl<'r, R: Into<Operand<'r>>> std::ops::$trait<R> for &Tensor {
            type Output = Result<Tensor>;

            #[track_caller]
            fn $method(self, other: R) -> Self::Output {
                self.$try_method(other)
            }
        }

        impl<'r, R: Into<Operand<'r>>> std::ops::$trait<R> for Tensor {
            type Output = Result<Tensor>;

            #[track_caller]
            fn $method(self, other: R) -> Self::Output {
                self.$try_method(other)
            }
        }
    )* };
}

impl_binary_op! {
    Add::add => try_add,
    Sub::sub => try_sub,
    Mul::mul => try_mul,
    Div::div => try_div,
    Rem::rem => try_mod,
    BitAnd::bitand => try_bitand,
    BitOr::bitor => try_bitor,
    BitXor::bitxor => try_bitxor,
    Shl::shl => try_shl,
    Shr::shr => try_shr,
}

/// `2.0 * &t` — the scalar is materialized in the tensor's dtype, so the
/// operation runs exactly as `t.dtype()` demands.
macro_rules! impl_scalar_lhs {
    ($($ty:ty),* $(,)?) => { $(
        impl_scalar_lhs!(@ops $ty: Add::add => try_add, Sub::sub => try_sub, Mul::mul => try_mul,
                                  Div::div => try_div, Rem::rem => try_mod);
    )* };
    (@ops $ty:ty: $($trait:ident :: $method:ident => $try_method:ident),* $(,)?) => { $(
        impl std::ops::$trait<&Tensor> for $ty {
            type Output = Result<Tensor>;

            #[track_caller]
            fn $method(self, other: &Tensor) -> Self::Output {
                Tensor::const_(self, other.dtype()).$try_method(other)
            }
        }

        impl std::ops::$trait<Tensor> for $ty {
            type Output = Result<Tensor>;

            #[track_caller]
            fn $method(self, other: Tensor) -> Self::Output {
                Tensor::const_(self, other.dtype()).$try_method(&other)
            }
        }
    )* };
}

impl_scalar_lhs!(f32, f64, i32, i64);

// Negation is infallible, so it stays a plain `Tensor`.
impl std::ops::Neg for &Tensor {
    type Output = Tensor;

    #[track_caller]
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;

    #[track_caller]
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}
