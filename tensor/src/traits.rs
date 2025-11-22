use super::*;

/// Macro to implement binary operation traits for Tensor.
///
/// Generates all 4 ownership combinations:
/// - &Tensor op &Tensor (primary implementation, calls try_* method)
/// - Tensor op Tensor (forwards to &self op &other)
/// - &Tensor op Tensor (forwards to self op &other)
/// - Tensor op &Tensor (forwards to &self op other)
macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $try_method:ident, $error_msg:expr) => {
        impl std::ops::$trait for &Tensor {
            type Output = Tensor;

            #[track_caller]
            fn $method(self, other: &Tensor) -> Tensor {
                self.$try_method(other).expect($error_msg)
            }
        }

        impl std::ops::$trait for Tensor {
            type Output = Tensor;

            #[track_caller]
            fn $method(self, other: Tensor) -> Tensor {
                (&self).$method(&other)
            }
        }

        impl std::ops::$trait<Tensor> for &Tensor {
            type Output = Tensor;

            #[track_caller]
            fn $method(self, other: Tensor) -> Tensor {
                self.$method(&other)
            }
        }

        impl std::ops::$trait<&Tensor> for Tensor {
            type Output = Tensor;

            #[track_caller]
            fn $method(self, other: &Tensor) -> Tensor {
                (&self).$method(other)
            }
        }
    };
}

// Binary arithmetic operations
impl_binary_op!(Add, add, try_add, "Addition failed");
impl_binary_op!(Sub, sub, try_sub, "Subtraction failed");
impl_binary_op!(Mul, mul, try_mul, "Multiplication failed");
impl_binary_op!(Div, div, try_div, "Division failed");

// Unary operations
impl std::ops::Neg for &Tensor {
    type Output = Tensor;

    #[track_caller]
    fn neg(self) -> Tensor {
        self.try_neg().expect("Negation failed")
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;

    #[track_caller]
    fn neg(self) -> Tensor {
        (&self).neg()
    }
}
