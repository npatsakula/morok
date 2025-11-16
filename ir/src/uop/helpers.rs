//! Helper methods for UOp pattern matching and simplification.
//!
//! These methods support symbolic pattern matching, based on Tinygrad's ops.py.

use std::rc::Rc;

use crate::op::Op;
use crate::types::{BinaryOp, ConstValue};
use crate::uop::UOp;

impl UOp {
    /// Returns the largest known integer that divides this UOp.
    ///
    /// Based on Tinygrad's `const_factor()` (ops.py lines 695-702).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = UOp::const_(DType::Int32, ConstValue::Int(6));
    /// assert_eq!(x.const_factor(), 6);
    ///
    /// let mul = UOp::new(Op::Binary(BinaryOp::Mul, x, y), DType::Int32);
    /// // Returns 6 if x.const_factor() == 6
    /// ```
    pub fn const_factor(&self) -> i64 {
        match &self.op {
            // Constants have their own value as factor
            Op::Const(cv) => match &cv.0 {
                ConstValue::Int(i) => *i,
                ConstValue::UInt(u) => *u as i64,
                _ => 1,
            },

            // Multiplication: product of const factors
            // If multiplying by a constant, the whole expression has that factor
            Op::Binary(BinaryOp::Mul, a, b) => {
                let fa = a.const_factor();
                let fb = b.const_factor();
                fa * fb
            }

            // Division: dividend factor divided by divisor factor
            Op::Binary(BinaryOp::Idiv, a, b) => {
                let fa = a.const_factor();
                let fb = b.const_factor();
                if fb != 0 && fa % fb == 0 { fa / fb } else { 1 }
            }

            // Modulo: GCD of both operands
            Op::Binary(BinaryOp::Mod, a, b) => {
                let fa = a.const_factor();
                let fb = b.const_factor();
                gcd(fa.abs(), fb.abs())
            }

            // Addition: GCD of all addends
            Op::Binary(BinaryOp::Add, a, b) => {
                let fa = a.const_factor();
                let fb = b.const_factor();
                gcd(fa.abs(), fb.abs())
            }

            _ => 1,
        }
    }

    /// Returns `self / v` if `v` divides `self` exactly, otherwise None.
    ///
    /// Based on Tinygrad's `divides()` (ops.py lines 703-711).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // (6*x).divides(3) = Some(2*x)
    /// // (5*x).divides(3) = None
    /// ```
    pub fn divides(self: &Rc<Self>, v: &Rc<Self>) -> Option<Rc<Self>> {
        // If v is a constant, check if const_factor is divisible
        if let Op::Const(cv) = v.op() {
            if let ConstValue::Int(divisor) = cv.0 {
                let factor = self.const_factor();
                if divisor != 0 && factor % divisor == 0 {
                    // If self is constant, return const result
                    if let Op::Const(self_cv) = self.op() {
                        if let ConstValue::Int(dividend) = self_cv.0 {
                            return Some(Self::const_(self.dtype(), ConstValue::Int(dividend / divisor)));
                        }
                    }

                    // If self is multiplication by constant
                    if let Op::Binary(BinaryOp::Mul, a, b) = self.op() {
                        // Check right operand for constant
                        if let Op::Const(const_cv) = b.op() {
                            if let ConstValue::Int(mult) = const_cv.0 {
                                if mult % divisor == 0 {
                                    return Some(Self::new(
                                        Op::Binary(
                                            BinaryOp::Mul,
                                            a.clone(),
                                            Self::const_(b.dtype(), ConstValue::Int(mult / divisor)),
                                        ),
                                        self.dtype(),
                                    ));
                                }
                            }
                        }

                        // Check left operand for constant (multiplication is commutative)
                        if let Op::Const(const_cv) = a.op() {
                            if let ConstValue::Int(mult) = const_cv.0 {
                                if mult % divisor == 0 {
                                    return Some(Self::new(
                                        Op::Binary(
                                            BinaryOp::Mul,
                                            Self::const_(a.dtype(), ConstValue::Int(mult / divisor)),
                                            b.clone(),
                                        ),
                                        self.dtype(),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Separates a constant term from a binary expression.
    ///
    /// Returns (non_const_part, const_value).
    /// Based on Tinygrad's `pop_const()` (ops.py lines 712-713).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // (x + 5).pop_const(ADD) = (x, Some(Int(5)))
    /// // (x + y).pop_const(ADD) = (x + y, None)
    /// // x.pop_const(ADD) = (x, None)
    /// ```
    pub fn pop_const(self: &Rc<Self>, op: BinaryOp) -> (Rc<Self>, Option<ConstValue>) {
        if let Op::Binary(self_op, a, b) = self.op() {
            if *self_op == op {
                // Check if right operand is constant
                if let Op::Const(cv) = b.op() {
                    return (a.clone(), Some(cv.0.clone()));
                }
                // Check if left operand is constant (for commutative ops)
                if op.is_commutative() {
                    if let Op::Const(cv) = a.op() {
                        return (b.clone(), Some(cv.0.clone()));
                    }
                }
            }
        }

        (self.clone(), None)
    }

    /// Splits an associative operation chain into its individual terms.
    ///
    /// Based on Tinygrad's `split_uop()` (ops.py lines 464-467).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // (x + y + z).split_uop(ADD) = [x, y, z]
    /// // (x + y).split_uop(ADD) = [x, y]
    /// // x.split_uop(ADD) = [x]
    /// ```
    pub fn split_uop(self: &Rc<Self>, sep: BinaryOp) -> Vec<Rc<Self>> {
        let mut result = Vec::new();
        let mut stack = vec![self.clone()];

        while let Some(node) = stack.pop() {
            if let Op::Binary(op, a, b) = node.op() {
                if *op == sep {
                    // Add operands to stack in reverse order to maintain left-to-right
                    stack.push(b.clone());
                    stack.push(a.clone());
                    continue;
                }
            }
            result.push(node);
        }

        result
    }
}

/// Computes the greatest common divisor using Euclid's algorithm.
fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Extension trait for BinaryOp to check if it's commutative.
#[allow(dead_code)] // Used in pop_const for commutative check
trait BinaryOpExt {
    fn is_commutative(&self) -> bool;
}

impl BinaryOpExt for BinaryOp {
    fn is_commutative(&self) -> bool {
        matches!(
            self,
            BinaryOp::Add
                | BinaryOp::Mul
                | BinaryOp::And
                | BinaryOp::Or
                | BinaryOp::Xor
                | BinaryOp::Max
                | BinaryOp::Eq
                | BinaryOp::Ne
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_dtype::DType;

    #[test]
    fn test_const_factor_constant() {
        let c = UOp::const_(DType::Int32, ConstValue::Int(6));
        assert_eq!(c.const_factor(), 6);
    }

    #[test]
    fn test_const_factor_multiplication() {
        let x = UOp::define_global(1, DType::Int32);
        let c = UOp::const_(DType::Int32, ConstValue::Int(6));
        let mul = UOp::new(Op::Binary(BinaryOp::Mul, x, c), DType::Int32);
        assert_eq!(mul.const_factor(), 6);
    }

    #[test]
    fn test_const_factor_addition() {
        let c1 = UOp::const_(DType::Int32, ConstValue::Int(6));
        let c2 = UOp::const_(DType::Int32, ConstValue::Int(9));
        let add = UOp::new(Op::Binary(BinaryOp::Add, c1, c2), DType::Int32);
        assert_eq!(add.const_factor(), 3); // GCD(6, 9) = 3
    }

    #[test]
    fn test_divides_constant_exact() {
        let c = UOp::const_(DType::Int32, ConstValue::Int(12));
        let divisor = UOp::const_(DType::Int32, ConstValue::Int(3));
        let result = c.divides(&divisor);

        assert!(result.is_some());
        if let Some(r) = result {
            if let Op::Const(cv) = r.op() {
                assert_eq!(cv.0, ConstValue::Int(4));
            } else {
                panic!("Expected constant result");
            }
        }
    }

    #[test]
    fn test_divides_constant_not_exact() {
        let c = UOp::const_(DType::Int32, ConstValue::Int(10));
        let divisor = UOp::const_(DType::Int32, ConstValue::Int(3));
        let result = c.divides(&divisor);

        assert!(result.is_none());
    }

    #[test]
    fn test_pop_const_with_constant() {
        let x = UOp::define_global(1, DType::Int32);
        let c = UOp::const_(DType::Int32, ConstValue::Int(5));
        let add = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), c), DType::Int32);

        let (rest, const_val) = add.pop_const(BinaryOp::Add);

        assert!(Rc::ptr_eq(&rest, &x));
        assert_eq!(const_val, Some(ConstValue::Int(5)));
    }

    #[test]
    fn test_pop_const_without_constant() {
        let x = UOp::define_global(1, DType::Int32);
        let y = UOp::define_global(2, DType::Int32);
        let add = UOp::new(Op::Binary(BinaryOp::Add, x, y), DType::Int32);

        let (rest, const_val) = add.pop_const(BinaryOp::Add);

        assert!(Rc::ptr_eq(&rest, &add));
        assert_eq!(const_val, None);
    }

    #[test]
    fn test_split_uop_chain() {
        let x = UOp::define_global(1, DType::Int32);
        let y = UOp::define_global(2, DType::Int32);
        let z = UOp::define_global(3, DType::Int32);

        // Build: x + y + z = (x + y) + z
        let xy = UOp::new(Op::Binary(BinaryOp::Add, x.clone(), y.clone()), DType::Int32);
        let xyz = UOp::new(Op::Binary(BinaryOp::Add, xy, z.clone()), DType::Int32);

        let terms = xyz.split_uop(BinaryOp::Add);

        assert_eq!(terms.len(), 3);
        assert!(Rc::ptr_eq(&terms[0], &x));
        assert!(Rc::ptr_eq(&terms[1], &y));
        assert!(Rc::ptr_eq(&terms[2], &z));
    }

    #[test]
    fn test_split_uop_single() {
        let x = UOp::define_global(1, DType::Int32);
        let terms = x.split_uop(BinaryOp::Add);

        assert_eq!(terms.len(), 1);
        assert!(Rc::ptr_eq(&terms[0], &x));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 19), 1);
        assert_eq!(gcd(100, 50), 50);
    }
}
