//! Helper methods for UOp pattern matching and simplification.
//!
//! These methods support symbolic pattern matching, based on Tinygrad's ops.py.

use std::collections::HashSet;
use std::sync::Arc;

use crate::op::Op;
use crate::types::{AxisType, BinaryOp, ConstValue};
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
    pub fn divides(self: &Arc<Self>, v: &Arc<Self>) -> Option<Arc<Self>> {
        // If v is a constant, check if const_factor is divisible
        if let Op::Const(cv) = v.op()
            && let ConstValue::Int(divisor) = cv.0
        {
            let factor = self.const_factor();
            if divisor != 0 && factor % divisor == 0 {
                // If self is constant, return const result
                if let Op::Const(self_cv) = self.op()
                    && let ConstValue::Int(dividend) = self_cv.0
                {
                    return Some(Self::const_(self.dtype(), ConstValue::Int(dividend / divisor)));
                }

                // If self is multiplication by constant
                if let Op::Binary(BinaryOp::Mul, a, b) = self.op() {
                    // Check right operand for constant
                    if let Op::Const(const_cv) = b.op()
                        && let ConstValue::Int(mult) = const_cv.0
                        && mult % divisor == 0
                    {
                        let new_const = Self::const_(b.dtype(), ConstValue::Int(mult / divisor));
                        return Some(a.try_mul(&new_const).expect("divides: mul should succeed with same dtype"));
                    }

                    // Check left operand for constant (multiplication is commutative)
                    if let Op::Const(const_cv) = a.op()
                        && let ConstValue::Int(mult) = const_cv.0
                        && mult % divisor == 0
                    {
                        let new_const = Self::const_(a.dtype(), ConstValue::Int(mult / divisor));
                        return Some(new_const.try_mul(b).expect("divides: mul should succeed with same dtype"));
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
    pub fn pop_const(self: &Arc<Self>, op: BinaryOp) -> (Arc<Self>, Option<ConstValue>) {
        if let Op::Binary(self_op, a, b) = self.op()
            && *self_op == op
        {
            // Check if right operand is constant
            if let Op::Const(cv) = b.op() {
                return (a.clone(), Some(cv.0));
            }
            // Check if left operand is constant (for commutative ops)
            if op.is_commutative()
                && let Op::Const(cv) = a.op()
            {
                return (b.clone(), Some(cv.0));
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
    pub fn split_uop(self: &Arc<Self>, sep: BinaryOp) -> Vec<Arc<Self>> {
        let mut result = Vec::new();
        let mut stack = vec![self.clone()];

        while let Some(node) = stack.pop() {
            if let Op::Binary(op, a, b) = node.op()
                && *op == sep
            {
                // Add operands to stack in reverse order to maintain left-to-right
                stack.push(b.clone());
                stack.push(a.clone());
                continue;
            }
            result.push(node);
        }

        result
    }

    /// Returns all nodes that this UOp depends on (backward slice / dependency set).
    ///
    /// This is used by the optimizer to check if a range appears in an expression's dependencies.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = UOp::var("x", DType::Int32, 0, 10);
    /// let y = UOp::var("y", DType::Int32, 0, 20);
    /// let expr = x.try_add_op(&y).unwrap();
    ///
    /// let deps = expr.backward_slice();
    /// // Check if x is in dependencies using pointer equality
    /// assert!(deps.iter().any(|d| Arc::ptr_eq(d, &x)));
    /// ```
    pub fn backward_slice(self: &Arc<Self>) -> Vec<Arc<Self>> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut stack = vec![self.clone()];

        while let Some(node) = stack.pop() {
            let ptr = Arc::as_ptr(&node);

            if visited.contains(&ptr) {
                continue;
            }

            visited.insert(ptr);
            result.push(node.clone());

            // Add all children to stack
            node.op.map_child(|child| {
                stack.push(child.clone());
            });
        }

        result
    }

    /// Check if this UOp's size is divisible by the given amount.
    ///
    /// Returns `Some(quotient)` if divisible, `None` otherwise.
    /// This is a convenience method for the optimizer to validate transformations.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let range = UOp::range(SInt::Const(16), 0, AxisType::Loop);
    /// assert_eq!(range.divisible_by(4), Some(4)); // 16 / 4 = 4
    /// assert_eq!(range.divisible_by(5), None);    // 16 not divisible by 5
    /// ```
    pub fn divisible_by(self: &Arc<Self>, amount: usize) -> Option<usize> {
        // For RANGE operations, check the end (size) field
        if let Op::Range { end, axis_id: _, axis_type: _ } = self.op() {
            // Check if end is a constant
            if let Op::Const(cv) = end.op()
                && let ConstValue::Int(sz) = cv.0
                && sz > 0
                && (sz as usize).is_multiple_of(amount)
            {
                return Some((sz as usize) / amount);
            }

            // Check using const_factor
            let factor = end.const_factor();
            if factor > 0 && (factor as usize).is_multiple_of(amount) {
                return Some((factor as usize) / amount);
            }
        }

        // For constants, check the value directly
        if let Op::Const(cv) = self.op()
            && let ConstValue::Int(val) = cv.0
            && val > 0
            && (val as usize).is_multiple_of(amount)
        {
            return Some((val as usize) / amount);
        }

        None
    }

    /// Create a new RANGE UOp with a different axis type.
    ///
    /// This is a convenience method for the optimizer to convert ranges between
    /// axis types (e.g., LOOP → GLOBAL for parallelization).
    ///
    /// # Panics
    ///
    /// Panics if called on a non-RANGE operation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let loop_range = UOp::range_axis(UOp::index_const(16), 0, AxisType::Loop);
    /// let global_range = loop_range.with_axis_type(AxisType::Global);
    /// // global_range has same size and axis_id, but different axis type
    /// ```
    pub fn with_axis_type(self: &Arc<Self>, new_type: AxisType) -> Arc<Self> {
        if let Op::Range { end, axis_id, axis_type: _ } = self.op() {
            Self::range_axis(end.clone(), *axis_id, new_type)
        } else {
            panic!("with_axis_type() called on non-RANGE operation: {:?}", self.op);
        }
    }

    /// Extract the actual index from a range, stripping validity checks.
    ///
    /// If the range is a WHERE(valid, idx, invalid_marker), returns idx.
    /// Otherwise, returns the range itself.
    ///
    /// This is used for range merging when comparing indexing patterns across
    /// multiple consumers.
    ///
    /// Based on Tinygrad's `get_idx()` (ops.py:438-439).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Range with padding: WHERE(i < 5, i, SENTINEL)
    /// let padded_range = UOp::where_op(valid, idx.clone(), invalid_marker)?;
    /// assert!(Arc::ptr_eq(&padded_range.get_idx(), &idx));
    ///
    /// // Plain range: returns itself
    /// let plain_range = UOp::range_axis(...);
    /// assert!(Arc::ptr_eq(&plain_range.get_idx(), &plain_range));
    /// ```
    pub fn get_idx(self: &Arc<Self>) -> Arc<Self> {
        use crate::types::TernaryOp;

        match self.op() {
            Op::Ternary(TernaryOp::Where, _, true_val, false_val) if Self::is_invalid_marker(false_val) => {
                // WHERE(valid, idx, INVALID) → return idx
                true_val.clone()
            }
            _ => self.clone(),
        }
    }

    /// Extract the validity mask from a range.
    ///
    /// If the range is a WHERE(valid, idx, invalid_marker), returns valid.
    /// Otherwise, returns constant true (always valid).
    ///
    /// This is used for range merging to combine validity conditions when
    /// multiple consumers share compatible indexing patterns.
    ///
    /// Based on Tinygrad's `get_valid()` (ops.py:440-441).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Range with padding: WHERE(i < 5, i, SENTINEL)
    /// let padded_range = UOp::where_op(valid.clone(), idx, invalid_marker)?;
    /// assert!(Arc::ptr_eq(&padded_range.get_valid(), &valid));
    ///
    /// // Plain range: returns constant true
    /// let plain_range = UOp::range_axis(...);
    /// if let Op::Const(cv) = plain_range.get_valid().op() {
    ///     assert_eq!(cv.0, ConstValue::Bool(true));
    /// }
    /// ```
    pub fn get_valid(self: &Arc<Self>) -> Arc<Self> {
        use crate::types::TernaryOp;
        use morok_dtype::DType;

        match self.op() {
            Op::Ternary(TernaryOp::Where, cond, _, false_val) if Self::is_invalid_marker(false_val) => {
                // WHERE(valid, idx, INVALID) → return valid
                cond.clone()
            }
            _ => {
                // Always valid - return constant true
                Self::const_(DType::Bool, ConstValue::Bool(true))
            }
        }
    }

    /// Check if a UOp represents an invalid index marker.
    ///
    /// Currently uses a sentinel value convention (i64::MIN for Index type).
    /// This will be replaced with proper ConstValue::Invalid in Phase 5.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let invalid = UOp::invalid_marker();
    /// assert!(UOp::is_invalid_marker(&invalid));
    ///
    /// let valid_idx = UOp::index_const(5);
    /// assert!(!UOp::is_invalid_marker(&valid_idx));
    /// ```
    fn is_invalid_marker(uop: &Arc<Self>) -> bool {
        matches!(uop.op(), Op::Invalid)
    }

    /// Create an invalid index marker.
    ///
    /// Invalid markers are used with WHERE operations to indicate out-of-bounds
    /// or padded regions. The value is undefined and should never be used directly -
    /// it exists only to be masked away by validity checks.
    ///
    /// # Returns
    ///
    /// A UOp representing an invalid index value.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Padding: WHERE(i < actual_size, i, invalid)
    /// let invalid = UOp::invalid_marker();
    /// let padded = UOp::where_op(valid, actual_idx, invalid)?;
    /// ```
    pub fn invalid_marker() -> Arc<Self> {
        use morok_dtype::DType;

        // Invalid marker for out-of-bounds indices (used in padding/masking)
        Self::new(Op::Invalid, DType::Index)
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
        let x = UOp::var("x", DType::Int32, 0, 100);
        let c = UOp::const_(DType::Int32, ConstValue::Int(6));
        let mul = x.try_mul(&c).unwrap();
        assert_eq!(mul.const_factor(), 6);
    }

    #[test]
    fn test_const_factor_addition() {
        let c1 = UOp::const_(DType::Int32, ConstValue::Int(6));
        let c2 = UOp::const_(DType::Int32, ConstValue::Int(9));
        let add = c1.try_add(&c2).unwrap();
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
        let x = UOp::var("x", DType::Int32, 0, 100);
        let c = UOp::const_(DType::Int32, ConstValue::Int(5));
        let add = x.try_add(&c).unwrap();

        let (rest, const_val) = add.pop_const(BinaryOp::Add);

        assert!(Arc::ptr_eq(&rest, &x));
        assert_eq!(const_val, Some(ConstValue::Int(5)));
    }

    #[test]
    fn test_pop_const_without_constant() {
        let x = UOp::var("x", DType::Int32, 0, 100);
        let y = UOp::var("y", DType::Int32, 0, 100);
        let add = x.try_add(&y).unwrap();

        let (rest, const_val) = add.pop_const(BinaryOp::Add);

        assert!(Arc::ptr_eq(&rest, &add));
        assert_eq!(const_val, None);
    }

    #[test]
    fn test_split_uop_chain() {
        let x = UOp::var("x", DType::Int32, 0, 100);
        let y = UOp::var("y", DType::Int32, 0, 100);
        let z = UOp::var("z", DType::Int32, 0, 100);

        // Build: x + y + z = (x + y) + z
        let xy = x.try_add(&y).unwrap();
        let xyz = xy.try_add(&z).unwrap();

        let terms = xyz.split_uop(BinaryOp::Add);

        assert_eq!(terms.len(), 3);
        assert!(Arc::ptr_eq(&terms[0], &x));
        assert!(Arc::ptr_eq(&terms[1], &y));
        assert!(Arc::ptr_eq(&terms[2], &z));
    }

    #[test]
    fn test_split_uop_single() {
        let x = UOp::var("x", DType::Int32, 0, 100);
        let terms = x.split_uop(BinaryOp::Add);

        assert_eq!(terms.len(), 1);
        assert!(Arc::ptr_eq(&terms[0], &x));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 19), 1);
        assert_eq!(gcd(100, 50), 50);
    }
}
