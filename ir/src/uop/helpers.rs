//! Helper methods for UOp pattern matching and simplification.
//!
//! These methods support symbolic pattern matching, based on Tinygrad's ops.py.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::op::Op;
use crate::types::{AxisType, BinaryOp, ConstValue};
use crate::uop::UOp;

impl UOp {
    /// Returns the largest known integer that divides this UOp.
    ///
    /// Based on Tinygrad's `const_factor()` (ops.py:693-700).
    /// For MUL, only checks immediate CONST children (not recursive).
    pub fn const_factor(&self) -> i64 {
        match &self.op {
            Op::Const(cv) => match &cv.0 {
                ConstValue::Int(i) => *i,
                ConstValue::UInt(u) => *u as i64,
                _ => 1,
            },
            // VCONST: GCD of all elements (Tinygrad ops.py:697)
            Op::VConst { values } => values
                .iter()
                .filter_map(|v| match v {
                    ConstValue::Int(i) => Some(*i),
                    ConstValue::UInt(u) => Some(*u as i64),
                    _ => None,
                })
                .map(|v| v.abs())
                .reduce(gcd)
                .unwrap_or(1),
            // MUL: only immediate CONST child, matching Tinygrad exactly
            Op::Binary(BinaryOp::Mul, a, b) => {
                if let Op::Const(cv) = &a.op
                    && let ConstValue::Int(i) = cv.0
                {
                    return i;
                }
                if let Op::Const(cv) = &b.op
                    && let ConstValue::Int(i) = cv.0
                {
                    return i;
                }
                1
            }
            Op::Binary(BinaryOp::Add, a, b) => gcd(a.const_factor().abs(), b.const_factor().abs()),
            _ => 1,
        }
    }

    /// Returns `self / v` if `v` divides `self` exactly, otherwise None.
    ///
    /// Based on Tinygrad's `divides()` (ops.py lines 703-711).
    /// Delegates to [`divides_int`] for constant divisors.
    pub fn divides(self: &Arc<Self>, v: &Arc<Self>) -> Option<Arc<Self>> {
        if let Op::Const(cv) = v.op()
            && let ConstValue::Int(divisor) = cv.0
        {
            return self.divides_int(divisor);
        }
        None
    }

    /// Returns `self / v` if integer `v` divides all terms exactly, otherwise None.
    ///
    /// Based on Tinygrad's `divides(v: int)` (ops.py:701-709).
    /// Recursively handles Const, Add, and Mul operations.
    pub fn divides_int(self: &Arc<Self>, v: i64) -> Option<Arc<Self>> {
        if v == 1 {
            return Some(Arc::clone(self));
        }
        if v == 0 {
            return None;
        }
        match self.op() {
            Op::Const(cv) => {
                let ConstValue::Int(val) = cv.0 else { return None };
                if val % v == 0 { Some(Self::const_(self.dtype(), ConstValue::Int(val / v))) } else { None }
            }
            // VCONST: divide each element if all are divisible (Tinygrad ops.py:704)
            Op::VConst { values } => {
                let divided: Option<Vec<ConstValue>> = values
                    .iter()
                    .map(|val| match val {
                        ConstValue::Int(i) if i % v == 0 => Some(ConstValue::Int(i / v)),
                        _ => None,
                    })
                    .collect();
                divided.map(UOp::vconst)
            }
            Op::Binary(BinaryOp::Add, a, b) => {
                let d0 = a.divides_int(v)?;
                let d1 = b.divides_int(v)?;
                d0.try_add(&d1).ok()
            }
            Op::Binary(BinaryOp::Mul, a, b) => {
                if let Some(d0) = a.divides_int(v) {
                    return d0.try_mul(b).ok();
                }
                if let Some(d1) = b.divides_int(v) {
                    return a.try_mul(&d1).ok();
                }
                None
            }
            _ => None,
        }
    }

    /// Returns `self / v` if exact division by UOp `v` is possible.
    ///
    /// Based on Tinygrad's `divide_exact(v: UOp)` (ops.py:717-726).
    /// Handles identity, constant divisors, Add recursion, and Mul factoring.
    pub fn divide_exact(self: &Arc<Self>, v: &Arc<Self>) -> Option<Arc<Self>> {
        if Arc::ptr_eq(self, v) {
            return Some(self.const_like(1i64));
        }
        if let Op::Const(cv) = v.op()
            && let ConstValue::Int(d) = cv.0
        {
            return self.divides_int(d);
        }
        if let Op::Binary(BinaryOp::Add, a, b) = self.op() {
            let d0 = a.divide_exact(v)?;
            let d1 = b.divide_exact(v)?;
            return d0.try_add(&d1).ok();
        }
        if let Op::Binary(BinaryOp::Mul, a, b) = self.op() {
            if let Some(d) = a.divide_exact(v) {
                return d.try_mul(b).ok();
            }
            if let Some(d) = b.divide_exact(v) {
                return a.try_mul(&d).ok();
            }
        }
        None
    }

    /// Computes the symbolic GCD of multiple UOps, returning a UOp.
    ///
    /// Based on Tinygrad's `UOp.gcd()` (ops.py:713-716).
    /// Finds both numeric GCD of const_factors AND common symbolic MUL factors.
    ///
    /// For inputs `6*a*b` and `4*a*c`, returns `2*a` (numeric GCD=2, common factor=a).
    pub fn symbolic_gcd(uops: &[Arc<Self>]) -> Arc<Self> {
        assert!(!uops.is_empty(), "symbolic_gcd requires at least one uop");

        // Step 1: decompose each uop into (term, factor) where term = uop / factor
        let decomp: Vec<(Arc<Self>, i64)> = uops
            .iter()
            .map(|u| {
                let f = u.const_factor();
                let term = if f == 1 || f == 0 {
                    Arc::clone(u)
                } else {
                    u.divides_int(f).unwrap_or_else(|| u.const_like(1i64))
                };
                (term, f)
            })
            .collect();

        // Step 2: split each term into MUL factors, build Counter (ptr → count)
        let counters: Vec<HashMap<*const Self, (Arc<Self>, usize)>> = decomp
            .iter()
            .map(|(term, _)| {
                let mut counter: HashMap<*const Self, (Arc<Self>, usize)> = HashMap::new();
                for factor in term.split_uop(BinaryOp::Mul) {
                    let ptr = Arc::as_ptr(&factor);
                    counter.entry(ptr).and_modify(|(_, c)| *c += 1).or_insert((factor, 1));
                }
                counter
            })
            .collect();

        // Step 3: intersect counters (keep factors present in ALL terms with min count)
        let mut common = counters[0].clone();
        for other in &counters[1..] {
            common.retain(|ptr, (_, count)| {
                if let Some((_, other_count)) = other.get(ptr) {
                    *count = (*count).min(*other_count);
                    true
                } else {
                    false
                }
            });
        }

        // Step 4: numeric GCD of all const_factors
        let numeric = decomp.iter().map(|(_, f)| f.abs()).reduce(gcd).unwrap_or(1);

        // Step 5: multiply common symbolic factors with numeric GCD
        let mut result = uops[0].const_like(numeric);
        for (factor, count) in common.values() {
            // Skip CONST(1) factors from divides_int normalization
            if let Op::Const(cv) = factor.op()
                && matches!(cv.0, ConstValue::Int(1))
            {
                continue;
            }
            for _ in 0..*count {
                result = result.try_mul(factor).expect("symbolic_gcd: mul failed");
            }
        }

        result
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

    /// Cached backward slice: set of all node IDs reachable from this UOp.
    ///
    /// O(1) membership test via `contains()`. Computed once and cached per-node.
    /// Prefer this over `backward_slice()` when you only need to check if a
    /// node is in the dependency set.
    pub fn backward_slice_ids(self: &Arc<Self>) -> &HashSet<u64> {
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::BackwardSliceProperty;
        BackwardSliceProperty::get(self)
    }

    /// Returns all nodes that this UOp depends on (backward slice / dependency set).
    ///
    /// For membership tests, prefer [`backward_slice_ids()`] which returns a
    /// cached `HashSet<u64>` with O(1) lookup.
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
        if let Op::Range { end, .. } = self.op() {
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
        if let Op::Range { end, axis_id, .. } = self.op() {
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
    /// Matches both scalar `Op::Invalid` and vectorized `VECTORIZE(Invalid, ..., Invalid)`
    /// where ALL elements are Invalid. The vectorized form appears after expansion
    /// broadcasts scalar Invalid across lanes.
    ///
    /// Uses `all()` semantics (entire vector must be Invalid). This differs from
    /// `has_invalid()` in symbolic patterns which uses `any()` for guard semantics.
    pub fn is_invalid_marker(uop: &Arc<Self>) -> bool {
        match uop.op() {
            Op::Invalid => true,
            Op::Vectorize { elements } => {
                !elements.is_empty() && elements.iter().all(|e| matches!(e.op(), Op::Invalid))
            }
            _ => false,
        }
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

    /// Check if this UOp is a monotonically increasing function of its inputs.
    ///
    /// Returns true for:
    /// - Irreducible ops (RANGE, CONST, DEFINE_VAR)
    /// - ADD of increasing ops
    /// - MUL/IDIV by non-negative constants
    ///
    /// Based on Tinygrad's `is_increasing()` (ops.py:689-694).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Constants are increasing
    /// let c = UOp::const_(DType::Int32, ConstValue::Int(5));
    /// assert!(c.is_increasing());
    ///
    /// // Range variables are increasing
    /// let range = UOp::range_axis(UOp::index_const(16), 0, AxisType::Loop);
    /// assert!(range.is_increasing());
    ///
    /// // x + y is increasing if both x and y are increasing
    /// let sum = range.try_add(&c).unwrap();
    /// assert!(sum.is_increasing());
    ///
    /// // x * 2 is increasing if x is increasing
    /// let two = UOp::const_(DType::Index, ConstValue::Int(2));
    /// let scaled = range.try_mul(&two).unwrap();
    /// assert!(scaled.is_increasing());
    /// ```
    pub fn is_increasing(self: &Arc<Self>) -> bool {
        match self.op() {
            // Irreducible: RANGE, CONST, DEFINE_VAR
            Op::Range { .. } | Op::Const(_) | Op::DefineVar { .. } => true,

            // ADD: both operands must be increasing
            Op::Binary(BinaryOp::Add, a, b) => a.is_increasing() && b.is_increasing(),

            // MUL/IDIV by non-negative constant
            Op::Binary(BinaryOp::Mul | BinaryOp::Idiv, a, b) => {
                if let Op::Const(cv) = b.op() {
                    matches!(cv.0, ConstValue::Int(n) if n >= 0) && a.is_increasing()
                } else {
                    false
                }
            }

            _ => false,
        }
    }
}

/// Computes the greatest common divisor using Euclid's algorithm.
/// Always returns a non-negative value.
pub fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
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
        assert_eq!(gcd(-12, 8), 4);
        assert_eq!(gcd(12, -8), 4);
        assert_eq!(gcd(-12, -8), 4);
    }

    #[test]
    fn test_symbolic_gcd_numeric_only() {
        // GCD of 6*x and 4*y → numeric GCD is 2
        let x = UOp::var("x", DType::Index, 0, 10);
        let y = UOp::var("y", DType::Index, 0, 10);
        let six = UOp::const_(DType::Index, ConstValue::Int(6));
        let four = UOp::const_(DType::Index, ConstValue::Int(4));
        let a = x.try_mul(&six).unwrap(); // 6*x
        let b = y.try_mul(&four).unwrap(); // 4*y
        let g = UOp::symbolic_gcd(&[a, b]);
        if let Op::Const(cv) = g.op() {
            assert_eq!(cv.0, ConstValue::Int(2));
        } else {
            panic!("Expected constant GCD, got: {}", g.tree());
        }
    }

    #[test]
    fn test_symbolic_gcd_with_common_factor() {
        // GCD of 6*x and 4*x → 2*x (common symbolic factor x, numeric GCD 2)
        let x = UOp::var("x", DType::Index, 0, 10);
        let six = UOp::const_(DType::Index, ConstValue::Int(6));
        let four = UOp::const_(DType::Index, ConstValue::Int(4));
        let a = x.try_mul(&six).unwrap(); // 6*x (= x*6 internally)
        let b = x.try_mul(&four).unwrap(); // 4*x (= x*4 internally)
        let g = UOp::symbolic_gcd(&[a, b]);
        // Should be 2*x — a MUL node
        assert!(matches!(g.op(), Op::Binary(BinaryOp::Mul, _, _)), "Expected MUL, got: {}", g.tree());
    }

    #[test]
    fn test_const_factor_mul_only_immediate() {
        // (x * 6) * (y * 4) — const_factor should be 1 (no immediate CONST child)
        let x = UOp::var("x", DType::Index, 0, 10);
        let y = UOp::var("y", DType::Index, 0, 10);
        let six = UOp::const_(DType::Index, ConstValue::Int(6));
        let four = UOp::const_(DType::Index, ConstValue::Int(4));
        let a = x.try_mul(&six).unwrap(); // x*6
        let b = y.try_mul(&four).unwrap(); // y*4
        let ab = a.try_mul(&b).unwrap(); // (x*6) * (y*4)
        // Tinygrad: neither immediate child is CONST → returns 1
        assert_eq!(ab.const_factor(), 1);
    }

    #[test]
    fn test_const_factor_vconst() {
        let vc = UOp::vconst(vec![ConstValue::Int(6), ConstValue::Int(12), ConstValue::Int(18), ConstValue::Int(24)]);
        assert_eq!(vc.const_factor(), 6); // GCD(6, 12, 18, 24) = 6
    }

    #[test]
    fn test_const_factor_vconst_no_common() {
        let vc = UOp::vconst(vec![ConstValue::Int(7), ConstValue::Int(11)]);
        assert_eq!(vc.const_factor(), 1); // GCD(7, 11) = 1
    }

    #[test]
    fn test_divides_int_vconst() {
        let vc = UOp::vconst(vec![ConstValue::Int(6), ConstValue::Int(12)]);
        let result = vc.divides_int(3);
        assert!(result.is_some());
        if let Some(r) = result {
            if let Op::VConst { values } = r.op() {
                assert_eq!(values, &[ConstValue::Int(2), ConstValue::Int(4)]);
            } else {
                panic!("Expected VConst result");
            }
        }
    }

    #[test]
    fn test_divides_int_vconst_not_divisible() {
        let vc = UOp::vconst(vec![
            ConstValue::Int(6),
            ConstValue::Int(7), // 7 not divisible by 3
        ]);
        assert!(vc.divides_int(3).is_none());
    }

    #[test]
    fn test_is_increasing_const() {
        let c = UOp::const_(DType::Int32, ConstValue::Int(5));
        assert!(c.is_increasing());

        let neg = UOp::const_(DType::Int32, ConstValue::Int(-5));
        assert!(neg.is_increasing()); // Constants are always "increasing" (irreducible)
    }

    #[test]
    fn test_is_increasing_add() {
        let a = UOp::const_(DType::Int32, ConstValue::Int(5));
        let b = UOp::const_(DType::Int32, ConstValue::Int(3));
        let sum = a.try_add(&b).unwrap();
        assert!(sum.is_increasing());
    }

    #[test]
    fn test_is_increasing_mul_positive_const() {
        let x = UOp::var("x", DType::Int32, 0, 100);
        let two = UOp::const_(DType::Int32, ConstValue::Int(2));
        let scaled = x.try_mul(&two).unwrap();
        assert!(scaled.is_increasing());
    }

    #[test]
    fn test_is_increasing_mul_negative_const() {
        let x = UOp::var("x", DType::Int32, 0, 100);
        let neg = UOp::const_(DType::Int32, ConstValue::Int(-2));
        let scaled = x.try_mul(&neg).unwrap();
        assert!(!scaled.is_increasing()); // Multiplying by negative is not increasing
    }

    #[test]
    fn test_is_increasing_idiv_positive_const() {
        let x = UOp::var("x", DType::Int32, 0, 100);
        let two = UOp::const_(DType::Int32, ConstValue::Int(2));
        let divided = x.idiv(&two);
        assert!(divided.is_increasing());
    }

    #[test]
    fn test_is_increasing_complex() {
        // (x + 5) * 2 should be increasing
        let x = UOp::var("x", DType::Int32, 0, 100);
        let five = UOp::const_(DType::Int32, ConstValue::Int(5));
        let two = UOp::const_(DType::Int32, ConstValue::Int(2));
        let sum = x.try_add(&five).unwrap();
        let scaled = sum.try_mul(&two).unwrap();
        assert!(scaled.is_increasing());
    }
}
