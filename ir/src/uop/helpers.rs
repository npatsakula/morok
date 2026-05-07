//! Helper methods for UOp pattern matching and simplification.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::op::Op;
use crate::types::{AxisType, BinaryOp, ConstValue};
use crate::uop::UOp;

impl UOp {
    /// Returns the largest known integer that divides this UOp.
    ///
    /// For MUL, only checks immediate CONST children (not recursive).
    pub fn const_factor(&self) -> i64 {
        match &self.op {
            Op::Const(cv) => match &cv.0 {
                ConstValue::Int(i) => *i,
                ConstValue::UInt(u) => *u as i64,
                _ => 1,
            },
            // VCONST: GCD of all elements
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
            // MUL: only immediate CONST child
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

    /// Returns `self / v` if integer `v` divides all terms exactly, otherwise None.
    ///
    /// Recursively handles CONST, VCONST, ADD, and MUL — preserves symbolic
    /// factors in the quotient (e.g. `(T*4).divides(4) == T`).
    pub fn divides(self: &Arc<Self>, v: i64) -> Option<Arc<Self>> {
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
            // VCONST: divide each element if all are divisible
            Op::VConst { values } => {
                let divided: Option<Vec<ConstValue>> = values
                    .iter()
                    .map(|val| match val {
                        ConstValue::Int(i) if i % v == 0 => Some(ConstValue::Int(i / v)),
                        _ => None,
                    })
                    .collect();
                divided.map(|v| UOp::vconst(v, self.dtype().scalar_dtype()))
            }
            Op::Binary(BinaryOp::Add, a, b) => {
                let d0 = a.divides(v)?;
                let d1 = b.divides(v)?;
                d0.try_add(&d1).ok()
            }
            Op::Binary(BinaryOp::Mul, a, b) => {
                if let Some(d0) = a.divides(v) {
                    return d0.try_mul(b).ok();
                }
                if let Some(d1) = b.divides(v) {
                    return a.try_mul(&d1).ok();
                }
                None
            }
            _ => None,
        }
    }

    /// Returns `self / v` if exact division by UOp `v` is possible.
    ///
    /// MUL uses multiset-Counter matching so factor ordering doesn't determine
    /// success — `(2*a*b).divide_exact(2*a)` yields `b` regardless of the Mul
    /// tree's associativity.
    pub fn divide_exact(self: &Arc<Self>, v: &Arc<Self>) -> Option<Arc<Self>> {
        if Arc::ptr_eq(self, v) {
            return Some(self.const_like(1i64));
        }
        if let Op::Const(cv) = v.op()
            && let ConstValue::Int(d) = cv.0
        {
            return self.divides(d);
        }
        if let Op::Binary(BinaryOp::Add, a, b) = self.op() {
            let d0 = a.divide_exact(v)?;
            let d1 = b.divide_exact(v)?;
            return d0.try_add(&d1).ok();
        }
        if matches!(self.op(), Op::Binary(BinaryOp::Mul, _, _)) {
            let (fac, c_self) = self.pop_const(BinaryOp::Mul);
            let (div_fac, c_v) = v.pop_const(BinaryOp::Mul);
            // `pop_const` seeds the const slot with the identity element
            // (`Int(1)` for MUL on integer dtypes), so a non-int return means
            // the expression has a non-integer const factor we cannot reason
            // about — bail.
            let const_self = c_self.try_int()?;
            let const_v = c_v.try_int()?;
            if const_v == 0 || const_self % const_v != 0 {
                return None;
            }
            // Multiset diff: build counts from `fac`, subtract `div_fac` factors.
            let mut counts: HashMap<u64, (Arc<Self>, i32)> = HashMap::new();
            for f in fac.split_uop(BinaryOp::Mul) {
                counts.entry(f.id).and_modify(|(_, c)| *c += 1).or_insert((f, 1));
            }
            for f in div_fac.split_uop(BinaryOp::Mul) {
                match counts.get_mut(&f.id) {
                    Some((_, c)) => *c -= 1,
                    None => return None,
                }
            }
            if counts.values().any(|(_, c)| *c < 0) {
                return None;
            }
            // Multiply remaining factors, seeded with the const quotient.
            let mut result = self.const_like(const_self / const_v);
            for (factor, count) in counts.values() {
                for _ in 0..*count {
                    result = result.try_mul(factor).ok()?;
                }
            }
            return Some(result);
        }
        None
    }

    /// Computes the symbolic GCD of multiple UOps, returning a UOp.
    ///
    /// Finds both numeric GCD of const_factors AND common symbolic MUL factors.
    /// For inputs `6*a*b` and `4*a*c`, returns `2*a` (numeric GCD=2, common factor=a).
    pub fn symbolic_gcd(uops: &[Arc<Self>]) -> Arc<Self> {
        assert!(!uops.is_empty(), "symbolic_gcd requires at least one uop");

        // Step 1: decompose each uop into (term, factor) where term = uop / factor
        let decomp: Vec<(Arc<Self>, i64)> = uops
            .iter()
            .map(|u| {
                let f = u.const_factor();
                let term =
                    if f == 1 || f == 0 { Arc::clone(u) } else { u.divides(f).unwrap_or_else(|| u.const_like(1i64)) };
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
            // Skip CONST(1) factors from divides normalization
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
    /// Returns `(non_const_part, const_value)` — when no const is present the
    /// const slot is the operation's identity element (`0` for ADD, `1` for
    /// MUL, `dtype.min` for MAX). Relies on the const-on-right canonicalization
    /// invariant, so only the right operand is checked.
    ///
    /// # Examples
    ///
    /// ```text
    /// (x + 5).pop_const(ADD) = (x, Int(5))
    /// (x + y).pop_const(ADD) = (x + y, Int(0))
    /// x.pop_const(ADD)       = (x, Int(0))
    /// ```
    pub fn pop_const(self: &Arc<Self>, op: BinaryOp) -> (Arc<Self>, ConstValue) {
        if let Op::Binary(self_op, a, b) = self.op()
            && *self_op == op
            && let Op::Const(cv) = b.op()
        {
            return (a.clone(), cv.0);
        }
        (self.clone(), op.identity_element(self.dtype()))
    }

    /// Splits an associative operation chain into its individual terms.
    ///
    /// # Examples
    ///
    /// ```text
    /// (x + y + z).split_uop(ADD) = [x, y, z]
    /// (x + y).split_uop(ADD) = [x, y]
    /// x.split_uop(ADD) = [x]
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
    /// For membership tests, prefer [`Self::backward_slice_ids`] which returns
    /// a cached `HashSet<u64>` with O(1) lookup.
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
            Op::Invalid => {
                // Bare Invalid is NOT valid
                Self::const_(DType::Bool, ConstValue::Bool(false))
            }
            _ => {
                // Non-Invalid, non-WHERE: always valid
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
            // Irreducible: RANGE, CONST, DEFINE_VAR, SPECIAL
            Op::Range { .. } | Op::Const(_) | Op::DefineVar { .. } | Op::Special { .. } => true,

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
#[path = "../test/unit/uop/helpers_internal.rs"]
mod tests;
