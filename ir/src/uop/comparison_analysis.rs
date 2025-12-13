//! Unified comparison analysis for range-based optimizations.
//!
//! This module provides a single source of truth for analyzing comparison
//! operations based on operand ranges, eliminating duplication between
//! VminVmax computation and DCE analysis.

use crate::UOp;
use crate::types::{BinaryOp, ConstValue};
use std::cmp::Ordering;
use std::sync::Arc;

/// Unified comparison analyzer that handles both constant and range-based analysis.
pub struct ComparisonAnalyzer;

impl ComparisonAnalyzer {
    /// Analyze a comparison operation and determine if it has a known result.
    ///
    /// Returns Some(true/false) if the comparison result is deterministic,
    /// or None if it could be either true or false.
    pub fn analyze(op: BinaryOp, x: &Arc<UOp>, y: &Arc<UOp>) -> Option<bool> {
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::VminVmaxProperty;

        // Fast path: self-comparison for non-float types
        if Arc::ptr_eq(x, y) && !x.dtype().is_float() {
            return match op {
                BinaryOp::Lt => Some(false), // x < x is always false
                BinaryOp::Eq => Some(true),  // x == x is always true
                BinaryOp::Ne => Some(false), // x != x is always false
                _ => None,
            };
        }

        // Check if comparison can be eliminated (handles float NaN cases)
        if !Self::can_eliminate_comparison(x, y) {
            return None;
        }

        // Get ranges for both operands
        let (x_min, x_max) = VminVmaxProperty::get(x);
        let (y_min, y_max) = VminVmaxProperty::get(y);

        // Analyze based on ranges
        Self::analyze_with_ranges(op, *x_min, *x_max, *y_min, *y_max)
    }

    /// Analyze comparison with explicit ranges.
    ///
    /// This is the core logic that determines comparison results based on ranges.
    pub fn analyze_with_ranges(
        op: BinaryOp,
        x_min: ConstValue,
        x_max: ConstValue,
        y_min: ConstValue,
        y_max: ConstValue,
    ) -> Option<bool> {
        use BinaryOp::*;

        match op {
            Lt => Self::analyze_lt(x_min, x_max, y_min, y_max),
            Le => Self::analyze_le(x_min, x_max, y_min, y_max),
            Eq => Self::analyze_eq(x_min, x_max, y_min, y_max),
            Ne => Self::analyze_ne(x_min, x_max, y_min, y_max),
            Gt => Self::analyze_gt(x_min, x_max, y_min, y_max),
            Ge => Self::analyze_ge(x_min, x_max, y_min, y_max),
            _ => None,
        }
    }

    /// Analyze comparison for additional operations not in BinaryOp.
    /// This is for internal use to support logical equivalents.
    pub fn analyze_extended(
        op_name: &str,
        x_min: ConstValue,
        x_max: ConstValue,
        y_min: ConstValue,
        y_max: ConstValue,
    ) -> Option<bool> {
        match op_name {
            "le" => Self::analyze_le(x_min, x_max, y_min, y_max),
            "gt" => Self::analyze_gt(x_min, x_max, y_min, y_max),
            "ge" => Self::analyze_ge(x_min, x_max, y_min, y_max),
            _ => None,
        }
    }

    /// Get the min/max range for a comparison result.
    ///
    /// Returns (min, max) where both are Bool ConstValues.
    pub fn get_comparison_range(
        op: BinaryOp,
        x_min: ConstValue,
        x_max: ConstValue,
        y_min: ConstValue,
        y_max: ConstValue,
    ) -> (ConstValue, ConstValue) {
        match Self::analyze_with_ranges(op, x_min, x_max, y_min, y_max) {
            Some(true) => (ConstValue::Bool(true), ConstValue::Bool(true)),
            Some(false) => (ConstValue::Bool(false), ConstValue::Bool(false)),
            None => (ConstValue::Bool(false), ConstValue::Bool(true)),
        }
    }

    /// Check if a comparison can be safely eliminated based on types.
    fn can_eliminate_comparison(x: &Arc<UOp>, y: &Arc<UOp>) -> bool {
        let dtype = x.dtype();

        // For non-float types, always safe to eliminate
        if !dtype.is_float() {
            return true;
        }

        // For floats, check if NaN is possible
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::VminVmaxProperty;

        let check_nan = |uop: &Arc<UOp>| {
            let (min, max) = VminVmaxProperty::get(uop);
            matches!(min, ConstValue::Float(f) if f.is_nan()) || matches!(max, ConstValue::Float(f) if f.is_nan())
        };

        !check_nan(x) && !check_nan(y)
    }

    /// Analyze x < y
    fn analyze_lt(x_min: ConstValue, x_max: ConstValue, y_min: ConstValue, y_max: ConstValue) -> Option<bool> {
        use Ordering::*;
        match (Self::compare_values(&x_max, &y_min), Self::compare_values(&x_min, &y_max)) {
            (Less, _) => Some(true),                // max(x) < min(y) => always true
            (_, ord) if ord != Less => Some(false), // min(x) >= max(y) => always false
            _ => None,
        }
    }

    /// Analyze x <= y
    fn analyze_le(x_min: ConstValue, x_max: ConstValue, y_min: ConstValue, y_max: ConstValue) -> Option<bool> {
        use Ordering::*;
        match (Self::compare_values(&x_max, &y_min), Self::compare_values(&x_min, &y_max)) {
            (ord, _) if ord != Greater => Some(true), // max(x) <= min(y) => always true
            (_, Greater) => Some(false),              // min(x) > max(y) => always false
            _ => None,
        }
    }

    /// Analyze x > y
    fn analyze_gt(x_min: ConstValue, x_max: ConstValue, y_min: ConstValue, y_max: ConstValue) -> Option<bool> {
        // x > y is equivalent to y < x
        Self::analyze_lt(y_min, y_max, x_min, x_max)
    }

    /// Analyze x >= y
    fn analyze_ge(x_min: ConstValue, x_max: ConstValue, y_min: ConstValue, y_max: ConstValue) -> Option<bool> {
        // x >= y is equivalent to y <= x
        Self::analyze_le(y_min, y_max, x_min, x_max)
    }

    /// Analyze x == y
    fn analyze_eq(x_min: ConstValue, x_max: ConstValue, y_min: ConstValue, y_max: ConstValue) -> Option<bool> {
        // Check if ranges overlap
        let no_overlap = Self::ranges_disjoint(x_min, x_max, y_min, y_max);

        if no_overlap {
            Some(false) // No overlap => always false
        } else if x_min == x_max && y_min == y_max && x_min == y_min {
            Some(true) // Both are same constant => always true
        } else {
            None // Could be either
        }
    }

    /// Analyze x != y
    fn analyze_ne(x_min: ConstValue, x_max: ConstValue, y_min: ConstValue, y_max: ConstValue) -> Option<bool> {
        // Opposite of ==
        Self::analyze_eq(x_min, x_max, y_min, y_max).map(|v| !v)
    }

    /// Check if two ranges are disjoint (don't overlap).
    fn ranges_disjoint(x_min: ConstValue, x_max: ConstValue, y_min: ConstValue, y_max: ConstValue) -> bool {
        use Ordering::*;
        matches!((Self::compare_values(&x_max, &y_min), Self::compare_values(&x_min, &y_max)), (Less, _) | (_, Greater))
    }

    /// Compare two ConstValues for ordering.
    ///
    /// # Panics
    /// Panics if NaN is encountered, as it should have been filtered by
    /// `can_eliminate_comparison` before reaching this point.
    fn compare_values(a: &ConstValue, b: &ConstValue) -> Ordering {
        match (a, b) {
            (ConstValue::Int(x), ConstValue::Int(y)) => x.cmp(y),
            (ConstValue::UInt(x), ConstValue::UInt(y)) => x.cmp(y),
            (ConstValue::Float(x), ConstValue::Float(y)) => {
                // NaN should have been filtered by can_eliminate_comparison
                debug_assert!(!x.is_nan() && !y.is_nan(), "NaN should have been filtered by can_eliminate_comparison");

                if x.is_nan() || y.is_nan() {
                    // In release builds, be conservative: treat as equal (no optimization)
                    // This prevents incorrect eliminations if NaN somehow reaches here
                    Ordering::Equal
                } else {
                    x.partial_cmp(y).unwrap_or(Ordering::Equal)
                }
            }
            (ConstValue::Bool(x), ConstValue::Bool(y)) => x.cmp(y),
            _ => Ordering::Equal, // Mixed types shouldn't happen
        }
    }
}

/// Convenience function for DCE to get all three comparison results at once.
pub fn analyze_all_comparisons(x: &Arc<UOp>, y: &Arc<UOp>) -> (Option<bool>, Option<bool>, Option<bool>) {
    let lt = ComparisonAnalyzer::analyze(BinaryOp::Lt, x, y);
    let eq = ComparisonAnalyzer::analyze(BinaryOp::Eq, x, y);
    let ne = ComparisonAnalyzer::analyze(BinaryOp::Ne, x, y);
    (lt, eq, ne)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lt_analysis() {
        // x in [100, 200], y in [50, 50]
        // x < y should be false
        assert_eq!(
            ComparisonAnalyzer::analyze_with_ranges(
                BinaryOp::Lt,
                ConstValue::Int(100),
                ConstValue::Int(200),
                ConstValue::Int(50),
                ConstValue::Int(50)
            ),
            Some(false)
        );

        // x in [0, 10], y in [20, 30]
        // x < y should be true
        assert_eq!(
            ComparisonAnalyzer::analyze_with_ranges(
                BinaryOp::Lt,
                ConstValue::Int(0),
                ConstValue::Int(10),
                ConstValue::Int(20),
                ConstValue::Int(30)
            ),
            Some(true)
        );
    }

    #[test]
    fn test_eq_analysis() {
        // Non-overlapping ranges
        assert_eq!(
            ComparisonAnalyzer::analyze_with_ranges(
                BinaryOp::Eq,
                ConstValue::Int(0),
                ConstValue::Int(10),
                ConstValue::Int(20),
                ConstValue::Int(30)
            ),
            Some(false)
        );

        // Same constant
        assert_eq!(
            ComparisonAnalyzer::analyze_with_ranges(
                BinaryOp::Eq,
                ConstValue::Int(5),
                ConstValue::Int(5),
                ConstValue::Int(5),
                ConstValue::Int(5)
            ),
            Some(true)
        );
    }
}
