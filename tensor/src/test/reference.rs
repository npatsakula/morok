//! Reference implementations for correctness validation.
//!
//! These are simple, straightforward implementations used to verify
//! that the optimized tensor operations produce correct results.

/// Reference operations on f32 slices.
pub mod ops {
    // =========================================================================
    // Binary Operations
    // =========================================================================

    /// Reference add implementation: element-wise addition.
    pub fn add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    /// Reference subtract implementation: element-wise subtraction.
    pub fn sub_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    }

    /// Reference multiply implementation: element-wise multiplication.
    pub fn mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x * y).collect()
    }

    /// Reference divide implementation: element-wise division.
    pub fn div_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x / y).collect()
    }

    // =========================================================================
    // Reduce Operations
    // =========================================================================

    /// Reference sum implementation: full reduction.
    pub fn sum_f32(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    /// Reference max implementation: full reduction.
    pub fn max_f32(data: &[f32]) -> f32 {
        data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Reference min implementation: full reduction.
    pub fn min_f32(data: &[f32]) -> f32 {
        data.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    /// Reference mean implementation: full reduction.
    pub fn mean_f32(data: &[f32]) -> f32 {
        sum_f32(data) / data.len() as f32
    }

    /// Reference argmax implementation: returns index of first maximum.
    pub fn argmax_f32(data: &[f32]) -> i32 {
        data.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as i32).unwrap_or(0)
    }

    /// Reference argmin implementation: returns index of first minimum.
    pub fn argmin_f32(data: &[f32]) -> i32 {
        data.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as i32).unwrap_or(0)
    }
}
