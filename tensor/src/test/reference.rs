//! Reference implementations for correctness validation.
//!
//! These are simple, straightforward implementations used to verify
//! that the optimized tensor operations produce correct results.

/// Reference operations on f32 slices.
pub mod ops {
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

    /// Reference sum implementation: sum all elements.
    pub fn sum_f32(a: &[f32]) -> f32 {
        a.iter().sum()
    }

    /// Reference product implementation: product of all elements.
    pub fn prod_f32(a: &[f32]) -> f32 {
        a.iter().product()
    }

    /// Reference max implementation: maximum of all elements.
    pub fn max_f32(a: &[f32]) -> f32 {
        a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Reference min implementation: minimum of all elements.
    pub fn min_f32(a: &[f32]) -> f32 {
        a.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Reference mean implementation: average of all elements.
    pub fn mean_f32(a: &[f32]) -> f32 {
        if a.is_empty() {
            0.0
        } else {
            a.iter().sum::<f32>() / a.len() as f32
        }
    }
}
