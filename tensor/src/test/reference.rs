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
}
