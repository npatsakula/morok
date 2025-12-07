//! Optimization strategy selection.
//!
//! Provides strategy abstraction for choosing between different optimization approaches:
//! - Heuristic-based (fast, reasonable performance)
//! - Beam search (slow, ML-quality performance) - future
//!
//! Strategy can be configured via environment variables.

/// Optimization strategy for kernel tuning.
///
/// Controls which optimization approach to use when transforming kernel ASTs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OptStrategy {
    /// No optimization (for debugging/regression testing).
    ///
    /// Passes the AST through unchanged.
    None,

    /// Hand-coded heuristics (default).
    ///
    /// Fast compilation with reasonable performance.
    /// Uses deterministic rules based on kernel structure.
    #[default]
    Heuristic,

    /// Beam search optimization (future).
    ///
    /// Slower compilation but better performance.
    /// Explores optimization space using search.
    Beam {
        /// Beam width - number of candidates to keep at each step.
        width: usize,
    },
}

impl OptStrategy {
    /// Get optimization strategy from environment variables.
    ///
    /// # Environment Variables
    ///
    /// * `MOROK_NOOPT=1` - Disable all optimizations
    /// * `MOROK_BEAM=N` - Use beam search with width N
    ///
    /// # Returns
    ///
    /// The selected strategy (defaults to Heuristic).
    pub fn from_env() -> Self {
        // Check for optimization bypass
        if std::env::var("MOROK_NOOPT").is_ok() {
            return Self::None;
        }

        // Check for beam search
        if let Ok(beam_str) = std::env::var("MOROK_BEAM")
            && let Ok(width) = beam_str.parse::<usize>()
            && width > 0
        {
            return Self::Beam { width };
        }

        Self::Heuristic
    }

    /// Check if this strategy disables optimization.
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Check if this strategy uses beam search.
    pub fn is_beam(&self) -> bool {
        matches!(self, Self::Beam { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_heuristic() {
        assert_eq!(OptStrategy::default(), OptStrategy::Heuristic);
    }

    #[test]
    fn test_is_none() {
        assert!(OptStrategy::None.is_none());
        assert!(!OptStrategy::Heuristic.is_none());
        assert!(!OptStrategy::Beam { width: 4 }.is_none());
    }

    #[test]
    fn test_is_beam() {
        assert!(!OptStrategy::None.is_beam());
        assert!(!OptStrategy::Heuristic.is_beam());
        assert!(OptStrategy::Beam { width: 4 }.is_beam());
    }
}
