//! Optimizer configuration types.
//!
//! Provides typed configuration for kernel optimization with bon builders.
//! Supports both explicit configuration and environment variable fallbacks.

use std::time::Duration;

use bon::bon;

// ============================================================================
// OPTIMIZATION STRATEGY
// ============================================================================

/// Optimization strategy for kernel tuning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OptStrategy {
    /// No optimization (for debugging/regression testing).
    None,

    /// Hand-coded heuristics (default).
    #[default]
    Heuristic,

    /// Beam search optimization.
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
    pub fn from_env() -> Self {
        if std::env::var("MOROK_NOOPT").is_ok() {
            return Self::None;
        }

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

// ============================================================================
// TENSOR CORE SETTINGS
// ============================================================================

/// Tensor core usage level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TcUsage {
    /// Disabled (USE_TC=0).
    Disabled,

    /// Enabled (USE_TC=1, default).
    #[default]
    Enabled,

    /// Shape-only mode (USE_TC=2).
    ShapeOnly,
}

impl TcUsage {
    /// Convert to integer value for internal APIs.
    pub fn as_usize(&self) -> usize {
        match self {
            Self::Disabled => 0,
            Self::Enabled => 1,
            Self::ShapeOnly => 2,
        }
    }
}

/// Tensor core optimization level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TcOpt {
    /// Strict matching (TC_OPT=0).
    Strict,

    /// Relaxed matching (TC_OPT=1).
    Relaxed,

    /// Padded matching (TC_OPT=2, default).
    #[default]
    Padded,
}

impl TcOpt {
    /// Convert to integer value for internal APIs.
    pub fn as_usize(&self) -> usize {
        match self {
            Self::Strict => 0,
            Self::Relaxed => 1,
            Self::Padded => 2,
        }
    }
}

/// Tensor core selection mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TcSelect {
    /// Auto-select best tensor core (TC_SELECT=-1, default).
    #[default]
    Auto,

    /// Use specific tensor core index.
    Index(usize),
}

impl TcSelect {
    /// Convert to integer value for internal APIs.
    pub fn as_i32(&self) -> i32 {
        match self {
            Self::Auto => -1,
            Self::Index(idx) => *idx as i32,
        }
    }
}

// ============================================================================
// BEAM SEARCH CONFIGURATION
// ============================================================================

/// Configuration for beam search auto-tuning.
#[derive(Debug, Clone)]
pub struct BeamConfig {
    /// Beam width - number of candidates to keep at each step.
    pub beam_width: usize,
    /// Maximum search time.
    pub timeout: Duration,
    /// Maximum upcast size (product of UPCAST/UNROLL dimensions).
    pub max_upcast: usize,
    /// Maximum local size (product of LOCAL/WARP/GROUP_REDUCE dimensions).
    pub max_local: usize,
    /// Maximum UOps in kernel before rejecting.
    pub max_uops: usize,
    /// Number of benchmark runs per kernel.
    pub num_runs: usize,
    /// Disable disk cache.
    pub disable_cache: bool,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            timeout: Duration::from_secs(60),
            max_upcast: 256,
            max_local: 1024,
            max_uops: 3000,
            num_runs: 3,
            disable_cache: false,
        }
    }
}

#[bon]
impl BeamConfig {
    /// Create a beam configuration with builder pattern.
    #[builder]
    pub fn builder(
        #[builder(default = 4)] beam_width: usize,
        #[builder(default = 60)] timeout_secs: u64,
        #[builder(default = 256)] max_upcast: usize,
        #[builder(default = 1024)] max_local: usize,
        #[builder(default = 3000)] max_uops: usize,
        #[builder(default = 3)] num_runs: usize,
        #[builder(default = false)] disable_cache: bool,
    ) -> Self {
        Self {
            beam_width,
            timeout: Duration::from_secs(timeout_secs),
            max_upcast,
            max_local,
            max_uops,
            num_runs,
            disable_cache,
        }
    }

    /// Create configuration from environment variables.
    ///
    /// # Environment Variables
    ///
    /// * `MOROK_BEAM` - Beam width (default: 4)
    /// * `MOROK_BEAM_TIMEOUT` - Max search time in seconds (default: 60)
    /// * `BEAM_UPCAST_MAX` - Max upcast size (default: 256)
    /// * `BEAM_LOCAL_MAX` - Max local memory elements (default: 1024)
    /// * `BEAM_UOPS_MAX` - Max UOps before rejecting (default: 3000)
    /// * `BEAM_RUNS` - Benchmark runs per kernel (default: 3)
    /// * `IGNORE_BEAM_CACHE` - Bypass disk cache if set
    pub fn from_env() -> Self {
        let beam_width = std::env::var("MOROK_BEAM").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        let timeout_secs = std::env::var("MOROK_BEAM_TIMEOUT").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
        let max_upcast = std::env::var("BEAM_UPCAST_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
        let max_local = std::env::var("BEAM_LOCAL_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
        let max_uops = std::env::var("BEAM_UOPS_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(3000);
        let num_runs = std::env::var("BEAM_RUNS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        let disable_cache = std::env::var("IGNORE_BEAM_CACHE").is_ok();

        Self {
            beam_width,
            timeout: Duration::from_secs(timeout_secs),
            max_upcast,
            max_local,
            max_uops,
            num_runs,
            disable_cache,
        }
    }

    /// Get beam width from strategy if applicable.
    pub fn with_strategy_width(mut self, strategy: &OptStrategy) -> Self {
        if let OptStrategy::Beam { width } = strategy {
            self.beam_width = *width;
        }
        self
    }
}

// ============================================================================
// HEURISTICS CONFIGURATION
// ============================================================================

/// Configuration for heuristic-based optimization.
#[derive(Debug, Clone)]
pub struct HeuristicsConfig {
    // Tensor cores
    /// Tensor core usage level.
    pub tc_enabled: TcUsage,
    /// Tensor core optimization level.
    pub tc_opt: TcOpt,
    /// Tensor core selection mode.
    pub tc_select: TcSelect,

    // Matrix-vector optimization
    /// Enable matrix-vector optimization.
    pub matvec_enabled: bool,
    /// Matrix-vector block size (rows per workgroup).
    pub matvec_blocksize: usize,

    // Reduction thresholds
    /// Threshold for applying grouped reduction.
    pub grouped_threshold: usize,
    /// Threshold for applying unroll.
    pub unroll_threshold: usize,

    // Local memory
    /// Disable local memory globally.
    pub disable_locals: bool,

    // Threading
    /// Maximum thread count for CPU parallelization.
    /// Default: std::thread::available_parallelism().
    /// Set to 1 to disable threading.
    pub thread_count: usize,

    // Vectorization
    /// Enable K-axis vectorization for matmul.
    /// When enabled, UPCAST is applied to the reduce (K) axis with scalar accumulators.
    /// This enables SLP vectorization but prevents direct FMA intrinsic generation.
    /// Default: true.
    pub k_vectorize: bool,

    // Debug
    /// Debug verbosity level.
    pub debug_level: u8,
}

/// Get default thread count from system (used by Default and builder).
fn default_thread_count() -> usize {
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8)
}

impl HeuristicsConfig {
    /// Create configuration from environment variables.
    ///
    /// # Environment Variables
    ///
    /// * `MOROK_THREADS` - Maximum thread count (default: available_parallelism)
    /// * `MOROK_NO_K_VECTORIZE` - Disable K-axis vectorization (enables direct FMA)
    pub fn from_env() -> Self {
        let thread_count =
            std::env::var("MOROK_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or_else(default_thread_count);
        let k_vectorize = std::env::var("MOROK_NO_K_VECTORIZE").is_err();

        Self { thread_count, k_vectorize, ..Default::default() }
    }
}

impl Default for HeuristicsConfig {
    fn default() -> Self {
        Self {
            tc_enabled: TcUsage::Enabled,
            tc_opt: TcOpt::Padded,
            tc_select: TcSelect::Auto,
            matvec_enabled: true,
            matvec_blocksize: 4,
            grouped_threshold: 256,
            unroll_threshold: 32,
            disable_locals: false,
            thread_count: default_thread_count(),
            k_vectorize: true,
            debug_level: 0,
        }
    }
}

#[bon]
impl HeuristicsConfig {
    /// Create a heuristics configuration with builder pattern.
    #[builder]
    pub fn builder(
        #[builder(default)] tc_enabled: TcUsage,
        #[builder(default)] tc_opt: TcOpt,
        #[builder(default)] tc_select: TcSelect,
        #[builder(default = true)] matvec_enabled: bool,
        #[builder(default = 4)] matvec_blocksize: usize,
        #[builder(default = 256)] grouped_threshold: usize,
        #[builder(default = 32)] unroll_threshold: usize,
        #[builder(default = false)] disable_locals: bool,
        #[builder(default = default_thread_count())] thread_count: usize,
        #[builder(default = true)] k_vectorize: bool,
        #[builder(default = 0)] debug_level: u8,
    ) -> Self {
        Self {
            tc_enabled,
            tc_opt,
            tc_select,
            matvec_enabled,
            matvec_blocksize,
            grouped_threshold,
            unroll_threshold,
            disable_locals,
            thread_count,
            k_vectorize,
            debug_level,
        }
    }
}

// ============================================================================
// TOP-LEVEL OPTIMIZER CONFIGURATION
// ============================================================================

/// Top-level optimizer configuration.
///
/// Combines strategy selection, beam search settings, and heuristic parameters.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct OptimizerConfig {
    /// Optimization strategy (None, Heuristic, or Beam).
    pub strategy: OptStrategy,
    /// Beam search configuration (used when strategy is Beam).
    pub beam: BeamConfig,
    /// Heuristics configuration (used when strategy is Heuristic).
    pub heuristics: HeuristicsConfig,
}

#[bon]
impl OptimizerConfig {
    /// Create an optimizer configuration with builder pattern.
    #[builder]
    pub fn builder(
        #[builder(default)] strategy: OptStrategy,
        #[builder(default)] beam: BeamConfig,
        #[builder(default)] heuristics: HeuristicsConfig,
    ) -> Self {
        Self { strategy, beam, heuristics }
    }

    /// Create configuration from environment variables.
    ///
    /// Reads strategy from env, then populates beam and heuristics config accordingly.
    pub fn from_env() -> Self {
        let strategy = OptStrategy::from_env();
        let beam = BeamConfig::from_env().with_strategy_width(&strategy);
        let heuristics = HeuristicsConfig::from_env();

        Self { strategy, beam, heuristics }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_strategy_default_is_heuristic() {
        assert_eq!(OptStrategy::default(), OptStrategy::Heuristic);
    }

    #[test]
    fn test_opt_strategy_is_none() {
        assert!(OptStrategy::None.is_none());
        assert!(!OptStrategy::Heuristic.is_none());
        assert!(!OptStrategy::Beam { width: 4 }.is_none());
    }

    #[test]
    fn test_opt_strategy_is_beam() {
        assert!(!OptStrategy::None.is_beam());
        assert!(!OptStrategy::Heuristic.is_beam());
        assert!(OptStrategy::Beam { width: 4 }.is_beam());
    }

    #[test]
    fn test_beam_config_default() {
        let config = BeamConfig::default();
        assert_eq!(config.beam_width, 4);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.max_upcast, 256);
        assert_eq!(config.max_local, 1024);
    }

    #[test]
    fn test_beam_config_builder() {
        let config = BeamConfig::builder().beam_width(8).timeout_secs(120).max_upcast(512).build();

        assert_eq!(config.beam_width, 8);
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.max_upcast, 512);
        assert_eq!(config.max_local, 1024); // default
    }

    #[test]
    fn test_heuristics_config_default() {
        let config = HeuristicsConfig::default();
        assert_eq!(config.tc_enabled, TcUsage::Enabled);
        assert_eq!(config.tc_opt, TcOpt::Padded);
        assert!(config.matvec_enabled);
        assert_eq!(config.grouped_threshold, 256);
    }

    #[test]
    fn test_heuristics_config_builder() {
        let config = HeuristicsConfig::builder()
            .tc_enabled(TcUsage::Disabled)
            .matvec_enabled(false)
            .grouped_threshold(128)
            .build();

        assert_eq!(config.tc_enabled, TcUsage::Disabled);
        assert!(!config.matvec_enabled);
        assert_eq!(config.grouped_threshold, 128);
    }

    #[test]
    fn test_optimizer_config_default() {
        let config = OptimizerConfig::default();
        assert_eq!(config.strategy, OptStrategy::Heuristic);
        assert_eq!(config.beam.beam_width, 4);
    }

    #[test]
    fn test_optimizer_config_builder() {
        let config = OptimizerConfig::builder()
            .strategy(OptStrategy::Beam { width: 8 })
            .beam(BeamConfig::builder().timeout_secs(120).build())
            .build();

        assert_eq!(config.strategy, OptStrategy::Beam { width: 8 });
        assert_eq!(config.beam.timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_tc_usage_as_usize() {
        assert_eq!(TcUsage::Disabled.as_usize(), 0);
        assert_eq!(TcUsage::Enabled.as_usize(), 1);
        assert_eq!(TcUsage::ShapeOnly.as_usize(), 2);
    }

    #[test]
    fn test_tc_opt_as_usize() {
        assert_eq!(TcOpt::Strict.as_usize(), 0);
        assert_eq!(TcOpt::Relaxed.as_usize(), 1);
        assert_eq!(TcOpt::Padded.as_usize(), 2);
    }

    #[test]
    fn test_tc_select_as_i32() {
        assert_eq!(TcSelect::Auto.as_i32(), -1);
        assert_eq!(TcSelect::Index(5).as_i32(), 5);
    }
}
