//! Backend-agnostic profiling data shared across the device boundary.
//!
//! These types live in `device` (not `runtime`) because the device traits
//! [`crate::DispatchTimestamps::counters`] and [`crate::Program::resource_usage`]
//! return them. `runtime` re-exports them through its `profiler` module so callers
//! see one profiling vocabulary.

use std::collections::BTreeMap;

/// A hardware performance counter selectable via PMC. Names are arch-neutral;
/// each backend maps them to its block/perf-select values. The SQ subset answers
/// the ILP/occupancy question (VALU vs SQ-busy cycles, waves launched); the
/// gfx942 (CDNA3) additions cover LDS bank conflicts, MFMA occupancy, and L2
/// hit/miss so derived-metric rows can be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PmcCounter {
    /// Cycles the shader sequencer (SQ) was busy.
    SqBusyCycles,
    /// Waves launched.
    SqWaves,
    /// VALU instructions issued.
    SqInstsValu,
    /// SALU instructions issued.
    SqInstsSalu,
    /// LDS bank-conflict cycles.
    LdsBankConflict,
    /// LDS index-unit active cycles (denominator for the conflict rate).
    LdsIdxActive,
    /// GRBM GUI-active cycles (denominator for MFMA utilization).
    GrbmGuiActive,
    /// L2 (TCC) cache hits.
    L2Hit,
    /// L2 (TCC) cache misses.
    L2Miss,
    /// Cycles the VALU was busy issuing MFMA (matrix) instructions.
    ValuMfmaBusyCycles,
    /// MFMA (matrix) instructions issued.
    InstsMfma,
}

impl PmcCounter {
    /// Short token used in `SVOD_PMC=…` selection and as a table header.
    pub fn token(self) -> &'static str {
        match self {
            Self::SqBusyCycles => "sqbusy",
            Self::SqWaves => "waves",
            Self::SqInstsValu => "valu",
            Self::SqInstsSalu => "salu",
            Self::LdsBankConflict => "bankconflict",
            Self::LdsIdxActive => "ldsact",
            Self::GrbmGuiActive => "gui",
            Self::L2Hit => "l2hit",
            Self::L2Miss => "l2miss",
            Self::ValuMfmaBusyCycles => "mfmabusy",
            Self::InstsMfma => "mfma",
        }
    }

    /// Parse a `SVOD_PMC` token; unknown tokens return `None`.
    pub fn from_token(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "sqbusy" | "busy" => Self::SqBusyCycles,
            "waves" => Self::SqWaves,
            "valu" => Self::SqInstsValu,
            "salu" => Self::SqInstsSalu,
            "bankconflict" => Self::LdsBankConflict,
            "ldsact" => Self::LdsIdxActive,
            "gui" => Self::GrbmGuiActive,
            "l2hit" => Self::L2Hit,
            "l2miss" => Self::L2Miss,
            "mfmabusy" => Self::ValuMfmaBusyCycles,
            "mfma" => Self::InstsMfma,
            _ => return None,
        })
    }

    /// The arch-neutral default selection (SQ occupancy triple). Kept small so a
    /// bare `SVOD_PMC=1` works on every supported arch; the gfx942-only counters
    /// are opt-in by token.
    pub fn all() -> [PmcCounter; 3] {
        [Self::SqBusyCycles, Self::SqWaves, Self::SqInstsValu]
    }
}

/// Hardware counter values harvested for one dispatch. Sparse: only the
/// requested counters are present.
#[derive(Debug, Clone, Default)]
pub struct CounterSet {
    pub values: BTreeMap<PmcCounter, u64>,
    /// Number of XCC compute engines the counters were summed over (0 = unknown).
    /// Needed to normalize cross-block derived metrics (e.g. MFMA utilization,
    /// which divides an SE-summed SQ counter by an XCC-summed GRBM counter).
    pub xcc_num: u32,
    /// Device-total SIMD lane count (`CU_NUM · SIMDs_per_CU`, 0 = unknown), the
    /// MFMA-utilization denominator's `CU_NUM · 4` term on gfx9.
    pub device_simds: u32,
    /// Peak engine clock in MHz (0 = unknown), the reference `F_peak` for
    /// achieved-clock (`sclk`) derivation and clock-normalized MFMA utilization.
    pub peak_clk_mhz: u32,
}

/// Per-kernel static GPU resource usage, decoded from the compiled program's
/// kernel descriptor. Pure static — no runtime cost.
#[derive(Debug, Clone, Copy)]
pub struct KernelResources {
    /// Vector GPRs allocated per lane.
    pub vgprs: u32,
    /// Scalar GPRs allocated.
    pub sgprs: u32,
    /// LDS (group segment) bytes per workgroup.
    pub lds_bytes: u32,
    /// Scratch (private segment) bytes per lane.
    pub scratch_bytes: u32,
    /// Wave width (32 or 64).
    pub wave_size: u32,
    /// VGPR-limited occupancy as a fraction of the SIMD's max resident waves
    /// (0.0–1.0), for architectures with known register-file geometry. `None`
    /// when the geometry is unknown — the first-order limiter only (LDS and
    /// workgroup limits are not modeled).
    pub occupancy: Option<f32>,
}
