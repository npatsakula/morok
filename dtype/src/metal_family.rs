//! Apple GPU family (`MTLGPUFamily`), the Metal counterpart of [`super::AmdArch`].
//!
//! Only the coarse family matters to the optimizer: `simdgroup_matrix`
//! (tensor cores) needs Apple7 or later, so Intel/AMD GPUs in Macs (`Mac2`)
//! and pre-M1 Apple GPUs run without them.

use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MetalFamily {
    /// A GPU that matched no known family (very old, or a future-only family).
    Unknown,
    /// `MTLGPUFamilyMac2`: Intel and AMD GPUs in Intel Macs.
    Mac2,
    /// `MTLGPUFamilyAppleN` (Apple7 = M1/A14, Apple8 = M2, Apple9 = M3/M4).
    Apple(u8),
}

impl MetalFamily {
    /// `simdgroup_multiply_accumulate` is available from Apple7 (M1) on.
    pub const fn has_simdgroup_matrix(self) -> bool {
        matches!(self, Self::Apple(generation) if generation >= 7)
    }

    /// The family's `MTLGPUFamily` label (`Apple9`, `Mac2`, `Unknown`).
    pub fn parse(label: &str) -> Option<Self> {
        match label {
            "Unknown" => Some(Self::Unknown),
            "Mac2" => Some(Self::Mac2),
            _ => label.strip_prefix("Apple").and_then(|generation| generation.parse().ok()).map(Self::Apple),
        }
    }
}

impl fmt::Display for MetalFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => f.write_str("Unknown"),
            Self::Mac2 => f.write_str("Mac2"),
            Self::Apple(generation) => write!(f, "Apple{generation}"),
        }
    }
}

#[cfg(test)]
#[path = "test/unit/metal_family.rs"]
mod tests;
