//! AMD GPU architecture targets (gfx-family).
//!
//! Covers the arch set handled by the AMD LLVM renderer and the `is_cdna`
//! predicate.

use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AmdArch {
    // CDNA — datacenter; MFMA, wave64.
    Gfx942,
    Gfx950,
    // RDNA2 — Radeon 6000; no matrix cores, wave32. gfx1030 = Navi 21 (RX 6900 XT).
    Gfx1030,
    Gfx1031,
    Gfx1032,
    Gfx1034,
    // RDNA3 — Radeon 7000; WMMA, wave32.
    Gfx1100,
    Gfx1101,
    Gfx1102,
    Gfx1151,
    // RDNA4 — next-gen Radeon; WMMA with bf16/fp8 packing, wave32.
    Gfx1200,
    Gfx1201,
}

impl AmdArch {
    /// CDNA family (datacenter; MFMA intrinsics, wave64).
    pub const fn is_cdna(self) -> bool {
        matches!(self, Self::Gfx942 | Self::Gfx950)
    }

    /// gfx major version (`9` = CDNA, `10` = RDNA2, `11` = RDNA3, `12` = RDNA4) —
    /// the discriminator several PM4 cache encodings branch on.
    pub const fn gfx_major(self) -> u32 {
        match self {
            Self::Gfx942 | Self::Gfx950 => 9,
            Self::Gfx1030 | Self::Gfx1031 | Self::Gfx1032 | Self::Gfx1034 => 10,
            Self::Gfx1100 | Self::Gfx1101 | Self::Gfx1102 | Self::Gfx1151 => 11,
            Self::Gfx1200 | Self::Gfx1201 => 12,
        }
    }

    /// RDNA2 family (Radeon 6000; no matrix cores, wave32).
    pub const fn is_rdna2(self) -> bool {
        matches!(self, Self::Gfx1030 | Self::Gfx1031 | Self::Gfx1032 | Self::Gfx1034)
    }

    /// RDNA3 family (Radeon 7000; WMMA intrinsics, wave32).
    pub const fn is_rdna3(self) -> bool {
        matches!(self, Self::Gfx1100 | Self::Gfx1101 | Self::Gfx1102 | Self::Gfx1151)
    }

    /// RDNA4 family (next-gen Radeon; WMMA with bf16/fp8 packing, wave32).
    pub const fn is_rdna4(self) -> bool {
        matches!(self, Self::Gfx1200 | Self::Gfx1201)
    }

    /// `true` when the arch has hardware matrix-multiply intrinsics
    /// (CDNA's MFMA or RDNA3+'s WMMA). RDNA2 (gfx10) has none — matmul lowers to
    /// scalar/vector FMA, driven by an empty `tensor_cores` list in its renderer
    /// profile (the optimizer then never emits a WMMA UOp).
    pub const fn has_matrix_cores(self) -> bool {
        self.is_cdna() || self.is_rdna3() || self.is_rdna4()
    }

    /// Default wave size for this arch (clang `-mcpu` selects this; the kernel
    /// descriptor's `kernel_code_properties` bit 10 encodes wave32 when set).
    pub const fn wave_size(self) -> u32 {
        if self.is_cdna() { 64 } else { 32 }
    }

    /// clang `-mcpu=...` string.
    pub const fn mcpu(self) -> &'static str {
        match self {
            Self::Gfx942 => "gfx942",
            Self::Gfx950 => "gfx950",
            Self::Gfx1030 => "gfx1030",
            Self::Gfx1031 => "gfx1031",
            Self::Gfx1032 => "gfx1032",
            Self::Gfx1034 => "gfx1034",
            Self::Gfx1100 => "gfx1100",
            Self::Gfx1101 => "gfx1101",
            Self::Gfx1102 => "gfx1102",
            Self::Gfx1151 => "gfx1151",
            Self::Gfx1200 => "gfx1200",
            Self::Gfx1201 => "gfx1201",
        }
    }

    /// Map KFD's `gfx_target_version` integer to an `AmdArch`.
    ///
    /// Format: `major*10000 + minor*100 + stepping` (the encoding used by
    /// `/sys/.../properties` `gfx_target_version`).
    pub fn from_gfx_target_version(v: u32) -> Option<Self> {
        // KFD's `gfx_target_version` encodes (major, minor, step) as
        //   v = major*10_000 + minor*100 + step
        // (decimal, not hex — the `gfx%d%x%x` string formatting re-hexifies
        // minor/step purely for display).
        Some(match v {
            90_402 => Self::Gfx942,
            90_500 => Self::Gfx950,
            100_300 => Self::Gfx1030,
            100_301 => Self::Gfx1031,
            100_302 => Self::Gfx1032,
            100_304 => Self::Gfx1034,
            110_000 => Self::Gfx1100,
            110_001 => Self::Gfx1101,
            110_002 => Self::Gfx1102,
            110_501 => Self::Gfx1151,
            120_000 => Self::Gfx1200,
            120_001 => Self::Gfx1201,
            _ => return None,
        })
    }

    /// Parse a `gfx{family}` string (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "gfx942" => Some(Self::Gfx942),
            "gfx950" => Some(Self::Gfx950),
            "gfx1030" => Some(Self::Gfx1030),
            "gfx1031" => Some(Self::Gfx1031),
            "gfx1032" => Some(Self::Gfx1032),
            "gfx1034" => Some(Self::Gfx1034),
            "gfx1100" => Some(Self::Gfx1100),
            "gfx1101" => Some(Self::Gfx1101),
            "gfx1102" => Some(Self::Gfx1102),
            "gfx1151" => Some(Self::Gfx1151),
            "gfx1200" => Some(Self::Gfx1200),
            "gfx1201" => Some(Self::Gfx1201),
            _ => None,
        }
    }
}

impl fmt::Display for AmdArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.mcpu())
    }
}

#[cfg(test)]
#[path = "test/unit/amd_arch.rs"]
mod tests;
