//! Arch-derived capability bundle for the tile DSL.
//!
//! `svod-tk` kernels are built for a specific GPU arch: the wave width, the
//! cross-lane reduce tree, and the per-lane WMMA-fragment row stride are all
//! arch properties, as is the WMMA descriptor itself ([`crate::group`] looks it
//! up from the shared `TensorCore` table by [`ArchCaps::arch`]). [`ArchCaps`] is
//! the single place those are derived from an [`AmdArch`], so the builders thread
//! one value instead of hardcoding gfx942 (wave64) literals.
//!
//! Both gfx942 (CDNA3, wave64) and gfx1151 (RDNA3.5, wave32) are *built* (see
//! `MATMUL_/FA_SUPPORTED_ARCHS`). gfx942 is the validated/calibrated target — the
//! register-tile fragment-layout tables ([`crate::tiles`] strides and
//! `group::mma`'s per-lane upcast counts) and the [`crate::WARP_THREADS`]
//! layout-table constant are pinned to it; gfx1151 is carried by the dedicated
//! wave32 paths described below.
//!
//! What generalizes cleanly to RDNA3.5 (gfx1151, wave32): [`ArchCaps::wave_size`]
//! (the control path — warp/lane math, launch block), the WMMA descriptor (sourced
//! by arch from the shared `TensorCore` table — RDNA routes to the 32-thread WMMA
//! core), and [`ArchCaps::reduce_tree`] (the `wave_size/16 − 1` sibling-fold formula
//! yields the correct `[16]` for the wave32 even/odd accumulator — see below). The
//! RDNA WMMA *fragment* layout is otherwise different (`ept=(16,16,8)`, inputs
//! replicated across the two wave-halves, an even/odd-interleaved `<8×float>`
//! accumulator), so it is carried by dedicated tile shapes (`RT_16X16_W32_*` in
//! [`crate::tiles`]) resolved per arch by [`ArchCaps::frag`] / [`ArchCaps::shared_default`]
//! / [`ArchCaps::shared_swizzled`] — the single arch→fragment table, so kernels stay
//! arch-blind (no `is_cdna()` shape branches). Both matmul and FA are now built for gfx1151 (in
//! `MATMUL_/FA_SUPPORTED_ARCHS`); [`ArchCaps::frag_row_stride`] is the one remaining
//! CDNA-only datum (the legacy direct-launch FA mask — the production rolled-db
//! kernel derives its mask from the accumulator's own `lane_rc` instead).

use smallvec::SmallVec;
use svod_dtype::AmdArch;

use crate::tiles::{
    RT_16X16, RT_16X16_W32_ACC, RT_16X16_W32_ACC_T, RT_16X16_W32_IN, RTBaseShape, ST_16X16, ST_16X16_SWIZZLED,
    ST_16X16_SWIZZLED_W32, STBaseShape,
};

/// Logical role of a 16×16 matrix-core fragment, independent of arch packing.
/// Resolved to a physical [`RTBaseShape`] by [`ArchCaps::frag`] — kernels select a
/// fragment by *role*, never by naming a per-arch constant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FragRole {
    /// f32 MMA output / online-softmax accumulator.
    Accumulator,
    /// f16/bf16 WMMA input operand (A or B).
    Operand,
    /// Accumulator transposed for an N-major store (e.g. the FA output `O[q,d]`
    /// from the `[d,q]` PV accumulator).
    AccumulatorT,
}

/// The arch-derived constants the tile builders thread instead of the wave64
/// literals. `Copy`; [`Self::for_arch`] is `const`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchCaps {
    /// The target GPU arch — drives the WMMA descriptor lookup
    /// (`Renderer::for_amd_arch`) in [`crate::group`].
    pub arch: AmdArch,
    /// Lanes per wave. `threadIdx` splits into warp = `idx / wave_size` and lane
    /// = `idx % wave_size`; the launch block is `warps * wave_size`. 64 on CDNA3,
    /// 32 on RDNA3/4.
    pub wave_size: usize,
}

impl ArchCaps {
    /// Derive the caps from `arch` (wave size from [`AmdArch::wave_size`]).
    pub const fn for_arch(arch: AmdArch) -> Self {
        Self { arch, wave_size: arch.wave_size() as usize }
    }

    /// The validated default target: gfx942 (CDNA3, wave64).
    pub const GFX942: ArchCaps = ArchCaps::for_arch(AmdArch::Gfx942);

    /// Cross-lane (`ds_bpermute`) reduce-tree offsets: a lane folds the partials of
    /// the `wave_size / 16` sibling row-groups, each one WMMA-column span (16 lanes)
    /// apart → wave64 `[16, 32, 48]`, wave32 `[16]`. This is correct for **both** the
    /// CDNA MFMA layout *and* the RDNA even/odd accumulator: at wave32 a softmax row
    /// (16 KV) is split across a lane's 8 in-register elements (the even/odd half)
    /// and its sibling lane `L+16` (the other half), so the single `[16]` fold plus
    /// the in-register reduce covers the whole row (HW-validated reduce structure).
    pub fn reduce_tree(&self) -> SmallVec<[i64; 3]> {
        (1..self.wave_size as i64 / 16).map(|i| i * 16).collect()
    }

    /// Per-lane row stride of a 16×16 **CDNA MFMA** accumulator fragment
    /// (`256 / wave_size`): wave64 → 4. Used only by the legacy direct-launch FA
    /// builders' causal/padding mask (which map each `laneid / 16` row-group to KV
    /// rows with this contiguous stride). The production rolled-db FA derives its
    /// mask from the att accumulator's own `lane_rc` instead — arch-correct for both
    /// the CDNA stride and the RDNA even/odd interleave — so it does not call this.
    pub const fn frag_row_stride(&self) -> i64 {
        (16 * 16 / self.wave_size) as i64
    }

    /// Physical register fragment for a logical [`FragRole`] on this arch — the
    /// single arch→fragment table the kernels resolve through. CDNA's MFMA
    /// accumulator and input fragments share a layout, so every role resolves to
    /// [`RT_16X16`]; RDNA (gfx11 WMMA) splits into the even/odd-interleaved
    /// accumulator, the replicated input, and the transposed accumulator.
    pub fn frag(&self, role: FragRole) -> RTBaseShape {
        if self.arch.is_cdna() {
            RT_16X16
        } else {
            match role {
                FragRole::Accumulator => RT_16X16_W32_ACC,
                FragRole::Operand => RT_16X16_W32_IN,
                FragRole::AccumulatorT => RT_16X16_W32_ACC_T,
            }
        }
    }

    /// The canonical LDS strip fragment: plain on CDNA. On RDNA the only ept-8 strip
    /// defined is swizzled, so it coincides with [`Self::shared_swizzled`]. Used by
    /// kernels whose LDS access does not itself need the XOR swizzle (flash-attention).
    pub fn shared_default(&self) -> STBaseShape {
        if self.arch.is_cdna() { ST_16X16 } else { ST_16X16_SWIZZLED_W32 }
    }

    /// The XOR-swizzled LDS strip fragment, for kernels that swizzle to avoid LDS
    /// bank conflicts (the matmul A/B strips).
    pub fn shared_swizzled(&self) -> STBaseShape {
        if self.arch.is_cdna() { ST_16X16_SWIZZLED } else { ST_16X16_SWIZZLED_W32 }
    }

    /// Whether an MMA accumulator fragment can be reused directly as a WMMA input via
    /// a register copy. True on CDNA (MFMA acc == input fragment); false on RDNA (the
    /// even/odd `<8×f32>` accumulator and the replicated `<16×in>` input differ), where
    /// the acc→input handoff must round-trip through LDS instead.
    pub fn acc_reusable_as_input(&self) -> bool {
        self.arch.is_cdna()
    }
}

/// What a launcher resolves from the **physical device**: the arch-derived
/// [`ArchCaps`] plus topology scalars that vary across SKUs and virtualized
/// partitions of the *same* arch (a partition exposes fewer CUs than the full
/// device). Kept distinct from [`ArchCaps`] so the latter stays a pure function of
/// the arch; these fields are not arch-derivable and come from the KFD probe in
/// [`crate::target::resolve_supported_profile`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceProfile {
    /// Arch-derived capabilities (wave size, fragment shapes).
    pub caps: ArchCaps,
    /// Whole-device compute-unit count (`simd_count / simd_per_cu` from KFD) — the
    /// grid-saturation target for the split-K / corpus-split / FA-tile heuristics.
    pub cu_count: usize,
}
