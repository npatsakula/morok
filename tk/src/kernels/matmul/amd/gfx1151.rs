//! The gfx1151 (RDNA3.5, wave32 WMMA) specialization home. Today gfx1151 runs the
//! generic [`gemm_core`](super::super::common::gemm_core) through [`GFX1151_CFG`] —
//! the occupancy-tuned config below, selected by
//! [`cfg_for_arch`](super::super::cfg_for_arch). A future tracked-WMMA pipeline (the
//! RDNA3.5 peer of gfx942's inline-`asm` microkernel) lands here.

use super::super::MatmulCfg;

/// gfx1151 (RDNA3.5, wave32) config: 64×64 block, 2×2
/// waves (4 waves / 128 threads), ONE
/// 32×32 accumulator/wave, 128-bit vec fills, no L2 swizzle (single-XCD APU), and
/// **`k_step = 32`**. The `reg=32` tile keeps accumulator VGPR ≈ 32/lane; the
/// `k_step=32` halves the live WMMA-input fragment VGPR vs the default 64 (the input
/// replicates all `k_step`/16 K-sub-steps per lane), raising occupancy. `k_step` is
/// the dominant occupancy lever on RDNA3.5/wave32; the single-buffered path has no
/// memory stall a double buffer could hide. gfx942 keeps `k_step = K_STEP` (64). A
/// smaller `k_step` lowers the WMMA-input VGPR but adds barriers, so the tuned value
/// trades occupancy against barrier overhead.
pub const GFX1151_CFG: MatmulCfg =
    MatmulCfg { block: 64, wave_rows: 2, wave_cols: 2, n_accum: 1, l2_swizzle: false, vec_load: true, k_step: 32 };
