//! Static GPU resource + occupancy decoding from the AMD kernel descriptor.
//!
//! All values come from `COMPUTE_PGM_RSRC1` and the descriptor's segment sizes
//! — no execution, no hardware access.

use crate::profile::KernelResources;

// COMPUTE_PGM_RSRC1 bit fields (amdhsa, stable across GFX6–GFX12).
const RSRC1_VGPR_SHIFT: u32 = 0; // GRANULATED_WORKITEM_VGPR_COUNT, bits [5:0]
const RSRC1_VGPR_MASK: u32 = 0x3f;
const RSRC1_SGPR_SHIFT: u32 = 6; // GRANULATED_WAVEFRONT_SGPR_COUNT, bits [9:6]
const RSRC1_SGPR_MASK: u32 = 0xf;

/// Decode allocated VGPR/SGPR/LDS/scratch and, for architectures with known
/// register-file geometry, the VGPR-limited occupancy.
///
/// `rsrc1` is the original `compute_pgm_rsrc1` from the descriptor (NOT Svod's
/// dispatch-patched copy). `lds_bytes`/`scratch_bytes` are the descriptor's exact
/// segment sizes. `wave_size` is 32 or 64; `target_major` is the gfx generation.
pub fn decode_resources(
    rsrc1: u32,
    lds_bytes: u32,
    scratch_bytes: u32,
    wave_size: u32,
    target_major: u32,
) -> KernelResources {
    // VGPRs are allocated/encoded in blocks: 8 per block in wave32, 4 in wave64
    // (GFX10+). Confirmed on RDNA3.5: field 24 → (24+1)*8 = 200 VGPRs.
    let vgpr_block = if wave_size == 32 { 8 } else { 4 };
    let vgprs = (((rsrc1 >> RSRC1_VGPR_SHIFT) & RSRC1_VGPR_MASK) + 1) * vgpr_block;
    // SGPRs are encoded in blocks of 16. Informational on GFX10+, where the HW
    // allocates a fixed pool, so this is reported but never an occupancy limiter.
    let sgprs = (((rsrc1 >> RSRC1_SGPR_SHIFT) & RSRC1_SGPR_MASK) + 1) * 16;
    let occupancy = vgpr_limited_occupancy(vgprs, wave_size, target_major);
    KernelResources {
        vgprs: Some(vgprs),
        sgprs: Some(sgprs),
        lds_bytes,
        scratch_bytes: Some(scratch_bytes),
        wave_size,
        occupancy,
    }
}

/// VGPR-limited occupancy as a fraction of the SIMD's max resident waves, for
/// architectures with known register-file geometry (`None` otherwise). This is
/// the first-order limiter for compute-bound kernels; LDS and workgroup limits
/// are not modeled.
fn vgpr_limited_occupancy(vgprs: u32, wave_size: u32, target_major: u32) -> Option<f32> {
    // (vgpr_file_per_simd, max_waves_per_simd, vgpr_alloc_granule)
    let (file, max_waves, granule) = match (target_major, wave_size) {
        // RDNA3 / RDNA3.5 (gfx11.x), wave32: 1536 VGPRs/SIMD, 16 waves/SIMD.
        (11, 32) => (1536u32, 16u32, 16u32),
        _ => return None,
    };
    let alloc = vgprs.max(1).div_ceil(granule) * granule;
    let waves = (file / alloc).min(max_waves);
    Some(waves as f32 / max_waves as f32)
}
