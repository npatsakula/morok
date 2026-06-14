use super::*;
use svod_ir::RendererDevice;

#[test]
fn test_renderer_cpu() {
    let r = Renderer::cpu();
    assert_eq!(r.device, RendererDevice::Cpu);
    assert!(!r.has_local);
    assert!(r.has_threads);
    assert_eq!(r.tensor_cores.len(), 0);
}

#[test]
fn test_renderer_cuda() {
    let r = Renderer::cuda();
    assert_eq!(r.device, RendererDevice::CudaSm80); // Default is SM80/Ampere
    assert!(r.has_local);
    assert!(r.has_shared);
    assert!(!r.has_threads);
    assert!(r.shared_max > 0);
    assert!(!r.tensor_cores.is_empty());
}

#[test]
fn test_for_amd_arch_maps_each_family() {
    use svod_dtype::AmdArch;
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx942).device, RendererDevice::AmdCdna3);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx950).device, RendererDevice::AmdCdna4);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1030).device, RendererDevice::AmdRdna2);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1034).device, RendererDevice::AmdRdna2);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1100).device, RendererDevice::AmdRdna3);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1151).device, RendererDevice::AmdRdna3);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1201).device, RendererDevice::AmdRdna4);
}

/// Load-bearing regression guard for the whole RDNA2 feature: the RDNA2 profile
/// MUST carry no tensor cores. If this ever becomes non-empty, the optimizer
/// will emit WMMA UOps that gfx10 hardware can't execute (it has no matrix
/// cores) and matmul will fail to compile/run. Every RDNA2 die must route here.
#[test]
fn test_rdna2_profile_has_no_tensor_cores() {
    use svod_dtype::AmdArch;
    for arch in [AmdArch::Gfx1030, AmdArch::Gfx1031, AmdArch::Gfx1032, AmdArch::Gfx1034] {
        let r = Renderer::for_amd_arch(arch);
        assert_eq!(r.device, RendererDevice::AmdRdna2, "{arch:?} must use the RDNA2 profile");
        assert!(r.tensor_cores.is_empty(), "{arch:?} must have no tensor cores (gfx10 has no WMMA)");
    }
    // Contrast: RDNA3 (the profile RDNA2 would wrongly fall through to) HAS them.
    assert!(!Renderer::for_amd_arch(AmdArch::Gfx1100).tensor_cores.is_empty());
}

#[test]
fn test_local_max_axes_amd_caps_z_at_64() {
    use svod_dtype::AmdArch;
    // AMD/HIP caps the 3rd local axis at 64 below the 1024-thread product limit
    // (tinygrad parity); other backends rely on the product cap alone.
    for arch in
        [AmdArch::Gfx1030, AmdArch::Gfx1151, AmdArch::Gfx1100, AmdArch::Gfx1201, AmdArch::Gfx942, AmdArch::Gfx950]
    {
        assert_eq!(Renderer::for_amd_arch(arch).local_max_axes(), Some([1024, 1024, 64]), "{arch:?}");
    }
    assert_eq!(Renderer::cpu().local_max_axes(), None);
    assert_eq!(Renderer::cuda().local_max_axes(), None);
    assert_eq!(Renderer::metal().local_max_axes(), None);
}

#[test]
fn test_tensor_core_cuda() {
    let tc = CUDA_81616.build(DType::Float16, DType::Float32);
    assert_eq!(tc.dims, (8, 16, 16));
    assert_eq!(tc.threads, 32);
    assert_eq!(tc.dtype_in, DType::Float16);
    assert_eq!(tc.dtype_out, DType::Float32);
    assert!(!tc.opts.is_empty());
}
