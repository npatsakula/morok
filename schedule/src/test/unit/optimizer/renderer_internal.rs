use super::*;
use svod_ir::RendererDevice;

/// `Renderer::cpu()` is the runtime CPU target (threads, no shared memory);
/// `tinygrad_base_cpu` is the reference `Renderer()` the parity tests compare against.
#[test]
fn test_cpu_renderers_are_distinct_targets() {
    let runtime = Renderer::cpu();
    assert_eq!(runtime.device, RendererDevice::Cpu);
    assert!(!runtime.has_local && !runtime.has_shared && runtime.has_threads);
    assert!(runtime.tensor_cores.is_empty());

    let reference = Renderer::tinygrad_base_cpu();
    assert!(reference.has_local && reference.has_shared && !reference.has_threads);
    assert_eq!(reference.shared_max, 32768);
    assert_eq!(reference.global_max, Some(vec![0x8fff_ffff; 3]));
    assert_eq!(reference.local_max, Some(0x8fff_ffff));
}

#[test]
fn test_renderer_cuda() {
    let r = Renderer::cuda();
    assert_eq!(r.device, RendererDevice::CudaSm80); // Default is SM80/Ampere
    assert!(r.has_local && r.has_shared && !r.has_threads);
    assert!(r.shared_max > 0);
    assert!(!r.tensor_cores.is_empty());
}

#[test]
fn test_for_amd_arch_maps_each_family() {
    use svod_dtype::AmdArch;
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx942).device, RendererDevice::AmdCdna3);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx950).device, RendererDevice::AmdCdna4);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1100).device, RendererDevice::AmdRdna3);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1151).device, RendererDevice::AmdRdna3);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1201).device, RendererDevice::AmdRdna4);
}

/// The fingerprint keys the compilation caches, so it must move with anything that
/// changes generated code — including two archs that share a `RendererDevice`.
#[test]
fn test_renderer_fingerprint_tracks_exact_target_and_capabilities() {
    use svod_dtype::AmdArch;

    let gfx1151 = Renderer::for_amd_arch(AmdArch::Gfx1151);
    assert_ne!(Renderer::for_amd_arch(AmdArch::Gfx1100).cache_fingerprint(), gfx1151.cache_fingerprint());

    let mut constrained = gfx1151.clone();
    constrained.upcast_max -= 1;
    assert_ne!(gfx1151.cache_fingerprint(), constrained.cache_fingerprint());

    constrained = gfx1151.clone();
    constrained.tensor_cores.clear();
    assert_ne!(gfx1151.cache_fingerprint(), constrained.cache_fingerprint());

    let all_ops = gfx1151.clone().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
    let mut fewer_ops = svod_ir::RendererOps::all();
    fewer_ops.binary.remove(&svod_ir::BinaryOp::Threefry);
    assert_ne!(
        all_ops.cache_fingerprint(),
        gfx1151.with_rewrite_capabilities(fewer_ops, None, None).cache_fingerprint()
    );
}

/// CDNA renders OCP FP8 natively but not the FNUZ encodings; RDNA renders neither.
#[test]
fn test_amd_fp8_dtype_capabilities_are_arch_specific() {
    use svod_dtype::{AmdArch, ScalarDType};

    for arch in [AmdArch::Gfx942, AmdArch::Gfx950] {
        let renderer = Renderer::for_amd_arch(arch);
        for dtype in [ScalarDType::FP8E4M3, ScalarDType::FP8E5M2] {
            assert!(renderer.supports_storage_dtype(dtype), "{arch} must keep OCP {dtype:?} storage");
            assert!(renderer.supports_conversion_dtype(dtype), "{arch} must keep OCP {dtype:?} conversion");
            assert!(renderer.supports_matrix_dtype(dtype), "{arch} must keep {dtype:?} matrix operands");
            assert!(!renderer.supports_alu_dtype(dtype), "{arch} must widen ordinary {dtype:?} ALU");
        }
        for dtype in [ScalarDType::FP8E4M3FNUZ, ScalarDType::FP8E5M2FNUZ] {
            assert!(!renderer.supports_dtype(dtype), "{arch} must decompose {dtype:?}");
        }
    }

    for arch in [AmdArch::Gfx1151, AmdArch::Gfx1201] {
        let renderer = Renderer::for_amd_arch(arch);
        for dtype in [ScalarDType::FP8E4M3, ScalarDType::FP8E5M2, ScalarDType::FP8E4M3FNUZ, ScalarDType::FP8E5M2FNUZ] {
            assert!(!renderer.supports_dtype(dtype), "{arch} must decompose {dtype:?} to f16");
        }
    }
}

#[test]
fn test_amd_tensor_core_tables_match_architecture() {
    use svod_dtype::AmdArch;

    // tinygrad `tc.py:132`: amd_cdna3 = amd_cdna_161632[:2] + amd_cdna_161616 -- four
    // cores, no fp32 input (the rate-neutral `v_mfma_f32_16x16x4_f32` is not offered).
    let gfx942 = Renderer::for_amd_arch(AmdArch::Gfx942);
    assert_eq!(gfx942.tensor_cores.len(), 4);
    assert!(gfx942.tensor_cores.iter().any(|tc| tc.dims == (16, 16, 32) && tc.dtype_in == DType::FP8E4M3));
    assert!(!gfx942.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Float32));

    let gfx950 = Renderer::for_amd_arch(AmdArch::Gfx950);
    assert_eq!(gfx950.tensor_cores.len(), 8);
    assert!(gfx950.tensor_cores.iter().any(|tc| tc.dims == (16, 16, 128) && tc.dtype_in == DType::FP8E4M3));

    let gfx1151 = Renderer::for_amd_arch(AmdArch::Gfx1151);
    assert_eq!(gfx1151.tensor_cores.len(), 4);
    assert!(!gfx1151.tensor_cores.iter().any(|tc| tc.dtype_in.scalar_dtype().is_fp8()));
    assert!(gfx1151.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Int8 && tc.dtype_out == DType::Int32));

    let gfx1201 = Renderer::for_amd_arch(AmdArch::Gfx1201);
    assert_eq!(gfx1201.tensor_cores.len(), 4);
    assert!(!gfx1201.tensor_cores.iter().any(|tc| tc.dtype_in.scalar_dtype().is_fp8()));
    assert!(gfx1201.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Float16 && tc.dtype_out == DType::Float32));
    assert!(gfx1201.tensor_cores.iter().any(|tc| tc.dtype_in == DType::BFloat16 && tc.dtype_out == DType::BFloat16));

    let cuda = CUDA_81616.build(DType::Float16, DType::Float32);
    assert_eq!((cuda.dims, cuda.threads), ((8, 16, 16), 32));
    assert!(!cuda.opts.is_empty());
}

/// Only targets with a per-axis local limit report one.
#[test]
fn test_local_max_axes_match_renderer_capabilities() {
    assert_eq!(Renderer::cuda().local_max_axes(), Some([1024, 1024, 64]));
    assert_eq!(Renderer::webgpu().local_max_axes(), Some([256, 256, 64]));
    assert_eq!(Renderer::amd_cdna3().local_max_axes(), None);
    assert_eq!(Renderer::cpu().local_max_axes(), None);
    assert_eq!(Renderer::metal().local_max_axes(), None);
}

/// Tensor cores follow the Apple GPU family: none below Apple7 or on Intel-Mac
/// GPUs, and the family is part of the profile's identity.
#[test]
fn test_metal_profile_follows_gpu_family() {
    use svod_dtype::MetalFamily;
    let m4 = Renderer::for_metal_family(MetalFamily::Apple(9));
    assert_eq!(m4.tensor_cores.len(), Renderer::metal().tensor_cores.len());
    let m1 = Renderer::for_metal_family(MetalFamily::Apple(7));
    assert_eq!(m1.tensor_cores.len(), 5);
    for family in [MetalFamily::Apple(6), MetalFamily::Mac2, MetalFamily::Unknown] {
        assert!(Renderer::for_metal_family(family).tensor_cores.is_empty(), "{family}");
    }
    assert_ne!(m4.cache_fingerprint(), m1.cache_fingerprint());
    assert_ne!(m4.cache_fingerprint(), Renderer::metal().cache_fingerprint());
    assert_eq!(Renderer::for_metal_family(MetalFamily::Apple(9)).cache_fingerprint(), m4.cache_fingerprint());
}
