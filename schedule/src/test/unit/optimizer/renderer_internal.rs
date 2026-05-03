use super::*;

#[test]
fn test_renderer_cpu() {
    let r = Renderer::cpu();
    assert_eq!(r.device, "CPU");
    assert!(!r.has_local);
    assert!(r.has_threads);
    assert_eq!(r.tensor_cores.len(), 0);
}

#[test]
fn test_renderer_cuda() {
    let r = Renderer::cuda();
    assert_eq!(r.device, "CUDA_SM80"); // Default is SM80/Ampere
    assert!(r.has_local);
    assert!(r.has_shared);
    assert!(!r.has_threads);
    assert!(r.shared_max > 0);
    assert!(!r.tensor_cores.is_empty());
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
