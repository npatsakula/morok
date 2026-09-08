//! Kernel count alignment tests against Tinygrad.
//!
//! Require downloading real ONNX models; `#[ignore]` by default.

use std::path::Path;

use svod_dtype::{DType, ScalarDType};
use svod_tensor::{CpuBackend, PrepareConfig, Tensor};

use crate::importer::OnnxImporter;

fn count_kernels(model_path: &Path) -> usize {
    let mut importer = OnnxImporter::new();
    let result = importer.import(model_path, &[]).unwrap();

    for (name, input_tensor) in &result.inputs {
        let shape = input_tensor.dims().unwrap_or_else(|e| panic!("input '{name}' shape: {e}"));
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let real_tensor =
            Tensor::from_raw_bytes(bytemuck::cast_slice(&data), &shape, DType::Scalar(ScalarDType::Float32))
                .unwrap_or_else(|e| panic!("input '{name}': {e}"));
        input_tensor.assign(&real_tensor);
    }

    let config = PrepareConfig::for_cpu_backend(CpuBackend::Clang);
    let outputs: Vec<Tensor> = result.outputs.values().cloned().collect();
    let plan = Tensor::prepare_batch_with(outputs.iter(), &config).unwrap();
    plan.prepared_kernels().len()
}

/// Tinygrad produces 58 kernels (1 is input preprocessing).
/// Svod: 57 (input assigned directly, no preprocessing kernel).
///
/// ```sh
/// curl -L -o /tmp/inception-v2-9.onnx \
///   https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx
/// ```
#[test]
#[ignore = "requires /tmp/inception-v2-9.onnx (45MB)"]
fn test_inception_v2_kernel_count() {
    let path = Path::new("/tmp/inception-v2-9.onnx");
    assert!(path.exists(), "{} not found", path.display());

    let n = count_kernels(path);
    assert!((57..=58).contains(&n), "kernel count {n}, expected 57-58 (tinygrad: 58)");
}
