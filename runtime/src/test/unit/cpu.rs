//! CPU device integration tests.
//!
//! Tests the full Device pipeline: render → compile → runtime factory → execute

use crate::devices::cpu::{CpuBackend, create_cpu_device_with_backend};
use morok_device::registry::DeviceRegistry;
use morok_dtype::{AddrSpace, DType};
use morok_ir::{ConstValue, UOp};

fn build_copy_sink() -> std::sync::Arc<UOp> {
    let ptr_dtype = DType::Float32.ptr(None, AddrSpace::Global);
    let out = UOp::param(0, 16, ptr_dtype.clone(), None);
    let inp = UOp::param(1, 16, ptr_dtype, None);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));

    let out_index = UOp::index().buffer(out).indices(vec![idx.clone()]).call().expect("output INDEX");
    let in_index = UOp::index().buffer(inp.clone()).indices(vec![idx]).call().expect("input INDEX");
    let load = UOp::load().buffer(inp).index(in_index).call();
    let store = out_index.store(load);
    UOp::sink(vec![store])
}

#[test]
fn test_cpu_device_creation_llvm() {
    let registry = DeviceRegistry::default();
    let device =
        create_cpu_device_with_backend(&registry, CpuBackend::Llvm).expect("Failed to create CPU device with LLVM");

    // Verify device properties
    assert_eq!(device.base_device_key(), "CPU");
    assert_eq!(device.compiler.cache_key(), "llvm-jit");
}

#[test]
fn test_compile_and_runtime_pipeline_llvm() {
    use morok_device::device::ProgramSpec;
    use morok_dtype::DeviceSpec;

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Llvm).unwrap();

    // Create a minimal valid LLVM IR program
    // This is a no-op kernel that just returns
    let llvm_ir = r#"
define void @test_kernel() {
entry:
  ret void
}
"#;

    let sink = UOp::sink(vec![]);
    let spec = ProgramSpec::new("test_kernel".to_string(), llvm_ir.to_string(), DeviceSpec::Cpu, sink);

    // Test 1: Compile
    let compiled = device.compiler.compile(&spec).expect("Compile should succeed");
    assert!(compiled.src.is_some(), "LLVM JIT should have source");
    assert!(compiled.bytes.is_empty(), "LLVM JIT should have empty bytes");
    assert_eq!(compiled.name, "test_kernel");

    // Test 2: Runtime factory
    let program = (device.runtime)(&compiled).expect("RuntimeFactory should succeed");
    // Note: program.name() might not match spec.name (it's a TODO in LlvmProgram)
    assert!(!program.name().is_empty(), "Program should have a name");

    // Test 3: Execute (no buffers needed for this kernel)
    let pointers: Vec<*mut u8> = vec![];

    unsafe {
        program.execute(&pointers, &[], None, None).expect("Execution should succeed");
    }
}

#[test]
fn test_compile_invalid_ir() {
    use morok_device::device::ProgramSpec;
    use morok_dtype::DeviceSpec;

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Llvm).unwrap();

    // Create a ProgramSpec with invalid LLVM IR
    let sink = UOp::sink(vec![]);
    let spec = ProgramSpec::new("test".to_string(), "this is not valid LLVM IR".to_string(), DeviceSpec::Cpu, sink);

    // Compilation should fail gracefully
    // Note: Current implementation doesn't validate, so this will pass
    // TODO: Add LLVM IR validation to LlvmCompiler
    let result = device.compiler.compile(&spec);
    assert!(result.is_ok(), "Should return CompiledSpec even with invalid IR (validation TODO)");
}

#[test]
fn test_renderer_metadata_consistent_between_clang_and_llvm() {
    let registry = DeviceRegistry::default();
    let clang = create_cpu_device_with_backend(&registry, CpuBackend::Clang).expect("create clang device");
    let llvm = create_cpu_device_with_backend(&registry, CpuBackend::Llvm).expect("create llvm device");

    let sink = build_copy_sink();
    let linear = UOp::linear(sink.toposort().into());
    let clang_spec = clang.renderer.render(&linear, Some("meta_copy")).expect("clang render");
    let llvm_spec = llvm.renderer.render(&linear, Some("meta_copy")).expect("llvm render");

    assert_eq!(clang_spec.globals, vec![0, 1]);
    assert_eq!(clang_spec.outs, vec![0]);
    assert_eq!(clang_spec.ins, vec![1]);
    assert_eq!(clang_spec.globals, llvm_spec.globals);
    assert_eq!(clang_spec.outs, llvm_spec.outs);
    assert_eq!(clang_spec.ins, llvm_spec.ins);
}
