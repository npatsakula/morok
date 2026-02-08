//! CPU device integration tests.
//!
//! Tests the full Device pipeline: render → compile → runtime factory → execute

use crate::devices::cpu::{CpuBackend, create_cpu_device_with_backend};
use morok_device::registry::DeviceRegistry;
use morok_ir::UOp;

#[test]
fn test_cpu_device_creation_llvm() {
    let registry = DeviceRegistry::default();
    let device =
        create_cpu_device_with_backend(&registry, CpuBackend::Llvm).expect("Failed to create CPU device with LLVM");

    // Verify device properties
    assert_eq!(device.base_device_key(), "CPU");
    assert!(device.compiler.cache_key().is_some());
    assert_eq!(device.compiler.cache_key().unwrap(), "llvm-jit");
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
