//! CPU device implementation using LLVM JIT.
//!
//! This module provides a Device instance for CPU execution using LLVM as the
//! compiler backend. It wraps the existing `morok_runtime::LlvmKernel` to implement
//! the new device abstraction traits.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use morok_device::device::{Compiler, Device, Program, ProgramSpec, Renderer, RuntimeFactory};
use morok_device::registry::DeviceRegistry;
use morok_device::Result;
use morok_dtype::DeviceSpec;
use morok_ir::UOp;

use crate::{CompiledKernel, LlvmKernel};

/// LLVM program wrapper implementing the Program trait.
///
/// This wraps `LlvmKernel` to provide a unified execution interface.
struct LlvmProgram {
    kernel: LlvmKernel,
}

impl Program for LlvmProgram {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vars: &HashMap<String, i64>,
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
    ) -> Result<()> {
        // CPU execution ignores global/local size (those are for GPU kernels)
        unsafe {
            CompiledKernel::execute_with_vars(&self.kernel, buffers, vars)
                .map_err(|e| morok_device::Error::Runtime {
                    message: format!("LLVM kernel execution failed: {}", e),
                })
        }
    }

    fn name(&self) -> &str {
        "llvm_cpu_kernel" // TODO: Extract from LlvmKernel
    }
}

/// LLVM compiler implementing the Compiler trait.
///
/// This wraps `morok_runtime::LlvmKernel::compile_ir` to provide a unified compilation interface.
struct LlvmCompiler;

impl Compiler for LlvmCompiler {
    fn compile(&self, _src: &str) -> Result<Vec<u8>> {
        // For LLVM JIT, we don't actually need to return bytes since compilation
        // and execution happen in one step. We return empty bytes as a placeholder.
        // The actual compilation happens in the RuntimeFactory.
        //
        // TODO: In the future, we might want to serialize the LLVM module here
        // for caching purposes.
        Ok(Vec::new())
    }

    fn cache_key(&self) -> Option<&str> {
        Some("llvm-jit")
    }
}

/// LLVM renderer wrapper implementing the Renderer trait.
///
/// This wraps `morok_codegen::llvm::render` to provide a unified rendering interface.
struct LlvmRendererWrapper {
    device: DeviceSpec,
}

impl Renderer for LlvmRendererWrapper {
    fn render(&self, ast: &Rc<UOp>) -> Result<ProgramSpec> {
        // Call the existing LLVM renderer
        let rendered = morok_codegen::llvm::render(ast, Some("kernel")).map_err(|e| {
            morok_device::Error::Runtime { message: format!("LLVM rendering failed: {}", e) }
        })?;

        // Convert RenderedKernel to ProgramSpec
        let mut spec = ProgramSpec::new(
            rendered.name.clone(),
            rendered.code.clone(),
            self.device.clone(),
            ast.clone(),
        );

        if let Some(global) = rendered.global_size {
            if let Some(local) = rendered.local_size {
                spec.set_work_sizes(global, local);
            }
        }

        // TODO: Extract variables from the kernel signature
        // For now, we don't populate vars since the existing LLVM renderer
        // doesn't extract them.

        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }
}

/// Runtime factory for creating LLVM programs.
///
/// This creates `LlvmProgram` instances from source code.
/// For LLVM JIT, compilation and loading happen in one step.
fn create_llvm_program(entry_point: &str, compiled_bytes: &[u8]) -> Result<Box<dyn Program>> {
    // For LLVM JIT, the compiled_bytes are empty (placeholder).
    // We need the source code to compile, but it's not passed here.
    //
    // This is a design issue: the RuntimeFactory signature assumes
    // separate compilation and loading steps, but LLVM JIT does both at once.
    //
    // WORKAROUND: For now, we'll need to change the flow. Instead of:
    //   1. render() -> source
    //   2. compile() -> bytes
    //   3. runtime() -> program
    //
    // We'll do:
    //   1. render() -> source
    //   2. compile() returns empty (no-op)
    //   3. runtime() compiles directly from source (passed via a hack)
    //
    // TODO: Refactor to pass source code through properly, or store it in
    // the CompiledRunner alongside the bytes.

    // TEMPORARY HACK: We can't actually create a program here without the source.
    // This will be fixed when we implement CompiledRunner properly.
    let _ = (entry_point, compiled_bytes);
    Err(morok_device::Error::Runtime {
        message: "LLVM runtime factory not fully implemented yet - needs source code".to_string(),
    })
}

/// Create a CPU device using LLVM as the backend.
///
/// This is the main entry point for creating CPU devices.
/// It wires up the LLVM renderer, compiler, and runtime into a Device instance.
pub fn create_cpu_device(registry: &DeviceRegistry) -> Result<Device> {
    let device_spec = DeviceSpec::Cpu;

    // Get the CPU allocator from the registry
    let allocator = registry.get(&device_spec)?;

    // Create the LLVM renderer
    let renderer = Arc::new(LlvmRendererWrapper { device: device_spec.clone() });

    // Create the LLVM compiler
    let compiler = Arc::new(LlvmCompiler);

    // Create the runtime factory
    let runtime: RuntimeFactory = Arc::new(|entry_point, bytes| create_llvm_program(entry_point, bytes));

    Ok(Device::new(device_spec, allocator, renderer, compiler, runtime))
}
