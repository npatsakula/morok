//! CPU device implementation with selectable JIT backends.
//!
//! This module provides a Device instance for CPU execution using either:
//! - LLVM JIT (maximum optimization, slower compilation)
//! - Cranelift JIT (faster compilation, good-enough optimization)
//!
//! The backend can be selected via:
//! - `MOROK_CPU_BACKEND` environment variable ("llvm" or "cranelift")
//! - Explicit `create_cpu_device_with_backend()` call

use std::collections::HashMap;
use std::sync::Arc;

use morok_device::Result;
use morok_device::device::{Compiler, Device, Program, ProgramSpec, Renderer, RuntimeFactory};
use morok_device::registry::DeviceRegistry;
use morok_dtype::DeviceSpec;
use morok_ir::UOp;

use crate::{CompiledKernel, LlvmKernel};

/// CPU backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CpuBackend {
    /// Cranelift JIT backend.
    /// Faster compilation, good-enough codegen quality.
    Cranelift,
    /// LLVM JIT backend (default).
    /// Maximum optimization, slower compilation.
    #[default]
    Llvm,
}

impl CpuBackend {
    /// Select backend from environment variable MOROK_CPU_BACKEND.
    pub fn from_env() -> Self {
        match std::env::var("MOROK_CPU_BACKEND").as_deref() {
            Ok("llvm") | Ok("LLVM") => CpuBackend::Llvm,
            Ok("cranelift") | Ok("CRANELIFT") => CpuBackend::Cranelift,
            _ => CpuBackend::default(),
        }
    }
}

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
                .map_err(|e| morok_device::Error::Runtime { message: format!("LLVM kernel execution failed: {}", e) })
        }
    }

    fn name(&self) -> &str {
        "llvm_cpu_kernel" // TODO: Extract from LlvmKernel
    }
}

/// LLVM compiler implementing the Compiler trait.
///
/// For LLVM JIT, we validate the IR and return it as source for runtime compilation.
/// The actual JIT compilation happens in the RuntimeFactory when creating the Program.
struct LlvmCompiler;

impl Compiler for LlvmCompiler {
    fn compile(&self, spec: &morok_device::device::ProgramSpec) -> Result<morok_device::device::CompiledSpec> {
        // For LLVM JIT, we validate the IR and pass it through to the runtime
        // The RuntimeFactory will perform the actual JIT compilation

        // TODO: Add LLVM IR validation here using inkwell
        // For now, we trust that the renderer produces valid IR

        // Return CompiledSpec with source for JIT compilation
        Ok(morok_device::device::CompiledSpec::from_source(spec.name.clone(), spec.src.clone(), spec.ast.clone()))
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
    fn render(&self, ast: &Arc<UOp>) -> Result<ProgramSpec> {
        // Call the existing LLVM renderer
        let rendered = morok_codegen::llvm::render(ast, Some("kernel"))
            .map_err(|e| morok_device::Error::Runtime { message: format!("LLVM rendering failed: {}", e) })?;

        // Convert RenderedKernel to ProgramSpec
        let mut spec = ProgramSpec::new(rendered.name.clone(), rendered.code.clone(), self.device.clone(), ast.clone());

        if let Some(global) = rendered.global_size
            && let Some(local) = rendered.local_size
        {
            spec.set_work_sizes(global, local);
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
/// This creates `LlvmProgram` instances from a CompiledSpec containing source code.
/// For LLVM JIT, compilation and loading happen in this step.
fn create_llvm_program(spec: &morok_device::device::CompiledSpec) -> Result<Box<dyn Program>> {
    // Extract source code from CompiledSpec
    let src = spec.src.as_ref().ok_or_else(|| morok_device::Error::Runtime {
        message: "LLVM JIT requires source code in CompiledSpec".to_string(),
    })?;

    // Use existing LlvmKernel to JIT compile the IR
    let kernel = crate::LlvmKernel::compile_ir(src, &spec.name, &spec.name)
        .map_err(|e| morok_device::Error::Runtime { message: format!("LLVM JIT compilation failed: {}", e) })?;

    Ok(Box::new(LlvmProgram { kernel }))
}

// =============================================================================
// Cranelift Backend
// =============================================================================

use crate::cranelift::CraneliftKernel;

/// Cranelift program wrapper implementing the Program trait.
struct CraneliftProgram {
    kernel: CraneliftKernel,
}

impl Program for CraneliftProgram {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vars: &HashMap<String, i64>,
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
    ) -> Result<()> {
        // SAFETY: Caller ensures buffers are valid
        unsafe {
            self.kernel.execute(buffers, vars).map_err(|e| morok_device::Error::Runtime { message: format!("{}", e) })
        }
    }

    fn name(&self) -> &str {
        self.kernel.name()
    }
}

/// Cranelift renderer wrapper implementing the Renderer trait.
struct CraneliftRendererWrapper {
    device: DeviceSpec,
}

impl Renderer for CraneliftRendererWrapper {
    fn render(&self, ast: &Arc<UOp>) -> Result<ProgramSpec> {
        // Create the Cranelift renderer and render
        let renderer = morok_codegen::cranelift::CraneliftRenderer::new();
        let rendered = renderer
            .render(ast, Some("kernel"))
            .map_err(|e| morok_device::Error::Runtime { message: format!("Cranelift rendering failed: {}", e) })?;

        // Convert RenderedKernel to ProgramSpec
        let mut spec = ProgramSpec::new(rendered.name.clone(), rendered.code.clone(), self.device.clone(), ast.clone());

        if let Some(global) = rendered.global_size
            && let Some(local) = rendered.local_size
        {
            spec.set_work_sizes(global, local);
        }

        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }

    fn decompositor(&self) -> Option<morok_ir::pattern::PatternMatcher<()>> {
        // Delegate to the codegen CraneliftRenderer
        let renderer = morok_codegen::cranelift::CraneliftRenderer::new();
        <morok_codegen::cranelift::CraneliftRenderer as morok_codegen::Renderer>::decompositor(&renderer)
    }
}

/// Cranelift compiler - passes IR through to runtime for JIT.
struct CraneliftCompiler;

impl Compiler for CraneliftCompiler {
    fn compile(&self, spec: &ProgramSpec) -> Result<morok_device::device::CompiledSpec> {
        // For Cranelift, we pass the IR text through to runtime
        Ok(morok_device::device::CompiledSpec::from_source(spec.name.clone(), spec.src.clone(), spec.ast.clone()))
    }

    fn cache_key(&self) -> Option<&str> {
        Some("cranelift-jit")
    }
}

/// Runtime factory for creating Cranelift programs.
fn create_cranelift_program(spec: &morok_device::device::CompiledSpec) -> Result<Box<dyn Program>> {
    let src = spec.src.as_ref().ok_or_else(|| morok_device::Error::Runtime {
        message: "Cranelift JIT requires source code in CompiledSpec".to_string(),
    })?;

    let kernel = CraneliftKernel::compile(src, &spec.name)
        .map_err(|e| morok_device::Error::Runtime { message: format!("Cranelift JIT compilation failed: {}", e) })?;

    Ok(Box::new(CraneliftProgram { kernel }))
}

// =============================================================================
// Public API
// =============================================================================

/// Create a CPU device with the default backend.
///
/// The default backend is selected by:
/// 1. `MOROK_CPU_BACKEND` environment variable ("llvm" or "cranelift")
/// 2. If not set, defaults to Cranelift (faster compilation)
pub fn create_cpu_device(registry: &DeviceRegistry) -> Result<Device> {
    create_cpu_device_with_backend(registry, CpuBackend::from_env())
}

/// Create a CPU device with a specific backend.
pub fn create_cpu_device_with_backend(registry: &DeviceRegistry, backend: CpuBackend) -> Result<Device> {
    let device_spec = DeviceSpec::Cpu;
    let allocator = registry.get(&device_spec)?;

    match backend {
        CpuBackend::Cranelift => {
            let renderer = Arc::new(CraneliftRendererWrapper { device: device_spec.clone() });
            let compiler = Arc::new(CraneliftCompiler);
            let runtime: RuntimeFactory = Arc::new(create_cranelift_program);
            Ok(Device::new(device_spec, allocator, renderer, compiler, runtime))
        }
        CpuBackend::Llvm => {
            let renderer = Arc::new(LlvmRendererWrapper { device: device_spec.clone() });
            let compiler = Arc::new(LlvmCompiler);
            let runtime: RuntimeFactory = Arc::new(create_llvm_program);
            Ok(Device::new(device_spec, allocator, renderer, compiler, runtime))
        }
    }
}
