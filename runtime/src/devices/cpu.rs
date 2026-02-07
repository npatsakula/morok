//! CPU device implementation with selectable JIT backends.
//!
//! This module provides a Device instance for CPU execution using either:
//! - Clang C codegen (default, human-readable, fast debug cycles)
//! - LLVM JIT (maximum optimization, slower compilation)
//!
//! The backend can be selected via:
//! - `MOROK_CPU_BACKEND` environment variable ("clang" or "llvm")
//! - Explicit `create_cpu_device_with_backend()` call

use std::sync::Arc;

use morok_device::Result;
use morok_device::device::{Compiler, Device, Program, ProgramSpec, Renderer, RuntimeFactory};
use morok_device::registry::DeviceRegistry;
use morok_dtype::DeviceSpec;
use morok_ir::UOp;

use crate::LlvmKernel;
use crate::clang::ClangKernel;

/// CPU backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CpuBackend {
    /// Clang C codegen backend (default).
    /// Generates C source, compiles with clang, loads via dlopen.
    #[default]
    Clang,
    /// LLVM JIT backend.
    /// Maximum optimization, slower compilation.
    Llvm,
    /// MLIR backend.
    /// Generates MLIR, lowers to LLVM IR, then JIT compiles.
    #[cfg(feature = "mlir")]
    Mlir,
}

impl CpuBackend {
    /// Select backend from environment variable MOROK_CPU_BACKEND.
    pub fn from_env() -> Self {
        match std::env::var("MOROK_CPU_BACKEND").as_deref() {
            Ok("clang") | Ok("CLANG") => CpuBackend::Clang,
            Ok("llvm") | Ok("LLVM") => CpuBackend::Llvm,
            #[cfg(feature = "mlir")]
            Ok("mlir") | Ok("MLIR") => CpuBackend::Mlir,
            _ => CpuBackend::default(),
        }
    }
}

// =============================================================================
// Shared parallel execution
// =============================================================================

/// Execute a kernel function pointer in parallel across multiple threads.
///
/// # Safety
///
/// Buffer safety is guaranteed by the shift_to() transformation:
/// - Each thread_id maps to disjoint output indices
/// - Index formula: `output[thread_id * chunk_size + local_idx]`
///
/// Same buffer pointers can be safely passed to all threads because:
/// 1. Input buffers: Read-only access (no data race)
/// 2. Output buffers: Disjoint write regions per thread
unsafe fn execute_parallel(
    fn_ptr: usize,
    buffers: &[*mut u8],
    vals: &[i64],
    var_names: &[String],
    thread_count: usize,
) -> Result<()> {
    use rayon::prelude::*;

    let thread_id_idx = var_names.iter().position(|n| n == "thread_id");
    let buffer_usizes: Vec<usize> = buffers.iter().map(|&ptr| ptr as usize).collect();

    type KernelFn = unsafe extern "C" fn(*const *mut u8, *const i64);

    (0..thread_count).into_par_iter().for_each(|thread_id| {
        let mut thread_vals = vals.to_vec();
        if let Some(idx) = thread_id_idx {
            thread_vals[idx] = thread_id as i64;
        }

        let bufs_ptr = buffer_usizes.as_ptr() as *const *mut u8;

        unsafe {
            let f: KernelFn = std::mem::transmute(fn_ptr);
            f(bufs_ptr, thread_vals.as_ptr());
        }
    });

    Ok(())
}

// =============================================================================
// Clang Backend
// =============================================================================

/// Clang program wrapper implementing the Program trait.
struct ClangProgram {
    kernel: ClangKernel,
}

impl Program for ClangProgram {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
    ) -> Result<()> {
        let thread_count = global_size.map(|[tc, _, _]| tc).filter(|&tc| tc > 1);

        if let Some(count) = thread_count {
            unsafe { execute_parallel(self.kernel.fn_ptr() as usize, buffers, vals, self.kernel.var_names(), count) }
        } else {
            unsafe { self.kernel.execute_with_vals(buffers, vals) }
                .map_err(|e| morok_device::Error::Runtime { message: format!("Clang kernel execution failed: {}", e) })
        }
    }

    fn name(&self) -> &str {
        self.kernel.name()
    }
}

/// Clang renderer wrapper implementing the Renderer trait.
struct ClangRendererWrapper {
    device: DeviceSpec,
}

impl Renderer for ClangRendererWrapper {
    fn render(&self, ast: &Arc<UOp>) -> Result<ProgramSpec> {
        let rendered = morok_codegen::c::render(ast, Some("kernel"))
            .map_err(|e| morok_device::Error::Runtime { message: format!("C rendering failed: {}", e) })?;

        let mut spec = ProgramSpec::new(rendered.name.clone(), rendered.code.clone(), self.device.clone(), ast.clone());

        if let Some(global) = rendered.global_size
            && let Some(local) = rendered.local_size
        {
            spec.set_work_sizes(global, local);
        }

        spec.set_var_names(rendered.var_names.clone());

        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }
}

/// Clang compiler - passes C source through for clang compilation.
struct ClangCompiler;

impl Compiler for ClangCompiler {
    fn compile(&self, spec: &ProgramSpec) -> Result<morok_device::device::CompiledSpec> {
        let mut compiled =
            morok_device::device::CompiledSpec::from_source(spec.name.clone(), spec.src.clone(), spec.ast.clone());
        compiled.var_names = spec.var_names.clone();
        compiled.global_size = spec.global_size;
        compiled.local_size = spec.local_size;
        Ok(compiled)
    }

    fn cache_key(&self) -> Option<&str> {
        Some("clang")
    }
}

/// Runtime factory for creating Clang programs.
fn create_clang_program(spec: &morok_device::device::CompiledSpec) -> Result<Box<dyn Program>> {
    let src = spec.src.as_ref().ok_or_else(|| morok_device::Error::Runtime {
        message: "Clang backend requires source code in CompiledSpec".to_string(),
    })?;

    let kernel = ClangKernel::compile(src, &spec.name, spec.var_names.clone())
        .map_err(|e| morok_device::Error::Runtime { message: format!("Clang compilation failed: {}", e) })?;

    Ok(Box::new(ClangProgram { kernel }))
}

// =============================================================================
// LLVM Backend
// =============================================================================

/// LLVM program wrapper implementing the Program trait.
struct LlvmProgram {
    kernel: LlvmKernel,
}

impl Program for LlvmProgram {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
    ) -> Result<()> {
        let thread_count = global_size.map(|[tc, _, _]| tc).filter(|&tc| tc > 1);

        if let Some(count) = thread_count {
            unsafe { execute_parallel(self.kernel.fn_ptr() as usize, buffers, vals, self.kernel.var_names(), count) }
        } else {
            unsafe { self.kernel.execute_with_vals(buffers, vals) }
                .map_err(|e| morok_device::Error::Runtime { message: format!("LLVM kernel execution failed: {}", e) })
        }
    }

    fn name(&self) -> &str {
        self.kernel.name()
    }
}

/// LLVM compiler implementing the Compiler trait.
struct LlvmCompiler;

impl Compiler for LlvmCompiler {
    fn compile(&self, spec: &morok_device::device::ProgramSpec) -> Result<morok_device::device::CompiledSpec> {
        let mut compiled =
            morok_device::device::CompiledSpec::from_source(spec.name.clone(), spec.src.clone(), spec.ast.clone());
        compiled.var_names = spec.var_names.clone();
        compiled.global_size = spec.global_size;
        compiled.local_size = spec.local_size;
        Ok(compiled)
    }

    fn cache_key(&self) -> Option<&str> {
        Some("llvm-jit")
    }
}

/// LLVM renderer wrapper implementing the Renderer trait.
struct LlvmRendererWrapper {
    device: DeviceSpec,
}

impl Renderer for LlvmRendererWrapper {
    fn render(&self, ast: &Arc<UOp>) -> Result<ProgramSpec> {
        let rendered = morok_codegen::llvm::text::render(ast, Some("kernel"))
            .map_err(|e| morok_device::Error::Runtime { message: format!("LLVM rendering failed: {}", e) })?;

        let mut spec = ProgramSpec::new(rendered.name.clone(), rendered.code.clone(), self.device.clone(), ast.clone());

        if let Some(global) = rendered.global_size
            && let Some(local) = rendered.local_size
        {
            spec.set_work_sizes(global, local);
        }

        spec.set_var_names(rendered.var_names.clone());

        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }
}

/// Runtime factory for creating LLVM programs.
fn create_llvm_program(spec: &morok_device::device::CompiledSpec) -> Result<Box<dyn Program>> {
    let src = spec.src.as_ref().ok_or_else(|| morok_device::Error::Runtime {
        message: "LLVM JIT requires source code in CompiledSpec".to_string(),
    })?;

    let kernel = crate::LlvmKernel::compile_ir(src, &spec.name, &spec.name, spec.var_names.clone())
        .map_err(|e| morok_device::Error::Runtime { message: format!("LLVM JIT compilation failed: {}", e) })?;

    Ok(Box::new(LlvmProgram { kernel }))
}

// =============================================================================
// MLIR Backend
// =============================================================================

#[cfg(feature = "mlir")]
mod mlir_backend {
    use super::*;

    /// MLIR program wrapper using ExecutionEngine.
    pub struct MlirProgram {
        pub kernel: crate::mlir::MlirKernel,
    }

    impl Program for MlirProgram {
        unsafe fn execute(
            &self,
            buffers: &[*mut u8],
            vals: &[i64],
            global_size: Option<[usize; 3]>,
            _local_size: Option<[usize; 3]>,
        ) -> Result<()> {
            let thread_count = global_size.map(|[tc, _, _]| tc).filter(|&tc| tc > 1);

            if let Some(count) = thread_count {
                unsafe {
                    execute_parallel(self.kernel.fn_ptr() as usize, buffers, vals, self.kernel.var_names(), count)
                }
            } else {
                unsafe { self.kernel.execute_with_vals(buffers, vals) }.map_err(|e| morok_device::Error::Runtime {
                    message: format!("MLIR kernel execution failed: {}", e),
                })
            }
        }

        fn name(&self) -> &str {
            self.kernel.name()
        }
    }

    /// MLIR renderer wrapper implementing the Renderer trait.
    pub struct MlirRendererWrapper {
        pub device: DeviceSpec,
    }

    impl Renderer for MlirRendererWrapper {
        fn render(&self, ast: &Arc<UOp>) -> Result<ProgramSpec> {
            let rendered = morok_codegen::mlir::render(ast, Some("kernel"))
                .map_err(|e| morok_device::Error::Runtime { message: format!("MLIR rendering failed: {}", e) })?;

            let mut spec =
                ProgramSpec::new(rendered.name.clone(), rendered.code.clone(), self.device.clone(), ast.clone());

            if let Some(global) = rendered.global_size
                && let Some(local) = rendered.local_size
            {
                spec.set_work_sizes(global, local);
            }

            spec.set_var_names(rendered.var_names.clone());

            Ok(spec)
        }

        fn device(&self) -> &DeviceSpec {
            &self.device
        }

        fn decompositor(&self) -> Option<morok_ir::pattern::TypedPatternMatcher<()>> {
            use morok_ir::decompositions::ptrcat_decomposition_patterns;
            Some(ptrcat_decomposition_patterns())
        }
    }

    /// MLIR compiler implementing the Compiler trait.
    pub struct MlirCompiler;

    impl Compiler for MlirCompiler {
        fn compile(&self, spec: &morok_device::device::ProgramSpec) -> Result<morok_device::device::CompiledSpec> {
            let mut compiled =
                morok_device::device::CompiledSpec::from_source(spec.name.clone(), spec.src.clone(), spec.ast.clone());
            compiled.var_names = spec.var_names.clone();
            compiled.global_size = spec.global_size;
            compiled.local_size = spec.local_size;
            Ok(compiled)
        }

        fn cache_key(&self) -> Option<&str> {
            Some("mlir-exec-engine")
        }
    }

    /// Runtime factory for creating MLIR programs.
    pub fn create_mlir_program(spec: &morok_device::device::CompiledSpec) -> Result<Box<dyn Program>> {
        let src = spec.src.as_ref().ok_or_else(|| morok_device::Error::Runtime {
            message: "MLIR backend requires source code (MLIR text) in CompiledSpec".to_string(),
        })?;

        let kernel = crate::mlir::MlirKernel::compile(src, &spec.name, spec.var_names.clone()).map_err(|e| {
            morok_device::Error::Runtime { message: format!("MLIR ExecutionEngine compilation failed: {}", e) }
        })?;

        Ok(Box::new(MlirProgram { kernel }))
    }
}

#[cfg(feature = "mlir")]
use mlir_backend::{MlirCompiler, MlirRendererWrapper, create_mlir_program};

// =============================================================================
// Public API
// =============================================================================

/// Create a CPU device with the default backend.
///
/// The default backend is selected by:
/// 1. `MOROK_CPU_BACKEND` environment variable ("clang" or "llvm")
/// 2. If not set, defaults to Clang
pub fn create_cpu_device(registry: &DeviceRegistry) -> Result<Device> {
    create_cpu_device_with_backend(registry, CpuBackend::from_env())
}

/// Create a CPU device with a specific backend.
pub fn create_cpu_device_with_backend(registry: &DeviceRegistry, backend: CpuBackend) -> Result<Device> {
    let device_spec = DeviceSpec::Cpu;
    let allocator = registry.get(&device_spec)?;

    match backend {
        CpuBackend::Clang => {
            let renderer = Arc::new(ClangRendererWrapper { device: device_spec.clone() });
            let compiler = Arc::new(ClangCompiler);
            let runtime: RuntimeFactory = Arc::new(create_clang_program);
            Ok(Device::new(device_spec, allocator, renderer, compiler, runtime))
        }
        CpuBackend::Llvm => {
            let renderer = Arc::new(LlvmRendererWrapper { device: device_spec.clone() });
            let compiler = Arc::new(LlvmCompiler);
            let runtime: RuntimeFactory = Arc::new(create_llvm_program);
            Ok(Device::new(device_spec, allocator, renderer, compiler, runtime))
        }
        #[cfg(feature = "mlir")]
        CpuBackend::Mlir => {
            let renderer = Arc::new(MlirRendererWrapper { device: device_spec.clone() });
            let compiler = Arc::new(MlirCompiler);
            let runtime: RuntimeFactory = Arc::new(create_mlir_program);
            Ok(Device::new(device_spec, allocator, renderer, compiler, runtime))
        }
    }
}
