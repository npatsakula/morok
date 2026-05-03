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
use crate::dispatch::KernelCif;

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
/// - Each core_id maps to disjoint output indices
/// - Index formula: `output[core_id * chunk_size + local_idx]`
///
/// Same buffer pointers can be safely passed to all threads because:
/// 1. Input buffers: Read-only access (no data race)
/// 2. Output buffers: Disjoint write regions per thread
unsafe fn execute_parallel(
    cif: &KernelCif,
    fn_ptr: *const (),
    buffers: &[*mut u8],
    vals: &[i64],
    var_names: &[String],
    core_count: usize,
) -> Result<()> {
    use rayon::prelude::*;

    let core_id_idx = var_names.iter().position(|n| n == "core_id").ok_or_else(|| morok_device::Error::Runtime {
        message: "parallel CPU launch requires core_id runtime variable".to_string(),
    })?;
    let fn_ptr_usize = fn_ptr as usize;

    // Convert raw pointers to usize for Send-safe cross-thread sharing.
    // Safety: buffer pointers are read-only and point to disjoint write
    // regions per thread (guaranteed by shift_to transformation).
    let buf_ptr = buffers.as_ptr() as usize;
    let buf_len = buffers.len();

    // Nested parallelism policy: if we're already inside rayon work, avoid
    // spawning another parallel loop for core_id kernels.
    if rayon::current_thread_index().is_some() {
        for core_id in 0..core_count {
            let bufs = unsafe { std::slice::from_raw_parts(buf_ptr as *const *mut u8, buf_len) };
            unsafe {
                cif.dispatch(fn_ptr_usize as *const (), bufs, vals, Some((core_id_idx, core_id)));
            }
        }
        return Ok(());
    }

    (0..core_count).into_par_iter().for_each(|core_id| {
        let bufs = unsafe { std::slice::from_raw_parts(buf_ptr as *const *mut u8, buf_len) };
        unsafe {
            cif.dispatch(fn_ptr_usize as *const (), bufs, vals, Some((core_id_idx, core_id)));
        }
    });

    Ok(())
}

// =============================================================================
// Shared kernel execution
// =============================================================================

/// Execute a kernel: parallel if global_size > 1, otherwise single-threaded.
unsafe fn execute_kernel(
    cif: &KernelCif,
    fn_ptr: *const (),
    buffers: &[*mut u8],
    vals: &[i64],
    var_names: &[String],
    global_size: Option<[usize; 3]>,
) -> Result<()> {
    let core_count = global_size.map(|[tc, _, _]| tc).filter(|&tc| tc > 1);
    if let Some(count) = core_count {
        unsafe { execute_parallel(cif, fn_ptr, buffers, vals, var_names, count) }
    } else {
        unsafe { cif.dispatch(fn_ptr, buffers, vals, None) };
        Ok(())
    }
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
        unsafe {
            execute_kernel(self.kernel.cif(), self.kernel.fn_ptr(), buffers, vals, self.kernel.var_names(), global_size)
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
    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> Result<ProgramSpec> {
        let rendered = morok_codegen::c::render(ast, name.or(Some("kernel")))
            .map_err(|e| morok_device::Error::Runtime { message: format!("C rendering failed: {}", e) })?;

        let mut spec = ProgramSpec::new(rendered.name.clone(), rendered.code.clone(), self.device.clone(), ast.clone());

        spec.set_var_names(rendered.var_names.clone());
        spec.apply_derived_metadata_from_ast();
        if spec.buf_count == 0 {
            spec.buf_count = rendered.buffer_args.len();
        }

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
        let mut compiled = morok_device::device::CompiledSpec::from_source(
            spec.name.clone(),
            spec.src.clone(),
            spec.ast.clone(),
            spec.buf_count,
        );
        compiled.var_names = spec.var_names.clone();
        compiled.global_size = spec.global_size.clone();
        compiled.local_size = spec.local_size.clone();
        Ok(compiled)
    }

    fn cache_key(&self) -> &'static str {
        "clang"
    }
}

/// Runtime factory for creating Clang programs.
fn create_clang_program(spec: &morok_device::device::CompiledSpec) -> Result<Box<dyn Program>> {
    let src = spec.src.as_ref().ok_or_else(|| morok_device::Error::Runtime {
        message: "Clang backend requires source code in CompiledSpec".to_string(),
    })?;

    let kernel = ClangKernel::compile(src, &spec.name, spec.var_names.clone(), spec.buf_count)
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
        unsafe {
            execute_kernel(self.kernel.cif(), self.kernel.fn_ptr(), buffers, vals, self.kernel.var_names(), global_size)
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
        let mut compiled = morok_device::device::CompiledSpec::from_source(
            spec.name.clone(),
            spec.src.clone(),
            spec.ast.clone(),
            spec.buf_count,
        );
        compiled.var_names = spec.var_names.clone();
        compiled.global_size = spec.global_size.clone();
        compiled.local_size = spec.local_size.clone();
        Ok(compiled)
    }

    fn cache_key(&self) -> &'static str {
        "llvm-jit"
    }
}

/// LLVM renderer wrapper implementing the Renderer trait.
struct LlvmRendererWrapper {
    device: DeviceSpec,
}

impl Renderer for LlvmRendererWrapper {
    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> Result<ProgramSpec> {
        let rendered = morok_codegen::llvm::text::render(ast, name.or(Some("kernel")))
            .map_err(|e| morok_device::Error::Runtime { message: format!("LLVM rendering failed: {}", e) })?;

        let mut spec = ProgramSpec::new(rendered.name.clone(), rendered.code.clone(), self.device.clone(), ast.clone());

        spec.set_var_names(rendered.var_names.clone());
        spec.apply_derived_metadata_from_ast();
        if spec.buf_count == 0 {
            spec.buf_count = rendered.buffer_args.len();
        }

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

    let kernel = crate::LlvmKernel::compile_ir(src, &spec.name, &spec.name, spec.var_names.clone(), spec.buf_count)
        .map_err(|e| morok_device::Error::Runtime { message: format!("LLVM JIT compilation failed: {}", e) })?;

    Ok(Box::new(LlvmProgram { kernel }))
}

// =============================================================================
// MLIR Backend
// =============================================================================

#[cfg(feature = "mlir")]
mod mlir_backend {
    use std::ffi::c_void;

    use super::*;

    type MlirKernelFn = unsafe extern "C" fn(*const *mut u8, *const i64);

    unsafe fn dispatch_mlir_fn(fn_ptr: *const c_void, buffers: &[*mut u8], vals: &[i64]) {
        let kernel: MlirKernelFn = unsafe { std::mem::transmute(fn_ptr) };
        let buffer_usizes: Vec<usize> = buffers.iter().map(|&ptr| ptr as usize).collect();
        let bufs_ptr = buffer_usizes.as_ptr() as *const *mut u8;
        unsafe {
            kernel(bufs_ptr, vals.as_ptr());
        }
    }

    unsafe fn execute_mlir_parallel(
        fn_ptr: *const c_void,
        buffers: &[*mut u8],
        vals: &[i64],
        var_names: &[String],
        core_count: usize,
    ) -> Result<()> {
        use rayon::prelude::*;

        let core_id_idx =
            var_names.iter().position(|n| n == "core_id").ok_or_else(|| morok_device::Error::Runtime {
                message: "parallel MLIR CPU launch requires core_id runtime variable".to_string(),
            })?;
        let fn_ptr_usize = fn_ptr as usize;

        // Convert raw pointers to usize for Send-safe cross-thread sharing.
        let buf_ptr = buffers.as_ptr() as usize;
        let buf_len = buffers.len();
        let vals = vals.to_vec();

        // Avoid nested parallelism when already executing inside rayon worker.
        if rayon::current_thread_index().is_some() {
            for core_id in 0..core_count {
                let bufs = unsafe { std::slice::from_raw_parts(buf_ptr as *const *mut u8, buf_len) };
                let mut thread_vals = vals.clone();
                thread_vals[core_id_idx] = core_id as i64;
                unsafe { dispatch_mlir_fn(fn_ptr_usize as *const c_void, bufs, &thread_vals) };
            }
            return Ok(());
        }

        (0..core_count).into_par_iter().for_each(|core_id| {
            let bufs = unsafe { std::slice::from_raw_parts(buf_ptr as *const *mut u8, buf_len) };
            let mut thread_vals = vals.clone();
            thread_vals[core_id_idx] = core_id as i64;
            unsafe { dispatch_mlir_fn(fn_ptr_usize as *const c_void, bufs, &thread_vals) };
        });

        Ok(())
    }

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
            let core_count = global_size.map(|[tc, _, _]| tc).filter(|&tc| tc > 1);
            let fn_ptr = self.kernel.fn_ptr();

            if let Some(count) = core_count {
                unsafe { execute_mlir_parallel(fn_ptr, buffers, vals, self.kernel.var_names(), count) }
            } else {
                unsafe { dispatch_mlir_fn(fn_ptr, buffers, vals) };
                Ok(())
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
        fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> Result<ProgramSpec> {
            let rendered = morok_codegen::mlir::render(ast, name.or(Some("kernel")))
                .map_err(|e| morok_device::Error::Runtime { message: format!("MLIR rendering failed: {}", e) })?;

            let mut spec =
                ProgramSpec::new(rendered.name.clone(), rendered.code.clone(), self.device.clone(), ast.clone());

            spec.set_var_names(rendered.var_names.clone());
            spec.apply_derived_metadata_from_ast();
            if spec.buf_count == 0 {
                spec.buf_count = rendered.buffer_args.len();
            }

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
            let mut compiled = morok_device::device::CompiledSpec::from_source(
                spec.name.clone(),
                spec.src.clone(),
                spec.ast.clone(),
                spec.buf_count,
            );
            compiled.var_names = spec.var_names.clone();
            compiled.global_size = spec.global_size.clone();
            compiled.local_size = spec.local_size.clone();
            Ok(compiled)
        }

        fn cache_key(&self) -> &'static str {
            "mlir-exec-engine"
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
