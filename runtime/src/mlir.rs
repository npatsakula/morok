//! MLIR ExecutionEngine-based kernel runtime.

use std::ffi::c_void;

use melior::dialect::DialectRegistry;
use melior::ir::Module;
use melior::utility::{register_all_dialects, register_all_llvm_translations};
use melior::{Context, ExecutionEngine};

use crate::error::Result;

/// MLIR kernel compiled with ExecutionEngine.
pub struct MlirKernel {
    engine: ExecutionEngine,
    name: String,
    var_names: Vec<String>,
}

impl MlirKernel {
    /// Compile MLIR text to executable kernel using ExecutionEngine.
    pub fn compile(mlir_text: &str, kernel_name: &str, var_names: Vec<String>) -> Result<Self> {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        register_all_llvm_translations(&context);

        // Parse MLIR module
        let module = Module::parse(&context, mlir_text)
            .ok_or_else(|| crate::error::Error::JitCompilation { reason: "Failed to parse MLIR module".to_string() })?;

        // Create ExecutionEngine
        let engine = ExecutionEngine::new(&module, 2, &[], false);

        Ok(Self { engine, name: kernel_name.to_string(), var_names })
    }

    /// Get kernel name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get variable names.
    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }

    /// Get function pointer for the kernel.
    pub fn fn_ptr(&self) -> *const c_void {
        let ptr = self.engine.lookup(&self.name);
        if ptr.is_null() {
            panic!("kernel function '{}' not found", self.name);
        }
        ptr as *const c_void
    }

    /// Execute kernel with raw buffer pointers and variable values.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - Buffer pointers are valid and properly aligned
    /// - Buffer sizes match kernel expectations
    /// - vals array matches kernel's variable count
    pub unsafe fn execute_with_vals(&self, buffers: &[*mut u8], vals: &[i64]) -> Result<()> {
        type KernelFn = unsafe extern "C" fn(*const *mut u8, *const i64);

        let fn_ptr = self.fn_ptr();
        let kernel: KernelFn = unsafe { std::mem::transmute(fn_ptr) };

        let buffer_usizes: Vec<usize> = buffers.iter().map(|&ptr| ptr as usize).collect();
        let bufs_ptr = buffer_usizes.as_ptr() as *const *mut u8;

        unsafe {
            kernel(bufs_ptr, vals.as_ptr());
        }

        Ok(())
    }
}
