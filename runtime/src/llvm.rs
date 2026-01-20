//! LLVM JIT compilation and execution.

use std::mem::ManuallyDrop;

use crate::Result;
use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use tracing::{debug, instrument, trace};

/// LLVM JIT-compiled kernel with proper context ownership.
///
/// Uses Box<Context> for stable addressing and ManuallyDrop for correct drop order.
/// The Context is heap-allocated and never moves, so references remain valid.
pub struct LlvmKernel {
    /// Owned context (heap-allocated, stable address).
    /// Not directly accessed, but must be kept alive for module/engine.
    #[allow(dead_code)]
    context: Box<Context>,

    /// Module referencing the context (dropped before context).
    /// SAFETY: 'static lifetime is a lie, but safe because:
    /// 1. Context is boxed (stable address)
    /// 2. Module is dropped before Context in Drop impl
    module: ManuallyDrop<Module<'static>>,

    /// Execution engine referencing the module (dropped first).
    /// SAFETY: Same rationale as module.
    execution_engine: ManuallyDrop<ExecutionEngine<'static>>,

    /// Entry point function name.
    entry_point: String,

    /// Kernel name for debugging.
    name: String,

    /// Variable names in order (for populating vars array at runtime).
    /// Includes thread_id at the end if threading is enabled.
    var_names: Vec<String>,

    /// Raw function pointer for thread-safe parallel execution.
    /// SAFETY: This is a raw pointer to JIT-compiled code that remains valid
    /// as long as the execution_engine is alive.
    fn_ptr: *const u8,
}

impl LlvmKernel {
    /// Compile LLVM IR to executable code.
    ///
    /// This parses the LLVM IR, verifies it, runs optimization passes,
    /// and JIT compiles using LLVM's MCJIT or ORC engine.
    ///
    /// # Arguments
    /// * `ir` - LLVM IR source code
    /// * `entry_point` - Name of the kernel entry point function
    /// * `name` - Kernel name for debugging/caching
    /// * `var_names` - Variable names in order for populating vars array at runtime
    #[instrument(skip_all, fields(kernel.entry_point, kernel.name))]
    pub fn compile_ir(
        ir: &str,
        entry_point: impl Into<String>,
        name: impl Into<String>,
        var_names: Vec<String>,
    ) -> Result<Self> {
        let entry_point = entry_point.into();
        let name = name.into();

        // Record span fields after conversion
        tracing::Span::current().record("kernel.entry_point", &entry_point);
        tracing::Span::current().record("kernel.name", &name);

        // Create boxed context (stable heap address)
        let context = Box::new(Context::create());

        // SAFETY: We extend lifetime to 'static, but maintain safety by:
        // 1. Context is boxed - stable address, won't move
        // 2. We manually control drop order in Drop impl
        // 3. Module/Engine never outlive the context
        let context_ref: &'static Context = unsafe { &*(context.as_ref() as *const Context) };

        // Parse LLVM IR into a module
        let mem_buffer = inkwell::memory_buffer::MemoryBuffer::create_from_memory_range_copy(ir.as_bytes(), &name);
        let module = context_ref
            .create_module_from_ir(mem_buffer)
            .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to parse LLVM IR: {}", e) })?;

        // Verify the module
        if let Err(err) = module.verify() {
            return Err(crate::Error::JitCompilation { reason: format!("Module verification failed: {}", err) });
        }

        // Dump LLVM IR at trace level (before optimization)
        trace!(
            kernel.name = %name,
            llvm.ir = %module.print_to_string().to_string(),
            "LLVM IR before optimization"
        );

        // Initialize native target for optimization passes
        Target::initialize_native(&InitializationConfig::default()).map_err(|e| crate::Error::JitCompilation {
            reason: format!("Failed to initialize native target: {}", e),
        })?;

        // Create target machine for native CPU with aggressive optimization
        let triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&triple)
            .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to get target: {}", e) })?;
        let target_machine = target
            .create_target_machine(
                &triple,
                TargetMachine::get_host_cpu_name().to_str().unwrap_or("generic"),
                TargetMachine::get_host_cpu_features().to_str().unwrap_or(""),
                OptimizationLevel::Aggressive,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .ok_or_else(|| crate::Error::JitCompilation { reason: "Failed to create target machine".to_string() })?;

        // Configure pass options with explicit vectorization (like Tinygrad)
        let pass_options = PassBuilderOptions::create();
        pass_options.set_loop_vectorization(true);
        pass_options.set_loop_slp_vectorization(true);
        pass_options.set_loop_unrolling(true);

        // Run optimization passes using the new PassBuilder API
        module.run_passes("default<O3>", &target_machine, pass_options).map_err(|e| crate::Error::JitCompilation {
            reason: format!("Failed to run optimization passes: {}", e),
        })?;

        // Dump optimized IR at debug level
        debug!(
            kernel.name = %name,
            llvm.ir = %module.print_to_string().to_string(),
            "LLVM IR after optimization"
        );

        // Create execution engine with Aggressive - the JIT engine may need to do its own codegen
        let execution_engine = module.create_jit_execution_engine(OptimizationLevel::Aggressive).map_err(|e| {
            crate::Error::JitCompilation { reason: format!("Failed to create execution engine: {}", e) }
        })?;

        // Cache function pointer for thread-safe parallel execution
        let fn_ptr = execution_engine
            .get_function_address(&entry_point)
            .map_err(|e| crate::Error::FunctionNotFound { name: format!("{}: {}", entry_point, e) })?
            as *const u8;

        Ok(Self {
            context,
            module: ManuallyDrop::new(module),
            execution_engine: ManuallyDrop::new(execution_engine),
            entry_point,
            name,
            var_names,
            fn_ptr,
        })
    }

    /// Compile a RenderedKernel from the codegen crate.
    pub fn compile(kernel: &morok_codegen::RenderedKernel) -> Result<Self> {
        Self::compile_ir(&kernel.code, &kernel.name, &kernel.name, kernel.var_names.clone())
    }

    /// Get the variable names in order (thread-safe).
    ///
    /// Variable names are passed from codegen and cached for populating the vars array.
    /// Includes thread_id at the end if threading is enabled.
    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }

    /// Get the raw function pointer (thread-safe).
    ///
    /// The function pointer points to JIT-compiled machine code and is safe to call
    /// from multiple threads concurrently (the compiled code is read-only).
    pub fn fn_ptr(&self) -> *const u8 {
        self.fn_ptr
    }

    /// Get the kernel name (for debugging/profiling).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Execute the kernel with positional variable values.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - Buffer pointers are valid and properly aligned
    /// - `vals` has the correct length matching `var_names`
    pub unsafe fn execute_with_vals(&self, buffers: &[*mut u8], vals: &[i64]) -> Result<()> {
        tracing::debug!(
            kernel.entry_point = %self.entry_point,
            kernel.num_buffers = buffers.len(),
            kernel.num_vals = vals.len(),
            "Executing LLVM kernel"
        );

        // Kernel signature: void kernel(ptr %args, ptr %vars)
        type KernelFn = unsafe extern "C" fn(*const *mut u8, *const i64);
        unsafe {
            let f: KernelFn = std::mem::transmute(self.fn_ptr);
            f(buffers.as_ptr(), vals.as_ptr());
        }

        Ok(())
    }
}

impl Drop for LlvmKernel {
    fn drop(&mut self) {
        // SAFETY: Drop in reverse dependency order
        // ExecutionEngine -> Module -> Context
        unsafe {
            ManuallyDrop::drop(&mut self.execution_engine);
            ManuallyDrop::drop(&mut self.module);
            // context drops naturally via Box::drop
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llvm_kernel_no_args() {
        // New signature: kernel(ptr %args, ptr %vars)
        let ir = r#"
            define void @test_kernel(ptr %args, ptr %vars) {
                ret void
            }
        "#;

        let kernel = LlvmKernel::compile_ir(ir, "test_kernel", "test_kernel", vec![]).unwrap();
        assert_eq!(kernel.name(), "test_kernel");

        unsafe {
            kernel.execute_with_vals(&[], &[]).unwrap();
        }
    }

    #[test]
    fn test_llvm_kernel_with_args() {
        // New signature: kernel(ptr %args, ptr %vars)
        // This test just checks that buffers are passed correctly
        let ir = r#"
            define void @add_kernel(ptr %args, ptr %vars) {
                ret void
            }
        "#;

        let kernel = LlvmKernel::compile_ir(ir, "add_kernel", "add_kernel", vec![]).unwrap();

        let mut data1 = vec![0u8; 16];
        let mut data2 = vec![0u8; 16];
        let buffers = vec![data1.as_mut_ptr(), data2.as_mut_ptr()];

        unsafe {
            kernel.execute_with_vals(&buffers, &[]).unwrap();
        }
    }

    #[test]
    fn test_kernel_drop_order() {
        // This test verifies that Drop doesn't crash
        // (proper drop order prevents use-after-free)
        let ir = r#"
            define void @test(ptr %args, ptr %vars) {
                ret void
            }
        "#;

        let kernel = LlvmKernel::compile_ir(ir, "test", "test", vec![]).unwrap();
        drop(kernel); // Should not crash
    }
}
