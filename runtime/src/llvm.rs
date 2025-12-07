//! LLVM JIT compilation and execution.

use std::mem::ManuallyDrop;

use crate::{CompiledKernel, Result};
use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;

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
}

impl LlvmKernel {
    /// Compile LLVM IR to executable code.
    ///
    /// This parses the LLVM IR, verifies it, runs optimization passes,
    /// and JIT compiles using LLVM's MCJIT or ORC engine.
    pub fn compile_ir(ir: &str, entry_point: impl Into<String>, name: impl Into<String>) -> Result<Self> {
        let entry_point = entry_point.into();
        let name = name.into();

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

        // Create execution engine with optimization
        let execution_engine = module.create_jit_execution_engine(OptimizationLevel::Aggressive).map_err(|e| {
            crate::Error::JitCompilation { reason: format!("Failed to create execution engine: {}", e) }
        })?;

        Ok(Self {
            context,
            module: ManuallyDrop::new(module),
            execution_engine: ManuallyDrop::new(execution_engine),
            entry_point,
            name,
        })
    }

    /// Compile a RenderedKernel from the codegen crate.
    pub fn compile(kernel: &morok_codegen::RenderedKernel) -> Result<Self> {
        Self::compile_ir(&kernel.code, &kernel.entry_point, &kernel.name)
    }
}

impl CompiledKernel for LlvmKernel {
    unsafe fn execute_with_vars(
        &self,
        buffers: &[*mut u8],
        vars: &std::collections::HashMap<String, i64>,
    ) -> Result<()> {
        // Get the function from the module to inspect its signature
        let function = self
            .module
            .get_function(&self.entry_point)
            .ok_or_else(|| crate::Error::FunctionNotFound { name: self.entry_point.clone() })?;

        let param_count = function.count_params() as usize;
        let var_count = param_count.saturating_sub(1); // Subtract 1 for buffer array pointer

        if var_count == 0 {
            // No variables - use simple bootstrap function
            type BootstrapFn0 = unsafe extern "C" fn(*const *mut u8);
            let func: JitFunction<BootstrapFn0> = unsafe {
                self.execution_engine
                    .get_function(&self.entry_point)
                    .map_err(|e| crate::Error::FunctionNotFound { name: format!("{}: {}", self.entry_point, e) })?
            };
            unsafe {
                func.call(buffers.as_ptr());
            }
        } else {
            // Extract variable values in parameter order
            // The function parameters are ordered: (ptr %args, i64 %var0, i64 %var1, ...)
            // We need to match variable names from the HashMap to parameter positions
            let mut var_values = Vec::new();
            eprintln!("EXECUTE_WITH_VARS: param_count={}, var_count={}", param_count, var_count);
            eprintln!("EXECUTE_WITH_VARS: vars={:?}", vars);
            for i in 1..param_count {
                if let Some(param) = function.get_nth_param(i as u32) {
                    let param_name = param.get_name().to_str().map_err(|_| crate::Error::JitCompilation {
                        reason: format!("Invalid UTF-8 in parameter name at index {}", i),
                    })?;

                    eprintln!("EXECUTE_WITH_VARS: param[{}] name='{}', looking up in vars", i, param_name);
                    let value = vars.get(param_name).copied().ok_or_else(|| crate::Error::JitCompilation {
                        reason: format!("Missing variable value for parameter '{}'", param_name),
                    })?;

                    eprintln!("EXECUTE_WITH_VARS: param[{}] name='{}', value={}", i, param_name, value);
                    var_values.push(value);
                }
            }

            // Call with the appropriate number of variable parameters
            match var_count {
                1 => {
                    type BootstrapFn1 = unsafe extern "C" fn(*const *mut u8, i64);
                    let func: JitFunction<BootstrapFn1> = unsafe {
                        self.execution_engine.get_function(&self.entry_point).map_err(|e| {
                            crate::Error::FunctionNotFound { name: format!("{}: {}", self.entry_point, e) }
                        })?
                    };
                    unsafe {
                        func.call(buffers.as_ptr(), var_values[0]);
                    }
                }
                2 => {
                    type BootstrapFn2 = unsafe extern "C" fn(*const *mut u8, i64, i64);
                    let func: JitFunction<BootstrapFn2> = unsafe {
                        self.execution_engine.get_function(&self.entry_point).map_err(|e| {
                            crate::Error::FunctionNotFound { name: format!("{}: {}", self.entry_point, e) }
                        })?
                    };
                    unsafe {
                        func.call(buffers.as_ptr(), var_values[0], var_values[1]);
                    }
                }
                3 => {
                    type BootstrapFn3 = unsafe extern "C" fn(*const *mut u8, i64, i64, i64);
                    let func: JitFunction<BootstrapFn3> = unsafe {
                        self.execution_engine.get_function(&self.entry_point).map_err(|e| {
                            crate::Error::FunctionNotFound { name: format!("{}: {}", self.entry_point, e) }
                        })?
                    };
                    unsafe {
                        func.call(buffers.as_ptr(), var_values[0], var_values[1], var_values[2]);
                    }
                }
                _ => {
                    return Err(crate::Error::JitCompilation {
                        reason: format!("Unsupported number of variables: {}. Max supported is 3.", var_count),
                    });
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
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
        let ir = r#"
            define void @test_kernel_impl() {
                ret void
            }

            define void @test_kernel(ptr %args) {
                call void @test_kernel_impl()
                ret void
            }
        "#;

        let kernel = LlvmKernel::compile_ir(ir, "test_kernel", "test_kernel").unwrap();
        assert_eq!(kernel.name(), "test_kernel");

        unsafe {
            kernel.execute(&[]).unwrap();
        }
    }

    #[test]
    fn test_llvm_kernel_with_args() {
        let ir = r#"
            define void @add_kernel_impl(ptr %buf0, ptr %buf1) {
                ret void
            }

            define void @add_kernel(ptr %args) {
                %buf0_ptr = getelementptr ptr, ptr %args, i64 0
                %buf0 = load ptr, ptr %buf0_ptr
                %buf1_ptr = getelementptr ptr, ptr %args, i64 1
                %buf1 = load ptr, ptr %buf1_ptr
                call void @add_kernel_impl(ptr %buf0, ptr %buf1)
                ret void
            }
        "#;

        let kernel = LlvmKernel::compile_ir(ir, "add_kernel", "add_kernel").unwrap();

        let mut data1 = vec![0u8; 16];
        let mut data2 = vec![0u8; 16];
        let buffers = vec![data1.as_mut_ptr(), data2.as_mut_ptr()];

        unsafe {
            kernel.execute(&buffers).unwrap();
        }
    }

    #[test]
    fn test_kernel_drop_order() {
        // This test verifies that Drop doesn't crash
        // (proper drop order prevents use-after-free)
        let ir = r#"
            define void @test(ptr %args) {
                ret void
            }
        "#;

        let kernel = LlvmKernel::compile_ir(ir, "test", "test").unwrap();
        drop(kernel); // Should not crash
    }
}
