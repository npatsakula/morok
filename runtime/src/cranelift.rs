//! Cranelift JIT compilation and execution.

use std::collections::HashMap;

use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, MemFlags, Signature, UserFuncName, types};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};

use crate::Result;

/// Cranelift JIT-compiled kernel.
pub struct CraneliftKernel {
    /// JIT module containing the compiled code.
    /// Must be kept alive for func_ptr to remain valid.
    #[allow(dead_code)]
    module: JITModule,

    /// Function pointer to the bootstrap function.
    func_ptr: *const u8,

    /// Number of buffer parameters (kept for debugging/introspection).
    #[allow(dead_code)]
    buffer_count: usize,

    /// Variable names in parameter order.
    var_names: Vec<String>,

    /// Kernel name.
    name: String,
}

// SAFETY: The JITModule owns the compiled code and the function pointer
// points into that code. As long as we keep the module alive, the pointer is valid.
unsafe impl Send for CraneliftKernel {}
unsafe impl Sync for CraneliftKernel {}

impl CraneliftKernel {
    /// Compile Cranelift IR to executable code.
    pub fn compile(ir_with_meta: &str, _entry_point: &str) -> Result<Self> {
        // Parse metadata from combined string
        let (ir_text, name, buffer_count, var_names) = morok_codegen::cranelift::parse_metadata(ir_with_meta)
            .ok_or_else(|| crate::Error::JitCompilation {
                reason: "Failed to parse Cranelift IR metadata".to_string(),
            })?;

        // Setup ISA
        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to set cranelift flag: {}", e) })?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to set cranelift flag: {}", e) })?;

        let isa = cranelift_native::builder()
            .map_err(|e| crate::Error::JitCompilation {
                reason: format!("Failed to create native ISA builder: {}", e),
            })?
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to finish ISA: {}", e) })?;

        // Create module with default libcall names
        let builder = JITBuilder::with_isa(isa.clone(), default_libcall_names());
        let mut module = JITModule::new(builder);

        // Parse the impl function from IR text using cranelift-reader
        let functions = cranelift_reader::parse_functions(&ir_text)
            .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to parse Cranelift IR: {}", e) })?;

        if functions.is_empty() {
            return Err(crate::Error::JitCompilation { reason: "No functions found in Cranelift IR".to_string() });
        }

        // We expect exactly one function: kernel_impl
        let impl_func = functions.into_iter().next().unwrap();
        let impl_name = get_function_name(&impl_func);

        // Declare the impl function
        let mut impl_sig = module.make_signature();
        impl_sig.params = impl_func.signature.params.clone();
        impl_sig.returns = impl_func.signature.returns.clone();
        impl_sig.call_conv = impl_func.signature.call_conv;

        let impl_func_id = module
            .declare_function(&impl_name, Linkage::Local, &impl_sig)
            .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to declare impl function: {}", e) })?;

        // Define the impl function
        let mut ctx = module.make_context();
        ctx.func = impl_func;

        module
            .define_function(impl_func_id, &mut ctx)
            .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to define impl function: {}", e) })?;

        // =======================================================================
        // Build the bootstrap function programmatically
        // Signature: fn kernel(args_ptr: i64, var0: i64, var1: i64, ...)
        // It loads buffer pointers from args_ptr array and calls kernel_impl
        // =======================================================================
        let bootstrap_func_id =
            build_bootstrap_function(&mut module, &name, impl_func_id, &impl_sig, buffer_count, &var_names)?;

        // Finalize all functions
        module
            .finalize_definitions()
            .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to finalize: {}", e) })?;

        // Get function pointer to the bootstrap function
        let func_ptr = module.get_finalized_function(bootstrap_func_id);

        Ok(Self { module, func_ptr, buffer_count, var_names, name })
    }

    /// Execute the kernel with the given buffers and variables.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - Buffer pointers are valid and properly aligned
    /// - Buffer count matches expected parameters
    /// - Variable values are valid
    pub unsafe fn execute(&self, buffers: &[*mut u8], vars: &HashMap<String, i64>) -> Result<()> {
        // The bootstrap function takes (ptr args_array, i64 var0, i64 var1, ...)
        // Build variable values in correct order
        let mut var_values: Vec<i64> = Vec::with_capacity(self.var_names.len());
        for name in &self.var_names {
            let val = vars.get(name).copied().unwrap_or(0);
            var_values.push(val);
        }

        // Call the bootstrap function based on variable count
        // The function signature is: fn(args_ptr: *const *mut u8, var0: i64, var1: i64, ...)
        let args_ptr = buffers.as_ptr();

        match self.var_names.len() {
            0 => {
                let func: extern "C" fn(*const *mut u8) = unsafe { std::mem::transmute(self.func_ptr) };
                func(args_ptr);
            }
            1 => {
                let func: extern "C" fn(*const *mut u8, i64) = unsafe { std::mem::transmute(self.func_ptr) };
                func(args_ptr, var_values[0]);
            }
            2 => {
                let func: extern "C" fn(*const *mut u8, i64, i64) = unsafe { std::mem::transmute(self.func_ptr) };
                func(args_ptr, var_values[0], var_values[1]);
            }
            3 => {
                let func: extern "C" fn(*const *mut u8, i64, i64, i64) = unsafe { std::mem::transmute(self.func_ptr) };
                func(args_ptr, var_values[0], var_values[1], var_values[2]);
            }
            4 => {
                let func: extern "C" fn(*const *mut u8, i64, i64, i64, i64) =
                    unsafe { std::mem::transmute(self.func_ptr) };
                func(args_ptr, var_values[0], var_values[1], var_values[2], var_values[3]);
            }
            5 => {
                let func: extern "C" fn(*const *mut u8, i64, i64, i64, i64, i64) =
                    unsafe { std::mem::transmute(self.func_ptr) };
                func(args_ptr, var_values[0], var_values[1], var_values[2], var_values[3], var_values[4]);
            }
            6 => {
                let func: extern "C" fn(*const *mut u8, i64, i64, i64, i64, i64, i64) =
                    unsafe { std::mem::transmute(self.func_ptr) };
                func(
                    args_ptr,
                    var_values[0],
                    var_values[1],
                    var_values[2],
                    var_values[3],
                    var_values[4],
                    var_values[5],
                );
            }
            7 => {
                let func: extern "C" fn(*const *mut u8, i64, i64, i64, i64, i64, i64, i64) =
                    unsafe { std::mem::transmute(self.func_ptr) };
                func(
                    args_ptr,
                    var_values[0],
                    var_values[1],
                    var_values[2],
                    var_values[3],
                    var_values[4],
                    var_values[5],
                    var_values[6],
                );
            }
            8 => {
                let func: extern "C" fn(*const *mut u8, i64, i64, i64, i64, i64, i64, i64, i64) =
                    unsafe { std::mem::transmute(self.func_ptr) };
                func(
                    args_ptr,
                    var_values[0],
                    var_values[1],
                    var_values[2],
                    var_values[3],
                    var_values[4],
                    var_values[5],
                    var_values[6],
                    var_values[7],
                );
            }
            _ => {
                return Err(crate::Error::JitCompilation {
                    reason: format!("Unsupported number of variables: {}. Max supported is 8.", self.var_names.len()),
                });
            }
        }

        Ok(())
    }

    /// Get the kernel name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Build the bootstrap function that loads buffer pointers and calls kernel_impl.
///
/// The bootstrap function signature is:
///   fn kernel(args_ptr: i64, var0: i64, var1: i64, ...) -> ()
///
/// It performs:
///   buf0 = load(args_ptr[0])
///   buf1 = load(args_ptr[1])
///   ...
///   call kernel_impl(buf0, buf1, ..., var0, var1, ...)
fn build_bootstrap_function(
    module: &mut JITModule,
    name: &str,
    impl_func_id: FuncId,
    _impl_sig: &Signature,
    buffer_count: usize,
    var_names: &[String],
) -> Result<FuncId> {
    // Build bootstrap signature: (i64 args_ptr, i64 var0, i64 var1, ...)
    let mut bootstrap_sig = module.make_signature();
    bootstrap_sig.call_conv = CallConv::SystemV;

    // First param: args_ptr (pointer to array of buffer pointers)
    bootstrap_sig.params.push(AbiParam::new(types::I64));

    // Variable parameters
    for _ in var_names {
        bootstrap_sig.params.push(AbiParam::new(types::I64));
    }

    // Declare bootstrap function
    let bootstrap_func_id = module
        .declare_function(name, Linkage::Local, &bootstrap_sig)
        .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to declare bootstrap function: {}", e) })?;

    // Build function body
    let mut func = Function::with_name_signature(UserFuncName::testcase(name), bootstrap_sig.clone());

    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

        // Create entry block with parameters
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Get parameter values
        let args_ptr = builder.block_params(entry_block)[0];
        let var_values: Vec<_> = (0..var_names.len()).map(|i| builder.block_params(entry_block)[i + 1]).collect();

        // Load buffer pointers from args array
        let mut buf_values = Vec::with_capacity(buffer_count);
        for i in 0..buffer_count {
            let offset = (i * 8) as i32; // 8 bytes per pointer
            let buf_ptr = builder.ins().load(types::I64, MemFlags::trusted(), args_ptr, offset);
            buf_values.push(buf_ptr);
        }

        // Declare the impl function for calling
        let impl_func_ref = module.declare_func_in_func(impl_func_id, builder.func);

        // Build call arguments: buffers first, then variables
        let mut call_args = Vec::with_capacity(buffer_count + var_names.len());
        call_args.extend(buf_values);
        call_args.extend(var_values);

        // Call impl function
        builder.ins().call(impl_func_ref, &call_args);

        // Return
        builder.ins().return_(&[]);

        builder.finalize();
    }

    // Define the bootstrap function
    let mut ctx = module.make_context();
    ctx.func = func;

    module
        .define_function(bootstrap_func_id, &mut ctx)
        .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to define bootstrap function: {}", e) })?;

    Ok(bootstrap_func_id)
}

/// Extract the function name from a Cranelift Function.
fn get_function_name(func: &Function) -> String {
    // The function name is stored in func.name as UserFuncName
    // For testcase names, it's UserFuncName::Testcase(name)
    match &func.name {
        cranelift_codegen::ir::UserFuncName::Testcase(name) => name.to_string(),
        cranelift_codegen::ir::UserFuncName::User(user_ref) => format!("user_{}", user_ref.index),
    }
}

/// CompiledKernel implementation for Cranelift kernels.
impl crate::CompiledKernel for CraneliftKernel {
    unsafe fn execute_with_vars(&self, buffers: &[*mut u8], vars: &HashMap<String, i64>) -> Result<()> {
        // SAFETY: Caller ensures buffers are valid
        unsafe { self.execute(buffers, vars) }
    }

    fn name(&self) -> &str {
        &self.name
    }
}
