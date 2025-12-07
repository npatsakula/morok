//! LLVM IR renderer implementation using inkwell.

use crate::{RenderedKernel, Renderer, Result, with_context};
use inkwell::context::Context;
use inkwell::module::Module;
use morok_ir::{Op, UOp};
use snafu::{OptionExt, ResultExt};
use std::sync::Arc;

use super::error::{BuildCallSnafu, BuildGepSnafu, BuildLoadSnafu, BuildReturnSnafu, InvalidFunctionParameterSnafu};
use super::helpers::ValueMap;

/// Render a UOp graph to LLVM IR using the thread-local context.
pub fn render(uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
    with_context(|context| {
        let renderer = LlvmRenderer::new(context);
        renderer.render(uop, name)
    })
}

/// Collect all buffer and variable parameters from a UOp graph.
///
/// After rangeify, BUFFER operations have been converted to DEFINE_GLOBAL/DEFINE_LOCAL,
/// and OUTER ranges have been converted to DEFINE_VAR.
///
/// This function collects:
/// - Buffers: DEFINE_GLOBAL, DEFINE_LOCAL, BUFFER (for non-rangeified graphs)
/// - Variables: DEFINE_VAR (for OUTER range iteration values)
///
/// Returns (buffers, variables) in a consistent order for deterministic function signatures:
/// - Buffers: DEFINE_GLOBAL sorted by internal ID, DEFINE_LOCAL sorted by internal ID, BUFFER sorted by UOp ID
/// - Variables: DEFINE_VAR sorted by variable name
fn collect_buffers_and_vars(root: &Arc<UOp>) -> (Vec<Arc<UOp>>, Vec<Arc<UOp>>) {
    let nodes = root.toposort();

    // Collect buffers
    let mut buffers = Vec::new();
    for node in &nodes {
        match node.op() {
            Op::Buffer { .. } | Op::DefineGlobal(_) | Op::DefineLocal(_) => {
                buffers.push(node.clone());
            }
            _ => {}
        }
    }

    // Sort buffers by internal ID (matches split_kernel.rs ordering)
    buffers.sort_by_key(|b| match b.op() {
        Op::DefineGlobal(id) => *id as u64,
        Op::DefineLocal(id) => (*id as u64) + (1u64 << 32), // Offset locals after globals
        Op::Buffer { .. } => b.id + (1u64 << 48),           // Offset input buffers after defines
        _ => b.id,
    });

    // Collect variables
    let mut variables = Vec::new();
    for node in &nodes {
        if let Op::DefineVar { .. } = node.op() {
            variables.push(node.clone());
        }
    }

    // Sort variables by name for deterministic ordering (matches split_kernel.rs ordering)
    variables.sort_by(|a, b| {
        let name_a = match a.op() {
            Op::DefineVar { name, .. } => name,
            _ => unreachable!("filtered to only DefineVar above"),
        };
        let name_b = match b.op() {
            Op::DefineVar { name, .. } => name,
            _ => unreachable!("filtered to only DefineVar above"),
        };
        name_a.cmp(name_b)
    });

    (buffers, variables)
}

/// Add parameter attributes to buffer pointer parameters.
///
/// Adds `noalias` and `align 32` attributes to buffer parameters only.
/// These attributes help LLVM optimize memory accesses by:
/// - `noalias`: Indicates buffers don't overlap, enabling vectorization
/// - `align 32`: Specifies 32-byte alignment for SIMD operations
///
/// Variable parameters (i64) don't get these attributes.
///
/// Based on Tinygrad's LLVM renderer (tinygrad/renderer/llvmir.py).
fn add_buffer_param_attributes<'ctx>(
    function: inkwell::values::FunctionValue<'ctx>,
    buffer_count: u32,
    context: &'ctx Context,
) {
    use inkwell::attributes::AttributeLoc;

    // Only add attributes to buffer parameters (not variable parameters)
    for i in 0..buffer_count {
        // Add noalias attribute (buffers don't alias)
        let noalias_attr_id = inkwell::attributes::Attribute::get_named_enum_kind_id("noalias");
        let noalias = context.create_enum_attribute(noalias_attr_id, 0);
        function.add_attribute(AttributeLoc::Param(i), noalias);

        // Add alignment attribute (32-byte aligned for SIMD)
        let align_attr_id = inkwell::attributes::Attribute::get_named_enum_kind_id("align");
        let align = context.create_enum_attribute(align_attr_id, 32);
        function.add_attribute(AttributeLoc::Param(i), align);
    }
}

/// Add function attributes to kernel function.
///
/// Adds critical function attributes for correctness and performance:
/// - `alwaysinline`: Always inline this function
/// - `nounwind`: Function doesn't throw exceptions
/// - `no-trapping-math`: Allows aggressive float optimizations
/// - `no-builtins`: Don't replace with builtin implementations
///
/// Based on Tinygrad's LLVM renderer (tinygrad/renderer/llvmir.py).
fn add_kernel_function_attributes<'ctx>(function: inkwell::values::FunctionValue<'ctx>, context: &'ctx Context) {
    use inkwell::attributes::AttributeLoc;

    // nounwind - function doesn't throw exceptions
    let nounwind_id = inkwell::attributes::Attribute::get_named_enum_kind_id("nounwind");
    let nounwind = context.create_enum_attribute(nounwind_id, 0);
    function.add_attribute(AttributeLoc::Function, nounwind);

    // alwaysinline - always inline this function
    let alwaysinline_id = inkwell::attributes::Attribute::get_named_enum_kind_id("alwaysinline");
    let alwaysinline = context.create_enum_attribute(alwaysinline_id, 0);
    function.add_attribute(AttributeLoc::Function, alwaysinline);

    // no-trapping-math - allows aggressive float optimizations
    let no_trap_math = context.create_string_attribute("no-trapping-math", "true");
    function.add_attribute(AttributeLoc::Function, no_trap_math);

    // no-builtins - don't replace with builtin implementations
    let no_builtins = context.create_string_attribute("no-builtins", "");
    function.add_attribute(AttributeLoc::Function, no_builtins);
}

/// LLVM IR code generator for CPU execution using inkwell.
pub struct LlvmRenderer<'ctx> {
    context: &'ctx Context,
}

impl<'ctx> LlvmRenderer<'ctx> {
    /// Create a new LLVM renderer with a given context.
    ///
    /// The context must outlive the renderer, typically the context
    /// is owned by the calling code and passed by reference.
    pub fn new(context: &'ctx Context) -> Self {
        Self { context }
    }

    /// Get the inkwell context.
    pub fn context(&self) -> &'ctx Context {
        self.context
    }

    /// Render a UOp graph into LLVM IR.
    ///
    /// This creates a module with:
    /// 1. The actual kernel function taking individual buffer pointers
    /// 2. A bootstrap function that unpacks an array of pointers and calls the kernel
    fn render_to_module(&self, uop: &Arc<UOp>, name: &str) -> Result<Module<'ctx>> {
        let module = self.context.create_module(name);
        let builder = self.context.create_builder();

        // Collect all buffers and variables from the graph
        let (buffers, variables) = collect_buffers_and_vars(uop);

        // Create kernel function signature: void kernel(ptr %buf0, ptr %buf1, ..., i64 %var0, i64 %var1, ...)
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        let i64_type = self.context.i64_type();

        let mut param_types: Vec<inkwell::types::BasicMetadataTypeEnum> = Vec::new();

        // Add buffer pointer parameters
        for _ in &buffers {
            param_types.push(ptr_type.into());
        }

        // Add variable i64 parameters
        for _ in &variables {
            param_types.push(i64_type.into());
        }

        let fn_type = self.context.void_type().fn_type(&param_types, false);
        let kernel_name = format!("{}_impl", name);
        let kernel_function = module.add_function(&kernel_name, fn_type, None);

        // Add LLVM attributes to buffer parameters only (not variables)
        add_buffer_param_attributes(kernel_function, buffers.len() as u32, self.context);
        add_kernel_function_attributes(kernel_function, self.context);

        // Create entry block for kernel
        let entry_block = self.context.append_basic_block(kernel_function, "entry");
        builder.position_at_end(entry_block);

        // Create ValueMap and populate with buffer and variable parameters
        let mut values = ValueMap::new();

        // Add buffer parameters
        for (i, buffer_uop) in buffers.iter().enumerate() {
            let param =
                kernel_function.get_nth_param(i as u32).context(InvalidFunctionParameterSnafu { index: i as u32 })?;
            param.set_name(&format!("buf{}", i));
            values.insert(buffer_uop.id, param);
        }

        // Add variable parameters (after buffers)
        let buffer_count = buffers.len();
        for (i, var_uop) in variables.iter().enumerate() {
            let param_idx = (buffer_count + i) as u32;
            let param =
                kernel_function.get_nth_param(param_idx).context(InvalidFunctionParameterSnafu { index: param_idx })?;
            if let Op::DefineVar { name, .. } = var_uop.op() {
                param.set_name(name);
            }
            values.insert(var_uop.id, param);
        }

        // Walk the UOp graph in topological order and generate code
        let nodes = uop.toposort();
        for node in &nodes {
            // Generate code for this node
            super::ops::codegen_uop(node, self.context, &module, &builder, &mut values)?;
        }

        // Return void
        builder.build_return(None).context(BuildReturnSnafu)?;

        // Create bootstrap function: void kernel_bootstrap(ptr %args, i64 %var0, i64 %var1, ...)
        // This unpacks the array of buffer pointers and forwards both buffers and variables to the kernel
        let mut bootstrap_param_types: Vec<inkwell::types::BasicMetadataTypeEnum> = Vec::new();
        bootstrap_param_types.push(ptr_type.into()); // buffer array pointer

        // Add variable parameters
        for _ in &variables {
            bootstrap_param_types.push(i64_type.into());
        }

        let bootstrap_fn_type = self.context.void_type().fn_type(&bootstrap_param_types, false);
        let bootstrap_function = module.add_function(name, bootstrap_fn_type, None);

        let bootstrap_entry = self.context.append_basic_block(bootstrap_function, "entry");
        builder.position_at_end(bootstrap_entry);

        let args_array = bootstrap_function
            .get_first_param()
            .context(InvalidFunctionParameterSnafu { index: 0u32 })?
            .into_pointer_value();
        args_array.set_name("args");

        // Extract each buffer pointer from the array
        let mut kernel_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::new();
        for i in 0..buffers.len() {
            // GEP to get pointer to args[i]
            let index = self.context.i64_type().const_int(i as u64, false);
            let ptr_to_ptr = unsafe {
                builder.build_gep(ptr_type, args_array, &[index], &format!("arg{}_ptr", i)).context(BuildGepSnafu)?
            };

            // Load the actual buffer pointer
            let buffer_ptr = builder.build_load(ptr_type, ptr_to_ptr, &format!("arg{}", i)).context(BuildLoadSnafu)?;

            kernel_args.push(buffer_ptr.into());
        }

        // Add variable parameters from bootstrap function parameters
        for i in 0..variables.len() {
            let param_idx = (1 + i) as u32; // +1 because first param is buffer array
            let var_param = bootstrap_function
                .get_nth_param(param_idx)
                .context(InvalidFunctionParameterSnafu { index: param_idx })?;
            if i < variables.len()
                && let Op::DefineVar { name, .. } = variables[i].op()
            {
                var_param.set_name(name);
            }
            kernel_args.push(var_param.into());
        }

        // Call the kernel with unpacked buffer pointers and variable values
        builder
            .build_call(kernel_function, &kernel_args, "")
            .context(BuildCallSnafu { intrinsic: kernel_name.to_string() })?;

        // Return void
        builder.build_return(None).context(BuildReturnSnafu)?;

        // Verify the module
        module.verify().map_err(|err| super::error::Error::ModuleVerification { message: err.to_string() })?;

        Ok(module)
    }
}

impl<'ctx> Renderer for LlvmRenderer<'ctx> {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
        let kernel_name = name.unwrap_or("kernel");

        // Generate LLVM IR module
        let module = self.render_to_module(uop, kernel_name)?;

        // Get LLVM IR as string
        let ir_string = module.print_to_string().to_string();

        Ok(RenderedKernel::new(ir_string, kernel_name.to_string(), kernel_name.to_string()))
    }

    fn backend_name(&self) -> &str {
        "llvm"
    }

    fn decompositor(&self) -> Option<morok_ir::pattern::PatternMatcher<()>> {
        // LLVM has native transcendentals, no decomposition needed
        None
    }
}
