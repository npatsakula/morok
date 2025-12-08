//! CPU-specific LLVM code generation.
//!
//! This module provides the CPU renderer which generates LLVM IR for CPU execution.

pub mod ops;

use inkwell::attributes::AttributeLoc;
use inkwell::context::Context;
use inkwell::module::Module;
use snafu::{OptionExt, ResultExt};
use std::sync::Arc;

use morok_ir::{Op, UOp};

use crate::llvm::error::*;
use crate::llvm::helpers::ValueMap;
use crate::{RenderedKernel, Renderer};

/// CPU LLVM renderer.
///
/// Generates LLVM IR for CPU execution. This renderer:
/// - Always inlines outer loops (LLVM can optimize entire loop nests)
/// - Uses all CPU-specific optimizations (noalias, alignment hints)
pub struct CpuLlvmRenderer<'ctx> {
    context: &'ctx Context,
}

impl<'ctx> CpuLlvmRenderer<'ctx> {
    /// Create a new CPU LLVM renderer with a given context.
    pub fn new(context: &'ctx Context) -> Self {
        Self { context }
    }

    /// Get the inkwell context.
    pub fn context(&self) -> &'ctx Context {
        self.context
    }

    /// Render a UOp graph into LLVM IR module.
    fn render_to_module(&self, uop: &Arc<UOp>, name: &str) -> Result<Module<'ctx>> {
        let module = self.context.create_module(name);
        let builder = self.context.create_builder();

        // Collect all buffers and variables from the graph
        let (buffers, variables) = collect_buffers_and_vars(uop);

        // Create kernel function signature: void kernel(ptr %buf0, ..., i64 %var0, ...)
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

        // Add LLVM attributes to buffer parameters only
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

        // Note: variables vector is empty for CPU with inlined outer loops
        // DefineVar values are generated as loop counters below

        // Walk the UOp graph in topological order
        let nodes = uop.toposort();

        // Pre-pass: Identify nodes that are in REDUCE source subgraphs
        // These nodes should NOT be processed in the main loop - REDUCE will handle them
        // This is critical because REDUCE needs to set up the loop counter BEFORE
        // its source is evaluated (e.g., INDEX needs the loop counter, not the end value)
        let reduce_src_nodes = find_reduce_source_nodes(&nodes);

        // Pre-pass: Generate loops for DefineVar nodes
        // For CPU with inlined outer loops, each DefineVar represents an outer loop
        // We generate the loop structure here and map DefineVar to the loop counter
        // Store the DefineVar IDs to close their loops at the end
        let mut outer_loop_ids: Vec<u64> = Vec::new();
        for node in &nodes {
            if let Op::DefineVar { max_val, .. } = node.op() {
                // Generate end value: max_val + 1 (DefineVar stores max value, we need exclusive upper bound)
                let end_int = self.context.i64_type().const_int((*max_val + 1) as u64, false);

                // Build loop structure
                let (loop_ctx, counter_val) = crate::llvm::common::loop_gen::build_loop(
                    self.context,
                    &builder,
                    kernel_function,
                    end_int,
                    node.id,
                )?;

                // Store loop context (using DefineVar id as key)
                values.insert_loop(node.id, loop_ctx);

                // Map DefineVar to loop counter
                values.insert(node.id, counter_val.into());

                // Track for closing at the end
                outer_loop_ids.push(node.id);
            }
        }

        // Main pass: process all remaining nodes
        // Skip nodes in REDUCE source subgraphs - REDUCE will handle them
        for node in &nodes {
            if reduce_src_nodes.contains(&node.id) {
                continue;
            }
            ops::codegen_uop(node, self.context, &module, &builder, &mut values)?;
        }

        // Close outer loops (innermost to outermost, so reverse order)
        for &loop_id in outer_loop_ids.iter().rev() {
            if let Some(loop_ctx) = values.get_loop(loop_id) {
                let loop_ctx = loop_ctx.clone();
                crate::llvm::common::loop_gen::close_loop(&builder, &loop_ctx)?;
            }
        }

        // Return void
        builder.build_return(None).context(BuildReturnSnafu)?;

        // Create bootstrap function: void kernel_bootstrap(ptr %args, i64 %var0, ...)
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
            let index = self.context.i64_type().const_int(i as u64, false);
            let ptr_to_ptr = unsafe {
                builder.build_gep(ptr_type, args_array, &[index], &format!("arg{}_ptr", i)).context(BuildGepSnafu)?
            };
            let buffer_ptr = builder.build_load(ptr_type, ptr_to_ptr, &format!("arg{}", i)).context(BuildLoadSnafu)?;
            kernel_args.push(buffer_ptr.into());
        }

        // Add variable parameters from bootstrap function parameters
        for (i, var) in variables.iter().enumerate() {
            let param_idx = (1 + i) as u32; // +1 because first param is buffer array
            let var_param = bootstrap_function
                .get_nth_param(param_idx)
                .context(InvalidFunctionParameterSnafu { index: param_idx })?;
            if let Op::DefineVar { name, .. } = var.op() {
                var_param.set_name(name);
            }
            kernel_args.push(var_param.into());
        }

        // Call the kernel
        builder
            .build_call(kernel_function, &kernel_args, "")
            .context(BuildCallSnafu { intrinsic: kernel_name.to_string() })?;

        // Return void
        builder.build_return(None).context(BuildReturnSnafu)?;

        // Verify the module
        module.verify().map_err(|err| Error::ModuleVerification { message: err.to_string() })?;

        Ok(module)
    }
}

impl<'ctx> Renderer for CpuLlvmRenderer<'ctx> {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> crate::Result<RenderedKernel> {
        let kernel_name = name.unwrap_or("kernel");
        let module = self.render_to_module(uop, kernel_name)?;
        let ir_string = module.print_to_string().to_string();
        Ok(RenderedKernel::new(ir_string, kernel_name.to_string(), kernel_name.to_string()))
    }

    fn backend_name(&self) -> &str {
        "llvm"
    }

    fn decompositor(&self) -> Option<morok_ir::pattern::PatternMatcher<()>> {
        None
    }
}

/// Collect buffer parameters from a UOp graph.
///
/// Collects:
/// - Buffers: DEFINE_GLOBAL, DEFINE_LOCAL, BUFFER operations
///
/// Note: DefineVar is NOT collected as a parameter. With inlined outer loops,
/// DefineVar values are generated as loop counters inside the kernel, not as
/// function parameters passed from the scheduler.
///
/// Returns buffers sorted for deterministic function signatures.
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
        Op::DefineLocal(id) => (*id as u64) + (1u64 << 32),
        Op::Buffer { .. } => b.id + (1u64 << 48),
        _ => b.id,
    });

    // For CPU with inlined outer loops, DefineVar becomes loop counters, not parameters
    // Return empty variables vector
    let variables = Vec::new();

    (buffers, variables)
}

/// Add parameter attributes to buffer pointer parameters.
fn add_buffer_param_attributes<'ctx>(
    function: inkwell::values::FunctionValue<'ctx>,
    buffer_count: u32,
    context: &'ctx Context,
) {
    for i in 0..buffer_count {
        let noalias_attr_id = inkwell::attributes::Attribute::get_named_enum_kind_id("noalias");
        let noalias = context.create_enum_attribute(noalias_attr_id, 0);
        function.add_attribute(AttributeLoc::Param(i), noalias);

        let align_attr_id = inkwell::attributes::Attribute::get_named_enum_kind_id("align");
        let align = context.create_enum_attribute(align_attr_id, 32);
        function.add_attribute(AttributeLoc::Param(i), align);
    }
}

/// Add function attributes to kernel function.
fn add_kernel_function_attributes<'ctx>(function: inkwell::values::FunctionValue<'ctx>, context: &'ctx Context) {
    let nounwind_id = inkwell::attributes::Attribute::get_named_enum_kind_id("nounwind");
    let nounwind = context.create_enum_attribute(nounwind_id, 0);
    function.add_attribute(AttributeLoc::Function, nounwind);

    let alwaysinline_id = inkwell::attributes::Attribute::get_named_enum_kind_id("alwaysinline");
    let alwaysinline = context.create_enum_attribute(alwaysinline_id, 0);
    function.add_attribute(AttributeLoc::Function, alwaysinline);

    let no_trap_math = context.create_string_attribute("no-trapping-math", "true");
    function.add_attribute(AttributeLoc::Function, no_trap_math);

    let no_builtins = context.create_string_attribute("no-builtins", "");
    function.add_attribute(AttributeLoc::Function, no_builtins);
}

/// Find all nodes that are in a REDUCE's source subgraph.
///
/// REDUCE operations need to set up their loop counter BEFORE evaluating their source.
/// This function identifies nodes that depend on Reduce RANGEs so they can be skipped
/// in the main codegen loop - REDUCE will handle them when it evaluates its source.
///
/// Returns a set of UOp IDs to skip in the main loop.
fn find_reduce_source_nodes(nodes: &[Arc<UOp>]) -> std::collections::HashSet<u64> {
    use std::collections::HashSet;

    // First, find all REDUCE nodes and collect their source subgraph nodes
    let mut reduce_src_ids: HashSet<u64> = HashSet::new();

    for node in nodes {
        if let Op::Reduce { src, ranges, .. } = node.op() {
            // Add all nodes in the source subgraph (excluding buffers/defines)
            collect_source_subgraph(src, &mut reduce_src_ids);

            // Add reduce ranges (but not their end values - those are constants)
            for range in ranges {
                if let Op::Range { axis_type: morok_ir::AxisType::Reduce, .. } = range.op() {
                    reduce_src_ids.insert(range.id);
                }
            }
        }
    }

    reduce_src_ids
}

/// Recursively collect all nodes in a subgraph, excluding buffers and defines.
fn collect_source_subgraph(uop: &Arc<UOp>, ids: &mut std::collections::HashSet<u64>) {
    // Don't re-process
    if ids.contains(&uop.id) {
        return;
    }

    // Skip buffer/define nodes - they're handled separately
    match uop.op() {
        Op::Buffer { .. } | Op::DefineGlobal(_) | Op::DefineLocal(_) | Op::DefineVar { .. } => {
            return;
        }
        // Skip constants - they can be generated anywhere
        Op::Const(_) => {
            return;
        }
        _ => {}
    }

    ids.insert(uop.id);

    // Recurse into sources
    for src in uop.op().sources() {
        collect_source_subgraph(&src, ids);
    }
}
