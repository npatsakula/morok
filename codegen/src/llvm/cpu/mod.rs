//! CPU-specific LLVM code generation.
//!
//! This module provides the CPU renderer which generates LLVM IR for CPU execution.

pub mod ops;

use inkwell::attributes::AttributeLoc;
use inkwell::context::Context;
use inkwell::module::Module;
use snafu::{OptionExt, ResultExt};
use std::sync::Arc;
use tracing::{debug, trace};

use morok_ir::{Op, UOp};

use crate::common::collect_buffers_and_vars;
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
    ///
    /// Returns `(module, var_names, thread_count)` where:
    /// - var_names: variable names in order (including thread_id if threading)
    /// - thread_count > 1 indicates a threaded kernel
    fn render_to_module(&self, uop: &Arc<UOp>, name: &str) -> Result<(Module<'ctx>, Vec<String>, usize)> {
        let module = self.context.create_module(name);
        let builder = self.context.create_builder();

        // Collect all buffers and variables from the graph
        let (buffers, variables) = collect_buffers_and_vars(uop);

        // Detect Thread ranges - these become dispatch dimensions, not loops
        // Thread ranges use thread_id parameter instead of creating loops
        let nodes = uop.toposort();
        let thread_info: Option<(Arc<UOp>, usize)> = nodes.iter().find_map(|n| {
            if let Op::Range { axis_type: morok_ir::AxisType::Thread, end, .. } = n.op() {
                if let Op::Const(cv) = end.op() {
                    if let morok_ir::ConstValue::Int(count) = cv.0 {
                        return Some((n.clone(), count as usize));
                    }
                }
            }
            None
        });

        let has_threading = thread_info.is_some();
        let thread_count = thread_info.as_ref().map(|(_, c)| *c).unwrap_or(1);

        // Calculate total vars count (variables + thread_id if threading)
        let vars_count = variables.len() + if has_threading { 1 } else { 0 };

        debug!(
            num_buffers = buffers.len(),
            num_variables = variables.len(),
            has_threading,
            vars_count,
            "Collected kernel parameters"
        );

        // Create kernel function signature: void kernel(ptr %args, ptr %vars)
        // - args: pointer to array of buffer pointers
        // - vars: pointer to array of i64 values (variables + optional thread_id)
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        let i64_type = self.context.i64_type();

        let param_types: Vec<inkwell::types::BasicMetadataTypeEnum> = vec![ptr_type.into(), ptr_type.into()];

        let fn_type = self.context.void_type().fn_type(&param_types, false);
        let kernel_function = module.add_function(name, fn_type, None);

        // Add function attributes
        add_kernel_function_attributes(kernel_function, self.context);

        // Create entry block for kernel
        let entry_block = self.context.append_basic_block(kernel_function, "entry");
        builder.position_at_end(entry_block);

        // Get args and vars array pointers
        let args_param = kernel_function
            .get_nth_param(0)
            .context(InvalidFunctionParameterSnafu { index: 0u32 })?
            .into_pointer_value();
        args_param.set_name("args");

        let vars_param = kernel_function
            .get_nth_param(1)
            .context(InvalidFunctionParameterSnafu { index: 1u32 })?
            .into_pointer_value();
        vars_param.set_name("vars");

        // Create ValueMap and populate with loaded buffer and variable values
        let mut values = ValueMap::new();

        // Load buffer pointers from args array
        for (i, buffer_uop) in buffers.iter().enumerate() {
            let index = i64_type.const_int(i as u64, false);
            let ptr_to_ptr = unsafe {
                builder.build_gep(ptr_type, args_param, &[index], &format!("buf{}_ptr", i)).context(BuildGepSnafu)?
            };
            let buffer_ptr = builder.build_load(ptr_type, ptr_to_ptr, &format!("buf{}", i)).context(BuildLoadSnafu)?;

            // Add noalias metadata via assume (buffer pointers don't alias)
            values.insert(buffer_uop.id, buffer_ptr);
        }

        // Load variable values from vars array
        for (i, var_uop) in variables.iter().enumerate() {
            let index = i64_type.const_int(i as u64, false);
            let var_ptr = unsafe {
                builder.build_gep(i64_type, vars_param, &[index], &format!("var{}_ptr", i)).context(BuildGepSnafu)?
            };
            let var_name = if let Op::DefineVar { name, .. } = var_uop.op() { name.as_str() } else { "var" };
            let var_val = builder.build_load(i64_type, var_ptr, var_name).context(BuildLoadSnafu)?;
            values.insert(var_uop.id, var_val);
        }

        // Load thread_id from vars array if threading is used
        // thread_id is at position variables.len() in the vars array
        if let Some((thread_range, _)) = &thread_info {
            let index = i64_type.const_int(variables.len() as u64, false);
            let thread_id_ptr =
                unsafe { builder.build_gep(i64_type, vars_param, &[index], "thread_id_ptr").context(BuildGepSnafu)? };
            let thread_id_val = builder.build_load(i64_type, thread_id_ptr, "thread_id").context(BuildLoadSnafu)?;

            // Store thread_id for Thread range lookup by axis_id
            if let Op::Range { axis_id, .. } = thread_range.op() {
                values.insert_range(axis_id.value(), thread_id_val);
            }
            // Also store by UOp ID for direct lookups
            values.insert(thread_range.id, thread_id_val);
        }

        // Build var_names list for runtime (variables sorted by name + optional thread_id)
        let mut var_names: Vec<String> = variables
            .iter()
            .filter_map(|v| if let Op::DefineVar { name, .. } = v.op() { Some(name.clone()) } else { None })
            .collect();
        if has_threading {
            var_names.push("thread_id".to_string());
        }

        // Walk the UOp graph in topological order (already computed above for thread detection)
        // Note: nodes was already computed above for thread detection

        debug!(num_nodes = nodes.len(), "walking toposort");

        // Pre-pass: Identify nodes that are in REDUCE source subgraphs
        // These nodes should NOT be processed in the main loop - REDUCE will handle them
        // This is critical because REDUCE needs to set up the loop counter BEFORE
        // its source is evaluated (e.g., INDEX needs the loop counter, not the end value)
        let reduce_src_nodes = find_reduce_source_nodes(&nodes);

        // Pre-pass: Process OUTER/GLOBAL/LOOP RANGE ops first to create loops
        // This ensures loop counters are available before other operations that use them.
        // Loops are created directly from RANGE ops (Tinygrad approach).
        for node in &nodes {
            if let Op::Range { axis_type, .. } = node.op()
                && matches!(
                    axis_type,
                    morok_ir::AxisType::Outer | morok_ir::AxisType::Global | morok_ir::AxisType::Loop
                )
            {
                trace!(range.id = node.id, axis_type = ?axis_type, "Pre-pass: Processing loop RANGE");
                ops::codegen_uop(node, self.context, &module, &builder, &mut values)?;
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

        // Close all remaining outer loops (those not closed by END operations)
        // END operations for Outer ranges are excluded from kernel bodies (Tinygrad pattern),
        // so outer loops must be closed here at the end.
        // Use take_loop to remove them, preventing any double-close issues.
        let remaining_ids = values.remaining_loop_ids();
        for loop_id in remaining_ids.into_iter().rev() {
            if let Some(loop_ctx) = values.take_loop(loop_id) {
                crate::llvm::common::loop_gen::close_loop(&builder, &loop_ctx)?;
            }
        }

        // Return void
        builder.build_return(None).context(BuildReturnSnafu)?;

        // No bootstrap function needed - kernel directly takes (ptr args, ptr vars)

        // Dump IR at trace level
        trace!(llvm.ir = %module.to_string(), "llvm ir before verification");

        // Verify the module
        module.verify().map_err(|err| Error::ModuleVerification { message: err.to_string() })?;

        Ok((module, var_names, thread_count))
    }
}

impl<'ctx> Renderer for CpuLlvmRenderer<'ctx> {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> crate::Result<RenderedKernel> {
        let kernel_name = name.unwrap_or("kernel");
        let (module, var_names, thread_count) = self.render_to_module(uop, kernel_name)?;
        let ir_string = module.print_to_string().to_string();

        let mut rendered = RenderedKernel::new(ir_string, kernel_name.to_string());

        // Set variable names for runtime to populate vars array
        rendered.var_names = var_names;

        // Set global_size for threaded kernels
        // This will be passed through to runtime for parallel dispatch
        if thread_count > 1 {
            rendered.global_size = Some([thread_count, 1, 1]);
            rendered.local_size = Some([1, 1, 1]);
        }

        Ok(rendered)
    }

    fn backend_name(&self) -> &str {
        "llvm"
    }

    fn decompositor(&self) -> Option<morok_ir::pattern::PatternMatcher<()>> {
        None
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

            // Add Reduce AND Loop type ranges (OUTER is handled by BIND)
            // These need to be skipped in main loop - REDUCE will handle them
            for range in ranges {
                if let Op::Range { axis_type, .. } = range.op()
                    && matches!(axis_type, morok_ir::AxisType::Reduce | morok_ir::AxisType::Loop)
                {
                    reduce_src_ids.insert(range.id);
                }
            }
        }
    }

    reduce_src_ids
}

/// Recursively collect all nodes in a subgraph, excluding buffers, defines, and REDUCEs.
///
/// IMPORTANT: We stop at REDUCE nodes because nested REDUCEs need to be evaluated
/// separately (they create their own loops).
///
/// INDEX nodes ARE included in the skip set - they must be evaluated by REDUCE
/// after it sets up the loop counters. However, INDEX results should not be cached
/// since they depend on loop context (handled by require_value not caching them).
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
        // Stop at REDUCE nodes - they have their own source subgraphs
        // that should be handled separately. Nested REDUCEs must be evaluated
        // BEFORE the outer REDUCE's loop, not inside it.
        Op::Reduce { .. } => {
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
