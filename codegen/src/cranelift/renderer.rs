//! Cranelift IR renderer implementation.

use std::collections::HashMap;
use std::sync::Arc;

use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, Signature, UserFuncName, Value, types as cl_types};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

use crate::{RenderedKernel, Result};
use morok_ir::{Op, UOp};

use super::helpers::LoopContext;
use super::ops::codegen_uop;
use crate::common::collect_buffers_and_vars;

/// Cranelift code generator for CPU execution.
pub struct CraneliftRenderer;

impl CraneliftRenderer {
    pub fn new() -> Self {
        Self
    }

    /// Render a UOp graph into Cranelift IR text.
    pub fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
        let kernel_name = name.unwrap_or("kernel");
        let impl_name = format!("{}_impl", kernel_name);

        // Collect buffers and variables
        let (buffers, variables) = collect_buffers_and_vars(uop);
        let buffer_count = buffers.len();
        let var_names: Vec<String> = variables
            .iter()
            .filter_map(|v| if let Op::DefineVar { name, .. } = v.op() { Some(name.clone()) } else { None })
            .collect();

        // =========================================================================
        // Generate kernel_impl: takes individual buffer pointers + variables
        // =========================================================================
        let mut impl_sig = Signature::new(CallConv::SystemV);

        // Add buffer pointer parameters (i64)
        for _ in &buffers {
            impl_sig.params.push(AbiParam::new(cl_types::I64));
        }

        // Add variable parameters (i64)
        for _ in &variables {
            impl_sig.params.push(AbiParam::new(cl_types::I64));
        }

        // Create the implementation function
        let mut impl_func = Function::with_name_signature(UserFuncName::testcase(&impl_name), impl_sig);

        // Create function builder context
        let mut func_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut impl_func, &mut func_ctx);

            // Create entry block
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Create value map for UOp IDs to Cranelift Values
            let mut values: HashMap<u64, Value> = HashMap::new();

            // Map buffer parameters
            for (i, buffer_uop) in buffers.iter().enumerate() {
                let param = builder.block_params(entry_block)[i];
                values.insert(buffer_uop.id, param);
            }

            // Map variable parameters (after buffers)
            for (i, var_uop) in variables.iter().enumerate() {
                let param_idx = buffer_count + i;
                let param = builder.block_params(entry_block)[param_idx];
                values.insert(var_uop.id, param);
            }

            // Create context for loop tracking
            let mut loop_contexts: HashMap<u64, LoopContext> = HashMap::new();

            // Generate code for all nodes in topological order
            let nodes = uop.toposort();
            for node in &nodes {
                codegen_uop(node, &mut builder, &mut values, &mut loop_contexts)?;
            }

            // Return void
            builder.ins().return_(&[]);

            // Finalize the function
            builder.finalize();
        }

        // Convert impl function to Cranelift IR text
        let impl_ir_text = impl_func.display().to_string();

        // Serialize metadata for runtime
        // NOTE: We only emit kernel_impl as CLIF text.
        // The runtime will programmatically build the bootstrap function
        // that loads buffer pointers from an array and calls kernel_impl.
        let metadata = KernelMetadata { name: kernel_name.to_string(), buffer_count, var_names: var_names.clone() };

        // Combine IR and metadata
        let combined = format!("{}METADATA:{}", impl_ir_text, serde_metadata(&metadata));

        let mut rendered = RenderedKernel::new(combined, kernel_name.to_string());
        rendered.var_names = var_names;
        Ok(rendered)
    }
}

impl Default for CraneliftRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::Renderer for CraneliftRenderer {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
        CraneliftRenderer::render(self, uop, name)
    }

    fn backend_name(&self) -> &str {
        "cranelift"
    }

    fn decompositor(&self) -> Option<morok_ir::pattern::TypedPatternMatcher<()>> {
        // Cranelift doesn't have native transcendentals - use all decomposition patterns
        Some(morok_ir::decompositions::all_decomposition_patterns())
    }
}

/// Metadata for kernel execution.
#[derive(Debug, Clone)]
struct KernelMetadata {
    name: String,
    buffer_count: usize,
    var_names: Vec<String>,
}

/// Simple metadata serialization.
fn serde_metadata(meta: &KernelMetadata) -> String {
    format!("{}:{}:{}", meta.name, meta.buffer_count, meta.var_names.join(","))
}

/// Parse metadata from combined string.
pub fn parse_metadata(combined: &str) -> Option<(String, String, usize, Vec<String>)> {
    let parts: Vec<&str> = combined.splitn(2, "METADATA:").collect();
    if parts.len() != 2 {
        return None;
    }

    let ir_text = parts[0].to_string();
    let meta_parts: Vec<&str> = parts[1].split(':').collect();
    if meta_parts.len() < 2 {
        return None;
    }

    let name = meta_parts[0].to_string();
    let buffer_count: usize = meta_parts[1].parse().ok()?;
    let var_names: Vec<String> = if meta_parts.len() > 2 && !meta_parts[2].is_empty() {
        meta_parts[2].split(',').map(|s| s.to_string()).collect()
    } else {
        Vec::new()
    };

    Some((ir_text, name, buffer_count, var_names))
}
