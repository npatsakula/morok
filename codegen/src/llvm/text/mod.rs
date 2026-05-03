//! Text-based LLVM IR code generation (main entry point).
//!
//! This module generates LLVM IR as plain strings using `format!` macros,
//! following Tinygrad's approach in `renderer/llvmir.py`.
//!
//! # Kernel Signature
//!
//! Generates a single function with direct typed parameters and `noalias align 32`
//! buffer annotations:
//! ```llvm
//! define void @kernel(ptr noalias align 32 %buf0, ..., i32 %N) #0 { ... }
//! ```

use std::sync::Arc;

use morok_ir::pattern::TypedPatternMatcher;
use morok_ir::{Op, prelude::*};

use crate::common::is_output_buffer;
use crate::llvm::common::{RenderContext, ldt};
use crate::llvm::cpu::{reduce_identity, render_uop};
use crate::{BufferArg, Error, RenderedKernel, Renderer, Result};

/// Text-based LLVM IR renderer.
///
/// Generates LLVM IR as strings, suitable for compilation via external clang.
/// Produces a single function with direct typed parameters.
pub struct LlvmTextRenderer;

impl LlvmTextRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LlvmTextRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for LlvmTextRenderer {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
        let kernel_name = name.unwrap_or("kernel");

        let nodes: Vec<Arc<UOp>> = match uop.op() {
            Op::Linear { ops } => ops.iter().cloned().collect(),
            other => {
                return Err(Error::InvalidGraph {
                    reason: format!("LLVM text renderer expects LINEAR input, got {other:?}"),
                });
            }
        };

        for (i, node) in nodes.iter().enumerate() {
            tracing::debug!(position = i, op = node.op().as_ref(), id = node.id, "linearized node");
            if matches!(node.op(), Op::Custom { .. } | Op::CustomI { .. }) {
                return Err(Error::InvalidGraph {
                    reason: format!(
                        "LLVM backend does not support CUSTOM/CUSTOMI templates (op id {}); use C backend for custom templates",
                        node.id
                    ),
                });
            }
        }

        let mut ctx = RenderContext::new();
        let mut kernel: Vec<String> = Vec::new();
        let mut buffer_args: Vec<BufferArg> = Vec::new();
        let mut var_names: Vec<String> = Vec::new();

        let mut buffers: Vec<Arc<UOp>> = Vec::new();
        let mut variables: Vec<Arc<UOp>> = Vec::new();

        for node in &nodes {
            match node.op() {
                Op::Param { device: None, .. } => {
                    buffers.push(node.clone());
                }
                Op::DefineVar { .. } => {
                    variables.push(node.clone());
                }
                _ => {}
            }
        }

        buffers.sort_by_key(|b| if let Op::Param { slot, device: None, .. } = b.op() { *slot } else { usize::MAX });

        for (i, buf) in buffers.iter().enumerate() {
            if let Op::Param { slot, device: None, .. } = buf.op() {
                let is_output = is_output_buffer(buf, &nodes);
                buffer_args.push(BufferArg { index: *slot, name: format!("data{i}"), dtype: buf.dtype(), is_output });
            }
        }

        for var in &variables {
            if let Op::DefineVar { name, .. } = var.op() {
                var_names.push(name.clone());
            }
        }
        // -- Build function parameters --
        let mut inner_params: Vec<String> = Vec::new();

        // Buffer pointer parameters
        for (i, buf) in buffers.iter().enumerate() {
            inner_params.push(format!("ptr noalias align 32 %buf{i}"));
            ctx.register(buf.id, format!("%buf{i}"));
        }

        // Variable parameters
        for var in &variables {
            let var_base_name =
                if let Op::DefineVar { name, .. } = var.op() { name.clone() } else { "var".to_string() };
            let var_dtype = var.dtype();
            let var_dtype_str = ldt(&var_dtype);
            inner_params.push(format!("{var_dtype_str} %{var_base_name}"));
            ctx.register(var.id, format!("%{var_base_name}"));
        }

        // -- Build function body --
        kernel.push("  ; Reduction accumulators".to_string());
        for node in &nodes {
            if let Op::Reduce { reduce_op, .. } = node.op() {
                let dtype = ldt(&node.dtype());
                let identity = reduce_identity(*reduce_op, &node.dtype());
                let acc_name = format!("%reduce_{}", node.id);
                kernel.push(format!("  {acc_name} = alloca {dtype}"));
                kernel.push(format!("  store {dtype} {identity}, ptr {acc_name}"));
                ctx.register(node.id, acc_name);
            }
        }
        kernel.push("".to_string());

        for node in &nodes {
            match node.op() {
                Op::Const(cv) => {
                    let val = crate::llvm::common::lconst(&cv.0, &node.dtype());
                    ctx.register(node.id, val);
                }
                Op::VConst { .. } => {
                    ctx.name(node);
                }
                _ => {}
            }
        }

        for node in &nodes {
            if let Op::Range { axis_id, .. } = node.op() {
                let name = format!("%r{}", axis_id.value());
                ctx.register(node.id, name);
            }
        }

        for node in &nodes {
            if matches!(node.op(), Op::Noop | Op::Group { .. }) {
                ctx.register(node.id, String::new());
                continue;
            }
            render_uop(node, &mut ctx, &mut kernel);
            if let Some(err) = ctx.take_error() {
                return Err(err);
            }
        }

        kernel.push("  ret void".to_string());

        let ir = format!(
            r#"; ModuleID = '{kernel_name}'
source_filename = "{kernel_name}"

{intrinsics}

define void @{kernel_name}({inner_params}) #0 {{
entry:
{inner_body}
}}

attributes #0 = {{ nounwind "no-builtins" "no-trapping-math"="true" }}
"#,
            intrinsics = generate_intrinsic_declarations(&kernel),
            inner_params = inner_params.join(", "),
            inner_body = kernel.join("\n"),
        );

        tracing::debug!(generated_code = ir, "llvm codegen: final generated code");

        let mut result = RenderedKernel::new(ir, kernel_name.to_string());
        result.buffer_args = buffer_args;
        result.var_names = var_names;

        Ok(result)
    }

    fn backend_name(&self) -> &str {
        "llvm-text"
    }

    fn decompositor(&self) -> Option<TypedPatternMatcher<()>> {
        None
    }
}

fn mangle_type(llvm_type: &str) -> String {
    match llvm_type {
        "float" => "f32".to_string(),
        "double" => "f64".to_string(),
        "half" => "f16".to_string(),
        "i8" => "i8".to_string(),
        "i16" => "i16".to_string(),
        "i32" => "i32".to_string(),
        "i64" => "i64".to_string(),
        _ if llvm_type.starts_with('<') && llvm_type.ends_with('>') => {
            let inner = &llvm_type[1..llvm_type.len() - 1];
            let parts: Vec<&str> = inner.split(" x ").collect();
            if parts.len() == 2 {
                let count = parts[0].trim();
                let base = mangle_type(parts[1].trim());
                format!("v{count}{base}")
            } else {
                llvm_type.to_string()
            }
        }
        _ => llvm_type.to_string(),
    }
}

fn generate_intrinsic_declarations(kernel: &[String]) -> String {
    let mut decls = Vec::new();
    let kernel_str = kernel.join("\n");

    for intrinsic in &[
        "sqrt", "exp", "exp2", "log", "log2", "sin", "cos", "pow", "fabs", "floor", "ceil", "trunc", "round", "maxnum",
        "minnum", "fmuladd", "erf",
    ] {
        for llvm_type in
            &["float", "double", "half", "<2 x float>", "<4 x float>", "<8 x float>", "<2 x double>", "<4 x double>"]
        {
            let mangled = mangle_type(llvm_type);
            let pattern = format!("@llvm.{intrinsic}.{mangled}");
            if kernel_str.contains(&pattern) {
                let decl = match *intrinsic {
                    "fmuladd" => format!(
                        "declare {llvm_type} @llvm.{intrinsic}.{mangled}({llvm_type}, {llvm_type}, {llvm_type})"
                    ),
                    "pow" | "maxnum" | "minnum" => {
                        format!("declare {llvm_type} @llvm.{intrinsic}.{mangled}({llvm_type}, {llvm_type})")
                    }
                    _ => format!("declare {llvm_type} @llvm.{intrinsic}.{mangled}({llvm_type})"),
                };
                decls.push(decl);
            }
        }
    }

    for bits in &["i8", "i16", "i32", "i64"] {
        let pattern = format!("@llvm.abs.{bits}");
        if kernel_str.contains(&pattern) {
            decls.push(format!("declare {bits} @llvm.abs.{bits}({bits}, i1)"));
        }
    }

    decls.join("\n")
}

pub fn render(uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
    let renderer = LlvmTextRenderer::new();
    renderer.render(uop, name)
}

#[cfg(test)]
#[path = "../../test/unit/llvm_text.rs"]
mod tests;
