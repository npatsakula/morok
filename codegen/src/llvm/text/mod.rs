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
use morok_ir::{AxisType, Op, prelude::*};
use morok_schedule::linearize::{line_rewrite_cleanups, linearize_with_cfg};

use crate::llvm::common::{RenderContext, ldt};
use crate::llvm::cpu::{reduce_identity, render_uop};
use crate::{BufferArg, RenderedKernel, Renderer, Result};

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

        let nodes = linearize_with_cfg(uop.clone());

        // Stage 22: Apply line rewrite cleanups to handle gated INDEX operations.
        // Converts gated STOREs to IF/STORE/ENDIF sequences.
        // Based on Tinygrad's pm_linearize_cleanups (codegen/__init__.py:107-113).
        let nodes = line_rewrite_cleanups(nodes);

        for (i, node) in nodes.iter().enumerate() {
            tracing::debug!(position = i, op = node.op().as_ref(), id = node.id, "linearized node");
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

        let thread_info: Option<(Arc<UOp>, usize)> = nodes.iter().find_map(|n| {
            if let Op::Range { axis_type, end, .. } = n.op()
                && matches!(axis_type, AxisType::Thread)
                && let Op::Const(cv) = end.op()
                && let ConstValue::Int(count) = cv.0
            {
                return Some((n.clone(), count as usize));
            }
            None
        });

        let has_threading = thread_info.is_some();
        let thread_count = thread_info.as_ref().map(|(_, c)| *c).unwrap_or(1);

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
        if has_threading {
            var_names.push("thread_id".to_string());
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

        // Thread ID parameter
        if let Some((thread_range, _)) = &thread_info {
            let range_dtype = thread_range.dtype();
            let range_dtype_str = ldt(&range_dtype);
            inner_params.push(format!("{range_dtype_str} %thread_id"));

            if let Op::Range { axis_id, .. } = thread_range.op() {
                ctx.register(thread_range.id, "%thread_id".to_string());
                ctx.register_range(axis_id.value(), "%thread_id".to_string());
            }
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
            if let Op::Range { axis_id, axis_type, .. } = node.op()
                && !matches!(axis_type, AxisType::Thread)
            {
                let name = format!("%r{}", axis_id.value());
                ctx.register(node.id, name);
            }
        }

        for node in &nodes {
            if matches!(node.op(), Op::Noop | Op::Group { .. }) {
                ctx.register(node.id, String::new());
                continue;
            }
            if let Op::Range { axis_type, .. } = node.op()
                && matches!(axis_type, AxisType::Thread)
            {
                continue;
            }
            render_uop(node, &mut ctx, &mut kernel);
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

        if thread_count > 1 {
            result.global_size = Some([thread_count, 1, 1]);
            result.local_size = Some([1, 1, 1]);
        }

        Ok(result)
    }

    fn backend_name(&self) -> &str {
        "llvm-text"
    }

    fn decompositor(&self) -> Option<TypedPatternMatcher<()>> {
        None
    }
}

fn is_output_buffer(def_global: &Arc<UOp>, nodes: &[Arc<UOp>]) -> bool {
    let buffer_id = def_global.id;

    for node in nodes {
        if let Some(buffer) = node.store_buffer() {
            if buffer.id == buffer_id {
                return true;
            }
            if let Op::Index { buffer: idx_buf, .. } = buffer.op()
                && idx_buf.id == buffer_id
            {
                return true;
            }
        }
    }
    false
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
mod tests {
    use super::*;
    use morok_dtype::{AddrSpace, DType};
    use morok_ir::{BinaryOp, Op};

    #[test]
    fn test_simple_add() {
        let a = UOp::param(0, 1, DType::Float32.ptr(Some(1), AddrSpace::Global), None);
        let b = UOp::param(1, 1, DType::Float32.ptr(Some(1), AddrSpace::Global), None);
        let out = UOp::param(2, 1, DType::Float32.ptr(Some(1), AddrSpace::Global), None);

        let idx = UOp::index_const(0);
        let a_idx = UOp::index().buffer(a.clone()).indices(vec![idx.clone()]).call().unwrap();
        let b_idx = UOp::index().buffer(b.clone()).indices(vec![idx.clone()]).call().unwrap();
        let out_idx = UOp::index().buffer(out.clone()).indices(vec![idx.clone()]).call().unwrap();

        let a_load = UOp::load().buffer(a.clone()).index(a_idx).call();
        let b_load = UOp::load().buffer(b.clone()).index(b_idx).call();

        let add = UOp::new(Op::Binary(BinaryOp::Add, a_load, b_load), DType::Float32);

        let store = out_idx.store(add);
        let sink = UOp::sink(vec![store]);

        let result = render(&sink, Some("test_add")).unwrap();
        println!("{}", result.code);

        assert!(result.code.contains("define void @test_add("));
        assert!(result.code.contains("noalias align 32"));
        assert!(!result.code.contains("_inner"));
        assert!(!result.code.contains("ptr %args"));
        assert!(result.code.contains("fadd"));
        assert!(result.code.contains("load"));
        assert!(result.code.contains("store"));
    }
}
