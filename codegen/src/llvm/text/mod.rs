//! Text-based LLVM IR code generation (main entry point).
//!
//! This module generates LLVM IR as plain strings using `format!` macros,
//! following Tinygrad's approach in `renderer/llvmir.py`.
//!
//! # Kernel Signature
//!
//! The generated kernel uses CPU signature:
//! ```llvm
//! void @kernel(ptr %args, ptr %vars)
//! ```
//! - `args`: pointer to array of buffer pointers
//! - `vars`: pointer to array of i64 values (variables + optional thread_id)

use std::sync::Arc;

use morok_ir::pattern::TypedPatternMatcher;
use morok_ir::rewrite::graph_rewrite_bottom_up;
use morok_ir::{AxisType, Op, prelude::*};
use morok_schedule::devectorize::pm_render;
use morok_schedule::linearize::linearize_with_cfg;
use morok_schedule::rangeify::patterns::pm_bool_devectorize;

use crate::llvm::common::{RenderContext, ldt};
use crate::llvm::cpu::{reduce_identity, render_uop};
use crate::{BufferArg, RenderedKernel, Renderer, Result};

/// Text-based LLVM IR renderer.
///
/// Generates LLVM IR as strings, suitable for compilation via inkwell's IR parser.
/// Produces kernel signature: `void @kernel(ptr %args, ptr %vars)`
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

        let uop = graph_rewrite_bottom_up(&pm_render(), uop.clone(), &mut ());

        tracing::debug!(ast_after_pm_render = %uop.tree(), "codegen: after pm_render");

        let uop = graph_rewrite_bottom_up(&pm_bool_devectorize(), uop, &mut ());

        tracing::debug!(ast_after_pm_bool_devectorize = %uop.tree(), "codegen: after pm_bool_devectorize");

        let nodes = linearize_with_cfg(uop);

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
                Op::DefineGlobal(_) => {
                    buffers.push(node.clone());
                }
                Op::DefineVar { .. } => {
                    variables.push(node.clone());
                }
                _ => {}
            }
        }

        buffers.sort_by_key(|b| if let Op::DefineGlobal(id) = b.op() { *id } else { usize::MAX });

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
            if let Op::DefineGlobal(id) = buf.op() {
                let is_output = is_output_buffer(buf, &nodes);
                buffer_args.push(BufferArg { index: *id, name: format!("data{i}"), dtype: buf.dtype(), is_output });
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

        kernel.push("  ; Load buffer pointers from args array".to_string());
        for (i, buf) in buffers.iter().enumerate() {
            let ptr_name = format!("%buf{i}_ptr");
            let buf_name = format!("%buf{i}");
            kernel.push(format!("  {ptr_name} = getelementptr ptr, ptr %args, i64 {i}"));
            kernel.push(format!("  {buf_name} = load ptr, ptr {ptr_name}"));
            ctx.register(buf.id, buf_name);
        }

        kernel.push("  ; Load variable values from vars array".to_string());
        for (i, var) in variables.iter().enumerate() {
            let var_ptr_name = format!("%var{i}_ptr");
            let var_val_name =
                if let Op::DefineVar { name, .. } = var.op() { format!("%{name}") } else { format!("%var{i}") };
            kernel.push(format!("  {var_ptr_name} = getelementptr i64, ptr %vars, i64 {i}"));
            kernel.push(format!("  {var_val_name} = load i64, ptr {var_ptr_name}"));
            ctx.register(var.id, var_val_name);
        }

        if let Some((thread_range, _)) = &thread_info {
            let thread_idx = variables.len();
            kernel.push(format!("  %thread_id_ptr = getelementptr i64, ptr %vars, i64 {thread_idx}"));
            kernel.push("  %thread_id = load i64, ptr %thread_id_ptr".to_string());

            if let Op::Range { axis_id, .. } = thread_range.op() {
                ctx.register(thread_range.id, "%thread_id".to_string());
                ctx.register_range(axis_id.value(), "%thread_id".to_string());
            }
        }

        kernel.push("".to_string());

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
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

{intrinsics}

define void @{kernel_name}(ptr %args, ptr %vars) #0 {{
entry:
{body}
}}

attributes #0 = {{ alwaysinline nounwind "no-builtins" "no-trapping-math"="true" }}
"#,
            intrinsics = generate_intrinsic_declarations(&kernel),
            body = kernel.join("\n")
        );

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
        let a = UOp::define_global(0, DType::Float32.ptr(Some(1), AddrSpace::Global));
        let b = UOp::define_global(1, DType::Float32.ptr(Some(1), AddrSpace::Global));
        let out = UOp::define_global(2, DType::Float32.ptr(Some(1), AddrSpace::Global));

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

        assert!(result.code.contains("define void @test_add(ptr %args, ptr %vars)"));
        assert!(result.code.contains("getelementptr ptr, ptr %args"));
        assert!(result.code.contains("fadd"));
        assert!(result.code.contains("load"));
        assert!(result.code.contains("store"));
    }
}
