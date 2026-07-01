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

use svod_dtype::AmdArch;
use svod_ir::pattern::TypedPatternMatcher;
use svod_ir::{Op, prelude::*};

use crate::common::is_output_buffer;
use crate::llvm::amd;
use crate::llvm::common::{LlvmTarget, RenderContext, ldt};
use crate::llvm::cpu;
use crate::{BufferArg, Error, RenderedKernel, Renderer, Result};

/// Text-based LLVM IR renderer.
///
/// Generates LLVM IR as strings, suitable for compilation via external clang.
/// Produces a single function with direct typed parameters. The active
/// [`LlvmTarget`] selects between the CPU emitter and the AMDGPU emitter
/// (`amdgpu_kernel` ABI, addrspace(3) LDS, amdgcn intrinsics).
pub struct LlvmTextRenderer {
    target: LlvmTarget,
}

impl LlvmTextRenderer {
    /// Renderer for the host CPU target (default for backwards compatibility).
    pub fn new() -> Self {
        Self { target: LlvmTarget::Cpu }
    }

    /// Renderer for an AMD GPU at the named `gfx{family}` target.
    pub fn amd(arch: AmdArch) -> Self {
        Self { target: LlvmTarget::Amd(arch) }
    }

    /// Construct with an explicit target.
    pub fn with_target(target: LlvmTarget) -> Self {
        Self { target }
    }

    pub fn target(&self) -> LlvmTarget {
        self.target
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

        // Instruction-scheduling pass: lower any `sched::pipeline` markers into the
        // gfx9 machine scheduling controls (s_setprio brackets, sched.barrier fences,
        // the attention interleave comb). No-op on non-CDNA targets / unmarked kernels.
        let nodes = crate::llvm::sched::apply_pipeline_scheduling(nodes, self.target);

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
        // WMMA scratch buffers — one alloca + ptrtoint per (A, B, C) operand.
        // Allocas placed in the entry block so LLVM's mem2reg can promote them
        // to vector registers across loop iterations. Without this, the WMMA
        // accumulator is materialized to memory every K iteration.
        //
        // CPU/AMX only: the AMX tensor cores can only load operands from memory,
        // so they need these scratch slots. The AMDGPU path lowers WMMA straight
        // to `llvm.amdgcn.wmma.*` intrinsics over SSA vectors (see `amd::wmma`),
        // so emitting these allocas there is dead IR. Matches tinygrad's
        // `AMDLLVMRenderer`, which only preallocates on the `tc.amx` path.
        let wmma_count = nodes.iter().filter(|n| matches!(n.op(), Op::Wmma { .. })).count();
        if wmma_count > 0 && matches!(self.target, LlvmTarget::Cpu) {
            kernel.push("  ; WMMA AMX scratch buffers".to_string());
            for node in &nodes {
                if let Op::Wmma { a, b, c, .. } = node.op() {
                    for (i, src) in [a, b, c].iter().enumerate() {
                        let dtype = ldt(&src.dtype());
                        let base = format!("%wmma_{}_amx{}", node.id, i);
                        let ptr_name = format!("%wmma_{}_ptr_amx{}", node.id, i);
                        let align = src.dtype().bytes();
                        kernel.push(format!("  {base} = alloca {dtype}, align {align}"));
                        kernel.push(format!("  {ptr_name} = ptrtoint ptr {base} to i64"));
                    }
                }
            }
        }
        kernel.push("".to_string());

        for node in &nodes {
            if matches!(node.op(), Op::Noop | Op::Group { .. }) {
                ctx.register(node.id, String::new());
                continue;
            }
            match self.target {
                LlvmTarget::Cpu => {
                    cpu::render_uop(node, &mut ctx, &mut kernel);
                }
                LlvmTarget::Amd(_) => {
                    amd::render_uop(node, &mut ctx, &mut kernel, self.target);
                }
            }
            if let Some(err) = ctx.take_error() {
                return Err(err);
            }
        }

        kernel.push("  ret void".to_string());

        let abi = match self.target {
            LlvmTarget::Cpu => "void",
            LlvmTarget::Amd(_) => "amdgpu_kernel void",
        };

        let attrs = build_function_attributes(&self.target, &nodes);

        // Module-level prefix:
        //   1. amdgcn intrinsic declarations + CPU intrinsic declarations
        //   2. fp8 helper (AMD-only, only when the kernel uses fp8)
        //   3. addrspace(3) LDS globals from `Op::DefineLocal` (AMD-only)
        let mut module_blocks: Vec<String> = Vec::new();
        module_blocks.push(generate_intrinsic_declarations(&kernel, &self.target));
        if self.target.is_amd()
            && let Some(helper) = amd::ops::fp8_helper_prefix(&nodes)
        {
            module_blocks.push(helper.to_string());
        }
        if !ctx.module_prefix().is_empty() {
            module_blocks.push(ctx.module_prefix().join("\n"));
        }

        // A `declare` can originate from both the auto-scan and a hoisted
        // CUSTOM body line; LLVM forbids redefining a function, so keep only
        // the first occurrence of each identical declaration.
        let module_prefix = dedup_declares(module_blocks.join("\n\n"));

        let target_triple_line = match self.target {
            LlvmTarget::Cpu => String::new(),
            LlvmTarget::Amd(_) => "target triple = \"amdgcn-amd-amdhsa\"\n".to_string(),
        };

        let ir = format!(
            r#"; ModuleID = '{kernel_name}'
source_filename = "{kernel_name}"
{target_triple_line}
{module_prefix}

define {abi} @{kernel_name}({inner_params}) #0 {{
entry:
{inner_body}
}}

attributes #0 = {{ {attrs} }}
"#,
            module_prefix = module_prefix,
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

/// Remove duplicate `declare ...` lines from an assembled module prefix,
/// keeping the first occurrence per **function name** (the `@name` token). Two
/// declares for the same intrinsic with different signatures — e.g. a wave64
/// and a wave32 `@llvm.amdgcn.wmma.*` call in the same kernel — are treated as
/// duplicates so the second is dropped, avoiding clang's "invalid redefinition"
/// error. Non-`declare` lines pass through unchanged.
fn dedup_declares(prefix: String) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<&str> = Vec::new();
    for line in prefix.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("declare ") {
            // Extract the function name: the token between `@` and `(`.
            let key = trimmed
                .find('@')
                .and_then(|at| trimmed[at + 1..].find('(').map(|p| &trimmed[at + 1..at + 1 + p]))
                .unwrap_or(trimmed);
            if !seen.insert(key.to_string()) {
                continue;
            }
        }
        out.push(line);
    }
    out.join("\n")
}

fn generate_intrinsic_declarations(kernel: &[String], target: &LlvmTarget) -> String {
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

    if target.is_amd() {
        // Scalar (non-mangled) amdgcn intrinsics; declared whenever referenced
        // in the kernel body. Source: AMDGPU LLVM intrinsic reference.
        for (pattern, decl) in [
            ("@llvm.amdgcn.s.barrier", "declare void @llvm.amdgcn.s.barrier()"),
            ("@llvm.amdgcn.workgroup.id.x", "declare i32 @llvm.amdgcn.workgroup.id.x()"),
            ("@llvm.amdgcn.workgroup.id.y", "declare i32 @llvm.amdgcn.workgroup.id.y()"),
            ("@llvm.amdgcn.workgroup.id.z", "declare i32 @llvm.amdgcn.workgroup.id.z()"),
            ("@llvm.amdgcn.workitem.id.x", "declare i32 @llvm.amdgcn.workitem.id.x()"),
            ("@llvm.amdgcn.workitem.id.y", "declare i32 @llvm.amdgcn.workitem.id.y()"),
            ("@llvm.amdgcn.workitem.id.z", "declare i32 @llvm.amdgcn.workitem.id.z()"),
            ("@llvm.amdgcn.cvt.f32.fp8", "declare float @llvm.amdgcn.cvt.f32.fp8(i32, i32)"),
            ("@llvm.amdgcn.cvt.f32.bf8", "declare float @llvm.amdgcn.cvt.f32.bf8(i32, i32)"),
            ("@llvm.amdgcn.cvt.pk.fp8.f32", "declare i32 @llvm.amdgcn.cvt.pk.fp8.f32(float, float, i32, i1)"),
            ("@llvm.amdgcn.cvt.pk.bf8.f32", "declare i32 @llvm.amdgcn.cvt.pk.bf8.f32(float, float, i32, i1)"),
            ("@llvm.amdgcn.fmed3.f32", "declare float @llvm.amdgcn.fmed3.f32(float, float, float)"),
        ] {
            if kernel_str.contains(pattern) {
                decls.push(decl.to_string());
            }
        }
        // WMMA / MFMA intrinsics: the signature varies by family and dtype, so
        // we synthesize each `declare` from its call site's operand types. The
        // operands already carry the intrinsic-required wire types (bf16 as
        // i16, fp8 as a packed integer — bitcast in `render_wmma_amd`), so the
        // declaration matches the call by construction. Dedup identical lines
        // (a tiled matmul emits many calls to the same intrinsic).
        for line in kernel.iter() {
            if let Some(decl) = wmma_declaration_from_call(line)
                && !decls.contains(&decl)
            {
                decls.push(decl);
            }
        }
    }

    decls.join("\n")
}

/// Synthesize a `declare` line for a `@llvm.amdgcn.{wmma,mfma}.*` call by
/// echoing the call's argument types. Returns `None` if the line isn't a
/// WMMA/MFMA call site.
fn wmma_declaration_from_call(line: &str) -> Option<String> {
    let needle_wmma = "@llvm.amdgcn.wmma.";
    let needle_mfma = "@llvm.amdgcn.mfma.";
    let needle = if line.contains(needle_wmma) {
        needle_wmma
    } else if line.contains(needle_mfma) {
        needle_mfma
    } else {
        return None;
    };
    // `  %vN = call [fast-math flags] <ret_ty> @llvm.amdgcn.wmma.<rest>(<args>)`
    let call_start = line.find("call ")?;
    let mut after_call = &line[call_start + "call ".len()..];
    // Skip the optional fast-math flags between `call` and the return type (the
    // float-accumulating WMMA/MFMA carries `nsz arcp contract afn`) so the `declare`
    // (which must NOT carry them) parses the real return type, not `nsz …`.
    const FM_FLAGS: &[&str] = &["fast", "nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc"];
    while let Some(tok) = after_call.split_whitespace().next() {
        if FM_FLAGS.contains(&tok) {
            after_call = after_call[tok.len()..].trim_start();
        } else {
            break;
        }
    }
    let ret_end = after_call.find(" @")?;
    let ret_ty = &after_call[..ret_end];
    // Everything below is relative to `after_call` (the flag-stripped slice), not
    // the raw line, so the leading fast-math flags don't offset the positions.
    let name_start = ret_end + 2; // skip " @"
    let paren = after_call[name_start..].find('(')?;
    let intrinsic_name = &after_call[name_start..name_start + paren];
    if !intrinsic_name.starts_with(&needle[1..]) {
        return None;
    }
    // Extract the argument list (between the matching parens).
    let args_start = name_start + paren + 1;
    let args_end = after_call[args_start..].rfind(')')?;
    let args_chunk = &after_call[args_start..args_start + args_end];
    // Pull out types — entries are `<ty> %name` or `<ty> <const>`.
    let mut param_types: Vec<String> = Vec::new();
    let mut depth = 0;
    let mut current = String::new();
    let mut parts: Vec<String> = Vec::new();
    for ch in args_chunk.chars() {
        match ch {
            '<' => {
                depth += 1;
                current.push(ch);
            }
            '>' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                parts.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    for part in parts {
        let trimmed = part.trim();
        // The leading *type* token. A `<…>` vector/aggregate type runs to its
        // matching `>` (the value or name follows it); a scalar type is the
        // token before the first space (`i32 0`, `i1 false`). Splitting on the
        // first space would truncate `<16 x half>` to `<16` — the bug this
        // replaces — since the type itself contains spaces.
        let ty = if trimmed.starts_with('<') {
            let mut depth = 0usize;
            let mut end = trimmed.len();
            for (i, ch) in trimmed.char_indices() {
                match ch {
                    '<' => depth += 1,
                    '>' => {
                        depth -= 1;
                        if depth == 0 {
                            end = i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            &trimmed[..end]
        } else {
            trimmed.split_whitespace().next().unwrap_or(trimmed)
        };
        param_types.push(ty.to_string());
    }
    Some(format!("declare {ret_ty} @{intrinsic_name}({})", param_types.join(", ")))
}

/// Build the per-target `attributes #0` body.
fn build_function_attributes(target: &LlvmTarget, nodes: &[Arc<UOp>]) -> String {
    match target {
        LlvmTarget::Cpu => "nounwind \"no-builtins\" \"no-trapping-math\"=\"true\"".to_string(),
        LlvmTarget::Amd(_) => {
            // Tinygrad `llvmir.py:259-263`: include the upper bound on the
            // local workgroup size so the AMDGPU backend can size scratch
            // allocations / waves correctly.
            let max_l = nodes
                .iter()
                .filter_map(|n| match n.op() {
                    Op::Special { name, end } if name.starts_with('l') => match end.vmax() {
                        svod_ir::ConstValue::Int(v) => Some(*v as u64),
                        svod_ir::ConstValue::UInt(v) => Some(*v),
                        _ => None,
                    },
                    _ => None,
                })
                .product::<u64>()
                .max(1);
            format!(
                "alwaysinline nounwind \"no-builtins\" \"amdgpu-flat-work-group-size\"=\"1,{max_l}\" \
                 \"no-trapping-math\"=\"true\""
            )
        }
    }
}

pub fn render(uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
    let renderer = LlvmTextRenderer::new();
    renderer.render(uop, name)
}

#[cfg(test)]
#[path = "../../test/unit/llvm_text.rs"]
mod tests;
