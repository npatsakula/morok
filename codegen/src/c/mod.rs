//! C source code generation backend.
//!
//! Generates C source code from linearized UOp IR, suitable for compilation
//! with `clang -shared -O2` and loading via `dlopen`.
//!
//! # Kernel Signature
//!
//! Emits a single function with typed `restrict` pointer params and const variable params:
//!
//! ```c
//! void kernel(float* restrict data0, const int N) { /* body */ }
//! ```

mod amx;
pub mod ops;
pub mod types;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use morok_ir::pattern::TypedPatternMatcher;
use morok_ir::{Op, prelude::*};

use crate::common::{is_output_buffer, validate_custom_template_strict};
use crate::{BufferArg, Error, RenderedKernel, Result};

use self::ops::{CContext, count_references, render_uop};
use self::types::{c_const, c_dtype, c_reduce_identity, c_vconst, collect_vector_typedefs};

/// C source code renderer for CPU execution via clang.
pub struct CRenderer;

impl CRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::Renderer for CRenderer {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
        let kernel_name = name.unwrap_or("kernel");

        let nodes: Vec<Arc<UOp>> = match uop.op() {
            Op::Linear { ops } => ops.iter().cloned().collect(),
            other => {
                return Err(Error::InvalidGraph { reason: format!("C renderer expects LINEAR input, got {other:?}") });
            }
        };

        for (i, node) in nodes.iter().enumerate() {
            tracing::debug!(position = i, op = node.op().as_ref(), id = node.id, "c linearized node");
            match node.op() {
                Op::Custom { deps, code } | Op::CustomI { deps, code } => {
                    validate_custom_template_strict(code, deps.len())?;
                }
                _ => {}
            }
        }

        // Collect buffers and variables from linearized stream
        let mut buffers: Vec<Arc<UOp>> = Vec::new();
        let mut variables: Vec<Arc<UOp>> = Vec::new();

        for node in &nodes {
            match node.op() {
                Op::Param { device: None, .. } => buffers.push(node.clone()),
                Op::DefineVar { .. } => variables.push(node.clone()),
                _ => {}
            }
        }

        buffers.sort_by_key(|b| if let Op::Param { slot, device: None, .. } = b.op() { *slot } else { usize::MAX });

        // Build buffer args metadata
        let mut buffer_args: Vec<BufferArg> = Vec::new();
        for (i, buf) in buffers.iter().enumerate() {
            if let Op::Param { slot, device: None, .. } = buf.op() {
                let is_output = is_output_buffer(buf, &nodes);
                buffer_args.push(BufferArg { index: *slot, name: format!("data{i}"), dtype: buf.dtype(), is_output });
            }
        }

        // Build var_names
        let mut var_names: Vec<String> = Vec::new();
        for var in &variables {
            if let Op::DefineVar { name, .. } = var.op() {
                var_names.push(name.clone());
            }
        }
        // Count references for SSA inlining decisions
        let ref_counts = count_references(&nodes);
        let scope_escaping = find_scope_escaping_vars(&nodes, &ref_counts);
        let mut ctx = CContext::new(ref_counts, scope_escaping);

        // === Build C source ===
        let mut code_lines: Vec<String> = Vec::new();

        // Includes
        code_lines.push("#include <stdbool.h>".to_string());
        code_lines.push("".to_string());

        // Vector typedefs
        let typedefs = collect_vector_typedefs(&nodes);
        for td in &typedefs {
            code_lines.push(td.clone());
        }
        if !typedefs.is_empty() {
            code_lines.push("".to_string());
        }

        // WMMA (AMX) defines and static functions
        let wmma_defines = amx::collect_wmma_defines(&nodes);
        for def in &wmma_defines {
            code_lines.push(def.clone());
        }
        if !wmma_defines.is_empty() {
            code_lines.push("".to_string());
        }

        // Build typed function params
        let mut params: Vec<String> = Vec::new();

        // Buffer parameters
        for (i, buf) in buffers.iter().enumerate() {
            let buf_dtype = buf.dtype();
            let elem_type = match &buf_dtype {
                DType::Ptr { base, .. } => c_dtype(base),
                _ => c_dtype(&buf_dtype),
            };
            let name = format!("data{i}");
            params.push(format!("{elem_type}* restrict {name}"));
            ctx.register(buf.id, name);
        }

        // Variable parameters
        for var in &variables {
            if let Op::DefineVar { name, .. } = var.op() {
                let var_dtype = &var.dtype();
                let c_type = c_dtype(var_dtype);
                params.push(format!("const {c_type} {name}"));
                ctx.register(var.id, name.clone());
            }
        }

        // Function signature
        code_lines.push(format!("void {kernel_name}({}) {{", params.join(", ")));

        // Local memory allocations (stack arrays on CPU)
        for node in &nodes {
            if let Op::DefineLocal(id) = node.op() {
                let (base, size) = match node.dtype() {
                    DType::Ptr { base, size, .. } => (c_dtype(&base), size.unwrap_or(1)),
                    other => (c_dtype(&other), 1),
                };
                let name = format!("local{id}");
                code_lines.push(format!("  {base} {name}[{size}];"));
                ctx.register(node.id, name);
            }
        }

        code_lines.push("".to_string());

        // Reduction accumulator declarations (need to be in outer scope)
        for node in &nodes {
            if let Op::Reduce { reduce_op, ranges, .. } = node.op() {
                if ranges.is_empty() {
                    continue;
                }
                let dtype = &node.dtype();
                let c_type = c_dtype(dtype);
                let identity = c_reduce_identity(*reduce_op, dtype);
                let acc_name = format!("acc{}", node.id);
                code_lines.push(format!("  {c_type} {acc_name} = {identity};"));
                // Pre-register so the ops.rs render_uop finds it
                ctx.register(node.id, acc_name);
            }
        }

        // Register constants
        for node in &nodes {
            match node.op() {
                Op::Const(cv) => {
                    let val = c_const(&cv.0, &node.dtype());
                    ctx.register(node.id, val);
                }
                Op::VConst { values } => {
                    let val = c_vconst(values, &node.dtype());
                    ctx.register(node.id, val);
                }
                _ => {}
            }
        }

        // Pre-register range variable names
        for node in &nodes {
            if let Op::Range { axis_id, .. } = node.op() {
                let name = format!("ridx{}", axis_id.value());
                ctx.register(node.id, name);
            }
        }

        // Render all instructions
        // Skip NOOP and GROUP — they are structural no-ops (Tinygrad cstyle.py:175)
        let mut kernel_body: Vec<String> = Vec::new();
        for node in &nodes {
            if matches!(node.op(), Op::Noop | Op::Group { .. }) {
                // Register with empty string so downstream UNROLL/CONTRACT can alias them.
                // Matches LLVM backend behavior — these are structural no-ops.
                ctx.register(node.id, String::new());
                continue;
            }
            render_uop(node, &mut ctx, &mut kernel_body);
            if let Some(err) = ctx.take_error() {
                return Err(err);
            }
        }

        // Emit hoisted declarations for scope-escaping variables (before kernel body)
        if !ctx.hoisted_declarations.is_empty() {
            code_lines.append(&mut ctx.hoisted_declarations);
        }
        code_lines.extend(kernel_body);
        code_lines.push("}".to_string());
        code_lines.push("".to_string());

        let code = code_lines.join("\n");

        tracing::debug!(generated_c = code, "c codegen: final generated code");

        let mut result = RenderedKernel::new(code, kernel_name.to_string());
        result.buffer_args = buffer_args;
        result.var_names = var_names;

        Ok(result)
    }

    fn backend_name(&self) -> &str {
        "clang"
    }

    fn decompositor(&self) -> Option<TypedPatternMatcher<()>> {
        // C uses __builtin_ math functions (sqrt, exp, sin, etc.) — no decomposition needed.
        // Threefry is handled by XOR in render.
        None
    }
}

/// Find variables that escape their declaration scope.
///
/// Walks the linearized instruction list tracking scope depth. A variable "escapes"
/// if it's defined at a deeper scope than where it's used. Returns the set of UOp IDs
/// that need function-scope declarations to avoid "use of undeclared identifier" errors.
///
/// This handles the case where pm_decomp creates sibling ENDs that share sub-DAG nodes.
/// The linearizer places the shared node inside one loop, but another consumer is outside.
fn find_scope_escaping_vars(nodes: &[Arc<UOp>], ref_counts: &HashMap<u64, usize>) -> HashSet<u64> {
    let mut depth = 0usize;
    let mut def_depth: HashMap<u64, usize> = HashMap::new();
    let mut min_use_depth: HashMap<u64, usize> = HashMap::new();

    for node in nodes {
        // Track scope depth changes
        match node.op() {
            Op::Range { .. } | Op::If { .. } => {
                // Definition of this node is at current depth (before entering)
                if ref_counts.get(&node.id).copied().unwrap_or(0) > 1 {
                    def_depth.entry(node.id).or_insert(depth);
                }
                // Record usages of sources at current depth
                for src in node.op().sources() {
                    min_use_depth.entry(src.id).and_modify(|d| *d = (*d).min(depth)).or_insert(depth);
                }
                depth += 1;
                continue;
            }
            Op::End { .. } | Op::EndIf { .. } => {
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }

        // Record definition depth for multi-use values
        if ref_counts.get(&node.id).copied().unwrap_or(0) > 1 {
            def_depth.entry(node.id).or_insert(depth);
        }

        // Record minimum usage depth for all source operands
        for src in node.op().sources() {
            min_use_depth.entry(src.id).and_modify(|d| *d = (*d).min(depth)).or_insert(depth);
        }
    }

    // Variables where any use is at a shallower depth than definition
    def_depth
        .into_iter()
        .filter(|(id, def_d)| min_use_depth.get(id).copied().unwrap_or(*def_d) < *def_d)
        .map(|(id, _)| id)
        .collect()
}

/// Public render function for the C backend.
pub fn render(uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
    let renderer = CRenderer::new();
    crate::Renderer::render(&renderer, uop, name)
}
