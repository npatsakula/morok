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

use std::sync::Arc;

use morok_ir::pattern::TypedPatternMatcher;
use morok_ir::rewrite::graph_rewrite_bottom_up;
use morok_ir::{AxisType, Op, prelude::*};
use morok_schedule::linearize::{line_rewrite_cleanups, linearize_with_cfg};
use morok_schedule::rangeify::patterns::pm_bool_devectorize;

use crate::{BufferArg, RenderedKernel, Result};

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

        // Apply pm_bool_devectorize as safety fallback
        let uop = graph_rewrite_bottom_up(pm_bool_devectorize(), uop.clone(), &mut ());

        tracing::debug!(ast_after_pm_bool_devectorize = %uop.tree(), "c codegen: after pm_bool_devectorize");

        // Linearize the UOp DAG
        let nodes = linearize_with_cfg(uop);

        // Apply line rewrite cleanups (gated stores → if/store/endif)
        let nodes = line_rewrite_cleanups(nodes);

        for (i, node) in nodes.iter().enumerate() {
            tracing::debug!(position = i, op = node.op().as_ref(), id = node.id, "c linearized node");
        }

        // Collect buffers and variables from linearized stream
        let mut buffers: Vec<Arc<UOp>> = Vec::new();
        let mut variables: Vec<Arc<UOp>> = Vec::new();

        for node in &nodes {
            match node.op() {
                Op::DefineGlobal(_) => buffers.push(node.clone()),
                Op::DefineVar { .. } => variables.push(node.clone()),
                _ => {}
            }
        }

        buffers.sort_by_key(|b| if let Op::DefineGlobal(id) = b.op() { *id } else { usize::MAX });

        // Detect threading
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

        // Build buffer args metadata
        let mut buffer_args: Vec<BufferArg> = Vec::new();
        for (i, buf) in buffers.iter().enumerate() {
            if let Op::DefineGlobal(id) = buf.op() {
                let is_output = is_output_buffer(buf, &nodes);
                buffer_args.push(BufferArg { index: *id, name: format!("data{i}"), dtype: buf.dtype(), is_output });
            }
        }

        // Build var_names
        let mut var_names: Vec<String> = Vec::new();
        for var in &variables {
            if let Op::DefineVar { name, .. } = var.op() {
                var_names.push(name.clone());
            }
        }
        if has_threading {
            var_names.push("thread_id".to_string());
        }

        // Count references for SSA inlining decisions
        let ref_counts = count_references(&nodes);
        let mut ctx = CContext::new(ref_counts);

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

        // Thread ID parameter
        if let Some((thread_range, _)) = &thread_info {
            let range_dtype = &thread_range.dtype();
            let c_type = c_dtype(range_dtype);
            params.push(format!("const {c_type} thread_id"));
            ctx.register(thread_range.id, "thread_id".to_string());
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
            if let Op::Range { axis_id, axis_type, .. } = node.op()
                && !matches!(axis_type, AxisType::Thread)
            {
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
            if let Op::Range { axis_type, .. } = node.op()
                && matches!(axis_type, AxisType::Thread)
            {
                continue;
            }
            render_uop(node, &mut ctx, &mut kernel_body);
        }

        code_lines.extend(kernel_body);
        code_lines.push("}".to_string());
        code_lines.push("".to_string());

        let code = code_lines.join("\n");

        tracing::debug!(generated_c = code, "c codegen: final generated code");

        let mut result = RenderedKernel::new(code, kernel_name.to_string());
        result.buffer_args = buffer_args;
        result.var_names = var_names;

        if thread_count > 1 {
            result.global_size = Some([thread_count, 1, 1]);
            result.local_size = Some([1, 1, 1]);
        }

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

/// Public render function for the C backend.
pub fn render(uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
    let renderer = CRenderer::new();
    crate::Renderer::render(&renderer, uop, name)
}
