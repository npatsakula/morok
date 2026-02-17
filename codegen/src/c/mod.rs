//! C source code generation backend.
//!
//! Generates C source code from linearized UOp IR, suitable for compilation
//! with `clang -shared -O2` and loading via `dlopen`.
//!
//! # Kernel Signature
//!
//! ```c
//! void kernel(void** args, long long* vars);
//! ```
//! - `args[i]` = buffer pointer (order from `DefineGlobal` index)
//! - `vars[i]` = i64 variable value (order from `var_names`)
//! - Thread ID as last var when `global_size[0] > 1`

pub mod ops;
pub mod types;

use std::collections::BTreeSet;
use std::sync::Arc;

use morok_ir::pattern::TypedPatternMatcher;
use morok_ir::rewrite::graph_rewrite_bottom_up;
use morok_ir::{AxisType, Op, WmmaMetadata, prelude::*};
use morok_schedule::linearize::{line_rewrite_cleanups, linearize_with_cfg};
use morok_schedule::rangeify::patterns::pm_bool_devectorize;

use crate::{BufferArg, RenderedKernel, Result};

use morok_dtype::ScalarDType;

use self::ops::{CContext, count_references, render_uop};
use self::types::{c_const, c_dtype, c_reduce_identity, c_vconst, collect_vector_typedefs};

/// Bit 62: Z output is f32 (for f16->f32 mixed-precision FMA).
const AMX_FMA_Z_F32: u64 = 1 << 62;

/// Bit 62: Load-pair mode for LDX/LDY (load 128 bytes instead of 64).
const AMX_LOAD_PAIR_BIT: u64 = 1 << 62;

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
        let uop = graph_rewrite_bottom_up(&pm_bool_devectorize(), uop.clone(), &mut ());

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
        code_lines.push("#include <math.h>".to_string());
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
        let wmma_defines = collect_wmma_defines(&nodes);
        for def in &wmma_defines {
            code_lines.push(def.clone());
        }
        if !wmma_defines.is_empty() {
            code_lines.push("".to_string());
        }

        // Function signature
        code_lines.push(format!("void {kernel_name}(void** args, long long* vars) {{"));

        // Buffer pointer casts
        for (i, buf) in buffers.iter().enumerate() {
            let buf_dtype = buf.dtype();
            let elem_type = match &buf_dtype {
                DType::Ptr { base, .. } => c_dtype(base),
                _ => c_dtype(&buf_dtype),
            };
            let name = format!("data{i}");
            code_lines.push(format!("  {elem_type}* {name} = ({elem_type}*)args[{i}];"));
            ctx.register(buf.id, name);
        }

        // Variable loads
        for (i, var) in variables.iter().enumerate() {
            if let Op::DefineVar { name, .. } = var.op() {
                let var_dtype = &var.dtype();
                let c_type = c_dtype(var_dtype);
                if c_type == "long long" {
                    code_lines.push(format!("  long long {name} = vars[{i}];"));
                } else {
                    code_lines.push(format!("  {c_type} {name} = ({c_type})vars[{i}];"));
                }
                ctx.register(var.id, name.clone());
            }
        }

        // Thread ID
        if let Some((thread_range, _)) = &thread_info {
            let thread_idx = variables.len();
            let range_dtype = &thread_range.dtype();
            let c_type = c_dtype(range_dtype);
            if c_type == "long long" {
                code_lines.push(format!("  long long thread_id = vars[{thread_idx}];"));
            } else {
                code_lines.push(format!("  {c_type} thread_id = ({c_type})vars[{thread_idx}];"));
            }

            if let Op::Range { axis_id, .. } = thread_range.op() {
                ctx.register(thread_range.id, "thread_id".to_string());
                let _ = axis_id; // Thread range is registered above
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
        let mut kernel_body: Vec<String> = Vec::new();
        for node in &nodes {
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
        // C has math.h with sqrt, exp, sin, etc. — no decomposition needed
        // for standard transcendentals. Threefry is handled by XOR in render.
        None
    }
}

/// Collect AMX WMMA macro definitions and static wrapper functions for the C preamble.
///
/// Scans linearized nodes for `Op::Wmma` and emits the necessary AMX inline assembly
/// macros and static wrapper functions that implement the matrix multiply-accumulate
/// via Apple's AMX coprocessor.
fn collect_wmma_defines(nodes: &[Arc<UOp>]) -> Vec<String> {
    // Collect unique WMMA signatures: (name, dims, dtype_in, dtype_out)
    let mut seen = BTreeSet::new();
    for node in nodes {
        if let Op::Wmma { metadata, .. } = node.op() {
            seen.insert(metadata.name.clone());
        }
    }

    if seen.is_empty() {
        return Vec::new();
    }

    let mut lines = Vec::new();

    // AMX macros (only emitted once)
    lines.push(r#"#define AMX_SET(imm5) __asm("nop\nnop\nnop\n.word (0x201000+(%0<<5)+%1)" : : "i"(17), "i"(imm5) : "memory")"#.to_string());
    lines.push(r#"#define AMX(op, gpr, btf) __asm(".word (0x201000+(%0 << 5)+0%1-((0%1>>4)*6))" : : "i"(op), "r"((unsigned long long)(gpr)+(btf)) : "memory")"#.to_string());

    // Emit a static wrapper for each unique WMMA signature
    for node in nodes {
        if let Op::Wmma { metadata, .. } = node.op()
            && seen.remove(&metadata.name)
        {
            lines.push(render_amx_wmma_function(metadata));
        }
    }

    lines
}

/// Render a static AMX WMMA wrapper function for a specific matrix multiply configuration.
///
/// Generates a C function that:
/// 1. Initializes AMX state (`AMX_SET(0)`)
/// 2. Loads the accumulator matrix into Z registers
/// 3. Loads A into X register, B into Y register (with load-pair for tile grids)
/// 4. Executes fused multiply-add(s) (multiple FMAs for tile grids)
/// 5. Stores Z registers back to the accumulator
/// 6. Finalizes AMX state (`AMX_SET(1)`)
///
/// # Tile Grid Support
///
/// When `tile_grid > (1,1)`, uses load-pair mode (128-byte loads) and emits multiple
/// FMAs to compute a 2×2 grid of output tiles in one call.
///
/// # Mixed-Precision Support
///
/// For f16×f16→f32, sets bit 62 in FMA encoding to produce f32 accumulator output.
fn render_amx_wmma_function(metadata: &WmmaMetadata) -> String {
    let (n, m, _k) = metadata.dims;
    let (tile_y_count, tile_x_count) = metadata.tile_grid;
    let use_tile_grid = tile_x_count > 1 || tile_y_count > 1;

    let in_scalar = c_dtype(&metadata.dtype_in.scalar_dtype());
    let out_type = format!("{}{}", in_scalar, n * m); // e.g. float256
    let a_type = format!("{}{}", in_scalar, n); // e.g. float16
    let b_type = format!("{}{}", in_scalar, m); // e.g. float16
    let bytes_per_elem = metadata.dtype_in.bytes();

    // AMX instruction opcode selection based on dtype
    let fma_op: u32 = match metadata.dtype_in.base() {
        ScalarDType::Float64 => 10, // fma64
        ScalarDType::Float32 => 12, // fma32
        ScalarDType::Int16 => 14,   // mac16
        ScalarDType::Float16 => 15, // fma16
        _ => 12,
    };

    // Mixed-precision flag: f16 input -> f32 output requires bit 62
    let fma_flags: u64 =
        if metadata.dtype_in.base() == ScalarDType::Float16 && metadata.dtype_out.base() == ScalarDType::Float32 {
            AMX_FMA_Z_F32
        } else {
            0
        };

    // Load-pair mode for tile grids (loads 128 bytes instead of 64)
    let (ldx_encoding, ldy_encoding) = if use_tile_grid { (AMX_LOAD_PAIR_BIT, AMX_LOAD_PAIR_BIT) } else { (0, 0) };

    // Generate FMA encoding(s)
    let fma_calls = if use_tile_grid {
        // Multiple FMAs for tile grid: each FMA targets a different output tile
        let bytes_per_tile_row: usize = 64;
        let mut calls = Vec::new();
        for ty in 0..tile_y_count {
            for tx in 0..tile_x_count {
                let z_row = (ty * tile_x_count + tx) as u64;
                let x_off = (tx * bytes_per_tile_row) as u64;
                let y_off = (ty * bytes_per_tile_row) as u64;
                let encoding = fma_flags | (z_row << 20) | (x_off << 10) | y_off;
                calls.push(format!("  AMX({fma_op}, 0, {encoding}ull);"));
            }
        }
        calls.join("\n")
    } else {
        format!("  AMX({fma_op}, 0, {fma_flags}ull);")
    };

    format!(
        "static {out_type} __{name}({a_type} data1, {b_type} data2, {out_type} data0){{\n  \
         AMX_SET(0);\n  \
         for(int ridx0 = 0; ridx0 < {n}; ridx0++){{ \
         AMX(4, (int *)(&data0), 0ull<<62 | (ridx0*{bytes_per_elem}ull)<<56 | ridx0*64ull); }}\n  \
         AMX(0, (int *)(&data2), {ldx_encoding}ull); \
         AMX(1, (int *)(&data1), {ldy_encoding}ull);\n\
         {fma_calls}\n  \
         for(int ridx0 = 0; ridx0 < {n}; ridx0++){{ \
         AMX(5, (int *)(&data0), 0ull<<62 | (ridx0*{bytes_per_elem}ull)<<56 | ridx0*64ull); }}\n  \
         AMX_SET(1);\n  \
         return data0;\n}}",
        name = metadata.name,
    )
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
