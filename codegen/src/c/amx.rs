//! AMX (Apple Matrix eXtensions) WMMA helper generation for the C preamble.
//!
//! Scans linearized nodes for `Op::Wmma` and emits AMX inline assembly
//! macros and static wrapper functions for matrix multiply-accumulate.

use std::collections::BTreeSet;
use std::sync::Arc;

use morok_dtype::ScalarDType;
use morok_ir::{Op, UOp, WmmaMetadata};

use super::types::c_dtype;

/// Bit 62: Z output is f32 (for f16->f32 mixed-precision FMA).
const AMX_FMA_Z_F32: u64 = 1 << 62;

/// Bit 62: Load-pair mode for LDX/LDY (load 128 bytes instead of 64).
const AMX_LOAD_PAIR_BIT: u64 = 1 << 62;

/// Collect AMX WMMA macro definitions and static wrapper functions for the C preamble.
///
/// Scans linearized nodes for `Op::Wmma` and emits the necessary AMX inline assembly
/// macros and static wrapper functions that implement the matrix multiply-accumulate
/// via Apple's AMX coprocessor.
pub fn collect_wmma_defines(nodes: &[Arc<UOp>]) -> Vec<String> {
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

    lines.push(r#"#define AMX_SET(imm5) __asm("nop\nnop\nnop\n.word (0x201000+(%0<<5)+%1)" : : "i"(17), "i"(imm5) : "memory")"#.to_string());
    lines.push(r#"#define AMX(op, gpr, btf) __asm(".word (0x201000+(%0 << 5)+0%1-((0%1>>4)*6))" : : "i"(op), "r"((unsigned long long)(gpr)+(btf)) : "memory")"#.to_string());

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
/// FMAs to compute a 2x2 grid of output tiles in one call.
///
/// # Mixed-Precision Support
///
/// For f16*f16->f32, sets bit 62 in FMA encoding to produce f32 accumulator output.
fn render_amx_wmma_function(metadata: &WmmaMetadata) -> String {
    let (n, m, _k) = metadata.dims;
    let (tile_y_count, tile_x_count) = metadata.tile_grid;
    let use_tile_grid = tile_x_count > 1 || tile_y_count > 1;

    let in_scalar = c_dtype(&metadata.dtype_in.scalar_dtype());
    let out_type = format!("{}{}", in_scalar, n * m);
    let a_type = format!("{}{}", in_scalar, n);
    let b_type = format!("{}{}", in_scalar, m);
    let bytes_per_elem = metadata.dtype_in.bytes();

    let fma_op: u32 = match metadata.dtype_in.base() {
        ScalarDType::Float64 => 10, // fma64
        ScalarDType::Float32 => 12, // fma32
        ScalarDType::Int16 => 14,   // mac16
        ScalarDType::Float16 => 15, // fma16
        _ => 12,
    };

    let fma_flags: u64 =
        if metadata.dtype_in.base() == ScalarDType::Float16 && metadata.dtype_out.base() == ScalarDType::Float32 {
            AMX_FMA_Z_F32
        } else {
            0
        };

    let (ldx_encoding, ldy_encoding) = if use_tile_grid { (AMX_LOAD_PAIR_BIT, AMX_LOAD_PAIR_BIT) } else { (0, 0) };

    let fma_calls = if use_tile_grid {
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
