//! AMX (Apple Matrix eXtensions) WMMA support for the MLIR backend.
//!
//! Emits AMX inline assembly via `llvm.inline_asm` operations to perform
//! matrix multiply-accumulate on Apple Silicon.
//!
//! Hoisted path (inside K-reduction loop):
//!   - LDZ from DEFINE_REG alloca before loop (accumulator init)
//!   - LDY/LDX directly from source memory pointers per iteration (no temp buffers)
//!   - STZ to DEFINE_REG alloca after loop (accumulator writeback)
//!
//! Non-hoisted path (fallback): full SET/LDZ/LDX/LDY/FMA/STZ/CLR per call.

use melior::Context;
use melior::dialect::{arith, llvm, ods};
use melior::ir::BlockLike;
use melior::ir::attribute::{IntegerAttribute, StringAttribute, TypeAttribute};
use melior::ir::r#type::IntegerType;
use melior::ir::{Attribute, Block, Location, Type, Value};
use morok_dtype::DType;
use morok_dtype::ScalarDType;
use morok_ir::WmmaMetadata;

use super::ops::const_i64;

/// AMX FMA encoding bits.
const AMX_FMA_Z_F32: u64 = 1 << 62; // Bit 62: Z is f32 (only for fma16 mixed-precision)

/// Validate AMX dtype combination.
///
/// AMX supports the following dtype combinations:
/// - f32 × f32 → f32 (fma32)
/// - f64 × f64 → f64 (fma64)
/// - f16 × f16 → f16 (fma16)
/// - f16 × f16 → f32 (fma16 mixed-precision, bit 62 set)
/// - i16 × i16 → i16 (mac16)
pub(crate) fn validate_amx_dtypes(dtype_in: &DType, dtype_out: &DType) -> crate::Result<()> {
    let (in_base, out_base) = (dtype_in.base(), dtype_out.base());
    match (in_base, out_base) {
        (ScalarDType::Float32, ScalarDType::Float32)
        | (ScalarDType::Float64, ScalarDType::Float64)
        | (ScalarDType::Float16, ScalarDType::Float16)
        | (ScalarDType::Float16, ScalarDType::Float32) // Mixed-precision
        | (ScalarDType::Int16, ScalarDType::Int16) => Ok(()),
        _ => Err(crate::error::Error::MlirError {
            reason: format!("Unsupported AMX dtype: {:?} x {:?} -> {:?}", in_base, in_base, out_base),
        }),
    }
}

/// Compute Z row stride for dtype combination.
///
/// The Z register row stride depends on the dtype combination:
/// - fma64: 8 bytes per row (64 bytes / 8 f64 elements)
/// - fma32: 4 bytes per row (64 bytes / 16 f32 elements)
/// - fma16/mac16: 2 bytes per row (64 bytes / 32 f16/i16 elements)
pub(crate) fn z_row_stride(dtype_in: &DType, dtype_out: &DType) -> usize {
    match (dtype_in.base(), dtype_out.base()) {
        (ScalarDType::Float64, _) => 8,
        (ScalarDType::Float32, _) => 4,
        (ScalarDType::Float16, _) | (ScalarDType::Int16, _) => 2,
        _ => unreachable!("validate_amx_dtypes prevents this"),
    }
}

/// State for an AMX WMMA operation hoisted out of a reduce loop.
///
/// When WMMA is inside a K-reduction loop, we hoist SET/LDZ before the loop
/// and STZ/CLR after it so Z registers accumulate across iterations.
/// A/B inputs are loaded directly from source memory pointers each iteration.
pub struct AmxLoopState<'c> {
    /// The DEFINE_REG alloca pointer for the accumulator (C matrix).
    /// Layout matches Z register rows: row i at bytes [i*64..i*64+63].
    pub acc_alloca: Value<'c, 'c>,
    /// UOp ID of the DEFINE_REG for the accumulator.
    pub acc_reg_id: u64,
    /// WMMA metadata (dims, dtypes, tile_grid).
    pub metadata: WmmaMetadata,
}

/// Bit 62: Load-pair mode for LDX/LDY (load 128 bytes instead of 64).
const AMX_LOAD_PAIR_BIT: u64 = 1 << 62;

/// Load accumulator rows from DEFINE_REG alloca into Z registers.
///
/// Emits `ptrtoint(alloca)` then LDZ for each row with encoding:
///   `(z_row << 56) | (row * 64)`
/// where z_row = i * z_row_stride (row spacing in Z register file).
pub fn render_amx_ldz<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    acc_alloca: Value<'c, 'c>,
    metadata: &WmmaMetadata,
    loc: Location<'c>,
) -> crate::Result<()> {
    validate_amx_dtypes(&metadata.dtype_in, &metadata.dtype_out)?;
    let (n, _m, _k) = metadata.dims;
    let stride = z_row_stride(&metadata.dtype_in, &metadata.dtype_out);
    let i64_type: Type = IntegerType::new(ctx, 64).into();

    let ptr_i64 = ptrtoint(ctx, block, acc_alloca, i64_type, loc);

    for i in 0..n {
        let offset = ((i * stride) as u64) << 56 | (i as u64 * 64);
        let offset_val = const_i64(ctx, block, offset as i64, loc);
        let gpr = block.append_operation(arith::addi(ptr_i64, offset_val, loc)).result(0).unwrap().into();
        amx_op(ctx, block, 4, gpr, loc);
    }
    Ok(())
}

/// Store Z register rows back to DEFINE_REG alloca.
///
/// Emits `ptrtoint(alloca)` then STZ for each row with encoding:
///   `(z_row << 56) | (row * 64)`
pub fn render_amx_stz<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    acc_alloca: Value<'c, 'c>,
    metadata: &WmmaMetadata,
    loc: Location<'c>,
) -> crate::Result<()> {
    validate_amx_dtypes(&metadata.dtype_in, &metadata.dtype_out)?;
    let (n, _m, _k) = metadata.dims;
    let stride = z_row_stride(&metadata.dtype_in, &metadata.dtype_out);
    let i64_type: Type = IntegerType::new(ctx, 64).into();

    let ptr_i64 = ptrtoint(ctx, block, acc_alloca, i64_type, loc);

    for i in 0..n {
        let offset = ((i * stride) as u64) << 56 | (i as u64 * 64);
        let offset_val = const_i64(ctx, block, offset as i64, loc);
        let gpr = block.append_operation(arith::addi(ptr_i64, offset_val, loc)).result(0).unwrap().into();
        amx_op(ctx, block, 5, gpr, loc);
    }
    Ok(())
}

/// Execute AMX LDY + LDX + FMA using i64 addresses for A and B.
///
/// `a_i64` and `b_i64` are i64 values representing memory addresses of the operands.
/// This function:
///   1. LDY(a_i64) — load A into Y register
///   2. LDX(b_i64) — load B into X register
///   3. FMA — Z += X * Y
///
/// When tile_grid > (1,1), uses load-pair mode (bit 62) for 128-byte loads
/// and emits multiple FMAs for the output tile grid.
fn render_amx_fma_core<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    a_i64: Value<'c, 'c>,
    b_i64: Value<'c, 'c>,
    metadata: &WmmaMetadata,
    loc: Location<'c>,
) -> crate::Result<()> {
    let (tile_y_count, tile_x_count) = metadata.tile_grid;
    let use_load_pair = tile_x_count > 1 || tile_y_count > 1;

    // LDX: load B into X register
    let ldx_ptr = if use_load_pair {
        block
            .append_operation(arith::addi(b_i64, const_i64(ctx, block, AMX_LOAD_PAIR_BIT as i64, loc), loc))
            .result(0)
            .unwrap()
            .into()
    } else {
        b_i64
    };
    amx_op(ctx, block, 0, ldx_ptr, loc);

    // LDY: load A into Y register
    let ldy_ptr = if use_load_pair {
        block
            .append_operation(arith::addi(a_i64, const_i64(ctx, block, AMX_LOAD_PAIR_BIT as i64, loc), loc))
            .result(0)
            .unwrap()
            .into()
    } else {
        a_i64
    };
    amx_op(ctx, block, 1, ldy_ptr, loc);

    // FMA: Z += X * Y
    let (fma_op, fma_flags) = fma_opcode_and_flags(metadata)?;
    if use_load_pair {
        let bytes_per_tile_row: usize = 64;
        for ty in 0..tile_y_count {
            for tx in 0..tile_x_count {
                let z_row = (ty * tile_x_count + tx) as u64;
                let x_off = (tx * bytes_per_tile_row) as u64;
                let y_off = (ty * bytes_per_tile_row) as u64;
                let encoding = fma_flags | (z_row << 20) | (x_off << 10) | y_off;
                let encoding_val = const_i64(ctx, block, encoding as i64, loc);
                amx_op(ctx, block, fma_op, encoding_val, loc);
            }
        }
    } else {
        let encoding_val = const_i64(ctx, block, fma_flags as i64, loc);
        amx_op(ctx, block, fma_op, encoding_val, loc);
    }
    Ok(())
}

/// Execute AMX FMA directly from source memory pointers (fast path).
///
/// `a_ptr` and `b_ptr` are MLIR pointer values (GEP results from INDEX rendering)
/// pointing to contiguous source data. Uses ptrtoint to get i64 addresses.
pub fn render_amx_direct_fma<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    a_ptr: Value<'c, 'c>,
    b_ptr: Value<'c, 'c>,
    metadata: &WmmaMetadata,
    loc: Location<'c>,
) -> crate::Result<()> {
    let i64_type: Type = IntegerType::new(ctx, 64).into();
    let a_i64 = ptrtoint(ctx, block, a_ptr, i64_type, loc);
    let b_i64 = ptrtoint(ctx, block, b_ptr, i64_type, loc);
    render_amx_fma_core(ctx, block, a_i64, b_i64, metadata, loc)
}

/// Execute AMX FMA via temp buffers for non-contiguous operands (fallback path).
///
/// When PADTO introduces gated loads, the devectorizer cannot merge them into a
/// single contiguous LOAD, so `resolve_to_load_index` fails. In this case, the
/// operands are already rendered as LLVM vector values. We store them into stack
/// allocas and use those addresses for AMX LDX/LDY.
///
/// Either operand can use direct pointer or temp buffer independently:
/// - `a`: Either `Direct(ptr)` or `TempBuffer(vec_value, vec_type)`.
/// - `b`: Either `Direct(ptr)` or `TempBuffer(vec_value, vec_type)`.
pub fn render_amx_fma<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    a: AmxOperand<'c>,
    b: AmxOperand<'c>,
    metadata: &WmmaMetadata,
    loc: Location<'c>,
) -> crate::Result<()> {
    let i64_type: Type = IntegerType::new(ctx, 64).into();
    let a_i64 = a.into_i64(ctx, block, i64_type, loc);
    let b_i64 = b.into_i64(ctx, block, i64_type, loc);
    render_amx_fma_core(ctx, block, a_i64, b_i64, metadata, loc)
}

/// AMX operand: either a direct memory pointer or a vector value needing a temp buffer.
pub enum AmxOperand<'c> {
    /// Direct pointer to contiguous source data (fast path).
    Direct(Value<'c, 'c>),
    /// LLVM vector value from non-contiguous loads (needs temp buffer).
    TempBuffer(Value<'c, 'c>, Type<'c>),
}

impl<'c> AmxOperand<'c> {
    /// Convert to i64 address for AMX load instructions.
    fn into_i64(self, ctx: &'c Context, block: &Block<'c>, i64_type: Type<'c>, loc: Location<'c>) -> Value<'c, 'c> {
        match self {
            AmxOperand::Direct(ptr) => ptrtoint(ctx, block, ptr, i64_type, loc),
            AmxOperand::TempBuffer(vec_val, vec_type) => {
                let ptr_type = super::types::mlir_ptr_type(ctx);
                let one = const_i64(ctx, block, 1, loc);
                let alloca = block
                    .append_operation(llvm::alloca(
                        ctx,
                        one,
                        ptr_type,
                        loc,
                        llvm::AllocaOptions::new().elem_type(Some(TypeAttribute::new(vec_type))),
                    ))
                    .result(0)
                    .unwrap()
                    .into();
                block.append_operation(llvm::store(ctx, vec_val, alloca, loc, Default::default()));
                ptrtoint(ctx, block, alloca, i64_type, loc)
            }
        }
    }
}

/// Emit AMX SET instruction (initialize AMX coprocessor state).
pub fn amx_set<'c>(ctx: &'c Context, block: &Block<'c>, loc: Location<'c>) {
    let asm = StringAttribute::new(ctx, "nop\nnop\nnop\n.word (0x201000+(17<<5)+0)");
    let constraints = StringAttribute::new(ctx, "~{memory}");
    emit_void_inline_asm(ctx, block, asm, constraints, &[], loc);
}

/// Emit AMX CLR instruction (finalize AMX coprocessor state).
pub fn amx_clr<'c>(ctx: &'c Context, block: &Block<'c>, loc: Location<'c>) {
    let asm = StringAttribute::new(ctx, "nop\nnop\nnop\n.word (0x201000+(17<<5)+1)");
    let constraints = StringAttribute::new(ctx, "~{memory}");
    emit_void_inline_asm(ctx, block, asm, constraints, &[], loc);
}

/// Emit a single AMX instruction (ldx/ldy/ldz/stz/fma).
///
/// The encoding is: `.word (0x201000+(op<<5)+0gpr-((0gpr>>4)*6))`
/// where `op` is the AMX opcode and `gpr` is the i64 register value
/// containing the pointer + bit-shifted offset.
fn amx_op<'c>(ctx: &'c Context, block: &Block<'c>, op: u32, gpr: Value<'c, 'c>, loc: Location<'c>) {
    let asm = StringAttribute::new(ctx, ".word (0x201000+($0<<5)+0$1-((0$1>>4)*6))");
    let constraints = StringAttribute::new(ctx, "i,r,~{memory}");
    let i32_type: Type = IntegerType::new(ctx, 32).into();
    let op_const = block
        .append_operation(arith::constant(ctx, IntegerAttribute::new(i32_type, op as i64).into(), loc))
        .result(0)
        .unwrap()
        .into();
    emit_void_inline_asm(ctx, block, asm, constraints, &[op_const, gpr], loc);
}

/// Select FMA opcode and flags based on dtypes.
///
/// Returns (opcode, flags) where:
/// - opcode: AMX instruction opcode (10=fma64, 12=fma32, 14=mac16, 15=fma16)
/// - flags: Additional encoding bits (bit 62 for f16→f32 mixed-precision)
pub(crate) fn fma_opcode_and_flags(metadata: &WmmaMetadata) -> crate::Result<(u32, u64)> {
    validate_amx_dtypes(&metadata.dtype_in, &metadata.dtype_out)?;
    let opcode = match metadata.dtype_in.base() {
        ScalarDType::Float64 => 10, // fma64
        ScalarDType::Float32 => 12, // fma32
        ScalarDType::Int16 => 14,   // mac16
        ScalarDType::Float16 => 15, // fma16
        _ => unreachable!("validate_amx_dtypes prevents this"),
    };
    // Bit 62 for f16->f32 mixed-precision
    let flags = if metadata.dtype_in.base() == ScalarDType::Float16 && metadata.dtype_out.base() == ScalarDType::Float32
    {
        AMX_FMA_Z_F32
    } else {
        0
    };
    Ok((opcode, flags))
}

/// Emit a void inline assembly operation with side effects.
fn emit_void_inline_asm<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    asm_string: StringAttribute<'c>,
    constraints: StringAttribute<'c>,
    operands: &[Value<'c, 'c>],
    loc: Location<'c>,
) {
    let mut op = ods::llvm::inline_asm(ctx, operands, asm_string, constraints, loc);
    let bool_true: Attribute = Attribute::unit(ctx);
    op.set_has_side_effects(bool_true);
    block.append_operation(op.into());
}

/// Convert a pointer value to i64 via llvm.ptrtoint.
fn ptrtoint<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    ptr: Value<'c, 'c>,
    i64_type: Type<'c>,
    loc: Location<'c>,
) -> Value<'c, 'c> {
    block.append_operation(ods::llvm::ptrtoint(ctx, i64_type, ptr, loc).into()).result(0).unwrap().into()
}
