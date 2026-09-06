//! NVPTX-specific UOp lowering.
//!
//! Intercepts the ops whose generic LLVM form the NVPTX backend cannot select
//! (`Special`, `Barrier`, LOCAL buffers, `Log2`, `Wmma`) and falls through to
//! the CPU emitter for everything else. Verified against clang 22 + ptxas
//! 13.3 at `sm_86`; see `nvidia_backend_plan.md` §2 for the lowering table.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{DType, ScalarDType};
use svod_ir::{BinaryOp, Op, UnaryOp, prelude::*};

use crate::common::shaped_dtype;
use crate::llvm::common::gpu::{AXIS_LETTERS, parse_special_axis, render_define_local};
use crate::llvm::common::{LlvmTarget, RenderContext, ldt};
use crate::llvm::cpu;
use crate::llvm::nvptx::wmma;
use svod_ir::ops;

/// Render a UOp for NVPTX: `Some(())` when handled (here or by
/// [`cpu::render_uop`]), `None` for meta-ops without instructions.
pub fn render_uop(uop: &Arc<UOp>, ctx: &mut RenderContext, kernel: &mut Vec<String>, target: LlvmTarget) -> Option<()> {
    let arch =
        target.cuda_arch().unwrap_or_else(|| panic!("nvptx::render_uop called with non-NVPTX target {target:?}"));

    match uop.op() {
        // ── NVPTX-specific overrides ─────────────────────────────────────
        Op::Special(ops::Special { name, .. }) => render_special(uop, name, ctx, kernel),
        Op::Barrier(..) => render_barrier(kernel),
        Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local) => {
            render_define_local(uop, ctx, kernel)
        }
        // REG buffers keep the CPU emitter's plain `alloca`: NVPTX allocas
        // live in the generic address space (its datalayout carries no
        // `A5`), so unlike AMD there is no addrspace(5) round trip.
        Op::Wmma(ops::Wmma { a, b, c, metadata }) => wmma::render_wmma_nvptx(uop, a, b, c, metadata, arch, ctx, kernel),
        Op::Unary(UnaryOp::Log2, src) => render_log2(uop, src, ctx, kernel),
        Op::Unary(UnaryOp::Exp2, src) if src.dtype().base() == ScalarDType::Float64 => {
            undecomposed(ctx, "f64 Exp2 (NVPTX lowers `@llvm.exp2` for f16/f32 only)")
        }
        // `@llvm.{exp,log,sin,cos,pow}` fail NVPTX instruction selection and
        // `@llvm.erf` becomes an external call that only ptxas rejects; the
        // scheduler decomposes all of them, so reaching here is a drift of
        // the capability lists. Fail at render time instead.
        Op::Unary(
            op @ (UnaryOp::Sin | UnaryOp::Exp | UnaryOp::Log | UnaryOp::Cos | UnaryOp::Tan | UnaryOp::Erf),
            _,
        ) => undecomposed(ctx, &format!("{op:?}")),
        Op::Binary(BinaryOp::Pow, ..) => undecomposed(ctx, "Pow"),
        // OCP fp8 conversions need the sm_89 `cvt.*.e4m3x2` intrinsics, which
        // this renderer does not lower yet; fp8 storage must stay out of the
        // profile's supported dtypes so the scheduler widens it upstream.
        Op::Cast(ops::Cast { src, dtype }) if dtype.is_fp8() || src.dtype().is_fp8() => {
            ctx.set_unsupported_op(format!("NVPTX fp8 cast {:?} -> {:?}", src.dtype(), dtype));
            None
        }
        // ── Everything else: shared CPU path (ALU, INDEX, LOAD, STORE, …) ─
        _ => cpu::render_uop(uop, ctx, kernel),
    }
}

fn undecomposed(ctx: &mut RenderContext, what: &str) -> Option<()> {
    ctx.set_invalid_graph(format!(
        "NVPTX renderer received an un-decomposed {what}; it has no PTX lowering and must be excluded from \
         supported_ops so the scheduler decomposes it"
    ));
    Some(())
}

// ── LOG2: lg2.approx.f32 ──────────────────────────────────────────────────

/// `@llvm.log2.*` has no NVPTX lowering, so `Log2` uses the hardware
/// `lg2.approx.f32` (tinygrad's PTX choice; `@llvm.exp2.f32` already selects
/// `ex2.approx.f32`). Scalar f32 only: f16 widens around it, vectors split
/// per lane, and f64 is decomposed upstream.
fn render_log2(uop: &Arc<UOp>, src: &Arc<UOp>, ctx: &mut RenderContext, kernel: &mut Vec<String>) -> Option<()> {
    let dtype = shaped_dtype(src);
    let scalar = dtype.base();
    if !matches!(scalar, ScalarDType::Float16 | ScalarDType::Float32) {
        return undecomposed(ctx, &format!("{scalar:?} Log2 (NVPTX `lg2.approx` is f32 only)"));
    }
    let dst = ctx.name(uop);
    let src_name = ctx.get(src).to_string();
    let lane_ty = ldt(&dtype.scalar_dtype());
    let emit_lane = |lane_dst: &str, lane_src: &str, kernel: &mut Vec<String>| {
        if scalar == ScalarDType::Float16 {
            kernel.push(format!("  {lane_dst}.w = fpext half {lane_src} to float"));
            kernel.push(format!("  {lane_dst}.l = call float @llvm.nvvm.lg2.approx.f(float {lane_dst}.w)"));
            kernel.push(format!("  {lane_dst} = fptrunc float {lane_dst}.l to half"));
        } else {
            kernel.push(format!("  {lane_dst} = call float @llvm.nvvm.lg2.approx.f(float {lane_src})"));
        }
    };

    let lanes = dtype.vcount();
    if lanes == 1 {
        emit_lane(&dst, &src_name, kernel);
        return Some(());
    }
    let vec_ty = ldt(&dtype);
    let mut acc = "poison".to_string();
    for lane in 0..lanes {
        let element = format!("{dst}.e{lane}");
        let value = format!("{dst}.r{lane}");
        kernel.push(format!("  {element} = extractelement {vec_ty} {src_name}, i32 {lane}"));
        emit_lane(&value, &element, kernel);
        let next = if lane + 1 == lanes { dst.clone() } else { format!("{dst}.v{lane}") };
        kernel.push(format!("  {next} = insertelement {vec_ty} {acc}, {lane_ty} {value}, i32 {lane}"));
        acc = next;
    }
    Some(())
}

// ── SPECIAL: ctaid / tid ──────────────────────────────────────────────────

fn render_special(uop: &Arc<UOp>, name: &str, ctx: &mut RenderContext, kernel: &mut Vec<String>) -> Option<()> {
    let dst = ctx.name(uop);
    let Some((kind, axis)) = parse_special_axis(name) else {
        ctx.set_invalid_graph(format!("NVPTX renderer: malformed SPECIAL axis name {name:?}"));
        return None;
    };
    let dim = AXIS_LETTERS[axis as usize];
    // `i` (direct-global) axes launch with a block size of 1, so the block
    // index is the flat axis — the same contract the AMD emitter relies on.
    let sreg = match kind {
        'g' | 'i' => "ctaid",
        'l' => "tid",
        _ => unreachable!("parse_special_axis only yields g/l/i"),
    };
    kernel.push(format!("  {dst} = tail call i32 @llvm.nvvm.read.ptx.sreg.{sreg}.{dim}()"));
    Some(())
}

// ── BARRIER: block-scope fences around bar.sync 0 ─────────────────────────

/// NVPTX names the work-group scope `"block"` (`syncscope("workgroup")` is
/// rejected). `@llvm.nvvm.barrier0` is the form every LLVM release lowers to
/// `bar.sync 0` (newer ones auto-upgrade it to `barrier.cta.sync.aligned.all`).
fn render_barrier(kernel: &mut Vec<String>) -> Option<()> {
    kernel.push("  fence syncscope(\"block\") release".to_string());
    kernel.push("  tail call void @llvm.nvvm.barrier0()".to_string());
    kernel.push("  fence syncscope(\"block\") acquire".to_string());
    Some(())
}

// ── Warp-level builders (typed CUSTOM nodes) ──────────────────────────────

/// `shfl.sync.bfly.b32` across the full warp: lane `L` receives `value` from
/// lane `L ^ lane_mask` (the butterfly step of a warp reduction). 32-bit
/// values only; f32 rides through i32 like tk's `ds_bpermute` shuffle. The
/// `declare` travels in the CUSTOM body and is hoisted to the module prefix.
pub fn shfl_bfly(value: &Arc<UOp>, lane_mask: &Arc<UOp>) -> Arc<UOp> {
    assert_eq!(value.dtype().bytes(), 4, "shfl_bfly moves one 32-bit register, got {:?}", value.dtype());
    let is_f32 = value.dtype() == DType::Float32;
    let word = if is_f32 { value.bitcast(DType::Int32) } else { value.clone() };
    let shuffled = UOp::custom(
        smallvec![word, lane_mask.cast(DType::Int32)],
        "declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)\n\
         call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 {0}, i32 {1}, i32 31)"
            .to_string(),
        DType::Int32,
    );
    if is_f32 { shuffled.bitcast(DType::Float32) } else { shuffled }
}

/// `%globaltimer`: the nanosecond GPU clock, for in-kernel stamp probes.
pub fn globaltimer() -> Arc<UOp> {
    UOp::custom(
        smallvec![],
        "declare i64 @llvm.nvvm.read.ptx.sreg.globaltimer()\n\
         call i64 @llvm.nvvm.read.ptx.sreg.globaltimer()"
            .to_string(),
        DType::UInt64,
    )
}

#[cfg(test)]
#[path = "../../test/unit/llvm_nvptx_ops.rs"]
pub(crate) mod tests;
