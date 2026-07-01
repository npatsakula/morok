//! AMD WMMA / MFMA intrinsic dispatch by gfx family.
//!
//! The IR-level matrix shape is encoded in `WmmaMetadata::dims` as `(N, M, K)`.
//! For each (arch, in_dtype, acc_dtype, dims) tuple we map to one of the
//! `@llvm.amdgcn.{wmma|mfma}.*` intrinsics, packing inputs as needed.

use std::sync::Arc;

use svod_dtype::{AmdArch, DType, ScalarDType};
use svod_ir::{WmmaMetadata, prelude::*};

use crate::llvm::common::{RenderContext, ldt};

/// Render a WMMA UOp for the AMD target. Returns `None` if the (arch, dtype,
/// shape) combination has no direct intrinsic; in that case the caller
/// surfaces an `InvalidGraph` error and the optimizer must decompose it
/// upstream.
#[allow(clippy::too_many_arguments)]
pub fn render_wmma_amd(
    uop: &Arc<UOp>,
    a: &Arc<UOp>,
    b: &Arc<UOp>,
    c: &Arc<UOp>,
    metadata: &WmmaMetadata,
    arch: AmdArch,
    ctx: &mut RenderContext,
    kernel: &mut Vec<String>,
) -> Option<()> {
    let dst = ctx.name(uop);
    let a_name = ctx.get(a).to_string();
    let b_name = ctx.get(b).to_string();
    let c_name = ctx.get(c).to_string();

    let (n, m, k) = metadata.dims;
    // WMMA operands are vectors (e.g. `<16 x half>` for inputs, `<8 x float>`
    // for accumulators). `DType::scalar()` returns `None` for `Vector{..}` —
    // we need `base()`, which unwraps both `Scalar` and `Vector` to the inner
    // ScalarDType. svod's `.scalar()` is stricter than that, so we use
    // `.base()` here. Wrapping in `Some` keeps the downstream API uniform.
    let in_scalar = Some(a.dtype().base());
    let acc_scalar = Some(uop.dtype().base());

    let intrinsic = match resolve_intrinsic(arch, in_scalar, acc_scalar, (n, m, k)) {
        Some(s) => s,
        None => {
            ctx.set_invalid_graph(format!(
                "AMD renderer: no WMMA/MFMA intrinsic for arch={arch} in={in_scalar:?} \
                 acc={acc_scalar:?} dims=({n},{m},{k})"
            ));
            return None;
        }
    };

    // The intrinsics take their operands as raw bit patterns, not the natural
    // float types: bf16 lanes as `<N x i16>`, fp8 lanes packed into a single
    // `iN`. Bitcast each operand to its wire type before the call (and the
    // accumulator + result back, when it too is reinterpreted). Mirrors
    // tinygrad's `AMDLLVMRenderer` operand rewrite (`llvmir.py:274-298`).
    // The K=32 dotted `.bf16` MFMA (CDNA4/gfx950) takes its operands as native
    // `<N x bfloat>`; the K=16 `.bf16.1k` form takes `<N x i16>`. gfx942 never
    // reaches here for K=32 bf16 — `resolve_intrinsic` returns `None` above —
    // so this only flips the wire type on the gfx950 path.
    let bf16_native = arch.is_cdna() && k == 32;

    let a_op = bitcast_operand(kernel, &dst, "a", &a.dtype(), &a_name, bf16_native);
    let b_op = bitcast_operand(kernel, &dst, "b", &b.dtype(), &b_name, bf16_native);
    let c_op = bitcast_operand(kernel, &dst, "c", &c.dtype(), &c_name, bf16_native);

    let (acc_wire, acc_reinterpreted) = wmma_wire_type(&uop.dtype(), bf16_native);
    let call_dst = if acc_reinterpreted { format!("{dst}.r") } else { dst.clone() };

    let tail = if arch.is_cdna() {
        // MFMA: trailing cbsz/abid/blgp immediates.
        ", i32 0, i32 0, i32 0"
    } else if matches!(acc_scalar, Some(ScalarDType::Float32)) {
        // f32-accumulating WMMAs take (A, B, C) only.
        ""
    } else {
        // Any other accumulator (f16/bf16/int) takes a trailing `i1 false`
        // (the clamp/opsel bit).
        ", i1 false"
    };

    // Inline-`asm sideeffect` MFMA path (tk asm-microkernel only): emit the matrix
    // op as opaque inline asm so the AMDGPU machine scheduler cannot re-batch the
    // inner loop's `ds_read`/MFMA program order (the intrinsic form gets reordered
    // into all-memory-then-all-compute). The `0` constraint ties the C accumulator
    // operand to the result register (in==out → the K-reduction chains in place),
    // exactly mirroring the proven gfx942 spike. Only valid for the f32-accumulating
    // bf16 K=16 MFMA — the only shape the flag is ever set for; any other (e.g. the
    // bf16→bf16 reinterpreted accumulator) falls through to the intrinsic.
    let asm_mfma = metadata.asm && arch.is_cdna() && !acc_reinterpreted;
    if asm_mfma {
        kernel.push(format!(
            "  {call_dst} = call {acc_wire} asm sideeffect \
             \"v_mfma_f32_16x16x16_bf16 $0, $1, $2, $3\", \"=v,v,v,0\"({a_op}, {b_op}, {c_op})"
        ));
    } else {
        // Carry svod's standard fp fast-math flags (`nsz arcp contract afn` — the
        // same set applied to every `fadd`/`fmul`/… , matching tinygrad
        // `llvmir.py`) on the matrix op, so the compiler may contract/fuse it with
        // surrounding fp like any other float op and match hipcc's `contract` MFMA.
        // Only for a **float**-accumulating WMMA: the bf16→bf16 reinterpreted form
        // and integer accumulators return integer vectors, which reject fp flags.
        let fm = if !acc_reinterpreted && matches!(acc_scalar, Some(ScalarDType::Float32 | ScalarDType::Float16)) {
            "nsz arcp contract afn "
        } else {
            ""
        };
        kernel.push(format!("  {call_dst} = call {fm}{acc_wire} @{intrinsic}({a_op}, {b_op}, {c_op}{tail})"));
    }

    if acc_reinterpreted {
        // bf16→bf16: the call returns `<N x i16>`; reinterpret it back to bf16.
        kernel.push(format!("  {dst} = bitcast {acc_wire} {call_dst} to {}", ldt(&uop.dtype())));
    }
    Some(())
}

/// The LLVM type a WMMA/MFMA operand must be passed as, plus whether that
/// differs from its natural `ldt` type (a bitcast is then required). bf16 lanes
/// go as `i16` (the `bf16.1k`/RDNA `.bf16` intrinsics), except for the CDNA4
/// K=32 `.bf16` form which takes native `<N x bfloat>` (`bf16_native`); fp8
/// lanes pack into one `iN`; every other dtype passes as-is.
fn wmma_wire_type(dtype: &DType, bf16_native: bool) -> (String, bool) {
    match dtype {
        DType::Vector { scalar: ScalarDType::BFloat16, count } if !bf16_native => (format!("<{count} x i16>"), true),
        DType::Vector { scalar: ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2, count } => {
            (format!("i{}", count * 8), true)
        }
        _ => (ldt(dtype), false),
    }
}

/// Emit the operand bitcast when its wire type differs from its natural type,
/// and return the `"<wire-ty> <value>"` fragment for the call's argument list.
/// The temp name is derived from the unique `dst` (`%vN.a` …) so no fresh-name
/// counter is needed.
fn bitcast_operand(
    kernel: &mut Vec<String>,
    dst: &str,
    suffix: &str,
    dtype: &DType,
    name: &str,
    bf16_native: bool,
) -> String {
    let (wire_ty, reinterpreted) = wmma_wire_type(dtype, bf16_native);
    if !reinterpreted {
        return format!("{wire_ty} {name}");
    }
    let tmp = format!("{dst}.{suffix}");
    kernel.push(format!("  {tmp} = bitcast {} {name} to {wire_ty}", ldt(dtype)));
    format!("{wire_ty} {tmp}")
}

/// Pick an amdgcn intrinsic name for a given (arch, dtype, shape) tuple.
///
/// Returns `None` for shapes/dtypes the renderer doesn't natively support
/// (the optimizer is expected to decompose those upstream).
///
/// Naming scheme:
/// - RDNA3/RDNA4: `llvm.amdgcn.wmma.<acc>.16x16x16.<in>` (with optional
///   `.tied` for vec(8) accumulators on gfx1100/1151 — we leave that to a
///   future pre-rewrite pass).
/// - CDNA: `llvm.amdgcn.mfma.<acc>.<N>x<M>x<K><in>`.
/// - RDNA2 and other non-matrix-core arches: `None` — the optimizer must
///   decompose WMMA UOps to scalar/vector loops before rendering.
fn resolve_intrinsic(
    arch: AmdArch,
    in_dt: Option<ScalarDType>,
    acc_dt: Option<ScalarDType>,
    dims: (usize, usize, usize),
) -> Option<String> {
    if !arch.has_matrix_cores() {
        return None;
    }

    let (n, m, k) = dims;
    let in_dt = in_dt?;
    let acc_dt = acc_dt?;

    if arch.is_cdna() {
        // Verified with `llc -mcpu=gfx942|gfx950` (ROCm 7.2): the f16/bf16 K=16
        // forms (`f16`/`bf16.1k`) select on both CDNA3 (gfx942) and CDNA4
        // (gfx950); the dotted K=32 double-rate forms (`.f16`/`.bf16`) select on
        // gfx950 only; fp8/bf8 select only at K=32; f32 selects only at K=4
        // (`v_mfma_f32_16x16x4_f32`, scalar A/B operands). Anything else has no
        // MFMA intrinsic — return `None` so the caller raises `InvalidGraph`
        // (and the optimizer decomposes it) instead of emitting a name LLVM
        // silently lowers to a no-op extern call.
        let is_cdna4 = matches!(arch, AmdArch::Gfx950);
        let in_suffix = match (in_dt, k) {
            (ScalarDType::Float16, 32) if is_cdna4 => ".f16",
            (ScalarDType::BFloat16, 32) if is_cdna4 => ".bf16",
            (ScalarDType::Float16 | ScalarDType::BFloat16, 32) => return None,
            (ScalarDType::Float16, _) => "f16",
            (ScalarDType::BFloat16, _) => "bf16.1k",
            (ScalarDType::Float32, 4) => "f32",
            (ScalarDType::Float32, _) => return None,
            (ScalarDType::FP8E4M3, 32) => ".fp8.fp8",
            (ScalarDType::FP8E5M2, 32) => ".bf8.bf8",
            _ => return None,
        };
        let acc_suffix = match acc_dt {
            ScalarDType::Float32 => "f32",
            ScalarDType::Float64 => "f64",
            ScalarDType::Int32 => "i32",
            _ => return None,
        };
        return Some(format!("llvm.amdgcn.mfma.{acc_suffix}.{n}x{m}x{k}{in_suffix}"));
    }

    // RDNA3 / RDNA4 WMMA — both families use 16x16x16 matmul; differ in input
    // dtype packing (handled by upstream pre-rewrites at the renderer level
    // when present; here we just name the intrinsic).
    let in_suffix = match in_dt {
        ScalarDType::Float16 => "f16",
        ScalarDType::BFloat16 => "bf16",
        ScalarDType::Int8 => "iu8",
        ScalarDType::FP8E4M3 => "fp8.fp8",
        ScalarDType::FP8E5M2 => "bf8.bf8",
        _ => return None,
    };
    let acc_suffix = match acc_dt {
        ScalarDType::Float32 => "f32",
        ScalarDType::Float16 => "f16",
        ScalarDType::BFloat16 => "bf16",
        ScalarDType::Int32 => "i32",
        _ => return None,
    };
    Some(format!("llvm.amdgcn.wmma.{acc_suffix}.{n}x{m}x{k}.{in_suffix}"))
}

#[cfg(test)]
#[path = "../../test/unit/llvm_amd_wmma.rs"]
mod tests;
