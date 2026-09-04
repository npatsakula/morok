//! CPU-specific LLVM IR operation rendering.
//!
//! Generates LLVM IR strings for individual UOp operations on CPU.
//! Based on Tinygrad's PatternMatcher templates in `llvmir.py`.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{BinaryOp, Op, TernaryOp, UnaryOp, prelude::*};

use crate::common::{access_dtype, shaped_dtype, value_width};
use crate::llvm::common::{RenderContext, lcast, ldt};
use svod_ir::ops;

/// LLVM type of a value, honouring the lane count carried in its shape.
/// Tinygrad spells this `ldt(u.dtype, u.max_numel())` (`llvmir.py`).
fn lshaped(value: &Arc<UOp>) -> String {
    ldt(&shaped_dtype(value))
}

/// Render a UOp to LLVM IR string.
///
/// Returns None for meta-ops that don't produce instructions.
pub fn render_uop(uop: &Arc<UOp>, ctx: &mut RenderContext, kernel: &mut Vec<String>) -> Option<()> {
    let dst = ctx.name(uop);

    match uop.op() {
        Op::Const(_)
        | Op::VConst(..)
        | Op::Param(..)
        | Op::DefineVar(..)
        | Op::Noop
        | Op::Sink(..)
        | Op::Group(..)
        | Op::Unique(_)
        | Op::Call(..)
        | Op::Barrier(..) => None,

        Op::Buffer(ops::Buffer { arg, .. })
            if matches!(arg.addrspace, Some(svod_ir::AddrSpace::Local | svod_ir::AddrSpace::Reg)) =>
        {
            let base_dtype = arg.dtype.clone();
            let alloc_size = uop.buffer_size().unwrap_or(1);
            let base = ldt(&base_dtype);
            let align = if arg.addrspace == Some(svod_ir::AddrSpace::Local) { ", align 16" } else { "" };
            kernel.push(format!("  {dst} = alloca [{alloc_size} x {base}]{align}"));
            Some(())
        }

        Op::Buffer(..) => None,

        Op::Index(ops::Index { buffer, indices, .. }) => {
            let buf = ctx.get(buffer).to_string();

            // An INDEX with no indices is the buffer pointer itself; under opaque
            // pointers there is nothing to bitcast (tinygrad renders no node at all).
            if indices.is_empty() {
                ctx.alias(uop.id, buf);
                return None;
            }

            let (final_idx, final_idx_type) = if indices.len() == 1 {
                (ctx.get(&indices[0]).to_string(), ldt(&indices[0].dtype()))
            } else {
                ctx.set_invalid_graph(format!(
                    "LLVM renderer requires linearized INDEX (single-axis), found {} indices on uop {}",
                    indices.len(),
                    uop.id
                ));
                return None;
            };

            // Same split as the C renderer and tinygrad `llvmir.py`: a buffer with
            // an address space is addressed by GEP over the element dtype; a
            // register-resident (ALU) value is lane-extracted.
            if buffer.addrspace().is_some() {
                // Gate is NOT handled here — matching Tinygrad's approach where INDEX
                // always emits a plain GEP. The gate is handled at LOAD level (branch+phi)
                // and at STORE level (IF/ENDIF via line_rewrite_cleanups).
                kernel.push(format!(
                    "  {dst} = getelementptr inbounds {}, ptr {buf}, {final_idx_type} {final_idx}",
                    ldt(&uop.dtype())
                ));
            } else {
                kernel
                    .push(format!("  {dst} = extractelement {} {buf}, {final_idx_type} {final_idx}", lshaped(buffer)));
            }
            Some(())
        }

        Op::Shrink(ops::Shrink { src, offsets, sizes: _ }) => {
            let buf = ctx.get(src);
            let idx = ctx.get(offsets);
            kernel.push(format!(
                "  {dst} = getelementptr inbounds {}, ptr {buf}, {} {idx}",
                ldt(&uop.dtype()),
                ldt(&offsets.dtype())
            ));
            Some(())
        }

        Op::Load(ops::Load { index, alt, gate }) => {
            // Defense-in-depth: `UOp::new` (ir hash_consing.rs `new_tagged`) already
            // asserts the alt/gate pairing, the bool gate and the alt dtype, so no
            // legal construction path reaches these branches.
            if alt.is_some() != gate.is_some() {
                ctx.set_invalid_graph(format!("LOAD on uop {} must have either neither or both alt and gate", uop.id));
                return None;
            }
            let load_dtype = shaped_dtype(uop);
            if let (Some(alt), Some(gate)) = (alt, gate) {
                if gate.dtype() != DType::Bool {
                    ctx.set_invalid_graph(format!("gated LOAD on uop {} requires a scalar bool gate", uop.id));
                    return None;
                }
                // The alt is either the full-width value or a scalar broadcast into
                // every lane; anything else cannot feed the phi.
                let alt_dtype = shaped_dtype(alt);
                if alt_dtype != load_dtype && alt_dtype != load_dtype.scalar_dtype() {
                    ctx.set_invalid_graph(format!(
                        "gated LOAD on uop {} requires alt dtype to match the load dtype",
                        uop.id
                    ));
                    return None;
                }
            }
            let idx = ctx.get(index);
            let dtype = ldt(&load_dtype);
            let idx_type = "ptr";
            let volatile = if is_volatile_access(index) { "volatile " } else { "" };

            let gate_info = match (alt, gate) {
                (None, None) => None,
                (Some(alt), Some(gate)) => {
                    let gate_name = ctx.get(gate).to_string();
                    let alt_name = ctx.get(alt).to_string();
                    // A scalar alt behind a grouped load splats into every lane
                    // before the branch, so the phi's incoming types agree.
                    let alt_name = if value_width(alt) < load_dtype.vcount() {
                        splat_or_literal(&alt_name, &load_dtype, kernel, &format!("{dst}.alt"))
                    } else {
                        alt_name
                    };
                    Some((gate_name, alt_name))
                }
                _ => unreachable!(),
            };

            if let Some((gate, alt_val)) = gate_info {
                let label_base = &dst[1..]; // strip leading %
                let entry_label = format!("{label_base}_entry");
                let load_label = format!("{label_base}_load");
                let exit_label = format!("{label_base}_exit");
                let load_val = format!("{dst}_yes");

                kernel.push(format!("  br label %{entry_label}"));
                kernel.push(format!("{entry_label}:"));
                kernel.push(format!("  br i1 {gate}, label %{load_label}, label %{exit_label}"));
                kernel.push(format!("{load_label}:"));
                kernel.push(format!("  {load_val} = load {volatile}{dtype}, {idx_type} {idx}"));
                kernel.push(format!("  br label %{exit_label}"));
                kernel.push(format!("{exit_label}:"));
                kernel.push(format!("  {dst} = phi {dtype} [{load_val}, %{load_label}], [{alt_val}, %{entry_label}]"));
            } else {
                kernel.push(format!("  {dst} = load {volatile}{dtype}, {idx_type} {idx}"));
            }
            Some(())
        }

        Op::Store(ops::Store { index, value, gate }) => {
            if gate.is_some() {
                ctx.set_invalid_graph(format!(
                    "gated STORE on uop {} reached LLVM codegen; linear cleanup must rewrite it to IF/STORE/ENDIF",
                    uop.id
                ));
                return None;
            }
            let idx = ctx.get(index);
            let val = ctx.get(value);
            let val_type = ldt(&access_dtype(index, value));
            let idx_type = "ptr";
            let volatile = if is_volatile_access(index) { "volatile " } else { "" };

            kernel.push(format!("  store {volatile}{val_type} {val}, {idx_type} {idx}"));
            Some(())
        }

        Op::Binary(op, lhs, rhs) => {
            let l = ctx.get(lhs);
            let r = ctx.get(rhs);
            let ltype = lshaped(lhs);
            let rtype = lshaped(rhs);

            // Detect type mismatch: emitting `op T %l, %r` with mismatched operand
            // types is invalid LLVM IR that the assembler rejects later. Surface it
            // as a typed error here (with full diagnostic context via tracing) rather
            // than producing a kernel that fails to compile.
            if ltype != rtype {
                tracing::error!(
                    uop_id = uop.id,
                    uop_dtype = ?uop.dtype(),
                    op = ?op,
                    lhs_id = lhs.id,
                    rhs_id = rhs.id,
                    lhs_dtype = ?lhs.dtype(),
                    rhs_dtype = ?rhs.dtype(),
                    lhs_op = ?lhs.op().as_ref(),
                    rhs_op = ?rhs.op().as_ref(),
                    "Binary op type mismatch - lhs and rhs have different dtypes"
                );
                ctx.set_invalid_graph(format!(
                    "binary {op:?} on uop {} has mismatched operand LLVM types ({ltype} vs {rtype}); \
                     lhs uop {} ({:?}), rhs uop {} ({:?})",
                    uop.id,
                    lhs.id,
                    lhs.dtype(),
                    rhs.id,
                    rhs.dtype(),
                ));
                return None;
            }

            if matches!(op, BinaryOp::Max) {
                render_binary_max(&dst, lhs, l, r, &ltype, kernel);
            } else if matches!(op, BinaryOp::Pow) {
                render_binary_pow(&dst, lhs, l, r, &ltype, kernel);
            } else {
                let instr = binary_instr(*op, &lhs.dtype());
                kernel.push(format!("  {dst} = {instr} {ltype} {l}, {r}"));
            }
            Some(())
        }

        Op::Unary(op, src) => {
            let s = ctx.get(src);
            let stype = lshaped(src);
            let src_dtype = shaped_dtype(src);

            match op {
                UnaryOp::Neg => {
                    if src.dtype().is_float() {
                        kernel.push(format!("  {dst} = fneg {stype} {s}"));
                    } else {
                        kernel.push(format!("  {dst} = sub {stype} 0, {s}"));
                    }
                }
                UnaryOp::Not => {
                    let all_ones = if src.dtype().is_bool() { "1".to_string() } else { "-1".to_string() };
                    kernel.push(format!("  {dst} = xor {stype} {s}, {all_ones}"));
                }
                UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Trunc | UnaryOp::Round if !src.dtype().is_float() => {
                    // Rounding is identity for integer types (defense-in-depth;
                    // symbolic_simple folds these away upstream).
                    kernel.push(format!("  {dst} = bitcast {stype} {s} to {stype}"));
                }
                UnaryOp::Sqrt
                | UnaryOp::Exp
                | UnaryOp::Exp2
                | UnaryOp::Log
                | UnaryOp::Log2
                | UnaryOp::Sin
                | UnaryOp::Cos
                | UnaryOp::Floor
                | UnaryOp::Ceil
                | UnaryOp::Trunc
                | UnaryOp::Round => {
                    let intrinsic = unary_instr(*op, &src.dtype()).unwrap();
                    render_intrinsic(&dst, intrinsic, &[(&stype, s)], &stype, kernel);
                }
                UnaryOp::Abs => {
                    if src.dtype().is_float() {
                        render_intrinsic(&dst, "fabs", &[(&stype, s)], &stype, kernel);
                    } else {
                        render_intrinsic(&dst, "abs", &[(&stype, s), ("i1", "1")], &stype, kernel);
                    }
                }
                UnaryOp::Rsqrt => {
                    let sqrt_dst = format!("{dst}.sqrt");
                    render_intrinsic(&sqrt_dst, "sqrt", &[(&stype, s)], &stype, kernel);
                    let one = splat_or_literal("1.0", &src_dtype, kernel, &dst);
                    kernel.push(format!("  {dst} = fdiv nsz arcp contract afn {stype} {one}, {sqrt_dst}"));
                }
                UnaryOp::Reciprocal => {
                    let one = splat_or_literal("1.0", &src_dtype, kernel, &dst);
                    kernel.push(format!("  {dst} = fdiv nsz arcp contract afn {stype} {one}, {s}"));
                }
                UnaryOp::Tan => {
                    let sin_dst = format!("{dst}.sin");
                    let cos_dst = format!("{dst}.cos");
                    render_intrinsic(&sin_dst, "sin", &[(&stype, s)], &stype, kernel);
                    render_intrinsic(&cos_dst, "cos", &[(&stype, s)], &stype, kernel);
                    kernel.push(format!("  {dst} = fdiv nsz arcp contract afn {stype} {sin_dst}, {cos_dst}"));
                }
                UnaryOp::Sign => {
                    if src.dtype().is_float() {
                        let gt_zero = format!("{dst}.gt");
                        let lt_zero = format!("{dst}.lt");
                        let gt_ext = format!("{dst}.gt_ext");
                        let lt_ext = format!("{dst}.lt_ext");
                        let zero = splat_or_literal("0.0", &src_dtype, kernel, &dst);
                        kernel.push(format!("  {gt_zero} = fcmp nsz arcp contract afn ogt {stype} {s}, {zero}"));
                        kernel.push(format!("  {lt_zero} = fcmp nsz arcp contract afn olt {stype} {s}, {zero}"));
                        kernel.push(format!("  {gt_ext} = uitofp i1 {gt_zero} to {stype}"));
                        kernel.push(format!("  {lt_ext} = uitofp i1 {lt_zero} to {stype}"));
                        kernel.push(format!("  {dst} = fsub nsz arcp contract afn {stype} {gt_ext}, {lt_ext}"));
                    } else if src.dtype().is_signed() {
                        let gt_zero = format!("{dst}.gt");
                        let lt_zero = format!("{dst}.lt");
                        let gt_ext = format!("{dst}.gt_ext");
                        let lt_ext = format!("{dst}.lt_ext");
                        let zero = splat_or_literal("0", &src_dtype, kernel, &dst);
                        kernel.push(format!("  {gt_zero} = icmp sgt {stype} {s}, {zero}"));
                        kernel.push(format!("  {lt_zero} = icmp slt {stype} {s}, {zero}"));
                        kernel.push(format!("  {gt_ext} = zext i1 {gt_zero} to {stype}"));
                        kernel.push(format!("  {lt_ext} = zext i1 {lt_zero} to {stype}"));
                        kernel.push(format!("  {dst} = sub {stype} {gt_ext}, {lt_ext}"));
                    } else {
                        // Unsigned: sign(x) = (x != 0) ? 1 : 0.
                        let ne_zero = format!("{dst}.ne");
                        let zero = splat_or_literal("0", &src_dtype, kernel, &dst);
                        kernel.push(format!("  {ne_zero} = icmp ne {stype} {s}, {zero}"));
                        kernel.push(format!("  {dst} = zext i1 {ne_zero} to {stype}"));
                    }
                }
                UnaryOp::Erf => {
                    render_intrinsic(&dst, "erf", &[(&stype, s)], &stype, kernel);
                }
                UnaryOp::Square => {
                    if src.dtype().is_float() {
                        kernel.push(format!("  {dst} = fmul nsz arcp contract afn {stype} {s}, {s}"));
                    } else {
                        kernel.push(format!("  {dst} = mul {stype} {s}, {s}"));
                    }
                }
            }
            Some(())
        }

        Op::Ternary(TernaryOp::Where, cond, t, f) => {
            let c = ctx.get(cond);
            let tv = ctx.get(t);
            let fv = ctx.get(f);
            kernel.push(format!("  {dst} = select {} {c}, {} {tv}, {} {fv}", lshaped(cond), lshaped(t), lshaped(f)));
            Some(())
        }

        Op::Ternary(TernaryOp::MulAcc, a, b, c) => {
            let av = ctx.get(a);
            let bv = ctx.get(b);
            let cv = ctx.get(c);
            let dtype = lshaped(a);

            if a.dtype().is_float() {
                render_intrinsic(&dst, "fmuladd", &[(&dtype, av), (&dtype, bv), (&dtype, cv)], &dtype, kernel);
            } else {
                let mul_dst = format!("{dst}.mul");
                kernel.push(format!("  {mul_dst} = mul {dtype} {av}, {bv}"));
                kernel.push(format!("  {dst} = add {dtype} {mul_dst}, {cv}"));
            }
            Some(())
        }

        Op::Cast(ops::Cast { src, dtype }) => {
            let src_llvm_type = lshaped(src);
            let dst_llvm_type = lshaped(uop);

            // Alias for noop casts: same LLVM type or target is Ptr.
            // Matches tinygrad llvmir.py:164-165.
            if src_llvm_type == dst_llvm_type || matches!(dtype, DType::Ptr { .. }) {
                ctx.alias(uop.id, ctx.get(src).to_string());
                return None;
            }

            let s = ctx.get(src);

            if dtype.is_bool() && !src.dtype().is_bool() {
                let cmp = if src.dtype().is_float() { "fcmp nsz arcp contract afn une" } else { "icmp ne" };
                kernel.push(format!("  {dst} = {cmp} {src_llvm_type} {s}, zeroinitializer"));
            } else {
                let cast_instr = lcast(&src.dtype(), dtype);
                kernel.push(format!("  {dst} = {cast_instr} {src_llvm_type} {s} to {dst_llvm_type}"));
            }
            Some(())
        }

        Op::BitCast(ops::BitCast { src, dtype: _ }) => {
            let s = ctx.get(src);
            kernel.push(format!("  {dst} = bitcast {} {s} to {}", lshaped(src), lshaped(uop)));
            Some(())
        }

        Op::Range(ops::Range { axis_id, end, .. }) => {
            let id = axis_id.name();
            let dtype = ldt(&uop.dtype());
            let end_val = ctx.get(end).to_string();

            // Track range nesting for correct END footer ordering.
            ctx.push_range(id.clone());

            // Matches Tinygrad llvmir.py:156-165 exactly:
            //   entry → loop_entry (preheader) → loop_latch (phi+incr+cmp) → loop_body / loop_exit
            //   loop_body contains body instructions
            //   END branches to loop_footer → loop_latch (back edge)
            kernel.push(format!("  br label %loop_entry_{id}"));
            kernel.push(format!("loop_entry_{id}:"));
            kernel.push(format!("  br label %loop_latch_{id}"));
            kernel.push(format!("loop_latch_{id}:"));
            kernel.push(format!("  {dst} = phi {dtype} [ 0, %loop_entry_{id} ], [ {dst}phi, %loop_footer_{id} ]"));
            kernel.push(format!("  {dst}phi = add {dtype} {dst}, 1"));
            kernel.push(format!("  {dst}cmp = icmp ult {dtype} {dst}, {end_val}"));
            kernel.push(format!("  br i1 {dst}cmp, label %loop_body_{id}, label %loop_exit_{id}"));
            kernel.push(format!("loop_body_{id}:"));
            Some(())
        }

        Op::End(ops::End { ranges, .. }) => {
            // After pm_split_ends each END closes the exact RANGE it names,
            // matching Tinygrad's END renderer rather than an anonymous stack pop.
            for range in ranges {
                let Op::Range(ops::Range { axis_id, .. }) = range.op() else { continue };
                let id = axis_id.name();
                if !ctx.close_range(&id) {
                    return Some(());
                }
                kernel.push(format!("  br label %loop_footer_{id}"));
                kernel.push(format!("loop_footer_{id}:"));
                kernel.push(format!("  br label %loop_latch_{id}"));
                kernel.push(format!("loop_exit_{id}:"));
            }

            Some(())
        }

        Op::Stack(ops::Stack { sources }) => {
            if sources.is_empty() {
                return None;
            }
            render_vectorize(&dst, sources, ctx, kernel);
            Some(())
        }

        Op::Detach(ops::Detach { src }) => {
            let s = ctx.get(src);
            ctx.alias(uop.id, s.to_string());
            None
        }

        Op::Wmma(ops::Wmma { a, b, c, metadata }) => {
            // Apple AMX matmul.
            //
            // Stack slots `wmma_<id>_amx{0,1,2}` were pre-allocated in the
            // function entry block (see `llvm/text/mod.rs`); LLVM's mem2reg
            // pass promotes them to registers across loop iterations, which
            // is the whole reason for using LLVM here over the C backend.
            //
            // Per call: store the 3 src vectors into their allocas, then
            // `ldz×16 + ldx + ldy + fma + stz×16` via AMX inline asm. The C
            // operand is a flat 256-elem accumulator; A and B are 16-elem
            // input vectors. The AMX(op, gpr) macro encodes the row index
            // and byte offset into the gpr for ldz/stz.
            let a_val = ctx.get(a);
            let b_val = ctx.get(b);
            let c_val = ctx.get(c);
            let a_dtype = ldt(&a.dtype());
            let b_dtype = ldt(&b.dtype());
            let c_dtype = ldt(&c.dtype());
            let a_align = a.dtype().bytes();
            let b_align = b.dtype().bytes();
            let c_align = c.dtype().bytes();

            let id = uop.id;
            let amx0 = format!("%wmma_{id}_amx0");
            let amx1 = format!("%wmma_{id}_amx1");
            let amx2 = format!("%wmma_{id}_amx2");
            let ptr0 = format!("%wmma_{id}_ptr_amx0");
            let ptr1 = format!("%wmma_{id}_ptr_amx1");
            let ptr2 = format!("%wmma_{id}_ptr_amx2");

            // 1. Store A, B, C into their pre-allocated stack slots.
            kernel.push(format!("  store {a_dtype} {a_val}, ptr {amx0}, align {a_align}"));
            kernel.push(format!("  store {b_dtype} {b_val}, ptr {amx1}, align {b_align}"));
            kernel.push(format!("  store {c_dtype} {c_val}, ptr {amx2}, align {c_align}"));

            // 2. AMX_SET(0): enable the AMX coprocessor on this thread.
            // Without this, every subsequent AMX instruction traps with
            // SIGILL because the coprocessor is in disabled state.
            // Encoding: `nop;nop;nop;.word (0x201000 + (17 << 5) + 0)`
            // = `0x201220`.
            kernel.push(amx_set_inline_asm(0));

            // 3. ldz × N rows of the C accumulator into Z registers.
            // AMX `ldz` op = 4. Each row is 64 bytes; row index is encoded in bits 56-59 (i*4<<56),
            // byte offset is bits 0-9 (i*64). The bytes_per_elem in the encoding is fixed at
            // 4 because AMX TC is fp32-only.
            let n_rows = metadata.dims.0; // typically 16 for fp32
            for i in 0..n_rows {
                let off = ((i as u64 * 4) << 56) | (i as u64 * 64);
                let ld_name = format!("%wmma_{id}_ld{i}");
                kernel.push(format!("  {ld_name} = add i64 {ptr2}, {off}"));
                kernel.push(amx_inline_asm(4, &ld_name));
            }

            // 4. ldx (A → X), ldy (B → Y), fma32.
            kernel.push(amx_inline_asm(0, &ptr1));
            kernel.push(amx_inline_asm(1, &ptr0));
            kernel.push(amx_inline_asm_imm(12, 0));

            // 5. stz × N rows of Z back into the C accumulator's stack slot.
            for i in 0..n_rows {
                let off = ((i as u64 * 4) << 56) | (i as u64 * 64);
                let st_name = format!("%wmma_{id}_st{i}");
                kernel.push(format!("  {st_name} = add i64 {ptr2}, {off}"));
                kernel.push(amx_inline_asm(5, &st_name));
            }

            // 6. AMX_SET(1): disable the AMX coprocessor. Pairs with the
            // enable above.
            kernel.push(amx_set_inline_asm(1));

            // 7. Load the WMMA result back from the C accumulator stack slot.
            kernel.push(format!("  {dst} = load {c_dtype}, ptr {amx2}, align {c_align}"));
            Some(())
        }

        Op::After(ops::After { passthrough, .. }) => {
            #[cfg(debug_assertions)]
            if matches!(passthrough.op(), Op::Range(..)) {
                panic!("AFTER passthrough is Range (id={}), this violates Tinygrad semantics", passthrough.id);
            }
            let s = ctx.get(passthrough);
            ctx.alias(uop.id, s.to_string());
            None
        }

        Op::Bind(ops::Bind { var, value }) => {
            let v = ctx.get(value);
            ctx.alias(var.id, v.to_string());
            None
        }

        Op::If(ops::If { condition, .. }) => {
            let cond = ctx.get(condition);
            let if_id = uop.id;
            kernel.push(format!("  br i1 {cond}, label %if_then_{if_id}, label %if_end_{if_id}"));
            kernel.push(format!("if_then_{if_id}:"));
            Some(())
        }

        Op::EndIf(ops::EndIf { if_op }) => {
            let if_id = if_op.id;
            kernel.push(format!("  br label %if_end_{if_id}"));
            kernel.push(format!("if_end_{if_id}:"));
            Some(())
        }

        // CUSTOMI is always inline: register the formatted template as this
        // uop's operand string so consumers substitute it directly. Unlike C,
        // LLVM SSA cannot inline a multi-instruction fragment, so the template
        // must format to a single valid operand (a constant, a constexpr like
        // `bitcast`/`getelementptr`, or an existing SSA value). For anything
        // that needs its own instruction, use a typed `Custom` statement.
        Op::CustomI(ops::CustomI { deps, code }) => {
            let args: Vec<String> = deps.iter().map(|dep| ctx.get(dep).to_string()).collect();
            let expr = match crate::common::format_custom_template_strict(code, &args) {
                Ok(s) => s,
                Err(e) => {
                    ctx.set_invalid_graph(format!("CUSTOMI template error on uop {}: {e}", uop.id));
                    return None;
                }
            };
            ctx.register(uop.id, expr);
            Some(())
        }

        // CUSTOM emits raw LLVM IR. Any `declare ...` lines are hoisted to the
        // module prefix (deduplicated) so custom bodies may reference intrinsics
        // not in the renderer's built-in declaration set; remaining lines form
        // the body. A `Void` custom is a bare statement block; a typed custom is
        // a single instruction whose rendered text is the assignment RHS
        // (e.g. `fmul float {0}, 2.0` → `%vN = fmul float %op, 2.0`) — the LLVM
        // type lives in the RHS, so unlike C there is no separate declaration.
        Op::Custom(ops::Custom { deps, code }) => {
            let args: Vec<String> = deps.iter().map(|dep| ctx.get(dep).to_string()).collect();
            let rendered = match crate::common::format_custom_template_strict(code, &args) {
                Ok(s) => s,
                Err(e) => {
                    ctx.set_invalid_graph(format!("CUSTOM template error on uop {}: {e}", uop.id));
                    return None;
                }
            };

            let mut body_lines: Vec<String> = Vec::new();
            for line in rendered.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.starts_with("declare ") {
                    let decl = trimmed.to_string();
                    if !ctx.module_prefix().iter().any(|l| l == &decl) {
                        ctx.push_module_prefix(decl);
                    }
                } else {
                    body_lines.push(trimmed.to_string());
                }
            }

            if uop.dtype() == DType::Void {
                for line in &body_lines {
                    kernel.push(format!("  {line}"));
                }
                ctx.register(uop.id, String::new());
            } else {
                if body_lines.len() != 1 {
                    ctx.set_invalid_graph(format!(
                        "typed CUSTOM on uop {} must render exactly one LLVM instruction RHS, got {} body line(s)",
                        uop.id,
                        body_lines.len()
                    ));
                    return None;
                }
                kernel.push(format!("  {dst} = {}", body_lines[0]));
                ctx.register(uop.id, dst.clone());
            }
            Some(())
        }

        op if op.is_movement() => {
            // Movement ops must be eliminated during rangeify (remove_movement_op /
            // apply_bufferize_transform). Reaching codegen means the graph is malformed.
            ctx.set_invalid_graph(format!(
                "movement op {} (uop {}) reached LLVM codegen; should have been eliminated during rangeify",
                op.as_ref(),
                uop.id,
            ));
            None
        }

        op => {
            // An op variant the LLVM backend has no lowering for. Surface it as a
            // typed error instead of emitting a comment + None that would detonate
            // later when a consumer calls `ctx.get` on this missing value.
            ctx.set_unsupported_op(op.as_ref());
            None
        }
    }
}

fn is_volatile_access(index: &Arc<UOp>) -> bool {
    let mut current = index;
    loop {
        match current.op() {
            Op::Param(ops::Param { arg, .. }) => return arg.volatile,
            Op::Index(ops::Index { buffer, .. }) => current = buffer,
            Op::Shrink(ops::Shrink { src, .. })
            | Op::Cast(ops::Cast { src, .. })
            | Op::After(ops::After { passthrough: src, .. })
            | Op::Reshape(ops::Reshape { src, .. })
            | Op::Permute(ops::Permute { src, .. })
            | Op::Expand(ops::Expand { src, .. })
            | Op::Pad(ops::Pad { src, .. })
            | Op::Flip(ops::Flip { src, .. }) => current = src,
            Op::MSelect(ops::MSelect { buffer, .. }) => current = buffer,
            _ => return false,
        }
    }
}

/// Materialize a scalar literal as a value usable in a `dtype`-typed
/// instruction. For scalar `dtype` returns the literal as-is; for vector
/// `dtype` emits a splat (insertelement + shufflevector) into `kernel`
/// and returns the resulting SSA name.
fn splat_or_literal(scalar_lit: &str, dtype: &DType, kernel: &mut Vec<String>, name_hint: &str) -> String {
    if dtype.vcount() <= 1 {
        return scalar_lit.to_string();
    }
    let scalar_ty = ldt(&dtype.scalar_dtype());
    let n = dtype.vcount();
    let splat_z = format!("{name_hint}.splat0");
    let splat_v = format!("{name_hint}.splat");
    kernel.push(format!("  {splat_z} = insertelement <1 x {scalar_ty}> poison, {scalar_ty} {scalar_lit}, i32 0"));
    kernel.push(format!(
        "  {splat_v} = shufflevector <1 x {scalar_ty}> {splat_z}, \
         <1 x {scalar_ty}> poison, <{n} x i32> zeroinitializer"
    ));
    splat_v
}

/// Emit an `AMX_SET` instruction that toggles the AMX coprocessor's
/// per-thread state. `imm5 = 0` enables AMX (must run before any other
/// AMX instruction); `imm5 = 1` disables it (must run when leaving the
/// AMX block to release the corruption surface).
///
/// Encoding: three NOP cycles to drain the pipeline, then a fixed 32-bit
/// word at `0x201000 + (17 << 5) + imm5`. `17` is the AMX_SET op slot.
/// Same encoding as the `AMX_SET` macro in svod's C backend
/// (`codegen/src/c/amx.rs:39`).
fn amx_set_inline_asm(imm5: u32) -> String {
    let opcode = 0x201000u32 + (17 << 5) + imm5;
    format!(
        "  tail call void asm sideeffect \"nop\\0Anop\\0Anop\\0A.word ({opcode})\", \
         \"~{{memory}}\"()"
    )
}

/// Emit an Apple AMX inline asm instruction that takes a 64-bit register
/// operand.
///
/// The `.word` directive emits the AMX-encoded instruction; the encoding
/// `0x201000+(op<<5)+gpr-...` selects the AMX op and which AArch64 GPR
/// carries the operand. `sideeffect` is required so LLVM doesn't DCE the
/// AMX state-mutating instruction.
fn amx_inline_asm(op: u32, gpr_name: &str) -> String {
    format!(
        "  tail call void asm sideeffect \".word (0x201000+($0<<5)+0$1-((0$1>>4)*6))\", \
         \"i,r,~{{memory}}\"(i32 {op}, i64 {gpr_name})"
    )
}

/// Emit an AMX inline asm instruction with an immediate operand instead of a
/// register (used for `fma32` where the operand encoding is `0`).
fn amx_inline_asm_imm(op: u32, imm: u64) -> String {
    format!(
        "  tail call void asm sideeffect \".word (0x201000+($0<<5)+0$1-((0$1>>4)*6))\", \
         \"i,r,~{{memory}}\"(i32 {op}, i64 {imm})"
    )
}

fn binary_instr(op: BinaryOp, dtype: &DType) -> &'static str {
    assert!(
        !matches!(dtype.base(), svod_dtype::ScalarDType::Index),
        "Index dtype reached LLVM codegen binary_instr({op:?}, {dtype:?}) — \
         pm_lower_index_dtype should have lowered it to i32/i64"
    );
    let is_float = dtype.is_float();
    let is_signed = dtype.is_signed();

    match op {
        BinaryOp::FloorDiv | BinaryOp::FloorMod => {
            unreachable!("floor div/mod must be decomposed before LLVM rendering")
        }
        BinaryOp::Add => {
            if is_float {
                "fadd nsz arcp contract afn"
            } else if is_signed {
                "add nsw"
            } else {
                "add"
            }
        }
        BinaryOp::Mul => {
            if is_float {
                "fmul nsz arcp contract afn"
            } else {
                "mul"
            }
        }
        BinaryOp::Sub => {
            if is_float {
                "fsub nsz arcp contract afn"
            } else {
                "sub"
            }
        }
        BinaryOp::Fdiv => "fdiv nsz arcp contract afn",
        BinaryOp::CDiv => {
            if is_signed {
                "sdiv"
            } else {
                "udiv"
            }
        }
        BinaryOp::CMod => {
            if is_float {
                "frem nsz arcp contract afn"
            } else if is_signed {
                "srem"
            } else {
                "urem"
            }
        }
        BinaryOp::Max => {
            if is_float {
                "maxnum"
            } else if is_signed {
                "smax"
            } else {
                "umax"
            }
        }
        // Ordered float predicates throughout. Tinygrad only has CMPLT/CMPNE/CMPEQ
        // (`llvmir.py` float_lop: olt/une/oeq); Le/Gt/Ge are svod-only ops, and
        // ordered matches the C backend, whose `<=`/`>`/`>=` are false on NaN.
        BinaryOp::Lt => {
            if is_float {
                "fcmp nsz arcp contract afn olt"
            } else if is_signed {
                "icmp slt"
            } else {
                "icmp ult"
            }
        }
        BinaryOp::Le => {
            if is_float {
                "fcmp nsz arcp contract afn ole"
            } else if is_signed {
                "icmp sle"
            } else {
                "icmp ule"
            }
        }
        BinaryOp::Gt => {
            if is_float {
                "fcmp nsz arcp contract afn ogt"
            } else if is_signed {
                "icmp sgt"
            } else {
                "icmp ugt"
            }
        }
        BinaryOp::Ge => {
            if is_float {
                "fcmp nsz arcp contract afn oge"
            } else if is_signed {
                "icmp sge"
            } else {
                "icmp uge"
            }
        }
        BinaryOp::Eq => {
            if is_float {
                "fcmp nsz arcp contract afn oeq"
            } else {
                "icmp eq"
            }
        }
        BinaryOp::Ne => {
            if is_float {
                "fcmp nsz arcp contract afn une"
            } else {
                "icmp ne"
            }
        }
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
        BinaryOp::Xor => "xor",
        BinaryOp::Shl => "shl",
        BinaryOp::Shr => {
            if is_signed {
                "ashr"
            } else {
                "lshr"
            }
        }
        BinaryOp::Pow => "pow",
        BinaryOp::Threefry => "xor",
    }
}

fn unary_instr(op: UnaryOp, dtype: &DType) -> Option<&'static str> {
    let is_float = dtype.is_float();

    match op {
        UnaryOp::Neg => Some(if is_float { "fneg" } else { "sub" }),
        UnaryOp::Not => Some("xor"),
        UnaryOp::Sqrt => Some("sqrt"),
        UnaryOp::Rsqrt => None,
        UnaryOp::Exp => Some("exp"),
        UnaryOp::Exp2 => Some("exp2"),
        UnaryOp::Log => Some("log"),
        UnaryOp::Log2 => Some("log2"),
        UnaryOp::Sin => Some("sin"),
        UnaryOp::Cos => Some("cos"),
        UnaryOp::Abs => Some(if is_float { "fabs" } else { "abs" }),
        UnaryOp::Floor => Some("floor"),
        UnaryOp::Ceil => Some("ceil"),
        UnaryOp::Trunc => Some("trunc"),
        UnaryOp::Round => Some("rint"),
        UnaryOp::Reciprocal => None,
        UnaryOp::Tan => None,
        UnaryOp::Sign => None,
        UnaryOp::Erf => None,
        UnaryOp::Square => None,
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

fn render_intrinsic(dst: &str, name: &str, args: &[(&str, &str)], ret_type: &str, kernel: &mut Vec<String>) {
    let args_str: String = args.iter().map(|(ty, val)| format!("{ty} {val}")).collect::<Vec<_>>().join(", ");
    let mangled = mangle_type(ret_type);
    kernel.push(format!("  {dst} = call {ret_type} @llvm.{name}.{mangled}({args_str})"));
}

fn render_binary_max(dst: &str, lhs: &Arc<UOp>, l: &str, r: &str, ltype: &str, kernel: &mut Vec<String>) {
    if lhs.dtype().is_float() {
        render_intrinsic(dst, "maxnum", &[(ltype, l), (ltype, r)], ltype, kernel);
    } else {
        let is_signed = lhs.dtype().is_signed();
        let cmp = if is_signed { "sgt" } else { "ugt" };
        let cmp_dst = format!("{dst}.cmp");
        kernel.push(format!("  {cmp_dst} = icmp {cmp} {ltype} {l}, {r}"));
        kernel.push(format!("  {dst} = select i1 {cmp_dst}, {ltype} {l}, {ltype} {r}"));
    }
}

fn render_binary_pow(dst: &str, lhs: &Arc<UOp>, l: &str, r: &str, ltype: &str, kernel: &mut Vec<String>) {
    if lhs.dtype().is_float() {
        render_intrinsic(dst, "pow", &[(ltype, l), (ltype, r)], ltype, kernel);
    } else {
        let l_float = format!("{dst}.lf");
        let r_float = format!("{dst}.rf");
        let pow_float = format!("{dst}.pf");
        kernel.push(format!("  {l_float} = sitofp {ltype} {l} to double"));
        kernel.push(format!("  {r_float} = sitofp {ltype} {r} to double"));
        render_intrinsic(&pow_float, "pow", &[("double", &l_float), ("double", &r_float)], "double", kernel);
        kernel.push(format!("  {dst} = fptosi double {pow_float} to {ltype}"));
    }
}

fn render_vectorize(dst: &str, elements: &[Arc<UOp>], ctx: &RenderContext, kernel: &mut Vec<String>) {
    if elements.is_empty() {
        return;
    }

    let scalar_type = ldt(&elements[0].dtype());
    let count = elements.len();
    let vec_type = format!("<{count} x {scalar_type}>");

    let mut prev = "poison".to_string();
    for (i, elem) in elements.iter().enumerate() {
        let val = ctx.get(elem);
        let next = if i == count - 1 { dst.to_string() } else { format!("{dst}.v{i}") };
        kernel.push(format!("  {next} = insertelement {vec_type} {prev}, {scalar_type} {val}, i32 {i}"));
        prev = next;
    }
}
