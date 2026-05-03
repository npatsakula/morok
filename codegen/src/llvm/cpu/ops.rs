//! CPU-specific LLVM IR operation rendering.
//!
//! Generates LLVM IR strings for individual UOp operations on CPU.
//! Based on Tinygrad's PatternMatcher templates in `llvmir.py`.

use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::{BinaryOp, Op, ReduceOp, TernaryOp, UnaryOp, prelude::*};

use crate::llvm::common::{RenderContext, lcast, ldt};

/// Extract a scalar `ptr` from a vectorized `<N x ptr>` via `extractelement ... i32 0`.
///
/// When the devectorize pipeline doesn't fully eliminate vectorized PARAM pointers
/// (see `no_vectorized_buf` / `no_vectorized_index` which only target DEFINE_LOCAL/DEFINE_REG),
/// the GEP result can be `<N x ptr>`. All elements are identical (broadcast of the same buffer
/// pointer), so extracting element 0 yields the correct scalar ptr for LLVM load/store.
fn maybe_extract_scalar_ptr(
    dst: &str,
    idx: &str,
    idx_type: &str,
    dtype: &DType,
    kernel: &mut Vec<String>,
) -> (String, String) {
    if matches!(dtype, DType::Ptr { vcount, .. } if *vcount > 1) {
        let extract = format!("{dst}.ptr");
        kernel.push(format!("  {extract} = extractelement {idx_type} {idx}, i32 0"));
        (extract, "ptr".to_string())
    } else {
        (idx.to_string(), idx_type.to_string())
    }
}

/// Render a UOp to LLVM IR string.
///
/// Returns None for meta-ops that don't produce instructions.
pub fn render_uop(uop: &Arc<UOp>, ctx: &mut RenderContext, kernel: &mut Vec<String>) -> Option<()> {
    let dst = ctx.name(uop);

    match uop.op() {
        Op::Const(_)
        | Op::VConst { .. }
        | Op::Param { device: None, .. }
        | Op::DefineVar { .. }
        | Op::Noop
        | Op::Sink { .. }
        | Op::Group { .. }
        | Op::Buffer { .. }
        | Op::Unique(_)
        | Op::Device(_)
        | Op::Call { .. }
        | Op::Barrier { .. } => None,

        Op::DefineLocal(_) | Op::DefineReg { .. } => {
            // Emit alloca for local/register memory.
            // Read base type and size from dtype (matching Tinygrad's x.dtype.base/x.dtype.size).
            // After devectorize's no_vectorized_buf, dtype is the canonical source of truth.
            let (base_dtype, alloc_size) = match uop.dtype() {
                DType::Ptr { base, size, .. } => (base.as_ref().clone(), size.unwrap_or(1)),
                other => (other, 1),
            };
            let base = ldt(&base_dtype);
            // Tinygrad: DEFINE_LOCAL gets align 16 (for SSE vector loads), DEFINE_REG gets default.
            let align = if matches!(uop.op(), Op::DefineLocal(_)) { ", align 16" } else { "" };
            kernel.push(format!("  {dst} = alloca [{alloc_size} x {base}]{align}"));
            Some(())
        }

        Op::Index { buffer, indices, .. } => {
            let buf = ctx.get(buffer);
            let buf_type = ldt(&buffer.dtype());

            if indices.is_empty() {
                kernel.push(format!("  {dst} = bitcast {buf_type} {buf} to {}", ldt(&uop.dtype())));
            } else {
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

                let elem_type = match uop.dtype() {
                    morok_dtype::DType::Ptr { ref base, .. } => ldt(base),
                    other => ldt(&other),
                };

                // Gate is NOT handled here — matching Tinygrad's approach where INDEX
                // always emits a plain GEP. The gate is handled at LOAD level (branch+phi)
                // and at STORE level (IF/ENDIF via line_rewrite_cleanups).
                kernel.push(format!(
                    "  {dst} = getelementptr inbounds {elem_type}, {buf_type} {buf}, {final_idx_type} {final_idx}"
                ));
            }
            Some(())
        }

        Op::PointerIndex { ptr, offset } => {
            let ptr_val = ctx.get(ptr);
            let off_val = ctx.get(offset);
            let elem_type = ldt(&uop.dtype());
            let ptr_type = ldt(&ptr.dtype());
            let off_type = ldt(&offset.dtype());

            kernel.push(format!(
                "  {dst} = getelementptr inbounds {elem_type}, {ptr_type} {ptr_val}, {off_type} {off_val}"
            ));
            Some(())
        }

        Op::Load { index, alt, .. } => {
            let idx = ctx.get(index);
            let dtype = ldt(&uop.dtype());
            let idx_type = ldt(&index.dtype());

            let (idx, idx_type) = maybe_extract_scalar_ptr(&dst, idx, &idx_type, &index.dtype(), kernel);

            // Gated LOAD: emit branch+phi to avoid null deref.
            // Matches Tinygrad's pattern (llvmir.py:123-129) which requires BOTH
            // a gated INDEX and an alt value on the LOAD. If gate exists without
            // alt, that's a pipeline bug (line_rewrite_cleanups should provide it).
            // Unwrap one CAST layer to find the INDEX gate (matches Tinygrad's .or_casted("idx")).
            // The pipeline CAN produce CAST(INDEX) — devectorize handles this shape explicitly.
            let actual_index = match index.op() {
                Op::Cast { src, .. } => src,
                _ => index,
            };
            let gate_info = if let Op::Index { gate: Some(gate_uop), .. } = actual_index.op() {
                let Some(alt_uop) = alt.as_ref() else {
                    ctx.set_invalid_graph(format!(
                        "gated LOAD on uop {} has no alt value; line_rewrite_cleanups must lift gated LOADs",
                        uop.id
                    ));
                    return None;
                };
                Some((ctx.get(gate_uop).to_string(), ctx.get(alt_uop).to_string()))
            } else {
                None
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
                kernel.push(format!("  {load_val} = load {dtype}, {idx_type} {idx}"));
                kernel.push(format!("  br label %{exit_label}"));
                kernel.push(format!("{exit_label}:"));
                kernel.push(format!("  {dst} = phi {dtype} [{load_val}, %{load_label}], [{alt_val}, %{entry_label}]"));
            } else {
                kernel.push(format!("  {dst} = load {dtype}, {idx_type} {idx}"));
            }
            Some(())
        }

        Op::Store { index, value, .. } => {
            let idx = ctx.get(index);
            let val = ctx.get(value);
            let val_type = ldt(&value.dtype());
            let idx_type = ldt(&index.dtype());

            let (idx, idx_type) = maybe_extract_scalar_ptr(&dst, idx, &idx_type, &index.dtype(), kernel);

            kernel.push(format!("  store {val_type} {val}, {idx_type} {idx}"));
            Some(())
        }

        Op::Binary(op, lhs, rhs) => {
            let l = ctx.get(lhs);
            let r = ctx.get(rhs);
            let ltype = ldt(&lhs.dtype());
            let rtype = ldt(&rhs.dtype());

            // Debug: detect type mismatch (logged via tracing)
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
            let stype = ldt(&src.dtype());

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
                    kernel.push(format!("  {dst} = fdiv nsz arcp contract afn {stype} 1.0, {sqrt_dst}"));
                }
                UnaryOp::Reciprocal => {
                    kernel.push(format!("  {dst} = fdiv nsz arcp contract afn {stype} 1.0, {s}"));
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
                        kernel.push(format!("  {gt_zero} = fcmp nsz arcp contract afn ogt {stype} {s}, 0.0"));
                        kernel.push(format!("  {lt_zero} = fcmp nsz arcp contract afn olt {stype} {s}, 0.0"));
                        kernel.push(format!("  {gt_ext} = uitofp i1 {gt_zero} to {stype}"));
                        kernel.push(format!("  {lt_ext} = uitofp i1 {lt_zero} to {stype}"));
                        kernel.push(format!("  {dst} = fsub nsz arcp contract afn {stype} {gt_ext}, {lt_ext}"));
                    } else if src.dtype().is_signed() {
                        let gt_zero = format!("{dst}.gt");
                        let lt_zero = format!("{dst}.lt");
                        let gt_ext = format!("{dst}.gt_ext");
                        let lt_ext = format!("{dst}.lt_ext");
                        kernel.push(format!("  {gt_zero} = icmp sgt {stype} {s}, 0"));
                        kernel.push(format!("  {lt_zero} = icmp slt {stype} {s}, 0"));
                        kernel.push(format!("  {gt_ext} = zext i1 {gt_zero} to {stype}"));
                        kernel.push(format!("  {lt_ext} = zext i1 {lt_zero} to {stype}"));
                        kernel.push(format!("  {dst} = sub {stype} {gt_ext}, {lt_ext}"));
                    } else {
                        // Unsigned: sign(x) = (x != 0) ? 1 : 0.
                        let ne_zero = format!("{dst}.ne");
                        kernel.push(format!("  {ne_zero} = icmp ne {stype} {s}, 0"));
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
            kernel.push(format!(
                "  {dst} = select {} {c}, {} {tv}, {} {fv}",
                ldt(&cond.dtype()),
                ldt(&t.dtype()),
                ldt(&f.dtype())
            ));
            Some(())
        }

        Op::Ternary(TernaryOp::MulAcc, a, b, c) => {
            let av = ctx.get(a);
            let bv = ctx.get(b);
            let cv = ctx.get(c);
            let dtype = ldt(&a.dtype());

            if a.dtype().is_float() {
                render_intrinsic(&dst, "fmuladd", &[(&dtype, av), (&dtype, bv), (&dtype, cv)], &dtype, kernel);
            } else {
                let mul_dst = format!("{dst}.mul");
                kernel.push(format!("  {mul_dst} = mul {dtype} {av}, {bv}"));
                kernel.push(format!("  {dst} = add {dtype} {mul_dst}, {cv}"));
            }
            Some(())
        }

        Op::Cast { src, dtype } => {
            let s = ctx.get(src);

            // INDEX always produces ptr in LLVM (via GEP), regardless of Morok dtype.
            // When source is INDEX, treat source LLVM type as ptr for cast selection.
            let is_index_src = matches!(src.op(), Op::Index { .. });
            let src_llvm_type = if is_index_src { "ptr".to_string() } else { ldt(&src.dtype()) };
            let dst_llvm_type = ldt(dtype);

            // CAST(INDEX) to Ptr is a no-op - INDEX already produces ptr via GEP.
            // This matches Tinygrad's approach (llvmir.py:189) where CAST to PtrDType
            // is register aliasing: r[u] = r[u.src[0]]
            if is_index_src && matches!(dtype, DType::Ptr { .. }) {
                // Emit a bitcast as a named no-op to maintain SSA form
                kernel.push(format!("  {dst} = bitcast ptr {s} to ptr"));
                return Some(());
            }

            if dtype.is_bool() && !src.dtype().is_bool() {
                // Cast to bool: compare != 0 (not trunc, which only takes the low bit).
                // Matches Tinygrad llvmir.py:99-101.
                let cmp = if src.dtype().is_float() { "fcmp nsz arcp contract afn une" } else { "icmp ne" };
                kernel.push(format!("  {dst} = {cmp} {src_llvm_type} {s}, zeroinitializer"));
            } else if src_llvm_type == dst_llvm_type {
                kernel.push(format!("  {dst} = bitcast {src_llvm_type} {s} to {dst_llvm_type}"));
            } else {
                let cast_instr = lcast(&src.dtype(), dtype);
                kernel.push(format!("  {dst} = {cast_instr} {src_llvm_type} {s} to {dst_llvm_type}"));
            }
            Some(())
        }

        Op::BitCast { src, dtype } => {
            let s = ctx.get(src);
            kernel.push(format!("  {dst} = bitcast {} {s} to {}", ldt(&src.dtype()), ldt(dtype)));
            Some(())
        }

        Op::Range { axis_id, end, .. } => {
            let id = axis_id.value();
            let dtype = ldt(&uop.dtype());
            let end_val = ctx.get(end).to_string();

            // Track range nesting for correct END footer ordering.
            ctx.push_range(id);

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

        Op::End { ranges, .. } => {
            // After pm_split_ends, each END has exactly one RANGE.
            // Use the range_stack to emit footer blocks in correct nesting order
            // (innermost first = LIFO), regardless of the END's ranges field order.
            let range_count = ranges.iter().filter(|r| matches!(r.op(), Op::Range { .. })).count();
            for _ in 0..range_count {
                if let Some(id) = ctx.pop_range() {
                    // Matches Tinygrad llvmir.py:166-170 exactly:
                    //   body → loop_footer → loop_latch (back edge)
                    //   loop_exit: falls through after loop
                    kernel.push(format!("  br label %loop_footer_{id}"));
                    kernel.push(format!("loop_footer_{id}:"));
                    kernel.push(format!("  br label %loop_latch_{id}"));
                    kernel.push(format!("loop_exit_{id}:"));
                }
            }

            let pending = ctx.take_pending_reduces();
            for (reduce_id, info) in pending {
                let result_name = format!("%reduce_{reduce_id}.final");
                kernel.push(format!("  {result_name} = load {}, ptr {}", info.dtype, info.acc_ptr));
                ctx.register(reduce_id, result_name);
            }
            Some(())
        }

        Op::Reduce { src, ranges, reduce_op } => {
            let src_val = ctx.get(src);
            let dtype = ldt(&uop.dtype());

            if ranges.is_empty() {
                kernel.push(format!("  {dst} = bitcast {dtype} {src_val} to {dtype}"));
            } else {
                let acc_ptr = format!("%reduce_{}", uop.id);
                let acc_load = format!("{acc_ptr}.load");
                let acc_new = format!("{acc_ptr}.new");
                let instr = reduce_instr(*reduce_op, &uop.dtype());

                kernel.push(format!("  {acc_load} = load {dtype}, ptr {acc_ptr}"));

                if matches!(reduce_op, ReduceOp::Max | ReduceOp::Min) {
                    render_reduce_minmax(&acc_new, *reduce_op, &uop.dtype(), &acc_load, src_val, &dtype, kernel);
                } else {
                    kernel.push(format!("  {acc_new} = {instr} {dtype} {acc_load}, {src_val}"));
                }

                kernel.push(format!("  store {dtype} {acc_new}, ptr {acc_ptr}"));
                ctx.register_reduce_pending(uop.id, acc_ptr.clone(), dtype.clone());
            }
            Some(())
        }

        Op::Gep { vector, indices } => {
            let vec = ctx.get(vector);
            let vec_type = ldt(&vector.dtype());
            let out_type = ldt(&uop.dtype());

            if indices.len() == 1 {
                kernel.push(format!("  {dst} = extractelement {vec_type} {vec}, i32 {}", indices[0]));
            } else {
                render_multi_gep(&dst, vec, &vector.dtype(), indices, &out_type, kernel);
            }
            Some(())
        }

        Op::Vectorize { elements } => {
            render_vectorize(&dst, elements, ctx, kernel);
            Some(())
        }

        Op::Cat { sources } => {
            render_cat(&dst, sources, ctx, kernel);
            Some(())
        }

        Op::PtrCat { .. } => {
            panic!(
                "PtrCat must be eliminated before codegen (devectorize should distribute it into scalar loads/stores)"
            );
        }

        Op::Contract { src, .. } | Op::Unroll { src, .. } | Op::Detach { src } => {
            let s = ctx.get(src);
            ctx.alias(uop.id, s.to_string());
            None
        }

        Op::After { passthrough, .. } => {
            #[cfg(debug_assertions)]
            if matches!(passthrough.op(), Op::Range { .. }) {
                panic!("AFTER passthrough is Range (id={}), this violates Tinygrad semantics", passthrough.id);
            }
            let s = ctx.get(passthrough);
            ctx.alias(uop.id, s.to_string());
            None
        }

        Op::Bind { var, value } => {
            let v = ctx.get(value);
            ctx.alias(var.id, v.to_string());
            None
        }

        Op::If { condition, .. } => {
            let cond = ctx.get(condition);
            let if_id = uop.id;
            kernel.push(format!("  br i1 {cond}, label %if_then_{if_id}, label %if_end_{if_id}"));
            kernel.push(format!("if_then_{if_id}:"));
            Some(())
        }

        Op::EndIf { if_op } => {
            let if_id = if_op.id;
            kernel.push(format!("  br label %if_end_{if_id}"));
            kernel.push(format!("if_end_{if_id}:"));
            Some(())
        }

        // CUSTOM / CUSTOMI are intentionally absent: tinygrad's `llvmir.py`
        // doesn't handle CUSTOM either, and the LLVM text renderer rejects
        // these ops at the entry point with a typed error before they reach
        // here (see `llvm/text/mod.rs`).
        op if op.is_movement() => {
            panic!(
                "movement op {:?} (id={}) reached LLVM codegen — \
                 should have been eliminated during rangeify. \
                 This indicates a bug in remove_movement_op or apply_bufferize_transform.",
                std::mem::discriminant(op),
                uop.id,
            );
        }

        _ => {
            kernel.push(format!("; UNSUPPORTED: {:?}", uop.op()));
            None
        }
    }
}

fn binary_instr(op: BinaryOp, dtype: &DType) -> &'static str {
    assert!(
        !matches!(dtype.base(), morok_dtype::ScalarDType::Index),
        "Index dtype reached LLVM codegen binary_instr({op:?}, {dtype:?}) — \
         pm_lower_index_dtype should have lowered it to i32/i64"
    );
    let is_float = dtype.is_float();
    let is_signed = dtype.is_signed();

    match op {
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
        BinaryOp::Idiv => {
            if is_signed {
                "sdiv"
            } else {
                "udiv"
            }
        }
        BinaryOp::Mod => {
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
        BinaryOp::Lt => {
            if is_float {
                "fcmp nsz arcp contract afn ult"
            } else if is_signed {
                "icmp slt"
            } else {
                "icmp ult"
            }
        }
        BinaryOp::Le => {
            if is_float {
                "fcmp nsz arcp contract afn ule"
            } else if is_signed {
                "icmp sle"
            } else {
                "icmp ule"
            }
        }
        BinaryOp::Gt => {
            if is_float {
                "fcmp nsz arcp contract afn ugt"
            } else if is_signed {
                "icmp sgt"
            } else {
                "icmp ugt"
            }
        }
        BinaryOp::Ge => {
            if is_float {
                "fcmp nsz arcp contract afn uge"
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

fn reduce_instr(op: ReduceOp, dtype: &DType) -> &'static str {
    let is_float = dtype.is_float();
    let is_signed = dtype.is_signed();

    match op {
        ReduceOp::Add => {
            if is_float {
                "fadd nsz arcp contract afn"
            } else {
                "add"
            }
        }
        ReduceOp::Mul => {
            if is_float {
                "fmul nsz arcp contract afn"
            } else {
                "mul"
            }
        }
        ReduceOp::Max => {
            if is_float {
                "maxnum"
            } else if is_signed {
                "smax"
            } else {
                "umax"
            }
        }
        ReduceOp::Min => {
            if is_float {
                "minnum"
            } else if is_signed {
                "smin"
            } else {
                "umin"
            }
        }
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

fn render_reduce_minmax(
    dst: &str,
    op: ReduceOp,
    dtype: &DType,
    acc: &str,
    val: &str,
    ltype: &str,
    kernel: &mut Vec<String>,
) {
    if dtype.is_float() {
        let intrinsic = match op {
            ReduceOp::Max => "maxnum",
            ReduceOp::Min => "minnum",
            _ => unreachable!(),
        };
        render_intrinsic(dst, intrinsic, &[(ltype, acc), (ltype, val)], ltype, kernel);
    } else {
        let is_signed = dtype.is_signed();
        let cmp = match op {
            ReduceOp::Max => {
                if is_signed {
                    "sgt"
                } else {
                    "ugt"
                }
            }
            ReduceOp::Min => {
                if is_signed {
                    "slt"
                } else {
                    "ult"
                }
            }
            _ => unreachable!(),
        };
        let cmp_dst = format!("{dst}.cmp");
        kernel.push(format!("  {cmp_dst} = icmp {cmp} {ltype} {acc}, {val}"));
        kernel.push(format!("  {dst} = select i1 {cmp_dst}, {ltype} {acc}, {ltype} {val}"));
    }
}

fn render_multi_gep(
    dst: &str,
    vec: &str,
    vec_dtype: &DType,
    indices: &[usize],
    out_type: &str,
    kernel: &mut Vec<String>,
) {
    let vec_type = ldt(vec_dtype);

    let elem_dtype = match vec_dtype {
        DType::Ptr { base, addrspace, size, .. } => {
            DType::Ptr { base: base.clone(), addrspace: *addrspace, size: *size, vcount: 1 }
        }
        DType::Vector { scalar, .. } => DType::Scalar(*scalar),
        _ => DType::Scalar(vec_dtype.base()),
    };
    let elem_type = ldt(&elem_dtype);

    for (i, &idx) in indices.iter().enumerate() {
        let elem = format!("{dst}.e{i}");
        kernel.push(format!("  {elem} = extractelement {vec_type} {vec}, i32 {idx}"));
    }

    if indices.len() == 1 {
        kernel.push(format!("  {dst} = bitcast {elem_type} {dst}.e0 to {out_type}"));
    } else {
        let count = indices.len();
        kernel.push(format!("  {dst}.undef = undef <{count} x {elem_type}>"));
        let mut prev = format!("{dst}.undef");
        for i in 0..count {
            let next = if i == count - 1 { dst.to_string() } else { format!("{dst}.v{i}") };
            kernel.push(format!(
                "  {next} = insertelement <{count} x {elem_type}> {prev}, {elem_type} {dst}.e{i}, i32 {i}"
            ));
            prev = next;
        }
    }
}

fn render_vectorize(dst: &str, elements: &[Arc<UOp>], ctx: &RenderContext, kernel: &mut Vec<String>) {
    if elements.is_empty() {
        return;
    }

    let scalar_type = ldt(&elements[0].dtype());
    let count = elements.len();
    let vec_type = format!("<{count} x {scalar_type}>");

    let mut prev = "undef".to_string();
    for (i, elem) in elements.iter().enumerate() {
        let val = ctx.get(elem);
        let next = if i == count - 1 { dst.to_string() } else { format!("{dst}.v{i}") };
        kernel.push(format!("  {next} = insertelement {vec_type} {prev}, {scalar_type} {val}, i32 {i}"));
        prev = next;
    }
}

fn render_cat(dst: &str, sources: &[Arc<UOp>], ctx: &RenderContext, kernel: &mut Vec<String>) {
    if sources.is_empty() {
        return;
    }

    let total_count: usize = sources.iter().map(|s| s.dtype().vcount()).sum();
    let scalar_type = ldt(&sources[0].dtype().scalar_dtype());
    let out_type = format!("<{total_count} x {scalar_type}>");

    let mut out_idx = 0;
    let mut prev = "undef".to_string();

    for src in sources.iter() {
        let src_val = ctx.get(src);
        let src_count = src.dtype().vcount();

        if src_count == 1 {
            let next = if out_idx == total_count - 1 { dst.to_string() } else { format!("{dst}.c{out_idx}") };
            kernel.push(format!("  {next} = insertelement {out_type} {prev}, {scalar_type} {src_val}, i32 {out_idx}"));
            prev = next;
            out_idx += 1;
        } else {
            let src_type = ldt(&src.dtype());
            for i in 0..src_count {
                let elem = format!("{dst}.e{out_idx}");
                kernel.push(format!("  {elem} = extractelement {src_type} {src_val}, i32 {i}"));

                let next = if out_idx == total_count - 1 { dst.to_string() } else { format!("{dst}.c{out_idx}") };
                kernel.push(format!("  {next} = insertelement {out_type} {prev}, {scalar_type} {elem}, i32 {out_idx}"));
                prev = next;
                out_idx += 1;
            }
        }
    }
}

/// Get identity element for reduce operation.
pub fn reduce_identity(op: ReduceOp, dtype: &DType) -> String {
    let is_vector = matches!(dtype, DType::Vector { .. });

    match op {
        ReduceOp::Add => {
            if is_vector {
                "zeroinitializer".to_string()
            } else if dtype.is_float() {
                "0.0".to_string()
            } else {
                "0".to_string()
            }
        }
        ReduceOp::Mul => {
            if is_vector {
                "zeroinitializer".to_string()
            } else if dtype.is_float() {
                "1.0".to_string()
            } else {
                "1".to_string()
            }
        }
        ReduceOp::Max => {
            if is_vector {
                "zeroinitializer".to_string()
            } else if dtype.is_float() {
                "-0x7FF0000000000000".to_string()
            } else if dtype.is_signed() {
                i64::MIN.to_string()
            } else {
                "0".to_string()
            }
        }
        ReduceOp::Min => {
            if is_vector {
                "zeroinitializer".to_string() // TODO: proper +inf splat
            } else if dtype.is_float() {
                "0x7FF0000000000000".to_string() // +inf
            } else if dtype.is_signed() {
                i64::MAX.to_string()
            } else {
                u64::MAX.to_string()
            }
        }
    }
}
