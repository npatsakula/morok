//! MLIR-based code generation using Melior (MLIR Rust bindings).
//!
//! Generates MLIR modules using arith + LLVM dialects, then lowers to pure LLVM
//! dialect for text serialization. Produces the same kernel ABI as the LLVM text backend:
//! `void @kernel(ptr %args, ptr %vars)`

pub mod ctx;
pub mod ops;
pub mod types;

use std::sync::Arc;

use melior::dialect::{arith, llvm, DialectRegistry};
use melior::ir::attribute::{IntegerAttribute, StringAttribute, TypeAttribute};
use melior::ir::block::BlockLike;
use melior::ir::operation::OperationBuilder;
use melior::ir::r#type::{FunctionType, IntegerType};
use melior::ir::{Block, Identifier, Location, Module, Region, Type};
use melior::pass::PassManager;
use melior::utility::{register_all_dialects, register_all_llvm_translations};
use melior::Context;
use mlir_sys::MlirBlock;
use morok_dtype::DType;
use morok_ir::pattern::TypedPatternMatcher;
use morok_ir::rewrite::graph_rewrite_bottom_up;
use morok_ir::{AxisType, ConstValue, Op, prelude::*};
use morok_schedule::linearize::{line_rewrite_cleanups, linearize_with_cfg};
use morok_schedule::rangeify::patterns::pm_bool_devectorize;

use self::ctx::{LoopBlocks, RenderContext};
use self::ops::*;
use self::types::{mlir_ptr_type, mlir_type};
use crate::{BufferArg, RenderedKernel, Renderer, Result};
use crate::error::MlirSnafu;
use snafu::OptionExt;

/// MLIR-based renderer using Melior bindings.
pub struct MlirRenderer;

impl MlirRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MlirRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for MlirRenderer {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
        let kernel_name = name.unwrap_or("kernel");

        let uop = graph_rewrite_bottom_up(&pm_bool_devectorize(), uop.clone(), &mut ());
        let nodes = linearize_with_cfg(uop);
        let nodes = line_rewrite_cleanups(nodes);

        // Collect buffers, variables, thread info
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

        let thread_info: Option<(Arc<UOp>, usize)> = nodes.iter().find_map(|n| {
            if let Op::Range { axis_type, end, .. } = n.op()
                && matches!(axis_type, AxisType::Thread)
                && let Op::Const(cv) = end.op()
                && let ConstValue::Int(count) = cv.0
            {
                Some((n.clone(), count as usize))
            } else {
                None
            }
        });

        let has_threading = thread_info.is_some();
        let thread_count = thread_info.as_ref().map(|(_, c)| *c).unwrap_or(1);

        // Build buffer_args and var_names for RenderedKernel metadata
        let mut buffer_args: Vec<BufferArg> = Vec::new();
        let mut var_names: Vec<String> = Vec::new();

        for (i, buf) in buffers.iter().enumerate() {
            if let Op::DefineGlobal(id) = buf.op() {
                let is_output = is_output_buffer(buf, &nodes);
                buffer_args.push(BufferArg {
                    index: *id,
                    name: format!("data{i}"),
                    dtype: buf.dtype(),
                    is_output,
                });
            }
        }
        for var in &variables {
            if let Op::DefineVar { name, .. } = var.op() {
                var_names.push(name.clone());
            }
        }
        if has_threading {
            var_names.push("thread_id".to_string());
        }

        // === Build MLIR module ===
        let context = Context::new();
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        register_all_llvm_translations(&context);

        let loc = Location::unknown(&context);
        let module = Module::new(loc);

        let ptr_type = mlir_ptr_type(&context);
        let func_type = FunctionType::new(&context, &[ptr_type, ptr_type], &[]).into();
        let i64_type: Type = IntegerType::new(&context, 64).into();

        // Build function body
        let entry_block = Block::new(&[(ptr_type, loc), (ptr_type, loc)]);
        let entry_raw: MlirBlock = unsafe {
            mlir_sys::mlirBlockFromRef(entry_block.to_raw())
        };

        let mut rctx = RenderContext::new(entry_raw);

        let args_ptr = entry_block.argument(0).unwrap().into();
        let vars_ptr = entry_block.argument(1).unwrap().into();

        // Load buffer pointers: GEP into %args + load
        for (i, buf) in buffers.iter().enumerate() {
            let block = &entry_block;
            let idx = const_i64(&context, block, i as i64, loc);
            let buf_ptr_ptr = block
                .append_operation(llvm::get_element_ptr_dynamic(
                    &context, args_ptr, &[idx], ptr_type, ptr_type, loc,
                ))
                .result(0)
                .unwrap()
                .into();
            let buf_ptr = block
                .append_operation(llvm::load(&context, buf_ptr_ptr, ptr_type, loc, Default::default()))
                .result(0)
                .unwrap()
                .into();
            rctx.register(buf.id, buf_ptr);
        }

        // Load variables: GEP into %vars + load i64 + optional trunc
        for (i, var) in variables.iter().enumerate() {
            let block = &entry_block;
            let idx = const_i64(&context, block, i as i64, loc);
            let var_ptr = block
                .append_operation(llvm::get_element_ptr_dynamic(
                    &context, vars_ptr, &[idx], i64_type, ptr_type, loc,
                ))
                .result(0)
                .unwrap()
                .into();
            let var_i64 = block
                .append_operation(llvm::load(&context, var_ptr, i64_type, loc, Default::default()))
                .result(0)
                .unwrap()
                .into();

            let var_dtype = var.dtype();
            let target_type = mlir_type(&context, &var_dtype);
            let var_val = if target_type == i64_type {
                var_i64
            } else {
                block
                    .append_operation(arith::trunci(var_i64, target_type, loc))
                    .result(0)
                    .unwrap()
                    .into()
            };
            rctx.register(var.id, var_val);
        }

        // Load thread_id if present
        if let Some((ref thread_range, _)) = thread_info {
            let block = &entry_block;
            let thread_idx = const_i64(&context, block, variables.len() as i64, loc);
            let thread_ptr = block
                .append_operation(llvm::get_element_ptr_dynamic(
                    &context, vars_ptr, &[thread_idx], i64_type, ptr_type, loc,
                ))
                .result(0)
                .unwrap()
                .into();
            let thread_i64 = block
                .append_operation(llvm::load(&context, thread_ptr, i64_type, loc, Default::default()))
                .result(0)
                .unwrap()
                .into();

            let range_dtype = thread_range.dtype();
            let range_type = mlir_type(&context, &range_dtype);
            let thread_val = if range_type == i64_type {
                thread_i64
            } else {
                block
                    .append_operation(arith::trunci(thread_i64, range_type, loc))
                    .result(0)
                    .unwrap()
                    .into()
            };
            rctx.register(thread_range.id, thread_val);
        }

        // Pre-allocate reduce accumulators
        for node in &nodes {
            if let Op::Reduce { reduce_op, .. } = node.op() {
                let block = &entry_block;
                let dtype = &node.dtype();
                let acc_type = mlir_type(&context, dtype);
                let result_type = mlir_type(&context, dtype);

                // alloca
                let one = const_i64(&context, block, 1, loc);
                let acc_ptr = block
                    .append_operation(llvm::alloca(&context, one, acc_type, loc, Default::default()))
                    .result(0)
                    .unwrap()
                    .into();

                // store identity
                let identity = build_reduce_identity(&context, block, *reduce_op, dtype, loc);
                block.append_operation(llvm::store(&context, identity, acc_ptr, loc, Default::default()));

                rctx.register_reduce_pending(node.id, acc_ptr, result_type);
            }
        }

        // Pre-register constants
        for node in &nodes {
            if let Op::Const(cv) = node.op() {
                let block = unsafe { rctx.current_block_ref() };
                let val = build_const(&context, &block, &cv.0, &node.dtype(), loc);
                rctx.register(node.id, val);
            }
        }

        // Pre-register range variables (non-thread)
        for node in &nodes {
            if let Op::Range { axis_type, .. } = node.op()
                && !matches!(axis_type, AxisType::Thread)
            {
                // Will be registered during Range processing with block argument
            }
        }

        // Build function region, move entry block into it
        let func_region = Region::new();
        let entry_ref = func_region.append_block(entry_block);
        rctx.set_current_block(unsafe { mlir_sys::mlirBlockFromRef(entry_ref.to_raw()) });

        // Process linearized nodes
        for node in &nodes {
            render_node(&context, &func_region, &mut rctx, node, &thread_info, loc);
        }

        // Emit return on the final current block
        unsafe {
            let final_block = rctx.current_block_ref();
            final_block.append_operation(llvm::r#return(None, loc));
        }

        // Build llvm.func operation
        let func_op = {
            let func_name = StringAttribute::new(&context, kernel_name);
            let func_type_attr = TypeAttribute::new(func_type);
            OperationBuilder::new("llvm.func", loc)
                .add_attributes(&[
                    (Identifier::new(&context, "sym_name"), func_name.into()),
                    (Identifier::new(&context, "function_type"), func_type_attr.into()),
                ])
                .add_regions(vec![func_region])
                .build()
                .expect("valid llvm.func")
        };

        module.body().append_operation(func_op);

        // Run pass pipeline: arith-to-llvm + index-to-llvm
        let mut module = module;
        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(melior::pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(melior::pass::conversion::create_index_to_llvm());
        pass_manager.add_pass(melior::pass::conversion::create_reconcile_unrealized_casts());
        pass_manager.run(&mut module).map_err(|e| {
            crate::error::Error::MlirError { reason: format!("pass pipeline failed: {e}") }
        })?;

        // Verify module
        if !module.as_operation().verify() {
            return Err(crate::error::Error::MlirError {
                reason: "MLIR module verification failed".to_string(),
            });
        }

        let code = module.as_operation().to_string();

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
        "mlir"
    }

    fn decompositor(&self) -> Option<TypedPatternMatcher<()>> {
        None
    }
}

/// Process a single linearized node into MLIR operations.
fn render_node<'c>(
    ctx: &'c Context,
    region: &Region<'c>,
    rctx: &mut RenderContext<'c>,
    node: &Arc<UOp>,
    thread_info: &Option<(Arc<UOp>, usize)>,
    loc: Location<'c>,
) {
    match node.op() {
        // Skip meta-ops and already-handled ops
        Op::Const(_)
        | Op::DefineGlobal(_)
        | Op::DefineLocal(_)
        | Op::DefineVar { .. }
        | Op::Noop
        | Op::Sink { .. }
        | Op::Group { .. }
        | Op::Buffer { .. }
        | Op::Unique(_)
        | Op::Device(_)
        | Op::Kernel { .. }
        | Op::Barrier { .. } => {}

        Op::VConst { values, .. } => {
            let block = unsafe { rctx.current_block_ref() };
            let val = build_vconst(ctx, &block, values, &node.dtype(), loc);
            rctx.register(node.id, val);
        }

        Op::DefineReg { size } => {
            let block = unsafe { rctx.current_block_ref() };
            let base_dtype = match node.dtype() {
                DType::Ptr { base, .. } => base.as_ref().clone(),
                other => other,
            };
            let elem_type = mlir_type(ctx, &base_dtype);
            let arr_type = Type::parse(ctx, &format!("!llvm.array<{size} x {}>",
                elem_type_to_mlir_string(ctx, &base_dtype))).unwrap_or(elem_type);
            let one = const_i64(ctx, &block, 1, loc);
            let alloca_val = block
                .append_operation(llvm::alloca(ctx, one, arr_type, loc, Default::default()))
                .result(0)
                .unwrap()
                .into();
            rctx.register(node.id, alloca_val);
        }

        Op::Index { buffer, indices, gate } => {
            let block = unsafe { rctx.current_block_ref() };
            let buf_val = rctx.get(buffer.id);

            if indices.is_empty() {
                rctx.register(node.id, buf_val);
            } else {
                let idx_val = rctx.get(indices[0].id);
                let elem_dtype = match node.dtype() {
                    DType::Ptr { ref base, .. } => base.as_ref().clone(),
                    other => other,
                };
                let elem_type = mlir_type(ctx, &elem_dtype);

                let gep = block
                    .append_operation(llvm::get_element_ptr_dynamic(
                        ctx, buf_val, &[idx_val], elem_type, mlir_ptr_type(ctx), loc,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                let result = if let Some(gate_node) = gate {
                    let gate_val = rctx.get(gate_node.id);
                    let null_ptr = block
                        .append_operation(llvm::zero(mlir_ptr_type(ctx), loc))
                        .result(0)
                        .unwrap()
                        .into();
                    block
                        .append_operation(arith::select(gate_val, gep, null_ptr, loc))
                        .result(0)
                        .unwrap()
                        .into()
                } else {
                    gep
                };
                rctx.register(node.id, result);
            }
        }

        Op::PointerIndex { ptr, offset } => {
            let block = unsafe { rctx.current_block_ref() };
            let ptr_val = rctx.get(ptr.id);
            let off_val = rctx.get(offset.id);
            let elem_dtype = match node.dtype() {
                DType::Ptr { ref base, .. } => base.as_ref().clone(),
                other => other,
            };
            let elem_type = mlir_type(ctx, &elem_dtype);
            let result = block
                .append_operation(llvm::get_element_ptr_dynamic(
                    ctx, ptr_val, &[off_val], elem_type, mlir_ptr_type(ctx), loc,
                ))
                .result(0)
                .unwrap()
                .into();
            rctx.register(node.id, result);
        }

        Op::Load { index, .. } => {
            let block = unsafe { rctx.current_block_ref() };
            let idx_val = rctx.get(index.id);
            let load_type = mlir_type(ctx, &node.dtype());
            let result = block
                .append_operation(llvm::load(ctx, idx_val, load_type, loc, Default::default()))
                .result(0)
                .unwrap()
                .into();
            rctx.register(node.id, result);
        }

        Op::Store { index, value, .. } => {
            let block = unsafe { rctx.current_block_ref() };
            let idx_val = rctx.get(index.id);
            let val = rctx.get(value.id);
            block.append_operation(llvm::store(ctx, val, idx_val, loc, Default::default()));
        }

        Op::Binary(op, lhs, rhs) => {
            let block = unsafe { rctx.current_block_ref() };
            let l = rctx.get(lhs.id);
            let r = rctx.get(rhs.id);
            let result = render_binary(ctx, &block, *op, l, r, &lhs.dtype(), loc);
            rctx.register(node.id, result);
        }

        Op::Unary(op, src) => {
            let block = unsafe { rctx.current_block_ref() };
            let s = rctx.get(src.id);
            let result = render_unary(ctx, &block, *op, s, &src.dtype(), loc);
            rctx.register(node.id, result);
        }

        Op::Ternary(op, a, b, c) => {
            let block = unsafe { rctx.current_block_ref() };
            let av = rctx.get(a.id);
            let bv = rctx.get(b.id);
            let cv = rctx.get(c.id);
            let result = render_ternary(ctx, &block, *op, av, bv, cv, &node.dtype(), loc);
            rctx.register(node.id, result);
        }

        Op::Cast { src, dtype } => {
            let block = unsafe { rctx.current_block_ref() };
            let s = rctx.get(src.id);
            let result = render_cast(ctx, &block, s, &src.dtype(), dtype, loc);
            rctx.register(node.id, result);
        }

        Op::BitCast { src, dtype } => {
            let block = unsafe { rctx.current_block_ref() };
            let s = rctx.get(src.id);
            let to_type = mlir_type(ctx, dtype);
            let result = block
                .append_operation(arith::bitcast(s, to_type, loc))
                .result(0)
                .unwrap()
                .into();
            rctx.register(node.id, result);
        }

        Op::Range { end, axis_id, axis_type, .. } => {
            if matches!(axis_type, AxisType::Thread) {
                // Thread ranges are pre-loaded from vars
                return;
            }

            let range_dtype = node.dtype();
            let range_type = mlir_type(ctx, &range_dtype);
            let end_val = rctx.get(end.id);

            // Create 4 blocks: latch, body, footer, exit
            let latch_block = Block::new(&[(range_type, loc)]);
            let body_block = Block::new(&[]);
            let footer_block = Block::new(&[]);
            let exit_block = Block::new(&[]);

            // Branch from current block to latch with initial IV = 0
            let current_block = unsafe { rctx.current_block_ref() };
            let zero = const_int(ctx, &current_block, 0, range_type, loc);
            br(&current_block, &latch_block, &[zero], loc);

            // Append blocks to region
            let latch_ref = region.append_block(latch_block);
            let body_ref = region.append_block(body_block);
            let footer_ref = region.append_block(footer_block);
            let exit_ref = region.append_block(exit_block);

            // Latch block: compare IV < end, branch to body or exit
            let iv = latch_ref.argument(0).unwrap().into();
            let cmp = icmp(ctx, &latch_ref, "ult", iv, end_val, loc);
            cond_br(ctx, &latch_ref, cmp, &body_ref, &exit_ref, &[], &[], loc);

            // Store raw block handles
            let loop_blocks = LoopBlocks {
                latch: unsafe { mlir_sys::mlirBlockFromRef(latch_ref.to_raw()) },
                body: unsafe { mlir_sys::mlirBlockFromRef(body_ref.to_raw()) },
                footer: unsafe { mlir_sys::mlirBlockFromRef(footer_ref.to_raw()) },
                exit: unsafe { mlir_sys::mlirBlockFromRef(exit_ref.to_raw()) },
            };

            rctx.register(node.id, iv);
            rctx.register_loop(axis_id.value(), loop_blocks);
            rctx.set_current_block(unsafe { mlir_sys::mlirBlockFromRef(body_ref.to_raw()) });
        }

        Op::End { ranges, .. } => {
            // Close each range's loop
            for range in ranges.iter() {
                if let Op::Range { axis_id, axis_type, .. } = range.op() {
                    if matches!(axis_type, AxisType::Thread) {
                        continue;
                    }
                    let id = axis_id.value();
                    let range_dtype = range.dtype();
                    let range_type = mlir_type(ctx, &range_dtype);

                    let loops = rctx.get_loop(id).expect("loop blocks for End");

                    // Branch from current block (body end) to footer
                    let current = unsafe { rctx.current_block_ref() };
                    let footer = unsafe { loops.footer_ref() };
                    br(&current, &footer, &[], loc);

                    // Footer: increment IV, branch back to latch
                    let latch = unsafe { loops.latch_ref() };
                    let iv = rctx.get(range.id);
                    let one = const_int(ctx, &footer, 1, range_type, loc);
                    let iv_next = footer
                        .append_operation(arith::addi(iv, one, loc))
                        .result(0)
                        .unwrap()
                        .into();
                    br(&footer, &latch, &[iv_next], loc);

                    // Set current block to exit
                    let exit_raw = loops.exit;
                    rctx.set_current_block(exit_raw);
                }
            }

            // Finalize pending reduces: load final values
            let pending = rctx.take_pending_reduces();
            for (reduce_id, info) in pending {
                let block = unsafe { rctx.current_block_ref() };
                let final_val = block
                    .append_operation(llvm::load(ctx, info.acc_ptr, info.result_type, loc, Default::default()))
                    .result(0)
                    .unwrap()
                    .into();
                rctx.register(reduce_id, final_val);
            }
        }

        Op::Reduce { src, ranges, reduce_op } => {
            if ranges.is_empty() {
                // Passthrough reduce (no ranges)
                let s = rctx.get(src.id);
                rctx.register(node.id, s);
            } else {
                let block = unsafe { rctx.current_block_ref() };
                let src_val = rctx.get(src.id);
                // The pending reduce was pre-allocated; look it up by node.id
                // We need the acc_ptr. Since take_pending_reduces isn't called yet,
                // we can look it up. But our API only has take_pending_reduces.
                // Solution: use the acc_ptr we stored.
                // Actually, we need direct access to the pending reduce's acc_ptr.
                // Let's get it from try_get - the acc_ptr was registered as the node value
                // during pre-allocation... No, we used register_reduce_pending.
                // We need to add a method to peek at pending reduces.
                // For now, the render_reduce_accumulate helper expects acc_ptr directly.
                // We registered the pending reduce during pre-allocation. We need to peek.
                // Let me just get the pending reduce info from the context.
                if let Some(info) = rctx.peek_pending_reduce(node.id) {
                    let acc_ptr = info.acc_ptr;
                    render_reduce_accumulate(ctx, &block, *reduce_op, src_val, acc_ptr, &node.dtype(), loc);
                }
            }
        }

        Op::Gep { vector, indices } => {
            let block = unsafe { rctx.current_block_ref() };
            let vec_val = rctx.get(vector.id);
            let scalar_type = mlir_type(ctx, &node.dtype());

            if indices.len() == 1 {
                let result = render_extractelement(ctx, &block, vec_val, indices[0], scalar_type, loc);
                rctx.register(node.id, result);
            } else {
                // Multi-index GEP: extract multiple elements and build a smaller vector
                let vec_type = mlir_type(ctx, &node.dtype());
                let mut elements = Vec::new();
                for &idx in indices {
                    let elem = render_extractelement(ctx, &block, vec_val, idx, scalar_type, loc);
                    elements.push(elem);
                }
                if elements.len() == 1 {
                    rctx.register(node.id, elements[0]);
                } else {
                    let result = render_vectorize(ctx, &block, &elements, vec_type, scalar_type, loc);
                    rctx.register(node.id, result);
                }
            }
        }

        Op::Vectorize { elements } => {
            let block = unsafe { rctx.current_block_ref() };
            let elem_vals: Vec<_> = elements.iter().map(|e| rctx.get(e.id)).collect();
            let vec_type = mlir_type(ctx, &node.dtype());
            let scalar_type = mlir_type(ctx, &node.dtype().scalar_dtype());
            let result = render_vectorize(ctx, &block, &elem_vals, vec_type, scalar_type, loc);
            rctx.register(node.id, result);
        }

        Op::Cat { sources } => {
            let block = unsafe { rctx.current_block_ref() };
            let vec_type = mlir_type(ctx, &node.dtype());
            let scalar_dtype = node.dtype().scalar_dtype();
            let scalar_type = mlir_type(ctx, &scalar_dtype);

            let mut current = block
                .append_operation(llvm::undef(vec_type, loc))
                .result(0)
                .unwrap()
                .into();
            let mut out_idx = 0usize;

            for src in sources {
                let src_val = rctx.get(src.id);
                let src_count = src.dtype().vcount();
                if src_count == 1 {
                    let idx = const_i32(ctx, &block, out_idx as i64, loc);
                    current = block
                        .append_operation(llvm::insert_element(current, src_val, idx, loc))
                        .result(0)
                        .unwrap()
                        .into();
                    out_idx += 1;
                } else {
                    let src_scalar_type = mlir_type(ctx, &src.dtype().scalar_dtype());
                    for i in 0..src_count {
                        let elem = render_extractelement(ctx, &block, src_val, i, src_scalar_type, loc);
                        let idx = const_i32(ctx, &block, out_idx as i64, loc);
                        current = block
                            .append_operation(llvm::insert_element(current, elem, idx, loc))
                            .result(0)
                            .unwrap()
                            .into();
                        out_idx += 1;
                    }
                }
            }
            rctx.register(node.id, current);
        }

        Op::PtrCat { sources } => {
            let block = unsafe { rctx.current_block_ref() };
            let vec_type = mlir_type(ctx, &node.dtype());

            let mut current = block
                .append_operation(llvm::undef(vec_type, loc))
                .result(0)
                .unwrap()
                .into();
            for (i, src) in sources.iter().enumerate() {
                let src_val = rctx.get(src.id);
                let idx = const_i32(ctx, &block, i as i64, loc);
                current = block
                    .append_operation(llvm::insert_element(current, src_val, idx, loc))
                    .result(0)
                    .unwrap()
                    .into();
            }
            rctx.register(node.id, current);
        }

        Op::Contract { src, .. } | Op::Unroll { src, .. } | Op::Detach { src } => {
            let s = rctx.get(src.id);
            rctx.register(node.id, s);
        }

        Op::After { passthrough, .. } => {
            let s = rctx.get(passthrough.id);
            rctx.register(node.id, s);
        }

        Op::Bind { var, value } => {
            let v = rctx.get(value.id);
            rctx.register(var.id, v);
        }

        Op::If { condition, .. } => {
            let current = unsafe { rctx.current_block_ref() };
            let cond_val = rctx.get(condition.id);

            let then_block = Block::new(&[]);
            let end_block = Block::new(&[]);

            let then_ref = region.append_block(then_block);
            let end_ref = region.append_block(end_block);

            cond_br(ctx, &current, cond_val, &then_ref, &end_ref, &[], &[], loc);

            let end_raw = unsafe { mlir_sys::mlirBlockFromRef(end_ref.to_raw()) };
            rctx.register_if_end(node.id, end_raw);
            rctx.set_current_block(unsafe { mlir_sys::mlirBlockFromRef(then_ref.to_raw()) });
        }

        Op::EndIf { if_op } => {
            let current = unsafe { rctx.current_block_ref() };

            if let Some(end_raw) = rctx.take_if_end(if_op.id) {
                let end_block = unsafe { melior::ir::BlockRef::from_raw(end_raw) };
                br(&current, &end_block, &[], loc);
                rctx.set_current_block(end_raw);
            }
        }

        op if op.is_movement() => {
            panic!(
                "movement op {:?} (id={}) reached MLIR codegen — should have been eliminated during rangeify",
                std::mem::discriminant(op),
                node.id,
            );
        }

        _ => {
            tracing::warn!(op = ?node.op(), id = node.id, "unsupported op in MLIR codegen");
        }
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

fn elem_type_to_mlir_string(ctx: &Context, dtype: &DType) -> String {
    let ty = mlir_type(ctx, dtype);
    format!("{ty}")
}
