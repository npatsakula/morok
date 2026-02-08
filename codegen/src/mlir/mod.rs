//! MLIR-based code generation using Melior (MLIR Rust bindings).
//!
//! Generates MLIR modules using arith + scf + LLVM dialects, then lowers to pure LLVM
//! dialect for text serialization. Produces the same kernel ABI as the LLVM text backend:
//! `void @kernel(ptr %args, ptr %vars)`

pub mod ctx;
pub mod ops;
pub mod types;

use std::collections::HashMap;
use std::sync::Arc;

use melior::Context;
use melior::dialect::{DialectRegistry, arith, llvm, scf};
use melior::ir::RegionLike;
use melior::ir::attribute::{StringAttribute, TypeAttribute};
use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;
use melior::ir::r#type::IntegerType;
use melior::ir::{Block, Location, Module, Region, Type};
use melior::pass::PassManager;
use melior::utility::{register_all_dialects, register_all_llvm_translations};
use morok_dtype::DType;
use morok_ir::pattern::TypedPatternMatcher;
use morok_ir::rewrite::graph_rewrite_bottom_up;
use morok_ir::{AxisType, ConstValue, Op, ReduceOp, prelude::*};
use morok_schedule::linearize::{line_rewrite_cleanups, linearize_with_cfg};
use morok_schedule::rangeify::patterns::pm_bool_devectorize;

use self::ctx::{RenderContext, ScfIfInfo, ScfLoopInfo};
use self::ops::*;
use self::types::{mlir_ptr_type, mlir_type};
use crate::{BufferArg, RenderedKernel, Renderer, Result};

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

/// Info about a reduce associated with a specific range, collected during pre-scan.
struct ReduceInfo {
    reduce_id: u64,
    reduce_op: ReduceOp,
    dtype: DType,
}

/// Pre-scan linearized nodes to build a map from Range axis_id to its associated reduces.
fn build_reduce_map(nodes: &[Arc<UOp>]) -> HashMap<usize, Vec<ReduceInfo>> {
    let mut map: HashMap<usize, Vec<ReduceInfo>> = HashMap::new();
    for node in nodes {
        if let Op::Reduce { ranges, reduce_op, .. } = node.op() {
            for range in ranges {
                if let Op::Range { axis_id, .. } = range.op() {
                    map.entry(axis_id.value()).or_default().push(ReduceInfo {
                        reduce_id: node.id,
                        reduce_op: *reduce_op,
                        dtype: node.dtype(),
                    });
                }
            }
        }
    }
    map
}

impl Renderer for MlirRenderer {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
        let kernel_name = name.unwrap_or("kernel");

        let uop = graph_rewrite_bottom_up(&pm_bool_devectorize(), uop.clone(), &mut ());

        tracing::debug!(ast_after_pm_bool_devectorize = %uop.tree(), "mlir codegen: after pm_bool_devectorize");

        let nodes = linearize_with_cfg(uop);
        let nodes = line_rewrite_cleanups(nodes);

        for (i, node) in nodes.iter().enumerate() {
            tracing::debug!(position = i, op = node.op().as_ref(), id = node.id, "mlir linearized node");
        }

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
                buffer_args.push(BufferArg { index: *id, name: format!("data{i}"), dtype: buf.dtype(), is_output });
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

        // Pre-scan: map Range axis_id -> associated reduces
        let reduce_map = build_reduce_map(&nodes);

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
        let void_type = llvm::r#type::void(&context);
        let func_type = llvm::r#type::function(void_type, &[ptr_type, ptr_type], false);
        let i64_type: Type = IntegerType::new(&context, 64).into();

        // Build function body and transfer to region
        let entry_block = Block::new(&[(ptr_type, loc), (ptr_type, loc)]);

        // Get arguments before transferring ownership
        let args_ptr = entry_block.argument(0).unwrap().into();
        let vars_ptr = entry_block.argument(1).unwrap().into();

        // Build function region, move entry block into it
        let func_region = Region::new();
        let entry_ref = func_region.append_block(entry_block);

        let mut rctx = RenderContext::new(entry_ref);

        // Load buffer pointers: GEP into %args + load
        for (i, buf) in buffers.iter().enumerate() {
            let idx = const_i64(&context, &entry_ref, i as i64, loc);
            let buf_ptr = gep_load(&context, &entry_ref, args_ptr, idx, ptr_type, ptr_type, loc);
            rctx.register(buf.id, buf_ptr);
        }

        // Load variables: GEP into %vars + load i64 + optional trunc
        for (i, var) in variables.iter().enumerate() {
            let idx = const_i64(&context, &entry_ref, i as i64, loc);
            let var_i64 = gep_load(&context, &entry_ref, vars_ptr, idx, i64_type, i64_type, loc);

            let target_type = mlir_type(&context, &var.dtype());
            let var_val = if target_type == i64_type {
                var_i64
            } else {
                entry_ref.append_operation(arith::trunci(var_i64, target_type, loc)).result(0).unwrap().into()
            };
            rctx.register(var.id, var_val);
        }

        // Load thread_id if present
        if let Some((ref thread_range, _)) = thread_info {
            let thread_idx = const_i64(&context, &entry_ref, variables.len() as i64, loc);
            let thread_i64 = gep_load(&context, &entry_ref, vars_ptr, thread_idx, i64_type, i64_type, loc);

            let range_type = mlir_type(&context, &thread_range.dtype());
            let thread_val = if range_type == i64_type {
                thread_i64
            } else {
                entry_ref.append_operation(arith::trunci(thread_i64, range_type, loc)).result(0).unwrap().into()
            };
            rctx.register(thread_range.id, thread_val);
        }

        // Pre-register constants
        for node in &nodes {
            if let Op::Const(cv) = node.op() {
                let block = rctx.current_block();
                let val = build_const(&context, &block, &cv.0, &node.dtype(), loc);
                rctx.register(node.id, val);
            }
        }

        // Process linearized nodes
        for node in &nodes {
            render_node(&context, &mut rctx, node, &thread_info, &reduce_map, loc);
        }

        // Emit return on the final current block
        rctx.current_block().append_operation(llvm::r#return(None, loc));

        // Build llvm.func operation
        let func_op = llvm::func(
            &context,
            StringAttribute::new(&context, kernel_name),
            TypeAttribute::new(func_type),
            func_region,
            &[],
            loc,
        );

        module.body().append_operation(func_op);

        tracing::debug!(mlir_module = %module.as_operation(), "mlir codegen: before passes");

        // Verify module before running passes
        if !module.as_operation().verify() {
            return Err(crate::error::Error::MlirError {
                reason: format!("module verification failed before passes\nModule:\n{}", module.as_operation()),
            });
        }

        // Run pass pipeline: lower scf → cf → llvm
        let mut module = module;
        let pass_manager = PassManager::new(&context);
        let nested = pass_manager.nested_under("llvm.func");
        nested.add_pass(melior::pass::conversion::create_scf_to_control_flow());
        nested.add_pass(melior::pass::conversion::create_vector_to_llvm());
        nested.add_pass(melior::pass::conversion::create_math_to_llvm());
        nested.add_pass(melior::pass::conversion::create_arith_to_llvm());
        nested.add_pass(melior::pass::conversion::create_index_to_llvm());
        // cf-to-llvm is a module-level pass
        pass_manager.add_pass(melior::pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(melior::pass::conversion::create_reconcile_unrealized_casts());

        if let Err(e) = pass_manager.run(&mut module) {
            return Err(crate::error::Error::MlirError {
                reason: format!("pass pipeline failed: {}\nModule before passes:\n{}", e, module.as_operation()),
            });
        }

        tracing::debug!(mlir_module = %module.as_operation(), "mlir codegen: after passes");

        // Return MLIR text (ExecutionEngine will be used at runtime instead of LLVM JIT)
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
        // MLIR LLVM dialect doesn't support vector<N x ptr> types
        // Eliminate bare PTRCAT (dead code not consumed by LOAD/STORE)
        use morok_ir::decompositions::ptrcat_decomposition_patterns;

        Some(ptrcat_decomposition_patterns())
    }
}

/// Process a single linearized node into MLIR operations.
fn render_node<'c, 'a: 'c>(
    ctx: &'c Context,
    rctx: &mut RenderContext<'c, 'a>,
    node: &Arc<UOp>,
    _thread_info: &Option<(Arc<UOp>, usize)>,
    reduce_map: &HashMap<usize, Vec<ReduceInfo>>,
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
            let block = rctx.current_block();
            let val = build_vconst(ctx, &block, values, &node.dtype(), loc);
            rctx.register(node.id, val);
        }

        Op::DefineReg { size } => {
            let block = rctx.current_block();
            let base_dtype = match node.dtype() {
                DType::Ptr { base, .. } => base.as_ref().clone(),
                other => other,
            };
            let elem_type = mlir_type(ctx, &base_dtype);
            let arr_type = llvm::r#type::array(elem_type, *size as u32);
            let one = const_i64(ctx, &block, 1, loc);
            let ptr_type = mlir_ptr_type(ctx);
            let alloca_val = block
                .append_operation(llvm::alloca(
                    ctx,
                    one,
                    ptr_type,
                    loc,
                    llvm::AllocaOptions::new().elem_type(Some(TypeAttribute::new(arr_type))),
                ))
                .result(0)
                .unwrap()
                .into();
            rctx.register(node.id, alloca_val);
        }

        Op::Index { buffer, indices, gate } => {
            let block = rctx.current_block();
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
                        ctx,
                        buf_val,
                        &[idx_val],
                        elem_type,
                        mlir_ptr_type(ctx),
                        loc,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                let result = if let Some(gate_node) = gate {
                    let gate_val = rctx.get(gate_node.id);
                    let null_ptr =
                        block.append_operation(llvm::zero(mlir_ptr_type(ctx), loc)).result(0).unwrap().into();
                    block.append_operation(arith::select(gate_val, gep, null_ptr, loc)).result(0).unwrap().into()
                } else {
                    gep
                };
                rctx.register(node.id, result);
            }
        }

        Op::PointerIndex { ptr, offset } => {
            let block = rctx.current_block();
            let ptr_val = rctx.get(ptr.id);
            let off_val = rctx.get(offset.id);
            let elem_dtype = match node.dtype() {
                DType::Ptr { ref base, .. } => base.as_ref().clone(),
                other => other,
            };
            let elem_type = mlir_type(ctx, &elem_dtype);
            let result = block
                .append_operation(llvm::get_element_ptr_dynamic(
                    ctx,
                    ptr_val,
                    &[off_val],
                    elem_type,
                    mlir_ptr_type(ctx),
                    loc,
                ))
                .result(0)
                .unwrap()
                .into();
            rctx.register(node.id, result);
        }

        Op::Load { index, .. } => {
            let block = rctx.current_block();
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
            let block = rctx.current_block();
            let idx_val = rctx.get(index.id);
            let val = rctx.get(value.id);
            block.append_operation(llvm::store(ctx, val, idx_val, loc, Default::default()));
        }

        Op::Binary(op, lhs, rhs) => {
            let block = rctx.current_block();
            let l = rctx.get(lhs.id);
            let r = rctx.get(rhs.id);
            let result = render_binary(ctx, &block, *op, l, r, &lhs.dtype(), loc);
            rctx.register(node.id, result);
        }

        Op::Unary(op, src) => {
            let block = rctx.current_block();
            let s = rctx.get(src.id);
            let result = render_unary(ctx, &block, *op, s, &src.dtype(), loc);
            rctx.register(node.id, result);
        }

        Op::Ternary(op, a, b, c) => {
            let block = rctx.current_block();
            let av = rctx.get(a.id);
            let bv = rctx.get(b.id);
            let cv = rctx.get(c.id);
            let result = render_ternary(ctx, &block, *op, av, bv, cv, &node.dtype(), loc);
            rctx.register(node.id, result);
        }

        Op::Cast { src, dtype } => {
            let block = rctx.current_block();
            let s = rctx.get(src.id);
            let result = render_cast(ctx, &block, s, &src.dtype(), dtype, loc);
            rctx.register(node.id, result);
        }

        Op::BitCast { src, dtype } => {
            let block = rctx.current_block();
            let s = rctx.get(src.id);
            let to_type = mlir_type(ctx, dtype);
            let result = block.append_operation(arith::bitcast(s, to_type, loc)).result(0).unwrap().into();
            rctx.register(node.id, result);
        }

        // =====================================================================
        // Range → scf.for setup
        // =====================================================================
        Op::Range { end, axis_id, axis_type, .. } => {
            if matches!(axis_type, AxisType::Thread) {
                return;
            }

            let range_dtype = node.dtype();
            let range_type = mlir_type(ctx, &range_dtype);
            let index_type = Type::index(ctx);
            let end_val = rctx.get(end.id);

            let parent_block = rctx.current_block();

            // Cast bounds to index type
            let lb = to_index(&parent_block, const_int(ctx, &parent_block, 0, range_type, loc), index_type, loc);
            let ub = to_index(&parent_block, end_val, index_type, loc);
            let step = to_index(&parent_block, const_int(ctx, &parent_block, 1, range_type, loc), index_type, loc);

            // Look up reduces for this loop
            let axis = axis_id.value();
            let reduces = reduce_map.get(&axis);

            let mut init_values = Vec::new();
            let mut result_types = Vec::new();
            let mut reduce_ids = Vec::new();

            if let Some(infos) = reduces {
                for info in infos {
                    let acc_type = mlir_type(ctx, &info.dtype);
                    let identity = build_reduce_identity(ctx, &parent_block, info.reduce_op, &info.dtype, loc);
                    init_values.push(identity);
                    result_types.push(acc_type);
                    reduce_ids.push(info.reduce_id);
                }
            }

            // Build body block: (index, iter_arg_types...)
            let mut block_arg_types: Vec<(Type, Location)> = vec![(index_type, loc)];
            for &rt in &result_types {
                block_arg_types.push((rt, loc));
            }
            let body_block = Block::new(&block_arg_types);

            let body_region = Region::new();
            let body_ref = body_region.append_block(body_block);

            // Register IV: cast from index back to the original loop type
            let iv_index = body_ref.argument(0).unwrap().into();
            let iv = body_ref.append_operation(arith::index_cast(iv_index, range_type, loc)).result(0).unwrap().into();
            rctx.register(node.id, iv);

            // Register iter_arg block arguments as the initial accumulator values
            // and set up yield_values from block args
            let mut yield_values = Vec::new();
            for (i, &reduce_id) in reduce_ids.iter().enumerate() {
                let arg = body_ref.argument(i + 1).unwrap().into();
                rctx.register(reduce_id, arg);
                yield_values.push(arg);
            }

            rctx.set_current_block(body_ref);
            rctx.push_scf_loop(ScfLoopInfo {
                parent_block,
                region: body_region,
                range_id: node.id,
                axis_id: axis,
                range_type,
                lb,
                ub,
                step,
                init_values,
                result_types,
                reduce_ids,
                yield_values,
            });
        }

        // =====================================================================
        // End → close scf.for loops
        // =====================================================================
        Op::End { ranges, .. } => {
            for range in ranges.iter() {
                if let Op::Range { axis_type, .. } = range.op() {
                    if matches!(axis_type, AxisType::Thread) {
                        continue;
                    }

                    // Emit scf.yield on the current body block
                    let body_block = rctx.current_block();
                    let loop_info = rctx.pop_scf_loop();
                    body_block.append_operation(scf::r#yield(&loop_info.yield_values, loc));

                    // Build scf.for on the parent block
                    let for_op = loop_info.parent_block.append_operation(
                        melior::dialect::ods::scf::r#for(
                            ctx,
                            &loop_info.result_types,
                            loop_info.lb,
                            loop_info.ub,
                            loop_info.step,
                            &loop_info.init_values,
                            loop_info.region,
                            loc,
                        )
                        .into(),
                    );

                    // Register scf.for results as final reduce values
                    for (i, &reduce_id) in loop_info.reduce_ids.iter().enumerate() {
                        let result_val = for_op.result(i).unwrap().into();
                        rctx.register(reduce_id, result_val);
                    }

                    rctx.set_current_block(loop_info.parent_block);
                }
            }
        }

        // =====================================================================
        // Reduce → pure SSA accumulation (no alloca/load/store)
        // =====================================================================
        Op::Reduce { src, ranges, reduce_op } => {
            if ranges.is_empty() {
                let s = rctx.get(src.id);
                rctx.register(node.id, s);
            } else {
                let block = rctx.current_block();
                let src_val = rctx.get(src.id);
                // Current accumulator is the value registered for this reduce ID
                // (initially the iter_arg block arg, updated by previous Reduce ops)
                let acc = rctx.get(node.id);
                let acc_new = render_reduce_accumulate(ctx, &block, *reduce_op, src_val, acc, &node.dtype(), loc);
                rctx.register(node.id, acc_new);
                rctx.update_reduce_yield(node.id, acc_new);
            }
        }

        Op::Gep { vector, indices } => {
            let block = rctx.current_block();
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
            let block = rctx.current_block();
            let elem_vals: Vec<_> = elements.iter().map(|e| rctx.get(e.id)).collect();
            let vec_type = mlir_type(ctx, &node.dtype());
            let scalar_type = mlir_type(ctx, &node.dtype().scalar_dtype());
            let result = render_vectorize(ctx, &block, &elem_vals, vec_type, scalar_type, loc);
            rctx.register(node.id, result);
        }

        Op::Cat { sources } => {
            let block = rctx.current_block();
            let vec_type = mlir_type(ctx, &node.dtype());

            let mut all_scalars: Vec<melior::ir::Value> = Vec::new();
            for src in sources {
                let src_val = rctx.get(src.id);
                let src_count = src.dtype().vcount();
                if src_count == 1 {
                    all_scalars.push(src_val);
                } else {
                    let src_scalar_type = mlir_type(ctx, &src.dtype().scalar_dtype());
                    for i in 0..src_count {
                        all_scalars.push(render_extractelement(ctx, &block, src_val, i, src_scalar_type, loc));
                    }
                }
            }
            let result = block
                .append_operation(melior::dialect::ods::vector::from_elements(ctx, vec_type, &all_scalars, loc).into())
                .result(0)
                .unwrap()
                .into();
            rctx.register(node.id, result);
        }

        Op::PtrCat { sources } => {
            let block = rctx.current_block();

            // Create vector of pointers: <N x ptr>
            let count = sources.len() as u64;
            let ptr_type = mlir_ptr_type(ctx);
            let vec_type = Type::vector(&[count], ptr_type);

            let mut current = block.append_operation(llvm::undef(vec_type, loc)).result(0).unwrap().into();
            for (i, src) in sources.iter().enumerate() {
                let src_val = rctx.get(src.id);
                let idx = const_i32(ctx, &block, i as i64, loc);
                current = block
                    .append_operation(insert_element(current, src_val, idx, vec_type, loc))
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

        // =====================================================================
        // If → scf.if setup
        // =====================================================================
        Op::If { condition, .. } => {
            let parent_block = rctx.current_block();
            let cond_val = rctx.get(condition.id);

            let then_region = Region::new();
            let then_block = Block::new(&[]);
            let then_ref = then_region.append_block(then_block);

            rctx.push_scf_if(node.id, ScfIfInfo { parent_block, condition: cond_val, then_region });
            rctx.set_current_block(then_ref);
        }

        // =====================================================================
        // EndIf → close scf.if
        // =====================================================================
        Op::EndIf { if_op } => {
            let then_block = rctx.current_block();
            then_block.append_operation(scf::r#yield(&[], loc));

            let if_info = rctx.pop_scf_if(if_op.id);

            // Create empty else region with scf.yield
            let else_region = Region::new();
            let else_block = Block::new(&[]);
            let else_ref = else_region.append_block(else_block);
            else_ref.append_operation(scf::r#yield(&[], loc));

            if_info.parent_block.append_operation(scf::r#if(
                if_info.condition,
                &[],
                if_info.then_region,
                else_region,
                loc,
            ));

            rctx.set_current_block(if_info.parent_block);
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

/// GEP into an array base pointer at `idx`, then load a value of `load_type`.
fn gep_load<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    base: melior::ir::Value<'c, 'c>,
    idx: melior::ir::Value<'c, 'c>,
    elem_type: Type<'c>,
    load_type: Type<'c>,
    loc: Location<'c>,
) -> melior::ir::Value<'c, 'c> {
    let ptr = block
        .append_operation(llvm::get_element_ptr_dynamic(ctx, base, &[idx], elem_type, mlir_ptr_type(ctx), loc))
        .result(0)
        .unwrap()
        .into();
    block.append_operation(llvm::load(ctx, ptr, load_type, loc, Default::default())).result(0).unwrap().into()
}

/// Cast a value to `index` type via `arith::index_cast`.
fn to_index<'c>(
    block: &Block<'c>,
    val: melior::ir::Value<'c, 'c>,
    index_type: Type<'c>,
    loc: Location<'c>,
) -> melior::ir::Value<'c, 'c> {
    block.append_operation(arith::index_cast(val, index_type, loc)).result(0).unwrap().into()
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

/// Public render function for the MLIR backend.
pub fn render(uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
    let renderer = MlirRenderer::new();
    crate::Renderer::render(&renderer, uop, name)
}
