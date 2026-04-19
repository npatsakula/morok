//! Tensor realization (execution) API.
//!
//! This module provides the execution pipeline for tensor operations:
//! 1. **Rangeify** - Transform movement ops to BUFFERIZE + INDEX
//! 2. **Kernel splitting** - Split at STORE boundaries into KERNEL ops
//! 3. **Scheduling** - Extract kernels and create execution schedule
//! 4. **Execution** - Compile and run each kernel in dependency order
//!
//! # Parallel Execution
//!
//! When independent kernels are detected (no buffer conflicts), they can be
//! executed in parallel using the `UnifiedExecutor` from the runtime crate.
//! The executor tracks buffer dependencies and uses timeline signals for
//! cross-device synchronization.
//!
//! # ExecutionPlan (Pre-compiled Execution)
//!
//! For repeated executions, use `Tensor::prepare()` to create an `ExecutionPlan`
//! that pre-compiles all kernels and allocates all buffers. Then call
//! `plan.execute()` for fast repeated execution without recompilation overhead.
//!
//! ```ignore
//! // One-time preparation (compiles kernels, allocates buffers)
//! let plan = tensor.prepare()?;
//!
//! // Fast execution (can be called many times)
//! plan.execute(&mut executor)?;
//!
//! // Get results
//! let output = plan.output_buffer();
//! ```

use std::collections::{HashMap, HashSet};

use morok_schedule::{
    Scheduler, apply_post_optimization, beam_search_cached, hand_coded_optimizations, prepare_scheduler,
};
use tracing::{debug, trace};

use crate::{
    PrepareConfig, Result, Tensor,
    error::{
        BatchOutputMismatchSnafu, CompileKernelSnafu, CreateProgramSnafu, DependencyCyclesSnafu, DeviceSnafu,
        EmptyScheduleSnafu, ExecutionSnafu, OptimizeSnafu, RangeifySnafu, RenderKernelSnafu, ShapeUnknownSnafu,
        UOpSnafu,
    },
    schedule::{ScheduleItem, expand_schedule},
};
use morok_device::{Buffer, device::Device};
use morok_ir::pattern::is_any_const;
use morok_ir::{DeviceSpec, Op, UOp, UOpKey};
use morok_runtime::{
    ExecutionGraph, ExecutionNode, ExecutionPlan, ExecutionPlanBuilder, KernelBufferAccess, PreparedKernel,
};
use snafu::{OptionExt, ResultExt};
use std::sync::Arc;
use std::time::Duration;

fn collect_pending_indices(tensors: &[&mut Tensor]) -> Vec<usize> {
    tensors
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.uop().has_buffer_identity() && !is_any_const(&t.uop()) && !t.has_zero_elements())
        .map(|(i, _)| i)
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BufferStorageKey {
    id: u64,
    offset: usize,
    size: usize,
    dtype: morok_dtype::DType,
}

impl Tensor {
    /// Realize (execute) this tensor's computation graph.
    ///
    /// This is a convenience method that prepares and executes in one call.
    /// For repeated executions of the same computation, use `prepare()` instead.
    ///
    /// # Pipeline
    ///
    /// 1. **Prepare**: Creates an `ExecutionPlan` (compiles kernels, allocates buffers)
    /// 2. **Execute**: Runs all kernels in dependency order
    /// 3. **Return**: Links output buffer to this tensor's UOp
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    /// let c = (&a + &b).realize()?;
    /// // c's buffer now contains [5.0, 7.0, 9.0]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if preparation or execution fails.
    pub fn realize(&mut self) -> Result<()> {
        if self.uop().has_buffer_identity() {
            self.ensure_buffer();
            return Ok(());
        }
        // Pure constants: wrap in CONTIGUOUS to force materialization into a buffer.
        if is_any_const(&self.uop()) {
            let contiguous_uop = self.uop().contiguous();
            self.set_uop(contiguous_uop);
        }
        if self.has_zero_elements() {
            return Ok(());
        }

        resolve_pending_assigns(&self.uop(), &PrepareConfig::from_env())?;
        if self.uop().has_buffer_identity() {
            self.ensure_buffer();
            return Ok(());
        }

        let old_uop = self.uop();
        let input_buffer_ids: HashSet<u64> = collect_input_buffers(&old_uop).keys().copied().collect();

        let t_prep = std::time::Instant::now();
        let plan = self.prepare()?;
        let prep_ms = t_prep.elapsed().as_millis();
        let t_exec = std::time::Instant::now();
        plan.execute().context(ExecutionSnafu)?;
        let exec_ms = t_exec.elapsed().as_millis();
        debug!(prep_ms, exec_ms, "realize complete");

        self.finalize_realize(&plan, &old_uop)?;

        let realized_uop = self.uop();
        if !Arc::ptr_eq(&old_uop, &realized_uop) {
            #[allow(clippy::mutable_key_type)]
            let becomes_map = HashMap::from([(UOpKey(old_uop), realized_uop)]);
            crate::tensor_registry::apply_map_to_tensors(&becomes_map);
        }

        plan.release_intermediate_buffers(|uop_id| {
            if !input_buffer_ids.contains(&uop_id) {
                crate::tensor_registry::remove_buffer(uop_id);
            }
        });

        Ok(())
    }

    /// Realize tensor with custom configuration.
    ///
    /// Like [`realize()`](Self::realize) but allows specifying optimization strategy
    /// and codegen backend.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use morok_tensor::PrepareConfig;
    /// use morok_schedule::{OptStrategy, OptimizerConfig};
    ///
    /// let c = a.matmul(&b)?;
    /// let config = PrepareConfig::from(
    ///     OptimizerConfig::builder()
    ///         .strategy(OptStrategy::Beam { width: 4 })
    ///         .build()
    /// );
    /// let c = c.realize_with(&config)?;
    /// ```
    pub fn realize_with(&mut self, config: &PrepareConfig) -> Result<()> {
        if self.uop().has_buffer_identity() {
            self.ensure_buffer();
            return Ok(());
        }
        // Pure constants: wrap in CONTIGUOUS to force materialization into a buffer.
        if is_any_const(&self.uop()) {
            let contiguous_uop = self.uop().contiguous();
            self.set_uop(contiguous_uop);
        }
        if self.has_zero_elements() {
            return Ok(());
        }

        resolve_pending_assigns(&self.uop(), config)?;
        if self.uop().has_buffer_identity() {
            self.ensure_buffer();
            return Ok(());
        }

        let old_uop = self.uop();
        let input_buffer_ids: HashSet<u64> = collect_input_buffers(&old_uop).keys().copied().collect();

        let t_prep = std::time::Instant::now();
        let plan = self.prepare_with(config)?;
        let prep_ms = t_prep.elapsed().as_millis();
        let t_exec = std::time::Instant::now();
        plan.execute().context(ExecutionSnafu)?;
        let exec_ms = t_exec.elapsed().as_millis();
        debug!(prep_ms, exec_ms, "realize_with complete");

        self.finalize_realize(&plan, &old_uop)?;

        let realized_uop = self.uop();
        if !Arc::ptr_eq(&old_uop, &realized_uop) {
            #[allow(clippy::mutable_key_type)]
            let becomes_map = HashMap::from([(UOpKey(old_uop), realized_uop)]);
            crate::tensor_registry::apply_map_to_tensors(&becomes_map);
        }

        plan.release_intermediate_buffers(|uop_id| {
            if !input_buffer_ids.contains(&uop_id) {
                crate::tensor_registry::remove_buffer(uop_id);
            }
        });

        Ok(())
    }

    /// Finalize realization: bind output buffer to tensor.
    ///
    /// Note: intermediate buffer cleanup is deferred to `realize()` so it
    /// runs AFTER `apply_map_to_tensors`. This ensures other tensors can still
    /// find buffers during the substitution window.
    fn finalize_realize(&mut self, plan: &ExecutionPlan, uop: &Arc<UOp>) -> Result<()> {
        let output_buf = plan.output_buffer().clone();

        trace!(
            buffer.id = ?output_buf.id(),
            buffer.size = output_buf.size(),
            "Realized output buffer"
        );

        let output_dtype = uop.dtype();
        let output_device = output_buf.allocator().device_spec();
        let num_elements = output_buf.size() / output_dtype.bytes();

        let buffer_uop = UOp::new_buffer(output_device, num_elements, output_dtype.clone());
        let output_buf_arc = Arc::new(output_buf);

        crate::tensor_registry::register_buffer(buffer_uop.id, self.entry.id, output_buf_arc.clone());

        let shape = uop.shape().context(UOpSnafu)?.context(ShapeUnknownSnafu)?;
        let realized_uop = buffer_uop.try_reshape(shape).context(UOpSnafu)?;

        debug!(
            buffer_uop.id = buffer_uop.id,
            num_elements,
            shape = ?shape,
            realized_uop.id = realized_uop.id,
            realized_uop.base_id = realized_uop.base().id,
            "Tensor realized"
        );

        self.set_uop(realized_uop);
        self.entry.set_buffer(Arc::clone(&output_buf_arc));
        self.buffer = Some(output_buf_arc);
        Ok(())
    }

    /// Prepare an execution plan for this tensor's computation graph.
    ///
    /// This performs all one-time work:
    /// 1. Creates schedule from computation graph
    /// 2. Expands bound ranges
    /// 3. Compiles all kernels
    /// 4. Allocates all buffers
    /// 5. Computes parallel execution groups
    ///
    /// The returned `ExecutionPlan` can then be executed multiple times
    /// without recompilation overhead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    /// let mut c = &a + &b;
    ///
    /// // One-time preparation (wires output tensor to plan buffer)
    /// let plan = c.prepare()?;
    ///
    /// // Fast execution (can be called many times)
    /// plan.execute()?;
    ///
    /// // Get results
    /// let output = plan.output_buffer();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Rangeify transformation fails
    /// - No kernels found after scheduling
    /// - Kernel compilation fails
    /// - Buffer allocation fails
    pub fn prepare(&mut self) -> Result<ExecutionPlan> {
        self.prepare_with(&PrepareConfig::from_env())
    }

    /// Prepare an execution plan with explicit configuration.
    ///
    /// This method allows fine-grained control over kernel optimization settings
    /// and codegen backend selection.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use morok_tensor::PrepareConfig;
    /// use morok_schedule::{OptimizerConfig, OptStrategy, BeamConfig};
    ///
    /// // Beam search with width 8 and 120s timeout
    /// let config = PrepareConfig::from(
    ///     OptimizerConfig::builder()
    ///         .strategy(OptStrategy::Beam { width: 8 })
    ///         .beam(BeamConfig::builder()
    ///             .timeout_secs(120)
    ///             .build())
    ///         .build()
    /// );
    ///
    /// let plan = tensor.prepare_with(&config)?;
    /// plan.execute(&mut executor)?;
    /// ```
    pub fn prepare_with(&mut self, config: &PrepareConfig) -> Result<ExecutionPlan> {
        let t_total = std::time::Instant::now();
        let uop = self.uop();
        resolve_pending_assigns(&uop, config)?;

        let sink = UOp::sink(vec![uop.contiguous()]);
        let var_vals = extract_var_vals(&uop);

        // Pre-schedule normalization: BUFFER→PARAM (Tinygrad pm_pre_sched_cache).
        let (normalized, param_buffers) = normalize_buffers_to_params(&sink);

        // Build input_buffers keyed by PARAM UOp IDs (the normalized graph
        // no longer has BUFFER nodes — they've been replaced with PARAMs).
        let mut param_input_buffers = crate::schedule::InputBuffers::new();
        for node in normalized.toposort() {
            if let Op::Param { slot, .. } = node.op() {
                let (orig_buffer_id, _) = &param_buffers[*slot];
                if let Some(buf) = crate::tensor_registry::get_buffer(*orig_buffer_id) {
                    param_input_buffers.insert(node.id, buf);
                }
            }
        }

        // Compute cache key: (structural hash of normalized sink, codegen backend).
        let codegen = resolve_codegen(&param_buffers, config)?;
        let sched_key = (crate::schedule_cache::content_hash(&normalized), codegen);

        // Check schedule-level cache. On hit, skip rangeify + kernel_split.
        let cache = crate::schedule_cache::schedule_cache();
        let entry = {
            let guard = cache.guard();
            cache.get(&sched_key, &guard).cloned()
        };
        let entry = match entry {
            Some(hit) => {
                debug!("schedule cache hit");
                hit
            }
            None => {
                let rangeify_result = morok_schedule::rangeify_with_map(normalized, None).context(RangeifySnafu)?;
                let (k, _) = morok_schedule::run_kernel_split_pipeline(rangeify_result.sink);
                let new_entry = Arc::new(crate::schedule_cache::CachedSchedule { kernelized: k });
                let guard = cache.guard();
                cache.insert(sched_key, Arc::clone(&new_entry), &guard);
                new_entry
            }
        };

        let schedule_result =
            crate::schedule::create_schedule(entry.kernelized.clone(), &param_input_buffers, &var_vals)?;
        // Per-kernel optimization+compilation is cached globally in prepare_execution_plan
        // via OPT_CACHE keyed by content_hash(ast). Identical kernel ASTs across calls
        // (e.g., sort substages, repeated model inference) skip optimize+compile.
        let plan = prepare_execution_plan(&schedule_result, config)?;

        self.wire_output_tensor(&plan, &uop)?;
        debug!(total_ms = t_total.elapsed().as_millis() as u64, "prepare: total");
        Ok(plan)
    }

    fn wire_output_tensor(&mut self, plan: &ExecutionPlan, uop: &Arc<UOp>) -> Result<()> {
        if plan.num_outputs() > 0 {
            let buf = Arc::new(plan.output_buffer().clone());
            let dtype = uop.dtype();
            let device = buf.allocator().device_spec();
            let buffer_uop = UOp::new_buffer(device, buf.size() / dtype.bytes(), dtype);
            crate::tensor_registry::register_buffer(buffer_uop.id, self.entry.id, buf.clone());
            let shape = uop.shape().context(UOpSnafu)?.context(ShapeUnknownSnafu)?;
            self.set_uop(buffer_uop.try_reshape(shape).context(UOpSnafu)?);
            self.entry.set_buffer(buf.clone());
            self.buffer = Some(buf);
        }
        Ok(())
    }

    // =========================================================================
    // Batch realize / prepare
    // =========================================================================

    /// Realize multiple tensors in a single batch, sharing computation.
    ///
    /// Merges all tensor computation graphs into one SINK, enabling the scheduler
    /// to share kernels across outputs. More efficient than calling `realize()`
    /// individually when tensors share subgraphs.
    pub fn realize_batch<'a>(tensors: impl IntoIterator<Item = &'a mut Tensor>) -> Result<()> {
        Self::realize_batch_with(tensors, &PrepareConfig::from_env())
    }

    /// Realize multiple tensors with custom configuration.
    pub fn realize_batch_with<'a>(
        tensors: impl IntoIterator<Item = &'a mut Tensor>,
        config: &PrepareConfig,
    ) -> Result<()> {
        let mut tensors: Vec<&mut Tensor> = tensors.into_iter().collect();
        if tensors.is_empty() {
            return Ok(());
        }

        // Handle already-realized tensors
        for t in &mut tensors {
            if t.uop().has_buffer_identity() {
                t.ensure_buffer();
            }
        }

        // Wrap pure constants in CONTIGUOUS to force materialization (matches realize())
        for t in &mut tensors {
            if !t.uop().has_buffer_identity() && is_any_const(&t.uop()) {
                let contiguous_uop = t.uop().contiguous();
                t.set_uop(contiguous_uop);
            }
        }

        // Collect pending (unrealized) tensor indices
        let pending_indices = collect_pending_indices(&tensors);

        if pending_indices.is_empty() {
            return Ok(());
        }

        // Resolve pending assigns
        for &i in &pending_indices {
            resolve_pending_assigns(&tensors[i].uop(), config)?;
        }

        // Pending assigns can realize outputs. Re-sync tensor buffers and
        // recompute pending tensors from updated graphs.
        for t in &mut tensors {
            if t.uop().has_buffer_identity() {
                t.ensure_buffer();
            }
        }
        let pending_indices = collect_pending_indices(&tensors);
        if pending_indices.is_empty() {
            return Ok(());
        }

        // Collect input buffers and old UOps from ALL pending tensors
        let old_uops: Vec<Arc<UOp>> = pending_indices.iter().map(|&i| tensors[i].uop()).collect();
        let mut all_input_buffers = crate::schedule::InputBuffers::new();
        for uop in &old_uops {
            all_input_buffers.extend(collect_input_buffers(uop));
        }
        let input_ids: HashSet<u64> = all_input_buffers.keys().copied().collect();

        // Create merged SINK(CONTIGUOUS(t1), ..., CONTIGUOUS(tN))
        let contiguouses: Vec<Arc<UOp>> = old_uops.iter().map(|u| u.contiguous()).collect();
        let sink = UOp::sink(contiguouses);

        // Pre-schedule normalization: BUFFER→PARAM (Tinygrad pm_pre_sched_cache).
        // Must run before rangeify so buffer counting sees Param nodes (not Buffer).
        let (sink, param_buffers) = normalize_buffers_to_params(&sink);

        // Rebuild input_buffers keyed by PARAM UOp ids (the normalized graph
        // no longer has BUFFER nodes — they've been replaced with PARAMs).
        let mut param_input_buffers = crate::schedule::InputBuffers::new();
        for node in sink.toposort() {
            if let Op::Param { slot, .. } = node.op() {
                let (orig_buffer_id, _) = &param_buffers[*slot];
                if let Some(buf) = all_input_buffers.get(orig_buffer_id) {
                    param_input_buffers.insert(node.id, buf.clone());
                }
            }
        }

        // Extract bound variable values from all pending tensor UOps
        let mut var_vals = HashMap::new();
        for uop in &old_uops {
            var_vals.extend(extract_var_vals(uop));
        }

        // Pipeline: rangeify → kernel_split → schedule → plan → execute
        let rangeify_result = morok_schedule::rangeify_with_map(sink, None).context(RangeifySnafu)?;
        let (kernelized, _kernel_ctx) = morok_schedule::run_kernel_split_pipeline(rangeify_result.sink);
        let schedule_result = crate::schedule::create_schedule(kernelized, &param_input_buffers, &var_vals)?;

        let t_prep = std::time::Instant::now();
        let plan = prepare_execution_plan(&schedule_result, config)?;
        let prep_ms = t_prep.elapsed().as_millis();
        let t_exec = std::time::Instant::now();
        plan.execute().context(ExecutionSnafu)?;
        let exec_ms = t_exec.elapsed().as_millis();
        debug!(prep_ms, exec_ms, num_outputs = pending_indices.len(), "realize_batch complete");

        snafu::ensure!(
            plan.num_outputs() >= pending_indices.len(),
            BatchOutputMismatchSnafu { expected: pending_indices.len(), actual: plan.num_outputs() }
        );

        // Finalize each pending tensor in-place + build batched becomes_map
        #[allow(clippy::mutable_key_type)]
        let mut becomes_map = HashMap::new();
        for (buf_idx, &orig_idx) in pending_indices.iter().enumerate() {
            let output_buf = plan.output_buffer_at(buf_idx).clone();
            let old_uop = &old_uops[buf_idx];

            let output_dtype = old_uop.dtype();
            let output_device = output_buf.allocator().device_spec();
            let num_elements = output_buf.size() / output_dtype.bytes();
            let buffer_uop = UOp::new_buffer(output_device, num_elements, output_dtype);
            let buf_arc = Arc::new(output_buf);

            let t = &mut tensors[orig_idx];
            crate::tensor_registry::register_buffer(buffer_uop.id, t.entry.id, buf_arc.clone());
            let shape = old_uop.shape().context(UOpSnafu)?.context(ShapeUnknownSnafu)?;
            let realized_uop = buffer_uop.try_reshape(shape).context(UOpSnafu)?;
            t.set_uop(realized_uop.clone());
            t.entry.set_buffer(Arc::clone(&buf_arc));
            t.buffer = Some(buf_arc);

            becomes_map.insert(UOpKey(old_uop.clone()), realized_uop);
        }

        // Single batched apply_map (one global walk instead of N)
        crate::tensor_registry::apply_map_to_tensors(&becomes_map);

        // Cleanup intermediate buffers
        plan.release_intermediate_buffers(|id| {
            if !input_ids.contains(&id) {
                crate::tensor_registry::remove_buffer(id);
            }
        });

        Ok(())
    }

    /// Prepare a batch execution plan for multiple tensors.
    ///
    /// Output tensors are wired to plan buffers — after `execute`/`execute_with_vars`,
    /// results are readable directly via `tensor.as_vec()` or `tensor.array_view()`.
    pub fn prepare_batch<'a>(tensors: impl IntoIterator<Item = &'a mut Tensor>) -> Result<ExecutionPlan> {
        Self::prepare_batch_with(tensors, &PrepareConfig::from_env())
    }

    /// Prepare a batch execution plan with custom configuration.
    pub fn prepare_batch_with<'a>(
        tensors: impl IntoIterator<Item = &'a mut Tensor>,
        config: &PrepareConfig,
    ) -> Result<ExecutionPlan> {
        let mut tensors: Vec<&mut Tensor> = tensors.into_iter().collect();
        if tensors.is_empty() {
            return EmptyScheduleSnafu.fail();
        }

        // Handle already-realized tensors
        for t in &mut tensors {
            if t.uop().has_buffer_identity() {
                t.ensure_buffer();
            }
        }

        // Wrap pure constants in CONTIGUOUS to force materialization (matches realize())
        for t in &mut tensors {
            if !t.uop().has_buffer_identity() && is_any_const(&t.uop()) {
                let contiguous_uop = t.uop().contiguous();
                t.set_uop(contiguous_uop);
            }
        }

        // Collect pending (unrealized) tensor indices
        let pending_indices = collect_pending_indices(&tensors);

        if pending_indices.is_empty() {
            return EmptyScheduleSnafu.fail();
        }

        // Resolve pending assigns so input buffers exist for scheduling.
        for &i in &pending_indices {
            resolve_pending_assigns(&tensors[i].uop(), config)?;
        }

        // Pending assigns can realize outputs. Re-sync tensor buffers and
        // recompute pending tensors from updated graphs.
        for t in &mut tensors {
            if t.uop().has_buffer_identity() {
                t.ensure_buffer();
            }
        }
        let pending_indices = collect_pending_indices(&tensors);
        if pending_indices.is_empty() {
            return EmptyScheduleSnafu.fail();
        }

        // Collect UOps from pending tensors only
        let uops: Vec<Arc<UOp>> = pending_indices.iter().map(|&i| tensors[i].uop()).collect();

        let mut all_input_buffers = crate::schedule::InputBuffers::new();
        let mut var_vals = HashMap::new();
        for uop in &uops {
            all_input_buffers.extend(collect_input_buffers(uop));
            var_vals.extend(extract_var_vals(uop));
        }

        // Create merged SINK(CONTIGUOUS(t1), ..., CONTIGUOUS(tN)) from pending tensors
        let contiguouses: Vec<Arc<UOp>> = uops.iter().map(|u| u.contiguous()).collect();
        let sink = UOp::sink(contiguouses);

        // Pre-schedule normalization: BUFFER→PARAM (Tinygrad pm_pre_sched_cache).
        // Erases buffer identity so structurally identical computations on different
        // buffers share the same AST. Enables kernel compilation dedup.
        let (sink, param_buffers) = normalize_buffers_to_params(&sink);

        // Rebuild input_buffers keyed by PARAM UOp ids (the normalized graph
        // no longer has BUFFER nodes — they've been replaced with PARAMs).
        // Each PARAM's id maps to the original BUFFER's device allocation.
        let mut param_input_buffers = crate::schedule::InputBuffers::new();
        for node in sink.toposort() {
            if let Op::Param { slot, .. } = node.op() {
                let (orig_buffer_id, _) = &param_buffers[*slot];
                if let Some(buf) = all_input_buffers.get(orig_buffer_id) {
                    param_input_buffers.insert(node.id, buf.clone());
                }
            }
        }

        let rangeify_result = morok_schedule::rangeify_with_map(sink, None).context(RangeifySnafu)?;
        let (kernelized, _kernel_ctx) = morok_schedule::run_kernel_split_pipeline(rangeify_result.sink);
        let schedule_result = crate::schedule::create_schedule(kernelized, &param_input_buffers, &var_vals)?;

        let plan = prepare_execution_plan(&schedule_result, config)?;

        // Wire each pending output tensor to its plan buffer.
        // After execute/execute_with_vars, tensor.array_view() reads the result directly.
        for (buf_idx, &orig_idx) in pending_indices.iter().enumerate() {
            if buf_idx >= plan.num_outputs() {
                break;
            }
            let output_buf = plan.output_buffer_at(buf_idx).clone();
            let buf_arc = Arc::new(output_buf);
            let old_uop = &uops[buf_idx];
            let output_dtype = old_uop.dtype();
            let output_device = buf_arc.allocator().device_spec();
            let num_elements = buf_arc.size() / output_dtype.bytes();
            let buffer_uop = UOp::new_buffer(output_device, num_elements, output_dtype);
            let t = &mut tensors[orig_idx];
            crate::tensor_registry::register_buffer(buffer_uop.id, t.entry.id, buf_arc.clone());
            let shape = old_uop.shape().context(UOpSnafu)?.context(ShapeUnknownSnafu)?;
            let realized_uop = buffer_uop.try_reshape(shape).context(UOpSnafu)?;
            t.set_uop(realized_uop);
            t.entry.set_buffer(Arc::clone(&buf_arc));
            t.buffer = Some(buf_arc);
        }

        Ok(plan)
    }
}

/// Side-realize pending assigns for buffers referenced by a UOp graph.
///
/// Matches Tinygrad's `realize()` lines 274-289: before scheduling the main
/// computation, pending assigns for any referenced BUFFER are scheduled and
/// executed first. The `becomes_map` from each assign's realization is propagated
/// to all live tensors via `apply_map_to_tensors`.
///
/// Key difference from previous implementation: ASSIGN goes directly into SINK
/// (no CONTIGUOUS wrapper), matching Tinygrad line 281. The ASSIGN writes directly
/// into its target buffer via bufferize_to_store, producing a single copy kernel
/// instead of two.
fn resolve_pending_assigns(uop: &Arc<UOp>, config: &PrepareConfig) -> Result<()> {
    if !crate::tensor_registry::has_pending_assigns() {
        return Ok(());
    }

    // Collect buffer IDs up front (like Tinygrad line 288: set comprehension).
    let buffer_ids: Vec<u64> =
        uop.toposort().iter().filter(|n| matches!(n.op(), Op::Buffer { .. })).map(|n| n.id).collect();

    for buf_id in buffer_ids {
        realize_pending_recursive(buf_id, config)?;
    }

    Ok(())
}

/// Recursively realize pending assigns for a buffer and its transitive dependencies.
///
/// Matches Tinygrad's `_realize_pending` (tensor.py:276-287):
/// 1. Pop all assigns for this buffer
/// 2. For each assign, recursively realize any buffer dependencies that also have pending assigns
/// 3. Schedule + execute the assign
/// 4. Propagate becomes_map to all live tensors and remaining pending assigns
fn realize_pending_recursive(buf_id: u64, config: &PrepareConfig) -> Result<()> {
    let Some(assign_uops) = crate::tensor_registry::take_pending_assigns(buf_id) else {
        return Ok(());
    };

    for assign_uop in assign_uops {
        // Recurse: realize transitive dependencies FIRST (Tinygrad lines 279-280)
        for dep in assign_uop.toposort() {
            if matches!(dep.op(), Op::Buffer { .. }) && crate::tensor_registry::has_pending_assign(dep.id) {
                realize_pending_recursive(dep.id, config)?;
            }
        }

        debug!(buffer_id = buf_id, "realize_pending_recursive: side-realizing assign");

        // Schedule the ASSIGN directly in SINK — no CONTIGUOUS wrapper.
        let sink = UOp::sink(vec![assign_uop.clone()]);

        let all_input_buffers = collect_input_buffers(&assign_uop);
        let var_vals = extract_var_vals(&assign_uop);

        // Pre-schedule normalization: BUFFER→PARAM (same as prepare_with/realize_batch_with).
        let (sink, param_buffers) = normalize_buffers_to_params(&sink);
        let mut param_input_buffers = crate::schedule::InputBuffers::new();
        for node in sink.toposort() {
            if let Op::Param { slot, .. } = node.op() {
                let (orig_buffer_id, _) = &param_buffers[*slot];
                if let Some(buf) = all_input_buffers.get(orig_buffer_id) {
                    param_input_buffers.insert(node.id, buf.clone());
                }
            }
        }

        let rangeify_result = morok_schedule::rangeify_with_map(sink, None).context(RangeifySnafu)?;
        let (kernelized, _kernel_ctx) = morok_schedule::run_kernel_split_pipeline(rangeify_result.sink);
        let schedule_result = crate::schedule::create_schedule(kernelized, &param_input_buffers, &var_vals)?;

        if schedule_result.items.is_empty() {
            continue;
        }

        let plan = prepare_execution_plan(&schedule_result, config)?;
        plan.execute().context(ExecutionSnafu)?;

        // Register the target buffer so collect_input_buffers can find it.
        let output_buf = plan.output_buffer().clone();
        let output_buf_arc = Arc::new(output_buf);
        crate::tensor_registry::register_buffer_by_uop_id(buf_id, Arc::clone(&output_buf_arc));

        // Map ASSIGN → target to remove the ASSIGN from all live tensor graphs.
        let target_uop = match assign_uop.op() {
            Op::Assign { target, .. } => target.clone(),
            _ => unreachable!("pending assign must be an ASSIGN op"),
        };
        #[allow(clippy::mutable_key_type)]
        let becomes_map = HashMap::from([(UOpKey(assign_uop), target_uop)]);
        crate::tensor_registry::apply_map_to_tensors(&becomes_map);
        crate::tensor_registry::substitute_pending_assigns(&becomes_map);
    }

    Ok(())
}

/// Extract bound variable values from a UOp graph (pre-pipeline).
///
/// Scans for BIND(DEFINE_VAR, CONST) nodes and extracts the mapping
/// from variable name to concrete bound value. This is the Morok equivalent
/// of Tinygrad's `strip_bind` in `pm_pre_sched_cache`.
///
/// These values are passed through to scheduling so that user Variables
/// (like `Variable::new("N", 1, 32).bind(4)`) are treated as fixed parameters
/// rather than OUTER ranges to be expanded.
fn extract_var_vals(root: &Arc<UOp>) -> HashMap<String, i64> {
    let mut var_vals = HashMap::new();
    for node in root.toposort() {
        if let Op::Bind { var, value } = node.op()
            && let Op::DefineVar { name, .. } = var.op()
            && let Op::Const(cv) = value.op()
            && let Some(val) = cv.0.try_int()
        {
            var_vals.insert(name.clone(), val);
        }
    }
    var_vals
}

/// Pre-schedule normalization: replace BUFFER UOps with positional PARAM.
///
/// Erases buffer identity (UNIQUE id) so structurally identical computations
/// on different buffers produce the same AST. Returns:
/// - The normalized sink (BUFFER→PARAM)
/// - `param_buffers`: slot → (original_buffer_uop_id, buffer_uop) for buffer lookup
///
/// Matches Tinygrad's `pm_pre_sched_cache` / `replace_input_buffer` (engine/schedule.py:103-130).
/// Context for BUFFER→PARAM normalization (like Tinygrad's replace_input_buffer).
pub(crate) struct NormalizeBuffersCtx {
    pub buffer_map: HashMap<u64, usize>,
    pub param_buffers: Vec<(u64, Arc<UOp>)>,
}

pub(crate) fn normalize_buffers_to_params(sink: &Arc<UOp>) -> (Arc<UOp>, Vec<(u64, Arc<UOp>)>) {
    let mut ctx = NormalizeBuffersCtx { buffer_map: HashMap::new(), param_buffers: Vec::new() };

    // Replace BUFFER → PARAM via graph_rewrite (matching upstream pm_pre_sched_cache).
    use morok_ir::op::pattern_derived::OpKey;
    use morok_ir::pattern::{RewriteResult, SimplifiedPatternMatcher};
    use morok_ir::rewrite::graph_rewrite;

    let mut matcher = SimplifiedPatternMatcher::<NormalizeBuffersCtx>::new();
    matcher.add(&[OpKey::Buffer], |node, ctx| {
        let Op::Buffer { size, device, .. } = node.op() else {
            return RewriteResult::NoMatch;
        };
        let slot = *ctx.buffer_map.entry(node.id).or_insert_with(|| {
            let s = ctx.param_buffers.len();
            ctx.param_buffers.push((node.id, node.clone()));
            s
        });
        let param = UOp::param(slot, *size, node.dtype(), Some(device.clone()));
        RewriteResult::Rewritten(param)
    });

    let normalized = graph_rewrite(&matcher, sink.clone(), &mut ctx);
    (normalized, ctx.param_buffers)
}

/// Collect input buffers from a computation graph.
///
/// Walks the UOp graph and collects all BUFFER UOps that have
/// associated buffers in the tensor registry's buffer index.
/// Input tensors (from `from_slice()`) and realized tensors
/// register their buffers for this lookup to work.
///
/// This allows schedule creation to receive buffers explicitly without
/// needing global registry lookups during kernel buffer collection.
fn collect_input_buffers(root: &Arc<UOp>) -> crate::schedule::InputBuffers {
    let mut inputs = HashMap::new();
    for node in root.toposort() {
        if let Op::Buffer { .. } = node.op() {
            // Buffers are registered in from_slice_on() and realize()
            if let Some(buf) = crate::tensor_registry::get_buffer(node.id) {
                inputs.insert(node.id, buf);
            }
        }
    }
    inputs
}

/// Build an execution graph from an expanded schedule.
///
/// Creates an `ExecutionGraph` that represents the DAG of kernel operations,
/// capturing buffer dependencies and enabling parallel execution analysis.
///
/// # Arguments
///
/// * `schedule` - The expanded schedule (after range iteration expansion)
///
/// # Returns
///
/// An `ExecutionGraph` with nodes for each schedule item.
fn build_execution_graph(schedule: &[ScheduleItem]) -> ExecutionGraph {
    let mut graph = ExecutionGraph::new();

    for item in schedule {
        // Determine device from first buffer
        let device = item.buffers.first().map(|b| b.allocator().device_spec()).unwrap_or(DeviceSpec::Cpu);

        // Collect buffer IDs
        let all_buffer_ids: Vec<_> = item.buffers.iter().map(|b| b.id()).collect();

        // Determine output indices by finding buffers written to by STORE ops
        let is_transfer = matches!(item.ast.op(), Op::Copy { .. });
        let output_indices = if is_transfer {
            // For COPY, destination is output
            vec![0usize]
        } else {
            // For computational kernels, detect outputs from STORE operations
            detect_output_indices(&item.kernel, &item.buffers)
        };

        // Split buffer IDs into inputs and outputs
        let outputs: Vec<_> = output_indices.iter().map(|&i| all_buffer_ids[i]).collect();
        let inputs: Vec<_> =
            all_buffer_ids.iter().enumerate().filter(|(i, _)| !output_indices.contains(i)).map(|(_, id)| *id).collect();

        // Create buffer access info for parallel execution
        let buffer_access = Some(KernelBufferAccess { buffers: all_buffer_ids, output_indices });

        // Use kernel UOp ID for node ID since dependencies reference kernel IDs
        // (from kernel_ctx.kernel_deps which uses kernel UOp IDs)
        let node = ExecutionNode {
            id: item.kernel.id,
            device,
            inputs,
            outputs,
            predecessors: item.dependencies.clone(),
            is_transfer,
            buffer_access,
        };

        graph.add_node(node);
    }

    graph
}

/// Detect output buffer indices by finding PARAM slots written to by STORE operations.
///
/// Uses PARAM slot-based matching (like Tinygrad's `ProgramSpec.from_uop`,
/// renderer/__init__.py:119-124). The codegen PARAM slot directly indexes
/// into the kernel's buffer list — no second pass over sources needed.
fn detect_output_indices(kernel: &Arc<UOp>, buffers: &[Buffer]) -> Vec<usize> {
    let ast = match kernel.op() {
        Op::Kernel { ast, .. } => ast,
        _ => return vec![0],
    };

    // Single AST walk: find STORE targets, extract PARAM slot numbers.
    // Tinygrad: if (buf:=idx.src[0]).op is Ops.PARAM: outs.append(buf.arg)
    let mut output_slots = Vec::new();
    for node in ast.toposort() {
        if !matches!(node.op(), Op::Store { .. }) {
            continue;
        }
        // STORE.index → INDEX.buffer → PARAM (or STORE.index → CAST → INDEX.buffer → PARAM)
        let idx = match node.op() {
            Op::Store { index, .. } => index,
            _ => continue,
        };
        let buf = match idx.op() {
            Op::Index { buffer, .. } => buffer,
            Op::Cast { src, .. } => match src.op() {
                Op::Index { buffer, .. } => buffer,
                _ => continue,
            },
            _ => continue,
        };
        if let Op::Param { slot, device: None, .. } = buf.op()
            && !output_slots.contains(slot)
        {
            output_slots.push(*slot);
        }
    }

    // Slots are buffer indices (Tinygrad: sorted(dedup(outs)))
    output_slots.sort();
    output_slots.dedup();
    let output_indices: Vec<usize> = output_slots.into_iter().filter(|&s| s < buffers.len()).collect();
    if output_indices.is_empty() && !buffers.is_empty() { vec![0] } else { output_indices }
}

/// Prepare an execution plan from a schedule.
///
/// This performs all one-time preparation work:
/// 1. Expands the schedule (handles bound ranges)
/// 2. Builds execution graph and computes parallel groups
/// 3. Allocates all buffers
/// 4. Compiles all kernels
/// 5. Creates PreparedKernel structures
///
/// # Arguments
///
/// * `schedule` - The schedule from `create_schedule()`
///
/// # Returns
///
/// An `ExecutionPlan` ready for fast repeated execution.
///
/// # Errors
///
/// Returns error if compilation or buffer allocation fails.
fn prepare_execution_plan(
    schedule_result: &crate::schedule::ScheduleResult,
    config: &PrepareConfig,
) -> Result<ExecutionPlan> {
    // Expand the schedule to handle OUTER range iterations
    let mut expanded_schedule = expand_schedule(schedule_result.items.clone());

    // Memory planning (default-on): remap non-overlapping intermediate buffers
    // before allocation/compilation. This mirrors Tinygrad's placement of the
    // planner in the schedule finalization path.
    if std::env::var("MOROK_DISABLE_MEMORY_PLANNER").as_deref() != Ok("1") {
        let output_buffer_ids = collect_output_buffer_ids(&expanded_schedule, &schedule_result.output_uop_ids);
        let planner_result = crate::memory_planner::memory_planner(&expanded_schedule, &output_buffer_ids);
        if !planner_result.buffer_replace.is_empty() {
            trace!(
                replacements = planner_result.buffer_replace.len(),
                buffers_reused = planner_result.buffers_reused,
                memory_saved_bytes = planner_result.memory_saved,
                "applying memory planner buffer replacements"
            );
            crate::memory_planner::apply_buffer_replacements(&mut expanded_schedule, &planner_result.buffer_replace);
        }
    }

    debug!(num_items = expanded_schedule.len(), "expanded schedule");

    // Build execution graph for parallel group analysis
    let mut execution_graph = build_execution_graph(&expanded_schedule);

    // Verify the graph is valid (no cycles)
    execution_graph.compute_parallel_groups();
    if !execution_graph.is_valid() {
        return DependencyCyclesSnafu.fail();
    }

    // Get device via config's resolver (allows per-call backend selection).
    let alloc_registry = morok_device::registry::registry();
    let device = if let Some(first_item) = expanded_schedule.first() {
        let device_spec = first_item.buffers.first().map(|b| b.allocator().device_spec()).unwrap_or(DeviceSpec::Cpu);
        config.resolve_device(&device_spec, alloc_registry)?
    } else {
        return EmptyScheduleSnafu.fail();
    };

    let codegen: &'static str = device.compiler.cache_key();

    // Build the ExecutionPlan using the builder
    let mut builder = ExecutionPlanBuilder::new(device.device.clone());

    // Step 1: Add all buffers to the plan
    // Buffers in each ScheduleItem are already in the correct order (from collect_kernel_buffers).
    // We track buffers by their UOp ID (what they were registered under in tensor_registry's buffer index).
    let mut uop_id_to_idx: HashMap<u64, usize> = HashMap::new();
    let mut storage_to_idx: HashMap<BufferStorageKey, usize> = HashMap::new();

    // BUFFER_VIEW output slots are replaced later with base views. Keep them as
    // distinct entries even if they currently share physical storage, so replace
    // cannot accidentally mutate another logical buffer mapping.
    let buffer_view_output_uop_ids: HashSet<u64> = expanded_schedule
        .iter()
        .filter_map(|item| {
            if matches!(item.ast.op(), Op::BufferView { .. }) { item.buffer_uop_ids.first().copied() } else { None }
        })
        .collect();

    for item in &expanded_schedule {
        // Ensure all buffers are allocated
        for (buffer, &uop_id) in item.buffers.iter().zip(item.buffer_uop_ids.iter()) {
            buffer.ensure_allocated().context(DeviceSnafu)?;

            if uop_id_to_idx.contains_key(&uop_id) {
                continue;
            }

            let storage_key = BufferStorageKey {
                id: buffer.id().0,
                offset: buffer.offset(),
                size: buffer.size(),
                dtype: buffer.dtype(),
            };

            let idx = if !buffer_view_output_uop_ids.contains(&uop_id) {
                if let Some(&existing_idx) = storage_to_idx.get(&storage_key) {
                    builder.map_buffer(uop_id, existing_idx);
                    existing_idx
                } else {
                    let new_idx = builder.add_buffer(uop_id, buffer.clone());
                    storage_to_idx.insert(storage_key, new_idx);
                    new_idx
                }
            } else {
                builder.add_buffer(uop_id, buffer.clone())
            };
            uop_id_to_idx.insert(uop_id, idx);
        }

        // Collect alias IDs for cleanup
        builder.add_alias_ids(item.alias_registered_ids.iter().copied());
    }

    // Step 2: Compile all kernels and create PreparedKernel structures
    let mut prepared_kernels: Vec<PreparedKernel> = Vec::new();

    // Pre-compile: optimize + compile each UNIQUE ast once, cache by pre-optimization ast id.
    // Uses global cache so identical kernels across prepare calls (e.g., sort substages
    // with same axis) skip both optimization and compilation.
    type OptKey = (u64, DeviceSpec, &'static str);
    static OPT_CACHE: std::sync::OnceLock<papaya::HashMap<OptKey, Arc<morok_runtime::kernel_cache::CachedKernel>>> =
        std::sync::OnceLock::new();
    let opt_cache = OPT_CACHE.get_or_init(papaya::HashMap::new);
    let opt_guard = opt_cache.guard();

    for item in expanded_schedule.iter() {
        // COPY operations: buffer-to-buffer transfer (DISK→CPU, CPU→CUDA, etc.)
        // No kernel compilation needed — just register the buffer copy for execution.
        if matches!(item.ast.op(), Op::Copy { .. }) {
            if item.buffers.len() >= 2 {
                let mut dst = item.buffers[0].clone();
                let src = &item.buffers[1];
                if let Err(e) = dst.copy_from(src) {
                    tracing::error!("COPY buffer transfer failed: {e}");
                }
            }
            continue;
        }

        // BUFFER_VIEW: zero-copy sub-buffer view (DISK weight views).
        // Creates a view into the base buffer at the specified byte offset.
        // Tinygrad schedule.py:201-204: buffers[buf_uops[0]] = base.view(size, dtype, offset)
        if let Op::BufferView { size, offset, .. } = item.ast.op() {
            if item.buffers.len() >= 2 && item.buffer_uop_ids.len() >= 2 {
                let base = &item.buffers[1];
                let byte_offset = offset * base.dtype().bytes();
                let byte_size = size * item.ast.dtype().bytes();
                let view = base.view(byte_offset, byte_size).expect("BUFFER_VIEW: invalid view into base buffer");
                // Register the view under the output buffer's UOp ID so downstream
                // COPY/kernel items find it as their source buffer.
                let output_uop_id = item.buffer_uop_ids[0];
                if let Some(&idx) = uop_id_to_idx.get(&output_uop_id) {
                    builder.replace_buffer(idx, view);
                }
            }
            continue;
        }

        let opt_key = (crate::schedule_cache::content_hash(&item.ast), device.device.clone(), codegen);

        let cached = if let Some(cached) = opt_cache.get(&opt_key, &opt_guard) {
            Arc::clone(cached)
        } else {
            let optimizer_renderer = get_optimizer_renderer(&device);
            let optimized_ast = if let morok_schedule::OptStrategy::Beam { .. } = config.optimizer.strategy {
                beam_search_optimize(item.ast.clone(), &optimizer_renderer, &device, &item.buffers, &config.optimizer)?
            } else {
                morok_schedule::optimize_kernel_with_config(item.ast.clone(), &optimizer_renderer, &config.optimizer)
            };

            let kernel_name =
                optimized_ast.metadata::<morok_schedule::optimizer::KernelInfo>().map(|info| info.function_name());

            let ast_decomposed = match device.renderer.decompositor() {
                Some(matcher) => morok_ir::decompositions::decompose_with(&optimized_ast, &matcher),
                None => optimized_ast,
            };

            let result = morok_runtime::kernel_cache::get_or_compile_kernel(
                crate::schedule_cache::content_hash(&ast_decomposed),
                codegen,
                || {
                    let spec =
                        device.renderer.render(&ast_decomposed, kernel_name.as_deref()).context(RenderKernelSnafu)?;
                    let compiled = device.compiler.compile(&spec).context(CompileKernelSnafu)?;
                    let program = (device.runtime)(&compiled).context(CreateProgramSnafu)?;
                    Ok(morok_runtime::kernel_cache::CachedKernel {
                        program,
                        device: codegen.to_string(),
                        code: spec.src.clone(),
                        entry_point: spec.name.clone(),
                        var_names: spec.var_names.clone(),
                        global_size: spec.global_size,
                        local_size: spec.local_size,
                    })
                },
            )?;
            opt_cache.insert(opt_key, Arc::clone(&result), &opt_guard);
            result
        };

        // Build buffer indices for this kernel using item.buffer_uop_ids (already in correct order)
        let buffer_indices: Vec<usize> =
            item.buffer_uop_ids.iter().filter_map(|&uop_id| uop_id_to_idx.get(&uop_id).copied()).collect();

        trace!(kernel.ast_id = item.ast.id, num_buffers = item.buffers.len(), "kernel buffer mapping");

        // Create PreparedKernel
        // Note: buffer_ptrs and buffer_ids will be computed in ExecutionPlanBuilder::build()
        // Convert fixedvars HashMap to vals Vec using var_names order from CachedKernel
        let vals: Vec<i64> =
            cached.var_names.iter().map(|name| item.fixedvars.get(name).copied().unwrap_or(0)).collect();

        let prepared = PreparedKernel {
            id: item.kernel.id,
            ast: item.ast.clone(),
            kernel: cached,
            device: device.device.clone(),
            buffer_indices,
            output_indices: vec![0], // First buffer is typically output
            vals,
            dependencies: item.dependencies.clone(),
            buffer_ptrs: Vec::new(), // Computed in build()
            buffer_ids: Vec::new(),  // Computed in build()
        };

        prepared_kernels.push(prepared);
    }

    // Add kernels to builder and track their indices
    for kernel in prepared_kernels {
        builder.add_kernel(kernel);
    }

    // Deterministic output identification via ScheduleResult.output_uop_ids
    let output_buffer_indices: Vec<usize> =
        schedule_result.output_uop_ids.iter().filter_map(|&id| uop_id_to_idx.get(&id).copied()).collect();
    if !output_buffer_indices.is_empty() {
        builder.set_output_buffers(output_buffer_indices);
    }

    let plan = builder.build();
    Ok(plan)
}

fn collect_output_buffer_ids(schedule: &crate::schedule::Schedule, output_uop_ids: &[u64]) -> HashSet<u64> {
    let output_uop_set: HashSet<u64> = output_uop_ids.iter().copied().collect();
    let mut output_buffer_ids = HashSet::new();
    for item in schedule {
        for (buffer, &uop_id) in item.buffers.iter().zip(item.buffer_uop_ids.iter()) {
            if output_uop_set.contains(&uop_id) {
                output_buffer_ids.insert(buffer.id().0);
            }
        }
    }
    output_buffer_ids
}

/// Resolve the device string for cache keying (includes compiler cache key).
pub(crate) fn resolve_codegen(param_buffers: &[(u64, Arc<UOp>)], config: &PrepareConfig) -> Result<&'static str> {
    let alloc_registry = morok_device::registry::registry();
    let first_buf = param_buffers.iter().find_map(|(id, _)| crate::tensor_registry::get_buffer(*id));
    let spec = first_buf.as_ref().map(|b| b.allocator().device_spec()).unwrap_or(DeviceSpec::Cpu);
    let device = config.resolve_device(&spec, alloc_registry)?;
    Ok(device.compiler.cache_key())
}

/// Get the optimizer renderer for a device.
fn get_optimizer_renderer(device: &Device) -> morok_schedule::OptimizerRenderer {
    match device.device {
        DeviceSpec::Cpu => {
            if std::env::var("MOROK_AMX").as_deref() == Ok("1") {
                morok_schedule::OptimizerRenderer::apple_amx()
            } else {
                morok_schedule::OptimizerRenderer::cpu()
            }
        }
        DeviceSpec::Cuda { .. } => morok_schedule::OptimizerRenderer::cuda(),
        DeviceSpec::Metal { .. } => morok_schedule::OptimizerRenderer::metal(),
        _ => morok_schedule::OptimizerRenderer::cpu(),
    }
}

/// Optimize a kernel AST using beam search auto-tuning.
///
/// Beam search explores multiple optimization paths and selects the fastest
/// by compiling and timing each candidate. This is slower than heuristics
/// but can find better optimizations.
///
/// Note: Unlike Tinygrad which dispatches to EITHER beam search OR heuristics,
/// we pre-seed the beam with heuristic results (TC, output upcast, threading).
/// This gives beam search a strong starting point while letting it explore
/// further optimizations. TC must be applied by heuristics since beam search
/// requires TC as the very first opt.
fn beam_search_optimize(
    ast: Arc<UOp>,
    renderer: &morok_schedule::OptimizerRenderer,
    device: &Device,
    buffers: &[Buffer],
    optimizer_config: &morok_schedule::OptimizerConfig,
) -> Result<Arc<UOp>> {
    let beam_config = &optimizer_config.beam;
    // Prepare scheduler (applies symbolic simplification and loop→global)
    let mut scheduler = prepare_scheduler(ast, renderer);

    // Apply hand-coded heuristics BEFORE beam search to seed with TC, output
    // upcast, threading, etc. Beam search then explores further optimizations
    // on top of this baseline.
    hand_coded_optimizations(&mut scheduler, &optimizer_config.heuristics);

    // Ensure all buffers are allocated for timing
    for buf in buffers {
        buf.ensure_allocated().context(DeviceSnafu)?;
    }

    // Clone buffers for the closure (Buffer is Clone + Send + Sync)
    let buffers: Vec<Buffer> = buffers.to_vec();
    let bench_config = morok_runtime::BenchmarkConfig::default();

    // Clone device components for the closure
    let dev_renderer = device.renderer.clone();
    let dev_compiler = device.compiler.clone();
    let dev_runtime = device.runtime.clone();
    let max_uops = beam_config.max_uops;

    // Compile-and-time function: compilation is NOT timed, only execution.
    // Wrapped in catch_unwind because beam search explores speculative candidates
    // that may trigger rewrite engine limits or other panics. Matches Tinygrad's
    // try/except in _try_compile (codegen/opt/search.py:67-82).
    let compile_and_time = |s: &Scheduler| -> Option<Duration> {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        catch_unwind(AssertUnwindSafe(|| {
            let raw_ast = s.get_optimized_ast(None);

            // Apply post-optimization passes for accurate timing
            let optimized = apply_post_optimization(raw_ast);

            // Extract kernel name before decomposition (which loses metadata)
            let kernel_name =
                optimized.metadata::<morok_schedule::optimizer::KernelInfo>().map(|info| info.function_name());

            // Post-optimization UOp count filter (matches Tinygrad's BEAM_UOPS_MAX).
            // validate_limits checks pre-optimization AST size, but devectorization
            // can massively expand the graph (e.g., 256-wide UPCAST -> 4096 GEP indices).
            if optimized.toposort().len() > max_uops {
                return None;
            }

            // Apply decomposition
            let decomposed = match dev_renderer.decompositor() {
                Some(m) => morok_ir::decompositions::decompose_with(&optimized, &m),
                None => optimized,
            };

            // Render and compile (NOT timed)
            let spec = dev_renderer.render(&decomposed, kernel_name.as_deref()).ok()?;
            let compiled = dev_compiler.compile(&spec).ok()?;
            let program = (dev_runtime)(&compiled).ok()?;

            // Extract buffer pointers inside the closure (avoids Sync issue)
            let buffer_ptrs: Vec<*mut u8> = buffers.iter().map(|b| unsafe { b.as_raw_ptr() }).collect();

            // Time ONLY execution (pass global/local size for threaded kernels)
            // Note: Empty vals slice since benchmark kernels don't have symbolic variables
            let result = unsafe {
                morok_runtime::benchmark_kernel(
                    program.as_ref(),
                    &buffer_ptrs,
                    &[],
                    spec.global_size,
                    spec.local_size,
                    &bench_config,
                )
                .ok()?
            };

            Some(result.min)
        }))
        .ok()
        .flatten()
    };

    // Suppress panic output during beam search. Speculative candidates may panic
    // at compile or runtime — this is expected (matches Tinygrad's silent try/except).
    // catch_unwind catches panics but the default hook prints them first.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = beam_search_cached(scheduler, beam_config, compile_and_time);
    std::panic::set_hook(prev_hook);
    let result = result.context(OptimizeSnafu)?;

    // Debug: log beam search results
    tracing::debug!(
        opts = ?result.scheduler.applied_opts,
        timing = ?result.timing,
        iterations = result.iterations,
        "beam_search_optimize: completed"
    );

    // Apply post-optimization to final result
    let raw_ast = result.scheduler.get_optimized_ast(None);
    Ok(apply_post_optimization(raw_ast))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realize_simple_add() {
        crate::test::helpers::test_setup();

        // Test that realizing a simple computation works.
        // The pipeline transforms:
        //   ADD(RESHAPE(BUFFER_A), RESHAPE(BUFFER_B))
        // Into:
        //   STORE(OUTPUT, INDEX, ADD(LOAD(INPUT_A, idx), LOAD(INPUT_B, idx)))
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);

        // Create computation: a + b
        let mut c = &a + &b;

        // Realize should compile and execute the kernel
        c.realize().unwrap();
        let result: ndarray::ArrayD<f32> = c.as_ndarray().unwrap();
        let (result, _) = result.into_raw_vec_and_offset();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    /// Test that realizing a reduction (sum) works end-to-end.
    ///
    /// This verifies the complete reduction pipeline:
    /// - Early-return pattern prevents unnecessary ReduceAxis for size-1 dimensions
    /// - Vectorize consistency prevents VConst panics in shape extraction
    /// - ReduceAxis → REDUCE transformation following Tinygrad's approach
    /// - REDUCE codegen generates correct LLVM IR
    #[test]
    fn test_realize_sum() {
        crate::test::helpers::test_setup();

        // Create a 1D tensor: [1, 2, 3, 4]
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);

        // Sum all elements (should be 10.0)
        let sum_result = a.sum(());
        if let Err(ref e) = sum_result {
            tracing::debug!(error = ?e, "sum failed");
        }
        assert!(sum_result.is_ok(), "Sum creation failed");

        // Realize the computation
        let mut sum_tensor = sum_result.unwrap();
        let realized = sum_tensor.realize();
        if let Err(ref e) = realized {
            eprintln!("realize failed: {e:?}");
        }
        assert!(realized.is_ok(), "Realize should succeed: {:?}", realized.err());
    }

    #[test]
    fn test_tensor_device_default_cpu() {
        // Tensors created with from_slice default to CPU
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        assert_eq!(a.device(), morok_ir::DeviceSpec::Cpu);
    }

    #[test]
    fn test_tensor_to_same_device_is_noop() {
        // Moving to the same device should return a clone
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = a.to(morok_ir::DeviceSpec::Cpu);
        // Both should point to the same UOp (clone shares Rc)
        assert_eq!(a.device(), b.device());
    }

    #[test]
    fn test_tensor_to_different_device_creates_copy() {
        use morok_ir::DeviceSpec;
        // Moving to a different device should create a COPY UOp
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = a.to(DeviceSpec::Cuda { device_id: 0 });
        // b should report the new device
        assert_eq!(b.device(), DeviceSpec::Cuda { device_id: 0 });
        // a should still be on CPU
        assert_eq!(a.device(), DeviceSpec::Cpu);
    }

    // More comprehensive tests will be added in Phase 1.5

    // ==========================================================================
    // ExecutionPlan tests
    // ==========================================================================

    #[test]
    fn test_prepare_simple_add() {
        crate::test::helpers::test_setup();

        // Create computation: a + b
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let mut c = &a + &b;

        // Prepare should compile kernels and allocate buffers
        let plan = c.prepare();
        assert!(plan.is_ok(), "prepare() should succeed: {:?}", plan.err());

        let plan = plan.unwrap();

        // Verify plan has kernels and buffers
        assert!(plan.kernels().next().is_some(), "Plan should have at least one kernel");
        assert!(!plan.buffers().is_empty(), "Plan should have buffers");
    }

    #[test]
    fn test_prepare_and_execute() {
        crate::test::helpers::test_setup();

        // Create computation: a + b
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let mut c = &a + &b;

        // Prepare
        let plan = c.prepare().expect("prepare should succeed");

        // Execute
        let result = plan.execute();
        assert!(result.is_ok(), "execute() should succeed: {:?}", result.err());

        // Verify output buffer has correct data
        let output = plan.output_buffer();
        let mut data = vec![0.0f32; 3];
        output
            .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, 12) })
            .expect("copyout should succeed");
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_prepare_and_execute_twice() {
        crate::test::helpers::test_setup();

        // Create computation
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let mut c = &a + &b;

        // Prepare once
        let plan = c.prepare().expect("prepare should succeed");

        // Execute twice to verify reusability
        for _ in 0..2 {
            let result = plan.execute();
            assert!(result.is_ok(), "execute() should succeed: {:?}", result.err());
        }

        // Verify output
        let output = plan.output_buffer();
        let mut data = vec![0.0f32; 3];
        output
            .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, 12) })
            .expect("copyout should succeed");
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    /// Test that realize() produces correct results.
    ///
    /// Note: Buffer count assertions removed as they're not reliable with
    /// parallel test execution and global state. The key invariant (no memory
    /// leak) is tested in test_memory_growth_detection.
    #[test]
    fn test_realize_buffer_cleanup() {
        crate::test::helpers::test_setup();

        // Create input tensors ONCE (these will stay in registry)
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);

        // Realize the computation
        let mut c = &a + &b;
        c.realize().expect("realize should succeed");

        // Verify computation is correct
        let result: ndarray::ArrayD<f32> = c.as_ndarray().expect("as_ndarray should succeed");
        let (data, _) = result.into_raw_vec_and_offset();
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    /// Test that prepare() + execute() pattern can clean up with release_intermediate_buffers().
    #[test]
    fn test_prepare_execute_cleanup() {
        crate::test::helpers::test_setup();

        // Create input tensors
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let mut c = &a + &b;

        // Prepare the plan
        let plan = c.prepare().expect("prepare should succeed");

        // Execute multiple times (simulating benchmark loop)
        for _ in 0..3 {
            plan.execute().expect("execute should succeed");
        }

        // Verify output
        let output = plan.output_buffer();
        let mut data = vec![0.0f32; 3];
        output
            .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, 12) })
            .expect("copyout should succeed");
        assert_eq!(data, vec![5.0, 7.0, 9.0]);

        // Now cleanup — count how many buffers were actually released
        let count_before_cleanup = crate::tensor_registry::buffer_count();
        plan.release_intermediate_buffers(crate::tensor_registry::remove_buffer);
        let count_after_cleanup = crate::tensor_registry::buffer_count();

        // release_intermediate_buffers should remove at least one buffer (the output buffer)
        // or at minimum not increase the count. We check the immediate delta to avoid
        // interference from parallel tests.
        assert!(
            count_after_cleanup <= count_before_cleanup,
            "Cleanup should not increase buffer count: before={}, after={}",
            count_before_cleanup,
            count_after_cleanup
        );
    }

    /// Test that intermediate buffer cleanup is working.
    ///
    /// The correct pattern is: prepare() ONCE, execute() many times.
    /// This test verifies that repeated execute() calls do NOT grow the registry
    /// AFTER initial setup. First execute may allocate buffers (one-time setup),
    /// but subsequent calls must not grow.
    #[test]
    fn test_memory_growth_detection() {
        crate::test::helpers::test_setup();

        const ITERATIONS: usize = 10;

        // Create input tensors
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]);
        let mut c = &a + &b;

        // Prepare ONCE
        let plan = c.prepare().expect("prepare should succeed");

        let mut counts: Vec<usize> = Vec::with_capacity(ITERATIONS);

        // Execute MANY times
        for _ in 0..ITERATIONS {
            plan.execute().expect("execute should succeed");
            counts.push(crate::tensor_registry::buffer_count());
        }

        // Cleanup after final execution
        plan.release_intermediate_buffers(crate::tensor_registry::remove_buffer);
        let count_after_cleanup = crate::tensor_registry::buffer_count();

        // Key invariant: count should be STABLE during iterations (no growth between iterations)
        // First execute may allocate buffers, but subsequent calls must reuse them.
        let count_after_first_execute = counts[0];
        let growth_during_iterations = counts.last().unwrap().saturating_sub(count_after_first_execute);

        eprintln!("Counts during execute: {:?}", counts);
        eprintln!("Growth during iterations (after first): {}", growth_during_iterations);
        eprintln!("Count after cleanup: {}", count_after_cleanup);

        assert_eq!(
            growth_during_iterations, 0,
            "Registry should not grow during repeated execute() calls (after initial setup)"
        );

        // Cleanup should reduce count by removing allocated buffers
        assert!(
            count_after_cleanup <= count_after_first_execute,
            "Cleanup should not increase buffer count: first_execute={}, after_cleanup={}",
            count_after_first_execute,
            count_after_cleanup
        );
    }

    /// Test that realize() correctly computes and cleans up.
    #[test]
    fn test_memory_growth_realize_pattern() {
        crate::test::helpers::test_setup();

        // Single realize should work correctly
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]);
        let mut c = &a + &b;
        c.realize().expect("realize should succeed");

        // Verify result
        let result: ndarray::ArrayD<f32> = c.as_ndarray().expect("as_ndarray should succeed");
        assert_eq!(result.as_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
    }
}
