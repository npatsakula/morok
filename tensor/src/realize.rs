//! Tensor realization (execution) API.
//!
//! This module provides the execution pipeline for tensor operations:
//! 1. **Rangeify** - Transform movement ops to BUFFERIZE + INDEX
//! 2. **Kernel splitting** - Split at STORE boundaries into CALL wrappers
//! 3. **Scheduling** - Extract callables and create execution schedule
//! 4. **Execution** - Compile and run each kernel in dependency order
//!
//! Runtime plan execution is dependency-ordered with conservative mixed-op
//! barriers and hazard-aware host parallelism for safe compiled kernels.
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
//! plan.execute()?;
//!
//! // Get results
//! let output = plan.output_buffer();
//! ```

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use morok_schedule::{
    Scheduler, apply_post_optimization, beam_search_cached, hand_coded_optimizations, prepare_scheduler,
};
use tracing::{debug, trace};

use crate::{
    PrepareConfig, Result, Tensor,
    error::{
        BatchOutputMismatchSnafu, CompileKernelSnafu, CreateProgramSnafu, DeviceSnafu, EmptyScheduleSnafu,
        ExecutionSnafu, IrConstructionSnafu, OptimizeSnafu, RangeifySnafu, RenderKernelSnafu, ShapeUnknownSnafu,
        UOpSnafu,
    },
    schedule::ScheduleItem,
};
use morok_device::{Buffer, device::Device};
use morok_ir::pattern::is_any_const;
use morok_ir::{AxisType, DeviceSpec, Op, UOp, UOpKey};
use morok_runtime::{
    ExecutionPlan, ExecutionPlanBuilder, PreparedBufferView, PreparedCopy, PreparedCustomFunction, PreparedKernel,
    PreparedOp,
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
        let output_buf = plan.output_buffer().expect("realized plan must have an output buffer").clone();

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
    /// 2. Instantiates strict range-expanded callable schedule items
    /// 3. Compiles all kernels
    /// 4. Allocates all buffers
    /// 5. Builds dependency-ordered prepared op execution plan
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
    /// plan.execute()?;
    /// ```
    pub fn prepare_with(&mut self, config: &PrepareConfig) -> Result<ExecutionPlan> {
        let t_total = std::time::Instant::now();
        let uop = self.uop();

        let sink = UOp::sink(vec![uop.contiguous()]);
        let schedule_result = schedule_result_from_sink_with_cache(sink, extract_var_vals(&uop)?, config)?;
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
            let buf = Arc::new(plan.output_buffer().expect("plan with num_outputs > 0 must expose output").clone());
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

        let mut var_vals = HashMap::new();
        for uop in &old_uops {
            let extracted = extract_var_vals(uop)?;
            merge_var_vals_checked(&mut var_vals, &extracted, "realize_batch input collection")?;
        }
        let schedule_result = schedule_result_from_sink_with_cache(sink, var_vals, config)?;

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
            let output_buf = plan.output_buffer_at(buf_idx).expect("buf_idx in range").clone();
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

        // Collect UOps from pending tensors only
        let uops: Vec<Arc<UOp>> = pending_indices.iter().map(|&i| tensors[i].uop()).collect();

        let mut var_vals = HashMap::new();
        for uop in &uops {
            let extracted = extract_var_vals(uop)?;
            merge_var_vals_checked(&mut var_vals, &extracted, "prepare_batch input collection")?;
        }

        // Create merged SINK(CONTIGUOUS(t1), ..., CONTIGUOUS(tN)) from pending tensors
        let contiguouses: Vec<Arc<UOp>> = uops.iter().map(|u| u.contiguous()).collect();
        let sink = UOp::sink(contiguouses);

        let schedule_result = schedule_result_from_sink_with_cache(sink, var_vals, config)?;

        let plan = prepare_execution_plan(&schedule_result, config)?;

        // Wire each pending output tensor to its plan buffer.
        // After execute/execute_with_vars, tensor.array_view() reads the result directly.
        for (buf_idx, &orig_idx) in pending_indices.iter().enumerate() {
            if buf_idx >= plan.num_outputs() {
                break;
            }
            let output_buf = plan.output_buffer_at(buf_idx).expect("buf_idx in range").clone();
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

/// Extract bound variable values from a UOp graph (pre-pipeline).
///
/// Scans for BIND(DEFINE_VAR, CONST) nodes and extracts the mapping
/// from variable name to concrete bound value. This is the Morok equivalent
/// of Tinygrad's `strip_bind` in `pm_pre_sched_cache`.
///
/// These values are passed through to scheduling so that user Variables
/// (like `Variable::new("N", 1, 32).bind(4)`) are treated as fixed parameters
/// rather than OUTER ranges to be expanded.
/// Insert `(name, val)` into `var_vals` if not present, otherwise check that
/// any existing binding agrees. Returns `Err((prev, val))` on mismatch so
/// callers can format the error in their own context.
fn try_bind_var_val(var_vals: &mut HashMap<String, i64>, name: &str, val: i64) -> std::result::Result<(), (i64, i64)> {
    if let Some(&prev) = var_vals.get(name) {
        if prev != val {
            return Err((prev, val));
        }
        return Ok(());
    }
    var_vals.insert(name.to_string(), val);
    Ok(())
}

fn insert_var_val_checked(var_vals: &mut HashMap<String, i64>, name: &str, val: i64, context: &str) -> Result<()> {
    match try_bind_var_val(var_vals, name, val) {
        Ok(()) => Ok(()),
        Err((prev, val)) => {
            IrConstructionSnafu { details: format!("bind mismatch on {name}, {prev} != {val} ({context})") }.fail()
        }
    }
}

fn merge_var_vals_checked(dst: &mut HashMap<String, i64>, src: &HashMap<String, i64>, context: &str) -> Result<()> {
    for (name, val) in src {
        insert_var_val_checked(dst, name, *val, context)?;
    }
    Ok(())
}

fn extract_var_vals(root: &Arc<UOp>) -> Result<HashMap<String, i64>> {
    let mut var_vals = HashMap::new();
    for node in root.toposort() {
        if let Op::Bind { var, value } = node.op()
            && let Op::DefineVar { name, .. } = var.op()
            && let Op::Const(cv) = value.op()
            && let Some(val) = cv.0.try_int()
        {
            insert_var_val_checked(&mut var_vals, name, val, "bind extraction")?;
        }
    }
    Ok(var_vals)
}

fn schedule_cache_disabled_by_env() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("MOROK_DISABLE_SCHEDULE_CACHE").as_deref() == Ok("1"))
}

fn schedule_result_from_sink_with_cache(
    sink: Arc<UOp>,
    mut var_vals: HashMap<String, i64>,
    config: &PrepareConfig,
) -> Result<crate::schedule::ScheduleResult> {
    if config.disable_schedule_cache || schedule_cache_disabled_by_env() {
        return schedule_result_from_sink_uncached(sink, var_vals, config);
    }

    let normalization = normalize_for_schedule_cache(&sink)?;
    merge_var_vals_checked(&mut var_vals, &normalization.var_vals, "schedule cache normalization")?;

    let codegen = resolve_codegen(&normalization.param_buffers, config)?;
    let sched_key = (crate::schedule_cache::content_hash(&normalization.normalized), codegen);

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
            let schedule_root = restore_bind_placeholders_for_schedule(&normalization.normalized, &normalization);
            let rangeify_result = morok_schedule::rangeify_with_map(schedule_root, None).context(RangeifySnafu)?;
            let (kernel_graph, _) =
                morok_schedule::try_get_kernel_graph(rangeify_result.sink).context(RangeifySnafu)?;
            let pre_schedule = crate::schedule::create_pre_schedule(kernel_graph)?;
            let new_entry = Arc::new(crate::schedule_cache::CachedSchedule { pre_schedule: Arc::new(pre_schedule) });
            let guard = cache.guard();
            cache.insert(sched_key, Arc::clone(&new_entry), &guard);
            new_entry
        }
    };

    let restored_pre_schedule = restore_post_schedule_pre_schedule(&entry.pre_schedule, &normalization);
    let schedule_input_buffers = build_schedule_input_buffers(&restored_pre_schedule, &normalization);
    let result = crate::schedule::instantiate_schedule(&restored_pre_schedule, &schedule_input_buffers, &var_vals)?;
    Ok(result)
}

fn schedule_result_from_sink_uncached(
    sink: Arc<UOp>,
    var_vals: HashMap<String, i64>,
    _config: &PrepareConfig,
) -> Result<crate::schedule::ScheduleResult> {
    let rangeify_result = morok_schedule::rangeify_with_map(sink, None).context(RangeifySnafu)?;
    let (kernel_graph, _) = morok_schedule::try_get_kernel_graph(rangeify_result.sink).context(RangeifySnafu)?;
    let pre_schedule = crate::schedule::create_pre_schedule(kernel_graph.clone())?;
    let input_buffers = collect_input_buffers(&kernel_graph);
    let result = crate::schedule::instantiate_schedule(&pre_schedule, &input_buffers, &var_vals)?;
    Ok(result)
}

/// Pre-schedule cache normalization result.
///
/// Mirrors Tinygrad `pm_pre_sched_cache` behavior:
/// - BUFFER -> PARAM
/// - BUFFER_VIEW identities normalized via recursive BUFFER -> PARAM
/// - strip runtime value from BIND(DEFINE_VAR, CONST)
/// - normalize standalone UNIQUE identity -> LUNIQUE
pub(crate) struct ScheduleCacheNormalization {
    pub normalized: Arc<UOp>,
    pub param_values: Vec<Arc<UOp>>,
    pub param_buffers: Vec<(u64, Arc<UOp>)>,
    pub unique_values: Vec<Arc<UOp>>,
    pub var_vals: HashMap<String, i64>,
}

/// Context for pre-schedule cache normalization.
pub(crate) struct NormalizeScheduleCacheCtx {
    pub param_map: HashMap<u64, usize>,
    pub param_values: Vec<Arc<UOp>>,
    pub param_buffers: Vec<(u64, Arc<UOp>)>,
    pub var_vals: HashMap<String, i64>,
    pub bind_mismatch: Option<String>,
}

/// Full pre-schedule cache normalization (Tinygrad parity).
pub(crate) fn normalize_for_schedule_cache(sink: &Arc<UOp>) -> Result<ScheduleCacheNormalization> {
    let mut ctx = NormalizeScheduleCacheCtx {
        param_map: HashMap::new(),
        param_values: Vec::new(),
        param_buffers: Vec::new(),
        var_vals: HashMap::new(),
        bind_mismatch: None,
    };

    use morok_ir::op::pattern_derived::OpKey;
    use morok_ir::pattern::{RewriteResult, SimplifiedPatternMatcher};
    use morok_ir::rewrite::graph_rewrite;

    let mut matcher = SimplifiedPatternMatcher::<NormalizeScheduleCacheCtx>::new();

    fn to_param(
        node: &Arc<UOp>,
        ctx: &mut NormalizeScheduleCacheCtx,
        size: usize,
        device: Option<Arc<UOp>>,
    ) -> Arc<UOp> {
        let slot = *ctx.param_map.entry(node.id).or_insert_with(|| {
            let s = ctx.param_values.len();
            ctx.param_values.push(node.clone());
            s
        });
        UOp::param(slot, size, node.dtype(), device)
    }

    // BUFFER -> PARAM (erase runtime buffer identity in cache key).
    matcher.add(&[OpKey::Buffer], |node, ctx| {
        let Op::Buffer { size, device, .. } = node.op() else {
            return RewriteResult::NoMatch;
        };
        let slot = *ctx.param_map.entry(node.id).or_insert_with(|| {
            let s = ctx.param_values.len();
            ctx.param_values.push(node.clone());
            s
        });
        ctx.param_buffers.push((node.id, node.clone()));
        RewriteResult::Rewritten(UOp::param(slot, *size, node.dtype(), Some(device.clone())))
    });

    // BUFFER_VIEW -> PARAM (Tinygrad pm_replace_buf parity).
    matcher.add(&[OpKey::BufferView], |node, ctx| {
        let Op::BufferView { size, .. } = node.op() else {
            return RewriteResult::NoMatch;
        };
        RewriteResult::Rewritten(to_param(node, ctx, *size, Some(UOp::device(DeviceSpec::Cpu))))
    });

    // Strip runtime value from BIND for cache-key stability and collect var_vals.
    // Replaced with PARAM(device=Some) so restoration stays reversible and
    // distinguishable from internal PARAM(device=None) nodes created by rangeify.
    matcher.add(&[OpKey::Bind], |node, ctx| {
        let Op::Bind { var, value } = node.op() else {
            return RewriteResult::NoMatch;
        };
        let Op::DefineVar { name, .. } = var.op() else {
            return RewriteResult::NoMatch;
        };
        let Op::Const(cv) = value.op() else {
            return RewriteResult::NoMatch;
        };
        let Some(val) = cv.0.try_int() else {
            return RewriteResult::NoMatch;
        };

        if let Err((prev, val)) = try_bind_var_val(&mut ctx.var_vals, name, val) {
            if ctx.bind_mismatch.is_none() {
                ctx.bind_mismatch = Some(format!("bind mismatch on variable {name}: {prev} vs {val}"));
            }
            return RewriteResult::NoMatch;
        }
        RewriteResult::Rewritten(to_param(node, ctx, 0, Some(UOp::device(DeviceSpec::Cpu))))
    });

    // Tinygrad pre-schedule cache normalization:
    // - BUFFER(UNIQUE, DEVICE) -> PARAM
    // - BUFFER_VIEW base identity normalized through child BUFFER -> PARAM
    // - BIND(DEFINE_VAR, CONST) -> PARAM + var_vals capture
    let normalized = graph_rewrite(&matcher, sink.clone(), &mut ctx);

    if let Some(details) = ctx.bind_mismatch.take() {
        return IrConstructionSnafu { details }.fail();
    }

    // Normalize standalone UNIQUE identity to deterministic LUNIQUE slots.
    // This runs after BUFFER/BUFFER_VIEW replacement to avoid capturing UNIQUE
    // nodes that are no longer present in the normalized graph.
    struct UniqueNormalizationCtx {
        unique_map: HashMap<u64, usize>,
        unique_values: Vec<Arc<UOp>>,
    }
    let mut unique_ctx = UniqueNormalizationCtx { unique_map: HashMap::new(), unique_values: Vec::new() };
    let mut unique_matcher = SimplifiedPatternMatcher::<UniqueNormalizationCtx>::new();
    unique_matcher.add(&[OpKey::Unique], |node, ctx| {
        let Op::Unique(_) = node.op() else {
            return RewriteResult::NoMatch;
        };
        let slot = *ctx.unique_map.entry(node.id).or_insert_with(|| {
            let s = ctx.unique_values.len();
            ctx.unique_values.push(node.clone());
            s
        });
        RewriteResult::Rewritten(UOp::lunique(Some(slot)))
    });
    let normalized = graph_rewrite(&unique_matcher, normalized, &mut unique_ctx);

    ctx.param_buffers.sort_unstable_by_key(|(id, _)| *id);
    ctx.param_buffers.dedup_by_key(|(id, _)| *id);

    Ok(ScheduleCacheNormalization {
        normalized,
        param_values: ctx.param_values,
        param_buffers: ctx.param_buffers,
        unique_values: unique_ctx.unique_values,
        var_vals: ctx.var_vals,
    })
}

/// Post-schedule cache restore (Tinygrad `pm_post_sched_cache` equivalent).
///
/// Restores normalized placeholders back to runtime graph form for this run:
/// - PARAM(slot, device=Some(_)) -> original source node for current invocation
/// - BUFFER(LUNIQUE, DEVICE, size) -> fresh runtime BUFFER (memoized by slot)
/// - standalone LUNIQUE(slot) -> original UNIQUE identity
///
/// BIND runtime values are carried separately through `var_vals` and applied
/// at execution-time via fixedvars, preserving `execute_with_vars` behavior.
#[allow(clippy::mutable_key_type)]
pub(crate) fn restore_post_schedule_cache(root: &Arc<UOp>, normalization: &ScheduleCacheNormalization) -> Arc<UOp> {
    let mut subs: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    let mut lunique_buffers: HashMap<usize, Arc<UOp>> = HashMap::new();

    for node in root.toposort() {
        match node.op() {
            Op::Param { slot, device: Some(_), .. } => {
                if let Some(original) = normalization.param_values.get(*slot) {
                    let restored_original = restore_post_schedule_cache(original, normalization);
                    subs.insert(UOpKey(node.clone()), restored_original);
                }
            }
            Op::Buffer { unique, device, size } => {
                let Op::LUnique(slot) = unique.op() else {
                    continue;
                };
                let restored = if let Some(existing) = lunique_buffers.get(slot) {
                    existing.clone()
                } else {
                    let runtime_unique = UOp::buffer_id(None);
                    let fresh = UOp::new(
                        Op::Buffer { unique: runtime_unique, device: device.clone(), size: *size },
                        node.dtype(),
                    );
                    lunique_buffers.insert(*slot, fresh.clone());
                    fresh
                };
                subs.insert(UOpKey(node.clone()), restored);
            }
            Op::LUnique(slot) => {
                let restored = normalization.unique_values.get(*slot).cloned().unwrap_or_else(|| UOp::buffer_id(None));
                subs.insert(UOpKey(node.clone()), restored);
            }
            _ => {}
        }
    }

    // Tinygrad parity: restore over the whole cached graph.
    // This allows PARAM/BIND placeholders to be rewritten before schedule extraction.
    root.substitute(&subs)
}

/// Restore only normalized BIND placeholders back to BIND nodes.
///
/// Cache keying strips bind runtime values (`BIND -> PARAM`) for key stability,
/// but rangeify needs BIND semantics to preserve variable tracking. This helper
/// rewrites just those placeholders while keeping BUFFER/PARAM normalization —
/// the kernel AST must stay parametric so the cached pre-schedule can be reused
/// across runs with different runtime buffers (post-cache restoration only
/// swaps the buffer-uop *lists*, not the deep kernel AST).
#[allow(clippy::mutable_key_type)]
fn restore_bind_placeholders_for_schedule(root: &Arc<UOp>, normalization: &ScheduleCacheNormalization) -> Arc<UOp> {
    let mut subs: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    for node in root.toposort() {
        let Op::Param { slot, device: Some(_), .. } = node.op() else {
            continue;
        };

        let Some(original) = normalization.param_values.get(*slot) else {
            continue;
        };
        if matches!(original.op(), Op::Bind { .. }) {
            subs.insert(UOpKey(node.clone()), original.clone());
        }
    }

    if subs.is_empty() { root.clone() } else { root.substitute(&subs) }
}

/// Restore cached pre-schedule buffer UOps for the current invocation.
///
/// Tinygrad caches `pre_schedule` with normalized PARAM placeholders and then
/// post-rewrites only buffer UOps (`buf_uops`). This helper mirrors that flow:
/// callable identities/ASTs remain cached, while source/output buffer UOps are
/// restored to run-specific BUFFER identities.
fn restore_post_schedule_pre_schedule(
    pre_schedule: &crate::schedule::PreSchedule,
    normalization: &ScheduleCacheNormalization,
) -> crate::schedule::PreSchedule {
    let mut flat_buf_uops = Vec::new();
    let mut source_counts = Vec::with_capacity(pre_schedule.items.len());

    for item in &pre_schedule.items {
        source_counts.push(item.sources.len());
        flat_buf_uops.extend(item.sources.iter().cloned());
    }
    let outputs_offset = flat_buf_uops.len();
    flat_buf_uops.extend(pre_schedule.output_buffer_uops.iter().cloned());

    if flat_buf_uops.is_empty() {
        return pre_schedule.clone();
    }

    let restored_flat = match restore_post_schedule_cache(&UOp::sink(flat_buf_uops), normalization).op() {
        Op::Sink { sources, .. } => sources.iter().cloned().collect::<Vec<_>>(),
        _ => unreachable!("sink substitution must preserve SINK root"),
    };

    let mut cursor = 0usize;
    let mut restored_items = Vec::with_capacity(pre_schedule.items.len());
    for (item, source_count) in pre_schedule.items.iter().zip(source_counts) {
        let end = cursor + source_count;
        let sources = restored_flat[cursor..end].to_vec();
        cursor = end;
        let ast = restore_post_schedule_cache(&item.ast, normalization);
        restored_items.push(crate::schedule::PreScheduleItem {
            kernel: item.kernel.clone(),
            ast,
            sources,
            dependencies: item.dependencies.clone(),
            bound_ranges: item.bound_ranges.clone(),
        });
    }

    let output_buffer_uops = restored_flat[outputs_offset..].to_vec();
    crate::schedule::PreSchedule {
        items: restored_items,
        linear_ops: pre_schedule.linear_ops.clone(),
        output_buffer_uops,
    }
}

fn build_schedule_input_buffers(
    pre_schedule: &crate::schedule::PreSchedule,
    _normalization: &ScheduleCacheNormalization,
) -> crate::schedule::InputBuffers {
    let mut inputs = crate::schedule::InputBuffers::new();

    for item in &pre_schedule.items {
        for src in &item.sources {
            let buf = src.buf_uop();
            if let Op::Buffer { .. } = buf.op()
                && let Some(buffer) = crate::tensor_registry::get_buffer(buf.id)
            {
                inputs.insert(buf.id, buffer);
            }
        }
    }

    inputs
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

fn output_indices_from_program_metadata(globals: &[usize], outs: &[usize], num_buffers: usize) -> Result<Vec<usize>> {
    if num_buffers == 0 {
        return IrConstructionSnafu { details: "cannot map outputs for kernel with zero buffers".to_string() }.fail();
    }
    if globals.is_empty() {
        return IrConstructionSnafu { details: "ProgramSpec.globals is empty".to_string() }.fail();
    }
    if outs.is_empty() {
        return IrConstructionSnafu { details: "ProgramSpec.outs is empty".to_string() }.fail();
    }

    let slot_to_position: HashMap<usize, usize> =
        globals.iter().copied().enumerate().map(|(position, slot)| (slot, position)).collect();

    let mut output_indices = Vec::with_capacity(outs.len());
    for &slot in outs {
        let Some(position) = slot_to_position.get(&slot).copied() else {
            return IrConstructionSnafu {
                details: format!("ProgramSpec.outs slot {slot} not found in ProgramSpec.globals={globals:?}"),
            }
            .fail();
        };
        if position >= num_buffers {
            return IrConstructionSnafu {
                details: format!(
                    "ProgramSpec output index {position} (slot {slot}) out of range for {num_buffers} buffers"
                ),
            }
            .fail();
        }
        output_indices.push(position);
    }

    output_indices.sort_unstable();
    output_indices.dedup();
    if output_indices.is_empty() {
        return IrConstructionSnafu { details: "ProgramSpec output mapping resolved to empty set".to_string() }.fail();
    }

    Ok(output_indices)
}

fn resolve_item_buffer_indices(item: &ScheduleItem, uop_id_to_idx: &HashMap<u64, usize>) -> Result<Vec<usize>> {
    let mut indices = Vec::with_capacity(item.buffer_uop_ids.len());
    for &uop_id in &item.buffer_uop_ids {
        let Some(idx) = uop_id_to_idx.get(&uop_id).copied() else {
            return Err(crate::error::Error::BufferNotFound { uop_id });
        };
        indices.push(idx);
    }
    Ok(indices)
}

fn resolve_compiled_kernel_buffer_indices(
    item: &ScheduleItem,
    uop_id_to_idx: &HashMap<u64, usize>,
    globals: &[usize],
) -> Result<Vec<usize>> {
    let buffer_indices = resolve_item_buffer_indices(item, uop_id_to_idx)?;

    let mut ordered = Vec::with_capacity(globals.len());
    for &position in globals {
        let Some(idx) = buffer_indices.get(position).copied() else {
            return IrConstructionSnafu {
                details: format!(
                    "ProgramSpec.globals position {position} out of range for CALL {} buffer list len {} (buffer_uop_ids={:?})",
                    item.kernel.id,
                    buffer_indices.len(),
                    item.buffer_uop_ids
                ),
            }
            .fail();
        };
        ordered.push(idx);
    }

    Ok(ordered)
}

type OptKey = (u64, DeviceSpec, &'static str, u64);

/// Bounded global cache for optimized + compiled kernels keyed by AST hash.
///
/// Reads are lock-free via the underlying `papaya::HashMap`; the FIFO side
/// structure is touched only on insert under a short-lived mutex. The cap is
/// read once via `MOROK_OPT_CACHE_MAX` (default 4096); when capacity is
/// exceeded, the oldest insertions are evicted from both the map and the
/// FIFO.
struct OptCacheState {
    map: papaya::HashMap<OptKey, Arc<morok_runtime::kernel_cache::CachedKernel>>,
    fifo: parking_lot::Mutex<std::collections::VecDeque<OptKey>>,
    cap: usize,
}

impl OptCacheState {
    const DEFAULT_CAP: usize = 4096;

    fn new() -> Self {
        let cap = std::env::var("MOROK_OPT_CACHE_MAX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(Self::DEFAULT_CAP);
        Self { map: papaya::HashMap::new(), fifo: parking_lot::Mutex::new(std::collections::VecDeque::new()), cap }
    }

    fn insert(&self, key: OptKey, val: Arc<morok_runtime::kernel_cache::CachedKernel>) {
        let guard = self.map.guard();
        let was_new = self.map.insert(key.clone(), val, &guard).is_none();
        if !was_new {
            return;
        }
        let mut fifo = self.fifo.lock();
        fifo.push_back(key);
        while fifo.len() > self.cap {
            if let Some(evict) = fifo.pop_front() {
                self.map.remove(&evict, &guard);
            }
        }
    }
}

pub(crate) fn runtime_effect_ast(ast: &Arc<UOp>) -> &Arc<UOp> {
    match ast.op() {
        Op::End { computation, .. }
            if matches!(computation.op(), Op::Copy { .. } | Op::BufferView { .. } | Op::CustomFunction { .. }) =>
        {
            computation
        }
        _ => ast,
    }
}

fn optimizer_config_fingerprint(config: &PrepareConfig) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    config.optimizer.hash(&mut hasher);
    hasher.finish()
}

/// Prepare an execution plan from a schedule.
///
/// This performs all one-time preparation work:
/// 1. Allocates all buffers
/// 2. Compiles callable kernels
/// 3. Creates prepared runtime ops (compiled program + copy/view/custom-function handling)
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
    // Schedule items are already fully expanded by strict scheduler unroll.
    let mut schedule_items = schedule_result.items.clone();

    // Liveness-based pool reuse. Equivalent in correctness to tinygrad's arena
    // planner; differs only in storage strategy (per-pool Arc<Buffer> swaps vs.
    // monolithic arena with byte-offset BUFFER_VIEWs). Arena would pack tighter
    // and reduce allocator calls — tracked as the deferred PlannerMode::Arena
    // branch. Mode is selected by `MOROK_MEMORY_PLANNER`; `Disabled` short-
    // circuits the planner cleanly so callers don't need to gate the call.
    let planner_mode = crate::memory_planner::mode_from_env();
    let output_buffer_ids = collect_output_buffer_ids(&schedule_items, &schedule_result.output_uop_ids);
    let planner_result = crate::memory_planner::memory_planner(&schedule_items, &output_buffer_ids, planner_mode);
    if !planner_result.buffer_replace.is_empty() {
        trace!(
            replacements = planner_result.buffer_replace.len(),
            buffers_reused = planner_result.buffers_reused,
            memory_saved_bytes = planner_result.memory_saved,
            "applying memory planner buffer replacements"
        );
        crate::memory_planner::apply_reuse_dependencies(&mut schedule_items, &planner_result.reuse_dependencies);
        crate::memory_planner::apply_buffer_replacements(&mut schedule_items, &planner_result.buffer_replace);
    }

    debug!(num_items = schedule_items.len(), "schedule items ready for execution plan");

    // Resolve primary plan device from the first schedule item for plan metadata.
    // Individual compiled kernels may still resolve/compile on per-item devices.
    let alloc_registry = morok_device::registry::registry();
    let plan_device = if !schedule_items.is_empty() {
        let device_spec = schedule_items
            .iter()
            .flat_map(|item| item.buffers.iter().map(|b| b.allocator().device_spec()))
            .find(|spec| !spec.is_disk())
            .unwrap_or(DeviceSpec::Cpu);
        config.resolve_device(&device_spec, alloc_registry)?
    } else {
        return EmptyScheduleSnafu.fail();
    };
    let optimizer_fingerprint = optimizer_config_fingerprint(config);

    // Build the ExecutionPlan using the builder
    let mut builder = ExecutionPlanBuilder::new(plan_device.device.clone());

    // Step 1: Add all buffers to the plan
    // Buffers in each ScheduleItem are already in the correct order (from collect_callable_buffers).
    // We track buffers by their UOp ID (what they were registered under in tensor_registry's buffer index).
    let mut uop_id_to_idx: HashMap<u64, usize> = HashMap::new();
    let mut storage_to_idx: HashMap<BufferStorageKey, usize> = HashMap::new();

    // BUFFER_VIEW output slots are replaced later with base views. Keep them as
    // distinct entries even if they currently share physical storage, so replace
    // cannot accidentally mutate another logical buffer mapping.
    let buffer_view_output_uop_ids: HashSet<u64> = schedule_items
        .iter()
        .filter_map(|item| {
            if matches!(runtime_effect_ast(&item.ast).op(), Op::BufferView { .. }) {
                item.buffer_uop_ids.first().copied()
            } else {
                None
            }
        })
        .collect();

    for item in &schedule_items {
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

    // Step 2: Compile callable kernels and create prepared runtime ops

    // Pre-compile: optimize + compile each UNIQUE ast once, cache by pre-optimization ast id.
    // Uses global cache so identical kernels across prepare calls (e.g., sort substages
    // with same axis) skip both optimization and compilation. Bounded via FIFO eviction
    // to keep long-running processes from accumulating dead kernel entries indefinitely.
    static OPT_CACHE: std::sync::OnceLock<OptCacheState> = std::sync::OnceLock::new();
    let opt_state = OPT_CACHE.get_or_init(OptCacheState::new);
    let opt_cache = &opt_state.map;
    let opt_guard = opt_cache.guard();

    for item in &schedule_items {
        // COPY operations: buffer-to-buffer transfer (DISK→CPU, CPU→CUDA, etc.)
        // No compilation needed — register as PreparedOp for runtime execution.
        let runtime_ast = runtime_effect_ast(&item.ast);

        if matches!(runtime_ast.op(), Op::Copy { .. }) {
            let buffer_indices = resolve_item_buffer_indices(item, &uop_id_to_idx)?;
            builder.add_op_with_instance_dependencies(
                PreparedOp::BufferCopy(PreparedCopy {
                    id: item.kernel.id,
                    buffer_indices,
                    dependencies: item.dependencies.clone(),
                }),
                item.instance_dependencies.clone(),
            );
            continue;
        }

        // BUFFER_VIEW: zero-copy sub-buffer view (DISK weight views).
        // Creates a view into the base buffer at the specified byte offset.
        // Tinygrad schedule.py:201-204: buffers[buf_uops[0]] = base.view(size, dtype, offset)
        if let Op::BufferView { size, offset, .. } = runtime_ast.op() {
            let buffer_indices = resolve_item_buffer_indices(item, &uop_id_to_idx)?;

            if item.buffers.len() >= 2 && item.buffer_uop_ids.len() >= 2 && buffer_indices.len() >= 2 {
                let base = &item.buffers[1];
                let byte_offset = offset * base.dtype().bytes();
                let byte_size = size * runtime_ast.dtype().bytes();
                let view = base.view(byte_offset, byte_size).map_err(|e| crate::error::Error::IrConstruction {
                    details: format!(
                        "BUFFER_VIEW failed for kernel {}: base_buffer_id={}, byte_offset={}, byte_size={}: {e}",
                        item.kernel.id,
                        base.id().0,
                        byte_offset,
                        byte_size
                    ),
                })?;
                // Register the view under the output buffer's UOp ID so downstream
                // COPY/kernel items find it as their source buffer.
                let output_uop_id = item.buffer_uop_ids[0];
                if let Some(&idx) = uop_id_to_idx.get(&output_uop_id) {
                    builder.replace_buffer(idx, view);
                }

                builder.add_op_with_instance_dependencies(
                    PreparedOp::BufferView(PreparedBufferView {
                        id: item.kernel.id,
                        buffer_indices,
                        byte_offset,
                        byte_size,
                        dependencies: item.dependencies.clone(),
                    }),
                    item.instance_dependencies.clone(),
                );
            }
            continue;
        }

        // Explicit CUSTOM_FUNCTION runtime operations (Tinygrad ExecItem lowerers).
        // CALL bodies rooted at CUSTOM_FUNCTION are lowered directly to runtime
        // PreparedOp::CustomFunction with typed dispatch. Match against the
        // unwrapped runtime AST so END(CustomFunction) reaches this branch
        // consistently with Copy/BufferView above.
        if let Op::CustomFunction { kind, attrs } = runtime_ast.op() {
            let buffer_indices = resolve_item_buffer_indices(item, &uop_id_to_idx)?;
            let runtime_vars = attrs.iter().flat_map(morok_runtime::execution_plan::collect_runtime_vars).collect();
            builder.add_op_with_instance_dependencies(
                PreparedOp::CustomFunction(PreparedCustomFunction {
                    id: item.kernel.id,
                    kind: kind.clone(),
                    attrs: attrs.clone(),
                    buffer_indices,
                    fixedvars: item.fixedvars.clone(),
                    dependencies: item.dependencies.clone(),
                    runtime_vars,
                }),
                item.instance_dependencies.clone(),
            );
            continue;
        }

        let item_device_spec = item
            .buffers
            .iter()
            .map(|b| b.allocator().device_spec())
            .find(|spec| !spec.is_disk())
            .unwrap_or(DeviceSpec::Cpu);
        let item_device = config.resolve_device(&item_device_spec, alloc_registry)?;
        let item_codegen: &'static str = item_device.compiler.cache_key();

        let opt_key = (
            crate::schedule_cache::content_hash(&item.ast),
            item_device.device.clone(),
            item_codegen,
            optimizer_fingerprint,
        );

        let cached = if let Some(cached) = opt_cache.get(&opt_key, &opt_guard) {
            Arc::clone(cached)
        } else {
            let optimizer_renderer = get_optimizer_renderer(&item_device);
            let optimized_ast = if let morok_schedule::OptStrategy::Beam { .. } = config.optimizer.strategy {
                beam_search_optimize(
                    item.ast.clone(),
                    &optimizer_renderer,
                    &item_device,
                    &item.buffers,
                    &config.optimizer,
                )?
            } else {
                morok_schedule::optimize_kernel_with_config(item.ast.clone(), &optimizer_renderer, &config.optimizer)
            };

            let kernel_name =
                optimized_ast.metadata::<morok_schedule::optimizer::KernelInfo>().map(|info| info.function_name());

            let ast_decomposed = match item_device.renderer.decompositor() {
                Some(matcher) => morok_ir::decompositions::decompose_with(&optimized_ast, &matcher),
                None => optimized_ast,
            };
            let program =
                morok_codegen::program_pipeline::program_from_sink(ast_decomposed, item_device.device.clone());

            let result = morok_runtime::kernel_cache::get_or_compile_kernel(
                crate::schedule_cache::content_hash(&program),
                item_codegen,
                || {
                    let (spec, compiled) = compile_with_program_pipeline_components(
                        program.clone(),
                        item_device.renderer.as_ref(),
                        item_device.compiler.as_ref(),
                        kernel_name.as_deref(),
                    )?;
                    let program = (item_device.runtime)(&compiled).context(CreateProgramSnafu)?;
                    Ok(morok_runtime::kernel_cache::CachedKernel {
                        program,
                        device: item_codegen.to_string(),
                        code: spec.src.clone(),
                        entry_point: spec.name.clone(),
                        var_names: spec.var_names.clone(),
                        globals: spec.globals.clone(),
                        outs: spec.outs.clone(),
                        ins: spec.ins.clone(),
                        host_parallel_safe: matches!(item_device.device, DeviceSpec::Cpu),
                        global_size: spec.global_size.clone(),
                        local_size: spec.local_size.clone(),
                    })
                },
            )?;
            opt_state.insert(opt_key, Arc::clone(&result));
            result
        };

        // Build buffer indices in compiled ABI order (`ProgramSpec.globals`), not necessarily CALL arg order.
        let buffer_indices = resolve_compiled_kernel_buffer_indices(item, &uop_id_to_idx, &cached.globals)?;

        trace!(kernel.ast_id = item.ast.id, num_buffers = item.buffers.len(), "kernel buffer mapping");

        // Create PreparedKernel
        // Note: buffer_ptrs and buffer_ids will be computed in ExecutionPlanBuilder::build()
        // Convert fixedvars HashMap to vals Vec using var_names order from CachedKernel
        let vals: Vec<i64> =
            cached.var_names.iter().map(|name| item.fixedvars.get(name).copied().unwrap_or(0)).collect();
        let non_overridable_fixedvars = collect_non_overridable_fixedvars(item);

        let output_indices = output_indices_from_program_metadata(&cached.globals, &cached.outs, buffer_indices.len())
            .map_err(|e| crate::error::Error::IrConstruction {
                details: format!(
                    "invalid ProgramSpec output metadata for kernel id {} (globals={:?}, outs={:?}, num_buffers={}): {e}",
                    item.kernel.id,
                    cached.globals,
                    cached.outs,
                    buffer_indices.len()
                ),
            })?;

        let runtime_vars = morok_runtime::execution_plan::collect_runtime_vars(&item.ast);
        let prepared = PreparedKernel {
            id: item.kernel.id,
            ast: item.ast.clone(),
            kernel: cached,
            device: item_device.device.clone(),
            buffer_indices,
            output_indices,
            vals,
            fixedvars: non_overridable_fixedvars,
            dependencies: item.dependencies.clone(),
            buffer_ptrs: Vec::new(), // Computed in build()
            buffer_ids: Vec::new(),  // Computed in build()
            runtime_vars,
        };

        builder.add_op_with_instance_dependencies(
            PreparedOp::CompiledProgram(prepared),
            item.instance_dependencies.clone(),
        );
    }

    // Deterministic output identification via ScheduleResult.output_uop_ids
    let mut output_buffer_indices = Vec::with_capacity(schedule_result.output_uop_ids.len());
    for &uop_id in &schedule_result.output_uop_ids {
        let Some(idx) = uop_id_to_idx.get(&uop_id).copied() else {
            return Err(crate::error::Error::BufferNotFound { uop_id });
        };
        output_buffer_indices.push(idx);
    }
    if output_buffer_indices.is_empty() {
        return IrConstructionSnafu { details: "prepare_execution_plan produced no output buffer indices".to_string() }
            .fail();
    }
    builder.set_output_buffers(output_buffer_indices);

    builder.build().context(ExecutionSnafu)
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

fn collect_non_overridable_fixedvars(item: &ScheduleItem) -> HashMap<String, i64> {
    let Op::Call { args, .. } = item.kernel.op() else {
        return HashMap::new();
    };

    let mut locked = HashMap::new();
    for arg in args {
        let Op::Bind { var, value } = arg.op() else {
            continue;
        };
        let Op::DefineVar { name, .. } = var.op() else {
            continue;
        };
        let Op::Range { axis_type, .. } = value.op() else {
            continue;
        };
        if *axis_type != AxisType::Outer {
            continue;
        }
        if let Some(v) = item.fixedvars.get(name) {
            locked.insert(name.clone(), *v);
        }
    }
    locked
}

/// Render/compile entrypoint backed by PROGRAM pipeline stages.
fn compile_with_program_pipeline_components(
    kernel_ast: Arc<UOp>,
    renderer: &dyn morok_device::device::Renderer,
    compiler: &dyn morok_device::device::Compiler,
    kernel_name: Option<&str>,
) -> Result<(morok_device::device::ProgramSpec, morok_device::device::CompiledSpec)> {
    let mut program = match kernel_ast.op() {
        Op::Program { .. } => kernel_ast,
        other => {
            return IrConstructionSnafu {
                details: format!("compile_with_program_pipeline_components expects PROGRAM input, got {other:?}"),
            }
            .fail();
        }
    };

    program = morok_codegen::program_pipeline::get_program(
        &program,
        renderer,
        compiler,
        kernel_name,
        morok_codegen::program_pipeline::ProgramTarget::Source,
    )
    .context(RenderKernelSnafu)?;

    let rendered_entry = morok_device::device::ProgramSpec::from_uop(&program).map(|spec| spec.name).map_err(|e| {
        crate::error::Error::IrConstruction { details: format!("PROGRAM pipeline produced invalid SOURCE stage: {e}") }
    })?;

    let (program, compiled) =
        morok_codegen::program_pipeline::do_compile(&program, compiler).context(CompileKernelSnafu)?;

    let spec =
        morok_device::device::ProgramSpec::from_uop(&program).map_err(|e| crate::error::Error::IrConstruction {
            details: format!(
                "PROGRAM pipeline produced invalid ProgramSpec after compile (entry='{}'): {e}",
                rendered_entry
            ),
        })?;
    Ok((spec, compiled))
}

/// Resolve the device string for cache keying (includes compiler cache key).
pub(crate) fn resolve_codegen(param_buffers: &[(u64, Arc<UOp>)], config: &PrepareConfig) -> Result<&'static str> {
    let alloc_registry = morok_device::registry::registry();
    let spec = param_buffers
        .iter()
        .find_map(|(id, _)| {
            let spec = crate::tensor_registry::get_buffer(*id)?.allocator().device_spec();
            (!spec.is_disk()).then_some(spec)
        })
        .or_else(|| {
            param_buffers.iter().find_map(|(_, u)| {
                let Op::Buffer { device, .. } = u.op() else {
                    return None;
                };
                let Op::Device(spec) = device.op() else {
                    return None;
                };
                (!spec.is_disk()).then_some(spec.clone())
            })
        })
        .unwrap_or(DeviceSpec::Cpu);
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
    let dev_device = device.device.clone();
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
            let program = morok_codegen::program_pipeline::program_from_sink(decomposed, dev_device.clone());

            // Render and compile through PROGRAM stages (NOT timed).
            let (spec, compiled) = compile_with_program_pipeline_components(
                program,
                dev_renderer.as_ref(),
                dev_compiler.as_ref(),
                kernel_name.as_deref(),
            )
            .ok()?;
            let program = (dev_runtime)(&compiled).ok()?;

            // Extract buffer pointers inside the closure (avoids Sync issue)
            let buffer_ptrs: Vec<*mut u8> = buffers.iter().map(|b| unsafe { b.as_raw_ptr() }).collect();

            // Time ONLY execution (pass resolved global/local size for threaded kernels)
            // Note: Empty vals slice since benchmark kernels don't have symbolic variables
            let launch_dims = spec.launch_dims(&HashMap::new()).ok()?;
            let result = unsafe {
                morok_runtime::benchmark_kernel(
                    program.as_ref(),
                    &buffer_ptrs,
                    &[],
                    Some(launch_dims.global_size),
                    launch_dims.local_size,
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
#[path = "test/unit/realize_internal.rs"]
mod tests;
