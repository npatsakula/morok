//! Pre-compiled execution plan for kernel execution.
//!
//! `ExecutionPlan` separates one-time preparation (kernel compilation, buffer
//! allocation) from fast repeated execution.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │              PREPARATION (one-time)                      │
//! │  Schedule → instantiate → compile_kernels → build()     │
//! │                       ↓                                  │
//! │                ExecutionPlan                             │
//! └─────────────────────────────────────────────────────────┘
//!                         ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │              EXECUTION (fast path)                       │
//! │  dependency-ordered PreparedOp execution                 │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! let plan = tensor.prepare()?;
//! plan.execute()?;
//! let output = plan.output_buffer();
//! ```

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use morok_device::device::ProgramSpec;
use morok_device::{Buffer, BufferId};
use morok_dtype::DeviceSpec;
use morok_ir::{CustomFunctionKind, Op, UOp};
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::error::Result;
use crate::kernel_cache::CachedKernel;
use crate::profiler::KernelProfile;

type RuntimeLaunchSizes = (Option<[usize; 3]>, Option<[usize; 3]>);

// ============================================================================
// Core Structures
// ============================================================================

/// A pre-compiled kernel ready for execution.
///
/// Variable values are stored as positional `vals: Vec<i64>` rather than a named
/// HashMap, matching Tinygrad's `vals: tuple[int, ...]` parameter style.
#[derive(Clone)]
pub struct PreparedKernel {
    /// Unique identifier (from original AST).
    pub id: u64,

    pub ast: Arc<UOp>,

    /// Compiled kernel program (Arc-shared from cache).
    pub kernel: Arc<CachedKernel>,

    /// Device this kernel executes on.
    pub device: DeviceSpec,

    /// Indices into `ExecutionPlan::buffers` for this kernel's buffers.
    /// Ordered as expected by the kernel (matches codegen buffer order).
    pub buffer_indices: Vec<usize>,

    /// Indices of output buffers within `buffer_indices`.
    pub output_indices: Vec<usize>,

    /// Variable values in positional order (matches `var_names` in CachedKernel).
    pub vals: Vec<i64>,

    /// Fixed variable bindings captured at prepare time.
    ///
    /// These mirror Tinygrad's `fixedvars` semantics: values fixed by scheduling
    /// (for example from bound ranges) are not overridden by `execute_with_vars`.
    pub fixedvars: HashMap<String, i64>,

    /// Kernel IDs that must complete before this one (dependencies).
    pub dependencies: Vec<u64>,

    /// Pre-computed raw buffer addresses for low-allocation execution.
    /// Computed once during prepare(), stable for the lifetime of ExecutionPlan.
    /// SAFETY: Pointers are valid as long as ExecutionPlan owns the buffers.
    pub buffer_ptrs: Vec<usize>,

    /// Pre-computed buffer IDs for dependency tracking.
    pub buffer_ids: Vec<BufferId>,

    /// Cached `(name, min_val, max_val)` triples for every `DefineVar` reachable
    /// from `ast`. Populated at construction so `validate_runtime_var_bounds`
    /// doesn't re-toposort on every execute call.
    pub runtime_vars: Vec<RuntimeVar>,
}

/// Bound description for one `DefineVar` consumed by a kernel.
#[derive(Clone, Debug)]
pub struct RuntimeVar {
    pub name: String,
    pub min_val: i64,
    pub max_val: i64,
}

/// Walk `root` and collect bounds for every reachable `DefineVar`.
pub fn collect_runtime_vars(root: &Arc<UOp>) -> Vec<RuntimeVar> {
    let mut vars = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for node in root.toposort() {
        if let Op::DefineVar { name, min_val, max_val } = node.op()
            && seen.insert(name.clone())
        {
            vars.push(RuntimeVar { name: name.clone(), min_val: *min_val, max_val: *max_val });
        }
    }
    vars
}

/// Prepared buffer-to-buffer copy operation.
#[derive(Clone, Debug)]
pub struct PreparedCopy {
    /// Unique operation identifier.
    pub id: u64,

    /// Buffer indices in ExecutionPlan order: [dst, src].
    pub buffer_indices: Vec<usize>,

    /// Operation IDs that must complete before this copy.
    pub dependencies: Vec<u64>,
}

/// Prepared zero-copy buffer view operation.
#[derive(Clone, Debug)]
pub struct PreparedBufferView {
    /// Unique operation identifier.
    pub id: u64,

    /// Output and base buffer indices in ExecutionPlan order.
    /// `buffer_indices[0]` is output view, `buffer_indices[1]` is base source.
    pub buffer_indices: Vec<usize>,

    /// Expected byte offset into base for the view.
    pub byte_offset: usize,

    /// Expected byte size of the view.
    pub byte_size: usize,

    /// Operation IDs that must complete before this view is consumed.
    pub dependencies: Vec<u64>,
}

/// Prepared custom runtime function operation.
#[derive(Clone, Debug)]
pub struct PreparedCustomFunction {
    /// Unique operation identifier.
    pub id: u64,

    /// Explicit custom function kind (for example: `EncDec`).
    pub kind: CustomFunctionKind,

    /// Runtime descriptor attributes encoded by the IR body.
    pub attrs: SmallVec<[Arc<UOp>; 4]>,

    /// Buffer indices in ExecutionPlan order.
    pub buffer_indices: Vec<usize>,

    /// Bound variable values for this operation.
    pub fixedvars: HashMap<String, i64>,

    /// Operation IDs that must complete before this custom function runs.
    pub dependencies: Vec<u64>,

    /// Cached `(name, min_val, max_val)` triples for every `DefineVar`
    /// reachable from `attrs`. Populated at construction so
    /// `validate_runtime_var_bounds` doesn't re-toposort on every execute call.
    pub runtime_vars: Vec<RuntimeVar>,
}

/// Prepared execution item.
#[derive(Clone, Debug)]
pub enum PreparedOp {
    /// Compiled kernel/program operation.
    CompiledProgram(PreparedKernel),

    /// Direct buffer copy operation.
    BufferCopy(PreparedCopy),

    /// Zero-copy view aliasing operation.
    BufferView(PreparedBufferView),

    /// Runtime custom function operation.
    CustomFunction(PreparedCustomFunction),
}

fn op_identity(op: &PreparedOp) -> (u64, Vec<u64>) {
    match op {
        PreparedOp::CompiledProgram(kernel) => (kernel.id, kernel.dependencies.clone()),
        PreparedOp::BufferCopy(copy) => (copy.id, copy.dependencies.clone()),
        PreparedOp::BufferView(view) => (view.id, view.dependencies.clone()),
        PreparedOp::CustomFunction(custom) => (custom.id, custom.dependencies.clone()),
    }
}

fn validate_var_bound(name: &str, value: i64, min_val: i64, max_val: i64) -> Result<()> {
    if value < min_val || value > max_val {
        return Err(crate::error::Error::Execution {
            reason: format!("variable {name}={value} is outside bounds [{min_val}, {max_val}]"),
        });
    }
    Ok(())
}

struct DependencyGraph {
    op_ids: Vec<u64>,
    in_degree: Vec<usize>,
    successors: Vec<Vec<usize>>,
}

fn build_dependency_graph(ops: &[PreparedOp], instance_deps_per_op: Option<&[Vec<usize>]>) -> Result<DependencyGraph> {
    if let Some(instance_deps) = instance_deps_per_op
        && instance_deps.len() != ops.len()
    {
        return Err(crate::error::Error::Execution {
            reason: format!(
                "prepared op instance dependency table length mismatch: ops={}, instance_deps={}",
                ops.len(),
                instance_deps.len()
            ),
        });
    }

    let mut op_ids = Vec::with_capacity(ops.len());
    let mut deps_per_op = Vec::with_capacity(ops.len());
    let mut id_counts: HashMap<u64, usize> = HashMap::with_capacity(ops.len());

    for op in ops {
        let (op_id, deps) = op_identity(op);
        op_ids.push(op_id);
        deps_per_op.push(deps);
        *id_counts.entry(op_id).or_insert(0) += 1;
    }

    let has_duplicate_ids = id_counts.values().any(|&count| count > 1);

    let mut in_degree = vec![0usize; ops.len()];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); ops.len()];

    if !has_duplicate_ids {
        let mut id_to_idx: HashMap<u64, usize> = HashMap::with_capacity(ops.len());
        for (idx, &op_id) in op_ids.iter().enumerate() {
            id_to_idx.insert(op_id, idx);
        }

        for (idx, deps) in deps_per_op.iter().enumerate() {
            for dep in deps {
                let Some(&dep_idx) = id_to_idx.get(dep) else {
                    return Err(crate::error::Error::Execution {
                        reason: format!("prepared op {} depends on unknown op id {}", op_ids[idx], dep),
                    });
                };
                in_degree[idx] += 1;
                successors[dep_idx].push(idx);
            }
        }
    } else {
        // Expanded schedules may contain repeated op IDs for per-iteration items.
        // Resolve dependencies against the most recent prior op with that ID.
        let mut last_seen: HashMap<u64, usize> = HashMap::with_capacity(ops.len());

        for (idx, deps) in deps_per_op.iter().enumerate() {
            for dep in deps {
                let Some(&dep_idx) = last_seen.get(dep) else {
                    return Err(crate::error::Error::Execution {
                        reason: format!(
                            "prepared op {} depends on unknown prior op id {} (duplicate-id schedule mode)",
                            op_ids[idx], dep
                        ),
                    });
                };
                in_degree[idx] += 1;
                successors[dep_idx].push(idx);
            }

            last_seen.insert(op_ids[idx], idx);
        }
    }

    if let Some(instance_deps_per_op) = instance_deps_per_op {
        for (idx, instance_deps) in instance_deps_per_op.iter().enumerate() {
            for &dep_idx in instance_deps {
                if dep_idx >= ops.len() {
                    return Err(crate::error::Error::Execution {
                        reason: format!("prepared op {} depends on unknown op index {}", op_ids[idx], dep_idx),
                    });
                }
                if dep_idx == idx {
                    return Err(crate::error::Error::Execution {
                        reason: format!("prepared op {} cannot depend on itself by op index {}", op_ids[idx], dep_idx),
                    });
                }
                in_degree[idx] += 1;
                successors[dep_idx].push(idx);
            }
        }
    }

    Ok(DependencyGraph { op_ids, in_degree, successors })
}

#[cfg(test)]
fn compute_mixed_op_order(ops: &[PreparedOp]) -> Result<Vec<usize>> {
    compute_mixed_op_order_with_instance_dependencies(ops, &[])
}

fn compute_mixed_op_order_with_instance_dependencies(
    ops: &[PreparedOp],
    instance_deps_per_op: &[Vec<usize>],
) -> Result<Vec<usize>> {
    let instance_deps = (!instance_deps_per_op.is_empty()).then_some(instance_deps_per_op);
    let DependencyGraph { op_ids, mut in_degree, successors } = build_dependency_graph(ops, instance_deps)?;

    let mut ready: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    for (idx, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            ready.push(Reverse(idx));
        }
    }

    let mut order = Vec::with_capacity(ops.len());
    while let Some(Reverse(idx)) = ready.pop() {
        order.push(idx);
        for &succ in &successors[idx] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                ready.push(Reverse(succ));
            }
        }
    }

    if order.len() != ops.len() {
        let blocked: Vec<u64> = in_degree
            .iter()
            .enumerate()
            .filter_map(|(idx, &deg)| if deg > 0 { Some(op_ids[idx]) } else { None })
            .collect();
        return Err(crate::error::Error::Execution {
            reason: format!("cycle detected in prepared op dependencies: blocked_ids={blocked:?}"),
        });
    }

    Ok(order)
}

#[cfg(test)]
fn compute_execution_levels(ops: &[PreparedOp]) -> Result<Vec<Vec<usize>>> {
    compute_execution_levels_with_instance_dependencies(ops, &[])
}

fn compute_execution_levels_with_instance_dependencies(
    ops: &[PreparedOp],
    instance_deps_per_op: &[Vec<usize>],
) -> Result<Vec<Vec<usize>>> {
    let instance_deps = (!instance_deps_per_op.is_empty()).then_some(instance_deps_per_op);
    let DependencyGraph { op_ids, mut in_degree, successors } = build_dependency_graph(ops, instance_deps)?;

    let mut ready: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    for (idx, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            ready.push(Reverse(idx));
        }
    }

    let mut levels: Vec<Vec<usize>> = Vec::new();
    let mut visited = 0usize;

    while !ready.is_empty() {
        let mut level: Vec<usize> = Vec::new();
        while let Some(Reverse(idx)) = ready.pop() {
            level.push(idx);
        }

        let mut next_ready: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
        for &idx in &level {
            visited += 1;
            for &succ in &successors[idx] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    next_ready.push(Reverse(succ));
                }
            }
        }

        levels.push(level);
        ready = next_ready;
    }

    if visited != ops.len() {
        let blocked: Vec<u64> = in_degree
            .iter()
            .enumerate()
            .filter_map(|(idx, &deg)| if deg > 0 { Some(op_ids[idx]) } else { None })
            .collect();
        return Err(crate::error::Error::Execution {
            reason: format!("cycle detected in prepared op dependencies: blocked_ids={blocked:?}"),
        });
    }

    Ok(levels)
}

/// Pre-compiled execution plan for a computation graph.
///
/// Created once via `prepare()`, then executed multiple times.
/// The plan owns all its buffers and compiled kernels.
pub struct ExecutionPlan {
    /// Prepared operations in schedule order.
    ops: Vec<PreparedOp>,

    /// Concrete op-index dependencies parallel to `ops`.
    op_instance_dependencies: Vec<Vec<usize>>,

    /// Precomputed dependency-safe operation order.
    op_order: Vec<usize>,

    /// Topological levels of dependency-independent operations.
    op_levels: Vec<Vec<usize>>,

    /// ALL buffers owned by this plan (inputs, intermediates, outputs).
    buffers: Vec<Buffer>,

    /// Mapping: AST id → buffer index (for kernel buffer binding).
    ast_to_buffer: HashMap<u64, usize>,

    /// Indices of output buffers in `buffers` (matches SINK source order).
    output_buffer_indices: Vec<usize>,

    /// Primary device for this plan.
    device: DeviceSpec,

    /// Last dynamic variable bindings supplied through `execute_with_vars`.
    runtime_var_vals: HashMap<String, i64>,

    /// Additional UOp IDs registered as aliases that need cleanup.
    alias_ids: Vec<u64>,
}

// ============================================================================
// ExecutionPlan Implementation
// ============================================================================

impl ExecutionPlan {
    fn kernel_launch_sizes(kernel: &PreparedKernel) -> Result<RuntimeLaunchSizes> {
        let mut vars: HashMap<&str, i64> =
            HashMap::with_capacity(kernel.kernel.var_names.len() + kernel.fixedvars.len());
        for (idx, name) in kernel.kernel.var_names.iter().enumerate() {
            let value = kernel.vals.get(idx).copied().ok_or_else(|| crate::error::Error::Execution {
                reason: format!(
                    "Kernel {} has {} var names but only {} values",
                    kernel.id,
                    kernel.kernel.var_names.len(),
                    kernel.vals.len()
                ),
            })?;
            vars.insert(name.as_str(), value);
        }
        for (name, value) in &kernel.fixedvars {
            vars.insert(name.as_str(), *value);
        }

        let dims =
            ProgramSpec::resolve_launch_dims(&kernel.kernel.global_size, kernel.kernel.local_size.as_ref(), &vars)
                .map_err(|e| crate::error::Error::Execution {
                    reason: format!("Kernel {} launch dimensions failed: {e}", kernel.id),
                })?;
        Ok((Some(dims.global_size), dims.local_size))
    }

    fn kernel_uses_cpu_threading(kernel: &PreparedKernel) -> Result<bool> {
        if !matches!(kernel.device, DeviceSpec::Cpu) {
            return Ok(false);
        }
        let (global_size, _) = Self::kernel_launch_sizes(kernel)?;
        Ok(global_size.map(|[x, _, _]| x > 1).unwrap_or(false))
    }

    #[inline]
    fn execute_kernel(kernel: &PreparedKernel) -> Result<()> {
        let buffer_ptrs: SmallVec<[*mut u8; 8]> = kernel.buffer_ptrs.iter().map(|&ptr| ptr as *mut u8).collect();
        let (global_size, local_size) = Self::kernel_launch_sizes(kernel)?;
        unsafe {
            kernel
                .kernel
                .program
                .execute(&buffer_ptrs, &kernel.vals, global_size, local_size)
                .map_err(|e| crate::error::Error::Execution { reason: format!("Kernel {} failed: {}", kernel.id, e) })
        }
    }

    fn validate_runtime_var_bounds(&self, var_vals: &[(&str, i64)]) -> Result<()> {
        let vals_map: HashMap<&str, i64> = var_vals.iter().copied().collect();
        for op in &self.ops {
            match op {
                PreparedOp::CompiledProgram(kernel) => {
                    for var in &kernel.runtime_vars {
                        if kernel.fixedvars.contains_key(&var.name) || var.name == "core_id" {
                            continue;
                        }
                        if let Some(&value) = vals_map.get(var.name.as_str()) {
                            validate_var_bound(&var.name, value, var.min_val, var.max_val)?;
                        }
                    }
                }
                PreparedOp::CustomFunction(custom) => {
                    for var in &custom.runtime_vars {
                        if custom.fixedvars.contains_key(&var.name) || var.name == "core_id" {
                            continue;
                        }
                        if let Some(&value) = vals_map.get(var.name.as_str()) {
                            validate_var_bound(&var.name, value, var.min_val, var.max_val)?;
                        }
                    }
                }
                PreparedOp::BufferCopy(_) | PreparedOp::BufferView(_) => {}
            }
        }
        Ok(())
    }

    fn update_runtime_var_vals(&mut self, var_vals: &[(&str, i64)]) -> Result<()> {
        self.validate_runtime_var_bounds(var_vals)?;

        let vals_map: HashMap<&str, i64> = var_vals.iter().copied().collect();
        for &(name, value) in var_vals {
            if name == "core_id" {
                continue;
            }
            self.runtime_var_vals.insert(name.to_string(), value);
        }
        for op in &mut self.ops {
            if let PreparedOp::CompiledProgram(kernel) = op {
                for (idx, name) in kernel.kernel.var_names.iter().enumerate() {
                    if kernel.fixedvars.contains_key(name) || name == "core_id" {
                        continue;
                    }
                    if let Some(&v) = vals_map.get(name.as_str()) {
                        let Some(slot) = kernel.vals.get_mut(idx) else {
                            return Err(crate::error::Error::Execution {
                                reason: format!(
                                    "Kernel {} has {} var names but only {} values",
                                    kernel.id,
                                    kernel.kernel.var_names.len(),
                                    kernel.vals.len()
                                ),
                            });
                        };
                        *slot = v;
                    }
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn execute_copy(&self, copy: &PreparedCopy) -> Result<()> {
        if copy.buffer_indices.len() < 2 {
            return Err(crate::error::Error::Execution {
                reason: format!(
                    "Copy op {} requires at least two buffer indices (dst, src), got {}",
                    copy.id,
                    copy.buffer_indices.len()
                ),
            });
        }
        let dst_idx = copy.buffer_indices[0];
        let src_idx = copy.buffer_indices[1];

        if dst_idx >= self.buffers.len() || src_idx >= self.buffers.len() {
            return Err(crate::error::Error::Execution {
                reason: format!(
                    "Copy op {} buffer index out of range: dst={}, src={}, total_buffers={}",
                    copy.id,
                    dst_idx,
                    src_idx,
                    self.buffers.len()
                ),
            });
        }

        let mut dst = self.buffers[dst_idx].clone();
        let src = &self.buffers[src_idx];
        dst.copy_from(src)
            .map_err(|e| crate::error::Error::Execution { reason: format!("Copy op {} failed: {}", copy.id, e) })
    }

    #[inline]
    fn execute_buffer_view(&self, view: &PreparedBufferView) -> Result<()> {
        if view.buffer_indices.len() < 2 {
            return Err(crate::error::Error::Execution {
                reason: format!(
                    "BufferView op {} requires at least two buffer indices (out, base), got {}",
                    view.id,
                    view.buffer_indices.len()
                ),
            });
        }
        let out_idx = view.buffer_indices[0];
        let base_idx = view.buffer_indices[1];

        if out_idx >= self.buffers.len() || base_idx >= self.buffers.len() {
            return Err(crate::error::Error::Execution {
                reason: format!(
                    "BufferView op {} buffer index out of range: out={}, base={}, total_buffers={}",
                    view.id,
                    out_idx,
                    base_idx,
                    self.buffers.len()
                ),
            });
        }

        let out = &self.buffers[out_idx];
        let base = &self.buffers[base_idx];
        let expected_offset = base.offset() + view.byte_offset;

        if out.storage_id() != base.storage_id() || out.offset() != expected_offset || out.size() != view.byte_size {
            return Err(crate::error::Error::Execution {
                reason: format!(
                    "BufferView op {} mismatch: out(storage={:?},off={},size={}) base(storage={:?},off={}) expected(off={},size={})",
                    view.id,
                    out.storage_id(),
                    out.offset(),
                    out.size(),
                    base.storage_id(),
                    base.offset(),
                    expected_offset,
                    view.byte_size,
                ),
            });
        }
        Ok(())
    }

    #[inline]
    fn execute_custom_function(&self, custom: &PreparedCustomFunction) -> Result<()> {
        let mut buffers = Vec::with_capacity(custom.buffer_indices.len());
        for &idx in &custom.buffer_indices {
            let Some(buffer) = self.buffers.get(idx) else {
                return Err(crate::error::Error::Execution {
                    reason: format!(
                        "Custom function op {} ({:?}) buffer index out of range: idx={}, total_buffers={}",
                        custom.id,
                        custom.kind,
                        idx,
                        self.buffers.len()
                    ),
                });
            };
            buffers.push(buffer.clone());
        }

        let mut vars = self.runtime_var_vals.clone();
        vars.extend(custom.fixedvars.iter().map(|(k, v)| (k.clone(), *v)));

        crate::custom_function::run_custom_function(&custom.kind, &custom.attrs, &mut buffers, &vars).map_err(|e| {
            // Pass typed `Unsupported` errors through unchanged so callers can match on `kind`.
            // Other errors are wrapped with op context for debugging.
            match e {
                crate::error::Error::Unsupported { .. } => e,
                other => crate::error::Error::Execution {
                    reason: format!("Custom function op {} ({:?}) failed: {other}", custom.id, custom.kind),
                },
            }
        })
    }

    #[inline]
    fn execute_op(&self, op: &PreparedOp) -> Result<()> {
        match op {
            PreparedOp::CompiledProgram(kernel) => Self::execute_kernel(kernel),
            PreparedOp::BufferCopy(copy) => self.execute_copy(copy),
            PreparedOp::BufferView(view) => self.execute_buffer_view(view),
            PreparedOp::CustomFunction(custom) => self.execute_custom_function(custom),
        }
    }

    #[inline]
    fn op_requires_serial(op: &PreparedOp) -> bool {
        match op {
            PreparedOp::CompiledProgram(kernel) => !kernel.kernel.host_parallel_safe,
            PreparedOp::BufferCopy(_) | PreparedOp::BufferView(_) | PreparedOp::CustomFunction(_) => true,
        }
    }

    #[inline]
    fn compiled_kernel_at(&self, idx: usize) -> Option<&PreparedKernel> {
        match &self.ops[idx] {
            PreparedOp::CompiledProgram(kernel) => Some(kernel),
            _ => None,
        }
    }

    fn kernels_conflict(lhs: &PreparedKernel, rhs: &PreparedKernel) -> bool {
        let lhs_outputs: HashSet<BufferId> =
            lhs.output_indices.iter().filter_map(|&out_idx| lhs.buffer_ids.get(out_idx).copied()).collect();
        let rhs_outputs: HashSet<BufferId> =
            rhs.output_indices.iter().filter_map(|&out_idx| rhs.buffer_ids.get(out_idx).copied()).collect();

        if !lhs_outputs.is_disjoint(&rhs_outputs) {
            return true;
        }

        let lhs_reads: HashSet<BufferId> = lhs
            .buffer_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &buf)| (!lhs.output_indices.contains(&idx)).then_some(buf))
            .collect();
        let rhs_reads: HashSet<BufferId> = rhs
            .buffer_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &buf)| (!rhs.output_indices.contains(&idx)).then_some(buf))
            .collect();

        !lhs_outputs.is_disjoint(&rhs_reads) || !rhs_outputs.is_disjoint(&lhs_reads)
    }

    fn partition_parallel_safe_group(&self, indices: &[usize]) -> Result<Vec<Vec<usize>>> {
        let mut groups: Vec<Vec<usize>> = Vec::new();

        for &idx in indices {
            let Some(kernel) = self.compiled_kernel_at(idx) else {
                return Err(crate::error::Error::Execution {
                    reason: format!("parallel partition expected compiled kernel at op index {idx}"),
                });
            };

            let mut placed = false;
            for group in &mut groups {
                let has_conflict = group.iter().any(|&existing_idx| {
                    self.compiled_kernel_at(existing_idx)
                        .map(|existing| Self::kernels_conflict(existing, kernel))
                        .unwrap_or(true)
                });
                if !has_conflict {
                    group.push(idx);
                    placed = true;
                    break;
                }
            }

            if !placed {
                groups.push(vec![idx]);
            }
        }

        Ok(groups)
    }

    fn execute_parallel_group(&self, indices: &[usize]) -> Result<()> {
        if indices.len() <= 1 {
            if let Some(&idx) = indices.first() {
                self.execute_op(&self.ops[idx])?;
            }
            return Ok(());
        }

        let has_threaded_cpu_kernel = indices.iter().try_fold(false, |acc, &idx| {
            let Some(kernel) = self.compiled_kernel_at(idx) else {
                return Err(crate::error::Error::Execution {
                    reason: format!("parallel execution expected compiled kernel at op index {idx}"),
                });
            };
            Ok(acc || Self::kernel_uses_cpu_threading(kernel)?)
        })?;

        if has_threaded_cpu_kernel {
            for &idx in indices {
                let Some(kernel) = self.compiled_kernel_at(idx) else {
                    return Err(crate::error::Error::Execution {
                        reason: format!("parallel execution expected compiled kernel at op index {idx}"),
                    });
                };
                Self::execute_kernel(kernel)?;
            }
            return Ok(());
        }

        indices
            .par_iter()
            .map(|&idx| {
                let Some(kernel) = self.compiled_kernel_at(idx) else {
                    return Err(crate::error::Error::Execution {
                        reason: format!("parallel execution expected compiled kernel at op index {idx}"),
                    });
                };
                Self::execute_kernel(kernel)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(())
    }

    fn execute_parallel_group_profiled(&self, indices: &[usize]) -> Result<Vec<(usize, KernelProfile)>> {
        if indices.len() <= 1 {
            let mut profiles = Vec::new();
            if let Some(&idx) = indices.first() {
                let Some(kernel) = self.compiled_kernel_at(idx) else {
                    return Err(crate::error::Error::Execution {
                        reason: format!("profiled execution expected compiled kernel at op index {idx}"),
                    });
                };
                let start = Instant::now();
                Self::execute_kernel(kernel)?;
                profiles.push((
                    idx,
                    KernelProfile {
                        kernel: Arc::clone(&kernel.kernel),
                        device: kernel.device.clone(),
                        num_buffers: kernel.buffer_ptrs.len(),
                        elapsed: start.elapsed(),
                    },
                ));
            }
            return Ok(profiles);
        }

        let has_threaded_cpu_kernel = indices.iter().try_fold(false, |acc, &idx| {
            let Some(kernel) = self.compiled_kernel_at(idx) else {
                return Err(crate::error::Error::Execution {
                    reason: format!("profiled execution expected compiled kernel at op index {idx}"),
                });
            };
            Ok(acc || Self::kernel_uses_cpu_threading(kernel)?)
        })?;

        if has_threaded_cpu_kernel {
            let mut profiles = Vec::with_capacity(indices.len());
            for &idx in indices {
                let Some(kernel) = self.compiled_kernel_at(idx) else {
                    return Err(crate::error::Error::Execution {
                        reason: format!("profiled execution expected compiled kernel at op index {idx}"),
                    });
                };
                let start = Instant::now();
                Self::execute_kernel(kernel)?;
                profiles.push((
                    idx,
                    KernelProfile {
                        kernel: Arc::clone(&kernel.kernel),
                        device: kernel.device.clone(),
                        num_buffers: kernel.buffer_ptrs.len(),
                        elapsed: start.elapsed(),
                    },
                ));
            }
            return Ok(profiles);
        }

        let mut profiles = indices
            .par_iter()
            .map(|&idx| {
                let Some(kernel) = self.compiled_kernel_at(idx) else {
                    return Err(crate::error::Error::Execution {
                        reason: format!("profiled execution expected compiled kernel at op index {idx}"),
                    });
                };
                let start = Instant::now();
                Self::execute_kernel(kernel)?;
                Ok((
                    idx,
                    KernelProfile {
                        kernel: Arc::clone(&kernel.kernel),
                        device: kernel.device.clone(),
                        num_buffers: kernel.buffer_ptrs.len(),
                        elapsed: start.elapsed(),
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        profiles.sort_by_key(|(idx, _)| *idx);
        Ok(profiles)
    }

    /// Get the first (or only) output buffer after execution.
    ///
    /// Returns `None` for plans with no output buffers (for example, plans
    /// constructed before `set_output_buffer*` is called).
    pub fn output_buffer(&self) -> Option<&Buffer> {
        self.output_buffer_indices.first().and_then(|&i| self.buffers.get(i))
    }

    /// Get output buffer by position (matches SINK source order for batch).
    ///
    /// Returns `None` if `position` is out of range.
    pub fn output_buffer_at(&self, position: usize) -> Option<&Buffer> {
        self.output_buffer_indices.get(position).and_then(|&i| self.buffers.get(i))
    }

    /// Get all output buffers.
    pub fn output_buffers(&self) -> Vec<&Buffer> {
        self.output_buffer_indices.iter().map(|&i| &self.buffers[i]).collect()
    }

    /// Number of outputs in this plan.
    pub fn num_outputs(&self) -> usize {
        self.output_buffer_indices.len()
    }

    /// Get a buffer by AST id (for reading intermediate results).
    pub fn buffer(&self, ast_id: u64) -> Option<&Buffer> {
        self.ast_to_buffer.get(&ast_id).map(|&idx| &self.buffers[idx])
    }

    /// Get a mutable buffer by AST id (for `copyin()` on input buffers).
    pub fn buffer_mut_by_id(&mut self, ast_id: u64) -> Option<&mut Buffer> {
        self.ast_to_buffer.get(&ast_id).copied().map(|idx| &mut self.buffers[idx])
    }

    /// Get the primary device for this plan.
    pub fn device(&self) -> &DeviceSpec {
        &self.device
    }

    /// Get all buffers owned by this plan.
    pub fn buffers(&self) -> &[Buffer] {
        &self.buffers
    }

    /// Get mutable access to all buffers owned by this plan.
    pub fn buffers_mut(&mut self) -> &mut [Buffer] {
        &mut self.buffers
    }

    /// Get a mutable buffer by its index in the buffers array.
    pub fn buffer_at_mut(&mut self, index: usize) -> Option<&mut Buffer> {
        self.buffers.get_mut(index)
    }

    /// Get all prepared kernels.
    pub fn prepared_kernels(&self) -> Vec<&PreparedKernel> {
        self.ops
            .iter()
            .filter_map(|op| match op {
                PreparedOp::CompiledProgram(kernel) => Some(kernel),
                _ => None,
            })
            .collect()
    }

    /// Get all prepared operations in schedule order.
    pub fn prepared_ops(&self) -> &[PreparedOp] {
        &self.ops
    }

    /// Iterate over compiled kernels (for inspecting generated source code).
    pub fn kernels(&self) -> impl Iterator<Item = &CachedKernel> {
        self.ops.iter().filter_map(|op| match op {
            PreparedOp::CompiledProgram(kernel) => Some(kernel.kernel.as_ref()),
            _ => None,
        })
    }

    /// Execute the plan.
    ///
    /// Uses dependency-aware operation ordering for all prepared op types.
    pub fn execute(&self) -> Result<()> {
        for level in &self.op_levels {
            let mut pending_parallel: Vec<usize> = Vec::new();

            for &idx in level {
                let op = &self.ops[idx];
                if Self::op_requires_serial(op) {
                    if !pending_parallel.is_empty() {
                        let groups = self.partition_parallel_safe_group(&pending_parallel)?;
                        for group in groups {
                            self.execute_parallel_group(&group)?;
                        }
                        pending_parallel.clear();
                    }
                    self.execute_op(op)?;
                } else {
                    pending_parallel.push(idx);
                }
            }

            if !pending_parallel.is_empty() {
                let groups = self.partition_parallel_safe_group(&pending_parallel)?;
                for group in groups {
                    self.execute_parallel_group(&group)?;
                }
            }
        }
        Ok(())
    }

    /// Execute the plan with per-kernel timing.
    ///
    /// Returns a [`KernelProfile`] for each kernel in execution order.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let plan = tensor.prepare()?;
    /// let profiles = plan.execute_profiled()?;
    ///
    /// // Sort by time descending
    /// let mut sorted = profiles;
    /// sorted.sort_by(|a, b| b.elapsed.cmp(&a.elapsed));
    /// for p in &sorted[..10.min(sorted.len())] {
    ///     println!("{:>8.3}ms  {}", p.elapsed.as_secs_f64() * 1000.0, p.kernel.entry_point);
    /// }
    /// ```
    pub fn execute_profiled(&self) -> Result<Vec<KernelProfile>> {
        let mut profiles = Vec::new();
        for level in &self.op_levels {
            let mut pending_parallel: Vec<usize> = Vec::new();

            for &idx in level {
                match &self.ops[idx] {
                    PreparedOp::CompiledProgram(kernel) if kernel.kernel.host_parallel_safe => {
                        pending_parallel.push(idx);
                    }
                    PreparedOp::CompiledProgram(kernel) => {
                        if !pending_parallel.is_empty() {
                            let groups = self.partition_parallel_safe_group(&pending_parallel)?;
                            for group in groups {
                                let mut prof = self.execute_parallel_group_profiled(&group)?;
                                profiles.extend(prof.drain(..).map(|(_, p)| p));
                            }
                            pending_parallel.clear();
                        }

                        let start = Instant::now();
                        Self::execute_kernel(kernel)?;
                        profiles.push(KernelProfile {
                            kernel: Arc::clone(&kernel.kernel),
                            device: kernel.device.clone(),
                            num_buffers: kernel.buffer_ptrs.len(),
                            elapsed: start.elapsed(),
                        });
                    }
                    PreparedOp::BufferCopy(copy) => {
                        if !pending_parallel.is_empty() {
                            let groups = self.partition_parallel_safe_group(&pending_parallel)?;
                            for group in groups {
                                let mut prof = self.execute_parallel_group_profiled(&group)?;
                                profiles.extend(prof.drain(..).map(|(_, p)| p));
                            }
                            pending_parallel.clear();
                        }
                        self.execute_copy(copy)?;
                    }
                    PreparedOp::BufferView(view) => {
                        if !pending_parallel.is_empty() {
                            let groups = self.partition_parallel_safe_group(&pending_parallel)?;
                            for group in groups {
                                let mut prof = self.execute_parallel_group_profiled(&group)?;
                                profiles.extend(prof.drain(..).map(|(_, p)| p));
                            }
                            pending_parallel.clear();
                        }
                        self.execute_buffer_view(view)?;
                    }
                    PreparedOp::CustomFunction(custom) => {
                        if !pending_parallel.is_empty() {
                            let groups = self.partition_parallel_safe_group(&pending_parallel)?;
                            for group in groups {
                                let mut prof = self.execute_parallel_group_profiled(&group)?;
                                profiles.extend(prof.drain(..).map(|(_, p)| p));
                            }
                            pending_parallel.clear();
                        }
                        self.execute_custom_function(custom)?;
                    }
                }
            }

            if !pending_parallel.is_empty() {
                let groups = self.partition_parallel_safe_group(&pending_parallel)?;
                for group in groups {
                    let mut prof = self.execute_parallel_group_profiled(&group)?;
                    profiles.extend(prof.drain(..).map(|(_, p)| p));
                }
            }
        }
        Ok(profiles)
    }

    /// Re-execute the plan with different variable bindings.
    ///
    /// The kernel code is NOT recompiled; only the `vals` passed to each kernel
    /// are updated. Buffers must be allocated to max variable values (which is
    /// the default when using `Variable::bind()`).
    ///
    /// # Safety contract
    ///
    /// Variable values **must** fall within `[min_val, max_val]` bounds defined
    /// at `Variable::new()` time. Exceeding `max_val` causes out-of-bounds buffer
    /// access (buffers are allocated to `max_val`). Use `Variable::bind()` to
    /// validate bounds before calling this method.
    ///
    /// Variables not present in `var_vals` keep their existing values from
    /// `prepare()` (or the previous `execute_with_vars` call). Internal
    /// variables like `core_id` are left untouched.
    pub fn execute_with_vars(&mut self, var_vals: &[(&str, i64)]) -> Result<()> {
        self.update_runtime_var_vals(var_vals)?;
        self.execute()
    }

    /// Re-execute the plan with different variable bindings and per-kernel timing.
    ///
    /// Updates kernel `vals` the same way as [`Self::execute_with_vars`] and then
    /// executes via [`Self::execute_profiled`].
    pub fn execute_with_vars_profiled(&mut self, var_vals: &[(&str, i64)]) -> Result<Vec<KernelProfile>> {
        self.update_runtime_var_vals(var_vals)?;
        self.execute_profiled()
    }

    /// Get the first output buffer index.
    pub fn output_buffer_idx(&self) -> usize {
        self.output_buffer_indices[0]
    }

    /// Get the AST ID to buffer index mapping.
    pub fn ast_to_buffer_map(&self) -> &HashMap<u64, usize> {
        &self.ast_to_buffer
    }

    /// Release intermediate buffers from the global buffer registry.
    ///
    /// Call this after you're done executing the plan to free intermediate
    /// buffers from the global registry. The output buffer is preserved.
    pub fn release_intermediate_buffers<F>(&self, remove_fn: F)
    where
        F: Fn(u64),
    {
        self.release_buffers_impl(remove_fn, true);
    }

    /// Release ALL buffers from the global registry, including the output.
    pub fn release_all_buffers<F>(&self, remove_fn: F)
    where
        F: Fn(u64),
    {
        self.release_buffers_impl(remove_fn, false);
    }

    fn release_buffers_impl<F>(&self, remove_fn: F, skip_output: bool)
    where
        F: Fn(u64),
    {
        let output_buf_ids: std::collections::HashSet<u64> = if skip_output {
            self.output_buffer_indices.iter().filter_map(|&idx| self.buffers.get(idx).map(|b| b.id().0)).collect()
        } else {
            std::collections::HashSet::new()
        };

        for (&ast_id, &buf_idx) in &self.ast_to_buffer {
            if skip_output && output_buf_ids.contains(&self.buffers[buf_idx].id().0) {
                continue;
            }
            remove_fn(ast_id);
        }

        for &alias_id in &self.alias_ids {
            remove_fn(alias_id);
        }
    }
}

impl std::fmt::Debug for ExecutionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kernel_count = self.ops.iter().filter(|op| matches!(op, PreparedOp::CompiledProgram(_))).count();
        f.debug_struct("ExecutionPlan")
            .field("ops", &self.ops.len())
            .field("op_instance_dependencies", &self.op_instance_dependencies.len())
            .field("op_order", &self.op_order.len())
            .field("kernels", &kernel_count)
            .field("buffers", &self.buffers.len())
            .field("device", &self.device)
            .finish()
    }
}

impl std::fmt::Debug for PreparedKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedKernel")
            .field("id", &self.id)
            .field("device", &self.device)
            .field("buffer_indices", &self.buffer_indices)
            .field("output_indices", &self.output_indices)
            .field("vals", &self.vals)
            .field("fixedvars", &self.fixedvars)
            .field("dependencies", &self.dependencies)
            .finish()
    }
}

// ============================================================================
// Builder for ExecutionPlan
// ============================================================================

/// Builder for creating ExecutionPlan from schedule data.
pub struct ExecutionPlanBuilder {
    ops: Vec<PreparedOp>,
    op_instance_dependencies: Vec<Vec<usize>>,
    buffers: Vec<Buffer>,
    ast_to_buffer: HashMap<u64, usize>,
    output_buffer_indices: Vec<usize>,
    device: DeviceSpec,
    alias_ids: Vec<u64>,
}

impl ExecutionPlanBuilder {
    /// Create a new builder.
    pub fn new(device: DeviceSpec) -> Self {
        Self {
            ops: Vec::new(),
            op_instance_dependencies: Vec::new(),
            buffers: Vec::new(),
            ast_to_buffer: HashMap::new(),
            output_buffer_indices: Vec::new(),
            device,
            alias_ids: Vec::new(),
        }
    }

    /// Add alias IDs that need cleanup.
    pub fn add_alias_ids(&mut self, ids: impl IntoIterator<Item = u64>) {
        self.alias_ids.extend(ids);
    }

    /// Add a buffer to the plan. Returns the buffer index.
    pub fn add_buffer(&mut self, ast_id: u64, buffer: Buffer) -> usize {
        let idx = self.buffers.len();
        self.buffers.push(buffer);
        self.ast_to_buffer.insert(ast_id, idx);
        idx
    }

    /// Map an additional AST/buffer UOp ID to an existing buffer index.
    pub fn map_buffer(&mut self, ast_id: u64, idx: usize) {
        self.ast_to_buffer.insert(ast_id, idx);
    }

    /// Replace a buffer at the given index (for BUFFER_VIEW sub-buffer views).
    pub fn replace_buffer(&mut self, idx: usize, buffer: Buffer) {
        self.buffers[idx] = buffer;
    }

    /// Set single output buffer index.
    pub fn set_output_buffer(&mut self, idx: usize) {
        self.output_buffer_indices = vec![idx];
    }

    /// Set multiple output buffer indices (batch scheduling).
    pub fn set_output_buffers(&mut self, indices: Vec<usize>) {
        self.output_buffer_indices = indices;
    }

    /// Compatibility helper: add a compiled kernel as a prepared operation.
    ///
    /// The canonical builder path is `add_op(PreparedOp::...)`.
    pub fn add_kernel(&mut self, kernel: PreparedKernel) {
        self.add_op(PreparedOp::CompiledProgram(kernel));
    }

    /// Add a prepared operation in schedule order.
    pub fn add_op(&mut self, op: PreparedOp) {
        self.add_op_with_instance_dependencies(op, Vec::new());
    }

    /// Add a prepared operation with concrete op-index dependencies.
    pub fn add_op_with_instance_dependencies(&mut self, op: PreparedOp, instance_dependencies: Vec<usize>) {
        self.ops.push(op);
        self.op_instance_dependencies.push(instance_dependencies);
    }

    /// Build the ExecutionPlan.
    ///
    /// Finalizes by computing pre-allocated buffer pointers and buffer IDs
    /// for zero-allocation execution.
    pub fn build(mut self) -> Result<ExecutionPlan> {
        for op in &mut self.ops {
            let PreparedOp::CompiledProgram(kernel) = op else {
                continue;
            };

            if kernel.output_indices.is_empty() {
                return Err(crate::error::Error::Execution {
                    reason: format!("CompiledProgram {} has no output indices", kernel.id),
                });
            }
            for &out_idx in &kernel.output_indices {
                if out_idx >= kernel.buffer_indices.len() {
                    return Err(crate::error::Error::Execution {
                        reason: format!(
                            "CompiledProgram {} output index out of range: output_idx={}, kernel_buffers={}",
                            kernel.id,
                            out_idx,
                            kernel.buffer_indices.len()
                        ),
                    });
                }
            }

            let mut buffer_ptrs = Vec::with_capacity(kernel.buffer_indices.len());
            let mut buffer_ids = Vec::with_capacity(kernel.buffer_indices.len());

            for &idx in &kernel.buffer_indices {
                let Some(buffer) = self.buffers.get(idx) else {
                    return Err(crate::error::Error::Execution {
                        reason: format!(
                            "CompiledProgram {} buffer index out of range: idx={}, total_buffers={}",
                            kernel.id,
                            idx,
                            self.buffers.len()
                        ),
                    });
                };
                buffer_ptrs.push(unsafe { buffer.as_raw_ptr() } as usize);
                buffer_ids.push(buffer.id());
            }

            kernel.buffer_ptrs = buffer_ptrs;
            kernel.buffer_ids = buffer_ids;
        }

        if self.output_buffer_indices.is_empty() && !self.buffers.is_empty() {
            return Err(crate::error::Error::Execution {
                reason: "execution plan output buffers must be set explicitly".to_string(),
            });
        }

        let op_order = compute_mixed_op_order_with_instance_dependencies(&self.ops, &self.op_instance_dependencies)?;
        let op_levels = compute_execution_levels_with_instance_dependencies(&self.ops, &self.op_instance_dependencies)?;

        Ok(ExecutionPlan {
            ops: self.ops,
            op_instance_dependencies: self.op_instance_dependencies,
            op_order,
            op_levels,
            buffers: self.buffers,
            ast_to_buffer: self.ast_to_buffer,
            output_buffer_indices: self.output_buffer_indices,
            device: self.device,
            runtime_var_vals: HashMap::new(),
            alias_ids: self.alias_ids,
        })
    }
}

#[cfg(test)]
#[path = "test/unit/execution_plan.rs"]
mod tests;
