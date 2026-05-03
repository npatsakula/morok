//! Callable scheduling types and execution.
//!
//! This module provides types and functions for managing the execution
//! schedule of tensor operations. After the rangeify pipeline transforms
//! the computation graph into callable operations (`CALL`), we need to:
//!
//! 1. Extract callable operations from the transformed graph
//! 2. Allocate buffers for intermediate results (PARAM/DEFINE_LOCAL)
//! 3. Execute callables in dependency order
//!
//! The scheduling process converts from lazy tensor operations to
//! executable callables with properly allocated device buffers.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use morok_device::Buffer;
use morok_device::device::Device;
use morok_device::registry;
use morok_dtype::{DType, DeviceSpec};
use morok_ir::{AxisType, Op, UOp};
use tracing::{debug, trace};

use crate::error::*;
use crate::{Error, Result};
use snafu::ResultExt;

fn canonicalize_callable_source(src: &Arc<UOp>) -> Arc<UOp> {
    let mut cur = src.clone();
    loop {
        match cur.op() {
            Op::After { .. }
            | Op::Buffer { .. }
            | Op::Param { .. }
            | Op::MSelect { .. }
            | Op::MStack { .. }
            | Op::Bind { .. } => return cur,
            _ => {
                let sources = cur.op().sources();
                let Some(next) = sources.first() else {
                    return cur;
                };
                if Arc::ptr_eq(&cur, next) {
                    return cur;
                }
                cur = (*next).clone();
            }
        }
    }
}

fn source_primary_buffer_id(src: &Arc<UOp>) -> Option<u64> {
    let src = canonicalize_callable_source(src);
    match src.op() {
        Op::Buffer { .. } | Op::Param { .. } | Op::After { .. } => Some(src.buf_uop().id),
        Op::Bind { .. } => None,
        Op::MSelect { buffer, device_index } => {
            if let Op::MStack { buffers } = buffer.op() {
                buffers.get(*device_index).map(|b| b.buf_uop().id).or_else(|| Some(src.buf_uop().id))
            } else {
                Some(src.buf_uop().id)
            }
        }
        Op::MStack { buffers } => buffers.first().map(|b| b.buf_uop().id),
        _ => None,
    }
}

fn collect_callable_dep_ids(dep: &Arc<UOp>, out: &mut HashSet<u64>) -> Result<()> {
    match dep.op() {
        Op::Call { .. } => {
            out.insert(dep.id);
            Ok(())
        }
        Op::End { computation, .. } => {
            if matches!(computation.op(), Op::Call { .. }) {
                out.insert(computation.id);
                Ok(())
            } else {
                IrConstructionSnafu {
                    details: format!("AFTER dependency END must wrap CALL, got {:?}", computation.op()),
                }
                .fail()
            }
        }
        Op::Store { .. } => Ok(()),
        Op::After { deps, .. } => {
            for nested in deps {
                collect_callable_dep_ids(nested, out)?;
            }
            Ok(())
        }
        other => IrConstructionSnafu {
            details: format!("AFTER dependency must be CALL/END(CALL)/STORE/AFTER, got {other:?}"),
        }
        .fail(),
    }
}

type AfterDependencySplit = (Vec<Arc<UOp>>, Vec<Arc<UOp>>);

fn split_after_dependencies(after: &Arc<UOp>) -> Result<AfterDependencySplit> {
    let Op::After { deps, .. } = after.op() else {
        return IrConstructionSnafu {
            details: format!("expected AFTER when splitting dependencies, got {:?}", after.op()),
        }
        .fail();
    };

    let mut kernels = Vec::new();
    let mut after_deps = Vec::new();
    for dep in deps {
        match dep.op() {
            Op::Call { .. } => kernels.push(dep.clone()),
            Op::End { computation, .. } if matches!(computation.op(), Op::Call { .. }) => kernels.push(dep.clone()),
            Op::After { .. } => after_deps.push(dep.clone()),
            Op::Store { .. } => {}
            other => {
                return IrConstructionSnafu {
                    details: format!("AFTER dependency must be CALL/END(CALL)/STORE/AFTER, got {other:?}"),
                }
                .fail();
            }
        }
    }

    Ok((kernels, after_deps))
}

fn collect_source_dependency_callable_ids(src: &Arc<UOp>, out: &mut HashSet<u64>) -> Result<()> {
    let src = canonicalize_callable_source(src);
    match src.op() {
        Op::After { .. } => {
            let (kernels, after_deps) = split_after_dependencies(&src)?;
            for kernel in kernels {
                collect_callable_dep_ids(&kernel, out)?;
            }
            for dep in after_deps {
                collect_source_dependency_callable_ids(&dep, out)?;
            }
            Ok(())
        }
        Op::MStack { buffers } => {
            for buffer in buffers {
                collect_source_dependency_callable_ids(buffer, out)?;
            }
            Ok(())
        }
        Op::MSelect { buffer, .. } => collect_source_dependency_callable_ids(buffer, out),
        Op::Buffer { .. } | Op::Param { .. } | Op::Bind { .. } => Ok(()),
        other => IrConstructionSnafu {
            details: format!("input to callable must resolve to AFTER/BUFFER/PARAM/MSELECT/MSTACK/BIND, got {other:?}"),
        }
        .fail(),
    }
}

fn callable_sources(callable: &Arc<UOp>) -> Option<Vec<Arc<UOp>>> {
    match callable.op() {
        Op::Call { args, .. } => Some(args.iter().cloned().collect()),
        _ => None,
    }
}

fn collect_call_bound_ranges(callable: &Arc<UOp>) -> Result<Vec<BoundRangeRef>> {
    let Op::Call { args, .. } = callable.op() else {
        return ExpectedCallableOpSnafu.fail();
    };

    let mut bound_ranges = Vec::new();
    for arg in args {
        let Op::Bind { var, value } = arg.op() else {
            continue;
        };
        let Op::DefineVar { name, .. } = var.op() else {
            return IrConstructionSnafu {
                details: format!("CALL BIND source must wrap DEFINE_VAR, got {:?}", var.op()),
            }
            .fail();
        };
        let Op::Range { axis_type, .. } = value.op() else {
            // User variable binds (`BIND(DEFINE_VAR, CONST)`) are not schedule loops.
            continue;
        };
        if *axis_type == AxisType::Outer {
            bound_ranges.push(BoundRangeRef { var_name: name.clone(), range_uop: value.clone() });
        }
    }
    Ok(bound_ranges)
}

fn collect_linear_sched_ops(root: &Arc<UOp>, callables: &[Arc<UOp>]) -> Result<Vec<LinearSchedOp>> {
    let callable_ids: HashSet<u64> = callables.iter().map(|c| c.id).collect();
    let mut linear_ops = Vec::new();

    for node in root.toposort_call_aware(false) {
        match node.op() {
            Op::Range { axis_type, .. } if *axis_type == AxisType::Outer => {
                linear_ops.push(LinearSchedOp::Range { range: node.clone() });
            }
            Op::Call { .. } if callable_ids.contains(&node.id) => {
                linear_ops.push(LinearSchedOp::Call { kernel_id: node.id });
            }
            Op::End { computation, ranges } if matches!(computation.op(), Op::Call { .. }) => {
                if !callable_ids.contains(&computation.id) {
                    continue;
                }
                let outer_ranges: Vec<Arc<UOp>> = ranges
                    .iter()
                    .filter_map(|r| match r.op() {
                        Op::Range { axis_type: AxisType::Outer, .. } => Some(r.clone()),
                        _ => None,
                    })
                    .collect();
                match outer_ranges.as_slice() {
                    [] => {}
                    [outer] => linear_ops.push(LinearSchedOp::End { range: outer.clone(), kernel_id: computation.id }),
                    _ => {
                        return IrConstructionSnafu {
                            details: format!(
                                "END(CALL) must close at most one OUTER range in strict scheduler, got {}",
                                outer_ranges.len()
                            ),
                        }
                        .fail();
                    }
                }
            }
            _ => {}
        }
    }

    if linear_ops.is_empty() {
        return IrConstructionSnafu { details: "strict scheduler produced empty linear control stream".to_string() }
            .fail();
    }
    Ok(linear_ops)
}

fn analyze_callable_dependencies(callables: &[Arc<UOp>], root: &Arc<UOp>) -> Result<Vec<HashSet<usize>>> {
    // Build callable ID → index mapping
    let callable_idx: HashMap<u64, usize> = callables.iter().enumerate().map(|(i, c)| (c.id, i)).collect();
    // Build dependency edges from source-driven dependency extraction.
    // This mirrors tinygrad scheduler semantics and avoids global writer-union heuristics.
    let mut dependencies: Vec<HashSet<usize>> = vec![HashSet::new(); callables.len()];

    for (consumer_idx, callable) in callables.iter().enumerate() {
        let mut dep_ids = HashSet::new();
        if let Some(sources) = callable_sources(callable) {
            for src in sources {
                collect_source_dependency_callable_ids(&src, &mut dep_ids)?;
            }
        }

        for dep_id in dep_ids {
            let Some(&producer_idx) = callable_idx.get(&dep_id) else {
                return IrConstructionSnafu {
                    details: format!("callable dependency references unknown callable id {dep_id}"),
                }
                .fail();
            };
            if producer_idx != consumer_idx {
                dependencies[consumer_idx].insert(producer_idx);
            }
        }
    }

    // Preserve ordering-only dependencies encoded through AFTER surfaces that
    // may include void/custom callables with no direct buffer edge.
    for node in root.toposort() {
        let Op::After { .. } = node.op() else {
            continue;
        };

        let (kernels, after_deps) = split_after_dependencies(&node)?;
        for kernel in kernels {
            let callable = match kernel.op() {
                Op::Call { .. } => kernel.clone(),
                Op::End { computation, .. } => computation.clone(),
                _ => unreachable!("split_after_dependencies only returns CALL/END(CALL) kernels"),
            };

            let Some(&consumer_idx) = callable_idx.get(&callable.id) else {
                return IrConstructionSnafu {
                    details: format!("AFTER dependency references unknown callable id {}", callable.id),
                }
                .fail();
            };

            let mut dep_ids = HashSet::new();
            for dep in &after_deps {
                collect_source_dependency_callable_ids(dep, &mut dep_ids)?;
            }

            for dep_id in dep_ids {
                let Some(&producer_idx) = callable_idx.get(&dep_id) else {
                    return IrConstructionSnafu {
                        details: format!("callable dependency references unknown callable id {dep_id}"),
                    }
                    .fail();
                };
                if producer_idx != consumer_idx {
                    dependencies[consumer_idx].insert(producer_idx);
                }
            }
        }
    }

    Ok(dependencies)
}

/// Input buffers collected before schedule creation.
///
/// Maps BUFFER UOp ID → Buffer for input tensors.
/// This keeps schedule instantiation explicit and avoids global lookups.
pub type InputBuffers = HashMap<u64, Buffer>;

/// Bound OUTER range reference extracted from CALL arguments.
#[derive(Clone, Debug)]
pub struct BoundRangeRef {
    /// Variable name (e.g., "range_0")
    pub var_name: String,
    /// RANGE UOp for this bound variable.
    pub range_uop: Arc<UOp>,
}

/// Linearized scheduling control op.
#[derive(Clone, Debug)]
pub enum LinearSchedOp {
    Range { range: Arc<UOp> },
    Call { kernel_id: u64 },
    End { range: Arc<UOp>, kernel_id: u64 },
}

/// A single executable callable with its buffers and variable bindings.
///
/// Each ScheduleItem represents one callable that needs to be compiled
/// and executed. The callable AST contains STORE operations that write
/// results to buffers.
///
/// Schedule items are fully expanded during schedule instantiation.
#[derive(Clone)]
pub struct ScheduleItem {
    /// The callable wrapper UOp (`CALL`) used for dependency identity.
    pub kernel: Arc<UOp>,

    /// The inner callable AST (typically SINK containing STORE ops) - for codegen
    pub ast: Arc<UOp>,

    /// Device buffers for this callable (in order expected by codegen)
    pub buffers: Vec<Buffer>,

    /// UOp IDs under which each buffer was registered in buffer index.
    /// Same length as `buffers`. Used for cleanup - to remove buffers from
    /// the global registry, we need to know what key they were registered under.
    pub buffer_uop_ids: Vec<u64>,

    /// Fixed variable values for this specific kernel invocation.
    /// Maps variable name (e.g., "range_0") to concrete i64 value.
    /// Always concrete in the strict scheduler path.
    pub fixedvars: HashMap<String, i64>,

    /// Callable UOp IDs that must complete before this item can execute.
    /// Empty for callables without dependencies (first in chain or independent).
    /// Dependencies are implicit in scheduling order after topological sort.
    pub dependencies: Vec<u64>,

    /// Concrete schedule-item indices that must complete before this item.
    /// Used for ordering constraints that cannot be represented by callable ID
    /// after strict unrolling creates repeated callable IDs.
    pub instance_dependencies: Vec<usize>,

    /// Additional UOp IDs registered as aliases in buffer index.
    /// These are IDs where the same buffer was registered under a different key
    /// for lookup convenience. They need to be cleaned up along with buffer_uop_ids.
    pub alias_registered_ids: Vec<u64>,
}

/// Full execution schedule (list of callables in dependency order).
pub type Schedule = Vec<ScheduleItem>;

/// Cached pre-schedule item (Tinygrad-style `pre_schedule` equivalent).
///
/// Contains callable identity/AST and argument UOps, but no concrete buffers.
#[derive(Clone)]
pub struct PreScheduleItem {
    /// Callable wrapper UOp (`CALL`) used for dependency identity.
    pub kernel: Arc<UOp>,
    /// Callable body AST used for codegen.
    pub ast: Arc<UOp>,
    /// Callable argument UOps in canonical order.
    pub sources: Vec<Arc<UOp>>,
    /// Callable dependencies by callable UOp ID.
    pub dependencies: Vec<u64>,
    /// OUTER range bindings (`BIND(DEFINE_VAR, RANGE)`) from CALL args.
    pub bound_ranges: Vec<BoundRangeRef>,
}

/// Cached pre-schedule artifact.
///
/// Mirrors Tinygrad's cached `pre_schedule + buf_uops` split:
/// - `items`: executable callable descriptors without buffers
/// - `output_buffer_uops`: output buffer UOps in sink source order
#[derive(Clone)]
pub struct PreSchedule {
    /// Pre-schedule callable items.
    pub items: Vec<PreScheduleItem>,
    /// Strict linear schedule control stream (`RANGE`, `CALL`, `END`).
    pub linear_ops: Vec<LinearSchedOp>,
    /// Output buffers in sink source order.
    pub output_buffer_uops: Vec<Arc<UOp>>,
}

type SortedCallables = (Vec<Arc<UOp>>, HashMap<u64, Vec<u64>>);

/// Result of schedule creation, including output buffer identification.
pub struct ScheduleResult {
    /// The schedule items in dependency order.
    pub items: Schedule,
    /// UOp IDs of output buffers, in SINK source order.
    /// Extracted directly from the SINK's sources via `buf_uop()`.
    /// For single-tensor realize, contains one ID.
    pub output_uop_ids: Vec<u64>,
}

/// Buffers collected for a single callable.
struct CallableBuffers {
    /// Device buffers in codegen order.
    buffers: Vec<Buffer>,
    /// UOp IDs for each buffer.
    uop_ids: Vec<u64>,
    /// Additional alias IDs for cleanup.
    alias_ids: Vec<u64>,
}

/// Sort callables by dependencies (producers before consumers).
///
/// Uses Kahn's algorithm for topological sort based on buffer dependencies
/// derived from the graph structure (AFTER nodes and callable sources).
/// This ensures producer callables are processed before consumers, which is
/// critical for buffer sharing: the producer allocates the buffer first,
/// then the consumer finds it in the registry via `get_or_create_buffer`.
fn sort_callables_by_dependencies(callables: &[Arc<UOp>], root: &Arc<UOp>) -> Result<SortedCallables> {
    debug!(num_callables = callables.len(), "sorting callables by dependencies");

    let dependencies = analyze_callable_dependencies(callables, root)?;

    // Kahn's algorithm for topological sort
    let mut in_degree: Vec<usize> = dependencies.iter().map(|deps| deps.len()).collect();
    let mut dependents: Vec<Vec<usize>> = vec![vec![]; callables.len()];

    for (consumer, deps) in dependencies.iter().enumerate() {
        for &producer in deps {
            dependents[producer].push(consumer);
        }
    }

    let mut queue: VecDeque<usize> =
        in_degree.iter().enumerate().filter(|&(_, &deg)| deg == 0).map(|(idx, _)| idx).collect();

    let mut sorted_indices = Vec::new();
    while let Some(idx) = queue.pop_front() {
        sorted_indices.push(idx);
        for &dependent in &dependents[idx] {
            in_degree[dependent] -= 1;
            if in_degree[dependent] == 0 {
                queue.push_back(dependent);
            }
        }
    }

    if sorted_indices.len() < callables.len() {
        return DependencyCyclesSnafu.fail();
    }

    let sorted: Vec<Arc<UOp>> = sorted_indices.iter().map(|&idx| callables[idx].clone()).collect();

    let dependency_ids_by_callable: HashMap<u64, Vec<u64>> = callables
        .iter()
        .enumerate()
        .map(|(idx, callable)| {
            let mut deps: Vec<u64> = dependencies[idx].iter().map(|&dep_idx| callables[dep_idx].id).collect();
            deps.sort_unstable();
            (callable.id, deps)
        })
        .collect();

    debug!(num_sorted = sorted.len(), "callables sorted");

    Ok((sorted, dependency_ids_by_callable))
}

/// Extract callables from transformed graph and create pre-schedule artifact.
///
/// This function walks the transformed UOp graph (after rangeify and
/// kernel splitting) and extracts all callable wrappers. For each callable,
/// it records callable identity, dependencies, and strict control-flow ops.
///
/// # Arguments
///
/// * `transformed` - The UOp graph after rangeify + kernel splitting
///
/// # Returns
///
/// A pre-schedule artifact ready for per-run instantiation.
///
/// # Errors
///
/// Returns error if:
/// - No callables found after scheduling pipeline
/// - Callable dependency graph contains cycles
pub fn create_pre_schedule(transformed: Arc<UOp>) -> Result<PreSchedule> {
    // Step 1: Find all callable wrappers (CALL) without descending into CALL bodies.
    let mut callables = Vec::new();
    for node in transformed.toposort_call_aware(false) {
        if matches!(node.op(), Op::Call { .. }) {
            callables.push(node);
        }
    }

    if callables.is_empty() {
        return NoKernelsFoundSnafu.fail();
    }

    // Step 1.5: Sort callables by dependencies (producers before consumers)
    let (callables, dependency_ids_by_callable) = sort_callables_by_dependencies(&callables, &transformed)?;

    // Step 2: Build pre-schedule items (AST + sources + dependencies + bound ranges).
    let mut items = Vec::with_capacity(callables.len());
    for callable_uop in callables {
        let Op::Call { body, args, .. } = callable_uop.op() else {
            unreachable!("filtered to only call wrappers above")
        };
        let dependencies = dependency_ids_by_callable.get(&callable_uop.id).cloned().unwrap_or_default();
        let bound_ranges = collect_call_bound_ranges(&callable_uop)?;
        items.push(PreScheduleItem {
            kernel: callable_uop.clone(),
            ast: body.clone(),
            sources: args.iter().cloned().collect(),
            dependencies,
            bound_ranges,
        });
    }

    // Step 3: Build strict linear control stream (`RANGE`, `CALL`, `END`).
    let callable_nodes: Vec<Arc<UOp>> = items.iter().map(|it| it.kernel.clone()).collect();
    let linear_ops = collect_linear_sched_ops(&transformed, &callable_nodes)?;

    // Output buffers in SINK source order.
    let output_buffer_uops: Vec<Arc<UOp>> = match transformed.op() {
        Op::Sink { sources, .. } => sources.iter().map(|src| src.buf_uop()).collect(),
        _ => vec![transformed.buf_uop()],
    };

    Ok(PreSchedule { items, linear_ops, output_buffer_uops })
}

/// Instantiate a concrete execution schedule from a pre-schedule artifact.
///
/// This is the per-run phase that attaches concrete buffers and runtime
/// variable bindings to cached callable descriptors.
pub fn instantiate_schedule(
    pre_schedule: &PreSchedule,
    input_buffers: &InputBuffers,
    var_vals: &HashMap<String, i64>,
) -> Result<ScheduleResult> {
    // Track allocated intermediate buffers locally (no global registry needed)
    let mut allocated_buffers: HashMap<u64, Buffer> = HashMap::new();

    let mut templates: HashMap<u64, ScheduleItemTemplate> = HashMap::with_capacity(pre_schedule.items.len());
    for item in &pre_schedule.items {
        let nodes = item.ast.toposort();

        // Map sources to actual Buffers.
        let kb = collect_callable_buffers(&item.sources, &item.ast, input_buffers, &mut allocated_buffers)?;

        debug!(callable.id = item.kernel.id, num_sources = item.sources.len(), "Schedule item created");

        // Populate fixedvars with only the user Variables referenced by this kernel's AST.
        let fixedvars: HashMap<String, i64> = if var_vals.is_empty() {
            HashMap::new()
        } else {
            let ast_var_names: HashSet<&str> = nodes
                .iter()
                .filter_map(|n| match n.op() {
                    Op::DefineVar { name, .. } => Some(name.as_str()),
                    _ => None,
                })
                .collect();
            var_vals
                .iter()
                .filter(|(name, _)| ast_var_names.contains(name.as_str()))
                .map(|(k, v)| (k.clone(), *v))
                .collect()
        };

        templates.insert(
            item.kernel.id,
            ScheduleItemTemplate {
                kernel: item.kernel.clone(),
                ast: item.ast.clone(),
                buffers: kb.buffers,
                buffer_uop_ids: kb.uop_ids,
                dependencies: item.dependencies.clone(),
                alias_registered_ids: kb.alias_ids,
                base_fixedvars: fixedvars,
                bound_ranges: item.bound_ranges.clone(),
            },
        );
    }

    let schedule = unroll_linear_schedule_strict(&pre_schedule.linear_ops, &templates)?;
    if schedule.is_empty() {
        return EmptyScheduleSnafu.fail();
    }

    let output_uop_ids: Vec<u64> = pre_schedule.output_buffer_uops.iter().map(|u| u.buf_uop().id).collect();
    Ok(ScheduleResult { items: schedule, output_uop_ids })
}

pub fn create_schedule(
    transformed: Arc<UOp>,
    input_buffers: &InputBuffers,
    var_vals: &HashMap<String, i64>,
) -> Result<ScheduleResult> {
    let pre = create_pre_schedule(transformed)?;
    instantiate_schedule(&pre, input_buffers, var_vals)
}

/// Extract device from the first input buffer in callable sources.
///
/// This follows Tinygrad's pattern where `ctx[0].device` (first buffer's device)
/// determines the device for codegen/compilation and output buffer allocation.
///
/// DISK buffers are skipped: a disk-resident input is never a viable execution
/// device (no compiler), so the search continues to find a compute-capable
/// source. When every source is on disk (e.g. fully-materialized parameter
/// graphs that exist only as safetensors mmaps before realization), we fall
/// back to CPU so the kernel still has somewhere to run; the runtime then
/// arranges the disk→CPU copies via the normal Copy ops.
fn find_first_input_buffer_device(
    sources: &[Arc<UOp>],
    input_buffers: &InputBuffers,
    allocated_buffers: &HashMap<u64, Buffer>,
) -> Result<Arc<Device>> {
    let alloc_registry = registry::registry();

    for src in sources {
        if let Some(buf_id) = source_primary_buffer_id(src) {
            let buffer = allocated_buffers.get(&buf_id).cloned().or_else(|| input_buffers.get(&buf_id).cloned());
            if let Some(buffer) = buffer {
                let device_spec = buffer.allocator().device_spec();
                if device_spec.is_disk() {
                    continue;
                }
                return morok_runtime::DEVICE_FACTORIES
                    .device(&device_spec, alloc_registry)
                    .context(DeviceFactorySnafu);
            }
        }
    }

    // Fallback to CPU if no input buffers found
    morok_runtime::DEVICE_FACTORIES.device(&DeviceSpec::Cpu, alloc_registry).context(DeviceFactorySnafu)
}

/// Collect buffers for a callable from its sources.
///
/// This walks the callable sources and identifies:
/// - Input buffers (Op::Buffer) - get from input_buffers
/// - Intermediate buffers (Op::Param) - allocate and track
/// - Shared buffers (Op::After) - look up from allocated_buffers (producer callable)
///
/// For input buffers (PARAM that maps to an original BUFFER),
/// we reuse the existing buffer from input_buffers instead of allocating.
///
/// For shared buffers (AFTER nodes), we look up the buffer using buf_uop()
/// which walks through AFTER chains to get the underlying buffer ID.
///
/// Output/intermediate buffers are allocated on the same device as the first input buffer
/// (following Tinygrad's pattern). Newly allocated buffers are tracked in `allocated_buffers`.
fn collect_callable_buffers(
    sources: &[Arc<UOp>],
    ast: &Arc<UOp>,
    input_buffers: &InputBuffers,
    allocated_buffers: &mut HashMap<u64, Buffer>,
) -> Result<CallableBuffers> {
    // Get target device from first input buffer (Tinygrad pattern: ctx[0].device)
    let target_device = find_first_input_buffer_device(sources, input_buffers, allocated_buffers)?;

    let mut buffers = Vec::new();
    let mut uop_ids = Vec::new();
    let mut alias_ids = Vec::new();

    for src in sources {
        let canonical_src = canonicalize_callable_source(src);
        if canonical_src.id != src.id {
            alias_ids.push(src.id);
        }

        match canonical_src.op() {
            Op::After { passthrough, .. } => {
                // Shared buffer from producer kernel.
                // Use buf_uop() to get underlying buffer ID (handles AFTER chains).
                let buf_id = passthrough.buf_uop().id;
                if buf_id != canonical_src.id {
                    alias_ids.push(canonical_src.id);
                }

                // Look up from allocated_buffers or input_buffers
                let existing = allocated_buffers.get(&buf_id).cloned().or_else(|| input_buffers.get(&buf_id).cloned());

                if let Some(buffer) = existing {
                    trace!(
                        buf_id,
                        buffer.id = ?buffer.id(),
                        "Found shared buffer from AFTER"
                    );

                    // Track under buf_id if not already tracked
                    allocated_buffers.entry(buf_id).or_insert_with(|| buffer.clone());

                    buffers.push(buffer);
                    uop_ids.push(buf_id);
                } else {
                    trace!(buf_id, "after buffer not found in allocated_buffers or input_buffers");
                    return Err(Error::BufferNotFound { uop_id: buf_id });
                }
            }
            Op::MSelect { .. } | Op::MStack { .. } => {
                let Some(canonical_id) = source_primary_buffer_id(&canonical_src) else {
                    return IrConstructionSnafu {
                        details: format!(
                            "multi-device callable source must resolve a primary buffer id: source_id={}, op={:?}",
                            canonical_src.id,
                            canonical_src.op()
                        ),
                    }
                    .fail();
                };
                if canonical_id != canonical_src.id {
                    alias_ids.push(canonical_src.id);
                }

                let existing =
                    allocated_buffers.get(&canonical_id).cloned().or_else(|| input_buffers.get(&canonical_id).cloned());

                if let Some(buffer) = existing {
                    trace!(canonical_id, buffer.id = ?buffer.id(), "Found shared buffer from MSELECT/MSTACK source");
                    allocated_buffers.entry(canonical_id).or_insert_with(|| buffer.clone());
                    buffers.push(buffer);
                    uop_ids.push(canonical_id);
                } else {
                    trace!(canonical_id, "multi-device source buffer not found in allocated_buffers or input_buffers");
                    return Err(Error::BufferNotFound { uop_id: canonical_id });
                }
            }
            // Callable args/sources are typically Buffer/Param/After/DefineLocal.
            Op::DefineLocal(_id) => {
                // Allocate local/shared memory buffer on same device as inputs
                let ptr_dtype = canonical_src.dtype();
                let size = compute_buffer_size(ast, &canonical_src)?;

                // Extract the base scalar dtype from the Ptr type
                let scalar_dtype = match ptr_dtype {
                    morok_dtype::DType::Ptr { base, .. } => *base,
                    other => {
                        return ExpectedPtrDtypeSnafu { context: "DEFINE_LOCAL", actual: other.clone() }.fail();
                    }
                };

                let buffer =
                    Buffer::new(target_device.allocator.clone(), scalar_dtype.clone(), vec![size], Default::default());

                // Track in allocated_buffers (no registry needed)
                allocated_buffers.insert(canonical_src.id, buffer.clone());

                buffers.push(buffer);
                uop_ids.push(canonical_src.id);
            }
            Op::Buffer { size, .. } | Op::Param { size, .. } => {
                let canonical_id = canonical_src.buf_uop().id;
                if canonical_id != canonical_src.id {
                    alias_ids.push(canonical_src.id);
                }

                // BUFFER/PARAM can be either input (from input_buffers) or output (needs allocation)
                // Try input_buffers first, then allocated_buffers, then allocate new
                if let Some(buffer) =
                    input_buffers.get(&canonical_id).cloned().or_else(|| input_buffers.get(&canonical_src.id).cloned())
                {
                    buffers.push(buffer);
                    uop_ids.push(canonical_id);
                } else if let Some(buffer) = allocated_buffers
                    .get(&canonical_id)
                    .cloned()
                    .or_else(|| allocated_buffers.get(&canonical_src.id).cloned())
                {
                    buffers.push(buffer);
                    uop_ids.push(canonical_id);
                } else {
                    // Output buffer - allocate new buffer
                    trace!(src.id = canonical_src.id, canonical_id, size, "Allocating output BUFFER/PARAM");
                    let scalar_dtype = canonical_src.dtype();

                    let buffer = Buffer::new(
                        target_device.allocator.clone(),
                        scalar_dtype.clone(),
                        vec![*size],
                        Default::default(),
                    );

                    // Track in allocated_buffers
                    allocated_buffers.insert(canonical_id, buffer.clone());
                    buffers.push(buffer);
                    uop_ids.push(canonical_id);
                }
            }
            Op::Bind { .. } => {
                // Variable binding - not a buffer, skip
                continue;
            }
            other => {
                return IrConstructionSnafu {
                    details: format!("unsupported callable source op for buffer collection: {other:?}"),
                }
                .fail();
            }
        }
    }

    alias_ids.sort_unstable();
    alias_ids.dedup();
    Ok(CallableBuffers { buffers, uop_ids, alias_ids })
}

#[derive(Clone)]
struct ScheduleItemTemplate {
    kernel: Arc<UOp>,
    ast: Arc<UOp>,
    buffers: Vec<Buffer>,
    buffer_uop_ids: Vec<u64>,
    dependencies: Vec<u64>,
    alias_registered_ids: Vec<u64>,
    base_fixedvars: HashMap<String, i64>,
    bound_ranges: Vec<BoundRangeRef>,
}

fn outer_range_bounds(range: &Arc<UOp>) -> Result<(i64, i64)> {
    let Op::Range { axis_type, .. } = range.op() else {
        return IrConstructionSnafu { details: format!("expected RANGE for outer loop control, got {:?}", range.op()) }
            .fail();
    };
    if *axis_type != AxisType::Outer {
        return IrConstructionSnafu {
            details: format!("strict scheduler expects OUTER RANGE control, got axis_type={axis_type:?}"),
        }
        .fail();
    }

    let Some(vmin) = range.vmin().try_int() else {
        return IrConstructionSnafu {
            details: format!("OUTER RANGE vmin must be concrete integer, got {:?}", range.vmin()),
        }
        .fail();
    };
    let Some(vmax) = range.vmax().try_int() else {
        return IrConstructionSnafu {
            details: format!("OUTER RANGE vmax must be concrete integer, got {:?}", range.vmax()),
        }
        .fail();
    };
    if vmax < vmin {
        return IrConstructionSnafu { details: format!("invalid OUTER RANGE bounds: vmin={vmin}, vmax={vmax}") }.fail();
    }
    Ok((vmin, vmax))
}

fn unroll_linear_schedule_strict(
    linear_ops: &[LinearSchedOp],
    templates: &HashMap<u64, ScheduleItemTemplate>,
) -> Result<Schedule> {
    let mut declared_ranges = HashSet::new();
    let mut ended_ranges = HashSet::new();
    for op in linear_ops {
        match op {
            LinearSchedOp::Range { range } => {
                declared_ranges.insert(range.id);
            }
            LinearSchedOp::End { range, .. } => {
                ended_ranges.insert(range.id);
            }
            LinearSchedOp::Call { .. } => {}
        }
    }

    for &rid in &declared_ranges {
        if !ended_ranges.contains(&rid) {
            return IrConstructionSnafu { details: format!("OUTER RANGE {rid} is missing END in strict scheduler") }
                .fail();
        }
    }
    for template in templates.values() {
        for br in &template.bound_ranges {
            if !declared_ranges.contains(&br.range_uop.id) {
                return IrConstructionSnafu {
                    details: format!(
                        "CALL {} bound variable '{}' references OUTER RANGE {} missing from linear schedule",
                        template.kernel.id, br.var_name, br.range_uop.id
                    ),
                }
                .fail();
            }
        }
    }

    let mut schedule = Vec::new();
    let mut in_ranges: HashMap<u64, i64> = HashMap::new();
    let mut range_ptrs: HashMap<u64, usize> = HashMap::new();
    let mut range_bounds: HashMap<u64, (i64, i64)> = HashMap::new();

    let mut sched_ptr = 0usize;
    while sched_ptr < linear_ops.len() {
        match &linear_ops[sched_ptr] {
            LinearSchedOp::Range { range } => {
                let bounds = if let Some(bounds) = range_bounds.get(&range.id).copied() {
                    bounds
                } else {
                    let bounds = outer_range_bounds(range)?;
                    range_bounds.insert(range.id, bounds);
                    bounds
                };
                in_ranges.insert(range.id, bounds.0);
                range_ptrs.insert(range.id, sched_ptr + 1);
            }
            LinearSchedOp::End { range, kernel_id } => {
                if !templates.contains_key(kernel_id) {
                    return IrConstructionSnafu {
                        details: format!("linear END references unknown CALL id {kernel_id}"),
                    }
                    .fail();
                }

                let (_, vmax) = if let Some(bounds) = range_bounds.get(&range.id).copied() {
                    bounds
                } else {
                    let bounds = outer_range_bounds(range)?;
                    range_bounds.insert(range.id, bounds);
                    bounds
                };

                let Some(cur) = in_ranges.get_mut(&range.id) else {
                    return IrConstructionSnafu {
                        details: format!("END references OUTER RANGE {} that is not active", range.id),
                    }
                    .fail();
                };

                if *cur < vmax {
                    *cur += 1;
                    let Some(jump_ptr) = range_ptrs.get(&range.id).copied() else {
                        return IrConstructionSnafu {
                            details: format!("missing loop jump pointer for OUTER RANGE {}", range.id),
                        }
                        .fail();
                    };
                    sched_ptr = jump_ptr;
                    continue;
                }
            }
            LinearSchedOp::Call { kernel_id } => {
                let Some(template) = templates.get(kernel_id) else {
                    return IrConstructionSnafu {
                        details: format!("linear CALL references unknown kernel id {kernel_id}"),
                    }
                    .fail();
                };

                let mut fixedvars = template.base_fixedvars.clone();
                for br in &template.bound_ranges {
                    let Some(value) = in_ranges.get(&br.range_uop.id).copied() else {
                        return IrConstructionSnafu {
                            details: format!(
                                "CALL {} bound variable '{}' references inactive OUTER RANGE {}",
                                kernel_id, br.var_name, br.range_uop.id
                            ),
                        }
                        .fail();
                    };
                    fixedvars.insert(br.var_name.clone(), value);
                }

                schedule.push(ScheduleItem {
                    kernel: template.kernel.clone(),
                    ast: template.ast.clone(),
                    buffers: template.buffers.clone(),
                    buffer_uop_ids: template.buffer_uop_ids.clone(),
                    fixedvars,
                    dependencies: template.dependencies.clone(),
                    instance_dependencies: Vec::new(),
                    alias_registered_ids: template.alias_registered_ids.clone(),
                });
            }
        }

        sched_ptr += 1;
    }

    Ok(schedule)
}

/// Compute buffer size from the buffer definition's dtype.
///
/// Buffer size is embedded in the Ptr dtype by debuf() during rangeify.
/// This follows Tinygrad's pattern where size is stored in `dtype.ptr(size=...)`.
fn compute_buffer_size(_ast: &Arc<UOp>, buffer_def: &Arc<UOp>) -> Result<usize> {
    // Extract size from Ptr dtype (set by debuf() in split_patterns.rs)
    match buffer_def.dtype() {
        DType::Ptr { size: Some(s), .. } => Ok(s),
        DType::Ptr { size: None, .. } => BufferPtrNoSizeSnafu.fail(),
        other => ExpectedPtrDtypeSnafu { context: "buffer_size", actual: other.clone() }.fail(),
    }
}
