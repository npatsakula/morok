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
use morok_ir::{Op, UOp};
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

/// Schedule-level RANGEs are uniquely identified by being paired with an
/// `END(Call)` in the transformed graph — the END structurally proves the
/// Range wraps a kernel call. Standalone `Bind(DefineVar, Range)` arguments
/// (user-supplied symbolic variable binds) reference Ranges with no END
/// pairing and must be skipped to avoid being mistaken for loop wrappers.
fn collect_scheduled_range_ids(root: &Arc<UOp>, callable_ids: &HashSet<u64>) -> HashSet<u64> {
    let mut ids = HashSet::new();
    for node in root.toposort_call_aware(false) {
        let Op::End { computation, ranges } = node.op() else { continue };
        if !matches!(computation.op(), Op::Call { .. }) || !callable_ids.contains(&computation.id) {
            continue;
        }
        for r in ranges {
            if matches!(r.op(), Op::Range { .. }) {
                ids.insert(r.id);
            }
        }
    }
    ids
}

fn collect_call_bound_ranges(callable: &Arc<UOp>, scheduled_range_ids: &HashSet<u64>) -> Result<Vec<BoundRangeRef>> {
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
        let Op::Range { .. } = value.op() else {
            // User variable binds (`BIND(DEFINE_VAR, CONST)`) are not schedule loops.
            continue;
        };
        // Only Ranges paired with an `END(Call)` are schedule-level wrappers;
        // standalone Range-valued binds carry runtime values, not loop counters.
        if !scheduled_range_ids.contains(&value.id) {
            continue;
        }
        bound_ranges.push(BoundRangeRef { var_name: name.clone(), range_uop: value.clone() });
    }
    Ok(bound_ranges)
}

fn collect_linear_sched_ops_internal(
    root: &Arc<UOp>,
    callable_ids: &HashSet<u64>,
    scheduled_range_ids: &HashSet<u64>,
) -> Result<Vec<LinearSchedOp>> {
    let mut linear_ops = Vec::new();

    for node in root.toposort_call_aware(false) {
        match node.op() {
            Op::Range { .. } if scheduled_range_ids.contains(&node.id) => {
                linear_ops.push(LinearSchedOp::Range { range: node.clone() });
            }
            Op::Call { .. } if callable_ids.contains(&node.id) => {
                linear_ops.push(LinearSchedOp::Call { kernel_id: node.id });
            }
            Op::End { computation, ranges } if matches!(computation.op(), Op::Call { .. }) => {
                if !callable_ids.contains(&computation.id) {
                    continue;
                }
                let wrapper_ranges: Vec<Arc<UOp>> =
                    ranges.iter().filter(|r| matches!(r.op(), Op::Range { .. })).cloned().collect();
                match wrapper_ranges.as_slice() {
                    [] => {}
                    [outer] => linear_ops.push(LinearSchedOp::End { range: outer.clone(), kernel_id: computation.id }),
                    _ => {
                        return IrConstructionSnafu {
                            details: format!(
                                "END(CALL) must close at most one wrapper range in strict scheduler, got {}",
                                wrapper_ranges.len()
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

/// Eagerly unroll the schedule control stream into a flat list of kernel
/// invocations.
///
/// Every outer loop iteration produces one invocation per kernel inside it,
/// with concrete `fixedvars` derived from the loop counters at that point.
/// Outer ranges must have concrete `vmin`/`vmax` (validated by
/// `schedule_range_bounds`) — there is no symbolic-iteration support today.
fn collect_kernel_invocations(
    root: &Arc<UOp>,
    items: &[PreScheduleItem],
    scheduled_range_ids: &HashSet<u64>,
) -> Result<Vec<KernelInvocation>> {
    let callable_ids: HashSet<u64> = items.iter().map(|it| it.kernel.id).collect();
    let linear_ops = collect_linear_sched_ops_internal(root, &callable_ids, scheduled_range_ids)?;

    let bound_ranges_by_kernel: HashMap<u64, &[BoundRangeRef]> =
        items.iter().map(|it| (it.kernel.id, it.bound_ranges.as_slice())).collect();

    // Pre-validation: every declared Range must have a matching End, and every
    // bound_range on every kernel must reference a declared Range.
    let mut declared_ranges: HashSet<u64> = HashSet::new();
    let mut ended_ranges: HashSet<u64> = HashSet::new();
    for op in &linear_ops {
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
            return IrConstructionSnafu { details: format!("schedule range {rid} is missing END in strict scheduler") }
                .fail();
        }
    }
    for item in items {
        for br in &item.bound_ranges {
            if !declared_ranges.contains(&br.range_uop.id) {
                return IrConstructionSnafu {
                    details: format!(
                        "CALL {} bound variable '{}' references schedule range {} missing from linear schedule",
                        item.kernel.id, br.var_name, br.range_uop.id
                    ),
                }
                .fail();
            }
        }
    }

    // Bytecode interpreter (range/end drives a counter, call emits an
    // invocation). The output is the eagerly-unrolled invocation list.
    let mut invocations = Vec::new();
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
                    let bounds = schedule_range_bounds(range)?;
                    range_bounds.insert(range.id, bounds);
                    bounds
                };
                in_ranges.insert(range.id, bounds.0);
                range_ptrs.insert(range.id, sched_ptr + 1);
            }
            LinearSchedOp::End { range, kernel_id } => {
                if !bound_ranges_by_kernel.contains_key(kernel_id) {
                    return IrConstructionSnafu {
                        details: format!("linear END references unknown CALL id {kernel_id}"),
                    }
                    .fail();
                }
                let (_, vmax) = if let Some(bounds) = range_bounds.get(&range.id).copied() {
                    bounds
                } else {
                    let bounds = schedule_range_bounds(range)?;
                    range_bounds.insert(range.id, bounds);
                    bounds
                };
                let Some(cur) = in_ranges.get_mut(&range.id) else {
                    return IrConstructionSnafu {
                        details: format!("END references schedule range {} that is not active", range.id),
                    }
                    .fail();
                };
                if *cur < vmax {
                    *cur += 1;
                    let Some(jump_ptr) = range_ptrs.get(&range.id).copied() else {
                        return IrConstructionSnafu {
                            details: format!("missing loop jump pointer for schedule range {}", range.id),
                        }
                        .fail();
                    };
                    sched_ptr = jump_ptr;
                    continue;
                }
            }
            LinearSchedOp::Call { kernel_id } => {
                let Some(bound_ranges) = bound_ranges_by_kernel.get(kernel_id) else {
                    return IrConstructionSnafu {
                        details: format!("linear CALL references unknown kernel id {kernel_id}"),
                    }
                    .fail();
                };
                let mut fixedvars = HashMap::new();
                for br in *bound_ranges {
                    let Some(value) = in_ranges.get(&br.range_uop.id).copied() else {
                        return IrConstructionSnafu {
                            details: format!(
                                "CALL {} bound variable '{}' references inactive schedule range {}",
                                kernel_id, br.var_name, br.range_uop.id
                            ),
                        }
                        .fail();
                    };
                    fixedvars.insert(br.var_name.clone(), value);
                }
                invocations.push(KernelInvocation { kernel_id: *kernel_id, fixedvars });
            }
        }
        sched_ptr += 1;
    }

    Ok(invocations)
}

fn analyze_callable_dependencies(callables: &[Arc<UOp>], root: &Arc<UOp>) -> Result<Vec<HashSet<usize>>> {
    // Build callable ID → index mapping
    let callable_idx: HashMap<u64, usize> = callables.iter().enumerate().map(|(i, c)| (c.id, i)).collect();
    // Build dependency edges from source-driven dependency extraction
    // (avoids global writer-union heuristics).
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

/// Schedule-level Range bound to a DEFINE_VAR via a CALL argument's
/// `Bind(DefineVar, Range)`. Each invocation of the wrapped CALL substitutes
/// the loop counter value into the kernel's variable.
#[derive(Clone, Debug)]
pub struct BoundRangeRef {
    /// Variable name (e.g., "range_0")
    pub var_name: String,
    /// RANGE UOp for this bound variable.
    pub range_uop: Arc<UOp>,
}

/// Linearized scheduling control op (internal to schedule construction).
///
/// The strict scheduler walks these as a small bytecode (Range/End drive a
/// loop counter, Call emits an invocation) — see `collect_kernel_invocations`.
/// Eager unrolling at pre-schedule time turns them into a flat
/// `Vec<KernelInvocation>`.
#[derive(Clone, Debug)]
enum LinearSchedOp {
    Range { range: Arc<UOp> },
    Call { kernel_id: u64 },
    End { range: Arc<UOp>, kernel_id: u64 },
}

/// One concrete kernel invocation: a kernel id + its loop-resolved variable bindings.
///
/// The schedule is a flat list of these (eagerly unrolled at
/// `create_pre_schedule` time); each element is an atomic kernel CALL with
/// concrete bindings.
#[derive(Clone, Debug)]
pub struct KernelInvocation {
    /// Kernel ID — looked up against the `PreScheduleItem.kernel.id` index.
    pub kernel_id: u64,
    /// Concrete `var_name → value` bindings produced by the surrounding loop
    /// counters at the moment of this invocation.
    pub fixedvars: HashMap<String, i64>,
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

    /// Names of variables in `fixedvars` whose values came from schedule-loop
    /// counters. User `var_vals` must not override these — see
    /// `collect_non_overridable_fixedvars`.
    pub loop_var_names: HashSet<String>,

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

/// Cached pre-schedule item.
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
    /// Schedule-level Range bindings (`BIND(DEFINE_VAR, RANGE)`) from CALL args.
    pub bound_ranges: Vec<BoundRangeRef>,
}

/// Cached pre-schedule artifact.
///
/// A flat list of kernel invocations with their concrete bindings. Outer
/// loops are eagerly unrolled at construction time, so there is no
/// schedule-level Range/End bytecode — just one entry per kernel call.
#[derive(Clone)]
pub struct PreSchedule {
    /// Per-kernel descriptor pool indexed by `kernel.id` from `KernelInvocation`.
    pub items: Vec<PreScheduleItem>,
    /// Flat sequence of kernel invocations after eager loop unrolling.
    pub invocations: Vec<KernelInvocation>,
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

    // Step 1.75: Compute the set of Range UOp IDs paired with `END(Call)` —
    // these are the schedule-level loop wrappers, identified structurally
    // (no axis_type filter).
    let callable_ids: HashSet<u64> = callables.iter().map(|c| c.id).collect();
    let scheduled_range_ids = collect_scheduled_range_ids(&transformed, &callable_ids);

    // Step 2: Build pre-schedule items (AST + sources + dependencies + bound ranges).
    let mut items = Vec::with_capacity(callables.len());
    for callable_uop in callables {
        let Op::Call { body, args, .. } = callable_uop.op() else {
            unreachable!("filtered to only call wrappers above")
        };
        let dependencies = dependency_ids_by_callable.get(&callable_uop.id).cloned().unwrap_or_default();
        let bound_ranges = collect_call_bound_ranges(&callable_uop, &scheduled_range_ids)?;
        items.push(PreScheduleItem {
            kernel: callable_uop.clone(),
            ast: body.clone(),
            sources: args.iter().cloned().collect(),
            dependencies,
            bound_ranges,
        });
    }

    // Step 3: Eagerly unroll outer loops into a flat list of kernel
    // invocations — each invocation carries the concrete `fixedvars` produced
    // by its enclosing loop counters.
    let invocations = collect_kernel_invocations(&transformed, &items, &scheduled_range_ids)?;

    // Output buffers in SINK source order.
    let output_buffer_uops: Vec<Arc<UOp>> = match transformed.op() {
        Op::Sink { sources, .. } => sources.iter().map(|src| src.buf_uop()).collect(),
        _ => vec![transformed.buf_uop()],
    };

    Ok(PreSchedule { items, invocations, output_buffer_uops })
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
            },
        );
    }

    let mut schedule = Vec::with_capacity(pre_schedule.invocations.len());
    for invocation in &pre_schedule.invocations {
        let Some(template) = templates.get(&invocation.kernel_id) else {
            return IrConstructionSnafu {
                details: format!("invocation references unknown kernel id {}", invocation.kernel_id),
            }
            .fail();
        };

        // Merge the kernel's user-Variable bindings (`base_fixedvars`) with
        // the loop-counter bindings produced at this iteration.
        let mut fixedvars = template.base_fixedvars.clone();
        fixedvars.extend(invocation.fixedvars.iter().map(|(k, v)| (k.clone(), *v)));
        let loop_var_names: HashSet<String> = invocation.fixedvars.keys().cloned().collect();

        schedule.push(ScheduleItem {
            kernel: template.kernel.clone(),
            ast: template.ast.clone(),
            buffers: template.buffers.clone(),
            buffer_uop_ids: template.buffer_uop_ids.clone(),
            fixedvars,
            loop_var_names,
            dependencies: template.dependencies.clone(),
            instance_dependencies: Vec::new(),
            alias_registered_ids: template.alias_registered_ids.clone(),
        });
    }

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
/// The first buffer's device determines the device for codegen/compilation and
/// output buffer allocation.
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
/// Output/intermediate buffers are allocated on the same device as the first
/// input buffer. Newly allocated buffers are tracked in `allocated_buffers`.
fn collect_callable_buffers(
    sources: &[Arc<UOp>],
    ast: &Arc<UOp>,
    input_buffers: &InputBuffers,
    allocated_buffers: &mut HashMap<u64, Buffer>,
) -> Result<CallableBuffers> {
    // Get target device from the first input buffer.
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
}

fn schedule_range_bounds(range: &Arc<UOp>) -> Result<(i64, i64)> {
    let Op::Range { .. } = range.op() else {
        return IrConstructionSnafu {
            details: format!("expected RANGE for schedule loop control, got {:?}", range.op()),
        }
        .fail();
    };

    let Some(vmin) = range.vmin().try_int() else {
        return IrConstructionSnafu {
            details: format!("schedule range vmin must be concrete integer, got {:?}", range.vmin()),
        }
        .fail();
    };
    let Some(vmax) = range.vmax().try_int() else {
        return IrConstructionSnafu {
            details: format!("schedule range vmax must be concrete integer, got {:?}", range.vmax()),
        }
        .fail();
    };
    if vmax < vmin {
        return IrConstructionSnafu { details: format!("invalid schedule range bounds: vmin={vmin}, vmax={vmax}") }
            .fail();
    }
    Ok((vmin, vmax))
}

/// Compute buffer size from the buffer definition's dtype.
///
/// Buffer size is embedded in the Ptr dtype by debuf() during rangeify
/// (`dtype.ptr(size=...)`).
fn compute_buffer_size(_ast: &Arc<UOp>, buffer_def: &Arc<UOp>) -> Result<usize> {
    // Extract size from Ptr dtype (set by debuf() in split_patterns.rs)
    match buffer_def.dtype() {
        DType::Ptr { size: Some(s), .. } => Ok(s),
        DType::Ptr { size: None, .. } => BufferPtrNoSizeSnafu.fail(),
        other => ExpectedPtrDtypeSnafu { context: "buffer_size", actual: other.clone() }.fail(),
    }
}
