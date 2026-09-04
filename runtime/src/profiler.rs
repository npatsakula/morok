//! Layered per-kernel execution profiling.
//!
//! Profiling is organized into four tiers, each adding more detail on top of the
//! one below; the per-item docs in this file refer to these tiers by number:
//!
//! - **Tier 1 — device time.** Per-kernel on-device execution time, taken from
//!   the GPU clock stamps the backend records around each dispatch
//!   ([`KernelProfile::gpu_or_wall`]). This is the baseline every profile carries.
//! - **Tier 2 — roofline.** Derived throughput: GFLOP/s and GB/s, computed from
//!   the Tier-1 device time and the static work estimates in
//!   [`KernelStaticInfo`].
//! - **Tier 3 — static occupancy.** VGPR/SGPR/LDS/scratch usage and the
//!   VGPR-limited occupancy, decoded from the compiled kernel descriptor without
//!   running anything ([`KernelResources`]).
//! - **Tier 4 — hardware counters.** AMD SQ performance counters (busy cycles,
//!   waves launched, VALU instructions issued) collected via PM4 perf-counter
//!   programming. This needs a stable GPU power state; when that is unavailable
//!   the run degrades to timing-only (Tiers 1–3).
//!
//! **Accumulate-and-min policy.** Repeated runs of the same plan are merged per
//! kernel by keeping the minimum device time ([`RunProfile::merge_min`]), which
//! is robust to outliers. The same policy backs both the
//! [`ExecutionPlan::profile`](crate::ExecutionPlan::profile) `iters` loop and the
//! Criterion `--profile-time` accumulator.
//!
//! **Entry points.** Profile a prepared plan with
//! [`ExecutionPlan::profile`](crate::ExecutionPlan::profile) or a tensor directly
//! with `Tensor::profile` (which wraps it). Behavior is configured
//! through [`ProfileOptions`] (and [`ProfileOptions::from_env`], reading
//! `SVOD_PMC` and `SVOD_PROFILE_ITERS`). The underlying per-kernel timing path is
//! [`ExecutionPlan::execute_profiled`](crate::ExecutionPlan::execute_profiled).
//! The library never prints; callers render a finished [`RunProfile`] with
//! [`RunProfile::render_table`].

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use svod_device::hcq::{CopyLeg, DeviceQueue, SemanticLinkedPlan};
use svod_device::{CounterSet, KernelResources, PmcCounter};
use svod_dtype::DeviceSpec;

use crate::kernel_cache::CachedKernel;

/// Measured duration for one command in a semantic linked plan. `copy_leg`
/// distinguishes the two commands emitted for a host-staged copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationTiming {
    pub operation: usize,
    pub copy_leg: Option<CopyLeg>,
    pub duration: Duration,
}

/// Critical-path measurements for one concrete device/queue lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneExecutionMetrics {
    pub lane: DeviceQueue,
    /// Completion time relative to the start of the linked plan.
    pub makespan: Duration,
    /// Time spent executing commands on this lane.
    pub busy: Duration,
    /// Lane-idle time forced by published cross-lane waits.
    pub wait: Duration,
    /// Busy time on this lane concurrent with at least one other lane.
    pub overlap: Duration,
}

/// Deterministic execution-lane simulation using the authoritative semantic
/// submissions and measured command durations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionLaneMetrics {
    pub makespan: Duration,
    pub busy: Duration,
    pub wait: Duration,
    /// Wall-clock time during which at least two lanes are busy.
    pub overlap: Duration,
    pub lanes: Vec<LaneExecutionMetrics>,
}

fn duration_from_nanos(nanos: u128) -> Duration {
    Duration::new((nanos / 1_000_000_000).min(u64::MAX as u128) as u64, (nanos % 1_000_000_000) as u32)
}

/// Compute makespan, busy, wait, and overlap from a semantic plan. Missing
/// command timings are treated as zero; this lets callers profile only the
/// command classes for which their backend exposes timestamps.
pub fn analyze_execution_lanes(plan: &SemanticLinkedPlan, timings: &[OperationTiming]) -> ExecutionLaneMetrics {
    #[derive(Default)]
    struct LaneState {
        end: u128,
        busy: u128,
        wait: u128,
        spans: Vec<(u128, u128)>,
    }

    let command_nanos = |operation: usize, copy_leg: Option<CopyLeg>| {
        timings
            .iter()
            .find(|timing| timing.operation == operation && timing.copy_leg == copy_leg)
            .map_or(0, |timing| timing.duration.as_nanos())
    };
    let mut states: std::collections::HashMap<DeviceQueue, LaneState> = std::collections::HashMap::new();
    let mut completions: std::collections::HashMap<(DeviceQueue, u64), u128> = std::collections::HashMap::new();

    for submission in plan.lanes() {
        let previous_end = states.get(&submission.lane).map_or(0, |state| state.end);
        let wait_until = submission
            .waits
            .iter()
            .filter_map(|wait| completions.get(&(wait.lane.clone(), wait.value)).copied())
            .max()
            .unwrap_or(0);
        let start = previous_end.max(wait_until);
        let busy =
            submission.commands.iter().map(|command| command_nanos(command.operation, command.copy_leg)).sum::<u128>();
        let end = start.saturating_add(busy);
        let state = states.entry(submission.lane.clone()).or_default();
        state.wait = state.wait.saturating_add(start.saturating_sub(previous_end));
        state.busy = state.busy.saturating_add(busy);
        state.end = end;
        if busy != 0 {
            state.spans.push((start, end));
        }
        completions.insert((submission.lane.clone(), submission.signal_value), end);
    }

    let mut all_spans = Vec::new();
    for (lane, state) in &states {
        all_spans.extend(state.spans.iter().map(|&span| (lane, span)));
    }
    let mut events = all_spans.iter().flat_map(|(_, span)| [(span.0, 1i32), (span.1, -1i32)]).collect::<Vec<_>>();
    events.sort_by_key(|&(time, delta)| (time, delta));
    let mut active = 0i32;
    let mut previous = 0u128;
    let mut overlap = 0u128;
    for (time, delta) in events {
        if active >= 2 {
            overlap = overlap.saturating_add(time.saturating_sub(previous));
        }
        active += delta;
        previous = time;
    }

    let mut lanes = states
        .iter()
        .map(|(lane, state)| {
            let mut intersections = Vec::new();
            for &own in &state.spans {
                for (other_lane, other) in &all_spans {
                    if *other_lane == lane {
                        continue;
                    }
                    let start = own.0.max(other.0);
                    let end = own.1.min(other.1);
                    if start < end {
                        intersections.push((start, end));
                    }
                }
            }
            intersections.sort_unstable();
            let mut merged: Vec<(u128, u128)> = Vec::new();
            for span in intersections {
                if let Some(last) = merged.last_mut()
                    && span.0 <= last.1
                {
                    last.1 = last.1.max(span.1);
                } else {
                    merged.push(span);
                }
            }
            LaneExecutionMetrics {
                lane: lane.clone(),
                makespan: duration_from_nanos(state.end),
                busy: duration_from_nanos(state.busy),
                wait: duration_from_nanos(state.wait),
                overlap: duration_from_nanos(merged.iter().map(|(start, end)| end - start).sum()),
            }
        })
        .collect::<Vec<_>>();
    lanes.sort_by_key(|metrics| {
        let queue = match metrics.lane.queue {
            svod_device::hcq::QueueKind::Compute(number) => (0, number),
            svod_device::hcq::QueueKind::Copy(number) => (1, number),
        };
        (metrics.lane.device.canonicalize(), queue)
    });

    ExecutionLaneMetrics {
        makespan: duration_from_nanos(states.values().map(|state| state.end).max().unwrap_or(0)),
        busy: duration_from_nanos(states.values().map(|state| state.busy).sum()),
        wait: duration_from_nanos(states.values().map(|state| state.wait).sum()),
        overlap: duration_from_nanos(overlap),
        lanes,
    }
}

/// Timestamp resources attached to one HCQ batch finalizer. Dispatch paths add
/// records in queue order; the terminal submission waits that same queue
/// timeline once, after which collection converts clocks and releases every
/// backend-owned handle together.
pub(crate) struct SubmissionProfileFinalizer {
    profiles: Vec<KernelProfile>,
    handles: Vec<Option<Arc<dyn svod_device::DispatchTimestamps>>>,
}

impl SubmissionProfileFinalizer {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self { profiles: Vec::with_capacity(capacity), handles: Vec::with_capacity(capacity) }
    }

    pub(crate) fn push(&mut self, profile: KernelProfile, handle: Option<Arc<dyn svod_device::DispatchTimestamps>>) {
        self.profiles.push(profile);
        self.handles.push(handle);
    }

    pub(crate) fn finish(
        mut self,
        synchronize: impl FnOnce() -> crate::error::Result<()>,
    ) -> crate::error::Result<Vec<KernelProfile>> {
        if self.handles.iter().any(Option::is_some) {
            synchronize()?;
        }
        for (profile, handle) in self.profiles.iter_mut().zip(&self.handles) {
            if let Some((start, end)) = handle.as_ref().and_then(|handle| handle.timestamps_ns()) {
                profile.gpu_start_ns = Some(start);
                profile.gpu_end_ns = Some(end);
            }
            profile.counters = handle.as_ref().and_then(|handle| handle.counters());
        }
        Ok(self.profiles)
    }
}

/// Per-kernel timing from a profiled execution.
///
/// Holds an `Arc<CachedKernel>` for zero-copy access to kernel metadata
/// (entry point, generated code, global/local size, variable names).
///
/// # Example
///
/// ```ignore
/// let plan = tensor.prepare()?;
/// let profiles = plan.execute_profiled()?;
///
/// for (i, p) in profiles.iter().enumerate() {
///     println!("{:4} {:>8.3}ms  {}  ({} bufs, {:?})",
///         i, p.wall.as_secs_f64() * 1000.0,
///         p.kernel.entry_point, p.num_buffers, p.device);
/// }
/// ```
pub struct KernelProfile {
    /// Compiled kernel (entry_point, code, global_size, local_size, var_names).
    /// Debug shows the entry point only — the code/program are not printable.
    pub kernel: Arc<CachedKernel>,
    /// Device this kernel executed on.
    pub device: DeviceSpec,
    /// Scope this dispatch is charged to, from the plan op that produced it.
    pub origin: Option<svod_ir::OriginId>,
    /// Every scope folded into this kernel.
    pub origins: svod_ir::OriginSet,
    /// Number of buffer arguments.
    pub num_buffers: usize,
    /// Host wall-clock around the dispatch submit. On async backends (GPU) this
    /// is mostly launch/submission overhead, NOT on-device execution time — for
    /// that use `gpu_start_ns`/`gpu_end_ns` (or [`Self::gpu_or_wall`]).
    pub wall: Duration,
    /// HW dispatch start/end on the GPU clock (ns), when the backend stamps
    /// dispatches ([`svod_device::DispatchTimestamps`]).
    pub gpu_start_ns: Option<u64>,
    pub gpu_end_ns: Option<u64>,
    /// Tier-2/3 static analysis (estimated flops/bytes + decoded GPU resources),
    /// populated by [`ExecutionPlan::profile`](crate::ExecutionPlan::profile) when
    /// `static_analysis` is set. `None` from [`ExecutionPlan::execute_profiled`].
    pub static_info: Option<KernelStaticInfo>,
    /// Tier-4 hardware performance counters, populated when PMC was enabled.
    pub counters: Option<CounterSet>,
}

/// Static per-kernel analysis: estimated work plus decoded GPU resources.
/// Computed without running the kernel (the estimates walk the kernel AST; the
/// resources are decoded from the compiled descriptor).
#[derive(Debug, Clone)]
pub struct KernelStaticInfo {
    /// Estimated floating-point ops, from the kernel AST. `None` when the
    /// estimate is unreliable — e.g. a hand-built kernel with unbounded symbolic
    /// ranges, where the AST walk saturates rather than forming a real count.
    pub est_flops: Option<u64>,
    /// Estimated bytes moved (distinct LOAD/STORE buffers).
    pub est_bytes: u64,
    /// Decoded GPU resource usage, when the backend exposes it.
    pub resources: Option<KernelResources>,
}

impl std::fmt::Debug for KernelProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelProfile")
            .field("kernel", &self.kernel.entry_point)
            .field("wall", &self.wall)
            .field("gpu_start_ns", &self.gpu_start_ns)
            .field("gpu_end_ns", &self.gpu_end_ns)
            .finish_non_exhaustive()
    }
}

impl KernelProfile {
    /// GPU execution time when stamped, else the host wall-clock fallback.
    pub fn gpu_or_wall(&self) -> Duration {
        match (self.gpu_start_ns, self.gpu_end_ns) {
            (Some(s), Some(e)) => Duration::from_nanos(e - s),
            _ => self.wall,
        }
    }
}

/// Which hardware counters to collect during a profiled run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum PmcSelection {
    /// No hardware counters (Tiers 1–3 only).
    #[default]
    None,
    /// The default set, [`PmcCounter::all`]: SQ busy cycles, waves launched, and
    /// VALU instructions issued.
    Default,
    /// An explicit counter list.
    Custom(Vec<PmcCounter>),
}

impl PmcSelection {
    /// Resolve to the concrete counter list (empty when disabled).
    pub fn counters(&self) -> Vec<PmcCounter> {
        match self {
            Self::None => Vec::new(),
            Self::Default => PmcCounter::all().to_vec(),
            Self::Custom(v) => v.clone(),
        }
    }

    /// Whether any counter is selected.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Options for [`ExecutionPlan::profile`](crate::ExecutionPlan::profile).
#[derive(Debug, Clone)]
pub struct ProfileOptions {
    /// Replays to run (clamped to at least one by `profile`); the per-kernel
    /// minimum device time is kept (robust to outliers).
    pub iters: u32,
    /// Collect Tier-2/3 static analysis (flops/bytes/resources). Cheap; on by default.
    pub static_analysis: bool,
    /// Hardware counter selection (Tier 4).
    pub counters: PmcSelection,
}

impl Default for ProfileOptions {
    fn default() -> Self {
        Self { iters: 1, static_analysis: true, counters: PmcSelection::None }
    }
}

impl ProfileOptions {
    /// Build from the `SVOD_PROFILE_ITERS` / `SVOD_PMC` env vars (defaults
    /// otherwise). This is the single place profiling env vars are read; callers
    /// that want explicit control construct [`ProfileOptions`] directly.
    pub fn from_env() -> Self {
        let mut o = Self::default();
        if let Ok(n) = std::env::var("SVOD_PROFILE_ITERS").unwrap_or_default().trim().parse::<u32>() {
            o.iters = n.max(1);
        }
        if let Ok(v) = std::env::var("SVOD_PMC") {
            o.counters = parse_pmc(&v);
        }
        o
    }
}

/// Parse a `SVOD_PMC` value: empty/`0` → none, `1` → default set, else a
/// comma-separated token list ([`PmcCounter::from_token`]).
pub(crate) fn parse_pmc(v: &str) -> PmcSelection {
    match v.trim() {
        "" | "0" => PmcSelection::None,
        "1" => PmcSelection::Default,
        list => {
            let counters: Vec<PmcCounter> = list.split(',').filter_map(PmcCounter::from_token).collect();
            if counters.is_empty() { PmcSelection::Default } else { PmcSelection::Custom(counters) }
        }
    }
}

/// Per-kernel-name aggregate over a profiled execution, sorted by total time
/// descending. Render with [`render_histogram`].
pub struct KernelAggregate {
    pub name: String,
    pub count: usize,
    pub total: Duration,
    pub mean: Duration,
}

/// Group profiles by entry point, sum GPU (or wall-fallback) durations.
pub fn aggregate_profiles(profiles: &[KernelProfile]) -> Vec<KernelAggregate> {
    let mut map: std::collections::HashMap<&str, (usize, Duration)> = std::collections::HashMap::new();
    for p in profiles {
        let e = map.entry(&p.kernel.entry_point).or_insert((0, Duration::ZERO));
        e.0 += 1;
        e.1 += p.gpu_or_wall();
    }
    let mut out: Vec<KernelAggregate> = map
        .into_iter()
        .map(|(name, (count, total))| KernelAggregate {
            name: name.to_string(),
            count,
            total,
            mean: total / count as u32,
        })
        .collect();
    out.sort_by_key(|p| std::cmp::Reverse(p.total));
    out
}

/// Multi-line histogram of the top-`n` kernels by total time.
pub fn render_histogram(profiles: &[KernelProfile], n: usize) -> String {
    let total: Duration = profiles.iter().map(KernelProfile::gpu_or_wall).sum();
    let stamped = profiles.iter().filter(|p| p.gpu_start_ns.is_some()).count();
    let mut s = format!(
        "{} dispatches ({} GPU-stamped), total {:.3} ms\n{:>10}  {:>5}  {:>9}  {:>5}  name\n",
        profiles.len(),
        stamped,
        total.as_secs_f64() * 1e3,
        "total ms",
        "count",
        "mean µs",
        "%",
    );
    for a in aggregate_profiles(profiles).into_iter().take(n) {
        let pct = 100.0 * a.total.as_secs_f64() / total.as_secs_f64().max(f64::EPSILON);
        s.push_str(&format!(
            "{:>10.3}  {:>5}  {:>9.1}  {:>5.1}  {}\n",
            a.total.as_secs_f64() * 1e3,
            a.count,
            a.mean.as_secs_f64() * 1e6,
            pct,
            a.name,
        ));
    }
    s
}

/// One stage of a profiled run: a named span owning the per-dispatch kernels of
/// ONE representative profiled execution (GPU-stamped when the backend supports
/// it), plus the host wall accumulated over the stage and an extensible metadata
/// bag. Host-only stages (no GPU work) carry empty `kernels`.
///
/// Model-agnostic by design: the stage identity is a free-form `name` (data, not
/// a typed enum), so any model populates the same shape and a generic UI /
/// histogram renders it uniformly. Stages are a flat, ordered list — any
/// grouping/hierarchy is a render-time concern, not stored here.
#[derive(Debug, Default)]
pub struct StageProfile {
    /// Stage identity, e.g. `"vad"`, `"mel"`, `"encoder"`, `"ctc_head"`.
    pub name: String,
    /// Host wall accumulated over the stage. On async GPUs this is mostly submit
    /// overhead; the on-device truth is in `kernels`.
    pub wall: Duration,
    /// Per-dispatch kernels of the profiled execution. Empty for host-only stages.
    pub kernels: Vec<KernelProfile>,
    /// Extensible per-stage metadata (rtf, chunk index, …). Keeps the format
    /// stable across models without schema churn; consumed as-is by the UI.
    pub meta: BTreeMap<String, String>,
}

impl StageProfile {
    /// A host-only stage (no GPU kernels).
    pub fn host(name: impl Into<String>, wall: Duration) -> Self {
        Self { name: name.into(), wall, kernels: Vec::new(), meta: BTreeMap::new() }
    }

    /// A GPU stage carrying one profiled execution's per-dispatch kernels.
    pub fn gpu(name: impl Into<String>, wall: Duration, kernels: Vec<KernelProfile>) -> Self {
        Self { name: name.into(), wall, kernels, meta: BTreeMap::new() }
    }

    /// Sum of GPU (or wall-fallback) time across the profiled execution.
    pub fn gpu_total(&self) -> Duration {
        self.kernels.iter().map(KernelProfile::gpu_or_wall).sum()
    }

    /// Top kernels by total time, aggregated by entry point.
    pub fn top(&self, n: usize) -> Vec<KernelAggregate> {
        let mut aggs = aggregate_profiles(&self.kernels);
        aggs.truncate(n);
        aggs
    }
}

/// A model-agnostic profile of one inference run: an ordered, flat list of named
/// stages. Any model emits this same shape, so a generic UI / the `Display`
/// histogram renders an arbitrary model's profile uniformly.
#[derive(Debug, Default)]
pub struct RunProfile {
    pub stages: Vec<StageProfile>,
}

impl RunProfile {
    /// Append a stage.
    pub fn push(&mut self, stage: StageProfile) {
        self.stages.push(stage);
    }

    /// First stage with the given name, if any.
    pub fn stage(&self, name: &str) -> Option<&StageProfile> {
        self.stages.iter().find(|s| s.name == name)
    }

    /// Merge another profiling pass of the SAME plan into this one, keeping each
    /// kernel's faster (min device-time) sample — and carrying that sample's
    /// counters/static analysis. Both must share stage + kernel ordering (extra
    /// stages/kernels in `other` are ignored); `self`'s stage metadata is kept.
    /// This is the single min-merge policy used to accumulate repeated passes
    /// (the `profile()` `iters` loop and the criterion `--profile-time` hook).
    pub fn merge_min(&mut self, other: RunProfile) {
        for (stage, incoming) in self.stages.iter_mut().zip(other.stages) {
            for (best, sample) in stage.kernels.iter_mut().zip(incoming.kernels) {
                if sample.gpu_or_wall() < best.gpu_or_wall() {
                    *best = sample;
                }
            }
        }
    }

    /// Rich per-kernel table: device time plus any populated Tier-2/3/4 metrics
    /// (GFLOP/s, GB/s, VGPR/SGPR/LDS, HW counters). Columns appear only when the
    /// underlying data is present, so a Tier-1-only run shows just timing.
    pub fn render_table(&self) -> String {
        let mut out = String::new();
        for s in &self.stages {
            if s.kernels.is_empty() {
                out.push_str(&format!("{}: wall {:.1} ms (host)\n", s.name, s.wall.as_secs_f64() * 1e3));
            } else {
                out.push_str(&render_stage_table(s));
            }
        }
        out
    }

    /// Fold another profile in, accumulating stages by name: matching stages
    /// sum their wall, concatenate kernels (so the histogram aggregates across
    /// runs), and merge metadata; new stage names are appended in order. Used
    /// to combine per-window profiles when transcribing a batch one window at
    /// a time.
    ///
    /// Same-named stages SUM their `wall`. A model that pre-accumulates a
    /// stage's `wall` to a whole-run total (rather than this window's slice)
    /// must therefore emit one profile for the run and not also rely on this
    /// per-window merge, or the total double-counts.
    pub fn merge(&mut self, other: RunProfile) {
        for stage in other.stages {
            match self.stages.iter_mut().find(|s| s.name == stage.name) {
                Some(existing) => {
                    existing.wall += stage.wall;
                    existing.kernels.extend(stage.kernels);
                    existing.meta.extend(stage.meta);
                }
                None => self.stages.push(stage),
            }
        }
    }
}

/// One aggregated table row (kernels grouped by entry point).
struct TableRow {
    name: String,
    count: usize,
    total: Duration,
    /// Summed flop estimate, or `None` if any dispatch's estimate was unreliable.
    flops: Option<u64>,
    bytes: u64,
    resources: Option<KernelResources>,
    counters: BTreeMap<PmcCounter, u64>,
    has_static: bool,
}

/// Format a roofline rate (GFLOP/s or GB/s) in giga-units/s; `-` when the count
/// is unknown, zero, or the elapsed time is non-positive.
fn roofline_rate(count: Option<u64>, secs: f64) -> String {
    match count {
        Some(c) if c > 0 && secs > 0.0 => format!("{:.1}", c as f64 / secs / 1e9),
        _ => "-".into(),
    }
}

/// Aggregate a stage's per-dispatch kernels by entry point and format a table
/// whose columns adapt to which tiers were collected.
fn render_stage_table(s: &StageProfile) -> String {
    let mut rows: Vec<TableRow> = Vec::new();
    let mut index: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for k in &s.kernels {
        let i = *index.entry(&k.kernel.entry_point).or_insert_with(|| {
            rows.push(TableRow {
                name: k.kernel.entry_point.clone(),
                count: 0,
                total: Duration::ZERO,
                flops: Some(0),
                bytes: 0,
                resources: None,
                counters: BTreeMap::new(),
                has_static: false,
            });
            rows.len() - 1
        });
        let r = &mut rows[i];
        r.count += 1;
        r.total += k.gpu_or_wall();
        if let Some(si) = &k.static_info {
            r.has_static = true;
            // Sum flops only while every dispatch had a reliable estimate.
            r.flops = match (r.flops, si.est_flops) {
                (Some(a), Some(b)) => Some(a.saturating_add(b)),
                _ => None,
            };
            r.bytes = r.bytes.saturating_add(si.est_bytes);
            if r.resources.is_none() {
                r.resources = si.resources;
            }
        }
        if let Some(cs) = &k.counters {
            for (&c, &v) in &cs.values {
                *r.counters.entry(c).or_insert(0) += v;
            }
        }
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.total));

    let any_static = rows.iter().any(|r| r.has_static);
    let any_res = rows.iter().any(|r| r.resources.is_some());
    let mut counter_cols: Vec<PmcCounter> = rows.iter().flat_map(|r| r.counters.keys().copied()).collect();
    counter_cols.sort();
    counter_cols.dedup();

    let mut header: Vec<String> = vec!["name".into(), "cnt".into(), "total ms".into(), "mean µs".into(), "%".into()];
    if any_static {
        header.push("GFLOP/s".into());
        header.push("GB/s".into());
    }
    if any_res {
        header.extend(["VGPR".into(), "SGPR".into(), "LDS".into(), "scratch".into(), "occ%".into()]);
    }
    for c in &counter_cols {
        header.push(c.token().into());
    }

    let grand: Duration = rows.iter().map(|r| r.total).sum();
    let mut body: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for r in &rows {
        let secs = r.total.as_secs_f64();
        let pct = 100.0 * secs / grand.as_secs_f64().max(f64::EPSILON);
        let mut cells = vec![
            r.name.clone(),
            r.count.to_string(),
            format!("{:.3}", secs * 1e3),
            format!("{:.1}", secs * 1e6 / r.count as f64),
            format!("{pct:.1}"),
        ];
        if any_static {
            cells.push(roofline_rate(r.flops, secs));
            cells.push(roofline_rate(Some(r.bytes), secs));
        }
        if any_res {
            match r.resources {
                Some(res) => {
                    let occ = res.occupancy.map(|o| format!("{:.0}", o * 100.0)).unwrap_or_else(|| "-".into());
                    cells.extend([
                        res.vgprs.to_string(),
                        res.sgprs.to_string(),
                        res.lds_bytes.to_string(),
                        res.scratch_bytes.to_string(),
                        occ,
                    ]);
                }
                None => cells.extend(["-".into(), "-".into(), "-".into(), "-".into(), "-".into()]),
            }
        }
        for c in &counter_cols {
            cells.push(r.counters.get(c).map(u64::to_string).unwrap_or_else(|| "-".into()));
        }
        body.push(cells);
    }

    let mut s_out = format!("{}: {} dispatches, GPU {:.3} ms\n", s.name, s.kernels.len(), grand.as_secs_f64() * 1e3);
    s_out.push_str(&fmt_columns(&header, &body));
    s_out
}

/// Format a table: column 0 left-aligned, the rest right-aligned, padded to the
/// widest cell per column.
fn fmt_columns(header: &[String], rows: &[Vec<String>]) -> String {
    let mut w: Vec<usize> = header.iter().map(String::len).collect();
    for r in rows {
        for (i, c) in r.iter().enumerate() {
            w[i] = w[i].max(c.len());
        }
    }
    let line = |cells: &[String]| -> String {
        let mut s = String::new();
        for (i, c) in cells.iter().enumerate() {
            if i == 0 {
                s.push_str(&format!("{:<width$}  ", c, width = w[i]));
            } else {
                s.push_str(&format!("{:>width$}  ", c, width = w[i]));
            }
        }
        s.push('\n');
        s
    };
    let mut out = line(header);
    for r in rows {
        out.push_str(&line(r));
    }
    out
}

impl std::fmt::Display for RunProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for s in &self.stages {
            if s.kernels.is_empty() {
                writeln!(f, "{}: wall {:.1} ms", s.name, s.wall.as_secs_f64() * 1e3)?;
            } else {
                writeln!(
                    f,
                    "{}: wall {:.1} ms, profiled exec GPU {:.3} ms\n{}",
                    s.name,
                    s.wall.as_secs_f64() * 1e3,
                    s.gpu_total().as_secs_f64() * 1e3,
                    render_histogram(&s.kernels, 20),
                )?;
            }
        }
        Ok(())
    }
}
