//! Memory planner for buffer reuse optimization.
//!
//! This module implements liveness-based memory planning following Tinygrad's approach.
//! The memory planner analyzes buffer lifetimes across the schedule and reuses buffers
//! with non-overlapping lifetimes, reducing memory consumption.
//!
//! # Algorithm
//!
//! 1. **Liveness Analysis**: Track first/last appearance of each buffer in schedule
//! 2. **Event Timeline**: Create sorted alloc/free events (frees before allocs at same step)
//! 3. **Pool-Based Allocation**: Reuse buffers by (device, dtype, size) key
//! 4. **Apply Replacements**: Map logical buffers to physical buffers

mod tlsf;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use morok_device::Buffer;
use morok_dtype::{DType, DeviceSpec};
use morok_ir::{Op, UOp};
use tracing::{debug, trace};

use crate::schedule::Schedule;

/// Minimum block size for buffer pooling (256-byte alignment, matching tinygrad).
const MIN_BLOCK_SIZE: usize = 256;

/// Selects the buffer-allocation strategy used by the planner entrypoint.
///
/// - `Disabled` short-circuits the planner and emits no replacements.
/// - `Remap` runs liveness-based pool reuse: groups buffers by
///   `(device, dtype, rounded_size)` and lets disjoint-lifetime buffers
///   share an underlying allocation.
/// - `Arena` packs all plannable buffers into one or two large
///   per-`(device, copy-lane)` arenas using a TLSF allocator and rewrites
///   each logical buffer as a `Buffer::view` into its lane's arena
///   (tinygrad parity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannerMode {
    /// Skip the planner entirely. Each `Buffer` keeps its original allocation
    /// and is freed by lazy `Drop`. Useful for memory-debugging baselines.
    Disabled,
    /// Liveness-based pool reuse: groups buffers by
    /// `(device, dtype, rounded_size)` and lets disjoint-lifetime buffers
    /// share an underlying allocation via `Arc<Buffer>` swap.
    Remap,
    /// Tinygrad-style packing: pack every plannable buffer into one or two
    /// per-`(device, copy-lane)` arenas using a TLSF allocator and rewrite
    /// each logical buffer as a fresh `Buffer::view` into its lane's arena.
    Arena,
}

/// Pure parser for the `MOROK_MEMORY_PLANNER` env var, exposed for testing.
///
/// Default (env unset) is [`PlannerMode::Arena`], matching tinygrad's
/// `NO_MEMORY_PLANNER=0` default — the arena planner runs unless the user
/// explicitly opts out. `remap` / `pool` keep the older liveness-based pool
/// reuse for parity with the previous default if a workload regresses.
pub fn parse_mode(raw: Option<&str>) -> PlannerMode {
    let Some(raw) = raw else {
        return PlannerMode::Arena;
    };
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "0" | "off" | "none" | "disabled" => PlannerMode::Disabled,
        "remap" | "pool" => PlannerMode::Remap,
        // "1" | "on" | "arena" | "" or any unrecognized → Arena (tinygrad default)
        _ => PlannerMode::Arena,
    }
}

/// Read `MOROK_MEMORY_PLANNER` from the environment and resolve to a [`PlannerMode`].
pub fn mode_from_env() -> PlannerMode {
    parse_mode(std::env::var("MOROK_MEMORY_PLANNER").ok().as_deref())
}

type LogicalBufferView = (usize, usize, DType, Vec<usize>);

/// Round up to the nearest multiple of block_size.
#[inline]
fn round_up(size: usize, block_size: usize) -> usize {
    size.div_ceil(block_size) * block_size
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Key for buffer pooling - groups buffers that can be reused interchangeably.
///
/// Buffer reuse is shape-agnostic: codegen reads logical shape from the UOp
/// graph, runtime dispatch passes raw `*mut u8` pointers, and the planner
/// skips output buffers (the only consumers of `Buffer::shape()` via
/// `as_array`/`as_array_mut`). Two non-output buffers with the same
/// `(device, dtype, rounded_size)` are interchangeable storage regardless
/// of their logical shapes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferPoolKey {
    /// Device where buffer is allocated.
    pub device: DeviceSpec,
    /// Data type of buffer elements.
    pub dtype: DType,
    /// Buffer size in bytes (rounded up to MIN_BLOCK_SIZE).
    pub size: usize,
}

/// Liveness information for a buffer.
#[derive(Debug, Clone)]
pub struct BufferLiveness {
    /// Index of first schedule step that uses this buffer.
    pub first_appearance: usize,
    /// Index of last schedule step that uses this buffer.
    pub last_appearance: usize,
    /// Pool key for buffer grouping.
    pub pool_key: BufferPoolKey,
    /// Representative logical buffer for this allocation ID.
    pub prototype: Buffer,
}

/// Buffer allocation/deallocation event for timeline scheduling.
#[derive(Debug, Clone)]
struct BufferEvent {
    /// Schedule item index when this event occurs.
    timestep: usize,
    /// True for allocation, false for deallocation.
    is_alloc: bool,
    /// Physical buffer allocation identifier.
    buffer_id: u64,
}

/// Schedule-order dependency introduced by a physical buffer reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReuseDependency {
    /// Schedule item that last uses the old logical buffer occupying the storage.
    pub predecessor_step: usize,
    /// Schedule item that first uses the new logical buffer reusing that storage.
    pub successor_step: usize,
}

struct ReusableBuffer {
    buffer: Buffer,
    released_by_step: usize,
}

/// Collected planner inputs derived from schedule traversal.
struct PlannerInput {
    /// Liveness keyed by physical buffer allocation ID.
    liveness: HashMap<u64, BufferLiveness>,
    /// Logical schedule slots that are eligible for replacement.
    occurrences: Vec<(BufferKey, u64)>,
}

/// Key to identify a buffer within a schedule.
///
/// We use (kernel_index, buffer_index) because the same UOp ID might appear
/// in multiple kernels due to buffer sharing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferKey {
    /// Index of kernel in the schedule.
    pub kernel_idx: usize,
    /// Index of buffer within that kernel's buffer list.
    pub buffer_idx: usize,
}

/// Result of memory planning.
#[derive(Debug)]
pub struct MemoryPlannerResult {
    /// Mapping from (kernel_idx, buffer_idx) to replacement buffer.
    /// Only contains entries for buffers that were replaced.
    pub buffer_replace: HashMap<BufferKey, Buffer>,
    /// Total memory saved through buffer reuse (in bytes).
    pub memory_saved: usize,
    /// Number of buffers that were reused.
    pub buffers_reused: usize,
    /// Execution ordering constraints required by reuse decisions.
    pub reuse_dependencies: Vec<ReuseDependency>,
}

// ============================================================================
// LIVENESS ANALYSIS
// ============================================================================

/// Analyze buffer liveness across the schedule.
///
/// Tracks first and last appearance of each buffer, skipping:
/// - Already allocated buffers (inputs)
/// - Output buffers
/// - Transfer operations
fn collect_noopt_buffer_ids(schedule: &Schedule) -> HashSet<u64> {
    // Alias detection groups views/buffers that share the same underlying
    // storage. Keying by `Buffer::id()` would miss views (since each view
    // mints a fresh handle id); keying by `storage_id()` correctly groups
    // every view of one allocation under one bucket.
    let mut by_storage: HashMap<u64, HashSet<LogicalBufferView>> = HashMap::new();
    let mut masked_store_ids = HashSet::new();
    for item in schedule {
        for buffer in &item.buffers {
            by_storage.entry(buffer.storage_id().0).or_default().insert((
                buffer.offset(),
                buffer.size(),
                buffer.dtype(),
                buffer.shape().to_vec(),
            ));
        }

        let uop_id_to_buffer_id: HashMap<u64, u64> =
            item.buffer_uop_ids.iter().copied().zip(item.buffers.iter().map(|b| b.id().0)).collect();
        for node in item.ast.toposort() {
            let Op::Store { index, .. } = node.op() else {
                continue;
            };
            collect_masked_store_buffer_ids(index, &uop_id_to_buffer_id, &mut masked_store_ids);
        }
    }

    // Map aliased storage ids back to handle ids — every Buffer in the
    // schedule whose storage has multiple distinct views is non-plannable.
    let aliased_storages: HashSet<u64> =
        by_storage.into_iter().filter_map(|(sid, views)| (views.len() > 1).then_some(sid)).collect();
    let aliased_ids = schedule.iter().flat_map(|item| {
        item.buffers
            .iter()
            .filter(|b| aliased_storages.contains(&b.storage_id().0))
            .map(|b| b.id().0)
            .collect::<Vec<_>>()
    });

    schedule
        .iter()
        .filter(|item| !matches!(item.ast.op(), Op::Sink { .. }))
        .flat_map(|item| item.buffers.iter().map(|b| b.id().0))
        .chain(aliased_ids)
        .chain(masked_store_ids)
        .collect()
}

fn collect_masked_store_buffer_ids(
    index: &Arc<UOp>,
    uop_id_to_buffer_id: &HashMap<u64, u64>,
    masked_store_ids: &mut HashSet<u64>,
) {
    match index.op() {
        Op::Index { buffer, gate: Some(_), .. } => {
            if let Some(buffer_id) = uop_id_to_buffer_id.get(&buffer.buf_uop().id) {
                masked_store_ids.insert(*buffer_id);
            }
        }
        Op::Index { .. } => {}
        other => {
            for child in other.children() {
                collect_masked_store_buffer_ids(child, uop_id_to_buffer_id, masked_store_ids);
            }
        }
    }
}

fn should_skip_buffer(buffer: &Buffer, output_buffer_ids: &HashSet<u64>, noopt_buffer_ids: &HashSet<u64>) -> bool {
    // Phase 1 planner only remaps full logical buffers. View/offset buffers require
    // an alias-preserving remap pass (planned for arena phase).
    buffer.allocator().device_spec().is_disk()
        || buffer.offset() != 0
        || buffer.is_allocated()
        || output_buffer_ids.contains(&buffer.id().0)
        || noopt_buffer_ids.contains(&buffer.id().0)
}

fn analyze_liveness(schedule: &Schedule, output_buffer_ids: &HashSet<u64>) -> PlannerInput {
    let noopt_buffer_ids = collect_noopt_buffer_ids(schedule);
    let mut liveness: HashMap<u64, BufferLiveness> = HashMap::new();
    let mut occurrences: Vec<(BufferKey, u64)> = Vec::new();

    for (step_idx, item) in schedule.iter().enumerate() {
        for (buf_idx, buffer) in item.buffers.iter().enumerate() {
            let key = BufferKey { kernel_idx: step_idx, buffer_idx: buf_idx };
            let buf_id = buffer.id().0;

            if should_skip_buffer(buffer, output_buffer_ids, &noopt_buffer_ids) {
                trace!(step_idx, buf_idx, buffer_id = buf_id, "skipping buffer in memory planner");
                continue;
            }

            occurrences.push((key, buf_id));

            let pool_key = BufferPoolKey {
                device: buffer.allocator().device_spec(),
                dtype: buffer.dtype(),
                size: round_up(buffer.size(), MIN_BLOCK_SIZE),
            };

            liveness
                .entry(buf_id)
                .and_modify(|info| {
                    info.first_appearance = info.first_appearance.min(step_idx);
                    info.last_appearance = info.last_appearance.max(step_idx);
                })
                .or_insert_with(|| BufferLiveness {
                    first_appearance: step_idx,
                    last_appearance: step_idx,
                    pool_key,
                    prototype: buffer.clone(),
                });
        }
    }

    debug!(num_optimizable = liveness.len(), "liveness analysis complete");

    PlannerInput { liveness, occurrences }
}

// ============================================================================
// EVENT TIMELINE
// ============================================================================

/// Build sorted event timeline from liveness information.
///
/// Events are sorted by (timestep, is_alloc) so that:
/// - Earlier timesteps come first
/// - At the same timestep, frees (is_alloc=false) come before allocs (is_alloc=true)
///
/// This ordering allows immediate reuse of freed buffers.
fn build_event_timeline(liveness: &HashMap<u64, BufferLiveness>) -> Vec<BufferEvent> {
    let mut events = Vec::with_capacity(liveness.len() * 2);

    for (&buf_id, info) in liveness {
        // Allocation event at first appearance
        events.push(BufferEvent { timestep: info.first_appearance, is_alloc: true, buffer_id: buf_id });

        // Deallocation event after last appearance
        events.push(BufferEvent { timestep: info.last_appearance + 1, is_alloc: false, buffer_id: buf_id });
    }

    // Sort by (timestep, is_alloc) - false < true ensures frees before allocs
    events.sort_by_key(|e| (e.timestep, e.is_alloc, e.buffer_id));

    events
}

// ============================================================================
// POOL-BASED ALLOCATION
// ============================================================================

/// Process events and compute buffer replacements using pool-based allocation.
///
/// For each allocation event:
/// - Try to reuse a buffer from the pool with matching key
/// - If no match, the buffer keeps its original allocation
///
/// For each deallocation event:
/// - Return the buffer to the pool for future reuse
fn process_events(
    events: &[BufferEvent],
    liveness: &HashMap<u64, BufferLiveness>,
    occurrences: &[(BufferKey, u64)],
) -> (HashMap<BufferKey, Buffer>, usize, usize, Vec<ReuseDependency>) {
    let mut free_pools: HashMap<BufferPoolKey, Vec<ReusableBuffer>> = HashMap::new();
    let mut memory_saved: usize = 0;
    let mut buffers_reused: usize = 0;
    let mut reuse_dependencies = Vec::new();
    let mut chosen_by_id: HashMap<u64, Buffer> = HashMap::new();

    // Track live assignment during timeline simulation.
    let mut active_buffers: HashMap<u64, Buffer> = HashMap::new();

    for event in events {
        let info = match liveness.get(&event.buffer_id) {
            Some(info) => info,
            None => continue,
        };
        let pool_key = &info.pool_key;

        if event.is_alloc {
            // Try to reuse a buffer from the pool
            if let Some(pool) = free_pools.get_mut(pool_key)
                && let Some(reused) = pool.pop()
            {
                trace!(timestep = event.timestep, reused_buffer_id = reused.buffer.id().0, "reusing buffer from pool");

                reuse_dependencies.push(ReuseDependency {
                    predecessor_step: reused.released_by_step,
                    successor_step: event.timestep,
                });
                chosen_by_id.insert(event.buffer_id, reused.buffer.clone());
                active_buffers.insert(event.buffer_id, reused.buffer);
                memory_saved += pool_key.size;
                buffers_reused += 1;
                continue;
            }

            // No reuse - use original buffer
            chosen_by_id.insert(event.buffer_id, info.prototype.clone());
            active_buffers.insert(event.buffer_id, info.prototype.clone());
        } else {
            // Deallocation - return buffer to pool
            if let Some(buffer) = active_buffers.remove(&event.buffer_id) {
                free_pools
                    .entry(pool_key.clone())
                    .or_default()
                    .push(ReusableBuffer { buffer, released_by_step: info.last_appearance });
            }
        }
    }

    let mut buffer_replace: HashMap<BufferKey, Buffer> = HashMap::new();
    for (key, buf_id) in occurrences {
        if let Some(chosen) = chosen_by_id.get(buf_id)
            && chosen.id().0 != *buf_id
        {
            buffer_replace.insert(*key, chosen.clone());
        }
    }

    (buffer_replace, memory_saved, buffers_reused, reuse_dependencies)
}

// ============================================================================
// ARENA-BASED ALLOCATION
// ============================================================================

/// Per-`(device, lane)` arena identifier. The `bool` separates the copy lane
/// (`true`) from the compute lane (`false`); this mirrors tinygrad's lane
/// keying and prevents introducing copy→compute→copy dependencies that
/// would force serialization.
type LaneKey = (DeviceSpec, bool);

/// Tinygrad-style arena planner: replaces every plannable buffer with a
/// `Buffer::view` into a per-lane arena allocated by [`tlsf::TlsfAllocator`].
///
/// Tinygrad rewrites the UOp graph to swap each `BUFFER` for a
/// `BUFFER_VIEW(arena, ...)`; Morok achieves the same effect at runtime by
/// populating [`MemoryPlannerResult::buffer_replace`] with arena views, which
/// the existing [`apply_buffer_replacements`] then swaps into the schedule.
fn memory_plan_arena(schedule: &Schedule, output_buffer_ids: &HashSet<u64>) -> MemoryPlannerResult {
    let empty_result = || MemoryPlannerResult {
        buffer_replace: HashMap::new(),
        memory_saved: 0,
        buffers_reused: 0,
        reuse_dependencies: Vec::new(),
    };

    let planner_input = analyze_liveness(schedule, output_buffer_ids);
    let liveness = planner_input.liveness;
    if liveness.is_empty() {
        return empty_result();
    }

    // Identify copy-lane buffers: any plannable buffer that appears as an
    // argument to a Copy schedule item.
    let mut copy_bufs: HashSet<u64> = HashSet::new();
    for item in schedule {
        let runtime_ast = crate::realize::runtime_effect_ast(&item.ast);
        if !matches!(runtime_ast.op(), Op::Copy { .. }) {
            continue;
        }
        for buffer in &item.buffers {
            let id = buffer.id().0;
            if liveness.contains_key(&id) {
                copy_bufs.insert(id);
            }
        }
    }

    let lane_key = |id: u64| -> LaneKey {
        let info = &liveness[&id];
        (info.prototype.allocator().device_spec(), copy_bufs.contains(&id))
    };

    // `buf_hold`: copy buffers stay live past their last appearance to avoid
    // clobbering before downstream copies finish.
    let buf_hold: HashMap<u64, usize> = copy_bufs
        .iter()
        .map(|&id| {
            let info = &liveness[&id];
            (id, info.last_appearance - info.first_appearance + 1)
        })
        .collect();

    // Per-buffer rounded size and byte size: round to `block_size` so the
    // TLSF allocator's bucket math stays correct.
    let nbytes_rounded: HashMap<u64, usize> =
        liveness.iter().map(|(&id, info)| (id, round_up(info.prototype.size(), MIN_BLOCK_SIZE))).collect();

    // Build event timeline with copy-lane hold extension on free events.
    let mut events: Vec<BufferEvent> = Vec::with_capacity(liveness.len() * 2);
    for (&id, info) in &liveness {
        events.push(BufferEvent { timestep: info.first_appearance, is_alloc: true, buffer_id: id });
        events.push(BufferEvent {
            timestep: info.last_appearance + 1 + buf_hold.get(&id).copied().unwrap_or(0),
            is_alloc: false,
            buffer_id: id,
        });
    }
    events.sort_by_key(|e| (e.timestep, e.is_alloc, e.buffer_id));

    // Per-lane TLSF allocators. Generous size budget = 2 × Σ(rounded sizes)
    // so even worst-case fragmentation can fit.
    let total_bytes: usize = nbytes_rounded.values().sum();
    let arena_budget = total_bytes.saturating_mul(2).max(MIN_BLOCK_SIZE);
    let mut tlsfs: HashMap<LaneKey, tlsf::TlsfAllocator> = HashMap::new();
    let mut offsets: HashMap<u64, usize> = HashMap::new();
    let mut peaks: HashMap<LaneKey, usize> = HashMap::new();
    // Track ranges freed within this lane so a later allocator-overlap
    // becomes an explicit `ReuseDependency`. We record `(offset, end, last_step)`
    // for every free; on alloc, any live entry whose `[offset, end)` overlaps
    // the new alloc emits a dep.
    let mut freed_ranges: HashMap<LaneKey, Vec<(usize, usize, usize)>> = HashMap::new();
    let mut reuse_dependencies: Vec<ReuseDependency> = Vec::new();

    for event in &events {
        let lane = lane_key(event.buffer_id);
        let info = &liveness[&event.buffer_id];
        let alloc =
            tlsfs.entry(lane.clone()).or_insert_with(|| tlsf::TlsfAllocator::new(arena_budget, 0, MIN_BLOCK_SIZE, 32));
        if event.is_alloc {
            let req = nbytes_rounded[&event.buffer_id];
            let off = match alloc.alloc(req, 1) {
                Ok(o) => o,
                Err(e) => {
                    tracing::warn!(?e, "arena planner: TLSF alloc failed; skipping arena rewrite");
                    return empty_result();
                }
            };
            offsets.insert(event.buffer_id, off);
            // Peak reflects actual byte usage (`buf.arg * itemsize`), not bucket-rounded size.
            let used_end = off + info.prototype.size();
            let peak = peaks.entry(lane.clone()).or_insert(0);
            if used_end > *peak {
                *peak = used_end;
            }
            // Emit reuse dependencies for any freed range this alloc
            // overlaps — they share storage now that the offset is reused.
            let alloc_end = off + req;
            if let Some(ranges) = freed_ranges.get(&lane) {
                for &(prev_off, prev_end, prev_last_step) in ranges {
                    let overlaps = off < prev_end && prev_off < alloc_end;
                    if overlaps {
                        reuse_dependencies.push(ReuseDependency {
                            predecessor_step: prev_last_step,
                            successor_step: info.first_appearance,
                        });
                    }
                }
            }
            // Drop freed ranges that this alloc overlaps. Their ReuseDependency
            // edges were just emitted above, and a future overlapping alloc
            // should depend on the *current* allocation's last appearance, not
            // the older eclipsed range — leaving them in would re-emit
            // redundant deps. The retain predicate keeps only ranges fully
            // disjoint from `[off, alloc_end)`.
            if let Some(ranges) = freed_ranges.get_mut(&lane) {
                ranges.retain(|&(o, e, _)| o >= alloc_end || e <= off);
            }
        } else if let Some(off) = offsets.get(&event.buffer_id).copied() {
            let req = nbytes_rounded[&event.buffer_id];
            if let Err(e) = alloc.free(off) {
                tracing::warn!(?e, "arena planner: TLSF free failed; skipping arena rewrite");
                return empty_result();
            }
            freed_ranges.entry(lane).or_default().push((off, off + req, info.last_appearance));
        }
    }

    // Allocate one arena buffer per lane, sized to the lane's peak. Precompute a
    // lane→prototype map so we don't re-scan `liveness` once per lane.
    let mut lane_proto: HashMap<LaneKey, Buffer> = HashMap::with_capacity(peaks.len());
    for (&id, info) in &liveness {
        lane_proto.entry(lane_key(id)).or_insert_with(|| info.prototype.clone());
    }
    let mut arenas: HashMap<LaneKey, Buffer> = HashMap::new();
    for (lane, &peak) in &peaks {
        if peak == 0 {
            continue;
        }
        let arena_size = round_up(peak, MIN_BLOCK_SIZE);
        let prototype = lane_proto.get(lane).expect("every populated lane must have a prototype");
        let arena = Buffer::new(
            prototype.allocator_arc(),
            morok_dtype::DType::UInt8,
            vec![arena_size],
            morok_device::allocator::BufferOptions::default(),
        );
        arenas.insert(lane.clone(), arena);
    }

    // Build buffer_replace by viewing each plannable buffer's slice of its
    // lane's arena. `Buffer::view` mints a fresh handle id per view (Path Y),
    // so disjoint views naturally appear as independent buffers to the
    // hazard model.
    let mut buffer_replace: HashMap<BufferKey, Buffer> = HashMap::new();
    let mut buffers_reused = 0usize;
    for (key, buf_id) in &planner_input.occurrences {
        let Some(&offset) = offsets.get(buf_id) else {
            continue;
        };
        let Some(arena) = arenas.get(&lane_key(*buf_id)) else {
            continue;
        };
        let info = &liveness[buf_id];
        let byte_size = info.prototype.size();
        let view = match arena.view(offset, byte_size) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(?e, "arena planner: view failed; skipping rewrite for one slot");
                continue;
            }
        };
        buffer_replace.insert(*key, view);
        buffers_reused += 1;
    }

    let arena_total: usize = peaks.values().map(|&p| round_up(p, MIN_BLOCK_SIZE)).sum();
    let memory_saved = total_bytes.saturating_sub(arena_total);

    debug!(
        buffers_planned = liveness.len(),
        buffers_replaced = buffers_reused,
        memory_saved_bytes = memory_saved,
        arena_count = arenas.len(),
        "arena memory planner complete"
    );

    MemoryPlannerResult { buffer_replace, memory_saved, buffers_reused, reuse_dependencies }
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

/// Run memory planner on a schedule.
///
/// Analyzes buffer lifetimes and identifies opportunities for buffer reuse.
/// Returns a mapping from logical buffers to physical buffers.
///
/// # Arguments
///
/// * `schedule` - The execution schedule to optimize
/// * `output_buffer_ids` - IDs of output buffers that must not be reused
/// * `mode` - Selects the planner strategy. [`PlannerMode::Disabled`] returns
///   an empty result without analyzing the schedule. [`PlannerMode::Remap`]
///   runs liveness-based pool reuse. [`PlannerMode::Arena`] runs the
///   tinygrad-style arena packing pass via [`memory_plan_arena`].
///
/// # Returns
///
/// `MemoryPlannerResult` containing buffer replacements and statistics.
#[allow(rustdoc::private_intra_doc_links)]
pub fn memory_planner(schedule: &Schedule, output_buffer_ids: &HashSet<u64>, mode: PlannerMode) -> MemoryPlannerResult {
    let empty_result = || MemoryPlannerResult {
        buffer_replace: HashMap::new(),
        memory_saved: 0,
        buffers_reused: 0,
        reuse_dependencies: Vec::new(),
    };

    if matches!(mode, PlannerMode::Disabled) {
        return empty_result();
    }

    if schedule.is_empty() {
        return empty_result();
    }

    if matches!(mode, PlannerMode::Arena) {
        return memory_plan_arena(schedule, output_buffer_ids);
    }

    // Phase 1: Liveness analysis
    let planner_input = analyze_liveness(schedule, output_buffer_ids);
    let liveness = planner_input.liveness;

    if liveness.is_empty() {
        debug!("no optimizable buffers found");
        return empty_result();
    }

    // Phase 2: Build event timeline
    let events = build_event_timeline(&liveness);

    // Phase 3: Process events and compute replacements
    let (buffer_replace, memory_saved, buffers_reused, reuse_dependencies) =
        process_events(&events, &liveness, &planner_input.occurrences);

    debug!(
        buffers_analyzed = liveness.len(),
        buffers_reused,
        memory_saved_bytes = memory_saved,
        "memory planner complete"
    );

    MemoryPlannerResult { buffer_replace, memory_saved, buffers_reused, reuse_dependencies }
}

/// Apply buffer replacements to the schedule.
///
/// Modifies the schedule in place, replacing logical buffers with their
/// physical replacements.
pub fn apply_buffer_replacements(schedule: &mut Schedule, replacements: &HashMap<BufferKey, Buffer>) {
    for (&key, replacement) in replacements {
        if let Some(item) = schedule.get_mut(key.kernel_idx)
            && let Some(buffer) = item.buffers.get_mut(key.buffer_idx)
        {
            *buffer = replacement.clone();
        }
    }
}

/// Add execution-order edges required by physical buffer reuse.
pub fn apply_reuse_dependencies(schedule: &mut Schedule, reuse_dependencies: &[ReuseDependency]) {
    for dep in reuse_dependencies {
        if dep.predecessor_step == dep.successor_step {
            continue;
        }

        debug_assert!(
            dep.successor_step > dep.predecessor_step,
            "reuse dependency must be forward-edge: predecessor={} >= successor={}",
            dep.predecessor_step,
            dep.successor_step,
        );

        if dep.predecessor_step >= schedule.len() {
            continue;
        }
        let Some(successor) = schedule.get_mut(dep.successor_step) else {
            continue;
        };
        if !successor.instance_dependencies.contains(&dep.predecessor_step) {
            successor.instance_dependencies.push(dep.predecessor_step);
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
#[path = "../test/unit/memory_planner.rs"]
mod tests;
