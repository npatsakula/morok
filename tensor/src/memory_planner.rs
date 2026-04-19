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

use std::collections::{HashMap, HashSet};

use morok_device::Buffer;
use morok_dtype::{DType, DeviceSpec};
use morok_ir::Op;
use tracing::{debug, trace};

use crate::schedule::Schedule;

/// Minimum block size for buffer pooling (4KB alignment).
const MIN_BLOCK_SIZE: usize = 0x1000;

/// Round up to the nearest multiple of block_size.
#[inline]
fn round_up(size: usize, block_size: usize) -> usize {
    size.div_ceil(block_size) * block_size
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Key for buffer pooling - groups buffers that can be reused interchangeably.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferPoolKey {
    /// Device where buffer is allocated.
    pub device: DeviceSpec,
    /// Data type of buffer elements.
    pub dtype: DType,
    /// Buffer size in bytes (rounded up to MIN_BLOCK_SIZE).
    pub size: usize,
    /// Logical shape metadata (must match for safe reuse in Morok codegen).
    pub shape: Vec<usize>,
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
    schedule
        .iter()
        .filter(|item| !matches!(item.ast.op(), Op::Sink { .. }))
        .flat_map(|item| item.buffers.iter().map(|b| b.id().0))
        .collect()
}

fn should_skip_buffer(buffer: &Buffer, output_buffer_ids: &HashSet<u64>, noopt_buffer_ids: &HashSet<u64>) -> bool {
    // Phase 1 planner only remaps full logical buffers. View/offset buffers require
    // an alias-preserving remap pass (planned for arena phase).
    buffer.offset() != 0
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
                shape: buffer.shape().to_vec(),
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
    events.sort_by_key(|e| (e.timestep, e.is_alloc));

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
) -> (HashMap<BufferKey, Buffer>, usize, usize) {
    let mut free_pools: HashMap<BufferPoolKey, Vec<Buffer>> = HashMap::new();
    let mut memory_saved: usize = 0;
    let mut buffers_reused: usize = 0;
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
                trace!(timestep = event.timestep, reused_buffer_id = reused.id().0, "reusing buffer from pool");

                chosen_by_id.insert(event.buffer_id, reused.clone());
                active_buffers.insert(event.buffer_id, reused);
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
                free_pools.entry(pool_key.clone()).or_default().push(buffer);
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

    (buffer_replace, memory_saved, buffers_reused)
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
///
/// # Returns
///
/// `MemoryPlannerResult` containing buffer replacements and statistics.
pub fn memory_planner(schedule: &Schedule, output_buffer_ids: &HashSet<u64>) -> MemoryPlannerResult {
    if schedule.is_empty() {
        return MemoryPlannerResult { buffer_replace: HashMap::new(), memory_saved: 0, buffers_reused: 0 };
    }

    // Phase 1: Liveness analysis
    let planner_input = analyze_liveness(schedule, output_buffer_ids);
    let liveness = planner_input.liveness;

    if liveness.is_empty() {
        debug!("no optimizable buffers found");
        return MemoryPlannerResult { buffer_replace: HashMap::new(), memory_saved: 0, buffers_reused: 0 };
    }

    // Phase 2: Build event timeline
    let events = build_event_timeline(&liveness);

    // Phase 3: Process events and compute replacements
    let (buffer_replace, memory_saved, buffers_reused) = process_events(&events, &liveness, &planner_input.occurrences);

    debug!(
        buffers_analyzed = liveness.len(),
        buffers_reused,
        memory_saved_bytes = memory_saved,
        "memory planner complete"
    );

    MemoryPlannerResult { buffer_replace, memory_saved, buffers_reused }
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

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::ScheduleItem;
    use morok_device::Buffer;
    use morok_ir::UOp;

    fn make_buffer(numel: usize) -> Buffer {
        let alloc = morok_device::registry::cpu().expect("cpu allocator");
        Buffer::new(alloc, DType::Float32, vec![numel], Default::default())
    }

    fn make_sink_item(id: u64, buffer: Buffer) -> ScheduleItem {
        let ast = UOp::sink(vec![UOp::native_const(0.0f32)]);
        ScheduleItem {
            kernel: ast.clone(),
            ast,
            buffers: vec![buffer],
            buffer_uop_ids: vec![id],
            fixedvars: HashMap::new(),
            bound_ranges: Vec::new(),
            dependencies: Vec::new(),
            alias_registered_ids: Vec::new(),
        }
    }

    fn make_nonsink_item(id: u64, buffer: Buffer) -> ScheduleItem {
        let ast = UOp::native_const(1.0f32);
        ScheduleItem {
            kernel: ast.clone(),
            ast,
            buffers: vec![buffer],
            buffer_uop_ids: vec![id],
            fixedvars: HashMap::new(),
            bound_ranges: Vec::new(),
            dependencies: Vec::new(),
            alias_registered_ids: Vec::new(),
        }
    }

    #[test]
    fn test_round_up() {
        assert_eq!(round_up(100, 0x1000), 0x1000);
        assert_eq!(round_up(0x1000, 0x1000), 0x1000);
        assert_eq!(round_up(0x1001, 0x1000), 0x2000);
        assert_eq!(round_up(0, 0x1000), 0);
    }

    #[test]
    fn test_buffer_pool_key_equality() {
        let key1 = BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x1000, shape: vec![256] };
        let key2 = BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x1000, shape: vec![256] };
        let key3 = BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x2000, shape: vec![512] };

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_event_timeline_ordering() {
        let mut liveness: HashMap<u64, BufferLiveness> = HashMap::new();
        liveness.insert(
            1,
            BufferLiveness {
                first_appearance: 0,
                last_appearance: 1,
                pool_key: BufferPoolKey {
                    device: DeviceSpec::Cpu,
                    dtype: DType::Float32,
                    size: 0x1000,
                    shape: vec![256],
                },
                prototype: make_buffer(256),
            },
        );
        liveness.insert(
            2,
            BufferLiveness {
                first_appearance: 2,
                last_appearance: 3,
                pool_key: BufferPoolKey {
                    device: DeviceSpec::Cpu,
                    dtype: DType::Float32,
                    size: 0x1000,
                    shape: vec![256],
                },
                prototype: make_buffer(256),
            },
        );

        let events = build_event_timeline(&liveness);

        // Events should be sorted by (timestep, is_alloc)
        // Free (false) comes before alloc (true) at same timestep
        assert_eq!(events.len(), 4);

        // Verify ordering: alloc@0, alloc@2 (free@2 comes before), free@2, alloc@2, free@4
        let mut prev_key = (0usize, false);
        for event in &events {
            let key = (event.timestep, event.is_alloc);
            assert!(key >= prev_key, "events not sorted: {:?} should come after {:?}", key, prev_key);
            prev_key = key;
        }
    }

    #[test]
    fn test_empty_schedule() {
        let schedule = vec![];
        let output_ids = HashSet::new();
        let result = memory_planner(&schedule, &output_ids);

        assert!(result.buffer_replace.is_empty());
        assert_eq!(result.memory_saved, 0);
        assert_eq!(result.buffers_reused, 0);
    }

    #[test]
    fn test_memory_planner_reuses_non_overlapping_buffers() {
        let b0 = make_buffer(256);
        let b1 = make_buffer(256);

        let schedule = vec![make_sink_item(10, b0.clone()), make_sink_item(11, b1.clone())];
        let result = memory_planner(&schedule, &HashSet::new());

        assert_eq!(result.buffers_reused, 1);
        let key = BufferKey { kernel_idx: 1, buffer_idx: 0 };
        let replacement = result.buffer_replace.get(&key).expect("second buffer should be remapped");
        assert_eq!(replacement.id(), b0.id());
    }

    #[test]
    fn test_memory_planner_skips_non_sink_noopt_buffers() {
        let b0 = make_buffer(256);
        let b1 = make_buffer(256);

        let schedule = vec![make_nonsink_item(20, b0), make_sink_item(21, b1)];
        let result = memory_planner(&schedule, &HashSet::new());

        assert_eq!(result.buffers_reused, 0);
        assert!(result.buffer_replace.is_empty());
    }
}
