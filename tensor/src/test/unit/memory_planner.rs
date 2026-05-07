use super::*;
use crate::schedule::ScheduleItem;
use morok_device::Buffer;
use morok_ir::UOp;
use std::collections::HashSet;
use std::sync::Arc;

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
        dependencies: Vec::new(),
        instance_dependencies: Vec::new(),
        alias_registered_ids: Vec::new(),
        loop_var_names: HashSet::new(),
    }
}

fn make_nonsink_item(id: u64, buffer: Buffer) -> ScheduleItem {
    let ast = UOp::native_const(id as f32);
    ScheduleItem {
        kernel: ast.clone(),
        ast,
        buffers: vec![buffer],
        buffer_uop_ids: vec![id],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        instance_dependencies: Vec::new(),
        alias_registered_ids: Vec::new(),
        loop_var_names: HashSet::new(),
    }
}

fn make_store_item(buffer_uop: &Arc<UOp>, buffer: Buffer, index: Arc<UOp>) -> ScheduleItem {
    let ast = UOp::sink(vec![index.store(UOp::native_const(1.0f32))]);
    ScheduleItem {
        kernel: ast.clone(),
        ast,
        buffers: vec![buffer],
        buffer_uop_ids: vec![buffer_uop.id],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        instance_dependencies: Vec::new(),
        alias_registered_ids: Vec::new(),
        loop_var_names: HashSet::new(),
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
fn test_round_up_256_block() {
    // 256-byte alignment.
    assert_eq!(round_up(1, 256), 256);
    assert_eq!(round_up(256, 256), 256);
    assert_eq!(round_up(257, 256), 512);
    assert_eq!(round_up(0, 256), 0);
}

#[test]
fn test_parse_mode_default_is_arena() {
    // Env unset (`NO_MEMORY_PLANNER=0`) → arena planner runs.
    assert_eq!(parse_mode(None), PlannerMode::Arena);
    assert_eq!(parse_mode(Some("")), PlannerMode::Arena);
}

#[test]
fn test_parse_mode_disabled_aliases() {
    for raw in ["0", "off", "none", "disabled", "OFF", " disabled ", "Disabled"] {
        assert_eq!(parse_mode(Some(raw)), PlannerMode::Disabled, "raw={raw:?}");
    }
}

#[test]
fn test_parse_mode_remap_aliases() {
    // `remap` / `pool` opt into the older liveness-based pool reuse.
    for raw in ["remap", "pool", "POOL", "Remap", " remap "] {
        assert_eq!(parse_mode(Some(raw)), PlannerMode::Remap, "raw={raw:?}");
    }
}

#[test]
fn test_parse_mode_arena_aliases() {
    for raw in ["arena", "ARENA", " arena ", "1", "on"] {
        assert_eq!(parse_mode(Some(raw)), PlannerMode::Arena, "raw={raw:?}");
    }
}

#[test]
fn test_parse_mode_unknown_falls_back_to_arena() {
    // Unknown values must not crash — default to the arena mode rather than
    // silently regressing to a different strategy.
    assert_eq!(parse_mode(Some("garbage")), PlannerMode::Arena);
}

#[test]
fn test_memory_planner_disabled_short_circuits() {
    // Same setup as test_memory_planner_reuses_non_overlapping_buffers — would
    // produce one reuse under Remap. Disabled must emit nothing.
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);

    let schedule = vec![make_sink_item(10, b0), make_sink_item(11, b1)];
    let result = memory_planner(&schedule, &HashSet::new(), PlannerMode::Disabled);

    assert!(result.buffer_replace.is_empty());
    assert_eq!(result.buffers_reused, 0);
    assert_eq!(result.memory_saved, 0);
    assert!(result.reuse_dependencies.is_empty());
}

#[test]
fn test_buffer_pool_key_equality() {
    let key1 = BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x1000 };
    let key2 = BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x1000 };
    let key3 = BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x2000 };

    assert_eq!(key1, key2);
    assert_ne!(key1, key3);
}

#[test]
fn test_buffer_pool_key_is_shape_agnostic() {
    // Two non-output buffers with the same (device, dtype, rounded_size) share
    // the same pool regardless of logical shape — codegen reads shape from the
    // UOp graph, not from the Buffer.
    let b_flat = make_buffer(256);
    let b_2d = morok_device::Buffer::new(
        morok_device::registry::cpu().expect("cpu"),
        DType::Float32,
        vec![16, 16],
        Default::default(),
    );

    let key_flat = BufferPoolKey {
        device: b_flat.allocator().device_spec(),
        dtype: b_flat.dtype(),
        size: round_up(b_flat.size(), 0x1000),
    };
    let key_2d = BufferPoolKey {
        device: b_2d.allocator().device_spec(),
        dtype: b_2d.dtype(),
        size: round_up(b_2d.size(), 0x1000),
    };

    assert_eq!(key_flat, key_2d, "shape-only differences must not split pools");
}

#[test]
fn test_event_timeline_ordering() {
    let mut liveness: HashMap<u64, BufferLiveness> = HashMap::new();
    liveness.insert(
        1,
        BufferLiveness {
            first_appearance: 0,
            last_appearance: 1,
            pool_key: BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x1000 },
            prototype: make_buffer(256),
        },
    );
    liveness.insert(
        2,
        BufferLiveness {
            first_appearance: 2,
            last_appearance: 3,
            pool_key: BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x1000 },
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
    let result = memory_planner(&schedule, &output_ids, PlannerMode::Remap);

    assert!(result.buffer_replace.is_empty());
    assert_eq!(result.memory_saved, 0);
    assert_eq!(result.buffers_reused, 0);
}

#[test]
fn test_memory_planner_reuses_non_overlapping_buffers() {
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);

    let schedule = vec![make_sink_item(10, b0.clone()), make_sink_item(11, b1.clone())];
    let result = memory_planner(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 1);
    let key = BufferKey { kernel_idx: 1, buffer_idx: 0 };
    let replacement = result.buffer_replace.get(&key).expect("second buffer should be remapped");
    assert_eq!(replacement.id(), b0.id());
    assert_eq!(result.reuse_dependencies, vec![ReuseDependency { predecessor_step: 0, successor_step: 1 }]);
}

#[test]
fn test_memory_planner_reuses_unmasked_store_outputs() {
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);
    let target = UOp::new_buffer(DeviceSpec::Cpu, 256, DType::Float32);
    let index = UOp::index().buffer(target.clone()).indices(vec![UOp::index_const(0)]).call().unwrap();

    let schedule = vec![make_store_item(&target, b0.clone(), index), make_sink_item(61, b1)];
    let result = memory_planner(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 1);
    let key = BufferKey { kernel_idx: 1, buffer_idx: 0 };
    let replacement = result.buffer_replace.get(&key).expect("second buffer should be remapped");
    assert_eq!(replacement.id(), b0.id());
}

#[test]
fn test_memory_planner_skips_masked_store_outputs() {
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);
    let target = UOp::new_buffer(DeviceSpec::Cpu, 256, DType::Float32);
    let index = UOp::index()
        .buffer(target.clone())
        .indices(vec![UOp::index_const(0)])
        .gate(UOp::native_const(true))
        .call()
        .unwrap();

    let schedule = vec![make_store_item(&target, b0, index), make_sink_item(62, b1)];
    let result = memory_planner(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 0);
    assert!(result.buffer_replace.is_empty());
}

#[test]
fn test_memory_planner_skips_wrapped_masked_store_outputs() {
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);
    let target = UOp::new_buffer(DeviceSpec::Cpu, 256, DType::Float32);
    let index = UOp::index()
        .buffer(target.clone())
        .indices(vec![UOp::index_const(0)])
        .gate(UOp::native_const(true))
        .call()
        .unwrap()
        .cast(DType::Index);

    let schedule = vec![make_store_item(&target, b0, index), make_sink_item(63, b1)];
    let result = memory_planner(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 0);
    assert!(result.buffer_replace.is_empty());
}

#[test]
fn test_apply_reuse_dependencies_adds_antidependency() {
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);
    let mut schedule = vec![make_nonsink_item(10, b0), make_nonsink_item(11, b1)];

    apply_reuse_dependencies(&mut schedule, &[ReuseDependency { predecessor_step: 0, successor_step: 1 }]);
    apply_reuse_dependencies(&mut schedule, &[ReuseDependency { predecessor_step: 0, successor_step: 1 }]);

    assert!(schedule[1].dependencies.is_empty());
    assert_eq!(schedule[1].instance_dependencies, vec![0]);
}

#[test]
fn test_memory_planner_skips_non_sink_noopt_buffers() {
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);

    let schedule = vec![make_nonsink_item(20, b0), make_sink_item(21, b1)];
    let result = memory_planner(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 0);
    assert!(result.buffer_replace.is_empty());
}

#[test]
fn test_memory_planner_skips_buffers_with_views() {
    let b0 = make_buffer(256);
    let b0_view = b0.view(4, b0.size() - 4).unwrap();
    let b1 = make_buffer(256);

    let schedule = vec![make_sink_item(30, b0), make_sink_item(31, b0_view), make_sink_item(32, b1)];
    let result = memory_planner(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 0);
    assert!(result.buffer_replace.is_empty());
}

#[test]
fn test_memory_planner_skips_lone_nonzero_offset_view() {
    let b0 = make_buffer(256);
    let b0_view = b0.view(4, b0.size() - 4).unwrap();
    let b1 = make_buffer(255);

    let schedule = vec![make_sink_item(35, b0_view), make_sink_item(36, b1)];
    let result = memory_planner(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 0);
    assert!(result.buffer_replace.is_empty());
}

#[test]
fn test_memory_planner_skips_buffers_with_shape_aliases() {
    let b0 = make_buffer(256);
    let b0_alias = b0.view(0, b0.size() - 4).unwrap();
    let b1 = make_buffer(256);

    let schedule = vec![make_sink_item(40, b0), make_sink_item(41, b0_alias), make_sink_item(42, b1)];
    let result = memory_planner(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 0);
    assert!(result.buffer_replace.is_empty());
}

// ============================================================================
// Arena planner (PlannerMode::Arena) tests
// ============================================================================

#[test]
fn test_arena_packs_disjoint_lifetimes_into_one_arena() {
    // Three disjoint-lifetime buffers on the same device should all collapse
    // into views over a single arena allocation.
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);
    let b2 = make_buffer(256);

    let schedule = vec![make_sink_item(50, b0.clone()), make_sink_item(51, b1.clone()), make_sink_item(52, b2.clone())];
    let result = memory_planner(&schedule, &HashSet::new(), PlannerMode::Arena);

    assert_eq!(result.buffer_replace.len(), 3, "every plannable buffer must get an arena view");

    let storage_ids: std::collections::HashSet<_> = result.buffer_replace.values().map(|b| b.storage_id().0).collect();
    assert_eq!(storage_ids.len(), 1, "all three views must share one underlying arena allocation");

    let handle_ids: std::collections::HashSet<_> = result.buffer_replace.values().map(|b| b.id().0).collect();
    assert_eq!(handle_ids.len(), 3, "each view must carry a distinct handle id (Path Y)");
}

#[test]
fn test_arena_excludes_output_buffers() {
    // Output buffers must never get rewritten — the runtime returns them
    // directly to the caller.
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);

    let schedule = vec![make_sink_item(60, b0.clone()), make_sink_item(61, b1.clone())];
    // Mark b0 as an output (its ScheduleItem id 60 maps to its uop id).
    let mut output_ids = HashSet::new();
    output_ids.insert(b0.id().0);

    let result = memory_planner(&schedule, &output_ids, PlannerMode::Arena);

    let key0 = BufferKey { kernel_idx: 0, buffer_idx: 0 };
    assert!(!result.buffer_replace.contains_key(&key0), "output buffer must not be rewritten into arena view");
}

#[test]
fn test_arena_mode_dispatches_to_arena_planner_not_remap() {
    // Construct a workload Remap would *not* reuse (different sizes → different
    // pool keys) but Arena *can* pack.
    let b0 = make_buffer(256);
    let b1 = make_buffer(512);

    let schedule = vec![make_sink_item(70, b0), make_sink_item(71, b1)];
    let remap = memory_planner(&schedule, &HashSet::new(), PlannerMode::Remap);
    let arena = memory_planner(&schedule, &HashSet::new(), PlannerMode::Arena);

    assert!(remap.buffer_replace.is_empty(), "Remap can't pack different-size buffers");
    assert_eq!(arena.buffer_replace.len(), 2, "Arena packs even when sizes differ");
}

#[test]
fn test_arena_disabled_mode_short_circuits_unchanged() {
    // Sanity: Disabled mode is unaffected by adding the arena path.
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);

    let schedule = vec![make_sink_item(80, b0), make_sink_item(81, b1)];
    let result = memory_planner(&schedule, &HashSet::new(), PlannerMode::Disabled);

    assert!(result.buffer_replace.is_empty());
    assert_eq!(result.buffers_reused, 0);
    assert!(result.reuse_dependencies.is_empty());
}

#[test]
fn test_arena_reports_memory_savings() {
    // Two disjoint-lifetime 256-byte buffers should pack into one ~256-byte
    // arena, saving roughly one buffer's worth of memory.
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);

    let schedule = vec![make_sink_item(90, b0), make_sink_item(91, b1)];
    let result = memory_planner(&schedule, &HashSet::new(), PlannerMode::Arena);

    assert!(
        result.memory_saved > 0,
        "arena packing of two disjoint-lifetime same-size buffers must report savings, got {} bytes saved",
        result.memory_saved
    );
}
