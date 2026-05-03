use super::super::types::OptOps;
use super::*;

#[test]
fn test_beam_config_default() {
    let config = BeamConfig::default();
    assert_eq!(config.beam_width, 4);
    assert_eq!(config.timeout, Duration::from_secs(60));
    assert_eq!(config.max_upcast, 256);
    assert_eq!(config.max_local, 1024);
}

#[test]
fn test_beam_actions_not_empty() {
    assert!(!BEAM_ACTIONS.is_empty());
    // Should have a reasonable number of actions
    // UPCAST: 8 axes * 6 amounts = 48
    // UNROLL: 5 axes * 3 amounts = 15
    // LOCAL: 6 axes * 7 amounts = 42
    // GROUPTOP: 3 axes * 8 amounts = 24
    // GROUP: 3 axes * 4 amounts = 12
    // TC: 1 + 9 = 10
    // SWAP: 10 pairs
    // NOLOCALS: 1
    // Total: ~162 actions
    assert!(BEAM_ACTIONS.len() > 100, "Expected >100 actions, got {}", BEAM_ACTIONS.len());
    assert!(BEAM_ACTIONS.len() < 500, "Expected <500 actions, got {}", BEAM_ACTIONS.len());
}

#[test]
fn test_beam_actions_contains_expected_types() {
    let has_upcast = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::UPCAST);
    let has_local = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::LOCAL);
    let has_unroll = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::UNROLL);
    let has_tc = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::TC);
    let has_swap = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::SWAP);
    let has_nolocals = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::NOLOCALS);

    assert!(has_upcast);
    assert!(has_local);
    assert!(has_unroll);
    assert!(has_tc);
    assert!(has_swap);
    assert!(has_nolocals);
}

#[test]
fn test_beam_search_with_mock_scoring() {
    use super::super::renderer::Renderer;
    use morok_ir::UOp;

    // Create a simple scheduler
    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val]);
    let renderer = Renderer::cpu();
    let scheduler = Scheduler::new(sink, renderer);

    let config = BeamConfig { beam_width: 2, timeout: Duration::from_millis(100), ..Default::default() };

    // Mock scoring: just return a constant time
    let mock_score = |_s: &Scheduler| Some(Duration::from_micros(100));

    let result = beam_search(scheduler, &config, mock_score);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.iterations > 0 || result.candidates_evaluated == 0);
}

#[test]
fn test_validate_limits() {
    use super::super::renderer::Renderer;
    use morok_ir::UOp;

    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val]);
    let renderer = Renderer::cpu();
    let scheduler = Scheduler::new(sink, renderer);

    let config = BeamConfig::default();

    // Simple scheduler should pass limits
    assert!(validate_limits(&scheduler, &config));

    // With very restrictive limits
    let strict_config = BeamConfig { max_upcast: 1, max_local: 1, max_uops: 1, ..Default::default() };

    // May or may not pass depending on UOp count
    let _result = validate_limits(&scheduler, &strict_config);
}

#[test]
fn test_replay_opts_empty() {
    use super::super::renderer::Renderer;
    use morok_ir::UOp;

    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val]);
    let renderer = Renderer::cpu();
    let scheduler = Scheduler::new(sink, renderer);

    // Empty replay should succeed
    let result = replay_opts(scheduler, &[]);
    assert!(result.is_ok());
}

#[test]
fn test_serialize_deserialize_opts_empty() {
    let opts: Vec<Opt> = vec![];
    let serialized = serialize_opts(&opts);
    let deserialized = deserialize_opts(&serialized);

    assert!(deserialized.is_some());
    assert!(deserialized.unwrap().is_empty());
}

#[test]
fn test_serialize_deserialize_opts_upcast() {
    let opts = vec![Opt::upcast(0, 4), Opt::upcast(1, 8)];
    let serialized = serialize_opts(&opts);
    let deserialized = deserialize_opts(&serialized);

    assert!(deserialized.is_some());
    let result = deserialized.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].op, OptOps::UPCAST);
    assert_eq!(result[0].axis, Some(0));
    assert_eq!(result[1].op, OptOps::UPCAST);
    assert_eq!(result[1].axis, Some(1));
}

#[test]
fn test_serialize_deserialize_opts_tc() {
    use super::super::types::OptArg;

    let opts = vec![Opt::tc(None, -1, 2, 1)];
    let serialized = serialize_opts(&opts);
    let deserialized = deserialize_opts(&serialized);

    assert!(deserialized.is_some());
    let result = deserialized.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].op, OptOps::TC);
    assert_eq!(result[0].axis, None);
    if let OptArg::TensorCore { tc_select, opt_level, use_tc } = &result[0].arg {
        assert_eq!(*tc_select, -1);
        assert_eq!(*opt_level, 2);
        assert_eq!(*use_tc, 1);
    } else {
        panic!("Expected TensorCore arg");
    }
}

#[test]
fn test_serialize_deserialize_opts_swap() {
    use super::super::types::OptArg;

    let opts = vec![Opt::swap(0, 2)];
    let serialized = serialize_opts(&opts);
    let deserialized = deserialize_opts(&serialized);

    assert!(deserialized.is_some());
    let result = deserialized.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].op, OptOps::SWAP);
    assert_eq!(result[0].axis, Some(0));
    if let OptArg::Swap { other_axis } = &result[0].arg {
        assert_eq!(*other_axis, 2);
    } else {
        panic!("Expected Swap arg");
    }
}

#[test]
fn test_serialize_deserialize_opts_mixed() {
    let opts = vec![Opt::upcast(0, 4), Opt::local(1, 16), Opt::unroll(0, 8), Opt::nolocals()];
    let serialized = serialize_opts(&opts);
    let deserialized = deserialize_opts(&serialized);

    assert!(deserialized.is_some());
    let result = deserialized.unwrap();
    assert_eq!(result.len(), 4);
    assert_eq!(result[0].op, OptOps::UPCAST);
    assert_eq!(result[1].op, OptOps::LOCAL);
    assert_eq!(result[2].op, OptOps::UNROLL);
    assert_eq!(result[3].op, OptOps::NOLOCALS);
}

#[test]
fn test_beam_actions_contains_thread() {
    let has_thread = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::THREAD);
    assert!(has_thread, "BEAM_ACTIONS should contain THREAD actions");

    // Count thread actions
    let thread_count = BEAM_ACTIONS.iter().filter(|a| a.op == OptOps::THREAD).count();
    assert!(thread_count >= 6, "Expected at least 6 THREAD actions (3 axes × 2+ amounts), got {}", thread_count);
}

#[test]
fn test_thread_action_applied_to_loop_axis() {
    use super::super::renderer::Renderer;
    use morok_ir::{AxisId, AxisType, UOp};

    // Create a kernel with Loop axis (CPU threading target)
    let end_64 = UOp::index_const(64);
    let r_loop = UOp::range_axis(end_64, AxisId::Renumbered(0), AxisType::Loop);
    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_loop]);

    let renderer = Renderer::cpu();
    let scheduler = Scheduler::new(sink, renderer);

    // Verify renderer supports threading
    assert!(scheduler.renderer().has_threads, "CPU renderer should have has_threads=true");

    // Try to apply THREAD opt with a divisor that fits available parallelism.
    let max_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
    let thread_count = [32usize, 16, 8, 4, 2].into_iter().find(|&t| t <= max_threads && 64 % t == 0).unwrap_or(1);
    if thread_count == 1 {
        return;
    }
    let mut test_scheduler = scheduler.clone();
    let result = apply_opt(&mut test_scheduler, &Opt::thread(0, thread_count), true);
    assert!(result.is_ok(), "THREAD(0, {}) should succeed on Loop axis: {:?}", thread_count, result);

    // Verify Thread axis was created
    let thread_axes = test_scheduler.axes_of(&[AxisType::Thread]);
    assert!(!thread_axes.is_empty(), "Should have Thread axis after THREAD opt");
}

#[test]
fn test_generate_actions_includes_thread_for_cpu() {
    use super::super::renderer::Renderer;
    use morok_ir::{AxisId, AxisType, UOp};

    // Create a kernel with Loop axis
    let end_64 = UOp::index_const(64);
    let r_loop = UOp::range_axis(end_64, AxisId::Renumbered(0), AxisType::Loop);
    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_loop]);

    let renderer = Renderer::cpu();
    let scheduler = Scheduler::new(sink, renderer);

    let config = BeamConfig::default();
    let candidates = generate_actions(&scheduler, &config);

    // Check if any candidate has a Thread axis
    let has_threaded = candidates.iter().any(|s| !s.axes_of(&[AxisType::Thread]).is_empty());
    assert!(has_threaded, "generate_actions should produce candidates with Thread axes for CPU");
}
