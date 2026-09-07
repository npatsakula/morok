use super::super::types::{OptArgExt, OptOps};
use super::*;

fn trivial_scheduler() -> Scheduler {
    Scheduler::new(UOp::sink(vec![UOp::native_const(1i32)]), crate::optimizer::Renderer::cpu())
}

/// A SINK with one WEAK axis, so `generate_actions` has something to split.
fn weak_axis_scheduler(constant: i32) -> Scheduler {
    use svod_ir::{AxisId, AxisType};

    let range = UOp::range_axis(UOp::index_const(64), AxisId::Renumbered(0), AxisType::Weak);
    // Pin the thread budget: `Renderer::cpu()` reads the host core count, and the
    // weak axis only offers THREAD splits up to that budget.
    let mut renderer = crate::optimizer::Renderer::cpu();
    renderer.global_max = Some(vec![32]);
    Scheduler::new(UOp::sink(vec![UOp::native_const(constant), range]), renderer)
}

/// `BEAM_ACTIONS` is tinygrad's `actions` grid: every opt kind is offered, and the
/// grid is amount-major (all axes at the first amount, then all axes at the next).
#[test]
fn beam_actions_cover_every_opt_kind_in_amount_major_order() {
    for op in [OptOps::UPCAST, OptOps::UNROLL, OptOps::LOCAL, OptOps::GROUP, OptOps::GROUPTOP, OptOps::TC, OptOps::SWAP]
    {
        assert!(BEAM_ACTIONS.iter().any(|action| action.op == op), "{op:?} must be offered");
    }
    // NOLOCALS is env-gated (`SVOD_NOLOCALS`) and absent by default.
    assert!(!BEAM_ACTIONS.iter().any(|action| action.op == OptOps::NOLOCALS));
    // 3 CPU-threadable axes x at least 2 divisors.
    assert!(BEAM_ACTIONS.iter().filter(|action| action.op == OptOps::THREAD).count() >= 6);

    let upcasts = BEAM_ACTIONS.iter().filter(|action| action.op == OptOps::UPCAST).take(9).collect::<Vec<_>>();
    assert_eq!(
        upcasts.iter().take(8).map(|action| action.axis).collect::<Vec<_>>(),
        (0..8).map(Some).collect::<Vec<_>>()
    );
    assert!(upcasts.iter().take(8).all(|action| action.arg.int() == Ok(0)));
    assert_eq!((upcasts[8].axis, upcasts[8].arg.int()), (Some(0), Ok(2)));

    // TC: the strict default plus one padded (`opt_level` 2) choice per axis.
    let tensor_cores = BEAM_ACTIONS.iter().filter(|action| action.op == OptOps::TC).collect::<Vec<_>>();
    assert_eq!(tensor_cores.len(), 10);
    assert!(tensor_cores.iter().any(|action| action.arg.tc().unwrap().1 == 0));
    assert_eq!(tensor_cores.iter().filter(|action| action.arg.tc().unwrap().1 == 2).count(), 9);
}

/// The persistent BEAM cache replays a winning plan, so its key must separate
/// everything that changes which plan wins — and nothing that only changes how the
/// search is executed.
#[test]
fn beam_cache_key_separates_behavior_and_ignores_execution_details() {
    use svod_dtype::AmdArch;

    let scheduler = trivial_scheduler();
    let base = BeamConfig::default();
    let key = |config: &BeamConfig, compiler: &str, ast_hash| {
        CacheKey::from_scheduler(&scheduler, config, compiler, ast_hash).to_bytes()
    };
    let base_key = key(&base, "compiler", 0);

    for (what, variant) in [
        ("ast hash", key(&base, "compiler", 1)),
        ("compiler identity", key(&base, "cpu-clang:18", 0)),
        ("min progress", key(&BeamConfig { min_progress_ns: base.min_progress_ns + 1, ..base.clone() }, "compiler", 0)),
        ("nolocals", key(&BeamConfig { enable_nolocals: !base.enable_nolocals, ..base.clone() }, "compiler", 0)),
        (
            "compile timeout",
            key(&BeamConfig { compile_timeout_secs: base.compile_timeout_secs + 1, ..base.clone() }, "compiler", 0),
        ),
        ("num runs", key(&BeamConfig { num_runs: base.num_runs + 1, ..base.clone() }, "compiler", 0)),
    ] {
        assert_ne!(base_key, variant, "{what} must change the key");
    }
    for (what, variant) in [
        (
            "compile workers",
            key(&BeamConfig { compile_workers: base.compile_workers + 1, ..base.clone() }, "compiler", 0),
        ),
        (
            "child recycling",
            key(&BeamConfig { max_tasks_per_child: base.max_tasks_per_child + 1, ..base.clone() }, "compiler", 0),
        ),
    ] {
        assert_eq!(base_key, variant, "{what} must not change the key");
    }

    let ast = UOp::sink(vec![UOp::native_const(1i32)]);
    let amd = |arch| Scheduler::new(ast.clone(), crate::optimizer::Renderer::for_amd_arch(arch));
    assert_ne!(
        CacheKey::from_scheduler(&amd(AmdArch::Gfx1100), &base, "amd", 0).to_bytes(),
        CacheKey::from_scheduler(&amd(AmdArch::Gfx1151), &base, "amd", 0).to_bytes(),
        "the exact AMD target must change the key"
    );

    // A replayed plan is only valid under the action space that produced it, and
    // `BEAM_ACTIONS` is built from `BEAM_PADTO` / `TC` / `TC_OPT`.
    let full = CacheKey::from_scheduler(&scheduler, &base, "compiler", 0);
    assert_eq!(full.action_space, action_space_hash(&BEAM_ACTIONS));
    assert_ne!(full.action_space, action_space_hash(&BEAM_ACTIONS[1..]));
    assert_ne!(base_key, CacheKey { action_space: full.action_space ^ 1, ..full }.to_bytes());
}

/// Every opt kind must survive the persistent-cache encoding unchanged.
#[test]
fn opts_survive_the_cache_encoding_roundtrip() {
    let every_kind = vec![
        Opt::upcast(0, 4),
        Opt::local(1, 16),
        Opt::unroll(0, 8),
        Opt::group(0, 4),
        Opt::grouptop(1, 8),
        Opt::thread(0, 4),
        Opt::padto(1, 32),
        Opt::swap(0, 2),
        Opt::tc(None, -1, 2, 1),
        Opt::nolocals(),
    ];
    for opts in [vec![], every_kind] {
        assert_eq!(deserialize_opts(&serialize_opts(&opts)), Some(opts));
    }
}

/// `validate_limits` rejects a candidate whose upcast product outgrows the target's.
#[test]
fn validate_limits_rejects_oversized_upcasts() {
    let mut scheduler = weak_axis_scheduler(0x1102);
    assert!(validate_limits(&scheduler, &BeamConfig::default()));

    apply_opt(&mut scheduler, &Opt::upcast(0, 4), true).expect("UPCAST(0, 4) on a 64-wide Weak axis");
    assert!(validate_limits(&scheduler, &BeamConfig::default()));
    assert!(!validate_limits(&scheduler, &BeamConfig { max_upcast: 2, ..Default::default() }));
}

/// The plain `beam_search` entry point drives the same loop as the staged ones with
/// an inline scorer.
#[test]
fn beam_search_runs_with_an_inline_scorer() {
    let config = BeamConfig { beam_width: 2, disable_cache: true, ..Default::default() };
    let score = |scheduler: &Scheduler, _early_stop: Option<Duration>| {
        Some(CandidateMetrics {
            timing: Duration::from_micros(100),
            // Vary the hash per candidate so dedup does not collapse them all.
            ir_hash: scheduler as *const Scheduler as u64,
            compute_ops: Some(1),
        })
    };

    let result = beam_search(weak_axis_scheduler(0x1103), &config, score).expect("beam search");
    assert!(result.candidates_evaluated > 0);
}

#[test]
fn generate_actions_includes_thread_for_cpu() {
    use svod_ir::AxisType;

    let candidates = generate_actions(&weak_axis_scheduler(0x1104), &BeamConfig::default());
    assert!(candidates.iter().any(|scheduler| !scheduler.axes_of(&[AxisType::Thread]).is_empty()));
}

#[test]
fn test_remote_beam_parent_tracks_only_opt_sequences() {
    let scheduler = weak_axis_scheduler(0x4b31);
    let config =
        BeamConfig { beam_width: 2, min_progress_ns: 1_000_000_000, disable_cache: true, ..Default::default() };
    let base_opt_count = scheduler.applied_opts.len();
    let worker_scheduler = scheduler.clone();
    let result = beam_search_remote_staged(
        scheduler,
        &config,
        |candidates, emit| {
            assert!(candidates.iter().all(|opts| opts.len() == base_opt_count + 1));
            for (index, opts) in candidates.iter().enumerate() {
                if apply_remote_candidate(worker_scheduler.clone(), base_opt_count, opts, &config).is_some() {
                    emit(
                        index,
                        CompiledCandidate {
                            artifact: index,
                            binary_key: index.to_le_bytes().to_vec(),
                            compute_ops: Some(1),
                            preparation: Duration::ZERO,
                            compilation: Duration::ZERO,
                        },
                    );
                }
            }
            Ok(())
        },
        |index, _| Some(Duration::from_nanos(10_000 - *index as u64)),
    )
    .unwrap();
    assert_eq!(result.iterations, 1);
    assert_eq!(result.scheduler.applied_opts.len(), base_opt_count + 1);
    assert!(result.compiled > 0);
}

#[test]
fn test_staged_beam_streams_unordered_compiles_dedups_and_serializes_timing() {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct FakeArtifact {
        index: usize,
    }

    let scheduler = weak_axis_scheduler(0x51a9);
    let config = BeamConfig {
        beam_width: 2,
        min_progress_ns: 1_000_000_000,
        compile_workers: 3,
        disable_cache: true,
        ..BeamConfig::default()
    };
    let opts_by_index = Arc::new(Mutex::new(HashMap::new()));
    let compile_calls = Arc::new(AtomicUsize::new(0));
    let benchmark_calls = Arc::new(AtomicUsize::new(0));
    let benchmark_active = Arc::new(AtomicUsize::new(0));
    let benchmark_max = Arc::new(AtomicUsize::new(0));

    let result = beam_search_staged(
        scheduler,
        &config,
        {
            let opts_by_index = Arc::clone(&opts_by_index);
            let calls = Arc::clone(&compile_calls);
            move |candidates: &[Scheduler], emit: &mut dyn FnMut(usize, CompiledCandidate<FakeArtifact>)| {
                for index in (0..candidates.len()).rev() {
                    calls.fetch_add(1, Ordering::SeqCst);
                    opts_by_index.lock().unwrap().insert(index, candidates[index].applied_opts.clone());
                    let binary_key = if matches!(index, 3 | 4) { vec![0xdd] } else { index.to_le_bytes().to_vec() };
                    emit(
                        index,
                        CompiledCandidate {
                            artifact: FakeArtifact { index },
                            binary_key,
                            compute_ops: Some(if index == 2 { 1001 } else { 1 }),
                            preparation: Duration::ZERO,
                            compilation: Duration::ZERO,
                        },
                    );
                }
            }
        },
        {
            let calls = Arc::clone(&benchmark_calls);
            let active = Arc::clone(&benchmark_active);
            let maximum = Arc::clone(&benchmark_max);
            move |artifact: &FakeArtifact, _| {
                calls.fetch_add(1, Ordering::SeqCst);
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                maximum.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(1));
                active.fetch_sub(1, Ordering::SeqCst);
                Some(Duration::from_nanos(10_000 - artifact.index as u64))
            }
        },
    )
    .unwrap();

    let generated = compile_calls.load(Ordering::SeqCst);
    assert!(generated > 6, "test scheduler must expose enough candidates");
    assert_eq!(result.generated, generated);
    assert_eq!(result.unique_ir, 0);
    assert_eq!(result.compiled, compile_calls.load(Ordering::SeqCst));
    assert_eq!(
        result.unique_binary,
        result.compiled - 2,
        "one excessive-compute and one duplicate binary must be removed"
    );
    assert_eq!(benchmark_calls.load(Ordering::SeqCst), result.unique_binary);
    assert_eq!(result.benchmarked, result.unique_binary);
    assert_eq!(benchmark_max.load(Ordering::SeqCst), 1, "backend timing must be serialized");

    let winning_index = opts_by_index
        .lock()
        .unwrap()
        .keys()
        .copied()
        .filter(|index| *index != 1 && *index != 2 && *index != 4)
        .max()
        .unwrap();
    assert_eq!(result.scheduler.applied_opts, opts_by_index.lock().unwrap()[&winning_index]);
}

/// Hash a candidate's opt sequence into a stable artifact id, so cold and warm runs
/// score the same plan identically.
fn plan_identity(opts: &[Opt]) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    opts.hash(&mut hasher);
    hasher.finish()
}

fn plan_timing(identity: u64) -> Option<Duration> {
    Some(Duration::from_nanos(1 + identity % 10_000))
}

#[test]
fn test_staged_beam_cache_cold_and_warm_choose_same_winner() {
    let scheduler = weak_axis_scheduler(0x6b17);
    let config =
        BeamConfig { min_progress_ns: 1_000_000_000, compile_workers: 2, disable_cache: false, ..Default::default() };
    let compiler_identity = "fake-compiler:beam-cold-warm-v1";
    let key = CacheKey::from_scheduler(&scheduler, &config, compiler_identity, 0x1234);
    cache_invalidate(&key);
    if CACHE_DB.is_none() {
        return; // Another Svod process may hold sled's exclusive database lock.
    }

    let run = |scheduler: Scheduler| {
        beam_search_cached_staged(
            scheduler,
            &config,
            compiler_identity,
            0x1234,
            |candidates, emit| {
                for (index, candidate) in candidates.iter().enumerate() {
                    let identity = plan_identity(&candidate.applied_opts);
                    emit(
                        index,
                        CompiledCandidate {
                            artifact: identity,
                            binary_key: identity.to_le_bytes().to_vec(),
                            compute_ops: Some(1),
                            preparation: Duration::ZERO,
                            compilation: Duration::ZERO,
                        },
                    );
                }
            },
            |identity, _| plan_timing(*identity),
        )
        .unwrap()
    };

    let cold = run(scheduler.clone());
    let warm = run(scheduler);
    cache_invalidate(&key);

    assert!(cold.iterations > 0);
    assert_eq!(warm.iterations, 0, "second search should replay the persistent BEAM entry");
    assert_eq!(cold.scheduler.applied_opts, warm.scheduler.applied_opts);
    assert_eq!(cold.timing, warm.timing);
}

#[test]
fn test_remote_beam_cache_reuses_winner_across_parallel_and_recycling_changes() {
    let scheduler = weak_axis_scheduler(0x7193);
    let cold_config = BeamConfig {
        min_progress_ns: 1_000_000_000,
        compile_workers: 1,
        max_tasks_per_child: 1,
        disable_cache: false,
        ..Default::default()
    };
    let warm_config = BeamConfig { compile_workers: 8, max_tasks_per_child: 99, ..cold_config.clone() };
    let identity = "fake-compiler:remote-cache-v1";
    let key = CacheKey::from_scheduler(&scheduler, &cold_config, identity, 0x22);
    cache_invalidate(&key);
    if CACHE_DB.is_none() {
        return;
    }

    let run = |config: &BeamConfig| {
        let worker_scheduler = scheduler.clone();
        let base_opt_count = scheduler.applied_opts.len();
        beam_search_cached_remote(
            scheduler.clone(),
            config,
            identity,
            0x22,
            |candidates, emit| {
                for (index, opts) in candidates.iter().enumerate() {
                    if apply_remote_candidate(worker_scheduler.clone(), base_opt_count, opts, config).is_none() {
                        continue;
                    }
                    let artifact = plan_identity(opts);
                    emit(
                        index,
                        CompiledCandidate {
                            artifact,
                            binary_key: artifact.to_le_bytes().to_vec(),
                            compute_ops: Some(1),
                            preparation: Duration::ZERO,
                            compilation: Duration::ZERO,
                        },
                    );
                }
                Ok(())
            },
            |artifact, _| plan_timing(*artifact),
        )
        .unwrap()
    };
    let cold = run(&cold_config);
    let warm = run(&warm_config);
    cache_invalidate(&key);
    assert!(cold.iterations > 0);
    assert_eq!(warm.iterations, 0);
    assert_eq!(cold.scheduler.applied_opts, warm.scheduler.applied_opts);
    assert_eq!(cold.timing, warm.timing);
}

#[test]
fn test_remote_beam_does_not_cache_unbenchmarked_search() {
    let scheduler = weak_axis_scheduler(0x7a21);
    let config = BeamConfig { min_progress_ns: 1_000_000_000, disable_cache: false, ..Default::default() };
    let identity = "fake-compiler:remote-no-empty-cache-v1";
    let key = CacheKey::from_scheduler(&scheduler, &config, identity, 0x31);
    cache_invalidate(&key);
    if CACHE_DB.is_none() {
        return;
    }

    let failed = beam_search_cached_remote(
        scheduler.clone(),
        &config,
        identity,
        0x31,
        |_candidates, _emit: &mut dyn FnMut(usize, CompiledCandidate<usize>)| Ok(()),
        |_artifact, _| Some(Duration::from_nanos(1)),
    )
    .unwrap();
    assert_eq!(failed.benchmarked, 0);
    assert_eq!(failed.timing, Duration::MAX);
    assert!(cache_get(&key).is_none());

    let cold = beam_search_cached_remote(
        scheduler,
        &config,
        identity,
        0x31,
        |candidates, emit| {
            for index in 0..candidates.len() {
                emit(
                    index,
                    CompiledCandidate {
                        artifact: index,
                        binary_key: index.to_le_bytes().to_vec(),
                        compute_ops: Some(1),
                        preparation: Duration::ZERO,
                        compilation: Duration::ZERO,
                    },
                );
            }
            Ok(())
        },
        |artifact, _| Some(Duration::from_nanos(10_000 - *artifact as u64)),
    )
    .unwrap();
    assert!(cold.iterations > 0, "the failed search must not create a cache hit");
    assert!(cache_get(&key).is_some());
    cache_invalidate(&key);
}

#[test]
fn test_remote_beam_worker_error_invalidates_cache() {
    let scheduler = weak_axis_scheduler(0x7a22);
    let config = BeamConfig { min_progress_ns: 1_000_000_000, disable_cache: false, ..Default::default() };
    let identity = "fake-compiler:remote-worker-error-v1";
    let key = CacheKey::from_scheduler(&scheduler, &config, identity, 0x32);
    cache_invalidate(&key);
    if CACHE_DB.is_none() {
        return;
    }

    cache_put(&key, &[Opt::upcast(0, 2)]);
    let result = beam_search_cached_remote(
        scheduler,
        &config,
        identity,
        0x32,
        |_candidates, _emit: &mut dyn FnMut(usize, CompiledCandidate<usize>)| {
            Err(OptError::BeamWorker { message: "disconnected".into() })
        },
        |_artifact, _| Some(Duration::from_nanos(1)),
    );
    assert!(matches!(result, Err(OptError::BeamWorker { .. })));
    assert!(cache_get(&key).is_none());
}
