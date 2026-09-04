use super::*;

#[test]
fn test_opt_strategy_predicates() {
    assert_eq!(OptStrategy::default(), OptStrategy::Heuristic);
    for (strategy, is_none, is_beam) in [
        (OptStrategy::None, true, false),
        (OptStrategy::Heuristic, false, false),
        (OptStrategy::Beam { width: 4 }, false, true),
    ] {
        assert_eq!((strategy.is_none(), strategy.is_beam()), (is_none, is_beam), "{strategy:?}");
    }
}

/// The defaults are tinygrad's, and a builder call reaches every field it exposes.
#[test]
fn test_beam_config_default_and_builder() {
    let config = BeamConfig::default();
    assert_eq!(config.beam_width, 4);
    assert_eq!(config.max_upcast, 256);
    assert_eq!(config.max_local, 1024);
    assert_eq!(config.min_progress_ns, 10);
    assert!(!config.enable_nolocals);
    assert_eq!(config.compile_workers, 0);
    assert_eq!(config.max_tasks_per_child, 16);
    assert_eq!(config.compile_timeout_secs, 10);

    let built = BeamConfig::builder()
        .beam_width(8)
        .max_upcast(512)
        .min_progress_ns(25)
        .enable_nolocals(true)
        .compile_workers(3)
        .max_tasks_per_child(5)
        .compile_timeout_secs(7)
        .build();
    assert_eq!(built.max_local, config.max_local, "an unset field keeps its default");
    assert_eq!((built.beam_width, built.max_upcast, built.min_progress_ns, built.enable_nolocals), (8, 512, 25, true));
    assert_eq!((built.compile_workers, built.max_tasks_per_child, built.compile_timeout_secs), (3, 5, 7));
}

/// tinygrad `helpers.py`: `BEAM_MIN_PROGRESS` is in microseconds.
#[test_case::test_case(None, 10; "unset falls back to the default")]
#[test_case::test_case(Some("0.01"), 10; "sub-nanosecond rounds to the default")]
#[test_case::test_case(Some("1"), 1_000; "one microsecond")]
#[test_case::test_case(Some("invalid"), 10; "unparseable falls back to the default")]
fn test_beam_min_progress_matches_tinygrad_microseconds_env(raw: Option<&str>, expected: u64) {
    assert_eq!(parse_beam_min_progress(raw), expected);
}

/// `SVOD_THREADS` is the one thread budget; only a positive integer overrides
/// the host's parallelism.
#[test_case::test_case(Some("4"), Some(4); "positive integer")]
#[test_case::test_case(Some("0"), None; "zero falls back")]
#[test_case::test_case(Some("many"), None; "unparseable falls back")]
#[test_case::test_case(None, None; "unset falls back")]
fn test_thread_budget_parsing(raw: Option<&str>, expected: Option<usize>) {
    let fallback = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    assert_eq!(parse_thread_budget(raw), expected.unwrap_or(fallback));
}

#[test]
fn test_heuristics_config_default_and_builder() {
    let config = HeuristicsConfig::default();
    assert_eq!(config.tc_enabled, TcUsage::Enabled);
    // tinygrad `helpers.py:238`: TC_OPT defaults to 0 on the heuristic path.
    assert_eq!(config.tc_opt, TcOpt::Strict);
    assert!(config.matvec_enabled);
    assert_eq!((config.threads_per_row, config.rows_per_thread, config.grouped_threshold), (8, 4, 256));

    let built = HeuristicsConfig::builder()
        .tc_enabled(TcUsage::Disabled)
        .matvec_enabled(false)
        .threads_per_row(16)
        .rows_per_thread(2)
        .grouped_threshold(128)
        .build();
    assert_eq!(built.tc_enabled, TcUsage::Disabled);
    assert!(!built.matvec_enabled);
    assert_eq!((built.threads_per_row, built.rows_per_thread, built.grouped_threshold), (16, 2, 128));
}

#[test]
fn test_optimizer_config_default_and_builder() {
    let config = OptimizerConfig::default();
    assert_eq!(config.strategy, OptStrategy::Heuristic);
    assert_eq!(config.beam.beam_width, 4);
    // tinygrad `helpers.py:245`: DISABLE_FAST_IDIV defaults to 1.
    assert!(config.disable_fast_idiv);

    let built = OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 8 })
        .beam(BeamConfig::builder().max_upcast(512).build())
        .build();
    assert_eq!(built.strategy, OptStrategy::Beam { width: 8 });
    assert_eq!((built.beam.beam_width, built.beam.max_upcast), (8, 512));
}

/// `disable_fast_idiv` gates the magic-multiply rewrite in the late pattern set.
#[test_case::test_case(true, true; "disabled keeps cdiv")]
#[test_case::test_case(false, false; "enabled rewrites cdiv")]
fn test_disable_fast_idiv_gates_late_rewrites(disable_fast_idiv: bool, expect_cdiv: bool) {
    use svod_ir::{BinaryOp, DType, Op, UOp};

    let x = UOp::var("x", DType::Int32, 0, 255);
    let cdiv = UOp::new(Op::Binary(BinaryOp::CDiv, x, UOp::native_const(3i32)), DType::Int32);
    let renderer = crate::optimizer::Renderer::cpu().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
    let patterns = crate::optimizer::get_late_rewrite_patterns(&renderer, disable_fast_idiv);
    let rewritten = crate::rewrite::graph_rewrite(&patterns, cdiv, &mut ());

    let has_cdiv = rewritten.toposort().iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::CDiv, ..)));
    assert_eq!(has_cdiv, expect_cdiv, "{}", rewritten.tree());
}

/// The numeric encodings tinygrad's `TC` / `TC_OPT` / `TC_SELECT` env vars use; they
/// reach the BEAM cache key and the remote worker protocol.
#[test]
fn test_tc_env_encodings_match_tinygrad() {
    assert_eq!([TcUsage::Disabled.as_usize(), TcUsage::Enabled.as_usize(), TcUsage::ShapeOnly.as_usize()], [0, 1, 2]);
    assert_eq!([TcOpt::Strict.as_usize(), TcOpt::Relaxed.as_usize(), TcOpt::Padded.as_usize()], [0, 1, 2]);
    assert_eq!([TcSelect::Auto.as_i32(), TcSelect::Index(5).as_i32()], [-1, 5]);
}
