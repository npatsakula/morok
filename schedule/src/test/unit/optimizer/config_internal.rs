use super::*;

#[test]
fn test_opt_strategy_default_is_heuristic() {
    assert_eq!(OptStrategy::default(), OptStrategy::Heuristic);
}

#[test]
fn test_opt_strategy_is_none() {
    assert!(OptStrategy::None.is_none());
    assert!(!OptStrategy::Heuristic.is_none());
    assert!(!OptStrategy::Beam { width: 4 }.is_none());
}

#[test]
fn test_opt_strategy_is_beam() {
    assert!(!OptStrategy::None.is_beam());
    assert!(!OptStrategy::Heuristic.is_beam());
    assert!(OptStrategy::Beam { width: 4 }.is_beam());
}

#[test]
fn test_beam_config_default() {
    let config = BeamConfig::default();
    assert_eq!(config.beam_width, 4);
    assert_eq!(config.timeout, Duration::from_secs(60));
    assert_eq!(config.max_upcast, 256);
    assert_eq!(config.max_local, 1024);
}

#[test]
fn test_beam_config_builder() {
    let config = BeamConfig::builder().beam_width(8).timeout_secs(120).max_upcast(512).build();

    assert_eq!(config.beam_width, 8);
    assert_eq!(config.timeout, Duration::from_secs(120));
    assert_eq!(config.max_upcast, 512);
    assert_eq!(config.max_local, 1024); // default
}

#[test]
fn test_heuristics_config_default() {
    let config = HeuristicsConfig::default();
    assert_eq!(config.tc_enabled, TcUsage::Enabled);
    assert_eq!(config.tc_opt, TcOpt::Padded);
    assert!(config.matvec_enabled);
    assert_eq!(config.threads_per_row, 8);
    assert_eq!(config.rows_per_thread, 4);
    assert_eq!(config.grouped_threshold, 256);
}

#[test]
fn test_heuristics_config_builder() {
    let config = HeuristicsConfig::builder()
        .tc_enabled(TcUsage::Disabled)
        .matvec_enabled(false)
        .threads_per_row(16)
        .rows_per_thread(2)
        .grouped_threshold(128)
        .build();

    assert_eq!(config.tc_enabled, TcUsage::Disabled);
    assert!(!config.matvec_enabled);
    assert_eq!(config.threads_per_row, 16);
    assert_eq!(config.rows_per_thread, 2);
    assert_eq!(config.grouped_threshold, 128);
}

#[test]
fn test_optimizer_config_default() {
    let config = OptimizerConfig::default();
    assert_eq!(config.strategy, OptStrategy::Heuristic);
    assert_eq!(config.beam.beam_width, 4);
}

#[test]
fn test_optimizer_config_builder() {
    let config = OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 8 })
        .beam(BeamConfig::builder().timeout_secs(120).build())
        .build();

    assert_eq!(config.strategy, OptStrategy::Beam { width: 8 });
    assert_eq!(config.beam.timeout, Duration::from_secs(120));
}

#[test]
fn test_tc_usage_as_usize() {
    assert_eq!(TcUsage::Disabled.as_usize(), 0);
    assert_eq!(TcUsage::Enabled.as_usize(), 1);
    assert_eq!(TcUsage::ShapeOnly.as_usize(), 2);
}

#[test]
fn test_tc_opt_as_usize() {
    assert_eq!(TcOpt::Strict.as_usize(), 0);
    assert_eq!(TcOpt::Relaxed.as_usize(), 1);
    assert_eq!(TcOpt::Padded.as_usize(), 2);
}

#[test]
fn test_tc_select_as_i32() {
    assert_eq!(TcSelect::Auto.as_i32(), -1);
    assert_eq!(TcSelect::Index(5).as_i32(), 5);
}
