//! Unit tests for the profiler data model, table rendering (GPU-free), and
//! [`RunProfile::merge`] accumulation semantics.

use std::collections::BTreeMap;
use std::time::Duration;

use svod_device::PmcCounter;

use crate::profiler::{DERIVED, DerivedCtx, PmcSelection, ProfileOptions, RunProfile, StageProfile, cratio, parse_pmc};

#[test]
fn pmc_counter_token_roundtrip() {
    for c in [PmcCounter::SqBusyCycles, PmcCounter::SqWaves, PmcCounter::SqInstsValu] {
        assert_eq!(PmcCounter::from_token(c.token()), Some(c), "token roundtrip for {c:?}");
    }
    assert_eq!(PmcCounter::from_token("nope"), None);
    assert_eq!(PmcCounter::from_token("BUSY"), Some(PmcCounter::SqBusyCycles), "case-insensitive alias");
}

#[test]
fn pmc_selection_resolution() {
    assert!(!PmcSelection::None.is_enabled());
    assert!(PmcSelection::None.counters().is_empty());
    assert!(PmcSelection::Default.is_enabled());
    assert_eq!(PmcSelection::Default.counters().len(), 3);
    let custom = PmcSelection::Custom(vec![PmcCounter::SqInstsValu]);
    assert_eq!(custom.counters(), vec![PmcCounter::SqInstsValu]);
}

#[test]
fn parse_pmc_values() {
    assert_eq!(parse_pmc(""), PmcSelection::None);
    assert_eq!(parse_pmc("0"), PmcSelection::None);
    assert_eq!(parse_pmc("1"), PmcSelection::Default);
    assert_eq!(parse_pmc("valu,waves"), PmcSelection::Custom(vec![PmcCounter::SqInstsValu, PmcCounter::SqWaves]));
    // All-unknown tokens fall back to the default set rather than an empty selection.
    assert_eq!(parse_pmc("bogus"), PmcSelection::Default);
}

#[test]
fn profile_options_default() {
    let o = ProfileOptions::default();
    assert_eq!(o.iters, 1);
    assert!(o.static_analysis);
    assert_eq!(o.counters, PmcSelection::None);
}

#[test]
fn cratio_requires_both_inputs_and_nonzero_denominator() {
    let mut m = BTreeMap::new();
    // Denominator absent → None.
    assert_eq!(cratio(&m, PmcCounter::SqInstsValu, PmcCounter::SqBusyCycles), None);
    m.insert(PmcCounter::SqBusyCycles, 0);
    // Numerator absent → None even with denominator present.
    assert_eq!(cratio(&m, PmcCounter::SqInstsValu, PmcCounter::SqBusyCycles), None);
    m.insert(PmcCounter::SqInstsValu, 50);
    // Zero denominator → None (no divide-by-zero).
    assert_eq!(cratio(&m, PmcCounter::SqInstsValu, PmcCounter::SqBusyCycles), None);
    m.insert(PmcCounter::SqBusyCycles, 200);
    assert_eq!(cratio(&m, PmcCounter::SqInstsValu, PmcCounter::SqBusyCycles), Some(0.25));
}

#[test]
fn derived_metric_formulas() {
    let lookup = |label: &str| DERIVED.iter().find(|(l, _)| *l == label).map(|(_, f)| *f).expect("derived col");
    let ctx = DerivedCtx { xcc_num: 8, device_simds: 1216, wall_secs: 1e-6, gpu_stamped: true };
    let mut m = BTreeMap::new();
    m.insert(PmcCounter::LdsBankConflict, 3);
    m.insert(PmcCounter::LdsIdxActive, 12);
    m.insert(PmcCounter::ValuMfmaBusyCycles, 40);
    m.insert(PmcCounter::SqBusyCycles, 80);
    m.insert(PmcCounter::GrbmGuiActive, 100);
    m.insert(PmcCounter::L2Hit, 30);
    m.insert(PmcCounter::L2Miss, 10);
    // rocprofiler bank-conflict rate: conflicts / (idx_active − conflicts) = 3/(12−3).
    assert_eq!(lookup("bankconf")(&m, ctx), Some(3.0 / 9.0));
    // rocprofiler MfmaUtil: mfma / (device_simds · GRBM_GUI_ACTIVE_per_XCD) = 40/(1216·100/8).
    assert_eq!(lookup("mfmautil")(&m, ctx), Some(40.0 / (1216.0 * 100.0 / 8.0)));
    // Achieved sclk (GHz): (gui / xcc) / wall / 1e9 = (100/8)/1e-6/1e9.
    assert_eq!(lookup("sclk")(&m, ctx), Some(100.0 / 8.0 / 1e-6 / 1e9));
    // svod matrix-duty (GRBM-free): mfma-busy / sq-busy = 40/80.
    assert_eq!(lookup("mfmaduty")(&m, ctx), Some(0.5));
    // L2 hit percentage: 100·30/(30+10). Column named `l2hitpct` (no `l2hit` clash).
    assert_eq!(lookup("l2hitpct")(&m, ctx), Some(75.0));
    // valuutil self-hides when its SQ inputs are absent.
    assert_eq!(lookup("valuutil")(&m, ctx), None);
    // mfmautil self-hides without the GRBM denominator (xcc_num = 0 in the default ctx).
    assert_eq!(lookup("mfmautil")(&m, DerivedCtx::default()), None);
    // sclk self-hides when the wall is host-derived (not GPU-stamped) — a host wall
    // (submit overhead) would yield a meaningless clock. (mfmautil is now clock-free.)
    assert_eq!(lookup("sclk")(&m, DerivedCtx { gpu_stamped: false, ..ctx }), None);
    // sclk self-hides without a captured wall time.
    assert_eq!(lookup("sclk")(&m, DerivedCtx { wall_secs: 0.0, ..ctx }), None);
    // L2 hit rate self-hides when a side is missing.
    m.remove(&PmcCounter::L2Miss);
    assert_eq!(lookup("l2hitpct")(&m, ctx), None);
}

#[test]
fn render_table_empty_and_host_only() {
    // Empty report renders nothing.
    assert_eq!(RunProfile::default().render_table(), "");

    // A host-only stage (no kernels) renders a single wall line, no metric table.
    let mut rp = RunProfile::default();
    rp.push(StageProfile::host("mel", Duration::from_millis(3)));
    let out = rp.render_table();
    assert!(out.contains("mel"), "host stage name present: {out:?}");
    assert!(out.contains("host"), "host stage tagged host: {out:?}");
    assert!(!out.contains("GFLOP/s"), "no metric columns for host-only: {out:?}");
}

#[test]
fn merge_accumulates_same_named_stages_and_appends_new() {
    let mut a = RunProfile::default();
    a.push(StageProfile::host("mel", Duration::from_millis(2)));
    let mut enc = StageProfile::host("encoder", Duration::from_millis(10));
    enc.meta.insert("rtf".into(), "0.02".into());
    a.push(enc);

    let mut b = RunProfile::default();
    let mut enc2 = StageProfile::host("encoder", Duration::from_millis(5)); // same name → sum wall + meta
    enc2.meta.insert("chunks".into(), "4".into());
    b.push(enc2);
    b.push(StageProfile::host("decode", Duration::from_millis(3))); // new name → appended

    a.merge(b);

    let names: Vec<&str> = a.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["mel", "encoder", "decode"], "matched stays in place, new appends");
    assert_eq!(a.stage("mel").unwrap().wall, Duration::from_millis(2), "untouched");

    let enc = a.stage("encoder").unwrap();
    assert_eq!(enc.wall, Duration::from_millis(15), "10 + 5 summed");
    assert_eq!(enc.meta.get("rtf").map(String::as_str), Some("0.02"), "kept");
    assert_eq!(enc.meta.get("chunks").map(String::as_str), Some("4"), "folded in");
}
