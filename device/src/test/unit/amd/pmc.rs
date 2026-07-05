//! Unit tests for PMC PM4 stream construction (GPU-free).
//!
//! `build_streams` resolves every perf-counter register and chooses SET_SH vs
//! SET_UCONFIG by absolute address — so this exercises the register table and
//! address windows without a GPU (it would have caught a missing `_HI` register).
//! Both the gfx11 (SQ-only, 32-bit LO) and gfx942 (multi-block, 64-bit LO+HI,
//! per-XCC PRED_EXEC) programming forks are covered.

use crate::amd::pmc::{PmcArch, PmcGrid, PmcLayout, build_streams};
use crate::profile::PmcCounter;
use svod_dtype::AmdArch;

fn gfx11() -> PmcArch {
    PmcArch::for_arch(AmdArch::Gfx1100).expect("gfx11 PmcArch")
}

fn gfx9() -> PmcArch {
    PmcArch::for_arch(AmdArch::Gfx942).expect("gfx9 PmcArch")
}

#[test]
fn pmc_supported_arches() {
    // gfx942 (CDNA3) and the whole gfx11 family are wired; other gfx9 parts
    // (gfx950) and RDNA4 are not — their register bases/offsets differ.
    assert!(crate::amd::pmc::pmc_supported(AmdArch::Gfx942));
    assert!(crate::amd::pmc::pmc_supported(AmdArch::Gfx1100));
    assert!(crate::amd::pmc::pmc_supported(AmdArch::Gfx1151));
    assert!(!crate::amd::pmc::pmc_supported(AmdArch::Gfx950));
    assert!(!crate::amd::pmc::pmc_supported(AmdArch::Gfx1200));
    // gfx942 takes the gfx9 fork; gfx950 (also major 9) has no PmcArch.
    assert!(PmcArch::for_arch(AmdArch::Gfx942).is_some_and(|a| a.is_gfx9()));
    assert!(PmcArch::for_arch(AmdArch::Gfx950).is_none());
    // gfx11 supports only the SQ occupancy triple; gfx942 supports every counter.
    let g11 = gfx11();
    assert!(g11.supports(PmcCounter::SqBusyCycles));
    assert!(!g11.supports(PmcCounter::L2Hit), "gfx11 has no L2 counter");
    assert!(!g11.supports(PmcCounter::ValuMfmaBusyCycles), "gfx11 has no MFMA counter");
    assert!(gfx9().supports(PmcCounter::L2Hit));
}

#[test]
fn gfx11_layout_is_counter_major_lo_only() {
    let arch = gfx11();
    let grid = PmcGrid { xcc: 1, se: 2, sa: 2, wgp: 5 };
    assert_eq!(grid.instances(), 20);
    let layout = PmcLayout::plan(&arch, &PmcCounter::all(), &grid);
    // 3 counters × 20 instances × 4 bytes (LO only), counter-major offsets.
    assert_eq!(layout.total_bytes, 3 * 20 * 4);
    for (i, rec) in layout.recs.iter().enumerate() {
        assert_eq!(rec.offset, i * 20 * 4);
        assert_eq!(rec.records, 20);
        assert!(!rec.has_hi, "gfx11 reads 32-bit LO only");
    }
}

#[test]
fn gfx11_build_streams_resolves_all_registers() {
    let arch = gfx11();
    let grid = PmcGrid { xcc: 1, se: 2, sa: 2, wgp: 5 };
    let counters = PmcCounter::all();
    let layout = PmcLayout::plan(&arch, &counters, &grid);
    // Must not panic on any register name/window resolution.
    let (start, read) = build_streams(&arch, &counters, &grid, &layout, 0x1_0000, None, false);
    assert!(!start.is_empty(), "start stream programs SELECTs + CTRL");
    assert!(!read.is_empty(), "read stream copies counters out");
    // gfx11 has a single XCC: no PRED_EXEC packets in the read stream.
    assert!(start.len() + read.len() < 900, "pmc streams = {} dwords", start.len() + read.len());
}

#[test]
fn gfx9_layout_per_block_records() {
    let arch = gfx9();
    // MI300X-scale: 8 XCCs, 4 SE per XCC.
    let grid = PmcGrid { xcc: 8, se: 4, sa: 1, wgp: 1 };
    let counters = [
        PmcCounter::SqWaves,            // SQ per-SE: xcc*se = 32 records
        PmcCounter::SqBusyCycles,       // SQ per-SE (simd_mask=0xf aggregates SIMDs): 32 records
        PmcCounter::ValuMfmaBusyCycles, // SQ per-SE: 32 records
        PmcCounter::GrbmGuiActive,      // GRBM per-XCC: xcc = 8 records
        PmcCounter::L2Hit,              // TCC: xcc*16 = 128 records
        PmcCounter::L2Miss,             // TCC: 128 records
    ];
    let layout = PmcLayout::plan(&arch, &counters, &grid);
    let expected = [32usize, 32, 32, 8, 128, 128];
    let mut off = 0usize;
    for (rec, &recs) in layout.recs.iter().zip(&expected) {
        assert_eq!(rec.records, recs);
        assert!(rec.has_hi, "gfx9 reads 64-bit LO+HI");
        assert_eq!(rec.offset, off);
        off += recs * 8; // 8 bytes per 64-bit record
    }
    assert_eq!(layout.total_bytes, off);
}

#[test]
fn gfx9_build_streams_resolves_all_blocks() {
    let arch = gfx9();
    let grid = PmcGrid { xcc: 8, se: 4, sa: 1, wgp: 1 };
    // Exercise every gfx942 counter across all three blocks.
    let counters = [
        PmcCounter::SqBusyCycles,
        PmcCounter::SqWaves,
        PmcCounter::SqInstsValu,
        PmcCounter::SqInstsSalu,
        PmcCounter::LdsBankConflict,
        PmcCounter::LdsIdxActive,
        PmcCounter::ValuMfmaBusyCycles,
        PmcCounter::InstsMfma,
        PmcCounter::GrbmGuiActive,
        PmcCounter::L2Hit,
        PmcCounter::L2Miss,
    ];
    let layout = PmcLayout::plan(&arch, &counters, &grid);
    // Must resolve every SQ/GRBM/TCC SELECT and LO/HI register without panicking.
    let (start, read) = build_streams(&arch, &counters, &grid, &layout, 0x1_0000, None, false);
    assert!(!start.is_empty());
    assert!(!read.is_empty());
    // Multi-XCC read must wrap copy-outs in PRED_EXEC (header word 0xC0023800).
    let pred_hdr = crate::amd::sys::pm4::pred_exec(1, 1)[0];
    assert!(read.contains(&pred_hdr), "multi-XCC read stream uses PRED_EXEC scoping");
}
