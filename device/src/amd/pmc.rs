//! AMD hardware performance-counter programming (PMC), arch-parameterized for
//! gfx11 (RDNA3.5) and gfx942 (CDNA3).
//!
//! The PM4 sequence programs `*_PERFCOUNTER*_SELECT`, starts/stops the perfmon via
//! `CP_PERFMON_CNTL`, and `COPY_DATA`s the per-instance counters into a GTT buffer
//! — summed over the block's instance grid when read back. gfx11 iterates the
//! SE/SA/WGP grid on a single XCC and reads 32-bit `_LO`; gfx942 iterates SE-only
//! for SQ (aggregating both SHs via `SQ_PERFCOUNTER_MASK`), the 16 TCC instances,
//! and a single GRBM, wraps each XCC in a `PRED_EXEC` predicate, and reads full
//! 64-bit `_LO`+`_HI`. Callers gate on [`pmc_supported`] and a stable power state
//! (`profile_standard`).

use std::ptr::NonNull;
use std::sync::Arc;

use crate::allocator::RawBuffer;
use crate::amd::am::regs::{self, RegDef};
use crate::amd::signal::AmdSignal;
use crate::amd::sys::pm4;
use crate::amd::topology::AmdNode;
use crate::profile::{CounterSet, PmcCounter};
use svod_dtype::AmdArch;

const SET_SH_START: u64 = pm4::PACKET3_SET_SH_REG_START as u64;
const SET_SH_END: u64 = 0x3000;
const SET_UCONFIG_START: u64 = pm4::PACKET3_SET_UCONFIG_REG_START as u64;

/// Number of TCC (L2) instances iterated for a gfx9 L2 counter.
const TCC_INSTANCES: u32 = 16;

/// `regRLC_PERFMON_CLK_CNTL` GC segment-1 offset (abs `0xDCBF`, UCONFIG). Absent
/// from the vendored register table. Writing `1` disables RLC perfmon
/// clock-gating (required on gfx9/CDNA or the SQ perfmon clock stays gated and
/// every SQ counter reads zero); `0` restores gating. Ref: AMD aqlprofile
/// `pmc_builder.h` (`Start`/`Stop`, gfx9-only).
const RLC_PERFMON_CLK_CNTL_OFFSET: u64 = 0x3cbf;

/// Whether PMC hardware-counter collection is implemented for this exact GPU.
/// Only gfx942 (CDNA3) is wired on the gfx9 side — the register bases/offsets in
/// [`PmcArch::for_arch`] are gfx942-specific and differ on gfx908/gfx90a/gfx950
/// — plus the whole gfx11 (RDNA3/3.5) family. Everything else falls back to
/// timing-only profiling.
pub fn pmc_supported(arch: AmdArch) -> bool {
    matches!(arch, AmdArch::Gfx942) || arch.gfx_major() == 11
}

/// Whether any AMD GPU is pinned to a stable power state (`profile_standard`),
/// required for meaningful perf-counter values. Scans the DRM sysfs nodes; on a
/// single-GPU host this is exact. Set it with `amd-smi set -l stable_std`.
pub fn stable_pstate() -> bool {
    // Opt-in bypass: on SR-IOV VFs the guest cannot pin the clock (the PF/hypervisor owns it),
    // so `profile_standard` is unreachable. Event counters (e.g. LDS_BANK_CONFLICT) are counts,
    // not rates, so they're valid on the `auto` clock — cycle-derived rates (SQ_BUSY %) are the
    // only ones the variable clock skews. `SVOD_PMC_FORCE=1` collects anyway, caveat accepted.
    if std::env::var("SVOD_PMC_FORCE").ok().map(|s| s != "0").unwrap_or(false) {
        return true;
    }
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return false;
    };
    entries.flatten().any(|e| {
        std::fs::read_to_string(e.path().join("device/power_dpm_force_performance_level"))
            .is_ok_and(|s| s.trim() == "profile_standard")
    })
}

/// The perf-counter block a [`PmcCounter`] lives in on gfx9. Each block has its
/// own `reg{PREFIX}_PERFCOUNTER{n}_{SELECT,LO,HI}` register bank and its own
/// instance grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PmcBlock {
    Sq,
    Grbm,
    Tcc,
}

impl PmcBlock {
    fn reg_prefix(self) -> &'static str {
        match self {
            Self::Sq => "SQ",
            Self::Grbm => "GRBM",
            Self::Tcc => "TCC",
        }
    }

    fn slot(self) -> usize {
        match self {
            Self::Sq => 0,
            Self::Grbm => 1,
            Self::Tcc => 2,
        }
    }
}

/// gfx942 (CDNA3) `(block, perf_sel)` for each counter. Values cross-checked
/// against tinygrad `runtime/autogen/am/pmc.py['gfx942']`.
fn gfx9_block_sel(c: PmcCounter) -> (PmcBlock, u32) {
    match c {
        PmcCounter::SqBusyCycles => (PmcBlock::Sq, 3),
        PmcCounter::SqWaves => (PmcBlock::Sq, 4),
        PmcCounter::SqInstsValu => (PmcBlock::Sq, 26),
        PmcCounter::SqInstsSalu => (PmcBlock::Sq, 60),
        PmcCounter::LdsIdxActive => (PmcBlock::Sq, 131),
        PmcCounter::LdsBankConflict => (PmcBlock::Sq, 126),
        PmcCounter::ValuMfmaBusyCycles => (PmcBlock::Sq, 77),
        PmcCounter::InstsMfma => (PmcBlock::Sq, 56),
        // gfx942 SQ wait/active selects (cross-checked vs tinygrad pmc.py['gfx942']).
        PmcCounter::SqWaitAny => (PmcBlock::Sq, 90),
        PmcCounter::SqWaitInstLds => (PmcBlock::Sq, 96),
        PmcCounter::SqActiveInstVmem => (PmcBlock::Sq, 102),
        PmcCounter::GrbmGuiActive => (PmcBlock::Grbm, 2),
        PmcCounter::L2Hit => (PmcBlock::Tcc, 17),
        PmcCounter::L2Miss => (PmcBlock::Tcc, 19),
    }
}

/// gfx11 SQ perf-counter selector (perf_sel) for each [`PmcCounter`], or `None`
/// when the counter has no gfx11 SQ definition. Only the arch-neutral SQ
/// occupancy triple is wired on gfx11; callers filter against
/// [`PmcArch::supports`] before emitting, so an unsupported counter never
/// reaches stream construction (no panic on user `SVOD_PMC=…` input).
fn gfx11_sq_perf_sel(c: PmcCounter) -> Option<u32> {
    Some(match c {
        PmcCounter::SqBusyCycles => 3,
        PmcCounter::SqWaves => 4,
        PmcCounter::SqInstsValu => 62,
        _ => return None,
    })
}

/// Arch parameters: the GC segment bases, the vendored register table, and the
/// gfx9-vs-gfx11 programming fork.
#[derive(Clone, Copy)]
pub struct PmcArch {
    /// GC IP segment bases (`[seg0, seg1]`); only seg0 differs between arches.
    bases: [u64; 2],
    regs: &'static [RegDef],
    is_gfx9: bool,
}

impl PmcArch {
    /// The arch parameters for an exact GPU, or `None` when PMC is unimplemented
    /// for it. The gfx9 register bases/offsets are gfx942 (gfx9_4_3) specific, so
    /// only [`AmdArch::Gfx942`] takes the gfx9 fork — other gfx9 parts
    /// (gfx908/gfx90a/gfx950) differ and must not be handed these values.
    pub fn for_arch(arch: AmdArch) -> Option<Self> {
        match arch {
            // gfx9_4_3 (CDNA3/MI300): only seg0 base differs (0x2000 vs 0x1260);
            // all seg1 perfcounter regs share PM4 addresses with gfx11.
            AmdArch::Gfx942 => Some(Self { bases: [0x2000, 0xA000], regs: regs::GC_9_4_3, is_gfx9: true }),
            // gfx11 family: verified vs Mesa (regGRBM_GFX_INDEX → abs 0xC200 dword;
            // regSQ_PERFCOUNTER0_SELECT → abs 0xD9C0 dword).
            a if a.gfx_major() == 11 => Some(Self { bases: [0x1260, 0xA000], regs: regs::GC_11_5_0, is_gfx9: false }),
            _ => None,
        }
    }

    /// Whether this arch's SQ counter is a gfx9 CDNA part (multi-XCC AQL PMC),
    /// versus a single-XCC gfx11 part (PM4-ring PMC).
    pub fn is_gfx9(&self) -> bool {
        self.is_gfx9
    }

    /// Whether a counter has a perf-select on this arch. gfx9 (gfx942) defines
    /// every [`PmcCounter`]; gfx11 only the SQ occupancy triple. Callers filter
    /// the requested set through this before stream construction.
    pub fn supports(&self, c: PmcCounter) -> bool {
        if self.is_gfx9 { true } else { gfx11_sq_perf_sel(c).is_some() }
    }

    fn reg(&self, name: &str) -> &'static RegDef {
        regs::find(self.regs, name).unwrap_or_else(|| panic!("GC register table missing {name}"))
    }

    /// Absolute dword address (`base[segment] + offset`).
    fn abs_addr(&self, r: &RegDef) -> u64 {
        self.bases[r.segment as usize] + r.offset as u64
    }

    /// Emit a raw-value register write at an absolute dword address, choosing the
    /// SET_SH vs SET_UCONFIG packet by window (same dispatch as [`wreg`]). For
    /// registers not present in the vendored table.
    fn wreg_abs(&self, out: &mut Vec<u32>, abs: u64, val: u32) {
        if (SET_SH_START..SET_SH_END).contains(&abs) {
            out.extend(pm4::set_sh_reg((abs - SET_SH_START) as u32, &[val]));
        } else {
            out.extend(pm4::set_uconfig_reg((abs - SET_UCONFIG_START) as u32, &[val]));
        }
    }

    /// Emit a register write as a SET_SH_REG or SET_UCONFIG_REG packet, chosen by
    /// the register's absolute address window.
    fn wreg(&self, out: &mut Vec<u32>, name: &str, fields: &[(&str, u32)]) {
        let r = self.reg(name);
        let val = r.encode(fields) as u32;
        let a = self.abs_addr(r);
        if (SET_SH_START..SET_SH_END).contains(&a) {
            out.extend(pm4::set_sh_reg((a - SET_SH_START) as u32, &[val]));
        } else {
            out.extend(pm4::set_uconfig_reg((a - SET_UCONFIG_START) as u32, &[val]));
        }
    }

    /// `GRBM_GFX_INDEX` set to broadcast all SE/SA(SH)/instances (setup/control).
    fn grbm_broadcast(&self, out: &mut Vec<u32>) {
        let sh = if self.is_gfx9 { "sh_broadcast_writes" } else { "sa_broadcast_writes" };
        self.wreg(out, "regGRBM_GFX_INDEX", &[("se_broadcast_writes", 1), (sh, 1), ("instance_broadcast_writes", 1)]);
    }

    /// gfx11: index one (se, sa, wgp) so the following COPY_DATA reads that WGP's
    /// counter. `instance_index` packs the WGP into bits [2+].
    fn grbm_index_gfx11(&self, out: &mut Vec<u32>, se: u32, sa: u32, wgp: u32) {
        self.wreg(out, "regGRBM_GFX_INDEX", &[("se_index", se), ("sa_index", sa), ("instance_index", wgp << 2)]);
    }

    /// gfx9: index one SE, broadcasting across SH and instance (SQ counters sum
    /// both SHs via `SQ_PERFCOUNTER_MASK`).
    fn grbm_index_se(&self, out: &mut Vec<u32>, se: u32) {
        self.wreg(
            out,
            "regGRBM_GFX_INDEX",
            &[("se_index", se), ("sh_broadcast_writes", 1), ("instance_broadcast_writes", 1)],
        );
    }

    /// gfx9: index one instance (a TCC channel), broadcasting across SE and SH.
    fn grbm_index_instance(&self, out: &mut Vec<u32>, inst: u32) {
        self.wreg(
            out,
            "regGRBM_GFX_INDEX",
            &[("instance_index", inst), ("se_broadcast_writes", 1), ("sh_broadcast_writes", 1)],
        );
    }
}

/// The instance grid to iterate for counter readback, derived from KFD topology.
/// gfx11 uses `se·sa·wgp` on a single XCC; gfx9 SQ counters are SE-only per XCC
/// (`sa`/`wgp` are `1`) with `xcc` XCCs iterated via `PRED_EXEC`.
#[derive(Clone, Copy, Debug)]
pub struct PmcGrid {
    pub xcc: u32,
    pub se: u32,
    pub sa: u32,
    pub wgp: u32,
}

impl PmcGrid {
    pub fn from_node(node: &AmdNode, arch: &PmcArch) -> Self {
        let xcc = node.num_xcc.max(1);
        let sa = node.simd_arrays_per_engine.max(1);
        let se = (node.array_count.max(1) / sa / xcc).max(1);
        if arch.is_gfx9 {
            // gfx9 SQ iterates SE-only (both SHs aggregated in-hardware); the XCC
            // dimension is the PRED_EXEC predicate, not part of the SE grid.
            return Self { xcc, se, sa: 1, wgp: 1 };
        }
        let cu_total = (node.simd_count.max(1) / node.simd_per_cu.max(1)) / xcc;
        let cu_per_sa = (cu_total / node.array_count.max(1)).max(1);
        let wgp = (cu_per_sa / 2).max(1); // WGP = 2 CU on RDNA
        // Bound the per-dispatch packet count against pathological topology.
        Self { xcc: 1, se, sa, wgp: wgp.min(16) }
    }

    /// gfx11 per-counter instance count (`se·sa·wgp`, single XCC).
    pub fn instances(&self) -> u32 {
        self.se * self.sa * self.wgp
    }
}

/// One counter's slice of the readback buffer: its first record's byte offset,
/// how many instance records follow, and whether each record carries a `_HI`
/// dword (a 64-bit counter) or is `_LO`-only (32-bit).
#[derive(Clone, Copy, Debug)]
pub struct CounterRec {
    pub offset: usize,
    pub records: usize,
    pub has_hi: bool,
}

impl CounterRec {
    /// Bytes between consecutive records (8 for 64-bit LO+HI, 4 for LO-only).
    fn stride(&self) -> usize {
        if self.has_hi { 8 } else { 4 }
    }
}

/// The readback-buffer layout: one [`CounterRec`] per requested counter (in
/// request order) plus the total buffer size. Per-block record counts can differ
/// (gfx9 SQ=xcc·se, GRBM=xcc, TCC=xcc·16), so this replaces a single uniform
/// instance count.
#[derive(Clone, Debug)]
pub struct PmcLayout {
    pub recs: Vec<CounterRec>,
    pub total_bytes: usize,
}

impl PmcLayout {
    /// Compute the buffer layout for `counters` over `grid` on `arch`.
    pub fn plan(arch: &PmcArch, counters: &[PmcCounter], grid: &PmcGrid) -> Self {
        if arch.is_gfx9 {
            let mut recs = Vec::with_capacity(counters.len());
            let mut off = 0usize;
            for s in gfx9_schedule(counters, grid) {
                let rec = CounterRec { offset: off, records: s.records, has_hi: true };
                off += rec.records * rec.stride();
                recs.push(rec);
            }
            Self { recs, total_bytes: off }
        } else {
            // gfx11: uniform `instances` LO-only 4-byte records, counter-major.
            let inst = grid.instances() as usize;
            let recs = (0..counters.len())
                .map(|i| CounterRec { offset: i * inst * 4, records: inst, has_hi: false })
                .collect();
            Self { recs, total_bytes: counters.len() * inst * 4 }
        }
    }
}

/// One gfx9 counter's program: which block/SELECT slot it occupies, its
/// perf-select value, and how many instance records it reads.
#[derive(Clone, Copy, Debug)]
struct Gfx9Sched {
    block: PmcBlock,
    pcid: u32,
    perf_sel: u32,
    records: usize,
}

/// Assign each counter a per-block SELECT slot (in request order) and its record
/// count. Shared by [`PmcLayout::plan`] and [`build_streams`] so the buffer
/// layout and the copy-out program always agree.
fn gfx9_schedule(counters: &[PmcCounter], grid: &PmcGrid) -> Vec<Gfx9Sched> {
    let mut next = [0u32; 3];
    counters
        .iter()
        .map(|&c| {
            let (block, perf_sel) = gfx9_block_sel(c);
            let pcid = next[block.slot()];
            next[block.slot()] += 1;
            let records = match block {
                // One read per (XCC, SE): the SQ SELECT's `simd_mask=0xf` aggregates all 4
                // SIMDs into the SE-global total, so a per-SE read already matches rocprofv3's
                // per-XCC aggregate (iterating `instance_index` per SIMD would 4×-overcount).
                PmcBlock::Sq => (grid.xcc * grid.se) as usize,
                PmcBlock::Grbm => grid.xcc as usize,
                PmcBlock::Tcc => (grid.xcc * TCC_INSTANCES) as usize,
            };
            Gfx9Sched { block, pcid, perf_sel, records }
        })
        .collect()
}

/// Emit a GPU-clock timestamp probe (`release_mem` in timestamp mode — a bare
/// data write, no cache flush and no signal decrement) writing to `addr`. On a
/// multi-XCC gang IB it is scoped to XCC 0 via `PRED_EXEC` so exactly one CP
/// writes the (globally-synchronized) clock — avoiding an 8-way write race while
/// still bracketing the kernel. Placed just before the kernel (start probe) and
/// just after it (end probe); `end − start` is device-stamped kernel time. Being
/// flush-free, it does not perturb the bracketed counter values.
fn push_ts_probe(out: &mut Vec<u32>, addr: u64, is_gfx9: bool, xcc: u32) {
    let probe = pm4::release_mem_timestamp(addr, is_gfx9);
    if xcc > 1 {
        out.extend_from_slice(&pm4::pred_exec(1 << 0, probe.len() as u32));
    }
    out.extend_from_slice(&probe);
}

/// Build the PMC start (program + arm) and read (sample + copy-out) PM4 streams.
/// `layout` must be the one [`PmcLayout::plan`] produced for the same
/// `arch`/`counters`/`grid` (it fixes the readback offsets).
///
/// `ts_addrs` (when set) injects `(start, end)` GPU-clock timestamp probes
/// bracketing the kernel so the caller can read device kernel time. `self_flush`
/// appends a final cache write-back so the copied counters reach memory without
/// an external completion flush (required on the AQL vendor-IB path; the gfx9
/// stream always self-flushes, the gfx11 PM4-ring stream relies on the
/// dispatch's completion `release_mem` and passes `false`).
pub fn build_streams(
    arch: &PmcArch,
    counters: &[PmcCounter],
    grid: &PmcGrid,
    layout: &PmcLayout,
    buf_va: u64,
    ts_addrs: Option<(u64, u64)>,
    self_flush: bool,
) -> (Vec<u32>, Vec<u32>) {
    if arch.is_gfx9 {
        build_streams_gfx9(arch, counters, grid, layout, buf_va, ts_addrs)
    } else {
        build_streams_gfx11(arch, counters, grid, buf_va, ts_addrs, self_flush)
    }
}

/// gfx11: single-XCC SE/SA/WGP grid, 32-bit `_LO` readback (even-numbered SQ
/// SELECT registers). Byte-identical to the original gfx11-only path.
fn build_streams_gfx11(
    arch: &PmcArch,
    counters: &[PmcCounter],
    grid: &PmcGrid,
    buf_va: u64,
    ts_addrs: Option<(u64, u64)>,
    self_flush: bool,
) -> (Vec<u32>, Vec<u32>) {
    let instances = grid.instances();

    // ── start: stop, program SELECTs + CTRL, enable, start ──
    let mut start = Vec::new();
    arch.grbm_broadcast(&mut start);
    arch.wreg(&mut start, "regCP_PERFMON_CNTL", &[("perfmon_state", 0)]);
    arch.wreg(&mut start, "regSQ_PERFCOUNTER_CTRL", &[("cs_en", 1), ("ps_en", 1), ("gs_en", 1), ("hs_en", 1)]);
    arch.wreg(&mut start, "regSQ_PERFCOUNTER_CTRL2", &[("force_en", 1), ("vmid_en", 0xffff)]);
    for (i, &c) in counters.iter().enumerate() {
        // gfx11 SQ uses even-numbered SELECT registers; readback uses index i.
        // Counters are pre-filtered to the arch's supported set (`PmcArch::supports`).
        let sel = gfx11_sq_perf_sel(c).expect("counter filtered to gfx11-supported set before build_streams");
        arch.wreg(&mut start, &format!("regSQ_PERFCOUNTER{}_SELECT", i * 2), &[("perf_sel", sel)]);
    }
    arch.wreg(&mut start, "regCOMPUTE_PERFCOUNT_ENABLE", &[("perfcount_enable", 1)]);
    arch.grbm_broadcast(&mut start);
    arch.wreg(&mut start, "regCP_PERFMON_CNTL", &[("perfmon_state", 0)]);
    arch.wreg(&mut start, "regCP_PERFMON_CNTL", &[("perfmon_state", 1)]);
    // Start-of-kernel timestamp: after the perfmon is armed, just before the
    // dispatch launches (single-XCC on gfx11 → no PRED_EXEC scoping).
    if let Some((start_ts, _)) = ts_addrs {
        push_ts_probe(&mut start, start_ts, /*is_gfx9=*/ false, grid.xcc);
    }

    // ── read: sample, then copy each counter's _LO at every instance ──
    let los: Vec<u32> = counters
        .iter()
        .enumerate()
        .map(|(i, _)| arch.abs_addr(arch.reg(&format!("regSQ_PERFCOUNTER{i}_LO"))) as u32)
        .collect();
    let mut read = Vec::new();
    // End-of-kernel timestamp: first thing in the read stream (which runs after
    // the kernel), before the perfmon sample.
    if let Some((_, end_ts)) = ts_addrs {
        push_ts_probe(&mut read, end_ts, /*is_gfx9=*/ false, grid.xcc);
    }
    arch.grbm_broadcast(&mut read);
    arch.wreg(&mut read, "regCP_PERFMON_CNTL", &[("perfmon_state", 1), ("perfmon_sample_enable", 1)]);
    let mut j = 0u32;
    for se in 0..grid.se {
        for sa in 0..grid.sa {
            for wgp in 0..grid.wgp {
                arch.grbm_index_gfx11(&mut read, se, sa, wgp);
                for (i, &lo) in los.iter().enumerate() {
                    let off = buf_va + ((i as u32 * instances + j) as u64) * 4;
                    read.extend(pm4::copy_data_reg_to_mem(lo, off));
                }
                j += 1;
            }
        }
    }
    // Stop the perfmon after the final sample (global CP state — leaving it on
    // perturbs other work).
    arch.grbm_broadcast(&mut read);
    arch.wreg(&mut read, "regCP_PERFMON_CNTL", &[("perfmon_state", 0)]);
    // On the AQL vendor-IB path there is no external completion flush, so write
    // back L2 here (mirrors the gfx9 stream's trailing flush); the PM4-ring path
    // relies on the dispatch's completion `release_mem` and passes `false`, so
    // the ring stream stays byte-identical.
    if self_flush {
        read.extend_from_slice(&pm4::acquire_mem());
    }
    (start, read)
}

/// gfx9 (CDNA3): consecutive SQ SELECT registers with SQ mask + inline vmid; the
/// read copies full 64-bit `_LO`+`_HI` per instance, iterating SE-only for SQ,
/// the 16 TCC channels for L2, and a single GRBM — each XCC wrapped in
/// `PRED_EXEC`.
fn build_streams_gfx9(
    arch: &PmcArch,
    counters: &[PmcCounter],
    grid: &PmcGrid,
    layout: &PmcLayout,
    buf_va: u64,
    ts_addrs: Option<(u64, u64)>,
) -> (Vec<u32>, Vec<u32>) {
    let sched = gfx9_schedule(counters, grid);

    // ── start: stop, program per-block SELECTs + CTRL/MASK, enable, start ──
    let mut start = Vec::new();
    // Drain the CS before (re)programming the perfmon (aqlprofile brackets Start
    // with CS_PARTIAL_FLUSH). Without it, waves still in flight from a prior
    // dispatch race the perfmon reset/start and the counts come out
    // non-deterministic (partial vs full) on the async AQL queue.
    start.extend_from_slice(&pm4::event_write(pm4::CS_PARTIAL_FLUSH, pm4::EVENT_INDEX_PARTIAL_FLUSH));
    arch.grbm_broadcast(&mut start);
    // gfx9/CDNA REQUIRES disabling RLC perfmon clock-gating or the SQ perfmon
    // clock stays gated and every SQ counter reads exactly zero (GRBM and other
    // always-clocked blocks still count — the tell-tale gfx9 symptom). Restored
    // at the end of the read stream. Ref: AMD aqlprofile `pmc_builder.h`
    // (`Start`/`Stop`, guarded `if GFXIP_LEVEL == 9`).
    arch.wreg_abs(&mut start, arch.bases[1] + RLC_PERFMON_CLK_CNTL_OFFSET, 1);
    arch.wreg(&mut start, "regCP_PERFMON_CNTL", &[("perfmon_state", 0)]);
    // gfx9 carries the vmid mask inline in CTRL (no CTRL2 register write).
    arch.wreg(
        &mut start,
        "regSQ_PERFCOUNTER_CTRL",
        &[("cs_en", 1), ("ps_en", 1), ("gs_en", 1), ("hs_en", 1), ("vmid_mask", 0xffff)],
    );
    for s in &sched {
        let name = format!("reg{}_PERFCOUNTER{}_SELECT", s.block.reg_prefix(), s.pcid);
        if s.block == PmcBlock::Sq {
            arch.wreg(
                &mut start,
                &name,
                &[("perf_sel", s.perf_sel), ("simd_mask", 0xf), ("sqc_bank_mask", 0xf), ("sqc_client_mask", 0xf)],
            );
        } else {
            arch.wreg(&mut start, &name, &[("perf_sel", s.perf_sel)]);
        }
    }
    arch.wreg(&mut start, "regSQ_PERFCOUNTER_MASK", &[("sh0_mask", 0xffff), ("sh1_mask", 0xffff)]);
    arch.wreg(&mut start, "regCOMPUTE_PERFCOUNT_ENABLE", &[("perfcount_enable", 1)]);
    arch.grbm_broadcast(&mut start);
    arch.wreg(&mut start, "regCP_PERFMON_CNTL", &[("perfmon_state", 0)]);
    arch.wreg(&mut start, "regCP_PERFMON_CNTL", &[("perfmon_state", 1)]);
    // Drain again so the perfmon is fully armed before the bracketed dispatch's
    // waves launch (aqlprofile's trailing Start barrier).
    start.extend_from_slice(&pm4::event_write(pm4::CS_PARTIAL_FLUSH, pm4::EVENT_INDEX_PARTIAL_FLUSH));
    // Start-of-kernel timestamp: perfmon is armed and the CS is drained, so this
    // stamps the instant just before the bracketed kernel launches.
    if let Some((start_ts, _)) = ts_addrs {
        push_ts_probe(&mut start, start_ts, /*is_gfx9=*/ true, grid.xcc);
    }

    // ── read: sample, then copy each counter's 64-bit value at every instance ──
    let mut read = Vec::new();
    // End-of-kernel timestamp: the read IB runs after the kernel (vendor packet
    // BARRIER bit), so this first probe stamps ~kernel end, before the sample.
    if let Some((_, end_ts)) = ts_addrs {
        push_ts_probe(&mut read, end_ts, /*is_gfx9=*/ true, grid.xcc);
    }
    arch.grbm_broadcast(&mut read);
    arch.wreg(&mut read, "regCP_PERFMON_CNTL", &[("perfmon_state", 1), ("perfmon_sample_enable", 1)]);
    // Drain the dispatch's waves and let the sample latch before copying the
    // counters out (aqlprofile's WaitIdle inside the read).
    read.extend_from_slice(&pm4::event_write(pm4::CS_PARTIAL_FLUSH, pm4::EVENT_INDEX_PARTIAL_FLUSH));
    for (s, rec) in sched.iter().zip(&layout.recs) {
        let lo = arch.abs_addr(arch.reg(&format!("reg{}_PERFCOUNTER{}_LO", s.block.reg_prefix(), s.pcid))) as u32;
        let hi = arch.abs_addr(arch.reg(&format!("reg{}_PERFCOUNTER{}_HI", s.block.reg_prefix(), s.pcid))) as u32;
        let mut j = 0u64;
        for xcc in 0..grid.xcc {
            // Build this XCC's copy-out packets, then scope them to the XCC with a
            // PRED_EXEC predicate whose payload count is their dword length.
            let mut inner = Vec::new();
            let mut copy_at = |inner: &mut Vec<u32>| {
                let off = buf_va + rec.offset as u64 + j * 8;
                inner.extend(pm4::copy_data_reg_to_mem(lo, off));
                inner.extend(pm4::copy_data_reg_to_mem(hi, off + 4));
                j += 1;
            };
            match s.block {
                PmcBlock::Sq => {
                    // Every SQ counter: one SE-broadcast read per SE (the `simd_mask=0xf`
                    // SELECT already aggregates the 4 SIMDs into the SE total).
                    for se in 0..grid.se {
                        arch.grbm_index_se(&mut inner, se);
                        copy_at(&mut inner);
                    }
                }
                PmcBlock::Grbm => {
                    // GRBM is per-XCC (rocprofv3 `INSTANCE[0:0] × XCC`); one
                    // broadcast read per XCC.
                    arch.grbm_broadcast(&mut inner);
                    copy_at(&mut inner);
                }
                PmcBlock::Tcc => {
                    for inst in 0..TCC_INSTANCES {
                        arch.grbm_index_instance(&mut inner, inst);
                        copy_at(&mut inner);
                    }
                }
            }
            if grid.xcc > 1 {
                read.extend(pm4::pred_exec(1 << xcc, inner.len() as u32));
            }
            read.extend(inner);
        }
    }
    arch.grbm_broadcast(&mut read);
    arch.wreg(&mut read, "regCP_PERFMON_CNTL", &[("perfmon_state", 0)]);
    // Restore RLC perfmon clock-gating disabled in the start stream (global
    // state — leaving it off perturbs power management of later, non-profiled work).
    arch.wreg_abs(&mut read, arch.bases[1] + RLC_PERFMON_CLK_CNTL_OFFSET, 0);
    // Write back L2 so the COPY_DATA counter values reach memory before the host
    // harvests them. On the PM4 path the external completion `release_mem`
    // (cache_flush) did this; the AQL vendor-packet path has no such external
    // flush, so the read stream must be self-contained (matches aqlprofile's
    // `BuildCacheFlushPacket` at the end of `Read`). gfx9 ACQUIRE_MEM with TC_WB.
    read.extend_from_slice(&pm4::acquire_mem_gfx9());
    (start, read)
}

/// A profiled-dispatch handle that also carries PMC counters. Holds the GTT
/// readback buffer alive until [`counters`](Self::counters) is read after sync,
/// and delegates timestamps to the dispatch's completion signal.
pub struct PmcHandle {
    ts: Arc<AmdSignal>,
    _buf: RawBuffer,
    host: NonNull<u8>,
    counters: Vec<PmcCounter>,
    layout: PmcLayout,
    /// Device topology carried into [`CounterSet`] so the profiler can normalize
    /// cross-block derived metrics (MFMA utilization). `(num_xcc, device_simds)`.
    device: (u32, u32),
    /// Extra GPU buffers kept alive until harvest — the AQL path's two vendor
    /// PM4-IB buffers (the CP reads them while the dispatch is in flight; freeing
    /// them early would let the CP walk unmapped VRAM). Empty on the PM4 path,
    /// where the streams are inlined into the ring.
    _ib_bufs: Vec<RawBuffer>,
}

// SAFETY: `host` points into a GTT allocation owned by `_buf` (kept alive for the
// handle's lifetime); reads happen only after the owning plan synchronizes.
unsafe impl Send for PmcHandle {}
unsafe impl Sync for PmcHandle {}

impl PmcHandle {
    pub fn new(
        ts: Arc<AmdSignal>,
        buf: RawBuffer,
        host: NonNull<u8>,
        counters: Vec<PmcCounter>,
        layout: PmcLayout,
    ) -> Self {
        Self { ts, _buf: buf, host, counters, layout, device: (0, 0), _ib_bufs: Vec::new() }
    }

    /// Attach the device topology used to normalize cross-block derived metrics:
    /// `num_xcc` and the device-total SIMD count (`simd_count`).
    pub fn with_device(mut self, num_xcc: u32, device_simds: u32) -> Self {
        self.device = (num_xcc, device_simds);
        self
    }

    /// Keep extra GPU buffers alive for the handle's lifetime (the AQL vendor
    /// PM4-IB buffers). Returns `self` for chaining at the call site.
    pub fn keep_alive(mut self, ib_bufs: Vec<RawBuffer>) -> Self {
        self._ib_bufs = ib_bufs;
        self
    }
}

impl crate::sync::DispatchTimestamps for PmcHandle {
    fn timestamps_ns(&self) -> Option<(u64, u64)> {
        self.ts.timestamps_ns()
    }

    fn counters(&self) -> Option<CounterSet> {
        let mut values = std::collections::BTreeMap::new();
        for (&c, rec) in self.counters.iter().zip(&self.layout.recs) {
            let stride = rec.stride();
            let mut sum = 0u64;
            for k in 0..rec.records {
                let base = rec.offset + k * stride;
                // SAFETY: `host` is a live GTT buffer of `total_bytes`, written by
                // COPY_DATA and flushed before the host reads post-sync.
                let v = unsafe {
                    let lo = (self.host.as_ptr().add(base) as *const u32).read_unaligned() as u64;
                    if rec.has_hi {
                        let hi = (self.host.as_ptr().add(base + 4) as *const u32).read_unaligned() as u64;
                        lo | (hi << 32)
                    } else {
                        lo
                    }
                };
                sum = sum.saturating_add(v);
            }
            values.insert(c, sum);
        }
        Some(CounterSet { values, xcc_num: self.device.0, device_simds: self.device.1 })
    }
}
