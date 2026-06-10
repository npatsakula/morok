use crate::amd::device::*;
use crate::error::Error;
use svod_dtype::AmdArch;

/// On hosts without `/dev/kfd` (or without a supported GPU), `open` must
/// surface a clean `Err` — never panic.
#[test]
fn open_without_gpu_or_unsupported_arch_is_clean_err() {
    let result = AmdDevice::open(0);
    match result {
        Ok(_) => {
            // Host has a supported AMD GPU — exercise the happy path too.
            // (We can't assert much without hardware-specific data.)
        }
        Err(Error::NoAmdGpu { .. }) | Err(Error::AmdAllocFailed { .. }) | Err(Error::AmdIoctl { .. }) => {
            // All acceptable.
        }
        Err(e) => panic!("unexpected error variant: {e:?}"),
    }
}

#[test]
fn aql_scratch_descriptor_gfx9_encoding() {
    // gfx9 SQ_BUF_RSRC scratch descriptor layout:
    //   WORD0 = lo32(va)
    //   WORD1 = hi32(va)[15:0] | SWIZZLE_ENABLE(bit31)
    //   WORD2 = lo32(size_per_xcc)   (NUM_RECORDS)
    //   WORD3 = SQ_BUF_RSRC: DST_SEL=XYZW, NUM_FORMAT=UINT, DATA_FORMAT=32,
    //           ELEMENT_SIZE=1, INDEX_STRIDE=3, ADD_TID_ENABLE=1 = 0x00EA4FAC
    let va: u64 = 0x1234_5678_9abc_d000;
    let d = AqlScratchDesc::gfx9(va, 0x0004_0000, 0xDEAD, 256);
    assert_eq!(d.resource_descriptor[0], 0x9abc_d000);
    assert_eq!(d.resource_descriptor[1], 0x8000_5678); // (0x12345678 & 0xFFFF) | 0x80000000
    assert_eq!(d.resource_descriptor[2], 0x0004_0000);
    assert_eq!(d.resource_descriptor[3], 0x00EA_4FAC);
    assert_eq!(d.backing_va, va);
    assert_eq!(d.tmpring_size, 0xDEAD);
    assert_eq!(d.wave64_lane_byte_size, 256); // wave64: priv_seg * 64 / 64
}

#[test]
fn aql_scratch_descriptor_gfx10_encoding() {
    // gfx10 (RDNA2 wave32) SRD — ROCr FillBufRsrcWord3_Gfx10: WORD0..2 as gfx9;
    // WORD3 = DST_SEL=XYZW(0xEAC) | FORMAT=BUF_FORMAT_32_UINT(0x14<<12)
    //       | ADD_TID_ENABLE(1<<23) | RESOURCE_LEVEL(1<<24) | OOB_SELECT(2<<28)
    //       = 0x21814EAC. Lane size for wave32 backing is priv_seg*32/64.
    let va: u64 = 0x1234_5678_9abc_d000;
    let d = AqlScratchDesc::gfx10(va, 0x0004_0000, 0xDEAD, 256 * 32 / 64);
    assert_eq!(d.resource_descriptor[0], 0x9abc_d000);
    assert_eq!(d.resource_descriptor[1], 0x8000_5678);
    assert_eq!(d.resource_descriptor[2], 0x0004_0000);
    assert_eq!(d.resource_descriptor[3], 0x2181_4EAC);
    assert_eq!(d.wave64_lane_byte_size, 128);
}

/// Cross-model parallelism (Pillar A): `assign_owner` must spread distinct
/// owners across distinct pool queues (fill-empty-first) up to `hw_queues`, and
/// the (hw_queues+1)-th owner must co-tenant an existing queue (the pool never
/// exceeds `hw_queues` — what kills the KFD-queue-exhaustion crash). Manual
/// probe: assumes isolated execution (the queue pool is process-global).
#[test]
#[ignore = "manual hardware probe; needs a real AMD GPU"]
fn assign_owner_spreads_then_cotenants() {
    use std::collections::HashSet;
    let Some(alloc) = super::test_support::amd_alloc_or_skip() else { return };
    let core = alloc.dev.core();
    if core.signal_pool().is_none() {
        core.install_signal_pool(crate::amd::signal::SignalPool::new(&alloc, 256).expect("signal pool"));
    }
    let n = core.hw_queues();
    assert!(n >= 1, "hw_queues must be >= 1");
    // Hold all owners alive so their queues stay referenced (strong_count > 1).
    let owners: Vec<_> = (0..n).map(|_| core.assign_owner(&alloc).expect("assign")).collect();
    let distinct: HashSet<_> = owners.iter().map(|o| o.queue_ptr()).collect();
    assert_eq!(distinct.len(), n, "first {n} owners must land on {n} distinct queues (fill-empty-first)");
    // The overflow owner has no idle queue left → co-tenants an existing one;
    // the pool stays capped at `n` (no unbounded KFD-queue creation).
    let extra = core.assign_owner(&alloc).expect("assign overflow");
    assert!(
        distinct.contains(&extra.queue_ptr()),
        "overflow owner must co-tenant an existing queue, not grow the pool"
    );
    eprintln!(
        "PROBE assign_owner: {n} owners → {n} distinct queues, overflow co-tenanted (pool capped at hw_queues={n})"
    );
}

#[test]
fn pack_tmpring_wavesize_width_by_arch() {
    // wave_scratch=0x3FFFF: cdna/rdna2(13b) truncate, rdna3(15b) truncates, rdna4(18b) keeps it.
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx942) >> 12, 0x1FFF);
    // RDNA2 (gfx10.3) shares the 13-bit WAVESIZE field with CDNA (gc_10_3_0 asic_reg).
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx1030) >> 12, 0x1FFF);
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx1100) >> 12, 0x7FFF);
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx1200) >> 12, 0x3FFFF);
    assert_eq!(pack_tmpring(0xABC, 0, &AmdArch::Gfx1100) & 0xFFF, 0xABC);
}
