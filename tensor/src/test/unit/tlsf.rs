use super::*;

#[test]
fn test_alloc_returns_zero_offset_for_first_block() {
    let mut a = TlsfAllocator::new(4096, 0, 256, 32);
    assert_eq!(a.alloc(256, 1).unwrap(), 0);
}

#[test]
fn test_alloc_free_roundtrip_returns_storage_to_free_pool() {
    let mut a = TlsfAllocator::new(4096, 0, 256, 32);
    let off1 = a.alloc(256, 1).unwrap();
    let off2 = a.alloc(256, 1).unwrap();
    let off3 = a.alloc(256, 1).unwrap();
    assert_ne!(off1, off2);
    assert_ne!(off2, off3);

    a.free(off1).unwrap();
    a.free(off2).unwrap();
    a.free(off3).unwrap();

    // After freeing all, the next allocation should fit even at the maximum
    // single-block size — coalescing must have rebuilt the original block.
    let big = a.alloc(4096, 1).unwrap();
    assert_eq!(big, 0, "fully-coalesced arena must hand back the whole region");
}

#[test]
fn test_merge_left_then_right() {
    let mut a = TlsfAllocator::new(4096, 0, 256, 32);
    let o1 = a.alloc(256, 1).unwrap();
    let o2 = a.alloc(256, 1).unwrap();
    let o3 = a.alloc(256, 1).unwrap();

    // Free middle then left — left+middle coalesces, then merging right
    // should pull in (originally) o3 once it's freed too.
    a.free(o2).unwrap();
    a.free(o1).unwrap();
    a.free(o3).unwrap();

    // Whole arena should be reusable.
    let big = a.alloc(4096, 1).unwrap();
    assert_eq!(big, 0);
}

#[test]
fn test_alignment_pads_offset_to_alignment() {
    let mut a = TlsfAllocator::new(8192, 0, 256, 32);
    let small = a.alloc(257, 1).unwrap();
    assert_eq!(small, 0);
    let aligned = a.alloc(256, 512).unwrap();
    assert_eq!(aligned % 512, 0, "alignment must be honored");
}

#[test]
fn test_oom_returns_error() {
    let mut a = TlsfAllocator::new(1024, 0, 256, 32);
    let _ok = a.alloc(1024, 1).unwrap();
    let err = a.alloc(256, 1).expect_err("expected OOM");
    matches!(err, TlsfError::OutOfMemory { .. });
}

#[test]
fn test_double_free_is_rejected() {
    let mut a = TlsfAllocator::new(1024, 0, 256, 32);
    let o = a.alloc(256, 1).unwrap();
    a.free(o).unwrap();
    let err = a.free(o).expect_err("double free must fail");
    matches!(err, TlsfError::DoubleFree { .. });
}

#[test]
fn test_unknown_offset_free_is_rejected() {
    let mut a = TlsfAllocator::new(1024, 0, 256, 32);
    let err = a.free(99999).expect_err("free of unknown offset must fail");
    matches!(err, TlsfError::UnknownOffset { .. });
}

#[test]
fn test_determinism_same_sequence_yields_same_offsets() {
    fn run() -> Vec<usize> {
        let mut a = TlsfAllocator::new(8192, 0, 256, 32);
        let mut offsets = Vec::new();
        for sz in [256usize, 512, 1024, 256, 768, 256] {
            offsets.push(a.alloc(sz, 1).unwrap());
        }
        // Free out-of-order, then re-alloc; the FIFO bucket ordering must
        // make this deterministic.
        a.free(offsets[2]).unwrap();
        a.free(offsets[0]).unwrap();
        offsets.push(a.alloc(256, 1).unwrap());
        offsets.push(a.alloc(512, 1).unwrap());
        offsets
    }

    let first = run();
    let second = run();
    assert_eq!(first, second, "same alloc/free sequence must produce same offsets");
}

#[test]
fn test_peak_used_tracks_high_water_mark() {
    let mut a = TlsfAllocator::new(8192, 0, 256, 32);
    let _o1 = a.alloc(256, 1).unwrap();
    let _o2 = a.alloc(512, 1).unwrap();
    assert_eq!(a.peak_used(), 768, "two consecutive allocs of 256 + 512 push peak to 768");

    let _o3 = a.alloc(256, 1).unwrap();
    assert_eq!(a.peak_used(), 1024);
}

#[test]
fn test_base_offset_applied_to_returned_addresses() {
    let mut a = TlsfAllocator::new(4096, 0x1000, 256, 32);
    let o = a.alloc(256, 1).unwrap();
    assert_eq!(o, 0x1000, "returned offset must be base-relative absolute");
    a.free(o).unwrap();
}

#[test]
fn test_random_alloc_free_no_overlap() {
    use std::collections::BTreeMap;
    let mut a = TlsfAllocator::new(0x10000, 0, 256, 32);
    let mut live: BTreeMap<usize, usize> = BTreeMap::new(); // offset -> size

    // Deterministic pseudo-random schedule.
    let mut state: u64 = 0xCAFEBABE;
    for _ in 0..200 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let action = (state >> 32) % 3;
        if action != 0 || live.is_empty() {
            // alloc
            let req = (((state >> 8) & 0x7) + 1) as usize * 256;
            if let Ok(off) = a.alloc(req, 1) {
                // verify no overlap with any live block
                for (&l_off, &l_sz) in &live {
                    let no_overlap = off + req <= l_off || l_off + l_sz <= off;
                    assert!(no_overlap, "alloc {off}+{req} overlaps live block {l_off}+{l_sz}");
                }
                live.insert(off, req);
            }
        } else {
            // free a random live block
            let idx = (state >> 16) as usize % live.len();
            let key = *live.keys().nth(idx).unwrap();
            let _sz = live.remove(&key).unwrap();
            a.free(key).unwrap();
        }
    }
}
