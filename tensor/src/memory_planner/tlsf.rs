//! Two-Level Segregated Fit (TLSF) allocator for offline arena offset planning.
//!
//! This is a planning-time allocator: it assigns byte offsets within an arena
//! to logical buffers given a sequence of `alloc(size) -> offset` and
//! `free(offset)` events. It does not own real memory.
//!
//! Each free block is bucketed by `(lv1, lv2)` where `lv1` is the
//! position of the most-significant bit of its size and `lv2` is a further
//! subdivision of that power-of-two range. Allocation searches buckets from
//! the requested size upward; free re-inserts and coalesces with adjacent
//! free blocks.
//!
//! Determinism: bucket free-lists are `Vec`s with FIFO push-back / pop-front,
//! so two runs with the same alloc/free sequence produce the same offset
//! sequence — required for reproducible memory plans across schedule
//! regenerations.
//!
//! Constants: arena planning uses `block_size = 256`, `lv2_cnt = 32` to match
//! tinygrad's planner.

use std::collections::HashMap;

use snafu::Snafu;

#[derive(Debug, Snafu)]
pub enum TlsfError {
    #[snafu(display("TLSF out of memory: requested {req} bytes"))]
    OutOfMemory { req: usize },
    #[snafu(display("TLSF: free of unknown offset {offset}"))]
    UnknownOffset { offset: usize },
    #[snafu(display("TLSF: double free at offset {offset}"))]
    DoubleFree { offset: usize },
}

#[derive(Debug, Clone, Copy)]
struct BlockMeta {
    size: usize,
    next: Option<usize>,
    prev: Option<usize>,
    is_free: bool,
}

pub struct TlsfAllocator {
    base: usize,
    block_size: usize,
    /// `lv2_cnt.bit_length()` (e.g. `lv2_cnt = 32` → `l2_bits = 6`).
    l2_bits: usize,
    /// Free-list buckets indexed by `[lv1][lv2]`, each bucket is a FIFO of
    /// block start offsets. `Vec`-of-`Vec` keeps insertion order
    /// deterministic across runs.
    storage: Vec<Vec<Vec<usize>>>,
    /// Per-`lv1` count of free blocks across all `lv2` slots. Skipped when
    /// zero during search to avoid scanning empty rows.
    lv1_entries: Vec<usize>,
    /// Block metadata keyed by start offset. HashMap is fine — used only for
    /// direct lookups, never iterated.
    blocks: HashMap<usize, BlockMeta>,
    /// Running maximum of `start + req_size` ever returned by `alloc`. The
    /// arena buffer can be sized to this value (rounded up to `block_size`).
    peak_used: usize,
}

impl TlsfAllocator {
    /// Create a new allocator for `size` bytes starting at `base`.
    ///
    /// `block_size` is the minimum allocation granule; `lv2_cnt` is the
    /// number of second-level buckets per power-of-two range. Arena
    /// planning uses `block_size = 256`, `lv2_cnt = 32` (tinygrad parity).
    pub fn new(size: usize, base: usize, block_size: usize, lv2_cnt: usize) -> Self {
        let lv2_cnt = lv2_cnt.max(1);
        let l2_bits = bit_length(lv2_cnt);
        let lv1_levels = bit_length(size) + 1;

        let mut s = Self {
            base,
            block_size,
            l2_bits,
            storage: (0..lv1_levels).map(|_| (0..(1 << l2_bits)).map(|_| Vec::new()).collect()).collect(),
            lv1_entries: vec![0; lv1_levels],
            blocks: HashMap::new(),
            peak_used: 0,
        };

        // Seed with one free block covering the whole arena.
        if size > 0 {
            s.blocks.insert(0, BlockMeta { size, next: None, prev: None, is_free: true });
            s.insert_block_index(0, size);
        }
        s
    }

    /// Returns the position of the most-significant bit of `size` (1-based).
    /// Mirrors Python's `int.bit_length()`.
    fn lv1(&self, size: usize) -> usize {
        bit_length(size)
    }

    /// Second-level bucket within `lv1(size)`'s power-of-two range.
    fn lv2(&self, size: usize) -> usize {
        let msb = bit_length(size);
        if msb == 0 {
            return 0;
        }
        let shift = msb.saturating_sub(self.l2_bits);
        (size - (1 << (msb - 1))) >> shift
    }

    fn insert_block_index(&mut self, start: usize, size: usize) {
        let l1 = self.lv1(size);
        let l2 = self.lv2(size);
        self.storage[l1][l2].push(start);
        self.lv1_entries[l1] += 1;
    }

    fn remove_block_index(&mut self, start: usize, size: usize) {
        let l1 = self.lv1(size);
        let l2 = self.lv2(size);
        let bucket = &mut self.storage[l1][l2];
        if let Some(pos) = bucket.iter().position(|&s| s == start) {
            bucket.remove(pos);
            self.lv1_entries[l1] -= 1;
        }
    }

    fn split_block(&mut self, start: usize, size: usize, new_size: usize) {
        let next = self.blocks[&start].next;
        debug_assert!(self.blocks[&start].is_free, "split target must be free");

        self.remove_block_index(start, size);
        self.blocks.insert(
            start,
            BlockMeta { size: new_size, next: Some(start + new_size), prev: self.blocks[&start].prev, is_free: true },
        );
        self.insert_block_index(start, new_size);

        let remainder_start = start + new_size;
        let remainder_size = size - new_size;
        self.blocks.insert(remainder_start, BlockMeta { size: remainder_size, next, prev: Some(start), is_free: true });
        self.insert_block_index(remainder_start, remainder_size);

        if let Some(nxt) = next
            && let Some(meta) = self.blocks.get_mut(&nxt)
        {
            meta.prev = Some(remainder_start);
        }
    }

    fn merge_right(&mut self, start: usize) {
        let mut size = self.blocks[&start].size;
        let mut nxt = self.blocks[&start].next;
        debug_assert!(self.blocks[&start].is_free, "merge_right target must be free");

        while let Some(n) = nxt {
            let Some(blk) = self.blocks.get(&n).copied() else { break };
            if !blk.is_free {
                break;
            }
            self.remove_block_index(start, size);
            self.remove_block_index(n, blk.size);
            size += blk.size;
            self.blocks
                .insert(start, BlockMeta { size, next: blk.next, prev: self.blocks[&start].prev, is_free: true });
            self.insert_block_index(start, size);
            nxt = blk.next;
            self.blocks.remove(&n);
        }

        if let Some(n) = nxt
            && let Some(meta) = self.blocks.get_mut(&n)
        {
            meta.prev = Some(start);
        }
    }

    fn merge_block(&mut self, start: usize) {
        // Walk left while neighbours are free, then merge right from there.
        let mut s = start;
        while let Some(prev) = self.blocks[&s].prev
            && self.blocks.get(&prev).is_some_and(|b| b.is_free)
        {
            s = prev;
        }
        self.merge_right(s);
    }

    /// Allocate `req_size` bytes with `align` alignment. Returns the offset
    /// (`base`-relative) or `OutOfMemory` if no fitting block exists.
    pub fn alloc(&mut self, req_size: usize, align: usize) -> Result<usize, TlsfError> {
        let req_size = req_size.max(self.block_size);
        let align = align.max(1);
        let mut size = req_size + align - 1;
        size = size.max(self.block_size);

        // Round size up to next bucket granularity so any block in that
        // bucket fits the request.
        let msb = bit_length(size);
        if msb > self.l2_bits {
            let granule_bits = msb - self.l2_bits;
            let granule = 1usize << granule_bits;
            size = (size + granule - 1) & !(granule - 1);
        }

        let target_l1 = self.lv1(size);
        let target_l2 = self.lv2(size);

        for l1 in target_l1..self.storage.len() {
            if self.lv1_entries[l1] == 0 {
                continue;
            }
            let l2_start = if l1 == target_l1 { target_l2 } else { 0 };
            for l2 in l2_start..self.storage[l1].len() {
                if self.storage[l1][l2].is_empty() {
                    continue;
                }
                let start = self.storage[l1][l2][0];
                let nsize = self.blocks[&start].size;
                debug_assert!(nsize >= size, "bucketed block must be ≥ requested size");

                // Handle alignment: if the natural start is misaligned,
                // split off the head so the next block starts aligned.
                let aligned_start = round_up(start, align);
                let (mut start, mut nsize) = (start, nsize);
                if aligned_start != start {
                    let head = aligned_start - start;
                    self.split_block(start, nsize, head);
                    start = aligned_start;
                    nsize = self.blocks[&start].size;
                }

                if nsize > req_size {
                    self.split_block(start, nsize, req_size);
                }
                self.remove_block_index(start, req_size);
                if let Some(meta) = self.blocks.get_mut(&start) {
                    meta.is_free = false;
                }
                let abs_offset = start + self.base;
                let used_end = start + req_size;
                if used_end > self.peak_used {
                    self.peak_used = used_end;
                }
                return Ok(abs_offset);
            }
        }

        OutOfMemorySnafu { req: req_size }.fail()
    }

    /// Free a previously-allocated offset. Coalesces with adjacent free
    /// blocks.
    pub fn free(&mut self, offset: usize) -> Result<(), TlsfError> {
        let start = offset.checked_sub(self.base).ok_or(TlsfError::UnknownOffset { offset })?;
        let blk = self.blocks.get(&start).copied().ok_or(TlsfError::UnknownOffset { offset })?;
        if blk.is_free {
            return DoubleFreeSnafu { offset }.fail();
        }
        self.blocks.insert(start, BlockMeta { is_free: true, ..blk });
        self.insert_block_index(start, blk.size);
        self.merge_block(start);
        Ok(())
    }

    /// Maximum end-offset ever returned by `alloc`. The arena needs at least
    /// this many bytes (rounded up to `block_size`). Used by tests; the
    /// production planner derives peak from `events` directly.
    #[allow(dead_code)]
    pub fn peak_used(&self) -> usize {
        self.peak_used
    }
}

/// Position of the most-significant bit of `n` (1-based). Returns 0 for `n == 0`.
/// Mirrors Python's `int.bit_length()`.
fn bit_length(n: usize) -> usize {
    if n == 0 { 0 } else { (usize::BITS - n.leading_zeros()) as usize }
}

fn round_up(n: usize, granule: usize) -> usize {
    if granule <= 1 { n } else { n.div_ceil(granule) * granule }
}

#[cfg(test)]
#[path = "../test/unit/tlsf.rs"]
mod tests;
