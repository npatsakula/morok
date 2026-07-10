//! HipKittens synchronization / scheduling leaves (`micro_tk` §E) as thin tk2 wrappers. All are
//! compiler builtins / inline asm in HK; each maps to an existing tk2 builder that renders the same
//! intrinsic / asm mnemonic (verified by test T6).

#![allow(non_snake_case)]

use crate::build::{Builder, Effect, Idx};
use crate::ir::TileId;

/// HK's `__builtin_amdgcn_s_barrier()` cluster seal — a bare `s.barrier()` + the baked positional
/// `sched.barrier(0)` wall ([`Builder::bare_barrier`]). HK's cluster tail is exactly
/// `s_barrier(); sched_barrier(0)`. `body` passes through; `deps` are happens-after anchors.
pub fn s_barrier(b: &mut Builder, body: Effect, deps: &[TileId]) -> Effect {
    b.bare_barrier(body, deps)
}

/// HK's `__builtin_amdgcn_s_setprio(level)` — raise/lower wave issue priority around an MFMA burst
/// (`@llvm.amdgcn.s.setprio(i16 level)`), positioned after `after`.
pub fn s_setprio(b: &mut Builder, level: i64, after: &[TileId]) -> Effect {
    b.set_prio(level, after)
}

/// HK's `__builtin_amdgcn_sched_barrier(0)` — the machine-scheduler fence (`sched.barrier(i32 0)`),
/// positioned after `anchors` (a total reorder fence, mask 0).
pub fn sched_barrier(b: &mut Builder, anchors: &[TileId]) -> Effect {
    b.sched_fence(0, anchors)
}

/// HK's `asm("s_waitcnt lgkmcnt(0)")` — the manual LDS drain, ordered after the last LDS op `prev`.
pub fn s_waitcnt_lgkmcnt(b: &mut Builder, prev: TileId) -> Effect {
    b.swait_lgkmcnt(prev)
}

/// HK's `asm("s_waitcnt vmcnt(0)")` — the VMEM drain (cooperative `G::load`), ordered after `prev`.
pub fn s_waitcnt_vmcnt(b: &mut Builder, prev: TileId) -> Effect {
    b.swait_vmcnt(prev)
}

/// HK's ping-pong phase barrier — `micro_tk`'s `if (warp_row == eq) { __builtin_amdgcn_s_barrier(); }`
/// (prologue `eq == 1` at `256_256_64_16.cpp:77`, epilogue `eq == 0` at `:221`). The ONE predicated
/// barrier that phase-offsets the two warp-rows by a cluster (so one row's MFMA clusters overlap the
/// other's memory clusters). tk2 forbids authoring `If`/`EndIf`, so the `warp_row == eq` predicate rides
/// INSIDE the [`Builder::wave_barrier`] asm block (`readfirstlane`+`s_cmp`+`s_cbranch`+`s_barrier`) —
/// the exact `warpid()`-uniform conditional HK compiles to. `after` are happens-after anchors; route the
/// returned [`Effect`] onward so it stays live + placed (a DCE'd barrier would unbalance the pair →
/// deadlock). The `eq == 1`/`eq == 0` pair MUST be balanced (one of each reachable) or the workgroup hangs.
pub fn wave_phase_barrier(b: &mut Builder, warp_row: Idx, eq: i64, after: &[TileId]) -> Effect {
    b.wave_barrier(warp_row, eq, after)
}

/// HK's `warpid()` (`common/util.cuh:69`) — `threadIdx.x >> 6`.
pub fn warpid(b: &mut Builder, tid: Idx) -> Idx {
    let six = b.idx_const(6);
    b.idx_shr(tid, six)
}

/// `warp_row = warpid() / 4` (∈ {0, 1} for `NUM_WARPS = 8`).
pub fn warp_row(b: &mut Builder, tid: Idx) -> Idx {
    let w = warpid(b, tid);
    let four = b.idx_const(4);
    b.idx_div(w, four)
}

/// `warp_col = warpid() % 4` (∈ {0, 1, 2, 3}).
pub fn warp_col(b: &mut Builder, tid: Idx) -> Idx {
    let w = warpid(b, tid);
    let four = b.idx_const(4);
    b.idx_mod(w, four)
}
