//! Host tests for the typed cluster pipeline (DESIGN.md §5c).
//!
//! `matmul_lds_kblock_mw_pipe2` authors its steady body through the typed `MemScope`/`ComputeScope`
//! cluster scopes ([`svod_tk2::schedule`]) as HipKittens' **8-cluster dot-slice pipeline + 8-wave
//! ping-pong**: gather slice s (mem cluster) → MFMA slice s (compute cluster), interleaved, the
//! accumulators chaining across the compute clusters, a register-staged prefetch split, a deferred
//! commit, and each compute cluster bracketed by `set_prio(1)`/`set_prio(0)`. The MFMAs are the
//! intrinsic `b.mma` (compiler-visible). The structural tests below assert the schedule's closed-form
//! shape: `2·ksteps` steering nodes (a bracket per dot-slice compute cluster) and the invariant MFMA
//! count `2·ri·cj·ksteps` (steady body once + the epilogue's last block). Device allclose + the perf
//! payoff are gated on the MI300X.

use crate::ir::{Node, TileId};
use crate::kernels::{Program, matmul_lds_kblock_mw_pipe2};

/// Count arena nodes matching `pred`.
fn count(p: &Program, pred: impl Fn(&Node) -> bool) -> usize {
    (0..p.ir.len()).filter(|&id| pred(p.ir.node(TileId(id.try_into().unwrap())))).count()
}

fn n_set_prio(p: &Program) -> usize {
    count(p, |n| matches!(n, Node::SetPrio { .. }))
}
fn n_mma(p: &Program) -> usize {
    count(p, |n| matches!(n, Node::Mma { .. }))
}
fn n_barrier(p: &Program) -> usize {
    count(p, |n| matches!(n, Node::Barrier { .. } | Node::BareBarrier { .. }))
}

/// `pipe2` brackets EACH of its `ksteps` dot-slice compute clusters with a `set_prio(1)`/`set_prio(0)`
/// pair → exactly `2·ksteps` set_prio nodes.
fn assert_steered(p: &Program, ksteps: usize, label: &str) {
    assert_eq!(
        n_set_prio(p),
        2 * ksteps,
        "{label}: must bracket each of its {ksteps} dot-slice compute clusters with set_prio(1)+set_prio(0)"
    );
}

/// The MFMA count is the schedule-invariant closed form: the steady body is emitted ONCE
/// (`ksteps·ri·cj` MMAs) and the epilogue MFMAs the last block (`ri·cj·ksteps`) → `2·ri·cj·ksteps`.
/// Barriers: each dot-slice seals a mem-cluster gather AND a compute cluster in the steady body, so at
/// least `2·ksteps` (plus prologue/commit/epilogue/wave barriers) — a robust lower bound.
fn assert_math(p: &Program, ri: usize, cj: usize, ksteps: usize, label: &str) {
    assert_eq!(n_mma(p), 2 * ri * cj * ksteps, "{label}: MFMA count = 2·ri·cj·ksteps (steady once + epilogue)");
    assert!(n_barrier(p) >= 2 * ksteps, "{label}: ≥1 seal per mem/compute cluster ({} < {})", n_barrier(p), 2 * ksteps);
}

/// 4×4 warps, 256² tile, k_step=64 (grid_m % 4 == 0 → L2 swizzle). ri=cj=4, ksteps=4.
#[test]
fn pipe2_is_steered_256() {
    let p = matmul_lds_kblock_mw_pipe2(4096, 4096, 4096, 64, 64, 4, 4, 64);
    assert_steered(&p, 4, "256²/4×4/k_step=64");
    assert_math(&p, 4, 4, 4, "256²/4×4/k_step=64");
}

/// HK's own tiling: bm=128,bn=64, 2×4 warps → 256² tile, k_step=64 (ping-pong valid). ri=8, cj=4, ksteps=4.
#[test]
fn pipe2_is_steered_hk_tiling() {
    let p = matmul_lds_kblock_mw_pipe2(4096, 4096, 4096, 128, 64, 2, 4, 64);
    assert_steered(&p, 4, "HK 128×64/2×4/k_step=64");
    assert_math(&p, 8, 4, 4, "HK 128×64/2×4/k_step=64");
}

/// Single-warp path (wm=wn=1, no runtime warp offset). ri=cj=4, ksteps=4.
#[test]
fn pipe2_is_steered_single_warp() {
    let p = matmul_lds_kblock_mw_pipe2(1024, 1024, 1024, 64, 64, 1, 1, 64);
    assert_steered(&p, 4, "64²/1×1/k_step=64");
    assert_math(&p, 4, 4, 4, "64²/1×1/k_step=64");
}

/// k_step=32 — a different ksteps count (2 dot-slice compute clusters vs 4). ri=cj=4, ksteps=2.
#[test]
fn pipe2_is_steered_kstep32() {
    let p = matmul_lds_kblock_mw_pipe2(2048, 2048, 2048, 64, 64, 2, 2, 32);
    assert_steered(&p, 2, "128²/2×2/k_step=32");
    assert_math(&p, 4, 4, 2, "128²/2×2/k_step=32");
}
