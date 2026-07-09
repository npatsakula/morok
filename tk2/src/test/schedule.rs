//! Host tests for the typed cluster pipeline (Increment 1, DESIGN.md §5c).
//!
//! `matmul_lds_kblock_mw_pipe2` authors its steady body through the typed `MemScope`/`ComputeScope`
//! cluster scopes ([`svod_tk2::schedule`]). The compute cluster is bracketed by `set_prio(1)`/
//! `set_prio(0)` — the first real HipKittens schedule steering — which **deliberately breaks
//! byte-identity** with the plain `matmul_lds_kblock_mw_pipe`. Since `set_prio` is an ordering-only
//! hint (it does not touch the MFMA data flow), the *math* is unchanged: the structural tests below
//! assert exactly that — the steering nodes were added, and nothing else about the computation moved.
//! Device allclose + the ISA `render_amd_ir` check (the perf payoff) are gated on the MI300X.

use crate::ir::{Node, TileId};
use crate::kernels::{Program, matmul_lds_kblock_mw_pipe, matmul_lds_kblock_mw_pipe2};

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

/// The steering was added: `pipe2`'s steady compute cluster brackets its MFMAs with exactly two
/// `set_prio` nodes (`level=1` before, `level=0` after), which the plain `pipe` has none of.
fn assert_steered(pipe: &Program, pipe2: &Program, label: &str) {
    assert_eq!(n_set_prio(pipe), 0, "{label}: plain pipe must have no set_prio");
    assert_eq!(
        n_set_prio(pipe2),
        2,
        "{label}: pipe2's compute cluster must bracket its MFMAs with set_prio(1)+set_prio(0)"
    );
}

/// The math is unchanged: `set_prio` is ordering-only, so `pipe2` has the SAME MFMA count and the
/// SAME barrier (cluster-seal) count as the plain `pipe` — only the schedule hints were added.
fn assert_math_preserved(pipe: &Program, pipe2: &Program, label: &str) {
    assert_eq!(n_mma(pipe2), n_mma(pipe), "{label}: MFMA count must match — steering can't change the math");
    assert_eq!(n_barrier(pipe2), n_barrier(pipe), "{label}: cluster-seal barrier count must match — no new barriers");
}

/// Production config: 4×4 warps, 256² tile, k_step=64 (grid_m % 4 == 0 → L2 swizzle).
#[test]
fn pipe2_is_steered_256() {
    let pipe = matmul_lds_kblock_mw_pipe(4096, 4096, 4096, 64, 64, 4, 4, 64);
    let pipe2 = matmul_lds_kblock_mw_pipe2(4096, 4096, 4096, 64, 64, 4, 4, 64);
    assert_steered(&pipe, &pipe2, "256²/4×4/k_step=64");
    assert_math_preserved(&pipe, &pipe2, "256²/4×4/k_step=64");
}

/// 2×2 warps, 128² tile — the multi-warp offset path with a different grid.
#[test]
fn pipe2_is_steered_128() {
    let pipe = matmul_lds_kblock_mw_pipe(2048, 2048, 2048, 64, 64, 2, 2, 64);
    let pipe2 = matmul_lds_kblock_mw_pipe2(2048, 2048, 2048, 64, 64, 2, 2, 64);
    assert_steered(&pipe, &pipe2, "128²/2×2/k_step=64");
    assert_math_preserved(&pipe, &pipe2, "128²/2×2/k_step=64");
}

/// Single-warp path (wm=wn=1, no runtime warp offset).
#[test]
fn pipe2_is_steered_single_warp() {
    let pipe = matmul_lds_kblock_mw_pipe(1024, 1024, 1024, 64, 64, 1, 1, 64);
    let pipe2 = matmul_lds_kblock_mw_pipe2(1024, 1024, 1024, 64, 64, 1, 1, 64);
    assert_steered(&pipe, &pipe2, "64²/1×1/k_step=64");
    assert_math_preserved(&pipe, &pipe2, "64²/1×1/k_step=64");
}

/// k_step=32 — a different ksteps count (2 MFMAs per accumulator vs 4).
#[test]
fn pipe2_is_steered_kstep32() {
    let pipe = matmul_lds_kblock_mw_pipe(2048, 2048, 2048, 64, 64, 2, 2, 32);
    let pipe2 = matmul_lds_kblock_mw_pipe2(2048, 2048, 2048, 64, 64, 2, 2, 32);
    assert_steered(&pipe, &pipe2, "128²/2×2/k_step=32");
    assert_math_preserved(&pipe, &pipe2, "128²/2×2/k_step=32");
}
