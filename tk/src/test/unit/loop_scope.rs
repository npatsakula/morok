//! Unit tests for [`crate::Loop`] — the loop-scope object that makes the
//! loop-carried re-init / close ordering edges declarative. These pin the
//! abstraction's contract: it must emit the *exact* nodes the hand-threaded form
//! does (`reinit` → `After([range])`, `close`/`close_carry` → the loop-closing
//! `END`), which is what lets the Gap-1 refactor stay graph-identical.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{Op, UOp};

use crate::tiles::{RT_16X16, TileLayout};
use crate::{ArchCaps, Kernel};
use svod_ir::ops;

fn deps_contain(deps: &[Arc<UOp>], needle: &Arc<UOp>) -> bool {
    deps.iter().any(|d| Arc::ptr_eq(d, needle))
}

/// `index()` is the loop RANGE; `reinit(t)` re-wraps the tile as `After([range])`
/// — the declarative form of the hand-threaded `t.after([loop_range])` re-init,
/// so a per-iteration re-init can never be silently dropped.
#[test]
fn reinit_wraps_tile_after_loop_range() {
    let ker = Kernel::new("lp", [1, 1, 1], 64, vec![], ArchCaps::GFX942);
    let lp = ker.loop_static(4);
    assert!(matches!(lp.index().op(), Op::Range(..)), "index() is the loop RANGE");

    let rt = ker.rt((16, 16), DType::Float32, TileLayout::Col, RT_16X16);
    let r = lp.reinit(rt);
    match r.uop().op() {
        Op::After(ops::After { deps, .. }) => assert!(deps_contain(deps, lp.index()), "reinit dep is the loop range"),
        other => panic!("reinit must wrap the tile in After([range]), got {other:?}"),
    }
}

/// `close()` ends the loop's terminal store, returning the loop-closing `END`
/// (whose `ranges` includes this loop's RANGE) — the multi-accumulator close used
/// by matmul (`ended` + per-accumulator `after([ended])`).
#[test]
fn close_returns_loop_closing_end() {
    let ker = Kernel::new("lp", [1, 1, 1], 64, vec![], ArchCaps::GFX942);
    let warp = ker.warp();
    let lp = ker.loop_static(4);
    let range = lp.index().clone();
    // A terminal store inside the loop (`zero` pushes an END(STORE) on the stack).
    let _acc = warp.zero(ker.rt((16, 16), DType::Float32, TileLayout::Col, RT_16X16));
    let end = lp.close();
    match end.op() {
        Op::End(ops::End { ranges, .. }) => assert!(deps_contain(ranges, &range), "close ends this loop's range"),
        other => panic!("close must return an END, got {other:?}"),
    }
}

/// `close_carry(t)` ends the loop and rebinds the carried tile to its post-loop
/// value: an `After` over the loop-closing `END` — the single-accumulator close
/// used by FA (`o_reg`).
#[test]
fn close_carry_rebinds_tile_after_end() {
    let ker = Kernel::new("lp", [1, 1, 1], 64, vec![], ArchCaps::GFX942);
    let warp = ker.warp();
    let lp = ker.loop_static(4);
    let acc = warp.zero(ker.rt((16, 16), DType::Float32, TileLayout::Col, RT_16X16));
    let acc = lp.close_carry(acc);
    match acc.uop().op() {
        Op::After(ops::After { deps, .. }) => {
            assert!(deps.iter().any(|d| matches!(d.op(), Op::End(..))), "close_carry deps on the loop END");
        }
        other => panic!("close_carry must rewrap the tile in After([END]), got {other:?}"),
    }
}
