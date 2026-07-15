//! Host tests: the pass-runner scaffold — the nanopass identity folder, the
//! banded/contract-checked pipeline, and the Elevate strategy combinators.

use crate::ir::{TileId, TileIr};
use crate::kernels::Program;
use crate::kernels::matmul::{Tiling, matmul_lds_kblock_mw_clustered};
use crate::pass::{
    AsStrategy, Band, Fail, Id, IdentityFold, IdentityPass, Pass, PassError, Pipeline, Strategy, fold, or_else,
    repeat_fixpoint, seq, top_down, try_,
};

/// A small kept-kernel program — a generic `Program` fixture for the pass-runner scaffold tests
/// (which assert only pass-runner behaviour, not kernel semantics).
fn fixture() -> Program {
    matmul_lds_kblock_mw_clustered(128, 128, 256, Tiling { bm: 64, bn: 64, wm: 2, wn: 2, k_step: 64 })
}

/// A no-op pass in a chosen band (for band-ordering tests).
struct BandPass(Band);
impl Pass for BandPass {
    fn name(&self) -> &str {
        "band"
    }
    fn band(&self) -> Band {
        self.0
    }
    fn apply(&self, _ir: &mut TileIr, root: TileId) -> Result<TileId, PassError> {
        Ok(root)
    }
}

/// A pass whose `ensures` contract always fails (for contract-check tests).
struct BadEnsures;
impl Pass for BadEnsures {
    fn name(&self) -> &str {
        "bad_ensures"
    }
    fn band(&self) -> Band {
        Band::Tiling
    }
    fn ensures(&self, _ir: &TileIr, _root: TileId) -> bool {
        false
    }
    fn apply(&self, _ir: &mut TileIr, root: TileId) -> Result<TileId, PassError> {
        Ok(root)
    }
}

// ── the nanopass identity folder ─────────────────────────────────────────────

#[test]
fn identity_fold_is_a_no_op() {
    let p = fixture();
    let mut ir = p.ir;
    let out = fold(&mut IdentityFold, &mut ir, p.sink);
    assert_eq!(out, p.sink, "re-interning identical nodes must return the same root");
}

// ── the banded, contract-checked pipeline ────────────────────────────────────

#[test]
fn identity_pipeline_runs_and_preserves_the_root() {
    let p = fixture();
    let mut ir = p.ir;
    let out = Pipeline::new().then(IdentityPass).run(&mut ir, p.sink).expect("identity pipeline runs");
    assert_eq!(out, p.sink);
}

#[test]
fn band_must_not_decrease() {
    let p = fixture();
    let mut ir = p.ir;
    // RegAlloc (late) then Tiling (early) — an illegal decrease.
    let err = Pipeline::new()
        .then(BandPass(Band::RegAlloc))
        .then(BandPass(Band::Tiling))
        .run(&mut ir, p.sink)
        .expect_err("a band decrease must be rejected");
    assert!(matches!(err, PassError::BandOrder { .. }), "got {err:?}");
}

#[test]
fn ensures_contract_is_enforced() {
    let p = fixture();
    let mut ir = p.ir;
    let err = Pipeline::new().then(BadEnsures).run(&mut ir, p.sink).expect_err("a violated ensures must fail the run");
    assert!(matches!(err, PassError::Ensures { .. }), "got {err:?}");
}

// ── the Elevate strategy combinators ─────────────────────────────────────────

fn fresh() -> (TileIr, TileId) {
    let p = fixture();
    (p.ir, p.sink)
}

#[test]
fn seq_succeeds_when_both_succeed() {
    let (mut ir, root) = fresh();
    assert_eq!(seq(Id, Id).apply(&mut ir, root), Ok(root));
}

#[test]
fn seq_fails_when_first_fails() {
    let (mut ir, root) = fresh();
    assert_eq!(seq(Fail, Id).apply(&mut ir, root), Err(root));
}

#[test]
fn or_else_falls_through_to_the_alternative() {
    let (mut ir, root) = fresh();
    assert_eq!(or_else(Fail, Id).apply(&mut ir, root), Ok(root));
}

#[test]
fn try_never_fails() {
    let (mut ir, root) = fresh();
    assert_eq!(try_(Fail).apply(&mut ir, root), Ok(root));
}

#[test]
fn repeat_fixpoint_terminates_at_a_fixpoint() {
    let (mut ir, root) = fresh();
    // Id makes no progress → the fixpoint is reached immediately, well under fuel.
    assert_eq!(repeat_fixpoint(Id, 1000).apply(&mut ir, root), Ok(root));
    // A non-applying pass-as-strategy likewise degrades gracefully.
    assert_eq!(repeat_fixpoint(AsStrategy(IdentityPass), 8).apply(&mut ir, root), Ok(root));
}

#[test]
fn top_down_over_identity_reconstructs_the_same_dag() {
    let (mut ir, root) = fresh();
    // Applying Id at every node re-interns identically → no structural change.
    assert_eq!(top_down(Id).apply(&mut ir, root), Err(root));
}

#[test]
fn pass_as_strategy_reports_no_change() {
    let (mut ir, root) = fresh();
    // IdentityPass does not change the root, so its strategy adapter "fails" (no
    // progress) — exactly what `or_else`/`repeat_fixpoint` key on.
    assert_eq!(AsStrategy(IdentityPass).apply(&mut ir, root), Err(root));
}
