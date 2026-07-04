//! The pass-runner scaffold (DESIGN.md §2.6): a strategy algebra with declared
//! contracts + phase bands + a fuel cap — NOT a raw `Vec<Pass>`. No real
//! optimization passes yet (those are Step 2); this proves the *shape* passes will
//! plug into:
//!
//! - a **nanopass identity-default folder** ([`Fold`]): the driver recurses
//!   identically and a pass overrides only the node arms it touches, so adding a
//!   [`Node`] variant does not break every pass;
//! - a [`Pass`] trait with `requires`/`ensures` predicate hooks and a phase [`Band`];
//! - **Elevate-style strategy combinators** ([`seq`], [`or_else`], [`try_`],
//!   [`repeat_fixpoint`], [`top_down`]) so a pass that does not apply degrades
//!   gracefully, with a **fuel cap** so a non-terminating rewrite is bounded;
//! - a banded [`Pipeline`] runner that checks band monotonicity + the per-pass
//!   contracts between steps.

use std::collections::HashMap;

use snafu::Snafu;

use crate::ir::{Node, TileId, TileIr};

// ============================================================================
// Nanopass identity-default folder
// ============================================================================

/// A structural rewrite over the tile-IR. The [`fold`] driver visits nodes in
/// dependency order and calls [`Fold::fold_node`] with a node whose children are
/// **already rewritten**; the default re-interns identically (the identity
/// folder), so a concrete pass overrides `fold_node`, matches only the arm(s) it
/// transforms, and delegates the rest to `ir.intern(node)`.
pub trait Fold {
    fn fold_node(&mut self, ir: &mut TileIr, node: Node) -> TileId {
        ir.intern(node)
    }
}

/// The identity folder — overrides nothing. `fold(IdentityFold, ir, root) == root`
/// because re-interning structurally-identical nodes returns their existing ids.
pub struct IdentityFold;
impl Fold for IdentityFold {}

/// Drive `f` over every node reachable in the arena (ascending id = dependency
/// order, since children are interned before parents), returning the rewritten
/// `root`. Children are remapped before their parent is folded.
pub fn fold(f: &mut impl Fold, ir: &mut TileIr, root: TileId) -> TileId {
    let n = ir.len();
    let mut remap: Vec<Option<TileId>> = vec![None; n];
    for i in 0..n {
        let node = ir.node(TileId(i as u32)).clone();
        let mapped = TileIr::map_children(&node, |c| remap[c.0 as usize].expect("child folded before parent"));
        remap[i] = Some(f.fold_node(ir, mapped));
    }
    remap[root.0 as usize].expect("root folded")
}

// ============================================================================
// Passes: phase bands + declared contracts
// ============================================================================

/// The ordered phase bands (DESIGN.md §2.6). Passes reorder freely *within* a
/// band; the runner verifies bands are non-decreasing *between* passes.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Band {
    Tiling,
    MemoryPlacement,
    Pipelining,
    RegAlloc,
}

/// A tile-IR → tile-IR pass with declared invariant contracts and a phase band.
pub trait Pass {
    fn name(&self) -> &str;
    fn band(&self) -> Band;
    /// Precondition over tile-IR invariants (checked by the runner before `apply`).
    fn requires(&self, _ir: &TileIr, _root: TileId) -> bool {
        true
    }
    /// Postcondition over tile-IR invariants (checked by the runner after `apply`).
    fn ensures(&self, _ir: &TileIr, _root: TileId) -> bool {
        true
    }
    /// Apply the pass, returning the new root.
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Result<TileId, PassError>;
}

/// The trivial pass — runs the identity folder. Proves the runner + contract
/// checks work end-to-end; real Tiling/MemoryPlacement/… passes land in Step 2.
pub struct IdentityPass;
impl Pass for IdentityPass {
    fn name(&self) -> &str {
        "identity"
    }
    fn band(&self) -> Band {
        Band::Tiling
    }
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Result<TileId, PassError> {
        Ok(fold(&mut IdentityFold, ir, root))
    }
}

/// Errors from running a pass pipeline.
#[derive(Debug, Snafu, PartialEq, Eq)]
#[snafu(visibility(pub))]
pub enum PassError {
    /// A pass's `requires` precondition did not hold.
    #[snafu(display("pass {pass:?}: `requires` contract not satisfied"))]
    Requires { pass: String },
    /// A pass's `ensures` postcondition did not hold after `apply`.
    #[snafu(display("pass {pass:?}: `ensures` contract violated"))]
    Ensures { pass: String },
    /// A pass ran out of its declared band order (bands must be non-decreasing).
    #[snafu(display("pass {pass:?} in band {band:?} runs after band {prev:?} (bands must not decrease)"))]
    BandOrder { pass: String, band: Band, prev: Band },
    /// A pass's own `apply` failed.
    #[snafu(display("pass {pass:?} apply failed: {reason}"))]
    Apply { pass: String, reason: String },
}

/// A banded, contract-checked pass pipeline.
#[derive(Default)]
pub struct Pipeline {
    passes: Vec<Box<dyn Pass>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a pass (builder style).
    pub fn then(mut self, pass: impl Pass + 'static) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Run every pass in order, verifying band monotonicity and each pass's
    /// `requires`/`ensures` contracts between steps.
    pub fn run(&self, ir: &mut TileIr, root: TileId) -> Result<TileId, PassError> {
        let mut cur = root;
        let mut prev_band: Option<Band> = None;
        for pass in &self.passes {
            if let Some(prev) = prev_band {
                snafu::ensure!(
                    pass.band() >= prev,
                    BandOrderSnafu { pass: pass.name().to_string(), band: pass.band(), prev }
                );
            }
            prev_band = Some(pass.band());
            snafu::ensure!(pass.requires(ir, cur), RequiresSnafu { pass: pass.name().to_string() });
            cur = pass.apply(ir, cur)?;
            snafu::ensure!(pass.ensures(ir, cur), EnsuresSnafu { pass: pass.name().to_string() });
        }
        Ok(cur)
    }
}

// ============================================================================
// Elevate-style strategy combinators
// ============================================================================

/// The Elevate outcome: `Ok(id)` = applied (the — possibly changed — result),
/// `Err(id)` = did not apply (the unchanged root; the tile-IR is persistent, so
/// "no change" is just returning the original handle — no rollback needed).
pub type Rewrite = Result<TileId, TileId>;

/// A composable rewrite strategy over the tile-IR.
pub trait Strategy {
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Rewrite;
}

impl Strategy for Box<dyn Strategy> {
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Rewrite {
        (**self).apply(ir, root)
    }
}

/// Always succeeds, unchanged (Elevate `id`).
pub struct Id;
impl Strategy for Id {
    fn apply(&self, _ir: &mut TileIr, root: TileId) -> Rewrite {
        Ok(root)
    }
}

/// Always fails, unchanged (Elevate `fail`).
pub struct Fail;
impl Strategy for Fail {
    fn apply(&self, _ir: &mut TileIr, root: TileId) -> Rewrite {
        Err(root)
    }
}

/// Adapt a [`Pass`] into a [`Strategy`]: succeeds iff `apply` changed the root.
pub struct AsStrategy<P>(pub P);
impl<P: Pass> Strategy for AsStrategy<P> {
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Rewrite {
        match self.0.apply(ir, root) {
            Ok(new) if new != root => Ok(new),
            _ => Err(root),
        }
    }
}

/// `a` then `b`; fails if either fails (Elevate `seq`).
pub struct Seq<A, B>(pub A, pub B);
impl<A: Strategy, B: Strategy> Strategy for Seq<A, B> {
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Rewrite {
        match self.0.apply(ir, root) {
            Ok(x) => self.1.apply(ir, x).map_err(|_| root),
            Err(_) => Err(root),
        }
    }
}

/// `a`, else `b` on the original root (Elevate `<+` / left-biased choice).
pub struct OrElse<A, B>(pub A, pub B);
impl<A: Strategy, B: Strategy> Strategy for OrElse<A, B> {
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Rewrite {
        match self.0.apply(ir, root) {
            Ok(x) => Ok(x),
            Err(_) => self.1.apply(ir, root),
        }
    }
}

/// `a`, but never fails (Elevate `try`): a failed `a` degrades to a no-op success.
pub struct Try<A>(pub A);
impl<A: Strategy> Strategy for Try<A> {
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Rewrite {
        Ok(match self.0.apply(ir, root) {
            Ok(x) | Err(x) => x,
        })
    }
}

/// Apply `a` to a fixpoint (until it stops making progress), capped at `fuel`
/// iterations so a non-terminating rewrite is bounded, not a hang (DESIGN.md §2.6,
/// the tinygrad derived-edge non-termination footgun). Always succeeds.
pub struct RepeatFixpoint<A> {
    pub strat: A,
    pub fuel: u32,
}
impl<A: Strategy> Strategy for RepeatFixpoint<A> {
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Rewrite {
        let mut cur = root;
        for _ in 0..self.fuel {
            match self.strat.apply(ir, cur) {
                Ok(next) if next != cur => cur = next, // progress: keep going
                _ => return Ok(cur),                   // fixpoint (or failure) reached
            }
        }
        Ok(cur) // fuel exhausted — bounded, not a hang
    }
}

/// Apply `a` top-down: at the root, then (memoized over the DAG) into each child,
/// reconstructing parents by re-interning with rewritten children. Succeeds iff
/// the root changed. Uses [`Try`] semantics at each node so a non-matching node is
/// skipped rather than aborting the traversal.
pub struct TopDown<A>(pub A);
impl<A: Strategy> Strategy for TopDown<A> {
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Rewrite {
        let mut memo: HashMap<TileId, TileId> = HashMap::new();
        let out = self.visit(ir, root, &mut memo);
        if out != root { Ok(out) } else { Err(root) }
    }
}
impl<A: Strategy> TopDown<A> {
    fn visit(&self, ir: &mut TileIr, id: TileId, memo: &mut HashMap<TileId, TileId>) -> TileId {
        if let Some(&m) = memo.get(&id) {
            return m;
        }
        // Apply at this node (Try semantics), then descend into the result's children.
        let here = match self.0.apply(ir, id) {
            Ok(x) | Err(x) => x,
        };
        let node = ir.node(here).clone();
        let mapped = TileIr::map_children(&node, |c| self.visit(ir, c, memo));
        let out = ir.intern(mapped);
        memo.insert(id, out);
        out
    }
}

// ── constructors ─────────────────────────────────────────────────────────────

pub fn seq<A: Strategy, B: Strategy>(a: A, b: B) -> Seq<A, B> {
    Seq(a, b)
}
pub fn or_else<A: Strategy, B: Strategy>(a: A, b: B) -> OrElse<A, B> {
    OrElse(a, b)
}
pub fn try_<A: Strategy>(a: A) -> Try<A> {
    Try(a)
}
pub fn repeat_fixpoint<A: Strategy>(strat: A, fuel: u32) -> RepeatFixpoint<A> {
    RepeatFixpoint { strat, fuel }
}
pub fn top_down<A: Strategy>(a: A) -> TopDown<A> {
    TopDown(a)
}
