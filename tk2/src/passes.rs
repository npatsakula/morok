//! The first real optimization passes (DESIGN.md §4, §2.4) — the foundational
//! addressing optimization, as two contract-checked [`Pass`](crate::pass::Pass)es
//! through the runner:
//!
//! 1. [`UnrollPass`] (Band::Tiling) — fully unroll the rolled loops (the init range
//!    and the K-fragment reduction) into a **flat** body, substituting each range
//!    counter with its per-iteration constant. This is the enabling transform: it
//!    turns the loop-carried address arithmetic (`tk·16`, the init store index) into
//!    compile-time constants. The loop-carried accumulator is preserved by chaining
//!    each copy's read after the previous copy's store (`After` edges + register
//!    persistence — the same carry the rolled loop had, made explicit); the shared
//!    A/B operand scratch fragments get **fresh per-copy registers** so the flattened
//!    body has no write-after-read hazard on them (the linearizer orders only by
//!    edges — a shared scratch reg with no WAR edge would miscompile).
//!
//! 2. [`ConstFoldPass`] (Band::MemoryPlacement) — const-fold the now-compile-time
//!    address arithmetic to immediates (DESIGN.md §2.4: "the fold only fires when the
//!    fragment step is a COMPILE-TIME CONSTANT → requires UNROLL", encoded as this
//!    pass's `requires` contract). The lane-dependent part (`lane % 16`, `lane / 16`)
//!    stays runtime — only all-constant `IndexAlu` subtrees collapse.
//!
//! Both are **semantics-preserving by construction** (§2.1/§2.6): unroll replicates
//! the exact body with a value-preserving counter→constant substitution and an
//! explicit carry chain; const-fold replaces integer arithmetic with its value.

use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;
use svod_dtype::DType;

use crate::ir::{Edges, IndexOp, Node, Scalar, TileId, TileIr};
use crate::pass::{Band, Fold, Pass, PassError, fold};

// ── shared analysis helpers ──────────────────────────────────────────────────

/// Every node id reachable from `root` (its dependency cone).
pub(crate) fn reachable(ir: &TileIr, root: TileId) -> Vec<TileId> {
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        order.push(id);
        for c in TileIr::children(ir.node(id)) {
            stack.push(c);
        }
    }
    order
}

/// Count reachable nodes matching `pred` (the structural before/after proof metric).
pub(crate) fn count_reachable(ir: &TileIr, root: TileId, pred: impl Fn(&Node) -> bool) -> usize {
    reachable(ir, root).into_iter().filter(|&id| pred(ir.node(id))).count()
}

/// True if `id` is an integer `Const`.
fn int_const(ir: &TileIr, id: TileId) -> Option<i64> {
    match ir.node(id) {
        Node::Const { scalar: Scalar::Int(v), .. } => Some(*v),
        _ => None,
    }
}

/// True once no rolled loops remain (the flat-body / "unroll ran" invariant).
fn is_flat(ir: &TileIr, root: TileId) -> bool {
    reachable(ir, root).into_iter().all(|id| !matches!(ir.node(id), Node::Range { .. } | Node::End { .. }))
}

// ============================================================================
// Pass 1 — Unroll
// ============================================================================

/// Fully unroll every rolled loop into a flat body (DESIGN.md §4).
pub struct UnrollPass;

impl Pass for UnrollPass {
    fn name(&self) -> &str {
        "unroll"
    }
    fn band(&self) -> Band {
        Band::Tiling
    }
    /// Postcondition: the result is flat — no `Range`/`End` reachable (the whole
    /// point of the pass, and the structural proof the fold's precondition now holds).
    fn ensures(&self, ir: &TileIr, root: TileId) -> bool {
        is_flat(ir, root)
    }
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Result<TileId, PassError> {
        Ok(Unroller::default().rebuild(ir, root))
    }
}

/// The unroll driver. `rebuilt` memoizes the loop-invariant / post-loop rebuild;
/// `end_carry` memoizes each unrolled loop's carry-out effect(s).
#[derive(Default)]
struct Unroller {
    rebuilt: HashMap<TileId, TileId>,
    end_carry: HashMap<TileId, Vec<TileId>>,
}

/// Per-copy substitution state for one unrolled iteration.
struct CloneCtx<'a> {
    /// This copy's iteration index (the counter's constant value).
    j: i64,
    /// The loop counter node being substituted.
    counter: TileId,
    /// The accumulator register (the terminal store's target) — never freshened.
    acc_reg: TileId,
    /// The loop-carried read (`After` with the counter in its deps), if any.
    carry: Option<TileId>,
    /// The previous copy's terminal store (the carry-in for copies `j > 0`).
    prev: Option<TileId>,
    /// The carry-in seed for copy 0 (the loop-invariant read of the accumulator).
    seed: &'a [TileId],
    /// Fresh per-copy register remap for the (non-accumulator) scratch fragments.
    fresh: HashMap<TileId, TileId>,
    /// Per-copy clone memo.
    memo: HashMap<TileId, TileId>,
}

impl Unroller {
    /// Rebuild the invariant / post-loop graph, expanding each `End` (reached only
    /// as an `After`-dep or `Sink`-root effect) into its unrolled carry-out(s).
    fn rebuild(&mut self, ir: &mut TileIr, id: TileId) -> TileId {
        if let Some(&m) = self.rebuilt.get(&id) {
            return m;
        }
        let node = ir.node(id).clone();
        let out = match node {
            Node::After { val, deps } => {
                let val = self.rebuild(ir, val);
                let deps = self.expand_effects(ir, &deps);
                ir.intern(Node::After { val, deps })
            }
            Node::Sink { roots } => {
                let roots = self.expand_effects(ir, &roots);
                ir.intern(Node::Sink { roots })
            }
            // An `End` reached directly (not via an effect list) is not a shape tk2
            // produces; expand defensively and take the final carry-out.
            Node::End { .. } => *self.unroll_end(ir, id).last().expect("End unrolls to ≥1 carry-out"),
            other => {
                let mapped = TileIr::map_children(&other, |c| self.rebuild(ir, c));
                ir.intern(mapped)
            }
        };
        self.rebuilt.insert(id, out);
        out
    }

    /// Rebuild an effect list, splicing each `End` into its unrolled carry-out(s).
    fn expand_effects(&mut self, ir: &mut TileIr, effects: &[TileId]) -> Edges {
        let mut out = Edges::new();
        for &e in effects {
            if matches!(ir.node(e), Node::End { .. }) {
                out.extend(self.unroll_end(ir, e));
            } else {
                out.push(self.rebuild(ir, e));
            }
        }
        out
    }

    /// Unroll `End { body, ranges: [r] }` over its range `r` (static `trips`),
    /// returning the effect(s) that replace it downstream: the single final store
    /// for a loop-carried loop, or every copy's store for a carry-free one.
    fn unroll_end(&mut self, ir: &mut TileIr, end_id: TileId) -> Vec<TileId> {
        if let Some(v) = self.end_carry.get(&end_id) {
            return v.clone();
        }
        let Node::End { body, ranges } = ir.node(end_id).clone() else { panic!("unroll_end on non-End") };
        assert_eq!(ranges.len(), 1, "tk2 unroll: exactly one range per END");
        let counter = ranges[0];
        let Node::Range { trips, .. } = *ir.node(counter) else { panic!("END range operand is not a RANGE") };

        // The accumulator register the terminal store writes (shared, never freshened).
        let acc_reg = self.store_target_reg(ir, body);
        // The loop-carried read: the `After` in the body cone with the counter as a dep.
        let carry = self.find_carry_read(ir, body, counter);
        // Copy 0's carry-in seed: the carry read's deps minus the counter, rebuilt in
        // the post-loop context (nested Ends among them are themselves unrolled).
        let seed: Vec<TileId> = match carry {
            Some(cr) => {
                let Node::After { deps, .. } = ir.node(cr).clone() else { unreachable!() };
                let outer: Vec<TileId> = deps.into_iter().filter(|&d| d != counter).collect();
                self.expand_effects(ir, &outer).into_iter().collect()
            }
            None => Vec::new(),
        };

        let mut stores = Vec::with_capacity(trips.max(0) as usize);
        let mut prev = None;
        for j in 0..trips {
            let mut ctx =
                CloneCtx { j, counter, acc_reg, carry, prev, seed: &seed, fresh: HashMap::new(), memo: HashMap::new() };
            let s_j = self.clone_subst(ir, body, &mut ctx);
            prev = Some(s_j);
            stores.push(s_j);
        }

        // Carried loop: downstream reads the FINAL accumulated store (the earlier
        // copies are reachable through the carry chain). Carry-free loop (init): all
        // copies must be ordered before the downstream read.
        let carry_outs = if carry.is_some() { vec![*stores.last().expect("a loop runs ≥1 trip")] } else { stores };
        self.end_carry.insert(end_id, carry_outs.clone());
        carry_outs
    }

    /// Deep-clone the body cone for one copy: substitute the counter with `Const(j)`,
    /// rewrite the carry read's carry-in, and give every non-accumulator scratch
    /// register a fresh per-copy id (WAR-safe once flattened).
    fn clone_subst(&mut self, ir: &mut TileIr, id: TileId, ctx: &mut CloneCtx) -> TileId {
        if let Some(&m) = ctx.memo.get(&id) {
            return m;
        }
        let out = if id == ctx.counter {
            // the loop counter → this copy's constant
            ir.intern(Node::Const { scalar: Scalar::Int(ctx.j), dtype: DType::Index })
        } else if Some(id) == ctx.carry {
            // the loop-carried read → carry-in from the seed (copy 0) or the previous
            // copy's terminal store (copy j > 0).
            let Node::After { val, .. } = ir.node(id).clone() else { unreachable!() };
            let val = self.clone_subst(ir, val, ctx);
            let deps: Edges = if ctx.j == 0 {
                ctx.seed.iter().copied().collect()
            } else {
                SmallVec::from_slice(&[ctx.prev.expect("carry copy j>0 has a previous store")])
            };
            ir.intern(Node::After { val, deps })
        } else {
            let node = ir.node(id).clone();
            match node {
                // A non-accumulator scratch register → a fresh per-copy allocation.
                Node::DefineFrag { .. } | Node::DefineReg { .. } if id != ctx.acc_reg => {
                    if let Some(&f) = ctx.fresh.get(&id) {
                        f
                    } else {
                        let f = self.fresh_reg(ir, &node);
                        ctx.fresh.insert(id, f);
                        f
                    }
                }
                other => {
                    let mapped = TileIr::map_children(&other, |c| self.clone_subst(ir, c, ctx));
                    ir.intern(mapped)
                }
            }
        };
        ctx.memo.insert(id, out);
        out
    }

    /// Intern a fresh-id twin of a register node (fresh disambiguator = a distinct
    /// physical register, so per-copy scratch never aliases).
    fn fresh_reg(&self, ir: &mut TileIr, node: &Node) -> TileId {
        let id = ir.fresh_reg_id();
        match node {
            Node::DefineFrag { dtype, frag, .. } => {
                ir.intern(Node::DefineFrag { id, dtype: dtype.clone(), frag: *frag })
            }
            Node::DefineReg { dtype, len, .. } => ir.intern(Node::DefineReg { id, dtype: dtype.clone(), len: *len }),
            _ => unreachable!("fresh_reg on non-register node"),
        }
    }

    /// The base register a store writes into (following any `After` wrap).
    fn store_target_reg(&self, ir: &TileIr, store: TileId) -> TileId {
        let buf = match ir.node(store) {
            Node::StoreGlobal { buf, .. } | Node::StoreRegVec { buf, .. } => *buf,
            other => panic!("END body is not a store: {other:?}"),
        };
        self.base_buffer(ir, buf)
    }

    /// Peel `After` wraps down to the underlying buffer node.
    fn base_buffer(&self, ir: &TileIr, id: TileId) -> TileId {
        match ir.node(id) {
            Node::After { val, .. } => self.base_buffer(ir, *val),
            _ => id,
        }
    }

    /// The loop-carried read: the (unique, for tk2's patterns) `After` in the body
    /// cone whose deps include the loop counter. `None` for a carry-free loop.
    fn find_carry_read(&self, ir: &TileIr, body: TileId, counter: TileId) -> Option<TileId> {
        reachable(ir, body).into_iter().find(|&id| match ir.node(id) {
            Node::After { deps, .. } => deps.contains(&counter),
            _ => false,
        })
    }
}

// ============================================================================
// Pass 2 — Const-fold addressing
// ============================================================================

/// Collapse all-constant `IndexAlu` address arithmetic to immediates (DESIGN.md §2.4).
pub struct ConstFoldPass;

impl Pass for ConstFoldPass {
    fn name(&self) -> &str {
        "const_fold_addressing"
    }
    fn band(&self) -> Band {
        Band::MemoryPlacement
    }
    /// Precondition (DESIGN.md §2.4 — "requires UNROLL"): the fold only fires on
    /// compile-time-constant fragment/loop steps, so the graph must already be flat
    /// (no rolled ranges). The runner rejects a const-fold placed before unroll.
    fn requires(&self, ir: &TileIr, root: TileId) -> bool {
        is_flat(ir, root)
    }
    /// Postcondition: no `IndexAlu` with two constant operands survives.
    fn ensures(&self, ir: &TileIr, root: TileId) -> bool {
        reachable(ir, root).into_iter().all(|id| match ir.node(id) {
            Node::IndexAlu { a, b, .. } => int_const(ir, *a).is_none() || int_const(ir, *b).is_none(),
            _ => true,
        })
    }
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Result<TileId, PassError> {
        Ok(fold(&mut ConstFold, ir, root))
    }
}

/// The nanopass folder: overrides only the `IndexAlu` arm. Because the driver visits
/// children before parents, one sweep propagates constants bottom-up to a fixpoint.
struct ConstFold;

impl Fold for ConstFold {
    fn fold_node(&mut self, ir: &mut TileIr, node: Node) -> TileId {
        let Node::IndexAlu { op, a, b } = node else {
            return ir.intern(node);
        };
        let (av, bv) = (int_const(ir, a), int_const(ir, b));
        // Both constant → evaluate (non-negative index arithmetic: floor div/mod).
        if let (Some(x), Some(y)) = (av, bv) {
            let v = match op {
                IndexOp::Add => x + y,
                IndexOp::Mul => x * y,
                IndexOp::Div => {
                    if y != 0 {
                        x.div_euclid(y)
                    } else {
                        0
                    }
                }
                IndexOp::Mod => {
                    if y != 0 {
                        x.rem_euclid(y)
                    } else {
                        0
                    }
                }
                IndexOp::Xor => x ^ y,
                IndexOp::Shr => x >> y,
                IndexOp::Shl => x << y,
            };
            return ir.intern(Node::Const { scalar: Scalar::Int(v), dtype: DType::Index });
        }
        // Algebraic identities that clean up the lane_rc `inner`/stride terms an
        // unrolled copy leaves behind (`x + 0`, `x · 1`, `x · 0`, `x / 1`, `x % 1`).
        let zero = |ir: &mut TileIr| ir.intern(Node::Const { scalar: Scalar::Int(0), dtype: DType::Index });
        match (op, av, bv) {
            (IndexOp::Add, _, Some(0)) => a,
            (IndexOp::Add, Some(0), _) => b,
            (IndexOp::Mul, _, Some(1)) => a,
            (IndexOp::Mul, Some(1), _) => b,
            (IndexOp::Mul, _, Some(0)) | (IndexOp::Mul, Some(0), _) => zero(ir),
            (IndexOp::Div, _, Some(1)) => a,
            (IndexOp::Mod, _, Some(1)) => zero(ir),
            _ => ir.intern(Node::IndexAlu { op, a, b }),
        }
    }
}

// ============================================================================
// The addressing pipeline
// ============================================================================

/// Run the two-pass addressing optimization (unroll → const-fold) through the
/// banded, contract-checked [`Pipeline`](crate::pass::Pipeline), returning the new
/// root. The pipeline verifies band monotonicity (`Tiling ≤ MemoryPlacement`) and
/// each pass's `requires`/`ensures` between steps.
pub fn optimize_addressing(ir: &mut TileIr, root: TileId) -> Result<TileId, PassError> {
    crate::pass::Pipeline::new().then(UnrollPass).then(ConstFoldPass).run(ir, root)
}
