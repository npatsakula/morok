//! The **pluggable clustered-schedule combinator** (DESIGN §5c) — the register-staged HK pipeline
//! as a composable driver over an OPEN set of cluster kinds, replacing the closed `Cluster` enum +
//! `run_clustered_body` + `pipeline_clustered` interpreter that used to live in [`crate::kernels`].
//!
//! The split of concerns:
//! - [`Hooks`] — the ONLY kernel-specific movement: `prefetch`/`commit`/`gather`. matmul's impl
//!   (`MatmulHooks`) rides the [`crate::movement`] handles; FA will supply its own. The compute math
//!   is NOT here — it rides each [`Compute`] cluster's body, so a new compute kind grows no `Hooks`.
//! - [`ClusterCx`] — the **safe-op layer**. A cluster body calls `prefetch`/`gather`/`commit`/`compute`
//!   and NEVER names a barrier or a dependency edge; each safe op threads the EXACT ordering edges
//!   the old `run_clustered_body` match-arm wove (the value-anchored `set_prio`, the WAR-over-all-
//!   gathers, the commit's war→commit→raw triple, the per-cluster boundary tokens). Dropping an edge
//!   is impossible: the op layer owns them.
//! - [`Cluster`] — an OPEN trait (new kinds = new `impl`s, not new enum arms). [`Mem`] and [`Compute`]
//!   are the two matmul kinds; the schedule is a heterogeneous `Vec<Box<dyn Cluster<H>>>`.
//! - [`Pipeline`] — owns the prologue/steady/epilogue bracket, the register+LDS carries, the warp-
//!   phase ping-pong, the End-fold, the resident fork, and runs the completeness [`verify`]er at
//!   `.build()`.
//!
//! **Byte-identical emission:** every `Builder` call is reproduced in the SAME order as the deleted
//! interpreter, so under hash-consing the arena is bit-for-bit the old one. The win is the STRUCTURE
//! (ordering-as-edges owned by the op layer, a build-time completeness check), not codegen.

use std::collections::HashSet;

use crate::build::{Builder, Effect, F32, Frag, Idx, Val};
use crate::ir::{Node, TileId, TileIr};

/// The kernel-specific hooks the pipeline drives — the ONLY kernel-specific part of the schedule.
/// For matmul: `Op` is the `(A-vecs, B-vecs)` operand bundle of one K-slice, `Reg` is the register-
/// staged fill (`FillRegs`). Each hook is a pure emission: it interns nodes and returns handles; ALL
/// ordering edges are threaded by the [`ClusterCx`] safe ops, never here.
pub(crate) trait Hooks {
    /// One K-slice's operand bundle (matmul: the `ri` A-fragments + `cj` B-fragments).
    type Op;
    /// The register-staged fill carried prefetch→commit (matmul: `FillRegs`).
    type Reg;

    /// Prefetch block `k_base` global→VGPR (the latency hide). Returns the staged registers and the
    /// load-pin anchors (unused by the compiler-visible schedule; kept for the API symmetry).
    fn prefetch(&mut self, b: &mut Builder, k_base: Idx) -> (Self::Reg, Vec<TileId>);
    /// Commit the staged registers VGPR→LDS behind `war`. Returns the store effects the RAW fences.
    fn commit(&mut self, b: &mut Builder, k_base: Idx, reg: &Self::Reg, war: &[TileId]) -> Vec<Effect>;
    /// Gather K-slice `slice` LDS→operand-frags after `raw`. Returns the operand bundle, its store-
    /// fence tokens (the WAR consumes them), and the value `op_anchor` the `set_prio` anchors on.
    fn gather(&mut self, b: &mut Builder, slice: usize, raw: &[TileId]) -> (Self::Op, Vec<TileId>, TileId);
}

/// A **compute body**: the kernel's per-cluster math, given the gathered operand bundle and the
/// carried accumulator reads → the new accumulator values. Edge-free (no barrier/dep) — the
/// [`ClusterCx::compute`] wrapper owns the `set_prio` bracket + the acc round-trip. This is what
/// makes the *compute* side pluggable: matmul's `Compute` carries the MFMA loop, FA's `Softmax`/`PV`
/// carry theirs, and `Hooks` never grows a per-kernel compute method.
pub(crate) type ComputeBody<H> = dyn Fn(&mut Builder, Option<&<H as Hooks>::Op>, &[Val<F32>]) -> Vec<Val<F32>>;

/// The **safe-op context** — holds the `Builder`, the `Hooks`, and the per-body carry state, and
/// exposes ONLY the four safe ops. Each op reproduces the corresponding `run_clustered_body` match-
/// arm's emission EXACTLY (same `Builder` calls, same order), threading the edges so a cluster body
/// never has to (and cannot) name a barrier or a dep. One `ClusterCx` is spun up per body pass
/// (steady / epilogue) and the driver walks the schedule through it.
pub(crate) struct ClusterCx<'a, H: Hooks> {
    b: &'a mut Builder,
    hooks: &'a mut H,
    accs: &'a [Frag<F32>],
    seed: &'a [TileId],
    carry: &'a [Vec<TileId>],
    k_next: Option<Idx>,
    n_acc: usize,
    // ── carries (persist across clusters within one body) ──
    entry: Vec<TileId>,
    prev_store: Vec<TileId>,
    all_gathers: Vec<TileId>,
    operands: Vec<Option<(H::Op, TileId)>>,
    reg: Option<H::Reg>,
    raw_next: Option<TileId>,
    tail_barrier: Option<TileId>,
    first_compute: bool,
    // ── per-cluster (reset by the driver before each cluster) ──
    this_gathers: Vec<TileId>,
    sealed: bool,
}

impl<H: Hooks> ClusterCx<'_, H> {
    /// The **prefetch** safe op = the `if mc.prefetch && Some(kn)=k_next { reg = ... }` arm: stage
    /// block k+1 global→VGPR (steady only — a no-op when `k_next` is `None`, e.g. epilogue/resident).
    pub(crate) fn prefetch(&mut self) {
        if let Some(kn) = self.k_next {
            let (r, _) = self.hooks.prefetch(self.b, kn);
            self.reg = Some(r);
        }
    }

    /// The **gather** safe op = one iteration of the gather loop: gdeps = `seed` + `entry`; the hook
    /// gathers slice `s`; its fence tokens extend both this cluster's list and the cumulative
    /// `all_gathers` (WAR-fenced by a later commit), and its operand lands at `operands[s]`.
    pub(crate) fn gather(&mut self, s: usize) {
        let mut gdeps = self.seed.to_vec();
        gdeps.extend(&self.entry);
        let (op, g, op_anchor) = self.hooks.gather(self.b, s, &gdeps);
        self.this_gathers.extend(g.iter().copied());
        self.all_gathers.extend(g.iter().copied());
        self.operands[s] = Some((op, op_anchor));
    }

    /// The **commit** safe op = the commit arm: WAR-fence EVERY gather so far, `commit`, RAW-fence
    /// the fill as the LDS carry-out, and SEAL the cluster (its own barrier is the boundary token).
    /// A no-op (leaving the cluster unsealed) when there is nothing to commit (`k_next`/`reg` absent).
    pub(crate) fn commit(&mut self) {
        if let Some(kn) = self.k_next
            && self.reg.is_some()
        {
            let war = self.b.barrier(Effect(self.all_gathers[0]), &self.all_gathers[1..]);
            let fill = self.hooks.commit(self.b, kn, self.reg.as_ref().unwrap(), &[war.dep()]);
            let fill_deps: Vec<TileId> = fill[1..].iter().map(|e| e.dep()).collect();
            let rn = self.b.barrier(fill[0], &fill_deps).dep();
            self.raw_next = Some(rn);
            self.tail_barrier = Some(rn);
            self.entry = vec![rn];
            self.sealed = true;
        }
    }

    /// The **compute** safe op = the Compute arm: `set_prio(1)` anchored on the operand VALUE (of
    /// gathered slice `s`); the acc reads route the carry (or the prior stores) + `entry` + the prio;
    /// run the cluster's `body` (the kernel math — MFMA for matmul, softmax/PV for FA) over the
    /// operand + acc reads; store back; `set_prio(0)` on the results; the closing `s_barrier` on the
    /// LAST store. Boundary token = `[bar, prio0]`. SEALED. The `body` is edge-free — this wrapper
    /// owns the bracket + round-trip, so the compute is pluggable without a per-kernel `Hooks` method.
    pub(crate) fn compute(&mut self, operand: Option<usize>, body: &ComputeBody<H>) {
        // Resolve the gathered operand (if any). `operand = Some(s)` for a compute that consumes a
        // gathered K-slice (matmul MFMA, FA QKᵀ/PV); `None` for one that consumes only the accumulator
        // carry (FA softmax). The `unwrap_or_else` is the BUILD-TIME coupling check: it panics during
        // kernel construction (not on device) if a `Compute(s)` is scheduled before slice `s` is gathered.
        let (op_ref, op_anchor) = match operand {
            Some(s) => {
                let (o, a) = self.operands[s].as_ref().unwrap_or_else(|| {
                    panic!(
                        "Compute over slice {s}: no gather populated it earlier in the schedule (reorder the clusters)"
                    )
                });
                (Some(o), Some(*a))
            }
            None => (None, None),
        };
        // `set_prio(1)` anchors on the operand VALUE, so it exists only when there IS an operand; an
        // operand-less compute skips it (nothing nameable before the reads) and relies on the closing
        // `set_prio(0)` + barrier.
        let prio1 = op_anchor.map(|a| self.b.set_prio(1, &[a]).dep());
        let mut reads: Vec<Val<F32>> = Vec::with_capacity(self.n_acc);
        for ij in 0..self.n_acc {
            let mut deps = if self.first_compute { self.carry[ij].clone() } else { vec![self.prev_store[ij]] };
            deps.extend(&self.entry);
            if let Some(p) = prio1 {
                deps.push(p);
            }
            reads.push(self.b.load_frag_vec_after(self.accs[ij], &deps));
        }
        let new = body(self.b, op_ref, &reads);
        let new_ids: Vec<TileId> = new.iter().map(|v| v.id).collect();
        let mut stores: Vec<Effect> = Vec::with_capacity(self.n_acc);
        for (a, v) in self.accs.iter().zip(&new) {
            stores.push(self.b.store_frag_vec(*a, *v));
        }
        self.prev_store = stores.iter().map(|e| e.dep()).collect();
        self.first_compute = false;
        let prio0 = self.b.set_prio(0, &new_ids).dep();
        let bar = self.b.barrier(stores[self.n_acc - 1], &self.prev_store[..self.n_acc - 1]).dep();
        self.tail_barrier = Some(bar);
        self.entry = vec![bar, prio0];
        self.sealed = true;
    }
}

/// An **open cluster kind** — the schedule-as-data unit. `build` calls only the [`ClusterCx`] safe
/// ops. Object-safe over a fixed `H`, so a schedule is a heterogeneous `Vec<Box<dyn Cluster<H>>>`.
pub(crate) trait Cluster<H: Hooks> {
    fn build(&self, cx: &mut ClusterCx<H>);
}

/// A boxed cluster is itself a cluster (so a pre-built `Box<dyn Cluster<H>>` — e.g. from a factory
/// that mints clusters dynamically — passes straight to [`Pipeline::cluster`]).
impl<H: Hooks> Cluster<H> for Box<dyn Cluster<H>> {
    fn build(&self, cx: &mut ClusterCx<H>) {
        (**self).build(cx);
    }
}

/// A **memory cluster**: optionally prefetch k+1, gather the listed K-slices, optionally commit k+1.
/// Built via `Mem::builder()` — every field defaults (no prefetch, no gathers, no commit), so a call
/// site names only what it sets: `Mem::builder().prefetch(true).gathers(vec![0]).build()`.
#[derive(bon::Builder)]
pub(crate) struct Mem {
    #[builder(default)]
    prefetch: bool,
    #[builder(default, into)]
    gathers: Vec<usize>,
    #[builder(default)]
    commit: bool,
}

impl<H: Hooks> Cluster<H> for Mem {
    fn build(&self, cx: &mut ClusterCx<H>) {
        if self.prefetch {
            cx.prefetch();
        }
        for &s in &self.gathers {
            cx.gather(s);
        }
        if self.commit {
            cx.commit();
        }
    }
}

/// A **compute cluster**, carrying its own edge-free `body` (the kernel math — matmul's MFMA loop,
/// FA's softmax / PV). The combinator brackets ANY body uniformly, so a new compute kind is a new
/// `Compute::new(operand, body)` — no `Hooks` growth. `operand` is the gathered K-slice the body
/// consumes: `Compute::new(3, body)` for slice 3, `Compute::new(None, body)` for an operand-less
/// compute (e.g. FA softmax, which consumes only the accumulator carry). The `body` receives
/// `Option<&Op>` accordingly.
pub(crate) struct Compute<H: Hooks> {
    operand: Option<usize>,
    body: Box<ComputeBody<H>>,
}

impl<H: Hooks> Compute<H> {
    pub(crate) fn new(
        operand: impl Into<Option<usize>>,
        body: impl Fn(&mut Builder, Option<&H::Op>, &[Val<F32>]) -> Vec<Val<F32>> + 'static,
    ) -> Self {
        Compute { operand: operand.into(), body: Box::new(body) }
    }
}

impl<H: Hooks> Cluster<H> for Compute<H> {
    fn build(&self, cx: &mut ClusterCx<H>) {
        cx.compute(self.operand, self.body.as_ref());
    }
}

/// The threaded result of one body pass (steady / epilogue).
struct BodyOut {
    prev_store: Vec<TileId>,
    raw_next: Option<TileId>,
    tail_barrier: Option<TileId>,
}

/// Walk a schedule once through a fresh [`ClusterCx`], emitting each cluster's bracket + carries.
/// Used for BOTH the steady body (`k_next=Some`, `carry=[inited,kr]`) and the epilogue
/// (`k_next=None`, `carry=[]`, reading the post-loop `acc_loop` frags). The DRIVER owns the pure-
/// gather seal: after `cluster.build`, if the cluster gathered but was NOT sealed by a commit, close
/// its `s_barrier` over THIS cluster's gathers; if nothing was emitted (epilogue commit cluster) skip.
#[allow(clippy::too_many_arguments)]
fn run_body<H: Hooks>(
    b: &mut Builder,
    clusters: &[Box<dyn Cluster<H>>],
    hooks: &mut H,
    ksteps: usize,
    n_acc: usize,
    accs: &[Frag<F32>],
    seed: &[TileId],
    carry: &[Vec<TileId>],
    k_next: Option<Idx>,
) -> BodyOut {
    let mut cx = ClusterCx {
        b,
        hooks,
        accs,
        seed,
        carry,
        k_next,
        n_acc,
        entry: Vec::new(),
        prev_store: Vec::new(),
        all_gathers: Vec::new(),
        operands: (0..ksteps).map(|_| None).collect(),
        reg: None,
        raw_next: None,
        tail_barrier: None,
        first_compute: true,
        this_gathers: Vec::new(),
        sealed: false,
    };
    for cluster in clusters {
        cx.this_gathers.clear();
        cx.sealed = false;
        cluster.build(&mut cx);
        // Seal a pure-gather Mem cluster (gathered, not commit-sealed): the workgroup barrier fences
        // this cluster's gather reads, its BODY the LAST gather so the sync lands after the whole
        // cluster. A cluster that emitted nothing (epilogue commit) is skipped.
        if !cx.this_gathers.is_empty() && !cx.sealed {
            let n = cx.this_gathers.len();
            let bar = cx.b.barrier(Effect(cx.this_gathers[n - 1]), &cx.this_gathers[..n - 1]).dep();
            cx.tail_barrier = Some(bar);
            cx.entry = vec![bar];
        }
    }
    BodyOut { prev_store: cx.prev_store, raw_next: cx.raw_next, tail_barrier: cx.tail_barrier }
}

/// The **clustered pipeline combinator** — construct with [`pipeline`], push clusters with
/// [`Self::cluster`], emit with [`Self::build`]. Owns the prologue/steady/epilogue bracket, the
/// register+LDS carries, the warp-phase ping-pong, the End-fold, the resident fork, and runs the
/// completeness [`verify`]er at build.
pub(crate) struct Pipeline<'a, H: Hooks> {
    b: &'a mut Builder,
    hooks: H,
    nblocks: usize,
    k_step: usize,
    ksteps: usize,
    accs: &'a [Frag<F32>],
    inited: &'a [Effect],
    warp_row: Option<Idx>,
    asm_gather: bool,
    resident: bool,
    clusters: Vec<Box<dyn Cluster<H>>>,
}

/// Open a clustered pipeline over `hooks`. `nblocks = k/k_step ≥ 2`; `warp_row = Some` enables the
/// wave-phase ping-pong; `resident` drops the steady prefetch/commit (compute-resident microkernel).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pipeline<'a, H: Hooks>(
    b: &'a mut Builder,
    nblocks: usize,
    k_step: usize,
    ksteps: usize,
    accs: &'a [Frag<F32>],
    inited: &'a [Effect],
    warp_row: Option<Idx>,
    asm_gather: bool,
    resident: bool,
    hooks: H,
) -> Pipeline<'a, H> {
    Pipeline { b, hooks, nblocks, k_step, ksteps, accs, inited, warp_row, asm_gather, resident, clusters: Vec::new() }
}

impl<'a, H: Hooks> Pipeline<'a, H> {
    /// Append a cluster to the schedule (fluent). Takes any `impl Cluster<H>` (boxed internally), so
    /// the call site is `.cluster(Mem { .. })` / `.cluster(Compute::new(3, body))` — no `Box::new`.
    pub(crate) fn cluster(mut self, c: impl Cluster<H> + 'static) -> Self {
        self.clusters.push(Box::new(c));
        self
    }

    /// Emit the pipeline: prologue commit + wave-phase seed, the steady body over the schedule, the
    /// End-fold of the carries, the epilogue body + rebalance, then the completeness check. Returns
    /// the post-loop accumulator frags (the scatter source). Byte-identical to the old
    /// `pipeline_clustered`.
    pub(crate) fn build(self) -> Vec<Frag<F32>> {
        let Pipeline { b, mut hooks, nblocks, k_step, ksteps, accs, inited, warp_row, asm_gather, resident, clusters } =
            self;
        assert!(nblocks >= 2, "pipeline needs nblocks ≥ 2");
        let n_acc = accs.len();
        let ks_c = b.idx_const(k_step as i64);
        let one = b.idx_const(1);

        // ── prologue: commit block 0; the eq=1 wave-phase barrier (ordered after the commit via the
        //    warp_row operand carrying the raw_seed edge) offsets warp-row 1 one cluster. ──
        let zero = b.idx_const(0);
        let (reg0, _) = hooks.prefetch(b, zero);
        let fill0 = hooks.commit(b, zero, &reg0, &[]);
        let fill0_deps: Vec<TileId> = fill0[1..].iter().map(|e| e.dep()).collect();
        let raw_seed = b.barrier(fill0[0], &fill0_deps);
        let loop_seed = match warp_row {
            Some(wr) => {
                let wr_seeded = b.idx_after(wr, &[raw_seed.dep()]);
                b.wave_barrier(wr_seeded, 1, &[]).dep()
            }
            None => raw_seed.dep(),
        };

        // ── steady loop: block k's gathers via the carried RAW; prefetch/commit block k+1. ──
        let kr = b.range((nblocks - 1) as i64);
        let tk = b.counter(kr);
        let k_next_idx = b.idx_add(tk, one);
        let k_next = b.idx_mul(k_next_idx, ks_c);
        let carry: Vec<Vec<TileId>> = (0..n_acc).map(|ij| vec![inited[ij].dep(), kr.dep()]).collect();
        // Compute-resident: the whole tile is staged ONCE in the prologue, so the steady loop drops
        // BOTH prefetch and commit (`k_next=None`); the gathers still fire, re-reading the resident
        // block via `[loop_seed, kr]`.
        let steady_k_next = if resident { None } else { Some(k_next) };
        let body =
            run_body(b, &clusters, &mut hooks, ksteps, n_acc, accs, &[loop_seed, kr.dep()], &carry, steady_k_next);

        // ── loop close: fold the last-slice stores, raw_next (LDS carry, streaming only), AND the
        //    final cluster's barrier (else DCE drops it → unbalanced count → deadlock) under one End. ──
        let last = Effect(body.prev_store[n_acc - 1]);
        let mut carried: Vec<TileId> = body.prev_store[..n_acc - 1].to_vec();
        match body.raw_next {
            Some(rn) => carried.push(rn),
            None => assert!(resident, "streaming schedule must contain a commit cluster (raw_next carry)"),
        }
        carried.push(body.tail_barrier.expect("steady body must end on a cluster barrier"));
        // HK positional wall lattice: the `sched.barrier(0)` paired with every `s_barrier` pins the
        // opaque asm `ds_read_b64`s inside their cluster (load-bearing for the asm gather's correctness).
        if asm_gather {
            carried.push(b.wall_marker().dep());
        }
        let combined = b.combine(last, &carried);
        let ended = b.end(combined, &[kr]);
        let acc_loop: Vec<Frag<F32>> = accs.iter().map(|a| b.frag_after(*a, &[ended.dep()])).collect();

        // ── epilogue: the same schedule for the LAST block (via the End's carried RAW), no
        //    prefetch/commit; then the eq=0 wave-phase barrier rebalances warp-row 0. ──
        let ep_carry: Vec<Vec<TileId>> = (0..n_acc).map(|_| Vec::new()).collect();
        let ep = run_body(b, &clusters, &mut hooks, ksteps, n_acc, &acc_loop, &[ended.dep()], &ep_carry, None);
        let scatter_seed = warp_row.map(|wr| {
            let anchor = ep.tail_barrier.expect("epilogue must end on a cluster barrier");
            let wr_seeded = b.idx_after(wr, &[anchor]);
            b.wave_barrier(wr_seeded, 0, &[]).dep()
        });
        let out: Vec<Frag<F32>> = acc_loop
            .iter()
            .enumerate()
            .map(|(ij, a)| {
                let mut deps = vec![ep.prev_store[ij]];
                deps.extend(scatter_seed);
                b.frag_after(*a, &deps)
            })
            .collect();

        // ── completeness check: carry-completeness (checked inline above via the End-fold) + the
        //    wave-phase balance over the emitted output cone. A build-time panic. ──
        let roots: Vec<TileId> = out.iter().map(|f| f.id).collect();
        verify(&b.ir, &roots, body.raw_next, body.tail_barrier, resident);
        out
    }
}

/// The **completeness verifier** (DESIGN §5c/3c), run at [`Pipeline::build`]:
/// - **carry-completeness:** a streaming (non-resident) schedule MUST carry a commit's `raw_next`
///   (else the LDS carry is dropped), and the body MUST end on a cluster barrier (`tail_barrier`).
/// - **wave-phase balance:** the asymmetric barrier pair must be balanced (equal eq=0 and eq=1 wave
///   barriers reachable from the output) or one warp-row waits on an `s_barrier` the other never
///   reaches and the workgroup deadlocks.
///
/// A build-time panic (a kernel-authoring bug, not recoverable) — so a dropped ordering edge is a
/// construction-time failure, not a silent device race.
pub(crate) fn verify(
    ir: &TileIr,
    roots: &[TileId],
    raw_next: Option<TileId>,
    tail_barrier: Option<TileId>,
    resident: bool,
) {
    assert!(raw_next.is_some() || resident, "streaming schedule must contain a commit cluster (raw_next carry)");
    assert!(tail_barrier.is_some(), "steady body must end on a cluster barrier");
    let mut reach: HashSet<TileId> = HashSet::new();
    for &r in roots {
        reach.extend(crate::passes::reachable(ir, r));
    }
    let count = |want: i64| {
        reach.iter().filter(|&&id| matches!(ir.node(id), Node::WaveBarrier { eq, .. } if *eq == want)).count()
    };
    let (n0, n1) = (count(0), count(1));
    assert_eq!(n0, n1, "wave-phase barriers unbalanced (eq=0: {n0}, eq=1: {n1}) — would deadlock the workgroup");
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::build::Builder;

    /// The dropped-commit-edge is now a BUILD-TIME failure: a streaming schedule whose body produced
    /// no commit (`raw_next=None`, `resident=false`) is rejected by the verifier (was a silent LDS-
    /// carry drop → device race).
    #[test]
    #[should_panic(expected = "must contain a commit cluster")]
    fn verify_rejects_streaming_without_commit() {
        let b = Builder::new("t");
        verify(&b.ir, &[], None, Some(TileId(0)), false);
    }

    /// An unbalanced wave-phase pair (one eq=1 seed, no eq=0 rebalance reachable) is rejected — the
    /// deadlock guard.
    #[test]
    #[should_panic(expected = "unbalanced")]
    fn verify_rejects_unbalanced_wave_phase() {
        let mut b = Builder::new("t");
        let wr = b.block_axis(64);
        let frag = b.define_frag::<F32>(crate::ir::FragMap::gfx942_16x16(true));
        let s = b.f32(0.0);
        let e = b.store_frag_vec(frag, s);
        let wb = b.wave_barrier(wr, 1, &[e.dep()]);
        // route the eq=1 barrier into a live root; no eq=0 anywhere → unbalanced.
        let root = b.idx_after(wr, &[wb.dep()]);
        verify(&b.ir, &[root.0], Some(TileId(0)), Some(TileId(0)), false);
    }
}
