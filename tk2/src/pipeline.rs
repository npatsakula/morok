//! The **pluggable clustered-schedule combinator** (DESIGN §5c) — the register-staged HK pipeline
//! as a composable driver over an OPEN set of cluster kinds, replacing the closed `Cluster` enum +
//! `run_clustered_body` + `pipeline_clustered` interpreter that used to live in [`crate::kernels`].
//!
//! The split of concerns:
//! - [`Hooks`] — the ONLY kernel-specific movement: `prefetch`/`commit`/`gather`. matmul's impl
//!   (`MatmulHooks`) rides the [`crate::tile_move`] handles; FA will supply its own. The compute math
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
use std::marker::PhantomData;

use crate::build::{Builder, Edge, Effect, Elem, F32, Frag, Idx, Scope, Val};
use crate::ir::{FragMap, Node, TileId, TileIr};

/// The commit's **drain placement policy** (DESIGN §5c) — WHERE the collaborative fill's LDS writes
/// are made visible before the next-iteration gather. [`CommitDrain::IntrinsicAuto`] is the
/// compiler-visible `ds_write` whose `lgkmcnt(0)` the C6 RAW `s_barrier` auto-drains. The asm variant
/// uses the waitcnt-opaque `asm ds_write_b64` (the barrier can NOT auto-drain it):
/// [`CommitDrain::AsmDeferred`] (Phase C-b, HK's deferred drain) leaves the C6 barrier BARE and moves the
/// manual drain to C7's tail — after the 32 MFMAs (hidden) and before C7's tail barrier (so the
/// drain-before-barrier still gives every wave cross-wave visibility of its own writes).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CommitDrain {
    IntrinsicAuto,
    /// Lever-3 LDS double-buffer: commit(k+1) writes the OTHER parity half than gather(k) reads, so the
    /// read-before-overwrite WAR hazard is gone — the WAR seal (an `s_barrier` per iteration) is DROPPED.
    /// The RAW (a later iteration's gather seeing these writes) is still carried by `raw_next`. Same
    /// intrinsic (compiler-visible `ds_write`) RAW-drain as `IntrinsicAuto`, one fewer workgroup barrier.
    IntrinsicNoWar,
    AsmDeferred,
    /// Disjoint K2/V3 asm writes are drained and published immediately. This leaves a later compute
    /// barrier in the body, which is required for safe asymmetric wave-phase progress.
    AsmPublishedNoWar,
}

/// What kind of machine completion a hook's commit effects require before LDS publication. This is
/// deliberately separate from [`CommitDrain`]: hooks classify what they emitted, while the schedule
/// chooses where and how that class is completed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CommitCompletion {
    /// Compiler-visible LDS stores; a fenced barrier supplies their LDS completion.
    Intrinsic,
    /// Waitcnt-opaque LDS stores; publication requires an explicit `lgkmcnt(0)`.
    Opaque,
}

/// One hook commit and the completion class required by its effects.
pub(crate) struct CommitBatch {
    effects: Vec<Effect>,
    completion: CommitCompletion,
}

impl CommitBatch {
    pub(crate) fn new(effects: Vec<Effect>, completion: CommitCompletion) -> Self {
        assert!(!effects.is_empty(), "a commit batch must contain at least one effect");
        Self { effects, completion }
    }
}

fn validate_commit_policy(completion: CommitCompletion, drain: CommitDrain, bare_seals: bool) {
    let expected = match drain {
        CommitDrain::IntrinsicAuto | CommitDrain::IntrinsicNoWar => CommitCompletion::Intrinsic,
        CommitDrain::AsmDeferred | CommitDrain::AsmPublishedNoWar => CommitCompletion::Opaque,
    };
    assert_eq!(
        completion, expected,
        "commit completion {completion:?} is incompatible with publication policy {drain:?}"
    );
    assert!(
        !bare_seals || completion == CommitCompletion::Opaque,
        "bare seals require an opaque-LDS commit with explicit completion"
    );
}

/// The **single LDS-publication decision** a kernel makes per commit. It fixes BOTH halves that used to
/// be chosen independently and reconciled at runtime by [`validate_commit_policy`]: the [`CommitDrain`]
/// the schedule uses and the [`CommitCompletion`] the commit hook emits. Deriving both from one value
/// makes the two structurally unable to drift — the drain/completion mismatch the validator guards is
/// now unrepresentable at the authoring layer (the validator stays as unfireable defense-in-depth).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Publication {
    /// Compiler-visible `ds_write` + a fenced barrier supplies LDS completion (single-buffered).
    IntrinsicWar,
    /// Intrinsic writes to the disjoint double-buffer half — the WAR seal is dropped.
    IntrinsicNoWar,
    /// Waitcnt-opaque asm writes, drained lazily (the hot-loop tail hides the `lgkmcnt(0)`).
    AsmDeferred,
    /// Opaque asm writes drained + published immediately (double-buffered, no WAR seal).
    AsmPublished,
}

impl Publication {
    /// The schedule's drain policy (fed to [`Sched::commit_drain`]).
    pub(crate) fn drain(self) -> CommitDrain {
        match self {
            Publication::IntrinsicWar => CommitDrain::IntrinsicAuto,
            Publication::IntrinsicNoWar => CommitDrain::IntrinsicNoWar,
            Publication::AsmDeferred => CommitDrain::AsmDeferred,
            Publication::AsmPublished => CommitDrain::AsmPublishedNoWar,
        }
    }

    /// The machine completion the commit hook's effects require (fed to [`CommitBatch::new`]).
    pub(crate) fn completion(self) -> CommitCompletion {
        match self {
            Publication::IntrinsicWar | Publication::IntrinsicNoWar => CommitCompletion::Intrinsic,
            Publication::AsmDeferred | Publication::AsmPublished => CommitCompletion::Opaque,
        }
    }
}

/// The kernel-specific hooks the pipeline drives — the ONLY kernel-specific part of the schedule.
/// For matmul: `Op` is the `(A-vecs, B-vecs)` operand bundle of one K-slice, `Reg` is the register-
/// staged fill (`FillRegs`). Each hook is a pure emission: it interns nodes and returns handles; ALL
/// ordering edges are threaded by the [`ClusterCx`] safe ops, never here.
pub(crate) trait Hooks {
    /// One K-slice's operand bundle (matmul: the `ri` A-fragments + `cj` B-fragments).
    type Op;
    /// The register-staged fill carried prefetch→commit (matmul: `FillRegs`).
    type Reg;

    /// The number of independently-prefetchable operand tiles (matmul: 2 = A, B). The prologue stages
    /// all; the steady schedule may SPLIT them across separate `Mem` clusters (HK loads A@C0, B@C4) so
    /// each load hides under a different compute cluster instead of bunching at the loop top — the
    /// `sched.barrier(0)` walls already emitted at each cluster boundary pin the split placement.
    const PREFETCH_TILES: usize;

    /// Prologue issue order. Streaming kernels normally use declaration order; a mixed direct/staged
    /// transfer may override this so older register loads precede younger direct-to-LDS operations.
    fn prologue_prefetch_tiles(&self) -> Vec<usize> {
        (0..Self::PREFETCH_TILES).collect()
    }

    /// Prefetch operand-tile `tile` of block `k_base` global→VGPR (the latency hide), folding the
    /// staged registers into `prev` (the partial fill accumulated by earlier prefetch clusters this
    /// iteration; `None` = start fresh). `order` is the cluster entry: the load is ordered after it so
    /// it lands in this cluster (the split pin) instead of floating to the loop top. Returns the fill
    /// AND the load result values — the `sched_fence(0)` load-pin anchors on them (see [`ClusterCx::prefetch`]).
    fn prefetch(
        &mut self,
        b: &mut Builder,
        k_base: Idx,
        tile: usize,
        prev: Option<Self::Reg>,
        order: &[Edge],
    ) -> (Self::Reg, Vec<Edge>);
    /// Commit the staged registers VGPR→LDS behind `war`, classifying the returned effects by the
    /// machine completion they require. The pipeline rejects a mismatched publication policy.
    fn commit(&mut self, b: &mut Builder, k_base: Idx, reg: &Self::Reg, war: &[Edge]) -> CommitBatch;
    /// Gather K-slice `slice` LDS→operand-frags after `raw`, for the CURRENT block `block` (its parity
    /// selects the read buffer under LDS double-buffering; single-buffered hooks ignore it). Returns the
    /// operand bundle, its store-fence tokens (the WAR consumes them), and the `op_anchor` `set_prio` uses.
    fn gather(
        &mut self,
        b: &mut Builder,
        slice: usize,
        block: BlockCounter,
        raw: &[Edge],
    ) -> (Self::Op, Vec<Edge>, Edge);

    /// Tie every waitcnt-opaque gather value through its completed `lgkmcnt(0)` wait. Intrinsic-only
    /// hooks must still implement this explicitly, normally by rejecting an impossible invocation.
    fn ready_after_lgkm(&mut self, b: &mut Builder, op: Self::Op, wait: TileId) -> Self::Op;
}

/// A **compute-channel value** (DESIGN §3.2 — gentle typing: the dtype rides as DATA, not a phantom type
/// param). A cluster body reads its declared slots as these and returns its declared writes as these.
/// Every current kernel carries f32 slots (`o`/`m`/`l`, QKᵀ scores); the enum keeps the dtype as data so
/// a second carried dtype is one variant away, not a type-parameter rewrite.
#[derive(Copy, Clone)]
pub(crate) enum SlotVal {
    F32(Val<F32>),
}

impl SlotVal {
    /// Unwrap the f32 channel value.
    pub(crate) fn f32(self) -> Val<F32> {
        let SlotVal::F32(v) = self;
        v
    }
    fn id(self) -> TileId {
        let SlotVal::F32(v) = self;
        v.id
    }
}

/// A pipeline **accumulator/temporary slot** — its register fragment, dtype+map riding as DATA (the
/// heterogeneous carry of DESIGN §3.2). Whether a slot is CARRIED (seeded + loop-carried + End-folded,
/// e.g. GEMM's C, FA's `o`/`m`/`l`) or a per-iteration TEMPORARY (no seed, not carried — produced and
/// consumed within one iteration, e.g. FA's `s`=QKᵀ scores, `p`=softmax weights) is set by the
/// pipeline's `inited` (`Some` seed ⇒ carried, `None` ⇒ temporary), NOT by the slot itself. FA-32's
/// rotated score tile is carried; its per-iteration probability tile is temporary.
#[derive(Copy, Clone)]
pub(crate) enum AccSlot {
    F32(Frag<F32>),
}

impl AccSlot {
    /// Unwrap the f32 fragment — the post-loop scatter source.
    pub(crate) fn f32(self) -> Frag<F32> {
        let AccSlot::F32(f) = self;
        f
    }
    fn load_after(self, b: &mut Builder, deps: &[Edge]) -> SlotVal {
        let AccSlot::F32(f) = self;
        SlotVal::F32(b.load_frag_vec_after(f, deps))
    }
    fn store(self, b: &mut Builder, v: SlotVal) -> Effect {
        let (AccSlot::F32(f), SlotVal::F32(x)) = (self, v);
        b.store_frag_vec(f, x)
    }
    fn after(self, b: &mut Builder, deps: &[Edge]) -> AccSlot {
        let AccSlot::F32(f) = self;
        AccSlot::F32(b.frag_after(f, deps))
    }
    fn id(self) -> TileId {
        let AccSlot::F32(f) = self;
        f.id
    }
}

// ── typed slot handles — newtypes over the runtime slot index the engine uses, with the
//    element dtype as a phantom and "carried vs temporary" encoded in the TYPE. They retire the bare
//    `usize` the kernels used to hand-number (`slot_m = dtiles`, `[slot_s, slot_m, slot_l].chain(…)`),
//    the documented mis-numbering hazard, while erasing to the SAME index the `ClusterCx`/End-fold
//    already consume — so the engine is untouched and emission stays byte-identical. ──────────────────

/// A **carried accumulator slot** (seeded + loop-carried + End-folded): GEMM's C tile, FA's `o`/`m`/`l`/`s`.
#[derive(Copy, Clone)]
pub(crate) struct Acc<E: Elem> {
    idx: usize,
    _e: PhantomData<E>,
}

/// A **per-iteration temporary slot** (produced+consumed within one pass, no seed, not carried): FA's `p`.
#[derive(Copy, Clone)]
pub(crate) struct Temp<E: Elem> {
    idx: usize,
    _e: PhantomData<E>,
}

/// A **contiguous carried group** sharing one seed: GEMM's `c[ri·cj]`, FA's `o[dtiles]`. Addressable as
/// a group (its slots feed a read/write set via [`AsSlots`]) or per-tile ([`Self::slot`]).
#[derive(Copy, Clone)]
pub(crate) struct AccArray<E: Elem> {
    base: usize,
    len: usize,
    _e: PhantomData<E>,
}

impl<E: Elem> Acc<E> {
    pub(crate) fn index(self) -> usize {
        self.idx
    }
}
impl<E: Elem> AccArray<E> {
    /// The `i`-th slot of the group (bounds-checked — cheap insurance against the off-by-one this targets).
    pub(crate) fn slot(self, i: usize) -> Acc<E> {
        assert!(i < self.len, "AccArray index {i} out of bounds (len {})", self.len);
        Acc { idx: self.base + i, _e: PhantomData }
    }
}

/// Flatten typed slot handles into the `Vec<usize>` read/write index list the engine consumes — the
/// typed replacement for hand-built `[slot_s, slot_m, slot_l].into_iter().chain(0..dtiles)` arithmetic.
pub(crate) trait AsSlots {
    fn push_slots(&self, out: &mut Vec<usize>);
}
impl<E: Elem> AsSlots for Acc<E> {
    fn push_slots(&self, out: &mut Vec<usize>) {
        out.push(self.idx);
    }
}
impl<E: Elem> AsSlots for Temp<E> {
    fn push_slots(&self, out: &mut Vec<usize>) {
        out.push(self.idx);
    }
}
impl<E: Elem> AsSlots for AccArray<E> {
    fn push_slots(&self, out: &mut Vec<usize>) {
        out.extend(self.base..self.base + self.len);
    }
}

/// Build a `reads`/`writes` slot-index list from typed handles in declaration order:
/// `slot_set![carry.s, carry.m, carry.l, carry.o]`.
macro_rules! slot_set {
    ($($h:expr),* $(,)?) => {{
        let mut v: ::std::vec::Vec<usize> = ::std::vec::Vec::new();
        $( $crate::pipeline::AsSlots::push_slots(&$h, &mut v); )*
        v
    }};
}
pub(crate) use slot_set;

/// The **seed policy** of a carried [`SlotSet`] slot — how [`SlotSet::finish`] initialises it (a
/// temporary carries no `Init` and finishes to `None`). Mirrors the two accumulator seeders:
/// [`Init::Zero`] → [`Builder::zero_init_frag`], [`Init::Const`] → [`Builder::const_init_frag`].
#[derive(Copy, Clone)]
pub(crate) enum Init {
    /// Seed to 0 (GEMM's C, FA's `o`/`l` running norm).
    Zero,
    /// Seed to a constant (FA's `m` running max = −∞).
    Const(f32),
}

/// A **declarative accumulator-slot builder** — derives the pipeline's `(accs, inited)`
/// vectors from a slot DECLARATION, retiring the hand-written `slot_m = dtiles`, `slot_l = dtiles + 1`,
/// … index bookkeeping FA-32 used to carry (the error-prone part: a mis-numbered slot silently reads
/// the wrong fragment).
///
/// **Deferred-init / byte-identity contract:** each `carried*`/`temp` call allocates its fragment
/// IMMEDIATELY (assigning the next slot index in declaration order) but RECORDS its init policy rather
/// than emitting it; [`SlotSet::finish`] then emits ALL the inits together, in declaration order. This
/// reproduces the old "all `define_frag`s first, then all inits" emission exactly, so under hash-consing
/// the arena is bit-for-bit unchanged. (Emitting an init at declaration time would interleave
/// define/init and change the node order — so `finish` owns every init.)
pub(crate) struct SlotSet {
    slots: Vec<(Frag<F32>, Option<Init>)>,
}

impl SlotSet {
    pub(crate) fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Allocate one f32 fragment at the next slot index, recording its init policy (`None` = temporary).
    fn push(&mut self, b: &mut Builder, map: FragMap, init: Option<Init>) -> usize {
        let idx = self.slots.len();
        let frag = b.define_frag::<F32>(map);
        self.slots.push((frag, init));
        idx
    }

    /// Declare one carried f32 accumulator → an [`Acc`] handle (loop-carried + seeded + End-folded).
    pub(crate) fn carried_typed(&mut self, b: &mut Builder, map: FragMap, init: Init) -> Acc<F32> {
        Acc { idx: self.push(b, map, Some(init)), _e: PhantomData }
    }

    /// Declare a run of `count` carried f32 accumulators sharing one `init` → an [`AccArray`] (GEMM's
    /// `c[ri·cj]`, FA's `o[dtiles]`). Slots are pushed in order, so emission matches the old per-slot loop.
    pub(crate) fn carried_array(&mut self, b: &mut Builder, count: usize, map: FragMap, init: Init) -> AccArray<F32> {
        let base = self.slots.len();
        for _ in 0..count {
            self.push(b, map, Some(init));
        }
        AccArray { base, len: count, _e: PhantomData }
    }

    /// Declare one temporary f32 slot (produced+consumed within a pass, no seed) → a [`Temp`] handle.
    pub(crate) fn temp_typed(&mut self, b: &mut Builder, map: FragMap) -> Temp<F32> {
        Temp { idx: self.push(b, map, None), _e: PhantomData }
    }

    /// Emit ALL inits (in declaration order; temporaries → `None`) and return the pipeline's
    /// `(accs, inited)` carry vectors.
    pub(crate) fn finish(self, b: &mut Builder) -> (Vec<AccSlot>, Vec<Option<Effect>>) {
        let accs = self.slots.iter().map(|&(f, _)| AccSlot::F32(f)).collect();
        let inited = self
            .slots
            .iter()
            .map(|&(f, init)| match init {
                Some(Init::Zero) => Some(b.zero_init_frag(f)),
                Some(Init::Const(v)) => Some(b.const_init_frag(f, v)),
                None => None,
            })
            .collect();
        (accs, inited)
    }
}

/// The **current KV-block counter** handed to each compute body — the 0-based streaming index.
/// LAZY on purpose: a body that doesn't mask never materialises it, so it emits NO node and the
/// emitted (reachable) IR of GEMM/FA-16 stays byte-identical (`test::byte_identity`). Only a body that
/// masks calls [`Self::idx`], which reuses the live loop counter in the steady pass and mints an
/// `idx_const(nblocks-1)` in the epilogue (the last, only-possibly-ragged block).
#[derive(Copy, Clone)]
pub(crate) enum BlockCounter {
    /// Steady loop: the live `counter(kr)` — an existing node, so reusing it emits nothing.
    Steady(Idx, Scope),
    /// Epilogue: the last block index `nblocks-1`, materialised to an `idx_const` only on demand.
    Epilogue(i64, Scope),
}

impl BlockCounter {
    /// Materialise the block index as an [`Idx`]. Steady = the existing loop counter (no new node);
    /// epilogue = a fresh `idx_const(nblocks-1)` (a new node — so a masking body pays for it, an
    /// unmasked one doesn't, keeping GEMM/FA-16 byte-identical).
    pub(crate) fn idx(self, b: &mut Builder) -> Idx {
        match self {
            BlockCounter::Steady(i, scope) => b.scope_idx(i, scope),
            BlockCounter::Epilogue(n, scope) => {
                let n = b.idx_const(n);
                b.scope_idx(n, scope)
            }
        }
    }

    /// The lexical region in which movement/address expressions for this block are authored.
    pub(crate) fn scope(self) -> Scope {
        match self {
            BlockCounter::Steady(_, scope) | BlockCounter::Epilogue(_, scope) => scope,
        }
    }
}

/// A **compute body**: the kernel's per-cluster math, given the gathered operand bundle, the values of
/// the slots the cluster DECLARED it reads (in `reads` order), and the current [`BlockCounter`] → the
/// new values for the slots it DECLARED it writes (in `writes` order). Edge-free (no barrier/dep) — the
/// [`ClusterCx::compute`] wrapper owns the `set_prio` bracket + the per-slot round-trip. The declared
/// read/write SUBSETS are what let a cluster touch only the state it uses (no dead round-trip) over a
/// heterogeneous slot set. The block counter is what a ragged-tail mask needs (FA's `global_kv_index =
/// block·kv_blk + lane_kv`); bodies that don't mask (GEMM, PV) simply ignore it — and because it is
/// materialised lazily, ignoring it references no node, so their emitted IR is byte-unchanged.
pub(crate) type ComputeBody<H> =
    dyn Fn(&mut Builder, Option<&<H as Hooks>::Op>, &[SlotVal], BlockCounter) -> Vec<SlotVal>;

/// The **safe-op context** — holds the `Builder`, the `Hooks`, and the per-body carry state, and
/// exposes ONLY the four safe ops. Each op reproduces the corresponding `run_clustered_body` match-
/// arm's emission EXACTLY (same `Builder` calls, same order), threading the edges so a cluster body
/// never has to (and cannot) name a barrier or a dep. One `ClusterCx` is spun up per body pass
/// (steady / epilogue) and the driver walks the schedule through it.
pub(crate) struct ClusterCx<'a, H: Hooks> {
    b: &'a mut Builder,
    hooks: &'a mut H,
    accs: &'a [AccSlot],
    /// Per-slot carry-in edge: `[inited, range]` (steady) / `[]` (epilogue, the `acc_loop` frag observes
    /// the loop `End`) for a CARRIED slot; `[]` for a TEMPORARY (which must be written before it is read).
    carry: &'a [Vec<Edge>],
    /// Per-slot carried flag (`inited[s].is_some()`) — a read of a not-yet-written slot is a carry-in
    /// (carried) or an authoring bug (temporary read before produced).
    is_carried: &'a [bool],
    seed: &'a [Edge],
    k_next: Option<Idx>,
    /// The **current KV-block counter** routed into each compute `body` (see [`BlockCounter`]): the loop
    /// counter in the steady pass, `nblocks-1` in the epilogue. Lazy — a masking body materialises it,
    /// GEMM/PV ignore it, so their emitted IR stays byte-identical.
    block: BlockCounter,
    commit_drain: CommitDrain,
    /// Gather hooks emit waitcnt-opaque `ds_read_b64`; explicit readiness is therefore mandatory.
    asm_gather: bool,
    /// The **HK bare-seal policy** (§5c): when set, cluster seals lower to a bare `s_barrier`
    /// ([`Builder::bare_barrier`]) instead of the acq-rel-fenced [`Builder::barrier`], and the LDS
    /// ordering the fence dropped is re-supplied by an explicit `s_waitcnt lgkmcnt(0)` before each
    /// compute cluster with undrained gathers (mirroring HK's 3-drains-per-K-block vs the fence's 9).
    bare_seals: bool,
    /// The **MFMA-cluster pin** (§5c, the ISA-diff fix): when set, each compute cluster's MFMA run is
    /// bracketed by a LEADING + trailing `sched.barrier(0)`, so LLVM cannot fracture the independent
    /// 32-MFMA run nor sink an `s_barrier`/`s_waitcnt lgkmcnt(0)` into the middle of it (the measured
    /// re-batch — `kernel_instr.md §2`: intrinsic MFMAs are NOT held by a single positional fence).
    pin_mfma: bool,
    /// **Seals carry the wave phase** (`WaveTopology::PingPong`, the GEMM path): the compute-cluster
    /// seals ARE the wave-phase carriers — the eq-offset warp-row pair rendezvous at each `s_barrier`, so
    /// the seal MUST stay a real workgroup barrier. `false` (FA: disjoint-Q warps, `WaveTopology::Disjoint`):
    /// the compute clusters exchange ONLY per-warp registers (V is gathered to VGPRs in the Mem cluster;
    /// the softmax reduce is a per-warp `ds_bpermute`; PV never touches LDS), so a workgroup barrier at
    /// a compute seal guards no cross-warp state AND walls the 0-MFMA softmax shadow — its seal drops to
    /// a pure ordering combine (no `s_barrier`). See [`Self::compute`]. The load-bearing Mem-cluster
    /// WAR/RAW seals ([`Self::commit`]) are UNCHANGED either way — they guard real shared-LDS traffic.
    seals_carry_phase: bool,
    // ── carries (persist across clusters within one body) ──
    entry: Vec<Edge>,
    /// Per-slot source of the NEXT read this body pass: `Some(store)` = the last cluster that wrote the
    /// slot (its store is the intra-iteration RAW), `None` = not yet written (read from the slot's
    /// carry-in). Replaces the old `first_compute` bool + full `prev_store`: with per-cluster write
    /// subsets each slot is threaded independently, so a cluster that skips a slot leaves its source
    /// intact (no dead round-trip). The first read of a carried slot (`None`) uses `carry[s]`.
    slot_src: Vec<Option<Edge>>,
    all_gathers: Vec<Edge>,
    operands: Vec<Option<(H::Op, Edge)>>,
    reg: Option<H::Reg>,
    committed: bool,
    raw_next: Option<Edge>,
    tail_barrier: Option<Edge>,
    /// Bare-seal drain bookkeeping: the set of gathered slices whose `ds_read`s are not yet covered by
    /// an `lgkmcnt(0)` drain. A compute over slice `s` drains ONLY if `s` is outstanding (its own
    /// operand), then clears ALL (the unified `lgkmcnt(0)` completes every read); a commit's read-drain
    /// clears it too. Per-slice (not a bool) so C5 — whose slice 2 was already drained at C3 while a
    /// later-gathered slice 3 is still outstanding — does NOT stall on a spurious drain (HK's C5 has
    /// none: it drains at C1/C3/C6 only). Cleared each body pass.
    undrained: Vec<usize>,
    /// Opaque gather stores not yet covered by a queue-wide `lgkmcnt(0)` readiness wait.
    undrained_reads: Vec<Edge>,
    // ── per-cluster (reset by the driver before each cluster) ──
    this_gathers: Vec<Edge>,
    sealed: bool,
}

/// True if `target` is in `from`'s dependency cone (short-circuiting DFS) — the WAR guard's
/// "does this new slot value already carry that slot's read?" test in [`ClusterCx::compute`].
fn depends_on(ir: &TileIr, from: TileId, target: TileId) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![from];
    while let Some(id) = stack.pop() {
        if id == target {
            return true;
        }
        if !seen.insert(id) {
            continue;
        }
        for c in TileIr::children(ir.node(id)) {
            stack.push(c);
        }
    }
    false
}

impl<H: Hooks> ClusterCx<'_, H> {
    fn ready_opaque_gathers(&mut self) {
        if !self.asm_gather || self.undrained_reads.is_empty() {
            return;
        }
        let last = self.undrained_reads[self.undrained_reads.len() - 1];
        let covered = self.b.combine(Effect(last.raw()), &self.undrained_reads[..self.undrained_reads.len() - 1]);
        let ready = self.b.swait_lgkmcnt(covered.dep());
        for operand in &mut self.operands {
            if let Some((op, anchor)) = operand.take() {
                *operand = Some((self.hooks.ready_after_lgkm(self.b, op, ready.dep().raw()), anchor));
            }
        }
        self.entry.push(ready.dep());
        self.undrained_reads.clear();
        self.undrained.clear();
    }

    /// A **cluster seal** — the workgroup `s_barrier` closing a cluster. Under the bare-seal policy it
    /// is a bare `s_barrier` (no acq-rel fence, so no forced `lgkmcnt(0)` and no MFMA-overlap throttle);
    /// otherwise the acq-rel-fenced barrier. The LDS ordering a bare seal drops is re-supplied by the
    /// explicit drains in [`Self::compute`]/[`Self::commit`], so callers pass the SAME `(body, deps)`.
    fn seal(&mut self, body: Effect, deps: &[Edge]) -> Effect {
        if self.bare_seals { self.b.bare_barrier(body, deps) } else { self.b.barrier(body, deps) }
    }

    /// **Enter a memory cluster at issue-priority 0** — HK's ping-pong steering. The co-resident
    /// LOADING warp-row must yield SIMD issue to the COMPUTE row during its memory phase. But the
    /// compute cluster's trailing `set_prio(0)` ([`Self::compute`]) is anchored on the 32-MFMA result
    /// values, so when LLVM slides that run across the (independent, deferred) commit the AMDGPU
    /// setprio-merge pass drops it — the memory phase stays stuck at prio 1 and the loading wave steals
    /// SIMD issue from the compute wave (the real cause of the ~46% MFMA-util, misread as barrier-bound).
    /// A `set_prio(0)` anchored on this cluster's `entry` (the preceding barrier) and routed INTO `entry`
    /// sits inside the memory phase — not adjacent to any `set_prio(1)`, so it survives the merge — and
    /// the mem ops depending on it keep it live (no DCE). Skipped at the loop-top mem cluster (empty
    /// `entry`, no raised priority yet). Pure issue-priority hint: bit-exact.
    pub(crate) fn mem_prio0(&mut self) {
        // Under the HK bare-seal chain the steering is carried by the COMPUTE clusters' own
        // `set_prio(0)` (each now survives — routed into its bare tail barrier, [`Self::compute`]), so
        // the loop-back memory phase already runs at prio 0. The clone (`hk/gemm.rs`) emits NO
        // per-memory-cluster `set_prio(0)`; a redundant one here is a wasted issue slot in the memory
        // phase AND breaks the exact 8-`setprio 0` census. Skip it on the bare-seal path.
        if self.bare_seals || self.entry.is_empty() {
            return;
        }
        let p0 = self.b.set_prio(0, &self.entry).dep();
        self.entry.push(p0);
    }

    /// The **prefetch** safe op = the `if mc.prefetch && Some(kn)=k_next { reg = ... }` arm: stage
    /// block k+1 global→VGPR (steady only — a no-op when `k_next` is `None`, e.g. epilogue/resident).
    pub(crate) fn prefetch(&mut self, tiles: &[usize]) {
        if let Some(kn) = self.k_next {
            // Pin the loads to THIS cluster's entry (the preceding cluster's boundary barrier), so a
            // split B@C4 lands between the right MFMA clusters instead of hoisting to the loop top.
            let order = self.entry.clone();
            let mut anchors: Vec<Edge> = Vec::new();
            for &t in tiles {
                let prev = self.reg.take();
                let (reg, load_ids) = self.hooks.prefetch(self.b, kn, t, prev, &order);
                self.reg = Some(reg);
                anchors.extend(load_ids);
            }
            // The **load-pin** (HK's load→use separation): a `sched.barrier(0)` positioned right after
            // the load results forbids LLVM's MachineScheduler sinking the load down to just before the
            // commit's `s_waitcnt vmcnt(0)`. Without it the load issues ~1 MFMA before its wait (DRAM
            // latency exposed); pinned here, the following compute cluster (32 MFMAs) wedges between the
            // load and the commit, hiding the latency. Folded into `entry` so it is kept live and every
            // downstream op (this cluster's gathers, the next compute cluster) orders after the wall.
            if !anchors.is_empty() {
                let pin = self.b.sched_fence(0, &anchors).dep();
                self.entry.push(pin);
            }
        }
    }

    /// Prefetch after this memory cluster's gathers. This is the issue-ordering seam needed for
    /// `V load -> current gather -> K direct-to-LDS`: no barrier is inserted between the operations.
    pub(crate) fn prefetch_after_gathers(&mut self, tiles: &[usize]) {
        if let Some(kn) = self.k_next {
            self.ready_opaque_gathers();
            let mut order = self.entry.clone();
            order.extend(&self.this_gathers);
            let mut anchors: Vec<Edge> = Vec::new();
            for &t in tiles {
                let prev = self.reg.take();
                let (reg, load_ids) = self.hooks.prefetch(self.b, kn, t, prev, &order);
                self.reg = Some(reg);
                anchors.extend(load_ids);
            }
            if !anchors.is_empty() {
                let pin = self.b.sched_fence(0, &anchors).dep();
                self.entry.push(pin);
            }
        }
    }

    /// The **gather** safe op = one iteration of the gather loop: gdeps = `seed` + `entry`; the hook
    /// gathers slice `s`; its fence tokens extend both this cluster's list and the cumulative
    /// `all_gathers` (WAR-fenced by a later commit), and its operand lands at `operands[s]`.
    pub(crate) fn gather(&mut self, s: usize) {
        let mut gdeps = self.seed.to_vec();
        gdeps.extend(&self.entry);
        let (op, g, op_anchor) = self.hooks.gather(self.b, s, self.block, &gdeps);
        self.this_gathers.extend(g.iter().copied());
        self.all_gathers.extend(g.iter().copied());
        self.operands[s] = Some((op, op_anchor));
        // A bare seal will NOT drain these `ds_read`s — flag slice `s` for the compute that consumes it.
        self.undrained.push(s);
        if self.asm_gather {
            self.undrained_reads.extend(g);
        }
    }

    /// The **commit** safe op = the commit arm: WAR-fence EVERY gather so far, `commit`, RAW-fence
    /// the fill as the LDS carry-out, and SEAL the cluster (its own barrier is the boundary token).
    /// A no-op (leaving the cluster unsealed) when there is nothing to commit (`k_next`/`reg` absent).
    pub(crate) fn commit(&mut self) {
        if let Some(kn) = self.k_next {
            assert!(!self.committed, "a schedule body supports only one commit");
            let reg = self.reg.take().expect("commit requires a staged fill");
            self.committed = true;
            self.ready_opaque_gathers();
            let war = if self.commit_drain == CommitDrain::AsmDeferred {
                let last = self
                    .all_gathers
                    .len()
                    .checked_sub(1)
                    .expect("single-buffer opaque commit requires at least one preceding gather");
                // C-c: drain the outstanding asm gather READS (esp. the early-gathered slice the NEXT
                // compute cluster consumes) HERE, before the commit writes. `lgkmcnt` is a UNIFIED
                // read+write counter: if that operand read-drain instead landed AFTER the writes (to
                // feed the shadowing cluster's first MFMA), it would drain the opaque writes with it at
                // 0-MFMA shadow — killing the deferral. The reads have had the earlier compute clusters'
                // MFMAs of latency, so this drain is ~free; draining them early leaves NO read-drain
                // between the commit and the shadowing cluster → the C7-tail write-drain finally defers.
                // This is HK's WAR-guard drain (hk.dis 0x2830), and it IS the read-before-overwrite WAR.
                // Anchor the read-drain on the PRIOR COMPUTE cluster's barrier (C5's `bar5`), not the
                // gathers — so `lgkmcnt(0)` (which drains ALL outstanding reads regardless of anchor)
                // is POSITIONED after C5's 32-MFMA run, exactly as the clone (hk/gemm.rs:279
                // `s_waitcnt_lgkmcnt(bar5)`). Anchored on the gathers it could float up before C5's run.
                // The read-drain `lgkmcnt(0)` IS the WAR token (HK `gemm.rs:279-280`): it drains every
                // outstanding gather read before the commit overwrites LDS — no SEPARATE WAR `s_barrier`.
                // The clone's C6 is a SINGLE barrier (`bar6`, after the writes); the combinator used to
                // emit two (a WAR seal + the C6 seal), an extra workgroup barrier per iteration. Anchor
                // the drain on the PRIOR COMPUTE cluster's barrier (C5's `bar5` = `tail_barrier`) so the
                // `lgkmcnt(0)` is POSITIONED after C5's 32-MFMA run (the C5→C6 wall) exactly as the clone,
                // NOT on the gathers (where it could float up before C5's run). It also covers the whole
                // bare-seal read set, so the shadowing C7 needs no drain (`undrained = false`).
                let rd_anchor = self.tail_barrier.unwrap_or(self.all_gathers[last]);
                let rd = self.b.swait_lgkmcnt(rd_anchor);
                self.undrained.clear();
                vec![rd.dep()]
            } else if matches!(self.commit_drain, CommitDrain::IntrinsicNoWar | CommitDrain::AsmPublishedNoWar) {
                // Double-buffered: no read-before-overwrite hazard (disjoint parity halves), so emit NO
                // WAR seal. The gather reads are still drained before the next compute by the RAW seal
                // (a fenced barrier after the commit writes), and the RAW carry is `raw_next` below.
                Vec::new()
            } else {
                assert!(!self.all_gathers.is_empty(), "single-buffer intrinsic commit requires a preceding gather");
                let deps: Vec<Edge> = self.all_gathers[1..].to_vec();
                vec![self.seal(Effect(self.all_gathers[0].raw()), &deps).dep()]
            };
            // `fill` = the commit's write effects (intrinsic `ds_write` stores OR asm `ds_write_b64`
            // writes). The DRAIN is owned here now (not by the hook), dispatched on the policy.
            let batch = self.hooks.commit(self.b, kn, &reg, &war);
            validate_commit_policy(batch.completion, self.commit_drain, self.bare_seals);
            let fill = batch.effects;
            let fill_deps: Vec<Edge> = fill[1..].iter().map(|e| e.dep()).collect();
            match self.commit_drain {
                CommitDrain::IntrinsicAuto | CommitDrain::IntrinsicNoWar => {
                    // The compiler-visible stores: this RAW barrier auto-drains their `lgkmcnt(0)`.
                    // (IntrinsicAuto is never paired with `bare_seals`, so `seal` = the fenced barrier
                    // whose acquire IS that auto-drain — the invariant is unchanged here.) `IntrinsicNoWar`
                    // shares this RAW seal; it differs only in dropping the WAR seal above (double-buffer).
                    let rn = self.seal(fill[0], &fill_deps).dep();
                    self.raw_next = Some(rn);
                    self.tail_barrier = Some(rn);
                    self.entry = vec![rn];
                }
                CommitDrain::AsmDeferred => {
                    // C-c (HK clone chain): the C6 RAW barrier is BARE and the C6 read-drain (`rd`,
                    // anchored on C5's `bar5`) already drained every outstanding gather read via the
                    // unified `lgkmcnt` counter (which also sweeps the prior iter's opaque writes). So
                    // the opaque asm writes need NO extra drain in the loop (HK `gemm.rs:284-289`): the
                    // C6 bare seal IS the LDS-RAW carry, kept live alongside C7's tail barrier by the
                    // End-fold's `combine` — exactly HK's `raw_next = combine(bar6, [bar7])`. No deferred
                    // C7-tail drain (that extra `lgkmcnt(0)` after C7's MFMAs is what the clone omits).
                    let rn = self.seal(fill[0], &fill_deps).dep();
                    self.raw_next = Some(rn);
                    self.tail_barrier = Some(rn);
                    self.entry = vec![rn];
                }
                CommitDrain::AsmPublishedNoWar => {
                    let n = fill.len();
                    let committed =
                        self.b.combine(fill[n - 1], &fill[..n - 1].iter().map(|e| e.dep()).collect::<Vec<_>>());
                    let ready = self.b.swait_lgkmcnt(committed.dep());
                    let rn = self.seal(ready, &[committed.dep()]).dep();
                    self.raw_next = Some(rn);
                    self.tail_barrier = Some(rn);
                    self.entry = vec![rn];
                }
            }
            self.sealed = true;
        }
    }

    /// The **compute** safe op = the Compute arm: `set_prio(1)` anchored on the operand VALUE (of
    /// gathered slice `s`); the acc reads route the carry (or the prior stores) + `entry` + the prio;
    /// run the cluster's `body` (the kernel math — MFMA for matmul, softmax/PV for FA) over the
    /// operand + acc reads; store back; `set_prio(0)` on the results; the closing `s_barrier` on the
    /// LAST store. Boundary token = `[bar, prio0]`. SEALED. The `body` is edge-free — this wrapper
    /// owns the bracket + round-trip, so the compute is pluggable without a per-kernel `Hooks` method.
    fn compute(
        &mut self,
        operand: Option<usize>,
        prioritize: bool,
        reads: &[usize],
        writes: &[usize],
        body: &ComputeBody<H>,
    ) {
        assert!(!writes.is_empty(), "a compute cluster must write ≥1 slot (its seal anchors on the last store)");
        self.ready_opaque_gathers();
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
        // Bare-seal LDS drain (HK's C1/C3 `s_waitcnt lgkmcnt(0)`): a bare seal did NOT drain the prior
        // mem cluster's gather `ds_read`s, so before a compute consuming an UNDRAINED slice emit ONE
        // `lgkmcnt(0)` (it covers every outstanding read) and route it into the acc reads below, so the
        // MFMAs order after the data has arrived — then clear ALL (the unified counter drained them).
        // Drain ONLY when THIS compute's operand slice is still outstanding: HK's C5 consumes slice 2
        // (already drained at C3) so it must NOT stall on slice 3's still-in-flight reads (`gemm.rs`
        // drains at C1/C3/C6 only). Skipped when a fenced seal already drained (`!bare_seals`).
        if self.bare_seals
            && let Some(s) = operand
            && self.undrained.contains(&s)
            && let Some(&last) = self.all_gathers.last()
        {
            let drain = self.b.swait_lgkmcnt(last).dep();
            self.entry.push(drain);
            self.undrained.clear();
        }
        // `set_prio(1)` brackets ONLY the MFMA burst: anchor it on the cluster `entry` (the mem-seal
        // `s_barrier` + this cluster's `lgkmcnt(0)` drain), NOT on the operand VALUE. Anchored on the
        // operand value it floats up to the gather — right through the barrier wait and (for C7) the C6
        // commit — so the wave holds raised priority during its memory phase, starving the co-resident
        // loading wave's ping-pong partner (the priority inversion HK's `gemm.rs:114-119` fixes by
        // anchoring on `pre = [entry barrier, lgkmcnt]`). Still gated on `operand.is_some()` (an
        // operand-less compute has no MFMA burst to bracket and relies on the closing `set_prio(0)`).
        let prio1 = (operand.is_some() && prioritize).then(|| {
            let pre = self.entry.clone();
            self.b.set_prio(1, &pre).dep()
        });
        // MFMA-cluster LEADING pin (§5c ISA fix): a `sched.barrier(0)` after the cluster entry (the
        // mem-seal `s_barrier`) + the gathered operand `ds_read`s, but BEFORE the acc reads → the MFMAs
        // order after it, so LLVM can neither sink the seal barrier / `lgkmcnt(0)` down into the 32-MFMA
        // run nor hoist the run's first MFMA above the reads. Paired with the trailing pin, the run is
        // indivisible (the measured re-batch cure — a single trailing fence did NOT hold it).
        if self.pin_mfma {
            let mut anchors: Vec<Edge> = self.entry.clone();
            anchors.extend(op_anchor);
            anchors.extend(prio1);
            let lead = self.b.sched_fence(0, &anchors).dep();
            self.entry.push(lead);
        }
        // Load ONLY the DECLARED read slots (the subset — a cluster that skips a slot leaves it out of
        // its round-trip entirely, the dead-round-trip cure §3.2). Each read's source is the slot's
        // last writer this iteration (`slot_src`) or, on its first touch, its carry-in (`carry[s]`);
        // a temporary read before it is produced is an authoring bug.
        let read_vals: Vec<SlotVal> = reads
            .iter()
            .map(|&s| {
                let mut deps = match self.slot_src[s] {
                    Some(t) => vec![t],
                    None => {
                        assert!(
                            self.is_carried[s],
                            "compute reads temporary slot {s} before it is written this iteration"
                        );
                        self.carry[s].clone() // carried: [inited, range]; epilogue: [] (frag observes End)
                    }
                };
                deps.extend(&self.entry);
                if let Some(p) = prio1 {
                    deps.push(p);
                }
                self.accs[s].load_after(self.b, &deps)
            })
            .collect();
        let new = body(self.b, op_ref, &read_vals, self.block);
        assert_eq!(
            new.len(),
            writes.len(),
            "compute body returned {} values for {} declared writes",
            new.len(),
            writes.len()
        );
        let new_ids: Vec<Edge> = new.iter().map(|v| Edge::anchor(v.id())).collect();
        // WAR guard on a READ-then-INDEPENDENTLY-WRITTEN slot (FA-32's carried QKᵀ scores: the fused
        // cluster reads s(i−1) then writes the INDEPENDENT s(i) = QKᵀ). Its new value does NOT carry that
        // read in its dependency cone, so in a STRAIGHT-LINE pass (the epilogue — no loop phi to serialize
        // the fragment alloca, unlike the steady body) nothing stops the store from being emitted BEFORE
        // the read: the epilogue's own read AND any post-loop consumer of the same slot then forward from
        // the just-written value → that block's softmax reduces over the WRONG scores (its cross-lane
        // `ds_bpermute`s CSE-collapse, a ~1/nblocks error). The store must therefore happen-AFTER the read.
        // A value-level `After([read])` does NOT work: the After-simplification pass inlines a non-side-
        // effecting dep (a `Load`) to its sources, dropping the edge. So anchor the guarded store on the
        // cluster's OTHER stores (real `Store` side-effects, NOT inlined) — every read-consuming result
        // (`m`/`l`/`o`) flows through one, so the guarded store lands after the read transitively. A
        // DEPENDENT write (GEMM's `C=mma(A,B,C)`, FA's `o/m/l`) already carries its read ⇒ NOT guarded ⇒
        // stored in the first pass unchanged (byte-identical, per `test::byte_identity`).
        let guarded: Vec<bool> = writes
            .iter()
            .zip(&new)
            .map(|(&s, &v)| {
                reads.iter().position(|&r| r == s).is_some_and(|ri| !depends_on(&self.b.ir, v.id(), read_vals[ri].id()))
            })
            .collect();
        let mut stores: Vec<Option<Effect>> = vec![None; writes.len()];
        let mut anchor: Vec<Edge> = Vec::new(); // the un-guarded stores that carry the reads
        for (i, (&s, &v)) in writes.iter().zip(&new).enumerate() {
            if !guarded[i] {
                let e = self.accs[s].store(self.b, v);
                self.slot_src[s] = Some(e.dep());
                anchor.push(e.dep());
                stores[i] = Some(e);
            }
        }
        for (i, (&s, &v)) in writes.iter().zip(&new).enumerate() {
            if guarded[i] {
                let acc = if anchor.is_empty() { self.accs[s] } else { self.accs[s].after(self.b, &anchor) };
                let e = acc.store(self.b, v);
                self.slot_src[s] = Some(e.dep());
                stores[i] = Some(e);
            }
        }
        let stores: Vec<Effect> = stores.into_iter().map(|e| e.expect("every write slot stored")).collect();
        let prio0 = self.b.set_prio(0, &new_ids).dep();
        // MFMA-cluster TRAILING pin (§5c ISA fix): a `sched.barrier(0)` on ALL MFMA RESULTS, so the
        // tail `s_barrier` cannot hoist up into the run.
        let body_eff = stores[writes.len() - 1];
        let mut deps: Vec<Edge> = stores[..writes.len() - 1].iter().map(|e| e.dep()).collect();
        // Route `set_prio(0)` INTO the cluster's seal (clone `gemm.rs:134-136`): the barrier then
        // happens-after the prio-drop, so on the LAST compute cluster (C7 — whose `entry` no later
        // cluster consumes) the `s_setprio 0` is kept live by the carried tail barrier instead of
        // being DCE'd. Without this the loop-back memory phase stays at prio 1 (steering lost).
        deps.push(prio0);
        // A READ-FREE compute (FA's QKᵀ re-zeros its output, reading no slot) never threaded `entry`
        // (the prior cluster's seal) through an acc read — so fold it into THIS seal explicitly to keep
        // the cluster-boundary barrier chain ordered. (A cluster with reads carries `entry` via them.)
        if reads.is_empty() {
            deps.extend(&self.entry);
        }
        if self.pin_mfma {
            deps.push(self.b.sched_fence(0, &new_ids).dep());
        }
        // The **compute seal**, gated on `self.seals_carry_phase` (see the field doc). With ping-pong
        // ON (GEMM) it is the workgroup `s_barrier` that doubles as the wave-phase carrier — kept exactly
        // as before (byte-identical emission). With ping-pong OFF (FA) the compute clusters share no
        // cross-warp LDS state, so the `s_barrier` guards nothing and only walls the softmax-under-MFMA
        // interleave: emit a pure ordering combine (`Node::After` → `val.after(deps)`, NO instruction)
        // instead. It folds `body_eff` (the last store) + `deps` (the other stores + `set_prio(0)`) into
        // ONE token — same shape as the barrier — so `entry`/`tail_barrier` thread on unchanged and the
        // accumulator carry stays live, but LLVM is free to interleave the softmax VALU under the MFMAs.
        // The real accumulator RAW is threaded per-slot via `slot_src` (the store effect directly), NOT
        // this seal, so dropping the barrier cannot break the cross-cluster data dependency.
        let bar = if self.seals_carry_phase {
            self.seal(body_eff, &deps).dep()
        } else {
            self.b.combine(body_eff, &deps).dep()
        };
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
    #[builder(default, into)]
    prefetch: Vec<usize>,
    #[builder(default, into)]
    prefetch_after_gathers: Vec<usize>,
    #[builder(default, into)]
    gathers: Vec<usize>,
    #[builder(default)]
    commit: bool,
}

impl<H: Hooks> Cluster<H> for Mem {
    fn build(&self, cx: &mut ClusterCx<H>) {
        cx.mem_prio0(); // drop issue-priority to 0 for this memory phase (HK ping-pong steering)
        if !self.prefetch.is_empty() {
            cx.prefetch(&self.prefetch);
        }
        for &s in &self.gathers {
            cx.gather(s);
        }
        if !self.prefetch_after_gathers.is_empty() {
            cx.prefetch_after_gathers(&self.prefetch_after_gathers);
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
    reads: Vec<usize>,
    writes: Vec<usize>,
    /// Whether this cluster raises issue priority (`s_setprio(1)`) around its MFMA burst. Default `true`
    /// (GEMM, PV). A FUSED QKᵀ∥softmax cluster sets it FALSE — HK's QKᵀ cluster (Cluster 0) carries NO
    /// `s_setprio`; only its P·V cluster (Cluster 2) does — so the two co-resident waves' priority bias
    /// lives on the P·V burst alone. Also restores the `s_setprio` balance the old operand-less softmax
    /// cluster used to supply (its lone `prio0`), which `verify_v2` checks.
    prioritize: bool,
    body: Box<ComputeBody<H>>,
}

impl<H: Hooks> Compute<H> {
    /// A compute cluster over gathered `operand` (or `None`), declaring the accumulator/temporary slots
    /// it `reads` and `writes` (by index into the pipeline's slot set). The `body` receives the read
    /// slots' values (in `reads` order) and returns the write slots' new values (in `writes` order) —
    /// so it touches ONLY its declared state, over a heterogeneous ([`SlotVal`]) channel. GEMM passes
    /// `reads = writes = 0..n` (the uniform full-acc special case); FA passes asymmetric subsets.
    pub(crate) fn new(
        operand: impl Into<Option<usize>>,
        reads: impl Into<Vec<usize>>,
        writes: impl Into<Vec<usize>>,
        body: impl Fn(&mut Builder, Option<&H::Op>, &[SlotVal], BlockCounter) -> Vec<SlotVal> + 'static,
    ) -> Self {
        Compute {
            operand: operand.into(),
            reads: reads.into(),
            writes: writes.into(),
            prioritize: true,
            body: Box::new(body),
        }
    }

    /// Mark this cluster as NOT raising issue priority (see [`Compute::prioritize`]). Fluent; used by the
    /// FA fused QKᵀ∥softmax cluster.
    pub(crate) fn no_prio(mut self) -> Self {
        self.prioritize = false;
        self
    }
}

impl<H: Hooks> Cluster<H> for Compute<H> {
    fn build(&self, cx: &mut ClusterCx<H>) {
        cx.compute(self.operand, self.prioritize, &self.reads, &self.writes, self.body.as_ref());
    }
}

/// The threaded result of one body pass (steady / epilogue).
struct BodyOut {
    /// Per-slot last store this pass (`None` = never written — only valid for a temporary the epilogue
    /// happens not to touch). The End-fold reads each CARRIED slot's last store from here.
    slot_src: Vec<Option<Edge>>,
    raw_next: Option<Edge>,
    tail_barrier: Option<Edge>,
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
    accs: &[AccSlot],
    is_carried: &[bool],
    seed: &[Edge],
    carry: &[Vec<Edge>],
    k_next: Option<Idx>,
    block: BlockCounter,
    commit_drain: CommitDrain,
    asm_gather: bool,
    bare_seals: bool,
    pin_mfma: bool,
    seals_carry_phase: bool,
) -> BodyOut {
    let mut cx = ClusterCx {
        b,
        hooks,
        accs,
        carry,
        is_carried,
        seed,
        k_next,
        block,
        commit_drain,
        asm_gather,
        bare_seals,
        pin_mfma,
        seals_carry_phase,
        entry: Vec::new(),
        slot_src: vec![None; accs.len()],
        all_gathers: Vec::new(),
        operands: (0..ksteps).map(|_| None).collect(),
        reg: None,
        committed: false,
        raw_next: None,
        tail_barrier: None,
        undrained: Vec::new(),
        undrained_reads: Vec::new(),
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
            let body = Effect(cx.this_gathers[n - 1].raw());
            let deps: Vec<Edge> = cx.this_gathers[..n - 1].to_vec();
            let bar = cx.seal(body, &deps).dep();
            cx.tail_barrier = Some(bar);
            cx.entry = vec![bar];
        }
    }
    if seals_carry_phase && let Some(raw) = cx.raw_next {
        let tail = cx.tail_barrier.expect("phased streaming body must end on a barrier");
        assert!(
            raw.raw() != tail.raw() && depends_on(&cx.b.ir, tail.raw(), raw.raw()),
            "phased body tail must actually depend on an earlier publication"
        );
    }
    BodyOut { slot_src: cx.slot_src, raw_next: cx.raw_next, tail_barrier: cx.tail_barrier }
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
    accs: &'a [AccSlot],
    inited: &'a [Option<Effect>],
    topology: WaveTopology,
    asm_gather: bool,
    resident: bool,
    commit_drain: CommitDrain,
    bare_seals: bool,
    pin_mfma: bool,
    /// Optional straight-line schedule between the block-0 LDS seed and the steady loop. FA-32 uses
    /// this to compute QK(0) and stage block 1 without executing an empty softmax/PV iteration.
    warmup_clusters: Vec<Box<dyn Cluster<H>>>,
    /// Carried slots the warmup promises to seed before the rolled loop starts.
    warmup_seed_slots: Vec<usize>,
    /// Opt in to distinct warmup/steady/epilogue value scopes. Needed when post-loop dynamic masking
    /// introduces otherwise-identical address/reduction DAGs; left off for static hot paths.
    scoped_regions: bool,
    clusters: Vec<Box<dyn Cluster<H>>>,
}

/// The pipeline's scheduling policy — the per-cluster placement knobs. Grouped so `pipeline()` names them
/// instead of a positional bool sequence.
#[derive(Copy, Clone)]
pub(crate) struct Sched {
    pub asm_gather: bool,
    pub resident: bool,
    pub commit_drain: CommitDrain,
    pub bare_seals: bool,
    pub pin_mfma: bool,
}

/// The workgroup's **wave topology** — how its warp-rows relate across the compute clusters. `Disjoint`
/// warps (FA single-crew) exchange only per-warp registers, so a compute seal guards no cross-warp state
/// and drops to a pure ordering combine. `PingPong` runs an eq-offset warp-row pair that rendezvous at
/// every compute seal (GEMM, FA-fast), so the seal MUST stay a real workgroup `s_barrier` (the phase
/// carrier). `groups`/`offset` describe the crew count and the one-cluster stagger — the emission is
/// driven by the balanced eq=1/eq=0 barrier pair, so they are self-documenting metadata, not operands.
#[derive(Copy, Clone)]
pub(crate) enum WaveTopology {
    Disjoint,
    PingPong { warp_row: Idx, groups: u8, offset: u8 },
}

impl WaveTopology {
    /// Ping-pong ⇒ the compute-cluster seals ARE the wave-phase carriers (must stay real workgroup
    /// barriers, not pure ordering combines). The renamed successor of `ping_pong = warp_row.is_some()`.
    fn seals_carry_phase(self) -> bool {
        matches!(self, WaveTopology::PingPong { .. })
    }

    /// The `(eq=0, eq=1)` wave-barrier count a balanced pipeline emits: `(1, 1)` phased, `(0, 0)` disjoint.
    /// The verifier keys its deadlock check on this instead of a bare bool.
    fn barrier_census(self) -> (usize, usize) {
        match self {
            WaveTopology::PingPong { .. } => (1, 1),
            WaveTopology::Disjoint => (0, 0),
        }
    }

    /// Emit the eq=1 **stagger** barrier ordered after `after`, offsetting one warp-row by a cluster so
    /// the crews ping-pong. Returns the ordering edge to seed the loop AND the linear [`WavePhase`]
    /// witness that obliges a matching [`Self::realign`]. Only meaningful on `PingPong`.
    fn stagger(self, b: &mut Builder, after: Edge) -> (Edge, WavePhase) {
        let WaveTopology::PingPong { warp_row, groups, offset } = self else {
            unreachable!("stagger on a disjoint topology");
        };
        // A ping-pong needs ≥2 crews and a stagger of a whole (non-empty, in-range) crew — else the
        // eq-offset pair cannot rendezvous. Debug-only: this validates the topology's self-description
        // and emits no IR (byte-identity is untouched).
        debug_assert!(groups >= 2, "ping-pong needs ≥2 crews, got {groups}");
        debug_assert!((1..groups).contains(&offset), "stagger offset {offset} must lie in 1..{groups}");
        let edge = b.wave_barrier(warp_row, 1, &[after]).dep();
        (edge, WavePhase { consumed: false })
    }

    /// Consume the [`WavePhase`] witness with the matching eq=0 **realign** barrier ordered after
    /// `after` — the half that lets the offset warp-row rejoin so the workgroup does not deadlock at exit.
    fn realign(self, mut phase: WavePhase, b: &mut Builder, after: Edge) -> Edge {
        let WaveTopology::PingPong { warp_row, .. } = self else {
            unreachable!("realign on a disjoint topology");
        };
        phase.consumed = true;
        b.wave_barrier(warp_row, 0, &[after]).dep()
    }
}

/// A **linear witness** that an eq=1 wave-phase stagger was emitted and MUST be balanced by a matching
/// eq=0 realign — otherwise one warp-row waits on an `s_barrier` the other never reaches and the
/// workgroup deadlocks. `#[must_use]` catches an ignored witness at compile time; the drop bomb catches a
/// runtime drop (an authoring path that staggered but forgot to realign). Sole producer:
/// [`WaveTopology::stagger`]; sole consumer: [`WaveTopology::realign`].
#[must_use]
struct WavePhase {
    consumed: bool,
}

impl Drop for WavePhase {
    fn drop(&mut self) {
        assert!(
            self.consumed || std::thread::panicking(),
            "a wave-phase stagger (eq=1) was never realigned (eq=0) — the workgroup would deadlock"
        );
    }
}

/// Open a clustered pipeline over `hooks`. `nblocks = k/k_step ≥ 2`; `topology = PingPong` enables the
/// wave-phase ping-pong; `sched.resident` drops the steady prefetch/commit (compute-resident microkernel);
/// `sched.bare_seals` swaps the acq-rel-fenced cluster barriers for HK's bare `s_barrier` + explicit drains.
/// `inited[s] = Some(seed)` marks slot `s` CARRIED (loop-carried + End-folded); `None` a per-iteration
/// TEMPORARY (not carried — produced and consumed within one pass, e.g. FA's softmax `P`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pipeline<'a, H: Hooks>(
    b: &'a mut Builder,
    nblocks: usize,
    k_step: usize,
    ksteps: usize,
    accs: &'a [AccSlot],
    inited: &'a [Option<Effect>],
    topology: WaveTopology,
    sched: Sched,
    hooks: H,
) -> Pipeline<'a, H> {
    let Sched { asm_gather, resident, commit_drain, bare_seals, pin_mfma } = sched;
    Pipeline {
        b,
        hooks,
        nblocks,
        k_step,
        ksteps,
        accs,
        inited,
        topology,
        asm_gather,
        resident,
        commit_drain,
        bare_seals,
        pin_mfma,
        warmup_clusters: Vec::new(),
        warmup_seed_slots: Vec::new(),
        scoped_regions: false,
        clusters: Vec::new(),
    }
}

impl<'a, H: Hooks> Pipeline<'a, H> {
    /// Append a cluster to a one-time block-0 warmup schedule. A warmup stages block 1, writes any
    /// initial carried state needed by the steady schedule, then starts the rolled loop at block 1.
    /// It is currently restricted to non-resident, non-ping-pong pipelines with at least two blocks.
    pub(crate) fn warmup_cluster(mut self, c: impl Cluster<H> + 'static) -> Self {
        self.warmup_clusters.push(Box::new(c));
        self
    }

    /// Require the warmup to write a carried slot. This turns a missing warmup compute cluster into a
    /// build-time failure rather than silently falling back to the slot's algebraic seed.
    pub(crate) fn warmup_seed(mut self, slot: usize) -> Self {
        self.warmup_seed_slots.push(slot);
        self
    }

    /// Keep warmup, steady, and epilogue value DAGs lexically distinct. This is a correctness contract
    /// for kernels whose dynamic tail work can otherwise reuse loop-local expressions.
    pub(crate) fn scoped_regions(mut self) -> Self {
        self.scoped_regions = true;
        self
    }

    /// Append a cluster to the schedule (fluent). Takes any `impl Cluster<H>` (boxed internally), so
    /// the call site is `.cluster(Mem { .. })` / `.cluster(Compute::new(3, body))` — no `Box::new`.
    pub(crate) fn cluster(mut self, c: impl Cluster<H> + 'static) -> Self {
        self.clusters.push(Box::new(c));
        self
    }

    /// Emit the pipeline: prologue commit + wave-phase seed, the steady body over the schedule, the
    /// End-fold of the CARRIED slots + the epilogue body + rebalance, then the completeness check.
    /// Returns the post-loop slot set ([`AccSlot`], the scatter source — the caller unwraps the carried
    /// slots it needs). GEMM's uniform full-acc schedule emits byte-identically to the pre-refactor
    /// `pipeline_clustered`; FA's asymmetric read/write subsets drop the dead round-trips.
    pub(crate) fn build(self) -> Vec<AccSlot> {
        let Pipeline {
            b,
            mut hooks,
            nblocks,
            k_step,
            ksteps,
            accs,
            inited,
            topology,
            asm_gather,
            resident,
            commit_drain,
            bare_seals,
            pin_mfma,
            warmup_clusters,
            warmup_seed_slots,
            scoped_regions,
            clusters,
        } = self;
        assert!(nblocks >= 2, "pipeline needs nblocks ≥ 2");
        let n_slots = accs.len();
        assert_eq!(inited.len(), n_slots, "one `inited` entry per slot (Some = carried, None = temporary)");
        // A slot is CARRIED (loop-carried + End-folded) iff it was given a seed; else a TEMPORARY.
        let is_carried: Vec<bool> = inited.iter().map(|e| e.is_some()).collect();
        let carried_slots: Vec<usize> = (0..n_slots).filter(|&s| is_carried[s]).collect();
        assert!(!carried_slots.is_empty(), "a pipeline must carry ≥1 accumulator across the loop");
        // Ping-pong ⇔ the topology carries the wave phase — the signal that the compute-cluster seals
        // must stay real workgroup `s_barrier`s (the phase carriers). See `ClusterCx::seals_carry_phase`.
        let seals_carry_phase = topology.seals_carry_phase();
        // The linear witness that an eq=1 stagger was emitted and owes a matching eq=0 realign. Set at
        // whichever stagger site fires (no-warmup prologue / post-warmup), consumed at the epilogue.
        let mut phase: Option<WavePhase> = None;
        let has_warmup = !warmup_clusters.is_empty();
        let ks_c = b.idx_const(k_step as i64);
        let one = b.idx_const(1);

        // ── prologue: commit block 0; the eq=1 wave-phase barrier (ordered after the commit via the
        //    warp_row operand carrying the raw_seed edge) offsets warp-row 1 one cluster. ──
        let zero = b.idx_const(0);
        // Stage ALL operand tiles of block 0 (the prologue is off the hot loop, so load placement here
        // is irrelevant — the split-across-clusters hide only matters in the steady body).
        let mut reg0 = None;
        for t in hooks.prologue_prefetch_tiles() {
            let (reg, _load_ids) = hooks.prefetch(b, zero, t, reg0, &[]);
            reg0 = Some(reg);
        }
        let reg0 = reg0.expect("a pipeline has ≥1 prefetch tile");
        let batch0 = hooks.commit(b, zero, &reg0, &[]);
        validate_commit_policy(batch0.completion, commit_drain, bare_seals);
        let fill0 = batch0.effects;
        // The prologue's block-0 drain is one-time (outside the steady loop), so both asm policies drain
        // it EXPOSED here — the deferral is specifically about hiding the hot-loop C6 drain, not this seed.
        let raw_seed = match commit_drain {
            CommitDrain::IntrinsicAuto | CommitDrain::IntrinsicNoWar => {
                let fill0_deps: Vec<Edge> = fill0[1..].iter().map(|e| e.dep()).collect();
                b.barrier(fill0[0], &fill0_deps)
            }
            CommitDrain::AsmDeferred | CommitDrain::AsmPublishedNoWar => {
                let sw = b.swait_lgkmcnt(fill0.last().expect("asm commit emits ≥1 write").dep());
                b.barrier(sw, &[])
            }
        };
        let initial_loop_seed = match topology {
            // No-warmup ping-pong (GEMM): the eq=1 wave-phase stagger ordered after the prologue commit
            // (`raw_seed`), offsetting warp-row 1 by one cluster. With a WARMUP present the stagger is
            // DEFERRED to after the lockstep warmup (see `loop_seed` below): QK(0) and the carried-score
            // seed run in lockstep BEFORE the crews offset — HK's pre-stagger prologue. Disjoint paths are
            // unchanged. The `WavePhase` witness threads the owed eq=0 realign to the epilogue.
            WaveTopology::PingPong { .. } if !has_warmup => {
                let (edge, ph) = topology.stagger(b, raw_seed.dep());
                phase = Some(ph);
                edge
            }
            _ => raw_seed.dep(),
        };

        // Optional straight-line warmup: block 0 is already in LDS. The warmup schedule gathers it,
        // stages block 1 (`k_next = k_step`), and may seed carried compute state such as FA-32's S(0).
        // Its block-1 commit becomes the steady loop's initial LDS-RAW token. Keeping this in Pipeline
        // reuses the same movement/barrier wrappers as the steady schedule and avoids a second hand-woven
        // FA movement path.
        let warmup = if warmup_clusters.is_empty() {
            None
        } else {
            assert!(!resident, "pipeline warmup is only supported for streaming schedules");
            // Ping-pong WITH a warmup runs the warmup in LOCKSTEP (both crews, `seals_carry_phase=false` below) and
            // defers the eq=1 stagger to after it. Needs ≥3 KV blocks so ≥1 steady iteration remains to
            // carry the eq=1 barrier into the loop (a 2-block warm pipeline skips the steady body).
            assert!(!seals_carry_phase || nblocks >= 3, "phased warm pipeline needs nblocks ≥ 3");
            assert!(nblocks >= 2, "pipeline warmup needs at least two blocks");
            let warmup_carry: Vec<Vec<Edge>> =
                inited.iter().map(|e| e.map(|x| vec![x.dep()]).unwrap_or_default()).collect();
            let warm_scope = if scoped_regions { b.scope(&[initial_loop_seed]) } else { Scope::ROOT };
            let warm = run_body(
                b,
                &warmup_clusters,
                &mut hooks,
                ksteps,
                accs,
                &is_carried,
                &[initial_loop_seed],
                &warmup_carry,
                Some(ks_c), // commit block 1 while block-0 operands are already gathered
                BlockCounter::Epilogue(0, warm_scope),
                commit_drain,
                asm_gather && matches!(commit_drain, CommitDrain::AsmPublishedNoWar),
                bare_seals,
                pin_mfma,
                false, // LOCKSTEP warmup: the eq=1 stagger is emitted after it, not within (compute seals stay combines)
            );
            for &slot in &warmup_seed_slots {
                assert!(slot < n_slots, "warmup seed slot {slot} is out of range");
                assert!(is_carried[slot], "warmup seed slot {slot} must be carried");
                assert!(warm.slot_src[slot].is_some(), "warmup did not write required carried slot {slot}");
            }
            Some(warm)
        };
        let steady_blocks = nblocks - if warmup.is_some() { 2 } else { 1 };
        let loop_seed = warmup
            .as_ref()
            .map(|w| w.raw_next.expect("warmup schedule must commit block 1"))
            .unwrap_or(initial_loop_seed);
        // Ping-pong WITH a warmup: emit the eq=1 stagger barrier HERE, after the lockstep warmup, ordered
        // after block 1's publication (`loop_seed`). The crews offset by one cluster only once QK(0) and the
        // carried-score seed are done in lockstep — HK's follower-only prologue `s_barrier`. (No-warmup
        // ping-pong already emitted its eq=1 in the prologue; unphased/warmup-less paths are untouched.)
        let loop_seed = match topology {
            WaveTopology::PingPong { .. } if has_warmup => {
                let (edge, ph) = topology.stagger(b, loop_seed);
                phase = Some(ph);
                edge
            }
            _ => loop_seed,
        };

        // ── steady loop: without a warmup, process blocks 0..last-1 as before. With a warmup, QK(0)
        //    is already carried and block 1 is staged, so process current blocks 1..last-1. ──
        let (carry_out, body_raw_next, body_tail_barrier) = if steady_blocks == 0 {
            // A two-block warm pipeline has no steady iteration: warmup computed block 0 and staged
            // block 1, so transition those carried stores and the LDS publication directly into the
            // epilogue. Do not emit `Range(0)`: dead-loop cleanup cannot make loop-local operand values
            // dominate the epilogue, and relying on it produced invalid LLVM.
            let warm = warmup.as_ref().expect("only a warmup can leave zero steady blocks");
            let carried_stores: Vec<Edge> = carried_slots
                .iter()
                .map(|&s| warm.slot_src[s].unwrap_or_else(|| inited[s].expect("carried slot has a seed").dep()))
                .collect();
            let last = Effect(carried_stores[carried_stores.len() - 1].raw());
            let mut fold = carried_stores[..carried_stores.len() - 1].to_vec();
            let raw = warm.raw_next.expect("warmup schedule must commit block 1");
            let tail = warm.tail_barrier.expect("warmup schedule must end on a cluster barrier");
            fold.extend([raw, tail]);
            if asm_gather {
                fold.push(b.wall_marker().dep());
            }
            (b.combine(last, &fold), Some(raw), Some(tail))
        } else {
            let kr = if scoped_regions && let Some(warm) = &warmup {
                b.range_after(
                    steady_blocks as i64,
                    &[
                        warm.raw_next.expect("warmup schedule must commit block 1"),
                        warm.tail_barrier.expect("warmup schedule must end on a cluster barrier"),
                    ],
                )
            } else {
                b.range(steady_blocks as i64)
            };
            let tk = b.counter(kr);
            let current = if warmup.is_some() { b.idx_add(tk, one) } else { tk };
            let k_next_idx = b.idx_add(current, one);
            let k_next = b.idx_mul(k_next_idx, ks_c);
            // Per-slot carry-in: a CARRIED slot's first read routes `[seed, range]`; a TEMPORARY has none
            // (it must be produced before it is read — enforced in `compute`).
            let carry: Vec<Vec<Edge>> = (0..n_slots)
                .map(|s| {
                    inited[s]
                        .map(|e| {
                            let source = warmup.as_ref().and_then(|w| w.slot_src[s]).unwrap_or(e.dep());
                            vec![source, kr.dep()]
                        })
                        .unwrap_or_default()
                })
                .collect();
            // Compute-resident: the whole tile is staged ONCE in the prologue, so the steady loop drops
            // BOTH prefetch and commit (`k_next=None`); the gathers still fire, re-reading the resident
            // block via `[loop_seed, kr]`.
            let steady_k_next = if resident { None } else { Some(k_next) };
            let steady_scope = if scoped_regions { b.scope(&[kr.dep()]) } else { Scope::ROOT };
            let body = run_body(
                b,
                &clusters,
                &mut hooks,
                ksteps,
                accs,
                &is_carried,
                &[loop_seed, kr.dep()],
                &carry,
                steady_k_next,
                BlockCounter::Steady(current, steady_scope), // warmup shifts the current block from 0 to 1
                commit_drain,
                asm_gather && matches!(commit_drain, CommitDrain::AsmPublishedNoWar),
                bare_seals,
                pin_mfma,
                seals_carry_phase,
            );

            // Fold every CARRIED slot's last store, raw_next, and the final cluster barrier under one End.
            let carried_stores: Vec<Edge> = carried_slots
                .iter()
                .map(|&s| body.slot_src[s].expect("carried slot must be written every iteration (the loop carry)"))
                .collect();
            let last = Effect(carried_stores[carried_stores.len() - 1].raw());
            let mut fold: Vec<Edge> = carried_stores[..carried_stores.len() - 1].to_vec();
            match body.raw_next {
                Some(rn) => fold.push(rn),
                None => assert!(resident, "streaming schedule must contain a commit cluster (raw_next carry)"),
            }
            fold.push(body.tail_barrier.expect("steady body must end on a cluster barrier"));
            // HK positional wall lattice: the `sched.barrier(0)` paired with every `s_barrier` pins the
            // opaque asm `ds_read_b64`s inside their cluster (load-bearing for the asm gather's correctness).
            if asm_gather {
                fold.push(b.wall_marker().dep());
            }
            let combined = b.combine(last, &fold);
            (b.end(combined, &[kr]), body.raw_next, body.tail_barrier)
        };
        // Every slot's post-pipeline handle observes either the loop End or the direct no-steady
        // transition. Carried slots feed the epilogue; temporaries receive a fresh region handle.
        let acc_loop: Vec<AccSlot> = accs.iter().map(|a| a.after(b, &[carry_out.dep()])).collect();

        // ── epilogue: the same schedule for the LAST block (via the End's carried RAW), no
        //    prefetch/commit; then the eq=0 wave-phase barrier rebalances warp-row 0. ──
        let ep_carry: Vec<Vec<Edge>> = (0..n_slots).map(|_| Vec::new()).collect();
        let ep_scope = if scoped_regions { b.scope(&[carry_out.dep()]) } else { Scope::ROOT };
        let ep = run_body(
            b,
            &clusters,
            &mut hooks,
            ksteps,
            &acc_loop,
            &is_carried,
            &[carry_out.dep()],
            &ep_carry,
            None,
            // The epilogue processes block `nblocks-1` (the last KV block — the only one that can be ragged).
            BlockCounter::Epilogue((nblocks - 1) as i64, ep_scope),
            commit_drain,
            asm_gather && matches!(commit_drain, CommitDrain::AsmPublishedNoWar),
            bare_seals,
            pin_mfma,
            seals_carry_phase,
        );
        // Consume the `WavePhase` witness with the matching eq=0 realign (or `None` on a disjoint topology
        // whose witness was never minted). `take` leaves `phase = None`, so its drop is a clean no-op.
        let scatter_seed = phase.take().map(|ph| {
            let anchor = ep.tail_barrier.expect("epilogue must end on a cluster barrier");
            topology.realign(ph, b, anchor)
        });
        let out: Vec<AccSlot> = acc_loop
            .iter()
            .enumerate()
            .map(|(s, a)| {
                let mut deps: Vec<Edge> = ep.slot_src[s].into_iter().collect();
                deps.extend(scatter_seed);
                a.after(b, &deps)
            })
            .collect();

        // ── completeness check: carry-completeness (a carried slot unwritten in a pass panics the
        //    End-fold above) + the wave-phase balance over the emitted output cone. A build-time panic. ──
        let roots: Vec<TileId> = out.iter().map(|a| a.id()).collect();
        verify(
            &b.ir,
            &roots,
            body_raw_next.map(|e| e.raw()),
            body_tail_barrier.map(|e| e.raw()),
            resident,
            topology.barrier_census(),
        );
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
    expected_census: (usize, usize),
) {
    assert!(raw_next.is_some() || resident, "streaming schedule must contain a commit cluster (raw_next carry)");
    assert!(tail_barrier.is_some(), "steady body must end on a cluster barrier");
    let mut reach: HashSet<TileId> = HashSet::new();
    for &r in roots {
        reach.extend(crate::passes::reachable(ir, r));
    }
    if let Some(raw) = raw_next {
        assert!(reach.contains(&raw), "publication raw_next is not reachable from pipeline outputs");
    }
    let tail = tail_barrier.expect("checked above");
    assert!(reach.contains(&tail), "tail_barrier is not reachable from pipeline outputs");
    let count = |want: i64| {
        reach.iter().filter(|&&id| matches!(ir.node(id), Node::WaveBarrier { eq, .. } if *eq == want)).count()
    };
    let (n0, n1) = (count(0), count(1));
    let (e0, e1) = expected_census;
    assert_eq!(
        (n0, n1),
        (e0, e1),
        "wave-phase barriers unbalanced (eq=0: {n0}, eq=1: {n1}; topology expects {e0}/{e1}) — would deadlock the workgroup"
    );
    if expected_census != (0, 0) {
        let raw = raw_next.expect("a phased streaming pipeline requires publication");
        assert!(raw != tail && depends_on(ir, tail, raw), "phased tail must depend on an earlier publication");
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::build::Builder;

    struct NoGatherHooks {
        out: crate::build::Buf<F32>,
        zero: Idx,
    }

    impl Hooks for NoGatherHooks {
        type Op = ();
        type Reg = Val<F32>;

        const PREFETCH_TILES: usize = 1;

        fn prefetch(
            &mut self,
            b: &mut Builder,
            _k_base: Idx,
            _tile: usize,
            prev: Option<Self::Reg>,
            _order: &[Edge],
        ) -> (Self::Reg, Vec<Edge>) {
            let value = prev.unwrap_or_else(|| b.f32(1.0));
            (value, vec![Edge::anchor(value.id)])
        }

        fn commit(&mut self, b: &mut Builder, _k_base: Idx, reg: &Self::Reg, war: &[Edge]) -> CommitBatch {
            assert!(war.is_empty(), "no-WAR test hook must not receive a gather seal");
            CommitBatch::new(vec![b.store(self.out, self.zero, *reg)], CommitCompletion::Intrinsic)
        }

        fn gather(
            &mut self,
            _b: &mut Builder,
            _slice: usize,
            _block: BlockCounter,
            _raw: &[Edge],
        ) -> (Self::Op, Vec<Edge>, Edge) {
            panic!("no-gather hook must not gather")
        }

        fn ready_after_lgkm(&mut self, _b: &mut Builder, _op: Self::Op, _wait: TileId) -> Self::Op {
            panic!("no-gather hook has no opaque readiness path")
        }
    }

    #[test]
    fn no_war_commit_accepts_an_empty_gather_set() {
        let mut b = Builder::new("empty_gather_no_war");
        let out = b.global::<F32>(1);
        let zero = b.idx_const(0);
        let frag = b.define_frag::<F32>(FragMap::gfx942_16x16(true));
        let accs = [AccSlot::F32(frag)];
        let inited = [Some(b.zero_init_frag(frag))];
        let compute = Compute::<NoGatherHooks>::new(None, vec![0], vec![0], |_b, _op, reads, _block| vec![reads[0]]);
        let final_accs = pipeline(
            &mut b,
            2,
            1,
            0,
            &accs,
            &inited,
            WaveTopology::Disjoint,
            Sched {
                asm_gather: false,
                resident: false,
                commit_drain: CommitDrain::IntrinsicNoWar,
                bare_seals: false,
                pin_mfma: false,
            },
            NoGatherHooks { out, zero },
        )
        .cluster(Mem::builder().prefetch([0]).commit(true).build())
        .cluster(compute)
        .build();
        assert_eq!(final_accs.len(), 1);
    }

    #[test]
    #[should_panic(expected = "incompatible with publication policy")]
    fn intrinsic_commit_rejects_opaque_publication_policy() {
        validate_commit_policy(CommitCompletion::Intrinsic, CommitDrain::AsmDeferred, true);
    }

    #[test]
    #[should_panic(expected = "incompatible with publication policy")]
    fn opaque_commit_rejects_intrinsic_publication_policy() {
        validate_commit_policy(CommitCompletion::Opaque, CommitDrain::IntrinsicAuto, false);
    }

    #[test]
    #[should_panic(expected = "bare seals require")]
    fn intrinsic_commit_rejects_bare_seals() {
        validate_commit_policy(CommitCompletion::Intrinsic, CommitDrain::IntrinsicAuto, true);
    }

    /// The dropped-commit-edge is now a BUILD-TIME failure: a streaming schedule whose body produced
    /// no commit (`raw_next=None`, `resident=false`) is rejected by the verifier (was a silent LDS-
    /// carry drop → device race).
    #[test]
    #[should_panic(expected = "must contain a commit cluster")]
    fn verify_rejects_streaming_without_commit() {
        let b = Builder::new("t");
        verify(&b.ir, &[], None, Some(TileId(0)), false, (0, 0));
    }

    #[test]
    #[should_panic(expected = "publication raw_next is not reachable")]
    fn verify_rejects_unreachable_publication() {
        let mut b = Builder::new("unreachable_publication");
        let root = b.idx_const(0).0;
        let raw = b.idx_const(1).0;
        let tail = b.idx_const(2).0;
        verify(&b.ir, &[root], Some(raw), Some(tail), false, (0, 0));
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
        verify(&b.ir, &[root.0], Some(e.dep().raw()), Some(wb.dep().raw()), false, (1, 1));
    }
}
