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

use crate::build::{BF16, Builder, Effect, F32, Frag, Idx, Val};
use crate::ir::{Node, TileId, TileIr};

/// The commit's **drain placement policy** (DESIGN §5c) — WHERE the collaborative fill's LDS writes
/// are made visible before the next-iteration gather. [`CommitDrain::IntrinsicAuto`] is the
/// compiler-visible `ds_write` whose `lgkmcnt(0)` the C6 RAW `s_barrier` auto-drains. The asm variants
/// use the waitcnt-opaque `asm ds_write_b64` (the barrier can NOT auto-drain it): [`CommitDrain::AsmExposed`]
/// (Phase C-a) drains at C6 with an EXPOSED manual `s_waitcnt lgkmcnt(0)` (0-MFMA shadow);
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
    /// Phase C-a, retained as the A/B baseline for the deferred drain — no entry wires it (the clustered
    /// entries jumped straight to `AsmDeferred`), so it is exercised only when comparing the two policies.
    #[allow(dead_code)]
    AsmExposed,
    AsmDeferred,
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
        order: &[TileId],
    ) -> (Self::Reg, Vec<TileId>);
    /// Commit the staged registers VGPR→LDS behind `war`. Returns the store effects the RAW fences.
    fn commit(&mut self, b: &mut Builder, k_base: Idx, reg: &Self::Reg, war: &[TileId]) -> Vec<Effect>;
    /// Gather K-slice `slice` LDS→operand-frags after `raw`, for the CURRENT block `block` (its parity
    /// selects the read buffer under LDS double-buffering; single-buffered hooks ignore it). Returns the
    /// operand bundle, its store-fence tokens (the WAR consumes them), and the `op_anchor` `set_prio` uses.
    fn gather(
        &mut self,
        b: &mut Builder,
        slice: usize,
        block: BlockCounter,
        raw: &[TileId],
    ) -> (Self::Op, Vec<TileId>, TileId);
}

/// A **heterogeneous compute-channel value** (DESIGN §3.2 — gentle typing: the dtype rides as DATA, not
/// a phantom type param) so the channel can carry bf16 `P` beside f32 `o`/`m`/`l`. A cluster body reads
/// its declared slots as these and returns its declared writes as these; the wrong-dtype accessor
/// panics at BUILD time (a kernel-authoring error, not a device fault).
#[derive(Copy, Clone)]
pub(crate) enum SlotVal {
    F32(Val<F32>),
    BF16(Val<BF16>),
}

impl SlotVal {
    /// Read this channel value as f32 (panics if it is bf16 — an author dtype mismatch).
    pub(crate) fn f32(self) -> Val<F32> {
        match self {
            SlotVal::F32(v) => v,
            SlotVal::BF16(_) => panic!("compute slot value is bf16 but was read as f32"),
        }
    }
    /// Read this channel value as bf16 (panics if it is f32).
    pub(crate) fn bf16(self) -> Val<BF16> {
        match self {
            SlotVal::BF16(v) => v,
            SlotVal::F32(_) => panic!("compute slot value is f32 but was read as bf16"),
        }
    }
    fn id(self) -> TileId {
        match self {
            SlotVal::F32(v) => v.id,
            SlotVal::BF16(v) => v.id,
        }
    }
}

/// A pipeline **accumulator/temporary slot** — its register fragment, dtype+map riding as DATA (the
/// heterogeneous carry of DESIGN §3.2). Whether a slot is CARRIED (seeded + loop-carried + End-folded,
/// e.g. GEMM's C, FA's `o`/`m`/`l`) or a per-iteration TEMPORARY (no seed, not carried — produced and
/// consumed within one iteration, e.g. FA's `s`=QKᵀ scores, `p`=softmax weights) is set by the
/// pipeline's `inited` (`Some` seed ⇒ carried, `None` ⇒ temporary), NOT by the slot itself.
#[derive(Copy, Clone)]
pub(crate) enum AccSlot {
    F32(Frag<F32>),
    BF16(Frag<BF16>),
}

impl AccSlot {
    /// Unwrap the f32 fragment (panics if bf16) — the post-loop scatter source.
    pub(crate) fn f32(self) -> Frag<F32> {
        match self {
            AccSlot::F32(f) => f,
            AccSlot::BF16(_) => panic!("acc slot is bf16 but was used as f32"),
        }
    }
    fn load_after(self, b: &mut Builder, deps: &[TileId]) -> SlotVal {
        match self {
            AccSlot::F32(f) => SlotVal::F32(b.load_frag_vec_after(f, deps)),
            AccSlot::BF16(f) => SlotVal::BF16(b.load_frag_vec_after(f, deps)),
        }
    }
    fn store(self, b: &mut Builder, v: SlotVal) -> Effect {
        match (self, v) {
            (AccSlot::F32(f), SlotVal::F32(x)) => b.store_frag_vec(f, x),
            (AccSlot::BF16(f), SlotVal::BF16(x)) => b.store_frag_vec(f, x),
            _ => panic!("acc slot / channel value dtype mismatch on store"),
        }
    }
    fn after(self, b: &mut Builder, deps: &[TileId]) -> AccSlot {
        match self {
            AccSlot::F32(f) => AccSlot::F32(b.frag_after(f, deps)),
            AccSlot::BF16(f) => AccSlot::BF16(b.frag_after(f, deps)),
        }
    }
    fn id(self) -> TileId {
        match self {
            AccSlot::F32(f) => f.id,
            AccSlot::BF16(f) => f.id,
        }
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
    Steady(Idx),
    /// Epilogue: the last block index `nblocks-1`, materialised to an `idx_const` only on demand.
    Epilogue(i64),
}

impl BlockCounter {
    /// Materialise the block index as an [`Idx`]. Steady = the existing loop counter (no new node);
    /// epilogue = a fresh `idx_const(nblocks-1)` (a new node — so a masking body pays for it, an
    /// unmasked one doesn't, keeping GEMM/FA-16 byte-identical).
    pub(crate) fn idx(self, b: &mut Builder) -> Idx {
        match self {
            BlockCounter::Steady(i) => i,
            BlockCounter::Epilogue(n) => b.idx_const(n),
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
    carry: &'a [Vec<TileId>],
    /// Per-slot carried flag (`inited[s].is_some()`) — a read of a not-yet-written slot is a carry-in
    /// (carried) or an authoring bug (temporary read before produced).
    is_carried: &'a [bool],
    seed: &'a [TileId],
    k_next: Option<Idx>,
    /// The **current KV-block counter** routed into each compute `body` (see [`BlockCounter`]): the loop
    /// counter in the steady pass, `nblocks-1` in the epilogue. Lazy — a masking body materialises it,
    /// GEMM/PV ignore it, so their emitted IR stays byte-identical.
    block: BlockCounter,
    commit_drain: CommitDrain,
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
    /// **Ping-pong on** (`warp_row.is_some()`, the GEMM path): the compute-cluster seals ARE the
    /// wave-phase carriers — the eq-offset warp-row pair rendezvous at each `s_barrier`, so the seal
    /// MUST stay a real workgroup barrier. Ping-pong OFF (FA: disjoint-Q warps, `warp_row.is_none()`):
    /// the compute clusters exchange ONLY per-warp registers (V is gathered to VGPRs in the Mem cluster;
    /// the softmax reduce is a per-warp `ds_bpermute`; PV never touches LDS), so a workgroup barrier at
    /// a compute seal guards no cross-warp state AND walls the 0-MFMA softmax shadow — its seal drops to
    /// a pure ordering combine (no `s_barrier`). See [`Self::compute`]. The load-bearing Mem-cluster
    /// WAR/RAW seals ([`Self::commit`]) are UNCHANGED either way — they guard real shared-LDS traffic.
    ping_pong: bool,
    // ── carries (persist across clusters within one body) ──
    entry: Vec<TileId>,
    /// Per-slot source of the NEXT read this body pass: `Some(store)` = the last cluster that wrote the
    /// slot (its store is the intra-iteration RAW), `None` = not yet written (read from the slot's
    /// carry-in). Replaces the old `first_compute` bool + full `prev_store`: with per-cluster write
    /// subsets each slot is threaded independently, so a cluster that skips a slot leaves its source
    /// intact (no dead round-trip). The first read of a carried slot (`None`) uses `carry[s]`.
    slot_src: Vec<Option<TileId>>,
    all_gathers: Vec<TileId>,
    operands: Vec<Option<(H::Op, TileId)>>,
    reg: Option<H::Reg>,
    raw_next: Option<TileId>,
    tail_barrier: Option<TileId>,
    /// Bare-seal drain bookkeeping: the set of gathered slices whose `ds_read`s are not yet covered by
    /// an `lgkmcnt(0)` drain. A compute over slice `s` drains ONLY if `s` is outstanding (its own
    /// operand), then clears ALL (the unified `lgkmcnt(0)` completes every read); a commit's read-drain
    /// clears it too. Per-slice (not a bool) so C5 — whose slice 2 was already drained at C3 while a
    /// later-gathered slice 3 is still outstanding — does NOT stall on a spurious drain (HK's C5 has
    /// none: it drains at C1/C3/C6 only). Cleared each body pass.
    undrained: Vec<usize>,
    // ── per-cluster (reset by the driver before each cluster) ──
    this_gathers: Vec<TileId>,
    sealed: bool,
}

impl<H: Hooks> ClusterCx<'_, H> {
    /// A **cluster seal** — the workgroup `s_barrier` closing a cluster. Under the bare-seal policy it
    /// is a bare `s_barrier` (no acq-rel fence, so no forced `lgkmcnt(0)` and no MFMA-overlap throttle);
    /// otherwise the acq-rel-fenced barrier. The LDS ordering a bare seal drops is re-supplied by the
    /// explicit drains in [`Self::compute`]/[`Self::commit`], so callers pass the SAME `(body, deps)`.
    fn seal(&mut self, body: Effect, deps: &[TileId]) -> Effect {
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
            let mut anchors: Vec<TileId> = Vec::new();
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
    }

    /// The **commit** safe op = the commit arm: WAR-fence EVERY gather so far, `commit`, RAW-fence
    /// the fill as the LDS carry-out, and SEAL the cluster (its own barrier is the boundary token).
    /// A no-op (leaving the cluster unsealed) when there is nothing to commit (`k_next`/`reg` absent).
    pub(crate) fn commit(&mut self) {
        if let Some(kn) = self.k_next
            && self.reg.is_some()
        {
            let last = self.all_gathers.len() - 1;
            let war = if self.commit_drain == CommitDrain::AsmDeferred {
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
            } else if self.commit_drain == CommitDrain::IntrinsicNoWar {
                // Double-buffered: no read-before-overwrite hazard (disjoint parity halves), so emit NO
                // WAR seal. The gather reads are still drained before the next compute by the RAW seal
                // (a fenced barrier after the commit writes), and the RAW carry is `raw_next` below.
                Vec::new()
            } else {
                let deps: Vec<TileId> = self.all_gathers[1..].to_vec();
                vec![self.seal(Effect(self.all_gathers[0]), &deps).dep()]
            };
            // `fill` = the commit's write effects (intrinsic `ds_write` stores OR asm `ds_write_b64`
            // writes). The DRAIN is owned here now (not by the hook), dispatched on the policy.
            let fill = self.hooks.commit(self.b, kn, self.reg.as_ref().unwrap(), &war);
            let fill_deps: Vec<TileId> = fill[1..].iter().map(|e| e.dep()).collect();
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
                CommitDrain::AsmExposed => {
                    // C-a: the opaque asm writes need an EXPOSED manual drain here at C6 (0-MFMA shadow).
                    let sw = self.b.swait_lgkmcnt(fill.last().expect("asm commit emits ≥1 write").dep());
                    let rn = self.seal(sw, &[]).dep();
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
    pub(crate) fn compute(&mut self, operand: Option<usize>, reads: &[usize], writes: &[usize], body: &ComputeBody<H>) {
        assert!(!writes.is_empty(), "a compute cluster must write ≥1 slot (its seal anchors on the last store)");
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
        let prio1 = operand.map(|_| {
            let pre = self.entry.clone();
            self.b.set_prio(1, &pre).dep()
        });
        // MFMA-cluster LEADING pin (§5c ISA fix): a `sched.barrier(0)` after the cluster entry (the
        // mem-seal `s_barrier`) + the gathered operand `ds_read`s, but BEFORE the acc reads → the MFMAs
        // order after it, so LLVM can neither sink the seal barrier / `lgkmcnt(0)` down into the 32-MFMA
        // run nor hoist the run's first MFMA above the reads. Paired with the trailing pin, the run is
        // indivisible (the measured re-batch cure — a single trailing fence did NOT hold it).
        if self.pin_mfma {
            let mut anchors: Vec<TileId> = self.entry.clone();
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
        let new_ids: Vec<TileId> = new.iter().map(|v| v.id()).collect();
        // Store ONLY the DECLARED write slots, threading each as the next reader's RAW (`slot_src`).
        let stores: Vec<Effect> = writes
            .iter()
            .zip(&new)
            .map(|(&s, &v)| {
                let e = self.accs[s].store(self.b, v);
                self.slot_src[s] = Some(e.dep());
                e
            })
            .collect();
        let prio0 = self.b.set_prio(0, &new_ids).dep();
        // MFMA-cluster TRAILING pin (§5c ISA fix): a `sched.barrier(0)` on ALL MFMA RESULTS, so the
        // tail `s_barrier` cannot hoist up into the run.
        let body_eff = stores[writes.len() - 1];
        let mut deps: Vec<TileId> = stores[..writes.len() - 1].iter().map(|e| e.dep()).collect();
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
        // The **compute seal**, gated on ping-pong (`self.ping_pong`, see the field doc). With ping-pong
        // ON (GEMM) it is the workgroup `s_barrier` that doubles as the wave-phase carrier — kept exactly
        // as before (byte-identical emission). With ping-pong OFF (FA) the compute clusters share no
        // cross-warp LDS state, so the `s_barrier` guards nothing and only walls the softmax-under-MFMA
        // interleave: emit a pure ordering combine (`Node::After` → `val.after(deps)`, NO instruction)
        // instead. It folds `body_eff` (the last store) + `deps` (the other stores + `set_prio(0)`) into
        // ONE token — same shape as the barrier — so `entry`/`tail_barrier` thread on unchanged and the
        // accumulator carry stays live, but LLVM is free to interleave the softmax VALU under the MFMAs.
        // The real accumulator RAW is threaded per-slot via `slot_src` (the store effect directly), NOT
        // this seal, so dropping the barrier cannot break the cross-cluster data dependency.
        let bar = if self.ping_pong { self.seal(body_eff, &deps).dep() } else { self.b.combine(body_eff, &deps).dep() };
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
        Compute { operand: operand.into(), reads: reads.into(), writes: writes.into(), body: Box::new(body) }
    }
}

impl<H: Hooks> Cluster<H> for Compute<H> {
    fn build(&self, cx: &mut ClusterCx<H>) {
        cx.compute(self.operand, &self.reads, &self.writes, self.body.as_ref());
    }
}

/// The threaded result of one body pass (steady / epilogue).
struct BodyOut {
    /// Per-slot last store this pass (`None` = never written — only valid for a temporary the epilogue
    /// happens not to touch). The End-fold reads each CARRIED slot's last store from here.
    slot_src: Vec<Option<TileId>>,
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
    accs: &[AccSlot],
    is_carried: &[bool],
    seed: &[TileId],
    carry: &[Vec<TileId>],
    k_next: Option<Idx>,
    block: BlockCounter,
    commit_drain: CommitDrain,
    bare_seals: bool,
    pin_mfma: bool,
    ping_pong: bool,
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
        bare_seals,
        pin_mfma,
        ping_pong,
        entry: Vec::new(),
        slot_src: vec![None; accs.len()],
        all_gathers: Vec::new(),
        operands: (0..ksteps).map(|_| None).collect(),
        reg: None,
        raw_next: None,
        tail_barrier: None,
        undrained: Vec::new(),
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
            let body = Effect(cx.this_gathers[n - 1]);
            let deps: Vec<TileId> = cx.this_gathers[..n - 1].to_vec();
            let bar = cx.seal(body, &deps).dep();
            cx.tail_barrier = Some(bar);
            cx.entry = vec![bar];
        }
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
    warp_row: Option<Idx>,
    asm_gather: bool,
    resident: bool,
    commit_drain: CommitDrain,
    bare_seals: bool,
    pin_mfma: bool,
    clusters: Vec<Box<dyn Cluster<H>>>,
}

/// Open a clustered pipeline over `hooks`. `nblocks = k/k_step ≥ 2`; `warp_row = Some` enables the
/// wave-phase ping-pong; `resident` drops the steady prefetch/commit (compute-resident microkernel);
/// `bare_seals` swaps the acq-rel-fenced cluster barriers for HK's bare `s_barrier` + explicit drains.
/// `inited[s] = Some(seed)` marks slot `s` CARRIED (loop-carried + End-folded); `None` a per-iteration
/// TEMPORARY (not carried — produced and consumed within one pass, e.g. FA's QKᵀ scores / softmax `P`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pipeline<'a, H: Hooks>(
    b: &'a mut Builder,
    nblocks: usize,
    k_step: usize,
    ksteps: usize,
    accs: &'a [AccSlot],
    inited: &'a [Option<Effect>],
    warp_row: Option<Idx>,
    asm_gather: bool,
    resident: bool,
    commit_drain: CommitDrain,
    bare_seals: bool,
    pin_mfma: bool,
    hooks: H,
) -> Pipeline<'a, H> {
    Pipeline {
        b,
        hooks,
        nblocks,
        k_step,
        ksteps,
        accs,
        inited,
        warp_row,
        asm_gather,
        resident,
        commit_drain,
        bare_seals,
        pin_mfma,
        clusters: Vec::new(),
    }
}

impl<'a, H: Hooks> Pipeline<'a, H> {
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
            warp_row,
            asm_gather,
            resident,
            commit_drain,
            bare_seals,
            pin_mfma,
            clusters,
        } = self;
        assert!(nblocks >= 2, "pipeline needs nblocks ≥ 2");
        let n_slots = accs.len();
        assert_eq!(inited.len(), n_slots, "one `inited` entry per slot (Some = carried, None = temporary)");
        // A slot is CARRIED (loop-carried + End-folded) iff it was given a seed; else a TEMPORARY.
        let is_carried: Vec<bool> = inited.iter().map(|e| e.is_some()).collect();
        let carried_slots: Vec<usize> = (0..n_slots).filter(|&s| is_carried[s]).collect();
        assert!(!carried_slots.is_empty(), "a pipeline must carry ≥1 accumulator across the loop");
        // Ping-pong ON ⇔ a wave-phase (`warp_row`) is supplied — the signal that the compute-cluster
        // seals must stay real workgroup `s_barrier`s (the phase carriers). See `ClusterCx::ping_pong`.
        let ping_pong = warp_row.is_some();
        let ks_c = b.idx_const(k_step as i64);
        let one = b.idx_const(1);

        // ── prologue: commit block 0; the eq=1 wave-phase barrier (ordered after the commit via the
        //    warp_row operand carrying the raw_seed edge) offsets warp-row 1 one cluster. ──
        let zero = b.idx_const(0);
        // Stage ALL operand tiles of block 0 (the prologue is off the hot loop, so load placement here
        // is irrelevant — the split-across-clusters hide only matters in the steady body).
        let mut reg0 = None;
        for t in 0..H::PREFETCH_TILES {
            let (reg, _load_ids) = hooks.prefetch(b, zero, t, reg0, &[]);
            reg0 = Some(reg);
        }
        let reg0 = reg0.expect("a pipeline has ≥1 prefetch tile");
        let fill0 = hooks.commit(b, zero, &reg0, &[]);
        // The prologue's block-0 drain is one-time (outside the steady loop), so both asm policies drain
        // it EXPOSED here — the deferral is specifically about hiding the hot-loop C6 drain, not this seed.
        let raw_seed = match commit_drain {
            CommitDrain::IntrinsicAuto | CommitDrain::IntrinsicNoWar => {
                let fill0_deps: Vec<TileId> = fill0[1..].iter().map(|e| e.dep()).collect();
                b.barrier(fill0[0], &fill0_deps)
            }
            CommitDrain::AsmExposed | CommitDrain::AsmDeferred => {
                let sw = b.swait_lgkmcnt(fill0.last().expect("asm commit emits ≥1 write").dep());
                b.barrier(sw, &[])
            }
        };
        let loop_seed = match warp_row {
            // The eq=1 wave-phase barrier ordered after the prologue commit (`raw_seed`). The barrier
            // rides as an ordering-only dep (`deps[1..]`, unreferenced by the WaveBarrier template) — no
            // longer laundered through `idx_after` into the warp_row operand now that a CUSTOM accepts
            // a happens-after edge on an effect (Stage A).
            Some(wr) => b.wave_barrier(wr, 1, &[raw_seed.dep()]).dep(),
            None => raw_seed.dep(),
        };

        // ── steady loop: block k's gathers via the carried RAW; prefetch/commit block k+1. ──
        let kr = b.range((nblocks - 1) as i64);
        let tk = b.counter(kr);
        let k_next_idx = b.idx_add(tk, one);
        let k_next = b.idx_mul(k_next_idx, ks_c);
        // Per-slot carry-in: a CARRIED slot's first read routes `[seed, range]`; a TEMPORARY has none
        // (it must be produced before it is read — enforced in `compute`).
        let carry: Vec<Vec<TileId>> =
            (0..n_slots).map(|s| inited[s].map(|e| vec![e.dep(), kr.dep()]).unwrap_or_default()).collect();
        // Compute-resident: the whole tile is staged ONCE in the prologue, so the steady loop drops
        // BOTH prefetch and commit (`k_next=None`); the gathers still fire, re-reading the resident
        // block via `[loop_seed, kr]`.
        let steady_k_next = if resident { None } else { Some(k_next) };
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
            BlockCounter::Steady(tk), // current KV-block counter (steady): the live loop counter
            commit_drain,
            bare_seals,
            pin_mfma,
            ping_pong,
        );

        // ── loop close: fold every CARRIED slot's last store (its carry-out — the writers may differ per
        //    slot now, so read them from `slot_src`, not one uniform last-cluster store), raw_next (LDS
        //    carry, streaming only), AND the final cluster's barrier (else DCE drops it → unbalanced
        //    count → deadlock) under one End. Temporaries are NOT folded — each is reached transitively
        //    through the carried slot its consumer writes, so it stays live + loop-scoped. ──
        let carried_stores: Vec<TileId> = carried_slots
            .iter()
            .map(|&s| body.slot_src[s].expect("carried slot must be written every iteration (the loop carry)"))
            .collect();
        let last = Effect(carried_stores[carried_stores.len() - 1]);
        let mut fold: Vec<TileId> = carried_stores[..carried_stores.len() - 1].to_vec();
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
        let ended = b.end(combined, &[kr]);
        // Every slot's post-loop handle observes the `End`; for carried slots this is the carry-in the
        // epilogue reads, for temporaries a fresh handle the epilogue re-writes (no collision with the
        // in-loop stores).
        let acc_loop: Vec<AccSlot> = accs.iter().map(|a| a.after(b, &[ended.dep()])).collect();

        // ── epilogue: the same schedule for the LAST block (via the End's carried RAW), no
        //    prefetch/commit; then the eq=0 wave-phase barrier rebalances warp-row 0. ──
        let ep_carry: Vec<Vec<TileId>> = (0..n_slots).map(|_| Vec::new()).collect();
        let ep = run_body(
            b,
            &clusters,
            &mut hooks,
            ksteps,
            &acc_loop,
            &is_carried,
            &[ended.dep()],
            &ep_carry,
            None,
            // The epilogue processes block `nblocks-1` (the last KV block — the only one that can be ragged).
            BlockCounter::Epilogue((nblocks - 1) as i64),
            commit_drain,
            bare_seals,
            pin_mfma,
            ping_pong,
        );
        let scatter_seed = warp_row.map(|wr| {
            // The eq=0 rebalance barrier ordered after the epilogue's last cluster barrier — the barrier
            // rides as an ordering-only dep (Stage A), not laundered through `idx_after` into warp_row.
            let anchor = ep.tail_barrier.expect("epilogue must end on a cluster barrier");
            b.wave_barrier(wr, 0, &[anchor]).dep()
        });
        let out: Vec<AccSlot> = acc_loop
            .iter()
            .enumerate()
            .map(|(s, a)| {
                let mut deps: Vec<TileId> = ep.slot_src[s].into_iter().collect();
                deps.extend(scatter_seed);
                a.after(b, &deps)
            })
            .collect();

        // ── completeness check: carry-completeness (a carried slot unwritten in a pass panics the
        //    End-fold above) + the wave-phase balance over the emitted output cone. A build-time panic. ──
        let roots: Vec<TileId> = out.iter().map(|a| a.id()).collect();
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
