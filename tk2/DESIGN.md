# tk2 — design synthesis (v0, living, co-authored)

Status: **v0 scaffold.** Settled decisions are marked ✔; genuine forks we still owe a decision are marked **OPEN** with a recommendation. Distilled from five parallel investigations — GHC/Haskell typed-IR tradition, tinygrad, AMD Composable Kernel, ThunderKittens/HipKittens, and svod's own stack. Where a claim is load-bearing, the source(s) that independently reached it are named — several were reached by 3+ sources, which is why they're marked settled.

---

## 0. Thesis (validated by all five)

A **staged builder** (keeping TK/HK's tile-as-value vocabulary) produces a **tile-IR (data)**, not eager output. The *schedule is a program*: composable, user-orchestrated **passes** rewrite the tile-IR (tile / swizzle / pipeline / async / AGPR / pin), then a **verified lowering** emits svod's existing device-UOp → linearizer → codegen → LLVM.

The five sources don't just permit this — they *demand* it. CK's forward FMHA is 40 files / ~26,700 LOC with 19 near-copy ~1000-line pipeline bodies (each schedule variant a *forked file*, not a transform) and a bespoke Sobol/LHS sampler built solely to survive the resulting blob — "CK screaming for a real IR." HK is a compile-time IR *already*, but written in C++ template metaprogramming and evaluated eagerly, so it can never be inspected, transformed, or reused (its `d64` kernel is a fork of `d128` where **all** 20 differing lines are `vmcnt` constants). Both vendors have independently reinvented "schedule as data" and been crushed by expressing it in a language without ADTs and without staging. tk2's entire premise is doing that in Rust.

---

## 1. Architecture

Three levels; we **keep the bottom two verbatim** and build the top.

```
   typed builder  ──►  TILE-IR  ──(user-orchestrated passes)──►  scheduled TILE-IR
                          │                                              │
                          └──────────  verified lowering  ──────────────┘
                                              │
                          device-UOp  ──►  linearizer + type_verify  ──►  codegen  ──►  LLVM
                          (KEEP: existing svod stack, unchanged except one extension — §6E)
```

- **Builder** (§A): ergonomic, typed front-end. Emits interned tile-IR handles, never eager UOp.
- **Tile-IR** (§B): one hash-consed DAG that carries *both* the tile algorithm *and* the schedule as data. Passes populate the schedule fields; a verifier checks invariants between phase-bands.
- **Verified lowering** (§D): tile-IR → device-UOp, deriving a *complete, verified* ordering-edge set. A missing edge is a verifier error, never a silent miscompile.
- **Below the boundary** (svod today): the linearizer (heap-toposort over DAG edges + `CFGContext`), `type_verify`/`spec.rs`, codegen (`format!` LLVM-text, SSA-assuming), and LLVM (owns register allocation + instruction scheduling). Unchanged — except the AGPR/register-class channel needs a genuine codegen extension (§6E).

**The reshaping finding (TK/HK):** the schedule is *not* tile-level. All of HK's performance lives in the warp/cluster **pipeline** — buffer slots, `vmcnt` thresholds, ping-pong, priority windows — which sits *below* the tile op vocabulary (tile ops have no concept of "which buffer slot" or "which pipeline stage"). So the tile-IR must be **richer than tile ops**: it must carry loop / pipeline-stage / buffer-rotation / barrier / ordering structure as first-class nodes, or the passes have nothing to rewrite. (This is why §2-OPEN-1 — "one IR that grows the schedule, vs a separate schedule-IR" — is the first decision.)

---

## 2. Settled design decisions (the through-lines)

**✔ 2.1 Ordering is a first-class edge the value routes through — correctness by construction, verified complete.**
*(tinygrad `AFTER` · Exo semantics-preserving passes · svod "missing edge = miscompile" · CK "re-verify cross-representation invariants after every rewrite")*
This is the direct cure for the #1 pain that killed the old approach twice. svod's linearizer orders *purely* by DAG edges (`linearize.rs:213-219`) — no alias analysis, no program-order fallback — so a forgotten ordering edge is a silent wrong answer. Fix: every ordering/pin/async/WAR constraint is a tile-IR edge that the dependent value is *routed through* (you cannot consume the result without consuming the edge), **and** a WAR/RAW effect-model pass emits the complete edge set, **and** a verifier proves completeness before lowering. Passes are semantics-preserving-by-construction (Exo): a legal rewrite *cannot* drop an edge, so validity is inductive (valid input + legal pass ⇒ valid output) and there is never a whole-kernel equivalence check.
*Default edge population (CuTile token trick):* thread a happens-before token through every load/store on a `&mut` tile view and leave `&`-reads free to reorder — this **auto-derives** most ordering edges from the ownership annotations we already have, so the completeness verifier checks a mostly-populated edge set for the missing cases rather than requiring every `After` hand-authored. (This is where "gentle ownership" pays a second dividend beyond single-writer safety.)

**✔ 2.2 Hard edges (correctness) vs soft priority/tags (schedule) are separate channels.**
*(tinygrad linearizer · svod confirms the linearizer is already a priority-biased toposort)*
Dependencies are hard constraints (edges); pipeline/AGPR/pin passes emit only *priorities and tags* that bias placement within the legal space. A schedule pass can make code slower, never wrong. tk2 rides the existing linearizer for this.

**✔ 2.3 Interned, hash-consed DAG from day one — with mandatory disambiguators.**
*(tinygrad interning · Haskell/Accelerate sharing-recovery · svod hash-cons obligation)*
A functional tile *tree* silently blows up exponentially and duplicates placement decisions for a reused tile (the Accelerate sharing trap). Builder emits into an arena and returns `Copy` `TileId` handles; identical nodes hash-cons. **Correctness obligation (svod):** `UOp::new` interns by structure, so structurally-identical values *collapse to one register* unless tk2 threads disambiguators (`DefineReg.id`, buffer `Unique`, `tag`). Preserve deterministic allocation order or lose compile-cache dedup.

**✔ 2.4 Layout = a typed transform-graph ADT + a const-fold pass. This pass is the addressing-VALU fix.**
*(CK crown jewel · svod's runtime `lane_rc`/875µs problem · TK/HK swizzle-as-computed-policy)*
`enum Transform { Embed, Unmerge, Merge, Xor, Pad, Replicate, Freeze, PassThrough }` over newtyped `DimId`; every length/stride is `Static(i64) | Dynamic(UOpId)`. A const-fold pass collapses all-`Static` arithmetic to immediates *before* UOp emission — turning the per-access `lane_rc` div/mod VALU (re-emitted at every site today, the ~875µs explosion) into `base + offset:immediate`. **Swizzle is just an `Xor` transform node composed by graph-append** — never a special-cased per-shape struct — and its constants are *computed* from (shape, dtype, access-pattern) bank-conflict constraints, not hand-tabled (TK/HK D4).

**✔ 2.5 Register class is an enum field, not a 2⁴ code explosion.**
*(TK/HK `art`+`+256`-bias · CK residency/reg-class enums · svod "no reg-class channel exists")*
`enum RegClass { Vgpr, Agpr }` (and `enum Residency { Reg, Lds, Global }`) as fields on tile-IR values. HK's `macros.cuh` is ~1300 lines dominated by 16-way `if constexpr` ladders emitting one asm string per {A,B,C,D}∈{v,a} combination *only because inline-asm register class is textual*; with a `RegClass` field the emitter prints `a[]`/`v[]` at render and the explosion collapses to one path. CK proves this axis must be *explicit and pinned* — leaving it to the backend gives non-deterministic wrong results across compiler versions ("hard coded here. Otherwise, it produces incorrect results").

**✔ 2.6 The pass runner is a strategy algebra with declared contracts + phase bands + fuel — not a raw `Vec<Pass>`.**
*(GHC phase-ordering · Elevate strategy combinators · Hoopl triples · nanopass identity-folder · tinygrad's folklore-order footgun)*
GHC's own authors could not hand-maintain a total pass order (their retreat: ~3 phases + activation windows); tinygrad keeps its order correct *by folklore* ("crashes without this?"). Since tk2 hands pass order to the *user*, this is a user-facing footgun unless:
- each pass declares `requires`/`ensures` predicates over tile-IR invariants, checked by the runner;
- passes live in **phase bands** (`Tiling → MemoryPlacement → Pipelining → RegAlloc`), ordered freely *within* a band, verified *between* bands;
- the runner is Elevate-style combinators (`seq`, `or_else`, `try`, `repeat_fixpoint(fuel)`, `top_down`) so a pass that doesn't apply degrades gracefully, with a **fuel cap** (GHC) so a non-terminating rewrite is a warning, not a hang;
- analysis-driven passes (liveness, register-pressure, LDS-lifetime, redundant-`ds_read`) are Hoopl **lattice × transfer × rewrite** triples with a shared fixpoint driver → free termination from a finite-height lattice;
- every pass is a **nanopass identity-default folder** (override only the arms it touches) so adding a node variant doesn't break every pass — the direct cure for "lore accretes linearly."

**✔ 2.7 Verification is a staged spec-as-matcher; the decidability line is explicit.**
*(tinygrad `type_verify` · svod `spec.rs` is the concrete target · Haskell decidability split · CK cross-representation desync)*
tk2's "verified lowering" targets svod's `spec.rs`/`type_verify` (movement ops + `PtrCat`/`Cat` lowered away, integer addresses, matched ALU dtypes, one `Range` per `End`). Two *separate* verification axes:
- **Resource legality** (LDS bytes, VGPR/AGPR count, occupancy) — decidable; build it; it can only fail *after* placement, so some pass orders are discovered illegal late (plan the error UX).
- **Semantic equivalence** — undecidable; do **not** build a whole-kernel checker. Exo's semantics-preserving-by-construction (§2.1) makes it *unnecessary*. Any pass that can't be semantics-preserving is flagged "unverified, benchmark-gated."
CK's recurring miscompile hazard is *two coupled representations of one fact* (pipeline depth expressed 4×; a distribution = 2 objects that must agree). Passes *introduce* desync that CK avoided by construction — so a verify pass must **re-check cross-representation invariants after every rewrite**. Verified lowering is not polish; it buys back the safety CK got free from its type system.

**✔ 2.8 Arch is a trait over the ~18% that actually varies — and it gates pass legality.**
*(TK/HK "cdna4≈udna1 82% identical" · CK arch-table · svod `ArchCaps`/`FragRole` already clean · Haskell "arch is a lattice, not a flag")*
`trait Arch { const WAVE: usize; type MatrixCore; type SwizzlePolicy; }`. The genuine per-arch variance is small and known: wave64/32 (a constant + lane-mask), MFMA-vs-WMMA (*one* op's lowering), XOR-swizzle-vs-padding (*one* policy), and wave32-only cluster helpers. Keep svod's clean `ArchCaps` + `FragRole` shape. **Critically, arch gates pass *legality*, not just codegen:** gfx1151 (RDNA) has *no AGPRs*, so the AGPR pass is *illegal* there, not merely different — each pass declares `supported_archs`, verified (a runtime-flag'd AGPR pass would silently no-op = a different silent-wrong-schedule). Lean: arch is a *value in the IR + a per-pass legality declaration*, **not** a type parameter (a type param reintroduces the monomorphization/heterogeneity tax of §OPEN-2).

**✔ 2.9 The schedule-as-data has a hard floor: instruction-mix + scheduler-opacity are empirical, not verifiable-optimal.**
*(CK `sched_group_barrier` floor · TK/HK ratio-magic + LLVM-reorders-across-fences · svod "asm-pin is load-bearing" · Haskell non-confluence/no-cost-model)*
Not everything reduces to declarative, verifiable data. The MFMA:VALU:EXP issue-mix ratios (HK's `<6,3>`, `<10,5>`; CK's hand-tuned-per-dtype `sched_group_barrier`) are *empirical* — wrong ratios lose perf, not correctness, so no pass can *verify* them; they're a **tunable policy + autotuning**, and there is *no cost model* that picks them (legal ≠ good). And scheduler-opacity is real: svod's 805-TF matmul depends on `asm sideeffect` opacity surviving LLVM's scheduler; tk2 emitting UOp→LLVM inherits this — a carefully-scheduled IR *will* be reordered by the backend unless tk2 models scheduler-opacity + precise `s_waitcnt` as a first-class attribute + emits comprehensive fences (which themselves cost issue slots). Budget for autotuning; the schedule isn't fully "yours" until you fence or bypass LLVM scheduling.

---

## 3. The tile vocabulary to keep (TK/HK)

~30 named ops, verbatim: MMA transpose variants (`mma_AB/ABt/AtB/AtBt`), elementwise maps (`exp2, log2, gelu, max, add, mul, …`), row/col reductions + broadcast duals (`row_max/sum`, `broadcast_col`, …), `transpose/copy/swap_layout`, masking (`make_causal, tril, right_fill`), space-to-space load/store. Warp-as-unit with a warpgroup tier. Hardware `exp2` (`v_exp_f32`) as a first-class lowered op (the +34% FA lever). **Drop** the `ducks::` tag-kind system (→ Rust sealed traits), the `_s`/`_l` alias soup (→ enum values), the `-1`=runtime-dim sentinel (→ `enum Dim{Static,Dynamic}`).

---

## 4. Per-kernel lore → reusable pass (svod, the payoff table)

| Today: hand-written per kernel | tk2 pass |
|---|---|
| Software pipeline / double-buffer / register-staged prefetch (matmul cluster schedule; FA `kv_idx%2` ping-pong) | **Pipeline{stages}** pass — generates prologue/steady/epilogue from a logical body (HK writes the FA epilogue by hand *twice*) |
| Async-load completion (`vmcnt(N)` hand-chosen) | **async-wait pass** — derives counter thresholds from pipeline depth |
| SGPR descriptor hoist + LDS-cursor bump (HK's `/* Readfirstlane hoisting */`) | **scalarization/LICM pass** |
| XOR bank swizzle (two hand-tuned code paths coaxing `ds_read offset:` folds) | **layout attribute + address-lowering pass** (§2.4) |
| Unroll (`unroll: Cell<bool>` forking every primitive) | **unroll pass** on loop-IR (one op def, not two) |
| asm-pin (scheduler-opaque MFMA, `s_waitcnt`/`s_setprio`/`s_barrier`) | **op attribute (`scheduler_opaque` + precise-waitcnt) + fence-insertion pass** |
| `sched_group_barrier` MFMA:VALU:EXP ratios | **scheduling policy** (tunable/searchable, §2.9) |
| Wave ping-pong (`warpid()/4` + asymmetric barriers) | **warp-specialization pass** |
| Single-END accumulator chaining | **deleted** by a loop-IR carrying multiple accumulators natively |
| Split-K (`gemm_core_splitk` copy) | **K-grid-tiling transform** |
| P→PV relayout (arch-forked reg-copy vs LDS round-trip) | **layout-unification pass** (§5) |

---

## 5. The FA keystone as a pass (CK)

CK makes the softmax-P → PV-A relayout **free** on gfx942 by dispatching QK with `TransposeC` so the QK accumulator's C-layout *equals* PV's required A-layout — softmax runs in-place and P is already in PV's A lanes; only WMMA (genuinely different C vs A layout) needs a real shuffle. tk2 models this as a **layout-unification pass**: (1) the consumer MMA imposes a required A-layout on the producer's C value; (2) the pass tries to satisfy it *for free* by choosing the producer's transpose-C decorator (emit only a cast); (3) only if the `Layout`s are structurally irreconcilable does it insert an explicit shuffle. **"Layouts equal ⇒ free" is the optimization, and where all FA perf lives** — but it must be a *structurally verified* equality with a real fallback, never CK's `static_assert`-and-hope.

---

## 6. Cross-cutting: registers, ownership, the AGPR gap

**6A — Register class + residency as enum fields on tile-IR values** (§2.5).

**6B — Ownership/borrows for single-writer & liveness — the standout Rust bet, feasible *with a caveat*.**
Model a tile-IR value as an owned handle over `(RegClass, Range)`. `&mut` = destination/single-writer (MMA `d`, map `dst`); `&` = read operand; a moved tile transfers residency; a dropped tile frees its range. Because tk2 is *staged*, the borrow checker rejects "wrote `d` while an aliasing live `a` occupies the same VGPRs" and "used a tile after its registers were reassigned" **at IR-build time** — exactly the spill/liveness class that bit us. This is Futhark's uniqueness typing, free from Rust ownership.
**The caveat (the crux, flagged by both TK/HK and Haskell):** ownership models *physical-slot exclusivity*, not *virtual-tile scheduling*. Loop-carried accumulators (FA's `o_reg`/`m`/`l`, GEMM's `d`) are `&mut` across *every* iteration while also read to rescale — naive move-semantics says "moved out mid-loop," and *rematerialization* (recompute rather than keep-live) violates "owned once." **This is the exact loop-carry-through-asm risk that killed the prior AGPR attempt (#46), resurfacing at the type level.** Decision in §OPEN-3.

**6C — The AGPR pin is a genuine below-tk codegen gap.** svod's codegen has *no* `=a` channel — `DefineReg`→`addrspace(5)` alloca, LLVM owns VGPR/AGPR, the only pin is the asm-MFMA `=v,v,v,0` (for scheduling). AGPR residency requires a **new codegen lowering channel** (asm `=a` pin as the keystone, or a codegen extension). This is the one place "keep UOp+codegen" needs real work, not just a cleaner caller. (And note the memory record: the asm `=a` pin *alone* is not sufficient — it hit a `GCNHazardRecognizer` numerical-`inf` wall; register-class control must be paired with correct hazard/`s_nop` emission.)

---

## DECISIONS (resolved 2026-07-04; OPEN-N labels kept as cross-ref anchors)

**OPEN-1 — One IR that grows the schedule, or a separate schedule-IR below tiles?**
TK/HK proves the schedule lives *below* the tile vocabulary. Two encodings:
(a) **one tile-IR** that carries loop/pipeline/buffer/barrier/ordering as first-class *nodes and data fields*, initially empty, populated by passes (GHC uniformity → reorderable passes; Trees-That-Grow *goal* via optional fields, **not** type families);
(b) **two IRs** — a tile/algorithm IR that lowers to a separate schedule/pipeline IR the passes rewrite (MLIR progressive-dialects style).
**DECIDED → (a), ONE IR.** One ADT keeps passes reorderable and a uniform driver possible; the schedule is *data on the tile-IR* added by passes, with a per-band verifier playing nanopass's grammar-conformance role. (b) risks rebuilding the linearizer one level up. Accepted risk: the tile-IR becoming a kitchen sink — the band/verifier discipline (§2.6/§2.7) is what keeps it sane, and is now mandatory, not optional.

**OPEN-2 — How much typestate (the over-typing cliff).**
Both Haskell and TK/HK independently prescribe the *same* line, so this is nearly settled — recording it as OPEN only to ratify:
- **In types** (const-generics + sealed traits, rustc-checked): tile dims & shape-match (`Tile<M,K>·Tile<K,N>→Tile<M,N>` — kills HK's 4 arithmetic asserts/mma), layout kind, legal dtype tuples (sealed trait, "no impl ⇒ no method"), residency *space*, vector-orientation duals.
- **In data + verifier**: hardware-shape instruction selection (legalization, not typing), register-range/GPR validity (*never* surface types — HK's single largest unreadability source), swizzle/capacity/bank-conflict legality, dependency-completeness, schedule legality.
Rust has *no type families*, so per-phase IR growth is **optional data fields on one type**, not associated-type phase indexing (which monomorphizes into a combinatorial pile and makes passes non-reorderable). **DECIDED → ratified as stated.**

**OPEN-3 — Ownership-for-liveness: how far do we push it?**
Given the loop-carry caveat (§6B):
(a) **ownership everywhere** (elegant, but the accumulator handle must be designed for re-borrow / index-into-a-register-file rather than owning-move, or the borrow checker becomes the enemy — and the #46 ghost is real);
(b) **ownership for the easy discipline** (single-writer, operand non-aliasing at build time) **+ a Hoopl liveness/allocation pass** for the hard part (loop-carry + rematerialization + spilling, which ownership *cannot* model — Haskell pitfall #10: ownership = physical-slot exclusivity only, virtual-tile scheduling is still classical analysis).
**DECIDED → (b), "be gentle".** Ownership only for the free/safe discipline (single-writer, operand non-aliasing, checked at IR-build time); a Hoopl liveness/allocation pass owns the hard part (loop-carry, rematerialization, spilling). Ownership does *not* model register allocation, and we do not force it to. **CuTile cross-check done → confirms, does not change.** CuTile-rs (arXiv 2606.15991) uses ownership for *exactly* the free/safe part (cross-program output disjointness, proven race-free) and does *nothing* with ownership for the loop-carried accumulator — it's a `let mut tile_z` in a `for k` loop handed to a closed backend (NVIDIA Tile IR) that owns residency/liveness/pipelining. It is living proof of our thesis that ownership models physical-slot exclusivity, not virtual-tile scheduling. It offers no cleaner accumulator primitive (no tile-lifetime/region model, no SSA-of-tiles — the paper disclaims all of it); it simply *buys* the Hoopl half from a proprietary backend. **We have no such backend, so we must build the liveness/allocation pass ourselves** — the split stands, the risk is unchanged. (Steal one trick → §2.1.)

**OPEN-4 — Ship a cost model, or punt to measured benchmarks?**
Non-confluence (Haskell) + the instruction-mix floor (CK/TK/HK) mean *legal ≠ good* and user-ordered passes have no built-in signal that a reorder helped.
**DECIDED → punt to the criterion + in-process gfx942 PMC harness** — no static cost model until we've felt the pain. But budget for autotuning of the §2.9 issue-mix policy from the start.

---

## 7. De-risking plan (before committing to the build)

The lesson of the two-kernel saga: things pass in isolation and break on the real shape. So **prove the design on the real shapes, in order:**

1. **Skeleton:** the tile-IR ADT (interned, disambiguated), the builder that emits it, the verified lowering to device-UOp, and the strategy-combinator pass runner with band/contract checking. No optimizations yet — just "naive tile-IR → correct (slow) UOp → runs."
2. **Matmul first, hold 97%.** Express the asm-pinned 805-TF matmul as *passes* (pipeline + swizzle-layout + asm-pin/fence + reg-class) over the tile-IR. If tk2 can reproduce `gemm_core_asm`'s number with the schedule as *reusable passes* rather than a bespoke hand-written kernel — and TK/HK's "the schedule isn't tile-level" means this is the real test of whether the tile-IR is expressive enough — the design holds. If not, we learn on decision OPEN-1, cheaply, without having ported FA.
3. **FA second.** The async direct-to-LDS pipeline (the biggest measured lever), the layout-unification P→PV relayout (§5), and — gated on the AGPR codegen channel + hazard fix (§6C) — the AGPR accumulator.

Only after matmul-at-97%-via-passes do we commit to the full build-out.

---

## Appendix — source-by-source top borrow / worst pitfall

- **GHC/Haskell:** *borrow* Exo semantics-preserving-by-construction passes (the ordering-bug cure); *pitfall* the over-typing cliff is a real Rust-specific tax (no type families).
- **tinygrad:** *borrow* ordering-as-`AFTER`-edge + spec-as-matcher + hard/soft ordering split; *pitfall* folklore pass order + fixed-point/derived-edge non-termination (our deadlock generalized).
- **CK:** *borrow* layout-as-transform-graph + const-fold (the addressing-VALU fix) + P→PV layout-unification; *pitfall* schedule-as-compile-time-params forces an instance-blob explosion + a sampler; two-coupled-representations desync that passes *worsen* without a re-verify.
- **TK/HK:** *borrow* the `art` register-range tile → data, reg-class enum kills the 2⁴ asm explosion, the tile vocabulary; *pitfall* the schedule isn't tile-level (needs its own altitude), loop-carried accumulators fight the borrow checker.
- **svod own stack:** *borrow/keep* the linearizer + `type_verify` + codegen + arch caps; *boundary* tk2 owns the decision & proves it, UOp mechanically realizes it; *pitfall* the linearizer can't express data-dependent ordering, `run_count`/`InScopeRanges` is the only loop-membership mechanism (LICM footgun), one-RANGE-one-END, and the wave-phase balanced-barrier-count is a deadlock-if-unbalanced correctness obligation.
- **CuTile (arXiv 2606.15991, consulted only for OPEN-3):** *borrow* derive-ordering-edges-from-mutability (a default the verifier checks, §2.1); *confirm* gentle-ownership is sound — it uses ownership solely for cross-program single-writer (proven race-free) and punts loop-carried-accumulator liveness to a closed backend, validating our ownership/Hoopl split; *vindication* CuTile's "give up SIMT control" cliff (it falls back to `unsafe` for warp-specialization / manual LDS staging — exactly where the AMD FA gap lives) confirms tk2 must keep residency/AGPR/warp-spec **first-class passes, not `unsafe`-shaped escape hatches**. Not an architectural source: it's a single-level *eager* safety front-end over a proprietary optimizer — the design tk2 is deliberately *not* building.
