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

## 5b. Matmul optimization ladder — tk+HK+CK synthesis (2026-07-04, from 3 parallel teardowns)

**All three references converge on the SAME design** (independent confirmation):

| Dimension | tk `gemm_core_asm` | HK cdna3 | CK `comp_v3` | tk2 target |
|---|---|---|---|---|
| Block tile M×N×K | 256×256×64 | 256×256×64 | 256×256×64 | 256×256×64 (grow into it) |
| Warps/WG | 8 (2×4) | 8 (2×4) | 4 (2×2) | start small |
| MFMA | `16×16×16` bf16 | `16×16×16` bf16 | `32×32×8` bf16 (K-iterated) | **16×16×16** (native, matches tk2 frag) |
| Accumulator | VGPR (C-in-reg), 2×64×64/wave | VGPR, 128 f32/lane | VGPR, 256 f32/lane | VGPR (**AGPR ruled out by all 3**) |
| Reuse factor | ≈256 (tile edge) | 256 | 128 MAC/elem | ≈256 (vs naive's **0**) |
| LDS | single buf, reg-staged prefetch | single buf (64KB full), reg-prefetch d1 | `comp_v3`: 2-stage reg prefetch, 1 LDS buf | single buf first |
| Swizzle | whole-tile XOR, b64-granular | XOR `addr^(((addr%rep)>>7)<<3)`, b64 | XOR at KPack=8-elem (128-bit), inner dim `pass_through` | XOR, KPack-granular pass-through |

**The verdict on asm vs compiler (the ceiling question) — RESOLVED by CK:** CK's `comp_v3` inner loop is **fully compiler-visible** — `__builtin_amdgcn_mfma_*` intrinsics + `__builtin_amdgcn_sched_group_barrier(mask,count,0)` groups whose counts are **analytically computed** from tile geometry + a per-arch latency model (`HotLoopScheduler`, comp_v3.hpp:261-426). **No inline asm in the pipeline loop.** Asm is only two optional escape hatches: (1) **AGPR-pin** accumulators (non-default `WGAttrCtlEnum`; NOT needed — VGPR accumulators are the default and all 3 use them for the 256² matmul), (2) **direct global→LDS** `buffer_load…lds` (only the `comp_async` 3-stage memory-hiding pipeline needs it). So the ≈450→805 TF gap tk hit with asm is **NOT intrinsically asm-only**: HK reaches 763 TF and CK reaches peak with intrinsics+`sched_group_barrier`; tk fell short (446 TF) because *svod's IR presented the schedule to LLVM worse than hipcc presents HK's C++*, not because intrinsics can't express it. **The open, harness-decidable bet for tk2: emit `sched_group_barrier` with analytic counts over a compiler-visible loop and see if the cleaner tk2 IR lets LLVM reach HK/CK-level — avoiding the asm channel entirely.**

**The increment ladder (each individually harness-gated; sizes from the tk perf ladder):**
1. **LDS + block-tiling reuse** — the big structural win (memory-bound naive → compute-bound; reuse 0→256×). *Needs the LDS IR extension (below).* This is the first build.
2. **XOR LDS swizzle** at b64/b128 KPack granularity, inner (vector) dim `pass_through` so `ds_read_b64` stays coalesced — **the single biggest MFMA-util lever** (tk: bank-conflicts 2e8→0, util 33%→54%). Models §2.4 Layout-as-transform-graph: swizzle = one `Xor` transform node, const-folded.
3. **Base+offset immediate gather** (one swizzled base VGPR + `offset:` immediate per K-row) — kills register spills; the fold that makes #2 pay.
4. **`sched_group_barrier` + `s_setprio` schedule steering** (compiler-visible; the CK `HotLoopScheduler` shape) — the asm-vs-intrinsic bet. Needs tk2 to emit sched intrinsics at chosen points + (eventually) a per-arch latency model for the group counts.
5. *(deferred — needs codegen escape hatches)* **direct-to-LDS async 3-stage pipeline** (`buffer_load…lds`) for true MI300X memory hiding, and the AGPR hatch only if a future kernel needs it. **NOT double-buffering-via-2nd-LDS-buffer early** — tk + CK agree it's neutral when occupancy (2 WG/CU) already hides global latency.

**LDS IR extension (prerequisite for step 1; the immediate build):** `Residency::Lds` is a declared-but-unwired enum today. Add: an IR `DefineLocal{id,dtype,len}` node (→ `UOp::define_local(id, dtype)`, `AddrSpace::Local` ptr), a `Barrier{deps}` node (→ `UOp::barrier(deps)`), and LDS-addressed load/store (reuse `LoadGlobal`/`StoreGlobal` against an LDS buffer, or add `Load/StoreLocal`). Builder primitives to stage a strip global→LDS + a workgroup barrier. **Linearizer danger zone:** LDS store→barrier→load must carry ordering as first-class `After`/`Barrier` edges (a missing edge = silent miscompile, the exact class §2.1 targets) — this is the correctness-critical part to own, not delegate.

**The one abstraction to steal (CK):** LDS layout as a **composable coordinate-transform descriptor** consulted by BOTH the store loop and the `ds_read` loop (so they can never disagree), with the invariant that the innermost KPack (vector) dim stays `pass_through`. This is precisely §2.4; build the swizzle as a transform value, not copy-loop arithmetic.

### Step 1a shipped: LDS-staged single-accumulator matmul (device-verified; the reuse lesson, quantified)

`matmul_lds` (kernels.rs) — one 16×16 output tile per WG, but A[16,K] and B[K,16] staged into LDS **once** before the K-loop (single fill barrier, NO per-K-step refill ⇒ NO single-buffer WAR), K-loop reads fragments from LDS, single f32 accumulator. **Bit-exact on gfx942 at 32/64/128 (`max_abs_err=0`)** — isolates and proves the LDS-in-matmul machinery (fill loops + barrier + LDS fragment gather + carry) apart from the multi-accumulator/WAR complexity of 1b. Two separate single-store fill loops avoid the multi-store-per-END issue; `barrier` wraps an `End` passthrough (tk-legal). **Harness result (the point): it is NOT a perf path — n=256 ~15% slower than rolled (LDS overhead, zero reuse); n=512 ~20× slower (K=512 ⇒ 32 KB LDS ⇒ occupancy 1).** Staging the full K-strip for ONE output tile is pathological — it confirms empirically that **the stage must be amortised over a big tile (reuse), which is exactly step 1b**: stage K_STEP (≈64, small LDS) and reuse it across a BM×BN accumulator grid. 1a is the correctness stone that de-risks the machinery; 1b is where the win is. (Kept in the bench only at n=256 as the correctness gate + that datapoint.)

### Step 1b-i shipped: multi-accumulator REUSE (device-verified) — but occupancy, not reuse, is the gate

`matmul_lds_tiled(m,n,k,bm,bn)` (kernels.rs) — one bm×bn output tile per WG, an `(bm/16)×(bn/16)` grid of loop-carried accumulators, A[bm,K]+B[K,bn] staged into LDS **once** and each fragment **reused** across the tile (A-frag across all bn/16 cols, B-frag across all bm/16 rows). **Bit-exact on gfx942** across 2×2 / 4×4 / rectangular grids (`max_abs_err=0`). The multi-accumulator loop-close is `Builder::combine` — an `After` folding the other accumulators' stores into the last one's, so ONE `End` closes the RANGE around all of them **without serialising the MMAs** (cleaner than tk's acc-read chaining; each accumulator reads its final value post-loop via `.after([end])`).

**Harness verdict (the finding): 1b-i LOSES to naive despite the reuse** — n=256: 64×64 tile 32µs, 32×32 tile 28µs, vs naive **17.8µs**. PMC root cause: `waves=16` for the 64×64 tile — a bm×bn tile at n=256 yields only `(256/bm)²` workgroups (16 for 64×64) on a **304-CU** GPU ⇒ grid-starved; and the full-K stage forces 64/32 KB LDS ⇒ occupancy 1–2. **Reuse alone is NOT the win. The gate is OCCUPANCY:** the winning kernel needs (a) **K-blocking** (stage K_STEP≈64, small LDS ⇒ many WGs resident) so occupancy stays high independent of K, AND (b) **large n** (enough tiles to fill 304 CUs) — the exact regime tk's `gemm_core_asm` targets (256² block only above n≈4096; below that it uses 128² or the naive path). This matches the vendor-gap memory ("wrong kernel SIZE"). **Step 1b-ii** is therefore the real perf work: K-blocking loop + single-buffer per-K-step WAR (mirror `gemm_core` edge structure) + register-staged prefetch, then sweep tile×n on the harness. 1b-i is the correctness+machinery stone that makes 1b-ii a pure scheduling/occupancy problem, not a correctness one.

### Step 1b-ii shipped: K-blocking — the FIRST tk2 matmul to beat naive on device

`matmul_lds_kblock(m,n,k,bm,bn)` (kernels.rs) — the 1b-i tile but the A[bm,16]/B[16,bn] strips are **re-staged per K-fragment inside the K-loop** (K_STEP=16), so LDS is a tiny `(bm·16+16·bn)·2` bytes **independent of K** (bm=bn=64 ⇒ 4 KB). The single buffer is reused every iteration ⇒ **two workgroup barriers per K-block**, mirroring tk `gemm_core`: a **RAW** fence after the fill + a **WAR** fence after the LDS reads (`barrier(read0,[other reads])`), the WAR fence routed into the accumulator reads to scope it in the K-loop. Unrolled fill (no inner range ⇒ no loop-nest). **Device bit-exact incl. K=512 (32 refills)** — the two-barrier single-buffer WAR is correct across many iterations (the hardest matmul pattern, done).

**Harness verdict — kblock BEATS naive at scale (the reuse+occupancy win, measured):** N=1024 naive 72µs vs kblock 84µs (kblock still grid-starved: (1024/64)²=256 WGs < 304 CUs); **N=2048 naive 389µs vs kblock 375µs (kblock wins)**; **N=4096 naive 2753µs vs kblock 2456µs (+12%, ~56 vs 50 real TFLOP/s)**. Mechanism in the PMC: at N=4096 kblock issues **344M VALU vs naive's 1215M (3.5×less)** — reuse collapses the per-element address VALU + global traffic that saturate naive. Crossover at N≈1900 (where (N/64)² fills 304 CUs) exactly as predicted; below that naive's max parallelism wins. **This is the first tk2 matmul kernel faster than naive on real hardware — the LDS→reuse→K-blocking arc validated.** It is the naive→LDS-reuse *rung*: ~56 TF is well under tk's ~380 TF compiler-visible ceiling, headroom in the remaining levers (bigger tile + more warps for occupancy at lower N; **K_STEP=64** to amortise the 2 barriers/16-K; vectorised `ds_read_b64`; the XOR swizzle [biggest util lever]; `sched_group_barrier`). The (tile, n) crossover also means production needs the **arch/shape-dispatch** tk already has (256² only above n≈4096; naive/128² below) — a policy the harness tunes.

### LDS bank swizzle added (`matmul_lds_kblock_sw`) — correct, but NOT the current bottleneck

The HK/CK/tk **XOR bank swizzle** (`swizzle_col`, kernels.rs): for a single-subtile bf16 tile map `(row,col) → col ^ (((row%16)·cols·2 >> 7 << 3) / 2)` — **cross-checked against all three sources**: HK `st.cuh:110` `addr ^ (((addr%repeat)>>7)<<3)`, tk `swizzle.rs::tile_offset` (a faithful HK port), CK `make_xor_transform` (same XOR-row-into-bank principle, KPack-granular for its b128 reads). Applied identically on fill-store + gather-load (a bijection ⇒ numerically transparent). Needed new IR/builder index ops `Xor`/`Shr`/`Shl`. **Device bit-exact** (incl. B's whole-column swizzle at bn=64/32). **Harness: it HALVES bank conflicts (bankconf 1.40 → 0.75) exactly as designed — but perf is ~neutral (4096: 2597→2558µs, ~1.5%; mfmautil 0.162→0.164).** The lesson: **bank conflicts are not this kernel's binding constraint yet** — mfmautil sits at 16% because of the K_STEP=16 *barrier flood* (2 barriers per 16-K ⇒ ~512 syncs at K=4096) and the *scalar per-element addressing VALU*, not LDS conflicts. tk got 33%→54% from the swizzle because THEIR kernel was already vectorised + K_STEP=64, so conflicts WERE binding. Ours isn't there. **The swizzle is banked correctly (halves conflicts, verified) and will pay once the bigger bottleneck is gone. Next lever = K_STEP=64** (amortise the barriers 4×; needs an inner K-fragment loop over the 64/16=4 fragments per block) → then vectorised `ds_read_b64` fill/gather (fewer address ops) → *then* the swizzle's conflict reduction becomes visible. Ordering matters: apply the swizzle when conflicts bind, not before.

### Step 1b-ii K_STEP=64 shipped — 1.8× over naive at scale (the diagnosed barrier lever paid)

`matmul_lds_kblock_ks(m,n,k,bm,bn,k_step,swizzle)` (kernels.rs): stage a `k_step`-wide strip per outer K-block and chain `k_step/16` MFMAs per accumulator (pre-gather all `ri·ksteps` A-frags + `cj·ksteps` B-frags, so the WAR fence still routes into the accumulator reads), **amortising the two per-block barriers over `k_step/16` K-fragments** — the K_STEP=16 barrier flood (~512 syncs @K=4096) was the measured matrix-starve. Device bit-exact (K_STEP=64/32, swizzle on/off, incl. K=512). **Harness (bm=bn=64, swizzled): K_STEP=64 vs 16 — N=1024 84→85µs (neutral, grid-starved); N=2048 382→217µs (1.76×); N=4096 2451→1514µs (1.62×). vs naive at N=4096: 2777→1514µs = 1.83×, ~50→~91 real TFLOP/s. mfmautil 0.17→0.28.** The VGPR rose 96→172 (pre-gather 32 operand frags ⇒ occ ~2) and LDS 4→16 KB, but the 4× barrier cut dominates at scale. Now ~24% of tk's ~380 TF ceiling (was ~15%). **mfmautil is still only 0.28** — barriers no longer dominate, so the NEXT bottleneck (and where swizzle + the next lever bite) is the **scalar per-element addressing/`ds_read` VALU**: vectorised `ds_read_b64`/b128 fill+gather (4× fewer LDS ops + address ops; shifts the swizzle to CK's KPack-granular form) is the next lever, then re-check the swizzle (now that barriers are gone, conflicts should start to bind), then bigger tile + register-staged prefetch (occupancy at lower N). The K_STEP is now a tunable the harness sweeps (16 for grid-starved small-N, 64 for large-N).

### `.apply()` restored — swizzle is a composable layout pass, not hand-woven (the leak, fixed)

**Decision (confirmed with user): structural tiling (`bm/bn`, `k_step`) stays builder-config; the orthogonal *refinements* (swizzle, vectorize, unroll) are `.apply`-able passes.** The hand-authored kernel-per-lever was the tk leak returning; this pulls back to the DESIGN §1 model. Added **`Program::apply(self, pass) -> Self`** (kernels.rs) — the top-level fluent composition `matmul_lds_kblock_ks(cfg).apply(SwizzlePass)` (runs the pass's `requires`/`ensures` around it). The swizzle is now the first real `.apply` refinement: the base kernel emits **`Node::LdsCol{row,col,cols}`** layout-application markers (flat `col` by default) at every LDS access; **`SwizzlePass`** (passes.rs, a `Fold`) materialises each to the bank XOR `col^delta(row)`; `Transform::Xor` added to the §2.4 `Layout` ADT. The `swizzle:bool` params are gone from the kernel signatures. Bit-exact (host `swizzle_pass_materialises_the_layout`: base has `LdsCol`, `.apply` leaves none; device correctness-gated).

**Finding the `.apply` A/B made trivial (base vs `.apply(SwizzlePass)`): at K_STEP=64 the FLAT base is FASTER than swizzled** — N=1024 45.6 vs 83.8µs, N=2048 179 vs 214µs. The swizzle's per-access XOR addressing bloats VGPR 96→172 and costs more than the conflicts it removes (bankconf 6.2→0.73, yet flat wins — conflicts are NOT binding). **The swizzle is currently a pessimisation** — only visible once it's a separable pass, not woven in. It pays later (swizzle-at-KPack-granularity *with* vectorisation); for now `.apply(SwizzlePass)` is optional and the harness says leave it off. (Also note: the flat ks64 base at 45.6µs@1024 is the fastest small-N kernel yet — I'd only ever benched the *swizzled* ks64 before.) **NEXT (unchanged): vectorisation** — emit vector-index mem ops + `devectorize` (below-boundary reuse; NOT `heuristics.rs`, which is the optimizer's axis-selection tk2 replaces), swizzle-at-vector-granularity, B[N,K] transpose. Then it too is an `.apply` refinement.

### Vectorised gathers (A + B) shipped — and swizzle FLIPS from pessimisation to 2× win (the composition thesis, measured)

Both operand gathers are now ONE `<ept×bf16>` LDS vector load (`Node::LoadVecAt` → **`ds_read_b64`**) instead of `ept` scalar `ds_read`s. A (Row) already ran contiguous along its `k_step` columns; **B (Col) needed a transpose** — stage `b_smem` as `[bn, k_step]` (LDS row = N, col = K; `fill_lds_unrolled(transpose=true)` maps global K→col, N→row) so B's ept run walks the contiguous inner axis too. One `gather_frag_lds_vec` now serves both via `map.transpose` (picks which lane-coord is the fixed `outer` vs the run). **Crucially the vec gather routes its run-start through `LdsCol`** (not a flat base) so vectorisation and swizzle *compose*: the swizzle `delta` is `ept`-aligned (`>>7<<3>>1` ⇒ ×4 = ept) and the run-start is ept-aligned, so `(wc+t)^delta == (wc^delta)+t` — the b64 chunk relocates as a unit, the run stays contiguous. (This also fixed a latent break: the earlier flat A-vec gather silently mismatched the swizzled A-fill — host tests can't see it, only the device correctness gate; the `LdsCol` routing repairs it. Bit-exact now, flat + swizzle, N=64…512, K_STEP 16/64.)

**Harness (bm=bn=64, base vs `.apply(SwizzlePass)`, both vectorised, both bit-exact):**

| N | flat-vec `ks64` | swizzle-vec `sw64` | speedup |
|---|---|---|---|
| 1024 | 50.8 µs | 47.6 µs | 1.07× |
| 2048 | 257 µs | **129 µs** | **1.99×** |
| 4096 | 1949 µs (70 TF) | **921 µs (149 TF)** | **2.12×** |

**The prior finding inverts.** With *scalar* gathers, swizzle was a pessimisation (its per-access XOR VALU exceeded the conflict benefit; conflicts weren't binding). With *vectorised b64* gathers, the wide reads make bank conflicts the binding constraint, and swizzle removes them: at fixed (vectorised) state, **swizzle is ~2× faster than flat at scale**. This is the design thesis — vectorisation and swizzle *compose* — validated on device: ~149 TF at 4096, roughly 2× the prior best (~70 TF) and ~39% of tk's ~380 TF compiler-visible ceiling (was ~24%). Production config is now **vectorised + swizzled**; the flat path only wins the grid-starved small-N corner marginally. **NEXT: fill vectorisation** (the fill still emits `epl` scalar global-load + `ds_write` per lane — emit wide vector runs + add `devectorize`/`pm_render` to the lowering path to split b128), then package the whole gather+fill+swizzle bundle as a `VectorizePass` refinement; then bigger tile + register-staged prefetch for occupancy at low N, then `sched_group_barrier`.

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

---

## Implementation findings (Step 2b — first real passes; 2026-07-04)

The unroll + const-fold addressing passes landed bit-exact on gfx942 and the pass-runner + contract model (`requires`/`ensures` + phase bands) worked and caught misordering (a const-fold placed before unroll is rejected). The architecture is de-risked. Four findings from *building* it refine the design:

1. **Expansion passes are not node-local `Fold`s.** Const-fold fits the nanopass identity-default folder perfectly (local, children-first). But unroll is a *structural expansion* (one `End` → N body copies + rewiring downstream `After`-deps) and needed a bespoke clone-with-substitution driver. The `Pass` trait accommodates it (`apply` is free-form), but the nanopass folder abstraction (§2.6) only covers *local* passes. **Action:** add a shared "subgraph-clone-with-substitution" helper to the runner before more unroll-like/expansion passes (pipeline, warp-spec) arrive.

2. **Unroll over the implicit loop-carry does NOT yet generalize to multi-accumulator loops.** The carry is register+edges, not SSA, so unroll must *recognize* the carried read (the `After` whose deps include the range counter) and rewire it. This works for **one accumulator / one range per `End`** (matmul) — the pass asserts exactly that. **FA carries three (`o_reg`, `m`, `l`)**, so this unroll breaks there: it needs the Hoopl-style liveness the design defers (§OPEN-3/§6B). This is a hard **prerequisite before FA**, and it vindicates the gentle-ownership + Hoopl-liveness split (ownership modeled the carry; a *transform* over it had to reconstruct the dataflow).

3. **WAR-safety is invisible to the structural contracts.** Once flattened, the re-gathered A/B scratch fragments would be clobbered by the edge-only linearizer, so the pass mints **fresh per-copy registers** — i.e. it must reason about physical-slot exclusivity (§6B), not just structure. The `requires`/`ensures` contracts caught *nothing* here; correctness rested entirely on the device allclose. **Verification gap:** contracts don't catch WAR/aliasing miscompiles. Either build the effect-model/WAR verifier (§2.1's completeness check, extended to register aliasing) or accept benchmark-gating for this class — the latter is exactly §2.7's "semantic equivalence is not verifiable; lean on semantics-preserving-by-construction + benchmark-gate."

4. **We have no kernel-time measurement.** `launch::run` recompiles every call, so the observed `70ms→140ms` at 128³ is *compile* time (the 8×-bigger unrolled body compiles slower), not dispatch time (which is microseconds). **The first supervised perf task is a compile-once/run-many timing + in-process gfx942 PMC harness** ([[svod-gfx942-pmc-inprocess]]) — until then every perf number is garbage and no lever can be honestly gated.

## Measurement harness + first profiler-gated result (2026-07-04)

**The harness is the shipped `Tensor::prepare_with → ExecutionPlan::profile` path, NOT a custom one** (the direct `launch::run` / a hand-rolled `CompiledLaunch` was the wrong layer — it only times one kernel and reinvents the profiler). A tk2 `Program` is wrapped as an **opaque `custom_kernel` (`Op::Call`) graph-node `Tensor`** (`graph::graph_kernel`; the closure lowers the tile-IR against the supplied PARAM placeholders — `lower::lower_as_graph_node`, `Node::Global{slot}→flat_param(ph[slot])`). `opts_to_apply=Some([])` keeps the scheduler's optimizer off the hand-lowered body; the scheduler only places/orders/times it. This (a) reports true on-device time, (b) arms the in-process gfx942 PMC every lever gates on (`SVOD_PMC=1`, `SVOD_PMC_FORCE=1` on this SR-IOV VF), (c) composes multiple kernels into one plan. **Wired into criterion from day one** (`tk2/benches/{common,matmul}.rs`, a tk-free port of `tk/benches/`): each variant is **correctness-gated** (`assert_correct` allclose vs an f32 reference) *before* it is timed, so the bench doubles as the device correctness test — **no `#[ignore]` device tests, no custom harness** (matmul's ignored tests deleted; only the two skeleton kernels, which have no bench, remain).

**First result overturns the Step-2b assumption: unroll+const-fold is a REGRESSION on the naive matmul.** Device time (min), gfx942, square `M=N=K`:

| shape | rolled | unroll+fold | VGPR (rolled→fold) | VALU (rolled→fold) |
|---|---|---|---|---|
| 256³  | 17.6µs | 16.8µs | 20→48 | 321K→273K |
| 512³  | 23.7µs | **26.2µs** | 20→48 | 2.47M→2.08M |
| 1024³ | 69.5µs | **86.4µs (+24%)** | 20→48 | 19.3M→16.2M |

The const-fold **did** cut addressing VALU as designed (−16% at 1024³) — but the unroll **doubled VGPR pressure (20→48)**, and this kernel is **memory-bound** (no LDS, per-lane global gather; GB/s climbs with N, VALU is not the limiter). Trading VALU for registers on a memory-bound kernel is a net loss. **Lessons:** (i) the harness *discriminates* (distinct VGPR/VALU/time per variant) — it can gate levers; (ii) **`UnrollPass` is not a blanket win — it must be gated on the kernel being VALU/issue-bound, not memory-bound** (a first entry for the "no cost model — measure" policy, §2.9 / DECISION-4); (iii) the passes are semantics-*preserving* (allclose held at every shape) but not perf-preserving; (iv) the real matmul levers are the memory ones (LDS placement, swizzle, async direct-to-LDS), which this naive kernel hasn't got yet. Tier-2 **GFLOP/s is unreliable for hand kernels** (the AST FLOP-walker estimates differently for the rolled vs unrolled IR — 1044 vs 258 at 1024³ for the *same* math); gate on device **time + VALU/VGPR**, per the profiling doc's stated limitation.
