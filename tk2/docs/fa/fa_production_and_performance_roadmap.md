# Flash Attention Production and Performance Roadmap

## Scope

This document consolidates two reviews:

1. What tk2 is missing in its DSL, IR, pipeline, and verification layers to make a
   production Flash Attention kernel straightforward to author.
2. How to improve the gfx942 kernel without copying a HipKittens or AITER schedule.

It intentionally keeps only proposals with a credible high payoff. Small scheduler
sweeps, speculative register controls, and changes that previously regressed are not
part of the main roadmap.

The live FA-32 kernel remains the performance and correctness oracle while these
capabilities are added.

## Current Position

The current kernel in `tk2/src/kernels/fa.rs` is already a serious gfx942
implementation:

- 8 wave64 waves and 512 threads per workgroup.
- 32 query rows per wave and 256 query rows per workgroup.
- `v_mfma_f32_32x32x8_bf16` for QK and PV.
- A deliberately narrow public domain: `d in {64,128}` and sequence length exactly divisible by the
  256-row query tile. Ragged and two-KV-block shapes are qualification-only until bounded FA views land.
- Loop-invariant Q in registers.
- Register-staged global -> VGPR -> LDS movement for d64 K/V and short d128 K/V.
- Ordered direct-to-LDS K plus register-staged, write-transposed V for d128 at `S >= 1024`.
- Single-buffered K for d64 and double-buffered K for d128.
- Triple-buffered, write-transposed, pitch-36 V.
- Online softmax with carried `m`, `l`, O, and score state.
- A QK(0)-only warmup for the public tile-exact domain, plus crate-private qualification coverage of
  ragged and two-KV-block domains.
- Softmax of score block `i-1` overlapped with QK of block `i`.
- A final post-loop softmax and PV drain.
- Zero scratch for d64 and d128.
- Generated scheduler cadence, code-object ISA ordering, and resource usage covered by ignored gfx942
  qualification tests.
- d64-only XCD-local query-tile ordering for private-L2 K/V reuse.

Production remains direct K plus scalar write-transposed V and compiler-visible intrinsic gathers on
long d128, with no wave phase. Packed V, waitcnt-opaque gathers, the register-staged long-shape oracle,
and register-asm publication are `cfg(test)` crate-private qualification paths; none is a public selector
option. After the host reboot on 2026-07-18, the expanded resource/ISA, numerical, 64-replay, and exact
physical-MFMA gates all passed on GPU 0. No phase path was restored or dispatched.

Measured long-sequence performance on gfx942:

| Shape | Compiler | FA-32 throughput | Allocated VGPR | LDS | Scratch |
|---|---|---:|---:|---:|---:|
| `bh=32, S=2048, d=64` | LLVM 23 | 281.1-285.9 TF/s | 144 | 17,920 B | 0 |
| `bh=32, S=2048, d=64` | LLVM 22 | 279.4-281.7 TF/s | 136 | 17,920 B | 0 |
| `bh=32, S=2048, d=128` | LLVM 23 | 297.6-303.1 TF/s | 232 | 44,032 B | 0 |
| `bh=32, S=2048, d=128` | LLVM 22 | 305.2-311.0 TF/s | 232 | 44,032 B | 0 |

The LLVM 23 rows report the post-reboot centered `U[-1,1]` rerun through the active unqualified `clang`.
d64 measured 281.1, 285.9, 283.4, 282.4, and 281.2 TF/s (282.4 median). d128 measured 299.1, 299.3,
303.1, 297.6, and 301.2 TF/s (299.3 median), about 2.7% below the accepted pre-deadlock median. Pinning
`PATH` to the retained ROCm LLVM 22 compiler changed no source and restored d128 to 305.2, 306.3, and
311.0 TF/s (306.3 median), within the accepted range. The apparent post-reboot regression is therefore
compiler-driven, not a demonstrated DSL or schedule regression.

The accepted d128 production runs remain 304.9, 307.7, and 308.6 TF/s, versus 296.9, 299.0, and 296.9
TF/s for the locked register-staged baseline. That accepted comparison is a 3.6% median improvement.
Short d128 stays register-staged because direct K regresses `S=256`; the measured crossover is
`S >= 1024`.

### Comparable External Baseline

The relevant gfx942 AITER forward range is approximately 450-475 TF/s on randomized
inputs, not the cited 630.43 TF/s. That number is MI355X/gfx950 BWD-a32 at a causal GQA
shape; the same AITER table reports MI300X forward at 452.03 TF/s. It is also not strict
API parity: the current tk2 kernel stores F32 O, while the inspected AITER kernel stores
BF16 RTZ O and omits LSE.

Both implementations are operand-sensitive despite fixed code objects and MFMA work.
For the production tk2 long-d128 path, one controlled matrix measured:

| Input distribution | TF/s |
|---|---:|
| Centered `U[-1,1]` | 307.7 median |
| `N(0,1)` | 297.1 |
| `U[0,1]` | 313.6 |
| `N(0,1) * 10` | 312.2 |
| Zeros | 395.0 |
| Ones | 384.3 |

Zeros and ones are diagnostic compressed-data ceilings, not throughput gates. The
benchmark defaults to centered inputs and accepts `SVOD_FA_INPUT=sym|normal|normal10|u01|zeros|ones`.
FA-16 and FA-32 consume the exact same timed Q/K/V tensors for each shape. The post-reboot LLVM 23
normal-input samples measured 288.9, 295.0, and 288.7 TF/s at `bh=32,S=2048,d=128` (288.9 median, about
2.8% below the historical 297.1 point); an LLVM 22 normal-input rerun was not taken.

The main remaining problems are not one universal hot-loop defect:

| Regime | Dominant problem |
|---|---|
| Long d64 | Softmax and reduction issue cost relative to only 16 useful MFMAs per KV block |
| Long d128 | Mixed issue latency, register pressure, and weak K/V cache reuse |
| Short sequence | Fixed prologue, rotation, drain, and Q256 granularity |
| Low workgroup count | Grid starvation on a 304-CU device |
| Production inputs | Compile-time d64/d128 and Q256-exact `n`; no public tails, GQA, causal, LSE, or BF16 output |

## Executive Decision

Do not replace the current mainloop with another externally derived schedule.

Build the minimum contracts that let tk2 express and verify shape-specific attention
kernels, then optimize the current algorithm in measured vertical slices.

The combined priority is:

1. Preserve the implemented startup warmup and its resource/loop-trip gates.
2. Add runtime attention domains and bounded strided buffers.
3. Add verified circular LDS stages and typed asynchronous transfers.
4. Add causal pruning, GQA mapping, BF16 output, and LSE.
5. Explore shared-K/V execution across GQA query heads only after the production
   addressing contract exists.

## Why The Current DSL Is Difficult

The live kernel manually encodes several invariants that should be checked contracts:

- K stage selection is `% 2` and V stage selection is `% 3`.
- Delayed V consumption is represented by manual block arithmetic.
- Stage overwrite safety is inferred from cluster order and barriers.
- Score rotation is encoded through integer slot indices and carried/temporary flags.
- The final drain manually rebinds indices and LDS handles through dependency nodes to
  avoid loop-dominance and hash-consing problems.
- `Pipeline::nblocks` is a host-time integer, preventing per-workgroup causal loop bounds.
- Global safety relies on padded allocations rather than bounded loads and stores.
- Producer and consumer LDS layouts agree by convention, not by a checked layout type.
- VMEM waits are queue-wide thresholds. Their typed anchor proves issue position after a dependency
  batch, but deliberately does not claim ownership/readiness of one transfer payload.
- Scheduler hints are checked against reachable whole-kernel operations rather than a
  local schedule region.
- Phase verification now runs over the output dependency cone and proves publication/tail reachability,
  balance, and phase-carrier ordering. It still cannot prove that compiler-visible LDS scheduling stays
  race-free under asymmetric progress; consequently FA has no callable phase path.

These are DSL and verification gaps. The absence of a gfx942 transpose LDS read is a
hardware limitation and must not be hidden behind a misleading abstraction.

## Minimal Coherent DSL

The goal is not a general tensor compiler. Five focused contracts are enough.

### 1. Attention Arguments And Bounded Views - Low-Level Implemented

Add runtime scalar and buffer arguments, then expose them through a small strided view:

```rust
let args = AttentionArgs::bind(&mut b, AttentionAbi {
    q: BF16,
    k: BF16,
    v: BF16,
    o: BF16,
    lse: OptionalF32,
});

let q = args.q.view_bshd(args.q_shape, args.q_strides);
let k = args.k.view_bshd(args.k_shape, args.k_strides);
let v = args.v.view_bshd(args.v_shape, args.v_strides);
```

Implemented low-level semantics:

```rust
ScalarParam { name, min, max }
Global { slot, dtype, len }
MakeBufferRsrcDyn { buffer, base, num_bytes }
LoadGlobalBounded { buffer, offset, bound, alt }
StoreGlobalBounded { buffer, offset, bound, value }
```

`BoundedBuf` supplies gated scalar loads/stores, safe elementwise partial-vector loads,
runtime strided addressing, and a runtime-bounded modern AMD resource descriptor.
Out-of-range loads return the author-provided alternate value and out-of-range stores
are dropped. `launch::run_with_vars` validates named i32 scalar bindings, while graph
kernels reuse `ExecutionPlan::execute_with_vars`. Score masking is still required before
the online maximum, because zero-filled K rows are not equivalent to negative-infinity
scores.

The remaining work in this section is the high-level `AttentionArgs` ABI and migrating
FA's Q/K/V/O accesses from static padded buffers to these views. Native runtime f32
scalars are also deferred because the current device scalar ABI is integer-only; a
one-element buffer or a typed kernarg contract is required for runtime softmax scale.

This unlocks:

- Safe unpadded ragged batches.
- Runtime B, Sq, Sk, Hq, Hkv, and integer strides.
- GQA/MQA head addressing.
- BF16 O and optional LSE output.

Head dimension, MFMA shape, wave count, stage count, and vector width should remain
compile-time specializations.

### 2. Dynamic Iteration Domains And Attention Partitioning - Low-Level Implemented

Replace loose runtime index arithmetic with an iteration-domain contract:

```rust
let pid = AttentionPartition::new(
    args.batch,
    args.q_heads,
    args.q_len,
    Q_ROWS_PER_WORKGROUP,
).decode(&mut b);

let kv_head = pid.q_head / args.gqa_ratio();
let valid_k_end = AttentionMask::valid_k_end(args.mask, &pid, &args);
let domain = b.iter_domain(ceil_div(valid_k_end, KV_ROWS));
```

Implemented authoring representation:

```rust
pub struct IterDomain {
    pub trips: Idx,
}
```

The IR now has runtime scalar parameters, dynamic grid and range bounds, `idx_min`,
`idx_ceil_div`, and range-entry dependencies. The lower symbolic pass now preserves
native `Op::Range` predecessors. What remains is the attention-specific partition and
mask layer that derives per-workgroup causal/GQA domains from these primitives.

This unlocks the highest-value production optimization: causal work pruning. A causal
workgroup should not iterate over KV blocks that cannot contribute. For balanced causal
attention this can approach a 2x reduction in QK/PV work before overheads.

Short domains should dispatch to a dedicated one- or two-tile kernel rather than adding
general device control flow to the main long-sequence kernel.

### 3. Online Softmax And Output Epilogue

Package the existing, correct recurrence as a reusable tile operation:

```rust
let valid = AttentionMask::new()
    .kv_bounds(global_k, args.k_len)
    .causal(args.mask, global_q, global_k, args.q_len, args.k_len);

let scores = scores.mask(valid, f32::NEG_INFINITY);
let (state, p) = state.update(scores);
let state = state.accumulate_pv(p, v_fragment);
```

The helper must preserve the critical rule that invalid scores are masked before the
row maximum is updated.

The final state should own output conversion and LSE:

```rust
let state = state.finish();
state.store_o(o_view, OutputType::BF16, Bf16Rounding::Rtz);
state.store_lse_if_present(lse_view);
```

The current base-2 recurrence produces natural-log LSE as:

```text
lse = (m + log2(l)) * ln(2)
```

This requires a `Log2` operation or an explicitly verified equivalent.

### 4. Verified Circular LDS And Async Transfers - Completion Typing Implemented

Replace raw parity arithmetic with stage handles:

```rust
let k_stages = b.circular_lds::<BF16, 2>(k_layout);
let v_stages = b.circular_lds::<BF16, 3>(v_layout);

let k_read = k_stages.read(domain.current);
let k_write = k_stages.write(domain.next);
let v_read = v_stages.read(domain.current.offset(-1));
let v_write = v_stages.write(domain.next);
```

The verifier must prove:

- A read follows the matching commit and rendezvous.
- An overwrite follows all reads of the aliased stage.
- Prologue and epilogue uses are covered.
- The modulo phase relation is valid.
- Stage accesses do not escape the loop scope.

Only a small layout algebra is needed:

```rust
Layout2D::row_major([rows, cols])
    .transpose()
    .pad(Axis::Col, amount)
    .xor_swizzle(bytes)
```

It should verify allocation size, vector alignment, collaborative-write injectivity,
producer/consumer agreement, and MFMA fragment compatibility.

Commit completion is now typed and validated. Each hook returns a `CommitBatch` classified as
`Intrinsic`, `Opaque`, or `DirectAndOpaque`; the pipeline rejects a completion class that does not match
its selected publication policy. This closes the previous gap where an intrinsic commit could be paired
with a bare/opaque publication schedule or vice versa.

Waitcnt-opaque gathers also have explicit value readiness. `ready_after_lgkm` wraps every packed b64
operand in `OpaqueReadyB64` tied to an `SWaitLgkmcnt`; this prevents an MFMA consumer from crossing the
manual wait in machine scheduling. The post-loop V drain now applies the same wrapping to every final
P*V A operand, rather than protecting only rolled-loop gathers.

The implemented VMEM primitive intentionally has queue semantics:

```rust
let anchor = VmemWaitAnchor::from(last_issue);
let ready_for_older_v = b.swait_vmcnt_allowed(anchor, 4);
```

The anchor places `vmcnt(4)` after the complete issue batch. It does not identify a payload or mean that
the four direct-K transfers themselves are complete. In the accepted d128 schedule, V loads are older,
four K direct-to-LDS dwords are younger, and the wait allows those four younger VMEM operations to remain.
Host dependency-cone tests prove the older-V/four-younger-K relation.

The eventual general async-copy API should identify the transfer made ready:

```rust
let copy = copy_async(k_stages.write(domain.next), k_view.tile(domain.next));
let ready = b.wait_vmem(copy, Vmcnt::KeepNewer(8));
```

Minimal IR:

```rust
BufferLoadLds { resource, global_offset, lds, lds_offset, bytes, async_id }
WaitCnt { counter, remaining, waits_for }
```

The accepted direct-K path is covered by the narrower anchor/cone contract above. General transfer tokens
with payload ownership are still required before exposing arbitrary partial `vmcnt(k)` tuning in the DSL.
Direct-to-LDS V is not part of this roadmap because the current V path requires a
write-side transpose and gfx942 has no transpose LDS read.

### 5. Schedule Regions And Compiled Resource Contracts

Scheduler directives need local membership and local instruction budgets:

```rust
let region = b.schedule_region("qk-softmax", |b| {
    let previous = state.update_previous(scores_previous);
    let next = qk(b, q, k);
    (previous, next)
});

region.interleave(
    OpClass::Mfma,
    1,
    OpClass::Exp,
    3,
    Repeat::AtMost(16),
);
```

`Repeat::AtMost` is important. The measured d64 kernel intentionally requests more
pairs than its local MFMA count because the exact count hit a compiler scheduler cliff.
The DSL should call this a best-effort policy rather than an exact static claim.

Resource and ISA expectations should be reusable program contracts:

```rust
program.expect_resources(ResourceContract {
    target: Gfx942,
    max_vgpr_allocated: 240,
    max_lds_bytes: 44_032,
    max_scratch_bytes_per_thread: 0,
});
```

The first reliable register-placement contract is compiled resource metadata. Do not
add `force_vgpr` or `force_agpr` APIs until lowering can enforce them without bypassing
MFMA hazard recognition.

The current pipeline verifier computes the dependency cone from emitted outputs. A publication token and
tail barrier must both be in that cone; for a phased schedule the tail must depend on the earlier
publication, and exactly one seed/rebalance pair must be present. An unphased schedule must contain zero
phase barriers. This is stronger than whole-arena counting, but it does not rehabilitate FA phase
staggering: the deadlocked experiment remains removed and no phase constructor is callable.

## High-Gain Kernel Work

### A. Remove Empty Startup Softmax And PV - Implemented

The current score rotation starts from `S(-1) = -inf` and `P(-1) = 0`. The first
softmax is numerically inert, and its complete PV burst multiplies zero probabilities,
but both still execute.

Current effective structure:

```text
iteration 0:
    softmax(S(-1)) || QK(0)
    PV(-1)

iteration i:
    softmax(S(i-1)) || QK(i)
    PV(i-1)

drain:
    softmax(S(last))
    PV(last)
```

Target structure:

```text
warmup:
    stage K0/V0
    QK(0)

steady i = 1..last:
    softmax(S(i-1)) || QK(i)
    PV(i-1)

drain:
    softmax(S(last))
    PV(last)
```

Expected payoff:

- Approximately 4-8% at S256.
- Less than 2% at S2048.
- No additional LDS or VGPR requirement.

Required gate:

- Dynamic MFMA count equals exactly `waves * nblocks * 2 * (d / 8)`.
- One fewer softmax copy and one fewer PV copy where code structure permits.
- Existing full, ragged, and repeated correctness tests pass.
- Zero scratch and no resource growth.
- No runtime branch in the steady loop.

Measured result:

- S256 d128 improved from about 131.5 TF/s to 136.3-137.5 TF/s.
- S2048 d64 improved to 281.3-283.0 TF/s.
- S2048 d128 improved to 300.4-301.3 TF/s.
- Allocated VGPR, LDS, scratch, and scheduler cadence are unchanged.
- Steady trips are now `nblocks - 2` and are host-gated for exact and ragged domains.
- Two-block domains emit no synthetic loop and transition directly from warmup to epilogue.
- PMC confirms exact useful work: 2,097,152 physical MFMAs at d64 and 4,194,304
  at d128 for `bh=32, S=2048`, with no dummy startup contraction.

Ragged inputs use explicit warmup/steady/epilogue lexical scopes plus a range-entry
predecessor, preventing loop-local LDS and reduction DAGs from being reused after the
loop. Two-block inputs use a real no-steady-body pipeline path. Repeated gfx942 d64/d128
correctness passes for `n=64` and `n=80`; tile-exact hot paths intentionally use the
allocation-free root scope to preserve their scheduler topology and measured throughput.

### B. Q128 Short-Sequence Specialization - Measured And Rejected

Q256 is a good long-sequence tile but produces too few workgroups for short shapes.

Add a four-wave, 256-thread Q128 specialization selected by sequence length and grid
density.

d64 candidate:

```text
Q128
K1 / V3
predicted about 144 allocated VGPR
17,920 B LDS
up to 3 workgroups per CU by resource model
```

d128 candidate:

```text
Q128
K1 / V2
predicted at most 256 allocated VGPR
26,624 B LDS
2 workgroups per CU if both limits hold
```

Q128 alone is not sufficient for d128 because the current 44,032-byte LDS footprint
still permits only one workgroup. K1/V2 is worth revisiting only because the combined
variant crosses the two-workgroup threshold. V2 at Q256 remains a measured negative.

Expected payoff:

- Approximately 8-30% on short or underfilled shapes.
- No requirement that Q128 replace Q256 at long sequence lengths.

Required gate:

- d64 allocated VGPR preferably <= 144, hard ceiling 168.
- d128 allocated VGPR <= 256 and LDS <= 26,624 B.
- Zero scratch.
- Repeated correctness to expose LDS overwrite races.
- Benchmark at 128, 256, and 512 Q256-equivalent workgroups.

Abort d128 before timing if two workgroups cannot fit by both VGPR and LDS limits.

The isolated d64 Q128 experiment was correct but regressed S256 from 131.9-132.6 TF/s
to 123.4 TF/s at the full-grid shape and from about 95 TF/s to 88.4 TF/s at the
underfilled shape. Extra K/V fill work and reduced per-workgroup query reuse outweighed
the additional workgroups. Q128 is no longer a main-roadmap item; d128 K1/V2 should not
be attempted without a different K/V reuse model.

### C. Segmented Direct-K Movement - Implemented For Long d128

The first prototype kept both K and V register-staged and moved their commit after the
QK/softmax and PV regions. It increased prefetch-to-use distance but extended register
lifetimes and added publication boundaries.

Rejected register-only structure:

```text
memory A:
    prefetch K/V(i+1)
    gather K(i), V(i-1)
    make current LDS reads ready

compute:
    softmax/QK
    PV

memory B:
    commit prefetched K/V(i+1)
    publish the stage for the next iteration
```

This transfers the successful matmul principle of register-staged latency cover without
copying matmul's tile or schedule.

Expected payoff:

- Approximately 3-10% on long d128 if global latency is currently exposed.
- Likely smaller on d64, where L2 hit rate is already high.

Required gate:

- Final ISA shows global loads before a substantial MFMA run.
- No additional hot-loop workgroup barrier.
- No full VMEM drain before compute unless required by a consumed transfer.
- d128 remains <= 256 allocated VGPR and zero scratch.
- MFMA utilization rises by at least 0.02 or wall time improves by at least 3%.

Abort if LLVM sinks loads back next to commit, register pressure spills, or a new barrier
cancels the latency cover.

The first register-only split was correct and reduced d128 allocation, but regressed
long-sequence throughput because the extra gather/publication boundaries outweighed the
shorter lifetime. It remains rejected.

The accepted path changes the transfer mechanism rather than merely splitting clusters:

```text
issue V(i+1) global -> VGPR
gather current K(i), V(i-1)
issue four wave-coalesced K(i+1) dwords global -> alternate LDS plane
tracked vmcnt(4)
commit V(i+1) with eight waitcnt-opaque ds_write_b16 stores
QK(i) / softmax(i-1)
PV(i-1)
vmcnt(0), lgkmcnt(0), one publication barrier
```

K uses inverse-XOR source addressing because direct-to-LDS writes must target
lane-contiguous physical LDS dwords. V remains register-staged because gfx942 lacks the
transpose LDS read needed for direct V. The V writes use one base VGPR plus pitch-spaced
immediates; making them waitcnt-opaque prevents LLVM from conservatively strengthening
the partial wait to `vmcnt(0)` due to LDS aliasing.

The production selector enables this only for `d=128 && S>=1024` within the public Q256-exact domain.
Qualification tests cover the direct primitive, the swizzled 32x128 K tile, scalar asm V writes,
two-block and ragged paths, repeated full-FA correctness, transfer structure, resources, and final ISA
ordering. Long d128 historically allocates 232 VGPR, 44,032 B LDS, and zero scratch. The production ISA
gate now checks the final code-object sequence, not source mnemonic counts: an older staged V load and
four direct-K dwords precede `vmcnt(4)`, scalar V writes precede a substantial d128 MFMA run, and
`vmcnt(0) -> lgkmcnt(0) -> s_barrier` publishes the stage afterward.

Rendered LLVM checks remain separately labeled as DSL-construction checks for scheduler-group cadence and
the presence/absence of movement forms. The code-object bytes are not golden-hashed: the repository's
existing stable signature framework hashes reachable tile IR, while clang-produced ELF bytes and metadata
are toolchain-dependent. The ordered disassembly sequence is the stable code-object structural gate.

Two follow-on movement experiments are implemented but not production-promoted:

- Packed transposed V remaps each lane to four KV-row dwords, uses four waitcnt-opaque
  `v_perm_b32` operations, and replaces eight `ds_write_b16` stores with two
  `ds_write_b64` stores sharing one address (`offset:72` for the second write). It is
  bit-exact on full, ragged, and two-block paths, compiles at 216 VGPR with zero scratch,
  and preserves all three final-ISA `vmcnt(4)` waits. Centered long-d128 measurements
  were 309.9, 317.0, and 314.7 TF/s (314.7 median), only 2.3% above the accepted
  307.7 TF/s median and below the predeclared 3% promotion gate. The final one-address
  form sampled 311.5 TF/s before a later rejected phase probe wedged the device.
- Waitcnt-opaque K/V gathers use explicit queue-wide `lgkmcnt(0)` readiness and tie the
  gathered b64 operands through side-effect asm, preventing any MFMA from crossing the
  wait in final ISA. They are bit-exact across 64 long-shape replays, but increase the
  direct path to 240 VGPR and measured 285.3 TF/s, so they remain qualification-only.

The ignored resource gate now compiles every exercised FA constructor. It requires zero scratch for
production direct K, forced direct K, packed V, opaque gather, register-staged oracle, and register-asm
publication. The post-reboot code objects allocate 144 VGPR/17,920 B LDS for production d64, 232/44,032
for production and forced direct d128, 216/44,032 for packed V, 240/44,032 for opaque gathers, and
232/44,032 for both the register-staged oracle and register-asm publication. Every variant has zero
scratch. The old 136-VGPR d64 result remains reproducible with AMD LLVM 22, including from the current
source. The active unqualified `clang` now selects LLVM 23, which allocates 144 VGPR from byte-identical
rendered LLVM; even the immutable pre-review snapshot moves from 136 to 144 under that compiler. The gate
therefore uses the measured LLVM 23 ceiling. Restoring a 136 ceiling would require pinning compilation to
LLVM 22, not reverting the DSL changes.

An asm-contained register-K2/V3 `warp/4` phase attempt was also rejected. The identical
asm read/write movement is exact without asymmetric progress. Early phase forms produced
wrong values; after moving the phase seed behind a synchronized QK(0) warmup and limiting
phase to the rolled body, the one-steady-iteration ragged case hard-deadlocked. The public
phase constructor and device path were removed. Production therefore retains
`warp_row=None`; the reusable opaque movement and publication-generation verifier remain
for a future architecture-level barrier investigation.

The final-drain qualification bug found in review is fixed structurally: final V `DsReadB64` values are
wrapped through `OpaqueReadyB64` after the drain's `lgkmcnt(0)`, and the host gate requires every d128
final P*V MFMA A operand to depend on that exact wait. After reboot, the opaque-gather constructor passed
64 long-d128 bit-exact replays against the register-staged oracle.

### D. Causal And GQA-Aware Work/Data Reuse

This is the highest potential production optimization after runtime domains exist.

First stage:

- Runtime Hq/Hkv mapping.
- Workgroup-specific causal KV limit.
- Mask before online maximum.
- BF16 O and LSE output.

The causal loop bound can eliminate close to half of QK/PV work in balanced causal
attention. This is a larger payoff than scheduler-level tuning.

Second stage, only after GQA semantics are correct:

- Assign multiple query heads sharing one KV head to a cooperative workgroup or
  persistent work unit.
- Stage each K/V block once and consume it for multiple Q-head tiles.
- Select the number of cooperating heads from LDS, VGPR, and grid-density constraints.

This is not a near-term mainloop patch. It requires bounded strided views, attention
partitioning, dynamic domains, and verified stage lifetimes first.

Success is measured by reduced physical K/V bytes and wall time, not only by a higher
cache hit rate.

## Integrated Migration Plan

| Slice | Change | Enabling or optimizing | Required result |
|---:|---|---|---|
| 0 | Freeze current d64/d128 oracle and resource/ISA gates | Enabling | Complete; preserve byte identity outside intentional variants |
| 1 | Remove empty startup softmax/PV | Optimization | Complete for all supported domains; measured win |
| 2 | Runtime arguments, bounded views, and `IterDomain` | Enabling | Low-level complete; FA `AttentionArgs` migration pending |
| 3 | Explicit loop-region identity and no-steady lowering | Enabling | Complete; warmup works for ragged and two-block inputs |
| 4 | Verified layouts and circular LDS stages | Enabling | Byte-identical current K1/K2/V3 movement |
| 5 | Typed async copies and waits | Enabling | Typed commit completion, explicit opaque readiness, VMEM wait anchors, and ordered d128 direct-K publication implemented; payload-owning transfer tokens pending |
| 6 | Attention partition, causal/GQA mask, BF16 O, LSE | Enabling and optimizing | Full production correctness matrix |
| 7 | Causal pruning and GQA shared-K/V research | Optimization | Material physical-work or memory reduction |
| 8 | Schedule regions and reusable resource contracts | Enabling | Existing cadence represented without codegen change |

The order allows the first two kernel wins to proceed with minimal infrastructure while
the production and verification layers are built for the riskier changes.

## Benchmark Matrix

Do not optimize only `bh=32, S=2048`.

For both d64 and d128, measure:

| S | bh values | Q256 workgroups | Purpose |
|---:|---|---:|---|
| 256 | 128, 256, 512 | 128, 256, 512 | Startup cost, grid starvation, Q128 dispatch |
| 512 | 64, 128, 256 | 128, 256, 512 | Q128/Q256 crossover |
| 1024 | 32, 64, 128 | 128, 256, 512 | Medium sequence |
| 2048 | 16, 32, 64 | 128, 256, 512 | Long sequence, L2 and prefetch behavior |

Every variant must report:

- Median and minimum GPU time after warmup.
- Centered-input TF/s.
- Exact physical MFMA count.
- Allocated VGPR, LDS, and scratch.
- MFMA utilization.
- L2 hit/miss count.
- LDS bank-conflict count.
- Wait, barrier, priority, and hazard-NOP census.

## Gfx942 Qualification Invocation

There is no ordinary-runner CI target for these ignored device tests. Run them only on a healthy gfx942
system with the ROCm AMDGPU clang target, KFD access, and ROCm `llvm-objdump` available. The resource/ISA
test compiles and disassembles code objects without dispatching the FA kernel, but it still resolves the
configured AMD device through the existing test path.

```bash
SVOD_DEVICE=AMD:0 \
SVOD_LLVM_OBJDUMP=/opt/rocm/llvm/bin/llvm-objdump \
cargo test -p svod-tk2 --lib device::fa32_stays_within_resource_and_schedule_budget \
  -- --ignored --nocapture --test-threads=1
```

After the resource/ISA gate passes, the primary numerical qualification commands are:

```bash
SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib device::flash_attention32_matches_reference_on_gfx942 \
  -- --ignored --nocapture --test-threads=1
SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib device::flash_attention32_long_direct_k_matches_register_k_on_gfx942 \
  -- --ignored --nocapture --test-threads=1
```

Do not run these commands while the device is wedged. The 2026-07-18 post-reboot run passed both commands,
the resource/ISA gate, and `fa32_warmup_has_no_dummy_mfma_work`; the latter measured exactly 2,097,152 d64
and 4,194,304 d128 physical MFMAs.

Correctness coverage must include:

- d64 and d128.
- Multiple batches and heads.
- Sq and Sk around 31/32/33, 63/64/65, and 255/256/257.
- Unpadded ragged buffers with canary regions.
- Hq == Hkv, GQA, and MQA.
- Noncausal, top-left causal, and bottom-right causal.
- BF16 O and LSE against an FP32 reference.
- Repeated launches for race detection.

## Explicitly Deferred

These are omitted from the main roadmap because their expected payoff is small,
uncertain, or blocked by missing contracts:

- Exact external scheduler ratios.
- The previously tested delayed HK-style rotation.
- V2 plus a WAR barrier at Q256.
- Raw `warpid()/4` phase staggering (latest asm-contained attempt failed the device race/deadlock gate).
- Direct-to-LDS V.
- Full d128 XCD head grouping.
- Temporary zero-based PV at d128.
- Opaque assembly 32x32x8 MFMA.
- Premature VGPR/AGPR placement controls.
- Dedicated loader waves.
- Q512, KV16, or KV64 geometry.
- Persistent Q traversal for short sequences.
- Q128 with the current K/V reuse model.
- Deferred K/V commit across the current QK/PV regions.
- Public ragged or non-Q256-exact sequence lengths before bounded FA views replace padded allocation
  assumptions.
- A deterministic code-object byte signature across ROCm/LLVM versions; use tile-IR signatures plus the
  final-ISA structural sequence instead.

Some may become valid after the roadmap changes the resource threshold or verification
model. They should not be retried in the current kernel by adding ad hoc waits or
barriers.

## Final Recommendation

The immediate implementation sequence is:

1. Bind the implemented runtime scalars, bounded views, and dynamic domains into an
   FA-specific `AttentionArgs`/partition ABI.
2. Make current K/V stage arithmetic a verified circular-stage contract without
   changing generated code.
3. Generalize the proven d128 direct-K issue/wait/publication contract into typed
   asynchronous transfer tokens without changing other movement paths.
4. Complete causal/GQA/BF16/LSE production semantics.
5. Explore GQA shared-K/V execution as the next high-upside algorithmic change.

This preserves the fast current specialization, attacks the known short-sequence and
long-d128 bottlenecks, and builds only the DSL machinery needed to make future attention
kernels safe and concise.
