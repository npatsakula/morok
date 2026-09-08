---
sidebar_label: Flash Attention
---

# Worked Example: Flash Attention

Flash Attention is the kernel that justifies `tk`'s existence — the one the
[Overview](./overview) named as *not* expressible as a single schedulable reduction, the reason
a hand-authoring surface exists at all. This chapter walks through it: what makes it hard, how
the tile abstractions answer that, and where the [Wave32 vs Wave64](./wave-portability) split
shows up in anger.

We're describing the forward kernel in `tk/src/kernels/fa.rs`, reached through the USE-face
`flash_attention(q, k, v)`. It is built for gfx942 (CDNA3), gfx1151 (RDNA3.5) and CUDA `sm_80+`;
the per-warp `(q_blk, kv_blk)` tile is chosen per device by `FaPolicy` — the taller tile only when
its shared-memory buffers fit and the launch grid already covers the device's compute units,
otherwise the baseline `{16, 32}`.

---

## Why attention can't be autotuned

Plain attention is `softmax(QKᵀ) · V`. Written naively, that means: form the full `N×N` score
matrix, softmax it, multiply by `V`. The score matrix is enormous and never needs to exist all
at once — so Flash Attention streams over blocks of keys and values, maintaining the softmax
*incrementally*.

That word — incrementally — is the problem. The softmax normalization depends on the maximum
and the sum over *all* keys, but we only see one block at a time. So we keep running statistics
and fix up the result as we go. This is **online softmax**, and it's a recurrence: each KV block
reads and updates state the previous block produced.

The optimizer's action space is "tile and unroll this `REDUCE`." There is no `REDUCE` here to
tile — there's a loop whose body depends on its own previous iteration. The search can't find
it. You have to write it.

---

## The algorithm, in tiles

The kernel assigns each wave a block of queries and walks the keys/values block by block. For
each KV block it runs this loop body, all in tiles:

```text
for each block of K, V:                          ┌─ everything here is a tile op
    S   = Q · Kᵀ                                 │  (mma into a register accumulator)
    S   = mask(S)                                │  causal + key-padding masks
    m'  = max(m, rowmax(S))                      │  update running max  (cross-lane reduce)
    P   = exp2(S - m')                           │  rescale to the new max (base-2 exp)
    l   = l * exp2(m - m') + rowsum(P)           │  update running sum
    O   = O * exp2(m - m') + P · V               │  rescale accumulator, accumulate
    m   = m'                                     │
O = O / l                                        └─ final normalize
```

Two matrix multiplies per block (`Q·Kᵀ` and `P·V`), two cross-lane row reductions (the max and
the sum), and a rescale of the output accumulator every time the running max moves. The
`exp2` — base-2 exponential — is deliberate: the temperature is folded into `Q` up front so the
hardware's fast `exp2` unit can be used directly.

Each of those lines is a `Group` operation on tiles: `mma` for the multiplies, a `RV`
(register-vector) reduction for the row max/sum, an elementwise `exp2`/`mul` map for the
rescale. No lane arithmetic in sight.

---

## Streaming: double-buffered KV

This is gap 2 from [Where the FLOPS Hide](./where-flops-hide) in action. While the matrix
core works on the current KV block, the next block should already be on its way into shared
memory. The kernel keeps **two** LDS buffers and alternates ("double-buffering" / software
pipelining): compute on buffer A while loading buffer B, then swap.

```text
   load K/V block 0 --> LDS[A]
   ┌─────────────────────────────────────────────────┐
   │ compute on LDS[A]   ║   load block 1 --> LDS[B] │   <- overlap
   │ compute on LDS[B]   ║   load block 2 --> LDS[A] │
   │ ...                                             │
   └─────────────────────────────────────────────────┘
```

The shared tiles carry their XOR swizzle (gap 3), so the cooperative fill and the per-lane read
are bank-conflict-free.

---

## The layout wrinkle: relayout between the two matmuls

Here's where [Wave32 vs Wave64](./wave-portability) stops being abstract. The kernel does two
matrix multiplies, and the output of the first (`S = Q·Kᵀ`, after softmax becomes `P`) is the
*input* of the second (`P·V`). Can the score accumulator be fed straight back in as an operand?

- **On CDNA and on CUDA** (`acc_reusable_as_input() == true`): yes. The MFMA accumulator *is* the
  input fragment on CDNA, and the two-half `mma.sync` f32 accumulator holds the m16n8 C fragments in
  exactly the A-operand register order — so it's a register copy. Cheap.
- **On RDNA** (`acc_reusable_as_input() == false`): no. The even/odd accumulator and the replicated
  operand differ, so `P` has to make a **round-trip through LDS** (the per-warp softmax band the
  policy's `att_band` allocates) to be relaid out before the second multiply.

The kernel branches on `ArchCaps` to do the right thing on each. Same algorithm, two physical
realizations — exactly the portability tax the previous chapter described, here in the hottest
loop of the most important kernel.

---

## Masking

Causal masking (a query can't attend to a future key) and key-padding masking (ignore padded
positions in a batch) are applied to the score tile `S` before the softmax. The mask is derived
from the tile's own lane/row coordinates rather than loaded from memory — the position of each
score element is implied by which fragment and lane holds it, so the mask is computed, not
fetched.

:::tip[For GPU experts]
The compute/memory overlap isn't hand-emitted as raw scheduling intrinsics in `tk` the way it is
in HipKittens' kernels. Instead the KV loop is annotated with
`sched::pipeline(SchedKind::Attention, …)` (`tk/src/kernels/fa.rs`), a marker that a
post-linearization scheduling pass in codegen consumes to interleave the matrix, memory, and
exponential instruction streams. This keeps the kernel body readable — it expresses *what* to
overlap, and a later pass decides the concrete instruction ordering, rather than the author
threading raw scheduling intrinsics through the algorithm by hand.
:::

---

## Why this matters

Flash Attention is the whole section condensed into one file:

- it exists because **online softmax is a recurrence**, not a tileable reduction
  ([Overview](./overview));
- it lives or dies on **streaming and overlap** ([Where the FLOPS Hide](./where-flops-hide));
- it's expressed entirely in **tiles and roles**, never lane indices ([What Tiling Is](./tiling));
- it compiles to **the same UOp IR** as everything else and joins the lazy graph as an
  `Op::Call` ([Authoring into the IR](./lowering));
- and it carries an explicit **accumulator-reuse branch** in its hot loop, one per fragment layout
  ([Wave32 vs Wave64](./wave-portability)).

That's why it's hand-written, and why `tk` exists to write it. To run it in isolation and check
its numbers, see [Debugging](./debugging).
