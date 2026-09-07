---
sidebar_label: Overview
---

# Why Hand-Written Kernels?

Svod is built around automation. You build a lazy graph, call `realize()`, and the optimizer
decides how to tile, vectorize, and parallelize every loop — with [beam search](../architecture/optimizations/kernel-search)
it will even compile and time hundreds of candidate schedules to find a fast one. You never write a loop.

So why does Svod ship a crate — `tk` — whose entire job is letting you write GPU kernels by hand?

Because some kernels can't be discovered by searching over loop transformations. The
optimizer's action space is "take this reduction and tile it, unroll it, put it in shared
memory." That's enough for matmul, for a fused feed-forward block, for a layernorm. It is
**not** enough for Flash Attention, whose math is a *recurrence*: each block of keys updates
a running maximum and a running sum, rescaling the accumulator as it goes. There is no single
`REDUCE` to tile — the loop body depends on the result of the previous iteration. No amount
of axis-shuffling produces it.

For kernels like that, you need to write the algorithm. `tk` is how you do it, without
leaving the compiler.

---

## `tk` is a builder, not a backend

The temptation, when you need a hand-written kernel, is to bolt on a second code path: a
little GPU DSL that emits its own assembly and gets launched separately from everything else.
Now you have two compilers, two debuggers, two mental models.

`tk` refuses that. It is, in its own words, *"a thin eager builder, not a backend."* When you
author a kernel with `tk`, it doesn't emit machine code — it emits the **same UOp IR** that
the rest of Svod already speaks: explicit `RANGE` loops, `INDEX`/`STORE` memory ops, `WMMA`
matrix-core ops. The exact intermediate representation described in
[One IR to Rule Them All](../architecture/ir-design).

That means a hand-written `tk` kernel and an autotuned graph kernel are the *same kind of
object* — two subgraphs in one UOp DAG, rendered by one renderer, run by one runtime.
[Authoring into the IR](./lowering) shows exactly how that works.

---

## The three faces of `tk`

Depending on who you are and what you're doing, `tk` presents one of three interfaces (all
re-exported from `tk/src/lib.rs`):

| Face | You are… | What you touch |
|------|----------|----------------|
| **USE** | an application author who just wants a fast kernel | `matmul`, `flash_attention`, `flash_attention_with`, `kmeans_assign` — they return lazy `Tensor`s, no kernel knowledge required |
| **AUTHOR** | writing a new tile kernel | the `Kernel` / `Group` builder, `ArchCaps`, the tile types (`GL`/`ST`/`RT`/`RV`), `Swizzle`, `graph_launch` |
| **DEBUG** | testing or benchmarking a kernel in isolation | `compile`, `launch`, `run_kernel`, `CompiledLaunch`, and structural `KernelFingerprint`s |

The USE face is the important one for most readers: `flash_attention(q, k, v)` gives you back
an ordinary `Tensor` that participates in the lazy graph like any other. You never see a tile.
[What Tiling Is](./tiling) opens up the AUTHOR face; [Debugging](./debugging) covers DEBUG.

---

## When to hand-write, and when to let BEAM do it

There's one rule, and it falls straight out of *what BEAM actually searches over*.

BEAM — and the heuristic optimizer it falls back to — search the space of **schedules** for a
*fixed* computation. Given a kernel's dataflow graph, they try ways to tile, vectorize, unroll,
parallelize, stage through shared memory, and map onto matrix cores (the `OptOps` actions:
`UPCAST`, `UNROLL`, `LOCAL`, `GROUP`, `TC`, …). What they never do is change *what* is computed:
the nodes of the graph — the adds, muls, and reductions — are fixed; only their arrangement is
up for grabs.

So:

> If a kernel needs only a good **schedule** of a fixed dataflow, let BEAM find it. If it needs
> a **different algorithm** than the naive one — something no reordering of the existing ops can
> produce — you have to write it.

| Property of the kernel | Built by | Examples |
|------------------------|----------|----------|
| **Fixed dataflow** — elementwise ops and reductions over a rectangular iteration space; only the *schedule* (tiling, vectorization, data placement, matrix-core mapping) is open | graph ops + **BEAM** | matmul / GEMM, feed-forward, layernorm, softmax |
| **Needs a reformulated algorithm** — a loop-carried recurrence, or restructured numerics, that no reschedule of the naive ops can produce | **hand-authored in `tk`** | Flash Attention (online softmax); brute-force k-means assignment (`kmeans_assign`) — a cross-term WMMA fused with a running argmin over streamed centroid tiles, so the full `[N, K]` distance matrix is never formed |

### What BEAM can't reach

Naive attention forms the entire `N×N` score matrix, takes a global softmax over it, then
multiplies by `V`. BEAM could tile and vectorize that, but it would still materialize the full
score matrix — the cost Flash Attention exists to avoid.

The fast version never forms that matrix. It streams over blocks of keys, keeping a running max
and sum and rescaling the output as each block arrives: online softmax. That isn't the naive
computation rescheduled; it's a different dataflow with a loop-carried dependency, where each
block reads state the previous block wrote. No `UPCAST`/`UNROLL`/`TC` sequence can introduce a
recurrence, so online softmax lies outside BEAM's search space. The gap is one of algorithm, not
schedule, and that is what `tk` fills.

`tk` also ships a hand-written `matmul`, but it belongs in the first row of the table: it is a
performance canary for the DSL, not the production matmul, which goes through the graph.

:::tip For GPU experts
The structural difference between a hand-authored kernel and a BEAM-tuned one is a single field
on the `SINK` UOp's `KernelInfo`: a graph kernel leaves `opts_to_apply: None`, a `tk` kernel sets
`Some(vec![])`. Same IR, same pipeline, one marker. [Authoring into the IR](./lowering) traces
this end to end.
:::

---

## Where this section goes

The rest of this section builds up from the hardware problem to the design comparison:

1. **[Where the FLOPS Hide](./where-flops-hide)** — why a matrix core is hard to saturate,
   and the handful of bottlenecks every fast kernel has to beat.
2. **[What Tiling Is](./tiling)** — the abstraction that answers those bottlenecks, and how
   `tk` represents tiles in the type system.
3. **[Authoring into the IR](./lowering)** — how a `tk` kernel becomes UOps and joins the
   lazy graph.
4. **[Writing a Kernel](./first-kernel)** — authoring and running the simplest kernel, step by step.
5. **[Wave32 vs Wave64](./wave-portability)** — keeping one kernel correct across AMD's two
   wave widths and NVIDIA's warp32.
6. **[Flash Attention](./flash-attention)** — the worked example that motivated all of this.
7. **[Debugging](./debugging)** — running and verifying kernels by hand.
8. **[Profiling & Benchmarking](./profiling)** — the layered profiler and criterion integration,
   for any `Tensor` or `ExecutionPlan`.
9. **[tk vs HipKittens vs CuTile](./comparison)** — where this design sits in the landscape.
