---
sidebar_label: tk vs HipKittens vs CuTile
---

# Three Ways to Write a Tile Kernel

`tk` did not invent the tile abstraction. It sits in a small family of tile-based kernel systems,
and the most useful way to understand `tk`'s design is to place it next to its two closest
relatives:

- **[HipKittens](https://github.com/HazyResearch/HipKittens)** — HazyResearch's C++ tile library
  for AMD matrix cores, the direct lineage of `tk`'s abstractions.
- **[CuTile](https://github.com/NVIDIA/cutile-rs)** (cutile-rs) — NVIDIA Research's Rust system
  for tile kernels on NVIDIA GPUs.

All three share the core idea from [What Tiling Is](./tiling): pull fragment-sized tiles into
registers, compute on them, write them back. Where they differ is *who controls the hardware
mapping* — and that difference is a spectrum.

---

## The spectrum: explicit control ↔ managed abstraction

The three systems sit on an axis. At one end you manage registers, shared memory, and instruction
scheduling yourself; at the other you write tile-level code and a downstream compiler decides how
it maps to threads, shared memory, and matrix instructions. HipKittens sits at the explicit end and
CuTile at the managed end. `tk` sits left-of-center: it gives you explicit register and shared tiles
like HipKittens, but instead of being a standalone backend it lowers into Svod's single UOp IR.

---

## Side by side

| Axis | **tk** | **HipKittens** | **CuTile** |
|------|--------|----------------|------------|
| **Authoring surface** | Rust *builder API* (`Kernel`/`Group` mint UOps) | C++ *templates* | Rust *macro DSL* — write plain Rust in `#[cutile::module]`, the macro captures the AST |
| **IR target** | Svod's **one UOp IR** — same as the whole compiler | none (templates → clang amdgcn) | a *separate* MLIR `cuda_tile` dialect, serialized to Tile IR bytecode |
| **Lowering** | Svod render → LLVM → AMD binary, or → PTX (assembled to a cubin by `ptxas`, else JIT by the driver) | clang | bytecode → external `tileiras` assembler → cubin (JIT at first launch) |
| **Memory model** | **explicit** register *and* shared tiles | explicit register *and* shared tiles | **one** tile type (register-resident); shared-mem staging is implicit, chosen by the compiler |
| **Matrix-core API** | explicit `WMMA` op + role-based fragments | typed tiles → `__builtin_amdgcn_mfma_*` | a single functional `mma()` intrinsic |
| **Compute/memory overlap** | a `sched::pipeline` marker + a codegen pass | hand-written per kernel (raw scheduling intrinsics) | delegated to `tileiras` |
| **Headline differentiator** | one IR ⇒ hand kernels and autotuned kernels are peers | "built from the hardware up" | memory safety across the launch boundary |
| **Target** | AMD CDNA / RDNA **and** NVIDIA `sm_80+` | AMD CDNA / RDNA | NVIDIA `sm_80+` only |

Each `tk` kernel declares its own arch set on top of that: matmul, Flash Attention and single-query
attention are built for gfx942, gfx1151 and CUDA `sm_80+`; the k-means and k-NN kernels are AMD-only.

---

## What the code looks like

The authoring surfaces are genuinely different in feel. These snippets are illustrative — they
convey the *shape* of each model, not an exact API.

**HipKittens** — C++ templates; you name tiles and call the multiply directly:

```cpp
using namespace kittens;
rt_bf<64, 32>      a, b;     // register tiles of bf16
rt_fl<64, 32, col> acc;      // fp32 accumulator, col layout (MFMA output)

load(a, a_global, {row, k});
load(b, b_global, {k, col});
mma_ABt(acc, a, b, acc);     // acc += a · bᵀ  → __builtin_amdgcn_mfma_*
```

**CuTile** — write ordinary Rust inside a module the macro captures; tiles are immutable, the
compiler stages shared memory for you:

```rust
#[cutile::module]
mod kernels {
    use cutile::core::*;
    pub fn gemm(a: &Tensor<f32, A>, b: &Tensor<f32, B>, c: &mut Tensor<f32, C>) {
        let (i, j) = (tile_block_id_x(), tile_block_id_y());
        let mut acc = Tile::<f32, ACC>::zeros();
        for k in 0..a.dim(1) / BK {
            acc = mma(a.partition(AK).load([i, k]),
                      b.partition(BK).load([k, j]),
                      acc);            // one functional intrinsic
        }
        c.partition(CC).store([i, j], acc);
    }
}
```

**tk** — a Rust builder that mints IR; you request fragments by role and emit `Group` ops:

```rust
let ker = Kernel::new(grid, block, caps);
let a   = ker.gl(a_spec);                       // global layout
let mut acc = ker.rt(FragRole::Accumulator);    // role, not a hardcoded shape
let g   = ker.group();

g.load(&shared_a, &a, idx);                      // global → LDS (swizzled)
g.mma(&mut acc, &operand_a, &operand_b);         // → WMMA UOp
let sink = ker.finish(stores);                   // SINK { opts_to_apply: Some(vec![]) }
```

The CuTile example reads like a normal program; the `tk` example reads like building a graph.
That's the trade: CuTile's macro captures your *syntax* and re-parses it, while `tk` is a library
whose method calls *are* the IR construction.

---

## The key conceptual difference

Two distinctions matter more than the rest.

**Who owns shared memory.** CuTile has exactly *one* tile concept — the register tile — and
deliberately hides shared-memory staging; its `tileiras` assembler decides how data flows through
LDS, caches, and matrix cores. `tk` and HipKittens expose *both* register and shared tiles and
make you stage explicitly. CuTile sits a level *above* the register/shared distinction; `tk` sits
*at* it. That's the price and the power of control: more to manage, but the
[overlap and swizzle decisions](./where-flops-hide) that win the performance are yours to make.

**Where the IR lives.** This is `tk`'s real distinguishing move. HipKittens is a standalone C++
framework — it produces kernels, full stop. CuTile lowers to a *separate* MLIR dialect that only
its own toolchain consumes. `tk` lowers into the **same UOp IR the rest of Svod already speaks**.
A `tk` kernel isn't an artifact handed to a different compiler — it's a subgraph in the one IR,
next to every autotuned kernel.

:::tip[For GPU experts]
The IR-target difference is concrete at the toolchain level. `tk` renders its `SINK` through
`svod-codegen` to LLVM IR and then to an AMD binary or to PTX (assembled by `ptxas`, else JIT-ed by
the driver) — the same path graph kernels take. CuTile instead
serializes its tile dialect to bytecode that an *external* `tileiras` assembler turns into a cubin,
JIT-compiled at first launch; HipKittens is C++ templates compiled by clang. So "one IR" for `tk`
literally means one render-and-compile pipeline, where the others bridge into a separate compiler.
:::

---

## Why this matters

This is what lets Svod offer two things that are usually mutually exclusive: letting the compiler
find the schedule, and writing the schedule yourself, with no second compiler.

A BEAM-autotuned matmul and a hand-written Flash Attention are both just `SINK` UOps in one DAG.
They render through one renderer, run on one runtime, and print with one debugger. The only thing
that distinguishes them is the `opts_to_apply` marker, whose home is
[Authoring into the IR](./lowering): the same IR carries both an optimizer-driven and a
hand-driven kernel.

HipKittens proves you can match vendor libraries by going hardware-up. CuTile proves you can make
GPU kernels safe and high-level. `tk`'s bet is narrower and, for Svod, more useful: take the
hardware-up tile model, and instead of building a new backend around it, *speak the IR the
compiler already has*. That's the whole reason `tk` is small — and the reason a hand-written
kernel feels like a first-class citizen rather than an escape hatch.
