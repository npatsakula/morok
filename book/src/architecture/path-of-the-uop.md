# Path of the UOp: The 22-Stage Codegen Pipeline

A UOp starts as a high-level tensor expression. By the time it reaches the hardware, it has been transformed through 22 distinct stages—each with a specific purpose, each building on the last. This chapter traces that journey.

The pipeline is a proven design for tensor compilation. Understanding it means understanding how tensor expressions become machine code.

---

## How to Read This Chapter

If you're not a compiler engineer, this chapter might seem intimidating. Here's what you need to understand before diving in.

### Key Concepts

**UOp (Micro-Operation)**
- Think of it as a node in a flowchart representing one computation
- Example: `ADD(a, b)` means "add a and b"

**Pattern**
- A find-and-replace rule for code structures (not text)
- Example: "If you see ADD(x, 0), replace with x"
- Patterns fire repeatedly until no more matches (fixpoint)

**Range**
- A loop iteration: `RANGE(0..10)` means "for i from 0 to 10"

**AxisType**
- What kind of loop is this?
  - Global: Parallel across GPU blocks / CPU threads
  - Local: Parallel within a workgroup
  - Reduce: Accumulator (sum, max, etc.)
  - Loop: Sequential iteration

**Stage**
- One transformation pass through the code
- Patterns fire until fixpoint, then move to the next stage

### Reading Strategy

1. **First pass**: Read just the "What This Does" and "Why This Matters" sections
2. **Second pass**: Look at the diagrams and examples
3. **Third pass** (if you want details): Read the pattern descriptions

### Questions to Ask

For each stage, ask:
- What does this stage accomplish? (High-level goal)
- Why do we need this stage? (Motivation)
- What would go wrong without it? (Consequences)

---

## Overview

The 22 stages fall into four phases:

```text
Tensor Expression
       │
       ▼
┌─────────────────────────────────────┐
│ RANGEIFY (Stages 1-7)               │
│ Movement ops → Explicit loops       │
│                                     │
│ [Make iteration explicit,           │
│  optimize ranges]                   │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ EXPANDER (Stages 8-10)              │
│ UNROLL/UPCAST → Explicit vectors    │
│                                     │
│ [Expand optimization primitives]    │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ DEVECTORIZER (Stages 11-15)         │
│ Vector ops → Scalar code            │
│                                     │
│ [Lower to hardware-specific ops]    │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ LINEARIZER (Stages 16-22)           │
│ IR → Linear instruction sequence    │
│                                     │
│ [Serialize to executable code]      │
└─────────────────────────────────────┘
       │
       ▼
  Machine Code
```

Each stage applies pattern-based rewrites. Patterns fire until fixpoint, then the next stage begins.

---

## Phase 1: Rangeify

**Goal**: Transform high-level movement operations into explicit loop structures and optimize ranges.

---

### Stage 1: Early Movement Ops

> **Stage at a Glance**
>
> **Goal**: Clean up movement operations before range assignment
> **Key Patterns**: Movement on INDEX, movement through wrappers, nested INDEX simplification
> **Impact**: Prevents missed optimizations later in the pipeline

**What This Does**: This stage cleans up movement operations by pushing index manipulations into places where they're actually needed. Think of it as organizing your desk before filing papers—move instructions closer to where the data is used.

**Why This Matters**: Movement operations (RESHAPE, PERMUTE, etc.) are convenient abstractions, but the hardware needs concrete index calculations. By cleaning them up early, we ensure patterns in later stages can match correctly.

**Pattern**: `pm_mops + pm_syntactic_sugar` (bottom-up)

| Pattern | Transformation | Visual | Location |
|----------|---------------|--------|----------|
| Movement on INDEX | Apply movement to index expressions | `INDEX(PERMUTE(arr), [i, j]) → INDEX(arr, [j, i])` | `movement_op_patterns()` |
| Movement through AFTER | Move RESHAPE through timing wrapper (Tinygrad-specific) | `AFTER(RESHAPE(x, arg), [dep1, dep2]) → RESHAPE(AFTER(x, [dep2]), arg)` | Tinygrad only |
| Movement through END | Unwrap movement from END wrapper (Tinygrad-specific) | `END(RESHAPE(x), ranges) → END(x, ranges)` | Tinygrad only |
| Nested INDEX simplification | Remove redundant nested INDEX (Morok) | `INDEX(INDEX(ptr, [i]), [i]) → INDEX(ptr, [i])` | `movement_op_patterns()` |
| Nested INDEX concat | Flatten nested INDEX for PtrDType | `INDEX(INDEX(ptr, i), j) → INDEX(ptr, i, j)` | `pm_syntactic_sugar` |

**Why bottom-up?** Child nodes must be clean before parents can match. Movement ops nest deeply; cleaning from bottom prevents missed patterns.

**Note**: Tinygrad and Morok have different approaches here. Tinygrad moves movement ops through wrappers (AFTER, END) because it re-applies movement ops during bufferization. Morok removes movement ops entirely by transforming indices during bufferization, so AFTER/END patterns are not needed.

**Morok**: `movement_op_patterns()` in `rangeify/patterns.rs`

---

### Stage 2: Load Collapse

> **Stage at a Glance**
>
> **Goal**: Eliminate REDUCE operations by detecting range-independent computation
> **Key Patterns**: Bounded sum, gated load collapse, general reduce elimination
> **Impact**: Converts loop iterations to arithmetic operations

**What This Does**: Eliminates REDUCE operations by recognizing when the computation can be done without iteration. Uses range-independent computation detection and symbolic simplification.

**Why This Matters**: Reducing iterations to arithmetic operations eliminates loop overhead. Instead of running a loop 1000 times, compute the answer directly.

**Pattern**: `pm_load_collapse`

```text
// Before: Sum with bounds check
sum(1 for k in 0..64 if k >= length)

// After: Compute count directly (NO LOOP!)
count = clamp(64 - length, 0, 64)
```

The mechanism works by:
1. Identifying subexpressions that don't depend on the REDUCE range
2. Creating DEFINE_VAR for those subexpressions (treats as loop-invariant)
3. Substituting the range with DEFINE_VAR and running symbolic simplification
4. If the simplified expression has no more ranges, the REDUCE is eliminated

**Note**: WHERE movement through INDEX (`pm_move_where_on_load` in Stage 8) is a separate optimization that places conditionals before loads to skip memory accesses, but it doesn't eliminate REDUCE operations.

**Morok**: `pm_load_collapse()` in `rangeify/patterns.rs`

---

### Stage 3: Split Ranges

> **Stage at a Glance**
>
> **Goal**: Enable better optimization through divmod decomposition
> **Key Patterns**: Split ranges with modulo, flatten ranges
> **Impact**: Inner ranges can vectorize, outer can parallelize

**What This Does**: Handles modulo patterns by splitting a range into outer and inner components.

**Why This Matters**: Splitting ranges is like dividing a large task among team members. If you have 12 items and each person does 4, you get 3 people × 4 items. Inner loops (one person's 4 items) can be fast; outer loops (3 people) can run in parallel.

**Pattern**: `pm_split_ranges + pm_flatten_range`

```text
Before:  RANGE(end=12) % 4  // One loop with modulo (slow)
             ↓ [Split into outer × inner]
After:   RANGE(end=3) * 4 + RANGE(end=4)
            ↑outer        ↑inner
            Parallel      Sequential
```

This enables:
- Inner ranges can vectorize (SIMD)
- Outer ranges can parallelize (GPU blocks / CPU threads)

`pm_flatten_range` merges nested ranges on REDUCE/STORE/END when beneficial.

**Context**: Requires dictionary context (`ctx={}`) to track substitutions at SINK.

**Note**: The split only applies when `end % mod == 0` (divisibility check).

**Morok**: `pm_split_ranges()` + `pm_flatten_range()` in `rangeify/transforms.rs`

---

### Stage 4: Initial Symbolic

> **Stage at a Glance**
>
> **Goal**: Simplify expressions using algebra rules
> **Key Patterns**: Constant folding, identity removal, div-mod recombine
> **Impact**: Eliminates expensive operations, reduces code size

**What This Does**: Applies 100+ constant folding and algebraic simplification rules.

**Why This Matters**: Computers are fast at simple math. Dividing and taking remainders are slow operations. This stage uses algebra rules to eliminate slow operations whenever possible.

**Pattern**: `sym + pm_flatten_range`

**Constant folding**:
```text
ADD(CONST(2), CONST(3)) → CONST(5)
MUL(x, CONST(1)) → x
ADD(x, CONST(0)) → x
```

**Div-mod recombination**:
```text
(x / c) * c + (x % c) → x
```
*Why?* Computes the same value as `x` but with 3 operations instead of 1. This pattern finds and removes the redundancy (common in stride calculations).

**Boolean algebra**:
```text
x AND x → x
x OR FALSE → x
NOT(NOT(x)) → x
```

**Additional categories**:
- Identity removal (self-folding, redundant operations)
- Comparison simplification
- Cast optimization
- GEP pushing (move address calculations through ALUs)
- Where folding (combine WHERE with same conditions)
- Reduce mul chain (move multiplications outside reduce)

**Morok**: `symbolic_patterns()` in `symbolic/patterns.rs`

---

### Stage 5: Simplify Ranges

> **Stage at a Glance**
>
> **Goal**: Merge adjacent ranges to reduce loop overhead
> **Key Patterns**: Range merging with cost analysis
> **Impact**: Fewer loops = less overhead

**What This Does**: Merges adjacent ranges when profitable.

**Why This Matters**: Merging ranges is like combining multiple small trips into one big one. Instead of going to the store 4 times for 4 items, go once for all 4 items. Saves the overhead of starting and stopping.

**Pattern**: `pm_simplify_ranges`

```text
// Before: two separate ranges
RANGE(0..4), RANGE(0..8)

// After: merged (if compatible)
RANGE(0..32)
```

Merge criteria:
1. Axis types must be compatible (both output, both reduce, etc.)
2. REDUCE scope must remain consistent
3. **Cost-based**: Accept only if divmod operation count does not increase

The compiler only merges if it saves operations. Merging might require division/modulo to recalculate indices. If that costs more than it saves, merge is skipped.

**Morok**: `simplify_merge_adjacent()` in `rangeify/transforms.rs`

---

### Stage 6: Split Store (CPU-only)

> **Stage at a Glance**
>
> **Goal**: Avoid branch misprediction by splitting conditional stores
> **Key Patterns**: Split store ranges at comparison boundaries
> **Impact**: More predictable CPU execution

**What This Does**: Splits store ranges at conditional boundaries when there are `CMPLT(range, const)` comparisons in the store's consumer map.

**Why This Matters**: Branch misprediction slows down CPUs. Instead of one loop with an `if` statement that the CPU can't predict, we create two loops without conditionals. Each loop does predictable work, so the CPU stays fast.

**Pattern**: `pm_split_store`

```text
// Before: Store with conditional (branch misprediction risk)
for i in 0..100:
    if i < 50:
        output[i] = data[i]

// After: Two unconditional stores (predictable)
for i in 0..50:   // First loop
    output[i] = data[i]
for i in 50..100: // Second loop
    output[i] = data[i]
```

The transformation finds constant comparison points in the store's consumer map and creates disjoint ranges for each segment.

Skipped for GPU devices—they handle conditionals differently.

**Morok**: `pm_split_store()` in `rangeify/transforms.rs`

---

### Stage 7: Apply Opts

> **Stage at a Glance**
>
> **Goal**: Find optimal combination of vectorization, unrolling, memory usage
> **Key Algorithm**: Beam search or heuristics
> **Impact**: Can significantly improve performance

**What This Does**: The optimization search—either beam search or heuristic—explores different combinations of optimization actions.

**Why This Matters**: The compiler tries different combinations of optimizations (vectorize here? unroll there?) and picks the fastest. Finding the right combination can make code 10x faster.

**Function**: `apply_opts(sink, renderer)`

**Optimization actions**:

| Action | Effect | Hardware Target |
|--------|--------|-----------------|
| | TC | Enable tensor core usage | NVIDIA GPUs |
| | UPCAST | Vectorize a dimension | All (SIMD) |
| | LOCAL | Use local/shared memory | GPU (LDS) / CPU (L1) |
| | UNROLL | Unroll a loop dimension | All (avoid loop overhead) |
| | GROUP | Group operations for cache | All |
| | GROUPTOP | Group for reduce ops | GPU tensor cores |
| | THREAD | Thread-based parallelism | CPU |
| | NOLOCALS | Disable local memory usage | All (constraint, prevents further LOCAL actions) |
| | SWAP | Swap range assignments | All (try different tiling) |
| | PADTO | Pad for alignment | All (memory alignment) |

**Optimization Search Explained**:

The compiler searches for the best combination:
- **Heuristic mode** (BEAM=0): Fast hand-coded optimization patterns, no compilation
- **Beam search** (BEAM≥1): Compiles and runs candidates to measure actual performance

```text
Optimization Search:
├── Heuristic mode (BEAM=0): Hand-coded optimizations
└── Beam search (BEAM≥1):
    ├── Generate all possible actions (193 combinations)
    ├── Apply to all top-K candidates in parallel
    ├── Filter based on constraints
    ├── Compile and run each candidate → Measure actual time
    └── Pick fastest
```

**Note**: NOLOCALS is a constraint that sets `dont_use_locals = True`, preventing further LOCAL actions and affecting shared memory usage decisions.

**Morok**: `optimizer/mod.rs`, `optimizer/opts.rs`

---

## Phase 2: Expander

**Goal**: Transform optimization primitives (UNROLL/UPCAST) into explicit operations.

---

### Stage 8: Post-Opt Symbolic

> **Stage at a Glance**
>
> **Goal**: Symbolic simplification after optimization
> **Key Patterns**: WHERE movement, constant folding
> **Impact**: Enables better load combining and vectorization

**What This Does**: Symbolic simplification after optimization, plus WHERE movement.

**Why This Matters**: WHERE operations are like `if` statements. This stage moves `if` checks from after a load to before the load. Hardware can skip loading when the condition is false, saving memory bandwidth.

**Pattern**: `sym + pm_move_where_on_load`

```text
// Before: WHERE guards a load
WHERE(valid, LOAD(index), alt)

// After: validity moved to INDEX
LOAD(INDEX(ptr, idx, valid=valid), alt)
```

Moving validity into INDEX enables better load combining and vectorization.

**Note**: This pattern only matches when the alternative value is `0`. The transformation involves complex clause analysis: duplicate detection, range dependency checks, and data-dependent load verification.

**Note**: The Morok implementation uses `gate=` instead of `valid=` (the Index struct has a `gate` field). The concept is identical.

**Morok**: `pm_move_where_on_load()` in `symbolic/patterns.rs`

---

### Stage 9: Expander

> **Stage at a Glance**
>
> **Goal**: Convert UNROLL/UPCAST to explicit operations
> **Key Concepts**: UNROLL, CONTRACT, pattern order
> **Impact**: Makes vectorization explicit and ready for hardware

**What This Does**: Transforms UNROLL/UPCAST optimization primitives into explicit operations.

**Why This Matters**: UPCAST and UNROLL mark intent—what we want to do. This stage makes that intent explicit so the hardware can actually do it.

**Pattern**: `sym + pm_pre_expander + pm_group_for_reduce + expander`

⚠️ **Important: Pattern Precedence**

The patterns are combined and run to fixpoint. The order affects which pattern is tried first when multiple could match:
1. `sym` first (symbolic simplification)
2. `pm_pre_expander` second (converts UPCAST/UNROLL ranges)
3. `pm_group_for_reduce` third (handles GROUP_REDUCE axis)
4. `expander` last (main expansion)

Wrong precedence can cause incorrect vectorization or reduction scoping.

**UNROLL and CONTRACT**:

UNROLL and CONTRACT work together:

```text
UNROLL: "Take this one thing and make N copies for different positions"
Example:  x → [x_0, x_1, x_2, x_3]

CONTRACT: "Take these N things and combine them back"
Example:  [a, b, c, d] → one vector containing all four
```

Together: UPCAST marks intent to vectorize → UNROLL expands → CONTRACT combines.

**UPCAST range → VECTORIZE**:
```text
// Before: UPCAST marks vectorization intent
RANGE(end=4, UPCAST)
      ↓ [pm_pre_expander]
// Step 1: Convert to UNROLL with constant indices
UNROLL(VCONST([0, 1, 2, 3]))
      ↓ [expander]
// Step 2: Expand operations with UNROLL sources
// Operations now have unrolled sources
      ↓ [CONTRACT or implicit]
// After: explicit VECTORIZE
VECTORIZE(op[0], op[1], op[2], op[3])
```

**UNROLL range → repeated operations**:

When we say "operations duplicated," it sounds like copy-paste. But that's not what happens. The compiler creates a single SIMD instruction that processes all N elements together. Think of a SIMD register as a box holding 4 numbers; adding two boxes adds all 8 numbers at once.

```text
// Before: UPCAST marks vectorization intent
RANGE(end=3, UPCAST)
      ↓ [pm_pre_expander]
// Step 1: Convert to UNROLL
UNROLL(VCONST([0, 1, 2]))
      ↓ [expander]
// Step 2: Operations expand to handle all positions
// After: operations processed together (not duplicated)
UNROLL([op_at_0, op_at_1, op_at_2])
```

**UNROLL/END/CONTRACT interaction**:
```text
Before: END(STORE(...), [RANGE(UPCAST)])
             ↓ [pm_pre_expander]
Step 1: END(STORE(...), [UNROLL(VCONST([0,1,2,3]))])
             ↓ [expander]
Step 2: END(CONTRACT(STORE(...×4)), [])
```

**Broadcast through AFTER/END**:
```text
// Broadcast VECTORIZE (all elements identical)
AFTER(VECTORIZE([x, x, x, x]), deps) → VECTORIZE([AFTER(x, deps), AFTER(x, deps), ...])
```

**GROUP_REDUCE Handling** (`pm_group_for_reduce`):

GROUP_REDUCE is a special axis type for tensor core reductions:

```text
// Before: REDUCE with GROUP_REDUCE ranges
REDUCE(src, [range(GROUP_REDUCE)])
           ↓ [pm_group_for_reduce]
// After: Shared memory reduction pattern
1. Track upstream LOCAL ranges
2. BUFFERIZE result with group ranges (AddrSpace.LOCAL)
3. INDEX into buffer with transformed ranges
4. Final REDUCE with axes (range_id+100, AxisType.REDUCE)
```

This enables efficient tensor core accumulation via shared memory.

**Morok**: `expand.rs`

---

### Stage 10: Add Local Buffers

> **Stage at a Glance**
>
> **Goal**: Prepare buffers for fast memory (shared / L1)
> **Key Patterns**: Bufferize with locals, extract hints
> **Impact**: Frequently-accessed data stays in fast memory

**What This Does**: Prepares buffers for local memory usage and applies codegen-specific cleanups.

**Why This Matters**: **Local buffers** = fast memory close to the compute unit:
- GPU: Shared memory (LDS) — 100x faster than global memory
- CPU: L1 cache — 10x faster than main memory

The compiler moves frequently-accessed data to local buffers, similar to keeping important files on your desktop instead of a network drive.

**Pattern**: `pm_add_buffers_local + rangeify_codegen`

| Transform | Purpose |
|-----------|---------|
| `bufferize_to_store` | Convert BUFFERIZE with `allow_locals=true` |
| `get_contiguous` | Extract optimization hints from CONTIGUOUS |
| NOOP removal | Clean up no-op operations |
| Strip arg from STORE | Remove redundant arguments |
| Fix broadcast dtype | Ensure consistent types in broadcasts |

**Morok**: `rangeify/kernel.rs`

---

## Phase 3: Devectorizer

**Goal**: Lower from hardware-agnostic vectors to hardware-specific instructions.

---

### Stage 11: Remove Reduce

> **Stage at a Glance**
>
> **Goal**: Convert declarative REDUCE to imperative accumulation
> **Key Patterns**: Reduce to accumulator, horizontal reduction
> **Impact**: Maps to hardware reduction instructions

**What This Does**: Converts high-level REDUCE to accumulator pattern.

**Why This Matters**: A declarative "sum these values" needs to become imperative instructions: initialize accumulator, loop, add each value.

**Pattern**: `pm_reduce + gep_pushing`

```text
// Before: declarative reduction
REDUCE(Add, values, range)

// After: imperative accumulation
acc = DEFINE_REG(0.0)
for i in range:
    acc = ADD(acc, values[i])
```

**Horizontal reduction**:

Before we loop through a reduction dimension, we first combine neighboring values. This creates larger reductions that map better to hardware instructions.

```text
Before:  [a, b, c, d, e, f, g, h]  // 8 values
             ↓ [Horizontal reduction]
Step 1:  [a+e, b+f, c+g, d+h]      // 4 partial sums
             ↓ [Accumulator pattern]
After:   acc = acc + (a+e) + (b+f) + (c+g) + (d+h)
```

**GEP pushing** pushes GEP (get element pointer) operations through ALUs for better vectorization:

```text
GEP(ADD(ptr_a, ptr_b), idx) → ADD(GEP(ptr_a, idx), GEP(ptr_b, idx))
```
*Why?* Enables SIMD on the two GEPs (can be computed in parallel).

**WMMA Tensor Core Fusion**:
```text
// Fuse tensor core accumulation inline
WMMA(a, b, c) + add → WMMA(a, b, c + add)
```
This pattern enables efficient FMA-style accumulation on NVIDIA tensor cores.

**Morok**: `devectorize.rs`

---

### Stage 12: Add GPU Dims

> **Stage at a Glance**
>
> **Goal**: Map abstract ranges to GPU thread indices
> **Key Patterns**: Range to SPECIAL replacement
> **Impact**: Enables parallel execution on GPU

**What This Does**: Replaces ranges with GPU thread indices.

**Why This Matters**: GPUs have hard limits: max 1024 threads per block, max 48KB shared memory. If your computation needs 2000 threads, the compiler must split it into multiple blocks. Dimension limiting handles this automatically.

**Pattern**: `pm_add_gpudims`

```text
// Before: abstract range
RANGE(end=256, Global)

// After: GPU-specific
SPECIAL(gidx0)  // global thread index
```

**Mapping**:

| Range Type | GPU Equivalent |
|------------|----------------|
| Global, THREAD | `gidx` (global index) |
| Local, WARP, GROUP_REDUCE | `lidx` (local/workgroup index) |
| Reduce | Loop (no mapping) |

**Dimension Limiting**:

GPUs have hardware limits (e.g., max 1024 threads per block). When ranges exceed these limits, the compiler:

1. **Groups** adjacent dimensions: `[256, 256, 256]` with max `[256, 256]` → `[65536, 256]`
2. **Splits** large dimensions: `[2048]` with max `[1024]` → `[2, 1024]`
3. **Reconstructs** indices via divmod

**Store Masking**:

Global stores that don't use all local dimensions are masked:
```text
// If STORE doesn't use lidx1, mask it:
STORE(INDEX(...), value) → STORE(INDEX(..., gate=(lidx1 == 0)), value)
```
This ensures stores only execute when unused local indices are 0.

**Morok**: `gpudims.rs`

---

### Stage 13: Add Loads

> **Stage at a Glance**
>
> **Goal**: Wrap INDEX operations in explicit LOAD
> **Key Patterns**: Add LOAD, remove redundant loads
> **Impact**: Makes memory operations explicit for codegen

**What This Does**: Wraps INDEX operations in explicit LOAD.

**Why This Matters**: Index operations compute addresses. LOAD actually reads memory. Making this explicit helps the code generator understand what memory accesses are needed.

**Pattern**: `pm_add_loads`

```text
// Before: bare index
INDEX(ptr, i)

// After: explicit load
LOAD(INDEX(ptr, i))
```

Also removes redundant loads from stores (write-only access).

Note: Not all INDEX operations get wrapped in LOAD. Pointer types (already addresses) and image textures (special hardware) use different access methods.

**Morok**: `devectorize.rs`

---

### Stage 14: Devectorize

> **Stage at a Glance**
>
> **Goal**: Convert abstract vectors to match hardware capabilities
> **Key Phases**: 4 coordinated passes
> **Impact**: Vectors work with actual hardware width

**What This Does**: Handles the transition from abstract vectors to hardware operations.

**Why This Matters**: Devectorize uses 4 conceptual phases within a single `graph_rewrite`:

1. **Phase 1**: Create PTRCAT to group consecutive pointer accesses, devectorize ALU/WMMA/buffers, expand vector INDEX → GEP(PTRCAT)
2. **Phase 2**: Move GEP through LOAD/STORE
3. **Phase 3**: Distribute PTRCAT through LOAD/STORE, creating CAT(LOADs), fix image buffers
4. **Phase 4**: Split CAT(LOADs) into smaller chunks matching hardware width

**PTRCAT Construction**:

PTRCAT groups consecutive pointer accesses:
1. Generate individual indexes for each vector element
2. Extract (valid, root_src) → [offsets] mapping
3. Group consecutive offsets by validity and source
4. Create PTRCAT from grouped pointers
5. Return with GEP permutation for correct element order

This reduces memory bus transactions.

**Device-Specific Fold Lengths**:

| Device | Fold Lengths | Notes |
|--------|--------------|-------|
| DSP | 128, 64, 32, 16, 8, 4 | Large vectors for DSP SIMD |
| GPU (float4) | 4, 2 | Standard GPU vectorization |
| GPU (half + ALLOW_HALF8) | 8, 4, 2 | Half precision with env var |
| GPU (AMX) | 16, 8, 4, 2 | Apple AMX support |
| Image | 4 | Fixed for image textures |
| Default | 1 | Scalar fallback |

**Environment Variable**: `DEVECTORIZE`
- `0`: Skip `devectorize` only (keeps `correct_load_store`)
- `1`: Full devectorization (default)
- `≥2`: Skip both `devectorize` and `correct_load_store`

**Pattern**: `devectorize + load_store_folding + correct_load_store + load_store_indexing`

**Split vectorized ALUs**:
```text
// If hardware doesn't support vec4 add
ADD(vec4_a, vec4_b) → [ADD(a[0], b[0]), ADD(a[1], b[1]), ...]
```

**Load/store chunk splitting**: Match hardware memory width.

**Image fixup**: Special handling for image tensor buffers.

**Morok**: `devectorize.rs`

---

### Stage 15: Lower Index Dtype (Tinygrad-specific)

> **Stage at a Glance**
>
> **Goal**: Convert abstract Index type to concrete integers (Tinygrad approach)
> **Key Patterns**: Operation-specific lowering
> **Impact**: Indices use hardware-native integer types

**What This Does**: Converts abstract `Index` type to concrete integers.

**Why This Matters**: The Index type is abstract—hardware doesn't have it. We need to convert to i32 or i64, which the hardware actually supports.

**Pattern**: `pm_lower_index_dtype + load_store_indexing` (Tinygrad)

```text
// Before: abstract index type
idx: Index

// After: concrete type
idx: i32  // or i64
```

**Operation-Specific Lowering**:

Index type lowering is NOT a single cast—each operation type has specific patterns:

| Operation | Before | After |
|-----------|--------|-------|
| Binary ops | `INDEX(ADD(a, b))` | `ADD(CAST(INDEX(a)), CAST(INDEX(b)))` |
| WHERE | `WHERE(c, INDEX(x), INDEX(y))` | `WHERE(c, CAST(INDEX(x)), CAST(INDEX(y)))` |
| RANGE | `RANGE(end: Index)` | Adopts end's concrete dtype |
| SPECIAL | `SPECIAL(gidx)` | Always int32 |
| VECTORIZE | `VECTORIZE(idx...)` | Cast each to scalar concrete type |
| Invalid | `WHERE(idx, Invalid)` | `INDEX(idx, gate=valid)` |

**Note**: The actual pattern matches operations with children cast to index type, performs the operation in concrete types, then casts back to index. The notation above is simplified for clarity.

The `select_dtype()` function determines int32 vs int64:
```text
dtype = i64 if value.overflows(i32) else i32
```

**Double-cast pattern**: Some operations don't support Index types. We cast Index to integer, operate, then cast back: Index → i32 → ADD → i32 → Index. Later stages optimize away redundant casts.

**Morok implementation**: Morok handles Index types differently. Index lowering happens during code generation in LLVM/Cranelift backends (Index → i64), not as a separate stage.

---

## Phase 4: Linearizer

**Goal**: Convert the DAG to a linear instruction sequence.

---

### Stage 16: Post-Index Symbolic

> **Stage at a Glance**
>
> **Goal**: Full symbolic simplification after index lowering
> **Key Patterns**: All symbolic rules (140+)
> **Impact**: Final cleanup before serialization

**What This Does**: Full symbolic simplification after index lowering.

**Why This Matters**: Now that indices are concrete integers (i32/i64), arithmetic can fully simplify. This is the last chance to clean up expressions before linearization.

**Pattern**: `symbolic`

Includes GEP pushing patterns—move address calculations through arithmetic:
```text
Before:  GEP(ADD(arr_a, arr_b), idx)
              ↓ [Push GEP through ADD]
After:   ADD(GEP(arr_a, idx), GEP(arr_b, idx))
```
*Why?* Enables parallel computation of GEPs and may enable downstream vectorization. (Note: The pattern only applies when GEP's dtype and ALU's dtype are NOT pointers.)

---

### Stage 17: Pre-Matcher (Optional)

> **Stage at a Glance**
>
> **Goal**: Backend-specific patterns before decomposition
> **Key Patterns**: Renderer-specific
> **Impact**: Hardware-specific optimizations

**What This Does**: Renderer-specific patterns applied before decomposition.

**Why This Matters**: Each backend can add its own patterns. For example, DSP backends use this to replace generic patterns with DSP-specific SIMD intrinsics. This allows hardware-specific optimizations without changing the generic pipeline.

**Pattern**: `renderer.pre_matcher`

Most backends (CPU, GPU) don't need this. Only specialized hardware uses it.

**Note**: Morok does not currently implement this stage. The `Renderer` trait has only a `decompositor()` method. This is a future enhancement for DSP and other specialized backends.

---

### Stage 18: Decompositions

> **Stage at a Glance**
>
> **Goal**: Rewrite operations the target doesn't support
> **Key Patterns**: Power-of-2, transcendental approximations
> **Impact**: Maps high-level ops to hardware instructions

**What This Does**: Late rewrites for operations the target doesn't support.

**Why This Matters**: Hardware doesn't have every operation. For example, most CPUs don't have a direct `sin` instruction. We approximate it with operations that do exist (addition, multiplication, etc.).

**Pattern**: `symbolic_simple + get_late_rewrite_patterns`

| Pattern | Example | When Used |
|----------|---------|----------|
| `MOD → AND` | `x % 8 → x & 7` | Power-of-2 divisor |
| `MUL → SHL` | `x * 16 → x << 4` | Power-of-2 multiplier |
| `NEG` | `x * -1 → NEG(x)` | When NEG supported |
| `MULACC` | `a * b + c → MULACC(a, b, c)` | When FMA supported |

**Note**: Tinygrad implements additional decomposition patterns (DIV→SHR for power-of-2 division, FDIV→MUL for reciprocal optimization, comparison simplifications including De Morgan's laws, fast integer division via magic numbers). Transcendental function approximations (SIN, EXP, LOG, etc.) are implemented in Morok via the `decompositor()` pathway (see `ir/src/decompositions/transcendentals.rs`).

**Morok**: `optimizer/mod.rs`

---

### Stage 19: Final Rewrite

> **Stage at a Glance**
>
> **Goal**: Prepare for linearization
> **Key Patterns**: CONST vectorization, GEP resolution, END splitting
> **Impact**: Clean representation ready for linearization

**What This Does**: Prepare for linearization.

**Why This Matters**: Some patterns are easier to apply after decomposition. This stage does final cleanup before converting to a linear sequence.

**Pattern**: `pm_decomp + pm_render + extra_matcher + pm_split_ends`

**CONST vectorization**:
```text
// Make vector constants explicit
CONST(1.0) used as vec4 → VECTORIZE(1.0, 1.0, 1.0, 1.0)
```

**CAT to VECTORIZE** (via `gep_pushing` in `symbolic`):
```text
CAT(a, b, c, d) → VECTORIZE(a, b, c, d)
```
CAT cannot be rendered directly; explicit VECTORIZE is required for codegen.

**GEP resolution**: Convert remaining GEP operations.

**Split multi-range ENDs**:
```text
// Before: END closing multiple ranges
END(op, [range_a, range_b])

// After: nested single ENDs
END(END(op, range_a), range_b)
```

**extra_matcher**: Each backend can add its own final patterns. This allows hardware-specific optimizations without changing the generic pipeline.

**Morok**: `devectorize.rs`, `linearize/mod.rs`, `optimizer/mod.rs`

---

### Stage 20: Add Control Flow

> **Stage at a Glance**
>
> **Goal**: Build control flow graph and add range dependencies
> **Key Concept**: Three relationship types (nested, dependent, independent)
> **Impact**: Correct instruction ordering

**What This Does**: Builds the control flow graph and adds range dependencies.

**Why This Matters**: Operations must execute in a valid order. If a load uses a RANGE's value, the RANGE must come first. This stage tracks and enforces these dependencies.

**Pattern**: `pm_add_control_flow` (bottom-up)

```text
// Analyze which END operations depend on which
END(computation, [RANGE_A]) and END(other_computation, [RANGE_B]) are siblings
→ Creates edge: RANGE_B.src += END(computation)

// Add explicit dependency
RANGE_B waits for RANGE_A to complete
```

**Three relationship types**:

| Relationship | Example | Meaning |
|--------------|---------|---------|
| Nested | RANGE_A inside RANGE_B | A must complete before B starts |
| Dependent | LOAD_A uses RANGE_A | RANGE_A must precede LOAD_A |
| Independent | RANGE_X and RANGE_Y don't interact | Can run in parallel |

Bottom-up traversal ensures dependencies flow correctly from leaves to roots.

**Morok**: `schedule/src/linearize/mod.rs`

---

### Stage 21: Linearize

> **Stage at a Glance**
>
> **Goal**: Convert DAG to linear instruction sequence
> **Key Algorithm**: Priority-aware topological sort
> **Impact**: Valid execution order

**What This Does**: Converts the DAG to a linear instruction sequence via priority-aware topological sort.

**Why This Matters**: The graph structure doesn't specify execution order. We need to flatten it while respecting dependencies. Priorities ensure sensible ordering (definitions before uses, loads before computation, stores after).

**Function**: `linearize(sink)`

| Operation | Priority | Why |
|-----------|----------|-----|
| DEFINE_GLOBAL | -20 | Arguments must be defined first |
| DEFINE_VAR | -19 | Variables must be defined first |
| DEFINE_LOCAL | -18 | Allocations first |
| DEFINE_REG | -17 | Registers first |
| CONST | -10 | Constants early for reuse |
| LOAD | -1 | Loads before use |
| END | -5 | Closes ranges |
| STORE | +1 | Stores after computation |
| RANGE | +5 | Ranges open before use |

Lower priority = earlier in sequence. This ensures:
- Definitions come first
- Loads happen before computation
- Stores happen last
- Ranges open before their contents, close after

**Run_count ordering**: Operations are sorted primarily by execution frequency (run_count), then by priority. Operations with lower execution frequency (outside inner loops) are scheduled first, while operations in inner loops (higher run_count) are scheduled later. Example: A CONST executed 100 times appears before a CONST executed 1M times.

**run_count Calculation**:
```text
run_count = prod(int(r.vmax) + 1 for r in u.ranges)
```
This computes how many times an operation executes based on its enclosing ranges.

**Morok**: `schedule/src/linearize/mod.rs`

---

### Stage 22: Cleanup IF/ENDIF

> **Stage at a Glance**
>
> **Goal**: Final cleanup of linear instruction list
> **Key Transformation**: Gated INDEX → IF/STORE/ENDIF
> **Impact**: Handles hardware without predicated stores

**What This Does**: Final cleanup of the linear instruction list.

**Why This Matters**: Some hardware (modern GPUs) supports "predicated stores"—write to memory only if condition is true. Older hardware doesn't. For those, we wrap store in an IF statement. This stage ONLY runs when hardware lacks predicated store support.

**Pattern**: `pm_linearize_cleanups` (via `line_rewrite`, not `graph_rewrite`)

```text
// Gated INDEX in STORE becomes conditional store
STORE(INDEX(ptr, idx, valid=cond), value)
→ IF(cond) { STORE(INDEX(ptr, idx), value) } ENDIF
```

**Note**: This stage uses `line_rewrite` instead of `graph_rewrite` because it operates on the already-linearized instruction list rather than a DAG.

At this point, the instruction list is ready for code generation.

**Morok**: `schedule/src/linearize/mod.rs` (predicated stores path)

---

## Worked Example: Tracing Through All 22 Stages

Let's trace `c = a + b` (where a, b are [100, 100] tensors) through the pipeline.

### Initial Tensor Graph
```
[ADD]
├── [BUFFER(a)] : Float32
└── [BUFFER(b)] : Float32
```

### After Stage 1: Early Movement Ops
(No change—no movement ops in this example)

### After Stage 2: Load Collapse
(No change—no reductions in this example)

### After Stage 3: Split Ranges
(No change—no modulo operations)

### After Stage 4: Initial Symbolic
(No change—no simplification needed)

### After Stage 5: Simplify Ranges
(No change—no adjacent ranges yet)

### After Stage 6: Split Store
(Not applicable—GPU backend)

### After Stage 7: Apply Opts
Optimization actions applied:
- UPCAST j dimension by 4 (vectorization)
- LOCAL for input buffers (if beneficial)

### After Stage 8: Post-Opt Symbolic
No changes—symbolic already clean.

### After Stage 9: Expander
UPCAST → UNROLL → CONTRACT:
```
[VECTORIZE]
├── [ADD]
│   ├── [LOAD(a)]
│   │   └── [INDEX]
│   │       ├── [BUFFER(a)]
│   │       ├── [RANGE(i, Global, 0..100)]
│   │       └── [UNROLL(VCONST([0,1,2,3]))]  // Converted from RANGE(j, UPCAST)
│   └── [LOAD(b)]
│       └── [INDEX]
│           ├── [BUFFER(b)]
│           ├── [RANGE(i)]  // Same RANGE via hash consing
│           └── [UNROLL(VCONST([0,1,2,3]))]  // Same UNROLL via hash consing
```

### After Stage 10: Add Local Buffers
(If LOCAL opt was chosen)

### After Stage 11: Remove Reduce
(No change—no reductions)

### After Stage 12: Add GPU Dims
```
[SPECIAL(gidx0)] : Index  // replaces RANGE(i)
```

### After Stage 13: Add Loads
(No change—loads already present)

### After Stage 14: Devectorize
Vector split to match hardware width:
```
[VECTORIZE] : <4 x Float32>
├── [ADD(a[0], b[0])]
├── [ADD(a[1], b[1])]
├── [ADD(a[2], b[2])]
└── [ADD(a[3], b[3])]
```

### After Stage 15: Lower Index Dtype
```
[SPECIAL(gidx0)] : i32  // concrete type
```

### After Stage 16: Post-Index Symbolic
No changes needed.

### After Stage 17: Pre-Matcher
(No patterns for standard backends)

### After Stage 18: Decompositions
No decompositions needed—all ops supported.

### After Stage 19: Final Rewrite
No changes needed.

### After Stage 20: Add Control Flow
Dependencies tracked—no issues.

### After Stage 21: Linearize
Linear instruction sequence (simplified):
```
1. DEFINE_GLOBAL(0)  // Output buffer c
2. DEFINE_GLOBAL(1)  // Input buffer a
3. DEFINE_GLOBAL(2)  // Input buffer b
4. RANGE(i, 0..100, Global)  // gidx0
5. RANGE(j, 0..25, Loop)  // Unrolled /4
6. LOAD(a, i, j*4+0)  // Vector load
7. LOAD(b, i, j*4+0)  // Vector load
8. ADD(vec_a, vec_b)  // Vector add
9. STORE(c, i, j*4+0, result)
10. END(RANGE(j))
11. END(RANGE(i))
```

### After Stage 22: Cleanup IF/ENDIF
No changes needed—no gated stores.

**Result**: Ready for code generation! The LLVM/CUDA/other backend will compile this to actual machine code.

---

## Pattern Application Strategy

Each stage uses one of two rewrite strategies:

**Top-down** (default): Process parents before children. Use when transformations create new matchable subterms.

**Bottom-up**: Process children before parents. Use when child state affects parent matching (stages 1, 20).

Both iterate to fixpoint—patterns fire until no more match.

---

## Debugging the Pipeline

When a kernel produces wrong results, the bug lives in one of these 22 stages. Use environment variables to extract IR at each stage:

```bash
# See IR after each transformation
MOROK_DEBUG=ir cargo test failing_test
```

### Quick Reference

| Symptom | Likely Stages | What to Check |
|---------|---------------|---------------|
| Wrong values in output | 4, 9, 11, 18 | Symbolic simplification, expansion, devectorization |
| Slow performance | 7, 9, 14, 21 | Optimization, expansion, devectorization, linearization |
| Crashes/panics | 11, 12 | Reduce, GPU dims |
| Wrong loop count | 3, 5, 12 | Split ranges, simplify ranges, GPU dims |
| Missing vectorization | 9, 14 | Expander, devectorize |

### Common Issues

1. **Stage 3-4**: Range splitting/symbolic may lose constraints
2. **Stage 9**: Expansion order affects vectorization correctness
3. **Stage 11**: Accumulator initialization must match reduction identity
4. **Stage 14**: Hardware width mismatch—check vector fold length
5. **Stage 18**: Missing decomposition—check supported_ops list for backend
6. **Stage 21**: Priority bugs cause data races—verify dependencies

---

## Summary

The 22-stage pipeline transforms tensor expressions into machine code through systematic refinement:

1. **Stages 1-7**: Make iteration explicit, optimize ranges
2. **Stages 8-10**: Expand optimization primitives
3. **Stages 11-15**: Lower to hardware-specific operations
4. **Stages 16-22**: Serialize to executable instructions

Each stage has a single responsibility. Each builds on the last. The result: high-level tensor code runs at near-optimal speed on diverse hardware.

---

## Tinygrad vs Morok: Architectural Differences

This chapter describes the "ideal" 22-stage pipeline based on Tinygrad's implementation. Morok follows the same overall design, but some architectural differences exist. These are deliberate design choices, not bugs.

### Valid Architectural Differences

| Stage | Tinygrad | Morok | Notes |
|--------|-----------|-------|--------|
| 1: Early Movement Ops | Moves movement ops through AFTER/END wrappers | Removes movement ops during bufferization | Both approaches achieve functional equivalence; Morok's is cleaner |
| 4: Initial Symbolic | GEP pushing in `sym` | GEP pushing only in Stage 8 and 11 | Morok may miss some GEP optimization opportunities; consider adding to `symbolic_simple()` |
| 11: Remove Reduce | Includes distributive pattern `(x+y).reduce() → x.reduce() + y.reduce()` | Missing distributive pattern | Morok's `reduce_collapse()` is a subset; no correctness impact, potential performance difference |
| 15: Index Dtype Lowering | Separate stage converts Index → i32/i64 | Index lowered at codegen time (Index → i64) | Both valid; Morok trades runtime cost (always i64) for pipeline simplicity |
| 18: Decompositions | Transcendentals in Stage 18 | Transcendentals via `decompositor()` | Both valid; Morok's approach separates generic patterns from backend-specific decompositions |
| 19: Final Rewrite | `pm_render` applied in Stage 19 | `pm_render` applied during codegen | Functionally equivalent; Morok applies rendering transformations during rendering |

### Tinygrad-Only Patterns

Morok intentionally does not implement these Tinygrad-specific patterns:

| Pattern | Purpose | Why Morok Doesn't Need It |
|----------|-----------|-----------------------------|
| `to_bufferview` | Avoid disk buffer copies for DISK/TINYFS devices | Morok doesn't support DISK/TINYFS; in-memory backends don't need this |

### Morok Enhancements

Morok has some patterns/enhancements not in Tinygrad:

| Enhancement | Location | Purpose |
|-------------|---------|---------|
| Nested INDEX flattening with identical indices | `movement_op_patterns()` (lines 699-713) | Removes redundant `INDEX(INDEX(ptr, [i]), [i])` |
| CAT → VECTORIZE | `pm_render` | Converts CAT to explicit VECTORIZE (can't render CAT directly) |
| PTRCAT([x]) unwrap | `pm_render` | Removes single-element PTRCAT wrappers |

---

## Glossary

| Term | Simple Definition | Example |
|------|------------------|---------|
| **Accumulator** | Variable holding running total | `acc = acc + value` (in reduction) |
| **Axis** | One dimension of a tensor | Shape [100, 200] has 2 axes |
| **AxisType** | How a loop executes | Global=parallel, Reduce=accumulate |
| **Buffer** | Allocated memory holding data | A tensor's data lives in a buffer |
| **Bufferize** | Store result in memory instead of computing on-demand | Materialize intermediate value |
| **CONTRACT** | Combine multiple values into one vector | `[a, b, c, d] → vec4(a,b,c,d)` |
| **Devectorize** | Split vectors to match hardware | `vec8 → vec4, vec4` |
| **Divmod** | Division and remainder operations | `x // 7, x % 7` |
| **Fixpoint** | When applying patterns no longer changes anything | Patterns fire until fixpoint |
| **GEP** | Get Element Pointer—compute address from indices | `arr[i][j] → base + i*stride + j` |
| **Hash consing** | Reuse identical expressions | `ADD(x, 0) + ADD(x, 0)` shares memory |
| **Index** | Integer type for array indices | i32 or i64, depending on device |
| **Load** | Read from memory | `value = arr[i]` |
| **Pattern** | Find-and-replace rule for code | `ADD(x, 0) → x` |
| **Predicated store** | Write to memory conditionally | Write if valid else skip |
| **Range** | Loop iteration specification | `for i in 0..100` |
| **Reduction** | Combine many values into one | Sum, max, min |
| **Store** | Write to memory | `arr[i] = value` |
| **Symbolic** | Simplify using algebra rules | `(x/4)*4 → x` (when `x%4=0`) |
| **Tensor core** | Hardware for fast matrix multiply | NVIDIA GPUs only |
| **Topological sort** | Order nodes respecting dependencies | A before B if B uses A's result |
| **UNROLL** | Expand one op into multiple positions | `x → [x_0, x_1, x_2, x_3]` |
| **UPCAST** | Mark intent to vectorize | `RANGE(0..4, UPCAST)` |
| **Vectorize** | Process multiple values together | SIMD: add 4 numbers at once |
| **WHERE** | Conditional selection | `WHERE(cond, x, y) = x if cond else y` |
