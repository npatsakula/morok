---
sidebar_label: Phase 4 — Linearizer
---

# Phase 4: Linearizer

**Goal**: Convert the DAG to a linear instruction sequence.

---

## Stage 16: Post-Index Symbolic

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

## Stage 17: Pre-Matcher (Optional)

> **Stage at a Glance**
>
> **Goal**: Backend-specific patterns before decomposition
> **Key Patterns**: Renderer-specific
> **Impact**: Hardware-specific optimizations

**What This Does**: Renderer-specific patterns applied before decomposition.

**Why This Matters**: Each backend can add its own patterns. For example, DSP backends use this to replace generic patterns with DSP-specific SIMD intrinsics. This allows hardware-specific optimizations without changing the generic pipeline.

**Pattern**: `renderer.pre_matcher`

Most backends (CPU, GPU) don't need this. Only specialized hardware uses it.

**Note**: Morok does not currently implement this stage. The `Renderer` trait has `render()`, `backend_name()`, and `decompositor()` methods, but no `pre_matcher` support yet. This is a future enhancement for DSP and other specialized backends.

---

## Stage 18: Decompositions

> **Stage at a Glance**
>
> **Goal**: Rewrite operations the target doesn't support
> **Key Patterns**: Power-of-2, transcendental approximations
> **Impact**: Maps high-level ops to hardware instructions

**What This Does**: Late rewrites for operations the target doesn't support.

**Why This Matters**: Hardware doesn't have every operation. For example, most CPUs don't have a direct `sin` instruction. We approximate it with operations that do exist (addition, multiplication, etc.).

**Pattern**: `symbolic_simple() + get_late_rewrite_patterns()`

Note: `pm_render()` and `pm_split_ends()` are not part of this combined pass—they run separately in Stage 19.

| Pattern | Example | When Used |
|----------|---------|----------|
| `MOD → AND` | `x % 8 → x & 7` | Power-of-2 divisor |
| `MUL → SHL` | `x * 16 → x << 4` | Power-of-2 multiplier |
| `DIV → SHR` | `x // 8 → x >> 3` | Power-of-2 divisor |
| `FDIV → MUL` | `x / 2.0 → x * 0.5` | Float constant divisor |
| `NEG` | `x * -1 → NEG(x)` | When NEG supported |
| `MULACC` | `a * b + c → MULACC(a, b, c)` | When FMA supported |
| Fast integer division | `x // 7 → (x * M) >> S` | Non-power-of-2 divisor |
| De Morgan's laws | `(!x) & (!y) → !(x \| y)` | Boolean simplification (both directions) |
| Comparison negations | `!(x < c) → (c-1) < x` | Integer comparisons |

Transcendental function approximations (SIN, EXP, LOG, etc.) are implemented via the `decompositor()` pathway (see `ir/src/decompositions/transcendentals.rs`).

**Morok**: `optimizer/mod.rs`

---

## Stage 19: Final Rewrite

> **Stage at a Glance**
>
> **Goal**: Prepare for linearization
> **Key Patterns**: CONST vectorization, GEP resolution, END splitting
> **Impact**: Clean representation ready for linearization

**What This Does**: Prepare for linearization.

**Why This Matters**: Some patterns are easier to apply after decomposition. This stage does final cleanup before converting to a linear sequence.

**Pattern**: `symbolic_simple() + get_late_rewrite_patterns() + pm_render()`

Note: `extra_matcher` and `pm_split_ends` run separately, not as part of this combined pass.

**CONST vectorization**:
```text
// Make vector constants explicit
CONST(1.0) used as vec4 → VECTORIZE(1.0, 1.0, 1.0, 1.0)
```

**CAT to VECTORIZE** (via `pm_render`):
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

## Stage 20: Add Control Flow

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
| Dependent | END_A and END_B are siblings | END_B must wait for END_A (sibling dependency) |
| Independent | RANGE_X and RANGE_Y don't interact | Can run in parallel |

Bottom-up traversal ensures dependencies flow correctly from leaves to roots.

**Morok**: `schedule/src/linearize/mod.rs`

---

## Stage 21: Linearize

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
| CONST | -10 | Constants early for reuse (Morok extension; Tinygrad defaults to 0) |
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

## Stage 22: Cleanup IF/ENDIF

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
