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

Svod has no GEP op—addressing is `INDEX(STACK(...))`—so Tinygrad's `gep_pushing`
has no counterpart here. The nearest analogue is `alu_vectorize_reorder_patterns`:
```text
Before:  ADD(STACK(x, x, x, x), STACK(y, y, y, y))
              ↓ [Reorder ALU over STACK]
After:   STACK(ADD(x, y), ADD(x, y), ADD(x, y), ADD(x, y))
```
*Why?* Enables constant folding and scalar optimization on the collapsed operation. That rule lives in tier-3 `sym()`, so it already fired at Stage 14, not here.

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

**Note**: Svod has no `pre_matcher`. Backend hooks live on the `svod_device::device::Renderer` trait (`device/src/device.rs`): `decompositor()`, `extra_matcher()`, `pre_isel_matcher()` and `isel_matcher()`. The last two run at the PROGRAM boundary, between Stages 20 and 21, not before decomposition. (`svod_codegen::traits::Renderer` is a separate, narrower trait with `render()`, `backend_name()` and `decompositor()`.)

---

## Stage 18: Decompositions

> **Stage at a Glance**
>
> **Goal**: Rewrite operations the target doesn't support
> **Key Patterns**: Power-of-2, transcendental approximations
> **Impact**: Maps high-level ops to hardware instructions

**What This Does**: Late rewrites for operations the target doesn't support.

**Why This Matters**: Hardware doesn't have every operation. For example, most CPUs don't have a direct `sin` instruction. We approximate it with operations that do exist (addition, multiplication, etc.).

**Pattern**: `early_decomposition_patterns() + get_late_rewrite_patterns() + get_transcendental_patterns()` (plus `renderer.decompositor()` when the backend supplies one). `early_decomposition_patterns()` itself starts with `symbolic_simple()`.

Note: `pm_split_ends()` is not part of this pass—it is folded into the Stage 19 matcher and runs again at the head of Stage 20.

| Pattern | Example | When Used |
|----------|---------|----------|
| `MOD → AND` | `x % 8 → x & 7` | Power-of-2 divisor |
| `MUL → SHL` | `x * 16 → x << 4` | Power-of-2 multiplier |
| `DIV → SHR` | `x / 8 → x >> 3` | Power-of-2 divisor (C-style CDIV) |
| `FDIV → MUL` | `x / 2.0 → x * 0.5` | Float constant divisor |
| `NEG` | `x * -1 → NEG(x)` | When NEG supported |
| `MULACC` | `a * b + c → MULACC(a, b, c)` | When FMA supported |
| Fast integer division | `x // 7 → (x * M) >> S` | Non-power-of-2 divisor |
| De Morgan's law | `(!x) & (!y) → !(x \| y)` | Boolean simplification (AND-of-NOTs only) |
| Comparison negations | `!(x < c) → (c-1) < x` | Integer comparisons |

Transcendental approximations (EXP2, LOG2, SIN, …) come from `get_transcendental_patterns()` (`ir/src/decompositions/mod.rs`, implementations in `ir/src/decompositions/transcendentals.rs`). They are enabled per operation when the renderer lacks the instruction, or for every operation when `TRANSCENDENTAL=2`. The optional `Renderer::decompositor()` hook adds backend-specific rules on top; no in-tree backend uses it.

**Svod**: `optimizer/mod.rs`

---

## Stage 19: Final Rewrite

> **Stage at a Glance**
>
> **Goal**: Prepare for linearization
> **Key Patterns**: Weak-cast commit, renderer rewrites, END splitting
> **Impact**: Clean representation ready for linearization

**What This Does**: Prepare for linearization.

**Why This Matters**: Some patterns are easier to apply after decomposition. This stage does final cleanup before converting to a linear sequence.

**Pattern**: `pm_commit_weak() + pm_cast_weak() + pm_decomp` (the Stage 18 decompositions), plus `renderer.extra_matcher()` and `pm_split_ends()`—all summed into one matcher. `pm_remove_invalid()` and `add_implicit_barriers()` then run as separate passes.

Note: `extra_matcher` and `pm_split_ends` are part of this combined matcher, not separate passes. Svod has no CONST-vectorization or GEP-resolution step; Tinygrad's `pm_render` has no counterpart here.

**Split multi-range ENDs**:
```text
// Before: END closing multiple ranges
END(op, [range_a, range_b])

// After: nested single ENDs
END(END(op, range_a), range_b)
```

The ranges are sorted descending by `(axis_id, axis_type.priority())`, so the innermost END is built first. Void/Bool "backedge" sources are partitioned out and re-attached to the outermost END, with the original tag preserved.

**extra_matcher**: Each backend can add its own final patterns. This allows hardware-specific optimizations without changing the generic pipeline.

**Svod**: `optimizer/mod.rs`, `linearize/mod.rs`

---

## Stage 20: Add Control Flow

> **Stage at a Glance**
>
> **Goal**: Build control flow graph and add range dependencies
> **Key Concept**: Three relationship types (nested, dependent, independent)
> **Impact**: Correct instruction ordering

**What This Does**: Builds the control flow graph and adds range dependencies.

**Why This Matters**: Operations must execute in a valid order. If a load uses a RANGE's value, the RANGE must come first. This stage tracks and enforces these dependencies.

**Pattern**: `pm_add_control_flow` (bottom-up), preceded by a second `pm_split_ends` run

```text
// Analyze which END operations depend on which
END(computation, [RANGE_A]) and END(other_computation, [RANGE_B]) are siblings
→ Creates edge: RANGE_B.src += END(computation)

// Add explicit dependency
RANGE_B waits for RANGE_A to complete
```

**Three relationship types**:

| Relationship | Condition | Meaning |
|--------------|-----------|---------|
| Nested | END_A is a dep of END_B **and** RANGE_B is a dep of END_A | A's loop sits inside B's, so A closes before B closes |
| Dependent | END_A is a dep of END_B without that nesting | B's loop must be emitted after A's |
| Independent | Neither END depends on the other | Order is free; can run in parallel |

Bottom-up traversal ensures dependencies flow correctly from leaves to roots.

**Svod**: `schedule/src/linearize/mod.rs`, `schedule/src/linearize/cfg_context.rs`

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
| PARAM | -20 | Kernel arguments (and symbolic variables) must be defined first; ties break on the parameter slot |
| BUFFER | -18 | Allocations first |
| BUFFER (`AddrSpace::Local`) | -17 | Local allocations right after the global ones |
| END | -5 | Closes ranges |
| LOAD | -1 | Loads before use |
| Everything else (CONST, ALU, …) | 0 | Sinks next to its consumer |
| STORE | +1 | Stores after computation |
| RANGE | +5 | Ranges open before use |

Lower priority = earlier in sequence. This ensures:
- Definitions come first
- Loads happen before computation
- Stores happen last
- Ranges open before their contents, close after

**Run_count ordering**: Operations are sorted primarily by execution frequency (run_count), then by priority, then by the PARAM slot and the tuplize rank. Operations with lower execution frequency (outside inner loops) are scheduled first, while operations in inner loops (higher run_count) are scheduled later. Example: A CONST executed 100 times appears before a CONST executed 1M times.

**run_count Calculation**:
```text
run_count = prod(int(r.vmax) + 1 for r in u.in_scope_ranges())
```
This computes how many times an operation executes based on its enclosing in-scope ranges; a range whose `vmax` isn't a concrete integer contributes 1.

**Svod**: `linearize()` in `schedule/src/linearize/linearize.rs`

---

## Stage 22: Cleanup IF/ENDIF

> **Stage at a Glance**
>
> **Goal**: Final cleanup of linear instruction list
> **Key Transformation**: Gated STORE → IF/STORE/ENDIF
> **Impact**: Handles hardware without predicated stores

**What This Does**: Final cleanup of the linear instruction list.

**Why This Matters**: Some hardware (modern GPUs) supports "predicated stores"—write to memory only if condition is true. Older hardware doesn't. For those, we wrap store in an IF statement. This stage is only needed by backends that lack predicated store support; LLVM, CUDA and Metal handle the gate natively, so `linearize_with_cfg()` does not run it.

**Pattern**: `line_rewrite_cleanups` (via `line_rewrite`, not `graph_rewrite`)

```text
// Gated STORE becomes a conditional store
STORE(INDEX(ptr, idx), value, gate=cond)
→ IF(cond) { STORE(INDEX(ptr, idx), value) } ENDIF
```

**Note**: This stage uses `line_rewrite` instead of `graph_rewrite` because it operates on the already-linearized instruction list rather than a DAG.

At this point, the instruction list is ready for code generation.

**Svod**: `line_rewrite_cleanups()` in `schedule/src/linearize/mod.rs`
