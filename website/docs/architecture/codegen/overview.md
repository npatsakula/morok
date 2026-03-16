---
sidebar_label: Overview
---

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

### Additional Passes

Several passes run between the numbered stages and don't have their own stage number:

| Pass | Between Stages | Purpose |
|------|---------------|---------|
| `linearize_multi_index` | Before Stage 8 | Flatten multi-dimensional indices to linear offsets |
| `pm_bool_devectorize` | 14–15 | Handle boolean vector patterns |
| `pm_reduce_devectorize` | 14–15 | Handle vector reductions (K-vec, bool, horizontal) |
| `merge_sibling_ends` | 14–15 | Merge adjacent END operations |
| `pm_float_decomp` | Post-opt | Decompose floating-point operations |
| `bool_storage_patterns` | Post-opt | Convert bool ↔ uint8 for memory operations |
