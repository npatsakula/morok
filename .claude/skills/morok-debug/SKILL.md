---
name: morok-debug
description: Debug Morok tensor pipeline issues by extracting IR at each stage, visualizing UOp trees, and comparing with Tinygrad. Use when tests fail, produce wrong results, or crash.
---

# Morok Pipeline Debugging

## Three-Step Analysis

There are three places where errors can occur:
  - Frontend: we can create incorrect IR.
  - Transformation pipeline: we can incorrectly transform IR between stages.
  - Codegen: we can incorrectly generate target IR.
Most of the issues are in the transformation pipeline, and unfortunately it's the hardest
to debug.

The first step during investigation is to isolate the place where the error occurs. This
will allow you to simplify the investigation and reduce the context. You can do it by
extracting IR from an operation (`tensor.uop().tree()`); by extracting IR before codegen;
by extracting kernel code before execution.

Sometimes it's hard to understand if IR is correct, but we have two sources of information:
  - Compare it with Tinygrad IR for the same code: use Python code; they should be identical.
  - Read the @book/src/path-of-the-uop.md to understand if it's correct.
  
### Step 1: Extract all pipeline stages
```bash
# Preferred: use the extraction script (test must have #[tracing_test::traced_test])
./scripts/extract-ir.sh test_name -p morok-tensor -o /tmp/debug_ir.txt
```

### Step 2: Check LLVM IR
```bash
RUST_LOG=morok_codegen::llvm::text=debug cargo test test_name -- --nocapture 2>&1 | rg 'linearized node'
```

### Step 3: Compare with Tinygrad
Compare with Tinygrad's output using the `/tinygrad-debug` skill to isolate broken patterns.

---

## Stage-by-Stage Tree Extraction

### Quick Reference Table

Tracing field: `uop.tree` for rangeify, `ast.optimized` for optimizer.
All markers are in `schedule/src/rangeify/transforms.rs` and `schedule/src/optimizer/mod.rs`.

| Phase | Stage | Field | Debug Message |
|-------|-------|-------|---------------|
| **RANGEIFY** | 0 | `uop.tree` | `Stage 0: range assignment complete` |
| | — | `uop.tree` | `early rewrites complete` |
| | — | `uop.tree` | `split reduceops complete` |
| | 1 | `uop.tree` | `Stage 1: rangeify + movement ops complete` |
| | — | `uop.tree` | `buffer folding + dead axis removal complete` |
| | — | `uop.tree` | `reshape to scalar complete` |
| | 2a | `uop.tree` | `Stage 2a: load collapse complete` |
| | 2b | `uop.tree` | `Stage 2b: reduction simplify complete` |
| | 3a | `uop.tree` | `Stage 3a: split ranges complete` |
| | 3b | `uop.tree` | `Stage 3b: flatten ranges complete` |
| | 4 | `uop.tree` | `Stage 4: initial symbolic complete` |
| | 5 | `uop.tree` | `Stage 5: simplify ranges complete` |
| | 6 | `uop.tree` | `Stage 6: split store complete` *(CPU only)* |
| | 7 | `uop.tree` | `Stage 7: buffer removal complete` |
| | 7b | `uop.tree` | `Stage 7b: buffer limit enforcement complete` *(conditional)* |
| **OPTIMIZER** | 8 | `ast.optimized` | `Stage 8: after post-opt symbolic` |
| | 9 | `ast.optimized` | `Stage 9: after pre_expand` |
| | 10 | `ast.optimized` | `Stage 10: after add local buffers` |
| | — | `ast.optimized` | `after pm_reduce` |
| | — | `ast.optimized` | `after pm_add_gpudims` |
| | — | `ast.optimized` | `after pm_pushing_patterns` |
| | — | `ast.optimized` | `after pm_add_loads` |
| | — | `ast.optimized` | `after devectorize` |
| | — | `ast.optimized` | `after pm_bool_devectorize` |
| | — | `ast.optimized` | `after pm_reduce_devectorize` |
| | — | `ast.optimized` | `after pm_lower_index_dtype` |
| | — | `ast.optimized` | `after post-index symbolic` |
| | 18 | `ast.optimized` | `Stage 18: after pm_decomp` |
| | 19 | `ast.optimized` | `Stage 19: after pm_render` |
| | — | `ast.optimized` | `after bool_storage_pattern` |
| **LINEARIZER** | 20-22 | — | *No tree tracing (linear instruction lists)* |

### Extract All Stages at Once (Recommended)

Use `scripts/extract-ir.sh` to extract all pipeline stage trees into a single readable file:

```bash
# Basic: extract IR for a specific test
./scripts/extract-ir.sh test_sum_axis1_value -p morok-tensor

# With custom output file
./scripts/extract-ir.sh test_argmax_value_1d -p morok-tensor -o /tmp/argmax_ir.txt
```

**Prerequisite**: the target test must have `#[tracing_test::traced_test]` attribute.

The script produces a structured output file with:
- **RANGEIFY PHASE** section: origin input tree + Stages 0-7b (single pass, pre-kernel-split)
- **KERNEL N** sections: initial optimizer input + Stages 8-19 + generated C code (one section per kernel)

The script handles:
- ANSI color stripping
- Unescaping `\n` from tracing's quoted field values back to real newlines
- Multi-kernel detection via "Stage 8:" boundary markers
- Correct extraction of `origin.tree` and `ast.initial` span fields

**Pipeline structure** (why the output is organized this way):
- `rangeify_with_map` runs **once** for the entire SINK before kernel splitting
- `apply_post_optimization_with_renderer` runs **once per kernel** after splitting
- Stages 20-22 (linearizer) produce linear instruction lists, not trees — no tree tracing

### Method 1: In-Code Tree Extraction

Add temporary debugging code directly in your test or tensor operation:

```rust
use morok_ir::prelude::*;

// After each pipeline stage
println!("--- After Stage N ---");
println!("{}", uop.tree());  // Compact tree with back-references
// println!("{}", uop.tree_full());  // Full tree expanding all nodes
```

**Note**: You'll need to manually insert these print statements at the appropriate stage location in the pipeline code (`schedule/src/rangeify/transforms.rs`, `schedule/src/optimizer/mod.rs`, etc.).

### Method 2: Programmatic IR Extraction

For debugging existing tests, use the tensor `prepare()` API:

```rust
let plan = tensor.prepare().expect("prepare should succeed");

for kernel in plan.kernels() {
    println!("--- {} ({}) ---", kernel.entry_point, kernel.device);
    println!("{}", kernel.code);  // LLVM IR or device code
}
```

See `tensor/src/test/unit/matmul.rs:180` for a complete example.

### Method 3: Manual Tracing with rg

For quick single-stage checks without the full extraction script:

```bash
# Extract IR after Stage 0 (Rangeify)
RUST_LOG=morok_schedule::rangeify::transforms=debug cargo test test_name -- --nocapture 2>&1 | rg 'range assignment complete'

# Extract IR after a specific optimizer stage
RUST_LOG=morok_schedule::optimizer=debug cargo test test_name -- --nocapture 2>&1 | rg 'Stage 19: after pm_render'
```

**Note**: tree content is embedded inline with literal `\n` in the tracing field value. Use `scripts/extract-ir.sh` for readable output.

### Single-Stage Quick Checks

For a quick check of one stage without the full extraction script, use `rg` directly.
The tree content is inlined with literal `\n` in the field value, so it reads as a single line.

```bash
# Rangeify stages (target: morok_schedule::rangeify::transforms)
RUST_LOG=morok_schedule::rangeify::transforms=debug cargo test test_name -- --nocapture 2>&1 | rg 'Stage 0:'

# Optimizer stages (target: morok_schedule::optimizer)
RUST_LOG=morok_schedule::optimizer=debug cargo test test_name -- --nocapture 2>&1 | rg 'Stage 19:'
```

### Linking to Documentation

Each stage corresponds to a chapter in `book/src/architecture/path-of-the-uop.md`:

- **Stages 1-7**: See "Phase 1: Rangeify" section
- **Stages 8-10**: See "Phase 2: Expander" section  
- **Stages 11-15**: See "Phase 3: Devectorizer" section
- **Stages 16-22**: See "Phase 4: Linearizer" section

When you identify which stage has the issue, read the corresponding section to understand the transformation rules and expected behavior.

---

## Enabling Traces in Tests

### Why traces don't appear

`RUST_LOG` sets a filter level but requires a tracing subscriber. Tests don't have one by default.

### Add traced_test attribute
```rust
#[test]
#[tracing_test::traced_test]
fn test_my_failing_test() {
    let _guard = test_setup();  // Required for cache isolation
    // ... test code
}
```

This registers a global subscriber that captures all trace!/debug! logs.

### When traced_test is needed
- Complex code paths (rangeify, kernel splitting, scheduling)
- Multi-threaded execution
- When you need to assert on log contents

### Run with output visible
```bash
cargo test test_name -- --nocapture
```

## UOp Visualization

### In code
```rust
use morok_ir::prelude::*;

// Compact tree with back-references for shared nodes
println!("{}", uop.tree());

// Full tree expanding all nodes
println!("{}", uop.tree_full());
```

### Example output
```
[42] STORE : Void
├── [10] DEFINE_GLOBAL(0) : Ptr<Float32>
├── [35] INDEX : Ptr<Float32>
│   ├── [10] → (see above)
│   └── [30] RANGE(0, Reduce) : Index
└── [40] REDUCE(Add) : Float32
    └── [35] → (see above)
```

## Programmatic LLVM IR Extraction

### Using render() API
```rust
use morok_codegen::llvm::text::render;

let rendered = render(&uop_graph, Some("my_kernel"))?;

// Access LLVM IR as string
println!("{}", rendered.code);

// Also available:
// rendered.name - kernel name
// rendered.buffer_args - buffer arguments
// rendered.var_names - variable names
```

### From tensor (like test_print_matmul_ir)
```rust
let plan = tensor.prepare().expect("prepare should succeed");

for kernel in plan.kernels() {
    println!("--- {} ({}) ---", kernel.entry_point, kernel.device);
    println!("{}", kernel.code);  // LLVM IR or device code
}
```

## RUST_LOG Targets

| Target | Information |
|--------|-------------|
| `morok_schedule::rangeify::transforms=debug` | Rangeify stages 0-7b (`uop.tree` field) |
| `morok_schedule::optimizer=debug` | Optimizer stages 8-19 (`ast.optimized` field) |
| `morok_schedule::expand=debug` | UNROLL/UPCAST expansion |
| `morok_schedule::devectorize=debug` | Memory access optimization |
| `morok_schedule::linearize=debug` | Linearization passes |
| `morok_codegen::llvm::text=debug` | LLVM rendering (IR generation) |
| `morok_ir::pattern::simplified=trace` | Pattern matching details |

The `scripts/extract-ir.sh` script sets all relevant targets automatically.

## Common Debug Scenarios

### Wrong numerical result

1. Extract IR before/after optimization
2. Check if UNROLL/UPCAST expansion is correct
3. Compare vector widths with expected
4. Check LLVM IR for correct horizontal reduce

### SIGSEGV / Memory error

1. Check buffer sizes and indices
2. Look for mismatched vector widths
3. Check CAT expansion creating oversized vectors

### Pattern not matching

```bash
RUST_LOG=morok_ir::pattern::simplified=trace cargo test test_name 2>&1 | rg 'pattern'
```

## Better Op Labels in Tracing

### Use as_ref() not discriminant
```rust
// GOOD: Clean variant name
tracing::debug!(op = uop.op().as_ref(), "processing");
// Output: op="Binary"

// BAD: Opaque discriminant
tracing::debug!(op = ?std::mem::discriminant(uop.op()), "processing");
// Output: op=Discriminant(...)
```

## Programmatic LLVM IR Extraction

### Using render() API
```rust
use morok_codegen::llvm::text::render;

let rendered = render(&uop_graph, Some("my_kernel"))?;

// Access LLVM IR as string
println!("{}", rendered.code);
```

## Key Files

| File | Purpose |
|------|----------|
| `scripts/extract-ir.sh` | Pipeline IR extraction script |
| `ir/src/uop/tree.rs` | UOp tree visualization, format_node() |
| `ir/src/op.rs` | Op enum with AsRefStr derive |
| `schedule/src/rangeify/transforms.rs` | Rangeify pipeline, Stages 0-7b (`uop.tree` tracing) |
| `schedule/src/optimizer/mod.rs` | Post-optimization, Stages 8-19 (`ast.optimized` tracing) |
| `schedule/src/expand.rs` | UNROLL/UPCAST expansion |
| `schedule/src/optimizer/heuristics.rs` | Optimization decisions |
| `codegen/src/llvm/text/mod.rs` | LLVM IR generation (render() function) |
| `codegen/src/llvm/cpu/ops.rs` | CPU operation rendering |
| `codegen/src/types.rs` | RenderedKernel struct |

## Module Structure

After refactoring, LLVM codegen is organized as:
```
codegen/src/llvm/
├── common/           # Shared utilities (types, ctx)
│   ├── mod.rs
│   ├── ctx.rs      # RenderContext
│   └── types.rs     # ldt, lconst, lcast, addr_space_num
├── cpu/              # CPU-specific rendering
│   ├── mod.rs       # Exports render_uop, reduce_identity
│   └── ops.rs       # CPU operation rendering
├── gpu/              # Future GPU support (placeholder)
│   ├── mod.rs
│   └── ops.rs       # Stub returns None
└── text/             # Main entry point
    └── mod.rs       # LlvmTextRenderer, render() function
```

## Tracing Output Examples

### Enable codegen tracing
```bash
RUST_LOG=morok_codegen::llvm::text=debug cargo test test_name
```

### Sample output showing linearization and IR generation
```
[2026-01-26...] DEBUG morok_codegen::llvm::text::linearized node: position=0, op=DefineGlobal, id=42 "codegen: after pm_render"
[2026-01-26...] DEBUG morok_codegen::llvm::text::linearized node: position=1, op=Index, id=35
[2026-01-26...] DEBUG morok_codegen::llvm::text::linearized node: position=2, op=Range, id=30
```
