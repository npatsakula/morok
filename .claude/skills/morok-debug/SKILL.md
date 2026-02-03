---
name: morok-debug
description: Debug Morok tensor pipeline issues by extracting IR at each stage, visualizing UOp trees, and comparing with Tinygrad. Use when tests fail, produce wrong results, or crash.
---

# Morok Pipeline Debugging

## Three-Step Analysis

When you don't know where an error occurs in the pipeline:

### Step 1: Check IR before optimization
```bash
RUST_LOG=morok_schedule::rangeify=debug cargo test test_name 2>&1 | rg 'range assignment complete'
```

### Step 2: Check IR after optimization
```bash
RUST_LOG=morok_schedule::optimizer=debug cargo test test_name 2>&1 | rg 'optimization complete'
```

### Step 3: Check LLVM IR
```bash
RUST_LOG=morok_codegen::llvm::text=debug cargo test test_name 2>&1 | rg 'linearized node'
```

Compare with Tinygrad's output using the `/tinygrad` skill to isolate broken patterns.

---

## Stage-by-Stage Tree Extraction

### Quick Reference Table

| Pipeline Phase | Stage | RUST_LOG Target | Debug Marker | What to Extract |
|----------------|--------|-------------------|--------------|------------------|
| **RANGEIFY** | 0: Range Assignment | `morok_schedule::rangeify=debug` | `"range assignment complete"` | Initial IR with movement ops |
| | 1: Early Movement Ops | Same as Stage 0 | `"early rewrites complete"` | After movement op cleanup |
| | 2: Load Collapse | Same as Stage 0 | `"Stage 2a: load collapse complete"` | After REDUCE elimination |
| | 2b: Reduction Simplify | Same as Stage 0 | `"Stage 2b: reduction simplify complete"` | After reduction patterns |
| | 3: Split Ranges | Same as Stage 0 | `"Stage 3a: split ranges complete"` | After range splitting |
| | 3b: Flatten Ranges | Same as Stage 0 | `"Stage 3b: flatten ranges complete"` | After flattening |
| | 4: Initial Symbolic | Same as Stage 0 | `"Stage 4: initial symbolic complete"` | After constant folding |
| | 5: Simplify Ranges | Same as Stage 0 | `"Stage 5: simplify ranges complete"` | After range merging |
| | 6: Split Store | Same as Stage 0 | `"Stage 6: split store complete"` | After store splitting (CPU) |
| | 7: Buffer Removal | Same as Stage 0 | `"Stage 7: buffer removal complete"` | After buffer optimization |
| | 7b: Buffer Limits | Same as Stage 0 | `"Stage 7b: buffer limit enforcement complete"` | After limit enforcement |
| **EXPANDER** | 8: Post-Opt Symbolic | (not logged separately) | After WHERE movement |
| | 9: Expander | `morok_schedule::expand=debug` | `"Stage 9: expander complete"` | After UNROLL/UPCAST expansion |
| | 10: Add Local Buffers | `morok_schedule::rangeify=debug` | `"buffer folding + dead axis removal complete"` | After bufferization |
| **DEVECTORIZER** | 11: Remove Reduce | `morok_schedule::devectorize=debug` | `"Stage 11: remove reduce complete"` | After REDUCE→ACCUMULATOR |
| | 12: Add GPU Dims | Same as Stage 11 | `"Stage 12: add gpudims complete"` | After SPECIAL injection |
| | 13: Add Loads | Same as Stage 11 | `"Stage 13: add loads complete"` | After LOAD wrapping |
| | 14: Devectorize | Same as Stage 11 | `"Stage 14: devectorize complete"` | After hardware lowering |
| | 15: Lower Index Dtype | `morok_schedule::optimizer=debug` | `"after pm_lower_index_dtype"` | After Index→i32/i64 |
| **LINEARIZER** | 16: Post-Index Symbolic | (not logged separately) | Final cleanup |
| | 17: Pre-Matcher | (not logged separately) | Backend-specific patterns |
| | 18: Decompositions | (not logged separately) | Late rewrites |
| | 19: Final Rewrite | `morok_schedule::optimizer=debug` | `"Stage 19: final rewrite complete"` | Before linearization |
| | 20: Add Control Flow | `morok_schedule::linearize=debug` | `"Stage 20: add control flow complete"` | After CFG edges |
| | 21: Linearize | Same as Stage 20 | `"Stage 21: linearize complete"` | Topological sort |
| | 22: Cleanup IF/ENDIF | (not logged separately) | Final cleanup |

### Extract All Stages at Once

```bash
# Capture all pipeline stages in a single command
RUST_LOG='morok_schedule::rangeify|morok_schedule::expand|morok_schedule::devectorize|morok_schedule::gpudims|morok_schedule::optimizer|morok_schedule::linearize=debug' \
  cargo test test_name -- --nocapture 2>&1 | rg -E 'complete' --color=always
```

This shows tree output with markers for all stages with their corresponding UOp trees.

### Method 1: In-Code Tree Extraction

Add temporary debugging code directly in your test or tensor operation:

```rust
use morok_ir::prelude::*;

// After each pipeline stage
println!("--- After Stage N ---");
println!("{}", uop.tree());  // Compact tree with back-references
// println!("{}", uop.tree_full());  // Full tree expanding all nodes
```

**Note**: You'll need to manually insert these print statements at the appropriate stage location in the pipeline code (`schedule/src/rangeify/transforms.rs`, `schedule/src/expand.rs`, etc.).

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

### Method 3: Tracing with Tree Output

Enable RUST_LOG to see tree structure at specific stages:

```bash
# Extract IR after Stage 0-7 (Rangeify)
RUST_LOG=morok_schedule::rangeify=debug cargo test test_name 2>&1 | rg -A5 'range assignment complete' | head -50

# Extract IR after Stage 9 (Expander)
RUST_LOG=morok_schedule::expand=debug cargo test test_name 2>&1 | rg -A5 'Stage 9: expander complete' | head -50

# Extract IR after Stage 19 (Final Rewrite)
RUST_LOG=morok_schedule::optimizer=debug cargo test test_name 2>&1 | rg -A5 'Stage 19: final rewrite complete' | head -50
```

The `-A5` option shows 5 lines of context after each match, helping you see the tree structure.

### Stage-Specific Debugging Commands

#### Rangeify Phase (Stages 0-7)

```bash
# Stage 0: Range assignment
RUST_LOG=morok_schedule::rangeify=debug cargo test_name 2>&1 | rg 'range assignment complete'

# Stage 1: Early movement ops
RUST_LOG=morok_schedule::rangeify=debug cargo test_name 2>&1 | rg 'early rewrites complete'

# Stage 2a: Load collapse
RUST_LOG=morok_schedule::rangeify=debug cargo test_name 2>&1 | rg 'Stage 2a: load collapse complete'

# Stage 2b: Reduction simplify
RUST_LOG=morok_schedule::rangeify=debug cargo test_name 2>&1 | rg 'Stage 2b: reduction simplify complete'

# Stage 3a: Split ranges
RUST_LOG=morok_schedule::rangeify=debug cargo test_name 2>&1 | rg 'Stage 3a: split ranges complete'

# Stage 3b: Flatten ranges
RUST_LOG=morok_schedule::rangeify=debug cargo test_name 2>&1 | rg 'Stage 3b: flatten ranges complete'

# Stage 4: Initial symbolic
RUST_LOG=morok_schedule::rangeify=debug cargo test_name 2>&1 | rg 'Stage 4: initial symbolic complete'

# Stage 5: Simplify ranges
RUST_LOG=morok_schedule::rangeify=debug cargo test_name 2>&1 | rg 'Stage 5: simplify ranges complete'

# Stage 6: Split store (CPU only)
RUST_LOG=morok_schedule::rangeify=debug cargo test_name 2>&1 | rg 'Stage 6: split store complete'

# Stage 7: Buffer removal
RUST_LOG=morok_schedule::rangeify=debug cargo test_name 2>&1 | rg 'Stage 7: buffer removal complete'

# Stage 7b: Buffer limits
RUST_LOG=morok_schedule::rangeify=debug cargo test_name 2>&1 | rg 'Stage 7b: buffer limit enforcement complete'
```

#### Expander Phase (Stages 8-10)

```bash
# Stage 9: UNROLL/UPCAST expansion
RUST_LOG=morok_schedule::expand=debug cargo test_name 2>&1 | rg 'Stage 9: expander complete'

# Stage 10: Add local buffers
RUST_LOG=morok_schedule::rangeify=debug cargo test_name 2>&1 | rg 'buffer folding + dead axis removal complete'
```

#### Devectorizer Phase (Stages 11-15)

```bash
# Stage 11: Remove Reduce (REDUCE→ACCUMULATOR)
RUST_LOG=morok_schedule::devectorize=debug cargo test_name 2>&1 | rg 'Stage 11: remove reduce complete'

# Stage 12: Add GPU Dims
RUST_LOG=morok_schedule::devectorize=debug cargo test_name 2>&1 | rg 'Stage 12: add gpudims complete'

# Stage 13: Add Loads
RUST_LOG=morok_schedule::devectorize=debug cargo test_name 2>&1 | rg 'Stage 13: add loads complete'

# Stage 14: Devectorize
RUST_LOG=morok_schedule::devectorize=debug cargo test_name 2>&1 | rg 'Stage 14: devectorize complete'

# Stage 15: Lower Index Dtype (no explicit marker)
# Check IR after Stage 14 to see Index→i32/i64 conversion
```

#### Linearizer Phase (Stages 16-22)

```bash
# Stage 19: Final Rewrite
RUST_LOG=morok_schedule::optimizer=debug cargo test_name 2>&1 | rg 'Stage 19: final rewrite complete'

# Stage 20: Add Control Flow
RUST_LOG=morok_schedule::linearize=debug cargo test_name 2>&1 | rg 'Stage 20: add control flow complete'

# Stage 21: Linearize (topological sort)
RUST_LOG=morok_schedule::linearize=debug cargo test_name 2>&1 | rg 'Stage 21: linearize complete'

# Stage 22: Cleanup IF/ENDIF (no explicit marker)
# Check linearized instruction list after Stage 21
```

### Example: Debugging a Vectorization Issue

```bash
# 1. Check IR before expander (Stage 8)
RUST_LOG=morok_schedule::rangeify=debug cargo test_vectorize 2>&1 | rg -A5 'Stage 4: initial symbolic complete'

# 2. Check IR after expander (Stage 9)
RUST_LOG=morok_schedule::expand=debug cargo test_vectorize 2>&1 | rg -A5 'Stage 9: expander complete'

# 3. Compare to see if UNROLL→CONTRACT is correct
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
| `morok_schedule::rangeify=debug` | Pipeline stages with tree output |
| `morok_schedule::expand=debug` | UNROLL/UPCAST expansion |
| `morok_schedule::devectorize=debug` | Memory access optimization |
| `morok_schedule::optimizer=debug` | Post-optimization passes |
| `morok_codegen::llvm::text=debug` | LLVM rendering (IR generation) |
| `morok_ir::pattern::simplified=trace` | Pattern matching details |

### Pipeline stage markers (filter with rg)
```
"range assignment complete"
"early rewrites complete"
"split reduceops complete"
"rangeify complete"
"buffer folding + dead axis removal complete"
"reshape to scalar complete"
"buffer removal complete"
"symbolic simplification complete"
"reduction simplification complete"
```

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
| `ir/src/uop/tree.rs` | UOp tree visualization, format_node() |
| `ir/src/op.rs` | Op enum with AsRefStr derive |
| `schedule/src/rangeify/transforms.rs` | Main pipeline with stage markers |
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
