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
