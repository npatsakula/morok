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
# Uses JSON structured logging (rg + jaq required)
./scripts/extract-ir.sh test_name -p morok-tensor -o /tmp/debug_ir.txt

# Filter to specific module (e.g., only optimizer stages)
./scripts/extract-ir.sh test_name -p morok-tensor -t optimizer
```

### Step 2: Check LLVM IR
```bash
RUST_LOG=morok_codegen::llvm::text=debug cargo test test_name -- --nocapture 2>&1 | rg 'linearized node'
```

### Step 3: Compare with Tinygrad
Compare with Tinygrad's output using the `/tinygrad-debug` skill to isolate broken patterns.

---

## Pipeline Structure

The pipeline has three phases. Rangeify runs once for the entire SINK.
After kernel splitting, pre-opt and post-opt run once per kernel.

1. **Rangeify** (`schedule/src/rangeify/transforms.rs`) — field: `uop.tree`
2. **Per-kernel pre-optimization** (`schedule/src/optimizer/mod.rs: apply_pre_optimization`) — field: `ast.pre`
3. **Per-kernel post-optimization** (`schedule/src/optimizer/mod.rs: apply_post_optimization_with_renderer`) — field: `ast.optimized`
4. **Linearizer** — no tree tracing (linear instruction lists)

## Stage-by-Stage Tree Extraction

### Quick Reference Table

| Phase | Field | Debug Message |
|-------|-------|---------------|
| **RANGEIFY** (`rangeify/transforms.rs`) | | |
| | `uop.tree` | `early rewrites + replace contiguous complete` |
| | `uop.tree` | `Stage 0: range assignment complete` |
| | `uop.tree` | `split reduceops complete` |
| | `uop.tree` | `Stage 1: rangeify + movement ops complete` |
| | — | `mega-pass complete` *(node_count, no tree)* |
| | `uop.tree` | `Stage 7b: buffer limit enforcement complete` *(conditional)* |
| **PRE-OPT** (`optimizer/mod.rs: apply_pre_optimization`) | | |
| | `ast.initial` | `kernel initial` |
| | `ast.pre` | `pre-opt: movement ops + syntactic sugar complete` |
| | `ast.pre` | `pre-opt: load collapse complete` |
| | `ast.pre` | `pre-opt: split ranges complete` |
| | `ast.pre` | `pre-opt: symbolic + flatten complete` |
| | `ast.pre` | `pre-opt: simplify ranges complete` |
| **POST-OPT** (`optimizer/mod.rs: apply_post_optimization_with_renderer`) | | |
| | `ast.initial` | `kernel initial` |
| | `ast.optimized` | `Stage 8: after post-opt symbolic` |
| | `ast.optimized` | `Stage 9: after pre_expand` |
| | `ast.optimized` | `Stage 10: after add local buffers` |
| | `ast.optimized` | `after pm_reduce` |
| | `ast.optimized` | `after pm_add_gpudims` |
| | — | `after pm_add_loads` *(elapsed_ms only, no tree)* |
| | `ast.optimized` | `after devectorize` |
| | `ast.optimized` | `after pm_bool_devectorize` |
| | `ast.optimized` | `after pm_reduce_devectorize` |
| | `ast.optimized` | `after pm_lower_index_dtype` |
| | `ast.optimized` | `after post-index symbolic` |
| | `ast.optimized` | `Stage 18-19: after pm_decomp + pm_render` |
| | `ast.optimized` | `after pm_float_decomp` |
| | `ast.optimized` | `after bool_storage_pattern` |
| **LINEARIZER** | — | *No tree tracing (linear instruction lists)* |

### Extract All Stages at Once (Recommended)

Use `scripts/extract-ir.sh` to extract all pipeline stage trees into a single readable file:

```bash
# Basic: extract IR for a specific test
./scripts/extract-ir.sh test_sum_axis1_value -p morok-tensor

# With custom output file
./scripts/extract-ir.sh test_argmax_value_1d -p morok-tensor -o /tmp/argmax_ir.txt

# ONNX model tests
./scripts/extract-ir.sh light_densenet121 -p morok-onnx

# Filter to specific pipeline phase
./scripts/extract-ir.sh test_name -p morok-tensor -t optimizer
```

**Prerequisites**: `rg` (ripgrep) and `jaq` must be installed. Tests use JSON
tracing via `morok_schedule::testing::setup_test_tracing()`.

The script extracts fields: `uop.tree`, `ast.pre`, `ast.optimized`, `ast.initial`, `generated_c`.

**Pipeline structure** (why the output is organized this way):
- `rangeify_with_map` runs **once** for the entire SINK before kernel splitting
- `apply_pre_optimization` and `apply_post_optimization_with_renderer` run **once per kernel** after splitting
- Linearizer produces linear instruction lists, not trees — no tree tracing

### Method 1: In-Code Tree Extraction

Add temporary debugging code directly in your test or tensor operation:

```rust
use morok_ir::prelude::*;

// After each pipeline stage
println!("--- After Stage N ---");
println!("{}", uop.tree());  // Compact tree with back-references
// println!("{}", uop.tree_full());  // Full tree expanding all nodes
```

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
RUST_LOG=morok_schedule::optimizer=debug cargo test test_name -- --nocapture 2>&1 | rg 'Stage 18-19'
```

**Note**: Output is JSON. Use `scripts/extract-ir.sh` for readable, formatted output.

---

## Enabling Traces in Tests

### Why traces don't appear

`RUST_LOG` sets a filter level but requires a tracing subscriber. Tests don't have one by default.

### Enable tracing in a test

Call the shared initializer from `morok-schedule` (requires `testing` feature):

```rust
#[test]
fn test_my_failing_test() {
    morok_schedule::testing::setup_test_tracing();
    // ... test code
}
```

This registers a JSON subscriber controlled by `RUST_LOG`.

### Run with output visible
```bash
cargo test test_name -- --nocapture
```

---

## ONNX Model Debugging

### Node-level tracing

The ONNX importer has per-node tracing spans (`onnx_node` with `idx` and `op` fields):

```bash
# Non-intrusive: log node names and ops
RUST_LOG=morok_onnx::importer=debug cargo test light_densenet121 -p morok-onnx -- --nocapture

# Intrusive bisection: realize every node, dump first 5 values
# WARNING: breaks fusion — use only for numerical debugging
RUST_LOG=morok_onnx::importer=trace cargo test light_densenet121 -p morok-onnx -- --nocapture
```

At `trace` level, each node output is realized and its shape + first 5 f32 values are logged
(`out_name`, `shape`, `first5` fields).

---

## UOp Visualization

### In code
```rust
use morok_ir::prelude::*;

println!("{}", uop.tree());       // Compact tree with back-references
println!("{}", uop.tree_full());  // Full tree expanding all nodes
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
println!("{}", rendered.code);
```

### From tensor
```rust
let plan = tensor.prepare().expect("prepare should succeed");

for kernel in plan.kernels() {
    println!("--- {} ({}) ---", kernel.entry_point, kernel.device);
    println!("{}", kernel.code);
}
```

## RUST_LOG Targets

| Target | Information |
|--------|-------------|
| `morok_onnx::importer=debug` | ONNX node processing (idx, op_type) |
| `morok_onnx::importer=trace` | ONNX node values (intrusive realization) |
| `morok_schedule::rangeify::transforms=debug` | Rangeify stages (`uop.tree` field) |
| `morok_schedule::rangeify::indexing=debug` | Range assignment details |
| `morok_schedule::rangeify::kernel=debug` | Kernel splitting |
| `morok_schedule::optimizer=debug` | Pre-opt (`ast.pre`) + Post-opt (`ast.optimized`) stages |
| `morok_schedule::linearize=debug` | Linearization passes |
| `morok_codegen=debug` | LLVM rendering and codegen |
| `morok_ir::pattern::simplified=trace` | Pattern matching details |

The `scripts/extract-ir.sh` script sets all relevant targets automatically.

## Common Debug Scenarios

### Wrong numerical result

1. Extract IR before/after optimization
2. Check if UNROLL/UPCAST expansion is correct
3. Compare vector widths with expected
4. Check LLVM IR for correct horizontal reduce

### ONNX model gives wrong output

1. Run with `RUST_LOG=morok_onnx::importer=trace` to dump per-node values
2. Compare values with ONNX runtime reference output
3. Find the first diverging node, then investigate its op implementation

### SIGSEGV / Memory error

1. Check buffer sizes and indices
2. Look for mismatched vector widths
3. Check CAT expansion creating oversized vectors

### Pattern not matching

```bash
RUST_LOG=morok_ir::pattern::simplified=trace cargo test test_name 2>&1 | rg 'pattern'
```

## Key Files

| File | Purpose |
|------|----------|
| `scripts/extract-ir.sh` | Pipeline IR extraction (JSON + rg + jaq) |
| `schedule/src/testing.rs` | Shared test tracing setup (`testing` feature) |
| `ir/src/uop/tree.rs` | UOp tree visualization, format_node() |
| `ir/src/op.rs` | Op enum with AsRefStr derive |
| `schedule/src/rangeify/transforms.rs` | Rangeify pipeline (`uop.tree` tracing) |
| `schedule/src/optimizer/mod.rs` | Pre-opt (`ast.pre`) + Post-opt (`ast.optimized`) tracing |
| `schedule/src/expand.rs` | UNROLL/UPCAST expansion |
| `schedule/src/optimizer/heuristics.rs` | Optimization decisions |
| `codegen/src/llvm/text/mod.rs` | LLVM IR generation (render() function) |
| `codegen/src/llvm/cpu/ops.rs` | CPU operation rendering |
| `codegen/src/types.rs` | RenderedKernel struct |
