# Rangeify Module

This module implements Tinygrad's RANGEIFY algorithm, which transforms high-level tensor operations into executable kernels.

## Overview

The RANGEIFY transformation is the core of the Morok compiler's scheduling system. It converts high-level movement operations (reshape, permute, expand, etc.) into explicit index transformations and buffer operations, then splits the computation graph into individual executable kernels.

## Architecture

The transformation happens in 5 main phases:

### Phase 1: Range Assignment
**Module:** `indexing`

Determines input/output ranges for each UOp by analyzing the computation graph and assigning symbolic range variables.

**Key Functions:**
- `run_rangeify()` - Main entry point for range analysis
- `IndexingContext` - Tracks range assignments and dependencies

### Phase 2: Core Transform
**Modules:** `patterns`, `transform`

Converts movement operations into BUFFERIZE+INDEX patterns through pattern matching.

**Key Transformations:**
- RESHAPE → BUFFERIZE with index manipulation
- PERMUTE → BUFFERIZE with axis reordering
- EXPAND → BUFFERIZE with broadcasting
- PAD/SHRINK → BUFFERIZE with offset calculations

**Pattern Categories:**
- Early rewrites (DETACH, CONTIGUOUS_BACKWARD cleanup)
- Movement operation conversion
- Movement operation removal (after conversion)

### Phase 3: Buffer Management
**Modules:** `patterns`, `helpers`

Applies cost-based optimizations to minimize buffer allocations and memory transfers.

**Optimizations:**
- Buffer folding (fuse adjacent bufferize operations)
- Dead axis removal (eliminate unused dimensions)
- Cost-based buffer removal (inline cheap operations)

### Phase 4: Symbolic Simplification
**Module:** `symbolic` (in parent module)

Optimizes index expressions through symbolic mathematics.

**Simplifications:**
- Constant folding
- Algebraic identities
- Common subexpression elimination

### Phase 5: Kernel Splitting
**Modules:** `bufferize_to_store`, `kernel_context`, `split_kernel`, `split_patterns`, `pipeline`

Converts BUFFERIZE operations into executable KERNEL operations.

#### 5.1 BUFFERIZE → STORE Conversion
**Module:** `bufferize_to_store`

Converts high-level BUFFERIZE operations into low-level STORE operations with explicit buffer allocation.

**Transformations:**
- `BUFFERIZE(compute, ranges, {addrspace: GLOBAL})` → `DEFINE_GLOBAL + STORE + END`
- `BUFFERIZE(compute, ranges, {addrspace: LOCAL})` → `DEFINE_LOCAL + STORE + END + BARRIER`

**Key Function:** `bufferize_to_store()`

#### 5.2 Graph Rewriting Patterns
**Module:** `split_patterns`

Pattern handlers for transforming the graph into kernel-ready IR.

**Patterns:**
- `debuf()` - BUFFER → DEFINE_GLOBAL/DEFINE_LOCAL
- `handle_after()` - Track buffer dependencies
- `unbind_kernel()` - Remove kernel-local BIND operations
- `renumber_range()` - Renumber ranges for deduplication
- `remove_zero_range()` - Optimize empty loops
- `cleanup_const()` - Remove spurious sources

#### 5.3 Kernel Splitting Logic
**Module:** `split_kernel`

Determines kernel boundaries and creates KERNEL operations.

**Algorithm:**
1. Filter operations (only split at OUTER range boundaries)
2. Apply transformation pipeline
3. Create SINK wrapper
4. Build kernel arguments (buffers + variables)

**Key Function:** `split_store()`

**Helper Functions:**
- `all_ranges_outer()` - Check if ready to split
- `is_outer_end()` - Detect control flow markers

#### 5.4 Pipeline Orchestration
**Module:** `pipeline`

Orchestrates the complete kernel splitting transformation.

**Pipeline Stages:**
1. BUFFERIZE → STORE conversion
2. Graph rewriting with to_define_global patterns
3. Kernel splitting at STORE boundaries

**Key Function:** `run_kernel_split_pipeline()`

#### 5.5 Kernel Context
**Module:** `kernel_context`

Tracks transformation state during kernel splitting.

**State Tracked:**
- `global_counter` - Next DEFINE_GLOBAL ID
- `local_counter` - Next DEFINE_LOCAL ID
- `buffer_map` - Original buffer → replacement mapping
- `vars` - BIND variable tracking
- `range_counter` - Next range ID for renumbering

**Type:** `KernelContext`

## Module Organization

```
rangeify/
├── Core Transformation (Phases 1-4)
│   ├── context.rs          # RangeifyContext for tracking transformations
│   ├── helpers.rs          # Utility functions
│   ├── indexing.rs         # Range assignment (Phase 1)
│   ├── patterns.rs         # Pattern definitions (Phases 2-3)
│   └── transform.rs        # Main rangeify entry point
│
├── Kernel Splitting (Phase 5)
│   ├── bufferize_to_store.rs   # BUFFERIZE → STORE conversion
│   ├── kernel_context.rs       # Kernel transformation state
│   ├── pipeline.rs             # Pipeline orchestration
│   ├── split_kernel.rs         # Kernel splitting logic
│   └── split_patterns.rs       # Pattern handlers
│
└── mod.rs                  # Module exports and documentation
```

## Public API

### Main Entry Points

```rust
use morok_schedule::rangeify;

// Full rangeify transformation (Phases 1-4)
let transformed = rangeify(root);

// Kernel splitting pipeline (Phase 5)
let kernels = run_kernel_split_pipeline(root);
```

### Context Types

```rust
use morok_schedule::rangeify::{RangeifyContext, KernelContext};

// For custom transformations
let mut ctx = RangeifyContext::new();
let mut kernel_ctx = KernelContext::new();
```

## Implementation Status

### ✅ Complete (Phases 1-4)
- Range assignment algorithm
- Movement operation conversion
- Buffer management optimizations
- Symbolic simplification integration

### 🚧 In Progress (Phase 5)
- BUFFERIZE → STORE conversion ✅
- Pattern handlers for kernel transformation ✅
- Kernel splitting infrastructure ✅
- Pipeline orchestration framework ✅

### ⏳ Future Work (Phase 6)
- Full graph_rewrite engine integration
- Pattern matcher-based transformations
- Context-aware rewriting
- Topological ordering and dependency tracking

## Testing

Each module has comprehensive unit tests:

```bash
# Run all rangeify tests
cargo test --package schedule rangeify

# Run specific module tests
cargo test --package schedule bufferize_to_store
cargo test --package schedule split_patterns
cargo test --package schedule split_kernel
cargo test --package schedule pipeline
```

## References

Based on Tinygrad's rangeify implementation:
- `tinygrad/schedule/rangeify.py` - Main algorithm
- `tinygrad/ops.py` - Operation definitions
- `tinygrad/schedule/__init__.py` - Integration

## Examples

### Example 1: Basic Movement Operation

```rust
use morok_ir::UOp;
use morok_schedule::rangeify;

// Create a RESHAPE operation
let input = /* ... */;
let reshape = UOp::reshape(input, new_shape);

// Apply rangeify transformation
let result = rangeify(reshape);

// Result contains BUFFERIZE + INDEX operations
```

### Example 2: Kernel Splitting

```rust
use morok_schedule::run_kernel_split_pipeline;

// Create computation with BUFFERIZE operations
let computation = /* build graph */;

// Split into kernels
let kernels = run_kernel_split_pipeline(computation);

// kernels now contains KERNEL operations ready for codegen
```

## Design Principles

1. **Separation of Concerns**: Each phase handles a distinct transformation
2. **Composability**: Phases can be applied independently or together
3. **Testability**: Each component has comprehensive unit tests
4. **Performance**: Cost-based optimizations minimize memory transfers
5. **Correctness**: Pattern matching ensures type safety

## Performance Considerations

- **Buffer Folding**: Reduces memory allocations by fusing operations
- **Cost Analysis**: Only materializes buffers when beneficial
- **Symbolic Simplification**: Reduces runtime computation overhead
- **Kernel Deduplication**: Range renumbering enables kernel caching

## Future Enhancements

1. **Multi-Device Support**: Handle cross-device transfers
2. **Advanced Fusion**: More aggressive operation merging
3. **Auto-Tuning**: Dynamic cost model adjustment
4. **Parallelization**: Multi-threaded graph analysis
