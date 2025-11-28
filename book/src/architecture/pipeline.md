# Execution Pipeline

From tensor definition to kernel execution, morok follows a multi-stage compilation pipeline inspired by Tinygrad.

## Stage Overview

```
Tensor API → UOp DAG → Rangeify → Kernel Split → Schedule → Codegen → Execute
```

## Stage 0: Tensor Creation

**Input:** Rust slice or data
**Output:** `Tensor { uop: Rc<UOp> }`

```rust
let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;
```

The tensor creates a `BUFFER` UOp and registers the actual device buffer in a thread-local registry. No computation happens yet.

## Stage 1: Lazy Operation Building

**Input:** Tensor operations
**Output:** UOp DAG (Directed Acyclic Graph)

```rust
let c = a.try_add(&b)?;  // Builds UOp graph, no execution
let d = c.sum();         // Adds REDUCE node to graph
```

Each operation appends nodes to the UOp graph. The graph captures:
- Arithmetic operations (ADD, MUL, etc.)
- Movement operations (RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP)
- Reductions (SUM, MAX, etc.)

## Stage 2: Rangeify

**Input:** UOp DAG with movement ops
**Output:** UOp DAG with BUFFERIZE + INDEX + explicit RANGE loops

**File:** `schedule/src/rangeify/transform.rs`

This is the core transformation that converts high-level tensor operations into explicit loop nests:

1. **Range Assignment:** Create RANGE UOps for each dimension
2. **Movement Op Transformation:** Convert movement ops to index expressions
   - `SHRINK` → offset adjustment
   - `PERMUTE` → axis reordering
   - `EXPAND` → broadcast (index becomes 0)
   - `PAD` → conditional with WHERE
   - `RESHAPE` → axis combinatorics
3. **Buffer Simplification:** Remove unnecessary intermediate buffers
4. **Symbolic Simplification:** Apply algebraic identities

```
# Before rangeify:
BUFFER.reshape([2,3]).expand([4,2,3]).sum(axis=0)

# After rangeify:
RANGE(4) -> RANGE(2) -> RANGE(3) ->
  LOAD(buffer, index_expr) -> REDUCE(ADD) -> STORE
```

## Stage 3: Kernel Splitting

**Input:** Rangeified UOp DAG
**Output:** Multiple KERNEL UOps

**File:** `schedule/src/rangeify/pipeline.rs`

Splits the graph at STORE boundaries into separate kernels:

1. **BUFFERIZE → STORE:** Convert buffer materializations to explicit stores
2. **STORE → KERNEL:** Group related stores into kernel operations
3. **Dependency Tracking:** AFTER operations mark cross-kernel dependencies

## Stage 4: Schedule Creation

**Input:** KERNEL UOps
**Output:** `Vec<ScheduleItem>` with ordered kernels

Collects kernels in execution order, gathering:
- Kernel AST (the computation graph)
- Buffer arguments (inputs/outputs)
- Symbolic variable values

## Stage 5: Codegen

**Input:** Kernel AST
**Output:** LLVM IR

**File:** `codegen/src/llvm/renderer.rs`

Currently implements direct rendering to LLVM IR. Tinygrad has 16+ optimization passes here that morok is still developing:
- Range splitting/flattening
- GPU dimension assignment
- Load/store optimization
- Devectorization
- Control flow insertion

## Stage 6: Execution

**Input:** LLVM IR
**Output:** Computed result

**File:** `runtime/src/llvm.rs`

JIT compiles the LLVM IR and executes with buffer pointers.

## Gap Analysis vs Tinygrad

| Stage | Morok Status | Missing |
|-------|--------------|---------|
| Tensor Creation | Basic | numpy, URL, disk loading |
| Lazy Ops | Partial | Many advanced ops |
| Rangeify | 7 passes | 13+ in Tinygrad |
| Kernel Split | Basic | Dependency-aware BFS |
| Schedule | Basic | Memory planning, var tracking |
| Codegen | Direct render | 16+ optimization passes |
| Execute | LLVM only | CUDA, Metal, WebGPU |
