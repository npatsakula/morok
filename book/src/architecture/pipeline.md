# From Tensor to Machine Code

In most ML frameworks, computation happens immediately. Write `a + b` in PyTorch and it runs *now*—the GPU crunches numbers before you can even inspect the result. This eager execution is simple to understand, but it leaves optimization opportunities on the table. How can a compiler optimize a computation it hasn't seen yet?

Morok takes the opposite approach: **lazy evaluation**. When you write `a.try_add(&b)?`, nothing computes. Morok builds a graph describing *what* to compute, not *when*. The magic happens when you call `realize()`—that single method triggers the entire compilation pipeline, from high-level tensor operations down to JIT-compiled machine code.

This chapter traces that journey.

```text
tensor.realize()
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  LAZY GRAPH                                             │
│  Tensor ops build UOp DAG (no computation yet)          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  RANGEIFY                                               │
│  Movement ops → explicit RANGE loops                    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  KERNEL SPLITTING                                       │
│  Split at STORE boundaries → multiple KERNELs          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  OPTIMIZATION & CODEGEN                                 │
│  Heuristics/beam → LLVM IR → JIT compile               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  EXECUTION                                              │
│  Parallel kernel launch → result buffer                │
└─────────────────────────────────────────────────────────┘
```

Each box is a distinct phase. Let's walk through them.

---

## Lazy Evaluation: Building the Graph

A `Tensor` in Morok is surprisingly lightweight:

```rust
pub struct Tensor {
    entry: Arc<TensorEntry>,      // Computation graph
    buffer: Option<Arc<Buffer>>,  // Materialized data (if any)
}
```

The `entry` holds a `TensorEntry` containing the UOp graph—the computation this tensor represents. The `buffer` is optional: lazy tensors don't have one, only realized tensors do.

### Three Ways to Create Tensors

**1. Input tensors** — buffer allocated immediately:

```rust
let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;
// `a.buffer` = Some(Arc<Buffer>) with actual data
```

When you create a tensor from data, Morok allocates device memory and copies your bytes. The UOp graph contains a `BUFFER` node pointing to this allocation.

**2. Lazy operations** — no buffer, only graph:

```rust
let b = a.try_add(&a)?;   // b.buffer = None
let c = b.try_mul(&a)?;   // c.buffer = None
```

Arithmetic operations don't compute anything. They build a UOp graph: `Binary(Add, a.uop, a.uop)`. The tensor exists purely as a description of future work.

**3. Movement operations** — shares the original buffer:

```rust
let d = a.try_reshape(&[1, 3])?;  // d.buffer = same as a.buffer
```

Reshape, permute, and similar operations create new *views* of existing data. The buffer is shared; only the UOp graph changes to describe the new indexing.

### The Global Registry

Morok maintains three global maps (lock-free, thread-safe):

| Map | Key → Value | Purpose |
|-----|-------------|---------|
| `TENSORS` | tensor_id → `Weak<TensorEntry>` | Track all tensors for graph substitution |
| `BUFFERS` | uop_id → `Arc<Buffer>` | Find buffers during scheduling |
| `UOP_TO_TENSOR` | uop_id → tensor_id | Secondary index for lookups |

This registry enables a critical feature: **global graph substitution**. When an optimization transforms a UOp, all tensors referencing that UOp automatically see the updated version. No stale references, no manual updates.

### Hash Consing in Action

Because UOps use hash consing (content-based deduplication), identical computations share memory:

```rust
let x = a.try_add(&b)?;
let y = a.try_add(&b)?;
// x.uop() and y.uop() point to the SAME Arc<UOp>
```

This matters for caching: when we compile kernels, we cache by UOp ID. Hash consing means identical computations automatically hit the cache, even if constructed separately.

---

## Rangeify: Making Loops Explicit

When you write `tensor.reshape([2, 3]).expand([4, 2, 3]).sum(axis=0)`, those movement operations (reshape, expand) are high-level descriptions. To generate actual loops, we need explicit iteration structure.

**Rangeify** transforms movement operations into `RANGE` loops and `INDEX` arithmetic. The entry point is `rangeify()` in `schedule/src/rangeify/transforms.rs`.

### The 8-Pass Pipeline

Rangeify isn't a single transformation—it's eight coordinated passes:

| Pass | Purpose |
|------|---------|
| **1. Range Assignment** | Create RANGE UOps for each tensor dimension |
| **2. Early Rewrites** | Remove DETACH, clean up trivial RESHAPE |
| **3. Split Large Reductions** | Two-stage reduce for huge arrays (ratio > 32768) |
| **4. Core Rangeify** | ReduceAxis → REDUCE, bufferization, movement removal |
| **5. Buffer Folding** | Constant propagation through buffer expressions |
| **6. Dead Axis Removal** | Filter ranges that don't affect the output |
| **7. Cost-Based Buffer Removal** | Inline buffers when profitable (PContig optimization) |
| **8. Reduction Simplification** | Lift range-independent code out of reductions |

Each pass uses pattern-based rewriting (see the [Pattern-Based Optimization](./optimizations.md) chapter). Patterns fire until no more match, then the next pass begins.

### Before and After

Consider this tensor expression:

```text
Before: BUFFER.reshape([2, 3]).expand([4, 2, 3]).sum(axis=0)
```

After rangeify, movement ops become explicit index computations:

```text
After:
STORE
├── INDEX[RANGE(0..2), RANGE(0..3)]
└── REDUCE(Add)
    ├── LOAD
    │   └── INDEX[RANGE(0..4), RANGE(0..2), RANGE(0..3)]
    └── RANGE(0..4, Reduce)
```

The `EXPAND` became a `RANGE(0..4)` that doesn't affect the buffer index—broadcasting. The `RESHAPE` became different index arithmetic. The `SUM` became `REDUCE(Add)` with the first range marked as `Reduce` type.

### Movement → Index Arithmetic

Each movement operation has a specific transformation:

| Operation | Transformation |
|-----------|----------------|
| **RESHAPE** | Flatten/unflatten index expressions |
| **PERMUTE** | Reorder dimensions in INDEX |
| **EXPAND** | Index becomes 0 (or range doesn't affect index) |
| **PAD** | WHERE(in_bounds, LOAD, pad_value) |
| **SHRINK** | Offset adjustment in INDEX |
| **FLIP** | `size - 1 - index` |

After rangeify, there are no more movement ops—just arithmetic operations on indices.

---

## Kernel Splitting: Finding the Boundaries

A computation graph might have multiple outputs, or intermediate values that need materialization. **Kernel splitting** identifies these boundaries and creates separate kernels.

The entry point is `run_kernel_split_pipeline()` in `schedule/src/rangeify/kernel.rs`.

### Two-Phase Transformation

**Phase 1: BUFFERIZE → STORE**

`BUFFERIZE` nodes mark where values should materialize. Phase 1 converts them to explicit `STORE` operations:

```text
Before: BUFFERIZE(computation, ranges)
After:  END(STORE(buffer, INDEX(...), computation), ranges)
```

The `END` wrapper captures which ranges scope this store. Buffers are allocated and assigned IDs during this phase.

**Phase 2: STORE → KERNEL**

Each `STORE` becomes its own kernel:

```text
Before: END(STORE(...), ranges)
After:  KERNEL(SINK(STORE(...)), ranges, buffer_list)
```

The `KERNEL` node wraps everything: the computation (as a `SINK`), the iteration ranges, and the list of buffers this kernel reads and writes.

### Tracking Dependencies

When one kernel's output feeds another kernel's input, we need dependency tracking:

1. `fix_assign()` maps each buffer_id to the kernel that writes it
2. When kernel B reads a buffer written by kernel A, B depends on A
3. `resolve_kernel_dependencies()` builds the dependency graph

Dependencies appear as `AFTER` nodes in the IR, ensuring kernels execute in valid order.

### Buffer Renumbering

Each kernel sees buffers in a specific order (outputs first, then inputs). `renumber_define_globals()` remaps buffer IDs to match this ordering:

```text
Original: buffer_3, buffer_1, buffer_7
Kernel view: buffer_0 (output), buffer_1, buffer_2 (inputs)
```

This simplifies code generation—buffer `N` is always argument `N`.

---

## Schedule Creation: Preparing for Execution

Once kernels are split, we need to **schedule** them: determine execution order, allocate buffers, and prepare for compilation.

`create_schedule()` in `tensor/src/schedule.rs` produces a `Vec<ScheduleItem>`:

```rust
pub struct ScheduleItem {
    pub kernel: Arc<UOp>,              // KERNEL wrapper
    pub ast: Arc<UOp>,                 // Inner computation (for codegen)
    pub buffers: Vec<Buffer>,          // Device buffers
    pub dependencies: Vec<u64>,        // Producer kernel IDs
    pub fixedvars: HashMap<String, i64>,  // Bound iteration variables
}
```

### Buffer Allocation Strategy

- **Input buffers**: Already allocated (from `Tensor::from_slice`)
- **Intermediate buffers**: Allocated during scheduling (for kernel outputs that feed other kernels)
- **Output buffer**: Allocated and registered with the final tensor

### Parallel Group Analysis

Not all kernels need sequential execution. Independent kernels can run in parallel:

```text
Kernel A (writes buf0)
Kernel B (writes buf1)  ─── no dependency ─── can run in parallel
Kernel C (reads buf0, buf1)  ─── depends on A and B
```

The scheduler uses **Kahn's algorithm** to find parallel groups:

1. Build the kernel dependency DAG
2. Find all kernels with no incoming edges → Group 1
3. Remove Group 1, repeat → Group 2, etc.

Each group's kernels execute in parallel, then the next group starts.

---

## Code Generation: From UOp to LLVM IR

With kernels scheduled, we generate actual code. Morok currently supports two backends:

| Backend | Compile Speed | Output Quality | Use Case |
|---------|---------------|----------------|----------|
| **LLVM** | Slower | Highly optimized | Production |
| **Cranelift** | Faster | Good | Development/testing |

The `Renderer` trait abstracts code generation:

```rust
pub trait Renderer {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel>;
}
```

### LLVM CPU Renderer

The LLVM renderer (`codegen/src/llvm/cpu/`) traverses the UOp graph and emits LLVM IR:

```llvm
define void @kernel_0(ptr %args, ptr %vars) {
entry:
  %buf0 = load ptr, ptr %args
  %buf1 = load ptr, ptr getelementptr(ptr, ptr %args, i64 1)
  ; ... loop nest ...
  br label %loop_0

loop_0:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop_0 ]
  ; ... computation ...
  %i.next = add i64 %i, 1
  %cond = icmp slt i64 %i.next, 128
  br i1 %cond, label %loop_0, label %exit

exit:
  ret void
}
```

The generated kernel takes two arguments:
- `args`: Array of buffer pointers
- `vars`: Array of symbolic variable values (for dynamic shapes)

### Post-Optimization Passes

Before code generation, 13+ pattern-based passes clean up the IR:

| Pass | Purpose |
|------|---------|
| `pm_add_loads` | Wrap INDEX operations in LOAD |
| `pre_expand` | Convert UNROLL/UPCAST ranges to explicit operations |
| `devectorize` | Group contiguous memory accesses |
| `pm_reduce_devectorize` | Handle vector reductions (K-vec, bool, horizontal) |
| `pm_fma_decomposition` | Convert `a*b+c` to fused multiply-add |
| `bool_storage_patterns` | Convert bool ↔ uint8 for memory operations |

These passes transform the optimized AST into a form suitable for code generation. The result is clean, vectorized code with proper memory access patterns.

---

## Execution: Running the Kernels

Code generation produces LLVM IR strings. Execution involves JIT compilation and kernel launch.

### The ExecutionPlan

`prepare_execution_plan()` builds an `ExecutionPlan`:

```rust
pub struct ExecutionPlan {
    kernels: Vec<PreparedKernel>,       // Compiled kernels
    parallel_groups: Vec<ParallelGroup>,
    buffers: Vec<Buffer>,
    output_buffer_idx: usize,
}
```

The plan is **reusable**: compile once, execute many times with different data.

### JIT Compilation

The LLVM runtime (`runtime/src/llvm.rs`) compiles IR to machine code:

1. **Parse** the LLVM IR string into a module
2. **Verify** the module is well-formed
3. **Optimize** with LLVM's O3 pass pipeline
4. **JIT compile** to native machine code
5. **Cache** by (AST ID, device) for reuse

```rust
// Simplified JIT flow
let module = Module::parse_ir(context, ir_string)?;
module.verify()?;
pass_manager.run(&module);  // O3 optimization
let function = execution_engine.get_function::<KernelFn>(&name)?;
// Cache: (ast_id, device) → function
```

### Parallel Execution

With kernels compiled, execution follows the parallel groups:

```rust
for group in &plan.parallel_groups {
    if group.kernel_indices.len() == 1 {
        // Single kernel: direct call
        execute_kernel(&kernels[group.kernel_indices[0]]);
    } else {
        // Multiple kernels: parallel execution
        rayon::scope(|s| {
            for &idx in &group.kernel_indices {
                s.spawn(|_| execute_kernel(&kernels[idx]));
            }
        });
    }
}
```

Independent kernels run in parallel using Rayon's work-stealing scheduler.

### Kernel Caching

Hash consing makes kernel caching highly effective:

- **Key**: `(UOp ID, device string)`
- **Storage**: Lock-free HashMap (papaya crate)
- **Hit rate**: High, because identical computations share UOp IDs

When you compute the same expression twice, the second call hits the cache—no recompilation.

---

## Worked Example: Matrix Multiply

Let's trace `C = A @ B` through the entire pipeline. Assume 4×4 matrices.

### Stage 1: Lazy Graph Construction

```rust
let a = Tensor::from_slice(&a_data, &[4, 4])?;  // Input buffer allocated
let b = Tensor::from_slice(&b_data, &[4, 4])?;  // Input buffer allocated
let c = a.matmul(&b)?;                           // Graph built, no computation
```

At this point, `c` is a lazy tensor with this UOp graph:

```text
REDUCE_AXIS(Add, axis=2)
└── MUL
    ├── EXPAND(A, [4, 4, 4])    — A: [4, 4] → [4, 1, 4] → [4, 4, 4]
    └── EXPAND(B, [4, 4, 4])    — B: [4, 4] → [1, 4, 4] → [4, 4, 4]
```

### Stage 2: Rangeify

Movement ops become explicit loops:

```text
STORE
├── BUFFER(C)
├── INDEX[RANGE(i, 0..4), RANGE(j, 0..4)]
└── REDUCE(Add)
    ├── MUL
    │   ├── LOAD(A)
    │   │   └── INDEX[RANGE(i), RANGE(k, 0..4, Reduce)]
    │   └── LOAD(B)
    │       └── INDEX[RANGE(k), RANGE(j)]
    └── RANGE(k, Reduce)
```

The `i` and `j` ranges are output dimensions. The `k` range is the reduction (contracted) dimension.

### Stage 3: Kernel Splitting

Single STORE → single KERNEL:

```text
KERNEL
├── SINK(STORE(...))
├── ranges: [i: 0..4, j: 0..4]
└── buffers: [C (output), A (input), B (input)]
```

### Stage 4: Schedule

One `ScheduleItem` with:
- `kernel`: The KERNEL UOp
- `ast`: The inner SINK/STORE
- `buffers`: [C, A, B]
- `dependencies`: [] (no prior kernels)

### Stage 5: Optimization

Heuristic optimizer applies:
- Vectorization: UPCAST j dimension by 4
- Loop ordering: Ensure good cache behavior

### Stage 6: Code Generation

Generated LLVM IR (simplified):

```llvm
define void @matmul(ptr %args, ptr %vars) {
entry:
  %C = load ptr, ptr %args
  %A = load ptr, ptr getelementptr(ptr, ptr %args, i64 1)
  %B = load ptr, ptr getelementptr(ptr, ptr %args, i64 2)
  br label %loop_i

loop_i:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop_i.end ]
  br label %loop_j

loop_j:
  %j = phi i64 [ 0, %loop_i ], [ %j.next, %loop_k.end ]
  %acc = ... ; initialize accumulator
  br label %loop_k

loop_k:
  %k = phi i64 [ 0, %loop_j ], [ %k.next, %loop_k ]
  %a_val = load float, ptr ...  ; A[i, k]
  %b_val = load float, ptr ...  ; B[k, j]
  %prod = fmul float %a_val, %b_val
  %acc.new = fadd float %acc, %prod
  %k.next = add i64 %k, 1
  %k.cond = icmp slt i64 %k.next, 4
  br i1 %k.cond, label %loop_k, label %loop_k.end

loop_k.end:
  store float %acc.new, ptr ...  ; C[i, j]
  ; ... continue j, i loops
}
```

### Stage 7: Execution

1. JIT compile the LLVM IR
2. Execute: `kernel([C_ptr, A_ptr, B_ptr], [])`
3. Result is in C buffer

Total: one function call, result ready.

---

## Comparison: How Other Frameworks Execute

| Aspect | PyTorch | JAX | TVM | **Morok** |
|--------|---------|-----|-----|-----------|
| **Evaluation** | Eager (immediate) | Traced (jit decorator) | Lazy (te.compute) | Lazy (realize) |
| **Graph capture** | torch.compile | jax.jit trace | Explicit schedule | Implicit via ops |
| **Compilation** | TorchInductor | XLA backend | Auto-scheduler | Pattern + beam |
| **Caching** | Per-graph hash | Per-trace | Per-schedule | Per-AST (hash consing) |
| **Parallelism** | DataParallel/DDP | pmap/pjit | Parallel schedule | Parallel groups |

**PyTorch**: Eager by default, torch.compile for optimization. TorchInductor generates Triton or C++ code.

**JAX**: Functional transformations (jit, grad, vmap) trace computations. XLA compiles to optimized kernels.

**TVM**: Explicit separation of computation and schedule. Auto-scheduler searches for good schedules.

**Morok**: Fully lazy—nothing executes until `realize()`. Hash consing provides automatic caching. Pattern-based optimization with optional beam search for production quality.

---

## The Deeper Insight

The pipeline embodies several design principles:

**Lazy evaluation enables global optimization.** By deferring computation, we see the entire graph before generating code. No local decision limits global optimization.

**Explicit loops enable hardware-specific scheduling.** Movement ops are convenient abstractions, but GPUs need loops. Rangeify bridges the gap.

**Hash consing makes caching automatic.** Identical computations share pointers, so cache keys are trivial. No complex graph hashing needed.

**Separation of concerns keeps each stage simple.** Rangeify doesn't know about LLVM. Code generation doesn't know about tensor semantics. Each stage does one thing well.

The result: a compilation pipeline that's both powerful and maintainable. From `tensor.realize()` to machine code, every step is visible, debuggable, and extensible.
