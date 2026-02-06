# Op Bestiary: A Field Guide to UOp Operations

When debugging Morok IR dumps, you'll encounter operations that aren't obvious from their names. This chapter documents non-trivial operations with signatures, field explanations, and examples.

**What's covered:** Operations that require explanation—loop control, reductions, memory operations, kernel structure, vectorization, tensor cores.

**What's NOT covered:** Trivial ALU operations (`Add`, `Mul`, `Sqrt`, etc.) that work exactly as you'd expect.

---

## Loop Control: RANGE and END

### RANGE — Loop Scope Opener

```rust
Range {
    end: Arc<UOp>,           // loop bound (exclusive)
    axis_id: AxisId,         // identifier for deduplication
    axis_type: AxisType,     // scheduling behavior
}
```

**Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `end` | `Arc<UOp>` | Upper bound (exclusive), typically a `CONST` |
| `axis_id` | `AxisId` | `Unrenumbered(n)` before kernel splitting, `Renumbered(n)` after |
| `axis_type` | `AxisType` | Determines how the loop is scheduled (see below) |

**AxisType Hierarchy:**

| Type | Priority | GPU Mapping | Purpose |
|------|----------|-------------|---------|
| `Outer` | -2 | — | Kernel boundary marker |
| `Loop` | -1 | `for` loop | Sequential iteration |
| `Global` | 0 | `blockIdx` | Grid parallelism |
| `Thread` | 0 | thread pool | CPU parallelism |
| `Warp` | 1 | warp/wavefront | Sub-group parallelism |
| `Local` | 2 | `threadIdx` | Workgroup parallelism |
| `GroupReduce` | 2 | shared memory | Two-stage reduction |
| `Upcast` | 3 | SIMD | Vectorization |
| `Reduce` | 4 | accumulator | Reduction dimension |
| `Unroll` | 5 | unrolled | Loop unrolling |

Priority determines loop nesting order—lower values are outer loops.

**Example:**
```text
RANGE(end=128, axis_id=R0, type=Global)
└── CONST(128) : Index
```

### END — Loop Scope Closer

```rust
End {
    computation: Arc<UOp>,              // value computed inside loop
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being closed
}
```

END closes one or more RANGE scopes and removes them from the active set. Multiple ranges can be closed simultaneously.

**Example:**
```text
END
├── STORE(...)           — computation
├── RANGE(R0, Global)    — first range closed
└── RANGE(R1, Local)     — second range closed
```

---

## Reduction: REDUCE vs REDUCE_AXIS

Two operations with similar names serve different purposes.

### REDUCE_AXIS — Tensor Dimension Reduction (High-Level)

```rust
ReduceAxis {
    src: Arc<UOp>,           // input tensor
    reduce_op: ReduceOp,     // Add, Mul, Max, Min
    axes: Vec<usize>,        // axes to reduce
}
```

Used **before** rangeify. Operates on tensor dimensions like NumPy's `.sum(axis=0)`.

**Example:**
```text
REDUCE_AXIS(Add, axes=[1])
└── BUFFER[10, 20] : Float32
```

This reduces a `[10, 20]` tensor to `[10]` by summing along axis 1.

### REDUCE — Range Iteration Reduction (Low-Level)

```rust
Reduce {
    src: Arc<UOp>,                      // value to accumulate
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being reduced
    reduce_op: ReduceOp,                // Add, Mul, Max, Min
}
```

Used **after** rangeify. Accumulates values across RANGE iterations and closes the specified ranges.

**ReduceOp Variants:**

| Op | Identity | Operation | Tinygrad |
|----|----------|-----------|----------|
| `Add` | 0 | `acc + value` | ✓ |
| `Mul` | 1 | `acc * value` | ✓ |
| `Max` | -∞ | `max(acc, value)` | ✓ |
| `Min` | +∞ | `min(acc, value)` | Morok-only |

> **Compatibility:** Tinygrad's spec restricts REDUCE_AXIS to `{Add, Mul, Max}`. Morok extends this with `Min`.

**Example:**
```text
REDUCE(Add)
├── MUL                      — value to accumulate
│   ├── LOAD(A, ...)
│   └── LOAD(B, ...)
└── RANGE(R2, Reduce)        — range being reduced
    └── CONST(64)
```

### ALLREDUCE — Cross-Device Reduction

```rust
AllReduce {
    src: Arc<UOp>,           // local partial result
    device: Arc<UOp>,        // device specification
    reduce_op: ReduceOp,     // reduction operation
}
```

Performs distributed reduction across multiple devices. Used for multi-GPU training.

---

## Buffer Operations

### BUFFER — Buffer Declaration

```rust
Buffer {
    unique: Arc<UOp>,        // UNIQUE op for identity
    device: Arc<UOp>,        // DEVICE op
    size: usize,             // total element count
}
```

Declares a buffer for tensor storage. The `unique` field ensures distinct buffers even with identical size/device.

### BUFFERIZE — Materialization Marker

```rust
Bufferize {
    compute: Arc<UOp>,                  // computation to materialize
    ranges: SmallVec<[Arc<UOp>; 4]>,    // output dimensions
    opts: BufferizeOpts,                // address space, device
}
```

Marks where computation should materialize to memory. Triggers kernel splitting.

**BufferizeOpts:**

| Field | Type | Purpose |
|-------|------|---------|
| `device` | `Option<DeviceSpec>` | Target device, `None` for local |
| `addrspace` | `AddrSpace` | `Global` (device) or `Local` (shared) |

**Example:**
```text
BUFFERIZE(opts={addrspace=Global})
├── REDUCE(Add, ...)         — computation
├── RANGE(R0, Global)        — output dim 0
└── RANGE(R1, Global)        — output dim 1
```

### INDEX — Multi-Dimensional Buffer Access

```rust
Index {
    buffer: Arc<UOp>,                   // BUFFER or DEFINE_GLOBAL
    indices: SmallVec<[Arc<UOp>; 4]>,   // index per dimension
    gate: Option<Arc<UOp>>,             // optional predicate
}
```

Computes memory address from multi-dimensional indices. Returns element dtype (not pointer).

**Example:**
```text
INDEX : Float32
├── DEFINE_GLOBAL(0)
├── RANGE(R0, Global)        — index for dim 0
├── RANGE(R1, Loop)          — index for dim 1
└── MUL(...)                 — index for dim 2
```

### POINTER_INDEX — Low-Level Pointer Arithmetic

```rust
PointerIndex {
    ptr: Arc<UOp>,           // base pointer
    offset: Arc<UOp>,        // byte offset
}
```

Direct pointer arithmetic. Used after linearization when indices are flattened.

> **Compatibility:** Tinygrad uses `INDEX` with a `ptr=True` flag instead of a separate operation.

### LOAD — Memory Read

```rust
Load {
    buffer: Arc<UOp>,        // buffer or pointer
    index: Arc<UOp>,         // INDEX op
}
```

Read value from buffer at index. For gated loads, use an INDEX with a gate (INDEX has an optional `gate` field).

**Example:**
```text
LOAD : Float32
├── DEFINE_GLOBAL(1)
└── INDEX
    ├── DEFINE_GLOBAL(1)
    ├── RANGE(R0)
    └── RANGE(R2)
```

### STORE — Memory Write

```rust
Store {
    buffer: Arc<UOp>,                   // output buffer
    index: Arc<UOp>,                    // INDEX op
    value: Arc<UOp>,                    // value to write
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being closed
}
```

Write value to buffer. STORE closes the specified ranges, which represent output iteration dimensions. The ranges field is used for output upcasting: when a `Range(Upcast)` is included, it becomes `UNROLL` during expansion, then contracted via `CONTRACT`.

For gated stores, use an INDEX with a gate (INDEX has an optional `gate` field).

> **Compatibility:** Morok's STORE has an explicit `index` field (sources: buffer=0, index=1, value=2, ranges=3+). Tinygrad's STORE combines buffer and value differently (range_start=2).

**Example:**
```text
STORE
├── DEFINE_GLOBAL(0)         — output buffer
├── INDEX[R0, R1]            — write address
├── REDUCE(Add, ...)         — value
├── RANGE(R0, Global)        — output dim 0 (closed)
└── RANGE(R1, Global)        — output dim 1 (closed)
```

---

## Kernel Structure

### KERNEL — Kernel Wrapper

```rust
Kernel {
    sources: SmallVec<[Arc<UOp>; 4]>,   // arguments
    ast: Arc<UOp>,                       // computation (usually SINK)
}
```

Wraps a complete kernel for code generation. Sources are kernel arguments (`DefineGlobal`, `DefineLocal`, `DefineVar`).

**Example:**
```text
KERNEL
├── DEFINE_GLOBAL(0)         — output buffer arg
├── DEFINE_GLOBAL(1)         — input A arg
├── DEFINE_GLOBAL(2)         — input B arg
└── SINK                     — computation
    └── STORE(...)
```

### SINK — Multiple Root Collector

```rust
Sink {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

Collects multiple outputs into a single root. Every kernel's `ast` is typically a SINK containing STORE operations.

**Example:**
```text
SINK
├── STORE(output_0, ...)
├── STORE(output_1, ...)
└── STORE(output_2, ...)
```

### AFTER — Dependency Marker

```rust
After {
    passthrough: Arc<UOp>,              // value that flows through
    deps: SmallVec<[Arc<UOp>; 4]>,      // operations that must complete
}
```

Expresses execution dependencies between kernels without data dependency. The `passthrough` value is returned unchanged, but only after all `deps` complete.

**Example:**
```text
SINK
├── AFTER
│   ├── DEFINE_GLOBAL(0)     — passthrough (buffer reference)
│   └── KERNEL(...)          — must complete first
└── KERNEL(...)              — can use buffer after AFTER
```

### BARRIER — Synchronization Fence

```rust
Barrier {
    src: Arc<UOp>,                      // value passing through
    deps: SmallVec<[Arc<UOp>; 4]>,      // operations to wait for
}
```

GPU workgroup synchronization. Ensures all threads in a workgroup reach the barrier before continuing.

---

## Vector Operations

### VECTORIZE — Create Vector from Scalars

```rust
Vectorize {
    elements: SmallVec<[Arc<UOp>; 4]>,
}
```

Combines N scalar values into a vector of size N. All elements must have the same base dtype.

**Example:**
```text
VECTORIZE : <4 x Float32>
├── CONST(1.0)
├── CONST(2.0)
├── CONST(3.0)
└── CONST(4.0)
```

### GEP — Get Element Pointer (Vector Extract)

```rust
Gep {
    vector: Arc<UOp>,        // source vector
    indices: Vec<usize>,     // positions to extract
}
```

Extracts elements from a vector:
- Single index → scalar
- Multiple indices → smaller vector

**Example:**
```text
GEP([0, 2]) : <2 x Float32>
└── VECTORIZE : <4 x Float32>
    └── ...
```

### VConst — Vector Constant

```rust
VConst {
    values: Vec<ConstValue>,
}
```

Vector of compile-time constants. More efficient than `VECTORIZE` of `CONST` nodes.

### CAT — Concatenate Vectors

```rust
Cat {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

Concatenates vectors into a larger vector. Output `vcount` = sum of input `vcount`s.

**Example:**
```text
CAT : <8 x Float32>
├── VECTORIZE : <4 x Float32>
└── VECTORIZE : <4 x Float32>
```

### PtrCat — Concatenate Pointers

```rust
PtrCat {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

Groups memory accesses for vectorized load/store. Used by the devectorizer pass.

---

## Expansion: UNROLL and CONTRACT

### UNROLL — Expand Computation Across Iterations

```rust
Unroll {
    src: Arc<UOp>,                       // computation to expand
    unroll_axes: Vec<(usize, usize)>,    // (axis_index, factor) pairs
}
```

Creates multiple versions of computation for different iteration values. Used for loop unrolling optimization.

**Example:** `UNROLL(unroll_axes=[(0, 4)])` expands computation 4 times with different index values.

### CONTRACT — Collapse Unrolled Values to Vector

```rust
Contract {
    src: Arc<UOp>,                       // unrolled computation
    upcast_ranges: Vec<(usize, usize)>,  // (axis_index, factor) pairs
}
```

The inverse of UNROLL—collects expanded scalar values into a vector. Output vector size = product of factors.

**Example:**
```text
CONTRACT(upcast_ranges=[(0, 4)]) : <4 x Float32>
└── UNROLL(unroll_axes=[(0, 4)])
    └── LOAD(...)
```

This pattern vectorizes a load: expand 4 iterations, then pack results into a 4-element vector.

---

## Tensor Cores: WMMA

### WMMA — Warp Matrix Multiply-Accumulate

```rust
Wmma {
    a: Arc<UOp>,             // matrix A fragment
    b: Arc<UOp>,             // matrix B fragment
    c: Arc<UOp>,             // accumulator C fragment
    metadata: WmmaMetadata,  // hardware configuration
}
```

Hardware tensor core operation: `D = A × B + C`. Requires specific matrix shapes and data layouts.

**WmmaMetadata Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `name` | `String` | Instruction name (e.g., `"__hmma..."`) |
| `dims` | `(N, M, K)` | Matrix dimensions (e.g., `(16, 16, 16)`) |
| `dtype_in` | `DType` | Input matrix precision (e.g., `Float16`) |
| `dtype_out` | `DType` | Output precision (e.g., `Float32`) |
| `device` | `String` | Target device string |
| `threads` | `usize` | Threads per warp (typically 32) |
| `upcast_axes` | `Vec<(usize, usize)>` | Vectorization for output |
| `reduce_axes` | `Vec<(usize, usize)>` | Contraction axes |

**Example:**
```text
WMMA(dims=(16, 16, 16), dtype_in=Float16, dtype_out=Float32)
├── A fragment : <8 x Float16>
├── B fragment : <8 x Float16>
└── C accumulator : <8 x Float32>
```

---

## Control Flow

### IF / ENDIF — Conditional Execution

```rust
If {
    condition: Arc<UOp>,                // boolean predicate
    body: SmallVec<[Arc<UOp>; 4]>,      // operations to execute
}

EndIf {
    if_op: Arc<UOp>,         // corresponding IF op
}
```

Execute body only when condition is true. Used for boundary checks and sparse operations.

**Example:**
```text
IF
├── LT(idx, bound)           — condition (src[0])
├── STORE(...)               — body[0]
└── STORE(...)               — body[1]

ENDIF
└── IF(...)                  — references IF op
```

---

## Definition Operations

### DEFINE_GLOBAL — Device Memory Argument

```rust
DefineGlobal(usize)          // argument index
```

Kernel argument for device (global) memory. Index refers to position in kernel argument list.

### DEFINE_LOCAL — Shared Memory Allocation

```rust
DefineLocal(usize)           // local memory index
```

GPU shared memory (LDS) allocation. Visible within a workgroup.

### DEFINE_VAR — Symbolic Runtime Variable

```rust
DefineVar {
    name: String,            // variable name
    min_val: i64,            // minimum bound
    max_val: i64,            // maximum bound
}
```

Runtime variable with known bounds. Used for dynamic shapes where bounds are known.

**Example:**
```text
DEFINE_VAR(name="batch_size", min=1, max=128) : Index
```

### DEFINE_REG — Register Allocation

```rust
DefineReg {
    size: usize,             // register size
}
```

Allocates a register for intermediate storage. Used in code generation.

### BIND — Variable Binding

```rust
Bind {
    var: Arc<UOp>,           // DEFINE_VAR
    value: Arc<UOp>,         // concrete value
}
```

Binds a symbolic variable to a concrete value at runtime.

---

## Special Operations

### SPECIAL — Hardware-Provided Values

```rust
Special {
    end: Arc<UOp>,           // upper bound for this dimension
    name: String,            // e.g., "blockIdx.x", "threadIdx.y"
}
```

Accesses hardware-provided values (thread/block indices). Not a loop—the hardware provides the value directly.

**Example:**
```text
SPECIAL(name="blockIdx.x", end=128) : Index
└── CONST(128)
```

### UNIQUE — Identity Marker

```rust
Unique(usize)                // unique identifier
```

Creates a unique identity for buffer disambiguation. Two buffers with different UNIQUE values are distinct even if otherwise identical.

### DEVICE — Device Specification

```rust
Device(DeviceSpec)           // device specification
```

Specifies target device for computation.

---

## Movement Operations

High-level tensor shape transformations. These are converted to explicit INDEX operations during rangeify.

| Operation | Signature | Purpose |
|-----------|-----------|---------|
| `Reshape` | `{ src, new_shape }` | Change shape, same elements |
| `Permute` | `{ src, axes: Vec<usize> }` | Transpose/reorder axes |
| `Expand` | `{ src, new_shape }` | Broadcast to larger shape |
| `Pad` | `{ src, begin_pads, end_pads }` | Add padding |
| `Shrink` | `{ src, begins, ends }` | Extract sub-region |
| `Flip` | `{ src, axes: Vec<bool> }` | Reverse along axes |

**Example:** RESHAPE
```text
RESHAPE(new_shape=[6, 4]) : Shape[6, 4]
├── BUFFER[2, 3, 4] : Float32
└── CONST([6, 4]) : Shape
```

---

## Quick Reference

### By Category

| Category | Operations |
|----------|------------|
| **Loop Control** | `RANGE`, `END` |
| **Reduction** | `REDUCE_AXIS`, `REDUCE`, `ALLREDUCE` |
| **Memory** | `BUFFER`, `BUFFERIZE`, `INDEX`, `POINTER_INDEX`, `LOAD`, `STORE` |
| **Kernel** | `KERNEL`, `SINK`, `AFTER`, `BARRIER` |
| **Vector** | `VECTORIZE`, `GEP`, `VCONST`, `CAT`, `PTRCAT` |
| **Expansion** | `UNROLL`, `CONTRACT` |
| **Hardware** | `WMMA`, `SPECIAL` |
| **Control** | `IF`, `ENDIF` |
| **Definition** | `DEFINE_GLOBAL`, `DEFINE_LOCAL`, `DEFINE_VAR`, `DEFINE_REG`, `BIND`, `UNIQUE`, `DEVICE` |
| **Movement** | `RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP` |
| **ALU** | `Unary(...)`, `Binary(...)`, `Ternary(...)`, `Cast`, `BitCast` |

### Range-Ending Operations

Operations that close RANGE scopes (remove ranges from active set):

| Operation | Range Start Index |
|-----------|-------------------|
| `BUFFERIZE` | 1 (compute=0, ranges=1+) |
| `REDUCE` | 1 (src=0, ranges=1+) |
| `STORE` | 3 (buffer=0, index=1, value=2, ranges=3+) |
| `WMMA` | 3 (a=0, b=1, c=2) |
| `END` | 1 (computation=0, ranges=1+) |

### Expandable Operations

Operations that propagate UNROLL through the computation graph:

- ALU: `Unary`, `Binary`, `Ternary`
- Type: `Cast`, `BitCast`
- Vector: `Gep`, `Vectorize`
- Memory: `Load`, `Store`, `Index`, `PointerIndex`
- Control: `Reduce`, `End`, `After`
- Buffer: `Bufferize`
- Hardware: `Wmma`
