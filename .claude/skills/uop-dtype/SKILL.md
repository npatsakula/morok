---
name: uop-dtype
description: Reference for UOp and DType constructors, methods, and usage patterns. Use when constructing IR, debugging UOp creation, or understanding type system.
---

# UOp and DType Reference

## UOp Constructors by Category

### Data Constructors (`ir/src/uop/constructors/data.rs`)

| Method | Purpose | Signature |
|---------|---------|------------|
| `const_(dtype, value)` | Create constant with explicit dtype | `pub fn const_(dtype: DType, value: ConstValue) -> Arc<Self>` |
| `native_const<T>(value)` | Auto-infer dtype from Rust type | `pub fn native_const<T: HasDType + IntoUOp>(value: T) -> Arc<Self>` |
| `index_const(value)` | Create Index constant | `pub fn index_const(value: i64) -> Arc<Self>` |
| `const_like<T>(self, value)` | Create constant with same dtype as self | `pub fn const_like<T: IntoUOp>(self: &Arc<Self>, value: T) -> Arc<Self>` |
| `vconst(values)` | Vector constant from multiple values | `pub fn vconst(values: Vec<ConstValue>) -> Arc<Self>` |
| `new_buffer(device, size, dtype)` | Create buffer with device spec | `pub fn new_buffer(device: DeviceSpec, size: usize, dtype: DType) -> Arc<Self>` |
| `view(self, size, offset)` | Create buffer view | `pub fn view(self: &Arc<Self>, size: usize, offset: usize) -> Arc<Self>` |
| `device(device_spec)` | Create device specification | `pub fn device(device: DeviceSpec) -> Arc<Self>` |
| `noop()` | No-operation | `pub fn noop() -> Arc<Self>` |
| `cast(self, dtype)` | Cast to different dtype | `pub fn cast(self: &Arc<Self>, dtype: DType) -> Arc<Self>` |
| `bitcast(self, dtype)` | Reinterpret bits as different type | `pub fn bitcast(self: &Arc<Self>, dtype: DType) -> Arc<Self>` |

### Compute Operations (`ir/src/uop/constructors/compute.rs`)

**Binary** (`Result<Arc<Self>>`): `try_add`, `try_sub`, `try_mul`, `try_div`, `try_mod`, `try_max`, `try_pow`, comparisons (`eq/ne/lt/le/gt/ge`), bitwise (`and_op/or_op/xor_op`), shifts (`shl_op/shr_op`), `try_mulacc(a,b,c)` (fused multiply-accumulate)

**Unary** (infallible, return `Arc<Self>`): `neg`, `abs`, `square`, `sign`, `not`

**Transcendental** (`Result<Arc<Self>>`, require float): `try_sqrt`, `try_rsqrt`, `try_exp/exp2`, `try_log/log2`, `try_sin/cos/tan`, `try_erf`, `try_reciprocal`

**Rounding** (infallible, return `Arc<Self>`): `trunc`, `floor`, `ceil`, `round`

### Control Flow (`ir/src/uop/constructors/control.rs`)

| Method | Purpose |
|---------|---------|
| `range_axis(end, axis_id, axis_type)` | Create RANGE with specific axis type |
| `range(end, axis_id)` | Convenience: Loop axis type |
| `range_const(end_value, axis_id)` | RANGE with constant end (Loop type) |
| `range_outer_const(end_value, axis_id)` | RANGE with Outer axis type |
| `if_(condition, body)` | Conditional block |
| `endif(if_op)` | End IF block |
| `end(self, ranges)` | End of range/reduce scope |
| `barrier(self, deps)` | Synchronization barrier |
| `var(name, dtype, min_val, max_val)` | Define symbolic variable with bounds |
| `define_var(name, min_val, max_val)` | Define variable (Index dtype) |
| `bind(self, value)` | Bind value to variable |
| `special(end, name)` | GPU dimension variable (e.g., blockIdx.x) |

### Memory Operations (`ir/src/uop/constructors/memory.rs`)

**Indexing**:
```rust
// Builder pattern for INDEX
UOp::index()
    .buffer(buffer_uop)
    .indices(vec![index1, index2])
    .gate(gate_uop)        // Optional: WHERE validity condition
    .ptr(true)             // Optional: keep Ptr dtype for STORE targets
    .call()?              // Returns Result<Arc<Self>>

// Index validity
idx.valid(condition)  // Equivalent to WHERE(condition, idx, INVALID)
```

**Load/Store**:
```rust
// Builder pattern for LOAD
UOp::load()
    .buffer(buffer_uop)
    .index(index_uop)
    .dtype(vec4_dtype)  // Optional: explicit dtype
    .alt(zero_uop)      // Optional: alternative value for gated loads
    .call()             // Returns Arc<Self>

// STORE operations
index_uop.store(value_uop)              // Simple store
index_uop.store_with_ranges(value_uop, ranges)  // With ranges for output upcasting
```

**Memory Definitions**:
- `define_global(id, dtype)` - Global memory allocation
- `define_local(id, dtype)` - Local/shared memory allocation
- `define_reg(size)` - Register memory (void pointer)
- `define_reg_typed(size, element_dtype)` - Typed register accumulator

**Bufferization**:
- `bufferize(compute, ranges, opts)` - Materialize to buffer
- `bufferize_global(compute, ranges)` - Bufferize to global memory
- `bufferize_local(compute, ranges)` - Bufferize to local/shared memory

### Shape Operations (`ir/src/uop/constructors/shape.rs`)

| Method | Validation | Purpose |
|---------|------------|---------|
| `try_reshape(new_shape)` | No negative dims, product matches | Change shape preserving elements |
| `try_permute(axes)` | Valid permutation (0..n each exactly once) | Reorder dimensions |
| `try_expand(new_shape)` | Size-1 dims can expand, same rank | Broadcast from size-1 dims |
| `try_pad(padding)` | Concrete padding values only | Add padding |
| `try_shrink(ranges)` | Concrete ranges, begin ≤ end | Slice/subset |
| `try_flip(axes)` | Length matches shape dims | Reverse axes |

### Reduction Operations (`ir/src/uop/constructors/reduce.rs`)

| Method | Behavior |
|---------|-----------|
| `try_reduce_axis(reduce_op, axes)` | Reduces along tensor axes; early-returns self if all axes have dim=1 |
| `reduce(ranges, reduce_op)` | Reduces across loop ranges (for kernels) |
| `allreduce(src, device, reduce_op)` | All-reduce across devices |

### Hardware Operations (`ir/src/uop/constructors/hardware.rs`)

**Vectorization**:
```rust
// Create vector from scalars
UOp::try_vectorize(vec![a, b, c, d])?  // Returns Result<Arc<Self>>

// Broadcast scalar to vector
scalar_uop.broadcast(count)  // Replicates scalar count times

// Extract element(s) from vector
vector_uop.gep(vec![0])          // Single element
vector_uop.gep(vec![0, 2])        // Multiple elements (sub-vector)

// Unroll expansion
UOp::unroll(src, vec![(0, 4), (1, 8)])  // For axes 0 and 1
```

**WMMA (Tensor Cores)**:
```rust
UOp::wmma(a, b, c, WmmaMetadata {
    dtype_out: DType::Float32,
    upcast_axes: vec![(0, 16), (1, 16)],
})
```

### Graph Organization (`ir/src/uop/constructors/graph.rs`)

| Method | Purpose |
|---------|---------|
| `sink(sources)` | Graph termination mark, all sources are dependencies |
| `group(sources)` | Organize related operations, passes through first source |
| `assign(target, value)` | In-place assignment at INDEX location |
| `assign_with_mops(target, value, movement_ops)` | Assignment with movement ops for bufferization |
| `after(self, deps)` | Ordering constraint: self depends on deps completing |
| `detach(self)` | Detach from gradient flow / force materialization |
| `contiguous(self)` | Ensure contiguous memory layout |
| `contiguous_with_opts(self, opts)` | Contiguous with optimization hints |
| `precast(self)` | Force materialization before BITCAST |
| `custom(deps, code, dtype)` | Inject custom code as statement |
| `customi(deps, code, dtype)` | Inject custom code as expression |

## DType Reference

### DType Variants

| Variant | Description |
|---------|-------------|
| `Scalar(dtype)` | Base scalar type (Float32, Int32, etc.) |
| `Ptr { base, addrspace, size, vcount }` | Pointer to memory |
| `Void` | No type (for operations like STORE) |
| `Index` | Abstract integer type for indices (lowered to i32/i64) |

### Address Spaces

```rust
use morok_dtype::AddrSpace::*;

AddrSpace::Global    // GPU global memory / CPU main memory
AddrSpace::Local     // GPU shared memory / CPU L1 cache
AddrSpace::Reg       // Register memory
AddrSpace::Texture   // Texture memory
AddrSpace::Constant  // Constant memory
```

### DType Methods

| Method | Returns | Description |
|---------|----------|-------------|
| `.is_float()` | `bool` | Is floating-point type? |
| `.is_int()` | `bool` | Is integer type? |
| `.is_bool()` | `bool` | Is boolean type? |
| `.scalar()` | `Option<ScalarDType>` | Extract scalar dtype |
| `.vcount()` | `usize` | Vector width (1 = scalar, >1 = vector) |
| `.bytes()` | `usize` | Size in bytes |
| `.base()` | `&DType` | Base type for Ptr, scalar for Scalar |

### DType Conversion

```rust
// Get vector type with specified width
let vec4_dtype = DType::Float32.vec(4);  // <4 x float32>

// Create pointer type
let ptr_dtype = DType::Float32.ptr(Some(1024), AddrSpace::Global);

// Type promotion (find common type)
let common = DType::least_upper_dtype(&[a.dtype(), b.dtype()]);
```

## Validation Helpers (`ir/src/uop/constructors/mod.rs`)

The constructors module provides validation helpers:

| Function | Validates | Example Usage |
|-----------|------------|---------------|
| `promote_and_cast(lhs, rhs)` | Type promotion for binary ops | `let (lhs, rhs, dtype) = UOp::promote_and_cast(a, b)?;` |
| `check_bitwise_dtype(dtype, op)` | Int/bool requirement | Bitwise ops only on int/bool types |
| `check_division_by_zero(divisor)` | Constant zero divisor check | Compile-time check for division by zero |
| `validate_binary_shapes(lhs, rhs, op)` | Shape matching | Binary ops require exact shape match (no broadcasting at IR level) |
| `validate_ternary_shapes(true_val, false_val)` | Branch shape matching | WHERE/MULACC branches must match shapes |
| `validate_permutation(axes, expected_dims)` | Permutation validity | Check each index 0..n appears exactly once |
| `validate_reduce_axes(axes, shape_dims)` | Axis bounds | All reduction axes must be < ndim |
| `validate_flip_axes(axes, expected_dims)` | Flip spec length | Must have exactly one bool per dimension |

## Common Patterns

### Creating Constants

```rust
use morok_ir::{UOp, ConstValue, DType};

// Float constant
let float_zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));

// Integer constant with auto-inference
let int_one = UOp::native_const(1i32);

// Vector constant
let vec4_zero = UOp::vconst(vec![
    ConstValue::Float(0.0),
    ConstValue::Float(0.0),
    ConstValue::Float(0.0),
    ConstValue::Float(0.0),
]);

// Const-like operation (same dtype as another UOp)
let like_zero = some_uop.const_like(0.0);
```

### Building Index Operations

```rust
use morok_ir::UOp;

// Simple index
let index = UOp::index()
    .buffer(buffer_uop)
    .indices(vec![idx_i, idx_j])
    .call()?;

// With gate (WHERE condition moved to INDEX)
let gated_index = UOp::index()
    .buffer(buffer_uop)
    .indices(vec![idx])
    .gate(gate_uop)
    .call()?;

// Ptr dtype for STORE targets (keeps pointer type)
let ptr_index = UOp::index()
    .buffer(buffer_uop)
    .indices(vec![idx])
    .ptr(true)  // Keep Ptr dtype
    .call()?;
```

### Building Buffers

```rust
use morok_ir::{UOp, DeviceSpec, DType, AddrSpace};
use morok_dtype::DeviceSpec::*;

// Global memory buffer
let global_buf = UOp::new_buffer(
    DeviceSpec::Cpu,
    1024,
    DType::Float32,
);

// Local (shared) memory buffer
let local_buf = UOp::bufferize_local(compute_uop, ranges);

// Pointer type
let ptr_dtype = DType::Float32.ptr(Some(1024), AddrSpace::Global);
```

### Vector Operations

```rust
use morok_ir::UOp;

// Create vector from elements
let vec4 = UOp::try_vectorize(vec![x, y, z, w])?;

// Broadcast scalar to vector
let vec8 = scalar_uop.broadcast(8);

// Extract element
let first = vec4.gep(vec![0]);

// Contract (combine after unroll)
let combined = unrolled_uop.contract(vec![(0, 16)]);
```

### Range Operations

```rust
use morok_ir::{UOp, AxisId, AxisType};

// Loop range (inside kernel)
let loop_range = UOp::range_const(64, 0);

// Outer range (wraps entire kernel)
let outer_range = UOp::range_outer_const(10, 0);

// With explicit axis type
let global_range = UOp::range_axis(
    end_uop,
    AxisId::Renumbered(0),
    AxisType::Global,
);

// GPU dimension variable
let block_idx = UOp::special(end_uop, "blockIdx.x".to_string());
```

### Control Flow

```rust
use morok_ir::UOp;
use smallvec::smallvec;

// Create computation with ranges
let computation = ...;

// Wrap in END to close ranges
let result = computation.end(smallvec![range_a, range_b]);

// IF block
let if_block = UOp::if_(condition, smallvec![body_op1, body_op2]);
let end_if = UOp::endif(if_block);
```

## Key Files

| File | Purpose |
|------|---------|
| `ir/src/uop/constructors/mod.rs` | Validation helpers and module organization |
| `ir/src/uop/constructors/compute.rs` | Arithmetic, transcendental, bitwise, comparison ops |
| `ir/src/uop/constructors/data.rs` | Constants, buffers, device specifications |
| `ir/src/uop/constructors/control.rs` | Loop constructs, conditionals, variables |
| `ir/src/uop/constructors/memory.rs` | Load/store, indexing, bufferization |
| `ir/src/uop/constructors/reduce.rs` | Reduction operations |
| `ir/src/uop/constructors/shape.rs` | Shape manipulation |
| `ir/src/uop/constructors/graph.rs` | Graph organization |
| `ir/src/uop/constructors/hardware.rs` | WMMA, vectorization |
| `morok_dtype/src/lib.rs` | DType enum and methods |

## Tensor API Layer (`tensor/src/`)

The tensor layer provides ergonomic APIs on top of UOp constructors:

```rust
use morok_tensor::Tensor;

// Arithmetic operations (auto-broadcasting)
let c = &a + &b;  // Calls UOp::try_add with broadcasting

// Shape operations
let reshaped = a.try_reshape(&[2, 3])?;
let transposed = a.try_transpose(0, 1)?;

// Reductions
let sum_result = a.sum(())?;
let max_result = a.max_with().axes(0).keepdim(true).call()?;

// Matrix multiplication
let c = a.matmul(&b)?;
let c = a.dot(&b)?;  // Alias

// Realization
let result = (&a + &b).realize()?;
```

## Error Handling Patterns

Most constructors return `Result<Arc<Self>>` for type validation:

```rust
use snafu::ResultExt;

// Use context() to add error context
let result = UOp::try_add(&a, &b).context(MyErrorSnafu)?;

// Use expect() for panics in rewrites (after validation)
let result = UOp::add(&a, &b);  // Panics on type mismatch
```

## Debugging UOp Creation

### Enable UOp logging

```bash
# Log UOp construction (useful for pattern matching debugging)
RUST_LOG=morok_ir::uop=trace cargo test test_name
```

### Validate IR structure

```rust
use morok_ir::UOp;

// Check UOp tree structure
println!("{}", uop.tree());           // Compact with back-references
println!("{}", uop.tree_full());      // Full expanded tree

// Check dtype
println!("dtype: {:?}", uop.dtype());

// Check shape
if let Some(shape) = uop.shape()? {
    println!("shape: {:?}", shape);
}
```

### Common Issues

| Issue | Likely Cause | Check |
|--------|--------------|--------|
| Type mismatch error | Wrong dtype for operation | Verify operands support the operation (e.g., bitwise needs int) |
| Shape validation failure | Broadcasting incompatible | Check if shapes can align |
| Division by zero at compile time | Constant zero divisor | Use `check_division_by_zero` before creating DIV |
| Invalid permutation | Duplicate or missing axis | Use `validate_permutation` or `normalize_axes` |
| Index type mismatch | Non-Index in indices | Ensure all indices have Index dtype |
| Buffer size mismatch | Size doesn't match allocation | Verify size × dtype.bytes() matches expected |
