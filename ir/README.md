# svod-ir

Core IR with UOp graph representation, operations, and symbolic integers.

## Example

```rust
use svod_ir::{UOp, ConstValue};
use svod_dtype::DType;

let a = UOp::const_(DType::float32(), ConstValue::Float(1.0));
let b = UOp::const_(DType::float32(), ConstValue::Float(2.0));

// Fallible API - returns Result, handles type mismatches gracefully
let sum = a.try_add_op(&b)?;

// Using .unwrap() will panic on type errors or invalid operations
let sum = a.try_add_op(&b).unwrap();  // panics if types incompatible
```

## Origin Tracking

Every UOp built inside an `OriginScope` records where it came from — a module path
segment, a public op's call site, an ONNX node, or a free label — as a 4-byte id into a
process-global arena. The id is part of the node's content hash, so identical subgraphs
built under different scopes stay distinct nodes; at the kernel cut the body is stripped
and the origins move onto the kernel `CALL`, so kernel caches still deduplicate. Capture
is off by default; `SVOD_ORIGIN=1` (or `origin::capture_for_thread(true)`) turns it on.

```rust
use svod_ir::origin::{self, OriginScope};

let _capture = origin::capture_for_thread(true);
let _encoder = OriginScope::module("encoder");
let _layer = OriginScope::module("layers.3");
let c = UOp::const_(DType::Float32, ConstValue::Float(1.5));

// Walks the parent chain in the arena.
assert_eq!(origin::path(c.origin().unwrap()), "encoder.layers.3");
```

## Features

**Supported:**

- 80+ operations (arithmetic, memory, control flow)
- UOp graph with topological traversal
- Symbolic integers (SInt) for shape expressions
- Origin tracking: scopes attribute nodes and kernels to module paths, call sites and ONNX nodes
- Tensor core ops (WMMA)

**Planned:**

- Custom kernel ops
- Graph visualization

## Constructors

| Category | Methods | File |
|----------|---------|------|
| **Constants** | `const_`, `var`, `unique`, `noop` | `uop/constructors.rs` |
| **Memory** | `new_buffer`, `contiguous_slice`, `define_global`, `define_local` | `uop/constructors.rs` |
| **Load/Store** | `load`, `load_gated`, `store`, `store_gated`, `index` | `uop/constructors.rs` |
| **Arithmetic** | `try_add_op`, `try_sub_op`, `try_mul_op`, `try_div_op`, `try_mod_op` | `ops/arithmetic.rs` |
| **Unary** | `neg`, `abs`, `not`, `try_sqrt`, `try_exp`, `try_log` | `uop/constructors.rs` |
| **Bitwise** | `try_and_op`, `try_or_op`, `try_xor_op`, `try_shl_op`, `try_shr_op` | `ops/bitwise.rs` |
| **Comparison** | `try_cmplt`, `try_cmple`, `try_cmpeq`, `try_cmpne`, `try_cmpgt` | `uop/constructors.rs` |
| **Movement** | `try_reshape`, `try_expand`, `try_permute`, `try_pad`, `try_shrink` | `ops/movement.rs` |
| **Reduction** | `reduce`, `try_reduce_axis`, `allreduce` | `ops/reduction.rs` |
| **Control** | `range`, `range_axis`, `end`, `if_`, `endif`, `barrier` | `ops/control.rs` |
| **Advanced** | `where_op`, `wmma`, `bufferize`, `kernel`, `sink` | `ops/advanced.rs` |
| **Vector** | `vectorize`, `gep` | `ops/vector.rs` |

## Testing

```bash
cargo test -p svod-ir
```
