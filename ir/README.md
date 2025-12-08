# morok-ir

Core IR with UOp graph representation, operations, and symbolic integers.

## Example

```rust
use morok_ir::{UOp, ConstValue};
use morok_dtype::DType;

let a = UOp::const_(DType::float32(), ConstValue::Float(1.0));
let b = UOp::const_(DType::float32(), ConstValue::Float(2.0));

// Fallible API - returns Result, handles type mismatches gracefully
let sum = a.try_add_op(&b)?;

// Using .unwrap() will panic on type errors or invalid operations
let sum = a.try_add_op(&b).unwrap();  // panics if types incompatible
```

## Provenance Tracking

UOp creation locations are automatically tracked via `#[track_caller]`. This enables debugging by tracing where each node originated in Rust source code.

```rust
use morok_ir::provenance::PROVENANCE_TRACKER;

let c = UOp::const_(DType::Float32, ConstValue::Float(1.5));

// Query provenance
PROVENANCE_TRACKER.with(|t| {
    let chain = t.borrow().get_chain(c.id());
    // Returns: file path, line, column where UOp was created
});
```

**Captured info:**

- Workspace-relative file path, line, column
- Transformation history (substitution, pattern rewrites)
- ONNX node info (for model import)

## Features

**Supported:**

- 80+ operations (arithmetic, memory, control flow)
- UOp graph with topological traversal
- Symbolic integers (SInt) for shape expressions
- Provenance tracking with `#[track_caller]`
- Tensor core ops (WMMA)

**Planned:**

- Custom kernel ops
- Graph visualization

## Constructors

| Category | Methods | File |
|----------|---------|------|
| **Constants** | `const_`, `var`, `unique`, `noop` | `uop/constructors.rs` |
| **Memory** | `new_buffer`, `buffer_view`, `define_global`, `define_local` | `uop/constructors.rs` |
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
cargo test -p morok-ir
```
